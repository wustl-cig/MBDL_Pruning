# Github: https://github.com/samuro95/Prox-PnP/blob/main/GS_denoising/lightning_denoiser.py
# ID: 029ecc1

import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim import lr_scheduler
import random
import torchmetrics
from argparse import ArgumentParser, Namespace
import cv2
import torchvision
import numpy as np
from .test_utils import test_mode
import matplotlib.pyplot as plt
from .GS_utils import normalize_min_max
from .models.network_unet import UNetRes
from .models import DNCNN
from .models.FFDNET import FFDNet
from ..unet import UNet


# Added the following class to simplify the API.
class GradMatchWG(nn.Module):
    def __init__(self, grad_matching=True, channels=2, model_name='DRUNET', DRUNET_nc=64, act_mode='E'):
        super().__init__()

        DRUNET_nc = [DRUNET_nc, DRUNET_nc * 2, DRUNET_nc * 4, DRUNET_nc * 8]

        self.hparams = Namespace(
            # WG: used to set cnn channels
            channels=channels,

            # from add_model_specific_args
            model_name=model_name,  # change to DnCNN
            start_from_checkpoint=False,
            resume_from_checkpoint=False,
            pretrained_checkpoint='ckpts/GS_DRUNet.ckpt',
            pretrained_student=False,
            n_channels=3,
            nc_in=3,
            nc_out=3,
            nc=64,
            nb=20,
            act_mode=act_mode,  # Note: GSPNP and ProxPnP are different here.
            no_bias=True,
            power_method_nb_step=50,
            power_method_error_threshold=1e-2,
            power_method_error_momentum=0.,
            power_method_mean_correction=False,
            DRUNET_nb=2,
            DRUNET_nc=DRUNET_nc,
            grad_matching=grad_matching,
            weight_Ds=1.,
            residual_learning=False,

            # from add_optim_specific_args
            optimizer_type='adam',
            optimizer_lr=1e-4,
            scheduler_type='MultiStepLR',
            scheduler_milestones=[300, 600, 900, 1200],
            scheduler_gamma=0.5,
            early_stopping_patiente=5,
            gradient_clip_val=1e-2,
            val_check_interval=1.,
            min_sigma_test=0,
            max_sigma_test=50,
            min_sigma_train=0,
            max_sigma_train=50,
            sigma_list_test=[0, 15, 25, 50],
            sigma_step=False,
            get_spectral_norm=False,
            jacobian_loss_weight=0,
            eps_jacobian_loss=0.1,
            jacobian_loss_type='max',
            n_step_eval=1,
            use_post_forward_clip=False,
            use_sigma_model=False,
            sigma_model=25,
            get_regularization=False
        )

        self.net = GradMatch(hparams=self.hparams)

    def forward(self, x, sigma):

        x_hat, Dg = self.net(x, sigma)

        return x_hat


class StudentGrad(pl.LightningModule):
    """
    Standard Denoiser model
    """

    def __init__(self, model_name, pretrained, pretrained_checkpoint, act_mode, DRUNET_nb, residual_learning, channels, DRUNET_nc):
        super().__init__()
        self.model_name = model_name
        self.residual_learning = residual_learning
        if self.model_name == 'DRUNET':
            # change in_nc and out_nc to 2
            self.model = UNetRes(in_nc=channels + 1, out_nc=channels, nc=DRUNET_nc, nb=DRUNET_nb, act_mode=act_mode,
                                 downsample_mode='strideconv', upsample_mode='convtranspose')
        elif self.model_name == 'DNCNN':
            self.model = DNCNN.dncnn(channels, channels, 17, 'C', act_mode, False)
        elif self.model_name == 'FFDNET':
            self.model = FFDNet(channels, channels, 64, 15, act_mode=act_mode)
        elif self.model_name == 'UNet':
            self.model = UNet(
                dimension=2,
                i_nc=channels,
                o_nc=channels,
                f_root=32,
                conv_times=2,
                up_down_times=3,
                is_spe_norm=False,
                activation='softplus'
            )

        self.model.to(self.device)
        if pretrained:
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, val in state_dict.items():
                new_state_dict[key[6:]] = val
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x, sigma):
        if self.model_name == 'FFDNET':
            n = self.model(x, torch.full((x.shape[0], 1, 1, 1), sigma).type_as(x))
        else:
            if self.model_name == 'DRUNET':
                noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(self.device)
                x = torch.cat((x, noise_level_map), 1)
            n = self.model(x)
        if self.residual_learning:
            return x - n
        else:
            return n


class GradMatch(pl.LightningModule):
    """
    Gradient Step Denoiser
    """

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.student_grad = StudentGrad(self.hparams.model_name, self.hparams.pretrained_student,
                                        self.hparams.pretrained_checkpoint, self.hparams.act_mode,
                                        self.hparams.DRUNET_nb, self.hparams.residual_learning, self.hparams.channels, self.hparams.DRUNET_nc)
        self.train_PSNR = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.val_PSNR = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.train_teacher_PSNR = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)

    @torch.enable_grad()
    def calculate_grad(self, x, sigma):
        """
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :param sigma: Denoiser level (std)
        :return: Dg(x), DRUNet output N(x)
        """
        x = x.float()
        x = x.requires_grad_()

        if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
            N = self.student_grad.forward(x, sigma)
        else:
            current_model = lambda v: self.student_grad.forward(v, sigma)
            N = test_mode(current_model, x, mode=5, refield=64, min_size=256)

        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=True, only_inputs=True)[0]
        Dg = x - N - JN
        return Dg, N

    def forward(self, x, sigma):
        """
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :param sigma: Denoiser level (std)
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        """
        if self.hparams.grad_matching:  # If gradient step denoising
            Dg, _ = self.calculate_grad(x, sigma)
            if self.hparams.sigma_step:  # possibility to multiply Dg by sigma
                x_hat = x - self.hparams.weight_Ds * sigma * Dg
            else:  # as realized in the paper (with weight_Ds=1)
                x_hat = x - self.hparams.weight_Ds * Dg
            return x_hat, Dg
        else:  # If denoising with standard forward CNN
            x_hat = self.student_grad.forward(x, sigma)
            Dg = x - x_hat
            return x_hat, Dg

    def lossfn(self, x, y):  # L2 loss
        criterion = nn.MSELoss(reduction='none')
        return criterion(x.view(x.size()[0], -1), y.view(y.size()[0], -1)).mean(dim=1)

    def training_step(self, batch, batch_idx):
        y, _ = batch

        sigma = random.uniform(self.hparams.min_sigma_train, self.hparams.max_sigma_train) / 255
        u = torch.randn(y.size(), device=self.device)
        noise_in = u * sigma
        x = y + noise_in
        x_hat, Dg = self.forward(x, sigma)
        loss = self.lossfn(x_hat, y)
        self.train_PSNR.update(x_hat, y)

        if self.hparams.jacobian_loss_weight > 0:
            jacobian_norm = self.jacobian_spectral_norm(x, x_hat, sigma, interpolation=False, training=True)
            self.log('train/jacobian_norm_max', jacobian_norm.max(), prog_bar=True)
            if self.hparams.jacobian_loss_type == 'max':
                jacobian_loss = torch.maximum(jacobian_norm,
                                              torch.ones_like(jacobian_norm) - self.hparams.eps_jacobian_loss)
            elif self.hparams.jacobian_loss_type == 'exp':
                jacobian_loss = self.hparams.eps_jacobian_loss * torch.exp(
                    jacobian_norm - torch.ones_like(jacobian_norm) * (
                                1 + self.hparams.eps_jacobian_loss)) / self.hparams.eps_jacobian_loss
            else:
                print("jacobian loss not available")
            jacobian_loss = torch.clip(jacobian_loss, 0, 1e3)
            self.log('train/jacobian_loss_max', jacobian_loss.max(), prog_bar=True)

            loss = (loss + self.hparams.jacobian_loss_weight * jacobian_loss)

        loss = loss.mean()

        psnr = self.train_PSNR.compute()
        self.log('train/train_loss', loss.detach())
        self.log('train/train_psnr', psnr.detach(), prog_bar=True)

        if batch_idx == 0:
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
            self.logger.experiment.add_image('train/noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('train/denoised', denoised_grid, self.current_epoch)

        return loss

    def training_epoch_end(self, outputs):
        print('train PSNR updated')
        self.train_PSNR.reset()

    def validation_step(self, batch, batch_idx):
        torch.manual_seed(0)
        y, _ = batch
        batch_dict = {}

        sigma_list = self.hparams.sigma_list_test
        for i, sigma in enumerate(sigma_list):
            x = y + torch.randn(y.size(), device=self.device) * sigma / 255
            if self.hparams.use_sigma_model:  # Possibility to test with sigma model different than input sigma
                sigma_model = self.hparams.sigma_model / 255
            else:
                sigma_model = sigma / 255
            torch.set_grad_enabled(True)
            if self.hparams.grad_matching:  # GS denoise
                x_hat = x
                for n in range(self.hparams.n_step_eval):  # 1 step in practice
                    current_model = lambda v: self.forward(v, sigma_model)[0]
                    x_hat = current_model(x_hat)
                if self.hparams.get_regularization:  # Extract reguralizer value g(x)
                    N = self.student_grad.forward(x, sigma_model)
                    g = 0.5 * torch.sum((x - N).view(x.shape[0], -1) ** 2)
                    batch_dict["g_" + str(sigma)] = g.detach()
                l = self.lossfn(x_hat, y)
                self.val_PSNR.reset()
                p = self.val_PSNR(x_hat, y)
                Dg = (x - x_hat)
                Dg_norm = torch.norm(Dg, p=2)
            else:
                for n in range(self.hparams.n_step_eval):
                    current_model = lambda v: self.forward(v, sigma / 255)[0]
                    x_hat = x
                    if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
                        x_hat = current_model(x_hat)
                    elif x.size(2) % 8 != 0 or x.size(3) % 8 != 0:
                        x_hat = test_mode(current_model, x_hat, refield=64, mode=5)
                Dg = (x - x_hat)
                Dg_norm = torch.norm(Dg, p=2)
                l = self.lossfn(x_hat, y)
                self.val_PSNR.reset()
                p = self.val_PSNR(x_hat, y)

            if self.hparams.get_spectral_norm:
                jacobian_norm = self.jacobian_spectral_norm(x, x_hat, sigma_model)
                batch_dict["max_jacobian_norm_" + str(sigma)] = jacobian_norm.max().detach()
                batch_dict["mean_jacobian_norm_" + str(sigma)] = jacobian_norm.mean().detach()

            batch_dict["psnr_" + str(sigma)] = p.detach()
            batch_dict["loss_" + str(sigma)] = l.detach()
            batch_dict["Dg_norm_" + str(sigma)] = Dg_norm.detach()

        if batch_idx == 0:  # logging for tensorboard
            clean_grid = torchvision.utils.make_grid(normalize_min_max(y.detach())[:1])
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
            self.logger.experiment.add_image('val/clean', clean_grid, self.current_epoch)
            self.logger.experiment.add_image('val/noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('val/denoised', denoised_grid, self.current_epoch)

        if self.hparams.save_images:
            save_dir = 'images/' + self.hparams.name

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                os.mkdir(save_dir + '/noisy')
                os.mkdir(save_dir + '/denoised')
                os.mkdir(save_dir + '/denoised_no_noise')
                os.mkdir(save_dir + '/clean')
            for i in range(len(x)):
                clean = y[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                noisy = x[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                denoised = x_hat[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                clean = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
                noisy = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
                denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)

                if sigma < 1:
                    cv2.imwrite(save_dir + '/denoised_no_noise/' + str(batch_idx) + '.png', denoised)
                else:
                    cv2.imwrite(save_dir + '/denoised/' + str(batch_idx) + '.png', denoised)
                    cv2.imwrite(save_dir + '/clean/' + str(batch_idx) + '.png', clean)
                    cv2.imwrite(save_dir + '/noisy/' + str(batch_idx) + '.png', noisy)

        return batch_dict

    def validation_epoch_end(self, outputs):

        self.val_PSNR.reset()

        sigma_list = self.hparams.sigma_list_test
        for i, sigma in enumerate(sigma_list):
            res_mean_SN = []
            res_max_SN = []
            res_psnr = []
            res_Dg = []
            if self.hparams.get_regularization:
                res_g = []
            for x in outputs:
                if x["psnr_" + str(sigma)] is not None:
                    res_psnr.append(x["psnr_" + str(sigma)])
                res_Dg.append(x["Dg_norm_" + str(sigma)])
                if self.hparams.get_regularization:
                    res_g.append(x["g_" + str(sigma)])
                if self.hparams.get_spectral_norm:
                    res_max_SN.append(x["max_jacobian_norm_" + str(sigma)])
                    res_mean_SN.append(x["mean_jacobian_norm_" + str(sigma)])
            avg_psnr_sigma = torch.stack(res_psnr).mean()
            avg_Dg_norm = torch.stack(res_Dg).mean()
            if self.hparams.get_regularization:
                avg_s = torch.stack(res_g).mean()
                self.log('val/val_g_sigma=' + str(sigma), avg_s)
            if self.hparams.get_spectral_norm:
                avg_mean_SN = torch.stack(res_mean_SN).mean()
                max_max_SN = torch.stack(res_max_SN).max()
                self.log('val/val_max_SN_sigma=' + str(sigma), max_max_SN)
                self.log('val/val_mean_SN_sigma=' + str(sigma), avg_mean_SN)
                res_max_SN = np.array([el.item() for el in res_max_SN])
                np.save('res_max_SN_sigma=' + str(sigma) + '.npy', res_max_SN)
                # plt.hist(res_max_SN, bins='auto', label=r'\sigma='+str(sigma), alpha=0.5)
            self.log('val/val_psnr_sigma=' + str(sigma), avg_psnr_sigma)
            self.log('val/val_Dg_norm_sigma=' + str(sigma), avg_Dg_norm)
        if self.hparams.get_spectral_norm:
            plt.grid(True)
            plt.legend()
            plt.savefig('histogram.png')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optim_params = []
        for k, v in self.student_grad.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = Adam(optim_params, lr=self.hparams.optimizer_lr, weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def power_iteration(self, operator, vector_size, steps=100, momentum=0.0, eps=1e-3,
                        init_vec=None, verbose=False):
        """
        Power iteration algorithm for spectral norm calculation
        """
        with torch.no_grad():
            if init_vec is None:
                vec = torch.rand(vector_size).to(self.device)
            else:
                vec = init_vec.to(self.device)
            vec /= torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1, 1)

            for i in range(steps):

                new_vec = operator(vec)
                new_vec = new_vec / torch.norm(new_vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1,
                                                                                                  1)
                if momentum > 0 and i > 1:
                    new_vec -= momentum * old_vec
                old_vec = vec
                vec = new_vec
                diff_vec = torch.norm(new_vec - old_vec, p=2)
                if diff_vec < eps:
                    if verbose:
                        print("Power iteration converged at iteration: ", i)
                    break

        new_vec = operator(vec)
        div = torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0])
        lambda_estimate = torch.abs(
            torch.sum(vec.view(vector_size[0], -1) * new_vec.view(vector_size[0], -1), dim=1)) / div

        return lambda_estimate

    def jacobian_spectral_norm(self, y, x_hat, sigma, interpolation=False, training=False):
        """
        Get spectral norm of Dg^2 the hessian of g
        :param y:
        :param x_hat:
        :param sigma:
        :param interpolation:
        :return:
        """
        torch.set_grad_enabled(True)
        if interpolation:
            eta = torch.rand(y.size(0), 1, 1, 1, requires_grad=True).to(self.device)
            x = eta * y.detach() + (1 - eta) * x_hat.detach()
            x = x.to(self.device)
        else:
            x = y

        x.requires_grad_()
        x_hat, Dg = self.forward(x, sigma)
        if self.hparams.grad_matching:
            # we calculate the lipschitz constant of the gradient operator Dg=Id-D
            operator = lambda vec: \
            torch.autograd.grad(Dg, x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            # we calculate the lipschitz constant of the denoiser operator D
            f = x_hat
            operator = lambda vec: \
            torch.autograd.grad(f, x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
        lambda_estimate = self.power_iteration(operator, x.size(), steps=self.hparams.power_method_nb_step,
                                               momentum=self.hparams.power_method_error_momentum,
                                               eps=self.hparams.power_method_error_threshold)
        return lambda_estimate

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='DRUNET')
        parser.add_argument('--start_from_checkpoint', dest='start_from_checkpoint', action='store_true')
        parser.set_defaults(start_from_checkpoint=False)
        parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument('--pretrained_checkpoint', type=str, default='ckpts/GS_DRUNet.ckpt')
        parser.add_argument('--pretrained_student', dest='pretrained_student', action='store_true')
        parser.set_defaults(pretrained_student=False)
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--nc_in', type=int, default=3)
        parser.add_argument('--nc_out', type=int, default=3)
        parser.add_argument('--nc', type=int, default=64)
        parser.add_argument('--nb', type=int, default=20)
        parser.add_argument('--act_mode', type=str, default='s')
        parser.add_argument('--no_bias', dest='no_bias', action='store_false')
        parser.set_defaults(use_bias=True)
        parser.add_argument('--power_method_nb_step', type=int, default=50)
        parser.add_argument('--power_method_error_threshold', type=float, default=1e-2)
        parser.add_argument('--power_method_error_momentum', type=float, default=0.)
        parser.add_argument('--power_method_mean_correction', dest='power_method_mean_correction', action='store_true')
        parser.add_argument('--DRUNET_nb', type=int, default=2)
        parser.set_defaults(power_method_mean_correction=False)
        parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        parser.set_defaults(grad_matching=True)
        parser.add_argument('--weight_Ds', type=float, default=1.)
        parser.add_argument('--residual_learning', dest='residual_learning', action='store_true')
        parser.set_defaults(residual_learning=False)
        return parser

    @staticmethod
    def add_optim_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--optimizer_type', type=str, default='adam')
        parser.add_argument('--optimizer_lr', type=float, default=1e-4)
        parser.add_argument('--scheduler_type', type=str, default='MultiStepLR')
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[300, 600, 900, 1200])
        parser.add_argument('--scheduler_gamma', type=float, default=0.5)
        parser.add_argument('--early_stopping_patiente', type=str, default=5)
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--val_check_interval', type=float, default=1.)
        parser.add_argument('--min_sigma_test', type=int, default=0)
        parser.add_argument('--max_sigma_test', type=int, default=50)
        parser.add_argument('--min_sigma_train', type=int, default=0)
        parser.add_argument('--max_sigma_train', type=int, default=50)
        parser.add_argument('--sigma_list_test', type=int, nargs='+', default=[0, 15, 25, 50])
        parser.add_argument('--sigma_step', dest='sigma_step', action='store_true')
        parser.set_defaults(sigma_step=False)
        parser.add_argument('--get_spectral_norm', dest='get_spectral_norm', action='store_true')
        parser.set_defaults(get_spectral_norm=False)
        parser.add_argument('--jacobian_loss_weight', type=float, default=0)
        parser.add_argument('--eps_jacobian_loss', type=float, default=0.1)
        parser.add_argument('--jacobian_loss_type', type=str, default='max')
        parser.add_argument('--n_step_eval', type=int, default=1)
        parser.add_argument('--use_post_forward_clip', dest='use_post_forward_clip', action='store_true')
        parser.set_defaults(use_post_forward_clip=False)
        parser.add_argument('--use_sigma_model', dest='use_sigma_model', action='store_true')
        parser.set_defaults(use_sigma_model=False)
        parser.add_argument('--sigma_model', type=int, default=25)
        parser.add_argument('--get_regularization', dest='get_regularization', action='store_true')
        parser.set_defaults(get_regularization=False)
        return parser
