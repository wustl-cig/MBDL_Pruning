import copy
import os
import json
import datetime
import fire

import h5py
import numpy as np
import torch
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torch_util.module import CNNBlock, EDSR, DeepUnfolding, DEQ, ResBlock, DnCNN, crop_images
from Torch_Pruning import torch_pruning as tp
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
from torch_util.module import single_ftran, single_fmult, mul_ftran, mul_fmult
from torch_util.metrics import Stack, Mean, compare_psnr, compare_ssim, compare_snr
from torch_util.common import write_pruning, to_tiff, check_and_mkdir
from method.DeCoLearn import inputDataDict, absolute_helper
from dataset.pmri_fastmri_brain import RealMeasurement, uniformly_cartesian_mask, fmult, ftran
from dataset.modl import MoDLDataset, CombinedDataset

from method.DeCoLearn import inputDataDict
from method import DeCoLearn as DeCoLearn
from method import PNPtuning as PNPtuning
from method.PNPtuning import pnp_tuning

from sota_module.method.baseline.istanetplus import ISTANetPlusLightening
from sota_module.method.baseline.e2evarnet import E2EVarNetWrapper, DatasetWrapper
# from sota_module.method.baseline import e2evarnet

from sota_module.baseline.e2e_varnet.varnet import VarNet


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class DictDataset(Dataset):
    def __init__(self, mode, data_dict, config):

        self.__data_dict = data_dict
        self.config = config

        assert mode in ['train', 'valid', 'test']

        self.__index_map = []

        n_slice = self.__data_dict['fixed_x'].shape[0]

        for slice_ in range(n_slice):
            self.__index_map.append([slice_])

        total_len = self.__index_map.__len__()
        print("total_len: ", total_len)

        if mode == 'train':
            self.__index_map = self.__index_map[0: 410]

        elif mode == 'valid':
            self.__index_map = self.__index_map[410: 470]

        else:
            self.__index_map = self.__index_map[470:]

    def __len__(self):
        return len(self.__index_map)

    def __getitem__(self, item):
        slice_, = self.__index_map[item]

        fixed_x = self.__data_dict['fixed_x'][slice_]
        moved_x = self.__data_dict['moved_x'][slice_]
        sensitivity_map = self.__data_dict['sensitivity_map'][slice_]

        fixed_y = self.__data_dict['fixed_y'][slice_]
        fixed_mask = self.__data_dict['fixed_mask'][slice_]
        fixed_y_tran = self.__data_dict['fixed_y_tran'][slice_]

        moved_y = self.__data_dict['moved_y'][slice_]
        moved_mask = self.__data_dict['moved_mask'][slice_]
        moved_y_tran = self.__data_dict['moved_y_tran'][slice_]

        mul_fixed_y = self.__data_dict['mul_fixed_y'][slice_]
        mul_fixed_mask = self.__data_dict['mul_fixed_mask'][slice_]
        mul_fixed_y_tran = self.__data_dict['mul_fixed_y_tran'][slice_]

        mul_moved_y = self.__data_dict['mul_moved_y'][slice_]
        mul_moved_y_tran = self.__data_dict['mul_moved_y_tran'][slice_]
        mul_moved_mask = self.__data_dict['mul_moved_mask'][slice_]

        return fixed_x, moved_x, sensitivity_map, \
            fixed_y, fixed_mask, fixed_y_tran, \
            moved_y, moved_mask, moved_y_tran, \
            mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, \
            mul_moved_y, mul_moved_y_tran, mul_moved_mask


def pruning_recon_module(model, importance, dataset, sparsity, module_name=None, fine_tuning=True):
    with open('config.json') as File:
        config = json.load(File)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    nf_enc = config['module']['regis']['nf_enc']
    nf_dec = config['module']['regis']['nf_dec']
    iteration_k = config['module']['recon']['iteration_k']
    mu_list = config['module']['recon']['mu_list']  # for RED
    gamma_list = config['module']['recon']['gamma_list']  # for pnp and RED
    alpha_list = config['module']['recon']['alpha_list']  # for pnp
    recon_module_type = config['module']['recon']['recon_module_type']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    mul_coil = config['dataset']['multi_coil']

    generator = torch.Generator()
    generator.manual_seed(0)
    # Dataset processing

    method_dict = {
        'DeCoLearn': DeCoLearn,
    }

    data_split_ratio = config['setting']['data_split_ratio']
    dataset_lengths = [int(len(dataset) * data_split_ratio[0]), int(len(dataset) * data_split_ratio[1]),
                       len(dataset) - int(len(dataset) * data_split_ratio[0]) - int(len(dataset) * data_split_ratio[1])]

    ########## Produce Dummy Input for the pruning ##########
    dummy_index = [0]
    dummy_dataset = MoDLDataset()

    if module_name == "VARNET":
        dummy_dataset = DatasetWrapper(dummy_dataset)

    _, dummy_mul_y_tran, _, dummy_mul_y, dummy_x, dummy_mask, dummy_sensitivity_map = \
        (i.cuda() for i in next(iter(DataLoader(Subset(dummy_dataset, dummy_index), batch_size=1))))

    dummy_inputs = inputDataDict(dummy_mul_y_tran, dummy_mask, dummy_sensitivity_map, dummy_mul_y, module_name=module_name)
    #########################################################

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, dataset_lengths, generator=generator)

    if module_name == "VARNET":
        train_dataset = DatasetWrapper(train_dataset);
        valid_dataset = DatasetWrapper(valid_dataset);
        test_dataset = DatasetWrapper(test_dataset)
    else:
        pass


    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / batch_size)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)
    valid_iter_total = int(valid_dataset.__len__() / 1)

    # %%%%%%% Example Input Data Processing %%%%%%%
    sample_indices = [2]

    valid_sample_single_y_tran, valid_sample_mul_y_tran, valid_sample_single_y, valid_sample_mul_y, valid_sample_x, valid_sample_mask, valid_sample_sensitivity_map = \
        (i.cuda() for i in next(iter(
            DataLoader(Subset(valid_dataset, sample_indices), batch_size=len(sample_indices)))))

    if mul_coil == False:
        valid_sample_y = valid_sample_single_y
        valid_sample_mask = valid_sample_mask
        valid_sample_y_tran = valid_sample_single_y_tran
    else:  # mul_coil == True
        valid_sample_y = valid_sample_mul_y
        valid_sample_mask = valid_sample_mask
        valid_sample_y_tran = valid_sample_mul_y_tran
    unpruned_model = copy.deepcopy(model)

    print("[trainer]valid_sample_fixed_x.shape: ", valid_sample_x.shape)

    example_inputs = inputDataDict(valid_sample_y_tran, valid_sample_mask, valid_sample_sensitivity_map, valid_sample_y,
                                   module_name=module_name)

    ignored_layers = []
    
    for ii, m in enumerate(model.modules()):

        if module_name == "VARNET":
            try:
                ignored_layers.append(m.varnet.sens_net)
            except:
                pass

        if (ii > sum(1 for _ in model.modules()) - 2):
            ignored_layers.append(m)
    print(len(ignored_layers))

    model = model.cuda()

    model_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("[1]Beginning Process")
    if module_name == "VARNET":
        indexOfWeight = [0, 1, 2, 3, 4, 5, 6, 7]
        detected_unwrapped = []
        for i in indexOfWeight:
            detected_unwrapped.append(model.varnet.cascades[i].dc_weight)
        unwrapped_parameters = []
        for ii, v in enumerate(detected_unwrapped):
            unwrapped_parameters.append((v, 0))

    elif module_name == "ISTANET":
        unwrapped_parameters = []
        indexOfWeight = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        detected_unwrapped = []
        for i in indexOfWeight:
            detected_unwrapped.append(model.net.fcs[i].lambda_step)
        detected_unwrapped.append(model.net.fcs[i].soft_thr)
        unwrapped_parameters = []
        for ii, v in enumerate(detected_unwrapped):
            unwrapped_parameters.append((v, 0))

    else:
        unwrapped_parameters = None

    print("[2]Second Process")

    iterative_steps = config['pruning']['iterative_steps']  # progressive pruning

    if importance == "random":
        imp = tp.importance.RandomImportance()
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            # unwrapped_parameters = unwrapped_parameters
        )
    elif importance == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers
        )
    elif importance == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers
        )
    elif importance == "slim":
        imp = tp.importance.BNScaleImportance()
        pruner = tp.pruner.BNScalePruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers
        )
    elif importance == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner = tp.pruner.GroupNormPruner(
            model,
            dummy_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )
        print("Pruner Update")

    print("[3]Third Process")
    if fine_tuning == False:
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
        initial_nparams = base_nparams
        final_nparams = 0
        for i in range(iterative_steps):
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
            if i == iterative_steps - 1:
                final_nparams = nparams
            # print(model)
            # print(model(example_inputs).shape)
            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )
    else:
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
        initial_nparams = base_nparams
        final_nparams = 0
        for i in range(iterative_steps):
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, dummy_inputs)
            # pruner.setModel(model)
            if i == iterative_steps - 1:
                final_nparams = nparams
            # print(model)
            # print(model(example_inputs).shape)
            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )
            if importance == "group_norm":
                pruner.getGroupUpdate()
                if config['pruning']["student_teacher"] == True:
                    model, pruner = method_dict[config['setting']['method']].recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'], module_name=module_name,
                        pruner=pruner, unpruned_recon_module=unpruned_model, pruning_importance=importance
                    )
                else:
                    model, pruner = method_dict[config['setting']['method']].recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'], module_name=module_name,
                        pruner=pruner, pruning_importance=importance
                    )
            else:
                if config['pruning']["student_teacher"] == True:
                    model = method_dict[config['setting']['method']].recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'],
                        module_name=module_name, unpruned_recon_module=unpruned_model, pruning_importance=importance
                    )
                else:
                    model = method_dict[config['setting']['method']].recon_or_prior_train(
                        dataset=dataset,
                        recon_module=model,
                        regis_module=None,
                        config=config, root_path=config['reconstruction']['save_root_path'],
                        module_name=module_name, pruning_importance=importance
                    )
    return model, unpruned_model, initial_nparams, final_nparams

def main(gpu_index=0):
    with open('config.json') as File:
        config = json.load(File)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    purposeOfProgram = config['setting']['purpose']
    print(purposeOfProgram)

    from torch_util.module import cvpr2018_net as voxelmorph
    from torch_util.module import EDSR, DeepUnfolding, DEQ
    from method import DeCoLearn as DeCoLearn
    from dataset.modl import load_synthetic_MoDL_dataset
    mu_list = config['module']['recon']['mu_list']  # for RED
    gamma_list = config['module']['recon']['gamma_list']  # for pnp and RED
    alpha_list = config['module']['recon']['alpha_list']  # for pnp
    iteration_k = config['module']['recon']['iteration_k']

    generator = torch.Generator()
    generator.manual_seed(0)

    if config['setting']['dataset_type'] == "MoDLDataset":
        MoDLdataset = MoDLDataset()
    elif config['setting']['dataset_type'] == "RealMeasurement":
        MoDLdataset = RealMeasurement(
            idx_list=range(1375),
            acceleration_rate=1,
            is_return_y_smps_hat=True)

    elif config['setting']['dataset_type'] == "Merge":
        MoDLdataset = CombinedDataset()

    else:
        print("MENTIONED DATASET DOESN'T EXIST")

    data_split_ratio = config['setting']['data_split_ratio']
    dataset_lengths = [int(len(MoDLdataset) * data_split_ratio[0]), int(len(MoDLdataset) * data_split_ratio[1]),
                       len(MoDLdataset) - int(len(MoDLdataset) * data_split_ratio[0]) - int(
                           len(MoDLdataset) * data_split_ratio[1])]

    sample_indices = []
    
    for count in range(int(len(MoDLdataset) * data_split_ratio[1])):
        if count % 10 == 0:
            sample_indices.append(count)
    print(f"sample_indices: {sample_indices}")
    print(f"sample_indices: {sample_indices}")
    print(f"sample_indices: {sample_indices}")
    print(f"sample_indices: {sample_indices}")

    _, _, example_dataset = torch.utils.data.random_split(MoDLdataset, dataset_lengths, generator=generator)
    sample_single_y_tran = []
    sample_mul_y_tran = []
    sample_single_y = []
    sample_mul_y = []
    sample_x = []
    sample_mask = []
    sample_sensitivity_map = []
    for sample in DataLoader(Subset(example_dataset, sample_indices), batch_size=1):
        sample_single_y_tran_instant, sample_mul_y_tran_instant, sample_single_y_instant, sample_mul_y_instant, sample_x_instant, sample_mask_instant, sample_sensitivity_map_instant = (
            i.cuda() for i in sample)
        sample_single_y_tran.append(sample_single_y_tran_instant)
        sample_mul_y_tran.append(sample_mul_y_tran_instant)
        sample_single_y.append(sample_single_y_instant)
        sample_mul_y.append(sample_mul_y_instant)
        sample_x.append(sample_x_instant)
        sample_mask.append(sample_mask_instant)
        sample_sensitivity_map.append(sample_sensitivity_map_instant)

    print(f"len(sample_x): {len(sample_x)}")
    print(f"len(sample_x): {len(sample_x)}")
    print(f"len(sample_x): {len(sample_x)}")
    print(f"len(sample_x): {len(sample_x)}")

    tiff_log_metrics = Stack()
    tiff_base_log_batch = {
        "Purpose": config['setting']['purpose'],
        "Mul_Coil": config['dataset']['multi_coil'],
    }
    excel_log_metrics = Stack()
    pruning_base_log_batch = {
        "Purpose": config['setting']['purpose'],
        "Mul_Coil": config['dataset']['multi_coil'],
        "Pruning_Iterative_Steps": config['pruning']['iterative_steps'],
        "Data_Fidelity": config['pruning']['data_fidelity']
    }
    tiff_path = config['setting']['root_path'] + config['setting']['tiff_path']
    check_and_mkdir(tiff_path)

    # %%%%%%%%%%%%%%%%%%%%%%%%%% [BEGIN] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%
    check_and_mkdir(tiff_path + 'ground_truth/')
    check_and_mkdir(tiff_path + 'under_sample/')

    for i in range(len(sample_indices)):
        ground_truth = sample_x[i].permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
        ground_truth = np.sqrt(ground_truth[:, :, :, 0] ** 2 + ground_truth[:, :, :, 1] ** 2)
        (sample_mul_y_tran[i])[sample_x[i] == 0] = 0
        under_sample = sample_mul_y_tran[i].permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
        under_sample = np.sqrt(under_sample[:, :, :, 0] ** 2 + under_sample[:, :, :, 1] ** 2)
        to_tiff(ground_truth, path=tiff_path + 'ground_truth/' + 'ground_truth' + "_" + str(i) + '.tiff',
                is_normalized=True)
        to_tiff(under_sample, path=tiff_path + 'under_sample/' + 'under_sample' + "_" + str(i) + '.tiff',
                is_normalized=True)

        tiff_base_log_batch = {**tiff_base_log_batch, 'Under_sample_ssim' + str(i): compare_ssim(
            absolute_helper(crop_images(sample_mul_y_tran[i])), absolute_helper(crop_images(sample_x[i]))).item()}
        tiff_base_log_batch = {**tiff_base_log_batch, 'Under_sample_psnr' + str(i): compare_psnr(
            absolute_helper(crop_images(sample_mul_y_tran[i])), absolute_helper(crop_images(sample_x[i]))).item()}

    # %%%%%%%%%%%%%%%%%%%%%%%%%% [Begin]Pruning Code %%%%%%%%%%%%%%%%%%%%%%%%%%%
    pruning_module_name_list = config['pruning']['module']
    # pruning_module_path = config['pruning']['module_path']
    pruning_module_path_du = config['pruning']['module_path_du']

    importance_list = config['pruning']['importance']
    sparsity_list = config['pruning']['ch_sparsity']

    print(f"%%%% [Pruning] Purpose: Pruing %%%%")
    print(f"%%%% [Pruning] Fine Tuning: {config['pruning']['fine_tuning']} %%%%")
    print(f"%%%% [Pruning] Fine Tuning Loss Type: {config['pruning']['fine_tune_loss_type']} %%%%")
    for ii, pruning_module_name in enumerate(pruning_module_name_list):
        print(f"%%%% [Pruning] Pruned Module: {pruning_module_name} %%%%")
        generator = torch.Generator()
        generator.manual_seed(0)
        _, _, example_dataset = torch.utils.data.random_split(MoDLdataset, dataset_lengths, generator=generator)
        if pruning_module_name == "VARNET":
            example_dataset = DatasetWrapper(example_dataset)
        sample_single_y_tran = []
        sample_mul_y_tran = []
        sample_single_y = []
        sample_mul_y = []
        sample_x = []
        sample_mask = []
        sample_sensitivity_map = []
        for sample in DataLoader(Subset(example_dataset, sample_indices), batch_size=1):
            sample_single_y_tran_instant, sample_mul_y_tran_instant, sample_single_y_instant, sample_mul_y_instant, sample_x_instant, sample_mask_instant, sample_sensitivity_map_instant = \
                (i.cuda() for i in sample)
            sample_single_y_tran.append(sample_single_y_tran_instant)
            sample_mul_y_tran.append(sample_mul_y_tran_instant)
            sample_single_y.append(sample_single_y_instant)
            sample_mul_y.append(sample_mul_y_instant)
            sample_x.append(sample_x_instant)
            sample_mask.append(sample_mask_instant)
            sample_sensitivity_map.append(sample_sensitivity_map_instant)

        if config['dataset']['multi_coil'] == True:
            test_time_inputs = inputDataDict(sample_mul_y_tran[0], sample_mask[0], sample_sensitivity_map[0],
                                                sample_mul_y[0], module_name=pruning_module_name)
            test_inputs = []
            for i in range(len(sample_indices)):
                test_inputs.append(
                    inputDataDict(sample_mul_y_tran[i], sample_mask[i], sample_sensitivity_map[i], sample_mul_y[i],
                                    module_name=pruning_module_name))

        else:
            test_time_inputs = inputDataDict(sample_single_y_tran[0], sample_mask[0], sample_sensitivity_map[0],
                                                sample_single_y[0], module_name=pruning_module_name)
            test_inputs = []
            for i in range(len(sample_indices)):
                test_inputs.append(inputDataDict(sample_single_y_tran[i], sample_mask[i], sample_sensitivity_map[i],
                                                    sample_single_y[i], module_name=pruning_module_name))

        pruning_instant_log_batch = pruning_base_log_batch
        tiff_instant_log_batch = tiff_base_log_batch
        tiff_instant_log_batch = {**tiff_instant_log_batch, 'Module': pruning_module_name}
        pruning_instant_log_batch = {**pruning_instant_log_batch, 'Module': pruning_module_name}

        print(f"%%%%%%% Test Unpruned Model %%%%%%%")
        method_dict = {
            'DeCoLearn': DeCoLearn,
        }

        for jj, importance in enumerate(importance_list):
            # Do pruning
            pruning_iinstant_log_batch = pruning_instant_log_batch
            tiff_iinstant_log_batch = tiff_instant_log_batch

            if pruning_module_name == "CNNBlock":
                pruning_module_architecture = CNNBlock()

            elif pruning_module_name == "EDSR":
                pruning_module_architecture = EDSR(
                    n_resblocks=config['module']['recon']['EDSR']['n_resblocks'],
                    n_feats=config['module']['recon']['EDSR']['n_feats'],
                    res_scale=config['module']['recon']['EDSR']['res_scale'],
                    in_channels=2,
                    out_channels=2,
                    dimension=2,
                )

            elif pruning_module_name == "DEQ":
                pruning_module_architecture = DEQ(mu_list[0], gamma_list[0], alpha_list[0], pruning_module_name)
                # I should code to get the mu & gamma & alpha for the parameter
                # model_architecture = DEQ(0.8, 0.1, 0.8)
            elif pruning_module_name == "DU":
                pruning_module_architecture = DeepUnfolding(iteration_k, mu_list[0], gamma_list[0], alpha_list[0],
                                                            pruning_module_name)
                # I should code to get the mu & gamma & alpha for the parameter
                # model_architecture = DEQ(0.8, 0.1, 0.8)

            elif pruning_module_name == "ISTANET":
                pruning_module_architecture = ISTANetPlusLightening(config)
                model_sum = sum(p.numel() for p in pruning_module_architecture.parameters() if p.requires_grad)
                print(f"{pruning_module_name} model_sum: {model_sum}")
                print(f"{pruning_module_name} model_sum: {model_sum}")
                print(f"{pruning_module_name} model_sum: {model_sum}")
                print(f"{pruning_module_name} model_sum: {model_sum}")
                print(f"{pruning_module_name} model_sum: {model_sum}")
                print(f"{pruning_module_name} model_sum: {model_sum}")
                print(f"{pruning_module_name} model_sum: {model_sum}")

            elif pruning_module_name == "VARNET":
                pruning_module_architecture = E2EVarNetWrapper(config)
                model_sum = sum(p.numel() for p in pruning_module_architecture.parameters() if p.requires_grad)
                print(f"{pruning_module_name} VARNET model_sum: {model_sum}")
                print(f"{pruning_module_name} VARNET model_sum: {model_sum}")
                print(f"{pruning_module_name} VARNET model_sum: {model_sum}")
                print(f"{pruning_module_name} VARNET model_sum: {model_sum}")
                print(f"{pruning_module_name} VARNET model_sum: {model_sum}")
                # %%%%%%% Model processing

            if config['pruning']['module_loading'] == False:
                # We don't need to use model path in this case
                print(f"%%%% [Pruning] Module Loading: False %%%%")
                model = pruning_module_architecture

            else:  # Load saved model
                print(f"%%%% [Pruning] Module Loading: True %%%%")
                if pruning_module_name == "EDSR":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_edsr'], map_location=torch.device(device)))
                elif pruning_module_name == "DU":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_du'], map_location=torch.device(device)))
                elif pruning_module_name == "DEQ":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_deq'], map_location=torch.device(device)))
                    """HERE"""
                    pruning_module_architecture.setAlpha(alpha=nn.Parameter(torch.tensor(pruning_module_architecture.getAlpha()),requires_grad=False))

                    pruning_module_architecture.setGamma(
                        gamma=nn.Parameter(torch.tensor(pruning_module_architecture.getGamma()),
                                            requires_grad=True))

                    pruning_module_architecture.setMu(mu=nn.Parameter(torch.tensor(mu_list[0]), requires_grad=True))
                elif pruning_module_name == "ISTANET":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_istanet'], map_location=torch.device(device)))
                elif pruning_module_name == "VARNET":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_varnet'], map_location=torch.device(device)))

                pruning_module_architecture.cuda()

                model = pruning_module_architecture

            print(f"\n\n%%%  Saved Model Performance  %%%")
            zero_ssim, zero_psnr, unpruned_ssim, unpruned_psnr = method_dict[
                config['setting']['method']].pruning_test(
                dataset=MoDLdataset,
                recon_module=pruning_module_architecture,
                regis_module=None,
                config=config, recon_module_type=pruning_module_name
            )

            # Measuring test time
            saved_starter, saved_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if pruning_module_name == "DEQ":
                saved_repetitions = 30
            else:
                saved_repetitions = 100

            saved_timings = np.zeros((saved_repetitions, 1))
            # GPU-WARM-UP
            for _ in range(10):
                _ = pruning_module_architecture(test_time_inputs)
            # MEASURE PERFORMANCE
            with torch.no_grad():
                for rep in range(saved_repetitions):
                    saved_starter.record()
                    _ = pruning_module_architecture(test_time_inputs)
                    saved_ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    saved_curr_time = saved_starter.elapsed_time(saved_ender)
                    saved_timings[rep] = saved_curr_time

            # %%%%%%%%%%%%%%%%%%%%%%%%%% [BEGIN] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%
            saved_model_path = tiff_path + pruning_module_name + '/' + importance + '/saved_model/'
            check_and_mkdir(saved_model_path)
            print(
                f"VARNET CHECKING memory_allocated: {torch.cuda.memory_allocated(device=None) / (1024 * 1024 * 1024):2f}")
            for i in range(len(sample_indices)):
                if pruning_module_name == "ISTANET":
                    [saved_recon, _] = pruning_module_architecture(test_inputs[i])
                    saved_recon = saved_recon.detach()
                    # saved_recon = (torch.view_as_real(saved_recon)).permute([0, 3, 1, 2]).contiguous()
                elif pruning_module_name == "VARNET":
                    [saved_recon, _] = pruning_module_architecture(test_inputs[i])
                    saved_recon = saved_recon.detach()
                    # saved_recon = saved_recon.permute([0, 3, 1, 2]).contiguous()
                else:
                    saved_recon = pruning_module_architecture(test_inputs[i]).detach()
                saved_recon[sample_x[i] == 0] = 0

                saved_recon_tiff = saved_recon.permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
                saved_recon_tiff = np.sqrt(saved_recon_tiff[:, :, :, 0] ** 2 + saved_recon_tiff[:, :, :, 1] ** 2)
                to_tiff(saved_recon_tiff, path=saved_model_path + "saved_model_" + str(i) + '.tiff',
                        is_normalized=True)

                tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'saved_ssim' + str(i): compare_ssim(
                    absolute_helper(crop_images(saved_recon)), absolute_helper(crop_images(sample_x[i]))).item()}
                tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'saved_psnr' + str(i): compare_psnr(
                    absolute_helper(crop_images(saved_recon)), absolute_helper(crop_images(sample_x[i]))).item()}

            tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'Pruning_Criteria': importance}
            tiff_iinstant_log_batch = {**tiff_iinstant_log_batch,
                                        'Saved_Model(Test Time)': np.sum(saved_timings) / saved_repetitions}
            # %%%%%%%%%%%%%%%%%%%%%%%%%% [END] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%

            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Pruning_Criteria': importance}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Zero_Filled(SSIM)': zero_ssim}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Zero_Filled(PSNR)': zero_psnr}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch,
                                            'Saved_Model(Test Time)': np.sum(saved_timings) / saved_repetitions}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Unpruned(SSIM)': unpruned_ssim}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Unpruned(PSNR)': unpruned_psnr}
            '''HERE'''

            for kk, sparsity in enumerate(sparsity_list):
                pruning_iiinstant_log_batch = pruning_iinstant_log_batch
                tiff_iiinstant_log_batch = tiff_iinstant_log_batch
                if pruning_module_name == "CNNBlock":
                    pruning_module_architecture = CNNBlock()

                elif pruning_module_name == "EDSR":
                    pruning_module_architecture = EDSR(
                        n_resblocks=config['module']['recon']['EDSR']['n_resblocks'],
                        n_feats=config['module']['recon']['EDSR']['n_feats'],
                        res_scale=config['module']['recon']['EDSR']['res_scale'],
                        in_channels=2,
                        out_channels=2,
                        dimension=2,
                    )

                elif pruning_module_name == "DEQ":
                    pruning_module_architecture = DEQ(mu_list[0], gamma_list[0], alpha_list[0], pruning_module_name)
                    # I should code to get the mu & gamma & alpha for the parameter
                    # model_architecture = DEQ(0.8, 0.1, 0.8)
                elif pruning_module_name == "DU":
                    pruning_module_architecture = DeepUnfolding(iteration_k, mu_list[0], gamma_list[0],
                                                                alpha_list[0], pruning_module_name)
                    # I should code to get the mu & gamma & alpha for the parameter
                    # model_architecture = DEQ(0.8, 0.1, 0.8)
                elif pruning_module_name == "PNP":
                    pruning_module_architecture = DeepUnfolding(iteration_k, mu_list[0], gamma_list[0],
                                                                alpha_list[0], pruning_module_name)
                    # %%%%%%% Model processing
                elif pruning_module_name == "ISTANET":
                    pruning_module_architecture = ISTANetPlusLightening(config)
                elif pruning_module_name == "VARNET":
                    pruning_module_architecture = E2EVarNetWrapper(config)

                if config['pruning']['module_loading'] == False:
                    # We don't need to use model path in this case
                    print(f"%%%% [Pruning] Module Loading: False %%%%")
                    model = pruning_module_architecture
                else:  # Load saved model
                    print(f"%%%% [Pruning] Module Loading: True %%%%")
                    if pruning_module_name == "EDSR":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_edsr'], map_location=torch.device(device)))
                    elif pruning_module_name == "DU":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_du'], map_location=torch.device(device)))
                    elif pruning_module_name == "PNP":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_pnp'], map_location=torch.device(device)))
                    elif pruning_module_name == "DEQ":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_deq'], map_location=torch.device(device)))
                        if config['pruning']['fine_tune_loss_type'] == "testdc":
                            """HERE"""
                            print('', end='')
                            pruning_module_architecture.setAlpha(
                                alpha=nn.Parameter(torch.tensor(pruning_module_architecture.getAlpha()),
                                                    requires_grad=False))
                            # pruning_module_architecture.setAlpha(alpha=nn.Parameter(torch.tensor(alpha_list[0]), requires_grad=False))
                            # pruning_module_architecture.setAlpha(alpha=nn.Parameter(torch.tensor(alpha_list[0]), requires_grad=True))
                            # pruning_module_architecture.setGamma(gamma=nn.Parameter(torch.tensor(gamma_list[0]), requires_grad=True))
                            pruning_module_architecture.setGamma(
                                gamma=nn.Parameter(torch.tensor(pruning_module_architecture.getGamma()),
                                                    requires_grad=True))
                            pruning_module_architecture.setMu(
                                mu=nn.Parameter(torch.tensor(mu_list[0]), requires_grad=True))
                        else:
                            pruning_module_architecture.setTol(tolValue=4e-3)

                    elif pruning_module_name == "ISTANET":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_istanet'], map_location=torch.device(device)))
                    elif pruning_module_name == "VARNET":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_varnet'], map_location=torch.device(device)))

                    pruning_module_architecture.cuda()
                    # pruning_module_architecture = torch.load(pruning_module_path)
                    # one_shot_model = copy.deepcopy(pruning_module_architecture)
                    # fine_tuned_model = copy.deepcopy(pruning_module_architecture)

                print(f"\n%%%% [Pruning - {jj}/{importance}] %%%%")
                one_shot_pruned_model, unpruned_model, initial_number_of_parameters, final_number_of_parameters = pruning_recon_module(
                    pruning_module_architecture, importance, dataset=MoDLdataset, sparsity=sparsity,
                    module_name=pruning_module_name, fine_tuning=False)
                # one_shot_pruned_model = copy.deepcopy(pruned_model)

                if config['pruning']['fine_tuning'] == False:
                    zero_ssim, zero_psnr, one_shot_pruned_ssim, one_shot_pruned_psnr = method_dict[
                        config['setting']['method']].pruning_test(
                        dataset=MoDLdataset,
                        recon_module=one_shot_pruned_model,
                        regis_module=None,
                        config=config, recon_module_type=pruning_module_name
                    )
                    print(f"%%%% [One-shot Pruning - {importance}] Pruning is Done %%%%\n\n\n")
                else:  # fine tuning mode

                    print(f"\n\n%%%  Pruned Model Performance - {importance} %%%")
                    zero_ssim, zero_psnr, one_shot_pruned_ssim, one_shot_pruned_psnr = method_dict[
                        config['setting']['method']].pruning_test(
                        dataset=MoDLdataset,
                        recon_module=one_shot_pruned_model,
                        regis_module=None,
                        config=config, recon_module_type=pruning_module_name
                    )
                    # %%%%%%%%%%%%%%%%%%%%%%%%%% [BEGIN] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%
                    one_shot_path = tiff_path + pruning_module_name + '/' + importance + '/' + str(
                        sparsity) + '/one_shot_model/'
                    check_and_mkdir(one_shot_path)
                    for i in range(len(sample_indices)):
                        if pruning_module_name == "ISTANET":
                            [saved_recon, _] = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()
                            # saved_recon = (torch.view_as_real(saved_recon)).permute([0, 3, 1, 2]).contiguous()
                        elif pruning_module_name == "VARNET":
                            [saved_recon, _] = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()
                        elif pruning_module_name == "DEQ":
                            saved_recon = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()
                        else:
                            saved_recon = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()

                        saved_recon[sample_x[i] == 0] = 0
                        saved_recon_tiff = saved_recon.permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
                        saved_recon_tiff = np.sqrt(
                            saved_recon_tiff[:, :, :, 0] ** 2 + saved_recon_tiff[:, :, :, 1] ** 2)
                        to_tiff(saved_recon_tiff,
                                path=one_shot_path + "one_shot_model_" + str(i) + '.tiff', is_normalized=True)
                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'one_shot_ssim' + str(i): compare_ssim(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(
                                                            crop_images(sample_x[i]))).item()}
                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'one_shot_psnr' + str(i): compare_psnr(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(crop_images(sample_x[i]))).item()}
                    # %%%%%%%%%%%%%%%%%%%%%%%%%% [END] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%

                    print(f"\n\n%%% Start Fine Tuning - {importance}%%%")


                    fine_tuned_pruned_model, unpruned_model, initial_number_of_parameters, final_number_of_parameters = pruning_recon_module(
                        unpruned_model, importance, dataset=MoDLdataset, sparsity=sparsity,
                        module_name=pruning_module_name, fine_tuning=True)

                    print(f"%%%End Fine Tuning - {importance}%%%")

                    print(f"\n\n%%%Fine-tuned Pruned Model - {importance}%%%")

                    zero_ssim, zero_psnr, fine_tuned_pruned_ssim, fine_tuned_pruned_psnr = method_dict[
                        config['setting']['method']].pruning_test(
                        dataset=MoDLdataset,
                        recon_module=fine_tuned_pruned_model,
                        regis_module=None,
                        config=config, recon_module_type=pruning_module_name
                    )
                    print(f"%%%% [Iterative Pruning - {importance}] Pruning is Done %%%%\n\n\n")

                    # %%%%%%%%%%%%%%%%%%%%%%%%%% [BEGIN] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%
                    fine_tuned_path = tiff_path + pruning_module_name + '/' + importance + '/' + str(
                        sparsity) + '/fine_tuned_model/'
                    check_and_mkdir(fine_tuned_path)
                    for i in range(len(sample_indices)):
                        if pruning_module_name == "ISTANET":
                            [saved_recon, _] = fine_tuned_pruned_model(test_inputs[i])
                            # saved_recon = (torch.view_as_real(saved_recon)).permute([0, 3, 1, 2]).contiguous()
                        elif pruning_module_name == "VARNET":
                            [saved_recon, _] = fine_tuned_pruned_model(test_inputs[i])
                        else:
                            saved_recon = fine_tuned_pruned_model(test_inputs[i])
                        # saved_recon = fine_tuned_pruned_model(test_inputs[i])
                        saved_recon[sample_x[i] == 0] = 0
                        saved_recon_tiff = saved_recon.permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
                        saved_recon_tiff = np.sqrt(
                            saved_recon_tiff[:, :, :, 0] ** 2 + saved_recon_tiff[:, :, :, 1] ** 2)
                        to_tiff(saved_recon_tiff,
                                path=fine_tuned_path + "fine_tuned_model_" + str(
                                    i) + '.tiff', is_normalized=True)

                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'fine_tuned_ssim' + str(i): compare_ssim(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(
                                                            crop_images(sample_x[i]))).item()}
                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'fine_tuned_psnr' + str(i): compare_psnr(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(crop_images(sample_x[i]))).item()}
                    # %%%%%%%%%%%%%%%%%%%%%%%%%% [END] TIFF SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%

                # Measuring test time
                fine_starter, fine_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True)
                if pruning_module_name == "DEQ":
                    fine_repetitions = 30
                else:
                    fine_repetitions = 100
                fine_timings = np.zeros((fine_repetitions, 1))
                # GPU-WARM-UP
                for _ in range(10):
                    _ = fine_tuned_pruned_model(test_time_inputs)
                # MEASURE PERFORMANCE
                with torch.no_grad():
                    for rep in range(fine_repetitions):
                        fine_starter.record()
                        _ = fine_tuned_pruned_model(test_time_inputs)
                        fine_ender.record()
                        # WAIT FOR GPU SYNC
                        torch.cuda.synchronize()
                        fine_curr_time = fine_starter.elapsed_time(fine_ender)
                        fine_timings[rep] = fine_curr_time

                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                            'Fine_Tuned(Test Time)': np.sum(fine_timings) / fine_repetitions}
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch, 'Pruning_Sparsity': sparsity}
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                            'Initial_nparams': initial_number_of_parameters}
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch, 'Final_nparams': final_number_of_parameters}

                torch.save(one_shot_pruned_model.state_dict(), one_shot_path + pruning_module_name + ".pt")
                torch.save(fine_tuned_pruned_model.state_dict(), fine_tuned_path + pruning_module_name + ".pt")

                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Pruning_Sparsity': sparsity}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Oneshot(SSIM)': one_shot_pruned_ssim}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Oneshot(PSNR)': one_shot_pruned_psnr}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Fine_Tuned(SSIM)': fine_tuned_pruned_ssim}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Fine_Tuned(PSNR)': fine_tuned_pruned_psnr}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Fine_Tuned(Test Time)': np.sum(fine_timings) / fine_repetitions}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Initial_nparams': initial_number_of_parameters}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Final_nparams': final_number_of_parameters}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Loss_Consensus': config['method']['loss_recon_consensus']}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch,
                                                'Prior_Type': config['pruning']['prior_path']}
                excel_log_metrics = Stack()
                excel_log_metrics.update_state(pruning_iiinstant_log_batch)
                pruing_excel_file_name = tiff_path + config['setting']['purpose'] + config['pruning'][
                    'fine_tune_loss_type'] + "tiff.csv"
                write_pruning(log_dict=excel_log_metrics.result(), save_path=pruing_excel_file_name)

                tiff_log_metrics = Stack()
                tiff_log_metrics.update_state(tiff_iiinstant_log_batch)
                write_pruning(log_dict=tiff_log_metrics.result(), save_path=tiff_path + "tiff.csv")

                del one_shot_pruned_model
                del unpruned_model
                del fine_tuned_pruned_model

                torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

if __name__ == '__main__':
    fire.Fire(main)

