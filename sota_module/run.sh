#!/bin/bash
host=$(hostname)

if [ $host = cigserver4.seas.wustl.edu ];then

      if [ $# -eq 0 ];then
        echo "no argument or the argument"

      else

        docker run -it -v /export/project/gan.weijie/project/2023_nips:/opt/project:ro \
        -v /export2/project/Dataset:/opt/dataset \
        -v /export/project/gan.weijie/experiment:/opt/experiment \
        --gpus '"device='$1'"' -u 1942396:1245315 -w /opt/project --shm-size 16GB \
        wjgan95/acal:beta /opt/miniconda3/bin/python $2 $3

      fi

elif [ $host = cigserver3.seas.wustl.edu ];then

if [ $# -eq 0 ];then
  echo "no argument or the argument"

else

echo $1
echo '"device='$1'"'

docker run -it -v /export/project/gan.weijie/000_project/deq_cal:/opt/project:ro \
-v /export/project/gan.weijie/011_raw:/opt/raw:ro \
-v /export/project/gan.weijie/010_dataset:/opt/dataset \
-v /export/project/gan.weijie/020_experiment:/opt/experiment \
--gpus '"device='$1'"' -u 1942396:1245315 -w /opt/project --shm-size 16GB \
wjgan95/acal:beta /opt/miniconda3/bin/python $2 $3

fi

elif [ $host = home1 ];then

      if [ $# -eq 0 ];then
        echo "no argument or the argument"

      else

        docker run -it -v /export/project/gan.weijie/project/2023_nips:/opt/project:ro \
        -v /export/project/gan.weijie/dataset:/opt/dataset \
        -v /export/project/gan.weijie/experiment:/opt/experiment \
        --gpus '"device='$1'"' -u 1000:1000 -w /opt/project --shm-size 16GB \
        wjgan95/acal:beta /opt/miniconda3/bin/python $2 $3

      fi

elif [[ $host == *engr.wustl.edu* ]]; then

      echo "Using the EIT cluster"

      singularity shell --nv \
      --bind /home/research/gan.weijie/project/2023_nips:/opt/project \
      --bind /project/cigserver5/export1/gan.weijie/experiment:/opt/experiment \
      --bind /project/cigserver4/export2/Dataset:/opt/dataset \
      -W /opt/project \
      /project/cigserver5/export/gan.weijie/images/acal_beta.sif

elif [[ $host == *engr.wustl.edu* ]]; then

      echo "Using the RIS cluster"

#my_var=`date +"%Y%m%d_%H%M%S_"`deq_cal
#cp -r /home/g.weijie/000_project/deq_cal /storage1/fs1/kamilov/Active/gan.weijie/009_running_ris/$my_var
#
#export LSF_DOCKER_VOLUMES="/storage1/fs1/kamilov/Active/gan.weijie/009_running_ris/$my_var:/opt/project,readonly /storage1/fs1/kamilov/Active/gan.weijie/010_dataset:/opt/dataset,readonly /storage1/fs1/kamilov/Active/gan.weijie/020_experiment:/opt/experiment"
#export LSF_DOCKER_WORKDIR="/opt/project"
#export LSF_DOCKER_SHM_SIZE=16g
#
#if [ $# -eq 0 ];then
#  echo "no argument or the argument"
#
#else
#
#if [ $1 = general ];then
#
#bsub -g /gan.weijie/lanl_vit -q general -R 'gpuhost rusage[mem=64GB]' -gpu "num=1:gmem=28G" -a 'docker(wjgan95/main:0.05-raytune)' /opt/miniconda3/bin/python /opt/project/main.py
#
#elif [ $1 = interactive ];then
#
#bsub -g /gan.weijie/lanl_vit -Is -q general-interactive -R 'gpuhost rusage[mem=48GB]' -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -a 'docker(wjgan95/main:0.05-raytune)' /opt/miniconda3/bin/python /opt/project/main.py
#
#else
#
#echo  "argument is not ``general`` or ``interactive``"
#
#fi
#
#fi

fi