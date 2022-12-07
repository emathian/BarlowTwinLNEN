#!/bin/bash
#SBATCH --job-name=E_BT256h_clustering
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# nombre de taches MPI par noeud
#SBATCH --time=10:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=BarlowTwins_alltiles_z256_hash_dataaug_eval_train.out          # nom du fichier de sortie
#SBATCH --error=BarlowTwins_alltiles_z256_hash_dataaug_eval_train.error     
#SBATCH --account uli@v100


module load pytorch-gpu/py3/1.9.0
mkdir /gpfsscratch/rech/uli/ueu39kt/barlowtwins/train_tiles_harsh_dataaug_z128_p13_0712
srun python /linkhome/rech/genkmw01/ueu39kt/barlowtwins/main.py --evaluate --list-dir /gpfsscratch/rech/uli/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2/infer_train_tiles_clustering.txt --projector 1024-512-256-128 --checkpoint_evaluation /gpfsscratch/rech/uli/ueu39kt/barlowtwins/train_tiles_harsh_dataaug_z128/checkpoint_40000.pth --projector-dir /gpfsscratch/rech/uli/ueu39kt/barlowtwins/projectors/train_tiles_harsh_dataaug_z128_p13_0712