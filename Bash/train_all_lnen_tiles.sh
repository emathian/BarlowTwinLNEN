#!/bin/bash
#SBATCH --job-name=BarlowTwins
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:2 # 4
# nombre de taches MPI par noeud
#SBATCH --time=00:30:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=BarlowTwinsDev.out          # nom du fichier de sortie
#SBATCH --error=BarlowTwinsDev.error     
#SBATCH --account uli@v100


module load pytorch-gpu/py3/1.9.0
mkdir /gpfsscratch/rech/uli/ueu39kt/barlowtwins/dev
srun python /linkhome/rech/genkmw01/ueu39kt/barlowtwins/main.py --list-dir /gpfsscratch/rech/uli/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2/dev_barlow_twins.txt