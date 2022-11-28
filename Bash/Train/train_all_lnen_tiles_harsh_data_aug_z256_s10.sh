#!/bin/bash
#SBATCH --job-name=BT256h_all
#SBATCH --qos=qos_gpu-t4
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:8
# nombre de taches MPI par noeud
#SBATCH --time=100:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=BarlowTwins_alltiles_z256_hash_dataaug_s10.out          # nom du fichier de sortie
#SBATCH --error=BarlowTwins_alltiles_z256_hash_dataaug_s10.error     
#SBATCH --account uli@v100


module load pytorch-gpu/py3/1.9.0
mkdir /gpfsscratch/rech/uli/ueu39kt/barlowtwins/train_tiles_harsh_dataaug_z256_s10
srun python /linkhome/rech/genkmw01/ueu39kt/barlowtwins/main.py --list-dir /gpfsscratch/rech/uli/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2/train_path_review_tiles_barlow_twins.txt --projector 1024-512-256-256 --batch-size 896 --checkpoint-dir /gpfsscratch/rech/uli/ueu39kt/barlowtwins/train_tiles_harsh_dataaug_z256_s10



# class Transform:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4,
#                                         saturation=0.2, hue=0.5)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2), # like in JLQ
#             GaussianBlur(p=0.0), # False like in JLQ
#             Solarization(p=0.0),
#             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                  std=[0.229, 0.224, 0.225])
#         ])
#         self.transform_prime = transforms.Compose([
#             transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4,
#                                         saturation=0.2, hue=0.5)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#             GaussianBlur(p=0.0),
#             Solarization(p=0.0),
#             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                  std=[0.229, 0.224, 0.225])
#         ])