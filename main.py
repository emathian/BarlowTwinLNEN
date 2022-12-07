# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import hostlist
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='1024-512-256-128', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--checkpoint-dir', default='/gpfsscratch/rech/uli/ueu39kt/barlowtwins/dev_nshape/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--list-dir',  default='TrainTumorNormal.txt', type=str, metavar='C',
                        help='List of files for LNEN dataset')
###############
# Evaluation 
###############
parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Parallelize the training on the data set')
parser.add_argument('--checkpoint_evaluation', default='/gpfsscratch/rech/uli/ueu39kt/barlowtwins/train_tiles_harsh_dataaug_z128/checkpoint_30000.pth', type=Path,
                    metavar='DIR', help='path to checkpoint to evaluate')
parser.add_argument('--projector-dir', default='/gpfsscratch/rech/uli/ueu39kt/barlowtwins/projectors/train_tiles_harsh_dataaug_z128', type=Path,
                    metavar='DIR', help='path to where projectors will be saved')


def main():
    args = parser.parse_args()
    idr_torch_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    idr_world_size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    torch.backends.cudnn.enabled = False

    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)
    

    
    dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_world_size, 
                            rank=idr_torch_rank)
    
    if idr_torch_rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats_eval.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(local_rank)
    
    torch.backends.cudnn.benchmark = True
    gpu = torch.device("cuda")
    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    
    dataset = LNENDataset(args)
    print('Load LNEN data')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % idr_world_size == 0
    per_device_batch_size = args.batch_size // idr_world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=0,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if idr_torch_rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
            if idr_torch_rank == 0 and step % args.save_freq == 0:
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_{epoch}_{step}.pth')
                torch.save(model.module.backbone.state_dict(),
                       args.checkpoint_dir / f'wide_resnet50_{epoch}_{step}.pth')
    if idr_torch_rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'wide_resnet50_final.pth')
        
def evaluate():
    args = parser.parse_args()
    idr_torch_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    idr_world_size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    torch.backends.cudnn.enabled = False

    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)
    
    dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_world_size, 
                            rank=idr_torch_rank)
    if idr_torch_rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    
    torch.cuda.set_device(local_rank)    
    torch.backends.cudnn.benchmark = True
    gpu = torch.device("cuda")

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    print("args.checkpoint_evaluation  ", args.checkpoint_evaluation)
    ckpt = torch.load(args.checkpoint_evaluation ,
                          map_location='cpu')
    optimizer.load_state_dict(ckpt['optimizer'])
    
    dataset = LNENDataset(args)
    print('Load LNEN data')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    assert args.batch_size % idr_world_size == 0
    per_device_batch_size = args.batch_size // idr_world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=0,
        pin_memory=True, sampler=None)

    print('Len loader ', len(loader))
    scaler = torch.cuda.amp.GradScaler()
    with torch.no_grad():
        for step, (y1, y2, path_to_imgs) in enumerate(loader):
            if step % 100 == 0:
                if idr_torch_rank == 0:
                    print('step ', step, 
                      '\n progression ' , (step ) /  len(loader))
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            z1, z2, loss = model.forward(y1, y2)
            if idr_torch_rank == 0:
                write_projectors(args, z1, path_to_imgs)
            
def write_projectors(args, z1, path_to_imgs):
    os.makedirs(os.path.join(args.projector_dir), exist_ok= True)   
    for i in range(len(path_to_imgs)):
        tne_id = path_to_imgs[i].split('/')[-3]
        os.makedirs(os.path.join(args.projector_dir, tne_id), exist_ok= True)
        img_name = path_to_imgs[i].split('/')[-1]
        z1_c = z1[i].squeeze().detach().cpu().numpy()
        np.save(os.path.join(args.projector_dir,tne_id,  img_name.split('.jpg')[0]), 
                z1_c)
        


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.wide_resnet50_2(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        testy = self.backbone(y1)
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)#

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        if not args.evaluate:
            return loss
        else:
            return z1, z2, loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.5)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2), # like in JLQ
            GaussianBlur(p=0.0), # False like in JLQ
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(384, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.5)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    
class Transform_Evaluation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(384, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return y1, y2

class LNENDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.list_file = args.list_dir
        self.evaluate = args.evaluate
        # load dataset
        self.x = self.load_dataset_folder()
        # set transforms
        if not args.evaluate:
            self.transform = Transform()
        else:
            self.transform = Transform_Evaluation()

    def __getitem__(self, idx):
        paths = self.x[idx]
        x = Image.open(paths)
        x1, x2 = self.transform(x)
        if not args.evaluate:
            return x1, x2
        else:
            return x1, x2, paths


    def __len__(self):
        return len(self.x)
    
    def load_dataset_folder(self):
        list_file =  self.list_file
        x = []
        img_dir = os.path.join(list_file)
        with open(img_dir, 'r') as f:
            content =  f.readlines()
        files_list = []
        for l in content:
            l =  l.strip()
            files_list.append(l)
        files_list = sorted(files_list)
        x.extend(files_list)
        return list(x)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        main()
