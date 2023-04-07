# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import sys
import copy

from collections import OrderedDict

import numpy as np

import torch

import src.resnet as resnet
import src.wide_resnet as wide_resnet
from src.utils import (
    init_distributed,
    WarmupCosineSchedule
)
from src.data_manager_clustervec import (
    init_data,
    make_transforms
)
from src.sgd import SGD
from torch.nn.parallel import DistributedDataParallel
from src.lars import LARS

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args,run):
    #run ="AAA"
    # -- META
    model_name = args['meta']['model_name']
    port = args['meta']['master_port']
    load_checkpoint = args['meta']['load_checkpoint']
    training = args['meta']['training']
    copy_data = args['meta']['copy_data']
    use_fp16 = args['meta']['use_fp16']
    device = torch.device(args['meta']['device'])
    torch.cuda.set_device(device)

    # -- DATA
    unlabeled_frac = args['data']['unlabeled_frac']
    normalize = args['data']['normalize']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    dataset_name = args['data']['dataset']
    subset_path = args['data']['subset_path']
    num_classes = args['data']['num_classes']
    data_seed = None
    if 'cifar10' in dataset_name:
        data_seed = args['data']['data_seed']
    crop_scale = (0.5, 1.0) if 'cifar10' in dataset_name else (0.08, 1.0)

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    ref_lr = args['optimization']['lr']
    use_lars = args['optimization']['use_lars']
    zero_init = args['optimization']['zero_init']
    num_epochs = args['optimization']['epochs']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    r_file_enc = args['logging']['pretrain_path']

    # -- log/checkpointing paths
    r_enc_path = os.path.join(folder, r_file_enc)
    w_enc_path = os.path.join(folder, f'{tag}-fine-tune.pth.tar')

    # -- init distributed
    world_size, rank = init_distributed(port)
    logger.info(f'initialized rank/world-size: {rank}/{world_size}')

    # -- optimization/evaluation params
    if training:
        batch_size = 128
    else:
        batch_size = 16
        unlabeled_frac = 0.0
        load_checkpoint = True
        num_epochs = 1

    # -- init loss
    criterion = torch.nn.CrossEntropyLoss()
    
    # -- make train data transforms and data loaders/samples
    transform, init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=unlabeled_frac,
        training=training,
        crop_scale=crop_scale,
        split_seed=data_seed,
        basic_augmentations=True,
        normalize=normalize)
    (data_loader,
     dist_sampler) = init_data(
         dataset_name=dataset_name,
         transform=transform,
         init_transform=init_transform,
         u_batch_size=None,
         s_batch_size=batch_size,
         classes_per_batch=None,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=training,
         copy_data=copy_data)

    ipe = len(data_loader)
    logger.info(f'initialized data-loader (ipe {ipe})')

    # -- make val data transforms and data loaders/samples
    val_transform, val_init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=-1,
        training=True,
        basic_augmentations=True,
        force_center_crop=True,
        normalize=normalize)
    (val_data_loader,
     val_dist_sampler) = init_data(
         dataset_name=dataset_name,
         transform=val_transform,
         init_transform=val_init_transform,
         u_batch_size=None,
         s_batch_size=batch_size,
         classes_per_batch=None,
         world_size=1,
         rank=0,
         root_path=root_path,
         image_folder=image_folder,
         training=False,
         copy_data=copy_data)
    logger.info(f'initialized val data-loader (ipe {len(val_data_loader)})')

    # -- init model and optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    encoder, optimizer, scheduler = init_model(
        device=device,
        device_str=args['meta']['device'],
        num_classes=num_classes,
        training=training,
        use_fp16=use_fp16,
        r_enc_path=r_enc_path,
        iterations_per_epoch=ipe,
        world_size=world_size,
        ref_lr=ref_lr,
        weight_decay=wd,
        use_lars=use_lars,
        zero_init=zero_init,
        num_epochs=num_epochs,
        model_name=model_name)

    best_acc = None
    start_epoch = 0
    # -- load checkpoint
    if not training or load_checkpoint:
        encoder, optimizer, scheduler, start_epoch, best_acc = load_from_path(
            r_path=w_enc_path,
            encoder=encoder,
            opt=optimizer,
            sched=scheduler,
            scaler=scaler,
            device_str=args['meta']['device'],
            use_fp16=use_fp16)
    if not training:
        logger.info('putting model in eval mode')
        encoder.eval()
        logger.info(sum(p.numel() for n, p in encoder.named_parameters()
                        if p.requires_grad and ('fc' not in n)))
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):

        def train_step():
            # -- update distributed-data-loader epoch
            dist_sampler.set_epoch(epoch)
            top1_correct, top5_correct, total = 0, 0, 0
            for i, data in enumerate(data_loader):
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = encoder(inputs)
                    loss = criterion(outputs, labels)
                total += inputs.shape[0]
                top5_correct += float(outputs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
                top1_correct += float(outputs.max(dim=1).indices.eq(labels).sum())
                top1_acc = 100. * top1_correct / total
                top5_acc = 100. * top5_correct / total
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                if i % log_freq == 0:
                    logger.info('[%d, %5d] %.3f%% %.3f%% (loss: %.3f)'
                                % (epoch + 1, i, top1_acc, top5_acc, loss))
                if run is not None: 
                    run.log({
                        "step":i,
                        "train_top1_acc":top1_acc,
                        "train_top5_acc":top5_acc,
                        "loss":loss
                    })
            return 100. * top1_correct / total

        def val_step():
            val_encoder = copy.deepcopy(encoder).eval()
            top1_correct, top5_correct, total = 0, 0, 0
            for i, data in enumerate(val_data_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = val_encoder(inputs)
                total += inputs.shape[0]                
                top5_correct += float(outputs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
                top1_correct += float(outputs.max(dim=1).indices.eq(labels).sum())
                top1_acc = 100. * top1_correct / total
                top5_acc = 100. * top5_correct / total


            logger.info('[%d, %5d] %.3f%%' % (epoch + 1, i, top1_acc))
            return 100. * top1_correct / total, 100. * top5_correct / total

        train_top1 = 0.
        train_top1 = train_step()
        with torch.no_grad():
            val_top1, val_top5 = val_step()

        log_str = 'train:' if training else 'test:'
        logger.info('[%d] (%s: %.3f%%) (val: %.3f%%)'
                    % (epoch + 1, log_str, train_top1, val_top1))
        if run is not None:
            run.log({
                    "epoch":epoch+1,
                    "train_top1":train_top1,
                    "val_top1":val_top1,
                    "val_top5":val_top5
                    })
        # -- logging/checkpointing
        if training and (rank == 0) and ((best_acc is None)
                                         or (best_acc < val_top1)):
            best_acc = val_top1
            save_dict = {
                'encoder': encoder.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'epoch': epoch + 1,
                'unlabel_prob': unlabeled_frac,
                'world_size': world_size,
                'best_top1_acc': best_acc,
                'batch_size': batch_size,
                'lr': ref_lr,
                'amp': scaler.state_dict()
            }
            torch.save(save_dict, w_enc_path)

    return train_top1, val_top1


def load_pretrained(
    r_path,
    encoder,
    device_str
):
    checkpoint = torch.load(r_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    #from pudb import forked; forked.set_trace() 
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                f'path: {r_path}')
    del checkpoint
    return encoder


def load_from_path(
    r_path,
    encoder,
    opt,
    sched,
    scaler,
    device_str,
    use_fp16=False
):
    encoder = load_pretrained(r_path, encoder, device_str)
    checkpoint = torch.load(r_path, map_location=device_str)

    best_acc = None
    if 'best_top1_acc' in checkpoint:
        best_acc = checkpoint['best_top1_acc']

    epoch = checkpoint['epoch']
    if opt is not None:
        if use_fp16:
            scaler.load_state_dict(checkpoint['amp'])
        opt.load_state_dict(checkpoint['opt'])
        sched.load_state_dict(checkpoint['sched'])
        logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, sched, epoch, best_acc


def init_model(
    device,
    device_str,
    num_classes,
    training,
    use_fp16,
    r_enc_path,
    iterations_per_epoch,
    world_size,
    ref_lr,
    num_epochs,
    use_lars=False,
    zero_init=True,
    model_name='resnet50',
    warmup_epochs=0,
    weight_decay=0
):
    # -- init model
    if 'wide_resnet' in model_name:
        encoder = wide_resnet.__dict__[model_name](dropout_rate=0.0)
        hidden_dim = 128
    else:
        encoder = resnet.__dict__[model_name]()
        hidden_dim = 2048
        if 'w2' in model_name:
            hidden_dim *= 2
        elif 'w4' in model_name:
            hidden_dim *= 4

    # -- projection head
    encoder.fc = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu1', torch.nn.ReLU(inplace=True)),
        ('fc2', torch.nn.Linear(hidden_dim, num_classes))
    ]))

    encoder.to(device)
    encoder = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        device_str=device_str)

    if zero_init:
        for p in encoder.fc.fc2.parameters():
            torch.nn.init.zeros_(p)

    # -- init optimizer
    optimizer, scheduler = None, None
    if training:
        param_groups = [
            {'params': (p for n, p in encoder.named_parameters()
                        if ('bias' not in n) and ('bn' not in n))},
            {'params': (p for n, p in encoder.named_parameters()
                        if ('bias' in n) or ('bn' in n)),
             'LARS_exclude': True,
             'weight_decay': 0}
        ]
        optimizer = SGD(
            param_groups,
            nesterov=True,
            weight_decay=weight_decay,
            momentum=0.9,
            lr=ref_lr)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_epochs*iterations_per_epoch,
            start_lr=ref_lr,
            ref_lr=ref_lr,
            T_max=num_epochs*iterations_per_epoch)
        if use_lars:
            optimizer = LARS(optimizer, trust_coefficient=0.001)
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, broadcast_buffers=False)

    return encoder, optimizer, scheduler


if __name__ == "__main__":
    main()
