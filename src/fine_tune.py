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

import numpy as np

import torch
import torch.optim as optim

import src.resnet as resnet
from src.utils import (
    init_distributed,
    WarmupCosineSchedule
)
from src.data_manager import (
    init_data,
    make_transforms
)
from torch.nn.parallel import DistributedDataParallel
from apex import amp

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


def main(args):

    # -- META
    model_name = args['meta']['model_name']
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

    # - LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    r_file_enc = args['logging']['pretrain_path']

    # -- log/checkpointing paths
    r_enc_path = os.path.join(folder, r_file_enc)
    w_enc_path = os.path.join(folder, f'{tag}-fine-tune.pth.tar')

    # -- init distributed
    world_size, rank = init_distributed()
    logger.info(f'initialized rank/world-size: {rank}/{world_size}')

    # -- optimization/evaluation params
    batch_size, loops, num_epochs, ref_lr, start_lr = None, None, None, None, None
    if training:
        batch_size = 256
        loops = max(1, 16*256 // (world_size*batch_size))
        logger.info(f'loops {loops} (batch-size {loops*batch_size*world_size})')
        num_epochs = 60 if unlabeled_frac == 0.99 else 30
        ref_lr = 0.05 * world_size * loops * batch_size / 256
        start_lr = 0.05
    else:
        unlabeled_frac = 0.0
        load_checkpoint = True
        batch_size = 32
        num_epochs = 1
        loops = 1

    # -- init loss
    criterion = torch.nn.CrossEntropyLoss()

    # -- make data transforms
    transform, init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=unlabeled_frac,
        training=training,
        basic_augmentations=True,
        normalize=normalize)

    # -- init data-loaders/samplers
    (data_loader,
     dist_sampler) = init_data(
         dataset_name=dataset_name,
         transform=transform,
         init_transform=init_transform,
         u_batch_size=None,
         s_batch_size=batch_size,
         cpb=None,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=training,
         copy_data=copy_data)
    ipe = len(data_loader) // loops
    logger.info(f'initialized data-loader (ipe {ipe})')

    # -- init model and optimizer
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
        start_lr=start_lr,
        num_epochs=num_epochs,
        model_name=model_name)

    start_epoch = 0
    # -- load checkpoint
    if not training or load_checkpoint:
        encoder, optimizer, scheduler, start_epoch = load_from_path(
            r_path=w_enc_path,
            encoder=encoder,
            opt=optimizer,
            sched=scheduler,
            device_str=args['meta']['device'],
            use_fp16=use_fp16)

    if not training:
        logger.info('putting model in eval mode')
        encoder.eval()
        start_epoch = 0

    top1_acc_log, top5_acc_log = [], []
    best_acc = None
    for epoch in range(start_epoch, num_epochs):

        # -- update distributed-data-loader epoch
        dist_sampler.set_epoch(epoch)

        top1_correct, top5_correct, total = 0, 0, 0
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = encoder(inputs)

            loss = criterion(outputs, labels)
            total += inputs.shape[0]
            top5_correct += float(outputs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
            top1_correct += float(outputs.max(dim=1).indices.eq(labels).sum())
            top1_acc = 100. * top1_correct / total
            top5_acc = 100. * top5_correct / total

            if training:
                loss /= loops
                if use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if (i+1) % loops == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if i % log_freq == 0:
                logger.info('[%d, %5d] %.3f%% %.3f%% (loss: %.3f)'
                            % (epoch + 1, i, top1_acc, top5_acc, loss))

        top1_acc_log.append(top1_acc)
        top5_acc_log.append(top5_acc)

        # -- logging/checkpointing
        if training and (rank == 0) and ((best_acc is None)
                                         or (best_acc < top1_acc)):
            best_acc = top1_acc
            save_dict = {
                'encoder': encoder.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'epoch': epoch + 1,
                'unlabel_prob': unlabeled_frac,
                'world_size': world_size,
                'best_top1_acc': top1_acc,
                'batch_size': batch_size,
                'lr': ref_lr,
                'amp': amp.state_dict() if use_fp16 else None
            }
            torch.save(save_dict, w_enc_path)

    return top1_acc_log, top5_acc_log


def load_pretrained(
    r_path,
    encoder,
    device_str
):
    checkpoint = torch.load(r_path, map_location=device_str)
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
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
    device_str,
    use_fp16=False
):
    encoder = load_pretrained(r_path, encoder, device_str)
    checkpoint = torch.load(r_path, map_location=device_str)
    epoch = checkpoint['epoch']
    if opt is not None:
        if use_fp16:
            amp.load_state_dict(checkpoint['amp'])
        opt.load_state_dict(checkpoint['opt'])
        sched.load_state_dict(checkpoint['sched'])
        logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, sched, epoch


def init_model(
    device,
    device_str,
    num_classes,
    training,
    use_fp16,
    r_enc_path,
    iterations_per_epoch,
    world_size,
    start_lr,
    ref_lr,
    num_epochs,
    model_name='resnet50',
    warmup_epochs=0
):
    # -- init model
    encoder = resnet.__dict__[model_name]().to(device)
    encoder = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        device_str=device_str
    )
    hidden_dim = 2048
    if 'w2' in model_name:
        hidden_dim *= 2
    elif 'w4' in model_name:
        hidden_dim *= 4
    encoder.fc = torch.nn.Linear(hidden_dim, num_classes).to(device)
    torch.nn.init.zeros_(encoder.fc.weight)

    # -- init optimizer
    optimizer, scheduler = None, None
    if training:
        optimizer = optim.SGD(
            encoder.parameters(),
            nesterov=True,
            lr=ref_lr,
            momentum=0.9)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_epochs*iterations_per_epoch,
            start_lr=start_lr,
            ref_lr=ref_lr,
            T_max=num_epochs*iterations_per_epoch)
        if use_fp16:
            encoder, optimizer = amp.initialize(
                encoder,
                optimizer,
                opt_level='O1')
            logger.info('initialized mixed-precision')
    if world_size > 1:
        encoder = DistributedDataParallel(encoder)

    return encoder, optimizer, scheduler


if __name__ == "__main__":
    main()
