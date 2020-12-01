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
    gpu_timer,
    init_distributed,
    WarmupCosineSchedule,
    CSVLogger
)
from src.losses import (
    init_suncet_loss,
    init_ntxent_loss,
)
from src.data_manager import (
    init_data,
    make_transforms
)


from torch.nn.parallel import DistributedDataParallel
from apex.parallel.LARC import LARC
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

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    # -- META
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint']
    copy_data = args['meta']['copy_data']
    use_fp16 = args['meta']['use_fp16']
    device = torch.device(args['meta']['device'])
    torch.cuda.set_device(device)

    # -- CRITERTION
    cpb = args['criterion']['classes_per_batch']
    s_batch_size = args['criterion']['supervised_batch_size']
    u_batch_size = args['criterion']['unsupervised_batch_size']
    temperature = args['criterion']['temperature']
    gather_tensors = args['criterion']['gather_tensors']

    # -- DATA
    unlabeled_frac = args['data']['unlabeled_frac']
    color_jitter = args['data']['color_jitter']
    normalize = args['data']['normalize']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    dataset_name = args['data']['dataset']
    subset_path = args['data']['subset_path']

    # -- OPTIMIZATION
    num_epochs = args['optimization']['epochs']
    s_epoch = args['optimization']['supervised_epochs']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    r_file = args['logging']['read_checkpoint']
    # ----------------------------------------------------------------------- #

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}' + '-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}' + '-best.pth.tar')

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 's-loss'),
                           ('%.5f', 'u-loss'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder = init_model(device=device, model_name=model_name)
    if world_size > 1:
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)

    # -- init losses
    ntxent = init_ntxent_loss(
        batch_size=u_batch_size,
        world_size=world_size,
        rank=rank,
        temperature=temperature,
        device=device,
        gather_tensors=gather_tensors)
    suncet = init_suncet_loss(
        num_classes=cpb,
        batch_size=s_batch_size,
        world_size=world_size,
        rank=rank,
        temperature=temperature,
        device=device,
        gather_tensors=gather_tensors)

    # -- make data transforms
    transform, init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=unlabeled_frac,
        training=True,
        basic_augmentations=False,
        color_jitter=color_jitter,
        normalize=normalize)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler,
     supervised_loader,
     supervised_sampler) = init_data(
         dataset_name=dataset_name,
         transform=transform,
         init_transform=init_transform,
         u_batch_size=u_batch_size,
         s_batch_size=s_batch_size,
         cpb=cpb,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=True,
         copy_data=copy_data)
    iter_supervised = None
    ipe = len(unsupervised_loader)
    logger.info(f'iterations per epoch: {ipe}')

    # -- init optimizer and scheduler
    encoder, optimizer, scheduler = init_opt(
        encoder=encoder,
        use_fp16=use_fp16,
        start_lr=start_lr,
        ref_lr=lr,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs)
    if world_size > 1:
        encoder = DistributedDataParallel(encoder)

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, optimizer, start_epoch = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            opt=optimizer,
            use_fp16=use_fp16)
        for _ in range(start_epoch):
            for _ in range(ipe):
                scheduler.step()

    # -- TRAINING LOOP
    best_loss = None
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        if supervised_sampler is not None:
            supervised_sampler.set_epoch(epoch)

        # -- whether to use supervised-batch in this epoch
        use_supervised = (supervised_sampler is not None) \
            and (len(supervised_loader) > 0) \
            and (epoch < s_epoch)

        running_loss, u_running_loss, s_running_loss, running_time, count = 0., 0., 0., 0., 0
        for itr, udata in enumerate(unsupervised_loader):

            # -- load data
            imgs = torch.cat([udata[0], udata[1]], dim=0)
            if use_supervised:
                try:
                    sdata = next(iter_supervised)
                except Exception:
                    iter_supervised = iter(supervised_loader)
                    logger.info(f'len.supervised_loader: {len(iter_supervised)}')
                    sdata = next(iter_supervised)
                imgs = torch.cat([imgs, sdata[0]], dim=0)
            imgs = imgs.to(device)

            def train_step():
                optimizer.zero_grad()
                z = encoder(imgs)
                uloss = ntxent(z[:2*u_batch_size])
                sloss = suncet(z[2*u_batch_size:]) if use_supervised else 0.
                loss = sloss + uloss
                if use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                return loss, uloss, sloss
            (loss, uloss, sloss), etime = gpu_timer(train_step)
            running_loss += float(loss)
            u_running_loss += float(uloss)
            s_running_loss += float(sloss)
            running_time += float(etime)
            count += 1

            if itr % log_freq == 0:
                avg_loss = running_loss / count
                avg_sloss = s_running_loss / count
                avg_uloss = u_running_loss / count
                avg_time = running_time / count
                csv_logger.log(epoch + 1, itr, avg_sloss, avg_uloss, avg_time)
                logger.info('[%d, %5d] loss: %.3f (%.3f, %.3f)'
                            % (epoch + 1, itr, avg_loss, avg_sloss, avg_uloss))

        # -- logging/checkpointing
        avg_loss = running_loss / count
        logger.info('avg. loss %.3f' % avg_loss)

        if rank == 0:
            save_dict = {
                'encoder': encoder.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch + 1,
                'unlabel_prob': unlabeled_frac,
                'loss': avg_loss,
                's_batch_size': s_batch_size,
                'u_batch_size': u_batch_size,
                'world_size': world_size,
                'lr': lr,
                'temperature': temperature,
                'amp': amp.state_dict() if use_fp16 else None
            }
            torch.save(save_dict, latest_path)
            if best_loss is None or best_loss > avg_loss:
                best_loss = avg_loss
                logger.info('updating "best" checkpoint')
                torch.save(save_dict, best_path)
            if (epoch + 1) % checkpoint_freq == 0 \
                    or (epoch + 1) % 10 == 0 and epoch < checkpoint_freq:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))


def load_checkpoint(
    r_path,
    encoder,
    opt,
    use_fp16=False
):
    checkpoint = torch.load(r_path)
    epoch = checkpoint['epoch']
    if use_fp16:
        amp.load_state_dict(checkpoint['amp'])
    # -- loading encoder
    encoder.load_state_dict(checkpoint['encoder'])
    logger.info(f'loaded encoder from epoch {epoch}')

    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, epoch


def init_model(device, model_name='resnet50'):
    encoder = resnet.__dict__[model_name]()
    hidden_dim = 2048
    if 'w2' in model_name:
        hidden_dim *= 2
    elif 'w4' in model_name:
        hidden_dim *= 4
    # -- projection head
    encoder.fc = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_dim, 128)
    )
    encoder.to(device)
    return encoder


def init_opt(
    encoder,
    use_fp16,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    num_epochs
):
    optimizer = optim.SGD(
        encoder.parameters(),
        weight_decay=1e-6,
        momentum=0.9,
        lr=ref_lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=10*iterations_per_epoch,
        start_lr=start_lr,
        ref_lr=ref_lr,
        T_max=num_epochs*iterations_per_epoch)
    optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False)
    if use_fp16:
        encoder, optimizer = amp.initialize(encoder, optimizer, opt_level='O1')
        logger.info('initialized mixed-precision')
    return encoder, optimizer, scheduler


if __name__ == "__main__":
    main()
