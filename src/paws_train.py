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
from collections import OrderedDict


import numpy as np


import torch

import src.resnet as resnet
import src.wide_resnet as wide_resnet
from src.utils import (
    gpu_timer,
    init_distributed,
    WarmupCosineSchedule,
    CSVLogger,
    AverageMeter
)
from src.losses import (
    init_paws_loss,
    make_labels_matrix
)
#TODO change before trainning
from src.data_manager_clustervec import (
    init_data,
    make_transforms,
    make_multicrop_transform
)
from src.sgd import SGD
from src.lars import LARS

import apex
from torch.nn.parallel import DistributedDataParallel

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
logger.setLevel(logging.DEBUG)

def main(args,run):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    # -- META
    model_name = args['meta']['model_name']
    output_dim = args['meta']['output_dim']
    load_model = args['meta']['load_checkpoint']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    use_fp16 = args['meta']['use_fp16']
    use_pred_head = args['meta']['use_pred_head']
    device = torch.device(args['meta']['device'])
    torch.cuda.set_device(device)

    # -- CRITERTION
    reg = args['criterion']['me_max']
    supervised_views = args['criterion']['supervised_views']
    classes_per_batch = args['criterion']['classes_per_batch']
    s_batch_size = args['criterion']['supervised_imgs_per_class']
    u_batch_size = args['criterion']['unsupervised_batch_size']
    temperature = args['criterion']['temperature']
    sharpen = args['criterion']['sharpen']

    # -- DATA
    unlabeled_frac = args['data']['unlabeled_frac']
    color_jitter = args['data']['color_jitter_strength']
    normalize = args['data']['normalize']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    dataset_name = args['data']['dataset']
    subset_path = args['data']['subset_path']
    unique_classes = args['data']['unique_classes_per_rank']
    multicrop = args['data']['multicrop']
    label_smoothing = args['data']['label_smoothing']
    data_seed = None
    if 'cifar10' in dataset_name:
        data_seed = args['data']['data_seed']
        crop_scale = (0.75, 1.0) if multicrop > 0 else (0.5, 1.0)
        mc_scale = (0.3, 0.75)
        mc_size = 18
    else:
        crop_scale = (0.14, 1.0) if multicrop > 0 else (0.08, 1.0)
        mc_scale = (0.05, 0.14)
        mc_size = 96

    # -- OPTIMIZATION
    new = args['optimization']['new_opt']
    wd = float(args['optimization']['weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    mom = args['optimization']['momentum']
    nesterov = args['optimization']['nesterov']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    # ----------------------------------------------------------------------- #

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}' + '-best.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'paws-xent-loss'),
                           ('%.5f', 'paws-me_max-reg'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder = init_model(
        device=device,
        model_name=model_name,
        use_pred=use_pred_head,
        output_dim=output_dim)
    if world_size > 1:
        process_group = apex.parallel.create_syncbn_process_group(0)
        encoder = apex.parallel.convert_syncbn_model(encoder, process_group=process_group)

    # -- init losses
    paws = init_paws_loss(
        multicrop=multicrop,
        tau=temperature,
        T=sharpen,
        me_max=reg)
    # -- assume support images are sampled with ClassStratifiedSampler
    labels_matrix = make_labels_matrix(
        num_classes=classes_per_batch,
        s_batch_size=s_batch_size,
        world_size=world_size,
        device=device,
        unique_classes=unique_classes,
        smoothing=label_smoothing)

    # -- make data transforms
    transform, init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=unlabeled_frac,
        training=True,
        split_seed=data_seed,
        crop_scale=crop_scale,
        basic_augmentations=False,
        color_jitter=color_jitter,
        normalize=normalize)
    multicrop_transform = (multicrop, None)
    if multicrop > 0:
        multicrop_transform = make_multicrop_transform(
                dataset_name=dataset_name,
                num_crops=multicrop,
                size=mc_size,
                crop_scale=mc_scale,
                normalize=normalize,
                color_distortion=color_jitter)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler,
     supervised_loader,
     supervised_sampler) = init_data(
         dataset_name=dataset_name,
         transform=transform,
         init_transform=init_transform,
         supervised_views=supervised_views,
         u_batch_size=u_batch_size,
         s_batch_size=s_batch_size,
         unique_classes=unique_classes,
         classes_per_batch=classes_per_batch,
         multicrop_transform=multicrop_transform,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    encoder, optimizer, scheduler = init_opt(
        encoder=encoder,
        weight_decay=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        ref_mom=mom,
        nesterov=nesterov,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs)
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, broadcast_buffers=False)

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, optimizer, start_epoch = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            opt=optimizer,
            scaler=scaler,
            new=new,
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

        loss_meter = AverageMeter()
        ploss_meter = AverageMeter()
        rloss_meter = AverageMeter()
        time_meter = AverageMeter()
        data_meter = AverageMeter()

        for itr, udata in enumerate(unsupervised_loader):

            def load_imgs():
                #from pudb import forked;forked.set_trace()
                # -- unsupervised imgs
                uimgs = [u.to(device, non_blocking=True) for u in udata[:-1]]
                # -- supervised imgs
                global iter_supervised 
                try:
                    sdata = next(iter_supervised)
                except Exception:
                    iter_supervised = iter(supervised_loader)
                    logger.info(f'len.supervised_loader: {len(iter_supervised)}')
                    sdata = next(iter_supervised)
                finally:
                    labels = torch.cat([labels_matrix for _ in range(supervised_views)])
                    simgs = [s.to(device, non_blocking=True) for s in sdata[:-1]]
                # -- concatenate supervised imgs and unsupervised imgs
                imgs = simgs + uimgs
                return imgs, labels
            (imgs, labels), dtime = gpu_timer(load_imgs)
            data_meter.update(dtime)

            def train_step():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    optimizer.zero_grad()

                    # --
                    # h: representations of 'imgs' before head
                    # z: representations of 'imgs' after head
                    # -- If use_pred_head=False, then encoder.pred (prediction
                    #    head) is None, and _forward_head just returns the
                    #    identity, z=h
                    h, z = encoder(imgs, return_before_head=True)

                    # Compute paws loss in full precision
                    with torch.cuda.amp.autocast(enabled=False):

                        # Step 1. convert representations to fp32
                        h, z = h.float(), z.float()

                        # Step 2. determine anchor views/supports and their
                        #         corresponding target views/supports
                        # --
                        num_support = supervised_views * s_batch_size * classes_per_batch
                        # --
                        anchor_supports = z[:num_support]
                        anchor_views = z[num_support:]
                        # --
                        target_supports = h[:num_support].detach()
                        target_views = h[num_support:].detach()
                        target_views = torch.cat([
                            target_views[u_batch_size:2*u_batch_size],
                            target_views[:u_batch_size]], dim=0)

                        # Step 3. compute paws loss with me-max regularization
                        (ploss, me_max) = paws(
                            anchor_views=anchor_views,
                            anchor_supports=anchor_supports,
                            anchor_support_labels=labels,
                            target_views=target_views,
                            target_supports=target_supports,
                            target_support_labels=labels)
                        loss = ploss + me_max

                scaler.scale(loss).backward()
                lr_stats = scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                return (float(loss), float(ploss), float(me_max), lr_stats)
            (loss, ploss, rloss, lr_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            ploss_meter.update(ploss)
            rloss_meter.update(rloss)
            time_meter.update(etime)
            rloss_meter.update(rloss)
            time_meter.update(etime)

            if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                csv_logger.log(epoch + 1, itr,
                               ploss_meter.avg,
                               rloss_meter.avg,
                               time_meter.avg)
                if dataset_name != "test":
                    run.log({
                            'itr':itr,
                            'paws-xent-loss': ploss_meter.avg,
                            'paws-me_max-reg': rloss_meter.avg,
                            'time (ms)': time_meter.avg
                            })
                logger.info('[%d, %5d] loss: %.3f (%.3f %.3f) '
                            '(%d ms; %d ms)'
                            % (epoch + 1, itr,
                               loss_meter.avg,
                               ploss_meter.avg,
                               rloss_meter.avg,
                               time_meter.avg,
                               data_meter.avg))
                if lr_stats is not None:
                    logger.info('[%d, %5d] lr_stats: %.3f (%.2e, %.2e)'
                                % (epoch + 1, itr,
                                   lr_stats.avg,
                                   lr_stats.min,
                                   lr_stats.max))

            assert not np.isnan(loss), 'loss is nan'
            

        # -- logging/checkpointing
        logger.info('avg. loss %.3f' % loss_meter.avg)
        if dataset_name != "test" :
            run.log({
                "epoch":epoch+1,
                "avg loss":loss_meter.avg
                })

        #run.watch(encoder)

        if rank == 0:
            save_dict = {
                'encoder': encoder.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch + 1,
                'unlabel_prob': unlabeled_frac,
                'loss': loss_meter.avg,
                's_batch_size': s_batch_size,
                'u_batch_size': u_batch_size,
                'world_size': world_size,
                'lr': lr,
                'temperature': temperature,
                'amp': scaler.state_dict()
            }
            torch.save(save_dict, latest_path)
            if best_loss is None or best_loss > loss_meter.avg:
                best_loss = loss_meter.avg
                logger.info('updating "best" checkpoint')
                torch.save(save_dict, best_path)
            if (epoch + 1) % checkpoint_freq == 0 \
                    or (epoch + 1) % 10 == 0 and epoch < checkpoint_freq:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))


def load_checkpoint(
    r_path,
    encoder,
    opt,
    scaler,
    new,
    use_fp16=False
):
    checkpoint = torch.load(r_path, map_location='cpu')
    epoch = checkpoint['epoch']
    #from pudb import forked; forked.set_trace()
    # -- loading encoder
    if encoder.model_name == "resnet18":
        encoder.load_state_dict(checkpoint['state_dict'], strict = False)
    else:
        encoder.load_state_dict(checkpoint['encoder'], strict = False)
    logger.info(f'loaded encoder from epoch {epoch}')

    # -- loading optimizer
    if not new:
        opt.load_state_dict(checkpoint['optimizer'])

    if use_fp16:
        scaler.load_state_dict(checkpoint['amp'])
    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, epoch


def init_model(
    device,
    model_name='resnet50',
    use_pred=False,
    output_dim=128
):
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
        ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu2', torch.nn.ReLU(inplace=True)),
        ('fc3', torch.nn.Linear(hidden_dim, output_dim))
    ]))

    # -- prediction head
    encoder.pred = None
    if use_pred:
        mx = 4  # 4x bottleneck prediction head
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim//mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim//mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim//mx, output_dim)
        encoder.pred = torch.nn.Sequential(pred_head)
    encoder.model_name = model_name
    encoder.to(device)
    logger.info(encoder)
    return encoder


def init_opt(
    encoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    ref_mom,
    nesterov,
    warmup,
    num_epochs,
    weight_decay=1e-6,
    final_lr=0.0
):
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
        weight_decay=weight_decay,
        momentum=0.9,
        nesterov=nesterov,
        lr=ref_lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup*iterations_per_epoch,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=num_epochs*iterations_per_epoch)
    optimizer = LARS(optimizer, trust_coefficient=0.001)
    return encoder, optimizer, scheduler


if __name__ == "__main__":
    main()
