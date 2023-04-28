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
import contextlib
import io
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
from src.losses import (
    init_suncet_loss,
    make_labels_matrix
)
from src.data_manager_clustervec import (
    init_data,
    make_transforms
)

from src.sgd import SGD
from src.lars import LARS

from torch.nn.parallel import DistributedDataParallel

from snn_eval import main as val_run

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

    # -- META
    model_name = args['meta']['model_name']
    load_checkpoint = args['meta']['load_checkpoint']
    copy_data = args['meta']['copy_data']
    output_dim = args['meta']['output_dim']
    use_pred_head = args['meta']['use_pred_head']
    use_fp16 = args['meta']['use_fp16']
    device = torch.device(args['meta']['device'])
    torch.cuda.set_device(device)

    # -- DATA
    unlabeled_frac = args['data']['unlabeled_frac']
    label_smoothing = args['data']['label_smoothing']
    normalize = args['data']['normalize']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    dataset_name = args['data']['dataset']
    subset_path = args['data']['subset_path']
    unique_classes = args['data']['unique_classes_per_rank']
    data_seed = args['data']['data_seed']

    # -- CRITERTION
    classes_per_batch = args['criterion']['classes_per_batch']
    supervised_views = args['criterion']['supervised_views']
    batch_size = args['criterion']['supervised_batch_size']
    temperature = args['criterion']['temperature']

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    num_epochs = args['optimization']['epochs']
    use_lars = args['optimization']['use_lars']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    ref_lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    momentum = args['optimization']['momentum']
    nesterov = args['optimization']['nesterov']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    r_file_enc = args['logging']['pretrain_path']

    # -- log/checkpointing paths
    r_enc_path = os.path.join(folder, r_file_enc)
    w_enc_path = os.path.join(folder, f'{tag}-fine-tune-SNN.pth.tar')

    # -- init distributed
    world_size, rank = init_distributed()
    logger.info(f'initialized rank/world-size: {rank}/{world_size}')

    # -- init loss
    suncet = init_suncet_loss(
        num_classes=classes_per_batch,
        batch_size=batch_size*supervised_views,
        world_size=world_size,
        rank=rank,
        temperature=temperature,
        device=device)
    labels_matrix = make_labels_matrix(
        num_classes=classes_per_batch,
        s_batch_size=batch_size,
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
        basic_augmentations=True,
        normalize=normalize)
    (data_loader,
     dist_sampler) = init_data(
         dataset_name=dataset_name,
         transform=transform,
         init_transform=init_transform,
         supervised_views=supervised_views,
         u_batch_size=None,
         stratify=True,
         s_batch_size=batch_size,
         classes_per_batch=classes_per_batch,
         unique_classes=unique_classes,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=True,
         copy_data=copy_data)

    # -- rough estimate of labeled imgs per class used to set the number of
    #    fine-tuning iterations
    imgs_per_class = int(1300*(1.-unlabeled_frac)) if 'imagenet' in dataset_name else int(5000*(1.-unlabeled_frac))
    dist_sampler.set_inner_epochs(imgs_per_class//batch_size)

    ipe = len(data_loader)
    logger.info(f'initialized data-loader (ipe {ipe})')

    # -- init model and optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    encoder, optimizer, scheduler = init_model(
        device=device,
        training=True,
        r_enc_path=r_enc_path,
        iterations_per_epoch=ipe,
        world_size=world_size,
        start_lr=start_lr,
        ref_lr=ref_lr,
        num_epochs=num_epochs,
        output_dim=output_dim,
        model_name=model_name,
        warmup_epochs=warmup,
        use_pred_head=use_pred_head,
        use_fp16=use_fp16,
        wd=wd,
        final_lr=final_lr,
        momentum=momentum,
        nesterov=nesterov,
        use_lars=use_lars)
    #from pudb import forked;forked.set_trace()
    best_acc, val_top1 = None, None
    start_epoch = 0
    # -- load checkpoint
    if load_checkpoint:
        encoder, optimizer, scaler, scheduler, start_epoch, best_acc = load_from_path(
            r_path=w_enc_path,
            encoder=encoder,
            opt=optimizer,
            scaler=scaler,
            sched=scheduler,
            device=device,
            use_fp16=use_fp16,
            ckp=True)

    for epoch in range(start_epoch, num_epochs):

        def train_step():
            # -- update distributed-data-loader epoch
            dist_sampler.set_epoch(epoch)

            #from pudb import forked; forked.set_trace()
            for i, data in enumerate(data_loader):
                imgs = torch.cat([s.to(device) for s in data[:-1]], 0)
                labels = torch.cat([labels_matrix for _ in range(supervised_views)])
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    optimizer.zero_grad()
                    z = encoder(imgs)
                    loss = suncet(z, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                run.log({'loss':loss,"batch":i})
                if i % log_freq == 0:
                    logger.info('[%d, %5d] (loss: %.3f)' % (epoch + 1, i, loss))


        with torch.no_grad():
            with nostdout():
                val_top1, _ = val_run(
                    pretrained=copy.deepcopy(encoder),
                    subset_path=subset_path,
                    unlabeled_frac=unlabeled_frac,
                    dataset_name=dataset_name,
                    root_path=root_path,
                    image_folder=image_folder,
                    use_pred=use_pred_head,
                    normalize=normalize,
                    split_seed=data_seed)
        logger.info('[%d] (val: %.3f%%)' % (epoch + 1, val_top1))
        run.log({"val top1":val_top1, "epoch": epoch+1})
        train_step()

        # -- logging/checkpointing
        if (rank == 0) and ((best_acc is None) or (best_acc < val_top1)):
            best_acc = val_top1
            save_dict = {
                'encoder': encoder.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'epoch': epoch + 1,
                'unlabel_prob': unlabeled_frac,
                'world_size': world_size,
                'batch_size': batch_size,
                'best_top1_acc': best_acc,
                'lr': ref_lr,
                'amp': scaler.state_dict()
            }
            torch.save(save_dict, w_enc_path)

    logger.info('[%d] (best-val: %.3f%%)' % (epoch + 1, best_acc))
    run.log({"best_val":best_acc, "epoch": epoch+1})


def load_pretrained(
    r_path,
    encoder,
    device,
    ckp=False
):
    checkpoint = torch.load(r_path, map_location=device)
    if ckp:
        pretrained_dict = {k: v for k, v in checkpoint['encoder'].items()}
    else:
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
    scaler,
    device,
    use_fp16=False,
    ckp=False
):
    encoder = load_pretrained(r_path, encoder, device, ckp)
    checkpoint = torch.load(r_path, map_location=device)
    epoch = checkpoint['epoch']
    best_acc = None
    if 'best_top1_acc' in checkpoint:
        best_acc = checkpoint['best_top1_acc']
    if opt is not None:
        if use_fp16:
            scaler.load_state_dict(checkpoint['amp'])
        opt.load_state_dict(checkpoint['opt'])
        sched.load_state_dict(checkpoint['sched'])
        logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, scaler, sched, epoch, best_acc


def init_model(
    device,
    training,
    use_fp16,
    r_enc_path,
    iterations_per_epoch,
    world_size,
    start_lr,
    ref_lr,
    num_epochs,
    output_dim=128,
    model_name='resnet50',
    warmup_epochs=0,
    use_pred_head=False,
    use_lars=False,
    wd=1e-6,
    final_lr=0.,
    momentum=0.9,
    nesterov=False
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
    if use_pred_head:
        mx = 4  # 4x bottleneck prediction head
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim//mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim//mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim//mx, output_dim)
        encoder.pred = torch.nn.Sequential(pred_head)

    for m in encoder.modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    encoder.to(device)
    encoder = load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        device=device)

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
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=wd,
            lr=ref_lr)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_epochs*iterations_per_epoch,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=num_epochs*iterations_per_epoch)
        if use_lars:
            optimizer = LARS(optimizer, trust_coefficient=0.001)
    if world_size > 1:
        encoder = DistributedDataParallel(encoder)

    return encoder, optimizer, scheduler


@contextlib.contextmanager
def nostdout():
    logger.disabled = True
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout
    logger.disabled = False


if __name__ == "__main__":
    main()
