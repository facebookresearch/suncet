# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import pprint

from collections import OrderedDict

import numpy as np
import torch

import src.resnet as resnet
import src.wide_resnet as wide_resnet
from src.data_manager_clustervec import (
    init_data,
    make_transforms
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--use-pred', action='store_true',
    help='whether to use a prediction head')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture',
    default='resnet50',
    choices=[
        "resnet18",
        'resnet50',
        'resnet50w2',
        'resnet50w4',
        'wide_resnet28w2'
    ])
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--split-seed', type=float,
    default=152,
    help='seed for labeled data-split')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--unlabeled-frac', type=float,
    default='0.9',
    help='fraction of training data unlabeled')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--root-path', type=str,
    default='/datasets/',
    help='root directory to data')
parser.add_argument(
    '--image-folder', type=str,
    default='imagenet_full_size/061417/',
    help='image directory inside root_path')
parser.add_argument(
    '--dataset-name', type=str,
    default='imagenet_fine_tune',
    help='name of dataset to evaluate on',
    choices=[
        'imagenet_fine_tune',
        'cifar10_fine_tune',
        'clustervec_fine_tune'
    ])
parser.add_argument(
    '--subset-path', type=str,
    default='imagenet_subsets/',
    help='name of dataset to evaluate on',
    choices=[
        'imagenet_subsets/',
        'cifar10_subsets/',
        'clustervec_subsets/',
        "BRANDS_subsets/"
    ])

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(
    pretrained,
    subset_path,
    unlabeled_frac,
    dataset_name,
    root_path,
    image_folder,
    model_name=None,
    use_pred=True,
    normalize=True,
    device_str='cuda:0',
    split_seed=152
):
    device = torch.device(device_str)
    torch.cuda.set_device(device)
    num_classes = 151#TODO chaneg to use the global parameter #1000 if 'imagenet' in dataset_name else 10

    def init_pipe(training):
        # -- make data transforms
        transform, init_transform = make_transforms(
            dataset_name=dataset_name,
            subset_path=subset_path,
            unlabeled_frac=unlabeled_frac if training else 0.,
            training=training,
            split_seed=split_seed,
            basic_augmentations=True,
            force_center_crop=True,
            normalize=normalize)

        # -- init data-loaders/samplers
        (data_loader,
         data_sampler) = init_data(
            dataset_name=dataset_name,
            transform=transform,
            init_transform=init_transform,
            u_batch_size=None,
            s_batch_size=16,
            stratify=False,
            classes_per_batch=None,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=False,
            drop_last=False)

        return transform, init_transform, data_loader, data_sampler

    if type(pretrained) == str:
        encoder = init_model(
            device=device,
            model_name=model_name,
            use_pred=use_pred,
            pretrained=pretrained)
    else:
        encoder = pretrained
    encoder.eval()

    transform, init_transform, data_loader, data_sampler = init_pipe(True)
    embs, labs = make_embeddings(
        device,
        data_loader,
        data_sampler,
        encoder=encoder)

    transform, init_transform, data_loader, data_sampler = init_pipe(False)
    return evaluate_embeddings(
        device,
        data_loader,
        encoder=encoder,
        labs=labs,
        embs=embs,
        num_classes=num_classes,
        temp=0.1)


def evaluate_embeddings(
    device,
    data_loader,
    encoder,
    labs,
    embs,
    num_classes,
    temp=0.1,
):
    ipe = len(data_loader)

    embs = embs.to(device)
    labs = labs.to(device)

    # -- make labels one-hot
    num_classes = num_classes
    labs = labs.long().view(-1, 1)
    labs = torch.full((labs.size()[0], num_classes), 0., device=device).scatter_(1, labs, 1.)

    snn = make_snn(embs, labs, temp)

    logger.info(embs.shape)
    logger.info(labs.shape)
    logger.info(len(data_loader))

    top1_correct, top5_correct, total = 0, 0, 0
    for itr, data in enumerate(data_loader):
        imgs, labels = data[0].to(device), data[1].to(device)

        z = encoder(imgs)
        probs = snn(z)
        total += imgs.shape[0]
        top5_correct += float(probs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
        top1_correct += float(probs.max(dim=1).indices.eq(labels).sum())
        top1_acc = 100. * top1_correct / total
        top5_acc = 100. * top5_correct / total

        if itr % 50 == 0:
            logger.info('[%5d/%d] %.3f%% %.3f%%' % (itr, ipe, top1_acc, top5_acc))

    logger.info(f'top1/top5: {top1_acc}/{top5_acc}')

    return top1_acc, top5_acc


def make_embeddings(
    device,
    data_loader,
    data_sampler,
    encoder
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for itr, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            z = encoder(imgs)
        z_mem.append(z.to('cpu'))
        l_mem.append(labels.to('cpu'))
        if itr % 50 == 0:
            logger.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)

    return z_mem, l_mem


def make_snn(embs, labs, temp=0.1):

    # --Normalize embeddings
    embs = embs.div(embs.norm(dim=1).unsqueeze(1)).detach_()

    softmax = torch.nn.Softmax(dim=1)

    def snn(h, h_train=embs, h_labs=labs):
        # -- normalize embeddings
        h = h.div(h.norm(dim=1).unsqueeze(1))
        return softmax(h @ h_train.T / temp) @ h_labs

    return snn


def load_pretrained(
    encoder,
    pretrained
):
    checkpoint = torch.load(pretrained, map_location='cpu')
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
                f'path: {pretrained}')
    del checkpoint
    return encoder


def init_model(
    device,
    pretrained,
    model_name='resnet50',
    output_dim=None,
    use_pred=True
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
    output_dim = hidden_dim if output_dim is None else output_dim

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

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained)

    return encoder


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    pp.pprint(args)
    args.num_classes = 10 if 'cifar10' in args.dataset_name else 1000
    main(pretrained=args.pretrained,
         subset_path=args.subset_path,
         root_path=args.root_path,
         image_folder=args.image_folder,
         unlabeled_frac=args.unlabeled_frac,
         dataset_name=args.dataset_name,
         model_name=args.model_name,
         use_pred=args.use_pred,
         normalize=args.normalize,
         device_str=args.device,
         split_seed=args.split_seed)
