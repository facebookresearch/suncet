# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

from logging import getLogger

import numpy as np
from math import ceil

import torch

import torchvision.transforms as transforms
import torchvision

import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    dataset_name,
    transform,
    init_transform,
    u_batch_size,
    s_batch_size,
    classes_per_batch,
    unique_classes=False,
    multicrop_transform=(0, None),
    supervised_views=1,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    stratify=False,
    drop_last=True
):
    """
    :param dataset_name: ['imagenet', 'cifar10', 'cifar10_fine_tune', 'imagenet_fine_tune']
    :param transform: torchvision transform to apply to each batch of data
    :param init_transform: transform to apply once to all data at the start
    :param u_batch_size: unsupervised batch-size
    :param s_batch_size: supervised batch-size (images per class)
    :param classes_per_batch: num. classes sampled in each supervised batch per gpu
    :param unique_classes: whether each GPU should load different classes
    :param multicrop_transform: number of smaller multi-crop images to return
    :param supervised_views: number of views to generate of each labeled imgs
    :param world_size: number of workers for distributed training
    :param rank: rank of worker in distributed training
    :param root_path: path to the root directory containing all dataset
    :param image_folder: name of folder in 'root_path' containing data to load
    :param training: whether to load training data
    :param stratify: whether to class stratify 'fine_tune' data loaders
    :param copy_data: whether to copy data locally to node at start of training
    """

    if dataset_name == 'imagenet':
        return _init_imgnt_data(
            transform=transform,
            init_transform=init_transform,
            u_batch_size=u_batch_size,
            s_batch_size=s_batch_size,
            classes_per_batch=classes_per_batch,
            unique_classes=unique_classes,
            multicrop_transform=multicrop_transform,
            supervised_views=supervised_views,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=copy_data)

    elif dataset_name == 'imagenet_fine_tune':
        batch_size = s_batch_size
        return _init_imgnt_ft_data(
            transform=transform,
            init_transform=init_transform,
            batch_size=batch_size,
            stratify=stratify,
            classes_per_batch=classes_per_batch,
            unique_classes=unique_classes,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            drop_last=drop_last,
            copy_data=copy_data)

    elif dataset_name == 'cifar10':
        return _init_cifar10_data(
            transform=transform,
            init_transform=init_transform,
            u_batch_size=u_batch_size,
            s_batch_size=s_batch_size,
            classes_per_batch=classes_per_batch,
            multicrop_transform=multicrop_transform,
            supervised_views=supervised_views,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=copy_data)

    elif dataset_name == 'cifar10_fine_tune':
        batch_size = s_batch_size
        return _init_cifar10_ft_data(
            transform=transform,
            init_transform=init_transform,
            supervised_views=supervised_views,
            batch_size=batch_size,
            stratify=stratify,
            classes_per_batch=classes_per_batch,
            unique_classes=unique_classes,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=copy_data)


def _init_cifar10_ft_data(
    transform,
    init_transform,
    batch_size,
    stratify=False,
    classes_per_batch=1,
    unique_classes=False,
    supervised_views=1,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='cifar-pytorch/11222017/',
    training=True,
    copy_data=False,
    drop_last=False
):
    dataset = TransCIFAR10(
        root=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        transform=transform,
        init_transform=init_transform,
        supervised_views=supervised_views,
        train=training,
        supervised=True)

    if not stratify:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=True,
            num_workers=8)
    else:
        dist_sampler = ClassStratifiedSampler(
            data_source=dataset,
            world_size=world_size,
            rank=rank,
            batch_size=batch_size,
            classes_per_batch=classes_per_batch,
            seed=_GLOBAL_SEED,
            unique_classes=unique_classes)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=dist_sampler,
            pin_memory=True,
            num_workers=8)

    return (data_loader, dist_sampler)


def _init_cifar10_data(
    transform,
    init_transform,
    u_batch_size,
    s_batch_size,
    classes_per_batch=10,
    supervised_transform=None,
    multicrop_transform=(0, None),
    supervised_views=1,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='cifar-pytorch/11222017/',
    training=True,
    copy_data=False
):
    unsupervised_set = TransCIFAR10(
        root=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        transform=transform,
        init_transform=init_transform,
        multicrop_transform=multicrop_transform,
        train=training,
        supervised=False)
    unsupervised_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=unsupervised_set,
        num_replicas=world_size,
        rank=rank)
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_set,
        sampler=unsupervised_sampler,
        batch_size=u_batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=8)

    supervised_sampler, supervised_loader = None, None
    if classes_per_batch > 0 and s_batch_size > 0:
        supervised_set = TransCIFAR10(
            root=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            transform=supervised_transform if supervised_transform is not None else transform,
            supervised_views=supervised_views,
            init_transform=init_transform,
            train=True,
            supervised=True)
        supervised_sampler = ClassStratifiedSampler(
            data_source=supervised_set,
            world_size=world_size,
            rank=rank,
            batch_size=s_batch_size,
            classes_per_batch=classes_per_batch,
            seed=_GLOBAL_SEED)
        supervised_loader = torch.utils.data.DataLoader(
            supervised_set,
            batch_sampler=supervised_sampler,
            num_workers=8)
        if len(supervised_loader) > 0:
            tmp = ceil(len(unsupervised_loader) / len(supervised_loader))
            supervised_sampler.set_inner_epochs(tmp)
            logger.debug(f'supervised-reset-period {tmp}')

    return (unsupervised_loader, unsupervised_sampler,
            supervised_loader, supervised_sampler)


def _init_imgnt_ft_data(
    transform,
    init_transform,
    batch_size,
    stratify=False,
    classes_per_batch=1,
    unique_classes=False,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='imagenet_full_size/061417/',
    training=True,
    copy_data=False,
    drop_last=True,
    tar_folder='imagenet_full_size/',
    tar_file='imagenet_full_size-061417.tar',
):
    imagenet = ImageNet(
        root=root_path,
        image_folder=image_folder,
        tar_folder=tar_folder,
        tar_file=tar_file,
        transform=transform,
        train=training,
        copy_data=copy_data)
    logger.info('ImageNet fine-tune dataset created')
    dataset = TransImageNet(
        dataset=imagenet,
        supervised=True,
        init_transform=init_transform,
        seed=_GLOBAL_SEED)

    if not stratify:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=True,
            num_workers=8)
    else:
        dist_sampler = ClassStratifiedSampler(
            data_source=dataset,
            world_size=world_size,
            rank=rank,
            batch_size=batch_size,
            classes_per_batch=classes_per_batch,
            seed=_GLOBAL_SEED,
            unique_classes=unique_classes)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=dist_sampler,
            pin_memory=True,
            num_workers=8)

    return (data_loader, dist_sampler)


def _init_imgnt_data(
    transform,
    init_transform,
    u_batch_size,
    s_batch_size,
    classes_per_batch,
    unique_classes=False,
    multicrop_transform=(0, None),
    supervised_views=1,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='imagenet_full_size/061417/',
    training=True,
    copy_data=False,
    tar_folder='imagenet_full_size/',
    tar_file='imagenet_full_size-061417.tar'
):
    imagenet = ImageNet(
        root=root_path,
        image_folder=image_folder,
        tar_folder=tar_folder,
        tar_file=tar_file,
        transform=transform,
        train=training,
        copy_data=copy_data)
    logger.info('ImageNet dataset created')
    unsupervised_set = TransImageNet(
        dataset=imagenet,
        supervised=False,
        init_transform=init_transform,
        multicrop_transform=multicrop_transform,
        seed=_GLOBAL_SEED)
    unsupervised_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=unsupervised_set,
        num_replicas=world_size,
        rank=rank)
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_set,
        sampler=unsupervised_sampler,
        batch_size=u_batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=8)
    logger.info('ImageNet unsupervised data loader created')

    supervised_sampler, supervised_loader = None, None
    if classes_per_batch > 0 and s_batch_size > 0:
        logger.info('Making supervised ImageNet data loader...')
        supervised_set = TransImageNet(
            dataset=imagenet,
            supervised=True,
            supervised_views=supervised_views,
            init_transform=init_transform,
            seed=_GLOBAL_SEED)
        supervised_sampler = ClassStratifiedSampler(
            data_source=supervised_set,
            world_size=world_size,
            rank=rank,
            batch_size=s_batch_size,
            classes_per_batch=classes_per_batch,
            unique_classes=unique_classes,
            seed=_GLOBAL_SEED)
        supervised_loader = torch.utils.data.DataLoader(
            supervised_set,
            batch_sampler=supervised_sampler,
            pin_memory=True,
            num_workers=8)
        if len(supervised_loader) > 0:
            tmp = ceil(len(unsupervised_loader) / len(supervised_loader))
            supervised_sampler.set_inner_epochs(tmp)
            logger.info(f'supervised-reset-period {tmp}')
        logger.info('ImageNet supervised data loader created')

    return (unsupervised_loader, unsupervised_sampler,
            supervised_loader, supervised_sampler)


def make_transforms(
    dataset_name,
    subset_path=None,
    unlabeled_frac=1.0,
    training=True,
    basic_augmentations=False,
    force_center_crop=False,
    crop_scale=(0.08, 1.0),
    color_jitter=1.0,
    normalize=False,
    split_seed=0
):
    """
    :param dataset_name: ['imagenet', 'cifar10']
    :param subset_path: path to .txt file denoting subset of data to use
    :param unlabeled_frac: fraction of data that is unlabeled
    :param training: whether to load training data
    :param basic_augmentations: whether to use simple data-augmentations
    :param force_center_crop: whether to force use of a center-crop
    :param color_jitter: strength of color-jitter
    :param normalize: whether to normalize color channels
    """

    if 'imagenet' in dataset_name:
        logger.info('making imagenet data transforms')

        # -- file identifying which imagenet labels to keep
        keep_file = None
        if subset_path is not None:
            if unlabeled_frac >= 0:
                keep_file = os.path.join(subset_path, f'{int(unlabeled_frac* 100)}percent.txt')
            else:
                keep_file = os.path.join(subset_path, 'val.txt')
            logger.info(f'keep file: {keep_file}')

        return _make_imgnt_transforms(
            unlabel_prob=unlabeled_frac,
            training=training,
            basic=basic_augmentations,
            force_center_crop=force_center_crop,
            normalize=normalize,
            color_distortion=color_jitter,
            scale=crop_scale,
            keep_file=keep_file)

    elif 'cifar10' in dataset_name:
        logger.info('making cifar10 data transforms')
        keep_file = None
        if subset_path is not None:
            if unlabeled_frac == 0.92:
                keep_file = os.path.join(subset_path, f'spc.4000_split.{split_seed}.txt')
            logger.info(f'keep file: {keep_file}')

        return _make_cifar10_transforms(
            unlabel_prob=unlabeled_frac,
            training=training,
            basic=basic_augmentations,
            force_center_crop=force_center_crop,
            normalize=normalize,
            scale=crop_scale,
            color_distortion=color_jitter,
            keep_file=keep_file)


def _make_cifar10_transforms(
    unlabel_prob,
    training=True,
    basic=False,
    force_center_crop=False,
    normalize=False,
    scale=(0.5, 1.0),
    color_distortion=0.5,
    keep_file=None
):
    """
    Make data transformations

    :param unlabel_prob:probability of sampling unlabeled data point
    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param force_center_crop: whether to override settings and apply center crop to image
    :param normalize: whether to normalize image means and stds
    :param scale: random scaling range for image before resizing
    :param color_distortion: strength of color distortion
    :param keep_file: file containing names of images to use for semisupervised
    """
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        color_distort = transforms.Compose([
            rnd_color_jitter,
            Solarize(p=0.2),
            Equalize(p=0.2)])
        return color_distort

    if training and (not force_center_crop):
        if basic:
            transform = transforms.Compose(
                [transforms.CenterCrop(size=32),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=32, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.CenterCrop(size=32),
             transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2023, 0.1994, 0.2010))])

    def init_transform(targets, samples, keep_file=keep_file, training=training):
        """ Transforms applied to dataset at the start of training """
        new_targets, new_samples = [], []
        if training and (keep_file is not None):
            assert os.path.exists(keep_file), 'keep file does not exist'
            logger.info(f'Using {keep_file}')
            with open(keep_file, 'r') as rfile:
                for line in rfile:
                    indx = int(line.split('\n')[0])
                    new_targets.append(targets[indx])
                    new_samples.append(samples[indx])
        else:
            new_targets, new_samples = targets, samples
        return np.array(new_targets), np.array(new_samples)

    return transform, init_transform


def _make_imgnt_transforms(
    unlabel_prob,
    training=True,
    basic=False,
    force_center_crop=False,
    normalize=False,
    scale=(0.08, 1.0),
    color_distortion=1.0,
    keep_file=None
):
    """
    Make data transformations

    :param unlabel_prob: probability of sampling unlabeled data point
    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param force_center_crop: whether to override settings and apply center crop to image
    :param normalize: whether to normalize image means and stds
    :param scale: random scaling range for image before resizing
    :param color_distortion: strength of color distortion
    :param keep_file: file containing names of images to use for semisupervised
    """
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    logger.debug(f'uprob: {unlabel_prob}\t training: {training}\t basic: {basic}\t normalize: {normalize}\t color_distortion: {color_distortion}')
    if training and (not force_center_crop):
        if basic:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            logger.debug('making training (non-basic) transforms')
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 GaussianBlur(p=0.5),
                 transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))])

    def init_transform(root, samples, class_to_idx, seed,
                       keep_file=keep_file,
                       training=training):
        """ Transforms applied to dataset at the start of training """

        new_targets, new_samples = [], []
        if training and (keep_file is not None) and os.path.exists(keep_file):
            logger.info(f'Using {keep_file}')
            with open(keep_file, 'r') as rfile:
                for line in rfile:
                    class_name = line.split('_')[0]
                    target = class_to_idx[class_name]
                    img = line.split('\n')[0]
                    new_samples.append(
                        (os.path.join(root, class_name, img),
                         target))
                    new_targets.append(target)
        else:
            logger.info('flipping coin to keep labels')
            g = torch.Generator()
            g.manual_seed(seed)
            for sample in samples:
                if torch.bernoulli(torch.tensor(unlabel_prob), generator=g) == 0:
                    target = sample[1]
                    new_samples.append((sample[0], target))
                    new_targets.append(target)

        return np.array(new_targets), np.array(new_samples)

    return transform, init_transform


def make_multicrop_transform(
    dataset_name,
    num_crops,
    size,
    crop_scale,
    normalize,
    color_distortion
):
    if 'imagenet' in dataset_name:
        return _make_multicrop_imgnt_transforms(
            num_crops=num_crops,
            size=size,
            scale=crop_scale,
            normalize=normalize,
            color_distortion=color_distortion)
    elif 'cifar10' in dataset_name:
        return _make_multicrop_cifar10_transforms(
            num_crops=num_crops,
            size=size,
            scale=crop_scale,
            normalize=normalize,
            color_distortion=color_distortion)


def _make_multicrop_cifar10_transforms(
    num_crops,
    size=18,
    scale=(0.3, 0.75),
    normalize=False,
    color_distortion=0.5
):

    def get_color_distortion(s=1.0):
        print('_make_multicrop_cifar10_transforms distortion strength', s)
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        color_distort = transforms.Compose([
            rnd_color_jitter,
            Solarize(p=0.2),
            Equalize(p=0.2)])
        return color_distort

    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=size, scale=scale),
         transforms.RandomHorizontalFlip(),
         get_color_distortion(s=color_distortion),
         transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2023, 0.1994, 0.2010))])

    return (num_crops, transform)


def _make_multicrop_imgnt_transforms(
    num_crops,
    size=96,
    scale=(0.05, 0.14),
    normalize=False,
    color_distortion=1.0,
):
    def get_color_distortion(s=1.0):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    logger.debug('making multicrop transforms')
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=size, scale=scale),
         transforms.RandomHorizontalFlip(),
         get_color_distortion(s=color_distortion),
         GaussianBlur(p=0.5),
         transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))])

    return (num_crops, transform)


class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        data_source,
        world_size,
        rank,
        batch_size=1,
        classes_per_batch=10,
        epochs=1,
        seed=0,
        unique_classes=False
    ):
        """
        ClassStratifiedSampler

        Batch-sampler that samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Sampler, samples images WITH REPLACEMENT (i.e., not epoch-based)

        :param data_source: dataset of type "TransImageNet" or "TransCIFAR10'
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        :param unique_classes: true ==> each worker samples a distinct set of classes; false ==> all workers sample the same classes
        """
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.unique_cpb = unique_classes
        self.batch_size = batch_size
        self.num_classes = len(data_source.classes)
        self.epochs = epochs
        self.outer_epoch = 0

        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

    def set_epoch(self, epoch):
        self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.data_source.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank*i_size:(self.rank+1)*i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank*self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_folder='imagenet_full_size/',
        tar_file='imagenet_full_size-061417.tar',
        train=True,
        transform=None,
        target_transform=None,
        job_id=None,
        local_rank=None,
        copy_data=True
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param transform: data-augmentations (applied in data-loader)
        :param target_transform: target-transform to apply in data-loader
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_folder=tar_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(
            root=data_path,
            transform=transform,
            target_transform=target_transform)
        logger.info('Initialized ImageNet')


class TransImageNet(ImageNet):

    def __init__(
        self,
        dataset,
        supervised=False,
        supervised_views=1,
        init_transform=None,
        multicrop_transform=(0, None),
        seed=0
    ):
        """
        TransImageNet

        Dataset that can apply transforms to images on initialization and can
        return multiple transformed copies of the same image in each call
        to __getitem__
        """
        self.dataset = dataset
        self.supervised = supervised
        self.supervised_views = supervised_views
        self.multicrop_transform = multicrop_transform

        self.targets, self.samples = dataset.targets, dataset.samples
        if self.supervised:
            self.targets, self.samples = init_transform(
                dataset.root,
                dataset.samples,
                dataset.class_to_idx,
                seed)
            logger.debug(f'num-labeled {len(self.samples)}')
            mint = None
            self.target_indices = []
            for t in range(len(dataset.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.debug(f'min. labeled indices {mint}')

    @property
    def classes(self):
        return self.dataset.classes

    def __getitem__(self, index):
        target = self.targets[index]
        path = self.samples[index][0]
        img = self.dataset.loader(path)

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        if self.dataset.transform is not None:
            if self.supervised:
                return *[self.dataset.transform(img) for _ in range(self.supervised_views)], target
            else:
                img_1 = self.dataset.transform(img)
                img_2 = self.dataset.transform(img)

                multicrop, mc_transform = self.multicrop_transform
                if multicrop > 0 and mc_transform is not None:
                    mc_imgs = [mc_transform(img) for _ in range(int(multicrop))]
                    return img_1, img_2, *mc_imgs, target

                return img_1, img_2, target

        return img, target


class TransCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(
        self,
        root,
        image_folder='cifar-pytorch/11222017/',
        tar_file='cifar-10-python.tar.gz',
        copy_data=False,
        train=True,
        transform=None,
        target_transform=None,
        init_transform=None,
        supervised=True,
        multicrop_transform=(0, None),
        supervised_views=1
    ):
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_cifar10_locally(
                root=root,
                image_folder=image_folder,
                tar_file=tar_file)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder)
        logger.info(f'data-path {data_path}')

        super().__init__(data_path, train, transform, target_transform, False)

        self.supervised_views = supervised_views
        self.multicrop_transform = multicrop_transform
        self.supervised = supervised
        if self.supervised:
            self.targets, self.data = init_transform(self.targets, self.data)
            logger.info(f'num-labeled {len(self.data)}')
            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.info(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:

            if self.supervised:
                return *[self.transform(img) for _ in range(self.supervised_views)], target

            else:
                img_1 = self.transform(img)
                img_2 = self.transform(img)

                multicrop, mc_transform = self.multicrop_transform
                if multicrop > 0 and mc_transform is not None:
                    mc_imgs = [mc_transform(img) for _ in range(int(multicrop))]
                    return img_1, img_2, *mc_imgs, target

                return img_1, img_2, target

        return img, target


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_folder='imagenet_full_size/',
    tar_file='imagenet_full_size-061417.tar',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_folder, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path


def copy_cifar10_locally(
    root,
    image_folder='cifar-pytorch/11222017/',
    tar_file='cifar-10-python.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, image_folder, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = target
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(tmp_sgnl_file):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
