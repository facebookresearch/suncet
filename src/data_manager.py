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

from PIL import Image
from PIL import ImageFilter

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    dataset_name,
    transform,
    init_transform,
    u_batch_size,
    s_batch_size,
    cpb,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False
):
    """
    :param dataset_name: ['imagenet', 'cifar10', 'cifar10_fine_tune', 'imagenet_fine_tune']
    :param transform: torchvision transform to apply to each batch of data
    :param init_transform: transform to apply once to all data at the start
    :param u_batch_size: unsupervised batch-size
    :param s_batch_size: supervised batch-size
    :param cpb: num. classes sampled in each supervised batch
    :param world_size: number of workers for distributed training
    :param rank: rank of worker in distributed training
    :param root_path: path to the root directory containing all dataset
    :param image_folder: name of folder in 'root_path' containing data to load
    :param training: whether to load training data
    :param copy_data: whether to copy data locally to node at start of training
    """

    if dataset_name == 'imagenet':
        return _init_imgnt_data(
            transform=transform,
            init_transform=init_transform,
            u_batch_size=u_batch_size,
            s_batch_size=s_batch_size,
            cpb=cpb,
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
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            copy_data=copy_data)

    elif dataset_name == 'cifar10':
        return _init_cifar10_data(
            transform=transform,
            init_transform=init_transform,
            u_batch_size=u_batch_size,
            s_batch_size=s_batch_size,
            cpb=cpb,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training)

    elif dataset_name == 'cifar10_fine_tune':
        batch_size = s_batch_size
        return _init_cifar10_ft_data(
            transform=transform,
            init_transform=init_transform,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=training)


def _init_cifar10_ft_data(
    transform,
    init_transform,
    batch_size,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='cifar-pytorch/11222017/',
    training=True
):
    root = os.path.join(root_path, image_folder)
    dataset = TransCIFAR10(
        root=root,
        transform=transform,
        init_transform=init_transform,
        train=training,
        multi_trans=False)
    distsampler = SupervisedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=distsampler,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=4)

    return (dataloader, distsampler)


def _init_cifar10_data(
    transform,
    init_transform,
    u_batch_size,
    s_batch_size,
    cpb=10,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='cifar-pytorch/11222017/',
    training=True
):
    root = os.path.join(root_path, image_folder)
    unsupervised_set = TransCIFAR10(
        root=root,
        transform=transform,
        init_transform=init_transform,
        train=training,
        multi_trans=True)
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
        num_workers=4)

    supervised_sampler, supervised_loader = None, None
    if cpb > 0 and s_batch_size > 0:
        supervised_set = TransCIFAR10(
            root=root,
            transform=transform,
            init_transform=init_transform,
            train=True,
            multi_trans=False)
        supervised_sampler = ClassStratifiedSampler(
            data_source=supervised_set,
            world_size=world_size,
            rank=rank,
            batch_size=s_batch_size,
            classes_per_batch=cpb,
            seed=_GLOBAL_SEED)
        supervised_loader = torch.utils.data.DataLoader(
            supervised_set,
            batch_sampler=supervised_sampler,
            num_workers=4)
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
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='imagenet_full_size/061417/',
    training=True,
    copy_data=False,
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
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=8)

    return (data_loader, dist_sampler)


def _init_imgnt_data(
    transform,
    init_transform,
    u_batch_size,
    s_batch_size,
    cpb,
    world_size=1,
    rank=0,
    root_path='/datasets/',
    image_folder='imagenet_full_size/061417/',
    training=True,
    copy_data=False,
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
    logger.info('ImageNet dataset created')
    unsupervised_set = TransImageNet(
        dataset=imagenet,
        supervised=False,
        init_transform=init_transform,
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
    if cpb > 0 and s_batch_size > 0:
        logger.info('Making supervised ImageNet data loader...')
        supervised_set = TransImageNet(
            dataset=imagenet,
            supervised=True,
            init_transform=init_transform,
            seed=_GLOBAL_SEED)
        supervised_sampler = ClassStratifiedSampler(
            data_source=supervised_set,
            world_size=world_size,
            rank=rank,
            batch_size=s_batch_size,
            classes_per_batch=cpb,
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
    color_jitter=1.0,
    normalize=False
):
    """
    :param dataset_name: ['imagenet', 'cifar10']
    :param subset_path: path to .txt file denoting subset of data to use
    :param unlabeled_frac: fraction of data that is unlabeled
    :param training: whether to load training data
    :param basic_augmentations: whether to use simple data-augmentations
    :param color_jitter: strength of color-jitter
    :param normalize: whether to normalize color channels
    """

    keep_file = None
    if subset_path is not None:
        keep_file = os.path.join(subset_path, f'{int(unlabeled_frac* 100)}percent.txt')

    if 'imagenet' in dataset_name:
        logger.info('making imagenet data transforms')
        return _make_imgnt_transforms(
            unlabel_prob=unlabeled_frac,
            training=training,
            basic=basic_augmentations,
            normalize=normalize,
            color_distortion=color_jitter,
            keep_file=keep_file)

    elif 'cifar10' in dataset_name:
        logger.info('making cifar10 data transforms')
        return _make_cifar10_transforms(
            unlabel_prob=unlabeled_frac,
            training=training,
            basic=basic_augmentations,
            normalize=normalize,
            color_distortion=color_jitter,
            keep_file=keep_file)


def _make_cifar10_transforms(
    unlabel_prob,
    training=True,
    basic=False,
    normalize=False,
    color_distortion=1.0,
    keep_file=None
):
    """
    Make data transformations

    :param unlabel_prob:probability of sampling unlabeled data point
    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param normalize: whether to normalize image means and stds
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

    if training:
        if basic:
            # -- basic set of transformations to apply to all images
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=32),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()
                 ])
        else:
            # -- basic set of transformations to apply to all images
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=32),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=0.5),
                 transforms.ToTensor()
                 ])
    else:
        transform = transforms.Compose(
            [transforms.CenterCrop(size=32),
             transforms.ToTensor()
             ])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))
             ])

    def init_transform(target, keep_file=None):
        """ Transforms applied to dataset at the start of training """
        # -- "keep_file" not supported yet
        labeled = 0
        if torch.bernoulli(torch.tensor(unlabel_prob)) == 0:
            labeled = 1
        return labeled, target

    return transform, init_transform


def _make_imgnt_transforms(
    unlabel_prob,
    training=True,
    basic=False,
    normalize=False,
    color_distortion=1.0,
    keep_file=None
):
    """
    Make data transformations

    :param unlabel_prob:probability of sampling unlabeled data point
    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param normalize: whether to normalize image means and stds
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

    def get_gaussian_blur(ks=25):
        def gaussian_blur(img):
            radius_min, radius_max = 0.1, 2.0
            return img.filter(ImageFilter.GaussianBlur(
                radius=np.random.uniform(radius_min, radius_max)))
        t_gaussian_blur = transforms.Lambda(gaussian_blur)
        rnd_gaussian_blur = transforms.RandomApply([t_gaussian_blur], p=0.5)
        return rnd_gaussian_blur

    logger.debug(f'uprob: {unlabel_prob}\t training: {training}\t basic: {basic}\t normalize: {normalize}\t color_distortion: {color_distortion}')
    if not training:
        transform = transforms.Compose(
            [transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor()
             ])
    else:
        if basic:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()
                 ])
        else:
            logger.debug('making training (non-basic) transforms')
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 get_gaussian_blur(ks=23),
                 transforms.ToTensor()
                 ])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))
             ])

    def init_transform(root, samples, class_to_idx,
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
            for sample in samples:
                # -- flip coin to determine labeled samples
                if torch.bernoulli(torch.tensor(unlabel_prob)) == 0:
                    target = sample[1]
                    new_samples.append((sample[0], target))
                    new_targets.append(target)

        return np.array(new_targets), np.array(new_samples)

    return transform, init_transform


class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        data_source,
        world_size,
        rank,
        batch_size=1,
        classes_per_batch=10,
        epochs=1,
        seed=0
    ):
        """
        ClassStratifiedSampler

        Batch-sampler samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Shuffle after each epoch
        Drop last batch

        :param data_source: dataset of type "TransImageNet"
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        """
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.batch_size = batch_size
        self.num_classes = len(data_source.classes)
        self.epochs = epochs
        self.outer_epoch = 0

        num_samples = None
        for t in range(self.num_classes):
            t_indices = data_source.target_indices[t]
            num_samples = len(t_indices) if num_samples is None \
                else min(num_samples, len(t_indices))
        self.num_samples = 0 if num_samples is None else \
            int(num_samples * self.num_classes / self.cpb)

        assert self.num_classes % self.cpb == 0
        # -- subsampled samplers index
        self.seed = seed
        self._ssi = 0
        self._next_perm()

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
        self._ssi = (self._ssi + self.cpb) % self.num_classes
        if self._ssi == 0:
            self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.data_source.target_indices[t])
            i_size = len(t_indices) // self.world_size
            if i_size > 0:
                t_indices = t_indices[self.rank * i_size: (self.rank + 1) * i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(torch.utils.data.SubsetRandomSampler(t_indices))
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
        # -- initialize data samplers
        subsampled_samplers = self._subsample_samplers(self._get_local_samplers(0))

        counter, batch = 0, []
        for epoch in range(self.epochs):

            # -- shuffle data in each epoch and get local samples
            samplers = self._get_local_samplers(epoch)

            for _ in range(self.num_samples):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter = 0
                    batch = []
                    del subsampled_samplers
                    subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0
        return (self.num_samples // self.world_size) // self.batch_size


class SupervisedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(
        self,
        dataset,
        num_replicas,
        rank,
        epochs=1
    ):
        super(SupervisedSampler, self).__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank)
        self.epochs = epochs

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def __iter__(self):

        start_epoch = self.epochs * self.epoch
        end_epoch = start_epoch + self.epochs

        all_indices = []
        for e in range(start_epoch, end_epoch):
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(e)
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            all_indices += indices

        return iter(all_indices)


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
                local_rank=local_rank
            )
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(
            root=data_path,
            transform=transform,
            target_transform=target_transform)
        logger.info(f'Initialized ImageNet')


class TransImageNet(ImageNet):

    def __init__(
        self,
        dataset,
        supervised=False,
        init_transform=None,
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
        self.targets, self.samples = dataset.targets, dataset.samples
        if self.supervised:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.targets, self.samples = init_transform(
                dataset.root,
                dataset.samples,
                dataset.class_to_idx)
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

        if (self.dataset.transform is not None) and (not self.supervised):
            img_1 = self.dataset.transform(img)
            img_2 = self.dataset.transform(img)
            return img_1, img_2, target

        if self.dataset.transform is not None:
            img = self.dataset.transform(img)

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


class TransCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False,
                 init_transform=None, multi_trans=False,
                 seed=0):
        super().__init__(
            root, train, transform, target_transform, download)

        self.multi_trans = multi_trans
        if not self.multi_trans:
            torch.manual_seed(seed)
            np.random.seed(seed)
            modified = np.array(list(map(init_transform, self.targets)))
            self.l_indices = modified[:, 0]
            self.targets = modified[:, 1]
            logger.info('num-labeledi {len(self.targets[self.l_indices == 1])}')

            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    np.logical_and((self.targets == t),
                                   (self.l_indices == 1)))).tolist()
                self.target_indices.append(indices)
                logger.info(f'num-labeled target {t} {len(indices)}')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None and self.multi_trans:
            img_1 = self.transform(img)
            img_2 = self.transform(img)
            return img_1, img_2, target

        if self.transform is not None:
            img = self.transform(img)

        return img, target
