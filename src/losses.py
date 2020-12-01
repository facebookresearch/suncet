# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

logger = getLogger()


def init_ntxent_loss(
    batch_size,
    world_size,
    rank,
    temperature,
    device,
    gather_tensors=True
):
    """
    Make NTXent loss with normalized embeddings and a temperature parameter

    NOTE: Assumes data is loaded with data-loaders constrcuted from 'init_data'
          method in data_manager.py

    :param batch_size: num. local original images per batch
    :param world_size: total number of workers in network
    :param rank: rank in network
    :param temperature: temp. param
    :param device: device to map tensors onto
    :param gather_tensors: whether to all-gather tensors across workers
    """
    total_images = 2*batch_size*world_size
    pos_mask = torch.zeros(2*batch_size, total_images).to(device)
    diag_mask = torch.ones(2*batch_size, total_images).to(device)
    offset = rank*2*batch_size if gather_tensors else 0
    for i in range(batch_size):
        pos_mask[i, offset + batch_size + i] = 1.
        pos_mask[batch_size + i, offset + i] = 1.
        diag_mask[i, offset + i] = 0.
        diag_mask[batch_size + i, offset + batch_size + i] = 0.

    def contrastive_loss(z):
        # Step 1: normalize embeddings
        z = z.div(z.norm(dim=1).unsqueeze(1))

        # Step 2: gather embeddings from all workers
        z_buffer = gather_from_all(z) if gather_tensors else z
        logger.debug(f'{z_buffer.shape}')

        # Step 3: compute similarity between local embeddings and all others
        exp_cs = torch.exp(z @ z_buffer.t() / temperature) * diag_mask

        # Step 4: separate positive sample from negatives and compute loss
        pos = torch.sum(exp_cs * pos_mask, dim=1)
        diag = torch.sum(exp_cs, dim=1)
        loss = - torch.sum(torch.log(pos.div(diag))) / (2.*batch_size)
        return loss

    return contrastive_loss


def init_suncet_loss(
    num_classes,
    batch_size,
    world_size,
    rank,
    temperature,
    device,
    gather_tensors=True
):
    """
    Make SuNCE loss with normalized embeddings and a temperature parameter

    NOTE: Assumes data is loaded with data-loaders constrcuted from 'init_data'
          method in data_manager.py

    :param num_classes: num. image classes per batch
    :param batch_size: num. images per class in each batch
    :param world_size: total number of workers in network
    :param rank: local rank in network
    :param temperature: temp. param
    :param device: device to map tensors onto
    :param gather_tensors: whether to all-gather tensors across workers
    """
    local_images = batch_size*num_classes
    total_images = local_images*world_size
    diag_mask = torch.ones(local_images, total_images).to(device)
    offset = rank*local_images if gather_tensors else 0
    for i in range(local_images):
        diag_mask[i, offset + i] = 0.

    def contrastive_loss(z):

        # Step 1: normalize embeddings
        z = z.div(z.norm(dim=1).unsqueeze(1))

        # Step 2: gather embeddings from all workers
        z_buffer = gather_from_all(z) if gather_tensors else z

        # Step 3: compute similarlity between local embeddings
        exp_cs = torch.exp(z @ z_buffer.t() / temperature) * diag_mask

        # Step 4: compute normalization
        den = torch.sum(exp_cs, dim=1)

        # Step 5: compute loss for each class and accumulate
        loss = torch.zeros(1).to(device)
        for i in range(num_classes):
            pos_cls = torch.sum(exp_cs[i::num_classes, i::num_classes], dim=1)
            den_cls = den[i::num_classes]
            loss += - torch.sum(torch.log(pos_cls.div(den_cls))) / batch_size
        loss /= num_classes
        return loss

    return contrastive_loss


def gather_from_all(tensor):
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def gather_tensors_from_all(tensor):
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    ):
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def convert_to_distributed_tensor(tensor):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = 'cpu' if not tensor.is_cuda else 'gpu'
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor, orig_device):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == 'cpu':
        tensor = tensor.cpu()
    return tensor
