# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import torch
from src.utils import (
    AllGather,
    AllReduce
)

logger = getLogger()


def init_paws_loss(
    # paws configs
    multicrop=6,
    tau=0.1,
    T=0.25,
    me_max=True,
    # ropaws configs
    ropaws=False,
    prior_tau=3.0,
    prior_pow=1.0,
    label_ratio=5.0,
    s_batch_size=6720,
    u_batch_size=4096,
    rank=0,
    world_size=1,
):
    """
    Make semi-supervised PAWS loss

    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        return softmax(query @ supports.T / tau) @ labels

    def snn_semi(query, supports, labels, n_views):
        """ Semi-supervised density estmation """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        probs = []
        for q in query.chunk(n_views):  # for each view
            p = _snn_semi_each(q, supports, labels)
            probs.append(p)
        probs = torch.cat(probs, dim=0)  # concat over views

        # Step 4: convert p_out probability to uniform
        M, C = probs.shape
        p_in = probs.sum(dim=1)
        unif = torch.ones(M, C, device=probs.device) / C
        probs = probs + (1 - p_in.view(M, 1)) * unif

        p_in = sum(p_in.chunk(n_views)) / n_views  # average over views

        return probs, p_in

    def _snn_semi_each(query, supports, labels):
        query = AllGather.apply(query)
        M = query.size(0)
        N, K = labels.size()

        device = query.device
        arange = lambda n: torch.arange(n, device=device)
        eye = lambda n: torch.eye(n, n, device=device)

        # compute similarity matrix
        s_sims = query @ supports.T  # cosine sims MxN
        u_sims = query @ query.T  # cosine sims MxM
        u_sims[arange(M), arange(M)] = -float('inf')  # remove self-sims

        # compute in-domain prior
        max_sim = s_sims.max(dim=1)[0]
        prior = ((max_sim - 1) / prior_tau).exp().view(M, 1)

        # compute pseudo-label
        r = label_ratio * u_batch_size / s_batch_size
        s_sims = s_sims + math.log(r) * tau  # upscale labeled batch

        C = softmax(torch.cat([s_sims, u_sims], dim=1) / tau) * prior  # Mx(N+M)
        C_L, C_U = C[:, :N], C[:, N:]  # MxN, MxM
        probs = torch.linalg.inv(eye(M) - C_U) @ C_L @ labels  # MxK

        # get values for this node
        probs = probs[ M//world_size * rank : M//world_size * (rank + 1) ]
        return probs

    def loss(
        anchor_views,
        anchor_supports,
        anchor_support_labels,
        target_views,
        target_supports,
        target_support_labels,
        sharpen=sharpen,
        snn=snn,
        snn_semi=snn_semi,
    ):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        batch_size = len(anchor_views) // (2+multicrop)

        # Step 1: compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            if ropaws:
                targets, p_in = snn_semi(target_views, target_supports, target_support_labels, n_views=2)
            else:
                targets = snn(target_views, target_supports, target_support_labels)
            targets = sharpen(targets)
            if multicrop > 0:
                mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
                targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        if ropaws:
            weight = p_in.repeat(2 + multicrop) ** prior_pow  # weighted loss
            loss = torch.mean(weight * torch.sum(-targets * torch.log(probs), dim=1))
        else:
            loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = AllReduce.apply(torch.mean(sharpen(probs), dim=0))
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))

        return loss, rloss

    return loss


def init_simclr_loss(
    batch_size,
    world_size,
    rank,
    temperature,
    device
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
    offset = rank*2*batch_size
    for i in range(batch_size):
        pos_mask[i, offset + batch_size + i] = 1.
        pos_mask[batch_size + i, offset + i] = 1.
        diag_mask[i, offset + i] = 0.
        diag_mask[batch_size + i, offset + batch_size + i] = 0.

    def contrastive_loss(z):
        # Step 1: normalize embeddings
        z = torch.nn.functional.normalize(z)

        # Step 2: gather embeddings from all workers
        z_buffer = AllGather.apply(z.detach())
        logger.debug(f'{z_buffer.shape}')

        # Step 3: compute similarity between local embeddings and all others
        exp_cs = torch.exp(z @ z_buffer.T / temperature) * diag_mask

        # Step 4: separate positive sample from negatives and compute loss
        pos = torch.sum(exp_cs * pos_mask, dim=1)
        diag = torch.sum(exp_cs, dim=1)
        loss = - torch.sum(torch.log(pos.div(diag))) / (2.*batch_size)

        return loss.squeeze()

    return contrastive_loss


def init_suncet_loss(
    num_classes,
    batch_size,
    world_size,
    rank,
    temperature,
    device,
    unique_classes=False
):
    """
    Make SuNCEt supervised contrastive loss

    NOTE: Assumes data is loaded with data-loaders constrcuted from 'init_data'
          method in data_manager.py

    :param num_classes: num. image classes per batch
    :param batch_size: num. images per class in each batch
    :param world_size: total number of workers in network
    :param rank: local rank in network
    :param temperature: temp. param
    :param device: device to map tensors onto
    :param unique_classes: whether each worker loads the same set of classes
    """
    local_images = batch_size*num_classes
    total_images = local_images*world_size
    diag_mask = torch.ones(local_images, total_images).to(device)
    offset = rank*local_images
    for i in range(local_images):
        diag_mask[i, offset + i] = 0.

    def contrastive_loss(z, labels):

        # Step 1: normalize embeddings
        z = torch.nn.functional.normalize(z)

        # Step 2: gather embeddings from all workers
        z_buffer = AllGather.apply(z)

        # Step 3: compute class predictions
        exp_cs = torch.exp(z @ z_buffer.T / temperature) * diag_mask
        probs = exp_cs.div(exp_cs.sum(dim=1, keepdim=True)) @ labels

        # Step 4: compute loss for predictions
        targets = labels[offset:offset+local_images]
        overlap = probs**(-targets)
        loss = torch.mean(torch.sum(torch.log(overlap), dim=1))
        return loss

    return contrastive_loss


def make_labels_matrix(
    num_classes,
    s_batch_size,
    world_size,
    device,
    unique_classes=False,
    smoothing=0.0
):
    """
    Make one-hot labels matrix for labeled samples

    NOTE: Assumes labeled data is loaded with ClassStratifiedSampler from
          src/data_manager.py
    """

    local_images = s_batch_size*num_classes
    total_images = local_images*world_size

    off_value = smoothing/(num_classes*world_size) if unique_classes else smoothing/num_classes

    if unique_classes:
        labels = torch.zeros(total_images, num_classes*world_size).to(device) + off_value
        for r in range(world_size):
            # -- index range for rank 'r' images
            s1 = r * local_images
            e1 = s1 + local_images
            # -- index offset for rank 'r' classes
            offset = r * num_classes
            for i in range(num_classes):
                labels[s1:e1][i::num_classes][:, offset+i] = 1. - smoothing + off_value
    else:
        labels = torch.zeros(total_images, num_classes*world_size).to(device) + off_value
        for i in range(num_classes):
            labels[i::num_classes][:, i] = 1. - smoothing + off_value

    return labels


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
