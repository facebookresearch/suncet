# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import torch.multiprocessing as mp

import pprint
import yaml

from src.paws_train import main as paws
from src.suncet_train import main as suncet
from src.fine_tune import main as fine_tune
from src.snn_fine_tune import main as snn_fine_tune

from src.utils import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--sel', type=str,
    help='which script to run',
    choices=[
        'paws_train',
        'suncet_train',
        'fine_tune',
        'snn_fine_tune'
    ])


def process_main(rank, sel, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()

    logger.info(f'called-params {sel} {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        if rank == 0:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

    if rank == 0:
        dump = os.path.join(params['logging']['folder'], f'params-{sel}.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running {sel} (rank: {rank}/{world_size})')

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if sel == 'paws_train':
        return paws(params)
    elif sel == 'suncet_train':
        return suncet(params)
    elif sel == 'fine_tune':
        return fine_tune(params)
    elif sel == 'snn_fine_tune':
        return snn_fine_tune(params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.spawn(
        process_main,
        nprocs=num_gpus,
        args=(args.sel, args.fname, num_gpus, args.devices))
