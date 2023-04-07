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
import wandb

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
        'snn_fine_tune',
        'paws_tests',
        'paws_finetune'
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
        
        if not os.path.isdir(params['logging']['folder']):
            os.makedirs(params['logging']['folder'])

        dump = os.path.join(params['logging']['folder'], f'params-{sel}.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))

    # -- make sure all processes correctly initialized torch-distributed
    logger.info(f'Running {sel} (rank: {rank}/{world_size})')

    # -- turn off info-logging for ranks > 0, otherwise too much std output
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if sel == 'paws_train':
        run = wandb.init(project="Paws_train", entity="arbezlo", name=params['logging']['folder'].split(os.sep)[-2], dir=params['logging']['folder'])
        run.config.update(params)
        return paws(params, run)
    if sel == 'paws_tests': 
        run= "AAA"    
        return paws(params,run) #paws(params, run)
    elif sel == 'suncet_train':
        return suncet(params)
    elif sel == 'fine_tune':
        run = wandb.init(project="Paws_resnet_ft", entity="arbezlo", name=params['logging']['folder'].split(os.sep)[-2], dir=params['logging']['folder'])
        run.config.update(params)
        return fine_tune(params,run)
    elif sel == 'snn_fine_tune':
        run = wandb.init(project="Paws_snn_finetune", entity="arbezlo", name=params['logging']['folder'].split(os.sep)[-2], dir=params['logging']['folder'])
        run.config.update(params)
        return snn_fine_tune(params,run)
    elif sel == "paws_finetune":
        run = wandb.init(project="Paws_finetune", entity="arbezlo", name=params['logging']['folder'].split(os.sep)[-2], dir=params['logging']['folder'])
        run.config.update(params)
        return paws(params,run)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.spawn(
        process_main,
        nprocs=num_gpus,
        args=(args.sel, args.fname, num_gpus, args.devices))
