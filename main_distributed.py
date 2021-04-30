# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import submitit
import argparse
import logging
import pprint
import yaml
import sys

from src.paws_train import main as paws
from src.suncet_train import main as suncet
from src.fine_tune import main as fine_tune
from src.snn_fine_tune import main as snn_fine_tune

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/checkpoint/submitit/')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--device', type=str,
    help='type of GPU to use')
parser.add_argument(
    '--partition', type=str,
    help='partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int,
    help='time in minutes to run job',
    default=5)
parser.add_argument(
    '--sel', type=str,
    help='which script to run',
    choices=[
        'paws_train',
        'suncet_train',
        'fine_tune',
        'snn_fine_tune'
    ])


class Trainer:

    def __init__(self, sel, fname='configs.yaml', load_model=None):
        self.sel = sel
        self.fname = fname
        self.load_model = load_model

    def __call__(self):
        sel = self.sel
        fname = self.fname
        load_model = self.load_model
        logger.info(f'called-params {sel} {fname} {load_model}')

        # -- load script params
        params = None
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            if load_model is not None:
                params['meta']['load_checkpoint'] = load_model
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

        logger.info('Running %s' % sel)
        if sel == 'paws_train':
            return paws(params)
        elif sel == 'suncet_train':
            return suncet(params)
        elif sel == 'fine_tune':
            return fine_tune(params)
        elif sel == 'snn_fine_tune':
            return snn_fine_tune(params)

    def checkpoint(self):
        fb_trainer = Trainer(self.sel, self.fname, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch():
    executor = submitit.AutoExecutor(folder=args.folder)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_constraint=args.device,
        slurm_comment='comms release April 30',
        slurm_mem='450G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)

    config_fnames = [args.fname]
    if args.batch_launch:
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)

    jobs, trainers = [], []
    with executor.batch():
        for cf in config_fnames:
            fb_trainer = Trainer(args.sel, cf)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
