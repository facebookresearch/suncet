# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging

import pprint
import yaml

from src.paws_train import main as paws
from src.suncet_train import main as suncet
from src.fine_tune import main as fine_tune
from src.snn_fine_tune import main as snn_fine_tune

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--sel', type=str,
    help='which script to run',
    choices=[
        'paws_train',
        'suncet_train',
        'fine_tune',
        'snn_fine_tune'
    ])


def main(sel, fname):
    logger.info(f'called-params {sel} {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.sel, args.fname)
