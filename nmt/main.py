"""
Encoder-Decoder with attention for neural machine translation

"""


import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import argparse
import logging
import time
import os
import mxnet as mx
import numpy as np
import xconfig

np.random.seed(65536)  # make it predictable
mx.random.seed(65535)  # 2333

sys.path.append('.')
sys.path.append('..')

logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=logging.INFO, datefmt='%H:%M:%S')
file_handler = logging.FileHandler(os.path.join(xconfig.log_root, time.strftime("%Y%m%d-%H%M%S") + '.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
# logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["train", "test"], default='train',
    help="The mode to run. In the `train` mode a model is trained."
         " In the `test` mode a trained model is used to translate")
args = parser.parse_args()

logging.info(xconfig.get_config_str())

if __name__ == "__main__":
    if args.mode == 'train':
        logging.info('In train mode.')
        from trainer import train

        train()
    elif args.mode == 'test':
        logging.info('In test mode.')
        from tester import test

        test()
