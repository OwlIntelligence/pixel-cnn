import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting

# self define modules
from configs import config_args, configs
from utils.mask import *

def test_random_masks():
    mgen = RandomRectangleMaskGenerator(8,8)
    masks = mgen.gen(5)
    for m in masks:
        print(m)

test_random_masks()
