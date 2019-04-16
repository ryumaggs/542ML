import keras
import argparse
import sys
import time
import os
import pickle
import numpy as np
import keras
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Logger import SummaryPrint
from torch.autograd import Variable
from torchvision import datasets, transforms
import misc
from misc import bcolors
from DataLoaders import PFPSampler

from Wrapper import Wrapper

from Networks import AutoConvNetwork

import os
import network
def main(args):
    network_types={'AEConv':AutoConvNetwork,'ClassConv':ClassConvNetwork,'LinReg' = LinearReg}
    network_type=args[2]

    network_class=network_types[network_type]


if __name__ == "__main__":
    model = base_model()
    #main(sys.argv[1:])



