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

def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('run_name', metavar='N', type=str, help='name of run')
    parser.add_argument('gpu_id', metavar='G', type=str, help='which gpu to use')

    parser.add_argument('network_type', metavar='N', type=str,default='AEConv', help='network type: AEConv, ClassConv, LSTM')
    parser.add_argument('--print_network', action='store_true', help='print_network for debugging')
    parser.add_argument('--test', action='store_true', help='test')

    parser.add_argument('--checkpoint_every', type=int, default=10, help='checkpoint every n epochs')
    parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint with same name')
    parser.add_argument('--resume', action='store_true', help='resume from epoch we left off of when loading')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to load')

    parser.add_argument('--epochs', metavar='N', type=int, help='number of epochs to run for', default=100)
    parser.add_argument('--batch_size', metavar='bs', type=int, default=512, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--rmsprop', action='store_true', help='use rmsprop optimizer')
    parser.add_argument('--sgd', action='store_true', help='use sgd optimizer')
    parser.add_argument('--l2_reg', metavar='lr', type=float, help='learning rate', default=0.0)


    network_types={'AEConv':AutoConvNetwork,'ClassConv':ClassConvNetwork,'LinReg' = LinearReg}
    network_type=args[2]

    network_class=network_types[network_type]
    PFPSampler.add_args(parser)
    network_class.add_args(parser)
    args = parser.parse_args(args)
    data_loader = PFPSampler(args, train=not args.test)
    #i think keras uses GPU by default. double check this
    '''if args.gpu_id == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
    else:
        print(bcolors.OKBLUE + 'Using GPU' + str(args.gpu_id) + bcolors.ENDC)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device('cuda')'''


    #have to change the following to not torch for this project nor the wrapper because no classes in
    #keras.

    #probably just call the network function
    print(args.test)

    run_wrapper = Wrapper(args, network_class, device)
    if args.test:
        run_wrapper.test()
    else:
        run_wrapper.train()

if __name__ == "__main__":
    main(sys.argv[1:])



