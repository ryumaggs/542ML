import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import network
import ast
def main(args):
    #defaults
    input_size = 13
    hidden_size = 13
    hidden_layers = [13,6]
    network_type= network.shallow
    
    #editable sizes
    if len(args) > 0:
        d = parse_args(args)
        network_types={'shallow':network.shallow,'deep':network.deep}
        network_type = network_types[d['network']]
        if "input_size" in d:
            input_size = d["input_size"]
        if "hidden_size" in d:
            hidden_size = d["hidden_size"]
        if "hidden_layers" in d:
            hidden_layers = aast.literal_eval(d["hidden_layers"])

    model = None
    if d['network'] = 'shallow':
        model=network_types[network_type](input_size,hidden_size)
    if d['network'] = 'deep'
        model=network_types[network_type](input_size,hidden_layers)
    #model = network.base_model(13)
    model.summary()

    #random testing data
    data = np.random.random((50,13))
    labels = np.random.randint(2, size=(50, 1))
    model.fit(data,labels,epochs=10,batch_size=50)

def parse_args(args):
    d = {}
    for i in range(len(args)):
        cur = args[i]
        print(cur)
        cur = args[i].split('=')
        print(cur)
        d[cur[0]] = cur[1]
    return d
        

if __name__ == "__main__":
    main(sys.argv[1:])



