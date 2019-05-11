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
import dataLoader
import random
from keras.models import load_model
def main(args):
    #defaults
    print(matplotlib.get_backend())
    matplotlib.use('QT5Agg',warn=False,force=True)
    print(matplotlib.get_backend())
    input_size = 11
    hidden_size = 13
    hidden_layers = [13,20,15,6]
    num_channels = 6
    num_timestamp = 60
    network_type= network.shallow
    network_types={'shallow':network.shallow,'deep':network.deep,'conv':network.deep_conv}
    #editable sizes
    d = parse_args(args)
    if "input_size" in d:
        input_size = d["input_size"]
    if "hidden_size" in d:
        hidden_size = d["hidden_size"]
    if "hidden_layers" in d:
        hidden_layers = aast.literal_eval(d["hidden_layers"])
    print(d)
    dL = dataLoader.Dataset(1,"test",'./data/fff_Bw_1min.dat')
    model = None
    if d['network'] == 'shallow':
        model=network_types['shallow'](input_size,hidden_size)
        dL = dataLoader.Dataset(1,"test",'./data/fff_Bw_1min.dat')
    if d['network'] == 'deep':
        model=network_types['deep'](input_size,hidden_layers)
        dL = dataLoader.Dataset(1,"test",'./data/fff_Bw_1min.dat')
    if d['network'] == 'conv':
        model=network_types['conv'](num_timestamp,num_channels,hidden_layers)
        dL = dataLoader.ConvDataset(1,"conv","./data/ConvData.dat")
    #model = network.base_model(13)
    
    #random testing data
    if d['type'] == 'train':
        path = train(model,dL)
    if d['type'] == 'test':
        test('./model.h5',dL)

def parse_args(args):
    d = {}
    for i in range(len(args)):
        cur = args[i]
        cur = args[i].split('=')
        d[cur[0]] = cur[1]
    return d

def train(model,dL):
    model.summary()
    epochs = 1100
    b_size = 10
    time_step = 60
    feature_number = 6

    print_every = 1000
    save_every = 100
    loss = []
    loss_every = 50
    out_of_data = False
    i = 0
    running_loss = 0
    previous_run_loss = 0
    while(True):
        X = np.zeros((b_size,feature_number,time_step))
        Y = np.zeros((b_size,time_step))
        for j in range(b_size):
            X_o,Y_o = dL.__getitem__()
            X[j,:,:] = np.array(X_o)
            Y[j,:] = np.array(Y_o)
        history = model.fit(X,Y,epochs=50,batch_size=b_size,verbose=0)
        previous_run_loss = running_loss
        running_loss += history.history['loss'][0]
        if i%loss_every==0:
            print("---------------------------")
            # pred = model.predict(X,batch_size=10)
            # pred = pred.flatten()
            # correct = Y.flatten()
            # a = range(time_step * b_size)
            # plt.plot(a,pred,'r')
            # plt.plot(a,correct,'b')
            # plt.title("True vs. Predicted")
            # plt.xlabel("Sample")
            # plt.ylabel("BW")
            # plt.show()
            print(i)
            print("adding loss")
            loss.append(running_loss/loss_every)
            print("running_loss: ", running_loss/loss_every)
            print("loss difference: ", running_loss/loss_every - previous_run_loss/loss_every )
            running_loss = 0
        if i%save_every == 0:
            print("attempting to save")
            model.save('model.h5')
            print("saved model to disk")
        i += 1
    print(loss)
    plt.plot(loss)
    plt.show()
    model.save('model.h5')
    print("done training")
    return "model.h5"

def test(model_path,dL):
    model = load_model(model_path)
    b_size = 10
    feature_number = 6
    i = 0
    time_step = 60
    out_of_data = False
    plt1 = []
    plt2 = []
    while(i > -1):
        X = np.zeros((b_size,feature_number,time_step))
        Y = np.zeros((b_size,time_step))
        print(i)
        y_printcopy = np.zeros((b_size,2))
        for j in range(b_size):
            X_o,Y_o = dL.__getitem__()
            X[j,:,:] = np.array(X_o)
            Y[j,:] = np.array(Y_o)
            plt1.extend(Y[j,:])

        pred = model.predict(X,batch_size=10,verbose=0)
        print(pred)
        for q in range(pred.shape[0]):
            plt2.extend(pred[q,:])
        break
        if out_of_data:
            break
        #y_printcopy[:,0] = np.reshape(Y,(10,))

        #y_printcopy[:,1] = np.reshape(pred,(10,))
        # with open('outfile.txt','a') as f:
        #     for line in y_printcopy:
        #         print(line[0],line[1],file=f)
        i += 1
    plt1 = np.array(plt1)
    plt2 = np.array(plt2)
    print(plt1.shape)
    print(plt2.shape)
    a = range(time_step * b_size)
    plt.plot(a,plt1,'r')
    plt.plot(a,plt2,'b')
    plt.title("True vs. Predicted")
    plt.xlabel("Sample")
    plt.ylabel("BW")
    plt.show()
    return
def split_data():
    split = .75
    f_in = open('./data/fff_Bw_1min.dat','r')
    f_out_train = open('./data/fff_Bw_1min_train.dat','a')
    f_out_test = open('./data/fff_Bw_1min_test.dat','a')
    while(True):
        line = f_in.readline()
        print(line)
        if line == None:
            break
        line = line[:-1]
        a = random.randint(1,4)
        if a == 4:
            print(line,file=f_out_test)
        else:
            print(line,file=f_out_train)
    print("done splitting")

if __name__ == "__main__":
    main(sys.argv[1:])
    #split_data()



