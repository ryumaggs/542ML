import keras
import tensorflow as tf
import os

import numpy as np

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#get training data here
num_imputs = 13
def shallow(num_inputs,hidden_size):
	model = Sequential()
	model.add(Dense(hidden_size, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal',activation='relu'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def deep(num_inputs,hidden_layers):
	model = Sequential()
	model.add(Dense(hidden_layers[0],input_dim=num_nputs,kernel_initializer='normal',activation='relu'))
	model.add(Dense(hidden_layers[1],kernel_initializer='normal',activation='relu'))
	model.add(Dense(1,kernel_initializer='normal',activation='relu'))
	#compile
	model.compile(loss='mean_squared_error',optimizer='adam')
	return model