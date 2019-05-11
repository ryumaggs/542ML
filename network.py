import keras
import tensorflow as tf
import os

import numpy as np

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, GlobalAveragePooling1D, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#get training data here
num_imputs = 11
def shallow(num_inputs,hidden_size):
	model = Sequential()
	model.add(Dense(hidden_size, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal',activation='relu'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
	return model

def deep(num_inputs,hidden_layers):
	model = Sequential()
	model.add(Dense(hidden_layers[0],input_dim=num_inputs,kernel_initializer='normal',activation='tanh'))
	model.add(Dense(hidden_layers[1],kernel_initializer='normal',activation='tanh'))
	model.add(Dense(hidden_layers[2],kernel_initializer='normal',activation='tanh'))
	model.add(Dense(hidden_layers[3],kernel_initializer='normal',activation='tanh'))
	model.add(Dense(1,kernel_initializer='normal',activation='tanh'))
	#compile
	model.compile(loss='mean_squared_error',optimizer='adam')
	return model

def deep_conv(n_channels,n_timesteps, hidden_layers):
	model = Sequential()
	#add model layers
	model.add(Conv1D(100, 2, activation='relu', input_shape=(n_timesteps,n_channels)))
	model.add(Conv1D(120, 2, activation='relu'))
	model.add(Conv1D(140, 2, activation='relu'))
	model.add(Conv1D(140,2, activation='relu'))
	model.add(GlobalAveragePooling1D())
	model.add(Dropout(0.5))
	model.add(Dense(60, activation='relu'))
	model.compile(loss='mean_squared_error',optimizer='adam')
	model.summary()
	return model