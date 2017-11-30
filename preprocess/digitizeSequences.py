#!/usr/bin/python
# -*- coding: <encoding name> -*-
"""
    File name: digitizeSequences.py
    Author: Mamie Wang
    Date created: 11/29/2017
    Date last modified: 11/29/2017
    Python version: 3.6

    Input: csv file of training and validation data
    Output: numpy memmap file for digitized (acgt -> 0123) training and validation sequences
"""


import os, errno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress irrelevent warning messages
import keras;
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
from keras.constraints import maxnorm;
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import pandas
import time
import pydot
import graphviz
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#import tensorflow as tf



start_time = time.time()

np.random.seed(1234) #sets random seed for consistency
seq_length = 2000 #2kb iput
num_labels = 1    #Our binary final prediction. The only label is the predicted probability that the gene is on or whatever
init_weights_file = None#'Weights/model_weights_MSE.h5' #load in pre-trained weights/parameters Anna gave us here if we're using them. Elsewise, leave as None.
##########################
 
#Train Chromosomes 1 2 3 4 7 8 9 10 11 12 13 14 15 17 19 20 21 22
#Val Chromosomes 16 18
#Test Chromosomes  5 6

train = pandas.read_csv("/data/processed/hsa/hsa_train.csv", sep = ',').values 
val = pandas.read_csv("/data/processed/hsa/hsa_val.csv", sep = ',').values
#test = pandas.read_csv("test.csv", sep = ',').values 

train = np.delete(train, 0, axis = 1)
val = np.delete(val, 0, axis =1)
#test = np.delete(test, 0, axis =1)

np.random.shuffle(train)
np.random.shuffle(val)
#np.random.shuffle(test)

print("Data read in after " + str(time.time()-start_time))

num_val_examples = len(val)
print(str(num_val_examples) + " validation examples")
filename = '/data/processed/hsa/validationDigitized.dat' 
val_nums = np.memmap(filename, dtype=np.int32, mode='w+', shape=(num_val_examples, 2001))

start_time = time.time()
print("convert acgt to 0123 for validation data....")
for i in range(num_val_examples):#len(train)):
	seq_list = list(val[i,0])
	for j in range(seq_length):
		if seq_list[j].lower() == 'a':
			val_nums[i, j] = 0
		elif seq_list[j].lower() == 'c':
			val_nums[i, j] = 1
		elif seq_list[j].lower() == 'g':
			val_nums[i, j] = 2
		elif seq_list[j].lower() == 't':
			val_nums[i, j] = 3
	val_nums[i, 2000] = val[i,1]
print("Val data converted to digits after " + str(time.time()-start_time))
print(val_nums.shape)
del val_nums


num_training_examples = len(train)
print(str(num_training_examples) + " training examples")
filename = '/data/processed/hsa/trainDigitized.dat'
train_nums = np.memmap(filename, dtype=np.int32, mode='w+', shape=(num_training_examples, 2001))
start_time = time.time()
print("convert acgt to 0123 for training data...")
for i in range(num_training_examples):
	seq_list = list(train[i,0])
	for j in range(seq_length):
		if seq_list[j].lower() == 'a':
			train_nums[i, j] = 0
		elif seq_list[j].lower() == 'c':
			train_nums[i, j] = 1
		elif seq_list[j].lower() == 'g':
			train_nums[i, j] = 2
		elif seq_list[j].lower() == 't':
			train_nums[i, j] = 3
	train_nums[i, 2000] = train[i,1]
print("Train data converted after " + str(time.time()-start_time))
print(train_nums.shape)
del train_nums

