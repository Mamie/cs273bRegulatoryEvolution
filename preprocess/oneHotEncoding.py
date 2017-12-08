#!/usr/bin/python
# -*- coding: <encoding name> -*-
"""
    File name: oneHotEncoding.py
    Author: Mamie Wang
    Date created: 11/29/2017
    Date last modified: 12/07/2017
    Python version: 3.6

    Usage:
	python oneHotEncoding.py "path to the training data" "path to the validation data"
    Input: numpy memmap of the validation and training data (digitized)
    Output: one hot encoded array of validation and training data
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
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('trainDat')
	parser.add_argument('valDat')
	arguments = parser.parse_args()
	trainPath = arguments.trainDat
	valPath = arguments.valDat

	start_time = time.time()
	print('Start reading in files...')

	np.random.seed(1234) #sets random seed for consistency
	seq_length = 2000 #2kb iput
	num_labels = 1    #Our binary final prediction. The only label is the predicted probability that the gene is on or whatever
	with open(os.path.split(valPath)[0] + '/fileSize.txt', 'r') as infile:
		num_val_examples = int(infile.readline().rstrip())
		num_train_examples = int(infile.readline().rstrip())

	print('Reading in validation data')
	val_nums = np.memmap(valPath, dtype=np.int32, mode='r', shape=(num_val_examples, 2001))
	print('Read in ' + str(time.time() - start_time))
	basename = os.path.split(valPath)[0]
	datafile = basename + '/validationOneHotEncoding.dat'
	datafile2 = basename + '/validationLabel.dat'
	X_val = np.memmap(datafile, dtype=np.uint8, mode='w+', shape=(num_val_examples, 1, 2000, 4))
	Y_val = np.memmap(datafile2, dtype=np.uint8, mode='w+', shape=(num_val_examples))
	print('Start one hot encoding on validation data')
	start_time = time.time()
	for i in range(num_val_examples):
		X_val[i, 0, :, :] = to_categorical(val_nums[i, 0:2000], num_classes=4)
		Y_val[i] = val_nums[i, 2000]
	print('Complete one hot encoding in ' + str(time.time() - start_time))
	print(X_val.shape)
	print(Y_val.shape)
	del X_val
	del Y_val
	del val_nums

	start_time = time.time()
	print('Reading in training data')
	train_nums = np.memmap(trainPath, dtype=np.int32, mode='r', shape=(num_train_examples, 2001))
	print('Read in ' + str(time.time() - start_time))
	basename = os.path.split(trainPath)[0]
	datafile = basename + '/trainOneHotEncoding.dat'
	datafile2 = basename + '/trainOneHotLabel.dat'
	X_train = np.memmap(datafile, dtype=np.uint8, mode='w+', shape=(num_train_examples, 1, 2000, 4))
	Y_train = np.memmap(datafile2, dtype=np.uint8, mode='w+', shape=(num_train_examples))
	print('Start one hot encoding on training data')
	start_time = time.time()
	for i in range(num_train_examples):
		X_train[i, 0, :, :] = to_categorical(train_nums[i, 0:2000], num_classes=4)
		Y_train[i] = train_nums[i, 2000]
	print('Complete one hot encoding in ' + str(time.time() - start_time))
	print(X_train.shape)
	print(Y_train.shape)
	del X_train
	del Y_train
	del train_nums


