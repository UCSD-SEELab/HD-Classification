'''
Converts the dataset to FPGA binary input file.
Supports choir (binary), csv, and pickle format.
For choir, we have two files (a train file and a test file), each of which includes both X and y.
For csv, we have four files (separate csv files for train and test X and y).
We also have the datasets in pickle format, where a single pickle file contains both train and test X and y data.
To simplify reading data, we assume choir dataset is saved as isolet_train.choir_dat and isolet_test.choir_dat,
csv is saved as isolet_trainX.csv, isolet_trainY.csv, isolet_testX.csv, and isolet_testY.csv,
and eventually, pickle as named as isolet.pickle
choir data format has the smallest size (similar to our .bin output) but is difficult to handle.
CSV, on the other hand, is pretty large but we can manipulate that easily (e.g. to create test cases of different size).
The outputs will be saved in the same folder of input.
'''

import struct
import numpy as np
import argparse
import time
import pickle

def readChoirDat(filename):
	""" Parse a choir_dat file """
	with open(filename, 'rb') as f:
		nFeatures = struct.unpack('i', f.read(4))[0]
		nClasses = struct.unpack('i', f.read(4))[0]
		
		X = []
		y = []

		while True:
			newDP = []
			for i in range(nFeatures):
				v_in_bytes = f.read(4)
				if v_in_bytes is None or len(v_in_bytes) == 0:
					return X, y

				v = struct.unpack('f', v_in_bytes)[0]
				newDP.append(v)

			l = struct.unpack('i', f.read(4))[0]
			X.append(newDP)
			y.append(l)
	return X, y


def convert_bin(format_, X_train_path='', y_train_path='', X_test_path='', y_test_path=''):

	pad_ = 8

	if format_ == 'csv':
		X_train = np.loadtxt(open(X_train_path, 'r'), delimiter=',', skiprows=0)
		y_train = np.loadtxt(open(y_train_path, 'r'), delimiter=',', skiprows=0)
		X_test = np.loadtxt(open(X_test_path, 'r'), delimiter=',', skiprows=0)
		y_test = np.loadtxt(open(y_test_path, 'r'), delimiter=',', skiprows=0)
		X_train_out = X_train_path.replace('.csv', '.bin')
		y_train_out = y_train_path.replace('.csv', '.bin')
		X_test_out = X_test_path.replace('.csv', '.bin')
		y_test_out = y_test_path.replace('.csv', '.bin')
	elif format_ == 'choir':
		X_train, y_train = readChoirDat(X_train_path)
		X_test, y_test = readChoirDat(X_test_path)
		X_train_out = X_train_path.replace('.choir', '_trainX.bin')
		y_train_out = X_train_path.replace('.choir', '_trainY.bin')
		X_test_out = X_test_path.replace('.choir', '_testX.bin')
		y_test_out = X_test_path.replace('.choir', '_testY.bin')
	elif format_ == 'pickle':
		with open(X_train_path, 'rb') as f:
			data_pack = pickle.load(f, encoding='latin1')	
		X_train, y_train, X_test, y_test = data_pack
		X_train_out = X_train_path.replace('.pickle', '_trainX.bin')
		y_train_out = X_train_path.replace('.pickle', '_trainY.bin')
		X_test_out = X_train_path.replace('.pickle', '_testX.bin')
		y_test_out = X_train_path.replace('.pickle', '_testY.bin')

	#normalize and quantize to 16 bit integers
	max_x = np.max(np.abs(X_train))
	X_train = ((X_train/max_x) * (2**15 - 1)).astype('int32')
	X_test = ((X_test/max_x) * (2**15 - 1)).astype('int32')

	n_sample = len(X_train)
	n_test = len(X_test)
	n_feat = len(X_train[0])

	pad = 0 if (n_feat % pad_ == 0) else pad_ - (n_feat % pad_)
	len_padded = n_feat + pad

	#The number of elements is equal to the number of samples times features per sample (after padding), plus one element to store total numbers.
	X_train_1d = [0] * (len_padded*n_sample + 1)
	for i in range(0, n_sample):
		X_train_1d[1 + i*len_padded : 1 + i*len_padded + n_feat] = list(X_train[i])
	#add the number of items as the first element
	X_train_1d[0] = len(X_train_1d) - 1

	X_test_1d = [0] * (len_padded*n_test + 1)
	for i in range(0, n_test):
		X_test_1d[1 + i*len_padded : 1 + i*len_padded + n_feat] = list(X_test[i])
	X_test_1d[0] = len(X_test_1d) - 1

	y_train_1d = [0] * (len(y_train) + 1)
	y_train_1d[0] = len(y_train)
	y_train_1d[1:] = y_train

	y_test_1d = [0] * (len(y_test) + 1)
	y_test_1d[0] = len(y_test)
	y_test_1d[1:] = y_test

	np.array(X_train_1d).astype('int32').tofile(X_train_out)
	np.array(y_train_1d).astype('int32').tofile(y_train_out)
	np.array(X_test_1d).astype('int32').tofile(X_test_out)
	np.array(y_test_1d).astype('int32').tofile(y_test_out)
