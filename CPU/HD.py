import numpy as np
import pickle
import sys
import os
import math
from copy import deepcopy
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNN


def binarize(base_matrix):
	return np.where(base_matrix < 0, -1, 1)

def encoding_rp(X_data, base_matrix, signed=False):
	enc_hvs = []
	for i in range(len(X_data)):
		if i % int(len(X_data)/20) == 0:
			sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
			sys.stdout.flush()
		hv = np.matmul(base_matrix, X_data[i])
		if signed:
			hv = binarize(hv)
		enc_hvs.append(hv)
	return enc_hvs

def encoding_idlv(X_data, lvl_hvs, id_hvs, D, bin_len, x_min, L=64):
	enc_hvs = []
	for i in range(len(X_data)):
		if i == int(len(X_data)/1):
			break
		if i % int(len(X_data)/20) == 0:
			sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
			sys.stdout.flush()
		sum_ = np.array([0] * D)
		for j in range(len(X_data[i])):
            # bin_ = min( np.round((X_data[i][j] - x_min)/bin_len), L-1)
			bin_ = min( np.floor((X_data[i][j] - x_min)/bin_len), L-1)
			bin_ = int(bin_)
			sum_ += lvl_hvs[bin_]*id_hvs[j]
		enc_hvs.append(sum_)
	return enc_hvs

def encoding_perm(X_data, lvl_hvs, D, bin_len, x_min, L=64):
	enc_hvs = []
	for i in range(len(X_data)):
		if i % int(len(X_data)/20) == 0:
			sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
			sys.stdout.flush()
		sum_ = np.array([0] * D)
		for j in range(len(X_data[i])):
            # bin_ = min( np.round((X_data[i][j] - x_min)/bin_len), L-1)
			bin_ = min( np.floor((X_data[i][j] - x_min)/bin_len), L-1)
			bin_ = int(bin_)
			sum_ += np.roll(lvl_hvs[bin_], j)
		enc_hvs.append(sum_)
	return enc_hvs

def max_match(class_hvs, enc_hv, class_norms):
		max_score = -np.inf
		max_index = -1
		for i in range(len(class_hvs)):
			score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
			#score = np.matmul(class_hvs[i], enc_hv)
			if score > max_score:
				max_score = score
				max_index = i
		return max_index

def train(X_train, y_train, X_test, y_test, D=500, alg='rp', epoch=20, lr=1.0, L=64):
	
	#randomly select 20% of train data as validation
	permvar = np.arange(0, len(X_train))
	np.random.shuffle(permvar)
	X_train = [X_train[i] for i in permvar]
	y_train = [y_train[i] for i in permvar]
	cnt_vld = int(0.2 * len(X_train))
	X_validation = X_train[0:cnt_vld]
	y_validation = y_train[0:cnt_vld]
	X_train = X_train[cnt_vld:]
	y_train = y_train[cnt_vld:]

	#encodings
	if alg in ['rp', 'rp-sign']:
		#create base matrix
		base_matrix = np.random.rand(D, len(X_train[0]))
		base_matrix = np.where(base_matrix > 0.5, 1, -1)
		base_matrix = np.array(base_matrix, np.int8)
		print('\nEncoding ' + str(len(X_train)) + ' train data')
		train_enc_hvs = encoding_rp(X_train, base_matrix, signed=(alg == 'rp-sign'))
		print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
		validation_enc_hvs = encoding_rp(X_validation, base_matrix, signed=(alg == 'rp-sign'))
	
	elif alg in ['idlv', 'perm']:
		#create level matrix
		lvl_hvs = []
		temp = [-1]*int(D/2) + [1]*int(D/2)
		np.random.shuffle(temp)
		lvl_hvs.append(temp)
		change_list = np.arange(0, D)
		np.random.shuffle(change_list)
		cnt_toChange = int(D/2 / (L-1))
		for i in range(1, L):
			temp = np.array(lvl_hvs[i-1])
			temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
			lvl_hvs.append(list(temp))
		lvl_hvs = np.array(lvl_hvs, dtype=np.int8)
		x_min = min( np.min(X_train), np.min(X_validation) )
		x_max = max( np.max(X_train), np.max(X_validation) )
		bin_len = (x_max - x_min)/float(L)
		
		#need to create id hypervectors if encoding is level-id
		if alg == 'idlv':
			cnt_id = len(X_train[0])
			id_hvs = []
			for i in range(cnt_id):
				temp = [-1]*int(D/2) + [1]*int(D/2)
				np.random.shuffle(temp)
				id_hvs.append(temp)
			id_hvs = np.array(id_hvs, dtype=np.int8)
			print('\nEncoding ' + str(len(X_train)) + ' train data')
			train_enc_hvs = encoding_idlv(X_train, lvl_hvs, id_hvs, D, bin_len, x_min, L)
			print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
			validation_enc_hvs = encoding_idlv(X_validation, lvl_hvs, id_hvs, D, bin_len, x_min, L)
		elif alg == 'perm':
			print('\nEncoding ' + str(len(X_train)) + ' train data')
			train_enc_hvs = encoding_perm(X_train, lvl_hvs, D, bin_len, x_min, L)
			print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
			validation_enc_hvs = encoding_perm(X_validation, lvl_hvs, D, bin_len, x_min, L)
	
	#training, initial model
	class_hvs = [[0.] * D] * (max(y_train) + 1)
	for i in range(len(train_enc_hvs)):
		class_hvs[y_train[i]] += train_enc_hvs[i]
	class_norms = [np.linalg.norm(hv) for hv in class_hvs]
	class_hvs_best = deepcopy(class_hvs)
	class_norms_best = deepcopy(class_norms)
	#retraining
	if epoch > 0:
		acc_max = -np.inf
		print('\n\n' + str(epoch) + ' retraining epochs')
		for i in range(epoch):
			sys.stdout.write('epoch ' + str(i) + ': ')
			sys.stdout.flush()
			#shuffle data during retraining
			pickList = np.arange(0, len(train_enc_hvs))
			np.random.shuffle(pickList)
			for j in pickList:
				predict = max_match(class_hvs, train_enc_hvs[j], class_norms)
				if predict != y_train[j]:
					class_hvs[predict] -= np.multiply(lr, train_enc_hvs[j])
					class_hvs[y_train[j]] += np.multiply(lr, train_enc_hvs[j])
			class_norms = [np.linalg.norm(hv) for hv in class_hvs]
			correct = 0
			for j in range(len(validation_enc_hvs)):
				predict = max_match(class_hvs, validation_enc_hvs[j], class_norms)
				if predict == y_validation[j]:
					correct += 1
			acc = float(correct)/len(validation_enc_hvs)
			sys.stdout.write("%.4f " %acc)
			sys.stdout.flush()
			if i > 0 and i%5 == 0:
				print('')
			if acc > acc_max:
				acc_max = acc
				class_hvs_best = deepcopy(class_hvs)
				class_norms_best = deepcopy(class_norms)
	
	del X_train
	del X_validation
	del train_enc_hvs
	del validation_enc_hvs

	print('\n\nEncoding ' + str(len(X_test)) + ' test data')
	if alg == 'rp' or alg == 'rp-sign':
		test_enc_hvs = encoding_rp(X_test, base_matrix, signed=(alg == 'rp-sign'))
	elif alg == 'idlv':
		test_enc_hvs = encoding_idlv(X_test, lvl_hvs, id_hvs, D, bin_len, x_min, L)
	elif alg == 'perm':
			test_enc_hvs = encoding_perm(X_test, lvl_hvs, D, bin_len, x_min, L)
	correct = 0
	for i in range(len(test_enc_hvs)):
		predict = max_match(class_hvs_best, test_enc_hvs[i], class_norms_best)
		if predict == y_test[i]:
			correct += 1
	acc = float(correct)/len(test_enc_hvs)
	return acc
