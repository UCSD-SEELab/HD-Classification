
#python main.py --path isolet.pickle --d 1000 --alg rp --epoch 20
#python main.py --path isolet.pickle --d 1000 --alg rp-sign --epoch 20
#python main.py --path isolet.pickle --d 1000 --alg idlv --epoch 20 --L 64 
#python main.py --path isolet.pickle --d 1000 --alg perm --epoch 20 --L 64 

import HD
import numpy as np
import sys
import time
from copy import deepcopy
import argparse
import pickle
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--path', action='store', type=str, help='path to pickle dataset', required=True)
parser.add_argument('--d', action='store', dest='D', type=int, default=500, help='number of dimensions (default 500)')
parser.add_argument('--alg', action='store', type=str, default='rp', help='encoding technique (rp, rp-sign, idlv, perm')
parser.add_argument('--epoch', action='store', type=int, default=20, help='number of retraining iterations (default 20)')
parser.add_argument('--lr', '-lr', action='store', type=float, default=1.0, help='learning rate (default 1.0)')
parser.add_argument('--L', action='store', type=int, default=64, help='number of levels (default 64)')

inputs = parser.parse_args()
path = inputs.path
D = inputs.D
alg = inputs.alg
epoch = inputs.epoch
lr = inputs.lr
L = inputs.L

assert alg in ['rp', 'rp-sign', 'idlv', 'perm']

with open(path, 'rb') as f:
	dataset = pickle.load(f, encoding='latin1')	

X_train, y_train, X_test, y_test = deepcopy(dataset)
acc = HD.train(X_train, y_train, X_test, y_test, D, alg, epoch, lr, L)
print('\n')
print(acc)
