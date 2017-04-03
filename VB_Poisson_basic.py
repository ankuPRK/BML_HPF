import os
import sys
import math
import random
import progressbar
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops

from edward.models import Gamma, Poisson, Normal
from scipy.stats import poisson

from scipy.special import digamma


def build_toy_dataset(U, V):
  R = np.dot(U, V)
  return R


def get_indicators(N, M, prob_std=0.7):
  ind = np.random.binomial(1, prob_std, (N, M))
  return ind

def build_Matrix(n1, n2):
	X = np.zeros((n1,n2), dtype=np.int)	
	for i in range(n1):
		k = random.randint(0, int(n2/2))
		indices = random.sample(range(0, n2-1), k)
		for index in indices:
			X[i][index] = random.randint(1,5)

	return X

V = int(sys.argv[1])
D = int(sys.argv[2])
K = int(sys.argv[3])  # number of latent factors

# true latent factors
U_true = build_Matrix(V, K)
V_true = build_Matrix(K, D)

# DATA
X_train = build_toy_dataset(U_true, V_true)
# I_train = get_indicators(N, M)
# I_test = 1 - I_train

epsillon = 0.1

a_u = np.random.rand(V, K)
b_u = np.random.rand(V, K)
a_v = np.random.rand(K, D)
b_v = np.random.rand(K, D)

au_0 = 0.1
bu_0 = 0.2
av_0 = 0.1
bv_0 = 0.2

print a_u

bar = progressbar.ProgressBar()
iteration = 0
for iteration in bar(range(1000)):
	for k in range(0, K):
		s = 0
		for d in range(0, D):
			s += a_v[k][d]/b_v[k][d]

		for v in range(0, V):
			p = 0
			for d in range(0, D):
				if X_train[v][d] > 0 and a_v[k][d] > 0 and b_v[k][d] > 0:
					p += X_train[v][d]*math.exp(digamma(a_v[k][d]) - np.log(b_v[k][d]))
			
			a_u[v][k] = au_0 + p
			b_u[v][k] = bu_0 + s

	for k in range(0, K):
		s = 0
		for v in range(0, V):
			s += a_u[v][k]/b_u[v][k]

		for d in range(0, D):
			a_v[k][d] += av_0
		
			p = 0
			for v in range(0, V):	
				if X_train[v][d] > 0 and a_u[v][k] > 0 and b_u[v][k] > 0:			
					p += X_train[v][d]*math.exp(digamma(a_u[v][k]) - np.log(b_u[v][k]))
			
			a_v[k][d] = av_0 + p
			b_v[k][d] = bv_0 + s
	

print a_u