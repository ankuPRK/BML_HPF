import os
import sys
import math
import random
import progressbar
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
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

def get_list_of_nonzeros(X):
	V = X.shape[0]
	D = X.shape[1]
	ls_nz_V = []
	ls_nz_D = []
	for i in range(V):
		a = []
		ls_nz_V.append(a)
		for j in range(D):
			if X[i][j] > 0:
				ls_nz_V[i].append(j)
	for j in range(D):
		a = []
		ls_nz_D.append(a)
		for i in range(V):
			if X[i][j] > 0:
				ls_nz_D[j].append(i)
	return ls_nz_V, ls_nz_D

def build_small_dataset():
		# 	D1	D2	D3	D4
		# U1	5	3	-	1
		# U2	4	-	-	1
		# U3	1	1	-	5
		# U4	1	-	-	4
		# U5	-	1	5	4	
	X_train=np.array([5,3,0,1,4,0,0,1,1,1,0,5,1,0,0,4,0,1,5,4]).reshape((5,4))
		# 	D1	D2	D3	D4
		# U1	4.97	2.98	2.18	0.98
		# U2	3.97	2.40	1.97	0.99
		# U3	1.02	0.93	5.32	4.93
		# U4	1.00	0.85	4.59	3.93
		# U5	1.36	1.07	4.89	4.12

	return X_train, 5,4

if __name__ == '__main__':

	V = int(sys.argv[1])
	D = int(sys.argv[2])
	K = int(sys.argv[3])  # number of latent factors

	# true latent factors
	U_true = build_Matrix(V, K)
	V_true = build_Matrix(K, D)

	# DATA
	# X_train = build_toy_dataset(U_true, V_true)
	X_train,V,D = build_small_dataset()
	# print X_train
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

	# print a_u

	ls_nz_V, ls_nz_D = get_list_of_nonzeros(X_train)
	print("list is obtained. Starting iterations...")
	bar = progressbar.ProgressBar()
	iteration = 0
	curr = time()
	for iteration in bar(range(5000)):
		for k in range(0, K):
			#s = 0
			#for d in range(0, D):
			s = np.sum(a_v[k,:]/b_v[k,:])

			for v in range(0, V):
				p = 0
				for d in ls_nz_V[v]:
					p += X_train[v][d]*math.exp(digamma(a_v[k][d]) - np.log(b_v[k][d]))
				
				a_u[v][k] = au_0 + p
			b_u[:,k] = bu_0 + s

		for k in range(0, K):
			# s = 0
			# for v in range(0, V):
			s = np.sum(a_u[:,k]/b_u[:,k])

			for d in range(0, D):
				# a_v[k][d] += av_0
			
				p = 0
				for v in ls_nz_D[d]:	
					p += X_train[v][d]*math.exp(digamma(a_u[v][k]) - np.log(b_u[v][k]))
				
				a_v[k][d] = av_0 + p
			b_v[k,:] = bv_0 + s
		prev = curr
		curr = time()
#		print("Iteration " + str(iteration) + ": time taken(in s): " + str(curr-prev))
	# print a_u
	# print b_u


#max vals:
	u_s = (a_u - 1.0) / b_u
	v_s = (a_v - 1.0) / b_v
	t = np.dot(u_s,v_s)

	X_new = np.random.poisson(t)
	print X_train
	print X_new

	print (np.sum((X_train - X_new)**2)/(V*D))

#Posterior
	qU = Gamma(alpha=a_u, beta=b_u)
	qV = Gamma(alpha=a_v, beta=b_v)