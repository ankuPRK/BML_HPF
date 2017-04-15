import os
import sys
import math
import random
import progressbar
import edward as ed
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.python.ops import random_ops

from edward.models import Gamma, Poisson, Normal
from scipy.stats import poisson

from scipy.special import digamma
from scipy import special

import pmf


def build_toy_dataset(U, V):
  R = np.dot(U, V)
  return R


def get_indicators(N, M, prob_std=0.7):
  ind = np.random.binomial(1, prob_std, (N, M))
  return ind

def build_Matrix(n1, n2):
	X = np.zeros((n1,n2), dtype=np.int)	
	for i in range(n1):
		k = random.randint(1, int(n2/2))
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
	# X_train = np.array([[0, 1, 5, 0, 4, 0, 0, 5, 3, 0, 0],
	# 					 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 					 [0, 2, 1, 3, 0, 0, 0, 2, 0, 0, 0],
	# 					 [0, 0, 5, 2, 0, 0, 0, 0, 0, 0, 0],
	# 					 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
	# 					 [4, 4, 0, 0, 4, 4, 0, 3, 0, 0, 0],
	# 					 [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0],
	# 					 [5, 0, 0, 0, 0, 3, 0, 1, 4, 0, 0],
	# 					 [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],	
	# 					 [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]]
	# 					)
		# 	D1	D2	D3	D4
		# U1	4.97	2.98	2.18	0.98
		# U2	3.97	2.40	1.97	0.99
		# U3	1.02	0.93	5.32	4.93
		# U4	1.00	0.85	4.59	3.93
		# U5	1.36	1.07	4.89	4.12

	return X_train, 5, 4


def FitUsingLibBatch(X_train, V, D, K):
	pmf_var = pmf.PoissonMF(n_components=K, max_iter=10000, tol=0.0005,
	                 smoothness=100, verbose=False)

	pmf_var.fit(X_train)

	a_u = pmf_var.gamma_b
	b_u = pmf_var.rho_b
	b_u = np.repeat(b_u, D, axis=1)

	a_v = pmf_var.gamma_t
	b_v = pmf_var.rho_t
	b_v = b_v.reshape((1, K))
	b_v = np.repeat(b_v, V, axis=0)

	print a_v
	# print b_v
	# print a_v.shape
	qU = Gamma(alpha=a_u, beta=b_u)
	qV = Gamma(alpha=a_v, beta=b_v)

	qU_sample = qU.sample()
	qV_sample = qV.sample()

	X_new = np.zeros((V,D))
	
	n_sample = 10000
	for i in range(n_sample):
		avg_U = qU_sample.eval()
		avg_V = qV_sample.eval()
		temp = np.dot(avg_V, avg_U)
		X_new += np.random.poisson(temp)
		
	# print X_train
	# print np.round(X_new / n_sample, 0)


def FitUsingOwnImplementation(X_train, V, D, K):
	epsillon = 1.0

	a_u = np.random.rand(V, K) + 0.2
	b_u = np.random.rand(V, K) + 0.2
	a_v = np.random.rand(K, D) + 0.2
	b_v = np.random.rand(K, D) + 0.2

	Exp_Gamma_U, Exp_Gamma_LogU = ComputeExpectationGamma(a_u, b_u)
	Exp_Gamma_V, Exp_Gamma_LogV = ComputeExpectationGamma(a_v, b_v)

	au_0 = 1.0
	bu_0 = 1.0
	av_0 = 1.0
	bv_0 = 1.0

	ls_nz_V, ls_nz_D = get_list_of_nonzeros(X_train)
	print("list is obtained. Starting iterations...")
	bar = progressbar.ProgressBar()
	iteration = 0
	# curr = time()
	for iteration in bar(range(10000)):
		for k in range(0, K):
			temp = np.exp(Exp_Gamma_LogV[k,:])
			den = np.sum(temp)
			for v in range(0, V):
				
				# den = np.sum(np.exp(special.psi(a_v[k,:]) - np.log(b_v[k,:])))				
				# a_u[v][k] = au_0 + np.sum(X_train[v,:] * np.exp(special.psi(a_v[k,:]) - np.log(b_v[k,:]))) / den
				# print math.exp(Exp_Gamma_LogU[v][k])
				if math.isnan(math.exp(Exp_Gamma_LogU[v][k])):
					print "nan1"
					print iteration
					print Exp_Gamma_LogU[v][k]
					sys.exit()


				a_u[v][k] = au_0 + (np.sum(X_train[v,:] * temp) / den)
				if a_u[v][k] == 0 or not np.isfinite(a_u[v][k]):
					print "N2"


			b_u[:,k] = bu_0 + np.sum(Exp_Gamma_V[k,:])
			# b_u[:,k] = bu_0 + np.sum(a_v[k,:]/b_v[k,:])

		# bu = bu_0 + np.sum(Exp_Gamma_V, axis=0, keepdims=True)
			
		Exp_Gamma_U, Exp_Gamma_LogU = ComputeExpectationGamma(a_u, b_u)

		for k in range(0, K):
			
			temp = np.exp(Exp_Gamma_LogU[:,k])
			den = np.sum(temp)
			for d in range(0, D):
				# den = np.sum(np.exp(special.psi(a_u[:,k]) - np.log(b_u[:,k])))
			
				# a_v[k][d] = av_0 + np.sum(X_train[:,d] * np.exp(special.psi(a_u[:,k]) - np.log(b_u[:,k]))) / den
				# print math.exp(Exp_Gamma_LogV[k][d])
				if math.isnan(math.exp(Exp_Gamma_LogV[k][d])):
					print "nan2"
					print iteration
					print Exp_Gamma_LogV[k][d]
					sys.exit()

				a_v[k][d] = av_0 + (np.sum(X_train[:,d] * temp) / den)
				if a_v[k][d] == 0 or not np.isfinite(a_v[k][d]):
					print "N2"

			# b_v[k,:] = bv_0 + np.sum(a_u[:,k]/b_u[:,k])
			b_v[k,:] = bv_0 + np.sum(Exp_Gamma_U[:,k])

		# b_v = bv_0 + np.sum(Exp_Gamma_U, axis=1)

		Exp_Gamma_V, Exp_Gamma_LogV = ComputeExpectationGamma(a_v, b_v)

	print a_u
	# print b_u
	return a_u, b_u, a_v, b_v

def ComputeExpectationGamma(alpha_mat, beta_mat):
	# a = np.nonzero(alpha_mat)
	# a = a[0].shape[0]
	# if a != 15:
	# 	print "NOk1"

	# b = np.nonzero(beta_mat)
	# b = b[0].shape[0]
	# if b != 12:
	# 	print "NOk2"
	return alpha_mat / beta_mat, (special.psi(alpha_mat) - np.log(beta_mat))




if __name__ == '__main__':

	sess = ed.get_session()
	# Poisson._sample_n = _sample_n

	V = int(sys.argv[1])
	D = int(sys.argv[2])
	K = int(sys.argv[3])  # number of latent factors

	# DATA
	U_true = build_Matrix(V, K)
	V_true = build_Matrix(K, D)
	X_train = build_toy_dataset(U_true, V_true)

	print X_train

	FitUsingLibBatch(X_train, V, D, K)
	
	a_u, b_u, a_v, b_v = FitUsingOwnImplementation(X_train, V, D, K)
	qU = Gamma(alpha=a_u, beta=b_u)
	qV = Gamma(alpha=a_v, beta=b_v)

	qU_sample = qU.sample()
	qV_sample = qV.sample()

	avg_U = np.zeros((V,K))
	avg_V = np.zeros((K,D))

	n_sample = 10000
	X_new = np.zeros((V,D))
	for i in range(n_sample):
		avg_U = qU_sample.eval()
		avg_V = qV_sample.eval()
		temp = np.dot(avg_U, avg_V)
		X_new += np.random.poisson(temp)

	
	# print np.round(X_new / n_sample, 0)

	