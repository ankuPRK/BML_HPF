from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import sys
import random
import edward as ed
# import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# from tensorflow.python.ops import random_ops

from edward.models import Gamma, Poisson, Normal
# from scipy.stats import poisson

import pmf

def build_toy_dataset(U, V):
  R = np.dot(V, U)
  return R

def build_Matrix(n1, n2):
	X = np.zeros((n1,n2), dtype=np.int)	
	for i in range(n1):
		k = random.randint(1, int(n2/2))
		indices = random.sample(range(0, n2-1), k)
		for index in indices:
			X[i][index] = random.randint(1,5)

	return X

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

sess = ed.get_session()

N = int(sys.argv[1])
M = int(sys.argv[2])
D = int(sys.argv[3])
# D = int(min(M,N)/2) # number of latent factors

U_t = build_Matrix(D, M)
V_t = build_Matrix(N, D)
X_train = build_toy_dataset(U_t, V_t)
# X_train = build_Matrix(N, M)
# X_train, N, M = build_small_dataset()


# pmf_var = pmf.PoissonMF(n_components=D, max_iter=10000, tol=0.0005,
#                  smoothness=100, verbose=False)

# pmf_var.fit(X_train)
# # print pmf_var.transform(X_train)

# a_u = pmf_var.gamma_b
# b_u = pmf_var.rho_b
# b_u = np.repeat(b_u, M, axis=1)

# # print a_u.shape
# # # print a_u

# # print b_u.shape
# # # print b_u

# a_v = pmf_var.gamma_t
# b_v = pmf_var.rho_t
# b_v = b_v.reshape((1, D))
# b_v = np.repeat(b_v, N, axis=0)

# # print a_v.shape
# # print a_v

# # print b_v.shape
# # print b_v



# qU = Gamma(alpha=a_u, beta=b_u)
# qV = Gamma(alpha=a_v, beta=b_v)

# qU_sample = qU.sample()
# qV_sample = qV.sample()

# print qU_sample.shape
# print qV_sample.shape

# # avg_U = np.zeros((D,M))
# # avg_V = np.zeros((N,D))
# X_new = np.zeros((N,M))
# # X_new2 = np.zeros((N,M))

# n_sample = 10000
# for i in range(n_sample):
# 	avg_U = qU_sample.eval()
# 	avg_V = qV_sample.eval()
# 	temp = np.dot(avg_V, avg_U)
# 	X_new += np.random.poisson(temp)
# 	# X_new2 += temp

# # print X_new / n_sample
# print np.round(X_new / n_sample, 0)


pmf_var = pmf.OnlinePoissonMF(n_components=D,  batch_size=2, max_iter=1000, tol=0.0005,
                 smoothness=100, verbose=False)

pmf_var.fit(X_train)
# print pmf_var.transform(X_train)

a_u = pmf_var.gamma_b
b_u = pmf_var.rho_b
# b_u = np.repeat(b_u, M, axis=1)

print a_u.shape
print a_u

print b_u.shape
print b_u

a_v = pmf_var.gamma_t
b_v = pmf_var.rho_t
b_v = b_v.reshape((1, D))
b_v = np.repeat(b_v, N, axis=0)

print a_v.shape
print a_v

print b_v.shape
print b_v



qU = Gamma(alpha=a_u, beta=b_u)
qV = Gamma(alpha=a_v, beta=b_v)

qU_sample = qU.sample()
qV_sample = qV.sample()

print qU_sample.shape
print qV_sample.shape

# avg_U = np.zeros((D,M))
# avg_V = np.zeros((N,D))
X_new = np.zeros((N,M))
# X_new2 = np.zeros((N,M))

n_sample = 10000
for i in range(n_sample):
	avg_U = qU_sample.eval()
	avg_V = qV_sample.eval()
	temp = np.dot(avg_V, avg_U)
	X_new += np.random.poisson(temp)
	# X_new2 += temp

# print X_new / n_sample
print X_train
print np.round(X_new / n_sample, 0)
# print np.round(X_new2 / n_sample, 0)
# avg_U /= n_sample
# avg_V /= n_sample

# temp = np.dot(avg_V, avg_U)
# temp = np.transpose(temp)

# for i in range(n_sample):
# 	X_new += np.random.poisson(temp)

# print X_train
# print np.round(X_new / n_sample, 0)

# u_s = (a_u - 1.0) / b_u
# v_s = (a_v - 1.0) / b_v
# temp = np.dot(v_s,u_s)
# temp = np.transpose(temp)
# X_new = np.zeros((V,D))
# for i in range(n_sample):
# 	X_new += np.random.poisson(temp)
	
# print np.round(X_new / n_sample, 0)

# Poisson._sample_n = _sample_n
# qR = Poisson(lam=tf.matmul(qU, qV))
# qR_sample = qR.sample()

# X_new = np.zeros((V,D))
# for i in range(n_sample):
# 	X_new += qR_sample.eval()

# print np.round(X_new / n_sample, 0)



