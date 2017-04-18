from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import sys
import math
import random
import edward as ed
# import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# from tensorflow.python.ops import random_ops

from edward.models import Gamma, Poisson, Normal
# from scipy.stats import poisson
from Read_Netflix_data import Create_Mapping_Customer, Create_Data_Set
import pmf_modified
import pmf_activity_popularity

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


Num_Movies = 200
Num_Customer = 10000
Latent_Factors = 150
B = 5
max_iter = 5
n_pass = 3
# # Create_Mapping_Customer(Num_Customer, Num_Movies)
# # Create_Data_Set(Num_Customer, Num_Movies)
Training_Data = "Netflix_train.npy"
X_train = np.load(Training_Data)
X_train = X_train[0:Num_Customer,0:Num_Movies]
# test_row_no = random.sample(range(0, Num_Customer), Num_Customer / 3 )
# test_col_no = random.sample(range(0, Num_Movies), Num_Movies / 3 )
# X_test = X_train[test_row_no, test_col_no]
# X_train[test_row_no, test_col_no] = 0

# pmf_var = pmf_modified.OnlinePoissonMF(n_components=Latent_Factors, batch_size=5, n_pass=2, max_iter=10, tol=0.0005, smoothness=100, verbose=True)

# # pmf_var.fit_Netflix(Num_Customer, Num_Movies)
# pmf_var.fit(X_train)

# a_u = pmf_var.gamma_b
# b_u = pmf_var.rho_b

# a_v = pmf_var.gamma_t
# b_v = pmf_var.rho_t

# qU = Gamma(alpha=a_u, beta=b_u)
# qV = Gamma(alpha=a_v, beta=b_v)

# qU_sample = qU.sample()
# qV_sample = qV.sample()

# X_new = np.zeros((Num_Customer,Num_Movies))

# n_sample = 1000
# for i in range(n_sample):
# 	avg_U = qU_sample.eval()
# 	avg_V = qV_sample.eval()
# 	temp = np.dot(avg_V, avg_U)
# 	X_new += np.random.poisson(temp)
# 	# X_new2 += temp

# X_new = np.round(X_new / n_sample, 1)

# rmse = 0
# tot = 0
# for i in range(Num_Customer):
# 	for j in range(Num_Movies):
# 		if X_train[i][j] == 0:
# 			continue
# 		rmse += (X_train[i][j] - X_new[i][j])**2
# 		tot += 1

# rmse = math.sqrt(1.0*rmse / tot)
# print rmse








# print X_new / n_sample
# print X_train
# print np.round(X_new / n_sample, 1)

# X_train, N, M = build_small_dataset()
# N = 5
# M = 4
# D = 3
# B = 2
# U = build_Matrix(D, M)
# V = build_Matrix(N, D)
# X_train = np.dot(V, U)
# X_train = build_Matrix(N, M)
# test_row_no = random.sample(range(0, N), int(N / 3) )
# test_col_no = random.sample(range(0, M), int(M / 3) )
# X_test = X_train[test_row_no, test_col_no]
# X_train[test_row_no, test_col_no] = 0

N = Num_Customer
M = Num_Movies
D = Latent_Factors

pmf_var = pmf_modified.OnlinePoissonMF(n_components=D,  batch_size=B, max_iter=max_iter, tol=0.0005,  smoothness=100, verbose=True, n_pass=n_pass)

pmf_var.fit(X_train)
# print pmf_var.transform(X_train)

a_u = pmf_var.gamma_b
b_u = pmf_var.rho_b

a_v = pmf_var.gamma_t
b_v = pmf_var.rho_t

qU = Gamma(alpha=a_u, beta=b_u)
qV = Gamma(alpha=a_v, beta=b_v)

qU_sample = qU.sample()
qV_sample = qV.sample()

X_new = np.zeros((N,M))

n_sample = 1000
for i in range(n_sample):
	avg_U = qU_sample.eval()
	avg_V = qV_sample.eval()
	temp = np.dot(avg_V, avg_U)
	X_new += np.random.poisson(temp)
	# X_new2 += temp

X_new2 = np.round(X_new / n_sample, 1)
X_new3 = np.array(np.round(X_new / n_sample, 0), dtype=np.int32)
# print X_train
# print X_new
corr = 0
rmse = 0
tot = 0
for i in range(N):
	for j in range(M):
		if X_train[i][j] == 0:
			continue
		rmse += (X_train[i][j] - X_new2[i][j])**2
		if X_train[i][j] == X_new3[i][j]:
			corr += 1

		tot += 1


rmse = math.sqrt(1.0*rmse / tot)
acc = 1.0*corr / tot


pmf_var = pmf_activity_popularity.OnlinePoissonMF(n_components=D,  batch_size=B, max_iter=100*max_iter, tol=0.0005,  smoothness=100, verbose=True, n_pass=3*n_pass)

pmf_var.fit(X_train)
# print pmf_var.transform(X_train)
a_u = pmf_var.gamma_b
b_u = pmf_var.rho_b

a_v = pmf_var.gamma_t
b_v = pmf_var.rho_t

qU = Gamma(alpha=a_u, beta=b_u)
qV = Gamma(alpha=a_v, beta=b_v)

qU_sample = qU.sample()
qV_sample = qV.sample()

X_new = np.zeros((N,M))

n_sample = 1000
for i in range(n_sample):
	avg_U = qU_sample.eval()
	avg_V = qV_sample.eval()
	temp = np.dot(avg_V, avg_U)
	X_new += np.random.poisson(temp)
	# X_new2 += temp

X_new2 = np.round(X_new / n_sample, 1)
X_new3 = np.array(np.round(X_new / n_sample, 0), dtype=np.int32)

corr = 0
rmse2 = 0
tot = 0
for i in range(N):
	for j in range(M):
		if X_train[i][j] == 0:
			continue
		rmse2 += (X_train[i][j] - X_new2[i][j])**2
		if X_train[i][j] == X_new3[i][j]:
			corr += 1
		tot += 1


rmse2 = math.sqrt(1.0*rmse2 / tot)
acc2 = 1.0*corr/tot

print rmse
print acc
print "\n"
print rmse2
print acc2
# print acc
# print acc2