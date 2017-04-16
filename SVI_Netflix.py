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
from Read_Netflix_data import Create_Mapping_Customer, Create_Data_Set
import pmf_modified

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


Num_Movies = 2000
Num_Customer = 30000
Latent_Factors = 100

Create_Mapping_Customer(Num_Customer, Num_Movies)
Create_Data_Set(Num_Customer, Num_Movies)
# pmf_var = pmf_modified.OnlinePoissonMF(n_components=D, batch_size=1, n_pass=4,max_iter=10, tol=0.0005, smoothness=100, verbose=True)

# pmf_var.fit_Netflix(N, M)
# pmf_var.fit(X_train)

# a_u = pmf_var.gamma_b
# b_u = pmf_var.rho_b

# a_v = pmf_var.gamma_t
# b_v = pmf_var.rho_t

# qU = Gamma(alpha=a_u, beta=b_u)
# qV = Gamma(alpha=a_v, beta=b_v)

# qU_sample = qU.sample()
# qV_sample = qV.sample()

# X_new = np.zeros((N,M))

# n_sample = 10000
# for i in range(n_sample):
# 	avg_U = qU_sample.eval()
# 	avg_V = qV_sample.eval()
# 	temp = np.dot(avg_V, avg_U)
# 	X_new += np.random.poisson(temp)
# 	# X_new2 += temp

# # print X_new / n_sample
# print X_train
# print np.round(X_new / n_sample, 0)

# pmf_var = pmf_modified.OnlinePoissonMF(n_components=D,  batch_size=2, max_iter=1000, tol=0.0005,
#                  smoothness=100, verbose=False)

# pmf_var.fit(X_train)
# # print pmf_var.transform(X_train)

# a_u = pmf_var.gamma_b
# b_u = pmf_var.rho_b

# a_v = pmf_var.gamma_t
# b_v = pmf_var.rho_t

# qU = Gamma(alpha=a_u, beta=b_u)
# qV = Gamma(alpha=a_v, beta=b_v)

# qU_sample = qU.sample()
# qV_sample = qV.sample()

# X_new = np.zeros((N,M))

# n_sample = 10000
# for i in range(n_sample):
# 	avg_U = qU_sample.eval()
# 	avg_V = qV_sample.eval()
# 	temp = np.dot(avg_V, avg_U)
# 	X_new += np.random.poisson(temp)
	
# # print X_train
# print np.round(X_new / n_sample, 0)