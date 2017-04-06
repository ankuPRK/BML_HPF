#!/usr/bin/env python
"""Probabilistic matrix factorization using variational inference.
Visualizes the actual and the estimated rating matrices as heatmaps.
"""
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import sys
import random
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops

from edward.models import Gamma, Poisson, Normal
from scipy.stats import poisson

sess = ed.get_session()

def build_toy_dataset(U, V):
  R = np.dot(U, np.transpose(V))
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
# def _sample_n(self, n, seed=None):
#     return random_ops.random_poisson(
#         self.rate, [n], dtype=self.dtype, seed=seed)

def _sample_n(self, n=1, seed=None):
  # define Python function which returns samples as a Numpy array
	np_sample = lambda lam, n: poisson.rvs(mu=lam, size=n, random_state=seed).astype(np.float32)
  # wrap python function as tensorflow op
	val = tf.py_func(np_sample, [self.lam, n], [tf.float32])[0]
  # set shape from unknown shape
	batch_event_shape = self.get_batch_shape().concatenate(self.get_event_shape())
	shape = tf.concat(0, [tf.expand_dims(n, 0),
					tf.constant(batch_event_shape.as_list(), 
						dtype=tf.int32)])
	# print batch_event_shape
	# print shape
	val = tf.reshape(val, shape)
	return val

# R_true, N, M = build_small_dataset()
M = int(sys.argv[1])
N = int(sys.argv[2])
D = int(min(M,N)/2) # number of latent factors
U_true = build_Matrix(N, D)
V_true = build_Matrix(M, D)
R_true = build_toy_dataset(U_true, V_true)
R_true = np.array(R_true, dtype=np.float32)

B = N
# true latent factors
# U_true = build_Matrix(N, D)
# V_true = build_Matrix(M, D)

# DATA
# print np.shape(R_true)
# I_train = get_indicators(N, M)
# I_test = 1 - I_train

# MODEL
I = tf.placeholder(tf.float32, [N, M])
# U = Gamma(alpha=tf.ones([N, M]), beta=tf.ones([N, M]))
U = Gamma(alpha=tf.ones([N, D]), beta=tf.ones([N, D]))
V = Gamma(alpha=tf.ones([M, D]), beta=tf.ones([M, D]))


# Poisson._sample_n = _sample_n
# sess = ed.get_session()
temp = tf.matmul(U, tf.transpose(V))
# temp = tf.gather(U, V)
# print temp
# temp2 = tf.matmul(tf.transpose(U), V)
# U_sample = U.sample()
# V_sample = V.sample()

# # print (U_sample)
# values = tf.matmul(tf.transpose(U_sample), V_sample)
# values = _sample_n(N, M)
# t = tf.matmul(tf.transpose(U), V)
# # print (t[0][0])
# values =  np.zeros((N, M))
# for i in range(0, N):
# 	for j in range(0, M):
# 		values[i][j] = int(poisson.rvs(t[i][j], size=1))
# temp = tf.tile(tf.matmul(tf.transpose(U), V),1)
R = Poisson(lam=temp, value=tf.zeros_like(temp))
R = tf.reshape(R, [N, M])
# R = Poisson(lam=U, value=tf.zeros_like(U))
# R = tf.cast(R, tf.int32)
# print R
# R = Poisson(lam=tf.constant(1.0))
# print R
# INFERENCE
qU = Gamma(alpha=tf.exp(tf.Variable(tf.ones([N, D]))), beta=tf.exp(tf.Variable(tf.ones([N, D]))))
qV = Gamma(alpha=tf.exp(tf.Variable(tf.ones([M, D]))), beta=tf.exp(tf.Variable(tf.ones([M, D]))))

#Define qU and qV differently
# qU = Gamma(alpha=tf.Variable(tf.ones([N, D])), beta=tf.exp(tf.Variable(tf.ones([N, D]))))
# qV = Gamma(alpha=tf.exp(tf.Variable(tf.ones([M, D]))), beta=tf.exp(tf.Variable(tf.ones([M, D]))))

# qU = Gamma(alpha=tf.Variable(tf.zeros([N, D])), beta=tf.Variable(tf.ones([N, D])))
# qV = Gamma(alpha=tf.Variable(tf.zeros([M, D])), beta=tf.Variable(tf.ones([M, D])))
# qU_var_alpha_U = tf.Variable(tf.exp(tf.ones([B, D])))
# qU_var_beta_U = tf.Variable(tf.exp(tf.ones([B, D])))

# qU_var_alpha_V = tf.Variable(tf.exp(tf.ones([M, D])))
# qU_var_beta_V = tf.Variable(tf.exp(tf.ones([M, D])))

# qU = Gamma(alpha=qU_var_alpha_U, beta=qU_var_beta_U)
# qV = Gamma(alpha=qU_var_alpha_V, beta=qU_var_beta_V)
# qV = Gamma(alpha=tf.exp(tf.Variable(tf.ones([M, D]))), beta=tf.exp(tf.Variable(tf.ones([M, D]))))

# R_ph = tf.placeholder(tf.float32, [B, M])
# inference_global = ed.KLqp({V: qV}, data={R: R_ph, U: qU})
# inference_local = ed.KLqp({U: qU}, data={R: R_ph, V: qV})

# # # # temp = []
# # # # for i in range(0, M):
# # # # 	temp.append(float(N)/B)

# # # # temp = tf.cast(temp, tf.float32)
# # # # print R
# # # # print float(N) / B
# inference_global.initialize()
# inference_local.initialize()

# qU_alpha_init = tf.variables_initializer([qU_var_alpha])
# qU_beta_init = tf.variables_initializer([qU_var_beta])
# inference = ed.KLqp({V: qV}, data={R: R_true, U: qU})
# inference.run(n_iter=2500)
# # inference = ed.KLqp({U: qU}, data={R: R_true, V: qV})
# # inference.run(n_iter=2500)
# # # # # k = 0
# for i in range(1000):
# # 	# R_batch = R_true[k:k+B+1][:]
# # 	# k += B + 1
# 	R_batch = R_true
# 	for j in range(B): # make local inferences
# 		inference_local.update(feed_dict={R_ph: R_batch, V: qV})
# # 	# update global parameters
# 	inference_global.update(feed_dict={R_ph: R_batch})
# # 	# reinitialize the local factors
# 	qU_alpha_init.run()
# 	qU_beta_init.run()
# print R_true

# print R_true
# print R_true
# print temp
# print R_true
# print R
# print U
# print qU
# print V
# print qV
inference = ed.KLqp({U: qU, V: qV}, data={R: R_true})
inference.run(n_iter=2500)

qU_sample = qU.sample()
qV_sample = qV.sample()

qU_sample = qU_sample.eval()
qV_sample = qV_sample.eval()

t = np.zeros((N, M), dtype=np.float32)
for n in range(0, N):
	for m in range(0, M):
		s = 0
		for d in range(0, D):
			s += qU_sample[n][d]*qV_sample[m][d]

		t[n][m] = float(s)
# t = np.dot(qU_sample,np.transpose(qU_sample))
R_new = np.random.poisson(t)
print R_true
print R_new

# print qV.eval()
# print qV.eval()
# print qU_var_alpha_U.eval()
# print qU_var_beta_U.eval()
# print qU_var_alpha_V.eval()
# print qU_var_beta_V.eval()
# CRITICISM
# qR = Normal(mu=tf.matmul(tf.transpose(qU), qV), sigma=tf.ones([N, M]))
# temp = tf.matmul(tf.transpose(qU), qV)
# temp = tf.matmul(qU, tf.transpose(qV))
# qR = Poisson(lam=temp, value=tf.zeros_like(temp))
# R_new = qR.sample()
# print R_new.eval()

# print("Mean squared error on test data:")
# print(ed.evaluate('mean_squared_error', data={qR: R_true}))

# f = plt.figure("True")
# plt.imshow(R_true)
# f.show()

# f = plt.figure("Estimated")
# R_est = tf.matmul(tf.transpose(qU), qV).eval()
# plt.imshow(R_est)
# f.show()

# raw_input()



# # U = tf.placeholder(shape=[M, K], dtype=tf.float32)
# # V = tf.placeholder(shape=[K, N], dtype=tf.float32)
# U = Gamma(alpha=tf.zeros([M, K]), beta=tf.ones([M, K]))
# V = Gamma(alpha=tf.zeros([K, N]), beta=tf.ones([K, N]))
# # U1 = tf.Variable(U)
# # V1 = tf.Variable(V)
# # V = tf.placeholder(tf.float32, [N, K])
# # t = U*V
# # Poisson._sample_n = _sample_n
# temp = tf.matmul(U,V)
# # X = tf.placeholder(tf.int32, Poisson(lam=temp, value=tf.zeros_like(temp)))
# X = Poisson(lam=temp, value=tf.zeros_like(temp))

# # print U
# # print V
# qU = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([M, K]))), beta=tf.exp(tf.Variable(tf.ones([M, K]))))
# qV = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([K, N]))), beta=tf.exp(tf.Variable(tf.ones([K, N]))))

# inference = ed.VariationalInference({U: qU, V: qV}, data={X: X_train})
# # inference.run(n_iter=1000)

# # X_post = ed.copy(X, {U: q_U, V: q_V})
# # print ed.evaluate('log_likelihood', data={X_post: X_train})
# # X = tf.placeholder(tf.float32, [M, N])
# # w = Normal(mu=tf.zeros(N), sigma=tf.ones(N))
# # b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
# # y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(M))
