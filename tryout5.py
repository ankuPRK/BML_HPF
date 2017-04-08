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

M = int(sys.argv[1])
N = int(sys.argv[2])
# R_true, N, M = build_small_dataset()
D = int(min(M,N)/2) # number of latent factors
# U_true = build_Matrix(N, D)
# V_true = build_Matrix(M, D)
# R_true = build_toy_dataset(U_true, V_true)
R_true = build_Matrix(N, M)
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
# I = tf.placeholder(tf.float32, [N, M])
# U = Gamma(alpha=tf.ones([N, M]), beta=tf.ones([N, M]))
U = Gamma(alpha=tf.ones([N, D]), beta=tf.ones([N, D]))
V = Gamma(alpha=tf.ones([M, D]), beta=tf.ones([M, D]))


# Poisson._sample_n = _sample_n
# sess = ed.get_session()
temp = tf.matmul(U, tf.transpose(V))

R = Poisson(lam=temp, value=tf.zeros_like(temp))
R = tf.reshape(R, [N, M])

# INFERENCE
# qU = Gamma(alpha=tf.exp(tf.Variable(tf.ones([N, D]))), beta=tf.exp(tf.Variable(tf.ones([N, D]))))
# qV = Gamma(alpha=tf.exp(tf.Variable(tf.ones([M, D]))), beta=tf.exp(tf.Variable(tf.ones([M, D]))))

#Define qU and qV differently
qU_var_alpha = tf.Variable(tf.ones([B, D]))
qU_var_beta = tf.Variable(tf.ones([B, D]))
qV_var_alpha = tf.Variable(tf.ones([M, D]))
qV_var_beta = tf.Variable(tf.ones([M, D]))

# qU = Gamma(alpha=tf.Variable(tf.ones([N, D])), beta=tf.Variable(tf.ones([N, D])))
# qV = Gamma(alpha=tf.Variable(tf.ones([M, D])), beta=tf.Variable(tf.ones([M, D])))

qU = Gamma(alpha=qU_var_alpha, beta=qU_var_beta)
qV = Gamma(alpha=qV_var_alpha, beta=qV_var_beta)


# qU = Gamma(alpha=qU_var_alpha_U, beta=qU_var_beta_U)
# qV = Gamma(alpha=qU_var_alpha_V, beta=qU_var_beta_V)
# qV = Gamma(alpha=tf.exp(tf.Variable(tf.ones([M, D]))), beta=tf.exp(tf.Variable(tf.ones([M, D]))))

R_ph = tf.placeholder(tf.float32, [B, M])
inference_local = ed.KLqp({U: qU}, data={R: R_ph, V: qV})
inference_global = ed.KLqp({V: qV}, data={R: R_ph, U: qU})

# # # # temp = []
# # # # for i in range(0, M):
# # # # 	temp.append(float(N)/B)

# # # # temp = tf.cast(temp, tf.float32)
# # # # print R
# # # # print float(N) / B
inference_global.initialize(debug=True)
inference_local.initialize(debug=True)

qU_alpha_init = tf.variables_initializer([qU_var_alpha])
qU_beta_init = tf.variables_initializer([qU_var_beta])

qV_alpha_init = tf.variables_initializer([qV_var_alpha])
qV_beta_init = tf.variables_initializer([qV_var_beta])
# inference = ed.KLqp({V: qV}, data={R: R_true, U: qU})
# inference.run(n_iter=2500)
# # inference = ed.KLqp({U: qU}, data={R: R_true, V: qV})
# # inference.run(n_iter=2500)
# # # # # k = 0
for i in range(1000):
# 	# R_batch = R_true[k:k+B+1][:]
# 	# k += B + 1
	R_batch = R_true
	for j in range(B): # make local inferences
		inference_local.update(feed_dict={R_ph: R_batch})
# 	# update global parameters
	inference_global.update(feed_dict={R_ph: R_batch})
# 	# reinitialize the local factors
	qU_alpha_init.run()
	qU_beta_init.run()

# inference = ed.KLqp({U: qU, V: qV}, data={R: R_true})
# inference.run(n_iter=2500)

# qU_sample = qU.sample()
# qV_sample = qV.sample()

# qU_sample = qU_sample.eval()
# qV_sample = qV_sample.eval()

# print qU_sample
# print qV_sample

# t = np.zeros((N, M), dtype=np.float32)
# for n in range(0, N):
# 	for m in range(0, M):
# 		s = 0
# 		for d in range(0, D):
# 			s += qU_sample[n][d]*qV_sample[m][d]

# 		t[n][m] = float(s)
# # t = np.dot(qU_sample,np.transpose(qU_sample))
# R_new = np.random.poisson(t)
# print R_true
# print R_new
