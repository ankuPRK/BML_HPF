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
                        tf.constant(batch_event_shape.as_list(), dtype=tf.int32)])
  val = tf.reshape(val, shape)
  return val

M = int(sys.argv[1])
N = int(sys.argv[2])

D = 25  # number of latent factors
B = 10
# true latent factors
U_true = build_Matrix(N, D)
V_true = build_Matrix(M, D)

# DATA
R_true = build_toy_dataset(U_true, V_true)
print np.shape(R_true)
# I_train = get_indicators(N, M)
# I_test = 1 - I_train

# MODEL
I = tf.placeholder(tf.float32, [N, M])
U = Gamma(alpha=tf.zeros([N, D]), beta=tf.ones([N, D]))
V = Gamma(alpha=tf.zeros([M, D]), beta=tf.ones([M, D]))

Poisson._sample_n = _sample_n
# sess = ed.get_session()
temp = tf.matmul(U, np.transpose(V))
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
R = Poisson(lam=temp)
print R
# INFERENCE
qU = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([N, D]))), beta=tf.exp(tf.Variable(tf.ones([N, D]))))
qV = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([M, D]))), beta=tf.exp(tf.Variable(tf.ones([M, D]))))

inference_global = ed.KLqp({U: qU}, data={R: R_tru, V: qV})
inference_local = ed.KLqp({V: qV}, data={x: x_ph, U: qU})

inference_global.initialize(scale={R: float(N) / B, V: float(N) / B})
inference_local.initialize(scale={R: float(N) / B, V: float(N) / B})

qz_init = tf.initialize_variables([qz_variables])
for _ in range(1000):
	x_batch = next_batch(size=M)
	for _ in range(10): # make local inferences
		inference_local.update(feed_dict={x_ph: x_batch})
	# update global parameters
	inference_global.update(feed_dict={x_ph: x_batch})
	# reinitialize the local factors
	qz_init.run()
# inference = ed.VariationalInference({U: qU, V: qV}, data={R: R_true})
# inference.run()

# CRITICISM
# qR = Normal(mu=tf.matmul(tf.transpose(qU), qV), sigma=tf.ones([N, M]))
temp = tf.matmul(tf.transpose(qU), qV)
qR = Poisson(lam=temp, value=temp)
# qR = tf.matmul(tf.transpose(qU), qV)

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={qR: R_true, I: I_test}))

f = plt.figure("True")
plt.imshow(R_true)
f.show()

f = plt.figure("Estimated")
R_est = tf.matmul(tf.transpose(qU), qV).eval()
plt.imshow(R_est)
f.show()

raw_input()



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
