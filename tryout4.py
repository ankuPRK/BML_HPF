#!/usr/bin/env python
"""Probabilistic matrix factorization using variational inference.
Visualizes the actual and the estimated rating matrices as heatmaps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

sess = tf.Session()

def build_toy_dataset(U, V, N, M, noise_std=0.1):
  R = np.dot(np.transpose(U), V) + np.random.normal(0, noise_std, size=(N, M))
  return R


def get_indicators(N, M, prob_std=0.5):
  ind = np.random.binomial(1, prob_std, (N, M))
  return ind


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

N = 100  # number of users
M = 200  # number of movies
D = 25  # number of latent factors

# true latent factors
U_true = np.random.randn(D, N)
V_true = np.random.randn(D, M)

# DATA
R_true, N, M = build_small_dataset()
# R_true = build_toy_dataset(U_true, V_true, N, M)
I_train = get_indicators(N, M)
I_test = 1 - I_train

# MODEL
# I = tf.placeholder(tf.float32, [N, M])
U = Normal(mu=tf.zeros([D, N]), sigma=tf.ones([D, N]))
V = Normal(mu=tf.zeros([D, M]), sigma=tf.ones([D, M]))
R = Normal(mu=tf.matmul(tf.transpose(U), V), sigma=tf.ones([N, M]))

# INFERENCE
qU = Normal(mu=tf.Variable(tf.random_normal([D, N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, N]))))
qV = Normal(mu=tf.Variable(tf.random_normal([D, M])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, M]))))

inference = ed.KLqp({U: qU, V: qV}, data={R: R_true})
inference.run(n_iter=10000)


# CRITICISM
qR = Normal(mu=tf.matmul(tf.transpose(qU), qV), sigma=tf.ones([N, M]))

R_new = qR.sample()
# R_new = sess.run(temp)
# R_new = tf.reshape(R_new, [5, 4])
# R_new = np.matrix(R_new)
print (R_true)
print (R_new.eval())
# print (R_new)
# t = np.array(R_true - R_new)

# print (t)
# print (np.sum((R_true - R_new)**2)/(V*D))

#Posterior
# qU = Gamma(alpha=a_u, beta=b_u)
# qV = Gamma(alpha=a_v, beta=b_v)
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