import os
import sys
import random
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Gamma, Poisson, Normal
from scipy.stats import poisson

# def _sample_n(self, n=1, seed=None):
#   # define Python function which returns samples as a Numpy array
#   def np_sample(lam, n):
#     return poisson.rvs(mu=lam, size=n, random_state=seed).astype(np.float32)

#   # wrap python function as tensorflow op
#   val = tf.py_func(np_sample, [self.lam, n], [tf.float32])[0]
#   # set shape from unknown shape
#   batch_event_shape = self.get_batch_shape().concatenate(self.get_event_shape())
#   shape = tf.concat([tf.expand_dims(n, 0),
#                      tf.constant(batch_event_shape.as_list(), dtype=tf.int32)],
#                      0)
#   # shape = tf.concat_v2(0,
#   # 					[tf.expand_dims(n, 0),tf.constant(batch_event_shape.as_list(), dtype=tf.int32)]
#   #                    )
#   val = tf.reshape(val, shape)
#   return val


def build_toy_dataset(n1, n2):
	X = np.zeros((n1,n2), dtype=np.int)	
	for i in range(n1):
		k = random.randint(0, int(n2/2))
		indices = random.sample(range(0, n2-1), k)
		for index in indices:
			X[i][index] = random.randint(1,5)

	return X

# X_test = build_toy_dataset(M/4,N/4)

M = int(sys.argv[1])
N = int(sys.argv[2])

K = 50

X_train = build_toy_dataset(M,N)

# U = tf.placeholder(shape=[M, K], dtype=tf.float32)
# V = tf.placeholder(shape=[K, N], dtype=tf.float32)
U = Gamma(alpha=tf.zeros([M, K]), beta=tf.ones([M, K]))
V = Gamma(alpha=tf.zeros([K, N]), beta=tf.ones([K, N]))
# U1 = tf.Variable(U)
# V1 = tf.Variable(V)
# V = tf.placeholder(tf.float32, [N, K])
# t = U*V
# Poisson._sample_n = _sample_n
temp = tf.matmul(U,V)
# X = tf.placeholder(tf.int32, Poisson(lam=temp, value=tf.zeros_like(temp)))
X = Poisson(lam=temp, value=tf.zeros_like(temp))

# print U
# print V
qU = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([M, K]))), beta=tf.exp(tf.Variable(tf.ones([M, K]))))
qV = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([K, N]))), beta=tf.exp(tf.Variable(tf.ones([K, N]))))

inference = ed.VariationalInference({U: qU, V: qV}, data={X: X_train})
# inference.run(n_iter=1000)

# X_post = ed.copy(X, {U: q_U, V: q_V})
# print ed.evaluate('log_likelihood', data={X_post: X_train})
# X = tf.placeholder(tf.float32, [M, N])
# w = Normal(mu=tf.zeros(N), sigma=tf.ones(N))
# b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
# y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(M))
