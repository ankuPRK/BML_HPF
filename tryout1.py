import os
import sys
import random
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Gamma, Poisson

M = int(sys.argv[1])
N = int(sys.argv[2])

def build_toy_dataset(n1, n2):
	X = np.zeros((n1,n2), dtype=np.int)	
	for i in range(n1):
		k = random.randint(0, int(n2/2))
		indices = random.sample(range(0, n2-1), k)
		for index in indices:
			X[i][index] = random.randint(1,5)

	return X

X_train = build_toy_dataset(M,N)
# X_test = build_toy_dataset(M/4,N/4)

K = 50

U = Gamma(alpha=tf.zeros([M, K]), beta=tf.ones([M, K]))
V = Gamma(alpha=tf.zeros([K, N]), beta=tf.ones([K, N]))
t = U*V
# X = Poisson(lam=tf.matmul(U,V))
# print U
# print V
# q_U = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([M, K]))), beta=tf.exp(tf.Variable(tf.ones([M, K]))))
# q_V = Gamma(alpha=tf.exp(tf.Variable(tf.zeros([K, N]))), beta=tf.exp(tf.Variable(tf.ones([K, N]))))

# inference = ed.VariationalInference({U: q_U, V: q_V}, data={X: X_train})
# inference.run(n_iter=1000)

# X_post = ed.copy(X, {U: q_U, V: q_V})
# print ed.evaluate('log_likelihood', data={X_post: X_train})