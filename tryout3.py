import os
import sys
import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal, Poisson

# filenames = os.listdir('Netflix/training_set')
# count = int(sys.argv[1])
# for filename in filenames:
# 	f = 'Netflix/training_set/' + filename
# 	fp = open(f, 'r')
# 	line = fp.readline()
# 	while line:
# 		line = fp.readline()	

# 	count -= 1
# 	if count == 0:
# 		break

x_train = np.load('celegans_brain.npy')

N = x_train.shape[0]  # number of data points
K = 3  # latent dimensionality

z = Normal(mu=tf.zeros([N, K]), sigma=tf.ones([N, K]))
print z
# Calculate N x N distance matrix.
# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2], and tile
# it to create N identical rows.
xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, N])
print xp
# 2. Create a N x N matrix where entry (i, j) is ||z_i||^2 + ||z_j||^2
# - 2 z_i^T z_j.
xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
print xp
# 3. Invert the pairwise distances and make rate along diagonals to
# be close to zero.
xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))
print xp
x = Poisson(lam=xp, value=tf.zeros_like(xp))
print x
qz = Normal(mu=tf.Variable(tf.random_normal([N, K])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N, K]))))

# print qz
# print z

inference = ed.KLqp({z: qz}, data={x: x_train})
inference.run(n_iter=2500)

