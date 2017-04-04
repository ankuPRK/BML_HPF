# Link: http://edwardlib.org/tutorials/unsupervised


import numpy as np
import tensorflow as tf
from edward.models import Categorical, InverseGamma, Normal, MultivariateNormalDiag
from edward import KLqp
from matplotlib import pyplot as plt



def build_toy_dataset(N):
	pi = np.array([0.4, 0.6])
	mus = [[1, 1], [-1, -1]]
	stds = [[0.1, 0.1], [0.1, 0.1]]
	x = np.zeros((N, 2), dtype=np.float32)
	for n in range(N):
		k = np.argmax(np.random.multinomial(1, pi))
		x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
	return x

if __name__ == '__main__':
	N = 500  # number of data points
	D = 2  # dimensionality of data
	x_train = build_toy_dataset(N)

	K = 2  # number of components

	#joint model construction
	# mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
	# sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
	# c = Categorical(logits=tf.zeros([N, K]))
	# x = Normal(mu=tf.gather(mu, c), sigma=tf.gather(sigma, c))	

	#Collapsed model construction
	mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
	sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
	cat = Categorical(logits=tf.zeros([N, K]))

	print mu
	print sigma
	print cat
	# components = []
	# for k in range(K):
	# 	components.append(MultivariateNormalDiag(mu=tf.ones([N, 1]) * mu[k,:],
 #                           diag_stdev=tf.ones([N, 1]) * sigma[k,:]))
    
	jjj = MultivariateNormalDiag(mu=tf.ones([N, K]) * mu,
                           diag_stdev=tf.ones([N, K]) * sigma)

	# x = Mixture(cat=cat, components=components)	

	# #Inference model
	# qmu = Normal(
	#     mu=tf.Variable(tf.random_normal([K, D])),
	#     sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
	# qsigma = InverseGamma(
	#     alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
	#     beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))	

	# #run the Inference for 4000 iterations
	# inference = KLqp({mu: qmu, sigma: qsigma}, data={x: x_train})
	# inference.run(n_iter=4000, n_samples=20)

	# #Criticism
	# # Calculate likelihood for each data point and cluster assignment,
	# # averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
	# mu_sample = qmu.sample(100)
	# sigma_sample = qsigma.sample(100)
	# x_post = Normal(mu=tf.ones([N, 1, 1, 1]) * mu_sample,
	#                 sigma=tf.ones([N, 1, 1, 1]) * sigma_sample)
	# x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

	# # Sum over latent dimension, then average over posterior samples.
	# # ``log_liks`` ends up with shape (N, K).
	# log_liks = x_post.log_prob(x_broadcasted)
	# log_liks = tf.reduce_sum(log_liks, 3)
	# log_liks = tf.reduce_mean(log_liks, 1)

	# #cluster assignment
	# clusters = tf.argmax(log_liks, 1).eval()

	#Plot original data
	plt.figure(1)
	plt.scatter(x_train[:, 0], x_train[:, 1])
	plt.axis([-3, 3, -3, 3])

	#Plot the colored data
	# plt.figure(2)
	# plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
	# plt.axis([-3, 3, -3, 3])
	# plt.title("Predicted cluster assignments")

	plt.show()
