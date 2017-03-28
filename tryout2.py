# %matplotlib inline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import numpy as np
import edward as ed
from edward.models import Categorical, InverseGamma, Normal
from edward.models import Categorical, InverseGamma, Mixture, \
    MultivariateNormalDiag, Normal
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def build_toy_dataset(N):
  pi = np.array([0.5, 0.5])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x

N = 1000  # number of data points
D = 2  # dimensionality of data

x_train = build_toy_dataset(N)

plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.show()

K = 2  # number of components

mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
cat = Categorical(logits=tf.zeros([N, K]))

t1 = tf.Variable(mu)
t2 = tf.Variable(sigma)
components = [
    MultivariateNormalDiag(mu=tf.ones([N, 1]) * t1[k],
                           diag_stdev=tf.ones([N, 1]) * t2[k])
    for k in range(K)]
# components = []
# a1 = mu.sample()
# a2 = sigma.sample()
# for k in range(K):
# 	t1 = a1[k]
# 	t2 = a2[k]
# 	components.append(MultivariateNormalDiag(mu=tf.ones([N, 1]) * t1,
#                            diag_stdev=tf.ones([N, 1]) * t2))

x = Mixture(cat=cat, components=components)

qmu = Normal(
    mu=tf.Variable(tf.random_normal([K, D])),
    sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
qsigma = InverseGamma(
    alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
    beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: x_train})
inference.run(n_iter=4000, n_samples=20)

sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("\nInferred cluster means:")
    print(sess.run(qmu.mean()))
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)
sigma_sample = qsigma.sample(100)
x_post = Normal(mu=tf.ones([N, 1, 1, 1]) * mu_sample,
                sigma=tf.ones([N, 1, 1, 1]) * sigma_sample)
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
# log_liks = tf.reduce_sum(log_liks)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

clusters = tf.argmax(log_liks, 1).eval()

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()