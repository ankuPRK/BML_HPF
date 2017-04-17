"""

Poisson matrix factorization with Batch inference and Stochastic inference

CREATED: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>

"""

import sys
import numpy as np
from scipy import special
from operator import attrgetter

from sklearn.base import BaseEstimator, TransformerMixin

from Read_Netflix_data import GetMiniBatch

class PoissonMF(BaseEstimator, TransformerMixin):
    ''' Poisson matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):
        ''' Poisson matrix factorization

        Arguments
        ---------
        n_components : int
            Number of latent components

        max_iter : int
            Maximal number of iterations to perform

        tol : float
            The threshold on the increase of the objective to stop the
            iteration

        smoothness : int
            Smoothness on the initialization variational parameters

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during model fitting

        **kwargs: dict
            Model hyperparameters
        '''

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.c = float(kwargs.get('c', 0.1))

    def _init_popularity(self, n_feats):
        # variational parameters for beta
        self.gamma_p = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(1, n_feats))
        self.rho_p = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(1, n_feats))
        self.Ep, self.Elogp = _compute_expectations(self.gamma_p, self.rho_p)

    def _init_components(self, n_feats):
        # variational parameters for beta
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def set_components(self, shape, rate):
        '''Set the latent components from variational parameters.

        Parameters
        ----------
        shape : numpy-array, shape (n_components, n_feats)
            Shape parameters for the variational distribution

        rate : numpy-array, shape (n_components, n_feats)
            Rate parameters for the variational distribution

        Returns
        -------
        self : object
            Return the instance itself.
        '''

        self.gamma_b, self.rho_b = shape, rate
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)
        return self

    def _init_activity(self, n_samples):
        # variational parameters for theta
        self.gamma_a = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, 1))
        self.rho_a = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, 1))
        self.Ea, self.Eloga = _compute_expectations(self.gamma_a, self.rho_a)
        # self.c = 1. / np.mean(self.Et)

    def _init_weights(self, n_samples):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        # self.c = 1. / np.mean(self.Et)

    def fit(self, X):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_samples, n_feats = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self._update(X)
        return self

    def transform(self, X, attr=None):
        '''Encode the data as a linear combination of the latent components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)

        attr: string
            The name of attribute, default 'Eb'. Can be changed to Elogb to
            obtain E_q[log beta] as transformed data.

        Returns
        -------
        X_new : array-like, shape(n_samples, n_filters)
            Transformed data, as specified by attr.
        '''

        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Et'
        self._init_weights(n_samples)
        self._update(X, update_beta=False)
        return getattr(self, attr)

    def _update(self, X, update_beta=True):
        # alternating between update latent components and weights
        old_bd = -np.inf
        for i in xrange(self.max_iter):
            self._update_theta(X)
            if update_beta:
                self._update_beta(X)
            bound = self._bound(X)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                sys.stdout.write('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
                sys.stdout.flush()
            if improvement < self.tol:
                break
            old_bd = bound
        if self.verbose:
            sys.stdout.write('\n')
        pass

    def _update_theta(self, X):
        ratio = X / self._xexplog()
        self.gamma_t = self.a + np.exp(self.Elogt) * np.dot(
            ratio, np.exp(self.Elogb).T)
        self.rho_t = self.a * self.c + np.sum(self.Eb, axis=1)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def _update_beta(self, X):
        ratio = X / self._xexplog()
        self.gamma_b = self.c + np.exp(self.Elogb) * np.dot(
            np.exp(self.Elogt).T, ratio)
        self.rho_b = self.c + np.sum(self.Et, axis=0, keepdims=True).T
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))

    def _bound(self, X):
        bound = np.sum(X * np.log(self._xexplog()) - self.Et.dot(self.Eb))
        bound += _gamma_term(self.a, self.a * self.c,
                             self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound += _gamma_term(self.c, self.c, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound


class OnlinePoissonMF(PoissonMF):
    ''' Poisson matrix factorization with stochastic inference '''
    def __init__(self, n_components=100, batch_size=10, n_pass=10,
                 max_iter=100, tol=0.0005, shuffle=True, smoothness=100,
                 random_state=None, verbose=False,
                 **kwargs):
        ''' Poisson matrix factorization

        Arguments
        ---------
        n_components : int
            Number of latent components

        batch_size : int
            The size of mini-batch

        n_pass : int
            The number of passes through the entire data

        max_iter : int
            Maximal number of iterations to perform for a single mini-batch

        tol : float
            The threshold on the increase of the objective to stop the
            iteration

        shuffle : bool
            Whether to shuffle the data or not

        smoothness : int
            Smoothness on the initialization variational parameters

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during model fitting

        **kwargs: dict
            Model hyperparameters and learning rate
        '''

        self.n_components = n_components
        self.catch_size = batch_size
        self.n_pass = n_pass
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a_prime = float(kwargs.get('a_prime', 0.1))
        self.c_prime = float(kwargs.get('b_prime', 0.1))
        self.a = float(kwargs.get('a', 0.1))
        self.c_prime = float(kwargs.get('c_prime', 0.1))
        self.d_prime = float(kwargs.get('d_prime', 0.1))
        self.c = float(kwargs.get('c', 0.1))
        self.t0 = float(kwargs.get('t0', 1.))
        self.rho = float(kwargs.get('rho', 0.6))

    def fit(self, X, est_total=None):
        '''Fit the model to the data in X. X has to be loaded into memory.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        est_total : int
            The estimated size of the entire data. Could be larger than the
            actual size.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_samples, n_feats = X.shape
        if est_total is None:
            self._scale = float(n_samples) / self.catch_size
        else:
            self._scale = float(est_total) / self.catch_size
        self._init_popularity(n_feats)
        self._init_components(n_feats)
        self._init_activity(n_samples)
        self._init_weights(n_samples)
        
        #Update Activity and Popularity shape parameter
        # self.gamma_p = np.repeat(self.c_prime + self.n_components*self.c, n_feats).reshape((1, n_feats))
        self.gamma_a = self.a_prime + self.n_components*self.a
        self.gamma_p = self.c_prime + self.n_components*self.c
        self.cound = list()
        for count in xrange(self.n_pass):
            if self.verbose:
                print 'Iteration %d: passing through the data...' % count
            for (i, istart) in enumerate(xrange(0, n_samples,
                                                self.catch_size), 1):
                # print '\tMinibatch %d:' % i, istart
                iend = min(istart + self.catch_size, n_samples)
                self.set_learning_rate(iter=i)
                mini_batch = X[istart:iend]
                self.partial_fit(mini_batch, istart, iend)
                self.cound.append(self._stoch_bound(mini_batch, istart, iend))
        return self

    
    def partial_fit(self, X, istart, iend):
        '''Fit the data in X as a mini-batch and update the parameter by taking
        a natural gradient step. Could be invoked from a high-level out-of-core
        wrapper.

        Parameters
        ----------
        X : array-like, shape (batch_size, n_feats)
            Mini-batch data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        self.transform(X, istart, iend)

        # take a (natural) gradient step
        ratio = X / self._xexplog(istart, iend)
        self.gamma_b = (1 - self.rho) * self.gamma_b + self.rho * \
            (self.c + self._scale * np.exp(self.Elogb) *
             np.dot(np.exp(self.Elogt[istart:iend]).T, ratio))
        #print self.c, self._scale * np.sum(self.Et, axis=0, keepdims=True).T
        # print "Partial Fit"
        # print self.rho_b.shape
        # print self.Ep.shape
        # print self._scale.shape
        # print np.sum(self.Et[istart:iend], axis=0, keepdims=True).shape
        # print self.gamma_p.shape
        # print self.rho_p.shape
        temp = (1 - self.rho) * self.rho_b.T + self.rho * \
            (self.Ep.T + self._scale * np.sum(self.Et[istart:iend].T, axis=1, keepdims=False))
        self.rho_b = temp.T
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)
        # print self.rho_p.shape
        # print np.sum(self.Eb, axis=0, keepdims=True).T.shape
        self.rho_p = (1 - self.rho)*self.rho_p + self.rho*((1.0*self.c_prime / self.d_prime) + np.sum(self.Eb, axis=0, keepdims=True))

        # print self.gamma_p.shape
        # print self.rho_p.shape
        self.Ep, self.Elogp = _compute_expectations(self.gamma_p, self.rho_p)

        return self

    def transform(self, X, istart, iend):      #only for Stochastic PMF
        '''Encode the data as a linear combination of the latent components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)

        attr: string
            The name of attribute, default 'Eb'. Can be changed to Elogb to
            obtain E_q[log beta] as transformed data.

        Returns
        -------
        X_new : array-like, shape(n_samples, n_filters)
            Transformed data, as specified by attr.
        '''

        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        self.Et[istart:iend], self.Elogt[istart:iend] = _compute_expectations(self.gamma_t[istart:iend], self.rho_t[istart:iend])
        # self.c = 1. / np.mean(self.Et[istart:iend])
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        self._update(X, istart, iend)
        return self


    def _update(self, X, istart, iend):
        # alternating between update latent components and weights
        old_bd = -np.inf
        for i in xrange(self.max_iter):
            self._update_theta(X, istart, iend)
            self._update_zai(X, istart, iend)
            bound = self._bound(X, istart, iend)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                sys.stdout.write('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
                sys.stdout.flush()
            if improvement < self.tol:
                break
            old_bd = bound
        if self.verbose:
            sys.stdout.write('\n')
        pass

    def set_learning_rate(self, iter=None, rho=None):
        '''Set the learning rate for the gradient step

        Parameters
        ----------
        iter : int
            The current iteration, used to compute a Robbins-Monro type
            learning rate
        rho : float
            Directly specify the learning rate. Will override the one computed
            from the current iteration.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        if rho is not None:
            self.rho = rho
        elif iter is not None:
            self.rho = (iter + self.t0)**(-self.rho)
        else:
            raise ValueError('invalid learning rate.')
        return self

    def _stoch_bound(self, X, istart, iend):
        bound = np.sum(X * np.log(self._xexplog(istart, iend)) - self.Et[istart:iend].dot(self.Eb))
        bound += _gamma_term(self.a, self.Ea[istart:iend], self.gamma_t[istart:iend], self.rho_t[istart:iend],
                             self.Et[istart:iend], self.Elogt[istart:iend])
        # bound += self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound *= self._scale
        bound += _gamma_term(self.c, self.Ep, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound

    def _xexplog(self, istart, iend):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt[istart:iend]), np.exp(self.Elogb))

    def _update_theta(self, X, istart, iend):
        ratio = X / self._xexplog(istart, iend)
        self.gamma_t[istart:iend] = self.a + np.exp(self.Elogt[istart:iend]) * np.dot(ratio, np.exp(self.Elogb).T)
        # self.rho_t[istart:iend] = self.a * self.c + np.sum(self.Eb, axis=1)
        # self.rho_t[istart:iend] = self.Ea[istart:iend] * self.c + np.sum(self.Eb, axis=1)
        self.rho_t[istart:iend] = self.Ea[istart:iend] + np.sum(self.Eb, axis=1)
        # print self.Et[istart:iend].shape
        self.Et[istart:iend], self.Elogt[istart:iend] = _compute_expectations(self.gamma_t[istart:iend], self.rho_t[istart:iend])
        # print self.Et[istart:iend].shape
        # self.c = 1. / np.mean(self.Et[istart:iend])

    def _update_zai(self, X, istart, iend):
        ratio = X / self._xexplog(istart, iend)
        # self.gamma_a[istart:iend] = self.a_prime + self.n_components*self.a
        # self.rho_t[istart:iend] = self.a * self.c + np.sum(self.Eb, axis=1)
        # print self.rho_a[istart:iend].shape
        # print self.rho_a[istart:iend]
        # print np.sum(self.Et[istart:iend], axis=1).shape
        # print np.sum(self.Et[istart:iend], axis=1) 
        # temp = self.Et[istart:iend]
        # temp =temp.reshape((iend-istart,1))
        self.rho_a[istart:iend] = (1.0*self.a_prime / self.c_prime) + np.sum(self.Et[istart:iend], axis=1).reshape((iend-istart, 1))
        self.Ea[istart:iend], self.Eloga[istart:iend] = _compute_expectations(self.gamma_a, self.rho_a[istart:iend])
        # self.c = 1. / np.mean(self.Et[istart:iend])

    def _bound(self, X, istart, iend):
        bound = np.sum(X * np.log(self._xexplog(istart, iend)) - self.Et[istart:iend].dot(self.Eb))
        bound += _gamma_term(self.a, self.Ea[istart:iend],
                             self.gamma_t[istart:iend], self.rho_t[istart:iend],
                             self.Et[istart:iend], self.Elogt[istart:iend])
        # bound += self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound += _gamma_term(self.c, self.Ep, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound

def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))