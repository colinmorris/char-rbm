"""Restricted Boltzmann Machine with softmax visible units.
Based on sklearn's BernoulliRBM class.
"""

# Authors: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
#          Vlad Niculae
#          Gabriel Synnaeve
#          Lars Buitinck
# License: BSD 3 clause

import time
import re

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot, softmax, log_logistic
from sklearn.utils.fixes import expit             # logistic function
from sklearn.utils.validation import check_is_fitted

from smh import softmax_and_sample
import common


class BernoulliRBM(BaseEstimator, TransformerMixin):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------

    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    intercept_hidden_ : array-like, shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like, shape (n_features,)
        Biases of the visible units.

    components_ : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=0)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """

    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, lr_backoff=False):
        self.n_components = n_components
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_backoff = lr_backoff
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.rng_ = check_random_state(self.random_state)

    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : array, shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self, "components_")

        X = check_array(X, accept_sparse='csr', dtype=np.float)
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)

    def _sample_hiddens(self, v):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        return (self.rng_.random_sample(size=p.shape) < p)

    def _sample_visibles(self, h, sample_max=False):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        sample_max : bool
            Does nothing.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return (self.rng_.random_sample(size=p.shape) < p)

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        return (- safe_sparse_dot(v, self.intercept_visible_)
                - np.logaddexp(0, safe_sparse_dot(v, self.components_.T)
                               + self.intercept_hidden_).sum(axis=1))

    def gibbs(self, v, sample_max=False):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to start from.

        sample_max : bool
            If this RBM uses softmax visible units, then take the unit with
            the highest probability per group, rather than randomly sampling.

        Returns
        -------
        v_new : array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self, "components_")
        h_ = self._sample_hiddens(v)
        v_ = self._sample_visibles(h_, sample_max)

        return v_

    def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.rng_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='fortran')
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X)

    def _fit(self, v_pos):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[self.rng_.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def corrupt(self, v):
        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               self.rng_.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]
        return v_, None

    def uncorrupt(self, visibles, state):
        pass

    @common.timeit
    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, "components_")

        v = check_array(X, accept_sparse='csr')
        fe = self._free_energy(v)

        v_, state = self.corrupt(v)
        # TODO: If I wanted to be really fancy here, I would do one of those "with..." things.
        fe_corrupted = self._free_energy(v)
        self.uncorrupt(v, state)
        return fe.mean(), fe_corrupted.mean()

        # TODO: I don't have a great intuition about this. Why multiply by n_features?
        # The overfitting section of "Practical Guide" just says to compare the
        # average free energy of train and validation and to compare them. Any reason
        # not to just do that here as well? Seems much simpler to interpret. Maybe
        # it's because we're supposed to be dealing with smaller deltas here?
        #return v.shape[1] * log_logistic(fe_corrupted - fe)

    @common.timeit
    def score_validation_data(self, train, validation):
        """Return the energy difference between the given validation data, and a
        subset of the training data. This is useful for monitoring overfitting.
        If the model isn't overfitting, the difference should be around 0. The
        greater the difference, the more the model is overfitting.
        """
        # It's important to use the same subset of the training data every time (per Hinton's "Practical Guide")
        return self._free_energy(train[:validation.shape[0]]).mean(), self._free_energy(validation).mean()

    def fit(self, X, validation=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.

        validation : {array-like, sparse matrix}

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float)
        n_samples = X.shape[0]

        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.rng_.normal(0, 0.01, (self.n_components, X.shape[1])),
                order='fortran')
            self.intercept_hidden_ = np.zeros(self.n_components, )
            #self.intercept_visible_ = np.zeros(X.shape[1], )
            # 'It is usually helpful to initialize the bias of visible unit i to log[p_i/(1-p_i)] where p_i is the prptn of training vectors where i is on' - Practical Guide
            counts = np.sum(X, axis=0)
            # There should be no units that are always on
            assert np.max(counts) < X.shape[0], "Found a visible unit always on in the training data. Fishy."
            # There might be some units never on. Add a pseudo-count of 1 to avoid inf
            vis_priors = (counts + 1) / float(X.shape[0])
            self.intercept_visible = np.log( vis_priors / (1 - vis_priors) )


        else:
            print "Resuing existing weights and biases"
        # Don't necessarily want to reuse h_samples if we have one leftover from before - batch size might have changed
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            if self.lr_backoff:
                # If, e.g., we're doing 10 epochs, use the full learning rate for
                # the first iteration, 90% of the base learning rate for the second
                # iteration... and 10% for the final iteration
                self.learning_rate = (self.n_iter - iteration - 1) / self.n_iter
                print "Using learning rate of {:.3f} (base LR={:.3f})".format(self.learning_rate, self.base_learning_rate)

            for batch_slice in batch_slices:
                self._fit(X[batch_slice])

            if verbose:
                end = time.time()
                self.wellness_check(iteration, end - begin, X, validation)
                begin = end

        return self

    def wellness_check(self, epoch, duration, train, validation):
        """Log some diagnostic information on how the model is doing so far."""
        validation_debug = ''
        if validation is not None:
            v_energy, t_energy = self.score_validation_data(train, validation)
            validation_debug = "\nE(vali):\t{:.2f}\tE(train):\t{:.2f}\tRelative difference: {:.2f}".format(
                t_energy, v_energy, t_energy / v_energy)

        # TODO: This is pretty expensive. Figure out why? Or just do less often.
        # TODO: Maybe some of this information should be attached to self for the
        # sake of pickle archaeology later?
        e_train, e_corrupted = self.score_samples(train)
        print re.sub('\n *', '\n', """[{}] Iteration {}\tt = {:.2f}s
                E(train):\t{:.2f}\tE(corrupt):\t{:.2f}\tRelative difference: {:.2f}{}""".format
                     (type(self).__name__, epoch, duration,
                      e_train, e_corrupted, e_corrupted / e_train, validation_debug,
                      ))


class CharBernoulliRBM(BernoulliRBM):

    def __init__(self, codec, *args, **kwargs):
        """softmax_shape : tuple
        (N, M) where N is the length of character sequences, and M is the number
        of distinct characters. N*M will be the number of visible 
        binary units. (Softmax visible units won't actually be used unless you use the Softmax subclass.)
        """
        self.codec = codec
        self.softmax_shape = codec.shape()
        # Old-style class :(
        BernoulliRBM.__init__(self, *args, **kwargs)

    def wellness_check(self, epoch, duration, train, validation):
        BernoulliRBM.wellness_check(self, epoch, duration, train, validation)
        fantasy_samples = '|'.join([self.codec.decode(vec) for vec in
                                    self._sample_visibles(self.h_samples_[:3], sample_max=True)])
        print "Fantasy samples: {}".format(fantasy_samples)


class CharBernoulliRBMSoftmax(CharBernoulliRBM):

    def _sample_visibles(self, h, sample_max=False):
        """Sample from the distribution P(v|h). This obeys the softmax constraint
        on visible units. i.e. sum(v) == softmax_shape[0] for any visible 
        configuration v.

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        sample_max : bool
            If True, then for each softmax unit, take the value with the highest
            probability, rather than sampling randomly. 

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        nsamples, nfeats = p.shape
        reshaped = np.reshape(p, (nsamples,) + self.softmax_shape)
        return softmax_and_sample(reshaped).reshape((nsamples, nfeats))

    def corrupt(self, v):
        # TODO: Should probably make this available in the parent class as well.
        # So that the two models can be compared on even ground.
        n_softmax, n_opts = self.softmax_shape
        # Select a random index in to the indices of the non-zero values of each input
        # TODO: In the char-RBM case, if I wanted to really challenge the model, I would avoid selecting any
        # trailing spaces here. Cause any dumb model can figure out that it should assign high energy to
        # any instance of /  [^ ]/
        meta_indices_to_corrupt = self.rng_.randint(0, n_softmax, v.shape[0]) + np.arange(0, n_softmax * v.shape[0], n_softmax)

        # Offset these indices by a random amount (but not 0 - we want to actually change them)
        offsets = self.rng_.randint(1, n_opts, v.shape[0])
        # Also, do some math to make sure we don't "spill over" into a different softmax.
        # E.g. if n_opts=5, and we're corrupting index 3, we should choose offsets from {-3, -2, -1, +1}
        # 1-d array that matches with meta_i_t_c but which contains the indices themselves
        indices_to_corrupt = v.indices[meta_indices_to_corrupt]
        # Sweet lucifer
        offsets = offsets - (n_opts * (((indices_to_corrupt % n_opts) + offsets.ravel()) >= n_opts))

        v.indices[meta_indices_to_corrupt] += offsets
        return v, (meta_indices_to_corrupt, offsets)

    def uncorrupt(self, visibles, state):
        mitc, offsets = state
        visibles.indices[mitc] -= offsets
