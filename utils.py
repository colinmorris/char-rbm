import numpy as np

# Adapted from sklearn.utils.extmath.softmax
def softmax_and_sample(X, copy=True):
    """
    Given an array of 2-d arrays, each having shape (M, N) representing M softmax
    units with N possible values each, return an array of the same shape where
    each N-dimensional inner array has a 1 at one index, and zero everywhere
    else. The 1 is assigned according to the corresponding softmax probabilities
    (i.e. np.exp(X) / np.sum(np.exp(X)) )
    
    Parameters
    ----------
    X: array-like, shape (n_samples, M, N), dtype=float
        Argument to the logistic function
    copy: bool, optional
        Copy X or not.
    Returns
    -------
    out: array of 0,1, shape (n_samples, M, N)
        Softmax function evaluated at every point in x and sampled
    """
    if copy:
        X = np.copy(X)
    X_shape = X.shape
    a, b, c = X_shape
    # This will cause overflow when large values are exponentiated.
    # Hence the largest value in each row is subtracted from each data
    max_prob = np.max(X, axis=2).reshape((X.shape[0], X.shape[1], 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=2).reshape((X.shape[0], X.shape[1], 1))
    X /= sum_prob

    # We've got our probabilities, now sample from them
    thresholds = np.random.rand(X.shape[0], X.shape[1], 1)
    cumsum = np.cumsum(X, axis=2, out=X)
    x, y, z = np.indices(cumsum.shape)
    # This relies on the fact that, if there are multiple instances of the max
    # value in an array, argmax returns the index of the first one
    to_select = np.argmax(cumsum > thresholds, axis=2).reshape(a, b, 1)
    bin_sample = np.zeros(X_shape)
    bin_sample[x, y, to_select] = 1

    return bin_sample
