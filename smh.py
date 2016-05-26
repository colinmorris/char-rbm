import numpy as np

def softmax(X, copy=True):
    """
    Calculate the softmax function.
    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)
    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.
    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function
    copy: bool, optional
        Copy X or not.
    Returns
    -------
    out: array, shape (M, N)
        Softmax function evaluated at every point in x
    """
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X
    
def softmax_and_sample(X, copy=True):
    """
    Calculate the softmax function.
    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)
    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.
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
    to_select = np.argmax(cumsum > thresholds, axis=2).reshape(a,b,1)
    bin_sample = np.zeros(X_shape)
    bin_sample[x,y,to_select] = 1
    
    return bin_sample
