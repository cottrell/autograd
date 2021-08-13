from __future__ import absolute_import
import scipy.stats
from ... import numpy as np
from ...numpy.numpy_vjps import unbroadcast_f
from ...extend import primitive, defvjp
pdf = primitive(scipy.stats.multivariate_normal.pdf)
logpdf = primitive(scipy.stats.multivariate_normal.logpdf)
entropy = primitive(scipy.stats.multivariate_normal.entropy)

def generalized_outer_product(x):
    if np.ndim(x) == 1:
        return np.outer(x, x)
    return np.matmul(x, np.swapaxes(x, -1, -2))

def covgrad(x, mean, cov, allow_singular=False):
    if allow_singular:
        raise NotImplementedError('The multivariate normal pdf is not differentiable w.r.t. a singular covariance matix')
    J = np.linalg.inv(cov)
    solved = np.matmul(J, np.expand_dims(x - mean, -1))
    return 1.0 / 2 * (generalized_outer_product(solved) - J)

def solve(allow_singular):
    if allow_singular:
        return lambda A, x: np.dot(np.linalg.pinv(A), x)
    else:
        return np.linalg.solve
defvjp(logpdf, lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(x, lambda g: -np.expand_dims(np.atleast_1d(g), 1) * solve(allow_singular)(cov, (x - mean).T).T), lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(mean, lambda g: np.expand_dims(np.atleast_1d(g), 1) * solve(allow_singular)(cov, (x - mean).T).T), lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(cov, lambda g: np.reshape(g, np.shape(g) + (1, 1)) * covgrad(x, mean, cov, allow_singular)))
defvjp(pdf, lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(x, lambda g: -np.expand_dims(np.atleast_1d(ans * g), 1) * solve(allow_singular)(cov, (x - mean).T).T), lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(mean, lambda g: np.expand_dims(np.atleast_1d(ans * g), 1) * solve(allow_singular)(cov, (x - mean).T).T), lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(cov, lambda g: np.reshape(ans * g, np.shape(g) + (1, 1)) * covgrad(x, mean, cov, allow_singular)))
defvjp(entropy, None, lambda ans, mean, cov: unbroadcast_f(cov, lambda g: 0.5 * g * np.linalg.inv(cov).T))