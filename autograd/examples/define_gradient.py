"""This example shows how to define the gradient of your own functions.
This can be useful for speed, numerical stability, or in cases where
your code depends on external library calls."""
from __future__ import absolute_import
from __future__ import print_function
from .. import numpy as np
from ..numpy import random as npr
from .. import grad
from ..extend import primitive, defvjp
from ..test_util import check_grads

@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x))), also defined in scipy.special"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def logsumexp_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))
defvjp(logsumexp, logsumexp_vjp)
if __name__ == '__main__':

    def example_func(y):
        z = y ** 2
        lse = logsumexp(z)
        return np.sum(lse)
    grad_of_example = grad(example_func)
    print('Gradient: \n', grad_of_example(npr.randn(10)))
    check_grads(example_func, modes=['rev'])(npr.randn(10))