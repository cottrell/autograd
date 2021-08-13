from __future__ import absolute_import
from __future__ import print_function
from .. import numpy as np
from .. import value_and_grad
from scipy.optimize import minimize

def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
rosenbrock_with_grad = value_and_grad(rosenbrock)
result = minimize(rosenbrock_with_grad, x0=np.array([0.0, 0.0]), jac=True, method='CG')
print('Found minimum at {0}'.format(result.x))