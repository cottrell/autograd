from __future__ import absolute_import
from .. import numpy as np
import matplotlib.pyplot as plt
from .. import elementwise_grad as egrad
"\nMathematically we can only take gradients of scalar-valued functions, but\nautograd's elementwise_grad function also handles numpy's familiar vectorization\nof scalar functions, which is used in this example.\n\nTo be precise, elementwise_grad(fun)(x) always returns the value of a\nvector-Jacobian product, where the Jacobian of fun is evaluated at x and the\nvector is an all-ones vector with the same size as the output of fun. When\nvectorizing a scalar-valued function over many arguments, the Jacobian of the\noverall vector-to-vector mapping is diagonal, and so this vector-Jacobian\nproduct simply returns the diagonal elements of the Jacobian, which is the\n(elementwise) gradient of the function at each input value over which the\nfunction is vectorized.\n"

def tanh(x):
    return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))
x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x), x, egrad(tanh)(x), x, egrad(egrad(tanh))(x), x, egrad(egrad(egrad(tanh)))(x), x, egrad(egrad(egrad(egrad(tanh))))(x), x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x), x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))
plt.axis('off')
plt.savefig('tanh.png')
plt.show()