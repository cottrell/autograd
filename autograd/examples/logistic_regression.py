from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from .. import numpy as np
from .. import grad
from ..test_util import check_grads

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)

def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))
inputs = np.array([[0.52, 1.12, 0.77], [0.88, -1.08, 0.15], [0.52, 0.06, -1.3], [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])
training_gradient_fun = grad(training_loss)
weights = np.array([0.0, 0.0, 0.0])
check_grads(training_loss, modes=['rev'])(weights)
print('Initial loss:', training_loss(weights))
for i in range(100):
    weights -= training_gradient_fun(weights) * 0.01
print('Trained loss:', training_loss(weights))