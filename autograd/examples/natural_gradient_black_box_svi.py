from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from .. import numpy as np
from ..scipy.stats import norm as norm
from ..misc.optimizers import adam, sgd
from black_box_svi import black_box_variational_inference
if __name__ == '__main__':
    np.random.seed(42)
    obs_dim = 20
    Y = np.random.randn(obs_dim, obs_dim).dot(np.random.randn(obs_dim))

    def log_density(x, t):
        (mu, log_sigma) = (x[:, :obs_dim], x[:, obs_dim:])
        sigma_density = np.sum(norm.logpdf(log_sigma, 0, 1.35), axis=1)
        mu_density = np.sum(norm.logpdf(Y, mu, np.exp(log_sigma)), axis=1)
        return sigma_density + mu_density
    D = obs_dim * 2
    (objective, gradient, unpack_params) = black_box_variational_inference(log_density, D, num_samples=2000)

    def fisher_diag(lam):
        (mu, log_sigma) = unpack_params(lam)
        return np.concatenate([np.exp(-2.0 * log_sigma), np.ones(len(log_sigma)) * 2])
    natural_gradient = lambda lam, i: 1.0 / fisher_diag(lam) * gradient(lam, i)

    def optimize_and_lls(optfun):
        num_iters = 200
        elbos = []

        def callback(params, t, g):
            elbo_val = -objective(params, t)
            elbos.append(elbo_val)
            if t % 50 == 0:
                print('Iteration {} lower bound {}'.format(t, elbo_val))
        init_mean = -1 * np.ones(D)
        init_log_std = -5 * np.ones(D)
        init_var_params = np.concatenate([init_mean, init_log_std])
        variational_params = optfun(num_iters, init_var_params, callback)
        return np.array(elbos)
    elbo_lists = []
    step_sizes = [0.1, 0.25, 0.5]
    for step_size in step_sizes:
        optfun = lambda n, init, cb: adam(gradient, init, step_size=step_size, num_iters=n, callback=cb)
        standard_lls = optimize_and_lls(optfun)
        optnat = lambda n, init, cb: sgd(natural_gradient, init, step_size=step_size, num_iters=n, callback=cb, mass=0.001)
        natural_lls = optimize_and_lls(optnat)
        elbo_lists.append((standard_lls, natural_lls))
    plt.figure(figsize=(12, 8))
    colors = ['b', 'k', 'g']
    for (col, ss, (stand_lls, nat_lls)) in zip(colors, step_sizes, elbo_lists):
        plt.plot(np.arange(len(stand_lls)), stand_lls, '--', label='standard (adam, step-size = %2.2f)' % ss, alpha=0.5, c=col)
        plt.plot(np.arange(len(nat_lls)), nat_lls, '-', label='natural (sgd, step-size = %2.2f)' % ss, c=col)
    llrange = natural_lls.max() - natural_lls.min()
    plt.ylim((natural_lls.max() - llrange * 0.1, natural_lls.max() + 10))
    plt.xlabel('optimization iteration')
    plt.ylabel('ELBO')
    plt.legend(loc='lower right')
    plt.title('%d dimensional posterior' % D)
    plt.show()