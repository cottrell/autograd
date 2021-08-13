from __future__ import absolute_import
from builtins import range
import scipy.integrate
from .. import numpy as np
from ..extend import primitive, defvjp_argnums
from .. import make_vjp
from ..misc import flatten
from ..builtins import tuple
odeint = primitive(scipy.integrate.odeint)

def grad_odeint(yt, func, y0, t, func_args, **kwargs):
    (T, D) = np.shape(yt)
    (flat_args, unflatten) = flatten(func_args)

    def flat_func(y, t, flat_args):
        return func(y, t, *unflatten(flat_args))

    def unpack(x):
        return (x[0:D], x[D:2 * D], x[2 * D], x[2 * D + 1:])

    def augmented_dynamics(augmented_state, t, flat_args):
        (y, vjp_y, _, _) = unpack(augmented_state)
        (vjp_all, dy_dt) = make_vjp(flat_func, argnum=(0, 1, 2))(y, t, flat_args)
        (vjp_y, vjp_t, vjp_args) = vjp_all(-vjp_y)
        return np.hstack((dy_dt, vjp_y, vjp_t, vjp_args))

    def vjp_all(g):
        vjp_y = g[-1, :]
        vjp_t0 = 0
        time_vjp_list = []
        vjp_args = np.zeros(np.size(flat_args))
        for i in range(T - 1, 0, -1):
            vjp_cur_t = np.dot(func(yt[i, :], t[i], *func_args), g[i, :])
            time_vjp_list.append(vjp_cur_t)
            vjp_t0 = vjp_t0 - vjp_cur_t
            aug_y0 = np.hstack((yt[i, :], vjp_y, vjp_t0, vjp_args))
            aug_ans = odeint(augmented_dynamics, aug_y0, np.array([t[i], t[i - 1]]), tuple((flat_args,)), **kwargs)
            (_, vjp_y, vjp_t0, vjp_args) = unpack(aug_ans[1])
            vjp_y = vjp_y + g[i - 1, :]
        time_vjp_list.append(vjp_t0)
        vjp_times = np.hstack(time_vjp_list)[::-1]
        return (None, vjp_y, vjp_times, unflatten(vjp_args))
    return vjp_all

def argnums_unpack(all_vjp_builder):

    def build_selected_vjps(argnums, ans, combined_args, kwargs):
        vjp_func = all_vjp_builder(ans, *combined_args, **kwargs)

        def chosen_vjps(g):
            all_vjps = vjp_func(g)
            return [all_vjps[argnum] for argnum in argnums]
        return chosen_vjps
    return build_selected_vjps
defvjp_argnums(odeint, argnums_unpack(grad_odeint))