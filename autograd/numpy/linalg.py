from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from ..extend import defvjp
wrap_namespace(npla.__dict__, globals())

def T(x):
    return anp.swapaxes(x, -1, -2)
_dot = partial(anp.einsum, '...ij,...jk->...ik')
_diag = lambda a: anp.eye(a.shape[-1]) * a

def _matrix_diag(a):
    reps = anp.array(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))

def add2d(x):
    return anp.reshape(x, anp.shape(x) + (1, 1))
defvjp(det, lambda ans, x: lambda g: add2d(g) * add2d(ans) * T(inv(x)))
defvjp(slogdet, lambda ans, x: lambda g: add2d(g[1]) * T(inv(x)))

def grad_inv(ans, x):
    return lambda g: -_dot(_dot(T(ans), g), T(ans))
defvjp(inv, grad_inv)

def grad_pinv(ans, x):
    return lambda g: T(-_dot(_dot(ans, T(g)), ans) + _dot(_dot(_dot(ans, T(ans)), g), anp.eye(x.shape[-2]) - _dot(x, ans)) + _dot(_dot(_dot(anp.eye(ans.shape[-2]) - _dot(ans, x), g), T(ans)), ans))
defvjp(pinv, grad_pinv)

def grad_solve(argnum, ans, a, b):
    updim = lambda x: x if x.ndim == a.ndim else x[..., None]
    if argnum == 0:
        return lambda g: -_dot(updim(solve(T(a), g)), T(updim(ans)))
    else:
        return lambda g: solve(T(a), g)
defvjp(solve, partial(grad_solve, 0), partial(grad_solve, 1))

def grad_norm(ans, x, ord=None, axis=None):

    def check_implemented():
        matrix_norm = x.ndim == 2 and axis is None or isinstance(axis, tuple)
        if matrix_norm:
            if not (ord is None or ord == 'fro' or ord == 'nuc'):
                raise NotImplementedError('Gradient of matrix norm not implemented for ord={}'.format(ord))
        elif not (ord is None or ord > 1):
            raise NotImplementedError('Gradient of norm not implemented for ord={}'.format(ord))
    if axis is None:
        expand = lambda a: a
    elif isinstance(axis, tuple):
        (row_axis, col_axis) = axis
        if row_axis > col_axis:
            row_axis = row_axis - 1
        expand = lambda a: anp.expand_dims(anp.expand_dims(a, row_axis), col_axis)
    else:
        expand = lambda a: anp.expand_dims(a, axis=axis)
    if ord == 'nuc':
        if axis is None:
            roll = lambda a: a
            unroll = lambda a: a
        else:
            (row_axis, col_axis) = axis
            if row_axis > col_axis:
                row_axis = row_axis - 1
            roll = lambda a: anp.rollaxis(anp.rollaxis(a, col_axis, a.ndim), row_axis, a.ndim - 1)
            unroll = lambda a: anp.rollaxis(anp.rollaxis(a, a.ndim - 2, row_axis), a.ndim - 1, col_axis)
    check_implemented()

    def vjp(g):
        if ord in (None, 2, 'fro'):
            return expand(g / ans) * x
        elif ord == 'nuc':
            x_rolled = roll(x)
            (u, s, vt) = svd(x_rolled, full_matrices=False)
            uvt_rolled = _dot(u, vt)
            uvt = unroll(uvt_rolled)
            g = expand(g)
            return g * uvt
        else:
            return expand(g / ans ** (ord - 1)) * x * anp.abs(x) ** (ord - 2)
    return vjp
defvjp(norm, grad_norm)

def grad_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a symmetric matrix."""
    N = x.shape[-1]
    (w, v) = ans
    vc = anp.conj(v)

    def vjp(g):
        (wg, vg) = g
        w_repeated = anp.repeat(w[..., anp.newaxis], N, axis=-1)
        vjp_temp = _dot(vc * wg[..., anp.newaxis, :], T(v))
        if anp.any(vg):
            off_diag = anp.ones((N, N)) - anp.eye(N)
            F = off_diag / (T(w_repeated) - w_repeated + anp.eye(N))
            vjp_temp += _dot(_dot(vc, F * _dot(T(v), vg)), T(v))
        reps = anp.array(x.shape)
        reps[-2:] = 1
        if UPLO == 'L':
            tri = anp.tile(anp.tril(anp.ones(N), -1), reps)
        elif UPLO == 'U':
            tri = anp.tile(anp.triu(anp.ones(N), 1), reps)
        return anp.real(vjp_temp) * anp.eye(vjp_temp.shape[-1]) + (vjp_temp + anp.conj(T(vjp_temp))) * tri
    return vjp
defvjp(eigh, grad_eigh)

def grad_eig(ans, x):
    """Gradient of a general square (complex valued) matrix"""
    (e, u) = ans
    n = e.shape[-1]

    def vjp(g):
        (ge, gu) = g
        ge = _matrix_diag(ge)
        f = 1 / (e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1e-20)
        f -= _diag(f)
        ut = anp.swapaxes(u, -1, -2)
        r1 = f * _dot(ut, gu)
        r2 = -f * _dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut, gu)) * anp.eye(n))
        r = _dot(_dot(inv(ut), ge + r1 + r2), ut)
        if not anp.iscomplexobj(x):
            r = anp.real(r)
        return r
    return vjp
defvjp(eig, grad_eig)

def grad_cholesky(L, A):
    solve_trans = lambda a, b: solve(T(a), b)
    phi = lambda X: anp.tril(X) / (1.0 + anp.eye(X.shape[-1]))

    def conjugate_solve(L, X):
        return solve_trans(L, T(solve_trans(L, T(X))))

    def vjp(g):
        S = conjugate_solve(L, phi(anp.einsum('...ki,...kj->...ij', L, g)))
        return (S + T(S)) / 2.0
    return vjp
defvjp(cholesky, grad_cholesky)

def grad_svd(usv_, a, full_matrices=True, compute_uv=True):

    def vjp(g):
        usv = usv_
        if not compute_uv:
            s = usv
            usv = svd(a, full_matrices=False)
            u = usv[0]
            v = anp.conj(T(usv[2]))
            return _dot(anp.conj(u) * g[..., anp.newaxis, :], T(v))
        elif full_matrices:
            raise NotImplementedError('Gradient of svd not implemented for full_matrices=True')
        else:
            u = usv[0]
            s = usv[1]
            v = anp.conj(T(usv[2]))
            (m, n) = a.shape[-2:]
            k = anp.min((m, n))
            i = anp.reshape(anp.eye(k), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (k, k))))
            f = 1 / (s[..., anp.newaxis, :] ** 2 - s[..., :, anp.newaxis] ** 2 + i)
            gu = g[0]
            gs = g[1]
            gv = anp.conj(T(g[2]))
            utgu = _dot(T(u), gu)
            vtgv = _dot(T(v), gv)
            t1 = f * (utgu - anp.conj(T(utgu))) * s[..., anp.newaxis, :]
            t1 = t1 + i * gs[..., :, anp.newaxis]
            t1 = t1 + s[..., :, anp.newaxis] * (f * (vtgv - anp.conj(T(vtgv))))
            if anp.iscomplexobj(u):
                t1 = t1 + 1j * anp.imag(_diag(utgu)) / s[..., anp.newaxis, :]
            t1 = _dot(_dot(anp.conj(u), t1), T(v))
            if m < n:
                i_minus_vvt = anp.reshape(anp.eye(n), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (n, n)))) - _dot(v, anp.conj(T(v)))
                t1 = t1 + anp.conj(_dot(_dot(u / s[..., anp.newaxis, :], T(gv)), i_minus_vvt))
                return t1
            elif m == n:
                return t1
            elif m > n:
                i_minus_uut = anp.reshape(anp.eye(m), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (m, m)))) - _dot(u, anp.conj(T(u)))
                t1 = t1 + T(_dot(_dot(v / s[..., anp.newaxis, :], T(gu)), i_minus_uut))
                return t1
    return vjp
defvjp(svd, grad_svd)