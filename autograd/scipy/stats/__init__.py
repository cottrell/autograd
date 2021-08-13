from __future__ import absolute_import
from . import chi2
from . import beta
from . import gamma
from . import norm
from . import poisson
from . import t
try:
    from . import multivariate_normal
except AttributeError:
    pass
try:
    from . import dirichlet
except AttributeError:
    pass