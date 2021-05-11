import numpy as np
from scipy.special import xlogy, gammaln

import poisson_gof as pg

def compute_aic(Xhat, X, n_params):
    """
    Compute Akaike Information Criterion (AIC) between data `X` and model
    `Xhat`.
    """
    loss = poisson_nll(Xhat, X)
    return 2 * n_params + 2 * loss.sum()

def poisson_nll(Xhat, X, complete=True, eps=1e-7):
    """
    Poisson negative log-likelihood
    """
    X, Xhat = X + eps, Xhat + eps
    loss = Xhat - xlogy(X, Xhat)
    if complete:
        loss += gammaln(X + 1)
    return loss
