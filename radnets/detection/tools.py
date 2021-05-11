from scipy.special import xlogy, gammaln
from radnets.utils.constants import EPS
from ..data.preprocess import PREPROCESS, INVERSE_PREPROCESS


def _preprocess(model, X, preprocess):
    if preprocess in ['standardize', 'mean_center']:
        Xhat = PREPROCESS[preprocess](X, model.mu, model.sigma)
    else:
        Xhat = PREPROCESS[preprocess](X)
    return Xhat


def _inv_preprocess(model, X, Xhat, preprocess):
    if preprocess in ['standardize', 'mean_center']:
        Xhat = INVERSE_PREPROCESS[preprocess](Xhat, model.mu, model.sigma)
    elif preprocess in ['normlog']:
        Xhat = INVERSE_PREPROCESS[preprocess](Xhat, X)
    else:
        Xhat = INVERSE_PREPROCESS[preprocess](Xhat)
    return Xhat


def compute_aic(Xhat, X, n_params):
    """
    Compute Akaike Information Criterion (AIC) between data `X` and model
    `Xhat`.
    """
    loss = poisson_nll(Xhat, X)
    return 2 * n_params + 2 * loss.sum()


def poisson_nll(Xhat, X, complete=True, eps=EPS):
    """
    Poisson negative log-likelihood
    """
    X, Xhat = X + eps, Xhat + eps
    loss = Xhat - xlogy(X, Xhat)
    if complete:
        loss += gammaln(X + 1)
    return loss


def get_deviance(X, Xhat, eps=EPS):
    """
    Poisson deviance.
    """
    return 2 * (Xhat - X + xlogy(X, X/(Xhat + EPS))).sum()
