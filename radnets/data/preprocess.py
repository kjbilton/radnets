import numpy as np

from ..utils.constants import EPS


def standardize_preprocess(X, mu, sigma):
    """
    Standardize the input data `X` by mean subtraction and scaling by the
    variance.
    """
    return (X - mu) / (sigma + EPS)


def inv_standardize_preprocess(Xprime, mu, sigma):
    """
    Inverse of standardization: transforms a standarized spectrum using
    mean and variance.
    """
    return Xprime * (sigma + EPS) + mu


def log_preprocess(X):
    """
    Log-preprocessing of a spectrum.
    """
    return np.log(X + 1)


def inv_log_preprocess(Xprime):
    """
    Inverse of log-preprocessing.
    """
    return np.exp(Xprime) - 1


def norm_log_preprocess(X):
    """
    Normalized log preprocessing.
    """
    A = X + 1
    if len(X.shape) > 1:
        B = A.sum(axis=1)[:, None]
    else:
        B = A.sum()
    C = np.log(A / B)
    if len(X.shape) > 1:
        D = - C.min(axis=1)[:, None]
    else:
        D = - C.min()
    Xprime = C / D + 1
    return Xprime


def inv_norm_log_preprocess(Xprime, X):
    """
    Inverse of normalized log preprocessing.
    """
    A = X + 1
    if len(X.shape) > 1:
        B = A.sum(axis=1)[:, None]
    else:
        B = A.sum()
    C = np.log(A / B)
    if len(X.shape) > 1:
        D = - C.min(axis=1)[:, None]
    else:
        D = - C.min()
    Cprime = (Xprime - 1) * D
    Aprime = np.exp(Cprime) * B
    Xhat = Aprime - 1
    return Xhat


PREPROCESS = {'none': lambda x: x,
              'standardize': standardize_preprocess,
              'mean_center': standardize_preprocess,
              'log': log_preprocess,
              'normlog': norm_log_preprocess}


INVERSE_PREPROCESS = {'none': lambda x: x,
                      'standardize': inv_standardize_preprocess,
                      'mean_center': inv_standardize_preprocess,
                      'log': inv_log_preprocess,
                      'normlog': inv_norm_log_preprocess}


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
