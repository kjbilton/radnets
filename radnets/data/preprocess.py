import numpy as np

EPS = 1e-7

def standardize_preprocess(X, mu, sigma):
    return (X - mu) / (sigma + EPS)

def log_preprocess(X):
    return np.log(X + 1)

def norm_log_preprocess(X):
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

# Inverse transforms
def inv_standardize_preprocess(Xprime, mu, sigma):
    return Xprime * (sigma + EPS) + mu

def inv_log_preprocess(Xprime):
    return np.exp(Xprime) - 1

def inv_norm_log_preprocess(Xprime, X):
    # Compute normalization terms for inverse
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
    # Invert
    Cprime = (Xprime - 1) * D
    Aprime = np.exp(Cprime) * B
    Xhat = Aprime - 1

    return Xhat

PREPROCESS = {'none' : lambda x : x,
              'standardize' : standardize_preprocess,
              'log' :log_preprocess,
              'normlog' : norm_log_preprocess}

INVERSE_PREPROCESS = {'none' : lambda x : x,
                      'standardize' : inv_standardize_preprocess,
                      'log' :inv_log_preprocess,
                      'normlog' : inv_norm_log_preprocess}
