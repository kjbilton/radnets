import numpy as np

EPS = 1e-7

def standardize_preprocess(X, mu, sigma):
    return (X - mu) / sigma

def log_preprocess(X):
    return np.log(X + 1)

def norm_preprocess(X):
    if len(X.shape) > 1:
        return X / X.sum(axis=1)[:, None]
    return X / X.sum()

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
    return Xprime * sigma + mu

def inv_log_preprocess(Xprime):
    return np.exp(Xprime) - 1

def inv_norm_preprocess(Xprime, X):
    if len(X.shape) > 1:
        return Xprime * X.sum(axis=1)[:, None]
    return Xprime * X.sum()

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
