"""
Tools for performing binary anomaly detection for a single spectrum
(feedforward), or a set of spectra (recurrent).
"""
import numpy as np
import torch
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

def feedforward_deviance(model, X, preprocess):
    """
    Detect anomalous spectra using a FeedforwardAutoencoder.
    """
    # Prepare spectra for model
    Xhat = _preprocess(model, X, preprocess)
    Xhat = torch.tensor(Xhat).float().to(model.device)

    # Perform inference
    Xhat = model(Xhat).detach().cpu().numpy() + EPS
    X = X.astype(float) + EPS

    # Inverse preprocessing
    Xhat = _inv_preprocess(model, X, Xhat, preprocess)
    Xhat = np.maximum(Xhat, EPS)

    # Perform detection
    deviance = pg.deviance(n=X, lam=Xhat)
    return int(any(deviance > model.threshold))

def recurrent_deviance(model, X, preprocess):
    """
    Detect anomalous spectra using a RecurrentAutoencoder.
    """
    # Prepare spectra for model
    Xhat = _preprocess(model, X, preprocess)
    Xhat = torch.tensor(Xhat).float().to(model.device)
    Xhat = Xhat.unsqueeze(0)
    X_lens = [Xhat.shape[1]]

    # Perform inference
    Xhat = model(Xhat, X_lens).detach().cpu().squeeze().numpy()
    X = X.astype(float)

    # Inverse preprocessing
    Xhat = _inv_preprocess(model, X, Xhat, preprocess)
    Xhat = np.maximum(Xhat, EPS)

    # Perform detection
    deviance = pg.deviance(n=X, lam=Xhat)
    return int(any(deviance > model.threshold))
