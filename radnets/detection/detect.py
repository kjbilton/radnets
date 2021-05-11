"""
Tools for performing binary anomaly detection for a single spectrum
(feedforward), or a set of spectra (recurrent).
"""
import numpy as np
import torch

from .tools import _preprocess, _inv_preprocess

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
