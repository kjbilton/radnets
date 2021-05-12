"""
Tools for performing binary anomaly detection for a single spectrum
(feedforward), or a set of spectra (recurrent).
"""
import numpy as np
import torch
from .tools import compute_deviance
from ..utils.constants import EPS
from ..data.preprocess import _preprocess, _inv_preprocess


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
    deviance = compute_deviance(X, Xhat)
    return int(any(deviance > model.threshold))


def recurrent_deviance_threshold(model, data_loader, far, preprocess):
    """
    Computes threshold for recurrent deviance-based models
    """
    deviance = []

    for X, X_lens in data_loader:
        # Prepare spectra for model
        Xhat = X.float().numpy()
        Xhat = _preprocess(model, X, preprocess)
        Xhat = torch.tensor(Xhat).float()

        Xhat = model(Xhat.to(model.device), X_lens).detach().cpu().numpy() + \
            EPS
        X = X.numpy()

        # Reshape
        X = np.vstack([X[idx][:X_lens[idx]] for idx in range(len(X))])
        Xhat = np.vstack([Xhat[idx][:X_lens[idx]] for idx in range(len(Xhat))])

        # Inverse preprocessing
        Xhat = _inv_preprocess(model, X, Xhat, preprocess)
        Xhat = np.maximum(Xhat, EPS)

        # Compute deviance
        dev = compute_deviance(X, Xhat)
        deviance.append(dev)

    deviance = np.hstack(deviance)

    n_fa = np.floor(len(deviance) * far).astype(int)
    deviance_sorted = np.sort(deviance)[::-1]
    thresh = deviance_sorted[n_fa]
    return thresh, deviance
