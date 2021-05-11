"""
Tools for performing identification for a single spectrum
(feedforward), or a set of spectra (recurrent).
"""
import numpy as np
import torch

from .tools import _preprocess, _inv_preprocess

def feedforward_regression_id(model, X, preprocess):
    """
    Detect anomalous spectra using a FeedForwardID.
    """
    # Prepare spectra for model
    X = _preprocess(model, X, preprocess)
    X = torch.tensor(X).float().to(model.device)

    # Perform inference
    Yhat = model(X)
    Yhat = F.softmax(Yhat, dim=1).detach().cpu().numpy()

    # Post-processing to get single predictions
    predictions = np.zeros(len(Yhat))
    outputs_norm = Yhat[:, 1:] / model.thresholds
    alarm_mask = (outputs_norm > 1).any(axis=1)
    predictions[alarm_mask] = outputs_norm[alarm_mask].argmax(axis=1) + 1
    return predictions

def recurrent_regression_id(model, X, preprocess):
    """
    Detect anomalous spectra using a RecurrentID.
    """
    # Prepare spectra for model
    X = _preprocess(model, X, preprocess)
    X = torch.tensor(X).float().to(model.device)
    X = X.unsqueeze(0)
    X_lens = [X.shape[1]]

    # Perform inference
    Yhat = model(X, X_lens).squeeze()
    Yhat = F.softmax(Yhat, dim=1).detach().cpu().numpy()

    # Post-processing to get single predictions
    predictions = np.zeros(len(Yhat))
    outputs_norm = Yhat[:, 1:] / model.thresholds
    alarm_mask = (outputs_norm > 1).any(axis=1)
    predictions[alarm_mask] = outputs_norm[alarm_mask].argmax(axis=1) + 1
    return predictions
