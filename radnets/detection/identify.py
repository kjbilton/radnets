"""
Tools for performing identification for a single spectrum
(feedforward), or a set of spectra (recurrent).
"""
import numpy as np
import torch
import torch.nn.functional as F


from .tools import _preprocess


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


def compute_id_thresholds(model, data_loader, far):
    """
    Computes a threshold for each source individually.
    """
    predictions = []

    for X, y in data_loader:
        X = X.float()
        X = X.to(model.device)
        yhat = F.softmax(model(X), dim=1)
        yhat = yhat.detach().cpu().numpy()
        predictions.append(yhat)

    predictions = np.vstack(predictions)[:, 1:]
    predictions_sorted = np.sort(predictions, axis=0)[::-1]
    n_fa = int(len(predictions_sorted) * far / (y.shape[1] - 1))
    thresh = predictions_sorted[n_fa]
    return thresh, predictions


def recurrent_id_threshold(model, data_loader, far):
    """
    Computes threshold for recurrent ID models
    """
    predictions = []

    for X, y, lens in data_loader:
        X = X.float().to(model.device)

        # Perform inference
        Yhat = model(X, lens).squeeze()
        Yhat = F.softmax(Yhat, dim=1).detach().cpu().numpy()
        predictions.append(Yhat)

    predictions = np.vstack(predictions)[:, 1:]
    predictions_sorted = np.sort(predictions, axis=0)[::-1]

    n_fa = int(len(predictions_sorted) * far / (y.shape[2] - 1))
    thresh = predictions_sorted[n_fa]
    return thresh, predictions
