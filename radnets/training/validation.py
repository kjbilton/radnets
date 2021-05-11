import numpy as np
import torch.nn.functional as F

def predict_regression(model, thresholds, data_loader):
    """
    Returns
    -------
    predictions
    labels
    """
    # Containers for output
    predictions = []
    labels = []
    
    for X, y in data_loader:

        # Run inference
        X = X.to(model.device)
        outputs = model(X)
        outputs = F.softmax(outputs, dim=1).detach().cpu().numpy()

        # Make predictions
        _predictions = np.zeros(len(outputs))
        outputs_norm = outputs[:, 1:] / thresholds
        alarm_mask = (outputs_norm > 1).any(axis=1)

        _predictions[alarm_mask] = outputs_norm[alarm_mask].argmax(axis=1) + 1
        predictions.append(_predictions)

        # Ground truth
        y = y.numpy()
        _labels = np.zeros(len(y))
        src_mask = y[:, 0] != 1
        _labels[src_mask] = y[src_mask][:, 1:].argmax(axis=1) + 1
        labels.append(_labels)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    return predictions, labels

def predict_single_label(model, data_loader):
    """
    Returns
    -------
    predictions
    labels
    """
    # Containers for output
    predictions = []
    labels = []
    
    for X, y in data_loader:

        # Run inference
        X = X.to(model.device)
        outputs = model(X)
        outputs = F.softmax(outputs, dim=1).detach().cpu().numpy()

        # Make predictions
        _predictions = outputs.argmax(axis=1)
        predictions.append(_predictions)

        # Ground truth
        labels.append(y.numpy())

    predictions = np.hstack(predictions)
    labels = np.array(labels).flatten()
    return predictions, labels

def generate_regression_confusion_matrix(model, thresholds, data_loader, normed=False):
    """
    Predict data and produce a confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    
    predictions, labels = predict_regression(model, thresholds, data_loader)

    cm = confusion_matrix(labels, predictions)
    if normed:
        cm = cm / cm.sum(axis=1)[:, None]
    return cm

def generate_single_label_confusion_matrix(model, data_loader, normed=False):
    """
    Predict data and produce a confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    
    predictions, labels = predict_single_label(model, data_loader)

    cm = confusion_matrix(labels, predictions)
    if normed:
        cm = cm / cm.sum(axis=1)[:, None]
    return cm