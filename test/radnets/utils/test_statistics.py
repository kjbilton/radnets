import numpy as np

import radnets.utils.statistics as stats


def test_compute_deviance():
    X = np.array([[1, 2, 3]])
    Xhat = np.array([[1.1, 1.9, 3.2]])
    deviance_true = 0.0273
    deviance = stats.compute_deviance(X, Xhat)
    assert np.isclose(deviance, deviance_true, atol=1.E-4)
