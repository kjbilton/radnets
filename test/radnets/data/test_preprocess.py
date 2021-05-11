import numpy as np
import pytest
from radnets.data import preprocess as pp


@pytest.fixture
def data():
    x = np.ones(128)
    return x


def test_standardize_constant(data):
    standarized = pp.standardize_preprocess(data, 1, 0)
    assert np.all(standarized == 0)


def test_standardize_reconstruction(data):
    x = data
    mu = data.mean()
    sigma = data.std()
    xprime = pp.standardize_preprocess(x, mu, sigma)
    xhat = pp.inv_standardize_preprocess(xprime, mu, sigma)
    assert np.allclose(x, xhat)
