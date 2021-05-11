import numpy as np
import pytest
from torch.utils.data import Dataset
from radnets.data import FeedforwardDataset


@pytest.fixture
def data():
    n_spectra = 100
    n_bins = 128
    spectra = np.ones((n_spectra, n_bins))
    indices = np.zeros(n_spectra)
    dataset = FeedforwardDataset(spectra, indices)
    values = {'dataset': dataset, 'n_spectra': n_spectra, 'n_bins': n_bins}
    return values


def test_dataset(data):
    assert isinstance(data['dataset'], Dataset)
    assert len(data['dataset']) == data['n_spectra']
