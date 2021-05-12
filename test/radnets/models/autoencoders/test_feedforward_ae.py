import os
import numpy as np
import pytest
import torch
from torch import nn

from radnets.models import FeedforwardAutoencoder
from radnets.utils.config import load_config


@pytest.fixture
def data():
    config = load_config('data/cae_example.yaml')
    model = FeedforwardAutoencoder(config)
    model.threshold = 0
    return {'model': model, 'config': config}


def test_init(data):
    model = data['model']
    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, nn.Sequential)


def test_architecture(data):
    model = data['model']
    config = data['config']

    arch = config['architecture']
    assert model.encoder[-2].out_features == arch['encoder'][-1]['n_nodes_out']
    assert model.decoder[0].in_features == arch['decoder'][0]['n_nodes_in']


def test_save_and_load(data):
    model = data['model']
    threshold = model.threshold
    filename = 'test/test.pth'
    model.save_model(filename)

    loaded_model = torch.load(filename)
    assert isinstance(loaded_model, FeedforwardAutoencoder)
    assert loaded_model.threshold == threshold
    os.remove(filename)


def test_load_weights(data):
    model = data['model']
    default_parameters = model.parameters()
    model.load_weights()
    assert model.parameters() != default_parameters


def test_encode(data):
    model = data['model']
    model.load_weights()
    config = data['config']
    n_spectra = 1
    latent_dimension = config['architecture']['encoder'][-1]['n_nodes_out']
    n_bins = 128
    x = torch.ones((n_spectra, n_bins))
    encoding = model.encode(x)
    assert encoding.shape == (1, latent_dimension)


def test_decode(data):
    model = data['model']
    model.load_weights()
    config = data['config']
    n_spectra = 1
    latent_dimension = config['architecture']['encoder'][-1]['n_nodes_out']
    n_bins = 128
    encoding = torch.ones((n_spectra, latent_dimension))
    reconstruction = model.decode(encoding)
    assert reconstruction.shape[0] == n_bins


def test_detect(data):
    model = data['model']
    model.load_weights()

    n_bins = 128
    x = np.ones((1, n_bins))
    mu = np.ones(n_bins)
    sigma = np.ones(n_bins)
    model.mu = mu
    model.sigma = sigma

    detection = model.detect(x)
    assert detection
