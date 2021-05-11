import pytest
from torch import nn
from radnets.models import FeedforwardAutoencoder
from radnets.utils.config import load_config


@pytest.fixture
def data():
    config = load_config('config/cae_example.yaml')
    model = FeedforwardAutoencoder(config)
    return {'model': model, 'config': config}


def test_init(data):
    model = data['model']
    config = data['config']

    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, nn.Sequential)

def test_architecture(data):
    model = data['model']
    config = data['config']

    arch = config['architecture']
    assert model.encoder[-2].out_features == arch['encoder'][-1]['n_nodes_out']
    assert model.decoder[0].in_features == arch['decoder'][0]['n_nodes_in']
