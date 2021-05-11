import pytest
from torch import nn
from radnets.models import FeedForwardID
from radnets.utils.config import load_config


@pytest.fixture
def data():
    config = load_config('config/id_example.yaml')
    model = FeedForwardID(config)
    return {'model': model, 'config': config}


def test_init(data):
    model = data['model']
    assert isinstance(model.front_end, nn.Sequential)
    assert isinstance(model.rear_end, nn.Sequential)


def test_architecture(data):
    model = data['model']
    config = data['config']

    arch = config['architecture']
    assert model.rear_end[-1].out_features \
        == arch['rear_end'][-1]['n_nodes_out']
