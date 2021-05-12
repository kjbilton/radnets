from torch import nn


EPS = 1.E-7


activations = {'relu': nn.ReLU, 'tanh': nn.Tanh}
recurrent_layers = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
