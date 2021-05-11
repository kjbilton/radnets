from torch import nn

activations = {'relu' : nn.ReLU, 'tanh' : nn.Tanh}
recurrent_layers = {'rnn' : nn.RNN, 'lstm' : nn.LSTM, 'gru' : nn.GRU}
