from torch import nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, X):
        return X.view(*self.shape)
