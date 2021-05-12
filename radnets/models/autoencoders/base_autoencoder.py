"""
Base class for all autoencoder models.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.view import View
from ...utils.config import get_filename
from ...utils.constants import activations
from ...training.early_stopping import EarlyStopping


class BaseAutoencoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        device = params['device']
        self.device = torch.device(device)
        self.n_bins = params['spectral']['n_bins']
        self.l1_lambda = params['training']['l1_lambda']
        self.preprocess = self.params['training']['preprocess']
        assert self.preprocess in ['none', 'standardize', 'log']

        # Set loss function
        loss = params['training']['loss']
        assert loss in ['poisson', 'mse']

        if loss == 'poisson':
            if self.preprocess == 'none':
                self.loss = self.poisson_loss
            elif self.preprocess == 'log':
                self.loss = self.poisson_log_loss
            elif self.preprocess == 'standardize':
                self.loss = self.poisson_std_loss
        else:
            self.loss = self.mse_loss

        # Determine if it's a convnet
        _l = [_layer['layer_type'] == 'convolutional'
              for v in params['architecture'].values() for _layer in v]
        self.convnet = any(_l)

        # Determine if it's a fully-convolutional network
        self.fcn = all(_l)

        # Build encoder and decoder modules
        assert all([x in params['architecture']
                    for x in ['encoder', 'decoder']])
        self.encoder = self.build_encoder(params)
        self.decoder = self.build_decoder(params)

    def save_model(self, path):
        torch.save(self, path)

    def load_weights(self):
        weights_path = get_filename(self.params, 'training')
        weights = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(weights)

    def encode(self, X):
        """
        Encode spectra.

        X : torch.Tensor
            Pytorch tensor of the shape (n_spectra, n_bins) of spectra to be
            encoded.
        """
        # Format the data in case it's a convnet and needs a channel dimension
        if self.convnet:
            # Add an empty channel, yielding shape (n_samples, 1, n_bins)
            X = X[None, :].transpose(0, 1)
        return self.encoder(X)

    def decode(self, X):
        """
        Decode spectra.
        """
        # Squeeze to convert shape (n_samples,1,n_bins) to (n_samples,n_bins)
        return self.decoder(X).squeeze()

    def get_n_params(self):
        """
        Get the number of parameters in the model.
        """
        n_params = 0

        for param in list(self.parameters()):
            num = 1
            for param_size in list(param.size()):
                num *= param_size
            n_params += num
        return n_params

    ####################################################################
    # Internal methods
    ####################################################################
    def build_encoder(self, params):
        """
        Build the encoder submodule from the specified parameters
        """
        # Container for encoder operations
        modules = []

        # Container for input sizes to each encoder layer
        n_features = params['spectral']['n_bins']
        input_sizes = [n_features]

        # Container for number of input channels to each encoder layer
        input_channels = [1]

        # Iterate over all layers in the encoder
        encoder_params = params['architecture']['encoder']
        for idx, layer in enumerate(encoder_params):

            # Determine if data needs to be flattened (conv -> dense)
            if idx > 0:
                if (encoder_params[idx - 1]['layer_type'] == 'convolutional') \
                  and (layer['layer_type'] == 'dense'):
                    modules.append(nn.Flatten())

            # Add dense layer
            if layer['layer_type'] == 'dense':
                n_nodes_out = layer['n_nodes_out']
                _layer = nn.Linear(input_sizes[idx], n_nodes_out,
                                   layer['bias'])
                modules.append(_layer)
                # Update the current number of features
                n_features = n_nodes_out

                # Update number of channels (for compatibility with Conv1d)
                input_channels.append(1)

            # Add convolutional layer and pooling
            elif layer['layer_type'] == 'convolutional':
                n_kernels = layer['n_kernels_out']
                kernel_size = layer['kernel_size']
                padding = (kernel_size - 1) // 2
                bias = layer['bias']

                conv = nn.Conv1d(in_channels=input_channels[idx],
                                 out_channels=n_kernels,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 bias=bias)
                modules.append(conv)

                # Add pooling
                pool_size = layer['pool_size']
                pool = nn.MaxPool1d(pool_size)
                modules.append(pool)

                # Update the current number of features (width * depth)
                n_features //= pool_size

                if idx != len(encoder_params) - 1:
                    if encoder_params[idx + 1]['layer_type'] == 'dense':
                        # Flatten features
                        n_features *= n_kernels

                # Update number of channels
                input_channels.append(n_kernels)

            # Update the number of features in each layer
            input_sizes.append(n_features)

            # Add the activation function
            if 'activation' in layer.keys():
                modules.append(activations[layer['activation']]())

            # Add batchnorm
            if 'batchnorm' in layer.keys():
                if layer['batchnorm']:
                    modules.append(nn.BatchNorm1d(layer['n_nodes_out']))

            # Add dropout
            if 'dropout' in layer.keys():
                if layer['dropout']:
                    modules.append(nn.Dropout(p=layer['dropout']))

        # Set the dimensionality of the choke point
        self.ndof = int(n_features * input_channels[-1])
        self.input_sizes = input_sizes
        self.feature_size_min = int(n_features)

        return nn.Sequential(*modules)

    def build_decoder(self, params):
        """
        Build the decoder submodule from the specified parameters
        """
        modules = []

        decoder_params = params['architecture']['decoder']

        # Iterate over all layers in the encoder
        for idx, layer in enumerate(decoder_params):

            # Add dense layer
            if layer['layer_type'] == 'dense':
                # Output size = input size to next layer
                if idx != len(decoder_params) - 1:
                    _next_layer_type = decoder_params[idx+1]['layer_type']
                    if _next_layer_type == 'dense':
                        output_size = decoder_params[idx+1]['n_nodes_in']
                    elif _next_layer_type == 'convolutional':
                        n_features = self.input_sizes[::-1][idx+1]
                        output_size = n_features
                else:
                    output_size = params['spectral']['n_bins']
                _layer = nn.Linear(layer['n_nodes_in'], output_size,
                                   layer['bias'])
                modules.append(_layer)

                # Reshape into shape needed for deconvolution
                if idx != len(decoder_params) - 1:
                    if decoder_params[idx+1]['layer_type'] == 'convolutional':
                        n_kernels_in = decoder_params[idx+1]['n_kernels_in']
                        n_features //= n_kernels_in
                        shape = (-1, n_kernels_in, n_features)
                        modules.append(View(shape))

            # Add convolutional layer and pooling
            elif layer['layer_type'] == 'convolutional':
                n_kernels = layer['n_kernels_in']
                pool_size = layer['pool_size']
                bias = layer['bias']

                if idx != len(decoder_params) - 1:
                    n_kernels_out = decoder_params[idx+1]['n_kernels_in']
                else:
                    n_kernels_out = 1

                conv = nn.ConvTranspose1d(in_channels=n_kernels,
                                          out_channels=n_kernels_out,
                                          kernel_size=pool_size,
                                          stride=pool_size,
                                          bias=bias)
                modules.append(conv)

            # Add the activation function
            if 'activation' in layer.keys():
                modules.append(activations[layer['activation']]())

            # Add batchnorm
            if 'batchnorm' in layer.keys():
                if layer['batchnorm']:
                    modules.append(nn.BatchNorm1d(output_size))

        return nn.Sequential(*modules)

    def setup_early_stopping(self, name):
        patience = self.params['training']['early_stopping']['patience']
        delta = self.params['training']['early_stopping']['delta']
        return EarlyStopping(name=name, patience=patience, verbose=False,
                             delta=delta)

    def set_standardize_params(self, loaders):
        self.mu = loaders['training'].dataset.mu
        self.sigma = loaders['training'].dataset.sigma
        self.sigma_tensor = torch.tensor(self.sigma).to(self.device)
        self.mu_tensor = torch.tensor(self.mu).to(self.device)

    ####################################################################
    # Loss functions
    ####################################################################
    def poisson_loss(self, Xhat, X, complete=True, eps=1.E-7):
        Xhat = F.relu(Xhat) + eps

        # Compute Poisson loss
        _loss = (Xhat - X * torch.log(Xhat))

        # Include factorial term
        if complete:
            _loss += torch.lgamma(X + 1)
        _loss = _loss.sum()

        # Add sparsity regularization
        if self.l1_lambda > 0.:
            for param in self.parameters():
                _loss += self.l1_lambda * torch.norm(param, p=1)
        return _loss

    def poisson_log_loss(self, Xhat, X, complete=True, eps=1.E-7):
        # Transform input and output spectra to count space
        Xhat = F.relu(torch.exp(Xhat) - 1)
        X = F.relu(torch.exp(X) - 1)
        return self.poisson_loss(Xhat, X)

    def poisson_std_loss(self, Xhat, X, complete=True, eps=1.E-7):
        # Transform input and output spectra to count space
        Xhat = F.relu(Xhat * self.sigma_tensor + self.mu_tensor)
        X = F.relu(X * self.sigma_tensor + self.mu_tensor)
        return self.poisson_loss(Xhat, X)

    def mse_loss(self, Xhat, X, complete=True, eps=1.E-7):
        _loss = F.mse_loss(Xhat, X, reduction='sum')
        # Add sparsity regularization
        if self.l1_lambda > 0.:
            for param in self.parameters():
                _loss += self.l1_lambda * torch.norm(param, p=1)
        return _loss
