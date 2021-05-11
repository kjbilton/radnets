
"""
Base class for identification pytorch models.
"""
import torch
from torch import nn
import torch.nn.functional as F

from radnets.models.tools.constants import activations

class BaseModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.device = torch.device('cuda' if params['device'] == 'gpu'
                                    else 'cpu')

        self.n_bins = params['spectral']['n_bins']
        self.preprocess = self.params['training']['preprocess']

        if 'clip_grad' in self.params['training']:
            self.clip_grad = self.params['training']['clip_grad']
        else:
            self.clip_grad = 0.

        # Define loss function
        loss = params['training']['loss']
        assert loss in ['ce', 'kl', 'mse']
        if loss == 'ce':
            self.loss = self.ce_loss
        elif loss == 'kl':
            self.loss = self.kl_loss
        else:
            self.loss = self.mse_loss

        # Determine if it's a convnet
        _l = [l['layer_type'] == 'convolutional'
              for l in params['architecture']['front_end']]
        self.convnet = True if any(_l) else False

        # Determine if it's a fully-convolutional network
        self.fcn = True if all(_l) else False

    def get_n_params(self):
        """
        Get the number of parameters in the model.
        """
        n_params = 0

        for param in list(self.parameters()):
            num = 1
            for param_size in list(param.size()):
                num *=  param_size
            n_params += num

        return n_params

    def kl_loss(self, Yhat, Y):
        Yhat = F.softmax(Yhat, dim=1)
        l = F.kl_div(Yhat, Y, reduction='sum')
        return l

    def ce_loss(self, Yhat, Y):
        """
        Loss used when the output is the fraction of spectrum associated with each source type.
        """
        Yhat = F.softmax(Yhat, dim=1)
        # l =  F.binary_cross_entropy(Yhat, Y, weight=self.weight,
        #                             reduction='sum')

        l =  F.binary_cross_entropy(Yhat, Y, reduction='sum')
        return l


    def mse_loss(self, Yhat, Y):
        """
        Loss used when the output is the fraction of spectrum associated with each source type.
        """
        Yhat = F.softmax(Yhat, dim=1)
        l =  F.mse_loss(Yhat, Y, reduction='sum')
        return l


    def _build_front_end(self, params):
        """
        Build the network from the specified parameters
        """
        arch_params = params['architecture']['front_end']
        n_features = params['spectral']['n_bins']
        input_sizes = [n_features]
        input_channels = [1]
        modules = []

        for idx, layer in enumerate(arch_params):
            type = layer['layer_type']
            bias = layer['bias']

            if type == 'dense':
                n_nodes_out = layer['n_nodes_out']
                l = nn.Linear(input_sizes[idx], n_nodes_out, bias)
                modules.append(l)

                n_features = n_nodes_out
                input_channels.append(1)

            # Add convolutional layer and pooling
            elif type == 'convolutional':
                n_kernels = layer['n_kernels_out']
                kernel_size = layer['kernel_size']
                padding = (kernel_size - 1) // 2

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

                # Update the current number of features
                n_features //= pool_size
                if idx == len(arch_params) - 1:
                    n_features *= n_kernels
                elif arch_params[idx + 1]['layer_type'] == 'dense':
                    n_features *= n_kernels

                # Update number of channels
                input_channels.append(n_kernels)

            # Update the number of features in each layer
            input_sizes.append(n_features)

            # Add batchnorm
            if 'batchnorm' in layer.keys():
                if layer['batchnorm']:
                    if  type == 'dense':
                        bn_size = layer['n_nodes_out']
                    else:
                        bn_size = layer['n_kernels_out']
                    modules.append(nn.BatchNorm1d(bn_size))

            # Add the activation function
            if 'activation' in layer.keys():
                modules.append(activations[layer['activation']]())

            # Determine if data needs to be flattened (conv -> dense)
            if type == 'convolutional':
                if idx == len(arch_params) - 1:
                    modules.append(nn.Flatten())
                elif arch_params[idx + 1]['layer_type'] == 'dense':
                    modules.append(nn.Flatten())

            # Add dropout
            if 'dropout' in layer.keys():
                if layer['dropout']:
                    modules.append(nn.Dropout(p=layer['dropout']))

        self.input_sizes = input_sizes
        return nn.Sequential(*modules)

    def _build_rear_end(self, params):
        """
        Builds the portion of the network following the recurrent layer,
        ostensibly where the classification occurs.
        """
        arch_params = params['architecture']['rear_end']
        input_sizes = [self.input_sizes[-1]]
        modules = []

        for idx, layer in enumerate(arch_params):

            n_nodes_out = layer['n_nodes_out']
            l = nn.Linear(input_sizes[idx], n_nodes_out, layer['bias'])
            input_sizes.append(n_nodes_out)
            modules.append(l)

            # Add the activation function
            if 'activation' in layer.keys():
                modules.append(activations[layer['activation']]())

        return nn.Sequential(*modules)
