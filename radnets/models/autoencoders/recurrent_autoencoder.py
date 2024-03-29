import numpy as np
import torch
from torch import nn
from torch import optim

from .base_autoencoder import BaseAutoencoder
from ...data.preprocess import _preprocess, _inv_preprocess
from ...utils.constants import EPS, recurrent_layers
from ...utils.statistics import compute_deviance


class RecurrentAutoencoder(BaseAutoencoder):
    def __init__(self, params):
        super().__init__(params)

        assert 'recurrent' in params['architecture']
        self.recurrent = self._build_recurrent(params)

    def train_and_validate(self, loaders, optimizer, name=None,
                           scheduler=None):
        """
        loaders : dict, keys = ['training', 'validation']
            Dictorary of DataLoaders for training and validation data.
        optimizer :
            pytorch optimizer
        """
        if name is None:
            name = ''

        if self.params['training']['preprocess'] == 'standardize':
            self.set_standardize_params(loaders)
        early_stopping = self.setup_early_stopping(name)

        # Iterate over max number of epochs
        n_epochs = self.params['training']['n_epochs']
        for epoch in range(n_epochs):
            msg = f'Epoch {str(epoch).zfill(3)}. '

            # Switch between training and validation
            for mode in ['training', 'validation']:

                if mode == 'training':
                    self.train()
                else:
                    self.eval()

                # Track loss over data partition
                total_loss = 0.
                n_samples = 0

                # Iterate over mini batches within loader
                for X, X_lens in loaders[mode]:

                    # Transfer data to device (e.g., GPU)
                    X = X.float().to(self.device)

                    # Perform inference -- estimate the background
                    Xhat = self(X, X_lens)

                    # Unpad
                    X = torch.cat([X[idx][:X_lens[idx]]
                                  for idx in range(len(X))])
                    Xhat = torch.cat([Xhat[idx][:X_lens[idx]]
                                     for idx in range(len(Xhat))])

                    # Compute loss
                    loss = self.loss(Xhat, X)
                    total_loss += loss.item()
                    n_samples += sum(X_lens)

                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                mean_loss = total_loss / n_samples / self.n_bins
                msg += f'Mean {mode} loss: {mean_loss:.7f}. '

                if mode == 'validation':
                    print(msg)
                    early_stopping(mean_loss, self)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        # Load best model
                        fname = f'checkpoint_{name}.pt'
                        self.load_state_dict(torch.load(fname))
                        self.eval()
                        return

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    prev_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(mean_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr < prev_lr:
                        print(f'Learning rate reduced to {current_lr}')
                else:
                    scheduler.step()

        # Put the model back in evaluation mode
        self.eval()

    def forward(self, X, X_lens):
        """
        X :
            tensor of runs
        X_lens :
            Contains length of each run
        """
        batch_size, run_len, n_bins = X.size()

        # Put runs from all batch in a single dimension
        if self.convnet:
            X = X.view(batch_size * run_len, 1, n_bins)
        else:
            X = X.view(-1, n_bins)
        X = self.encoder(X)

        # Flatten features into single feature vectors for each run/spectrum
        X = X.view(batch_size, run_len, -1)

        X = nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True,
                                              enforce_sorted=False)
        X, _ = self.recurrent(X)
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # Reshape for decoding
        if not self.fcn:
            X = X.view(-1, self.ndof)
        else:
            X = X.view(batch_size * run_len, self.n_kernels_inner,
                       self.feature_size_min)

        X = self.decode(X)

        # Reshape back into runs
        X = X.view(batch_size, run_len, -1)
        return X

    def detect(self, X):
        Xhat = _preprocess(self, X, self.preprocess)
        Xhat = torch.tensor(Xhat).float().to(self.device)
        Xhat = Xhat.unsqueeze(0)
        X_lens = [Xhat.shape[1]]

        # Perform inference
        Xhat = self.forward(Xhat, X_lens).detach().cpu().squeeze().numpy()
        X = X.astype(float)

        # Inverse preprocessing
        Xhat = _inv_preprocess(self, X, Xhat, self.preprocess)
        Xhat = np.maximum(Xhat, EPS)

        # Perform detection
        deviance = compute_deviance(Xhat, X)
        return int(any(deviance > self.threshold))

    def recurrent_deviance_threshold(self, data_loader, far):
        """
        Computes threshold for recurrent deviance-based models
        """
        data_loader.dataset.preprocess = 'none'

        deviance = []

        for X, X_lens in data_loader:
            Xhat = X.float().numpy()
            Xhat = _preprocess(self, X, self.preprocess)
            Xhat = torch.tensor(Xhat).float().to(self.device)

            Xhat = self.forward(Xhat, X_lens)
            Xhat = Xhat.detach().cpu().numpy() + EPS
            X = X.numpy()

            # Reshape
            X = np.vstack([X[idx][:X_lens[idx]] for idx in range(len(X))])
            Xhat = np.vstack([Xhat[idx][:X_lens[idx]]
                              for idx in range(len(Xhat))])

            # Inverse preprocessing
            Xhat = _inv_preprocess(self, X, Xhat, self.preprocess)
            Xhat = np.maximum(Xhat, EPS)

            # Compute deviance
            dev = compute_deviance(Xhat, X)
            deviance.append(dev)

        deviance = np.hstack(deviance)

        n_fa = np.floor(len(deviance) * far).astype(int)
        deviance_sorted = np.sort(deviance)[::-1]
        thresh = deviance_sorted[n_fa]
        return thresh, deviance

    def _build_recurrent(self, params):
        """Build recurrent module that goes between encoder and decoder."""
        params = params['architecture']['recurrent'][0]
        rnn_type = params['rnn_type']
        activation = params['activation']
        assert rnn_type in recurrent_layers

        n_nodes_out = params['n_nodes_out']
        input_size = self.input_sizes[-1]

        if rnn_type == 'rnn':
            _layer = recurrent_layers[rnn_type](
                input_size, n_nodes_out, bias=params['bias'],
                num_layers=params['num_layers'], nonlinearity=activation)
        else:
            _layer = recurrent_layers[rnn_type](
                input_size, n_nodes_out, num_layers=params['num_layers'],
                bias=params['bias'])

        self.input_sizes.append(n_nodes_out)

        # Update the current number of features
        self.ndof = n_nodes_out
        self.feature_size_min = n_nodes_out
        return _layer
