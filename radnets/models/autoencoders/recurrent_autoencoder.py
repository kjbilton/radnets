import torch
from torch import nn
from torch import optim
from .base_autoencoder import BaseAutoencoder
from ..utils.constants import recurrent_layers


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
        # Grab dimensions for reshaping later
        batch_size, run_len, n_bins = X.size()

        # Put runs from all batch in a single dimension
        if self.convnet:
            X = X.view(batch_size * run_len, 1, n_bins)
        else:
            X = X.view(-1, n_bins)

        # Encode using convolutional features
        X = self.encoder(X)

        # Flatten features into single feature vectors for each run/spectrum
        X = X.view(batch_size, run_len, -1)

        # Apply dropout
        # X = self.recurrent_dropout(X)

        # Pack to pass to recurrent layer
        X = nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True,
                                              enforce_sorted=False)

        # Recurrent layer
        X, _ = self.recurrent(X)

        # Unpack
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # Reshape for decoding
        if not self.fcn:
            X = X.view(-1, self.ndof)
        else:
            X = X.view(batch_size * run_len, self.n_kernels_inner,
                       self.feature_size_min)

        # Decoder
        X = self.decode(X)

        # Reshape for turning back into runs
        X = X.view(batch_size, run_len, -1)
        return X

    def _build_recurrent(self, params):
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
