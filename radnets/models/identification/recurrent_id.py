import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from .base_id import BaseModel
from ...data.preprocess import _preprocess
from ...training.early_stopping import EarlyStopping
from ...utils.constants import recurrent_layers


class RecurrentID(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.front_end = self._build_front_end(params)
        self.recurrent, self.recurrent_post = self._build_recurrent(params)
        self.rear_end = self._build_rear_end(params)

    def train_and_validate(self, loaders, optimizer, filename=None, name=None,
                           scheduler=None):
        """
        loaders : dict, keys = ['training', 'validation']
            Dictorary of DataLoaders for training and validation data.
        scheduler : torch.optim.lr_scheduler
        """
        if name is None:
            name = ''

        if filename is not None:
            assert isinstance(filename, str)

        if loaders['training'].dataset.preprocess == 'standardize':
            self.mu = loaders['training'].dataset.mu
            self.sigma = loaders['training'].dataset.sigma

        patience = self.params['training']['early_stopping']['patience']
        delta = self.params['training']['early_stopping']['delta']
        early_stopping = EarlyStopping(name=name, patience=patience,
                                       verbose=False, delta=delta)

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
                for (X, y, lens) in loaders[mode]:

                    # Transfer data to device (e.g., GPU)
                    X = X.float().to(self.device)
                    y = y.to(self.device)

                    # Perform inference
                    yhat = self(X, lens)

                    X = torch.cat([X[idx][:lens[idx]]
                                  for idx in range(len(X))])
                    y = torch.cat([y[idx][:lens[idx]]for idx in range(len(y))])
                    yhat = torch.cat([yhat[idx][:lens[idx]]
                                      for idx in range(len(yhat))])

                    # Compute loss
                    loss = self.loss(yhat, y)
                    total_loss += loss.item()
                    n_samples += len(X)

                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        if self.clip_grad != 0:
                            nn.utils.clip_grad_norm_(self.parameters(),
                                                     self.clip_grad)
                        optimizer.step()

                mean_loss = total_loss / n_samples
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

        if filename is not None:
            torch.save(self.state_dict(), filename)

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
        X = self.front_end(X)

        # Flatten features into single feature vectors for each run/spectrum
        X = X.view(batch_size, run_len, -1)

        # Pack to pass to recurrent layer
        X = nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True,
                                              enforce_sorted=False)

        # Recurrent layer
        X, _ = self.recurrent(X)

        # Unpack
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = self.recurrent_post(X)

        X = X.view(-1, self.input_sizes[-1])
        X = self.rear_end(X)

        # Reshape for turning back into runs
        X = X.view(batch_size, run_len, -1)

        return X

    def identify(self, X):
        """
        Detect anomalous spectra using a RecurrentID.
        """
        # Prepare spectra for model
        X = _preprocess(self, X, self.preprocess)
        X = torch.tensor(X).float().to(self.device)
        X = X.unsqueeze(0)
        X_lens = [X.shape[1]]

        # Perform inference
        Yhat = self.forward(X, X_lens).squeeze()
        Yhat = F.softmax(Yhat, dim=1).detach().cpu().numpy()

        # Post-processing to get single predictions
        predictions = np.zeros(len(Yhat))
        outputs_norm = Yhat[:, 1:] / self.thresholds
        alarm_mask = (outputs_norm > 1).any(axis=1)
        predictions[alarm_mask] = outputs_norm[alarm_mask].argmax(axis=1) + 1
        return predictions

    def compute_thresholds(self, data_loader, far):
        """
        Computes threshold for recurrent ID models
        """
        predictions = []

        for X, y, lens in data_loader:
            X = X.float().to(self.device)

            # Perform inference
            Yhat = self.forward(X, lens).squeeze()
            Yhat = F.softmax(Yhat, dim=1).detach().cpu().numpy()
            predictions.append(Yhat)

        predictions = np.vstack(predictions)[:, 1:]
        predictions_sorted = np.sort(predictions, axis=0)[::-1]

        n_fa = int(len(predictions_sorted) * far / (y.shape[2] - 1))
        thresh = predictions_sorted[n_fa]
        self.thresholds = thresh
        return predictions

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

        # Add dropout
        modules = []
        if 'dropout' in params.keys():
            if params['dropout']:
                modules.append(nn.Dropout(p=params['dropout']))

        return _layer, nn.Sequential(*modules)
