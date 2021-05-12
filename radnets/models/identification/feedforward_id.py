import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ...data.preprocess import _preprocess
from ...training.early_stopping import EarlyStopping
from .base_id import BaseModel


class FeedForwardID(BaseModel):
    """
    """
    def __init__(self, params):
        super().__init__(params)
        self.front_end = self._build_front_end(params)
        self.rear_end = self._build_rear_end(params)

    def train_and_validate(self, loaders, optimizer, name=None,
                           scheduler=None):
        """
        loaders : dict, keys = ['training', 'validation']
            Dictorary of DataLoaders for training and validation data.
        """
        if name is None:
            name = ''

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
                for X, y in loaders[mode]:

                    # Transfer data to device (e.g., GPU)
                    X = X.float().to(self.device)
                    y = y.to(self.device)

                    # Perform inference
                    yhat = self(X)

                    # Compute loss
                    loss = self.loss(yhat, y)
                    total_loss += loss.item()
                    n_samples += len(X)

                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
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

        self.eval()

    def forward(self, X):
        if self.convnet:
            # Add an empty channel, yielding shape (n_samples, 1, n_bins)
            X = X[None, :].transpose(0, 1)
        X = self.front_end(X)
        return self.rear_end(X)

    def identify(self, X):
        # Prepare spectra for model
        X = _preprocess(self, X, self.preprocess)
        X = torch.tensor(X).float().to(self.device)

        # Perform inference
        Yhat = self.forward(X)
        Yhat = F.softmax(Yhat, dim=1).detach().cpu().numpy()

        # Post-processing to get single predictions
        predictions = np.zeros(len(Yhat))
        outputs_norm = Yhat[:, 1:] / self.thresholds
        alarm_mask = (outputs_norm > 1).any(axis=1)
        predictions[alarm_mask] = outputs_norm[alarm_mask].argmax(axis=1) + 1
        return predictions

    def compute_thresholds(self, data_loader, far):
        predictions = []

        for X, y in data_loader:
            X = X.float()
            X = X.to(self.device)
            yhat = F.softmax(self.forward(X), dim=1)
            yhat = yhat.detach().cpu().numpy()
            predictions.append(yhat)

        predictions = np.vstack(predictions)[:, 1:]
        predictions_sorted = np.sort(predictions, axis=0)[::-1]
        n_fa = int(len(predictions_sorted) * far / (y.shape[1] - 1))
        thresh = predictions_sorted[n_fa]
        self.thresholds = thresh
        return predictions
