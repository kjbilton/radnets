import numpy as np
import torch
from torch import optim
from ...data.preprocess import _preprocess, _inv_preprocess
from ...utils.constants import EPS
from ...detection.tools import compute_deviance
from .base_autoencoder import BaseAutoencoder


class FeedforwardAutoencoder(BaseAutoencoder):
    def __init__(self, params):
        super().__init__(params)

    def train_and_validate(self, loaders, optimizer, name=None,
                           scheduler=None):
        """
        loaders : dict, keys = ['training', 'validation']
            Dictorary of DataLoaders for training and validation data.
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
                for X in loaders[mode]:

                    # Transfer data to device (e.g., GPU)
                    X = X.float().to(self.device)

                    # Perform inference
                    Xhat = self(X)

                    # Compute loss
                    loss = self.loss(Xhat, X)
                    total_loss += loss.item()
                    n_samples += len(X)

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

    def forward(self, X):
        X = self.encode(X)
        X = self.decode(X)
        return X

    def detect(self, X):
        # Prepare data for inference
        Xhat = _preprocess(self, X, self.preprocess)
        Xhat = torch.tensor(Xhat).float().to(self.device)

        # Predict reconstruction
        Xhat = self.forward(Xhat).detach().cpu().numpy() + EPS

        # Inverse preprocessing
        X = X.astype(float) + EPS
        Xhat = _inv_preprocess(self, X, Xhat, self.preprocess)
        Xhat = np.maximum(Xhat, EPS)

        # Perform detection
        deviance = compute_deviance(X, Xhat)
        return int(any(deviance > self.threshold))

    def compute_threshold(self, data_loader, far):
        """
        Compute detection threshold.

        Parameters
        ----------
        data_loader
            pytorch data loader containing background data to evaluate on.
        far : float
            False alarm rate in units of inverse seconds.

        Returns
        -------
        threshold : float
        deviance : numpy array
        """
        data_loader.dataset.preprocess = 'none'

        deviance = []

        for X in data_loader:
            # Prepare spectra for model
            Xhat = X.float().numpy()
            Xhat = _preprocess(self, X, self.preprocess)
            Xhat = torch.tensor(Xhat).float().to(self.device)

            # Run inference
            Xhat = self.forward(Xhat)
            Xhat = Xhat.detach().cpu().numpy() + EPS
            X = X.numpy() + EPS

            # Inverse preprocessing
            Xhat = _inv_preprocess(self, X, Xhat, self.preprocess)
            Xhat = np.maximum(Xhat, EPS)

            # Compute p-values for Poisson deviance
            _deviance = compute_deviance(X, Xhat)
            deviance.append(_deviance)

        deviance = np.hstack(deviance)

        # Compute threshold
        n_fa = np.floor(len(deviance) * far).astype(int)
        deviance_sorted = np.sort(deviance)[::-1]
        thresh = deviance_sorted[n_fa]

        data_loader.dataset.preprocess = self.preprocess
        self.threshold = thresh
        return deviance
