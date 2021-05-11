import torch
from torch import optim


from radnets.detection.thresholds import feedforward_deviance_threshold
from .base_autoencoder import BaseAutoencoder


class FeedforwardAutoencoder(BaseAutoencoder):
    def __init__(self, params):
        super().__init__(params)

    ####################################################################
    # High-level methods
    ####################################################################
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

    def compute_threshold(self, data_loader, far):
        data_loader.dataset.preprocess = 'none'
        thresh, metrics = feedforward_deviance_threshold(self, data_loader,
                                                         far, self.preprocess)
        data_loader.dataset.preprocess = self.preprocess
        self.threshold = thresh
        return metrics
