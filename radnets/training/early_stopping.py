import numpy as np
import torch


class EarlyStopping:
    def __init__(self, name, patience=10, verbose=False, delta=0):
        """
        Args:
            name (str): Name appended to `checkpoint_` for saving checkpoint
            file.
            patience (int): How long to wait after last time validation loss
            improved. Default: 10
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify
            as an improvement. Default: 0
        """
        self.name = name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased \
                ({self.val_loss_min:.6f} --> {val_loss:.6f}). \
                Saving model ...')
        fname = f'checkpoint_{self.name}.pt'
        torch.save(model.state_dict(), fname)
        self.val_loss_min = val_loss
