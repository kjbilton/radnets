import numpy as np

from .base_dataset import BaseDataset


class FeedforwardDataset(BaseDataset):
    def _compute_mean_variance(self, spectra, labels):
        if labels is None:
            background = spectra
        else:
            background_mask = labels[:, 0] == 1.
            background = spectra[background_mask]
        self.mu = background.mean(axis=0)

        if self.preprocess == 'standardize':
            self.sigma = background.std(axis=0, ddof=1).values
        elif self.preprocess == 'mean_center':
            self.sigma = np.ones_like(self.mu)

    def __getitem__(self, idx):
        X = self.spectra[idx]
        X = self.preprocess_function(X)

        if self.labels is not None:
            y = self.labels[idx]
        else:
            y = None

        return (X, y)
