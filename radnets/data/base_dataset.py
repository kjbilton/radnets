from functools import partial
from torch.utils.data import Dataset

from .preprocess import (log_preprocess, norm_log_preprocess,
                         standardize_preprocess, EPS)
PREPROCESS_FUNCTIONS = {'none': lambda x: x, 'log': log_preprocess,
                        'normlog': norm_log_preprocess,
                        'standardize': standardize_preprocess,
                        'mean_center': standardize_preprocess}

class BaseDataset(Dataset):
    def __init__(self, spectra, indices, preprocess=None, labels=None):
        """
        Parameters
        ==========
        spectra : numpy array, shape (n_samples, n_bins)
            Array of spectra.
        labels : numpy array, shape (n_samples,)
            Labels of source type present.
        indices:
        preprocess : str, ('none', 'log', 'normlog', 'standardize',
                           'mean_center')
            Type of preprocessing to perform on spectra.
        """
        assert preprocess in PREPROCESS_FUNCTIONS, "Invalid preprocessing type."

        self.spectra = spectra
        self.indices = indices
        self.preprocess = preprocess
        self.labels = labels
        self.preprocess_function = PREPROCESS_FUNCTIONS[preprocess]

        if preprocess in ['standardize', 'mean_center']:
            self._compute_mean_variance(spectra, labels)
            self.preprocess_function = partial(self.preprocess_function,
                                               mu=self.mu, sigma=self.sigma)
        else:
            self.mu = None
            self.sigma = None

    def __len__(self):
        return len(self.spectra)
