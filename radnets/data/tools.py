import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


from . import FeedforwardDataset, RecurrentDataset
from ..utils.config import get_filename, load_config


def get_ff_autoencoder_dataset(data_config, preprocess='none', val_idx=0):
    """
    Return DataSet for background and validation data

    Parameters
    ----------
    data_config: dict
        Data configuration dictionary
    preprocess: str
        Type of preprocessing used ('none', 'standardize', ...)
    val_idx:
        Index of cross-validation
    """

    # Get training and validation indices
    training_splits, validation_splits = get_splits(data_config, val_idx)

    # Load in data
    bkg_file = get_filename(data_config, 'background/spectra')
    data = pd.read_hdf(bkg_file, '/spectra')

    # Split by indices
    training = data.loc[training_splits]
    training_data = FeedforwardDataset(training.values, training.index,
                                       preprocess)

    validation = data.loc[validation_splits]
    validation_data = FeedforwardDataset(validation.values, validation.index,
                                         preprocess)
    return training_data, validation_data


def get_ae_loaders(preprocess, model_config, n_runs_inject=100):
    data_config_file = model_config['data']['config']
    data_config = load_config(data_config_file)

    # Get background data
    training_data, validation_data = get_ff_autoencoder_dataset(
        data_config, preprocess=preprocess)

    # Create dataloaders
    batch_size = model_config['training']['batch_size']
    training_loader = DataLoader(training_data, batch_size=batch_size,
                                 shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size,
                                   shuffle=True)

    # Create an injection dataset
    injection = pd.DataFrame(data=validation_data.spectra,
                             index=validation_data.indices)
    injection_indices = injection.groupby(level=0).size().index.values
    injection_indices = np.random.choice(injection_indices, n_runs_inject,
                                         replace=False)
    injection_data = (injection_indices, injection)

    loaders = {'training': training_loader,
               'validation': validation_loader}
    return loaders, injection_data


def get_rnn_autoencoder_dataset(data_config, preprocess=None, val_idx=0):
    """
    Return DataSet for background and validation data

    Parameters
    ----------
    data_config: dict
        Data configuration dictionary
    val_idx:
        Index of cross-validation
    mode: str
        'bkg', 'src', 'inject'
    """

    # Get training and validation indices
    training_splits, validation_splits = get_splits(data_config, val_idx)

    # Load in data
    bkg_path = data_config['background']['spectra']['path']
    bkg_file = data_config['background']['spectra']['file']
    bkg_file = f'{bkg_path}/{bkg_file}'
    background = pd.read_hdf(bkg_file, '/spectra')

    # Split by indices
    training = background.loc[training_splits]
    train_indices = training.groupby(level=0).size().index.values
    training_data = RecurrentDataset(training, train_indices, preprocess)

    validation = background.loc[validation_splits]
    valid_indices = validation.groupby(level=0).size().index.values
    validation_data = RecurrentDataset(validation, valid_indices, preprocess)
    return training_data, validation_data


def get_ae_recurrent_loaders(preprocess, model_config, n_runs_inject=100):
    data_config_file = model_config['data']['config']
    data_config = load_config(data_config_file)

    # Get background data
    training_data, validation_data = get_rnn_autoencoder_dataset(
        data_config, preprocess=preprocess)

    # Create dataloaders
    batch_size = model_config['training']['batch_size']
    training_loader = DataLoader(training_data, batch_size=batch_size,
                                 shuffle=True, collate_fn=pad_autoencoder_run)
    validation_loader = DataLoader(validation_data, batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=pad_autoencoder_run)

    # Create an injection dataset
    injection = pd.DataFrame(data=validation_data.spectra)
    injection_indices = injection.groupby(level=0).size().index.values
    injection_indices = np.random.choice(injection_indices, n_runs_inject,
                                         replace=False)
    injection_data = (injection_indices, injection)

    loaders = {'training': training_loader,
               'validation': validation_loader}
    return loaders, injection_data


def pad_autoencoder_run(batch):
    runs = [torch.FloatTensor(x) for x in batch]
    run_lens = [len(x) for x in runs]
    runs_padded = pad_sequence(runs, batch_first=True, padding_value=0)
    return runs_padded, run_lens


def get_ff_identification_dataset(data_config, val_idx, preprocess=None):
    """
    Return DataSet for background and validation data

    Parameters
    ----------
    data_config: dict
        Data configuration dictionary
    val_idx:
        Index of cross-validation
    preprocess: str
    """
    # Get training and validation indices
    training_splits, validation_splits = get_splits(data_config, val_idx)

    # Load in data
    bkg_file = get_filename(data_config, 'background/spectra')
    src_file = get_filename(data_config, 'injection/feedforward')

    background = pd.read_hdf(bkg_file, '/spectra')
    source = pd.read_hdf(src_file, '/source')
    data = background + source

    source_fraction = pd.read_hdf(src_file, '/source_fraction')

    # Split by indices
    training = data.loc[training_splits]
    training_frac = source_fraction.loc[training_splits]
    training_data = FeedforwardDataset(training.values,
                                       training.index,
                                       preprocess,
                                       training_frac.values)

    validation = data.loc[validation_splits]
    validation_frac = source_fraction.loc[validation_splits]
    validation_data = FeedforwardDataset(validation.values,
                                         validation.index,
                                         preprocess,
                                         validation_frac.values)

    thresh = background.loc[validation_splits]
    thresh_frac = np.zeros_like(validation_frac.values)
    thresh_frac[:, 0] = 1.
    thresh_data = FeedforwardDataset(thresh.values,
                                     validation.index,
                                     preprocess,
                                     thresh_frac)
    # Generate a dataset
    return training_data, validation_data, thresh_data


def get_recurrent_id_dataset(data_config, val_idx, preprocess=None):
    """
    Return DataSet for background and validation data

    Parameters
    ----------
    data_config: dict
        Data configuration dictionary
    val_idx:
        Index of cross-validation
    """

    # Get training and validation indices
    training_splits, validation_splits = get_splits(data_config, val_idx)

    bkg_file = get_filename(data_config, 'background/augmented')
    bkg_file = get_filename(data_config, 'background/spectra')
    src_file = get_filename(data_config, 'injection/recurrent')

    background = pd.read_hdf(bkg_file, '/spectra')
    source = pd.read_hdf(src_file, '/source')
    labels = pd.read_hdf(src_file, '/source_fraction')

    data = background + source

    # run_indices = data.index.levels[0].values
    # n_runs = len(run_indices)
    # n_training_splits = int(0.8 * n_runs)
    # training_splits = np.random.choice(run_indices, size=n_training_splits,
    #                                    replace=False)
    # mask = np.isin(run_indices, training_splits)
    # validation_splits = run_indices[~mask]

    # Split by indices
    training = data.loc[training_splits]
    training_labels = labels.loc[training_splits]
    train_indices = training.groupby(level=0).size().index.values
    training_data = RecurrentDataset(training, train_indices, preprocess,
                                     labels=training_labels)

    validation = data.loc[validation_splits]
    validation_labels = labels.loc[validation_splits]
    valid_indices = validation.groupby(level=0).size().index.values
    validation_data = RecurrentDataset(validation, valid_indices, preprocess,
                                       labels=validation_labels)

    thresh = background.loc[validation_splits]
    thresh_frac = np.zeros_like(validation_labels.values)
    thresh_frac[:, 0] = 1.
    thresh_frac = pd.DataFrame(thresh_frac, index=validation_labels.index)
    thresh_data = RecurrentDataset(thresh, valid_indices, preprocess,
                                   labels=thresh_frac)

    # Generate a dataset
    return training_data, validation_data, thresh_data


def pad_labeled_run(batch):
    (xx, yy) = zip(*batch)

    xx = [torch.FloatTensor(x) for x in xx]
    yy = [torch.FloatTensor(y) for y in yy]

    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens


def get_splits(data_config, val_idx):
    """
    Get cross-validation indices for the specified validation index.
    """
    # Grab filename
    path = data_config['background']['splits']['path']
    filename = data_config['background']['splits']['cv_file']
    filename = f'{path}/{filename}'

    # Load file
    splits = np.load(filename)

    # Grab indices
    validation = splits[val_idx]
    training = np.delete(splits, val_idx, axis=0).flatten()

    return training, validation


def get_training_data(fname):
    """
    Parameters
    ----------
    fname: str
        File containing training data.

    Returns
    -------
    spectra: Pandas DataFrame
        Spectra.
    labels: Pandas DataFrame
        Labels of source identity in each spectrum.
    """
    background = pd.read_hdf(fname, 'background')
    source = pd.read_hdf(fname, 'source')
    labels = pd.read_hdf(fname, 'labels')
    spectra = background + source
    return spectra, labels


def get_train_test_splits(fname):
    """
    Get the run indices for training and testing.

    Parameters
    ----------
    fname: str
        Filename of file containing split information.

    Returns
    -------
    splits: dict
        Dictionary containing numpy arrays of run indices used in training and
        testing.
    """
    return {k: pd.read_hdf(fname, f'/splits/{k}').values
            for k in ['training', 'testing']}
