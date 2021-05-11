# `radnets`
`PyTorch`-based spectral gamma-ray anomaly detection and identification, used in the paper "Neural Network Approaches for Mobile Spectroscopic Gamma-ray Source Detection," Bilton et al., 2021.

## Overview
A gamma-ray spectrum can be treated as a d-dimensional vector `x`, which consists of benign background and potentially sources of interest.
The goal of spectral anomaly detection is to flag spectra which deviate from ordinary background.
Additionally, spectral identification classifies which type of radioactive sources are found in `x`, if any.
This project uses `pytorch` for both spectral anomaly detection and identification.

## Structure
- `radnets.data`
  - Tools for preprocessing data and custom `pytorch` `Datasets` and `Dataloaders` used in training models.
- `radnets.detection`
  - Tools for computing detection thresholds and performing detection.
- `radnets.models`
  - `pytorch` model classes for detection and identification. A distinction is made betweeen feedforward and recurrent models since they require a different treatment of data.
- `radnets.training`
  - Tools used in training models.
- `radnets.utils`
  - General utilities.

## Setup
Note: this has been developed and tested using `python 3.7`.
Install dependencies:
```
pip install -r requirements.txt
```

Install the `radnets` package:
```
pip install -e .
```

If you're developing, install the development dependencies:
```
pip install -r requirements-dev.txt
```
