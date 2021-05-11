# `radnets`
`PyTorch`-based spectral gamma-ray anomaly detection and identification, used in the paper "Neural Network Approaches for Mobile Spectroscopic Gamma-ray Source Detection," Bilton et al., 2021.

## Overview
A gamma-ray spectrum can be treated as a d-dimensional vector `x`.
The goal of spectral anomaly detection is to flag spectra which deviate from normal background.
Additionally, spectral identification classifies which type of radioactive sources are found in `x`, if any.
This project uses `pytorch` for both spectral anomaly detection and identification.

## Structure

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
