HydroForecast

This repository contains the code and supporting materials for the manuscript:

A Diffusion-Based Probabilistic Framework for Flood Forecasting in a Large-Sample Evaluation

HydroForecast is a diffusion-based probabilistic flood forecasting framework. The repository includes the main model implementation, training script, sampling script, configuration files, and evaluation/plotting notebook used in the study.

Repository structure
train.py: training script for HydroForecast.
sample.py: sampling/inference script for generating ensemble discharge forecasts.
ddpm.py: diffusion model implementation.
ddim.py: DDIM sampling implementation.
model.py: conditional denoising network.
dataset.py: dataset and data-loading utilities.
util.py: utility functions, including configuration-based class instantiation.
config/train.yaml: example configuration file for model training.
requirements.txt: Python dependencies.
evaluation.ipynb: notebook containing the plotting code and associated data used to generate the manuscript figures.
Installation

The main dependencies are listed in requirements.txt. A Python environment can be created using:

pip install -r requirements.txt
Training

To train the model, run:

python train.py --base configs/train.yaml --name your_name

Here, configs/train.yaml specifies the model architecture, data module, training settings, and diffusion configuration. The argument --name defines the experiment name used for saving logs and checkpoints.

Evaluation and figure reproduction

The file evaluation.ipynb contains the plotting code for each figure in the manuscript. The notebook also includes the processed data required for plotting, so it can be run directly to reproduce the main figure results.
