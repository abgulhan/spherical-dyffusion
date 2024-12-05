#!/bin/bash

python3 -m pip install --upgrade pip
# ====================
# Install dependencies. Note: Depending on your system, you may need to install PyTorch manually.
# ====================
# PyTorch:
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# General dependencies:
python3 -m pip install dacite h5py huggingface_hub matplotlib scipy einops wandb lightning xarray netCDF4 dask hydra-core cachey tensordict timm boto3 black ruff
# For SFNO:
python3 -m pip install --no-deps nvidia-modulus@git+https://github.com/ai2cm/modulus.git@94f62e1ce2083640829ec12d80b00619c40a47f8
python3 -m pip install torch-harmonics tensorly tensorly-torch
