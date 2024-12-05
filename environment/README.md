# Environment Setup
This project is developed in Python 3.9.

### 1. Create a virtual environment

You can use either Python's built-in `venv` or `conda` to create a virtual environment:

#### 1a. Python venv environment

To create a python virtual environment, run the following commands from the root of the project:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 1b. Conda env:

Start from a clean environment, e.g. with conda do:

    conda create -n spherical-dyffusion python=3.9
    conda activate spherical-dyffusion  # activate the environment called spherical-dyffusion

### 2. Install dependencies

After creating your virtual environment, run

    bash environment/install-dependencies.sh

Note that depending on your CUDA version, you may need to install PyTorch differently than in the bash file.
For more details about installing [PyTorch](https://pytorch.org/get-started/locally/), please refer to their official documentation.