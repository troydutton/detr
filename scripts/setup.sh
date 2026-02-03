# !/bin/bash

# Fail fast
set -eou pipefail

ENV_NAME="detr"

# Environment creation
conda env create -f environment.yaml

# We source conda.sh to bring 'activate' into this shell's scope
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Package installation
pip install --editable .

# Hooks
pre-commit install