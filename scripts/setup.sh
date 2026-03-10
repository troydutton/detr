# !/bin/bash

# Fail fast
set -eou pipefail

ENV_NAME="detr"

# Environment creation
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    conda env remove -y --name "$ENV_NAME"
fi
conda env create -f environment.yaml

# Add lib to the search path
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# Package installation
pip install --editable .

# Hooks
pre-commit install