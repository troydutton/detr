# !/bin/bash

# Fail fast
set -eo pipefail

ENV_NAME="detr"

# Bring conda into the path
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Environment creation
conda deactivate
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    conda env remove -y --name "$ENV_NAME"
fi
conda env create -f environment.yaml

# Add lib to the search path
conda activate "$ENV_NAME"
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# Package installation
pip install --editable .

# Hooks
pre-commit install
