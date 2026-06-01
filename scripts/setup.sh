# !/bin/bash

# Fail fast
set -eo pipefail

ENV_NAME="detr"

# Bring conda into the path
if [ -d "$HOME/miniconda3/" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3/" ]; then
    CONDA_PATH="$HOME/anaconda3"
else
    echo "No conda installation found."; exit 1
fi
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
