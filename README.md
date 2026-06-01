# DEtection TRansformers
Transformer-based object detection

# Development
The instructions for setting up the development environment are provided below.

## Setup
The setup script creates a new conda environment, installs the required dependencies, and sets up pre-commit hooks. It should be run from the root of the repository, and can be run multiple times without issue.

```bash
./scripts/setup.sh
```

After execution, the environment can be activated using `conda`.

```bash
conda activate detr
```

## Testing
The codebase uses `pytest` for testing. It is run as part of the pre-commit hooks, but can also be run manually.

```bash
pytest tests/
```

## Formatting
The codebase uses `black` for formatting. It is run as part of the pre-commit hooks, but can also be run manually.

```bash
black src/ tests/
```

## Linting
The codebase uses `ruff` for linting. It is run as part of the pre-commit hooks, but can also be run manually.

```bash
ruff check --fix src/ tests/
```

# Usage
The instructions for training, evaluating, exporting, and benchmarking models are provided below.

## Training
To train a model, update `config/train.yaml` and run `train.py`.

```bash
accelerate launch src/train.py <additional-args>
```

## Evaluation
To evaluate a model, update `config/evaluate.yaml` and run `evaluate.py`.

```bash
accelerate launch src/evaluate.py <additional-args>
```

## Exporting
To export a model, update `config/export.yaml` and run `export.py`.

```bash
accelerate launch src/export.py <additional-args>
```

## Benchmarking
To benchmark a model, first export the model as described above, then use `trtexec` to run inference.

```bash
trtexec --onnx=<path-to-onnx> --memPoolSize=workspace:4096 --fp16 --useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10
```