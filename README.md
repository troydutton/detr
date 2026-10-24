# DEtection TRansformers
Transformer-based object detection

# Development
The instructions for setting up the development environment are provided below.

## Setup
Dependencies are managed with [uv](https://docs.astral.sh/uv/). Syncing creates a `.venv` with the GPU dependencies, and installing the pre-commit hooks runs formatting, linting, and tests on every commit. Commands are then run inside the environment with `uv run`.

```bash
uv sync
uv run pre-commit install
```

A CPU-only environment can be created instead with `uv sync --no-default-groups --group cpu --group dev`.

## Testing
The codebase uses `pytest` for testing. It is run as part of the pre-commit hooks, but can also be run manually.

```bash
uv run pytest tests/
```

## Formatting
The codebase uses `black` for formatting. It is run as part of the pre-commit hooks, but can also be run manually.

```bash
uv run black src/ tests/
```

## Linting
The codebase uses `ruff` for linting. It is run as part of the pre-commit hooks, but can also be run manually.

```bash
uv run ruff check --fix src/ tests/
```

# Usage
The instructions for training, evaluating, exporting, and benchmarking models are provided below.

## Training
To train a model, update `configs/train.yaml` and run `train.py`.

```bash
uv run accelerate launch src/train.py <additional-args>
```

## Evaluation
To evaluate a model, update `configs/evaluate.yaml` and run `evaluate.py`.

```bash
uv run accelerate launch src/evaluate.py <additional-args>
```

## Exporting
To export a model, update `configs/export.yaml` and run `export.py`.

```bash
uv run accelerate launch src/export.py <additional-args>
```

## Benchmarking
To benchmark a model, first export the model as described above, then use `trtexec` to run inference.

```bash
uv run trtexec --onnx=<path-to-onnx> --memPoolSize=workspace:4096 --warmUp=500 --avgRuns=1000 --duration=10
```