# DEtection TRansformers
Transformer-based object detection

# Development
The instructions for setting up the development environment and running unit tests are provided below.

## Setup
The setup script initializes the developer environment in an idempotent manner.

```bash
bash ./scripts/setup.sh
```

After execution, the environment can be activated using `conda`.

```bash
conda activate detr
```

## Testing
Unit tests use the `pytest` framework and can be found in `tests/`.

Command:
```bash
conda activate detr && pytest tests/
```

# Usage
The instructions for training, evaluating, exporting, and benchmarking models are provided below.

## Training
To train a model, update `config/train.yaml` and run `train.py`.

```bash
accelerate launch src/train.py <additional-args>
```

## Evaluation
To evaluate a model, update `config/evaluate.yaml` and run `eval.py`.

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