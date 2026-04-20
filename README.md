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
To train a model, use the `train.py` script.

```bash
accelerate launch src/train.py --config-name <config-name> <additional-args>
```

## Evaluation
To evaluate a model, use the `eval.py` script.

```bash
accelerate launch src/eval.py --config-name <config-name> model.pretrained_weights=<path-to-weights> <additional-args>
```

## Exporting
To export a model, use the `export.py` script.

```bash
python src/export.py --config-name <config-name> model.pretrained_weights=<path-to-weights> <additional-args>
```

## Benchmarking
To benchmark a model, first export the model as described above, then use `trtexec` to run inference.

```bash
trtexec --onnx=<path-to-onnx> --memPoolSize=workspace:4096 --fp16 --useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10
```