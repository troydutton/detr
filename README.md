# DEtection TRansformers
Transformer-based object detection

# Setup
The setup script initializes the developer environment in an idempotent manner.

```bash
bash ./scripts/setup.sh
```

After execution, the environment can be activated using `conda`.

```bash
conda activate detr
```

# Testing
Unit tests use the `pytest` framework and can be found in `tests/`.

Command:
```bash
conda activate detr && pytest tests/
```