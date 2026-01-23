# DETR Project Instructions

## Key Workflows & Commands
### Environment Management
**ALWAYS** activate the conda environment before running any programs, tests, or python commands.
- Command: `conda activate detr`

### Testing
This project uses `pytest` unit tests located in `tests/` to verify functionality.
- Command: `conda activate detr && pytest tests/`

## Code & Style Conventions
### Type Hinting
Type hinting is **strictly enforced** for all arguments and return values.
- Use the `typing` module for complex types (e.g., `Tuple`, `List`, `Dict`, `Optional`).
- Use `Tensor` from `torch` for tensor types.
- `nn.Module` subclasses must use a decorator to copy type annotations from `forward` to `__call__`.

  ```python
  from utils.misc import take_annotation_from
  ...
  @take_annotation_from(forward)
  def __call__(self, *args, **kwargs):
      return nn.Module.__call__(self, *args, **kwargs)
  ```

### Docstrings
Google Style docstrings are required for all public modules, functions, classes, and methods.
- Types should only be specified in the function signature, not in the argument description.
- Docstrings for classes should document the arguments for creation under the class definition, above `__init__`.
- Functions with multiple return values should use #### for each return value after the first.
- Example:

  ```python
  class Backbone(nn.Module):
      """
      Args:
          name: Name of the backbone to load from timm.
          embed_dim: Dimension of the output embeddings.
          pretrained: Whether to load pretrained weights, optional.
      """

      def __init__(self, name: str, embed_dim: int, *, pretrained: bool = True) -> None:
          ...

      def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
          """
          Args:
              images: Image with shape (batch_size, 3, height, width).

          Returns:
              features: Features with shape (batch_size, feature_height, feature_width, embed_dim).
              #### feature_pos
              Positional embeddings with shape (batch_size, feature_height, feature_width, embed_dim).
          """
          ...
  ```

### Naming Conventions
Use descriptive yet succinct variable names that provide immediate semantic context: strictly avoid generic terms (no `data`, `input`) and single-letter names (no `x`, `y`) except in industry standard cases (i.e. `i` in a loop) 

- Use `images` (not `x`), `feature_pos` (not `pos` or `feature_positional_embeddings`), `embed_dim` (not `d`), and `in_channels` (not `c`).
- Use plurals for collections (`boxes`), `_mask` for booleans, `_logits` for raw outputs, and `_indices` for indices.
- Differentiate data location using `target_classes`, `encoder_boxes`, or `decoder_queries`.
