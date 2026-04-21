import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import to_dtype, to_image

from data.transforms import Transformation

Args = Dict[str, Union[Any, "Args"]]


class ImageDataset(Dataset):
    """
    Basic image dataset useful for inference on a set of images without annotations.

    Expects the following directory structure:
    root/
    ├── image0.jpg
    ├── image1.jpg
    ├── ...

    Args
        root: Root directory containing the images.
        transforms: Transformations to apply to each image, optional.
    """

    def __init__(self, root: str, transforms: Transformation = None) -> None:
        self.image_paths = sorted([p for p in Path(root).rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.transforms = transforms

        logging.info(f"Loaded {len(self.image_paths):,} images from `{root}`.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Retrieve an image and its corresponding metadata.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            image: Image with shape (channels, height, width).
            #### image_info
            Image information, including
                - `file_name`: Original file name.
                - `orig_size`: Original image size (height, width).
                - `size`: Transformed image size (height, width).
        """

        # Load the image
        file_name = str(self.image_paths[idx])
        image = Image.open(file_name).convert("RGB")

        # Apply the transformations, converting to tensor if no transforms are provided
        if self.transforms is not None:
            transformed_image = self.transforms(image)
        else:
            transformed_image = to_dtype(to_image(image), torch.float32, scale=True)

        # Add image information
        image_info = {
            "file_name": file_name,
            "orig_size": torch.tensor(image.size[::-1], dtype=torch.int64),
            "size": torch.tensor(transformed_image.shape[-2:], dtype=torch.int64),
        }

        return transformed_image, image_info
