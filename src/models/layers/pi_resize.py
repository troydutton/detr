import logging
from math import sqrt
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def pi_resize_patch_embed(patch_embeddings: Tensor, patch_size: Tuple[int, int]) -> Tensor:
    """
    Pseudoinverse resize of a patch embedding kernel, see [FlexiViT](https://arxiv.org/abs/2212.08013).

    Args:
        patch_embeddings: Patch embedding kernel with shape (embed_dim, in_channels, old_height, old_width).
        patch_size: Target (height, width) of the kernel.

    Returns:
        patch_embeddings: Resized kernel with shape (embed_dim, in_channels, height, width).
    """

    embed_dim, in_channels, old_height, old_width = patch_embeddings.shape

    # Linear map from old to new patch pixels, built by resizing each pixel basis vector
    basis = torch.eye(old_height * old_width).reshape(-1, 1, old_height, old_width)
    resize_matrix = F.interpolate(basis, size=patch_size, mode="bilinear", align_corners=False).reshape(old_height * old_width, -1).T

    # Choose the kernel that reproduces the original embeddings on resized patches
    patch_embeddings = patch_embeddings.float().reshape(embed_dim * in_channels, -1) @ torch.linalg.pinv(resize_matrix)

    return patch_embeddings.reshape(embed_dim, in_channels, *patch_size)


@torch.no_grad()
def resize_embeddings(state_dict: Dict[str, Tensor], model_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Resize the positional and patch embeddings in a state dict to match a model (i.e. if the image or patch size changes).

    Args:
        state_dict: State dict to be loaded.
        model_state_dict: State dict of the model being loaded into.

    Returns:
        state_dict: State dict with resized embeddings.
    """

    for key in list(state_dict.keys()):
        if key not in model_state_dict or state_dict[key].shape == model_state_dict[key].shape:
            continue

        # Interpolate positional embeddings (i.e. if the image or patch size changes)
        if key.endswith("embeddings.position_embeddings"):
            old_pos_embed, new_pos_embed = state_dict[key], model_state_dict[key]

            # Separate the class and patch tokens
            old_cls_pos_embed, old_patch_pos_embed = old_pos_embed[:, :1], old_pos_embed[:, 1:]
            new_patch_pos_embed = new_pos_embed[:, 1:]

            # Calculate the side lengths
            _, old_num_patches, embed_dim = old_patch_pos_embed.shape
            _, new_num_patches, _ = new_patch_pos_embed.shape

            old_size = int(sqrt(old_num_patches))
            new_size = int(sqrt(new_num_patches))

            logging.info(f"Interpolating positional embeddings: ({old_size}, {old_size}) -> ({new_size}, {new_size}).")

            # Restore spatial dimensions, interpolate, and flatten again
            old_patch_pos_embed = old_patch_pos_embed.reshape(1, old_size, old_size, embed_dim)
            old_patch_pos_embed = old_patch_pos_embed.permute(0, 3, 1, 2)

            patch_pos_embed = F.interpolate(
                old_patch_pos_embed.float(),
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            ).to(old_patch_pos_embed.dtype)

            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
            patch_pos_embed = patch_pos_embed.reshape(1, new_num_patches, embed_dim)

            state_dict[key] = torch.cat((old_cls_pos_embed, patch_pos_embed), dim=1)

        # Resize patch embeddings (i.e. if the patch size changes)
        elif key.endswith("patch_embeddings.projection.weight"):
            old_patch_embeddings = state_dict[key]
            old_patch_size = tuple(old_patch_embeddings.shape[-2:])
            new_patch_size = tuple(model_state_dict[key].shape[-2:])

            logging.info(f"Interpolating patch embeddings: {old_patch_size} -> {new_patch_size}.")

            state_dict[key] = pi_resize_patch_embed(old_patch_embeddings, new_patch_size).to(old_patch_embeddings.dtype)

    return state_dict
