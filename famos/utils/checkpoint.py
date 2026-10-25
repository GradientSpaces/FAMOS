"""Checkpoint loading for FAMOS modules."""

import logging

import torch
from safetensors.torch import load_file

logger = logging.getLogger("Utils")


def load_checkpoint_for_module(
    module: torch.nn.Module,
    checkpoint_path: str,
    prefix_to_remove: str | None = None,
    strict: bool = True,
):
    """Load a .safetensors or Lightning checkpoint into `module`.

    Only keys starting with `prefix_to_remove` are kept, with that prefix stripped.
    """
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )["state_dict"]

    if prefix_to_remove is not None:
        state_dict = {
            k[len(prefix_to_remove):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix_to_remove)
        }

    load_result = module.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}. {load_result}")
    return load_result
