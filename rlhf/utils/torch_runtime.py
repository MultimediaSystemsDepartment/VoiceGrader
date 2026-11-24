"""Shared low-risk PyTorch runtime acceleration for GPU model stages."""

from __future__ import annotations


def configure_torch_runtime() -> None:
    """Enable Ampere-friendly math and fixed-shape convolution autotuning."""
    import torch

    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def use_fused_optimizer(device: str) -> bool:
    """Return whether CUDA fused Adam/AdamW kernels can be used."""
    import torch

    return str(device).startswith("cuda") and torch.cuda.is_available()