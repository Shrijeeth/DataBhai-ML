from typing import Dict

import torch
import weaviate


def get_device_type(cpu_strict=False) -> str:
    """Gets default device configuration available for the user"""
    if cpu_strict:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.0)
        return "mps"
    if torch.backends.mkl.is_available():
        return "mkl"
    return "cpu"
