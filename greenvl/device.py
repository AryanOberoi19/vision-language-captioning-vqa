"""Device selection kept in one place so the same code runs on the Mac (MPS), a CUDA GPU or CPU."""
import os

# PyTorch's MPS allocator may by default grow to 1.7x Metal's recommended working set, which on the 24 GB M4 Pro
# is more than the machine's memory: the first Flickr8k run (29 Sep 2026) reached 29 GB and stalled. Cap it at the
# recommended working set and start freeing cached blocks earlier. Must be set before the first MPS allocation;
# an environment variable set by the user takes precedence.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")

import torch  # noqa: E402


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_mb(device: torch.device) -> float | None:
    """Memory currently held by the allocator on the device, in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None
