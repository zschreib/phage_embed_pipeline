from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

# ---------------------------
# Embedding utilities
# ---------------------------

# ESM2-3B pre-trained (esm2_t36_3B_UR50D) layer 36 = pooled embedding final layer (default)
def load_embedding(file_path: str, layer: int = 36) -> Tuple[np.ndarray, str]:

    data = torch.load(file_path, map_location="cpu")
    if "mean_representations" not in data:
        raise KeyError(f"'mean_representations' missing in {os.path.basename(file_path)}")

    if layer not in data["mean_representations"]:
        raise KeyError(f"Layer {layer} missing in {os.path.basename(file_path)}")

    if "label" not in data:
        raise KeyError(f"'label' missing in {os.path.basename(file_path)}")

    emb = data["mean_representations"][layer]
    if hasattr(emb, "numpy"):
        emb = emb.numpy()
    emb = np.asarray(emb, dtype=np.float32).ravel()
    if emb.ndim != 1:
        raise ValueError(f"Embedding must be 1D after ravel; got {emb.shape} in {os.path.basename(file_path)}")

    return emb, data["label"]

# ---------------------------
# Embedding processing
# ---------------------------
