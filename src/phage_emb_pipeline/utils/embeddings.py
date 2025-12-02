from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import umap
import hdbscan

# ---------------------------
# Embedding utilities
# ---------------------------

# ESM2-3B pre-trained (esm2_t36_3B_UR50D) layer 36 = pooled embedding final layer (default)
def load_embedding(file_path: str | Path, layer: int = 36) -> Tuple[np.ndarray, str]:

    data = torch.load(file_path, map_location="cpu")
    basename = os.path.basename(file_path)

    if "mean_representations" not in data:
        raise KeyError(f"'mean_representations' missing in {basename}")

    if layer not in data["mean_representations"]:
        raise KeyError(f"Layer {layer} missing in {basename}")

    if "label" not in data:
        raise KeyError(f"'label' missing in {basename}")

    emb = data["mean_representations"][layer]
    if hasattr(emb, "numpy"):
        emb = emb.numpy()
    emb = np.asarray(emb, dtype=np.float32).ravel()
    if emb.ndim != 1:
        raise ValueError(
            f"Embedding must be 1D after ravel; got {emb.shape} in {basename}"
        )

    return emb, data["label"]

def load_embeddings_from_dir(dir_path: str | Path, layer: int = 36) -> Tuple[np.ndarray, List[str]]:

    dir_path = Path(dir_path)
    emb_list: List[np.ndarray] = []
    labels: List[str] = []

    pt_files = sorted(dir_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {dir_path}")

    for fp in pt_files:
        emb, label = load_embedding(fp, layer=layer)
        emb_list.append(emb)
        labels.append(label)

    X = np.vstack(emb_list).astype(np.float32)
    return X, labels

# ---------------------------
# Embedding processing
# ---------------------------

def run_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = "cosine", random_state: int = 42, **kwargs) -> np.ndarray:

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **kwargs,
    )
    X_2d = reducer.fit_transform(X)
    return X_2d

def run_hdbscan(X_low_dim: np.ndarray, min_cluster_size: int = 10, min_samples: Optional[int] = None, **kwargs) -> np.ndarray:

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        **kwargs,
    )
    labels = clusterer.fit_predict(X_low_dim)
    return labels

def build_embedding_table(
    emb_dir: str | Path,
    metadata_tsv: str | Path,
    layer: int = 36,
    umap_cluster_kwargs: Optional[dict] = None,
    umap_viz_kwargs: Optional[dict] = None,
    hdbscan_kwargs: Optional[dict] = None,
) -> pd.DataFrame:

    umap_cluster_kwargs = umap_cluster_kwargs or {}
    umap_viz_kwargs = umap_viz_kwargs or {}
    hdbscan_kwargs = hdbscan_kwargs or {}

    # Load embeddings
    X, labels = load_embeddings_from_dir(emb_dir, layer=layer)
    emb_dict: Dict[str, np.ndarray] = {lab: X[i, :] for i, lab in enumerate(labels)}

    # Load metadata and join
    meta_df = load_metadata_tsv(metadata_tsv)
    df = meta_df[meta_df["name"].isin(emb_dict.keys())].reset_index(drop=True)

    if df.empty:
        raise ValueError("No overlap between metadata 'name' and embedding labels.")

    # Build array in metadata order
    X_ordered = np.stack([emb_dict[n] for n in df["name"].tolist()], axis=0)

    # UMAP for clustering (cluster space)
    X_cluster = run_umap(X_ordered, **umap_cluster_kwargs)
    cluster_labels = run_hdbscan(X_cluster, **hdbscan_kwargs)
    df["cluster"] = cluster_labels

    # UMAP for visualization (low dim plots to show)
    X_viz = run_umap(X_ordered, **umap_viz_kwargs)
    df["x"] = X_viz[:, 0]
    df["y"] = X_viz[:, 1]

    # Rename color column (todo: fix metadata structure input)
    df = df.rename(columns={"762_color": "color"})

    return df

# ---------------------------
# Embedding metadata
# ---------------------------

def load_metadata_tsv(path: str | Path) -> pd.DataFrame:

    path = Path(path)
    df = pd.read_csv(path, sep="\t")
    #ORF_ID, feature, hexcolor (todo: adjust later to handle other feature types)
    required_cols = {"name", "genofeature", "762_color"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")

    return df
