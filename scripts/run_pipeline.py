from __future__ import annotations

import argparse
from pathlib import Path

from phage_emb_pipeline.utils.common import set_seed
from phage_emb_pipeline.utils.embeddings import build_embedding_table
from phage_emb_pipeline.utils.network import write_cosmograph_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Run phage embedding pipeline.")
    parser.add_argument(
        "--emb-dir",
        type=str,
        required=True,
        help="Directory with ESM .pt embedding files.",
    )
    parser.add_argument(
        "--metadata-tsv",
        type=str,
        required=True,
        help="TSV file with metadata (name, genofeature, 762_color, ...).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="outputs/phage_embeddings_cosmograph.csv",
        help="Output CSV file for Cosmograph (embedding mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP, etc.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    df = build_embedding_table(
        emb_dir=Path(args.emb_dir),
        metadata_tsv=Path(args.metadata_tsv),
        layer=36,
        umap_cluster_kwargs={
            "n_neighbors": 30,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": args.seed,
            "n_components": 2,
        },
        umap_viz_kwargs={
            "n_neighbors": 15,
            "min_dist": 0.5,
            "metric": "cosine",
            "random_state": args.seed,
            "n_components": 2,
        },
        hdbscan_kwargs={
            "min_samples": 5,
            "min_cluster_size": 2,
        },
    )

    out_path = Path(args.out_file)
    write_cosmograph_embedding(df, out_path)

    print(f"Wrote Cosmograph embedding file to {out_path}")

if __name__ == "__main__":
    main()
