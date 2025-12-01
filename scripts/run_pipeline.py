# scripts/run_pipeline.py

from __future__ import annotations

import argparse
from pathlib import Path

from phage_emb_pipeline.pipeline import run_embedding_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phage embedding pipeline.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with ESM .pt files.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=36,
        help="Layer index from mean_representations to use (default: 36).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    X, labels = run_embedding_pipeline(
        input_dir=Path(args.input_dir),
        layer=args.layer,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
