from __future__ import annotations

from pathlib import Path

import pandas as pd

#(todo: add more features/analysis)
def write_cosmograph_embedding(
    df: pd.DataFrame,
    out_path: str | Path,
) -> None:

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"name", "x", "y", "color", "genofeature", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for Cosmograph embedding: {missing}")

    out_df = pd.DataFrame(
        {
            "id": df["name"],
            "x": df["x"],
            "y": df["y"],
            "color": df["color"],
            "genofeature": df["genofeature"],
            "cluster": df["cluster"],
        }
    )

    out_df.to_csv(out_path, index=False, sep=";")
