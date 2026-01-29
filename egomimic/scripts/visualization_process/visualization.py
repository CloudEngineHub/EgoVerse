"""
Gigantic 2D t-SNE scatter plot colored by a chosen metadata column.

Reads:
- manifest.json (for zarr + metadata paths)
- metadata.parquet (label column is configurable; defaults to lab-like columns)
- embeddings zarr group (expects dataset 'tsne_2d' by default)

Writes:
- a large PNG scatter plot to the data directory
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


def _pick_label_column(df: pd.DataFrame, label_col: str) -> str:
    """
    Resolve which metadata column to use as labels/colors.
    - If label_col is provided, require it to exist.
    - Else, fall back to common "lab" column names.
    """
    if label_col:
        if label_col not in df.columns:
            raise KeyError(
                "Requested --label-col '{}' not found in metadata. Available columns (truncated): {}".format(
                    label_col, list(df.columns)[:50]
                )
            )
        return label_col

    for c in ("lab", "db.lab", "metadata.lab"):
        if c in df.columns:
            return c
    raise KeyError(
        "Could not infer a default label column. Tried: lab, db.lab, metadata.lab. "
        "Pass --label-col to choose a column explicitly. Available columns (truncated): {}".format(
            list(df.columns)[:50]
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="egomimic/scripts/visualization_process/fold_clothes_aria_eva_all_labs/manifest.json",
    )
    ap.add_argument("--image-key", type=str, default="", help="Defaults to first manifest image key.")
    ap.add_argument("--tsne-name", type=str, default="tsne_2d", help="Dataset name inside the zarr group.")
    ap.add_argument(
        "--label-col",
        type=str,
        default="robot_name",
        help=(
            "Metadata column to color points by (e.g. 'lab', 'db.operator', 'task', 'episode_hash'). "
            "If omitted, tries lab-like columns: lab, db.lab, metadata.lab."
        ),
    )
    ap.add_argument("--out", type=str, default="", help="Output png path (defaults next to manifest).")
    ap.add_argument("--figsize", type=float, nargs=2, default=(20, 14), help="Figure size in inches (W H).")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--point-size", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.8)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    if args.image_key:
        image_key = args.image_key
    else:
        image_key = manifest["image_keys"][0]

    zarr_path = Path(manifest["embeddings"][image_key])
    meta_path = Path(manifest["metadata_parquet"])

    meta_df = pd.read_parquet(meta_path)
    label_col = _pick_label_column(meta_df, args.label_col)

    root = zarr.open_group(str(zarr_path), mode="r")
    if args.tsne_name not in root:
        raise KeyError(
            "Could not find '{}' in zarr group. Available arrays: {}".format(
                args.tsne_name, list(root.array_keys())
            )
        )
    y = np.asarray(root[args.tsne_name][:])  # (N,2)
    if y.ndim != 2 or y.shape[1] != 2:
        raise RuntimeError("Unexpected t-SNE shape: {}".format(y.shape))

    if len(meta_df) != y.shape[0]:
        raise RuntimeError(
            "Row mismatch: metadata has {} rows but tsne has {} rows".format(
                len(meta_df), y.shape[0]
            )
        )

    labels = meta_df[label_col].astype(str).fillna("unknown").to_numpy()
    uniq_labels, label_codes = np.unique(labels, return_inverse=True)

    # Build a categorical colormap with enough distinct colors
    cmap = plt.get_cmap("tab20", max(1, len(uniq_labels)))

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    ax.scatter(
        y[:, 0],
        y[:, 1],
        c=label_codes,
        cmap=cmap,
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0,
        rasterized=True,
    )

    ax.set_title("t-SNE of embeddings (colored by {}: {})".format("label", label_col))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(False)

    # Legend (can be large; place outside)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(i), markersize=6)
        for i in range(len(uniq_labels))
    ]
    ax.legend(
        handles,
        uniq_labels.tolist(),
        title=label_col,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        safe_label = label_col.replace("/", "_").replace(".", "_")
        out_path = manifest_path.parent / f"tsne_by_{safe_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print("[DONE] wrote", out_path)


if __name__ == "__main__":
    main()
