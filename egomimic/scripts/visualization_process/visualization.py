"""
Gigantic 2D t-SNE scatter plot colored by lab.

Reads:
- manifest.json (for zarr + metadata paths)
- metadata.parquet (expects a column for lab: tries 'lab', then 'db.lab')
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


def _pick_lab_column(df: pd.DataFrame) -> str:
    for c in ("lab", "db.lab", "metadata.lab"):
        if c in df.columns:
            return c
    raise KeyError(
        "Could not find a lab column. Tried: lab, db.lab, metadata.lab. "
        "Available columns (truncated): {}".format(list(df.columns)[:50])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="egomimic/scripts/visualization_process/data/manifest.json",
    )
    ap.add_argument("--image-key", type=str, default="", help="Defaults to first manifest image key.")
    ap.add_argument("--tsne-name", type=str, default="tsne_2d", help="Dataset name inside the zarr group.")
    ap.add_argument("--out", type=str, default="", help="Output png path (defaults next to manifest).")
    ap.add_argument("--figsize", type=float, nargs=2, default=(40, 28), help="Figure size in inches (W H).")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--point-size", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
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
    lab_col = _pick_lab_column(meta_df)

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

    labs = meta_df[lab_col].astype(str).fillna("unknown").to_numpy()
    uniq_labs, lab_codes = np.unique(labs, return_inverse=True)

    # Build a categorical colormap with enough distinct colors
    cmap = plt.get_cmap("tab20", max(1, len(uniq_labs)))

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    ax.scatter(
        y[:, 0],
        y[:, 1],
        c=lab_codes,
        cmap=cmap,
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0,
        rasterized=True,
    )

    ax.set_title("t-SNE of embeddings (colored by lab: {})".format(lab_col))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(False)

    # Legend (can be large; place outside)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(i), markersize=6)
        for i in range(len(uniq_labs))
    ]
    ax.legend(
        handles,
        uniq_labs.tolist(),
        title="lab",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = manifest_path.parent / "tsne_by_lab.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print("[DONE] wrote", out_path)


if __name__ == "__main__":
    main()
