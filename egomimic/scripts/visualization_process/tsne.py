"""
Run t-SNE (GPU) on saved embedding latents and store 2D coords back into the zarr.

Reads:
- manifest.json (to find the embeddings zarr path)
- embeddings zarr group (expects dataset name "embeddings")

Writes:
- dataset "tsne_2d" into the same zarr group, shape (N, 2), float32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def _load_embeddings(zarr_path: Path) -> np.ndarray:
    root = zarr.open_group(str(zarr_path), mode="r")
    if "embeddings" not in root:
        raise KeyError(
            "Expected dataset 'embeddings' in zarr group. Found keys: {}".format(
                list(root.array_keys())
            )
        )
    arr = root["embeddings"]
    # Load entire array into memory for t-SNE
    x = arr[:]
    print("x.shape =", x.shape)
    # cuML prefers float32
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _run_cuml_tsne(
    x: np.ndarray, *, perplexity: float, random_state: int, learning_rate: float
) -> np.ndarray:
    try:
        from cuml import TSNE
    except Exception as e:
        raise RuntimeError(
            "cuml is required. Make sure RAPIDS/cuML is installed in this environment."
        ) from e

    # cuML TSNE returns a (N, 2) array-like (often cupy-backed); convert to numpy.
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        # NOTE: scikit-learn supports learning_rate="auto", but cuML expects numeric.
        learning_rate=float(learning_rate),
    )
    y = tsne.fit_transform(x)

    # Convert cupy -> numpy if needed
    try:
        import cupy as cp

        if isinstance(y, cp.ndarray):
            y = cp.asnumpy(y)
    except Exception:
        pass

    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 2:
        raise RuntimeError("Unexpected TSNE output shape: {}".format(y.shape))
    return y.astype(np.float32, copy=False)


def _write_tsne(zarr_path: Path, *, y2d: np.ndarray, name: str, overwrite: bool) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")

    if name in root and not overwrite:
        raise FileExistsError(
            "Zarr dataset '{}' already exists at {}. Use --overwrite to replace.".format(
                name, zarr_path
            )
        )

    chunks = (min(8192, y2d.shape[0]), 2)
    root.create_dataset(
        name,
        shape=y2d.shape,
        chunks=chunks,
        dtype=np.float32,
        overwrite=overwrite,
    )
    root[name][:] = y2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="egomimic/scripts/visualization_process/fold_clothes_aria_eva_all_labs/manifest.json",
        help="Path to manifest.json produced by process_image.py",
    )
    ap.add_argument(
        "--image-key",
        type=str,
        default="",
        help="Optional image key to select from manifest['embeddings'] (defaults to first).",
    )
    ap.add_argument("--out-name", type=str, default="tsne_2d", help="Dataset name to write in zarr")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="cuML TSNE learning rate (must be numeric; sklearn's 'auto' is not supported).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("embed_store") != "zarr":
        raise RuntimeError("This script expects manifest embed_store == 'zarr'.")

    if args.image_key:
        image_key = args.image_key
    else:
        image_key = manifest["image_keys"][0]

    zarr_path = Path(manifest["embeddings"][image_key])
    print("[INFO] zarr_path =", zarr_path)
    print("[INFO] reading embeddings for key =", image_key)

    x = _load_embeddings(zarr_path)
    print("[INFO] embeddings shape/dtype =", x.shape, x.dtype)

    y2d = _run_cuml_tsne(
        x, perplexity=args.perplexity, random_state=args.seed, learning_rate=args.learning_rate
    )
    print("[INFO] tsne_2d shape/dtype =", y2d.shape, y2d.dtype)

    _write_tsne(zarr_path, y2d=y2d, name=args.out_name, overwrite=args.overwrite)
    print("[DONE] wrote {} into {}".format(args.out_name, zarr_path))


if __name__ == "__main__":
    main()
