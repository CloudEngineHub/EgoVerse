"""
Zero-row scanner for local Zarr episodes.

For every numeric array in each episode, finds frame indices where all values
are exactly zero.  Zero quaternions (0,0,0,0) embedded in a pose vec are the
classic sign of data corruption this was designed to catch.

Uses LocalEpisodeResolver to discover only valid, readable zarr stores.

Fast by design:
  - ThreadPool parallel reads (mostly I/O-bound, no pickling overhead).
  - Reads each array as a single numpy slice then checks with .all(axis=1).
  - tqdm bar advances per episode.

--pct samples uniformly over all frames across all episodes (not just episodes).
A fast metadata-only first pass reads frame counts, then globally samples pct%
of all frame slots and maps them back to per-episode indices.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from egomimic.rldb.zarr import LocalEpisodeResolver


# ── per-episode workers ───────────────────────────────────────────────────────


def _get_frame_count(ep_path: Path, episode_hash: str) -> tuple[str, int]:
    """Read only zarr attrs to get total_frames (no array data loaded)."""
    try:
        g = zarr.open_group(str(ep_path), mode="r")
        return episode_hash, int(g.attrs.get("total_frames", 0) or 0)
    except Exception:
        return episode_hash, 0


def _scan_episode(
    ep_path: Path,
    episode_hash: str,
    frame_indices: list[int] | None = None,
) -> dict:
    """
    Open the zarr store, check every numeric (T, ...) array for all-zero rows.

    Args:
        frame_indices: If given, only these frame indices are checked and
                       zero_rows entries contain the original frame numbers.
                       If None, all frames are checked.

    Returns:
        {episode_hash, total_frames, frames_scanned, zero_rows, error}
    """
    eh = episode_hash
    try:
        g = zarr.open_group(str(ep_path), mode="r")
        total_frames = int(g.attrs.get("total_frames", 0) or 0)

        idx = np.array(sorted(frame_indices), dtype=int) if frame_indices is not None else None
        frames_scanned = len(idx) if idx is not None else total_frames

        zero_rows: dict[str, list[int]] = {}
        for key in g.keys():
            arr = g[key]
            # Skip non-numeric or 1-D-only (annotations, jpeg stores)
            if arr.ndim < 2 or not np.issubdtype(arr.dtype, np.number):
                continue
            data: np.ndarray = arr.oindex[idx] if idx is not None else arr[:]
            T = data.shape[0]
            flat = data.reshape(T, -1)          # (T, features)
            zero_mask = (flat == 0).all(axis=1)
            bad_local = np.where(zero_mask)[0]
            # Map back to original frame numbers
            bad = (idx[bad_local].tolist() if idx is not None else bad_local.tolist())
            if bad:
                zero_rows[key] = bad

        return {
            "episode_hash": eh,
            "total_frames": total_frames,
            "frames_scanned": frames_scanned,
            "zero_rows": zero_rows,
            "error": None,
        }

    except Exception as e:
        return {
            "episode_hash": eh,
            "total_frames": 0,
            "frames_scanned": 0,
            "zero_rows": {},
            "error": str(e),
        }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan local Zarr episodes for all-zero rows in numeric arrays."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=str,
        help="Directory containing episode dirs.",
    )
    parser.add_argument(
        "--pct",
        default=100.0,
        type=float,
        help=(
            "Percentage of frames to scan (default 100 = all). "
            "Frames are sampled uniformly over the full frame pool across all episodes."
        ),
    )
    parser.add_argument(
        "--workers",
        default=32,
        type=int,
        help="Parallel threads (default 32).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"Error: not a directory: {dataset_root}")
        return 2

    # Use the resolver to enumerate only valid, readable zarr stores.
    print("Resolving valid episodes...")
    raw = LocalEpisodeResolver._get_local_filtered_paths(dataset_root, filters={})
    if not raw:
        print("No valid zarr episodes found.")
        return 2

    workers = max(1, args.workers)
    eps: list[tuple[Path, str]] = [(Path(p), eh) for p, eh in raw]

    # ── frame-level sampling ──────────────────────────────────────────────────
    # per_ep_frames: maps episode_hash -> sorted list of local frame indices to scan.
    # Empty dict means scan all frames.
    per_ep_frames: dict[str, list[int]] = {}

    if args.pct < 100.0:
        # Fast metadata-only pass to get each episode's frame count.
        print(f"Reading frame counts for {len(eps)} episodes...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fc_futures = {ex.submit(_get_frame_count, p, eh): eh for p, eh in eps}
            frame_counts: dict[str, int] = {}
            for fut in tqdm(
                concurrent.futures.as_completed(fc_futures),
                total=len(fc_futures),
                desc="frame counts",
                unit="ep",
                dynamic_ncols=True,
            ):
                eh = fc_futures[fut]
                _, n = fut.result()
                frame_counts[eh] = n

        # Build the global frame pool and sample pct% uniformly.
        # Each episode occupies a contiguous range [offset, offset+n_frames).
        ep_ranges: list[tuple[Path, str, int, int]] = []  # (path, hash, global_start, n_frames)
        offset = 0
        for p, eh in eps:
            n = frame_counts.get(eh, 0)
            if n > 0:
                ep_ranges.append((p, eh, offset, n))
                offset += n
        total_frames_global = offset

        n_sample = max(1, int(round(total_frames_global * args.pct / 100.0)))
        rng = random.Random(args.seed)
        global_sample = sorted(
            rng.sample(range(total_frames_global), min(n_sample, total_frames_global))
        )

        # Map global indices back to per-episode local indices (single linear scan).
        gi = 0
        for p, eh, ep_start, ep_n in ep_ranges:
            ep_end = ep_start + ep_n
            local: list[int] = []
            while gi < len(global_sample) and global_sample[gi] < ep_end:
                local.append(global_sample[gi] - ep_start)
                gi += 1
            if local:
                per_ep_frames[eh] = local

        # Only visit episodes that received at least one sampled frame.
        eps = [(p, eh) for p, eh in eps if eh in per_ep_frames]
        print(
            f"Uniform frame sample: {len(global_sample):,} / {total_frames_global:,} frames "
            f"({args.pct:.1f}%) across {len(eps)} episodes."
        )
    else:
        print(f"Scanning all frames in {len(eps)} episodes with {workers} threads...")

    # ── parallel scan ─────────────────────────────────────────────────────────
    total_episodes_with_zeros = 0
    total_zero_frames = 0
    total_frames_scanned = 0
    scan_errors: list[str] = []

    # {episode_hash: {key: [bad_frame_indices]}}
    results_with_zeros: dict[str, dict[str, list[int]]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(_scan_episode, p, eh, per_ep_frames.get(eh)): eh
            for p, eh in eps
        }
        pbar = tqdm(
            total=len(eps),
            desc="scanning",
            unit="ep",
            dynamic_ncols=True,
        )
        for fut in concurrent.futures.as_completed(future_map):
            pbar.update(1)
            res = fut.result()
            eh = res["episode_hash"]

            if res["error"]:
                scan_errors.append(f"{eh}: {res['error']}")
                tqdm.write(f"[ERROR] {eh}: {res['error']}")
                continue

            total_frames_scanned += res["frames_scanned"]

            if res["zero_rows"]:
                total_episodes_with_zeros += 1
                results_with_zeros[eh] = res["zero_rows"]
                all_bad = set()
                for bad_idx in res["zero_rows"].values():
                    all_bad.update(bad_idx)
                n_bad = len(all_bad)
                total_zero_frames += n_bad
                keys_str = ", ".join(
                    f"{k}({len(v)})" for k, v in res["zero_rows"].items()
                )
                tqdm.write(
                    f"[ZEROS] ep={eh}  "
                    f"{n_bad} frame(s) with all-zero rows  [{keys_str}]"
                )
    pbar.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    frame_pct = (total_zero_frames / total_frames_scanned * 100) if total_frames_scanned > 0 else 0.0
    ep_pct = (total_episodes_with_zeros / len(eps) * 100) if eps else 0.0

    print("=" * 60)
    print(f"Episodes scanned       : {len(eps)}")
    print(f"Scan errors            : {len(scan_errors)}")
    print(f"Episodes with zeros    : {total_episodes_with_zeros}  ({ep_pct:.1f}%)")
    print(f"Total frames scanned   : {total_frames_scanned:,}")
    print(f"Total bad frame slots  : {total_zero_frames}  ({frame_pct:.2f}% of scanned frames)")

    if results_with_zeros:
        print()
        print("Episodes with zero rows (sorted by bad-frame count):")
        sorted_eps = sorted(
            results_with_zeros.items(),
            key=lambda x: -len(set().union(*x[1].values())),
        )
        for eh, zero_rows in sorted_eps:
            all_bad = sorted(set().union(*zero_rows.values()))
            keys_str = ", ".join(zero_rows.keys())
            print(f"  {eh}")
            print(f"    keys   : {keys_str}")
            print(f"    frames : {len(all_bad)}  {all_bad[:20]}{'...' if len(all_bad) > 20 else ''}")
    print("=" * 60)

    return 1 if (results_with_zeros or scan_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
