"""
Reference converter: your raw episodes -> EgoVerse zarr episodes.

Copy this file and replace the two functions marked YOUR CODE:

- ``find_raw_episodes(raw_root)`` — list one entry per raw episode.
- ``load_episode(raw_path)`` — read one raw episode into a ``RawEpisode``:
  arrays in the shapes / frames below plus that episode's metadata.

Everything else (writing the zarr, the episode hash name, validation, the MP4
preview, batch conversion) is done for you. As shipped, ``load_episode``
returns synthetic data with ``lab="template"``, which validation rejects, so
every episode fails until you put in your own metadata:

    python egomimic/scripts/convert_template/template_to_zarr.py \\
        --raw-root /path/to/raw --output-dir /path/to/out --workers 8

or from your own batch script:

    from template_to_zarr import convert_all
    ok, failed = convert_all(paths, output_dir, workers=8)

Each episode is validated right after it is written (CONTRIBUTING_DATA.md
§11.1); an episode that fails is deleted and reported, so whatever ends up in
``output_dir`` is ready to upload. An episode with scene or objects set to
MissingMetadata() fails unless you pass ``--allow-missing``
(``allow_missing=True``). The MP4 preview needs ffmpeg; if it fails the episode
is kept and a warning is logged (``--no-mp4`` skips it). Spec:
CONTRIBUTING_DATA.md §4 (episode hash), §6 (arrays + metadata), §7 (coordinate
frames).
"""

import argparse
import fcntl
import hashlib
import logging
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from egomimic.rldb.zarr.episode_rules import start_time_error
from egomimic.rldb.zarr.required_metadata import (
    MissingMetadata,  # noqa: F401 (see load_episode)
)
from egomimic.rldb.zarr.validate import validate_episode
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.video_utils import save_preview_mp4

logger = logging.getLogger(__name__)


@dataclass
class RawEpisode:
    """One episode, already in EgoVerse conventions.

    Poses are (T, 7) = xyz + quaternion **wxyz**, expressed in your SLAM world
    frame (§7.1), at the RGB frame rate — interpolate first if your tracker is
    slower (§3.1). Leave a hand as None if it isn't tracked.
    """

    start_time_ms: int  # UTC epoch ms at recording start -> episode hash (§4)
    fps: int  # actual capture rate of `images` (30 or 60)
    images: np.ndarray  # (T, H, W, 3) uint8 RGB, egocentric camera
    intrinsics: np.ndarray  # 3x3 or 3x4 pinhole K at the (H, W) of `images`
    head_pose: np.ndarray  # (T, 7) world <- camera (§6.2)
    metadata: dict[str, Any]  # task_name, operator, lab, scene, objects (§6.3)
    # Your ID for the raw recording this episode was cut from, so it can be
    # re-converted from the raw upload later (§6.6).
    source_recording_id: str
    left_ee_pose: np.ndarray | None = None  # (T, 7)
    right_ee_pose: np.ndarray | None = None  # (T, 7)
    left_keypoints: np.ndarray | None = None  # (T, 63) = 21 MANO kps x xyz
    right_keypoints: np.ndarray | None = None  # (T, 63)
    task_description: str = ""
    # Language spans: (text, start_frame, end_frame), end exclusive (§8.1)
    annotations: list[tuple[str, int, int]] = field(default_factory=list)


# ── YOUR CODE ───────────────────────────────────────────────────────────────


def find_raw_episodes(raw_root: Path) -> list[Path]:
    """Return one path per raw episode under ``raw_root``.

    Replace with however your raw data is laid out (a folder per episode,
    one file per episode, rows of a manifest, ...).
    """
    return sorted(p for p in Path(raw_root).iterdir() if not p.name.startswith("."))


def load_episode(raw_path: Path) -> RawEpisode:
    """Read one raw episode. Replace this body with your own loader.

    ``metadata`` is per episode: read the operator, scene and objects from your
    own capture logs rather than hardcoding them for the whole batch — they
    are what lets people filter and train on the data. task_name, operator
    and lab are always required. If your source truly has no scene or
    objects, use MissingMetadata() for it — only if absolutely necessary, and
    convert with --allow-missing.
    """
    # Synthetic stand-in so the template runs as shipped.
    seed = int(hashlib.sha256(Path(raw_path).name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    T, H, W = 60, 360, 640
    return RawEpisode(
        start_time_ms=1_767_225_600_000 + seed,  # 2026-01-01 UTC + offset
        fps=30,
        images=rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8),
        intrinsics=np.array(
            [[250.0, 0.0, W / 2], [0.0, 250.0, H / 2], [0.0, 0.0, 1.0]]
        ),
        head_pose=_synthetic_trajectory(T, rng),
        left_ee_pose=_synthetic_trajectory(T, rng),
        right_ee_pose=_synthetic_trajectory(T, rng),
        task_description="folding a towel on a table",
        source_recording_id=Path(raw_path).name,
        annotations=[("fold the towel", 0, T)],
        metadata={
            "task_name": "fold_clothes",
            "operator": f"operator_{seed % 5}",
            "lab": "template",  # rejected: replace with your lab id
            "scene": "kitchen_a",
            "objects": ["towel"],
            # Only if absolutely necessary, e.g. your capture logs have no scene:
            # "scene": MissingMetadata(),
        },
    )


# ── Conversion (no changes needed below) ───────────────────────────────────


def convert_episode(
    raw_path: Path,
    output_dir: Path,
    *,
    save_mp4: bool = True,
    allow_missing: bool = False,
) -> Path:
    """Convert and validate one raw episode; returns the written .zarr path.

    Raises (and leaves nothing behind) if the episode fails validation.
    """
    ep = load_episode(Path(raw_path))
    if (err := start_time_error(ep.start_time_ms)) is not None:
        raise ValueError(f"{raw_path}: {err}")
    output_dir = Path(output_dir)
    episode_hash = timestamp_ms_to_episode_hash(ep.start_time_ms)
    zarr_path = output_dir / f"{episode_hash}.zarr"
    mp4_path = output_dir / f"{episode_hash}.mp4"

    numeric = {"obs_head_pose": ep.head_pose}
    for side in ("left", "right"):
        if getattr(ep, f"{side}_ee_pose") is not None:
            numeric[f"{side}.obs_ee_pose"] = getattr(ep, f"{side}_ee_pose")
        if getattr(ep, f"{side}_keypoints") is not None:
            numeric[f"{side}.obs_keypoints"] = getattr(ep, f"{side}_keypoints")

    K = np.asarray(ep.intrinsics, dtype=np.float64)
    if K.shape == (3, 3):
        K = np.hstack([K, np.zeros((3, 1))])

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f"{episode_hash}.lock"
    with _hash_lock(lock_path, zarr_path):
        if zarr_path.exists():
            raise FileExistsError(f"{zarr_path} already exists (duplicate start time?)")
        _write_and_validate(
            ep, zarr_path, mp4_path, numeric, K, save_mp4, allow_missing
        )
    return zarr_path


@contextmanager
def _hash_lock(lock_path: Path, zarr_path: Path):
    """Exclusive per-hash lock so parallel workers that share a start time can't
    overwrite each other. flock is released if the worker is killed, so a crash
    never blocks a rerun."""
    while True:
        lock = open(lock_path, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise FileExistsError(
                f"{zarr_path} is being written by another worker "
                "(duplicate start time?)"
            ) from None
        # Another worker may have unlinked the file between our open and flock;
        # a lock on an unlinked inode excludes nobody, so take a fresh one.
        try:
            if os.stat(lock_path).st_ino == os.fstat(lock.fileno()).st_ino:
                break
        except FileNotFoundError:
            pass
        lock.close()
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)
        lock.close()


def _write_and_validate(
    ep: RawEpisode,
    zarr_path: Path,
    mp4_path: Path,
    numeric: dict[str, np.ndarray],
    K: np.ndarray,
    save_mp4: bool,
    allow_missing: bool,
) -> None:
    """Write and validate one episode, removing it on failure; then the preview."""
    try:
        ZarrWriter.create_and_write(
            episode_path=zarr_path,
            numeric_data={k: np.asarray(v) for k, v in numeric.items()},
            image_data={"images.front_1": ep.images},
            embodiment=_embodiment(ep),
            fps=ep.fps,
            task_description=ep.task_description,
            annotations=ep.annotations or None,
            intrinsics={"front_1": K},
            source_recording_id=ep.source_recording_id,
            **ep.metadata,
        )
        report = validate_episode(zarr_path, allow_missing=allow_missing)
        if not report.ok:
            raise ValueError(
                f"{zarr_path.name} failed validation:\n  " + "\n  ".join(report.errors)
            )
        for warning in report.warnings:
            logger.warning("%s: %s", zarr_path.name, warning)
    except BaseException:
        shutil.rmtree(zarr_path, ignore_errors=True)
        raise
    if save_mp4:  # sibling preview for review (§6.5); the zarr is valid without it
        try:
            save_preview_mp4(ep.images.transpose(0, 3, 1, 2), mp4_path, ep.fps)
        except Exception as e:
            mp4_path.unlink(missing_ok=True)
            logger.warning("%s: MP4 preview not written: %s", zarr_path.name, e)


def convert_all(
    raw_paths: list[Path],
    output_dir: Path,
    *,
    workers: int = 1,
    save_mp4: bool = True,
    allow_missing: bool = False,
) -> tuple[list[Path], dict[Path, str]]:
    """Convert many episodes; one bad episode never stops the batch.

    Returns (written zarr paths, {raw_path: error message} for failures).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failed: dict[Path, str] = {}

    def _record(raw_path, fn):
        try:
            written.append(fn())
            logger.info("ok      %s -> %s", raw_path, written[-1].name)
        except Exception:
            failed[raw_path] = traceback.format_exc()
            logger.error("FAILED  %s\n%s", raw_path, failed[raw_path])

    opts = {"save_mp4": save_mp4, "allow_missing": allow_missing}
    if workers <= 1:
        for p in raw_paths:
            _record(p, lambda p=p: convert_episode(p, output_dir, **opts))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                p: pool.submit(convert_episode, p, output_dir, **opts)
                for p in raw_paths
            }
            for p, fut in futures.items():
                _record(p, fut.result)
    return written, failed


def _embodiment(ep: RawEpisode) -> str:
    # All human data shares one embodiment; your lab goes in `lab` (§9).
    has_left, has_right = ep.left_ee_pose is not None, ep.right_ee_pose is not None
    if has_left and has_right:
        return "human_bimanual"
    if has_left or has_right:
        return "human_left_arm" if has_left else "human_right_arm"
    raise ValueError("episode has no tracked hand (left_ee_pose / right_ee_pose)")


def _synthetic_trajectory(T: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth (T, 7) xyz + wxyz trajectory: drift plus a slow yaw."""
    xyz = np.cumsum(rng.normal(0, 0.002, (T, 3)), axis=0) + rng.normal(0, 0.3, 3)
    yaw = np.linspace(0, rng.uniform(-0.5, 0.5), T)
    quat = np.stack(
        [np.cos(yaw / 2), np.zeros(T), np.zeros(T), np.sin(yaw / 2)], axis=1
    )
    return np.concatenate([xyz, quat], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--no-mp4", action="store_true", help="Skip MP4 previews (need ffmpeg)"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep episodes with scene/objects set to MissingMetadata(). Only "
        "if absolutely necessary.",
    )
    args = parser.parse_args(argv)

    raw_paths = find_raw_episodes(args.raw_root)
    written, failed = convert_all(
        raw_paths,
        args.output_dir,
        workers=args.workers,
        save_mp4=not args.no_mp4,
        allow_missing=args.allow_missing,
    )
    print(f"{len(written)}/{len(raw_paths)} episodes written to {args.output_dir}")
    for p in failed:
        print(f"  FAILED {p}")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
