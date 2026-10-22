"""
Validate a contributed Zarr episode against CONTRIBUTING_DATA.md §6 / §11.1.

Run on every episode before uploading (vendors) and on ingest (RL2):

    python -m egomimic.rldb.zarr.validate ep1.zarr ep2.zarr ...

Exits nonzero if any episode has errors. Required metadata fields written as
MissingMetadata() (stored as null) are errors unless --allow-missing is passed.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simplejpeg
import zarr

from egomimic.rldb.zarr.required_metadata import (
    REQUIRED_METADATA_FIELDS,
    required_metadata_error,
)

_POSE_KEYS = (
    "left.obs_ee_pose",
    "right.obs_ee_pose",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "obs_head_pose",
    "left.cmd_ee_pose",
    "right.cmd_ee_pose",
)
_GRIPPER_KEYS = (
    "left.obs_gripper",
    "right.obs_gripper",
    "left.gripper",
    "right.gripper",
)
_KEYPOINT_KEYS = ("left.obs_keypoints", "right.obs_keypoints")


@dataclass
class ValidationReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    successes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _unwrap_bytes(raw):
    """Unwrap nested 0-d object/bytes ndarrays down to raw bytes."""
    while isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.flat[0]
    if isinstance(raw, np.bytes_):
        raw = bytes(raw)
    return raw


def _required_arm_keys(embodiment: str) -> list[str]:
    emb = (embodiment or "").lower()
    if emb.endswith("_left_arm"):
        return ["left.obs_ee_pose"]
    if emb.endswith("_right_arm"):
        return ["right.obs_ee_pose"]
    return ["left.obs_ee_pose", "right.obs_ee_pose"]


def _check_required_metadata(meta: dict, report: ValidationReport, allow_missing: bool):
    for name in REQUIRED_METADATA_FIELDS:
        if name not in meta:
            report.errors.append(
                f"Missing metadata field: {name}. See CONTRIBUTING_DATA.md §6.3."
            )
            continue
        if meta[name] is None:  # written as MissingMetadata()
            msg = f"{name}: marked as not available (MissingMetadata)"
            if allow_missing:
                report.warnings.append(msg)
            else:
                report.errors.append(
                    msg + ". We heavily recommend providing it; pass "
                    "--allow-missing to accept the episode anyway."
                )
            continue
        err = required_metadata_error(name, meta[name])
        if err is not None:
            report.errors.append(err)
        else:
            report.successes.append(f"{name}={meta[name]!r}")


def validate_episode(
    zarr_path: str | Path, *, allow_missing: bool = False
) -> ValidationReport:
    """Check a finished .zarr episode. Empty ``report.errors`` = pass."""
    report = ValidationReport(path=Path(zarr_path))
    errors, successes = report.errors, report.successes
    try:
        store = zarr.open_group(str(zarr_path), mode="r", zarr_format=3)
    except Exception as e:
        errors.append(f"cannot open as a zarr v3 group: {e}")
        return report
    meta = dict(store.attrs)

    # ── Metadata ────────────────────────────────────────────────────────────
    for name in ("embodiment", "total_frames", "fps", "features"):
        if name not in meta:
            errors.append(f"Missing metadata field: {name}")
    if "total_frames" not in meta:
        return report
    T = int(meta["total_frames"])

    if meta.get("fps", 0) not in (30, 60):
        errors.append(f"Unexpected fps={meta.get('fps')}. Expected 30 or 60.")
    else:
        successes.append(f"fps={meta['fps']}")

    _check_required_metadata(meta, report, allow_missing)

    # ── Embodiment identifier (must resolve to a valid id; see §9) ──────────
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    try:
        get_embodiment_id(meta.get("embodiment", ""))
        successes.append(f"embodiment={meta.get('embodiment')}")
    except (KeyError, AttributeError):
        errors.append(
            f"embodiment={meta.get('embodiment')!r} is not a valid identifier (see §9)"
        )

    # ── Camera intrinsics (MANDATORY; {camera_key: 3x4 K matrix} dict) ──────
    intr = meta.get("intrinsics")
    if not isinstance(intr, dict) or not intr:
        errors.append("intrinsics: missing or not a non-empty {camera_key: 3x4} dict")
    else:
        if not any("front" in str(k).lower() for k in intr):
            errors.append(f"intrinsics: no front-camera entry (keys: {list(intr)})")
        for cam, K in intr.items():
            shape = np.asarray(K, dtype=float).shape
            if shape != (3, 4):
                errors.append(f"intrinsics[{cam!r}]: expected 3x4 K, got {shape}")
            else:
                successes.append(f"intrinsics[{cam!r}]: 3x4")

    # ── Camera extrinsics (OPTIONAL; None, or a non-empty dict of 4x4) ──────
    extr = meta.get("extrinsics")
    if extr is not None:
        if not isinstance(extr, dict) or not extr:
            errors.append("extrinsics: present but not a non-empty dict")
        else:
            for k, T_ext in extr.items():
                shape = np.asarray(T_ext, dtype=float).shape
                if shape != (4, 4):
                    errors.append(f"extrinsics[{k!r}]: expected 4x4, got {shape}")

    # ── Features ↔ arrays, frame counts ─────────────────────────────────────
    features = meta.get("features", {}) or {}
    array_keys = [k for k in store.array_keys()]
    for key in array_keys:
        if key not in features:
            errors.append(f"{key}: array has no entry in features")
        if features.get(key, {}).get("dtype") == "json":
            continue
        arr_len = store[key].shape[0]
        if arr_len < T:
            errors.append(f"{key}: array length {arr_len} < total_frames {T}")
    for key in features:
        if key not in array_keys:
            errors.append(f"features[{key!r}] has no matching array")

    # ── Required keys ───────────────────────────────────────────────────────
    for key in ["images.front_1"] + _required_arm_keys(meta.get("embodiment", "")):
        if key not in store:
            errors.append(f"Missing required key: {key}")

    # ── Pose shapes and quaternion norms ────────────────────────────────────
    for key in _POSE_KEYS:
        if key not in store:
            continue
        arr = store[key][:T]
        if arr.ndim != 2 or arr.shape[-1] != 7:
            errors.append(f"{key}: expected shape (T, 7), got {arr.shape}")
            continue
        norms = np.linalg.norm(arr[:, 3:7], axis=1)
        bad = np.where(np.abs(norms - 1.0) > 1e-4)[0]
        if len(bad):
            errors.append(
                f"{key}: {len(bad)} frames with non-unit quaternions "
                f"(e.g. frame {bad[0]}, norm={norms[bad[0]]:.6f})"
            )

    for key in _GRIPPER_KEYS:
        if key in store:
            shape = store[key].shape
            if len(shape) != 2 or shape[-1] != 1:
                errors.append(f"{key}: expected shape (T, 1), got {shape}")

    for key in _KEYPOINT_KEYS:
        if key in store and store[key].shape[-1] != 63:
            errors.append(
                f"{key}: expected last dim 63 (21x3), got {store[key].shape[-1]}"
            )

    # ── Annotation format (JSON-encoded records) ────────────────────────────
    for key, feat in features.items():
        if feat.get("dtype") != "json" or key not in store:
            continue
        node = store[key]
        bad, first_err = 0, None
        for i in range(node.shape[0]):
            raw = _unwrap_bytes(node[i])
            try:
                rec = json.loads(
                    raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                )
                if not isinstance(rec, dict):
                    raise ValueError(f"record is {type(rec).__name__}, expected dict")
                for name, expected in (
                    ("text", str),
                    ("start_idx", int),
                    ("end_idx", int),
                ):
                    if not isinstance(rec.get(name), expected):
                        raise ValueError(f"field {name!r} missing or not {expected}")
                if not (0 <= rec["start_idx"] <= rec["end_idx"] <= T):
                    raise ValueError(
                        f"index range invalid: start={rec['start_idx']}, "
                        f"end={rec['end_idx']}, T={T}"
                    )
            except Exception as e:
                bad += 1
                first_err = first_err or (i, str(e))
        if bad:
            errors.append(
                f"{key}: {bad}/{node.shape[0]} annotations malformed "
                f"(e.g. index {first_err[0]}: {first_err[1]})"
            )

    # ── Image decodability (spot-check frame 0 of each JPEG key) ────────────
    for key, feat in features.items():
        if feat.get("dtype") != "jpeg" or key not in store:
            continue
        try:
            frame = simplejpeg.decode_jpeg(
                bytes(_unwrap_bytes(store[key][0])), colorspace="RGB"
            )
            if frame.ndim != 3 or frame.shape[2] != 3:
                errors.append(f"{key}: decoded frame has shape {frame.shape}")
        except Exception as e:
            errors.append(f"{key}: failed to decode frame 0: {e}")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("episodes", nargs="+", type=Path, help=".zarr episode dirs")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Accept required metadata fields written as MissingMetadata (warn instead of fail)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print passes")
    args = parser.parse_args(argv)

    n_failed = 0
    for path in args.episodes:
        report = validate_episode(path, allow_missing=args.allow_missing)
        print(f"{'PASS' if report.ok else 'FAIL'} {path}")
        if args.verbose:
            for s in report.successes:
                print(f"  ok:      {s}")
        for w in report.warnings:
            print(f"  warning: {w}")
        for e in report.errors:
            print(f"  error:   {e}")
        n_failed += not report.ok
    print(f"{len(args.episodes) - n_failed}/{len(args.episodes)} episodes passed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
