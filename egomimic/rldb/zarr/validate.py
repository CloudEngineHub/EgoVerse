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

from egomimic.rldb.zarr.episode_rules import (
    arm_sides,
    camera_errors,
    embodiment_error,
    fps_error,
)
from egomimic.rldb.zarr.required_metadata import check_stored_metadata

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


def validate_episode(
    zarr_path: str | Path, *, allow_missing: bool = False
) -> ValidationReport:
    """Check a finished .zarr episode. Empty ``report.errors`` = pass."""
    report = ValidationReport(path=Path(zarr_path))
    try:
        store = zarr.open_group(str(zarr_path), mode="r", zarr_format=3)
    except Exception as e:
        report.errors.append(f"cannot open as a zarr v3 group: {e}")
        return report
    try:
        _validate_store(store, report, allow_missing)
    except Exception as e:  # malformed attrs the checks below assume are well-formed
        report.errors.append(f"could not finish validating: {type(e).__name__}: {e}")
    return report


def _validate_store(
    store: zarr.Group, report: ValidationReport, allow_missing: bool
) -> None:
    meta = dict(store.attrs)
    _check_attrs(meta, report, allow_missing)
    if "features" not in meta:
        report.errors.append("Missing metadata field: features")
    T = meta.get("total_frames")
    if type(T) is not int or T <= 0:
        report.errors.append(f"total_frames must be a positive int, got {T!r}")
        return
    features = meta.get("features") or {}
    _check_arrays(store, features, meta.get("embodiment"), T, report.errors)
    _check_annotations(store, features, T, report.errors)
    _check_images(store, features, report.errors)


def _check_attrs(meta: dict, report: ValidationReport, allow_missing: bool) -> None:
    for name, err in (
        ("fps", fps_error(meta.get("fps"))),
        ("embodiment", embodiment_error(meta.get("embodiment"))),
    ):
        if err is None:
            report.successes.append(f"{name}={meta[name]!r}")
        else:
            report.errors.append(err)
    report.errors += camera_errors(meta.get("intrinsics"), meta.get("extrinsics"))
    errors, warnings, passed = check_stored_metadata(meta, allow_missing=allow_missing)
    report.errors += errors
    report.warnings += warnings
    report.successes += [f"{name}={meta[name]!r}" for name in passed]


def _check_arrays(
    store: zarr.Group, features: dict, embodiment, T: int, errors: list[str]
) -> None:
    array_keys = list(store.array_keys())
    for key in array_keys:
        if key not in features:
            errors.append(f"{key}: array has no entry in features")
        if features.get(key, {}).get("dtype") == "json":
            continue
        if not store[key].shape:
            errors.append(f"{key}: array is 0-dimensional, expected a time axis")
            continue
        if (arr_len := store[key].shape[0]) < T:
            errors.append(f"{key}: array length {arr_len} < total_frames {T}")
    for key in features:
        if key not in array_keys:
            errors.append(f"features[{key!r}] has no matching array")

    arm_keys = [f"{side}.obs_ee_pose" for side in arm_sides(str(embodiment or ""))]
    for key in ["images.front_1", *arm_keys]:
        if key not in store:
            errors.append(f"Missing required key: {key}")

    for key in _POSE_KEYS:
        if key not in store:
            continue
        arr = store[key][:T]
        if arr.ndim != 2 or arr.shape[-1] != 7:
            errors.append(f"{key}: expected shape (T, 7), got {arr.shape}")
            continue
        if not np.isfinite(arr).all():
            errors.append(f"{key}: contains NaN or inf")
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
    for key in _GRIPPER_KEYS + _KEYPOINT_KEYS:
        if key in store and not np.isfinite(store[key][:T]).all():
            errors.append(f"{key}: contains NaN or inf")


def _annotation_error(raw, T: int) -> str | None:
    try:
        rec = json.loads(
            raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        )
    except (TypeError, ValueError) as e:
        return str(e)
    if not isinstance(rec, dict):
        return f"record is {type(rec).__name__}, expected dict"
    for name, expected in (("text", str), ("start_idx", int), ("end_idx", int)):
        if not isinstance(rec.get(name), expected):
            return f"field {name!r} missing or not {expected}"
    if not (0 <= rec["start_idx"] <= rec["end_idx"] <= T):
        return (
            f"index range invalid: start={rec['start_idx']}, "
            f"end={rec['end_idx']}, T={T}"
        )
    return None


def _check_annotations(
    store: zarr.Group, features: dict, T: int, errors: list[str]
) -> None:
    for key, feat in features.items():
        if feat.get("dtype") != "json" or key not in store:
            continue
        node = store[key]
        found = [
            (i, err)
            for i in range(node.shape[0])
            if (err := _annotation_error(_unwrap_bytes(node[i]), T)) is not None
        ]
        if found:
            errors.append(
                f"{key}: {len(found)}/{node.shape[0]} annotations malformed "
                f"(e.g. index {found[0][0]}: {found[0][1]})"
            )


def _check_images(store: zarr.Group, features: dict, errors: list[str]) -> None:
    """Spot-check that frame 0 of each JPEG key decodes to RGB."""
    for key, feat in features.items():
        if feat.get("dtype") != "jpeg" or key not in store:
            continue
        try:
            frame = simplejpeg.decode_jpeg(
                bytes(_unwrap_bytes(store[key][0])), colorspace="RGB"
            )
        except Exception as e:
            errors.append(f"{key}: failed to decode frame 0: {e}")
            continue
        if frame.ndim != 3 or frame.shape[2] != 3:
            errors.append(f"{key}: decoded frame has shape {frame.shape}")


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
