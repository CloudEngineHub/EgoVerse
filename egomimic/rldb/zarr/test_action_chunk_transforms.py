#!/usr/bin/env python3
"""
Verification script for action chunk transforms.

Tests:
  1. Unit tests — _interpolate_euler matches egomimicUtils.interpolate_arr_euler
  2. Unit tests — transform classes produce correct shapes and handle edge cases
  3. Integration test — ZarrDataset with action_horizon + chunk_length on real zarr data
  4. Backwards compatibility — chunk_length=None leaves behaviour unchanged

Run:
    python test_action_chunk_transforms.py
"""

import sys
import os
import traceback

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

ZARR_SINGLE_ARM = os.path.join(
    SCRIPT_DIR, "zarr", "1ts", "1762544332273_dup58_dup7.zarr"
)
ZARR_BIMANUAL = os.path.join(
    SCRIPT_DIR, "zarr", "bimanual", "1769460905119.zarr"
)

# ── Helpers ────────────────────────────────────────────────────────────────
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unit: _interpolate_euler matches interpolate_arr_euler
# ═══════════════════════════════════════════════════════════════════════════
def test_match_reference():
    """Our _interpolate_euler should produce identical output to the
    reference interpolate_arr_euler from egomimicUtils (which takes (B,T,D))."""
    print("\n── Test: _interpolate_euler matches reference ──")
    from egomimic.rldb.zarr.action_chunk_transforms import _interpolate_euler
    from egomimic.utils.egomimicUtils import interpolate_arr_euler

    rng = np.random.RandomState(42)
    chunk_length = 100

    # 6D case
    seq_6d = rng.randn(10, 6).astype(np.float64)
    # Ensure angles wrap around to test unwrapping
    seq_6d[:, 3:6] = np.linspace(-3.5, 3.5, 10)[:, None] * np.array([1.0, 0.7, -0.5])

    ours_6d = _interpolate_euler(seq_6d, chunk_length)
    ref_6d = interpolate_arr_euler(seq_6d[np.newaxis], chunk_length)[0]

    check("6D shape", ours_6d.shape == (chunk_length, 6), f"got {ours_6d.shape}")
    check("6D values match", np.allclose(ours_6d, ref_6d, atol=1e-12),
          f"max diff = {np.max(np.abs(ours_6d - ref_6d))}")

    # 7D case (with gripper)
    seq_7d = rng.randn(10, 7).astype(np.float64)
    seq_7d[:, 3:6] = np.linspace(-3.5, 3.5, 10)[:, None] * np.array([1.0, 0.7, -0.5])

    ours_7d = _interpolate_euler(seq_7d, chunk_length)
    ref_7d = interpolate_arr_euler(seq_7d[np.newaxis], chunk_length)[0]

    check("7D shape", ours_7d.shape == (chunk_length, 7), f"got {ours_7d.shape}")
    check("7D values match", np.allclose(ours_7d, ref_7d, atol=1e-12),
          f"max diff = {np.max(np.abs(ours_7d - ref_7d))}")

    # Bad-data sentinel
    seq_bad = seq_6d.copy()
    seq_bad[3, 0] = 1e9
    ours_bad = _interpolate_euler(seq_bad, chunk_length)
    ref_bad = interpolate_arr_euler(seq_bad[np.newaxis], chunk_length)[0]

    check("bad-data sentinel shape", ours_bad.shape == (chunk_length, 6))
    check("bad-data sentinel all 1e9", np.all(ours_bad == 1e9))
    check("bad-data sentinel matches ref", np.array_equal(ours_bad, ref_bad))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Unit: Transform classes — shapes and dispatch
# ═══════════════════════════════════════════════════════════════════════════
def test_transform_classes():
    print("\n── Test: Transform class shapes and dispatch ──")
    from egomimic.rldb.zarr.action_chunk_transforms import (
        SingleArmEulerTransform,
        SingleArmEulerGripTransform,
        BimanualEulerTransform,
        BimanualEulerGripTransform,
    )

    rng = np.random.RandomState(99)
    T, CL = 10, 100

    # SingleArmEulerTransform (6D)
    t = SingleArmEulerTransform()
    a6 = rng.randn(T, 6)
    out_cart = t.transform(a6, CL, key="actions_cartesian")
    out_joint = t.transform(a6, CL, key="actions_joints")
    check("SingleArm6D cartesian shape", out_cart.shape == (CL, 6), f"got {out_cart.shape}")
    check("SingleArm6D joints shape", out_joint.shape == (CL, 6), f"got {out_joint.shape}")

    # SingleArmEulerGripTransform (7D)
    t7 = SingleArmEulerGripTransform()
    a7 = rng.randn(T, 7)
    out7_cart = t7.transform(a7, CL, key="actions_cartesian")
    out7_joint = t7.transform(a7, CL, key="actions_joints")
    check("SingleArm7D cartesian shape", out7_cart.shape == (CL, 7), f"got {out7_cart.shape}")
    check("SingleArm7D joints shape", out7_joint.shape == (CL, 7), f"got {out7_joint.shape}")

    # BimanualEulerTransform (12D)
    tb = BimanualEulerTransform()
    a12 = rng.randn(T, 12)
    out12_cart = tb.transform(a12, CL, key="actions_cartesian")
    out12_joint = tb.transform(a12, CL, key="actions_joints")
    check("Bimanual12D cartesian shape", out12_cart.shape == (CL, 12), f"got {out12_cart.shape}")
    check("Bimanual12D joints shape", out12_joint.shape == (CL, 12), f"got {out12_joint.shape}")

    # BimanualEulerGripTransform (14D)
    tbg = BimanualEulerGripTransform()
    a14 = rng.randn(T, 14)
    out14_cart = tbg.transform(a14, CL, key="actions_cartesian")
    out14_joint = tbg.transform(a14, CL, key="actions_joints")
    check("Bimanual14D cartesian shape", out14_cart.shape == (CL, 14), f"got {out14_cart.shape}")
    check("Bimanual14D joints shape", out14_joint.shape == (CL, 14), f"got {out14_joint.shape}")

    # Verify endpoints are preserved (first and last input frame)
    a6_clean = rng.randn(T, 6) * 0.5  # small values, no wrapping issues
    out_ep = t.transform(a6_clean, CL, key="actions_joints")
    check("endpoints preserved (first)", np.allclose(out_ep[0], a6_clean[0], atol=1e-10))
    check("endpoints preserved (last)", np.allclose(out_ep[-1], a6_clean[-1], atol=1e-10))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Unit: Registry covers all embodiments
# ═══════════════════════════════════════════════════════════════════════════
def test_registry():
    print("\n── Test: Registry covers all embodiments ──")
    from egomimic.rldb.zarr.action_chunk_transforms import get_action_chunk_transform, ACTION_CHUNK_TRANSFORMS
    from egomimic.rldb.zarr.zarr_dataset_multi import EMBODIMENT

    for member in EMBODIMENT:
        try:
            t = get_action_chunk_transform(member.value)
            check(f"registry has {member.name} ({member.value})", t is not None)
        except KeyError:
            check(f"registry has {member.name} ({member.value})", False, "KeyError")

    # Invalid ID should raise
    try:
        get_action_chunk_transform(999)
        check("invalid ID raises KeyError", False, "no exception raised")
    except KeyError:
        check("invalid ID raises KeyError", True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration: ZarrDataset with real zarr data
# ═══════════════════════════════════════════════════════════════════════════
def test_zarr_dataset_integration():
    print("\n── Test: ZarrDataset integration ──")
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    ACTION_HORIZON = 10
    CHUNK_LENGTH = 100

    # --- Single-arm 7D (eva_left_arm) ---
    if os.path.isdir(ZARR_SINGLE_ARM):
        ds = ZarrDataset(ZARR_SINGLE_ARM, action_horizon=ACTION_HORIZON, chunk_length=CHUNK_LENGTH)
        check("single-arm dataset loaded", True)
        check(f"single-arm len > 0", len(ds) > 0, f"len={len(ds)}")

        sample = ds[0]
        if "actions_cartesian" in sample:
            shape = tuple(sample["actions_cartesian"].shape)
            check("single-arm actions_cartesian shape", shape == (CHUNK_LENGTH, 7),
                  f"expected (100, 7), got {shape}")
        if "actions_joints" in sample:
            shape = tuple(sample["actions_joints"].shape)
            check("single-arm actions_joints shape", shape == (CHUNK_LENGTH, 7),
                  f"expected (100, 7), got {shape}")

        # Check near end-of-episode (padding + transform)
        sample_end = ds[len(ds) - 2]
        if "actions_cartesian" in sample_end:
            shape = tuple(sample_end["actions_cartesian"].shape)
            check("single-arm end-of-ep cartesian shape", shape == (CHUNK_LENGTH, 7),
                  f"expected (100, 7), got {shape}")
    else:
        print(f"  SKIP  single-arm zarr not found at {ZARR_SINGLE_ARM}")

    # --- Bimanual 14D (eva_bimanual) ---
    if os.path.isdir(ZARR_BIMANUAL):
        ds_bi = ZarrDataset(ZARR_BIMANUAL, action_horizon=ACTION_HORIZON, chunk_length=CHUNK_LENGTH)
        check("bimanual dataset loaded", True)

        sample_bi = ds_bi[0]
        if "actions_cartesian" in sample_bi:
            shape = tuple(sample_bi["actions_cartesian"].shape)
            check("bimanual actions_cartesian shape", shape == (CHUNK_LENGTH, 14),
                  f"expected (100, 14), got {shape}")
        if "actions_joints" in sample_bi:
            shape = tuple(sample_bi["actions_joints"].shape)
            check("bimanual actions_joints shape", shape == (CHUNK_LENGTH, 14),
                  f"expected (100, 14), got {shape}")
    else:
        print(f"  SKIP  bimanual zarr not found at {ZARR_BIMANUAL}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Backwards compatibility: chunk_length=None
# ═══════════════════════════════════════════════════════════════════════════
def test_backwards_compat():
    print("\n── Test: Backwards compatibility (chunk_length=None) ──")
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    if not os.path.isdir(ZARR_SINGLE_ARM):
        print(f"  SKIP  zarr not found at {ZARR_SINGLE_ARM}")
        return

    # No action chunking at all
    ds_none = ZarrDataset(ZARR_SINGLE_ARM)
    sample = ds_none[0]
    if "actions_cartesian" in sample:
        check("no-chunk: actions_cartesian is 1D", sample["actions_cartesian"].ndim == 1,
              f"ndim={sample['actions_cartesian'].ndim}")

    # action_horizon only, no chunk_length
    ds_horizon = ZarrDataset(ZARR_SINGLE_ARM, action_horizon=10)
    sample_h = ds_horizon[0]
    if "actions_cartesian" in sample_h:
        shape = tuple(sample_h["actions_cartesian"].shape)
        check("horizon-only: actions_cartesian shape (10, 7)", shape == (10, 7),
              f"got {shape}")

    # Both set
    ds_both = ZarrDataset(ZARR_SINGLE_ARM, action_horizon=10, chunk_length=100)
    sample_b = ds_both[0]
    if "actions_cartesian" in sample_b:
        shape = tuple(sample_b["actions_cartesian"].shape)
        check("both: actions_cartesian shape (100, 7)", shape == (100, 7),
              f"got {shape}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Angle wrapping correctness
# ═══════════════════════════════════════════════════════════════════════════
def test_angle_wrapping():
    """Verify that interpolating across -pi/+pi boundary produces smooth results."""
    print("\n── Test: Angle wrapping correctness ──")
    from egomimic.rldb.zarr.action_chunk_transforms import _interpolate_euler

    T, CL = 5, 50
    seq = np.zeros((T, 6))
    # Translation stays at 0
    # Yaw goes from +3.0 to -3.0 (crossing the ±pi boundary the short way)
    seq[:, 3] = np.array([3.0, 3.05, 3.1, -3.1, -3.0])

    out = _interpolate_euler(seq, CL)

    # All interpolated yaw values should stay near ±pi, not swing through 0
    yaw = out[:, 3]
    check("angle wrapping: yaw stays near ±pi",
          np.all(np.abs(yaw) > 2.5),
          f"min |yaw| = {np.min(np.abs(yaw)):.4f}")
    check("angle wrapping: output in [-pi, pi)",
          np.all(yaw >= -np.pi) and np.all(yaw < np.pi),
          f"range = [{yaw.min():.4f}, {yaw.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        test_match_reference,
        test_transform_classes,
        test_registry,
        test_angle_wrapping,
        test_zarr_dataset_integration,
        test_backwards_compat,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            print(f"\n  ERROR in {test_fn.__name__}:")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
