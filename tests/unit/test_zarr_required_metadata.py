"""Required metadata (operator/lab/scene/objects/task_name) in
ZarrWriter.create_and_write and the episode validator."""

import argparse

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_episode

from egomimic.rldb.zarr import validate
from egomimic.rldb.zarr.required_metadata import (
    MissingMetadata,
    add_required_metadata_args,
    build_required_metadata,
    required_metadata_from_args,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

T = 8
K = np.array([[100.0, 0, 32, 0], [0, 100.0, 32, 0], [0, 0, 1.0, 0]])
GOOD = {
    "task_name": "fold_clothes",
    "operator": "op_7",
    "lab": "rl2",
    "scene": "kitchen_a",
    "objects": ["shirt"],
}


def _write(path, **overrides):
    kwargs = {**GOOD, **overrides}
    kwargs = {k: v for k, v in kwargs.items() if v is not _DROP}
    return ZarrWriter.create_and_write(
        path,
        numeric_data={"actions": np.zeros((T, 3), np.float32)},
        embodiment="human_bimanual",
        intrinsics={"front_1": K},
        **kwargs,
    )


_DROP = object()


def _attrs(path):
    return dict(zarr.open_group(str(path), mode="r").attrs)


@pytest.mark.parametrize("field", list(GOOD))
def test_omitting_a_field_is_a_type_error(tmp_path, field):
    with pytest.raises(TypeError, match=field):
        _write(tmp_path / "ep.zarr", **{field: _DROP})


@pytest.mark.parametrize("field", list(GOOD))
@pytest.mark.parametrize("bad", [None, "", "   ", "unknown", "N/A", "debug", 3])
def test_placeholders_are_rejected(tmp_path, field, bad):
    with pytest.raises(ValueError, match=field):
        _write(tmp_path / "ep.zarr", **{field: bad})
    assert not (tmp_path / "ep.zarr").exists()


@pytest.mark.parametrize("bad", ["RL2", "rl 2", "_rl2"])
def test_lab_must_be_lowercase_identifier(tmp_path, bad):
    with pytest.raises(ValueError, match="lab"):
        _write(tmp_path / "ep.zarr", lab=bad)


def test_objects_list_rejects_empty_and_placeholder_entries(tmp_path):
    with pytest.raises(ValueError, match="objects"):
        _write(tmp_path / "a.zarr", objects=[])
    with pytest.raises(ValueError, match="objects"):
        _write(tmp_path / "b.zarr", objects=["shirt", "unknown"])


def test_values_land_in_attrs(tmp_path):
    attrs = _attrs(_write(tmp_path / "ep.zarr"))
    for k, v in GOOD.items():
        assert attrs[k] == v
    assert "missing_metadata" not in attrs


@pytest.mark.parametrize("reason", ["", "unknown", "too short", None])
def test_missing_metadata_needs_a_real_reason(reason):
    with pytest.raises(ValueError):
        MissingMetadata(reason)


def test_missing_metadata_round_trips(tmp_path):
    reason = "Vendor export has no per-episode collector ID"
    attrs = _attrs(_write(tmp_path / "ep.zarr", operator=MissingMetadata(reason)))
    assert attrs["operator"] is None
    assert attrs["missing_metadata"] == {"operator": reason}


def test_metadata_override_cannot_clobber_required_metadata(tmp_path):
    path = _write(
        tmp_path / "ep.zarr",
        metadata_override={"operator": "", "lab": "x", "missing_metadata": {}},
    )
    attrs = _attrs(path)
    assert attrs["operator"] == GOOD["operator"]
    assert attrs["lab"] == GOOD["lab"]
    assert "missing_metadata" not in attrs


def test_build_required_metadata_rejects_value_and_waiver_together():
    with pytest.raises(ValueError, match="both"):
        build_required_metadata(
            {"operator": "op"}, {"operator": "a long enough reason"}
        )
    with pytest.raises(ValueError, match="unknown fields"):
        build_required_metadata({}, {"opertor": "a long enough reason"})


def test_cli_args_feed_create_and_write(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name")
    add_required_metadata_args(parser)
    args = parser.parse_args(
        [
            "--task-name=fold_clothes",
            "--operator=op_7",
            "--lab=rl2",
            "--objects=shirt",
            "--missing-metadata",
            "scene=capture rig logs no scene label",
        ]
    )
    prov = required_metadata_from_args(args)
    assert isinstance(prov["scene"], MissingMetadata)
    attrs = _attrs(_write(tmp_path / "ep.zarr", **prov))
    assert attrs["missing_metadata"] == {"scene": "capture rig logs no scene label"}

    # Forgetting a flag (no waiver) fails at write time.
    bare = parser.parse_args(["--task-name=fold_clothes"])
    with pytest.raises(ValueError, match="operator"):
        _write(tmp_path / "bare.zarr", **required_metadata_from_args(bare))


def test_validator_passes_fixture_episode(tmp_path):
    ep = write_episode(tmp_path, "aria", T=16, H=32, W=32)
    report = validate.validate_episode(ep)
    assert report.ok, report.errors
    assert validate.main([str(ep)]) == 0


def test_validator_flags_waived_and_absent_fields(tmp_path):
    ep = write_episode(tmp_path, "aria", T=16, H=32, W=32)
    group = zarr.open_group(str(ep), mode="r+")
    group.attrs["operator"] = None
    group.attrs["missing_metadata"] = {"operator": "vendor export has no operator"}
    del group.attrs["scene"]

    strict = validate.validate_episode(ep)
    assert any("operator" in e for e in strict.errors)
    assert any("scene" in e for e in strict.errors)

    lenient = validate.validate_episode(ep, allow_missing=True)
    assert any("operator" in w for w in lenient.warnings)
    assert not any("operator" in e for e in lenient.errors)
    assert any("scene" in e for e in lenient.errors)  # absent != waived
    assert validate.main([str(ep), "--allow-missing"]) == 1


def test_validator_accepts_waiver_with_allow_missing(tmp_path):
    ep = write_episode(tmp_path, "aria", T=16, H=32, W=32)
    group = zarr.open_group(str(ep), mode="r+")
    group.attrs["operator"] = None
    group.attrs["missing_metadata"] = {"operator": "vendor export has no operator"}
    assert validate.main([str(ep)]) == 1
    assert validate.main([str(ep), "--allow-missing"]) == 0
