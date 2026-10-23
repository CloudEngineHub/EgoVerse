"""Required metadata (operator/lab/scene/objects/task_name) in
ZarrWriter.write and the episode validator."""

import argparse

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_episode

from egomimic.rldb.zarr import validate
from egomimic.rldb.zarr.required_metadata import (
    MissingMetadata,
    add_required_metadata_args,
    required_metadata_from_args,
)
from egomimic.rldb.zarr.zarr_writer import (
    ZarrWriter,
    key_exists,
    remove_key,
    write_annotations,
)

T = 8
K = np.array([[100.0, 0, 32, 0], [0, 100.0, 32, 0], [0, 0, 1.0, 0]])
GOOD = {
    "task_name": "fold_clothes",
    "operator": "op_7",
    "lab": "rl2",
    "scene": "kitchen_a",
    "objects": ["shirt"],
}


_DROP = object()


def _write(path, metadata_override=None, **overrides):
    kwargs = {**GOOD, **overrides}
    kwargs = {k: v for k, v in kwargs.items() if v is not _DROP}
    return ZarrWriter(
        path,
        embodiment="human_bimanual",
        intrinsics={"front_1": K},
        **kwargs,
    ).write(
        numeric_data={"actions": np.zeros((T, 3), np.float32)},
        metadata_override=metadata_override,
    )


def _attrs(path):
    return dict(zarr.open_group(str(path), mode="r").attrs)


@pytest.mark.parametrize("field", list(GOOD))
def test_omitting_a_field_is_a_type_error(tmp_path, field):
    with pytest.raises(TypeError, match=field):
        _write(tmp_path / "ep.zarr", **{field: _DROP})
    assert not (tmp_path / "ep.zarr").exists()


def test_create_and_write_wraps_write(tmp_path):
    ep = ZarrWriter.create_and_write(
        tmp_path / "ep.zarr",
        numeric_data={"actions": np.zeros((T, 3), np.float32)},
        embodiment="human_bimanual",
        intrinsics={"front_1": K},
        **GOOD,
    )
    assert _attrs(ep)["operator"] == GOOD["operator"]
    with pytest.raises(ValueError, match="operator"):
        ZarrWriter.create_and_write(
            tmp_path / "bad.zarr",
            numeric_data={"actions": np.zeros((T, 3), np.float32)},
            embodiment="human_bimanual",
            intrinsics={"front_1": K},
            **{**GOOD, "operator": "unknown"},
        )
    assert not (tmp_path / "bad.zarr").exists()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"embodiment": "mecka_bimanual"}, "embodiment"),
        ({"intrinsics": None}, "intrinsics"),
        ({"intrinsics": {"front_1": np.eye(3)}}, "3x4"),
        ({"extrinsics": {}}, "extrinsics"),
        ({"extrinsics": {"left": np.eye(3)}}, "4x4"),
        ({"intrinsics": {"cam0": K}}, "front-camera"),
        ({"fps": 25}, "fps"),
        ({"fps": np.int64(30)}, "fps"),
        ({"fps": 30.0}, "fps"),
    ],
)
def test_write_validates_camera_and_embodiment(tmp_path, kwargs, match):
    base = {"embodiment": "human_bimanual", "intrinsics": {"front_1": K}, **GOOD}
    writer = ZarrWriter(tmp_path / "ep.zarr", **{**base, **kwargs})
    with pytest.raises(ValueError, match=match):
        writer.write(numeric_data={"actions": np.zeros((T, 3), np.float32)})
    assert not (tmp_path / "ep.zarr").exists()


def test_bare_writer_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        ZarrWriter(tmp_path / "ep.zarr")


def test_store_edits_keep_features_in_sync(tmp_path):
    ep = write_episode(tmp_path, "aria", T=16, H=32, W=32)
    write_annotations(ep, [("pick", 0, 3)], "annotations", mode="w")
    write_annotations(ep, [("place", 4, 9)], "annotations_v2")
    assert key_exists(ep, "annotations_v2")
    assert _attrs(ep)["features"]["annotations_v2"]["shape"] == [1]
    assert validate.validate_episode(ep).ok

    with pytest.raises(ValueError, match="already exists"):
        write_annotations(ep, [("x", 0, 1)], "annotations_v2", mode="a")

    remove_key(ep, "annotations_v2")
    assert not key_exists(ep, "annotations_v2")
    assert "annotations_v2" not in _attrs(ep)["features"]
    assert validate.validate_episode(ep).ok


@pytest.mark.parametrize("field", list(GOOD))
@pytest.mark.parametrize("bad", [None, "", "   ", "unknown", "N/A", "debug", 3])
def test_placeholders_are_rejected(tmp_path, field, bad):
    with pytest.raises(ValueError, match=field):
        _write(tmp_path / "ep.zarr", **{field: bad})
    assert not (tmp_path / "ep.zarr").exists()


@pytest.mark.parametrize("bad", ["RL2", "rl 2", "_rl2", "rl2\n", "rl2 "])
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


def test_missing_metadata_is_stored_as_null(tmp_path):
    attrs = _attrs(_write(tmp_path / "ep.zarr", operator=MissingMetadata()))
    assert attrs["operator"] is None
    assert "missing_metadata" not in attrs


@pytest.mark.parametrize("key", ["operator", "intrinsics", "embodiment", "features"])
def test_metadata_override_rejects_writer_owned_keys(tmp_path, key):
    with pytest.raises(ValueError, match=f"must not set \\['{key}'\\]"):
        _write(tmp_path / "ep.zarr", metadata_override={key: "x"})
    assert not (tmp_path / "ep.zarr").exists()


def test_metadata_override_adds_extra_keys(tmp_path):
    attrs = _attrs(
        _write(tmp_path / "ep.zarr", metadata_override={"hand_used": "left"})
    )
    assert attrs["hand_used"] == "left"


def test_failed_rewrite_keeps_existing_episode(tmp_path):
    path = _write(tmp_path / "ep.zarr")
    before = _attrs(path)
    writer = ZarrWriter(
        path, embodiment="human_bimanual", intrinsics={"front_1": K}, **GOOD
    )
    with pytest.raises(ValueError, match="images.front_1"):
        writer.write(
            numeric_data={"actions": np.zeros((T, 3), np.float32)},
            image_data={"images.front_1": np.zeros((T, 4, 4), np.uint8)},
        )
    with pytest.raises(ValueError, match="JSON-serializable"):
        writer.write(
            numeric_data={"actions": np.zeros((T, 3), np.float32)},
            metadata_override={"x": object()},
        )
    assert _attrs(path) == before
    assert [p.name for p in tmp_path.iterdir()] == ["ep.zarr"]


def test_rewrite_does_not_keep_stale_features(tmp_path):
    writer = ZarrWriter(
        tmp_path / "ep.zarr",
        embodiment="human_bimanual",
        intrinsics={"front_1": K},
        **GOOD,
    )
    writer.write(numeric_data={"a": np.zeros((T, 3), np.float32)})
    path = writer.write(numeric_data={"b": np.zeros((T, 3), np.float32)})
    assert set(_attrs(path)["features"]) == {"b", "annotations"}


def test_waiver_rejects_value_and_unknown_fields():
    with pytest.raises(ValueError, match="both"):
        required_metadata_from_args(
            argparse.Namespace(operator="op", missing_metadata=["operator"])
        )
    with pytest.raises(ValueError, match="unknown fields"):
        required_metadata_from_args(argparse.Namespace(missing_metadata=["opertor"]))


def test_cli_args_feed_writer(tmp_path):
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
            "scene",
        ]
    )
    prov = required_metadata_from_args(args)
    assert isinstance(prov["scene"], MissingMetadata)
    attrs = _attrs(_write(tmp_path / "ep.zarr", **prov))
    assert attrs["scene"] is None

    with pytest.raises(SystemExit):  # typo'd field name is an argparse error
        parser.parse_args(["--missing-metadata", "scen"])

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
    assert validate.main([str(ep)]) == 1
    assert validate.main([str(ep), "--allow-missing"]) == 0


@pytest.mark.parametrize("field", ["task_name", "operator", "scene"])
def test_surrounding_whitespace_is_rejected(tmp_path, field):
    with pytest.raises(ValueError, match="whitespace"):
        _write(tmp_path / "ep.zarr", **{field: f" {GOOD[field]}\n"})


def test_objects_string_is_split_and_stored_as_list(tmp_path):
    attrs = _attrs(_write(tmp_path / "ep.zarr", objects="shirt, basket"))
    assert attrs["objects"] == ["shirt", "basket"]
    for bad in ("shirt, n/a", "shirt,,basket", ",", " shirt"):
        with pytest.raises(ValueError, match="objects"):
            _write(tmp_path / "bad.zarr", objects=bad)


def _corrupt(ep, **attrs):
    zarr.open_group(str(ep), mode="a").attrs.update(attrs)
    return validate.validate_episode(ep)


@pytest.mark.parametrize(
    "attrs, match",
    [
        ({"total_frames": None}, "total_frames"),
        ({"intrinsics": {"front_1": [[1, 2], [3]]}}, "must be a 3x4 K matrix"),
        ({"features": {"images.front_1": "jpeg"}}, "could not finish"),
    ],
)
def test_validator_reports_malformed_attrs(tmp_path, attrs, match):
    report = _corrupt(write_episode(tmp_path, "aria", T=16, H=32, W=32), **attrs)
    assert not report.ok
    assert any(match in e for e in report.errors)


def test_validator_rejects_nan_poses(tmp_path):
    ep = write_episode(tmp_path, "aria", T=16, H=32, W=32)
    store = zarr.open_group(str(ep), mode="a")
    pose = store["obs_head_pose"][:]
    pose[3] = np.nan
    store["obs_head_pose"][:] = pose
    report = validate.validate_episode(ep)
    assert any("obs_head_pose: contains NaN" in e for e in report.errors)


def test_float32_intrinsics_lists_are_stored(tmp_path):
    K32 = [[np.float32(v) for v in row] for row in K]
    path = ZarrWriter(
        tmp_path / "ep.zarr",
        embodiment="human_bimanual",
        intrinsics={"front_1": K32},
        **GOOD,
    ).write(numeric_data={"actions": np.zeros((T, 3), np.float32)})
    assert _attrs(path)["intrinsics"]["front_1"] == K.tolist()


def test_stale_temp_dirs_are_swept_and_non_dirs_rejected(tmp_path):
    (tmp_path / "ep.zarr.tmp-1").mkdir()
    (tmp_path / "ep.zarr.tmp-2").write_text("x")
    (tmp_path / "ep.zarr.old-3").mkdir()
    _write(tmp_path / "ep.zarr")
    assert [p.name for p in tmp_path.iterdir()] == ["ep.zarr"]

    (tmp_path / "file.zarr").write_text("x")
    (tmp_path / "link.zarr").symlink_to(tmp_path / "ep.zarr")
    for name in ("file.zarr", "link.zarr"):
        with pytest.raises(ValueError, match="not an episode directory"):
            _write(tmp_path / name)


def test_orphaned_old_copy_is_restored_not_swept(tmp_path):
    good = _write(tmp_path / "ep.zarr")
    good.rename(tmp_path / "ep.zarr.old-99")  # killed between the two renames
    with pytest.raises(ValueError):
        _write(tmp_path / "ep.zarr", operator="unknown")
    assert _attrs(tmp_path / "ep.zarr")["operator"] == GOOD["operator"]
