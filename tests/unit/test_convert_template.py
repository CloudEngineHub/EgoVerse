"""The reference converter vendors copy must keep producing valid episodes."""

import numpy as np
import pytest
import zarr

from egomimic.scripts.convert_template import template_to_zarr as tmpl


@pytest.fixture(autouse=True)
def _demo_lab(monkeypatch):
    """The shipped lab="template" is rejected on purpose; tests stand in a lab."""
    shipped = tmpl.load_episode

    def load(raw_path):
        ep = shipped(raw_path)
        ep.metadata["lab"] = "demo_lab"
        return ep

    monkeypatch.setattr(tmpl, "load_episode", load)
    return shipped


def test_shipped_template_metadata_is_rejected(tmp_path, monkeypatch, _demo_lab):
    monkeypatch.setattr(tmpl, "load_episode", _demo_lab)
    raw = _raw_paths(tmp_path, 1)
    written, failed = tmpl.convert_all(raw, tmp_path / "out", save_mp4=False)
    assert not written
    assert "lab='template' is a placeholder" in failed[raw[0]]


def _raw_paths(tmp_path, n):
    paths = [tmp_path / "raw" / f"session_{i}" for i in range(n)]
    for p in paths:
        p.mkdir(parents=True)
    return paths


def test_template_converts_and_validates(tmp_path):
    raw = _raw_paths(tmp_path, 3)
    assert tmpl.find_raw_episodes(tmp_path / "raw") == raw
    written, failed = tmpl.convert_all(raw, tmp_path / "out", save_mp4=False)
    assert not failed
    assert len(written) == 3
    for path in written:
        # named by the §4 episode hash, e.g. 2026-01-01-00-00-12-345000.zarr
        assert len(path.stem.split("-")) == 7
        attrs = zarr.open_group(str(path), mode="r").attrs
        assert attrs["embodiment"] == "human_bimanual"
        assert attrs["lab"] == "demo_lab"


def test_template_cli_and_workers(tmp_path):
    _raw_paths(tmp_path, 2)
    argv = ["--raw-root", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "out")]
    assert tmpl.main([*argv, "--no-mp4", "--workers", "2"]) == 0
    assert len(list((tmp_path / "out").glob("*.zarr"))) == 2


def test_bad_episode_is_reported_and_removed(tmp_path, monkeypatch):
    good_load = tmpl.load_episode

    def bad_load(raw_path):
        ep = good_load(raw_path)
        if raw_path.name == "session_1":
            ep.head_pose = ep.head_pose * np.array(
                [1, 1, 1, 2, 2, 2, 2]
            )  # non-unit quats
        return ep

    monkeypatch.setattr(tmpl, "load_episode", bad_load)
    raw = _raw_paths(tmp_path, 2)
    written, failed = tmpl.convert_all(raw, tmp_path / "out", save_mp4=False)
    assert len(written) == 1
    assert list(failed) == [raw[1]]
    assert "quaternion" in failed[raw[1]]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [written[0].name]


def test_waived_field_needs_allow_missing(tmp_path, monkeypatch):
    good_load = tmpl.load_episode

    def waived_load(raw_path):
        ep = good_load(raw_path)
        ep.metadata["scene"] = tmpl.MissingMetadata()
        return ep

    monkeypatch.setattr(tmpl, "load_episode", waived_load)
    raw = _raw_paths(tmp_path, 1)
    written, failed = tmpl.convert_all(raw, tmp_path / "strict", save_mp4=False)
    assert not written
    assert "--allow-missing" in failed[raw[0]]
    assert not list((tmp_path / "strict").iterdir())

    written, failed = tmpl.convert_all(
        raw, tmp_path / "waived", save_mp4=False, allow_missing=True
    )
    assert not failed
    assert zarr.open_group(str(written[0]), mode="r").attrs["scene"] is None


def test_operator_cannot_be_waived(tmp_path, monkeypatch):
    good_load = tmpl.load_episode

    def waived_load(raw_path):
        ep = good_load(raw_path)
        ep.metadata["operator"] = tmpl.MissingMetadata()
        return ep

    monkeypatch.setattr(tmpl, "load_episode", waived_load)
    raw = _raw_paths(tmp_path, 1)
    written, failed = tmpl.convert_all(
        raw, tmp_path / "out", save_mp4=False, allow_missing=True
    )
    assert not written
    assert "cannot be waived" in failed[raw[0]]


def test_whole_episode_annotation_covers_every_frame(tmp_path):
    written, _ = tmpl.convert_all(
        _raw_paths(tmp_path, 1), tmp_path / "out", save_mp4=False
    )
    attrs = zarr.open_group(str(written[0]), mode="r").attrs
    ep = tmpl.load_episode(tmp_path / "raw" / "session_0")
    assert ep.annotations[0][1:] == (0, attrs["total_frames"])


def test_failed_preview_keeps_the_episode(tmp_path, monkeypatch):
    def no_ffmpeg(*args, **kwargs):
        raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr(tmpl, "save_preview_mp4", no_ffmpeg)
    raw = _raw_paths(tmp_path, 1)
    written, failed = tmpl.convert_all(raw, tmp_path / "out")
    assert not failed
    assert [p.name for p in (tmp_path / "out").iterdir()] == [written[0].name]


@pytest.mark.parametrize("scale", [1e-3, 1e6])  # seconds, nanoseconds
def test_start_time_in_wrong_unit_is_rejected(tmp_path, monkeypatch, scale):
    good_load = tmpl.load_episode

    def load(raw_path):
        ep = good_load(raw_path)
        ep.start_time_ms = int(ep.start_time_ms * scale)
        return ep

    monkeypatch.setattr(tmpl, "load_episode", load)
    raw = _raw_paths(tmp_path, 1)
    written, failed = tmpl.convert_all(raw, tmp_path / "out", save_mp4=False)
    assert not written
    assert "epoch milliseconds" in failed[raw[0]]
