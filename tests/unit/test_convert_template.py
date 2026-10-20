"""The reference converter vendors copy must keep producing valid episodes."""

import numpy as np
import zarr

from egomimic.scripts.convert_template import template_to_zarr as tmpl


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
        assert attrs["lab"] == "template"


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
