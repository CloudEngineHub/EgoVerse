import copy
from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    LocalEpisodeResolver,
    S3EpisodeResolver,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def _resolver_key_map() -> dict:
    return {"observations.state": {"zarr_key": "state"}}


def _write_valid_episode(path: Path, frames: int = 4) -> None:
    images = np.zeros((frames, 4, 4, 3), dtype=np.uint8)
    for idx in range(frames):
        images[idx] = np.uint8((idx * 10) % 256)

    numeric = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2)
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data={"state": numeric},
        image_data={"images.front_1": images},
        embodiment="scale",
        task_name="integrity_test",
    )


def _write_invalid_empty_jpeg_episode(path: Path, frames: int = 4) -> None:
    _write_valid_episode(path, frames=frames)
    store = zarr.open_group(str(path), mode="a")
    store["images.front_1"][frames // 2] = b""


def _update_store_attrs(path: Path, mutate) -> None:
    store = zarr.open_group(str(path), mode="a")
    attrs = copy.deepcopy(dict(store.attrs))
    mutate(attrs)
    store.attrs.clear()
    store.attrs.update(attrs)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"abc", b"abc"),
        (bytearray(b"abc"), b"abc"),
        (memoryview(b"abc"), b"abc"),
        (np.bytes_(b"abc"), b"abc"),
    ],
)
def test_coerce_bytes_payload_supported_types(value, expected: bytes) -> None:
    assert EpisodeResolver._coerce_bytes_payload(value) == expected


@pytest.mark.parametrize("value", ["abc", 123, np.array([1, 2, 3])])
def test_coerce_bytes_payload_rejects_non_bytes(value) -> None:
    assert EpisodeResolver._coerce_bytes_payload(value) is None


@pytest.mark.parametrize(
    ("total_frames", "expected"),
    [
        (0, []),
        (1, [(0, 1)]),
        (256, [(0, 256)]),
        (257, [(0, 256), (256, 257)]),
        (513, [(0, 256), (256, 512), (512, 513)]),
    ],
)
def test_validation_ranges_cover_total_frames(
    total_frames: int, expected: list[tuple[int, int]]
) -> None:
    assert EpisodeResolver._validation_ranges(total_frames) == expected


@pytest.mark.parametrize(
    ("processed_path", "expected"),
    [
        ("s3://rldb/path/episode_1", "s3://rldb/path/episode_1/*"),
        ("processed_v3/scale/episode_1", "s3://rldb/processed_v3/scale/episode_1/*"),
    ],
)
def test_build_src_prefix(processed_path: str, expected: str) -> None:
    assert S3EpisodeResolver._build_src_prefix("rldb", processed_path) == expected


def test_build_sync_jobs_uses_final_and_partial_paths(tmp_path: Path) -> None:
    jobs = S3EpisodeResolver._build_sync_jobs(
        bucket_name="rldb",
        s3_paths=[("processed_v3/scale/episode_1", "episode_1")],
        local_dir=tmp_path,
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.episode_hash == "episode_1"
    assert job.src_prefix == "s3://rldb/processed_v3/scale/episode_1/*"
    assert job.final_path == tmp_path / "episode_1"
    assert job.partial_path == tmp_path / "episode_1.partial"


def test_validate_episode_store_accepts_valid_episode(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_ok"
    _write_valid_episode(episode_path)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert is_valid, reason


def test_validate_episode_store_rejects_empty_jpeg_bytes(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_bad_jpeg"
    _write_invalid_empty_jpeg_episode(episode_path)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "empty_jpeg_bytes" in reason


def test_validate_episode_store_rejects_missing_feature_array(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_missing_array"
    _write_valid_episode(episode_path)
    shutil.rmtree(episode_path / "state")

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "missing_array" in reason


def test_validate_episode_store_rejects_missing_episode_dir(tmp_path: Path) -> None:
    is_valid, reason = EpisodeResolver._validate_episode_store(tmp_path / "missing_ep")

    assert not is_valid
    assert "missing_episode_dir" in reason


def test_validate_episode_store_rejects_non_zarr_directory(tmp_path: Path) -> None:
    episode_path = tmp_path / "not_a_zarr"
    episode_path.mkdir()

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "open_error" in reason


def test_validate_episode_store_rejects_array_shorter_than_total_frames(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode_short_array"
    _write_valid_episode(episode_path)

    store = zarr.open_group(str(episode_path), mode="a")
    store.attrs.update({"total_frames": 101})

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "array_too_short" in reason


@pytest.mark.parametrize("missing_key", ["total_frames", "features", "embodiment"])
def test_validate_episode_store_rejects_missing_required_metadata(
    tmp_path: Path, missing_key: str
) -> None:
    episode_path = tmp_path / f"episode_missing_metadata_{missing_key}"
    _write_valid_episode(episode_path)

    def mutate(attrs: dict) -> None:
        attrs.pop(missing_key, None)

    _update_store_attrs(episode_path, mutate)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "missing_metadata" in reason
    assert missing_key in reason


@pytest.mark.parametrize("bad_total_frames", ["not_an_int", 0, -5])
def test_validate_episode_store_rejects_invalid_total_frames(
    tmp_path: Path, bad_total_frames
) -> None:
    episode_path = tmp_path / "episode_bad_total_frames"
    _write_valid_episode(episode_path)

    def mutate(attrs: dict) -> None:
        attrs["total_frames"] = bad_total_frames

    _update_store_attrs(episode_path, mutate)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "invalid_total_frames" in reason


@pytest.mark.parametrize("bad_features", [None, [], {}, "bad"])
def test_validate_episode_store_rejects_invalid_features_metadata(
    tmp_path: Path, bad_features
) -> None:
    episode_path = tmp_path / "episode_bad_features"
    _write_valid_episode(episode_path)

    def mutate(attrs: dict) -> None:
        attrs["features"] = bad_features

    _update_store_attrs(episode_path, mutate)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "invalid_features_metadata" in reason


def test_validate_episode_store_rejects_invalid_feature_info(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_bad_feature_info"
    _write_valid_episode(episode_path)

    def mutate(attrs: dict) -> None:
        attrs["features"]["state"] = "not_a_dict"

    _update_store_attrs(episode_path, mutate)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "invalid_feature_info" in reason


def test_validate_episode_store_rejects_invalid_jpeg_type(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_invalid_jpeg_type"
    _write_valid_episode(episode_path)

    def mutate(attrs: dict) -> None:
        attrs["features"]["state"]["dtype"] = "jpeg"

    _update_store_attrs(episode_path, mutate)

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "invalid_jpeg_type" in reason


def test_validate_episode_store_rejects_mid_episode_corruption_in_long_episode(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode_long_mid_corrupt"
    _write_valid_episode(episode_path, frames=600)
    store = zarr.open_group(str(episode_path), mode="a")
    store["images.front_1"][333] = b""

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "frame=333" in reason
    assert "empty_jpeg_bytes" in reason


def test_validate_episode_store_rejects_zero_size_chunk_file(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_zero_chunk"
    _write_valid_episode(episode_path)

    chunk_file = episode_path / "images.front_1" / "c" / "0"
    chunk_file.write_bytes(b"")

    is_valid, reason = EpisodeResolver._validate_episode_store(episode_path)

    assert not is_valid
    assert "zero_size_chunks" in reason


def test_episode_already_present_removes_stale_partial_dir(tmp_path: Path) -> None:
    final_path = tmp_path / "episode_1"
    partial_path = tmp_path / "episode_1.partial"
    _write_valid_episode(final_path)
    _write_valid_episode(partial_path)

    assert S3EpisodeResolver._episode_already_present(tmp_path, "episode_1") is True
    assert final_path.is_dir()
    assert not partial_path.exists()


def test_episode_already_present_rejects_invalid_final_episode(tmp_path: Path) -> None:
    _write_invalid_empty_jpeg_episode(tmp_path / "episode_1")

    assert S3EpisodeResolver._episode_already_present(tmp_path, "episode_1") is False


def test_prepare_sync_job_removes_stale_partial_and_invalid_final(tmp_path: Path) -> None:
    _write_valid_episode(tmp_path / "episode_1.partial")
    _write_invalid_empty_jpeg_episode(tmp_path / "episode_1")
    job = S3EpisodeResolver._build_sync_jobs(
        bucket_name="rldb",
        s3_paths=[("processed_v3/scale/episode_1", "episode_1")],
        local_dir=tmp_path,
    )[0]

    S3EpisodeResolver._prepare_sync_job(job)

    assert not job.partial_path.exists()
    assert not job.final_path.exists()


def test_validate_and_promote_jobs_promotes_valid_and_collects_invalid(
    tmp_path: Path,
) -> None:
    valid_job, invalid_job = S3EpisodeResolver._build_sync_jobs(
        bucket_name="rldb",
        s3_paths=[
            ("processed_v3/scale/episode_valid", "episode_valid"),
            ("processed_v3/scale/episode_invalid", "episode_invalid"),
        ],
        local_dir=tmp_path,
    )
    _write_valid_episode(valid_job.partial_path)
    _write_invalid_empty_jpeg_episode(invalid_job.partial_path)

    failures = S3EpisodeResolver._validate_and_promote_jobs([valid_job, invalid_job])

    assert valid_job.final_path.is_dir()
    assert not valid_job.partial_path.exists()
    assert len(failures) == 1
    assert failures[0][0].episode_hash == "episode_invalid"
    assert "empty_jpeg_bytes" in failures[0][1]
    assert not invalid_job.partial_path.exists()
    assert not invalid_job.final_path.exists()


def test_sync_s3_to_local_noops_for_empty_paths(tmp_path: Path) -> None:
    S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[],
        local_dir=tmp_path,
        numworkers=4,
    )

    assert list(tmp_path.iterdir()) == []


def test_sync_s3_to_local_retries_invalid_episode_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: dict[str, int] = {}

    def fake_run(cls, jobs, numworkers=10):
        for job in jobs:
            attempts[job.episode_hash] = attempts.get(job.episode_hash, 0) + 1
            if attempts[job.episode_hash] == 1:
                _write_invalid_empty_jpeg_episode(job.partial_path)
            else:
                _write_valid_episode(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[("s3://rldb/path/episode_1", "episode_1")],
        local_dir=tmp_path,
        numworkers=4,
    )

    assert attempts == {"episode_1": 2}
    assert (tmp_path / "episode_1").is_dir()
    assert not (tmp_path / "episode_1.partial").exists()


def test_sync_s3_to_local_raises_after_failed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: dict[str, int] = {}

    def fake_run(cls, jobs, numworkers=10):
        for job in jobs:
            attempts[job.episode_hash] = attempts.get(job.episode_hash, 0) + 1
            _write_invalid_empty_jpeg_episode(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    with pytest.raises(RuntimeError, match="episode_1"):
        S3EpisodeResolver._sync_s3_to_local(
            bucket_name="rldb",
            s3_paths=[("s3://rldb/path/episode_1", "episode_1")],
            local_dir=tmp_path,
            numworkers=4,
        )

    assert attempts == {"episode_1": 2}
    assert not (tmp_path / "episode_1").exists()
    assert not (tmp_path / "episode_1.partial").exists()


def test_sync_s3_to_local_only_syncs_invalid_and_missing_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_episode(tmp_path / "episode_keep")
    _write_invalid_empty_jpeg_episode(tmp_path / "episode_replace")
    sync_calls: list[list[str]] = []

    def fake_run(cls, jobs, numworkers=10):
        sync_calls.append([job.episode_hash for job in jobs])
        for job in jobs:
            _write_valid_episode(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[
            ("processed_v3/scale/episode_keep", "episode_keep"),
            ("processed_v3/scale/episode_replace", "episode_replace"),
            ("processed_v3/scale/episode_new", "episode_new"),
        ],
        local_dir=tmp_path,
        numworkers=4,
    )

    assert sync_calls == [["episode_replace", "episode_new"]]
    assert EpisodeResolver._validate_episode_store(tmp_path / "episode_keep")[0] is True
    assert EpisodeResolver._validate_episode_store(tmp_path / "episode_replace")[0] is True
    assert EpisodeResolver._validate_episode_store(tmp_path / "episode_new")[0] is True


def test_sync_from_filters_returns_empty_without_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_calls: list[tuple] = []

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(lambda filters=None, debug=False: []),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_sync_s3_to_local",
        classmethod(
            lambda cls, bucket_name, s3_paths, local_dir, numworkers=10: sync_calls.append(
                (bucket_name, s3_paths, local_dir, numworkers)
            )
        ),
    )

    result = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters={"episode_hash": "missing"},
        local_dir=tmp_path,
        numworkers=17,
    )

    assert result == []
    assert sync_calls == []


def test_sync_from_filters_forwards_numworkers_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict = {}
    filtered_paths = [("processed_v3/scale/episode_1", "episode_1")]

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(lambda filters=None, debug=False: filtered_paths),
    )

    def fake_sync(cls, bucket_name, s3_paths, local_dir, numworkers=10):
        observed.update(
            {
                "bucket_name": bucket_name,
                "s3_paths": s3_paths,
                "local_dir": local_dir,
                "numworkers": numworkers,
            }
        )

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_sync_s3_to_local",
        classmethod(fake_sync),
    )

    result = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters={"episode_hash": "episode_1"},
        local_dir=tmp_path,
        numworkers=23,
    )

    assert result == filtered_paths
    assert observed == {
        "bucket_name": "rldb",
        "s3_paths": filtered_paths,
        "local_dir": tmp_path,
        "numworkers": 23,
    }


def test_resolve_skips_sync_for_valid_cached_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_episode(tmp_path / "episode_1")
    sync_calls: list[list[str]] = []

    def fake_get_filtered_paths(filters=None, debug=False):
        return [("s3://rldb/path/episode_1", "episode_1")]

    def fake_run(cls, jobs, numworkers=10):
        sync_calls.append([job.episode_hash for job in jobs])
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(fake_get_filtered_paths),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    resolver = S3EpisodeResolver(
        folder_path=tmp_path,
        key_map=_resolver_key_map(),
    )
    datasets = resolver.resolve(filters={"episode_hash": "episode_1"})

    assert set(datasets.keys()) == {"episode_1"}
    assert len(datasets["episode_1"]) == 4
    assert sync_calls == []


def test_resolve_replaces_invalid_cached_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_invalid_empty_jpeg_episode(tmp_path / "episode_1")
    sync_calls: list[list[str]] = []

    def fake_get_filtered_paths(filters=None, debug=False):
        return [("s3://rldb/path/episode_1", "episode_1")]

    def fake_run(cls, jobs, numworkers=10):
        sync_calls.append([job.episode_hash for job in jobs])
        for job in jobs:
            _write_valid_episode(job.partial_path)
        return 0

    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(fake_get_filtered_paths),
    )
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_run_s5cmd_batch",
        classmethod(fake_run),
    )

    resolver = S3EpisodeResolver(
        folder_path=tmp_path,
        key_map=_resolver_key_map(),
    )
    datasets = resolver.resolve(filters={"episode_hash": "episode_1"})

    assert set(datasets.keys()) == {"episode_1"}
    assert len(datasets["episode_1"]) == 4
    assert sync_calls == [["episode_1"]]
    assert EpisodeResolver._validate_episode_store(tmp_path / "episode_1")[0] is True


def test_resolve_raises_when_no_filtered_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        S3EpisodeResolver,
        "_get_filtered_paths",
        staticmethod(lambda filters=None, debug=False: []),
    )

    resolver = S3EpisodeResolver(folder_path=tmp_path, key_map=_resolver_key_map())

    with pytest.raises(ValueError, match="filters matched no episodes"):
        resolver.resolve(filters={"episode_hash": "missing"})


def test_local_resolver_resolve_filters_and_loads_valid_episode(tmp_path: Path) -> None:
    valid_path = tmp_path / "episode_1"
    other_path = tmp_path / "episode_2"
    _write_valid_episode(valid_path)
    _write_valid_episode(other_path)

    _update_store_attrs(
        valid_path,
        lambda attrs: attrs.update({"task_name": "pick", "is_deleted": False}),
    )
    _update_store_attrs(
        other_path,
        lambda attrs: attrs.update({"task_name": "place", "is_deleted": False}),
    )

    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=_resolver_key_map(),
    )
    datasets = resolver.resolve(filters={"task_name": "pick"})

    assert set(datasets.keys()) == {"episode_1"}
    assert len(datasets["episode_1"]) == 4
