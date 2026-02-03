"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observations.images.{cam}  (JPEG compressed)
        ├── observations.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations
import json
import logging
import os
import random
import time
from functools import cached_property
from pathlib import Path
from typing import Callable
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
import boto3
import subprocess
import tempfile

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    text,
)

from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    create_default_engine,
    delete_all_episodes,
    delete_episodes,
    episode_hash_to_table_row,
    episode_table_to_df,
    update_episode,
)


from lerobot.common.datasets.utils import (
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    get_delta_indices,
    check_delta_timestamps,
    load_json,
    load_jsonlines,
    flatten_dict,
    unflatten_dict,
)

from egomimic.rldb.data_utils import (
    _ypr_to_quat,
    _quat_to_ypr,
    _slow_down_slerp_quat,
)

logger = logging.getLogger(__name__)

SEED = 42




class EpisodeResolver:
    """
    Filters SQL table for zarr episode paths/ downloads from S3.
    resolve returns processed_path. 
    """
    def __init__(self, bucket_name: str = "rldb"):
        self.bucket_name = bucket_name
        
    def resolve(
        self,
        filters: dict,
        folder_path: Path,
        *,
        bucket_name: str = "rldb",
        sync_from_s3: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Outputs hashes to folders to build zarr dataset from. 
        If sync_from_s3 is True, sync S3 paths to local_root before indexing.
        If not True, assuming that folders already exist within folder.
        """
        if sync_from_s3:
            filtered_paths = self.sync_from_filters(
                bucket_name=bucket_name,
                filters=filters,
                local_dir=folder_path,
            )
            valid_collection_paths = set()
            for processed_path, episode_hash in filtered_paths:
                valid_collection_names.add(episode_hash)
            if not valid_collection_names:
                raise ValueError(
                    "No valid collection names after sync_from_filters: "
                    "filters matched no episodes or sync produced no local collections."
                )
            return valid_collection_names
        else:
            paths = self._get_processed_path(filters)
            valid_collection_names = set()
            for processed_path, episode_hash in paths:
                valid_collection_names.add(episode_hash)
            if not valid_collection_names:
                raise ValueError(
                    "No valid collection names from _get_processed_path: "
                    "filters matched no episodes in the SQL table."
                )
            return valid_collection_names

    @staticmethod
    def _get_processed_path(filters):
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        series = pd.Series(filters)

        output = df.loc[
            (df[list(filters)] == series).all(axis=1),
            ["processed_path", "episode_hash"],
        ]
        skipped = df[df["processed_path"].isnull()]["episode_hash"].tolist()
        logger.info(
            f"Skipped {len(skipped)} episodes with null processed_path: {skipped}"
        )
        output = output[~output["episode_hash"].isin(skipped)]

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @staticmethod
    def _download_files(bucket_name, s3_prefix, local_dir):
        """
        Downloads all files from a specific S3 prefix to a local directory.
        """

    def decode_jpeg(data) -> np.ndarray:
        """Decode JPEG image to numpy array using OpenCV.

        Args:
            bucket_name (str): The AWS S3 bucket name
            s3_prefix (str): The S3 prefix path to download from (e.g., "processed/fold_cloth/dataset1/meta/")
            local_dir (Path): The local directory to save files to
        """
        s3 = boto3.client("s3")

        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
        objects = response.get("Contents", [])

        if not objects:
            logger.warning(f"No objects found for prefix: {s3_prefix}")
            return

        for obj in objects:
            key = obj["Key"]

            if key.endswith("/"):
                logger.debug(f"Skipping directory: {key}")
                continue

            if key == s3_prefix or key == s3_prefix.rstrip("/"):
                logger.debug(f"Skipping prefix path: {key}")
                continue

            local_file_path = local_dir / Path(key).name

            # Check if file already exists and is not empty, solves race condition of multiple processes downloading the same file
            try:
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    logger.debug(
                        f"File already exists, skipping: {key} -> {local_file_path}"
                    )
                    continue

                s3.download_file(bucket_name, key, str(local_file_path))
                logger.debug(f"Successfully downloaded: {key}")
            except FileNotFoundError as e:
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    logger.debug(f"File downloaded by another process, skipping: {key}")
                else:
                    logger.error(f"Failed to download {key}: {e}")
            except Exception as e:
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    logger.debug(
                        f"File downloaded by another process after error: {key}"
                    )
                else:
                    logger.error(f"Failed to download {key}: {e}")

    @classmethod
    def _sync_s3_to_local(cls, bucket_name, s3_paths, local_dir: Path):
        if not s3_paths:
            return

        # 0) Skip episodes already present locally
        to_sync = []
        already = []
        for processed_path, episode_hash in s3_paths:
            if cls._episode_already_present(local_dir, episode_hash):
                already.append(episode_hash)
            else:
                to_sync.append((processed_path, episode_hash))

        if already:
            logger.info("Skipping %d episodes already present locally.", len(already))

        if not to_sync:
            logger.info("Nothing to sync from S3 (all episodes already present).")
            return

        # 1) Build s5cmd batch script (one line per episode)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_sync_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            batch_path = Path(tmp_file.name)

        lines = []
        for processed_path, episode_hash in to_sync:
            # processed_path like: s3://rldb/processed_v2/eva/<hash>/
            if processed_path.startswith("s3://"):
                src_prefix = processed_path.rstrip("/") + "/*"
            else:
                src_prefix = (
                    f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"
                    + "/*"
                )

            # Destination is the root local_dir; s5cmd will preserve <hash>/... under it
            dst = local_dir / episode_hash
            lines.append(f'sync "{src_prefix}" "{str(dst)}/"')

        try:
            batch_path.write_text("\n".join(lines) + "\n")

            cmd = ["s5cmd", "run", str(batch_path)]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            subprocess.run(cmd, check=True)

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        ep = local_dir / episode_hash
        meta = ep / "meta"
        chunk0 = ep / "data" / "chunk-000"

        if not meta.is_dir() or not chunk0.is_dir():
            return False

        try:
            if not any(meta.iterdir()):
                return False
            if not any(chunk0.iterdir()):
                return False
        except FileNotFoundError:
            return False

        return True

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: dict,
        local_dir: Path,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir


        Returns:
            List[(processed_path, episode_hash)]
        """

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_processed_path(filters)
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        # 2) Logging
        logger.info(
            f"Syncing S3 datasets with filters {filters} to local directory {local_dir}..."
        )

        # 3) Sync
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=filtered_paths,
            local_dir=local_dir,
        )

        return filtered_paths


class MultiZarrDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiZarr Dataset
    """

    def __init__(
        self,
        filters: dict,
        local_root: str | Path,
        *,
        use_episode_resolver: bool = False,
        bucket_name: str = "rldb",
        sync_from_s3: bool = True,
    ):
        """
        Args:
            episode_resolver: EpisodeResolver or callable(filters) -> list[(processed_path, episode_hash)].
            filters: Dict passed to resolver, e.g. {"task": "fold_cloth/", "lab": "eth", "robot_name": "eva_bimanual", "is_deleted": False}.
            local_root: Local directory where episodes are (or will be) synced.
            bucket_name: S3 bucket for sync_from_s3.
            sync_from_s3: If True, sync resolved S3 paths to local_root before indexing.
        """
        self.local_root = Path(local_root)
        self.filters = dict(filters)
        self.bucket_name = bucket_name
        self.sync_from_s3 = sync_from_s3
        self.use_episode_resolver = use_episode_resolver
        if self.use_episode_resolver:
            self.episode_resolver = EpisodeResolver(bucket_name=bucket_name)
        else:
            self.episode_resolver = None

        if isinstance(self.episode_resolver, EpisodeResolver):
            self.filtered_hashes = self.episode_resolver.resolve(
                self.filters,
                local_dir=self.local_root,
                bucket_name=bucket_name,
                sync_from_s3=sync_from_s3,
            )
        else:
            self.filtered_hashes = sorted(p.name for p in local_root.iterdir() if p.is_dir())

        self._index_map = self._build_index_map(self.filtered_hashes)

    def _build_index_map(self, filtered_hashes: list[String]) -> list[tuple[Path, int]]:
        """Build flat (zarr_episode_path, local_frame_idx) for each frame."""
        index_map: list[tuple[Path, int]] = []
        for episode_hash in filtered_hashes:
            episode_root = self.local_root / episode_hash
            if not episode_root.is_dir():
                logger.warning(f"file root {episode_root} doesn't exist")
                continue
            json_path = episode_root / "zarr.json"
            if not json_path.exists():
                logger.error("json path does not exist: %s", json_path)
                continue
            


        return index_map

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_path, local_idx = self._index_map[idx]
        episode_reader = DummyZarrEpisode(
            ep_path,
            indices=[local_idx],
            image_shape=self.image_shape,
            num_state_dims=self.num_state_dims,
            num_action_dims=self.num_action_dims,
            fps=self.fps,
        )
        data = episode_reader.read()
        # Squeeze batch dim to single frame
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out


def load_info(local_dir: Path) -> dict:
    """Load info.json and convert shape lists to tuples.

    Supports both legacy global metadata and per-episode metadata structures.
    """
    # Try legacy global metadata first
    global_info_path = local_dir / INFO_PATH
    if global_info_path.exists():
        info = load_json(global_info_path)
        for ft in info.get("features", {}).values():
            if "shape" in ft:
                ft["shape"] = tuple(ft["shape"])
        return info

    # Fall back to per-episode metadata - aggregate from first episode
    return load_info_from_episodes(local_dir)


def load_info_from_episodes(local_dir: Path) -> dict:
    """Load and aggregate info from per-episode metadata."""
    import json

    # Find episode directories
    episode_dirs = sorted([
        p for p in local_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and p.name.endswith(".zarr")
    ])

    if not episode_dirs:
        return {"fps": 30, "features": {}, "total_episodes": 0, "total_frames": 0}

    # Load info from first episode for common fields (zarr.json lives in episode root)
    base_info = {}
    episode_root = episode_dirs[0]
    json_path = episode_root / "zarr.json"
    first_ep_info_path = episode_root / "meta" / "info.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            base_info = json.load(f).get("attributes", {})
    elif first_ep_info_path.exists():
        with open(first_ep_info_path, "r") as f:
            base_info = json.load(f)

    # Aggregate totals across all episodes
    total_frames = 0
    tasks = set()

    for ep_dir in episode_dirs:
        json_path = ep_dir / "zarr.json"
        ep_info_path = ep_dir / "meta" / "info.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                ep_info = json.load(f).get("attributes", {})
        elif ep_info_path.exists():
            with open(ep_info_path, "r") as f:
                ep_info = json.load(f)
        else:
            ep_info = {}

        if ep_info:
            total_frames += ep_info.get("total_frames", 0)
            if ep_info.get("task"):
                tasks.add(ep_info["task"])

    # Build aggregated info
    info = {
        "fps": base_info.get("fps", 30),
        "robot_type": base_info.get("robot_type"),
        "features": base_info.get("features", {}),
        "total_episodes": len(episode_dirs),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
    }

    # Convert shape lists to tuples
    for ft in info.get("features", {}).values():
        if "shape" in ft:
            ft["shape"] = tuple(ft["shape"])

    return info


def load_stats(local_dir: Path) -> dict | None:
    """Load stats.json if it exists."""
    stats_path = local_dir / STATS_PATH
    if not stats_path.exists():
        return None
    stats = load_json(stats_path)
    stats = {key: torch.tensor(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_tasks(local_dir: Path) -> dict:
    """Load tasks.jsonl."""
    tasks_path = local_dir / TASKS_PATH
    if not tasks_path.exists():
        return {}
    tasks = load_jsonlines(tasks_path)
    return {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}





class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path:str
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
        """
        self.episode_path = Episode_path


    def __len__(self) -> int:
        if self.total_frames.exists():
            return self.total_frames
        else:
            json_path = self.episode_path / "zarr.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    ep_info = json.load(f).get("attributes", {})
                    self.total_frames = ep_info.get("total_frames", 0)


    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_reader = ZarrEpisode(self.episode_path)
        data = episode_reader.read(idx)
        # Squeeze batch dim to single frame
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out



