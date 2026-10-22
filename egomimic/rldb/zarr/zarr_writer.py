"""
ZarrWriter: General-purpose Zarr episode writer.

This module provides a reusable writer for creating Zarr v3 episode stores
compatible with the ZarrEpisode reader.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import simplejpeg
import zarr
from zarr.core.dtype import VariableLengthBytes

from egomimic.rldb.zarr.required_metadata import (
    REQUIRED_METADATA_FIELDS,
    MissingMetadata,
    required_metadata_error,
)

logger = logging.getLogger(__name__)


def _intrinsics_to_jsonable(
    intrinsics: np.ndarray | list | dict,
) -> list | dict:
    """Coerce intrinsics into a JSON-serializable form for zarr.json attrs."""
    if isinstance(intrinsics, np.ndarray):
        return intrinsics.tolist()
    if isinstance(intrinsics, dict):
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in intrinsics.items()
        }
    return intrinsics


class ZarrWriter:
    """
    General-purpose writer for Zarr v3 episode stores.

    Creates episodes compatible with the ZarrEpisode reader, handling both
    numeric and image data with intelligent chunking and optional sharding.
    """

    JPEG_QUALITY = 85  # Fixed JPEG quality for image compression

    def __init__(
        self,
        episode_path: str | Path,
        *,
        embodiment: str,
        intrinsics: dict,
        task_name: "str | MissingMetadata",
        operator: "str | MissingMetadata",
        lab: "str | MissingMetadata",
        scene: "str | MissingMetadata",
        objects: "str | list[str] | MissingMetadata",
        fps: int = 30,
        task_description: str = "",
        annotations: list[tuple[str, int, int]] | None = None,
        chunk_timesteps: int = 100,
        extrinsics: dict | None = None,
    ):
        """
        Initialize ZarrWriter.

        The camera and metadata arguments are keyword-only with no default, so
        forgetting one is a TypeError at the call site; their values are
        checked by write() before anything touches disk. To edit an existing
        episode (annotations, removing a key) use the module-level
        write_annotations / remove_key / key_exists instead.

        Args:
            episode_path: Path to episode .zarr directory.
            embodiment: Embodiment identifier (e.g. "human_bimanual"); must be
                a valid EMBODIMENT name (CONTRIBUTING_DATA.md §9).
            fps: Frames per second for playback (default: 30).
            task_description: Task description.
            annotations: List of (text, start_idx, end_idx) tuples describing language annotations.
            chunk_timesteps: Number of timesteps per chunk for numeric arrays (default: 100).
            intrinsics: REQUIRED: {camera_key: 3x4 K matrix} dict
                (single-camera = one entry, e.g. {"front_1": K}). Stored under the
                "intrinsics" key in zarr.json metadata.
            extrinsics: Optional camera extrinsics. None (egocentric human data) or
                a non-empty dict of 4x4 transforms; robots key per-arm
                (e.g. {"left": world<-cam, "right": world<-cam}). Stored under the
                "extrinsics" key in zarr.json metadata.
            task_name: REQUIRED: high-level task (must match the DB
                row's task).
            operator: REQUIRED: per-episode ID of the collector.
            lab: REQUIRED: short, stable, lowercase source id.
            scene: REQUIRED: scene / location label.
            objects: REQUIRED: objects manipulated (str or list[str]).
                write() rejects an empty string or a placeholder
                such as "unknown" for any of these five. If the source data
                genuinely has no value, pass MissingMetadata(); it is stored
                as null.
        """
        self.episode_path = Path(episode_path)

        # Store parameters
        self.fps = fps
        self.embodiment = embodiment
        self.task_description = task_description
        self.annotations = annotations if annotations is not None else []
        self.chunk_timesteps = chunk_timesteps
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.required_metadata = {
            "task_name": task_name,
            "operator": operator,
            "lab": lab,
            "scene": scene,
            "objects": objects,
        }
        # Track image shapes for metadata
        self._features: dict[str, dict[str, Any]] = {}

    def write(
        self,
        numeric_data: dict[str, np.ndarray] | None = None,
        image_data: dict[str, np.ndarray] | None = None,
        pre_encoded_image_data: dict[str, tuple[np.ndarray, list[int]]] | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> Path:
        """
        Validate the episode configuration, then write it to the Zarr store.

        Args:
            numeric_data: Dictionary of numeric arrays (state, actions, etc.).
                All arrays must have same length along axis 0.
            image_data: Dictionary of image arrays with shape (T, H, W, 3).
                Images will be JPEG-compressed.
            pre_encoded_image_data: Dictionary mapping key to (encoded_array, image_shape).
                encoded_array is np.ndarray(dtype=object) of JPEG bytes.
                image_shape is [H, W, 3]. Skips internal JPEG encoding.
            metadata_override: Optional metadata overrides to apply after
                building metadata. Cannot overwrite intrinsics/extrinsics or the
                required metadata fields.

        Returns:
            Path to the written episode.

        Raises:
            ValueError: If no data is provided or arrays have inconsistent frame
                counts, if embodiment is not a valid identifier (§9), if
                intrinsics is not a non-empty {camera_key: 3x4 K matrix} dict,
                if extrinsics is neither None nor a non-empty {key: 4x4} dict, or
                if a required metadata field (task_name/operator/lab/scene/
                objects) is missing, empty, a placeholder, or malformed.
        """
        self._validate()
        numeric_data = numeric_data or {}
        image_data = image_data or {}
        pre_encoded_image_data = pre_encoded_image_data or {}

        if not numeric_data and not image_data and not pre_encoded_image_data:
            raise ValueError(
                "Must provide at least one of numeric_data, image_data, "
                "or pre_encoded_image_data"
            )

        # Infer total_frames from data
        all_lengths: list[int] = []
        for arr in numeric_data.values():
            all_lengths.append(len(arr))
        for arr in image_data.values():
            all_lengths.append(len(arr))
        for enc_arr, _shape in pre_encoded_image_data.values():
            all_lengths.append(len(enc_arr))

        if len(set(all_lengths)) > 1:
            raise ValueError(f"Inconsistent frame counts across arrays: {all_lengths}")

        self.total_frames = all_lengths[0]

        # Calculate padded frame count for sharding alignment
        padded_frames = self.total_frames
        if self.total_frames % self.chunk_timesteps != 0:
            padded_frames = (
                (self.total_frames + self.chunk_timesteps - 1) // self.chunk_timesteps
            ) * self.chunk_timesteps

        # Create parent directory
        self.episode_path.parent.mkdir(parents=True, exist_ok=True)

        # Open Zarr v3 store
        mode = "w" if self.episode_path.exists() else "w"
        store = zarr.open(str(self.episode_path), mode=mode, zarr_format=3)

        # Write numeric arrays
        for key, arr in numeric_data.items():
            self._write_numeric_array(store, key, arr, padded_frames)

        # Write image arrays (with internal JPEG encoding)
        for key, arr in image_data.items():
            self._write_image_array(store, key, arr, padded_frames)

        # Write pre-encoded image arrays (skip JPEG encoding)
        for key, (enc_arr, img_shape) in pre_encoded_image_data.items():
            self._write_pre_encoded_image_array(
                store, key, enc_arr, img_shape, padded_frames
            )

        # Write language annotations if provided
        if self.annotations is not None:
            self._features["annotations"] = _write_annotation_array(
                store, self.annotations, "annotations"
            )

        # Build and attach metadata
        metadata = self._build_metadata(metadata_override)
        store.attrs.update(metadata)
        return self.episode_path

    def _validate(self) -> None:
        """Check embodiment, camera metadata and required metadata (see write)."""
        # Validate the embodiment up front (it is guaranteed to be consumed by
        # the reader via get_embodiment_id) so a bad/empty/typo'd value fails
        # here with an actionable message instead of an opaque KeyError at load.
        from egomimic.rldb.embodiment.embodiment import EMBODIMENT

        embodiment = self.embodiment
        if not embodiment or embodiment.upper() not in EMBODIMENT.__members__:
            raise ValueError(
                "embodiment must be one of "
                f"{[m.name.lower() for m in EMBODIMENT]}, got {embodiment!r}. "
                "See CONTRIBUTING_DATA.md §9."
            )
        intrinsics = self.intrinsics
        if not isinstance(intrinsics, dict) or not intrinsics:
            raise ValueError(
                "Camera intrinsics must be a non-empty {camera_key: 3x4 K matrix} "
                'dict for a contributed zarr episode, e.g. {"front_1": K}. '
                "See CONTRIBUTING_DATA.md §6.4."
            )
        for cam_key, K in intrinsics.items():
            try:
                K_arr = np.asarray(K, dtype=np.float64)
            except (TypeError, ValueError):
                K_arr = None
            if K_arr is None or K_arr.shape != (3, 4):
                raise ValueError(
                    f"intrinsics[{cam_key!r}] must be a 3x4 K matrix (the 3x3 "
                    "pinhole K with a zero last column; pad a bare 3x3 with "
                    "np.hstack([K, np.zeros((3, 1))])), got "
                    f"{getattr(K_arr, 'shape', type(K).__name__)}. "
                    "See CONTRIBUTING_DATA.md §6.4."
                )
        extrinsics = self.extrinsics
        if extrinsics is not None and (
            not isinstance(extrinsics, dict) or not extrinsics
        ):
            raise ValueError(
                "Camera extrinsics must be None or a non-empty dict mapping a key "
                'to its transform matrix (robots key per-arm, e.g. {"left": T, '
                '"right": T}). See CONTRIBUTING_DATA.md §6.3.'
            )
        if extrinsics is not None:
            for ext_key, T in extrinsics.items():
                try:
                    T_arr = np.asarray(T, dtype=np.float64)
                except (TypeError, ValueError):
                    T_arr = None
                if T_arr is None or T_arr.shape != (4, 4):
                    raise ValueError(
                        f"extrinsics[{ext_key!r}] must be a 4x4 transform matrix, "
                        f"got {getattr(T_arr, 'shape', type(T).__name__)}. "
                        "See CONTRIBUTING_DATA.md §6.3."
                    )
        for field, value in self.required_metadata.items():
            if isinstance(value, MissingMetadata):
                logger.warning("Writing %s without %s", self.episode_path, field)
                continue
            err = required_metadata_error(field, value)
            if err is not None:
                raise ValueError(err)

    def _write_numeric_array(
        self, store: zarr.Group, key: str, arr: np.ndarray, padded_frames: int
    ) -> None:
        """
        Write a numeric array to the Zarr store.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            arr: Numeric array with shape (T, ...).
            padded_frames: Target frame count after padding (for sharding alignment).
        """
        num_frames = len(arr)

        # Store original shape and dtype before padding for metadata
        original_shape = arr.shape[1:]  # Shape excluding time dimension
        dtype_str = str(arr.dtype)

        # Pad array if needed
        if padded_frames > num_frames:
            pad_len = padded_frames - num_frames
            pad_shape = (pad_len,) + arr.shape[1:]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

        # Use chunk_timesteps for frames per chunk
        frames_per_chunk = min(self.chunk_timesteps, padded_frames)
        frames_per_chunk = max(1, frames_per_chunk)

        # Chunk shape: (frames, ...) - keep other dimensions intact
        chunk_shape = (frames_per_chunk,) + arr.shape[1:]

        store.create_array(
            key,
            data=arr,
            chunks=chunk_shape,
            shards=arr.shape,
        )

        # Track shape and dtype for metadata
        dimension_names = [f"dim_{i}" for i in range(len(original_shape))]
        self._features[key] = {
            "dtype": dtype_str,
            "shape": list(original_shape),
            "names": dimension_names,
        }

    def _write_image_array(
        self, store: zarr.Group, key: str, image_arr: np.ndarray, padded_frames: int
    ) -> None:
        """
        Write an image array to the Zarr store with JPEG compression.

        Images are always chunked 1 per timestep for efficient random access,
        regardless of chunk_timesteps setting.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            image_arr: Image array with shape (T, H, W, 3).
            padded_frames: Target frame count after padding (for sharding alignment).
        """
        # Validate shape
        if image_arr.ndim != 4 or image_arr.shape[-1] != 3:
            raise ValueError(
                f"Image array '{key}' must have shape (T, H, W, 3), got {image_arr.shape}"
            )

        # Encode each frame as JPEG
        num_frames = len(image_arr)

        # Encode to padded_frames length (pad with duplicate of last frame if needed)
        encoded = np.empty((padded_frames,), dtype=object)
        for i in range(padded_frames):
            # Use last frame for padding
            frame_idx = min(i, num_frames - 1)
            img = image_arr[frame_idx]
            jpeg_bytes = simplejpeg.encode_jpeg(
                img, quality=self.JPEG_QUALITY, colorspace="RGB"
            )
            encoded[i] = jpeg_bytes

        # Images are always chunked 1 per timestep, regardless of chunk_timesteps
        chunk_shape = (1,)

        store.create_array(
            key,
            shape=encoded.shape,
            chunks=chunk_shape,
            shards=encoded.shape,
            dtype=VariableLengthBytes(),
        )

        # Assign data after creation (required for VariableLengthBytes)
        store[key][:] = encoded

        # Track shape for metadata
        self._features[key] = {
            "dtype": "jpeg",
            "shape": list(image_arr.shape[1:]),  # [H, W, 3]
            "names": ["height", "width", "channel"],
        }

    def _write_pre_encoded_image_array(
        self,
        store: zarr.Group,
        key: str,
        encoded_arr: np.ndarray,
        image_shape: list[int],
        padded_frames: int,
    ) -> None:
        """
        Write already-JPEG-encoded image data to the Zarr store.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            encoded_arr: Object array of JPEG bytes with shape (T,).
            image_shape: Original image dimensions [H, W, 3].
            padded_frames: Target frame count after padding.
        """
        num_frames = len(encoded_arr)

        if padded_frames > num_frames:
            padded = np.empty((padded_frames,), dtype=object)
            padded[:num_frames] = encoded_arr
            last_jpeg = encoded_arr[-1]
            for i in range(num_frames, padded_frames):
                padded[i] = last_jpeg
            encoded_arr = padded

        chunk_shape = (1,)

        store.create_array(
            key,
            shape=encoded_arr.shape,
            chunks=chunk_shape,
            shards=encoded_arr.shape,
            dtype=VariableLengthBytes(),
        )

        store[key][:] = encoded_arr

        self._features[key] = {
            "dtype": "jpeg",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    def _build_metadata(
        self, metadata_override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Build episode metadata dictionary.

        Args:
            metadata_override: Optional overrides to apply.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "embodiment": self.embodiment,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "task_description": self.task_description,
            "features": self._features,
        }

        if self.intrinsics is not None:
            metadata["intrinsics"] = _intrinsics_to_jsonable(self.intrinsics)

        if self.extrinsics is not None:
            metadata["extrinsics"] = _intrinsics_to_jsonable(self.extrinsics)

        for field in REQUIRED_METADATA_FIELDS:
            value = self.required_metadata[field]
            if isinstance(value, MissingMetadata):
                value = None
            elif isinstance(value, tuple):
                value = list(value)
            metadata[field] = value

        # Apply overrides — but NEVER let them clobber the validated camera
        # metadata. intrinsics/extrinsics are validated in _validate; a
        # converter's metadata_override that happens to carry a stale or empty
        # "intrinsics"/"extrinsics" key must not silently overwrite the validated
        # values (this was the Mecka clobber bug → empty intrinsics in zarr.json).
        # The same goes for the validated required metadata fields.
        if metadata_override:
            protected = {"intrinsics", "extrinsics"}
            protected.update(REQUIRED_METADATA_FIELDS)
            override = {
                k: v for k, v in metadata_override.items() if k not in protected
            }
            metadata.update(override)

        return metadata

    @staticmethod
    def create_and_write(
        episode_path: str | Path,
        numeric_data: dict[str, np.ndarray] | None = None,
        image_data: dict[str, np.ndarray] | None = None,
        pre_encoded_image_data: dict[str, tuple[np.ndarray, list[int]]] | None = None,
        *,
        embodiment: str,
        intrinsics: dict,
        task_name: "str | MissingMetadata",
        operator: "str | MissingMetadata",
        lab: "str | MissingMetadata",
        scene: "str | MissingMetadata",
        objects: "str | list[str] | MissingMetadata",
        fps: int = 30,
        task_description: str = "",
        annotations: list[tuple[str, int, int]] | None = None,
        chunk_timesteps: int = 100,
        extrinsics: dict | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> Path:
        """Build a ZarrWriter and write the episode in one call.

        Arguments are as in ZarrWriter.__init__ and ZarrWriter.write; all
        validation happens in write(). Returns the episode path.
        """
        return ZarrWriter(
            episode_path,
            embodiment=embodiment,
            intrinsics=intrinsics,
            task_name=task_name,
            operator=operator,
            lab=lab,
            scene=scene,
            objects=objects,
            fps=fps,
            task_description=task_description,
            annotations=annotations,
            chunk_timesteps=chunk_timesteps,
            extrinsics=extrinsics,
        ).write(
            numeric_data=numeric_data,
            image_data=image_data,
            pre_encoded_image_data=pre_encoded_image_data,
            metadata_override=metadata_override,
        )


def _write_annotation_array(
    store: zarr.Group,
    annotations: list[tuple[str, int, int]],
    annotation_key: str,
) -> dict[str, Any]:
    """Write (text, start_idx, end_idx) annotations as JSON-encoded bytes.

    Returns the array's ``features`` entry.
    """
    encoded = np.array(
        [
            json.dumps(
                {
                    "text": text,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            for text, start_idx, end_idx in annotations
        ],
        dtype=object,
    )

    n_annotations = len(annotations)
    # Both the chunk and the SHARD edge must be >= 1 even when there are no
    # annotations: zarr v3 rejects a zero-length chunk grid, so an episode
    # written with shards=(0,) cannot be enumerated (group.keys() raises
    # "integer chunk edge length must be >= 1") even though the array itself
    # reads back fine. shape=(0,) with a >=1 edge is the correct encoding of
    # "empty" -- no chunk objects are written either way.
    store.create_array(
        annotation_key,
        shape=encoded.shape,
        chunks=(max(1, n_annotations),),
        shards=(max(1, n_annotations),),
        dtype=VariableLengthBytes(),
    )
    if n_annotations > 0:
        store[annotation_key][:] = encoded

    return {
        "dtype": "json",
        "shape": [n_annotations],
        "names": ["json"],
        "format": "annotation_v1",
    }


# ── Editing an existing episode ─────────────────────────────────────────────
# These work on an episode that is already written, so they need none of the
# metadata a ZarrWriter requires. Each keeps zarr.attrs["features"] in sync
# with the arrays (the episode validator checks that).


def key_exists(episode_path: str | Path, key: str) -> bool:
    """Whether ``key`` is an array/group in the episode store."""
    return key in zarr.open(str(episode_path), mode="r", zarr_format=3)


def remove_key(episode_path: str | Path, key: str) -> None:
    """Delete ``key`` from the episode store and from its features."""
    store = zarr.open_group(str(episode_path), mode="a", zarr_format=3)
    del store[key]
    features = dict(store.attrs.get("features", {}))
    if features.pop(key, None) is not None:
        store.attrs["features"] = features


def write_annotations(
    episode_path: str | Path,
    annotations: list[tuple[str, int, int]],
    annotation_key: str = "annotations",
    mode: Literal["w", "a"] = "a",
) -> None:
    """Write language annotations into an existing episode.

    Args:
        episode_path: Path to the episode .zarr directory.
        annotations: List of (text, start_idx, end_idx) tuples.
        annotation_key: Array key to write (default "annotations").
        mode: "w" replaces an existing ``annotation_key``; "a" only adds a new
            key and raises if it already exists.
    """
    if key_exists(episode_path, annotation_key):
        if mode != "w":
            raise ValueError(
                f"{annotation_key!r} already exists in {episode_path}; "
                'pass mode="w" to replace it'
            )
        remove_key(episode_path, annotation_key)
    store = zarr.open_group(str(episode_path), mode="a", zarr_format=3)
    feature = _write_annotation_array(store, annotations, annotation_key)
    features = dict(store.attrs.get("features", {}))
    features[annotation_key] = feature
    store.attrs["features"] = features
