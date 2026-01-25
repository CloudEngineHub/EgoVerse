from egomimic.rldb.utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader


dataset = MultiRLDBDataset(
    datasets={
        "indomain": S3RLDBDataset(
            bucket_name="rldb",
            mode="train",
            embodiment="aria_right_arm",
            cache_root="/coc/flash7/rpunamiya6/.cache",
            filters={
                "task": "object_in_container_indomain",
            },
            local_files_only=True,
        ),
        "everse_song": S3RLDBDataset(
            bucket_name="rldb",
            mode="train",
            embodiment="aria_right_arm",
            cache_root="/coc/flash7/rpunamiya6/.cache",
            filters={"task": "object in container", "lab": "song"},
            local_files_only=True,
        ),
        "everse_wang": S3RLDBDataset(
            bucket_name="rldb",
            mode="train",
            embodiment="aria_right_arm",
            cache_root="/coc/flash7/rpunamiya6/.cache",
            filters={"task": "object in container", "lab": "wang"},
            local_files_only=True,
        ),
    },
    embodiment="aria_right_arm",
)


def is_bad_tensor(x, max_abs=1e4):
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return True, "empty tensor"

        if not torch.isfinite(x).all():
            return True, "non-finite values"

        if x.dtype.is_floating_point:
            if torch.abs(x).max().item() > max_abs:
                return True, f"abs value > {max_abs}"

        return False, None

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return True, "empty array"

        if not np.isfinite(x).all():
            return True, "non-finite values"

        if np.issubdtype(x.dtype, np.floating):
            if np.abs(x).max() > max_abs:
                return True, f"abs value > {max_abs}"

        return False, None

    return False, None


def check_batch(obj, prefix=""):
    """
    Recursively check a batch for bad data.
    Returns a list of (key_path, reason) tuples.
    """
    bad = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(check_batch(v, f"{prefix}.{k}" if prefix else k))

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(check_batch(v, f"{prefix}[{i}]"))

    else:
        is_bad, reason = is_bad_tensor(obj)
        if is_bad:
            bad.append((prefix, reason))

    return bad


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
)

for batch_idx, batch in enumerate(loader):
    bad_entries = check_batch(batch)
    print(batch_idx)
    if bad_entries:
        print(f"\n[BAD BATCH] batch_idx={batch_idx}")
        for key, reason in bad_entries:
            print(f"  - {key}: {reason}")
        break  # or continue, depending on how aggressive you want to be
