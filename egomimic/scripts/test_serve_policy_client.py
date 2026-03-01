#!/usr/bin/env python3
"""
Test script for EgoVerse policy serving.

Connects to a running policy server (or uses local policy), loads dataset samples,
sends observations to the server, and compares predicted actions with ground truth.
Produces visualizations and statistics.

Usage:
    # With server running (start with: python egomimic/scripts/serve_policy.py --checkpoint path/to/last.ckpt --port 8000)
    python egomimic/scripts/test_serve_policy_client.py --host localhost --port 8000 --data-config egomimic/hydra_configs/data/test_RBY1.yaml

    # Local mode (no server; loads checkpoint directly)
    python egomimic/scripts/test_serve_policy_client.py --local --checkpoint logs/.../last.ckpt

Example:
    1. Local: test_serve_policy_client.py --local --checkpoint logs/RBY_test/test_2026-02-27_11-39-37/checkpoints/last.ckpt --episode-idx 1 --max-steps 20 --save-trajectory-imgs
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from egomimic.rldb.utils import FolderRLDBDataset, get_embodiment_id
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Dataset key mapping: lerobot_key -> server obs key (data_schematic batch key)
# RBY1 from train.yaml schematic_dict. Robomimic conversion uses obs.*
RBY1_LEROBOT_TO_OBS = {
    "obs.aria_image": "front_img_1",
    "obs.robot0_joint_pos": "robot0_joint_pos",
}


def _dataset_sample_to_obs(sample: dict) -> dict:
    """
    Convert dataset sample to server observation format.

    Dataset keys (from test_RBY1 / RLDB): obs.aria_image, obs.robot0_joint_pos, etc.
    Server expects: front_img_1 (H,W,3 uint8), robot0_joint_pos (D,) float32
    """
    obs = {}
    for lerobot_key, obs_key in RBY1_LEROBOT_TO_OBS.items():
        if lerobot_key not in sample:
            continue

        # INSERT_YOUR_CODE
        # Save aria_image if present
        # if lerobot_key == "obs.aria_image":
        #     img = sample[lerobot_key]
        #     # TODO: tensor is 0-1
        #     breakpoint()
        #     if torch.is_tensor(img):
        #         img = img.numpy()
        #     if img.ndim == 3 and img.shape[0] == 3:
        #         img = np.transpose(img, (1, 2, 0))
        #     if img.dtype != np.uint8:
        #         img = np.clip(img, 0, 255).astype(np.uint8)
        #     import imageio.v2 as imageio
        #     imageio.imwrite("obs_aria_image.png", img)
        # # breakpoint()
        # assert False
        val = sample[lerobot_key]
        if torch.is_tensor(val):
            val = val.numpy()
        if obs_key == "front_img_1":
            # Image: ensure (H,W,3) uint8. LeRobot/robomimic may store (C,H,W)
            if val.ndim == 3:
                if val.shape[0] == 3:
                    val = np.transpose(val, (1, 2, 0))
                # if val.dtype != np.uint8:
                #     val = np.clip(val, 0, 255).astype(np.uint8)
                val.astype(np.float32)
            obs[obs_key] = val
        else:
            obs[obs_key] = val.astype(np.float32).ravel()
    return obs


def _get_gt_actions(sample: dict) -> np.ndarray:
    """Extract ground truth actions from sample. Shape (action_horizon, action_dim)."""
    ac = sample.get("actions")
    if ac is None:
        raise KeyError("Sample has no 'actions' key")
    if torch.is_tensor(ac):
        ac = ac.numpy()
    return np.asarray(ac, dtype=np.float32)


# --- Client ---


def create_websocket_client(host: str, port: int):
    """Create WebSocket client for policy inference."""
    import msgpack_numpy
    import websockets.sync.client

    msgpack_numpy.patch()
    uri = f"ws://{host}:{port}"

    logger.info("Connecting to %s...", uri)
    conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
    metadata = msgpack_numpy.unpackb(conn.recv())
    logger.info("Connected. Server metadata: %s", metadata)

    def infer(obs: dict) -> dict:
        packed = msgpack_numpy.packb(obs)
        conn.send(packed)
        response = conn.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error: {response}")
        return msgpack_numpy.unpackb(response)

    def close():
        conn.close()

    return infer, close, metadata


def create_local_policy(checkpoint_path: str):
    """Load policy locally for inference (no server)."""
    from egomimic.models.denoising_policy import DenoisingPolicy
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.serving.egoverse_policy import EgoVersePolicy

    logger.info("Loading policy from %s", checkpoint_path)
    model = ModelWrapper.load_from_checkpoint(checkpoint_path, weights_only=False)
    if getattr(model.model, "diffusion", False):
        for head in model.model.nets["policy"].heads.values():
            if isinstance(head, DenoisingPolicy):
                head.num_inference_steps = 10
        logger.info("Set diffusion num_inference_steps=10")

    policy = EgoVersePolicy(model)

    def infer(obs: dict) -> dict:
        return policy.infer(obs)

    def close():
        pass

    return infer, close, policy.metadata


# --- Metrics ---


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute MSE, MAE, and per-dimension stats."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    diff = pred - gt
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    # Per-dimension MAE
    mae_per_dim = np.mean(np.abs(diff), axis=(0, 1))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mae_per_dim": mae_per_dim,
        "mae_per_dim_mean": float(np.mean(mae_per_dim)),
        "mae_per_dim_max": float(np.max(mae_per_dim)),
    }


# --- Visualization ---


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB if needed."""
    if img.shape[-1] == 3:
        # Assume BGR from OpenCV
        return img[..., [2, 1, 0]].copy()
    return img


def save_visualization(
    out_dir: Path,
    sample_idx: int,
    obs_img: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    metrics: dict,
):
    """Save input image, GT vs pred action plots, and metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Input image
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    rgb = _ensure_rgb(obs_img)
    ax.imshow(rgb)
    ax.set_title(f"Sample {sample_idx}: Input Observation")
    ax.axis("off")
    fig.savefig(out_dir / f"sample_{sample_idx:04d}_input.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2. Action comparison: plot a subset of action dims over horizon
    H, D = gt_actions.shape  # H=horizon, D=dim
    n_plots = min(8, D)  # Plot up to 8 dims
    dims_to_plot = np.linspace(0, D - 1, n_plots, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i, dim in enumerate(dims_to_plot):
        ax = axes[i]
        ax.plot(gt_actions[:, dim], "b-o", label="GT", markersize=4)
        ax.plot(pred_actions[:, dim], "r--x", label="Pred", markersize=4)
        ax.set_title(f"Dim {dim}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Sample {sample_idx}: GT vs Pred Actions (selected dims)")
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{sample_idx:04d}_actions.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 3. Trajectory vs time for ALL dimensions (GT and pred)
    time_steps = np.arange(H)
    n_cols = min(7, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)
    for dim in range(D):
        row, col = dim // n_cols, dim % n_cols
        ax = axes[row, col]
        ax.plot(time_steps, gt_actions[:, dim], "b-o", label="GT", markersize=3)
        ax.plot(time_steps, pred_actions[:, dim], "r--x", label="Pred", markersize=3)
        ax.set_title(f"Dim {dim}", fontsize=9)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=7)
    for dim in range(D, axes.size):
        row, col = dim // n_cols, dim % n_cols
        axes[row, col].axis("off")
    fig.suptitle(f"Sample {sample_idx}: Trajectory vs Time (all {D} dims)")
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{sample_idx:04d}_trajectory_all_dims.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 4. Metrics text
    with open(out_dir / f"sample_{sample_idx:04d}_metrics.txt", "w") as f:
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")


def _get_episode_ranges(dataset) -> list[tuple[int, int]]:
    """Scan dataset and return [(start_idx, length), ...] for each episode."""
    ranges = []
    start = None
    prev_ep = None
    for i in range(len(dataset)):
        sample = dataset[i]
        ep = int(sample["episode_index"])
        if start is None:
            start = i
            prev_ep = ep
        elif ep != prev_ep:
            ranges.append((start, i - start))
            start = i
            prev_ep = ep
    if start is not None:
        ranges.append((start, len(dataset) - start))
    return ranges


def test_trajectory(
    infer_fn,
    dataset,
    episode_idx: int,
    output_dir: Path,
    max_steps: int | None = None,
    save_imgs: bool = False,
) -> dict:
    """
    Test policy on a full episode trajectory.

    1. Load episode frames; for each timestep t, feed obs_t to policy and get action chunk.
    2. Use the first step of each chunk as the action for that timestep (action chunking).
    3. Plot GT vs predicted trajectory for each action dimension over all timesteps.

    Returns:
        dict with trajectory metrics (mse, mae, etc.).
    """
    from tqdm import tqdm

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranges = _get_episode_ranges(dataset)
    if episode_idx >= len(ranges):
        raise ValueError(
            f"episode_idx={episode_idx} out of range; dataset has {len(ranges)} episodes"
        )
    start, length = ranges[episode_idx]
    if max_steps is not None:
        length = min(length, max_steps)

    required = {"front_img_1", "robot0_joint_pos"}
    preds = []
    gts = []
    imgs_dir = None
    if save_imgs:
        imgs_dir = Path(output_dir) / "viz" / f"trajectory_ep{episode_idx:04d}_frames"
        imgs_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(length), desc="Trajectory frame", unit="frame"):
        sample = dataset[start + i]
        obs = _dataset_sample_to_obs(sample)
        if not required.issubset(obs.keys()):
            logger.warning("Frame %d missing required keys, stopping trajectory", i)
            break

        if save_imgs and "front_img_1" in obs:
            img = obs["front_img_1"]
            rgb = _ensure_rgb(img)
            plt.imsave(imgs_dir / f"frame_{i:04d}.png", rgb)

        result = infer_fn(obs)
        ac = result["actions"]
        if ac.ndim == 3:
            ac = ac[0]
        # First step of chunk = action for this timestep (action chunking)
        preds.append(ac[0, :])

        gt_chunk = _get_gt_actions(sample)
        gts.append(gt_chunk[0, :])

    pred_traj = np.stack(preds, axis=0).astype(np.float32)  # (T, D)
    gt_traj = np.stack(gts, axis=0).astype(np.float32)  # (T, D)

    T, D = pred_traj.shape
    logger.info("Trajectory: episode %d, %d steps, %d action dims", episode_idx, T, D)

    metrics = compute_metrics(pred_traj, gt_traj)

    out_dir = Path(output_dir) / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot each action dim: trajectory vs timestep (GT and Pred)
    time_steps = np.arange(T)
    n_cols = min(7, D)
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)
    gt_handles, pred_handles = None, None
    for dim in range(D):
        row, col = dim // n_cols, dim % n_cols
        ax = axes[row, col]
        ln_gt, = ax.plot(time_steps, gt_traj[:, dim], "b-o", label="GT (ground truth)", markersize=3)
        ln_pred, = ax.plot(time_steps, pred_traj[:, dim], "r--x", label="Pred (policy)", markersize=3)
        if gt_handles is None:
            gt_handles, pred_handles = ln_gt, ln_pred
        ax.set_title(f"Dim {dim}", fontsize=9)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)
    fig.legend(
        [gt_handles, pred_handles],
        ["GT (ground truth)", "Pred (policy)"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    for idx in range(D, axes.size):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].axis("off")
    fig.suptitle(f"Episode {episode_idx}: Trajectory vs Timestep (all {D} dims, T={T})")
    fig.tight_layout()
    out_path = out_dir / f"trajectory_ep{episode_idx:04d}_all_dims.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trajectory plot saved to %s", out_path)
    if save_imgs and imgs_dir is not None:
        logger.info("Trajectory input images saved to %s", imgs_dir)

    # Save trajectory metrics
    stats_path = Path(output_dir) / f"trajectory_ep{episode_idx:04d}_metrics.txt"
    with open(stats_path, "w") as f:
        f.write(f"Episode {episode_idx}, T={T}, D={D}\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
    logger.info("Trajectory metrics: MSE=%.6f MAE=%.6f", metrics["mse"], metrics["mae"])

    return metrics


def run_test(
    infer_fn,
    dataset,
    num_samples: int,
    output_dir: Path,
    save_viz: bool = True,
):
    """Run inference on dataset samples and compute statistics."""
    all_metrics = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
        # Unbatch
        sample = {k: v[0] for k, v in batch.items()}

        obs = _dataset_sample_to_obs(sample)
        gt = _get_gt_actions(sample)

        # API requires front_img_1 and robot0_joint_pos for RBY1
        required = {"front_img_1", "robot0_joint_pos"}
        if not required.issubset(obs.keys()):
            logger.warning(
                "Sample %d missing required keys (need %s, got %s), skipping",
                i, required, set(obs.keys()),
            )
            continue

        result = infer_fn(obs)
        pred = result["actions"]
        if pred.ndim == 3:
            pred = pred[0]
        pred = np.asarray(pred, dtype=np.float32)

        if pred.shape != gt.shape:
            logger.warning(
                "Shape mismatch sample %d: pred %s vs gt %s",
                i, pred.shape, gt.shape
            )
            # Truncate/pad to match
            min_h = min(pred.shape[0], gt.shape[0])
            min_d = min(pred.shape[1], gt.shape[1])
            pred = pred[:min_h, :min_d]
            gt = gt[:min_h, :min_d]

        metrics = compute_metrics(pred, gt)
        all_metrics.append(metrics)

        if save_viz and "front_img_1" in obs:
            save_visualization(
                output_dir / "viz",
                i,
                obs["front_img_1"],
                gt,
                pred,
                metrics,
            )

        if (i + 1) % 10 == 0:
            logger.info("Processed %d / %d samples", i + 1, num_samples)

    # Aggregate statistics
    agg = {
        "mse_mean": np.mean([m["mse"] for m in all_metrics]),
        "mse_std": np.std([m["mse"] for m in all_metrics]),
        "mae_mean": np.mean([m["mae"] for m in all_metrics]),
        "mae_std": np.std([m["mae"] for m in all_metrics]),
        "rmse_mean": np.mean([m["rmse"] for m in all_metrics]),
        "mae_per_dim_mean": np.mean([m["mae_per_dim_mean"] for m in all_metrics]),
        "n_samples": len(all_metrics),
    }

    # Save aggregate stats
    stats_path = output_dir / "test_stats.txt"
    with open(stats_path, "w") as f:
        f.write("=== EgoVerse Policy Test Statistics ===\n\n")
        f.write(f"N samples: {agg['n_samples']}\n\n")
        f.write(f"MSE:  mean={agg['mse_mean']:.6f} std={agg['mse_std']:.6f}\n")
        f.write(f"MAE:  mean={agg['mae_mean']:.6f} std={agg['mae_std']:.6f}\n")
        f.write(f"RMSE: mean={agg['rmse_mean']:.6f}\n")
        f.write(f"MAE per-dim (mean over samples): {agg['mae_per_dim_mean']:.6f}\n")

    # Summary visualization (always when we have metrics)
    if all_metrics:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        (output_dir / "viz").mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        mses = [m["mse"] for m in all_metrics]
        maes = [m["mae"] for m in all_metrics]
        rmses = [m["rmse"] for m in all_metrics]
        axes[0].hist(mses, bins=min(30, len(mses)), edgecolor="black", alpha=0.7)
        axes[0].set_title("MSE distribution")
        axes[0].set_xlabel("MSE")
        axes[1].hist(maes, bins=min(30, len(maes)), edgecolor="black", alpha=0.7)
        axes[1].set_title("MAE distribution")
        axes[1].set_xlabel("MAE")
        axes[2].hist(rmses, bins=min(30, len(rmses)), edgecolor="black", alpha=0.7)
        axes[2].set_title("RMSE distribution")
        axes[2].set_xlabel("RMSE")
        fig.suptitle(f"EgoVerse Policy Test (n={agg['n_samples']})")
        fig.tight_layout()
        fig.savefig(output_dir / "viz" / "summary_distributions.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Summary plot saved to viz/summary_distributions.png")

    logger.info("Statistics written to %s", stats_path)
    logger.info(
        "MSE=%.6f MAE=%.6f RMSE=%.6f (n=%d)",
        agg["mse_mean"], agg["mae_mean"], agg["rmse_mean"], agg["n_samples"],
    )

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Test EgoVerse policy serving: client + dataset eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Policy server host (ignored in --local mode)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Policy server port",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local policy (load checkpoint) instead of WebSocket client",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (required for --local)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="egomimic/hydra_configs/data/test_RBY1.yaml",
        help="Path to data config YAML (used for dataset structure)",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="/coc/flash7/zhenyang/EgoVerse/datasets",
        help="Folder containing RBY1 dataset subdirs (eth_lab, rl2_lab, etc.)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="valid",
        choices=["train", "valid"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for viz and stats (default: logs/test_serve_policy_<timestamp>)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable per-sample visualization (faster)",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Run trajectory test: feed observations 1-by-1, plot full episode actions per dim",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=0,
        help="Episode index for --trajectory mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max timesteps per episode in --trajectory mode (default: full episode)",
    )
    parser.add_argument(
        "--save-trajectory-imgs",
        action="store_true",
        help="Save all input images from trajectory frames (only in --trajectory mode)",
    )
    args = parser.parse_args()

    if args.local and not args.checkpoint:
        parser.error("--checkpoint is required when using --local")

    from datetime import datetime
    out_dir = Path(
        args.output_dir
        or f"logs/test_serve_policy_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (mirror test_RBY1 structure)
    dataset = FolderRLDBDataset(
        folder_path=Path(args.dataset_folder),
        embodiment="rby1",
        mode=args.mode,
        local_files_only=True,
        delta_timestamps={
            "actions": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    )
    logger.info("Loaded dataset: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("Dataset is empty. Check --dataset-folder and --mode.")
        sys.exit(1)

    # Create inference client or local policy
    if args.local:
        infer_fn, close_fn, metadata = create_local_policy(args.checkpoint)
    else:
        infer_fn, close_fn, metadata = create_websocket_client(args.host, args.port)

    try:
        if args.trajectory:
            test_trajectory(
                infer_fn,
                dataset,
                episode_idx=args.episode_idx,
                output_dir=out_dir,
                max_steps=args.max_steps,
                save_imgs=args.save_trajectory_imgs,
            )
        else:
            run_test(
                infer_fn,
                dataset,
                num_samples=args.num_samples,
                output_dir=out_dir,
                save_viz=not args.no_viz,
            )
    finally:
        close_fn()

    logger.info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()
