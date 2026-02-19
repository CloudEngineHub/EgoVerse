"""
Preprocessing script to convert egomimic single-frame dataset to cosmos_policy video sequence format.

This script can be used if runtime transformation in process_batch_for_training is insufficient
or if offline preprocessing is preferred for performance.

Usage:
    python egomimic/scripts/wm_process/egomimic_to_cosmos_policy.py \
        --input_dir /path/to/egomimic/data \
        --output_dir /path/to/cosmos_policy/data \
        --chunk_size 8 \
        --num_history_frames 1
"""

import argparse
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

from egomimic.rldb.utils import RLDBDataset


def load_egomimic_episode(dataset: RLDBDataset, episode_idx: int) -> Dict:
    """
    Load a single episode from egomimic dataset.
    
    Args:
        dataset: RLDBDataset instance
        episode_idx: Episode index
        
    Returns:
        dict: Episode data with images, actions, proprio, etc.
    """
    # Get episode frame indices
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_idx].item()
    to_idx = episode_data_index["to"][episode_idx].item()
    
    episode_data = {
        "images": [],
        "actions": [],
        "proprio": [],
        "annotations": [],
        "num_steps": to_idx - from_idx,
    }
    
    # Load all frames in episode
    for frame_idx in range(from_idx, to_idx):
        frame_data = dataset[frame_idx]
        
        # Collect images (assuming first camera key)
        camera_keys = dataset.meta.camera_keys
        if camera_keys:
            img_key = camera_keys[0]
            if img_key in frame_data:
                img = frame_data[img_key]  # (C, H, W)
                episode_data["images"].append(img.numpy() if torch.is_tensor(img) else img)
        
        # Collect actions
        if "action" in frame_data:
            action = frame_data["action"]  # (D,)
            episode_data["actions"].append(action.numpy() if torch.is_tensor(action) else action)
        
        # Collect proprioception
        if "observation.state" in frame_data:
            state = frame_data["observation.state"]  # (D,)
            episode_data["proprio"].append(state.numpy() if torch.is_tensor(state) else state)
        
        # Collect annotations
        if "annotations" in frame_data:
            episode_data["annotations"].append(frame_data["annotations"])
    
    # Convert lists to numpy arrays
    if episode_data["images"]:
        episode_data["images"] = np.stack(episode_data["images"], axis=0)  # (T, C, H, W)
    if episode_data["actions"]:
        episode_data["actions"] = np.stack(episode_data["actions"], axis=0)  # (T, D)
    if episode_data["proprio"]:
        episode_data["proprio"] = np.stack(episode_data["proprio"], axis=0)  # (T, D)
    
    return episode_data


def create_video_sequence(
    images: np.ndarray,
    num_history_frames: int,
    history_spacing: int = 1
) -> np.ndarray:
    """
    Create video sequence from single frames by repeating/using history.
    
    Args:
        images: (T, C, H, W) array of images
        num_history_frames: Number of history frames to include
        history_spacing: Spacing between history frames
        
    Returns:
        (T, num_history_frames+1, C, H, W) array of video sequences
    """
    T = images.shape[0]
    video_sequences = []
    
    for t in range(T):
        # Collect history frames
        history = []
        for i in range(num_history_frames):
            hist_idx = max(0, t - (num_history_frames - i) * history_spacing)
            history.append(images[hist_idx])
        
        # Add current frame
        history.append(images[t])
        
        # Stack to create sequence: (num_history_frames+1, C, H, W)
        sequence = np.stack(history, axis=0)
        video_sequences.append(sequence)
    
    # Stack all sequences: (T, num_history_frames+1, C, H, W)
    return np.stack(video_sequences, axis=0)


def extract_action_chunks(
    actions: np.ndarray,
    chunk_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract action chunks from action sequence.
    
    Args:
        actions: (T, D) array of actions
        chunk_size: Size of action chunk
        
    Returns:
        action_chunks: (T, chunk_size, D) array of action chunks
        valid_mask: (T,) boolean array indicating valid chunks
    """
    T = actions.shape[0]
    action_chunks = []
    valid_mask = []
    
    for t in range(T):
        if t + chunk_size <= T:
            chunk = actions[t:t+chunk_size]  # (chunk_size, D)
            action_chunks.append(chunk)
            valid_mask.append(True)
        else:
            # Pad with last action if needed
            chunk = actions[t:]
            padding = np.tile(actions[-1:], (chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, padding], axis=0)
            action_chunks.append(chunk)
            valid_mask.append(False)
    
    return np.stack(action_chunks, axis=0), np.array(valid_mask)


def save_cosmos_policy_episode(
    output_path: str,
    video_sequences: np.ndarray,
    action_chunks: np.ndarray,
    proprio: Optional[np.ndarray],
    annotations: Optional[List[str]],
    metadata: Dict
):
    """
    Save episode in cosmos_policy HDF5 format.
    
    Args:
        output_path: Path to output HDF5 file
        video_sequences: (T, num_frames, C, H, W) video sequences
        action_chunks: (T, chunk_size, D) action chunks
        proprio: (T, D_proprio) proprioception or None
        annotations: List of annotation strings or None
        metadata: Additional metadata dict
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        # Save video sequences
        f.create_dataset("video_sequences", data=video_sequences, compression="gzip")
        
        # Save action chunks
        f.create_dataset("actions", data=action_chunks, compression="gzip")
        
        # Save proprioception if available
        if proprio is not None:
            f.create_dataset("proprio", data=proprio, compression="gzip")
        
        # Save annotations if available
        if annotations:
            # Store as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("annotations", data=annotations, dtype=dt)
        
        # Save metadata
        metadata_group = f.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                metadata_group.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                metadata_group.attrs[key] = json.dumps(value)
            else:
                metadata_group.attrs[key] = str(value)


def process_dataset(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 8,
    num_history_frames: int = 1,
    history_spacing: int = 1,
    final_image_size: int = 224,
    use_proprio: bool = True,
    max_episodes: Optional[int] = None
):
    """
    Process egomimic dataset to cosmos_policy format.
    
    Args:
        input_dir: Path to egomimic dataset directory
        output_dir: Path to output directory for cosmos_policy data
        chunk_size: Action chunk size
        num_history_frames: Number of history frames in video sequence
        history_spacing: Spacing between history frames
        final_image_size: Target image size (will resize if needed)
        use_proprio: Whether to include proprioception
        max_episodes: Maximum number of episodes to process (None for all)
    """
    print(f"Loading egomimic dataset from: {input_dir}")
    
    # Load egomimic dataset
    dataset = RLDBDataset(
        repo_id=None,
        root=input_dir,
        local_files_only=True,
        mode="train"
    )
    
    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)
    
    print(f"Processing {num_episodes} episodes...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each episode
    episode_files = []
    for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        try:
            # Load episode
            episode_data = load_egomimic_episode(dataset, episode_idx)
            
            if episode_data["num_steps"] < chunk_size:
                print(f"Skipping episode {episode_idx}: too short ({episode_data['num_steps']} < {chunk_size})")
                continue
            
            # Create video sequences
            if len(episode_data["images"]) == 0:
                print(f"Skipping episode {episode_idx}: no images")
                continue
            
            images = episode_data["images"]  # (T, C, H, W)
            
            # Resize images if needed
            if images.shape[2] != final_image_size or images.shape[3] != final_image_size:
                from PIL import Image
                resized_images = []
                for img in images:
                    # Convert to PIL Image and resize
                    img_pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
                    img_pil = img_pil.resize((final_image_size, final_image_size))
                    img_resized = np.array(img_pil).transpose(2, 0, 1) / 255.0
                    resized_images.append(img_resized)
                images = np.stack(resized_images, axis=0)
            
            video_sequences = create_video_sequence(
                images, num_history_frames, history_spacing
            )  # (T, num_history_frames+1, C, H, W)
            
            # Extract action chunks
            if len(episode_data["actions"]) == 0:
                print(f"Skipping episode {episode_idx}: no actions")
                continue
            
            actions = episode_data["actions"]  # (T, D)
            action_chunks, valid_mask = extract_action_chunks(actions, chunk_size)
            
            # Get proprioception
            proprio = None
            if use_proprio and len(episode_data["proprio"]) > 0:
                proprio = episode_data["proprio"]  # (T, D_proprio)
            
            # Get annotations
            annotations = episode_data.get("annotations", None)
            
            # Create metadata
            metadata = {
                "episode_idx": episode_idx,
                "num_steps": episode_data["num_steps"],
                "chunk_size": chunk_size,
                "num_history_frames": num_history_frames,
                "final_image_size": final_image_size,
                "use_proprio": use_proprio,
            }
            
            # Save episode
            output_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.hdf5")
            save_cosmos_policy_episode(
                output_path,
                video_sequences,
                action_chunks,
                proprio,
                annotations,
                metadata
            )
            
            episode_files.append(output_path)
        
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            continue
    
    # Save dataset metadata
    dataset_metadata = {
        "num_episodes": len(episode_files),
        "chunk_size": chunk_size,
        "num_history_frames": num_history_frames,
        "final_image_size": final_image_size,
        "use_proprio": use_proprio,
        "episode_files": episode_files,
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"\nProcessed {len(episode_files)} episodes")
    print(f"Output saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert egomimic dataset to cosmos_policy format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to egomimic dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for cosmos_policy data"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="Action chunk size (default: 8)"
    )
    parser.add_argument(
        "--num_history_frames",
        type=int,
        default=1,
        help="Number of history frames in video sequence (default: 1)"
    )
    parser.add_argument(
        "--history_spacing",
        type=int,
        default=1,
        help="Spacing between history frames (default: 1)"
    )
    parser.add_argument(
        "--final_image_size",
        type=int,
        default=224,
        help="Target image size (default: 224)"
    )
    parser.add_argument(
        "--use_proprio",
        action="store_true",
        help="Include proprioception in output"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (default: all)"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        num_history_frames=args.num_history_frames,
        history_spacing=args.history_spacing,
        final_image_size=args.final_image_size,
        use_proprio=args.use_proprio,
        max_episodes=args.max_episodes
    )


if __name__ == "__main__":
    main()

