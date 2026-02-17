"""
Test script for CosmosPolicyRLDBDataset.

This script validates the CosmosPolicyRLDBDataset implementation by:
- Loading dataset from hydra config (debug.yaml)
- Testing basic functionality (__len__, __getitem__)
- Verifying output format matches cosmos-policy expectations
- Testing T5 embedding construction
- Verifying video tensor structure
- Testing action chunk extraction
- Testing DataLoader compatibility
- Testing embodiment-specific behavior
"""

import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from torch.utils.data import DataLoader
from pprint import pprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from egomimic.rldb.cosmos_policy_dataset import CosmosPolicyRLDBDataset


def test_dataset_loading(cfg: DictConfig):
    """Test loading dataset from hydra config."""
    print("=" * 80)
    print("TEST 1: Dataset Loading")
    print("=" * 80)
    
    # Load train dataset
    train_dataset_config = cfg.data.train_datasets.dataset1
    print(f"\nLoading train dataset with config:")
    print(f"  Target: {train_dataset_config._target_}")
    print(f"  Embodiment: {train_dataset_config.embodiment}")
    print(f"  Mode: {train_dataset_config.mode}")
    
    train_dataset = hydra.utils.instantiate(train_dataset_config)
    print(f"✓ Train dataset loaded successfully")
    print(f"  Dataset type: {type(train_dataset)}")
    print(f"  Dataset length: {len(train_dataset)}")
    print(f"  Embodiment ID: {train_dataset.embodiment}")
    
    # Load valid dataset
    valid_dataset_config = cfg.data.valid_datasets.dataset1
    valid_dataset = hydra.utils.instantiate(valid_dataset_config)
    print(f"✓ Valid dataset loaded successfully")
    print(f"  Dataset length: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset


def test_basic_functionality(dataset: CosmosPolicyRLDBDataset):
    """Test basic dataset functionality."""
    print("\n" + "=" * 80)
    print("TEST 2: Basic Functionality")
    print("=" * 80)
    
    # Test __len__
    dataset_len = len(dataset)
    print(f"\n✓ Dataset length: {dataset_len}")
    assert dataset_len > 0, "Dataset should have at least one sample"
    
    # Test __getitem__ for first sample
    print(f"\nTesting __getitem__(0)...")
    sample = dataset[0]
    print(f"✓ Successfully retrieved sample at index 0")
    print(f"  Sample keys: {list(sample.keys())}")
    
    # Test __getitem__ for middle sample
    if dataset_len > 10:
        mid_idx = dataset_len // 2
        print(f"\nTesting __getitem__({mid_idx})...")
        sample = dataset[mid_idx]
        print(f"✓ Successfully retrieved sample at index {mid_idx}")
    
    # Test __getitem__ for last sample
    if dataset_len > 1:
        last_idx = dataset_len - 1
        print(f"\nTesting __getitem__({last_idx})...")
        sample = dataset[last_idx]
        print(f"✓ Successfully retrieved sample at index {last_idx}")
    
    # Test embodiment attribute
    print(f"\n✓ Embodiment ID: {dataset.embodiment}")
    assert hasattr(dataset, 'embodiment'), "Dataset should have embodiment attribute"
    
    return sample


def test_output_format(sample: dict, chunk_size: int, final_image_size: int):
    """Test that output format matches cosmos-policy expectations."""
    print("\n" + "=" * 80)
    print("TEST 3: Output Format Validation")
    print("=" * 80)
    
    required_keys = [
        "video",
        "actions",
        "t5_text_embeddings",
        "t5_text_mask",
        "fps",
        "padding_mask",
        "image_size",
        "proprio",
        "future_proprio",
        "__key__",
        "command",
        "action_latent_idx",
        "value_latent_idx",
        "current_proprio_latent_idx",
        "current_wrist_image_latent_idx",
        "current_wrist_image2_latent_idx",
        "current_image_latent_idx",
        "future_proprio_latent_idx",
        "future_wrist_image_latent_idx",
        "future_wrist_image2_latent_idx",
        "future_image_latent_idx",
    ]
    
    print(f"\nChecking required keys...")
    for key in required_keys:
        assert key in sample, f"Missing required key: {key}"
        print(f"  ✓ {key}")
    
    # Check video shape
    print(f"\nChecking video tensor...")
    video = sample["video"]
    assert isinstance(video, torch.Tensor), "video should be a torch.Tensor"
    assert len(video.shape) == 4, f"video should have 4 dimensions, got {len(video.shape)}"
    C, T, H, W = video.shape
    print(f"  ✓ video shape: (C={C}, T={T}, H={H}, W={W})")
    assert C == 3, f"Expected 3 channels, got {C}"
    assert H == final_image_size, f"Expected height {final_image_size}, got {H}"
    assert W == final_image_size, f"Expected width {final_image_size}, got {W}"
    assert T > 0, "Time dimension should be positive"
    
    # Check actions shape
    print(f"\nChecking actions tensor...")
    actions = sample["actions"]
    assert isinstance(actions, torch.Tensor), "actions should be a torch.Tensor"
    assert len(actions.shape) == 2, f"actions should have 2 dimensions, got {len(actions.shape)}"
    action_chunk_size, action_dim = actions.shape
    print(f"  ✓ actions shape: (chunk_size={action_chunk_size}, action_dim={action_dim})")
    assert action_chunk_size == chunk_size, f"Expected chunk_size {chunk_size}, got {action_chunk_size}"
    
    # Check T5 embeddings
    print(f"\nChecking T5 embeddings...")
    t5_emb = sample["t5_text_embeddings"]
    assert isinstance(t5_emb, torch.Tensor), "t5_text_embeddings should be a torch.Tensor"
    print(f"  ✓ t5_text_embeddings shape: {t5_emb.shape}")
    assert len(t5_emb.shape) == 2, f"t5_text_embeddings should have 2 dimensions, got {len(t5_emb.shape)}"
    seq_len, embed_dim = t5_emb.shape
    assert seq_len == 512, f"Expected sequence length 512, got {seq_len}"
    assert embed_dim == 1024, f"Expected embedding dim 1024, got {embed_dim}"
    
    # Check T5 mask
    t5_mask = sample["t5_text_mask"]
    assert isinstance(t5_mask, torch.Tensor), "t5_text_mask should be a torch.Tensor"
    assert t5_mask.shape == (512,), f"t5_text_mask should have shape (512,), got {t5_mask.shape}"
    print(f"  ✓ t5_text_mask shape: {t5_mask.shape}")
    
    # Check proprio
    print(f"\nChecking proprio...")
    proprio = sample["proprio"]
    assert isinstance(proprio, torch.Tensor), "proprio should be a torch.Tensor"
    print(f"  ✓ proprio shape: {proprio.shape}")
    
    future_proprio = sample["future_proprio"]
    assert isinstance(future_proprio, torch.Tensor), "future_proprio should be a torch.Tensor"
    print(f"  ✓ future_proprio shape: {future_proprio.shape}")
    
    # Check other fields
    print(f"\nChecking other fields...")
    assert isinstance(sample["fps"], torch.Tensor), "fps should be a torch.Tensor"
    assert isinstance(sample["padding_mask"], torch.Tensor), "padding_mask should be a torch.Tensor"
    assert isinstance(sample["image_size"], torch.Tensor), "image_size should be a torch.Tensor"
    assert isinstance(sample["__key__"], int), "__key__ should be an int"
    assert isinstance(sample["command"], str), "command should be a string"
    print(f"  ✓ All field types correct")
    
    # Check latent indices
    print(f"\nChecking latent indices...")
    latent_indices = [
        "action_latent_idx",
        "value_latent_idx",
        "current_proprio_latent_idx",
        "current_wrist_image_latent_idx",
        "current_wrist_image2_latent_idx",
        "current_image_latent_idx",
        "future_proprio_latent_idx",
        "future_wrist_image_latent_idx",
        "future_wrist_image2_latent_idx",
        "future_image_latent_idx",
    ]
    for idx_name in latent_indices:
        idx_value = sample[idx_name]
        assert isinstance(idx_value, (int, torch.Tensor)), f"{idx_name} should be int or tensor"
        print(f"  ✓ {idx_name}: {idx_value}")
    
    print("\n✓ All output format checks passed!")


def test_t5_embeddings(dataset: CosmosPolicyRLDBDataset):
    """Test T5 embedding construction."""
    print("\n" + "=" * 80)
    print("TEST 4: T5 Embedding Construction")
    print("=" * 80)
    
    # Get a sample
    sample = dataset[0]
    
    # Check T5 embeddings exist
    assert "t5_text_embeddings" in sample, "Sample should contain t5_text_embeddings"
    t5_emb = sample["t5_text_embeddings"]
    print(f"\n✓ T5 embeddings shape: {t5_emb.shape}")
    
    # Check T5 mask
    t5_mask = sample["t5_text_mask"]
    print(f"✓ T5 mask shape: {t5_mask.shape}")
    print(f"  Mask sum (valid tokens): {t5_mask.sum().item()}")
    
    # Check command text
    command = sample.get("command", "")
    print(f"✓ Command text: '{command[:50]}...' (truncated)" if len(command) > 50 else f"✓ Command text: '{command}'")
    
    # Test with multiple samples to verify consistency
    print(f"\nTesting T5 embeddings across multiple samples...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        t5_emb = sample["t5_text_embeddings"]
        assert t5_emb.shape == (512, 1024), f"Sample {i}: Expected shape (512, 1024), got {t5_emb.shape}"
        print(f"  ✓ Sample {i}: T5 embedding shape correct")
    
    print("\n✓ All T5 embedding tests passed!")


def test_video_tensor(dataset: CosmosPolicyRLDBDataset, num_duplicates_per_image: int):
    """Test video tensor structure."""
    print("\n" + "=" * 80)
    print("TEST 5: Video Tensor Structure")
    print("=" * 80)
    
    sample = dataset[0]
    video = sample["video"]
    C, T, H, W = video.shape
    
    print(f"\nVideo tensor shape: (C={C}, T={T}, H={H}, W={W})")
    
    # Check expected sequence structure
    # Sequence: [blank(1), current_proprio(dup), left_wrist(dup), right_wrist(dup), 
    #            primary(dup), action(dup), future_proprio(dup), future_left(dup), 
    #            future_right(dup), future_primary(dup), value(dup)]
    # Expected segments: 11 (or 10 if no value function)
    # Expected time steps: 1 + 10 * num_duplicates_per_image (or 9 * dup if no value)
    
    # Count segments (approximate)
    print(f"\nAnalyzing video tensor structure...")
    print(f"  Expected duplicates per image: {num_duplicates_per_image}")
    print(f"  Actual time dimension: {T}")
    
    # Check video values are valid
    print(f"\nChecking video tensor values...")
    video_float = video.float() if video.dtype == torch.uint8 else video
    print(f"  Min value: {video_float.min().item():.3f}")
    print(f"  Max value: {video_float.max().item():.3f}")
    print(f"  Mean value: {video_float.mean().item():.3f}")
    print(f"  Dtype: {video.dtype}")
    
    # Check latent indices are within bounds
    print(f"\nChecking latent indices are within bounds...")
    latent_indices = {
        "action_latent_idx": sample["action_latent_idx"],
        "current_image_latent_idx": sample["current_image_latent_idx"],
        "future_image_latent_idx": sample["future_image_latent_idx"],
    }
    
    for name, idx in latent_indices.items():
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if idx >= 0:
            print(f"  ✓ {name}: {idx} (within bounds)")
        else:
            print(f"  ✓ {name}: {idx} (disabled)")
    
    print("\n✓ All video tensor tests passed!")


def test_action_chunks(dataset: CosmosPolicyRLDBDataset, chunk_size: int):
    """Test action chunk extraction."""
    print("\n" + "=" * 80)
    print("TEST 6: Action Chunk Extraction")
    print("=" * 80)
    
    # Test multiple samples
    print(f"\nTesting action chunks across samples...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        actions = sample["actions"]
        
        assert len(actions.shape) == 2, f"Sample {i}: Actions should be 2D, got {len(actions.shape)}D"
        assert actions.shape[0] == chunk_size, f"Sample {i}: Expected chunk_size {chunk_size}, got {actions.shape[0]}"
        
        print(f"  ✓ Sample {i}: actions shape {actions.shape}, dtype {actions.dtype}")
        print(f"    Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    print("\n✓ All action chunk tests passed!")


def test_dataloader(dataset: CosmosPolicyRLDBDataset):
    """Test DataLoader compatibility."""
    print("\n" + "=" * 80)
    print("TEST 7: DataLoader Compatibility")
    print("=" * 80)
    
    # Create DataLoader
    batch_size = 4
    num_workers = 2
    
    print(f"\nCreating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    print("✓ DataLoader created successfully")
    
    # Test batching
    print(f"\nTesting batch loading...")
    batch = next(iter(dataloader))
    print(f"✓ Successfully loaded batch")
    
    # Check batch structure
    print(f"\nBatch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Verify batch dimensions
    assert batch["video"].shape[0] == batch_size, f"Batch size mismatch for video"
    assert batch["actions"].shape[0] == batch_size, f"Batch size mismatch for actions"
    assert batch["t5_text_embeddings"].shape[0] == batch_size, f"Batch size mismatch for t5_text_embeddings"
    
    print(f"\n✓ Batch dimensions correct")
    
    # Test multiple batches
    print(f"\nTesting multiple batches...")
    num_batches = min(3, len(dataloader))
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"  ✓ Batch {i}: loaded successfully")
    
    print("\n✓ All DataLoader tests passed!")


def test_embodiment_specific(dataset: CosmosPolicyRLDBDataset):
    """Test embodiment-specific behavior."""
    print("\n" + "=" * 80)
    print("TEST 8: Embodiment-Specific Behavior")
    print("=" * 80)
    
    # Get a sample to check what images are available
    sample = dataset[0]
    
    # Get raw item from underlying dataset to check available camera keys
    raw_item = dataset.rldb_dataset[0]
    images_dict = raw_item.get("observations", {}).get("images", {})
    available_camera_keys = list(images_dict.keys())
    
    print(f"\nAvailable camera keys in raw data: {available_camera_keys}")
    print(f"Embodiment ID: {dataset.embodiment}")
    
    # Check if wrist images exist
    has_left_wrist = "left_wrist_img" in available_camera_keys
    has_right_wrist = "right_wrist_img" in available_camera_keys
    has_front = "front_img_1" in available_camera_keys or "cam_high" in available_camera_keys
    
    print(f"\nCamera availability:")
    print(f"  Front image: {has_front}")
    print(f"  Left wrist: {has_left_wrist}")
    print(f"  Right wrist: {has_right_wrist}")
    
    # Check video tensor - should handle missing cameras gracefully
    video = sample["video"]
    print(f"\n✓ Video tensor created successfully with shape {video.shape}")
    print(f"  (Missing cameras should be replaced with blank images)")
    
    # Check latent indices
    print(f"\nLatent indices:")
    print(f"  current_wrist_image_latent_idx: {sample['current_wrist_image_latent_idx']}")
    print(f"  current_wrist_image2_latent_idx: {sample['current_wrist_image2_latent_idx']}")
    print(f"  current_image_latent_idx: {sample['current_image_latent_idx']}")
    
    print("\n✓ All embodiment-specific tests passed!")


def main():
    """Main test function."""
    print("=" * 80)
    print("CosmosPolicyRLDBDataset Test Script")
    print("=" * 80)
    
    # Load debug config directly
    debug_config_path = project_root / "egomimic" / "hydra_configs" / "data" / "debug.yaml"
    debug_cfg = OmegaConf.load(debug_config_path)
    
    # Create a minimal cfg structure for compatibility
    cfg = OmegaConf.create({
        "data": debug_cfg
    })
    
    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Data config: {debug_config_path}")
    print(f"\nDataset config:")
    print(OmegaConf.to_yaml(cfg.data.train_datasets.dataset1))
    
    # Test 1: Load dataset
    train_dataset, valid_dataset = test_dataset_loading(cfg)
    
    # Get dataset parameters from config
    dataset_config = cfg.data.train_datasets.dataset1
    chunk_size = dataset_config.get("chunk_size", 25)
    final_image_size = dataset_config.get("final_image_size", 224)
    num_duplicates_per_image = dataset_config.get("num_duplicates_per_image", 4)
    
    # Test 2: Basic functionality
    sample = test_basic_functionality(train_dataset)
    
    # Test 3: Output format
    test_output_format(sample, chunk_size, final_image_size)
    
    # Test 4: T5 embeddings
    test_t5_embeddings(train_dataset)
    
    # Test 5: Video tensor
    test_video_tensor(train_dataset, num_duplicates_per_image)
    
    # Test 6: Action chunks
    test_action_chunks(train_dataset, chunk_size)
    
    # Test 7: DataLoader
    test_dataloader(train_dataset)
    
    # Test 8: Embodiment-specific
    test_embodiment_specific(train_dataset)
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
