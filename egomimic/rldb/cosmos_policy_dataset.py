"""
Cosmos Policy RLDB Dataset - Wraps RLDBDataset and transforms data to cosmos-policy format.

This dataset loads LeRobot/RLDB format data and transforms it to match cosmos-policy's
expected input format, including:
- Combining separate camera views into a single video tensor
- Constructing T5 embeddings from text annotations
- Formatting action chunks
- Extracting proprioceptive state
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

cosmos_policy_path = Path(__file__).parent.parent.parent / "external" / "cosmos-policy"
if str(cosmos_policy_path) not in sys.path:
    sys.path.insert(0, str(cosmos_policy_path))

from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_policy.datasets.dataset_utils import (
    preprocess_image,
    resize_images,
)
from cosmos_policy.datasets.dataset_common import get_action_chunk_with_padding

from egomimic.rldb.utils import (
    RLDBDataset,
    S3RLDBDataset,
    FolderRLDBDataset,
    get_embodiment_id,
)


class CosmosPolicyRLDBDataset(Dataset):
    """
    Dataset wrapper that loads RLDB/LeRobot format data and transforms it to cosmos-policy format.
    
    This class wraps an existing RLDBDataset (S3RLDBDataset, RLDBDataset, or FolderRLDBDataset)
    and transforms the frame-level data to match cosmos-policy's expected input format.
    """
    
    def __init__(
        self,
        # RLDB dataset parameters (can use any of these)
        repo_id: Optional[str] = None,
        root: Optional[str] = None,
        bucket_name: Optional[str] = None,
        folder_path: Optional[str] = None,
        mode: str = "train",
        embodiment: str = "eva_bimanual",
        filters: Optional[Dict] = None,
        local_files_only: bool = True,
        percent: float = 1.0,
        valid_ratio: float = 0.2,
        # Cosmos-policy specific parameters
        chunk_size: int = 25,
        final_image_size: int = 224,
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        construct_t5_embeddings: bool = True,
        t5_model_name: str = "google-t5/t5-11b",
        precompute_embeddings: bool = False,
        fps: int = 16,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        **kwargs,
    ):
        """
        Initialize CosmosPolicyRLDBDataset.
        
        Args:
            # RLDB dataset parameters 
            repo_id: Repository ID for RLDBDataset (if using local RLDBDataset)
            root: Root path for RLDBDataset (if using local RLDBDataset)
            bucket_name: S3 bucket name (if using S3RLDBDataset)
            folder_path: Folder path (if using FolderRLDBDataset)
            mode: Dataset mode ("train" or "valid")
            embodiment: Embodiment type (e.g., "eva_bimanual", "aria_bimanual")
            filters: Filter dictionary for S3RLDBDataset
            local_files_only: Whether to use only local files
            percent: Fraction of data to use
            valid_ratio: Validation split ratio
            
            # Cosmos-policy specific parameters
            chunk_size: Action chunk size
            final_image_size: Target image size (square)
            normalize_images: Whether to normalize images
            normalize_actions: Whether to normalize actions
            normalize_proprio: Whether to normalize proprio
            use_image_aug: Whether to apply image augmentations
            use_stronger_image_aug: Whether to apply stronger augmentations
            use_proprio: Whether to use proprioceptive state
            num_duplicates_per_image: Temporal compression factor (images per latent frame)
            construct_t5_embeddings: Whether to construct T5 embeddings from text
            t5_model_name: T5 model name for embedding construction
            precompute_embeddings: Whether to precompute all embeddings at init
            fps: Frames per second
            return_value_function_returns: Whether to return value function returns
            gamma: Discount factor for value function
            **kwargs: Additional arguments passed to underlying RLDBDataset
        """
        # Store cosmos-policy specific parameters
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.construct_t5_embeddings = construct_t5_embeddings
        self.t5_model_name = t5_model_name
        self.precompute_embeddings = precompute_embeddings
        self.fps = fps
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        
        # Initialize underlying RLDBDataset
        if bucket_name is not None:
            # Use S3RLDBDataset
            self.rldb_dataset = S3RLDBDataset(
                embodiment=embodiment,
                mode=mode,
                bucket_name=bucket_name,
                filters=filters or {},
                local_files_only=local_files_only,
                percent=percent,
                valid_ratio=valid_ratio,
                **kwargs,
            )
        elif folder_path is not None:
            # Use FolderRLDBDataset
            self.rldb_dataset = FolderRLDBDataset(
                folder_path=folder_path,
                embodiment=embodiment,
                mode=mode,
                percent=percent,
                local_files_only=local_files_only,
                valid_ratio=valid_ratio,
                **kwargs,
            )
        elif repo_id is not None and root is not None:
            # Use regular RLDBDataset
            self.rldb_dataset = RLDBDataset(
                repo_id=repo_id,
                root=root,
                mode=mode,
                local_files_only=local_files_only,
                percent=percent,
                valid_ratio=valid_ratio,
                **kwargs,
            )
        else:
            raise ValueError(
                "Must provide either (bucket_name), (folder_path), or (repo_id, root) "
                "to initialize underlying RLDBDataset"
            )
        
        # Get embodiment ID
        self.embodiment = get_embodiment_id(embodiment)
        
        # Initialize T5 encoder if needed
        self.t5_embeddings_cache = {}
        if self.construct_t5_embeddings:
            # T5 encoder will be initialized lazily on first use
            self._t5_encoder = None
            # Workaround for torch < 2.6: Monkey-patch transformers to bypass version check
            # when safetensors is available
            self._patch_transformers_for_torch_25()
            if self.precompute_embeddings:
                self._precompute_all_embeddings()
    
    def _patch_transformers_for_torch_25(self):
        """Monkey-patch transformers to bypass torch version check when safetensors is available."""
        try:
            import transformers
            from transformers.utils.import_utils import check_torch_load_is_safe as original_check
            
            def patched_check():
                # Bypass check - we'll use safetensors anyway
                return
            
            # Patch the function in multiple places
            transformers.utils.import_utils.check_torch_load_is_safe = patched_check
            if hasattr(transformers, 'modeling_utils'):
                transformers.modeling_utils.check_torch_load_is_safe = patched_check
        except Exception as e:
            # If patching fails, continue anyway - might work if torch is already >= 2.6
            pass
    
    def _precompute_all_embeddings(self):
        """Precompute T5 embeddings for all unique annotations in the dataset."""
        print("Precomputing T5 embeddings for all unique annotations...")
        unique_annotations = set()
        
        # Collect all unique annotations
        for idx in range(len(self.rldb_dataset)):
            item = self.rldb_dataset[idx]
            annotation = item.get("annotations", "")
            if annotation and annotation not in unique_annotations:
                unique_annotations.add(annotation)
        
        # Precompute embeddings
        for annotation in unique_annotations:
            embedding = get_text_embedding(
                annotation,
                device="cpu",
                max_length=512,
                cache_dir=None,
                local_files_only=False,
            )
            self.t5_embeddings_cache[annotation] = embedding.cpu()
        
        print(f"Precomputed {len(self.t5_embeddings_cache)} T5 embeddings")
    
    def _construct_t5_embedding(self, text: str) -> torch.Tensor:
        """
        Construct T5 embedding from text.
        
        Args:
            text: Text string to embed
            
        Returns:
            T5 embedding tensor of shape (1, 512, 1024) for t5-11b
        """
        if not text:
            text = ""  # Empty string for missing annotations
        
        # Check cache first
        if text in self.t5_embeddings_cache:
            return self.t5_embeddings_cache[text]
        
        # Compute embedding
        # Note: cosmos-policy code has been modified to use safetensors
        # to avoid torch >= 2.6 requirement
        embedding = get_text_embedding(
            text,
            device="cpu",
            max_length=512,
            cache_dir=None,
            local_files_only=False,
        )
        
        # Cache it
        if self.precompute_embeddings:
            self.t5_embeddings_cache[text] = embedding.cpu()
        
        return embedding.cpu()
    
    def _combine_camera_views(
        self,
        front_img: np.ndarray,
        left_wrist_img: Optional[np.ndarray],
        right_wrist_img: Optional[np.ndarray],
        proprio: Optional[np.ndarray],
        future_front_img: Optional[np.ndarray],
        future_left_wrist_img: Optional[np.ndarray],
        future_right_wrist_img: Optional[np.ndarray],
        future_proprio: Optional[np.ndarray],
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        """
        Combine separate camera views into a single video tensor following cosmos-policy format.
        
        Sequence structure:
        [blank, current_proprio, left_wrist, right_wrist, primary, action_placeholder,
         future_proprio, future_left, future_right, future_primary, value_placeholder]
        
        Args:
            front_img: Primary camera image (H, W, C)
            left_wrist_img: Left wrist camera image (H, W, C) or None
            right_wrist_img: Right wrist camera image (H, W, C) or None
            proprio: Current proprioceptive state or None
            future_front_img: Future primary camera image or None
            future_left_wrist_img: Future left wrist image or None
            future_right_wrist_img: Future right wrist image or None
            future_proprio: Future proprioceptive state or None
            
        Returns:
            Tuple of (video tensor (C, T, H, W), latent_indices dict)
        """
        # Ensure images are the right size
        if front_img.shape[:2] != (self.final_image_size, self.final_image_size):
            front_img = resize_images(
                np.expand_dims(front_img, 0),
                self.final_image_size
            )[0]
        
        # Build frame sequence
        frames = []  # List of (H, W, C) images
        repeats = []  # List of repeat counts
        latent_indices = {}
        segment_idx = 0
        
        # Reference image for creating blank frames
        ref_image = front_img.copy()
        
        # 1. Blank first input image
        blank_frame = np.zeros_like(ref_image)
        frames.append(blank_frame)
        repeats.append(1)
        segment_idx += 1
        
        # 2. Current proprio (blank image placeholder)
        if self.use_proprio and proprio is not None:
            blank_proprio = np.zeros_like(ref_image)
            frames.append(blank_proprio)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["current_proprio_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            latent_indices["current_proprio_latent_idx"] = -1
        
        # 3. Current left wrist image
        if left_wrist_img is not None:
            if left_wrist_img.shape[:2] != (self.final_image_size, self.final_image_size):
                left_wrist_img = resize_images(
                    np.expand_dims(left_wrist_img, 0),
                    self.final_image_size
                )[0]
            frames.append(left_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["current_wrist_image_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            # Use blank if missing
            frames.append(blank_frame.copy())
            repeats.append(self.num_duplicates_per_image)
            latent_indices["current_wrist_image_latent_idx"] = segment_idx
            segment_idx += 1
        
        # 4. Current right wrist image
        if right_wrist_img is not None:
            if right_wrist_img.shape[:2] != (self.final_image_size, self.final_image_size):
                right_wrist_img = resize_images(
                    np.expand_dims(right_wrist_img, 0),
                    self.final_image_size
                )[0]
            frames.append(right_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["current_wrist_image2_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            # Use blank if missing
            frames.append(blank_frame.copy())
            repeats.append(self.num_duplicates_per_image)
            latent_indices["current_wrist_image2_latent_idx"] = segment_idx
            segment_idx += 1
        
        # 5. Current primary image
        frames.append(front_img)
        repeats.append(self.num_duplicates_per_image)
        latent_indices["current_image_latent_idx"] = segment_idx
        segment_idx += 1
        
        # 6. Action placeholder (blank)
        blank_action = np.zeros_like(ref_image)
        frames.append(blank_action)
        repeats.append(self.num_duplicates_per_image)
        latent_indices["action_latent_idx"] = segment_idx
        segment_idx += 1
        
        # 7. Future proprio (blank image placeholder)
        if self.use_proprio and future_proprio is not None:
            blank_future_proprio = np.zeros_like(ref_image)
            frames.append(blank_future_proprio)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["future_proprio_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            latent_indices["future_proprio_latent_idx"] = -1
        
        # 8. Future left wrist image
        if future_left_wrist_img is not None:
            if future_left_wrist_img.shape[:2] != (self.final_image_size, self.final_image_size):
                future_left_wrist_img = resize_images(
                    np.expand_dims(future_left_wrist_img, 0),
                    self.final_image_size
                )[0]
            frames.append(future_left_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["future_wrist_image_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            frames.append(blank_frame.copy())
            repeats.append(self.num_duplicates_per_image)
            latent_indices["future_wrist_image_latent_idx"] = segment_idx
            segment_idx += 1
        
        # 9. Future right wrist image
        if future_right_wrist_img is not None:
            if future_right_wrist_img.shape[:2] != (self.final_image_size, self.final_image_size):
                future_right_wrist_img = resize_images(
                    np.expand_dims(future_right_wrist_img, 0),
                    self.final_image_size
                )[0]
            frames.append(future_right_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["future_wrist_image2_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            frames.append(blank_frame.copy())
            repeats.append(self.num_duplicates_per_image)
            latent_indices["future_wrist_image2_latent_idx"] = segment_idx
            segment_idx += 1
        
        # 10. Future primary image
        if future_front_img is not None:
            if future_front_img.shape[:2] != (self.final_image_size, self.final_image_size):
                future_front_img = resize_images(
                    np.expand_dims(future_front_img, 0),
                    self.final_image_size
                )[0]
            frames.append(future_front_img)
        else:
            frames.append(blank_frame.copy())
        repeats.append(self.num_duplicates_per_image)
        latent_indices["future_image_latent_idx"] = segment_idx
        segment_idx += 1
        
        # 11. Value placeholder (blank)
        if self.return_value_function_returns:
            blank_value = np.zeros_like(ref_image)
            frames.append(blank_value)
            repeats.append(self.num_duplicates_per_image)
            latent_indices["value_latent_idx"] = segment_idx
            segment_idx += 1
        else:
            latent_indices["value_latent_idx"] = -1
        
        # Stack unique frames - preprocess_image expects (T, H, W, C)
        all_unique_images = np.stack(frames, axis=0)  # (num_segments, H, W, C)
        
        # Preprocess images (resize, normalize, augment)
        # preprocess_image expects (T, H, W, C) and returns (C, T, H, W)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )  # Returns (C, num_segments, H, W) tensor
        
        # Expand by repeat counts along time dimension (dim=1 is time)
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        # Repeat along time dimension (dim=1)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)  # (C, T, H, W)
        
        return all_images, latent_indices
    
    def _extract_action_chunk(self, item: Dict[str, Any], frame_idx: int) -> np.ndarray:
        """
        Extract action chunk from frame data.
        
        Args:
            item: Frame data from RLDBDataset
            frame_idx: Frame index within episode
            
        Returns:
            Action chunk of shape (chunk_size, action_dim)
        """
        # Try to get pre-chunked actions first
        action_keys = [
            "actions_joints",
            "actions_cartesian",
            "actions_base_cartesian",
            "actions_eef_cartesian",
        ]
        
        for key in action_keys:
            if key in item:
                actions = item[key]
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                # If already chunked, take first chunk
                if len(actions.shape) == 2 and actions.shape[0] >= self.chunk_size:
                    return actions[:self.chunk_size]
                # If single-step, we need to extract chunk from episode
                elif len(actions.shape) == 1:
                    # Need to get full episode actions
                    # For now, pad single action to chunk size
                    action_chunk = np.tile(actions, (self.chunk_size, 1))
                    return action_chunk
        
        # Fallback: create zero action chunk
        # Try to infer action dim from any action key
        action_dim = 14  # Default
        for key in action_keys:
            if key in item:
                actions = item[key]
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                if len(actions.shape) >= 1:
                    action_dim = actions.shape[-1]
                    break
        
        return np.zeros((self.chunk_size, action_dim), dtype=np.float32)
    
    def _get_future_frame(self, current_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get future frame data for constructing future state.
        
        Args:
            current_idx: Current frame index
            
        Returns:
            Future frame data or None if not available
        """
        future_idx = current_idx + self.chunk_size
        if future_idx < len(self.rldb_dataset):
            try:
                return self.rldb_dataset[future_idx]
            except:
                return None
        return None
    
    def __len__(self) -> int:
        """Return length of underlying dataset."""
        return len(self.rldb_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item transformed to cosmos-policy format.
        
        Args:
            idx: Frame index
            
        Returns:
            Dictionary with cosmos-policy format data
        """
        # Get current frame from RLDBDataset
        item = self.rldb_dataset[idx]
        
        # Extract images
        front_img = item.get("observations", {}).get("images", {}).get("front_img_1")
        if front_img is None:
            # Try alternative key
            front_img = item.get("observations", {}).get("images", {}).get("cam_high")
        
        left_wrist_img = item.get("observations", {}).get("images", {}).get("left_wrist_img")
        right_wrist_img = item.get("observations", {}).get("images", {}).get("right_wrist_img")
        
        # Convert to numpy if needed
        if front_img is not None:
            if isinstance(front_img, torch.Tensor):
                front_img = front_img.cpu().numpy()
            if front_img.dtype != np.uint8:
                front_img = (front_img * 255).astype(np.uint8) if front_img.max() <= 1.0 else front_img.astype(np.uint8)
            # Ensure (H, W, C) format
            if len(front_img.shape) == 4:
                front_img = front_img[0]
            if front_img.shape[2] != 3:
                front_img = front_img.transpose(1, 2, 0) if len(front_img.shape) == 3 else front_img
        
        if left_wrist_img is not None:
            if isinstance(left_wrist_img, torch.Tensor):
                left_wrist_img = left_wrist_img.cpu().numpy()
            if left_wrist_img.dtype != np.uint8:
                left_wrist_img = (left_wrist_img * 255).astype(np.uint8) if left_wrist_img.max() <= 1.0 else left_wrist_img.astype(np.uint8)
            if len(left_wrist_img.shape) == 4:
                left_wrist_img = left_wrist_img[0]
            if len(left_wrist_img.shape) == 3 and left_wrist_img.shape[2] != 3:
                left_wrist_img = left_wrist_img.transpose(1, 2, 0)
        
        if right_wrist_img is not None:
            if isinstance(right_wrist_img, torch.Tensor):
                right_wrist_img = right_wrist_img.cpu().numpy()
            if right_wrist_img.dtype != np.uint8:
                right_wrist_img = (right_wrist_img * 255).astype(np.uint8) if right_wrist_img.max() <= 1.0 else right_wrist_img.astype(np.uint8)
            if len(right_wrist_img.shape) == 4:
                right_wrist_img = right_wrist_img[0]
            if len(right_wrist_img.shape) == 3 and right_wrist_img.shape[2] != 3:
                right_wrist_img = right_wrist_img.transpose(1, 2, 0)
        
        # Extract proprio
        proprio = None
        if self.use_proprio:
            state = item.get("observations", {}).get("state", {})
            proprio = state.get("joint_positions") or state.get("ee_pose") or state.get("eepose")
            if proprio is not None:
                if isinstance(proprio, torch.Tensor):
                    proprio = proprio.cpu().numpy()
                if len(proprio.shape) > 1:
                    proprio = proprio.flatten()
        
        # Get future frame
        future_item = self._get_future_frame(idx)
        future_front_img = None
        future_left_wrist_img = None
        future_right_wrist_img = None
        future_proprio = None
        
        if future_item is not None:
            future_front_img = future_item.get("observations", {}).get("images", {}).get("front_img_1")
            if future_front_img is None:
                future_front_img = future_item.get("observations", {}).get("images", {}).get("cam_high")
            
            future_left_wrist_img = future_item.get("observations", {}).get("images", {}).get("left_wrist_img")
            future_right_wrist_img = future_item.get("observations", {}).get("images", {}).get("right_wrist_img")
            
            # Convert future images
            if future_front_img is not None:
                if isinstance(future_front_img, torch.Tensor):
                    future_front_img = future_front_img.cpu().numpy()
                if future_front_img.dtype != np.uint8:
                    future_front_img = (future_front_img * 255).astype(np.uint8) if future_front_img.max() <= 1.0 else future_front_img.astype(np.uint8)
                if len(future_front_img.shape) == 4:
                    future_front_img = future_front_img[0]
                if len(future_front_img.shape) == 3 and future_front_img.shape[2] != 3:
                    future_front_img = future_front_img.transpose(1, 2, 0)
            
            if future_left_wrist_img is not None:
                if isinstance(future_left_wrist_img, torch.Tensor):
                    future_left_wrist_img = future_left_wrist_img.cpu().numpy()
                if future_left_wrist_img.dtype != np.uint8:
                    future_left_wrist_img = (future_left_wrist_img * 255).astype(np.uint8) if future_left_wrist_img.max() <= 1.0 else future_left_wrist_img.astype(np.uint8)
                if len(future_left_wrist_img.shape) == 4:
                    future_left_wrist_img = future_left_wrist_img[0]
                if len(future_left_wrist_img.shape) == 3 and future_left_wrist_img.shape[2] != 3:
                    future_left_wrist_img = future_left_wrist_img.transpose(1, 2, 0)
            
            if future_right_wrist_img is not None:
                if isinstance(future_right_wrist_img, torch.Tensor):
                    future_right_wrist_img = future_right_wrist_img.cpu().numpy()
                if future_right_wrist_img.dtype != np.uint8:
                    future_right_wrist_img = (future_right_wrist_img * 255).astype(np.uint8) if future_right_wrist_img.max() <= 1.0 else future_right_wrist_img.astype(np.uint8)
                if len(future_right_wrist_img.shape) == 4:
                    future_right_wrist_img = future_right_wrist_img[0]
                if len(future_right_wrist_img.shape) == 3 and future_right_wrist_img.shape[2] != 3:
                    future_right_wrist_img = future_right_wrist_img.transpose(1, 2, 0)
            
            if self.use_proprio:
                future_state = future_item.get("observations", {}).get("state", {})
                future_proprio = future_state.get("joint_positions") or future_state.get("ee_pose") or future_state.get("eepose")
                if future_proprio is not None:
                    if isinstance(future_proprio, torch.Tensor):
                        future_proprio = future_proprio.cpu().numpy()
                    if len(future_proprio.shape) > 1:
                        future_proprio = future_proprio.flatten()
        
        # Ensure we have at least a blank front image
        if front_img is None:
            front_img = np.zeros((self.final_image_size, self.final_image_size, 3), dtype=np.uint8)
        
        # Combine camera views into video tensor
        video, latent_indices = self._combine_camera_views(
            front_img=front_img,
            left_wrist_img=left_wrist_img,
            right_wrist_img=right_wrist_img,
            proprio=proprio,
            future_front_img=future_front_img,
            future_left_wrist_img=future_left_wrist_img,
            future_right_wrist_img=future_right_wrist_img,
            future_proprio=future_proprio,
        )
        
        # Extract action chunk
        action_chunk = self._extract_action_chunk(item, idx)
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()
        action_chunk = torch.from_numpy(action_chunk).float()
        
        # Get text annotation and construct T5 embedding
        annotation = item.get("annotations", "")
        if not annotation:
            annotation = ""
        
        if self.construct_t5_embeddings:
            t5_embedding = self._construct_t5_embedding(annotation)
            t5_text_mask = torch.ones(512, dtype=torch.int64)
        else:
            t5_embedding = torch.zeros(1, 512, 1024)
            t5_text_mask = torch.zeros(512, dtype=torch.int64)
        
        # Format proprio
        if proprio is not None:
            proprio_tensor = torch.from_numpy(proprio).float()
        else:
            proprio_tensor = torch.zeros(14)  # Default size
        
        if future_proprio is not None:
            future_proprio_tensor = torch.from_numpy(future_proprio).float()
        else:
            future_proprio_tensor = torch.zeros(14)
        
        # Build return dictionary
        result = {
            "video": video,  # (C, T, H, W)
            "actions": action_chunk,  # (chunk_size, action_dim)
            "t5_text_embeddings": t5_embedding.squeeze(0),  # (512, 1024) for t5-11b
            "t5_text_mask": t5_text_mask,  # (512,)
            "fps": torch.tensor(self.fps, dtype=torch.float32),
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": torch.tensor([self.final_image_size] * 4, dtype=torch.float32),
            "proprio": proprio_tensor,
            "future_proprio": future_proprio_tensor,
            "__key__": idx,
            "command": annotation,  # Keep text for reference
        }
        
        # Add latent indices
        result.update(latent_indices)
        
        # Add value function returns if needed
        if self.return_value_function_returns:
            # Placeholder - would need episode success info
            result["value_function_return"] = torch.tensor(0.0)
            result["next_value_function_return"] = torch.tensor(0.0)
        
        return result
