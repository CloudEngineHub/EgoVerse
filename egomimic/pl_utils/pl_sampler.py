from torch.utils.data import Sampler
from torch.utils.data import DataLoader, random_split, default_collate
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning import LightningDataModule
from transformers import AutoTokenizer
from egomimic.utils.egomimicUtils import nds
import json
import os
import logging
from egomimic.rldb.utils import RLDBDataset
from termcolor import cprint
import torch

class EpisodeValBatchSampler(Sampler[list[int]]):
    """
    Validation: each step corresponds to one full episode (all its frames).
    Each rank gets a disjoint subset of episodes.
    """
    def __init__(self, dataset, rank=0, world_size=1):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

        episode_to_frames = {}
        for i in range(len(dataset)):
            ep = dataset[i]["episode_index"]
            episode_to_frames.setdefault(ep, []).append(i)

        self.episode_to_frames = episode_to_frames
        self.episodes = sorted(episode_to_frames.keys())
        self.num_episodes = len(self.episodes)

        self.episodes_rank = [ep for ep in self.episodes if (ep % world_size) == rank]

    def __iter__(self):
        for ep in self.episodes_rank:
            yield self.episode_to_frames[ep]

    def __len__(self):
        return len(self.episodes_rank)
