#!/usr/bin/env python3
"""Analyze statistics for scene_diversity_16 config files."""

import yaml
import re
from collections import defaultdict

# Config files to analyze
configs = [
    ('scene_diversity_16_60', 60),
    ('scene_diversity_16_30', 30),
    ('scene_diversity_16_15', 15),
    ('scene_diversity_16_7_5', 7.5),
    ('scene_diversity_16_3_75', 3.75),
]

print("=" * 80)
print("Statistics for Scene Diversity 16 Config Files")
print("=" * 80)
print()

for config_name, minutes in configs:
    filename = f'egomimic/hydra_configs/data/{config_name}.yaml'
    
    try:
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        datasets = config['train_datasets']['dataset1']['datasets']
        
        # Count entries per scene
        scene_counts = defaultdict(int)
        scene_operators = defaultdict(set)
        
        for key, value in datasets.items():
            if key.startswith('#'):
                continue
            match = re.match(r'op(\d+)_scene(\d+)_ep(\d+)', key)
            if match:
                op = int(match.group(1))
                scene = int(match.group(2))
                scene_counts[scene] += 1
                scene_operators[scene].add(op)
        
        # Sort scenes
        sorted_scenes = sorted(scene_counts.keys())
        
        total_entries = sum(scene_counts.values())
        total_scenes = len(sorted_scenes)
        
        print(f"Config: {config_name} ({minutes} minutes per scene)")
        print("-" * 80)
        print(f"Total entries: {total_entries}")
        print(f"Total scenes: {total_scenes}")
        print()
        print("Entries per scene:")
        print(f"{'Scene':<8} {'Entries':<10} {'Operators':<12} {'Avg Episodes/Op':<15}")
        print("-" * 80)
        
        for scene in sorted_scenes:
            entries = scene_counts[scene]
            operators = len(scene_operators[scene])
            avg_episodes = entries / operators if operators > 0 else 0
            print(f"{scene:<8} {entries:<10} {operators:<12} {avg_episodes:<15.2f}")
        
        print()
        print()
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}")
        print()
    except Exception as e:
        print(f"ERROR processing {filename}: {e}")
        print()

print("=" * 80)
