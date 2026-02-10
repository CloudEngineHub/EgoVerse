#!/usr/bin/env python3
"""
Generate 25 scene diversity config files for cup_on_saucer task.
Uses hierarchical subsampling with operator diversification.
Operator restrictions: op1-7 for scenes 1-16, op14 for scenes 9-16, op17 for scenes 1-8
"""

import yaml
import re
import json
from collections import defaultdict
from pathlib import Path

def parse_source_config(config_path):
    """Parse source config and extract episode data filtered by operator-scene rules."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    episode_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    train_datasets = config.get('train_datasets', {})
    for dataset_name, dataset_config in train_datasets.items():
        datasets = dataset_config.get('datasets', {})
        for key, value in datasets.items():
            match = re.match(r'op(\d+)_scene(\d+)_ep(\d+)', key)
            if match:
                op = int(match.group(1))
                scene = int(match.group(2))
                ep = int(match.group(3))
                episode_hash = value.get('filters', {}).get('episode_hash', '')
                
                # Filter by operator-scene rules (UPDATED: op1-7 instead of op1-11)
                if scene >= 1 and scene <= 16:
                    if op >= 1 and op <= 7:  # Operators 1-7 for scenes 1-16
                        episode_data[scene][op][ep] = episode_hash
                    elif op == 14 and scene >= 9:  # Operator 14 for scenes 9-16
                        episode_data[scene][op][ep] = episode_hash
                    elif op == 17 and scene <= 8:  # Operator 17 for scenes 1-8
                        episode_data[scene][op][ep] = episode_hash
    
    return episode_data

def select_episodes_with_diversity(episode_data, scene, num_episodes):
    """
    Select episodes for a scene prioritizing operator diversity.
    Returns list of (operator, episode_num, episode_hash) tuples.
    """
    if scene not in episode_data:
        return []
    
    # Get all available operators for this scene
    available_ops = {}
    for op, episodes in episode_data[scene].items():
        available_ops[op] = sorted(episodes.items())  # List of (ep_num, hash) tuples
    
    # Strategy: First take one episode from each operator, then fill remaining
    selected = []
    operators = sorted(available_ops.keys())
    
    # First pass: one episode from each operator
    for op in operators:
        if available_ops[op] and len(selected) < num_episodes:
            ep_num, ep_hash = available_ops[op][0]
            selected.append((op, ep_num, ep_hash))
            available_ops[op] = available_ops[op][1:]  # Remove used episode
    
    # Second pass: fill remaining by taking from operators with most available
    while len(selected) < num_episodes:
        # Find operators with remaining episodes
        ops_with_remaining = [(op, len(eps)) for op, eps in available_ops.items() if eps]
        if not ops_with_remaining:
            break
        
        # Sort by number of remaining episodes (descending)
        ops_with_remaining.sort(key=lambda x: x[1], reverse=True)
        
        # Take one from the operator with most remaining
        op = ops_with_remaining[0][0]
        if available_ops[op]:
            ep_num, ep_hash = available_ops[op][0]
            selected.append((op, ep_num, ep_hash))
            available_ops[op] = available_ops[op][1:]
    
    return selected[:num_episodes]

def subsample_episodes_by_diversity(base_episodes, target_num):
    """
    Subsample episodes maintaining operator diversity.
    Uses proportional distribution.
    """
    if target_num >= len(base_episodes):
        return base_episodes
    
    # Count episodes per operator
    op_counts = defaultdict(int)
    op_episodes = defaultdict(list)
    
    for op, ep_num, ep_hash in base_episodes:
        op_counts[op] += 1
        op_episodes[op].append((op, ep_num, ep_hash))
    
    # Calculate proportional distribution
    total = len(base_episodes)
    target_per_op = {}
    for op, count in op_counts.items():
        target_per_op[op] = max(1, int(count * target_num / total))
    
    # Adjust to ensure total equals target_num
    current_total = sum(target_per_op.values())
    if current_total < target_num:
        # Add to operators with most episodes
        ops_sorted = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        for op, _ in ops_sorted:
            if current_total >= target_num:
                break
            if target_per_op[op] < op_counts[op]:
                target_per_op[op] += 1
                current_total += 1
    elif current_total > target_num:
        # Remove from operators with least episodes
        ops_sorted = sorted(op_counts.items(), key=lambda x: x[1])
        for op, _ in ops_sorted:
            if current_total <= target_num:
                break
            if target_per_op[op] > 1:
                target_per_op[op] -= 1
                current_total -= 1
    
    # Select episodes proportionally
    selected = []
    for op, episodes in op_episodes.items():
        num_to_take = target_per_op[op]
        selected.extend(episodes[:num_to_take])
    
    return selected[:target_num]

def build_base_16_60(episode_data):
    """Build base config with 16 scenes × 16 episodes."""
    base_config = {}
    
    for scene in range(1, 17):
        episodes = select_episodes_with_diversity(episode_data, scene, 16)
        base_config[scene] = episodes
    
    return base_config

def generate_config_yaml(scene_episodes, output_path):
    """Generate YAML config file."""
    valid_datasets = {
        'dataset1': {
            '_target_': 'egomimic.rldb.utils.MultiRLDBDataset',
            'datasets': {
                'valid_ep1': {
                    '_target_': 'egomimic.rldb.utils.S3RLDBDataset',
                    'bucket_name': 'rldb',
                    'mode': 'total',
                    'embodiment': 'aria_bimanual',
                    'local_files_only': True,
                    'temp_root': '/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity_cup_on_saucer',
                    'filters': {
                        'episode_hash': '2025-11-14-17-24-52-962000'
                    }
                }
            },
            'embodiment': 'aria_bimanual'
        }
    }
    
    train_datasets = {
        'dataset1': {
            '_target_': 'egomimic.rldb.utils.MultiRLDBDataset',
            'datasets': {}
        }
    }
    
    # Add episodes to train_datasets
    for scene in sorted(scene_episodes.keys()):
        for op, ep_num, ep_hash in scene_episodes[scene]:
            key = f'op{op}_scene{scene}_ep{ep_num}'
            train_datasets['dataset1']['datasets'][key] = {
                '_target_': 'egomimic.rldb.utils.S3RLDBDataset',
                'bucket_name': 'rldb',
                'mode': 'total',
                'embodiment': 'aria_bimanual',
                'local_files_only': True,
                'temp_root': '/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity_cup_on_saucer',
                'filters': {
                    'episode_hash': ep_hash
                }
            }
    
    train_datasets['dataset1']['embodiment'] = 'aria_bimanual'
    
    config = {
        '_target_': 'egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper',
        'valid_datasets': valid_datasets,
        'train_datasets': train_datasets,
        'train_dataloader_params': {
            'dataset1': {
                'batch_size': 32,
                'num_workers': 10
            }
        },
        'valid_dataloader_params': {
            'dataset1': {
                'batch_size': 32,
                'num_workers': 10
            }
        }
    }
    
    # Write YAML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Generated: {output_path}")

def main():
    # Paths
    source_config = Path('egomimic/hydra_configs/data/diversity_cup_on_saucer_all.yaml')
    output_dir = Path('egomimic/hydra_configs/data/cup_saucer/scene_diversity')
    
    # Parse source config
    print("Parsing source config...")
    episode_data = parse_source_config(source_config)
    
    # Build base 16_60 config
    print("Building base 16_60 config...")
    base_16_60 = build_base_16_60(episode_data)
    
    # Generate all 25 configs
    print("Generating all config files...")
    
    # Scene counts and episode counts per scene
    scene_counts = [1, 2, 4, 8, 16]
    episode_counts = [1, 2, 4, 8, 16]  # 3.75, 7.5, 15, 30, 60 min per scene
    
    for num_scenes in scene_counts:
        for num_episodes_per_scene in episode_counts:
            # Get scenes for this config
            scenes = list(range(1, num_scenes + 1))
            
            # Build episode selection
            scene_episodes = {}
            for scene in scenes:
                if scene in base_16_60:
                    base_episodes = base_16_60[scene]
                    # Subsample episodes if needed
                    if num_episodes_per_scene < len(base_episodes):
                        selected_episodes = subsample_episodes_by_diversity(
                            base_episodes, num_episodes_per_scene
                        )
                    else:
                        selected_episodes = base_episodes[:num_episodes_per_scene]
                    scene_episodes[scene] = selected_episodes
            
            # Generate filename
            minutes = num_episodes_per_scene * 3.75
            if minutes == int(minutes):
                minutes_str = str(int(minutes))
            else:
                # Convert 3.75 to 3_75, 7.5 to 7_5
                minutes_str = str(minutes).replace('.', '_')
            
            filename = f'scene_diversity_{num_scenes}_{minutes_str}.yaml'
            output_path = output_dir / filename
            
            # Generate config
            generate_config_yaml(scene_episodes, output_path)
    
    print(f"\nGenerated 25 config files in {output_dir}")

if __name__ == '__main__':
    main()

