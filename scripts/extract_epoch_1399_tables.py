#!/usr/bin/env python3
"""
Sync wandb runs and extract validation metrics for a specified epoch.
Create tables similar to scene_diversity_fold_clothes format.

Usage:
    python extract_epoch_1399_tables.py --epoch 999
    python extract_epoch_1399_tables.py --epoch 1399 --output-dir /path/to/output
"""

import json
import re
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
import pandas as pd
import wandb
from wandb import Api

def parse_directory_name(dirname: str) -> Optional[Tuple[int, float]]:
    """
    Parse directory name to extract scenes and time.
    
    Expected format: scenes-{number}-time-{minutes}_{timestamp}
    Note: Time may use underscores instead of decimal points (e.g., "7_5" = 7.5, "3_75" = 3.75)
    
    Returns:
        Tuple of (scenes, time) or None if parsing fails
    """
    # Pattern to match: scenes-{number}-time-{minutes}_{timestamp}
    # Handle both decimal points and underscores in time (e.g., "7_5" = 7.5, "3_75" = 3.75)
    pattern = r'scenes-(\d+)-time-([\d._]+)-'
    match = re.match(pattern, dirname)
    if match:
        scenes = int(match.group(1))
        time_str = match.group(2)
        # Convert underscore to decimal point for time
        if '_' in time_str:
            time = float(time_str.replace('_', '.'))
        else:
            time = float(time_str)
        return (scenes, time)
    return None

def find_wandb_run_dir(subdir_path: Path) -> Optional[Path]:
    """Find wandb run directory, prioritizing run-* over offline-run-*."""
    wandb_dir = subdir_path / "wandb"
    if not wandb_dir.exists():
        return None
    
    # Try latest-run symlink first
    latest_run = wandb_dir / "latest-run"
    if latest_run.exists() and latest_run.is_symlink():
        resolved = latest_run.resolve()
        if resolved.exists() and "run-" in resolved.name and "offline-run" not in resolved.name:
            return resolved
    
    # Look for run-* directories (prioritize these over offline-run-*)
    run_dirs = [d for d in wandb_dir.glob("run-*") if "offline-run" not in d.name]
    if run_dirs:
        return run_dirs[0]
    
    # Fallback to offline-run-* if no run-* found
    offline_dirs = [d for d in wandb_dir.glob("offline-run-*")]
    if offline_dirs:
        return offline_dirs[0]
    
    return None

def sync_wandb_run(wandb_run_dir: Path) -> bool:
    """Sync a wandb run to generate summary files."""
    # Check if already synced (wandb-summary.json exists and is non-empty)
    summary_path = wandb_run_dir / "files" / "wandb-summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                content = f.read().strip()
                if content:
                    return True  # Already synced
        except:
            pass
    
    try:
        print(f"      Syncing...", end='', flush=True)
        result = subprocess.run(
            ['wandb', 'sync', str(wandb_run_dir)],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print(" done", flush=True)
            return True
        else:
            print(f" failed (code {result.returncode})", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(" timeout", flush=True)
        return False
    except Exception as e:
        print(f" error: {e}", flush=True)
        return False

def extract_epoch_metrics(wandb_dir: Path, target_epoch: int, sync_first: bool = True) -> Tuple[Dict[str, float], Optional[int]]:
    """
    Extract validation metrics for target_epoch, or nearest validation epoch with metrics.
    
    Args:
        wandb_dir: Path to wandb run directory
        target_epoch: Target epoch number to extract metrics for
        sync_first: Whether to sync wandb run first
    
    Returns:
        Tuple of (metrics_dict, actual_epoch_used)
        actual_epoch_used is None if no metrics found, otherwise the epoch number used
    """
    metrics = {}
    actual_epoch_used = None
    
    # Sync if needed
    if sync_first:
        sync_wandb_run(wandb_dir)
        time.sleep(2)  # Give it time to write files
    
    # Get project and run name from metadata
    metadata_path = wandb_dir / "files" / "wandb-metadata.json"
    project = None
    run_name = None
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                args = metadata.get('args', [])
                for arg in args:
                    if 'project=' in arg:
                        project = arg.split('project=')[-1].split()[0]
                        break
                # Extract run name from directory
                # Directory format: run-{timestamp}-{name}
                # We need the full name after the timestamp
                if 'run-' in wandb_dir.name:
                    parts = wandb_dir.name.split('run-', 1)[1].split('-', 1)
                    if len(parts) > 1:
                        # Take everything after the timestamp (first part)
                        run_name = parts[1]
                    else:
                        run_name = wandb_dir.name.split('run-')[1]
                else:
                    run_name = None
        except:
            pass
    
    # Try to get metrics from wandb API
    if project and run_name:
        try:
            api = Api()
            # Try different entity/project combinations
            run_paths = [
                f"rl2-group/{project}/{run_name}",
                f"local/{project}/{run_name}",
                f"{project}/{run_name}",
            ]
            
            for run_path in run_paths:
                try:
                    run = api.run(run_path)
                    # Scan history for target epoch
                    history = list(run.scan_history())
                    
                    # First, try to find target epoch with validation metrics
                    target_epoch_row = None
                    epochs_with_valid = []
                    
                    for row in history:
                        epoch = row.get('epoch', None)
                        if epoch is not None:
                            # Check if this row has validation metrics
                            valid_metrics = {
                                k: v for k, v in row.items() 
                                if k.startswith('Valid/') and v is not None
                            }
                            
                            if epoch == target_epoch:
                                target_epoch_row = row
                                if valid_metrics:
                                    # Found target epoch with metrics - use it!
                                    metrics.update(valid_metrics)
                                    actual_epoch_used = target_epoch
                                    return metrics, actual_epoch_used
                            
                            # Collect all epochs with validation metrics
                            if valid_metrics:
                                epochs_with_valid.append((epoch, row))
                    
                    # If target epoch has no metrics, find nearest validation epoch
                    if not metrics and epochs_with_valid:
                        # Find closest epoch to target that has validation metrics
                        closest_epoch, closest_row = min(
                            epochs_with_valid, 
                            key=lambda x: abs(x[0] - target_epoch)
                        )
                        
                        valid_metrics = {
                            k: v for k, v in closest_row.items() 
                            if k.startswith('Valid/') and v is not None
                        }
                        if valid_metrics:
                            metrics.update(valid_metrics)
                            actual_epoch_used = closest_epoch
                            return metrics, actual_epoch_used
                    
                    break
                except Exception as e:
                    # Try next run_path
                    continue
        except Exception as e:
            # API initialization failed or other error
            pass
    
    return metrics, actual_epoch_used

def sanitize_filename(metric_name: str) -> str:
    """
    Sanitize metric name for use as filename.
    
    Replaces invalid filesystem characters with underscores.
    """
    # Remove "Valid/" prefix
    name = metric_name.replace("Valid/", "")
    # Replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace slashes with underscores
    name = name.replace('/', '_')
    return name

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract validation metrics for a specific epoch and generate CSV tables"
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=999,
        help='Target epoch number to extract metrics for (default: 999)'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default="/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold-clothes-cotrain-2",
        help='Directory containing experiment subdirectories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/coc/flash7/bli678/Shared/EgoVerse/results/scene_diversity_cotrain_fold_clothes",
        help='Output directory for CSV tables'
    )
    
    args = parser.parse_args()
    
    target_epoch = args.epoch
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not logs_dir.exists():
        print(f"Error: Directory not found: {logs_dir}")
        return
    
    # Step 1: List directories with target epoch checkpoints
    print("="*80)
    print(f"STEP 1: Finding directories with epoch {target_epoch} checkpoints")
    print("="*80)
    
    subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    directories_with_epoch = []
    
    for subdir in sorted(subdirs):
        ckpt_path = subdir / "checkpoints" / f"epoch_epoch={target_epoch}.ckpt"
        if ckpt_path.exists():
            directories_with_epoch.append(subdir.name)
    
    print(f"\nFound {len(directories_with_epoch)} directories with epoch {target_epoch} checkpoints:")
    for dirname in directories_with_epoch:
        print(f"  - {dirname}")
    
    # Step 2: Collect metrics by configuration
    print("\n" + "="*80)
    print(f"STEP 2: Syncing wandb runs and extracting epoch {target_epoch} metrics")
    print("="*80)
    
    metrics_by_config: Dict[Tuple[int, float], Dict[str, float]] = {}
    
    for i, subdir_name in enumerate(sorted(directories_with_epoch), 1):
        subdir = logs_dir / subdir_name
        
        # Parse directory name
        parsed = parse_directory_name(subdir_name)
        if parsed is None:
            print(f"[{i}/{len(directories_with_epoch)}] Skipping {subdir_name}: Could not parse directory name")
            continue
        
        scenes, time = parsed
        print(f"\n[{i}/{len(directories_with_epoch)}] Processing: {subdir_name}")
        print(f"  -> {scenes} scenes, {time} minutes")
        
        # Find wandb run directory
        wandb_run_dir = find_wandb_run_dir(subdir)
        if wandb_run_dir is None:
            print(f"  [WARN] wandb directory not found")
            continue
        
        print(f"  Found wandb run: {wandb_run_dir.name}")
        
        # Extract metrics (with syncing)
        print(f"  Extracting metrics...", end='', flush=True)
        metrics, actual_epoch = extract_epoch_metrics(wandb_run_dir, target_epoch, sync_first=True)
        print("", flush=True)  # New line after extraction
        
        if metrics:
            config = (scenes, time)
            metrics_by_config[config] = metrics
            if actual_epoch == target_epoch:
                print(f"  [OK] Found {len(metrics)} validation metrics at epoch {target_epoch}")
            else:
                print(f"  [OK] Found {len(metrics)} validation metrics at epoch {actual_epoch} (nearest to {target_epoch})")
        else:
            print(f"  [FAIL] No metrics found")
            # Debug: Check if run is accessible
            try:
                import wandb
                from wandb import Api
                metadata_path = wandb_run_dir / "files" / "wandb-metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        args = metadata.get('args', [])
                        project = None
                        for arg in args:
                            if 'project=' in arg:
                                project = arg.split('project=')[-1].split()[0]
                                break
                        if 'run-' in wandb_run_dir.name:
                            parts = wandb_run_dir.name.split('run-', 1)[1].split('-', 1)
                            run_name = parts[1] if len(parts) > 1 else wandb_run_dir.name.split('run-')[1]
                            if project and run_name:
                                api = Api()
                                try:
                                    run = api.run(f"rl2-group/{project}/{run_name}")
                                    history = list(run.scan_history())
                                    epochs = [row.get('epoch') for row in history if row.get('epoch') is not None]
                                    if epochs:
                                        print(f"      Debug: Epoch range {min(epochs)}-{max(epochs)}, 999 exists: {999 in epochs}")
                                except:
                                    print(f"      Debug: Cannot access run via API (may not be synced)")
            except:
                pass
    
    if not metrics_by_config:
        print("\nError: No metrics found in any directory!")
        return
    
    # Step 3: Generate tables
    print("\n" + "="*80)
    print("STEP 3: Generating CSV tables")
    print("="*80)
    
    # Collect all unique metric names
    all_metrics = set()
    for metrics in metrics_by_config.values():
        all_metrics.update(metrics.keys())
    
    print(f"\nFound {len(all_metrics)} unique validation metrics")
    print(f"Found {len(metrics_by_config)} experiment configurations")
    
    # Define expected scenes and time values
    expected_scenes = [1, 2, 4, 8, 16]
    expected_times = [3.75, 7.5, 15, 30, 60]
    
    # Report which configurations were found
    print("\nExperiment configurations found:")
    for config in sorted(metrics_by_config.keys()):
        scenes, time = config
        num_metrics = len(metrics_by_config[config])
        print(f"  {scenes} scenes, {time} minutes: {num_metrics} metrics")
    
    # Create a table for each metric
    for metric_name in sorted(all_metrics):
        print(f"\nProcessing metric: {metric_name}")
        
        # Create DataFrame with scenes as rows and time as columns
        data = {}
        missing_values = []
        for time in expected_times:
            data[time] = []
            for scenes in expected_scenes:
                config = (scenes, time)
                value = metrics_by_config.get(config, {}).get(metric_name, None)
                data[time].append(value)
                if value is None:
                    missing_values.append(f"({scenes}s, {time}m)")
        
        df = pd.DataFrame(data, index=expected_scenes)
        df.index.name = "Scenes"
        df.columns.name = "Time (minutes)"
        
        if missing_values:
            print(f"  Warning: Missing values for {metric_name} at: {', '.join(missing_values[:5])}")
            if len(missing_values) > 5:
                print(f"    ... and {len(missing_values) - 5} more")
        
        # Sanitize metric name for filename
        filename = sanitize_filename(metric_name)
        output_path = output_dir / f"{filename}.csv"
        
        # Save to CSV
        df.to_csv(output_path)
        print(f"  [OK] Saved to: {output_path}")
        print(f"  Table shape: {df.shape}")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  - Processed {len(metrics_by_config)} experiment configurations")
    print(f"  - Found {len(all_metrics)} unique validation metrics")
    print(f"  - Created {len(all_metrics)} CSV files")
    print(f"  - Output directory: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
