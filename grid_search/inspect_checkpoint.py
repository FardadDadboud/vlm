#!/usr/bin/env python3
"""
Checkpoint Inspector - Check grid search progress and status
"""

import json
import argparse
from pathlib import Path


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint file and show progress"""
    checkpoint_file = Path(checkpoint_path)
    
    if not checkpoint_file.exists():
        print(f"No checkpoint found at: {checkpoint_path}")
        print("This is a fresh start - no previous progress to resume")
        return
    
    # Load checkpoint
    with open(checkpoint_file, 'r') as f:
        results = json.load(f)
    
    # Analyze
    total = len(results)
    successful = len([r for r in results if r.get('status') == 'success'])
    failed = len([r for r in results if r.get('status') == 'failed'])
    oom = len([r for r in results if r.get('status') == 'oom_killed'])
    
    print(f"{'='*80}")
    print(f"Checkpoint Status: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    print(f"Progress:")
    print(f"  Completed: {total} experiments")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  OOM Killed: {oom}\n")
    
    if successful > 0:
        # Show best result so far
        successful_results = [r for r in results if r.get('status') == 'success']
        best = max(successful_results, key=lambda x: x['metrics']['mAP_50'])
        print(f"Best Result So Far:")
        print(f"  Exp ID: {best['exp_id']}")
        print(f"  Parameters: {best['params']}")
        print(f"  mAP@50: {best['metrics']['mAP_50']:.4f}\n")
    
    if oom > 0:
        print(f"{'='*80}")
        print(f"WARNING: {oom} experiments killed by OOM")
        print(f"{'='*80}")
        print("Consider:")
        print("  1. Reduce dataset_subset.max_samples in config")
        print("  2. Increase SLURM memory allocation (#SBATCH --mem=XXG)")
        print("  3. Increase execution.gpu_memory_per_job_gb (reduces parallelism)")
        print("  4. Request more GPUs with more memory\n")
        
        # Show which parameter combinations cause OOM
        oom_results = [r for r in results if r.get('status') == 'oom_killed']
        print(f"OOM Killed Experiments:")
        for r in oom_results[:10]:
            print(f"  Exp {r['exp_id']}: {r['params']}")
        if len(oom_results) > 10:
            print(f"  ... and {len(oom_results)-10} more\n")
    
    if failed > 0:
        print(f"Recent Failures:")
        failed_results = [r for r in results if r.get('status') == 'failed']
        for r in failed_results[-5:]:
            error = r.get('error', 'Unknown error')
            print(f"  Exp {r['exp_id']}: {error[:100]}...")
        print()
    
    print(f"{'='*80}")
    print("To Resume:")
    print(f"  sbatch your_slurm_script.sh")
    print("  (or) python grid_search/grid_search_runner.py --config <config.json>")
    print("  The script will automatically resume from this checkpoint")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect grid search checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint.json')
    args = parser.parse_args()
    
    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()