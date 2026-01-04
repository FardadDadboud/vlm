"""
Grid Search Runner with Adaptive GPU Parallelization
"""

import os
import sys
import json
import argparse
import concurrent.futures
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Dict, Any
import time

from gpu_manager import GPUManager
from experiment_queue import ExperimentQueue


def run_single_experiment(exp: Dict[str, Any], gpu_id: int, output_dir: Path) -> Dict[str, Any]:
    """
    Run a single experiment on specified GPU
    
    Args:
        exp: Experiment configuration
        gpu_id: GPU ID to use
        output_dir: Output directory for results
        
    Returns:
        Results dictionary
    """
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create experiment-specific output directory
    exp_dir = output_dir / f"exp_{exp['id']:04d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(exp['config'], f, indent=2)
    
    try:
        # Import here to avoid conflicts
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from shift_comprehensive_evaluator import main as run_evaluation
        from vlm_shift_dataset import VLMSHIFTDataset
        from vlm_shift_domain_evaluator import VLMSHIFTDomainEvaluator
        from vlm_detector_system_new import GroundingDINODetector, check_gpu_status
        from adapters import create_adapter
        from PIL import Image
        
        # Load dataset
        dataset = VLMSHIFTDataset(
            data_root=exp['config']['data_root'],
            split=exp['config']['split'],
            filters=exp['config']['filters']
        )
        
        # Initialize detector
        device = "cuda" if check_gpu_status() else "cpu"
        base_detector = GroundingDINODetector(
            model_path=exp['config']['detector']['model_path'],
            device=device
        )
        
        # Create adapter
        detector = create_adapter(
            adaptation_type=exp['config']['adaptation']['type'],
            detector=base_detector,
            config=exp['config']
        )
        
        # Run detection
        predictions = []
        max_samples = exp['config'].get('max_samples')
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        for i in range(num_samples):
            sample = dataset[i]
            image = Image.open(sample['image_path'])
            
            result = detector.adapt_and_detect(
                image,
                exp['config']['detector']['target_classes'],
                threshold=exp['config']['detector']['threshold']
            )
            
            predictions.append({
                'image_id': sample['image_info']['id'],
                'boxes': result.boxes,
                'scores': result.scores,
                'labels': result.labels
            })
        
        # Evaluate
        evaluator = VLMSHIFTDomainEvaluator(
            dataset=dataset,
            output_dir=str(exp_dir)
        )
        
        results = evaluator.evaluate_detections(
            predictions=predictions,
            visualize=False,
            save_visualizations=0
        )
        
        # Extract key metrics
        return {
            'exp_id': exp['id'],
            'params': exp['params'],
            'metrics': {
                'mAP': results['overall']['mAP'],
                'mAP_50': results['overall']['mAP_50'],
                'mAP_75': results['overall']['mAP_75']
            },
            'gpu_id': gpu_id,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'exp_id': exp['id'],
            'params': exp['params'],
            'error': str(e),
            'gpu_id': gpu_id,
            'status': 'failed'
        }


def run_grid_search(grid_config_path: str):
    """
    Run grid search with adaptive GPU parallelization
    
    Args:
        grid_config_path: Path to grid search config JSON
    """
    # Load grid config
    with open(grid_config_path, 'r') as f:
        grid_config = json.load(f)
    
    # Create output directory
    output_dir = Path(grid_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grid config
    with open(output_dir / 'grid_config.json', 'w') as f:
        json.dump(grid_config, f, indent=2)
    
    # Initialize GPU manager and experiment queue
    gpu_manager = GPUManager(safety_margin_gb=grid_config['execution']['safety_margin_gb'])
    queue = ExperimentQueue(grid_config)
    
    memory_per_job = grid_config['execution']['gpu_memory_per_job_gb']
    max_workers = gpu_manager.get_max_parallel_jobs(memory_per_job)
    
    print(f"\nGrid Search Configuration:")
    print(f"  Total experiments: {len(queue.experiments)}")
    print(f"  Max parallel jobs: {max_workers}")
    print(f"  Memory per job: {memory_per_job} GB")
    print(f"  Output directory: {output_dir}\n")
    
    # Track results
    all_results = []
    checkpoint_file = output_dir / 'checkpoint.json'
    
    # Resume from checkpoint if exists
    completed_exp_ids = set()
    if checkpoint_file.exists():
        print(f"\n{'='*80}")
        print("CHECKPOINT FOUND - Resuming from previous run")
        print(f"{'='*80}\n")
        with open(checkpoint_file, 'r') as f:
            all_results = json.load(f)
        completed_exp_ids = {r['exp_id'] for r in all_results}
        print(f"Loaded {len(all_results)} completed experiments")
        print(f"Remaining: {len(queue.experiments) - len(all_results)}\n")
        
        # Remove completed experiments from queue
        queue.pending = [eid for eid in queue.pending if eid not in completed_exp_ids]
        queue.completed = list(completed_exp_ids)
    
    # Main execution loop
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        start_time = time.time()
        
        while queue.has_pending() or futures:
            # Submit new jobs when GPU available
            while queue.has_pending():
                gpu_id = gpu_manager.allocate_gpu(memory_per_job)
                if gpu_id is None:
                    break  # All GPUs full
                
                exp = queue.get_next()
                future = executor.submit(run_single_experiment, exp, gpu_id, output_dir)
                futures[future] = (exp['id'], gpu_id)
                
                print(f"Started experiment {exp['id']}/{len(queue.experiments)} on GPU {gpu_id}")
            
            # Wait for any job to complete
            if futures:
                done, pending = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                
                for future in done:
                    exp_id, gpu_id = futures.pop(future)
                    
                    # Robust error handling for OOM and broken processes
                    try:
                        result = future.result()
                    except concurrent.futures.process.BrokenProcessPool as e:
                        print(f"\n{'='*80}")
                        print(f"WARNING: Process pool broken (likely OOM kill)")
                        print(f"Experiment {exp_id} failed - marking as OOM error")
                        print(f"Saving checkpoint and continuing...")
                        print(f"{'='*80}\n")
                        result = {
                            'exp_id': exp_id,
                            'params': queue.experiments[exp_id]['params'],
                            'error': 'OOM_KILL - Process terminated by system',
                            'gpu_id': gpu_id,
                            'status': 'oom_killed'
                        }
                    except Exception as e:
                        print(f"\n{'='*80}")
                        print(f"WARNING: Unexpected error in experiment {exp_id}")
                        print(f"Error: {str(e)}")
                        print(f"{'='*80}\n")
                        result = {
                            'exp_id': exp_id,
                            'params': queue.experiments[exp_id]['params'],
                            'error': str(e),
                            'gpu_id': gpu_id,
                            'status': 'failed'
                        }
                    
                    all_results.append(result)
                    queue.mark_completed(exp_id)
                    gpu_manager.release_gpu(gpu_id, memory_per_job)
                    
                    # Save checkpoint immediately after each result
                    with open(checkpoint_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                    # Print progress
                    progress = queue.get_progress()
                    elapsed = time.time() - start_time
                    rate = len(all_results) / elapsed if elapsed > 0 else 0
                    eta = (progress['total'] - len(all_results)) / rate if rate > 0 else 0
                    
                    print(f"Completed {len(all_results)}/{progress['total']} "
                          f"({100*len(all_results)/progress['total']:.1f}%) | "
                          f"Rate: {rate:.2f} exp/s | ETA: {eta/60:.1f} min")
                    
                    if result['status'] == 'success':
                        print(f"  mAP@50: {result['metrics']['mAP_50']:.4f}")
                    else:
                        print(f"  Status: {result['status']}")
                        if 'error' in result:
                            print(f"  Error: {result['error'][:100]}...")  # Truncate long errors
    
    # Save final results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Grid Search Complete!")
    print(f"{'='*80}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved to: {results_file}")
    
    # Analyze results
    successful_results = [r for r in all_results if r['status'] == 'success']
    failed_results = [r for r in all_results if r['status'] == 'failed']
    oom_results = [r for r in all_results if r['status'] == 'oom_killed']
    
    print(f"\nExperiment Summary:")
    print(f"  Total: {len(all_results)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    print(f"  OOM Killed: {len(oom_results)}")
    
    if oom_results:
        print(f"\n{'='*80}")
        print("OOM Killed Experiments (consider reducing memory usage):")
        print(f"{'='*80}")
        for r in oom_results[:5]:  # Show first 5
            print(f"  Exp {r['exp_id']}: {r['params']}")
        if len(oom_results) > 5:
            print(f"  ... and {len(oom_results)-5} more")
    
    if failed_results:
        print(f"\n{'='*80}")
        print("Failed Experiments:")
        print(f"{'='*80}")
        for r in failed_results[:5]:  # Show first 5
            error_msg = r.get('error', 'Unknown error')
            print(f"  Exp {r['exp_id']}: {error_msg[:80]}...")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results)-5} more")
    
    # Find best configuration
    if successful_results:
        best = max(successful_results, key=lambda x: x['metrics']['mAP_50'])
        print(f"\nBest Configuration:")
        print(f"  Parameters: {best['params']}")
        print(f"  mAP@50: {best['metrics']['mAP_50']:.4f}")
    else:
        print(f"\n{'='*80}")
        print("WARNING: No successful experiments!")
        print("All experiments failed - check errors above")
        print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid Search for Hyperparameter Tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to grid search config JSON')
    args = parser.parse_args()
    
    run_grid_search(args.config)