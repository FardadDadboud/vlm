"""
grid_search_runner_multidet.py  —  multi-detector grid runner.

Faithful mirror of grid_search/grid_search_runner.py (same ExperimentQueue,
GPUManager, checkpoint.json resume, pooled results['overall']['mAP_50'] metric,
OOM-safe loop) with ONE fix: the base detector is dispatched by
config['detector']['name'] (grounding-dino | owlv2 | yolo-world), mirroring
shift_comprehensive_evaluator.py:378-389. The stock runner hardcodes
GroundingDINODetector and therefore cannot run OWLv2 / YOLO-World grids.

New file — does NOT modify the stock runner. Reuses grid_search/ modules on
sys.path.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # grid_search/
from gpu_manager import GPUManager
from experiment_queue import ExperimentQueue


def _build_detector(cfg, device):
    """Dispatch by detector name — mirrors shift_comprehensive_evaluator.py."""
    from vlm_detector_system_new import (GroundingDINODetector, OWLv2Detector,
                                         YOLOWorldDetector)
    name = cfg['detector']['name']
    mp = cfg['detector']['model_path']
    if name == 'owlv2':
        return OWLv2Detector(model_path=mp, device=device)
    elif name == 'yolo-world':
        return YOLOWorldDetector(model_path=mp, device=device)
    else:
        return GroundingDINODetector(model_path=mp, device=device)


def run_single_experiment(exp: Dict[str, Any], gpu_id: int, output_dir: Path) -> Dict[str, Any]:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    exp_dir = output_dir / f"exp_{exp['id']:04d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(exp['config'], f, indent=2)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # VLM dir
        from vlm_shift_dataset import VLMSHIFTDataset
        from vlm_shift_domain_evaluator import VLMSHIFTDomainEvaluator
        from vlm_detector_system_new import check_gpu_status
        from adapters import create_adapter
        from PIL import Image

        dataset = VLMSHIFTDataset(
            data_root=exp['config']['data_root'],
            split=exp['config']['split'],
            filters=exp['config']['filters'])

        device = "cuda" if check_gpu_status() else "cpu"
        base_detector = _build_detector(exp['config'], device)   # <-- the fix
        detector = create_adapter(
            adaptation_type=exp['config']['adaptation']['type'],
            detector=base_detector, config=exp['config'])

        predictions = []
        max_samples = exp['config'].get('max_samples')
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        for i in range(num_samples):
            sample = dataset[i]
            image = Image.open(sample['image_path'])
            result = detector.adapt_and_detect(
                image, exp['config']['detector']['target_classes'],
                threshold=exp['config']['detector']['threshold'])
            predictions.append({'image_id': sample['image_info']['id'],
                                'boxes': result.boxes, 'scores': result.scores,
                                'labels': result.labels})

        evaluator = VLMSHIFTDomainEvaluator(dataset=dataset, output_dir=str(exp_dir))
        results = evaluator.evaluate_detections(
            predictions=predictions, visualize=False, save_visualizations=0)

        return {'exp_id': exp['id'], 'params': exp['params'],
                'metrics': {'mAP': results['overall']['mAP'],
                            'mAP_50': results['overall']['mAP_50'],
                            'mAP_75': results['overall']['mAP_75']},
                'gpu_id': gpu_id, 'status': 'success'}
    except Exception as e:
        return {'exp_id': exp['id'], 'params': exp['params'], 'error': str(e),
                'gpu_id': gpu_id, 'status': 'failed'}


def run_grid_search(grid_config_path: str):
    with open(grid_config_path) as f:
        grid_config = json.load(f)
    output_dir = Path(grid_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'grid_config.json', 'w') as f:
        json.dump(grid_config, f, indent=2)

    gpu_manager = GPUManager(safety_margin_gb=grid_config['execution']['safety_margin_gb'])
    queue = ExperimentQueue(grid_config)
    memory_per_job = grid_config['execution']['gpu_memory_per_job_gb']
    max_workers = gpu_manager.get_max_parallel_jobs(memory_per_job)
    print(f"Total experiments: {len(queue.experiments)} | max parallel: {max_workers}")

    all_results = []
    checkpoint_file = output_dir / 'checkpoint.json'
    if checkpoint_file.exists():
        all_results = json.load(open(checkpoint_file))
        done_ids = {r['exp_id'] for r in all_results}
        queue.pending = [eid for eid in queue.pending if eid not in done_ids]
        queue.completed = list(done_ids)
        print(f"RESUME: {len(all_results)} done, {len(queue.pending)} remaining")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        t0 = time.time()
        while queue.has_pending() or futures:
            while queue.has_pending():
                gpu_id = gpu_manager.allocate_gpu(memory_per_job)
                if gpu_id is None:
                    break
                exp = queue.get_next()
                futures[executor.submit(run_single_experiment, exp, gpu_id, output_dir)] = (exp['id'], gpu_id)
                print(f"Started exp {exp['id']} on GPU {gpu_id}")
            if futures:
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for fut in done:
                    exp_id, gpu_id = futures.pop(fut)
                    try:
                        result = fut.result()
                    except concurrent.futures.process.BrokenProcessPool:
                        result = {'exp_id': exp_id, 'params': queue.experiments[exp_id]['params'],
                                  'error': 'OOM_KILL', 'gpu_id': gpu_id, 'status': 'oom_killed'}
                    except Exception as e:
                        result = {'exp_id': exp_id, 'params': queue.experiments[exp_id]['params'],
                                  'error': str(e), 'gpu_id': gpu_id, 'status': 'failed'}
                    all_results.append(result)
                    queue.mark_completed(exp_id)
                    gpu_manager.release_gpu(gpu_id, memory_per_job)
                    json.dump(all_results, open(checkpoint_file, 'w'), indent=2)
                    n = len(all_results); tot = len(queue.experiments)
                    print(f"[{n}/{tot}] status={result['status']}"
                          + (f" mAP@50={result['metrics']['mAP_50']:.4f}" if result['status'] == 'success' else ""))

    json.dump(all_results, open(output_dir / 'results.json', 'w'), indent=2)
    ok = [r for r in all_results if r['status'] == 'success']
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min | success {len(ok)}/{len(all_results)}")
    if ok:
        best = max(ok, key=lambda x: x['metrics']['mAP_50'])
        print(f"BEST pooled mAP@50={best['metrics']['mAP_50']:.4f} params={best['params']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    run_grid_search(args.config)
