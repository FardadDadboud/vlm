#!/usr/bin/env python3
"""
SHIFT Comprehensive Evaluation Script
Runs full baseline VLM evaluation with mAP metrics, visualizations, and logs

Updated: Added video boundary reset for temporal methods
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import time

# Import components
from vlm_shift_dataset import VLMSHIFTDataset
from vlm_shift_domain_evaluator import VLMSHIFTDomainEvaluator
from vlm_detector_system_new import OWLv2Detector, GroundingDINODetector, check_gpu_status
from adapters import create_adapter


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_detector_on_dataset(detector, dataset, config: Dict[str, Any], 
                            output_dir: Path,
                            max_samples: int = None):
    """
    Run detector on dataset and return predictions in COCO format.
    
    Now handles video boundaries for temporal methods.
    """
    target_classes = config['detector']['target_classes']
    threshold = config['detector']['threshold']
    reset_on_video_change = config.get('reset_on_video_change', True)
    
    predictions = []
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"\nRunning {detector.model_path} on {num_samples} samples...")
    print(f"Target classes: {target_classes}")
    print(f"Threshold: {threshold}")
    print(f"Reset on video change: {reset_on_video_change}\n")
    
    start_time = time.time()
    
    # Track video boundaries
    prev_video_id = None
    video_count = 0
    reset_count = 0
    
    # Adaptation statistics
    adaptation_stats = {
        'frames_processed': 0,
        'frames_with_updates': 0,
        'total_confident_detections': 0,
        'resets': 0,
        'videos_processed': set()
    }
    
    for i in range(num_samples):
        sample = dataset[i]
        image_path = sample['image_path']
        
        # Get video ID for boundary detection
        video_id = sample.get('video_id') or sample.get('sequence_id')
        if video_id is None:
            # Try to extract from image path (e.g., .../video_name/frame.jpg)
            path_parts = Path(image_path).parts
            if len(path_parts) >= 2:
                video_id = path_parts[-2]  # Parent folder as video ID
        
        # Check for video boundary
        if reset_on_video_change and video_id is not None:
            if prev_video_id is not None and video_id != prev_video_id:
                # Video changed - reset temporal state
                if hasattr(detector, 'reset'):
                    detector.reset()
                    reset_count += 1
                video_count += 1
            prev_video_id = video_id
            adaptation_stats['videos_processed'].add(video_id)

        


        # Load image
        image = Image.open(image_path)
        
        # Use adapter (works for both vanilla and adapted)
        detection_result = detector.adapt_and_detect(image, target_classes, threshold=threshold)
        
        # Collect adaptation statistics if available
        if hasattr(detector, 'get_adaptation_stats'):
            frame_stats = detector.get_adaptation_stats()
            adaptation_stats['frames_processed'] += 1
            if frame_stats.get('num_updates', 0) > 0:
                adaptation_stats['frames_with_updates'] += 1
            adaptation_stats['total_confident_detections'] += frame_stats.get('num_confident', 0)
        
        # Format prediction
        pred = {
            'image_id': sample['image_info']['id'],
            'boxes': [[float(x) for x in box] for box in detection_result.boxes],
            'scores': [float(s) for s in detection_result.scores],
            'labels': [str(l) for l in detection_result.labels]
        }
        predictions.append(pred)
        
        # Progress logging
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            eta = (num_samples - i - 1) / fps if fps > 0 else 0
            
            # Build info string
            info_parts = [f"Progress: {i+1}/{num_samples} ({100*(i+1)/num_samples:.1f}%)"]
            info_parts.append(f"FPS: {fps:.2f}")
            info_parts.append(f"ETA: {eta:.1f}s")
            
            if hasattr(detector, 'cache') and detector.cache is not None:
                info_parts.append(f"Cache: {detector.cache.M}")
            if hasattr(detector, 'ssm') and detector.ssm is not None:
                info_parts.append(f"SSM: active")
            if reset_count > 0:
                info_parts.append(f"Resets: {reset_count}")
                
            print(f"  {' | '.join(info_parts)}")
    
    total_time = time.time() - start_time
    print(f"\n✓ Detection complete: {num_samples} images in {total_time:.2f}s ({num_samples/total_time:.2f} FPS)")
    
    # Print adaptation statistics
    if adaptation_stats['frames_processed'] > 0:
        print(f"\nAdaptation Statistics:")
        print(f"  Videos processed: {len(adaptation_stats['videos_processed'])}")
        print(f"  Adapter resets: {reset_count}")
        print(f"  Frames with updates: {adaptation_stats['frames_with_updates']}/{adaptation_stats['frames_processed']} "
              f"({100*adaptation_stats['frames_with_updates']/adaptation_stats['frames_processed']:.1f}%)")
        print(f"  Total confident detections: {adaptation_stats['total_confident_detections']}")
    
    # Save predictions
    pred_file = output_dir / "predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Predictions saved to: {pred_file}")
    
    return predictions


def run_detector_by_video(detector, dataset, config: Dict[str, Any],
                          output_dir: Path, max_samples: int = None):
    """
    Run detector video-by-video with proper resets between videos.
    
    This ensures temporal methods get clean sequences.
    """
    target_classes = config['detector']['target_classes']
    threshold = config['detector']['threshold']
    
    # Group samples by video
    video_samples = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        video_id = sample.get('video_id') or sample.get('sequence_id')
        if video_id is None:
            path_parts = Path(sample['image_path']).parts
            video_id = path_parts[-2] if len(path_parts) >= 2 else f"video_{i//100}"
        
        if video_id not in video_samples:
            video_samples[video_id] = []
        video_samples[video_id].append((i, sample))
    
    print(f"\nFound {len(video_samples)} videos in dataset")
    print(f"Target classes: {target_classes}")
    print(f"Threshold: {threshold}\n")
    
    predictions = []
    total_processed = 0
    start_time = time.time()
    
    # Process each video
    for video_idx, (video_id, samples) in enumerate(video_samples.items()):
        # Reset adapter for new video
        if hasattr(detector, 'reset'):
            detector.reset()
        
        # Sort by frame number if available
        samples.sort(key=lambda x: x[1].get('frame_idx', x[0]))
        
        video_start = time.time()
        
        for frame_idx, (dataset_idx, sample) in enumerate(samples):
            if max_samples is not None and total_processed >= max_samples:
                break
            image_path = sample['image_path']
            image = Image.open(image_path)
            
            detection_result = detector.adapt_and_detect(image, target_classes, threshold=threshold)
            
            pred = {
                'image_id': sample['image_info']['id'],
                'boxes': [[float(x) for x in box] for box in detection_result.boxes],
                'scores': [float(s) for s in detection_result.scores],
                'labels': [str(l) for l in detection_result.labels],
                'video_id': video_id,
                'frame_idx': frame_idx
            }
            predictions.append(pred)
            total_processed += 1
        
        if max_samples is not None and total_processed >= max_samples:
            print(f"  Reached max_samples limit ({max_samples})")
            break
        
        video_time = time.time() - video_start
        if (video_idx + 1) % 1 == 0:
            elapsed = time.time() - start_time
            print(f"  Videos: {video_idx+1}/{len(video_samples)} | "
                  f"Frames: {total_processed} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n✓ Detection complete: {total_processed} frames from {video_idx+1} videos "
          f"in {total_time:.2f}s ({total_processed/total_time:.2f} FPS)")
    
    # Save predictions
    pred_file = output_dir / "predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Predictions saved to: {pred_file}")
    
    return predictions


def main():
    """Run comprehensive evaluation"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='SHIFT Comprehensive Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--by-video', action='store_true', help='Process video-by-video with resets')
    args = parser.parse_args()
    
    # Load configuration
    CONFIG = load_config(args.config)
    
    print("="*80)
    print("SHIFT Comprehensive Baseline Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(json.dumps(CONFIG, indent=2))
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir']) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Step 1: Load dataset
    print("\n" + "="*80)
    print("[Step 1/4] Loading SHIFT Dataset")
    print("="*80)
    
    dataset = VLMSHIFTDataset(
        data_root=CONFIG['data_root'],
        split=CONFIG['split'],
        filters=CONFIG['filters']
    )
    
    print(f"\n✓ Loaded {len(dataset)} frames")
    
    # Get statistics
    stats = dataset.get_domain_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Weather distribution: {dict(stats['weather_distribution'])}")
    print(f"  Time distribution: {dict(stats['time_distribution'])}")
    print(f"  Category distribution: {dict(stats['category_distribution'])}")
    
    # Step 2: Initialize detector
    print("\n" + "="*80)
    print("[Step 2/4] Initializing VLM Detector")
    print("="*80)

    gpu_available = check_gpu_status()
    device = "cuda" if gpu_available else "cpu"

    if CONFIG['detector']['name'] == 'owlv2':
        base_detector = OWLv2Detector(
            model_path=CONFIG['detector']['model_path'],
            device=device
        )
    elif CONFIG['detector']['name'] == 'grounding-dino':
        base_detector = GroundingDINODetector(
            model_path=CONFIG['detector']['model_path'],
            device=device
        )
    else:
        raise ValueError(f"Unknown detector: {CONFIG['detector']['name']}")

    # Wrap with adapter
    detector = create_adapter(
        adaptation_type=CONFIG['adaptation']['type'],
        detector=base_detector,
        config=CONFIG
    )

    print(f"\n✓ Detector ready: {detector.model_path}")
    if CONFIG['adaptation']['type'] != 'none':
        print(f"✓ Adaptation: {CONFIG['adaptation']['type']}")
        print(f"  Parameters: {CONFIG['adaptation']['params']}")
    
    # Step 3: Run detection
    print("\n" + "="*80)
    print("[Step 3/4] Running Detection")
    print("="*80)
    
    # Choose processing mode
    if args.by_video or CONFIG.get('process_by_video', False):
        print("Processing mode: VIDEO-BY-VIDEO (with resets)")
        predictions = run_detector_by_video(
            detector=detector,
            dataset=dataset,
            config=CONFIG,
            output_dir=output_dir,
            max_samples=CONFIG['max_samples']
        )
    else:
        print("Processing mode: SEQUENTIAL (with boundary detection)")
        predictions = run_detector_on_dataset(
            detector=detector,
            dataset=dataset,
            config=CONFIG,
            output_dir=output_dir,
            max_samples=CONFIG['max_samples']
        )
    
    # Step 4: Evaluate with domain evaluator
    print("\n" + "="*80)
    print("[Step 4/4] Computing Comprehensive Metrics")
    print("="*80)
    
    evaluator = VLMSHIFTDomainEvaluator(
        dataset=dataset,
        output_dir=str(output_dir)
    )
    
    results = evaluator.evaluate_detections(
        predictions=predictions,
        visualize=True,
        save_visualizations=CONFIG['num_visualizations']
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    if 'overall' in results:
        print(f"\nOverall Performance:")
        print(f"  mAP @ IoU=0.5:0.95: {results['overall']['mAP']:.4f}")
        print(f"  mAP @ IoU=0.50:     {results['overall']['mAP_50']:.4f}")
        print(f"  mAP (small):        {results['overall']['mAP_small']:.4f}")
        print(f"  mAP (medium):       {results['overall']['mAP_medium']:.4f}")
        print(f"  mAP (large):        {results['overall']['mAP_large']:.4f}")
    
    print(f"\nTop 5 Domains by mAP@50:")
    domain_scores = [(d, r['mAP_50']) for d, r in results.items() 
                     if d not in ['overall', 'size_analysis'] and 'mAP_50' in r]
    domain_scores.sort(key=lambda x: x[1], reverse=True)
    for domain, score in domain_scores[:5]:
        print(f"  {domain:30s}: {score:.4f}")
    
    print(f"\nBottom 5 Domains by mAP@50:")
    for domain, score in domain_scores[-5:]:
        print(f"  {domain:30s}: {score:.4f}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"  - domain_evaluation_results.json : Full metrics")
    print(f"  - evaluation_summary.txt         : Human-readable summary")
    print(f"  - predictions.json               : Raw predictions")
    print(f"  - visualizations/                : Detection visualizations")
    print(f"  - config.json                    : Evaluation configuration")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Review visualizations in: {}/visualizations/".format(output_dir))
    print("  2. Analyze per-domain performance in evaluation_summary.txt")
    print("  3. Identify challenging domains for TTA methods")
    print("="*80)


if __name__ == "__main__":
    main()