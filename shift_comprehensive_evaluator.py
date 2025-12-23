#!/usr/bin/env python3
"""
SHIFT Comprehensive Evaluation Script
Runs full baseline VLM evaluation with mAP metrics, visualizations, and logs
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
    Run detector on dataset and return predictions in COCO format
    """
    target_classes = config['detector']['target_classes']
    threshold = config['detector']['threshold']
    
    predictions = []
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"\nRunning {detector.model_path} on {num_samples} samples...")
    print(f"Target classes: {target_classes}")
    print(f"Threshold: {threshold}\n")
    
    start_time = time.time()
    
    
    for i in range(num_samples):
        sample = dataset[i]
        image_path = sample['image_path']

        # print(f"***********************DEBUGGING***********************")
        # print(f"image_path: {image_path}")
        # print(f"***********************DEBUGGING***********************")
        
        # Load image
        image = Image.open(image_path)
        
        # Use adapter (works for both vanilla and BCA+)
        detection_result = detector.adapt_and_detect(image, target_classes, threshold=threshold)

        # Debug: compare with original detection result
        # original_detection_result = detector.detector.detect(image, target_classes, threshold=0.10)
        # print(f"**************************DEBUGGING**************************")
        # print(f"Original detection result: {original_detection_result}")
        # print(f"Adapted detection result: {detection_result}")
        # print(f"**************************DEBUGGING**************************")
        
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
            cache_info = f" | Cache: {detector.cache.M}" if hasattr(detector, 'cache') else ""
            print(f"  Progress: {i+1}/{num_samples} ({100*(i+1)/num_samples:.1f}%) | "
                  f"FPS: {fps:.2f} | ETA: {eta:.1f}s{cache_info}")
    
    total_time = time.time() - start_time
    print(f"\n✓ Detection complete: {num_samples} images in {total_time:.2f}s ({num_samples/total_time:.2f} FPS)")
    
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
    args = parser.parse_args()
    
    # Load configuration
    CONFIG = load_config(args.config)
    
    print("="*80)
    print("SHIFT Comprehensive Baseline Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(json.dumps(CONFIG, indent=2))
    
    # Create output directory
    # output_dir = Path(CONFIG['output_dir']) + "/" + time.strftime("%Y%m%d_%H%M%S")
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

    # Wrap with BCA+ ONCE
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