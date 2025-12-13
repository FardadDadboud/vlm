#!/usr/bin/env python3
"""
Smoke Test: Run Vanilla VLM Detector on SHIFT Dataset
This demonstrates Step 1c-iii: Running baseline VLM detection

Usage:
    python shift_vanilla_baseline_smoke_test.py
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np

# Import your existing components
from vlm_shift_dataset import VLMSHIFTDataset
from vlm_detector_system_new import OWLv2Detector, check_gpu_status


def run_detector_on_samples(detector, dataset, num_samples: int = 10, target_classes: List[str] = None):
    """
    Run detector on a small number of samples
    
    Args:
        detector: VLM detector instance
        dataset: SHIFT dataset instance
        num_samples: Number of samples to process
        target_classes: List of class names to detect
    
    Returns:
        List of detection results
    """
    if target_classes is None:
        # SHIFT categories
        target_classes = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    
    results = []
    print(f"\nRunning {detector.model_name} on {num_samples} samples...")
    print(f"Target classes: {target_classes}")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image_path = sample['image_path']
        
        # Load image
        image = Image.open(image_path)
        
        # Run detection
        detection_result = detector.detect(image, target_classes, threshold=0.1)
        
        # Format result for evaluation (convert to JSON-serializable types)
        result = {
            'image_id': sample['image_info']['id'],
            'video_name': sample['image_info']['video_name'],
            'frame_index': sample['image_info']['frame_index'],
            'weather': sample['image_info']['weather_coarse'],
            'time': sample['image_info']['timeofday_coarse'],
            'boxes': [[float(x) for x in box] for box in detection_result.boxes],
            'scores': [float(s) for s in detection_result.scores],
            'labels': [str(l) for l in detection_result.labels],
            'gt_annotations': len(sample['annotations'])
        }
        results.append(result)
        
        print(f"  [{i+1}/{num_samples}] {sample['image_info']['video_name']}/{sample['image_info']['frame_index']}: "
              f"{len(detection_result.boxes)} detections (GT: {len(sample['annotations'])})")
    
    return results


def compute_basic_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute basic detection statistics"""
    total_detections = sum(len(r['boxes']) for r in results)
    total_gt = sum(r['gt_annotations'] for r in results)
    
    # Group by domain
    domain_stats = {}
    for result in results:
        domain = f"{result['weather']}_{result['time']}"
        if domain not in domain_stats:
            domain_stats[domain] = {'detections': 0, 'gt': 0, 'frames': 0}
        
        domain_stats[domain]['detections'] += len(result['boxes'])
        domain_stats[domain]['gt'] += result['gt_annotations']
        domain_stats[domain]['frames'] += 1
    
    return {
        'total_frames': len(results),
        'total_detections': total_detections,
        'total_gt': total_gt,
        'avg_detections_per_frame': total_detections / len(results) if results else 0,
        'avg_gt_per_frame': total_gt / len(results) if results else 0,
        'domain_statistics': domain_stats
    }


def main():
    """Run smoke test"""
    print("=" * 80)
    print("SHIFT Vanilla VLM Baseline - Smoke Test")
    print("=" * 80)
    
    # Configuration
    DATA_ROOT = "/media/fardad/T7 Touch/SHIFT/shift-dev/data"  # Update this path
    NUM_SAMPLES = 20  # Small number for smoke test
    
    # Step 1: Load SHIFT dataset (small subset)
    print("\n[Step 1] Loading SHIFT dataset...")
    dataset = VLMSHIFTDataset(
        data_root=DATA_ROOT,
        split="val",
        filters={
            'weather': ['clear', 'overcast'],  # Filter to 2 weather conditions
            'max_frames_per_video': 10  # Only 10 frames per video
        }
    )
    print(f"✓ Loaded {len(dataset)} frames")
    
    # Get dataset statistics
    stats = dataset.get_domain_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Videos: {stats['total_videos']}")
    print(f"  Weather: {dict(stats['weather_distribution'])}")
    print(f"  Categories: {dict(stats['category_distribution'])}")
    
    # Step 2: Initialize detector
    print("\n[Step 2] Initializing VLM detector...")
    gpu_available = check_gpu_status()
    device = "cuda" if gpu_available else "cpu"
    
    detector = OWLv2Detector(
        model_name="google/owlv2-base-patch16-ensemble",
        device=device
    )
    print(f"✓ Detector ready on {device}")
    
    # Step 3: Run detection on samples
    print("\n[Step 3] Running detection...")
    results = run_detector_on_samples(
        detector=detector,
        dataset=dataset,
        num_samples=NUM_SAMPLES,
        target_classes=["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    )
    
    # Step 4: Compute basic metrics
    print("\n[Step 4] Computing metrics...")
    metrics = compute_basic_metrics(results)
    
    print("\n" + "=" * 80)
    print("SMOKE TEST RESULTS")
    print("=" * 80)
    print(f"Frames processed: {metrics['total_frames']}")
    print(f"Total detections: {metrics['total_detections']}")
    print(f"Total GT objects: {metrics['total_gt']}")
    print(f"Avg detections/frame: {metrics['avg_detections_per_frame']:.2f}")
    print(f"Avg GT/frame: {metrics['avg_gt_per_frame']:.2f}")
    
    print(f"\nDomain-wise Statistics:")
    for domain, stats in metrics['domain_statistics'].items():
        print(f"  {domain}:")
        print(f"    Frames: {stats['frames']}")
        print(f"    Detections: {stats['detections']}")
        print(f"    GT: {stats['gt']}")
        print(f"    Avg detections/frame: {stats['detections']/stats['frames']:.2f}")
    
    # Save results
    output_file = "shift_smoke_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✓ Smoke test completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review results to ensure detector is working")
    print("  2. Run full evaluation with more samples")
    print("  3. Compute mAP metrics using pycocotools")


if __name__ == "__main__":
    main()