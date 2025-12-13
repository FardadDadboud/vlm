#!/usr/bin/env python3
"""
SHIFT Comprehensive Evaluation Script
Runs full baseline VLM evaluation with mAP metrics, visualizations, and logs
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import time

# Import components
from vlm_shift_dataset import VLMSHIFTDataset
from vlm_shift_domain_evaluator import VLMSHIFTDomainEvaluator
from vlm_detector_system_new import OWLv2Detector, GroundingDINODetector, check_gpu_status
from bca_plus_adapter import BCAPlusAdapter


def run_detector_on_dataset(detector, dataset, output_dir: Path, 
                            target_classes: List[str] = None,
                            max_samples: int = None):
    """
    Run detector on dataset and return predictions in COCO format
    """
    if target_classes is None:
        target_classes = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    
    predictions = []
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"\nRunning {detector.model_path} on {num_samples} samples...")
    print(f"Target classes: {target_classes}\n")
    
    start_time = time.time()
    
    for i in range(num_samples):
        sample = dataset[i]
        image_path = sample['image_path']
        
        # Load image
        image = Image.open(image_path)
        
        # FIXED: Use detector directly (it's already wrapped with BCA+)
        detection_result = detector.adapt_and_detect(image, target_classes, threshold=0.50)

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
            print(f"  Progress: {i+1}/{num_samples} ({100*(i+1)/num_samples:.1f}%) | "
                  f"FPS: {fps:.2f} | ETA: {eta:.1f}s | Cache size: {detector.cache.M}")
    
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
    
    # Configuration
    CONFIG = {
        'data_root': "/media/fardad/T7 Touch/SHIFT/shift-dev/data",  # UPDATE THIS
        'split': 'val',
        'filters': {
            'weather': ['clear'],  # Test 3 weather conditions
            'time': ['daytime'],  # Daytime only for initial test
            'max_frames_per_video': None  # 50 frames per video
        },
        "max_samples": None,
        'detector': {
            'name': 'grounding-dino',
            'model_path': 'IDEA-Research/grounding-dino-base'
        },
        'output_dir': './shift_comprehensive_eval_results',
        'num_visualizations': 1000  # Visualizations per domain
    }
    
    print("="*80)
    print("SHIFT Comprehensive Baseline Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(json.dumps(CONFIG, indent=2))
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
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
    detector = BCAPlusAdapter(
        detector=base_detector,
        tau1=0.85,
        tau2=0.85,
        ws=0.3,
        num_classes=6
    )

    print(f"\n✓ Detector ready: {detector.model_path}")
    print(f"✓ BCA+ adapter configured: tau1={detector.tau1}, tau2={detector.tau2}, ws={detector.ws}")
    
    # Step 3: Run detection
    print("\n" + "="*80)
    print("[Step 3/4] Running Detection")
    print("="*80)
    
    predictions = run_detector_on_dataset(
        detector=detector,
        dataset=dataset,
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