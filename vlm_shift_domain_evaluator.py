#!/usr/bin/env python3
"""
VLM SHIFT Domain Evaluator
Evaluates VLM detector performance across different domain conditions (weather/time)
Similar to background evaluator but adapted for SHIFT dataset
"""

import os
import json
import time
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from copy import deepcopy
import cv2
from PIL import Image, ImageDraw, ImageFont

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False


class VLMSHIFTDomainEvaluator:
    """
    Domain-wise evaluator for VLM detectors on SHIFT dataset
    Groups GT and predictions by domain (weather × time) and evaluates separately
    """
    
    def __init__(self, dataset, output_dir: str = "./shift_evaluation_results"):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain configuration - SHIFT has weather × time combinations
        self.weather_types = ["clear", "overcast", "rainy", "foggy", "cloudy"]
        self.time_types = ["daytime", "dawn/dusk", "night"]
        
        # Create domain combinations
        self.domains = []
        for weather in self.weather_types:
            for time_period in self.time_types:
                self.domains.append(f"{weather}_{time_period}")
        
        # Evaluation data storage
        self.domain_data = {
            domain: {"annotations": [], "predictions": [], "images": {}}
            for domain in self.domains
        }
        
        # Overall data (all domains combined)
        self.domain_data["overall"] = {"annotations": [], "predictions": [], "images": {}}
        
        self._gt_processed = False
        
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools required. Install with: pip install pycocotools")
    
    def evaluate_detections(self, predictions: List[Dict[str, Any]], 
                           visualize: bool = True,
                           save_visualizations: int = 10) -> Dict[str, Any]:
        """
        Main evaluation function
        
        Args:
            predictions: List of prediction results
            visualize: Whether to create visualization images
            save_visualizations: Number of visualization images to save per domain
        """
        start_time = time.time()
        print("\n" + "="*80)
        print("SHIFT Domain-wise Evaluation")
        print("="*80)
        
        # Group predictions and GT by domain
        print("\n[1/5] Grouping annotations by domain...")
        self._group_by_domain(predictions)
        
        # Compute mAP metrics per domain
        print("\n[2/5] Computing mAP metrics...")
        results = self._compute_domain_metrics()
        
        # Add size-based analysis
        print("\n[3/5] Computing size-based metrics...")
        size_metrics = self._compute_size_metrics(predictions)
        results['size_analysis'] = size_metrics
        
        # Create visualizations
        if visualize:
            print(f"\n[4/5] Creating visualizations ({save_visualizations} per domain)...")
            self._create_visualizations(predictions, max_per_domain=save_visualizations)
        
        # Save comprehensive results
        print("\n[5/5] Saving results...")
        self._save_comprehensive_results(results)
        
        evaluation_time = time.time() - start_time
        print(f"\n✓ Domain evaluation completed in {evaluation_time:.2f}s")
        print(f"✓ Results saved to: {self.output_dir}")
        
        return results
    
    def _group_by_domain(self, predictions: List[Dict[str, Any]]):
        """Group predictions and GT by domain (weather × time)"""
        # Process ground truth (only once)
        if not self._gt_processed:
            print("  Processing ground truth annotations...")
            for sample in self.dataset:
                weather = sample['image_info']['weather_coarse']
                time_period = sample['image_info']['timeofday_coarse']
                domain = f"{weather}_{time_period}"
                
                # Store image info
                image_id = sample['image_info']['id']
                if domain in self.domain_data:
                    self.domain_data[domain]['images'][image_id] = {
                        'width': sample['image_info']['width'],
                        'height': sample['image_info']['height'],
                        'file_name': str(sample['image_path'].name),
                        'id': image_id
                    }
                    
                    # Store annotations
                    for anno in sample['annotations']:
                        anno_copy = anno.copy()
                        anno_copy['image_id'] = image_id
                        self.domain_data[domain]['annotations'].append(anno_copy)
                
                # Also add to overall
                self.domain_data['overall']['images'][image_id] = {
                    'width': sample['image_info']['width'],
                    'height': sample['image_info']['height'],
                    'file_name': str(sample['image_path'].name),
                    'id': image_id
                }
                for anno in sample['annotations']:
                    anno_copy = anno.copy()
                    anno_copy['image_id'] = image_id
                    self.domain_data['overall']['annotations'].append(anno_copy)
            
            self._gt_processed = True
            print(f"  ✓ Processed {len(self.dataset)} samples")
        
        # Process predictions
        print("  Processing predictions...")
        for pred in predictions:
            image_id = pred['image_id']
            
            # Find domain for this image
            sample = next((s for s in self.dataset if s['image_info']['id'] == image_id), None)
            if sample is None:
                continue
            
            weather = sample['image_info']['weather_coarse']
            time_period = sample['image_info']['timeofday_coarse']
            domain = f"{weather}_{time_period}"
            
            # Convert predictions to COCO format
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                # Get category_id
                cat_id = self._get_category_id(label)
                
                pred_dict = {
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': box,  # [x1, y1, x2, y2]
                    'score': float(score)
                }
                
                if domain in self.domain_data:
                    self.domain_data[domain]['predictions'].append(pred_dict)
                self.domain_data['overall']['predictions'].append(pred_dict)
        
        # Print statistics
        print(f"\n  Domain Statistics:")
        for domain in self.domains + ['overall']:
            n_images = len(self.domain_data[domain]['images'])
            n_gt = len(self.domain_data[domain]['annotations'])
            n_pred = len(self.domain_data[domain]['predictions'])
            if n_images > 0:
                print(f"    {domain:30s}: {n_images:4d} images, {n_gt:5d} GT, {n_pred:5d} predictions")
    
    def _get_category_id(self, label: str) -> int:
        """Map label name to category ID"""
        category_map = {
            "pedestrian": 1,
            "car": 2,
            "truck": 3,
            "bus": 4,
            "motorcycle": 5,
            "bicycle": 6
        }
        return category_map.get(label.lower(), 1)
    
    def _compute_domain_metrics(self) -> Dict[str, Any]:
        """Compute mAP metrics for each domain"""
        results = {}
        
        for domain in self.domains + ['overall']:
            annotations = self.domain_data[domain]['annotations']
            predictions = self.domain_data[domain]['predictions']
            images = self.domain_data[domain]['images']
            
            if not annotations or not images:
                continue
            
            print(f"  Evaluating {domain}...")
            domain_results = self._evaluate_single_domain(domain, annotations, predictions, images)
            
            if domain_results:
                results[domain] = domain_results
        
        return results
    
    def _evaluate_single_domain(self, domain_name: str, 
                                annotations: List[Dict],
                                predictions: List[Dict],
                                images: Dict) -> Optional[Dict[str, Any]]:
        """Evaluate single domain using COCO metrics"""
        
        # Create COCO-format dataset with required fields
        coco_gt = {
            'info': {
                'description': f'SHIFT {domain_name} Domain',
                'version': '1.0',
                'year': 2024
            },
            'licenses': [],
            'images': list(images.values()),
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'pedestrian'},
                {'id': 2, 'name': 'car'},
                {'id': 3, 'name': 'truck'},
                {'id': 4, 'name': 'bus'},
                {'id': 5, 'name': 'motorcycle'},
                {'id': 6, 'name': 'bicycle'}
            ]
        }
        
        # Add annotation IDs
        for idx, anno in enumerate(annotations):
            anno_copy = anno.copy()
            anno_copy['id'] = idx
            # Convert bbox from [x1,y1,x2,y2] to [x,y,w,h] for COCO
            x1, y1, x2, y2 = anno_copy['bbox']
            anno_copy['bbox'] = [x1, y1, x2-x1, y2-y1]
            coco_gt['annotations'].append(anno_copy)
        
        # Save GT to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt, f)
            gt_file = f.name
        
        try:
            # Load COCO GT
            coco = COCO(gt_file)
            
            # Convert predictions to COCO format
            coco_preds = []
            for pred in predictions:
                pred_copy = pred.copy()
                # Convert bbox from [x1,y1,x2,y2] to [x,y,w,h]
                x1, y1, x2, y2 = pred_copy['bbox']
                pred_copy['bbox'] = [x1, y1, x2-x1, y2-y1]
                coco_preds.append(pred_copy)
            
            if not coco_preds:
                return None
            
            # Run COCO evaluation
            coco_dt = coco.loadRes(coco_preds)
            coco_eval = COCOeval(coco, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'mAP': float(coco_eval.stats[0]),  # AP @ IoU=0.5:0.95
                'mAP_50': float(coco_eval.stats[1]),  # AP @ IoU=0.5
                'mAP_75': float(coco_eval.stats[2]),  # AP @ IoU=0.75
                'mAP_small': float(coco_eval.stats[3]),  # AP for small objects
                'mAP_medium': float(coco_eval.stats[4]),  # AP for medium objects
                'mAP_large': float(coco_eval.stats[5]),  # AP for large objects
                'AR_1': float(coco_eval.stats[6]),  # AR with 1 detection
                'AR_10': float(coco_eval.stats[7]),  # AR with 10 detections
                'AR_100': float(coco_eval.stats[8]),  # AR with 100 detections
                'num_images': len(images),
                'num_gt': len(annotations),
                'num_predictions': len(predictions)
            }
            
            return metrics
            
        finally:
            # Clean up temp file
            os.unlink(gt_file)
    
    def _compute_size_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Compute metrics by object size"""
        # COCO size definitions (in pixels²)
        SMALL_THRESHOLD = 32 * 32  # < 1024 pixels
        MEDIUM_THRESHOLD = 96 * 96  # < 9216 pixels
        
        size_stats = {
            'small': {'count': 0, 'total_area': 0},
            'medium': {'count': 0, 'total_area': 0},
            'large': {'count': 0, 'total_area': 0}
        }
        
        for sample in self.dataset:
            for anno in sample['annotations']:
                area = anno['area']
                if area < SMALL_THRESHOLD:
                    size_stats['small']['count'] += 1
                    size_stats['small']['total_area'] += area
                elif area < MEDIUM_THRESHOLD:
                    size_stats['medium']['count'] += 1
                    size_stats['medium']['total_area'] += area
                else:
                    size_stats['large']['count'] += 1
                    size_stats['large']['total_area'] += area
        
        # Compute averages
        for size_cat in ['small', 'medium', 'large']:
            count = size_stats[size_cat]['count']
            if count > 0:
                size_stats[size_cat]['avg_area'] = size_stats[size_cat]['total_area'] / count
            else:
                size_stats[size_cat]['avg_area'] = 0
        
        return size_stats
    
    def _create_visualizations(self, predictions: List[Dict], max_per_domain: int = 10):
        """Create visualization images with detection boxes"""
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Group predictions by domain
        domain_vis_counts = defaultdict(int)
        
        for pred in predictions:
            image_id = pred['image_id']
            
            # Find sample
            sample = next((s for s in self.dataset if s['image_info']['id'] == image_id), None)
            if sample is None:
                continue
            
            weather = sample['image_info']['weather_coarse']
            time_period = sample['image_info']['timeofday_coarse']
            domain = f"{weather}_{time_period}"
            
            # Check if we've saved enough for this domain
            if domain_vis_counts[domain] >= max_per_domain:
                continue
            
            # Load image
            image_path = sample['image_path']
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Draw GT boxes (green)
            for anno in sample['annotations']:
                x1, y1, x2, y2 = [int(c) for c in anno['bbox']]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, anno['category_name'], (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw prediction boxes (red)
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                x1, y1, x2, y2 = [int(c) for c in box]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = f"{label}: {score:.2f}"
                cv2.putText(image, text, (x1, y1-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Add domain info
            text = f"{domain} | GT: {len(sample['annotations'])} | Pred: {len(pred['boxes'])}"
            cv2.putText(image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Save
            filename = f"{domain}_{domain_vis_counts[domain]:03d}_{sample['image_info']['video_name']}_{sample['image_info']['frame_index']:06d}.jpg"
            save_path = vis_dir / filename
            cv2.imwrite(str(save_path), image)
            
            domain_vis_counts[domain] += 1
        
        print(f"  ✓ Saved {sum(domain_vis_counts.values())} visualization images to {vis_dir}")
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save detailed results to JSON"""
        
        # Main results file
        results_file = self.output_dir / "domain_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary_file = self.output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SHIFT Domain Evaluation Summary\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            if 'overall' in results:
                f.write("Overall Performance:\n")
                f.write(f"  mAP @ IoU=0.5:0.95: {results['overall']['mAP']:.4f}\n")
                f.write(f"  mAP @ IoU=0.50:     {results['overall']['mAP_50']:.4f}\n")
                f.write(f"  mAP @ IoU=0.75:     {results['overall']['mAP_75']:.4f}\n")
                f.write(f"  mAP (small):        {results['overall']['mAP_small']:.4f}\n")
                f.write(f"  mAP (medium):       {results['overall']['mAP_medium']:.4f}\n")
                f.write(f"  mAP (large):        {results['overall']['mAP_large']:.4f}\n\n")
            
            # Per-domain metrics
            f.write("Per-Domain Performance:\n")
            f.write(f"{'Domain':<30s} {'mAP':>8s} {'mAP@50':>8s} {'Images':>8s} {'GT':>8s} {'Pred':>8s}\n")
            f.write("-"*80 + "\n")
            
            for domain in sorted(results.keys()):
                if domain == 'overall' or domain == 'size_analysis':
                    continue
                metrics = results[domain]
                f.write(f"{domain:<30s} "
                       f"{metrics['mAP']:>8.4f} "
                       f"{metrics['mAP_50']:>8.4f} "
                       f"{metrics['num_images']:>8d} "
                       f"{metrics['num_gt']:>8d} "
                       f"{metrics['num_predictions']:>8d}\n")
            
            # Size analysis
            if 'size_analysis' in results:
                f.write("\n\nObject Size Distribution:\n")
                for size_cat in ['small', 'medium', 'large']:
                    stats = results['size_analysis'][size_cat]
                    f.write(f"  {size_cat.capitalize()}:  {stats['count']:6d} objects, "
                           f"avg area: {stats['avg_area']:.1f} px²\n")
        
        print(f"  ✓ Saved results to {results_file}")
        print(f"  ✓ Saved summary to {summary_file}")


def create_shift_domain_evaluator(dataset, output_dir: str = "./shift_evaluation_results"):
    """Factory function"""
    return VLMSHIFTDomainEvaluator(dataset, output_dir)