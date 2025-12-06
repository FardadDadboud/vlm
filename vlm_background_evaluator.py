#!/usr/bin/env python3
"""
VLM Background-wise Evaluator
Evaluates VLM detector performance separately for each background type
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

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False


class VLMBackgroundEvaluator:
    """
    Background-wise evaluator for VLM detectors
    Groups GT and predictions by background overlap and evaluates separately
    """
    
    def __init__(self, dataset, output_dir: str = "./vlm_evaluation_results"):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Background configuration
        self.background_types = ["sky", "tree", "ground"]
        self.overlap_threshold = 0.5

        # Add GT processing flag
        self._gt_processed = False
        
        # Evaluation data storage
        # Store image dimensions for accurate COCO evaluation
        self.bg_data = {
            bg_type: {"annotations": [], "predictions": [], "images": {}}
            for bg_type in self.background_types
        }
        
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools required for evaluation. Install with: pip install pycocotools")
    
    def evaluate_detections(self, predictions: List[Dict[str, Any]],
                        expand_boxes: bool = False,  # Default to False
                        scale_factor_h: float = 1.2, scale_factor_w: float = 1.2,
                        reset_predictions: bool = True,
                        size_aware_thresholds: bool = True) -> Dict[str, Any]:
        """
        Main evaluation function with size-aware threshold filtering
        
        Args:
            predictions: List of prediction results from VLM detectors
            expand_boxes: Whether to expand prediction boxes (deprecated)
            reset_predictions: Whether to clear previous predictions
            size_aware_thresholds: Apply different confidence thresholds based on object size
        """
        start_time = time.time()
        print("Starting VLM background-wise evaluation...")
        
        # Clear previous predictions if needed (but keep GT)
        if reset_predictions:
            for bg_type in self.background_types:
                self.bg_data[bg_type]["predictions"] = []
        
        # Apply size-aware threshold filtering before evaluation
        if size_aware_thresholds:
            predictions = self._apply_size_aware_filtering(predictions)
        
        # Group predictions and GT by background
        self._group_by_background(predictions)
        
        # Run evaluation for each background with standard IoU (0.5)
        results = {}
        
        for bg_type in self.background_types:
            print(f"Evaluating {bg_type} background...")
            bg_results = self._evaluate_background_single_iou(bg_type, iou_threshold=0.5)
            if bg_results:
                results[f"{bg_type}_bbox"] = bg_results['bbox']
                results[f"{bg_type}_recall"] = bg_results['recall'] 
                results[f"{bg_type}_samples"] = len(self.bg_data[bg_type]["images"])
        
        # Compute overall statistics
        overall_results = self._compute_overall_statistics(results)
        results.update(overall_results)
        
        # Save and visualize results
        self._save_results(results)
        
        evaluation_time = time.time() - start_time
        print(f"Background-wise evaluation completed in {evaluation_time:.2f}s")
        
        return results

    def _apply_size_aware_filtering(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply size-aware confidence thresholds"""
        filtered_predictions = []
        
        for pred in predictions:
            filtered_boxes, filtered_scores, filtered_labels = [], [], []
            
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Size-adaptive thresholds
                    if area < 1024:  # Small objects (< 32x32 pixels)
                        min_threshold = 0.4
                    elif area < 9216:  # Medium objects (< 96x96 pixels)  
                        min_threshold = 0.25
                    else:  # Large objects
                        min_threshold = 0.15
                    
                    # Also filter out unreasonably large detections (likely whole-image errors)
                    image_area = 1024 * 1024  # Assume roughly 1MP images
                    if area > 0.5 * image_area:  # Reject detections > 50% of image
                        continue
                    
                    if score >= min_threshold:
                        filtered_boxes.append(box)
                        filtered_scores.append(score)
                        filtered_labels.append(label)
            
            filtered_pred = pred.copy()
            filtered_pred['boxes'] = filtered_boxes
            filtered_pred['scores'] = filtered_scores  
            filtered_pred['labels'] = filtered_labels
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions

    def _evaluate_background_multi_iou(self, bg_type: str, iou_thresholds: List[float]) -> Optional[Dict[str, Any]]:
        """Evaluate single background type using multiple IoU thresholds"""
        annotations = self.bg_data[bg_type]["annotations"]
        predictions = self.bg_data[bg_type]["predictions"]
        images = self.bg_data[bg_type]["images"]
        
        if not annotations:
            return None
        
        try:
            return self._run_coco_evaluation_multi_iou(images, annotations, predictions, bg_type, iou_thresholds)
        except Exception as e:
            print(f"Evaluation failed for {bg_type}: {e}")
            return None

    def _run_coco_evaluation_multi_iou(self, images: Dict[int, Tuple[int, int]], annotations: List[Dict],
                                    predictions: List[Dict], bg_type: str, iou_thresholds: List[float]) -> Dict[str, Any]:
        """Run COCO evaluation with multiple IoU thresholds"""
        # Create image entries with actual dimensions
        image_entries = []
        for img_id, (width, height) in images.items():
            image_entries.append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": f"{img_id}.jpg"
            })
        
        # Create COCO format data
        coco_gt_data = {
            "info": {
                "description": "VLM Background Evaluation",
                "version": "1.0",
                "year": 2025
            },
            "images": image_entries,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "drone"}],
            "licenses": []
        }
        
        # Write temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
            json.dump(coco_gt_data, gt_file)
            gt_file_path = gt_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
            json.dump(predictions, pred_file)
            pred_file_path = pred_file.name
        
        try:
            # Load COCO and run evaluation
            coco_gt = COCO(gt_file_path)
            
            if predictions:
                coco_dt = coco_gt.loadRes(pred_file_path)
            else:
                coco_dt = coco_gt.loadRes([])
            
            # Evaluate at multiple IoU thresholds
            results = {'bbox': {}, 'recall': {}}
            
            for iou_thresh in iou_thresholds:
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                
                # Set custom IoU threshold
                coco_eval.params.iouThrs = np.array([iou_thresh])
                coco_eval.evaluate()
                coco_eval.accumulate()
                
                # Suppress output
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    coco_eval.summarize()
                finally:
                    sys.stdout = old_stdout
                
                # Extract metrics
                if coco_eval.stats is not None and len(coco_eval.stats) >= 9:
                    ap = coco_eval.stats[0] * 100  # AP at specified IoU
                    ar1 = coco_eval.stats[6] * 100  # AR@1
                    ar10 = coco_eval.stats[7] * 100  # AR@10
                    ar100 = coco_eval.stats[8] * 100  # AR@100
                    
                    results['bbox'][f'AP@{iou_thresh}'] = ap
                    results['recall'][f'AR@{iou_thresh}@1'] = ar1
                    results['recall'][f'AR@{iou_thresh}@10'] = ar10
                    results['recall'][f'AR@{iou_thresh}@100'] = ar100
            
            # For backward compatibility, use AP@0.5 as main AP if available
            if 'AP@0.5' in results['bbox']:
                results['bbox']['AP'] = results['bbox']['AP@0.5']
                results['bbox']['AP50'] = results['bbox']['AP@0.5']
                results['recall']['AR@1'] = results['recall']['AR@0.5@1']
                results['recall']['AR@10'] = results['recall']['AR@0.5@10']
                results['recall']['AR@100'] = results['recall']['AR@0.5@100']
            elif 'AP@0.1' in results['bbox']:
                # Use lowest IoU as fallback
                results['bbox']['AP'] = results['bbox']['AP@0.1']
                results['bbox']['AP50'] = results['bbox'].get('AP@0.5', results['bbox']['AP@0.1'])
                results['recall']['AR@1'] = results['recall']['AR@0.1@1']
                results['recall']['AR@10'] = results['recall']['AR@0.1@10']
                results['recall']['AR@100'] = results['recall']['AR@0.1@100']
            
            return results
        
        finally:
            # Clean up temp files
            os.unlink(gt_file_path)
            os.unlink(pred_file_path)
    
    def _expand_prediction_boxes(self, predictions: List[Dict[str, Any]], 
                           scale_factor_h: float = 1.2, scale_factor_w: float = 1.2) -> List[Dict[str, Any]]:
        """Expand prediction boxes while keeping center point fixed"""
        expanded_predictions = []
        
        for pred in predictions:
            expanded_pred = deepcopy(pred)
            expanded_boxes = []
            
            for box in pred['boxes']:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                old_w = x2 - x1
                old_h = y2 - y1
                new_w = old_w * scale_factor_w
                new_h = old_h * scale_factor_h
                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2
                expanded_boxes.append([new_x1, new_y1, new_x2, new_y2])
            expanded_pred['boxes'] = expanded_boxes
            expanded_predictions.append(expanded_pred)
        
        return expanded_predictions
    
    def _group_by_background(self, predictions: List[Dict[str, Any]]):
        """Group predictions and GT annotations by background overlap"""
        print("Grouping samples by background...")
        
        # Process GT annotations only once
        if not self._gt_processed:
            print("Processing GT annotations...")
            self._process_gt_annotations()
            self._gt_processed = True
        
        # Always process predictions
        print("Processing predictions...")
        self._process_predictions(predictions)
        
        # Print grouping statistics
        for bg_type in self.background_types:
            gt_count = len(self.bg_data[bg_type]["annotations"])
            pred_count = len(self.bg_data[bg_type]["predictions"])
            img_count = len(self.bg_data[bg_type]["images"])
            print(f"{bg_type.upper()}: {img_count} images, {gt_count} GT, {pred_count} predictions")

    def _process_gt_annotations(self):
        """Process GT annotations once and group by background"""
        for idx in range(len(self.dataset)):
            sample = self.dataset.get_sample(idx)
            image_info = sample['image_info']
            
            # Get background masks
            bg_masks = self.dataset.create_background_masks(sample)
            if not bg_masks:
                continue
            
            # Group GT annotations
            grouped_gt = self._group_annotations_by_background(
                sample['annotations'], bg_masks, 
                image_info['height'], image_info['width']
            )
            
            # Store data for each background
            for bg_type in self.background_types:
                if bg_type in grouped_gt:
                    # Add image info with dimensions (only if has GT)
                    self.bg_data[bg_type]["images"][image_info['id']] = (
                        image_info['width'], image_info['height']
                    )
                    
                    # Add GT annotations
                    for ann in grouped_gt[bg_type]:
                        ann_entry = {
                            "id": len(self.bg_data[bg_type]["annotations"]) + 1,
                            "image_id": image_info['id'],
                            "category_id": 0,
                            "bbox": ann['bbox'],
                            "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                            "iscrowd": ann.get('iscrowd', 0)
                        }
                        self.bg_data[bg_type]["annotations"].append(ann_entry)

    def _process_predictions(self, predictions: List[Dict[str, Any]]):
        """Process predictions and group by background"""
        # Create image_id to prediction mapping
        pred_by_image = {pred['image_id']: pred for pred in predictions}
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.get_sample(idx)
            image_info = sample['image_info']
            image_id = str(image_info['file_name']).split('.')[0]
            
            # Skip if no predictions for this image
            if image_id not in pred_by_image:
                continue
                
            # Get background masks
            bg_masks = self.dataset.create_background_masks(sample)
            if not bg_masks:
                continue
            
            # Group predictions
            pred_result = pred_by_image[image_id]
            grouped_pred = self._group_predictions_by_background(
                pred_result, bg_masks,
                image_info['height'], image_info['width']
            )
            
            # Add predictions to each background
            for bg_type in self.background_types:
                if bg_type in grouped_pred:
                    for pred in grouped_pred[bg_type]:
                        pred_entry = {
                            "image_id": image_info['id'],
                            "category_id": 0,
                            "bbox": pred['bbox'],
                            "score": pred['score']
                        }
                        self.bg_data[bg_type]["predictions"].append(pred_entry)
    
    def _group_annotations_by_background(self, annotations: List[Dict], bg_masks: Dict[str, np.ndarray],
                                        height: int, width: int) -> Dict[str, List[Dict]]:
        """Group GT annotations by background overlap"""
        grouped = {}
        
        for ann in annotations:
            bbox = ann.get("bbox", [])
            if not bbox:
                continue
            
            overlaps = self._calculate_bbox_background_overlap(bbox, bg_masks, height, width)
            
            if overlaps:
                best_bg = max(overlaps.items(), key=lambda x: x[1])
                if best_bg[1] > self.overlap_threshold:
                    bg_type = best_bg[0]
                    if bg_type not in grouped:
                        grouped[bg_type] = []
                    grouped[bg_type].append(ann)
        
        return grouped
    
    def _group_predictions_by_background(self, pred_result: Dict, bg_masks: Dict[str, np.ndarray],
                                        height: int, width: int) -> Dict[str, List[Dict]]:
        """Group predictions by background overlap"""
        grouped = {}
        
        boxes = pred_result.get('boxes', [])
        scores = pred_result.get('scores', [])
        labels = pred_result.get('labels', [])
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Convert box format if needed
            if len(box) == 4:
                x1, y1, x2, y2 = box
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # Convert to COCO format
                
                overlaps = self._calculate_bbox_background_overlap(bbox, bg_masks, height, width)
                
                if overlaps:
                    best_bg = max(overlaps.items(), key=lambda x: x[1])
                    if best_bg[1] > self.overlap_threshold:
                        bg_type = best_bg[0]
                        if bg_type not in grouped:
                            grouped[bg_type] = []
                        
                        grouped[bg_type].append({
                            # pred['bbox'] = [round(coord) for coord in pred['bbox']]
                            "bbox": [round(coord) for coord in bbox],
                            "score": float(score),
                            "label": label
                        })
        
        return grouped
    
    def _calculate_bbox_background_overlap(self, bbox: List[float], bg_masks: Dict[str, np.ndarray],
                                          height: int, width: int) -> Dict[str, float]:
        """Calculate overlap between bbox and background masks"""
        try:
            # Handle COCO format [x, y, w, h]
            x, y, w, h = bbox
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2, y2 = int(min(width, x + w)), int(min(height, y + h))
            
            if x2 <= x1 or y2 <= y1:
                return {}
            
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area == 0:
                return {}
            
            overlaps = {}
            for bg_type, bg_mask in bg_masks.items():
                bbox_region = bg_mask[y1:y2, x1:x2]
                overlap_area = np.sum(bbox_region)
                overlap_ratio = overlap_area / bbox_area
                overlaps[bg_type] = overlap_ratio
            
            return overlaps
        
        except Exception as e:
            print(f"Error calculating bbox overlap: {e}")
            return {}
    
    def _evaluate_background(self, bg_type: str) -> Optional[Dict[str, Any]]:
        """Evaluate single background type using COCO metrics"""
        annotations = self.bg_data[bg_type]["annotations"]
        predictions = self.bg_data[bg_type]["predictions"]
        images = self.bg_data[bg_type]["images"]
        
        if not annotations:
            return None
        
        try:
            # Add this before creating coco_gt_data
            print(f"DEBUG: First few GT boxes: {[ann['bbox'] for ann in annotations[:3]]}")
            print(f"DEBUG: First few pred boxes: {[pred['bbox'] for pred in predictions[:3]]}")
            print(f"DEBUG: GT box types: {[type(coord) for coord in annotations[0]['bbox']]}")
            print(f"DEBUG: Pred box types: {[type(coord) for coord in predictions[0]['bbox']]}")
            return self._run_coco_evaluation(images, annotations, predictions, bg_type)
        except Exception as e:
            print(f"Evaluation failed for {bg_type}: {e}")
            print(f"DEBUG: COCO evaluation exception: {str(e)}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_coco_evaluation(self, images: Dict[int, Tuple[int, int]], annotations: List[Dict],
                           predictions: List[Dict], bg_type: str) -> Dict[str, Any]:
        """Run COCO evaluation for background subset"""
        # DEBUG: Check inputs
        print(f"DEBUG: COCO eval for {bg_type}")
        print(f"DEBUG: Images: {len(images)}")
        print(f"DEBUG: Annotations: {len(annotations)}")
        print(f"DEBUG: Predictions: {len(predictions)}")
        
        if annotations:
            print(f"DEBUG: Sample annotation: {annotations[0]}")
        if predictions:
            print(f"DEBUG: Sample prediction: {predictions[0]}")
        # Create image entries with actual dimensions
        image_entries = []
        for img_id, (width, height) in images.items():
            image_entries.append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": f"{img_id}.jpg"
            })
        
        # Create COCO format data
        # NEW (with required info):
        coco_gt_data = {
            "info": {
                "description": "VLM Background Evaluation",
                "version": "1.0",
                "year": 2025
            },
            "images": image_entries,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "drone"}],
            "licenses": []
        }
        # Write temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
            json.dump(coco_gt_data, gt_file)
            gt_file_path = gt_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
            json.dump(predictions, pred_file)
            pred_file_path = pred_file.name
        
        try:
            # Load COCO and run evaluation
            coco_gt = COCO(gt_file_path)
            
            if predictions:
                coco_dt = coco_gt.loadRes(pred_file_path)
            else:
                coco_dt = coco_gt.loadRes([])
            
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()

            # DEBUG: Check if evaluation succeeded
            if coco_eval.stats is None:
                print(f"DEBUG: COCO evaluation failed - stats is None")
                print(f"DEBUG: GT count: {len(annotations)}, Pred count: {len(predictions)}")
                return {
                    'bbox': {'AP': 0, 'AP50': 0, 'AP75': 0},
                    'recall': {'AR@1': 0, 'AR@10': 0, 'AR@100': 0}
                }
            
            # Suppress output and get results
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                coco_eval.summarize()
            finally:
                sys.stdout = old_stdout
            
            # Extract metrics with safety checks
            stats = coco_eval.stats
            if len(stats) < 9:
                print(f"DEBUG: Insufficient stats length: {len(stats)}")
                return {
                    'bbox': {'AP': 0, 'AP50': 0, 'AP75': 0},
                    'recall': {'AR@1': 0, 'AR@10': 0, 'AR@100': 0}
                }
            ap = coco_eval.stats[0] * 100  # AP@0.5:0.95
            ap50 = coco_eval.stats[1] * 100  # AP@0.5
            ap75 = coco_eval.stats[2] * 100  # AP@0.75
            ar1 = coco_eval.stats[6] * 100  # AR@1
            ar10 = coco_eval.stats[7] * 100  # AR@10
            ar100 = coco_eval.stats[8] * 100  # AR@100
            
            return {
                'bbox': {'AP': ap, 'AP50': ap50, 'AP75': ap75},
                'recall': {'AR@1': ar1, 'AR@10': ar10, 'AR@100': ar100}
            }
        
        finally:
            # Clean up temp files
            os.unlink(gt_file_path)
            os.unlink(pred_file_path)
    
    def _compute_overall_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall statistics across backgrounds"""
        overall = {}
        
        # Collect AP values
        aps = []
        ap50s = []
        recalls = []
        weights = []
        
        for bg_type in self.background_types:
            bbox_key = f"{bg_type}_bbox"
            recall_key = f"{bg_type}_recall"
            
            if bbox_key in results:
                aps.append(results[bbox_key]['AP'])
                ap50s.append(results[bbox_key]['AP50'])
                
                # Use GT count as weight
                gt_count = len(self.bg_data[bg_type]["annotations"])
                weights.append(gt_count)
            
            if recall_key in results:
                recalls.append(results[recall_key]['AR@100'])
        
        if aps and weights:
            # Weighted means
            weights_array = np.array(weights)
            normalized_weights = weights_array / np.sum(weights_array)
            
            overall["background_wise_weighted_mean_AP"] = np.average(aps, weights=normalized_weights)
            overall["background_wise_weighted_mean_AP50"] = np.average(ap50s, weights=normalized_weights)
            overall["background_wise_mean_AP"] = np.mean(aps)
            overall["background_wise_mean_AP50"] = np.mean(ap50s)
            
            # Overall metrics for compatibility
            overall["overall_mAP"] = overall["background_wise_weighted_mean_AP"]
            overall["mAP50"] = overall["background_wise_weighted_mean_AP50"]
        
        if recalls:
            overall["background_wise_mean_recall"] = np.mean(recalls)
            overall["overall_recall"] = np.mean(recalls)
        
        return overall
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results with comprehensive outputs"""
        try:
            # Create output subdirectories
            (self.output_dir / "metrics").mkdir(exist_ok=True)
            (self.output_dir / "visualizations").mkdir(exist_ok=True)
            (self.output_dir / "sample_images").mkdir(exist_ok=True)
            (self.output_dir / "logs").mkdir(exist_ok=True)
            
            # Save main results
            results_file = self.output_dir / "metrics" / "background_evaluation_results.json"
            
            results_with_metadata = {
                "evaluation_results": results,
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_samples": len(self.dataset),
                    "background_types": self.background_types,
                    "overlap_threshold": self.overlap_threshold
                },
                "background_data_counts": {
                    bg_type: {
                        "images": len(data["images"]),
                        "annotations": len(data["annotations"]),
                        "predictions": len(data["predictions"])
                    }
                    for bg_type, data in self.bg_data.items()
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            
            # Create visualizations
            self._create_background_visualizations(results)
            
            # Save sample images with predictions
            self._save_sample_predictions()
            
            print(f"Results and visualizations saved to {self.output_dir}")
        
        except Exception as e:
            print(f"Failed to save results: {e}")

    def _create_background_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive background analysis plots"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            viz_dir = self.output_dir / "visualizations"
            
            # 1. Background Performance Comparison
            self._plot_background_performance(results, viz_dir)
            
            # 2. Background Distribution
            self._plot_background_distribution(viz_dir)
            
            # 3. Performance vs Sample Count Scatter
            self._plot_performance_vs_samples(results, viz_dir)
            
        except Exception as e:
            print(f"Failed to create visualizations: {e}")

    def _plot_background_performance(self, results: Dict[str, Any], viz_dir: Path):
        """Plot background performance comparison"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        backgrounds = []
        aps = []
        ap50s = []
        recalls = []
        
        for bg_type in self.background_types:
            bbox_key = f"{bg_type}_bbox"
            recall_key = f"{bg_type}_recall"
            
            if bbox_key in results:
                backgrounds.append(bg_type.title())
                aps.append(results[bbox_key]['AP'])
                ap50s.append(results[bbox_key]['AP50'])
                
                if recall_key in results:
                    recalls.append(results[recall_key]['AR@100'])
                else:
                    recalls.append(0)
        
        if not backgrounds:
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Background-wise Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. mAP Comparison
        x = np.arange(len(backgrounds))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, aps, width, label='mAP@0.5:0.95', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, ap50s, width, label='mAP@0.5', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Background Type')
        ax1.set_ylabel('mAP (%)')
        ax1.set_title('Background-wise mAP Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(backgrounds)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 2. Recall Comparison
        if recalls:
            bars3 = ax2.bar(backgrounds, recalls, alpha=0.8, color='lightgreen')
            ax2.set_xlabel('Background Type')
            ax2.set_ylabel('Recall@100 (%)')
            ax2.set_title('Background-wise Recall Comparison')
            ax2.grid(True, alpha=0.3)
            
            for bar in bars3:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # 3. Performance Radar Chart
        if len(backgrounds) >= 3:
            angles = np.linspace(0, 2*np.pi, len(backgrounds), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            aps_radar = aps + [aps[0]]
            ap50s_radar = ap50s + [ap50s[0]]
            
            ax3 = plt.subplot(2, 2, 3, projection='polar')
            ax3.plot(angles, aps_radar, 'o-', linewidth=2, label='mAP@0.5:0.95', color='blue')
            ax3.fill(angles, aps_radar, alpha=0.25, color='blue')
            ax3.plot(angles, ap50s_radar, 'o-', linewidth=2, label='mAP@0.5', color='red')
            ax3.fill(angles, ap50s_radar, alpha=0.25, color='red')
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(backgrounds)
            ax3.set_ylim(0, max(max(aps), max(ap50s)) * 1.1)
            ax3.set_title('Performance Radar Chart')
            ax3.legend()
        
        # 4. Performance Summary Table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        headers = ['Background', 'mAP@0.5:0.95', 'mAP@0.5', 'Recall@100', 'Samples']
        
        for i, bg in enumerate(backgrounds):
            bg_type = bg.lower()
            samples = len(self.bg_data[bg_type]["images"])
            recall_val = recalls[i] if i < len(recalls) else 0
            
            table_data.append([
                bg, f'{aps[i]:.2f}%', f'{ap50s[i]:.2f}%', 
                f'{recall_val:.2f}%', str(samples)
            ])
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary Table')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "background_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_background_distribution(self, viz_dir: Path):
        """Plot background sample and annotation distribution"""
        import matplotlib.pyplot as plt
        
        backgrounds = []
        sample_counts = []
        annotation_counts = []
        
        for bg_type in self.background_types:
            backgrounds.append(bg_type.title())
            sample_counts.append(len(self.bg_data[bg_type]["images"]))
            annotation_counts.append(len(self.bg_data[bg_type]["annotations"]))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Background Data Distribution', fontsize=14, fontweight='bold')
        
        # Sample distribution
        bars1 = ax1.bar(backgrounds, sample_counts, alpha=0.8, color='lightblue')
        ax1.set_xlabel('Background Type')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Sample Distribution')
        ax1.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts) * 0.01,
                    str(sample_counts[i]), ha='center', va='bottom')
        
        # Annotation distribution
        bars2 = ax2.bar(backgrounds, annotation_counts, alpha=0.8, color='lightsalmon')
        ax2.set_xlabel('Background Type')
        ax2.set_ylabel('Number of Annotations')
        ax2.set_title('Annotation Distribution')
        ax2.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(annotation_counts) * 0.01,
                    str(annotation_counts[i]), ha='center', va='bottom')
        
        # Pie chart of total distribution
        total_annotations = sum(annotation_counts)
        if total_annotations > 0:
            percentages = [count/total_annotations * 100 for count in annotation_counts]
            ax3.pie(percentages, labels=backgrounds, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Annotation Distribution (%)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "background_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_vs_samples(self, results: Dict[str, Any], viz_dir: Path):
        """Plot performance vs sample count scatter"""
        import matplotlib.pyplot as plt
        
        sample_counts = []
        performances = []
        bg_labels = []
        
        for bg_type in self.background_types:
            bbox_key = f"{bg_type}_bbox"
            if bbox_key in results:
                sample_counts.append(len(self.bg_data[bg_type]["annotations"]))
                performances.append(results[bbox_key]['AP'])
                bg_labels.append(bg_type.title())
        
        if len(sample_counts) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(sample_counts, performances, s=100, alpha=0.7, c=range(len(bg_labels)), cmap='viridis')
        
        # Add labels for each point
        for i, label in enumerate(bg_labels):
            ax.annotate(label, (sample_counts[i], performances[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Number of GT Annotations')
        ax.set_ylabel('mAP@0.5:0.95 (%)')
        ax.set_title('Performance vs Sample Count')
        ax.grid(True, alpha=0.3)
        
        # Add correlation line if enough points
        if len(sample_counts) >= 3:
            import numpy as np
            z = np.polyfit(sample_counts, performances, 1)
            p = np.poly1d(z)
            ax.plot(sample_counts, p(sample_counts), "--", alpha=0.8, color='red')
            
            # Calculate correlation
            correlation = np.corrcoef(sample_counts, performances)[0,1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_vs_samples.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _save_single_sample_prediction(self, image_id: int, bg_type: str, output_dir: Path, 
                                 detector_name: str = None, prompt_info: str = None):
        """Save single image with predictions and GT"""
        try:
            # Find corresponding dataset sample
            dataset_sample = None
            for idx in range(len(self.dataset)):
                sample = self.dataset.get_sample(idx)
                if sample['image_info']['id'] == image_id:
                    dataset_sample = sample
                    break
            
            if dataset_sample is None:
                return
            
            # Load and copy image
            from PIL import Image, ImageDraw, ImageFont
            image = Image.open(dataset_sample['image_path']).convert('RGB')
            img_width, img_height = image.size
            
            # Create visualization
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw GT annotations (green boxes)
            gt_annotations = [ann for ann in self.bg_data[bg_type]["annotations"] 
                            if ann["image_id"] == image_id]
            
            for ann in gt_annotations:
                bbox = ann["bbox"]  # [x, y, w, h]
                x, y, w, h = bbox
                draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
                draw.text((x, y-20), 'GT', fill='green', font=font)
            
            # Draw predictions (red boxes)
            pred_annotations = [pred for pred in self.bg_data[bg_type]["predictions"]
                                if pred["image_id"] == image_id]
            
            for pred in pred_annotations:
                bbox = pred["bbox"]  # [x, y, w, h]
                score = pred["score"]
                x, y, w, h = bbox
                draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                draw.text((x, y+h+5), f'Pred: {score:.2f}', fill='red', font=font)
            
            # # Generate comprehensive filename
            # detector_str = detector_name if detector_name else "unknown"
            # prompt_str = prompt_info if prompt_info else "mixed"
            
            # # Clean prompt string for filename
            # prompt_clean = prompt_str.replace(' ', '_').replace(',', '-')[:50]  # Limit length
            
            # filename = f"{detector_str}_{prompt_clean}_{bg_type}_img{image_id}_gt{len(gt_annotations)}_pred{len(pred_annotations)}.jpg"
            
            # Fix filename generation
            detector_str = detector_name if detector_name else "detector"
            prompt_str = prompt_info if prompt_info else "default_prompt"
            
            # Clean and limit prompt string for filename
            prompt_clean = ''.join(c for c in prompt_str if c.isalnum() or c in '-_')[:30]
            
            # Generate proper filename without "unknown_mixed"
            filename = f"{detector_str}_{prompt_clean}_{bg_type}_sample{i}.jpg"

            
            image.save(output_dir / filename, 'JPEG', quality=95)
            
        except Exception as e:
            print(f"Failed to save sample {image_id}: {e}")

    def _save_sample_predictions(self, max_samples: int = 10, detector_name: str = None, prompt_info: str = None):
        """Save random sample images with predictions and GT overlaid"""
        
        try:
            import random
            from PIL import Image, ImageDraw, ImageFont
            
            sample_dir = self.output_dir / "sample_images"
            
            # Get random samples from each background type
            for bg_type in self.background_types:
                if not self.bg_data[bg_type]["images"]:
                    continue
                
                bg_sample_dir = sample_dir / bg_type
                bg_sample_dir.mkdir(exist_ok=True)
                
                # Get random samples
                all_image_ids = list(self.bg_data[bg_type]["images"])
                sample_image_ids = random.sample(all_image_ids, min(max_samples, len(all_image_ids)))
                
                for img_id in sample_image_ids:
                    self._save_single_sample_prediction(img_id, bg_type, bg_sample_dir, detector_name, prompt_info)
                
                print(f"Saved {len(sample_image_ids)} sample images for {bg_type} background")

        except Exception as e:
            print(f"Failed to save sample predictions: {e}")

    def reset_evaluator(self):
        """Reset evaluator for new detector evaluation"""
        self._gt_processed = False
        for bg_type in self.background_types:
            self.bg_data[bg_type] = {"annotations": [], "predictions": [], "images": {}}
