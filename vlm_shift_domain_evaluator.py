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
        # PERF-1 FIX: build image_id → sample lookup once (O(N)), not O(N) per prediction
        if not hasattr(self, '_image_id_to_sample'):
            self._image_id_to_sample = {
                s['image_info']['id']: s for s in self.dataset
            }
        for pred in predictions:
            image_id = pred['image_id']
            
            # Find domain for this image — O(1) dict lookup
            sample = self._image_id_to_sample.get(image_id, None)
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
        cat_id = category_map.get(label.lower().strip(), None)
        if cat_id is None:
            if not hasattr(self, '_unknown_label_warnings'):
                self._unknown_label_warnings = set()
            if label not in self._unknown_label_warnings:
                print(f"  WARNING: unknown label '{label}' has no category mapping, "
                      f"defaulting to pedestrian(1). Check label normalization.")
                self._unknown_label_warnings.add(label)
            return 1
        return cat_id
    
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
        
        # Add annotation IDs — MUST start at 1, not 0.
        # pycocotools uses 0 as the "unmatched" sentinel in dtMatches/gtMatches,
        # so any GT annotation with id=0 can never register as a true positive.
        for idx, anno in enumerate(annotations):
            anno_copy = anno.copy()
            anno_copy['id'] = idx + 1  # 1-indexed to avoid id=0 bug
            # Convert bbox from [x1,y1,x2,y2] to [x,y,w,h] for COCO
            x1, y1, x2, y2 = anno_copy['bbox']
            w, h = x2 - x1, y2 - y1
            # BUG-6 FIX: validate bbox dimensions
            if w <= 0 or h <= 0:
                print(f"  WARNING: degenerate bbox id={anno_copy['id']} "
                      f"[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] → w={w:.1f},h={h:.1f}, skipping")
                continue
            anno_copy['bbox'] = [x1, y1, w, h]
            # BUG-4 FIX: COCO GT requires explicit 'area' for size-based AP
            anno_copy['area'] = w * h
            # In _evaluate_single_domain, after anno_copy['bbox'] conversion (around line 251):
            if 'iscrowd' not in anno_copy:
                anno_copy['iscrowd'] = 0  # IMPORTANT: COCO requires this field
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

            # After line 280 (coco_eval.summarize()), add:
            self._debug_evaluation_details(coco, coco_dt, coco_eval, domain_name)
            best_threshold = self._analyze_score_distribution_by_match(coco, coco_dt, coco_eval, domain_name)
            self._analyze_nms_threshold(predictions, coco, domain_name, confidence_threshold=best_threshold)  # Use best conf threshold from previous analysis
            
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
            
            # ── Per-class metrics extraction ──
            per_class = self._extract_per_class_metrics(coco, coco_dt, coco_eval, domain_name)
            metrics['per_class'] = per_class
            
            return metrics
            
        finally:
            # Clean up temp file
            os.unlink(gt_file)

    def _extract_per_class_metrics(self, coco, coco_dt, coco_eval, domain_name):
        """
        Extract per-class AP (mAP, AP@50, AP@75), AR, and TP/FP/FN from COCO eval.
        
        coco_eval.eval['precision'] shape: [T, R, K, A, M]
            T = 10 IoU thresholds (0.50:0.05:0.95)
            R = 101 recall thresholds (0.00:0.01:1.00)
            K = number of categories
            A = 4 area ranges (all, small, medium, large)
            M = 3 maxDets values (1, 10, 100)
        
        coco_eval.eval['recall'] shape: [T, K, A, M]
        """
        CATEGORY_NAMES = {1: 'pedestrian', 2: 'car', 3: 'truck',
                          4: 'bus', 5: 'motorcycle', 6: 'bicycle'}
        
        precision = coco_eval.eval['precision']   # [T, R, K, A, M]
        recall = coco_eval.eval['recall']          # [T, K, A, M]
        
        cat_ids = coco_eval.params.catIds
        area_idx = 0    # 'all'
        maxdet_idx = 2  # maxDets=100
        
        n_imgs = len(coco_eval.params.imgIds)
        n_cats = len(cat_ids)
        n_areas = len(coco_eval.params.areaRng)
        
        per_class = {}
        
        print(f"\n  ── Per-Class Metrics ({domain_name}) ──")
        print(f"  {'Class':<14s} {'GT':>5s} {'Pred':>6s} {'AP':>7s} {'AP@50':>7s} "
              f"{'AP@75':>7s} {'AR@100':>7s} {'TP':>5s} {'FP':>6s} {'FN':>5s}")
        print(f"  {'-'*80}")
        
        for k_idx, cat_id in enumerate(cat_ids):
            cat_name = CATEGORY_NAMES.get(int(cat_id), f'cat_{cat_id}')
            
            # ── AP metrics ──
            # mAP (average over all 10 IoU thresholds)
            p_all = precision[:, :, k_idx, area_idx, maxdet_idx]
            valid = p_all[p_all > -1]
            ap = float(np.mean(valid)) if len(valid) > 0 else -1.0
            
            # AP @ IoU=0.50 (index 0)
            p_50 = precision[0, :, k_idx, area_idx, maxdet_idx]
            valid_50 = p_50[p_50 > -1]
            ap_50 = float(np.mean(valid_50)) if len(valid_50) > 0 else -1.0
            
            # AP @ IoU=0.75 (index 5)
            p_75 = precision[5, :, k_idx, area_idx, maxdet_idx]
            valid_75 = p_75[p_75 > -1]
            ap_75 = float(np.mean(valid_75)) if len(valid_75) > 0 else -1.0
            
            # ── AR @ maxDets=100 (average over IoU thresholds) ──
            r_vals = recall[:, k_idx, area_idx, maxdet_idx]
            valid_r = r_vals[r_vals > -1]
            ar_100 = float(np.mean(valid_r)) if len(valid_r) > 0 else -1.0
            
            # ── TP / FP / FN at IoU=0.5 ──
            tp, fp, fn = 0, 0, 0
            iou_idx = 0  # IoU=0.5
            
            for img_idx in range(n_imgs):
                eval_idx = k_idx * (n_areas * n_imgs) + area_idx * n_imgs + img_idx
                evalImg = coco_eval.evalImgs[eval_idx]
                if evalImg is None:
                    continue
                
                dtMatches = evalImg.get('dtMatches', None)
                if dtMatches is not None and len(dtMatches) > 0:
                    matches = dtMatches[iou_idx]
                    tp += int((matches > 0).sum())
                    fp += int((matches == 0).sum())
                
                gtMatches = evalImg.get('gtMatches', None)
                if gtMatches is not None and len(gtMatches) > 0:
                    fn += int((gtMatches[iou_idx] == 0).sum())
            
            # ── GT / Pred counts ──
            n_gt = len(coco.getAnnIds(catIds=[int(cat_id)]))
            n_dt = len(coco_dt.getAnnIds(catIds=[int(cat_id)]))
            
            per_class[cat_name] = {
                'category_id': int(cat_id),
                'AP': round(ap, 4),
                'AP_50': round(ap_50, 4),
                'AP_75': round(ap_75, 4),
                'AR_100': round(ar_100, 4),
                'num_gt': n_gt,
                'num_predictions': n_dt,
                'TP': tp,
                'FP': fp,
                'FN': fn,
            }
            
            # Precision / Recall at IoU=0.5
            prec_at_50 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_at_50 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class[cat_name]['precision_at_50'] = round(prec_at_50, 4)
            per_class[cat_name]['recall_at_50'] = round(rec_at_50, 4)
            
            # Print row
            ap_str = f"{ap:.4f}" if ap >= 0 else "  N/A "
            ap50_str = f"{ap_50:.4f}" if ap_50 >= 0 else "  N/A "
            ap75_str = f"{ap_75:.4f}" if ap_75 >= 0 else "  N/A "
            ar_str = f"{ar_100:.4f}" if ar_100 >= 0 else "  N/A "
            print(f"  {cat_name:<14s} {n_gt:>5d} {n_dt:>6d} {ap_str:>7s} {ap50_str:>7s} "
                  f"{ap75_str:>7s} {ar_str:>7s} {tp:>5d} {fp:>6d} {fn:>5d}")
        
        print(f"  {'-'*80}")
        
        return per_class

    def _analyze_score_distribution_by_match(self, coco, coco_dt, coco_eval, domain_name):
        """Analyze score distribution for TPs vs FPs to find optimal threshold"""
        import numpy as np
        
        n_imgs = len(coco_eval.params.imgIds)
        n_cats = len(coco_eval.params.catIds)
        n_areas = len(coco_eval.params.areaRng)
        
        tp_scores = []
        fp_scores = []
        
        area_idx = 0  # 'all'
        iou_idx = 0   # IoU=0.5
        
        for cat_idx in range(n_cats):
            for img_idx in range(n_imgs):
                eval_idx = cat_idx * (n_areas * n_imgs) + area_idx * n_imgs + img_idx
                evalImg = coco_eval.evalImgs[eval_idx]
                
                if evalImg is None:
                    continue
                
                dtMatches = evalImg.get('dtMatches', None)
                dtScores = evalImg.get('dtScores', None)
                
                if dtMatches is not None and dtScores is not None and len(dtScores) > 0:
                    matches_at_iou50 = dtMatches[iou_idx]
                    for i, (match, score) in enumerate(zip(matches_at_iou50, dtScores)):
                        if match > 0:
                            tp_scores.append(score)
                        else:
                            fp_scores.append(score)
        
        tp_scores = np.array(tp_scores)
        fp_scores = np.array(fp_scores)
        
        print(f"\n  === SCORE DISTRIBUTION ANALYSIS ===")
        print(f"  True Positive scores:")
        print(f"    Count: {len(tp_scores)}")
        # BUG-3 FIX: guard empty arrays before calling .mean()/.min()
        if len(tp_scores) > 0:
            print(f"    Mean: {tp_scores.mean():.4f}, Median: {np.median(tp_scores):.4f}")
            print(f"    Min: {tp_scores.min():.4f}, Max: {tp_scores.max():.4f}")
            print(f"    Percentiles [25, 50, 75, 90]: {np.percentile(tp_scores, [25, 50, 75, 90])}")
        else:
            print(f"    (no true positives in this domain)")
        
        print(f"\n  False Positive scores:")
        print(f"    Count: {len(fp_scores)}")
        if len(fp_scores) > 0:
            print(f"    Mean: {fp_scores.mean():.4f}, Median: {np.median(fp_scores):.4f}")
            print(f"    Min: {fp_scores.min():.4f}, Max: {fp_scores.max():.4f}")
            print(f"    Percentiles [25, 50, 75, 90]: {np.percentile(fp_scores, [25, 50, 75, 90])}")
        else:
            print(f"    (no false positives in this domain)")
        
        # Find optimal threshold using F1
        print(f"\n  === THRESHOLD ANALYSIS ===")
        
        best_f1 = 0
        best_threshold = 0
        
        if len(tp_scores) == 0 and len(fp_scores) == 0:
            print(f"  (no detections to analyze)")
            print(f"  === END SCORE ANALYSIS ===\n")
            return best_threshold
        
        print(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'TP':>8s} {'FP':>8s}")
        print(f"  {'-'*58}")
        
        for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            tp = (tp_scores >= thresh).sum()
            fp = (fp_scores >= thresh).sum()
            fn = (tp_scores < thresh).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {thresh:>10.2f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {tp:>8d} {fp:>8d}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        print(f"\n  Best F1={best_f1:.4f} at threshold={best_threshold}")
        print(f"  === END SCORE ANALYSIS ===\n")

        return best_threshold

    def _analyze_nms_threshold(self, predictions: List[Dict], coco, domain_name: str, 
                            confidence_threshold: float = 0.15):
        """
        Analyze different NMS IoU thresholds to find optimal for deployment.
        """
        import numpy as np
        from collections import defaultdict
        
        try:
            from torchvision.ops import nms
            import torch
            USE_TORCH_NMS = True
        except ImportError:
            USE_TORCH_NMS = False
            print("  Warning: torchvision not available, using custom NMS")
        
        print(f"\n  === NMS THRESHOLD ANALYSIS (conf_thresh={confidence_threshold}) ===")
        
        # DEBUG: Quick sanity check
        print(f"  COCO GT: {len(coco.getAnnIds())} annotations, {len(coco.getImgIds())} images")
        print(f"  Predictions: {len(predictions)} total")
        
        # Step 1: Group predictions by image_id
        # predictions bbox is in [x1, y1, x2, y2] format (NOT yet converted to COCO)
        preds_by_image = defaultdict(list)
        for pred in predictions:
            # predictions are in [x1, y1, x2, y2] format here!
            preds_by_image[pred['image_id']].append({
                'bbox': pred['bbox'],  # Keep as [x1, y1, x2, y2]
                'score': pred['score'],
                'category_id': pred['category_id']
            })
        
        # NMS IoU thresholds to test
        nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        
        for nms_thresh in nms_thresholds:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_preds_after_nms = 0
            
            for image_id in coco.getImgIds():
                image_preds = preds_by_image.get(image_id, [])
                
                # Get GT annotations - NOTE: imgIds must be a LIST!
                gt_ann_ids = coco.getAnnIds(imgIds=[image_id])
                gt_anns = coco.loadAnns(gt_ann_ids)
                
                if len(image_preds) == 0:
                    # Count all GT as FN
                    total_fn += len(gt_anns)
                    continue
                
                # Extract arrays - bbox is in [x1, y1, x2, y2] format
                boxes = np.array([p['bbox'] for p in image_preds])
                scores = np.array([p['score'] for p in image_preds])
                cat_ids = np.array([p['category_id'] for p in image_preds])
                
                # Step 1: Apply confidence threshold
                conf_mask = scores >= confidence_threshold
                boxes = boxes[conf_mask]
                scores = scores[conf_mask]
                cat_ids = cat_ids[conf_mask]
                
                if len(boxes) == 0:
                    total_fn += len(gt_anns)
                    continue
                
                # Step 2: Apply NMS per class
                final_boxes = []
                final_scores = []
                final_cat_ids = []
                
                unique_cats = np.unique(cat_ids)
                for cat_id in unique_cats:
                    cat_mask = cat_ids == cat_id
                    cat_boxes = boxes[cat_mask]
                    cat_scores = scores[cat_mask]
                    
                    if nms_thresh < 1.0 and len(cat_boxes) > 0:
                        if USE_TORCH_NMS:
                            boxes_tensor = torch.tensor(cat_boxes, dtype=torch.float32)
                            scores_tensor = torch.tensor(cat_scores, dtype=torch.float32)
                            keep_indices = nms(boxes_tensor, scores_tensor, nms_thresh)
                            keep_indices = keep_indices.numpy()
                        else:
                            keep_indices = self._custom_nms(cat_boxes, cat_scores, nms_thresh)
                        
                        final_boxes.extend(cat_boxes[keep_indices])
                        final_scores.extend(cat_scores[keep_indices])
                        final_cat_ids.extend([cat_id] * len(keep_indices))
                    else:
                        final_boxes.extend(cat_boxes)
                        final_scores.extend(cat_scores)
                        final_cat_ids.extend([cat_id] * len(cat_boxes))
                
                final_boxes = np.array(final_boxes) if final_boxes else np.array([]).reshape(0, 4)
                final_scores = np.array(final_scores) if final_scores else np.array([])
                final_cat_ids = np.array(final_cat_ids) if final_cat_ids else np.array([])
                
                total_preds_after_nms += len(final_boxes)
                
                # Step 3: Prepare GT boxes
                gt_boxes = []
                gt_cats = []
                gt_matched = []
                
                for ann in gt_anns:
                    # COCO GT bbox is [x, y, w, h], convert to [x1, y1, x2, y2]
                    x, y, w, h = ann['bbox']
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_cats.append(ann['category_id'])
                    gt_matched.append(False)
                
                gt_boxes = np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 4)
                
                # Sort predictions by score (descending)
                if len(final_scores) > 0:
                    sort_idx = np.argsort(-final_scores)
                    final_boxes = final_boxes[sort_idx]
                    final_scores = final_scores[sort_idx]
                    final_cat_ids = final_cat_ids[sort_idx]
                
                # Greedy matching
                for pred_box, pred_cat_id in zip(final_boxes, final_cat_ids):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_cat) in enumerate(zip(gt_boxes, gt_cats)):
                        if gt_matched[gt_idx]:
                            continue
                        if gt_cat != pred_cat_id:
                            continue
                        
                        iou = self._compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= 0.5 and best_gt_idx >= 0:
                        total_tp += 1
                        gt_matched[best_gt_idx] = True
                    else:
                        total_fp += 1
                
                # Count unmatched GT as FN
                total_fn += sum(1 for m in gt_matched if not m)
            
            # Compute metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'nms_thresh': nms_thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'total_preds': total_preds_after_nms
            })
        
        # Print results
        print(f"  {'NMS_IoU':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'TP':>8s} {'FP':>8s} {'#Preds':>10s}")
        print(f"  {'-'*68}")
        
        best_f1 = 0
        best_nms_thresh = 1.0
        
        for r in results:
            print(f"  {r['nms_thresh']:>10.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1']:>10.4f} {r['tp']:>8d} {r['fp']:>8d} {r['total_preds']:>10d}")
            
            if r['f1'] > best_f1:
                best_f1 = r['f1']
                best_nms_thresh = r['nms_thresh']
        
        print(f"\n  Best F1={best_f1:.4f} at NMS_IoU={best_nms_thresh}")
        print(f"  === END NMS ANALYSIS ===\n")
        
        return results, best_nms_thresh

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    def _custom_nms(self, boxes, scores, iou_threshold):
        """Custom NMS implementation if torchvision not available"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        # Sort by score descending
        order = np.argsort(-scores)
        keep = []
        
        while len(order) > 0:
            idx = order[0]
            keep.append(idx)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = np.array([self._compute_iou(boxes[idx], boxes[r]) for r in remaining])
            
            # Keep boxes with IoU below threshold
            mask = ious < iou_threshold
            order = remaining[mask]
        
        return np.array(keep, dtype=int)

    def _debug_evaluation_details(self, coco, coco_dt, coco_eval, domain_name):
        """Debug: Analyze TP/FP distribution - CORRECTED VERSION"""
        import numpy as np
        
        print(f"\n  === DEBUG for {domain_name} ===")
        
        # 1. Score distribution analysis
        all_scores = []
        for img_id in coco.getImgIds():
            dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
            all_scores.extend([ann['score'] for ann in dt_anns])
        
        if all_scores:
            scores_arr = np.array(all_scores)
            print(f"  Score distribution:")
            print(f"    Min: {scores_arr.min():.4f}, Max: {scores_arr.max():.4f}")
            print(f"    Mean: {scores_arr.mean():.4f}, Median: {np.median(scores_arr):.4f}")
            print(f"    Scores > 0.5: {(scores_arr > 0.5).sum()} ({100*(scores_arr > 0.5).mean():.1f}%)")
            print(f"    Scores > 0.3: {(scores_arr > 0.3).sum()} ({100*(scores_arr > 0.3).mean():.1f}%)")
            print(f"    Scores > 0.1: {(scores_arr > 0.1).sum()} ({100*(scores_arr > 0.1).mean():.1f}%)")
        
        # 2. Per-category breakdown
        print(f"\n  Per-category breakdown:")
        for cat_id, cat_name in [(1, 'pedestrian'), (2, 'car'), (3, 'truck'), 
                                (4, 'bus'), (5, 'motorcycle'), (6, 'bicycle')]:
            gt_count = len(coco.getAnnIds(catIds=[cat_id]))
            dt_count = len(coco_dt.getAnnIds(catIds=[cat_id]))
            if gt_count > 0 or dt_count > 0:
                ratio = dt_count / gt_count if gt_count > 0 else float('inf')
                print(f"    {cat_name:12s}: GT={gt_count:5d}, Pred={dt_count:6d}, Ratio={ratio:.2f}x")
        
        # 3. CORRECTED: Analyze TP/FP/FN - filter to areaRng='all' only
        # coco_eval.params.areaRng = [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
        # areaRngLbl = ['all', 'small', 'medium', 'large']
        # We only want index 0 (area='all')
        
        tp_count = 0
        fp_count = 0
        fn_count = 0
        
        # evalImgs is organized as: [img0_cat0_area0, img0_cat0_area1, ..., img0_cat1_area0, ...]
        # Shape is effectively: n_cats × n_areas × n_imgs (flattened)
        n_imgs = len(coco_eval.params.imgIds)
        n_cats = len(coco_eval.params.catIds)
        n_areas = len(coco_eval.params.areaRng)  # 4: all, small, medium, large
        
        print(f"\n  Eval structure: {n_imgs} imgs × {n_cats} cats × {n_areas} areas = {len(coco_eval.evalImgs)} evalImgs")
        
        # Only process area='all' (index 0)
        area_idx = 0  # 'all'
        iou_idx = 0   # IoU=0.5 (first threshold)
        
        for cat_idx in range(n_cats):
            for img_idx in range(n_imgs):
                # Index calculation: cat_idx * (n_areas * n_imgs) + area_idx * n_imgs + img_idx
                eval_idx = cat_idx * (n_areas * n_imgs) + area_idx * n_imgs + img_idx
                
                evalImg = coco_eval.evalImgs[eval_idx]
                if evalImg is None:
                    continue
                
                dtMatches = evalImg.get('dtMatches', None)
                if dtMatches is not None and len(dtMatches) > 0:
                    matches_at_iou50 = dtMatches[iou_idx]
                    tp_count += (matches_at_iou50 > 0).sum()
                    fp_count += (matches_at_iou50 == 0).sum()
                
                gtMatches = evalImg.get('gtMatches', None)
                if gtMatches is not None and len(gtMatches) > 0:
                    fn_count += (gtMatches[iou_idx] == 0).sum()
        
        total_gt = len(coco.getAnnIds())
        total_pred = len(coco_dt.getAnnIds())
        
        print(f"\n  Sanity check:")
        print(f"    Total GT annotations: {total_gt}")
        print(f"    Total predictions: {total_pred}")
        print(f"    TP + FN = {tp_count + fn_count} (should ≈ {total_gt})")
        print(f"    TP + FP = {tp_count + fp_count} (should ≈ {total_pred})")
        
        print(f"\n  TP/FP/FN at IoU=0.5 (area='all'):")
        print(f"    True Positives:  {tp_count}")
        print(f"    False Positives: {fp_count}")
        print(f"    False Negatives: {fn_count}")
        if tp_count + fp_count > 0:
            print(f"    Precision: {tp_count / (tp_count + fp_count):.4f}")
        if tp_count + fn_count > 0:
            print(f"    Recall: {tp_count / (tp_count + fn_count):.4f}")
        
        # 4. Additional: Check maxDets truncation impact
        print(f"\n  maxDets analysis:")
        imgs_over_100_preds = 0
        total_truncated = 0
        for img_id in coco.getImgIds():
            n_preds = len(coco_dt.getAnnIds(imgIds=img_id))
            if n_preds > 100:
                imgs_over_100_preds += 1
                total_truncated += (n_preds - 100)
        print(f"    Images with >100 predictions: {imgs_over_100_preds}")
        print(f"    Total predictions truncated by maxDets=100: {total_truncated}")
        
        print(f"  === END DEBUG ===\n")
    
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
            sample = self._image_id_to_sample.get(image_id, None)
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
        """Save detailed results to JSON and human-readable summary"""
        
        # Main results file
        results_file = self.output_dir / "domain_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary_file = self.output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("SHIFT Domain Evaluation Summary\n")
            f.write("="*100 + "\n\n")
            
            # Overall metrics
            if 'overall' in results:
                f.write("Overall Performance:\n")
                f.write(f"  mAP @ IoU=0.5:0.95: {results['overall']['mAP']:.4f}\n")
                f.write(f"  mAP @ IoU=0.50:     {results['overall']['mAP_50']:.4f}\n")
                f.write(f"  mAP @ IoU=0.75:     {results['overall']['mAP_75']:.4f}\n")
                f.write(f"  mAP (small):        {results['overall']['mAP_small']:.4f}\n")
                f.write(f"  mAP (medium):       {results['overall']['mAP_medium']:.4f}\n")
                f.write(f"  mAP (large):        {results['overall']['mAP_large']:.4f}\n\n")
                
                # Overall per-class table
                if 'per_class' in results['overall']:
                    f.write("Overall Per-Class Performance:\n")
                    self._write_per_class_table(f, results['overall']['per_class'])
                    f.write("\n")
            
            # Per-domain metrics
            f.write("\nPer-Domain Performance:\n")
            f.write(f"{'Domain':<30s} {'mAP':>8s} {'mAP@50':>8s} {'Images':>8s} {'GT':>8s} {'Pred':>8s}\n")
            f.write("-"*80 + "\n")
            
            for domain in sorted(results.keys()):
                if domain in ('overall', 'size_analysis'):
                    continue
                metrics = results[domain]
                f.write(f"{domain:<30s} "
                       f"{metrics['mAP']:>8.4f} "
                       f"{metrics['mAP_50']:>8.4f} "
                       f"{metrics['num_images']:>8d} "
                       f"{metrics['num_gt']:>8d} "
                       f"{metrics['num_predictions']:>8d}\n")
            
            # Per-domain per-class breakdown
            f.write("\n\n" + "="*100 + "\n")
            f.write("Per-Domain Per-Class Breakdown\n")
            f.write("="*100 + "\n")
            
            for domain in sorted(results.keys()):
                if domain in ('overall', 'size_analysis'):
                    continue
                metrics = results[domain]
                if 'per_class' not in metrics:
                    continue
                
                f.write(f"\n── {domain} (mAP={metrics['mAP']:.4f}, mAP@50={metrics['mAP_50']:.4f}) ──\n")
                self._write_per_class_table(f, metrics['per_class'])
            
            # Size analysis
            if 'size_analysis' in results:
                f.write("\n\nObject Size Distribution:\n")
                for size_cat in ['small', 'medium', 'large']:
                    stats = results['size_analysis'][size_cat]
                    f.write(f"  {size_cat.capitalize()}:  {stats['count']:6d} objects, "
                           f"avg area: {stats['avg_area']:.1f} px²\n")
        
        print(f"  ✓ Saved results to {results_file}")
        print(f"  ✓ Saved summary to {summary_file}")
    
    @staticmethod
    def _write_per_class_table(f, per_class: Dict[str, Dict]):
        """Write a per-class metrics table to file handle f."""
        header = (f"  {'Class':<14s} {'GT':>5s} {'Pred':>6s} {'AP':>7s} {'AP@50':>7s} "
                  f"{'AP@75':>7s} {'AR@100':>7s} {'TP':>5s} {'FP':>6s} {'FN':>5s} "
                  f"{'Prec@50':>8s} {'Rec@50':>8s}")
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        
        for cls_name, m in per_class.items():
            def _fmt(v): return f"{v:.4f}" if v >= 0 else "  N/A "
            
            f.write(f"  {cls_name:<14s} "
                   f"{m['num_gt']:>5d} {m['num_predictions']:>6d} "
                   f"{_fmt(m['AP']):>7s} {_fmt(m['AP_50']):>7s} "
                   f"{_fmt(m['AP_75']):>7s} {_fmt(m['AR_100']):>7s} "
                   f"{m['TP']:>5d} {m['FP']:>6d} {m['FN']:>5d} "
                   f"{m['precision_at_50']:>8.4f} {m['recall_at_50']:>8.4f}\n")


def create_shift_domain_evaluator(dataset, output_dir: str = "./shift_evaluation_results"):
    """Factory function"""
    return VLMSHIFTDomainEvaluator(dataset, output_dir)