#!/usr/bin/env python3
"""
VLM Prompt Evaluation Strategy
Implements fair evaluation strategies for drone detection across multiple prompts
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict


class VLMPromptEvaluator:
    """
    Comprehensive prompt evaluation strategy for VLM drone detection
    Provides multiple evaluation approaches for fair comparison
    """
    
    def __init__(self, dataset, detector_system, background_evaluator, output_dir: str = "./prompt_evaluation"):
        self.dataset = dataset
        self.detector_system = detector_system
        self.background_evaluator = background_evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define prompt categories for fair evaluation
        # self.prompt_categories = {
        #     "drone_specific": [
        #         "drone", "quadcopter", "UAV", "unmanned aerial vehicle",
        #         "multirotor", "rotorcraft", "flying drone", "aerial drone",
        #         "quadrotor", "rc drone", "unmanned aircraft"
        #     ],
        #     "general_aircraft": [
        #         "aircraft", "airplane", "helicopter", "small aircraft"
        #     ],
        #     "ambiguous_objects": [
        #         "bird", "flying bird", "kite", "flying object"
        #     ]
        # }

        self.prompt_categories = {
            "drone_specific": [
                "small drone flying in the sky", "quadcopter hovering above ground",
                "UAV aircraft in aerial view", "small unmanned aerial vehicle in flight",
                "multirotor drone against sky background", "flying drone with rotors visible",
                "aerial drone captured from ground view", "quadrotor flying outdoors",
                "small rc drone in daylight", "unmanned aircraft in outdoor environment"
            ],
            "contextual_aircraft": [
                "small aircraft flying in clear sky", "helicopter in aerial photography",
                "small airplane at distance", "aircraft visible in sky background"  
            ],
            "size_aware": [
                "tiny drone dot in sky", "distant small flying object",
                "small dark object against bright sky", "compact flying vehicle"
            ]
        }
        
        # All prompts for comprehensive testing
        self.all_prompts = []
        for category_prompts in self.prompt_categories.values():
            self.all_prompts.extend(category_prompts)
    
    def evaluate_comprehensive(self, detector_names: List[str], 
                             evaluation_strategies: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation across multiple prompts and strategies
        
        Args:
            detector_names: List of detector names to evaluate
            evaluation_strategies: List of strategies ['single_best', 'average', 'ensemble', 'sensitivity']
        
        Returns:
            Complete evaluation results across all strategies
        """
        if evaluation_strategies is None:
            evaluation_strategies = ['single_best', 'average', 'ensemble', 'sensitivity']
        
        print(f"Starting comprehensive prompt evaluation for {len(detector_names)} detectors...")
        
        results = {
            "evaluation_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detectors": detector_names,
                "strategies": evaluation_strategies,
                "dataset_size": len(self.dataset),
                "prompt_categories": self.prompt_categories
            },
            "results_by_detector": {},
            "comparison_summary": {}
        }
        
        # Evaluate each detector with all strategies
        for detector_name in detector_names:
            print(f"\nEvaluating detector: {detector_name}")
            detector_results = self._evaluate_detector_all_strategies(
                detector_name, evaluation_strategies
            )
            results["results_by_detector"][detector_name] = detector_results
        
        # Create comparison summary
        results["comparison_summary"] = self._create_comparison_summary(
            results["results_by_detector"], evaluation_strategies
        )
        
        # Save comprehensive results
        self._save_comprehensive_results(results)
        
        return results
    
    def _evaluate_detector_all_strategies(self, detector_name: str, 
                                    strategies: List[str]) -> Dict[str, Any]:
        """Evaluate single detector with all strategies"""
        self.background_evaluator.reset_evaluator()
        detector_results = {
            "prompt_individual_results": {},
            "strategy_results": {}
        }
        
        # Step 1: Get predictions for all individual prompts
        print(f"  Testing {len(self.all_prompts)} individual prompts...")
        for prompt in self.all_prompts:
            predictions = self._run_detection_with_prompt(detector_name, [prompt])
            bg_results = self.background_evaluator.evaluate_detections(
                predictions, 
                expand_boxes=False,  # Disable expansion
                # iou_thresholds=[0.1, 0.3, 0.5, 0.7]  # Use multiple IoU thresholds
                size_aware_thresholds=True
            )
            detector_results["prompt_individual_results"][prompt] = bg_results
            
            # Save samples with proper naming
            self.background_evaluator._save_sample_predictions(
                max_samples=3, 
                detector_name=detector_name, 
                prompt_info=prompt
            )
        
        # Step 2: Apply evaluation strategies
        for strategy in strategies:
            print(f"  Applying strategy: {strategy}")
            strategy_result = self._apply_evaluation_strategy(
                strategy, detector_name, detector_results["prompt_individual_results"]
            )
            detector_results["strategy_results"][strategy] = strategy_result
        
        return detector_results
    
    def _apply_evaluation_strategy(self, strategy: str, detector_name: str,
                                 individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific evaluation strategy"""
        
        if strategy == "single_best":
            return self._single_best_prompt_strategy(individual_results)
        
        elif strategy == "average":
            return self._average_prompt_strategy(individual_results)
        
        elif strategy == "ensemble":
            return self._ensemble_prompt_strategy(detector_name)
        
        elif strategy == "sensitivity":
            return self._prompt_sensitivity_strategy(individual_results)
        
        else:
            raise ValueError(f"Unknown evaluation strategy: {strategy}")
    
    def _single_best_prompt_strategy(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find and report the best-performing single prompt"""
        best_prompt = None
        best_performance = -1
        best_results = None
        
        # Find best prompt based on overall mAP
        for prompt, results in individual_results.items():
            if "overall_mAP" in results:
                if results["overall_mAP"] > best_performance:
                    best_performance = results["overall_mAP"]
                    best_prompt = prompt
                    best_results = results
        
        return {
            "strategy": "single_best",
            "best_prompt": best_prompt,
            "best_performance": best_performance,
            "results": best_results,
            "category": self._get_prompt_category(best_prompt)
        }
    
    def _average_prompt_strategy(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute average performance across all prompts"""
        # Separate by category for more detailed analysis
        category_results = {}
        
        for category, prompts in self.prompt_categories.items():
            category_maps = []
            category_map50s = []
            category_recalls = []
            valid_results = []
            
            for prompt in prompts:
                if prompt in individual_results and "overall_mAP" in individual_results[prompt]:
                    results = individual_results[prompt]
                    category_maps.append(results["overall_mAP"])
                    category_map50s.append(results.get("mAP50", 0))
                    category_recalls.append(results.get("overall_recall", 0))
                    valid_results.append(results)
            
            if category_maps:
                category_results[category] = {
                    "mean_mAP": np.mean(category_maps),
                    "mean_mAP50": np.mean(category_map50s),
                    "mean_recall": np.mean(category_recalls),
                    "std_mAP": np.std(category_maps),
                    "prompt_count": len(category_maps),
                    "individual_performances": dict(zip(prompts[:len(category_maps)], category_maps))
                }
        
        # Overall average across all prompts
        all_maps = []
        all_map50s = []
        all_recalls = []
        
        for results in individual_results.values():
            if "overall_mAP" in results:
                all_maps.append(results["overall_mAP"])
                all_map50s.append(results.get("mAP50", 0))
                all_recalls.append(results.get("overall_recall", 0))
        
        return {
            "strategy": "average",
            "overall_average": {
                "mean_mAP": np.mean(all_maps) if all_maps else 0,
                "mean_mAP50": np.mean(all_map50s) if all_map50s else 0,
                "mean_recall": np.mean(all_recalls) if all_recalls else 0,
                "std_mAP": np.std(all_maps) if all_maps else 0,
                "prompt_count": len(all_maps)
            },
            "category_breakdown": category_results
        }
    
    def _ensemble_prompt_strategy(self, detector_name: str) -> Dict[str, Any]:
        """Improved ensemble with better diversity and NMS"""
        ensemble_configs = [
            {
                "name": "contextual_ensemble", 
                "prompts": ["small drone in sky", "quadcopter hovering", "UAV flying outdoors"],
                "weight": 1.2  # Higher weight for contextual prompts
            },
            {
                "name": "size_aware_ensemble",
                "prompts": ["tiny drone dot", "distant flying object", "small aircraft"],
                "weight": 1.0
            },
            {
                "name": "technical_ensemble", 
                "prompts": ["drone", "quadcopter", "UAV", "multirotor"],
                "weight": 0.8  # Lower weight for generic terms
            }
        ]
        
        ensemble_results = {}
        
        for config in ensemble_configs:
            weighted_predictions = []
            
            for prompt in config["prompts"]:
                predictions = self._run_detection_with_prompt(detector_name, [prompt])
                
                # Apply weight to scores
                for pred in predictions:
                    weighted_scores = [s * config["weight"] for s in pred['scores']]
                    pred['scores'] = weighted_scores
                
                weighted_predictions.extend(predictions)
            
            # Apply improved NMS with lower threshold for small objects
            nms_predictions = self._apply_improved_nms(weighted_predictions, nms_threshold=0.3)
            
            # Evaluate
            bg_results = self.background_evaluator.evaluate_detections(
                nms_predictions,
                expand_boxes=False,
                size_aware_thresholds=True)
            ensemble_results[config["name"]] = {
                "results": bg_results,
                "total_predictions": sum(len(p['boxes']) for p in nms_predictions)
            }
        
        return {"strategy": "ensemble", "ensemble_results": ensemble_results}

    def _apply_cross_prompt_nms(self, all_predictions: List[Dict[str, Any]], nms_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression across predictions from different prompts"""
        # Group predictions by image_id
        image_predictions = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': [], 'prompt_ids': []})
        
        for prompt_id, pred in enumerate(all_predictions):
            image_id = pred['image_id']
            image_predictions[image_id]['boxes'].extend(pred['boxes'])
            image_predictions[image_id]['scores'].extend(pred['scores'])
            image_predictions[image_id]['labels'].extend(pred['labels'])
            image_predictions[image_id]['prompt_ids'].extend([prompt_id] * len(pred['boxes']))
        
        # Apply NMS per image
        nms_predictions = []
        
        for image_id, preds in image_predictions.items():
            if not preds['boxes']:
                nms_predictions.append({
                    'image_id': image_id,
                    'boxes': [],
                    'scores': [],
                    'labels': []
                })
                continue
            
            # Convert to numpy arrays for NMS
            boxes = np.array(preds['boxes'])
            scores = np.array(preds['scores'])
            labels = preds['labels']
            
            # Convert boxes to [x1, y1, x2, y2] format if needed
            if boxes.shape[1] == 4:
                # Check if already in x1,y1,x2,y2 format or x,y,w,h format
                if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
                    # Likely x,y,w,h format, convert to x1,y1,x2,y2
                    boxes_xyxy = boxes.copy()
                    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
                    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
                    boxes = boxes_xyxy
            
            # Apply NMS
            keep_indices = self._nms_numpy(boxes, scores, nms_threshold)
            
            nms_predictions.append({
                'image_id': image_id,
                'boxes': boxes[keep_indices].tolist(),
                'scores': scores[keep_indices].tolist(),
                'labels': [labels[i] for i in keep_indices]
            })
        
        return nms_predictions

    def _nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Numpy implementation of Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        # Convert to float
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by scores
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            # Pick the last index
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            # Calculate IoU
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Remove boxes with IoU > threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _prompt_sensitivity_strategy(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensitivity to prompt variations"""
        # Extract performance values
        performances = []
        prompt_performances = {}
        
        for prompt, results in individual_results.items():
            if "overall_mAP" in results:
                perf = results["overall_mAP"]
                performances.append(perf)
                prompt_performances[prompt] = perf
        
        if not performances:
            return {"strategy": "sensitivity", "error": "No valid results for sensitivity analysis"}
        
        # Calculate sensitivity metrics
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        min_perf = np.min(performances)
        max_perf = np.max(performances)
        
        # Find most and least effective prompts
        best_prompt = max(prompt_performances.items(), key=lambda x: x[1])
        worst_prompt = min(prompt_performances.items(), key=lambda x: x[1])
        
        # Calculate coefficient of variation (relative stability)
        cv = (std_perf / mean_perf * 100) if mean_perf > 0 else float('inf')
        
        # Category-wise sensitivity
        category_sensitivity = {}
        for category, prompts in self.prompt_categories.items():
            category_perfs = [prompt_performances[p] for p in prompts if p in prompt_performances]
            if category_perfs:
                category_sensitivity[category] = {
                    "mean": np.mean(category_perfs),
                    "std": np.std(category_perfs),
                    "coefficient_of_variation": np.std(category_perfs) / np.mean(category_perfs) * 100 if np.mean(category_perfs) > 0 else float('inf'),
                    "range": max(category_perfs) - min(category_perfs),
                    "prompt_count": len(category_perfs)
                }
        
        return {
            "strategy": "sensitivity",
            "sensitivity_metrics": {
                "mean_performance": mean_perf,
                "std_performance": std_perf,
                "coefficient_of_variation": cv,
                "performance_range": max_perf - min_perf,
                "min_performance": min_perf,
                "max_performance": max_perf,
                "best_prompt": {"prompt": best_prompt[0], "performance": best_prompt[1]},
                "worst_prompt": {"prompt": worst_prompt[0], "performance": worst_prompt[1]}
            },
            "category_sensitivity": category_sensitivity,
            "individual_performances": prompt_performances
        }
    
    def _run_detection_with_prompt(self, detector_name: str, prompts: List[str]) -> List[Dict[str, Any]]:
        """Run detection on dataset with given prompts"""
        predictions = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.get_sample(idx)
            image_info = sample['image_info']
            image_id = str(image_info['file_name']).replace('.png', '').replace('.jpg', '')
            
            try:
                # Load image
                from PIL import Image
                image = Image.open(sample['image_path']).convert('RGB')
                
                # Run detection
                result = self.detector_system.detectors[detector_name].detect(
                    image, prompts, threshold=0.1
                )
                
                # Format prediction
                pred = {
                    'image_id': image_id,
                    'boxes': result.boxes,
                    'scores': result.scores,
                    'labels': result.labels
                }
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                # Add empty prediction
                predictions.append({
                    'image_id': image_id,
                    'boxes': [],
                    'scores': [],
                    'labels': []
                })
        
        return predictions
    
    def _get_prompt_category(self, prompt: str) -> str:
        """Get category for a given prompt"""
        for category, prompts in self.prompt_categories.items():
            if prompt in prompts:
                return category
        return "unknown"
    
    def _create_comparison_summary(self, detector_results: Dict[str, Any], 
                                 strategies: List[str]) -> Dict[str, Any]:
        """Create cross-detector comparison summary"""
        summary = {}
        
        # Compare detectors by strategy
        for strategy in strategies:
            strategy_comparison = {}
            
            if strategy == "single_best":
                # Compare best performances
                best_perfs = {}
                for detector, results in detector_results.items():
                    if strategy in results["strategy_results"]:
                        best_perfs[detector] = results["strategy_results"][strategy]["best_performance"]
                
                if best_perfs:
                    best_detector = max(best_perfs.items(), key=lambda x: x[1])
                    strategy_comparison = {
                        "best_detector": {"name": best_detector[0], "performance": best_detector[1]},
                        "all_performances": best_perfs,
                        "performance_ranking": sorted(best_perfs.items(), key=lambda x: x[1], reverse=True)
                    }
            
            elif strategy == "average":
                # Compare average performances
                avg_perfs = {}
                for detector, results in detector_results.items():
                    if strategy in results["strategy_results"]:
                        avg_perfs[detector] = results["strategy_results"][strategy]["overall_average"]["mean_mAP"]
                
                if avg_perfs:
                    best_avg_detector = max(avg_perfs.items(), key=lambda x: x[1])
                    strategy_comparison = {
                        "best_average_detector": {"name": best_avg_detector[0], "performance": best_avg_detector[1]},
                        "all_averages": avg_perfs,
                        "average_ranking": sorted(avg_perfs.items(), key=lambda x: x[1], reverse=True)
                    }
            
            summary[strategy] = strategy_comparison
        
        return summary
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results with visualizations"""
        try:
            # Create output subdirectories
            (self.output_dir / "metrics").mkdir(exist_ok=True)
            (self.output_dir / "visualizations").mkdir(exist_ok=True)
            (self.output_dir / "logs").mkdir(exist_ok=True)
            (self.output_dir / "prompt_analysis").mkdir(exist_ok=True)
            
            # Save full results
            results_file = self.output_dir / "metrics" / "comprehensive_prompt_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Create all visualizations
            self._create_prompt_visualizations(results)
            
            # Create summary report
            self._create_summary_report(results)
            
            # Create detailed logs
            self._save_detailed_logs(results)
            
            print(f"Comprehensive results saved to {self.output_dir}")
        
        except Exception as e:
            print(f"Failed to save results: {e}")

    def _create_prompt_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive prompt analysis visualizations"""
        try:
            viz_dir = self.output_dir / "visualizations"
            
            # 1. Prompt Importance Analysis
            self._plot_prompt_importance(results, viz_dir)
            
            # 2. Strategy Comparison
            self._plot_strategy_comparison(results, viz_dir)
            
            # 3. Detector Performance Matrix
            self._plot_detector_performance_matrix(results, viz_dir)
            
            # 4. Prompt Category Analysis
            self._plot_prompt_category_analysis(results, viz_dir)
            
            # 5. Sensitivity Analysis
            self._plot_sensitivity_analysis(results, viz_dir)
            
        except Exception as e:
            print(f"Failed to create visualizations: {e}")

    def _plot_prompt_importance(self, results: Dict[str, Any], viz_dir: Path):
        """Create prompt importance analysis plots"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Collect prompt performances across all detectors
        prompt_performances = {}
        
        for detector, detector_results in results["results_by_detector"].items():
            individual_results = detector_results.get("prompt_individual_results", {})
            
            for prompt, result in individual_results.items():
                if "overall_mAP" in result:
                    if prompt not in prompt_performances:
                        prompt_performances[prompt] = []
                    prompt_performances[prompt].append(result["overall_mAP"])
        
        if not prompt_performances:
            return
        
        # Calculate statistics for each prompt
        prompt_stats = {}
        for prompt, performances in prompt_performances.items():
            prompt_stats[prompt] = {
                'mean': np.mean(performances),
                'std': np.std(performances),
                'max': np.max(performances),
                'min': np.min(performances),
                'count': len(performances),
                'category': self._get_prompt_category(prompt)
            }
        
        # Sort prompts by mean performance
        sorted_prompts = sorted(prompt_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        # Create comprehensive prompt analysis figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Prompt Performance Ranking
        ax1 = plt.subplot(3, 2, 1)
        prompts = [item[0] for item in sorted_prompts[:15]]  # Top 15
        means = [item[1]['mean'] for item in sorted_prompts[:15]]
        stds = [item[1]['std'] for item in sorted_prompts[:15]]
        colors = [self._get_category_color(prompt_stats[p]['category']) for p in prompts]
        
        bars = ax1.barh(range(len(prompts)), means, xerr=stds, alpha=0.8, color=colors)
        ax1.set_yticks(range(len(prompts)))
        ax1.set_yticklabels(prompts)
        ax1.set_xlabel('Mean mAP (%)')
        ax1.set_title('Prompt Performance Ranking (Top 15)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax1.text(bar.get_width() + std + 1, bar.get_y() + bar.get_height()/2,
                    f'{mean:.1f}±{std:.1f}', va='center', ha='left', fontsize=8)
        
        # 2. Category Performance Comparison
        ax2 = plt.subplot(3, 2, 2)
        category_stats = {}
        for prompt, stats in prompt_stats.items():
            cat = stats['category']
            if cat not in category_stats:
                category_stats[cat] = []
            category_stats[cat].append(stats['mean'])
        
        categories = list(category_stats.keys())
        cat_means = [np.mean(category_stats[cat]) for cat in categories]
        cat_stds = [np.std(category_stats[cat]) for cat in categories]
        cat_colors = [self._get_category_color(cat) for cat in categories]
        
        bars2 = ax2.bar(categories, cat_means, yerr=cat_stds, alpha=0.8, color=cat_colors)
        ax2.set_ylabel('Mean mAP (%)')
        ax2.set_title('Performance by Prompt Category', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for bar, mean, std in zip(bars2, cat_means, cat_stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.1f}', ha='center', va='bottom')
        
        # 3. Prompt Stability vs Performance Scatter
        ax3 = plt.subplot(3, 2, 3)
        scatter_means = [stats['mean'] for stats in prompt_stats.values()]
        scatter_stds = [stats['std'] for stats in prompt_stats.values()]
        scatter_cats = [stats['category'] for stats in prompt_stats.values()]
        scatter_colors = [self._get_category_color(cat) for cat in scatter_cats]
        
        scatter = ax3.scatter(scatter_stds, scatter_means, c=scatter_colors, alpha=0.7, s=60)
        ax3.set_xlabel('Performance Std Dev (%)')
        ax3.set_ylabel('Mean Performance (%)')
        ax3.set_title('Prompt Stability vs Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax3.axhline(y=np.mean(scatter_means), color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=np.mean(scatter_stds), color='gray', linestyle='--', alpha=0.5)
        
        # 4. Performance Distribution
        ax4 = plt.subplot(3, 2, 4)
        all_performances = []
        for performances in prompt_performances.values():
            all_performances.extend(performances)
        
        ax4.hist(all_performances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=np.mean(all_performances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_performances):.1f}%')
        ax4.set_xlabel('mAP (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Overall Performance Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Top vs Bottom Prompts Comparison
        ax5 = plt.subplot(3, 2, 5)
        top_3_prompts = [item[0] for item in sorted_prompts[:3]]
        bottom_3_prompts = [item[0] for item in sorted_prompts[-3:]]
        
        top_3_means = [item[1]['mean'] for item in sorted_prompts[:3]]
        bottom_3_means = [item[1]['mean'] for item in sorted_prompts[-3:]]
        
        x_pos = np.arange(len(top_3_prompts))
        width = 0.35
        
        ax5.bar(x_pos - width/2, top_3_means, width, label='Top 3 Prompts', alpha=0.8, color='green')
        ax5.bar(x_pos + width/2, bottom_3_means, width, label='Bottom 3 Prompts', alpha=0.8, color='red')
        
        ax5.set_xlabel('Prompt Rank')
        ax5.set_ylabel('Mean mAP (%)')
        ax5.set_title('Top vs Bottom Performing Prompts', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'#{i+1}' for i in range(len(top_3_prompts))])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Prompt Length vs Performance
        ax6 = plt.subplot(3, 2, 6)
        prompt_lengths = [len(prompt.split()) for prompt in prompt_stats.keys()]
        length_performances = [stats['mean'] for stats in prompt_stats.values()]
        
        ax6.scatter(prompt_lengths, length_performances, alpha=0.7, s=60, color='purple')
        ax6.set_xlabel('Prompt Length (words)')
        ax6.set_ylabel('Mean Performance (%)')
        ax6.set_title('Prompt Length vs Performance', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add trend line
        if len(prompt_lengths) > 3:
            z = np.polyfit(prompt_lengths, length_performances, 1)
            p = np.poly1d(z)
            ax6.plot(sorted(prompt_lengths), p(sorted(prompt_lengths)), "--", alpha=0.8, color='red')
            
            correlation = np.corrcoef(prompt_lengths, length_performances)[0,1]
            ax6.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax6.transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "prompt_importance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed prompt analysis
        analysis_dir = self.output_dir / "prompt_analysis"
        with open(analysis_dir / "prompt_statistics.json", 'w') as f:
            json.dump(prompt_stats, f, indent=2, default=str)

    def _plot_strategy_comparison(self, results: Dict[str, Any], viz_dir: Path):
        """Plot evaluation strategy comparison"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        strategies_data = {}
        
        # Collect strategy results across detectors
        for detector, detector_results in results["results_by_detector"].items():
            strategy_results = detector_results.get("strategy_results", {})
            
            for strategy, strategy_result in strategy_results.items():
                if strategy not in strategies_data:
                    strategies_data[strategy] = {}
                
                if strategy == "single_best":
                    strategies_data[strategy][detector] = strategy_result.get("best_performance", 0)
                elif strategy == "average":
                    overall_avg = strategy_result.get("overall_average", {})
                    strategies_data[strategy][detector] = overall_avg.get("mean_mAP", 0)
                elif strategy == "sensitivity":
                    sens_metrics = strategy_result.get("sensitivity_metrics", {})
                    strategies_data[strategy][detector] = sens_metrics.get("mean_performance", 0)
        
        if not strategies_data:
            return
        
        # Create strategy comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evaluation Strategy Comparison', fontsize=16, fontweight='bold')
        
        # 1. Strategy Performance Comparison
        detectors = list(next(iter(strategies_data.values())).keys())
        strategies = list(strategies_data.keys())
        
        x = np.arange(len(detectors))
        width = 0.2
        
        for i, strategy in enumerate(strategies):
            if strategy in ["single_best", "average", "sensitivity"]:
                values = [strategies_data[strategy].get(det, 0) for det in detectors]
                ax1.bar(x + i*width, values, width, label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        ax1.set_xlabel('Detectors')
        ax1.set_ylabel('mAP (%)')
        ax1.set_title('Strategy Performance by Detector')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(detectors)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Strategy Consistency Analysis
        if "average" in strategies_data and "single_best" in strategies_data:
            avg_performances = list(strategies_data["average"].values())
            best_performances = list(strategies_data["single_best"].values())
            
            ax2.scatter(avg_performances, best_performances, s=100, alpha=0.7)
            
            for i, det in enumerate(detectors):
                ax2.annotate(det, (avg_performances[i], best_performances[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            # Add diagonal line
            min_val = min(min(avg_performances), min(best_performances))
            max_val = max(max(avg_performances), max(best_performances))
            ax2.plot([min_val, max_val], [min_val, max_val], '--', alpha=0.5, color='red')
            
            ax2.set_xlabel('Average Strategy mAP (%)')
            ax2.set_ylabel('Single Best Strategy mAP (%)')
            ax2.set_title('Strategy Consistency Analysis')
            ax2.grid(True, alpha=0.3)
        
        # 3. Strategy Ranking
        strategy_means = {}
        for strategy, detector_data in strategies_data.items():
            strategy_means[strategy] = np.mean(list(detector_data.values()))
        
        sorted_strategies = sorted(strategy_means.items(), key=lambda x: x[1], reverse=True)
        
        ax3.bar([s[0].replace('_', ' ').title() for s in sorted_strategies],
               [s[1] for s in sorted_strategies], alpha=0.8, color='lightcoral')
        ax3.set_ylabel('Mean mAP (%)')
        ax3.set_title('Strategy Ranking (Average across Detectors)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (_, value) in enumerate(sorted_strategies):
            ax3.text(i, value + 0.5, f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Strategy Variability
        strategy_stds = {}
        for strategy, detector_data in strategies_data.items():
            strategy_stds[strategy] = np.std(list(detector_data.values()))
        
        ax4.bar([s.replace('_', ' ').title() for s in strategy_stds.keys()],
               list(strategy_stds.values()), alpha=0.8, color='lightgreen')
        ax4.set_ylabel('Standard Deviation (%)')
        ax4.set_title('Strategy Variability (Across Detectors)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_detector_performance_matrix(self, results: Dict[str, Any], viz_dir: Path):
        """Create detector performance comparison matrix"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create performance matrix
        detectors = list(results["results_by_detector"].keys())
        strategies = ["single_best", "average"]
        
        performance_matrix = np.zeros((len(detectors), len(strategies)))
        
        for i, detector in enumerate(detectors):
            detector_results = results["results_by_detector"][detector]
            strategy_results = detector_results.get("strategy_results", {})
            
            for j, strategy in enumerate(strategies):
                if strategy in strategy_results:
                    if strategy == "single_best":
                        performance_matrix[i, j] = strategy_results[strategy].get("best_performance", 0)
                    elif strategy == "average":
                        overall_avg = strategy_results[strategy].get("overall_average", {})
                        performance_matrix[i, j] = overall_avg.get("mean_mAP", 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Add text annotations
        for i in range(len(detectors)):
            for j in range(len(strategies)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xticks(np.arange(len(strategies)))
        ax.set_yticks(np.arange(len(detectors)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
        ax.set_yticklabels(detectors)
        
        ax.set_title('Detector Performance Matrix (mAP %)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('mAP (%)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "detector_performance_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prompt_category_analysis(self, results: Dict[str, Any], viz_dir: Path):
        """Plot detailed prompt category analysis"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Collect category performance data
        category_data = {}
        
        for detector, detector_results in results["results_by_detector"].items():
            individual_results = detector_results.get("prompt_individual_results", {})
            
            for prompt, result in individual_results.items():
                if "overall_mAP" in result:
                    category = self._get_prompt_category(prompt)
                    
                    if category not in category_data:
                        category_data[category] = {'performances': [], 'prompts': []}
                    
                    category_data[category]['performances'].append(result["overall_mAP"])
                    category_data[category]['prompts'].append(prompt)
        
        if not category_data:
            return
        
        # Create category analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prompt Category Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category Performance Box Plot
        categories = list(category_data.keys())
        performances_by_category = [category_data[cat]['performances'] for cat in categories]
        
        bp = ax1.boxplot(performances_by_category, labels=categories, patch_artist=True)
        colors = [self._get_category_color(cat) for cat in categories]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('mAP (%)')
        ax1.set_title('Performance Distribution by Category')
        ax1.grid(True, alpha=0.3)
        
        # 2. Category Statistics
        category_stats = {}
        for cat, data in category_data.items():
            category_stats[cat] = {
                'mean': np.mean(data['performances']),
                'std': np.std(data['performances']),
                'count': len(data['performances'])
            }
        
        cat_names = list(category_stats.keys())
        means = [category_stats[cat]['mean'] for cat in cat_names]
        stds = [category_stats[cat]['std'] for cat in cat_names]
        colors = [self._get_category_color(cat) for cat in cat_names]
        
        bars = ax2.bar(cat_names, means, yerr=stds, alpha=0.8, color=colors)
        ax2.set_ylabel('Mean mAP (%)')
        ax2.set_title('Category Performance Summary')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
        
        # 3. Prompt Count by Category
        counts = [category_stats[cat]['count'] for cat in cat_names]
        ax3.pie(counts, labels=cat_names, autopct='%1.1f%%', startangle=90,
               colors=[self._get_category_color(cat) for cat in cat_names])
        ax3.set_title('Prompt Distribution by Category')
        
        # 4. Category Performance Range
        ranges = [max(data['performances']) - min(data['performances']) 
                 for data in category_data.values()]
        
        ax4.bar(cat_names, ranges, alpha=0.8, color=colors)
        ax4.set_ylabel('Performance Range (%)')
        ax4.set_title('Category Performance Variability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "prompt_category_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_sensitivity_analysis(self, results: Dict[str, Any], viz_dir: Path):
        """Plot prompt sensitivity analysis"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        sensitivity_data = {}
        
        # Extract sensitivity data
        for detector, detector_results in results["results_by_detector"].items():
            strategy_results = detector_results.get("strategy_results", {})
            
            if "sensitivity" in strategy_results:
                sens_result = strategy_results["sensitivity"]
                sens_metrics = sens_result.get("sensitivity_metrics", {})
                
                sensitivity_data[detector] = {
                    'cv': sens_metrics.get('coefficient_of_variation', 0),
                    'range': sens_metrics.get('performance_range', 0),
                    'mean': sens_metrics.get('mean_performance', 0),
                    'std': sens_metrics.get('std_performance', 0)
                }
        
        if not sensitivity_data:
            return
        
        detectors = list(sensitivity_data.keys())
        
        # Create sensitivity analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prompt Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Coefficient of Variation Comparison
        cvs = [sensitivity_data[det]['cv'] for det in detectors]
        bars1 = ax1.bar(detectors, cvs, alpha=0.8, color='lightcoral')
        ax1.set_ylabel('Coefficient of Variation (%)')
        ax1.set_title('Prompt Sensitivity (Lower = More Stable)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels and interpretation
        for bar, cv in zip(bars1, cvs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{cv:.1f}%', ha='center', va='bottom')
        
        # Add stability threshold line
        ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Moderate Stability')
        ax1.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='High Stability')
        ax1.legend()
        
        # 2. Performance Range
        ranges = [sensitivity_data[det]['range'] for det in detectors]
        bars2 = ax2.bar(detectors, ranges, alpha=0.8, color='lightblue')
        ax2.set_ylabel('Performance Range (%)')
        ax2.set_title('Performance Variability Across Prompts')
        ax2.grid(True, alpha=0.3)
        
        for bar, range_val in zip(bars2, ranges):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{range_val:.1f}%', ha='center', va='bottom')
        
        # 3. Stability vs Performance Scatter
        means = [sensitivity_data[det]['mean'] for det in detectors]
        ax3.scatter(cvs, means, s=100, alpha=0.7, color='purple')
        
        for i, det in enumerate(detectors):
            ax3.annotate(det, (cvs[i], means[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Coefficient of Variation (%)')
        ax3.set_ylabel('Mean Performance (%)')
        ax3.set_title('Stability vs Performance Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax3.axhline(y=np.mean(means), color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=np.mean(cvs), color='gray', linestyle='--', alpha=0.5)
        
        # 4. Stability Ranking
        stability_scores = [100/max(cv, 1) for cv in cvs]  # Higher is more stable
        sorted_detectors = sorted(zip(detectors, stability_scores), key=lambda x: x[1], reverse=True)
        
        det_names = [item[0] for item in sorted_detectors]
        scores = [item[1] for item in sorted_detectors]
        
        bars4 = ax4.bar(det_names, scores, alpha=0.8, color='lightgreen')
        ax4.set_ylabel('Stability Score')
        ax4.set_title('Detector Stability Ranking')
        ax4.grid(True, alpha=0.3)
        
        for bar, score in zip(bars4, scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _get_category_color(self, category: str) -> str:
        """Get consistent color for prompt category"""
        color_map = {
            'drone_specific': '#1f77b4',    # Blue
            'general_aircraft': '#ff7f0e',  # Orange  
            'ambiguous_objects': '#2ca02c', # Green
            'unknown': '#d62728'            # Red
        }
        return color_map.get(category, '#808080')  # Gray default

    def _save_detailed_logs(self, results: Dict[str, Any]):
        """Save detailed evaluation logs"""
        logs_dir = self.output_dir / "logs"
        
        # Save evaluation log
        with open(logs_dir / "evaluation_log.txt", 'w') as f:
            f.write("VLM Comprehensive Evaluation Log\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            meta = results["evaluation_metadata"]
            f.write(f"Timestamp: {meta['timestamp']}\n")
            f.write(f"Dataset Size: {meta['dataset_size']} samples\n")
            f.write(f"Detectors Evaluated: {', '.join(meta['detectors'])}\n")
            f.write(f"Strategies Applied: {', '.join(meta['strategies'])}\n")
            f.write(f"Total Prompts Tested: {len(self.all_prompts)}\n\n")
            
            # Detailed results for each detector
            for detector, detector_results in results["results_by_detector"].items():
                f.write(f"\n{'='*20} {detector.upper()} {'='*20}\n")
                
                # Individual prompt results summary
                individual_results = detector_results.get("prompt_individual_results", {})
                if individual_results:
                    f.write(f"\nIndividual Prompt Results ({len(individual_results)} prompts):\n")
                    f.write("-" * 40 + "\n")
                    
                    # Sort by performance
                    sorted_prompts = sorted(individual_results.items(), 
                                          key=lambda x: x[1].get("overall_mAP", 0), reverse=True)
                    
                    for prompt, result in sorted_prompts:
                        map_score = result.get("overall_mAP", 0)
                        category = self._get_prompt_category(prompt)
                        f.write(f"{prompt:<30} | {map_score:>6.2f}% | {category}\n")
                
                # Strategy results summary
                strategy_results = detector_results.get("strategy_results", {})
                for strategy, strategy_result in strategy_results.items():
                    f.write(f"\n{strategy.replace('_', ' ').title()} Strategy:\n")
                    f.write("-" * 30 + "\n")
                    
                    if strategy == "single_best":
                        f.write(f"Best Prompt: {strategy_result.get('best_prompt', 'N/A')}\n")
                        f.write(f"Best Performance: {strategy_result.get('best_performance', 0):.3f}% mAP\n")
                        f.write(f"Category: {strategy_result.get('category', 'N/A')}\n")
                    
                    elif strategy == "average":
                        overall = strategy_result.get("overall_average", {})
                        f.write(f"Mean Performance: {overall.get('mean_mAP', 0):.3f}% ± {overall.get('std_mAP', 0):.3f}%\n")
                        f.write(f"Prompts Tested: {overall.get('prompt_count', 0)}\n")
                        
                        # Category breakdown
                        category_breakdown = strategy_result.get("category_breakdown", {})
                        for category, cat_results in category_breakdown.items():
                            f.write(f"  {category}: {cat_results.get('mean_mAP', 0):.3f}% ± {cat_results.get('std_mAP', 0):.3f}%\n")
                    
                    elif strategy == "sensitivity":
                        sens_metrics = strategy_result.get("sensitivity_metrics", {})
                        f.write(f"Mean Performance: {sens_metrics.get('mean_performance', 0):.3f}%\n")
                        f.write(f"Coefficient of Variation: {sens_metrics.get('coefficient_of_variation', 0):.1f}%\n")
                        f.write(f"Performance Range: {sens_metrics.get('performance_range', 0):.3f}%\n")
                        
                        best_prompt = sens_metrics.get('best_prompt', {})
                        worst_prompt = sens_metrics.get('worst_prompt', {})
                        f.write(f"Most Effective: {best_prompt.get('prompt', 'N/A')} ({best_prompt.get('performance', 0):.3f}%)\n")
                        f.write(f"Least Effective: {worst_prompt.get('prompt', 'N/A')} ({worst_prompt.get('performance', 0):.3f}%)\n")
            
            # Overall comparison summary
            f.write(f"\n{'='*20} COMPARISON SUMMARY {'='*20}\n")
            comparison = results.get("comparison_summary", {})
            
            for strategy, comp_results in comparison.items():
                f.write(f"\n{strategy.replace('_', ' ').title()} Strategy Winner:\n")
                
                if "best_detector" in comp_results:
                    best = comp_results["best_detector"]
                    f.write(f"  {best['name']}: {best['performance']:.3f}% mAP\n")
                
                if "performance_ranking" in comp_results:
                    f.write("  Complete Ranking:\n")
                    for i, (detector, perf) in enumerate(comp_results["performance_ranking"]):
                        f.write(f"    {i+1}. {detector}: {perf:.3f}%\n")
    
    def _create_summary_report(self, results: Dict[str, Any]):
        """Create human-readable summary report"""
        summary_file = self.output_dir / "evaluation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("VLM Prompt Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            meta = results["evaluation_metadata"]
            f.write(f"Evaluation Date: {meta['timestamp']}\n")
            f.write(f"Detectors Tested: {', '.join(meta['detectors'])}\n")
            f.write(f"Dataset Size: {meta['dataset_size']} samples\n")
            f.write(f"Total Prompts Tested: {len(self.all_prompts)}\n\n")
            
            # Strategy results for each detector
            for detector, detector_results in results["results_by_detector"].items():
                f.write(f"\n{detector.upper()} RESULTS\n")
                f.write("-" * 30 + "\n")
                
                for strategy, strategy_results in detector_results["strategy_results"].items():
                    f.write(f"\n{strategy.replace('_', ' ').title()}:\n")
                    
                    if strategy == "single_best":
                        f.write(f"  Best Prompt: {strategy_results['best_prompt']}\n")
                        f.write(f"  Best Performance: {strategy_results['best_performance']:.3f} mAP\n")
                        f.write(f"  Category: {strategy_results['category']}\n")
                    
                    elif strategy == "average":
                        overall = strategy_results["overall_average"]
                        f.write(f"  Average mAP: {overall['mean_mAP']:.3f} ± {overall['std_mAP']:.3f}\n")
                        f.write(f"  Prompts Tested: {overall['prompt_count']}\n")
                        
                        for category, cat_results in strategy_results["category_breakdown"].items():
                            f.write(f"    {category}: {cat_results['mean_mAP']:.3f} ± {cat_results['std_mAP']:.3f}\n")
            
            # Comparison summary
            f.write(f"\n\nCOMPARISON SUMMARY\n")
            f.write("=" * 30 + "\n")
            
            for strategy, comparison in results["comparison_summary"].items():
                f.write(f"\n{strategy.replace('_', ' ').title()}:\n")
                
                if "best_detector" in comparison:
                    best = comparison["best_detector"]
                    f.write(f"  Winner: {best['name']} ({best['performance']:.3f} mAP)\n")
                
                if "performance_ranking" in comparison:
                    f.write("  Ranking:\n")
                    for i, (detector, perf) in enumerate(comparison["performance_ranking"][:3]):
                        f.write(f"    {i+1}. {detector}: {perf:.3f}\n")
