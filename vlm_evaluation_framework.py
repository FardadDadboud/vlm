#!/usr/bin/env python3
"""
VLM Evaluation Framework for Drone Detection
Integrates VLM detectors with DrIFT dataset and comprehensive evaluation
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import existing VLM system
from vlm_detector_system_new import (
    MultiModalDetector, OWLv2Detector, YOLOWorldDetector,
    GroundingDINODetector, DETRDetector, check_gpu_status
)

# Import new components (these should be in the same directory or properly installed)
from vlm_drift_dataset import VLMDrIFTDataset
from vlm_background_evaluator import VLMBackgroundEvaluator  
from vlm_prompt_evaluator import VLMPromptEvaluator


class VLMEvaluationFramework:
    """
    Comprehensive VLM evaluation framework for drone detection
    Integrates detection, dataset loading, and evaluation components
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else self._get_default_config()
        self.detector_system = None
        self.dataset = None
        self.background_evaluator = None
        self.prompt_evaluator = None
        
        # Initialize components
        self._setup_detector_system()
        self._setup_dataset()
        self._setup_evaluators()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                return json.load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "dataset": {
                "data_root": "/path/to/DrIFT_dataset",
                "split": "val",
                "filters": {
                    "image_filters": {
                        "view": ["Aerial"],
                        "source": ["Real"]
                    },
                    "annotation_filters": {
                        "background": [0, 1, 2]  # sky, tree, ground
                    }
                }
            },
            "detectors": {
                "owlv2": {
                    "enabled": True,
                    "model_path": "google/owlv2-base-patch16-ensemble"
                },
                "yolo-world": {
                    "enabled": True,
                    "model_path": "yolov8s-world.pt"
                },
                "grounding-dino": {
                    "enabled": False,  # Disabled by default due to installation complexity
                    "model_path": "IDEA-Research/grounding-dino-base"
                },
                
                
            },
            "evaluation": {
                "output_dir": "./vlm_evaluation_results",
                "strategies": ["single_best", "average", "ensemble", "sensitivity"]
            }
        }
    
    def _setup_detector_system(self):
        """Setup VLM detector system with configured models"""
        print("Setting up VLM detector system...")
        
        # Check GPU status
        gpu_available = check_gpu_status()
        device = "cuda" if gpu_available else "cpu"
        
        # Initialize detector system
        self.detector_system = MultiModalDetector()
        
        # Add configured detectors
        for detector_name, detector_config in self.config["detectors"].items():
            if not detector_config.get("enabled", False):
                continue
            
            try:
                if "owlv2" in detector_name.lower():
                    detector = OWLv2Detector(detector_config["model_path"], device=device)
                    self.detector_system.add_detector(detector_name, detector)
                
                elif "yolo" in detector_name.lower():
                    detector = YOLOWorldDetector(detector_config["model_path"], device=device)
                    self.detector_system.add_detector(detector_name, detector)
                
                elif "grounding" in detector_name.lower():
                    detector = GroundingDINODetector(detector_config["model_path"], device=device)
                    self.detector_system.add_detector(detector_name, detector)

                print(f"Loaded detector: {detector_name}")
                
            except Exception as e:
                print(f"Failed to load {detector_name}: {e}")
        
        if not self.detector_system.detectors:
            raise RuntimeError("No detectors loaded successfully")
        
        print(f"Detector system ready with {len(self.detector_system.detectors)} models")
    
    def _setup_dataset(self):
        """Setup DrIFT dataset with filtering"""
        print("Setting up dataset...")
        
        dataset_config = self.config["dataset"]
        self.dataset = VLMDrIFTDataset(
            data_root=dataset_config["data_root"],
            split=dataset_config.get("split", "val"),
            filters=dataset_config.get("filters")
        )
        
        print(f"Dataset ready with {len(self.dataset)} samples")
    
    def _setup_evaluators(self):
        """Setup evaluation components"""
        eval_config = self.config["evaluation"]
        output_dir = eval_config.get("output_dir", "./vlm_evaluation_results")
        
        # Background evaluator
        self.background_evaluator = VLMBackgroundEvaluator(
            self.dataset, 
            output_dir=f"{output_dir}/background"
        )
        
        # Prompt evaluator
        self.prompt_evaluator = VLMPromptEvaluator(
            self.dataset,
            self.detector_system,
            self.background_evaluator,
            output_dir=f"{output_dir}/prompt"
        )
        
        print("Evaluators ready")
    
    def run_quick_evaluation(self, detector_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run quick evaluation with comprehensive output generation"""
        if detector_names is None:
            detector_names = list(self.detector_system.detectors.keys())
        
        print(f"Running quick evaluation for {detector_names}...")
        
        # Create timestamped output directory
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_output_dir = Path(self.config["evaluation"]["output_dir"]) / f"quick_eval_{timestamp}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        all_predictions = {}
        
        for detector_name in detector_names:
            print(f"Evaluating {detector_name}...")
            
            # Use drone-specific prompts for quick eval
            drone_prompts = ["drone", "quadcopter", "UAV"]
            
            # Run detection
            predictions = self._run_detection(detector_name, drone_prompts)
            all_predictions[detector_name] = predictions
            
            # Evaluate with background-wise metrics (no expansion, multiple IoUs)
            bg_results = self.background_evaluator.evaluate_detections(
                predictions, 
                expand_boxes=False,  # Disable expansion
                iou_thresholds=[0.1, 0.3, 0.5, 0.7]  # Multiple IoU thresholds
            )
            
            results[detector_name] = {
                "prompts_used": drone_prompts,
                "background_results": bg_results,
                "total_predictions": sum(len(pred['boxes']) for pred in predictions),
                "images_processed": len(predictions)
            }
            
            # Save samples with proper naming
            prompt_str = "-".join(drone_prompts)
            self.background_evaluator._save_sample_predictions(
                max_samples=5,
                detector_name=detector_name,
                prompt_info=prompt_str
            )
        
        # Save comprehensive results
        self._save_quick_results_comprehensive(results, all_predictions, eval_output_dir)
        
        print(f"Quick evaluation completed! Results saved to: {eval_output_dir}")
        return results
    
    def run_comprehensive_evaluation(self, detector_names: Optional[List[str]] = None,
                                   strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with full output generation
        
        Args:
            detector_names: List of detector names to evaluate (default: all available)
            strategies: List of evaluation strategies (default: from config)
            
        Returns:
            Comprehensive evaluation results with saved outputs
        """
        if detector_names is None:
            detector_names = list(self.detector_system.detectors.keys())
        
        if strategies is None:
            strategies = self.config["evaluation"].get("strategies", ["single_best", "average"])
        
        print(f"Running comprehensive evaluation...")
        print(f"Detectors: {detector_names}")
        print(f"Strategies: {strategies}")
        
        # Create timestamped output directory
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_output_dir = Path(self.config["evaluation"]["output_dir"]) / f"comprehensive_eval_{timestamp}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update evaluators with new output directory
        self.background_evaluator.output_dir = eval_output_dir / "background"
        self.prompt_evaluator.output_dir = eval_output_dir / "prompt"

        self.background_evaluator.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_evaluator.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run comprehensive prompt evaluation
        results = self.prompt_evaluator.evaluate_comprehensive(detector_names, strategies)
        
        # Generate additional analysis outputs
        self._generate_additional_analysis(results, eval_output_dir)
        
        print(f"Comprehensive evaluation completed! Results saved to: {eval_output_dir}")
        
        return results

    def _save_quick_results_comprehensive(self, results: Dict[str, Any], 
                                        predictions: Dict[str, List], 
                                        output_dir: Path):
        """Save comprehensive quick evaluation results with visualizations"""
        # Create output subdirectories
        (output_dir / "metrics").mkdir(exist_ok=True)
        (output_dir / "visualizations").mkdir(exist_ok=True)
        (output_dir / "sample_images").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        
        # Save main results
        with open(output_dir / "metrics" / "quick_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create quick evaluation visualizations
        self._create_quick_evaluation_plots(results, output_dir / "visualizations")
        
        # Save sample predictions
        self._save_sample_predictions_quick(predictions, output_dir / "sample_images")
        
        # Create evaluation report
        self._create_quick_evaluation_report(results, output_dir)
        
        # Save evaluation log
        self._save_evaluation_log(results, output_dir / "logs")

    def _create_quick_evaluation_plots(self, results: Dict[str, Any], viz_dir: Path):
        """Create visualization plots for quick evaluation"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 1. Overall Performance Comparison
            detectors = list(results.keys())
            overall_maps = []
            sky_maps = []
            tree_maps = []
            ground_maps = []
            
            for detector in detectors:
                bg_results = results[detector]["background_results"]
                overall_maps.append(bg_results.get("overall_mAP", 0))
                sky_maps.append(bg_results.get("sky_bbox", {}).get("AP", 0))
                tree_maps.append(bg_results.get("tree_bbox", {}).get("AP", 0))
                ground_maps.append(bg_results.get("ground_bbox", {}).get("AP", 0))
            
            # Create comprehensive comparison plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Quick Evaluation Results Summary', fontsize=16, fontweight='bold')
            
            # Overall performance comparison
            bars1 = ax1.bar(detectors, overall_maps, alpha=0.8, color='skyblue')
            ax1.set_ylabel('Overall mAP (%)')
            ax1.set_title('Overall Performance Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars1, overall_maps):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            # Background-wise performance
            x = np.arange(len(detectors))
            width = 0.25
            
            ax2.bar(x - width, sky_maps, width, label='Sky', alpha=0.8, color='lightblue')
            ax2.bar(x, tree_maps, width, label='Tree', alpha=0.8, color='lightgreen')
            ax2.bar(x + width, ground_maps, width, label='Ground', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('Detectors')
            ax2.set_ylabel('mAP (%)')
            ax2.set_title('Background-wise Performance')
            ax2.set_xticks(x)
            ax2.set_xticklabels(detectors)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Detection statistics
            total_preds = [results[det]["total_predictions"] for det in detectors]
            bars3 = ax3.bar(detectors, total_preds, alpha=0.8, color='orange')
            ax3.set_ylabel('Total Detections')
            ax3.set_title('Detection Count Comparison')
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, total_preds):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_preds)*0.01,
                        str(value), ha='center', va='bottom')
            
            # Performance ranking
            detector_performance = list(zip(detectors, overall_maps))
            detector_performance.sort(key=lambda x: x[1], reverse=True)
            
            ranked_detectors = [x[0] for x in detector_performance]
            ranked_scores = [x[1] for x in detector_performance]
            
            bars4 = ax4.bar(range(len(ranked_detectors)), ranked_scores, 
                           alpha=0.8, color='gold')
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('mAP (%)')
            ax4.set_title('Performance Ranking')
            ax4.set_xticks(range(len(ranked_detectors)))
            ax4.set_xticklabels([f'{i+1}. {det}' for i, det in enumerate(ranked_detectors)], 
                               rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "quick_evaluation_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Detailed background analysis
            self._create_background_analysis_plot(results, viz_dir)
            
        except Exception as e:
            print(f"Failed to create quick evaluation plots: {e}")

    def _create_background_analysis_plot(self, results: Dict[str, Any], viz_dir: Path):
        """Create detailed background analysis plot"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Background-wise Detailed Analysis', fontsize=16, fontweight='bold')
            
            detectors = list(results.keys())
            backgrounds = ['sky', 'tree', 'ground']
            
            # Create performance matrix
            performance_matrix = np.zeros((len(detectors), len(backgrounds)))
            sample_matrix = np.zeros((len(detectors), len(backgrounds)))
            
            for i, detector in enumerate(detectors):
                bg_results = results[detector]["background_results"]
                for j, bg in enumerate(backgrounds):
                    bg_key = f"{bg}_bbox"
                    sample_key = f"{bg}_samples"
                    
                    if bg_key in bg_results:
                        performance_matrix[i, j] = bg_results[bg_key]["AP"]
                    if sample_key in bg_results:
                        sample_matrix[i, j] = bg_results[sample_key]
            
            # Performance heatmap
            im1 = ax1.imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto')
            ax1.set_xticks(np.arange(len(backgrounds)))
            ax1.set_yticks(np.arange(len(detectors)))
            ax1.set_xticklabels([bg.title() for bg in backgrounds])
            ax1.set_yticklabels(detectors)
            ax1.set_title('Performance Heatmap (mAP %)')
            
            # Add text annotations
            for i in range(len(detectors)):
                for j in range(len(backgrounds)):
                    text = ax1.text(j, i, f'{performance_matrix[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im1, ax=ax1)
            
            # Sample count heatmap
            im2 = ax2.imshow(sample_matrix, cmap='Blues', aspect='auto')
            ax2.set_xticks(np.arange(len(backgrounds)))
            ax2.set_yticks(np.arange(len(detectors)))
            ax2.set_xticklabels([bg.title() for bg in backgrounds])
            ax2.set_yticklabels(detectors)
            ax2.set_title('Sample Count by Background')
            
            # Add text annotations
            for i in range(len(detectors)):
                for j in range(len(backgrounds)):
                    text = ax2.text(j, i, f'{int(sample_matrix[i, j])}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im2, ax=ax2)
            
            # Background performance comparison
            bg_means = np.mean(performance_matrix, axis=0)
            bg_stds = np.std(performance_matrix, axis=0)
            
            bars = ax3.bar(backgrounds, bg_means, yerr=bg_stds, alpha=0.8, 
                          color=['lightblue', 'lightgreen', 'lightcoral'])
            ax3.set_ylabel('Mean mAP (%)')
            ax3.set_title('Average Performance by Background')
            ax3.grid(True, alpha=0.3)
            
            for bar, mean, std in zip(bars, bg_means, bg_stds):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                        f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
            
            # Detector performance consistency
            detector_stds = np.std(performance_matrix, axis=1)
            detector_means = np.mean(performance_matrix, axis=1)
            
            ax4.scatter(detector_stds, detector_means, s=100, alpha=0.7, color='purple')
            
            for i, detector in enumerate(detectors):
                ax4.annotate(detector, (detector_stds[i], detector_means[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel('Performance Std Dev Across Backgrounds (%)')
            ax4.set_ylabel('Mean Performance (%)')
            ax4.set_title('Detector Consistency Analysis')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "background_detailed_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Failed to create background analysis plot: {e}")

    def _save_sample_predictions_quick(self, predictions: Dict[str, List], output_dir: Path, max_samples: int = 5):
        """Save sample prediction images for quick evaluation"""
        try:
            import random
            from PIL import Image, ImageDraw, ImageFont
            
            # Create detector subdirectories
            for detector in predictions.keys():
                detector_dir = output_dir / detector
                detector_dir.mkdir(exist_ok=True)
            
            # Get random sample indices
            total_samples = len(next(iter(predictions.values())))
            sample_indices = random.sample(range(total_samples), min(max_samples, total_samples))
            
            for detector, pred_list in predictions.items():
                detector_dir = output_dir / detector
                saved_count = 0
                
                for idx in sample_indices:
                    if idx >= len(pred_list):
                        continue
                    
                    pred = pred_list[idx]
                    image_id = pred['image_id']
                    
                    # Find corresponding dataset sample
                    dataset_sample = None
                    for dataset_idx in range(len(self.dataset)):
                        sample = self.dataset.get_sample(dataset_idx)
                        sample_id = str(sample['image_info']['file_name']).replace('.png', '').replace('.jpg', '')
                        if sample_id == image_id:
                            dataset_sample = sample
                            break
                    
                    if dataset_sample is None:
                        continue
                    
                    # Create visualization
                    try:
                        image = Image.open(dataset_sample['image_path']).convert('RGB')
                        draw = ImageDraw.Draw(image)
                        
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                        except:
                            font = ImageFont.load_default()
                        
                        # Draw GT annotations (green boxes)
                        for ann in dataset_sample['annotations']:
                            bbox = ann['bbox']  # [x, y, w, h]
                            x, y, w, h = bbox
                            draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
                            draw.text((x, y-18), 'GT', fill='green', font=font)
                        
                        # Draw predictions (red boxes)
                        boxes = pred['boxes']
                        scores = pred['scores']
                        labels = pred['labels']
                        
                        for box, score, label in zip(boxes, scores, labels):
                            if len(box) == 4:
                                x1, y1, x2, y2 = box
                                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                                draw.text((x1, y2+5), f'{label}: {score:.2f}', fill='red', font=font)
                        
                        # Save image
                        filename = f"sample_{saved_count+1}_{image_id}_gt{len(dataset_sample['annotations'])}_pred{len(boxes)}.jpg"
                        image.save(detector_dir / filename, 'JPEG', quality=95)
                        saved_count += 1
                        
                    except Exception as e:
                        print(f"Error saving sample {image_id} for {detector}: {e}")
                        continue
                
                print(f"Saved {saved_count} sample images for {detector}")
        
        except Exception as e:
            print(f"Failed to save sample predictions: {e}")

    def _create_quick_evaluation_report(self, results: Dict[str, Any], output_dir: Path):
        """Create comprehensive quick evaluation report"""
        report_file = output_dir / "evaluation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# VLM Quick Evaluation Report\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            
            best_detector = max(results.items(), 
                              key=lambda x: x[1]["background_results"].get("overall_mAP", 0))
            best_performance = best_detector[1]["background_results"].get("overall_mAP", 0)
            
            f.write(f"**Best Performing Model:** {best_detector[0]} ({best_performance:.2f}% mAP)\n\n")
            f.write(f"**Total Models Evaluated:** {len(results)}\n")
            f.write(f"**Dataset Size:** {len(self.dataset)} samples\n")
            f.write(f"**Prompts Used:** drone, quadcopter, UAV\n\n")
            
            # Performance summary table
            f.write("## Performance Summary\n\n")
            f.write("| Model | Overall mAP | Sky mAP | Tree mAP | Ground mAP | Total Detections |\n")
            f.write("|-------|-------------|---------|----------|------------|------------------|\n")
            
            for detector, result in results.items():
                bg_results = result["background_results"]
                overall_map = bg_results.get("overall_mAP", 0)
                sky_map = bg_results.get("sky_bbox", {}).get("AP", 0)
                tree_map = bg_results.get("tree_bbox", {}).get("AP", 0)
                ground_map = bg_results.get("ground_bbox", {}).get("AP", 0)
                total_preds = result["total_predictions"]
                
                f.write(f"| {detector} | {overall_map:.2f}% | {sky_map:.2f}% | {tree_map:.2f}% | {ground_map:.2f}% | {total_preds} |\n")
            
            # Background analysis
            f.write("\n## Background-wise Analysis\n\n")
            
            # Calculate background statistics
            bg_performances = {'sky': [], 'tree': [], 'ground': []}
            for result in results.values():
                bg_results = result["background_results"]
                for bg in bg_performances.keys():
                    bg_key = f"{bg}_bbox"
                    if bg_key in bg_results:
                        bg_performances[bg].append(bg_results[bg_key]["AP"])
            
            for bg, performances in bg_performances.items():
                if performances:
                    import numpy as np
                    mean_perf = np.mean(performances)
                    std_perf = np.std(performances)
                    f.write(f"- **{bg.title()} Background:** {mean_perf:.2f}% ± {std_perf:.2f}% mAP (across {len(performances)} models)\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            f.write("1. **For Best Overall Performance:** Use " + best_detector[0] + "\n")
            
            # Find most consistent detector
            detector_consistency = {}
            for detector, result in results.items():
                bg_results = result["background_results"]
                bg_aps = []
                for bg in ['sky', 'tree', 'ground']:
                    bg_key = f"{bg}_bbox"
                    if bg_key in bg_results:
                        bg_aps.append(bg_results[bg_key]["AP"])
                
                if len(bg_aps) > 1:
                    import numpy as np
                    detector_consistency[detector] = np.std(bg_aps)
            
            if detector_consistency:
                most_consistent = min(detector_consistency.items(), key=lambda x: x[1])
                f.write(f"2. **For Consistency Across Backgrounds:** Use {most_consistent[0]} (std: {most_consistent[1]:.2f}%)\n")
            
            f.write("3. **For Further Analysis:** Run comprehensive evaluation for detailed prompt sensitivity analysis\n")
            f.write("4. **For Production:** Consider inference speed and memory requirements alongside accuracy\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("- `metrics/`: JSON files with detailed metrics\n")
            f.write("- `visualizations/`: Performance comparison plots\n")
            f.write("- `sample_images/`: Random samples with predictions and GT overlaid\n")
            f.write("- `logs/`: Detailed evaluation logs\n")

    def _save_evaluation_log(self, results: Dict[str, Any], logs_dir: Path):
        """Save detailed evaluation log"""
        log_file = logs_dir / "evaluation_log.txt"
        
        with open(log_file, 'w') as f:
            import time
            f.write("VLM Quick Evaluation Log\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {len(self.dataset)} samples\n")
            f.write(f"Detectors: {list(results.keys())}\n")
            f.write(f"Prompts: drone, quadcopter, UAV\n\n")
            
            for detector, result in results.items():
                f.write(f"{detector.upper()} RESULTS:\n")
                f.write("-" * 20 + "\n")
                
                bg_results = result["background_results"]
                f.write(f"Overall mAP: {bg_results.get('overall_mAP', 0):.3f}%\n")
                f.write(f"Images processed: {result['images_processed']}\n")
                f.write(f"Total predictions: {result['total_predictions']}\n")
                
                # Background breakdown
                for bg in ['sky', 'tree', 'ground']:
                    bg_key = f"{bg}_bbox"
                    sample_key = f"{bg}_samples"
                    if bg_key in bg_results:
                        ap = bg_results[bg_key]["AP"]
                        samples = bg_results.get(sample_key, 0)
                        f.write(f"{bg.title()} - mAP: {ap:.3f}%, Samples: {samples}\n")
                
                f.write("\n")

    def _generate_additional_analysis(self, results: Dict[str, Any], output_dir: Path):
        """Generate additional analysis outputs for comprehensive evaluation"""
        try:
            analysis_dir = output_dir / "additional_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # 1. Create model comparison matrix
            self._create_model_comparison_matrix(results, analysis_dir)
            
            # 2. Generate recommendation engine
            self._generate_model_recommendations(results, analysis_dir)
            
            # 3. Create performance timeline (if multiple evaluations exist)
            self._create_performance_timeline(analysis_dir)
            
        except Exception as e:
            print(f"Failed to generate additional analysis: {e}")

    def _create_model_comparison_matrix(self, results: Dict[str, Any], analysis_dir: Path):
        """Create comprehensive model comparison matrix"""
        try:
            import pandas as pd
            import numpy as np
            
            # Collect all metrics for comparison
            comparison_data = []
            
            for detector, detector_results in results["results_by_detector"].items():
                row_data = {"Detector": detector}
                
                # Single best strategy
                if "single_best" in detector_results.get("strategy_results", {}):
                    single_best = detector_results["strategy_results"]["single_best"]
                    row_data["Best_Single_Prompt"] = single_best.get("best_prompt", "N/A")
                    row_data["Best_Single_Performance"] = single_best.get("best_performance", 0)
                
                # Average strategy
                if "average" in detector_results.get("strategy_results", {}):
                    average = detector_results["strategy_results"]["average"]
                    overall_avg = average.get("overall_average", {})
                    row_data["Average_Performance"] = overall_avg.get("mean_mAP", 0)
                    row_data["Performance_Std"] = overall_avg.get("std_mAP", 0)
                
                # Sensitivity
                if "sensitivity" in detector_results.get("strategy_results", {}):
                    sensitivity = detector_results["strategy_results"]["sensitivity"]
                    sens_metrics = sensitivity.get("sensitivity_metrics", {})
                    row_data["Coefficient_of_Variation"] = sens_metrics.get("coefficient_of_variation", 0)
                    row_data["Performance_Range"] = sens_metrics.get("performance_range", 0)
                
                comparison_data.append(row_data)
            
            # Create DataFrame and save
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                df.to_csv(analysis_dir / "model_comparison_matrix.csv", index=False)
                
                # Create summary statistics
                summary_stats = df.describe()
                summary_stats.to_csv(analysis_dir / "model_statistics_summary.csv")
        
        except Exception as e:
            print(f"Failed to create model comparison matrix: {e}")

    def _generate_model_recommendations(self, results: Dict[str, Any], analysis_dir: Path):
        """Generate personalized model recommendations based on use cases"""
        try:
            recommendations = {
                "use_case_recommendations": {
                    "highest_accuracy": {"detector": None, "score": 0, "reason": ""},
                    "most_consistent": {"detector": None, "score": 0, "reason": ""},
                    "most_stable": {"detector": None, "score": 0, "reason": ""},
                    "best_for_production": {"detector": None, "score": 0, "reason": ""}
                },
                "background_specialists": {
                    "sky_specialist": {"detector": None, "score": 0},
                    "tree_specialist": {"detector": None, "score": 0},
                    "ground_specialist": {"detector": None, "score": 0}
                },
                "overall_ranking": []
            }
            
            # Analyze results for recommendations
            detector_scores = {}
            
            for detector, detector_results in results["results_by_detector"].items():
                scores = {"detector": detector}
                
                # Get average performance
                if "average" in detector_results.get("strategy_results", {}):
                    avg_results = detector_results["strategy_results"]["average"]
                    overall_avg = avg_results.get("overall_average", {})
                    scores["accuracy"] = overall_avg.get("mean_mAP", 0)
                    scores["consistency"] = 1 / max(overall_avg.get("std_mAP", 1), 0.1)  # Lower std = higher consistency
                
                # Get stability (inverse of coefficient of variation)
                if "sensitivity" in detector_results.get("strategy_results", {}):
                    sens_results = detector_results["strategy_results"]["sensitivity"]
                    sens_metrics = sens_results.get("sensitivity_metrics", {})
                    cv = sens_metrics.get("coefficient_of_variation", 100)
                    scores["stability"] = 100 / max(cv, 1)  # Lower CV = higher stability
                
                detector_scores[detector] = scores
            
            # Find best for each use case
            if detector_scores:
                # Highest accuracy
                best_accuracy = max(detector_scores.items(), key=lambda x: x[1].get("accuracy", 0))
                recommendations["use_case_recommendations"]["highest_accuracy"] = {
                    "detector": best_accuracy[0],
                    "score": best_accuracy[1].get("accuracy", 0),
                    "reason": f"Achieved {best_accuracy[1].get('accuracy', 0):.2f}% average mAP across all prompts"
                }
                
                # Most consistent
                best_consistency = max(detector_scores.items(), key=lambda x: x[1].get("consistency", 0))
                recommendations["use_case_recommendations"]["most_consistent"] = {
                    "detector": best_consistency[0],
                    "score": best_consistency[1].get("consistency", 0),
                    "reason": "Shows lowest variance across different prompt types"
                }
                
                # Most stable
                best_stability = max(detector_scores.items(), key=lambda x: x[1].get("stability", 0))
                recommendations["use_case_recommendations"]["most_stable"] = {
                    "detector": best_stability[0],
                    "score": best_stability[1].get("stability", 0),
                    "reason": "Least sensitive to prompt variations"
                }
                
                # Overall ranking (weighted combination)
                overall_scores = []
                for detector, scores in detector_scores.items():
                    # Weighted score: 40% accuracy, 30% consistency, 30% stability
                    weighted_score = (
                        0.4 * scores.get("accuracy", 0) +
                        0.3 * (scores.get("consistency", 0) * 10) +  # Scale consistency
                        0.3 * scores.get("stability", 0)
                    )
                    overall_scores.append((detector, weighted_score))
                
                overall_scores.sort(key=lambda x: x[1], reverse=True)
                recommendations["overall_ranking"] = [
                    {"rank": i+1, "detector": detector, "score": score}
                    for i, (detector, score) in enumerate(overall_scores)
                ]
            
            # Save recommendations
            with open(analysis_dir / "model_recommendations.json", 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            # Create human-readable recommendations
            with open(analysis_dir / "recommendations.txt", 'w') as f:
                f.write("VLM Model Recommendations\n")
                f.write("=" * 30 + "\n\n")
                
                f.write("RECOMMENDED MODELS BY USE CASE:\n")
                f.write("-" * 35 + "\n")
                
                for use_case, rec in recommendations["use_case_recommendations"].items():
                    if rec["detector"]:
                        f.write(f"{use_case.replace('_', ' ').title()}: {rec['detector']}\n")
                        f.write(f"  Score: {rec['score']:.2f}\n")
                        f.write(f"  Reason: {rec['reason']}\n\n")
                
                f.write("OVERALL RANKING:\n")
                f.write("-" * 20 + "\n")
                for item in recommendations["overall_ranking"]:
                    f.write(f"{item['rank']}. {item['detector']} (Score: {item['score']:.2f})\n")
        
        except Exception as e:
            print(f"Failed to generate model recommendations: {e}")

    def _create_performance_timeline(self, analysis_dir: Path):
        """Create performance timeline if multiple evaluations exist"""
        # This is a placeholder for future functionality to track performance over time
        # Could be implemented to compare results across multiple evaluation runs
        pass
    
    def _run_detection(self, detector_name: str, prompts: List[str]) -> List[Dict[str, Any]]:
        """Enhanced detection with SAHI and multi-scale"""
        predictions = []
        
        print(f"Running enhanced detection with prompts: {prompts}")
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.get_sample(idx)
            image_info = sample['image_info']
            image_id = str(image_info['file_name']).replace('.png', '').replace('.jpg', '')
            
            try:
                # Use integrated SAHI + multi-scale detection
                result = self._detect_with_sahi_multiscale(
                    sample['image_path'], 
                    detector_name, 
                    prompts,
                    scales=[0.8, 1.0, 1.2],
                    tile_size=512,
                    overlap_ratio=0.25
                )
                
                pred = {
                    'image_id': image_id,
                    'boxes': result.boxes,
                    'scores': result.scores,
                    'labels': result.labels
                }
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                predictions.append({
                    'image_id': image_id,
                    'boxes': [],
                    'scores': [],
                    'labels': []
                })
        
        return predictions

    def _detect_with_sahi_multiscale(self, image_path: str, detector_name: str, 
                                    prompts: List[str], scales: List[float],
                                    tile_size: int = 512, overlap_ratio: float = 0.25):
        """Integrated SAHI + Multi-scale detection"""
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        all_detections = []
        
        for scale in scales:
            # Resize image for this scale
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply SAHI on scaled image
            scale_detections = self._apply_sahi_detection(
                scaled_image, detector_name, prompts, tile_size, overlap_ratio
            )
            
            # Rescale boxes back to original image coordinates
            scale_factor = 1.0 / scale
            rescaled_boxes = []
            for box in scale_detections.boxes:
                x1, y1, x2, y2 = box
                rescaled_box = [x1 * scale_factor, y1 * scale_factor, 
                            x2 * scale_factor, y2 * scale_factor]
                rescaled_boxes.append(rescaled_box)
            
            scale_detections.boxes = rescaled_boxes
            all_detections.append(scale_detections)
        
        # Merge all scales and apply NMS
        return self._merge_multiscale_results(all_detections)
    

    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detector names"""
        return list(self.detector_system.detectors.keys()) if self.detector_system else []
    
    def update_dataset_filters(self, filters: Dict[str, Any]):
        """Update dataset filters and reload"""
        self.config["dataset"]["filters"] = filters
        self._setup_dataset()
        self._setup_evaluators()  # Need to recreate evaluators with new dataset


def create_example_config():
    """Create example configuration file"""
    example_config = {
        "dataset": {
            "data_root": "/path/to/DrIFT_dataset",
            "split": "val",
            "filters": {
                "image_filters": {
                    "view": ["Aerial"],
                    "source": ["Real"],
                    "weather": ["Normal"]
                },
                "annotation_filters": {
                    "background": [0, 1, 2]
                }
            }
        },
        "detectors": {
            "owlv2": {
                "enabled": True,
                "model_path": "google/owlv2-base-patch16-ensemble"
            },
            "yolo-world": {
                "enabled": True, 
                "model_path": "yolov8s-world.pt"
            },
            "grounding-dino": {
                "enabled": False,
                "model_path": "IDEA-Research/grounding-dino-base"
            }
        },
        "evaluation": {
            "output_dir": "./vlm_evaluation_results",
            "strategies": ["single_best", "average", "ensemble", "sensitivity"]
        }
    }
    
    with open("vlm_evaluation_config.json", 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print("Created example config: vlm_evaluation_config.json")


# Example usage
if __name__ == "__main__":
    # Create example config if it doesn't exist
    if not Path("vlm_evaluation_config.json").exists():
        create_example_config()
        print("Update the dataset path in vlm_evaluation_config.json and run again")
    else:
        # Load and run evaluation
        framework = VLMEvaluationFramework("vlm_evaluation_config.json")
        
        print("\nAvailable detectors:", framework.get_available_detectors())
        
        # Run quick evaluation first
        print("\n" + "="*50)
        print("RUNNING QUICK EVALUATION")
        print("="*50)
        quick_results = framework.run_quick_evaluation()
        
        # Optionally run comprehensive evaluation
        run_comprehensive = input("\nRun comprehensive evaluation? (y/n): ").lower() == 'y'
        
        if run_comprehensive:
            print("\n" + "="*50)
            print("RUNNING COMPREHENSIVE EVALUATION")
            print("="*50)
            comprehensive_results = framework.run_comprehensive_evaluation()
            
            print("Comprehensive evaluation completed!")
            print(f"Results saved to: {framework.config['evaluation']['output_dir']}")
