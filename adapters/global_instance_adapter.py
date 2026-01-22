"""
Global + Instance Temporal BCA+ Adapter

Main adapter that combines:
1. Global Cache: Class-level prototypes and statistics
2. Instance Cache: Per-object tracking with Kalman filtering
3. Multiple data association methods
4. Bayesian fusion of VLM + Global + Instance predictions

Key features:
- Backpropagation-free adaptation
- Modular components for ablation study
- Multiple fusion strategies
- Comprehensive debugging support
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image

try:
    from .base_adapter import BaseAdapter
    from .global_cache import GlobalCache
    from .instance_cache import InstanceCache, Track
    from .trackers import (
        create_associator, 
        BaseAssociator, 
        AssociationResult
    )
except ImportError:
    from base_adapter import BaseAdapter
    from global_cache import GlobalCache
    from instance_cache import InstanceCache, Track
    from trackers import (
        create_associator, 
        BaseAssociator, 
        AssociationResult
    )


class GlobalInstanceAdapter(BaseAdapter):
    """
    Global + Instance Temporal BCA+ Adapter.
    
    Combines class-level global cache with instance-level tracking
    for robust test-time adaptation.
    """
    
    def __init__(self, detector, config: dict):
        """
        Initialize adapter from config.
        
        Config structure:
        {
            "adaptation": {
                "type": "global_instance",
                "params": {
                    # Global cache params
                    "global": {
                        "enabled": true,
                        "prototype_momentum": 0.9,
                        "scale_momentum": 0.95,
                        "use_instance_clusters": true,
                        "max_clusters_per_class": 10,
                        "tau_global": 0.6  # Confidence threshold for global update
                    },
                    # Instance cache params
                    "instance": {
                        "enabled": true,
                        "max_tracks": 100,
                        "min_hits_to_confirm": 3,
                        "max_age": 10,
                        "feature_smoothing": 0.7,
                        "class_prob_smoothing": 0.8
                    },
                    # Tracker params
                    "tracker": {
                        "method": "hungarian",  # hungarian, bytetrack, jpda, deepsort
                        "iou_threshold": 0.3,
                        "feature_weight": 0.3,
                        "high_threshold": 0.6,  # For bytetrack
                        "low_threshold": 0.1    # For bytetrack
                    },
                    # Fusion params
                    "fusion": {
                        "mode": "entropy",  # entropy, weighted, vlm_only, global_only, instance_only
                        "vlm_weight": 0.4,
                        "global_weight": 0.3,
                        "instance_weight": 0.3,
                        "temperature": 1.0
                    },
                    # General params
                    "tau_update": 0.6,  # Confidence threshold for updates
                    "update_after_nms": true
                }
            }
        }
        """
        super().__init__(detector, config)
        
        # Parse config
        params = config.get('adaptation', {}).get('params', {})
        
        # Component enables (for ablation)
        global_params = params.get('global', {})
        instance_params = params.get('instance', {})
        tracker_params = params.get('tracker', {})
        fusion_params = params.get('fusion', {})
        
        self.use_global = global_params.get('enabled', True)
        self.use_instance = instance_params.get('enabled', True)
        
        # Class info
        self.num_classes = len(config['detector']['target_classes'])
        self.class_names = config['detector']['target_classes']
        self.feature_dim = None  # Will be set on first detection
        
        # Global cache config
        self.prototype_momentum = global_params.get('prototype_momentum', 0.9)
        self.scale_momentum = global_params.get('scale_momentum', 0.95)
        self.use_instance_clusters = global_params.get('use_instance_clusters', True)
        self.max_clusters_per_class = global_params.get('max_clusters_per_class', 10)
        self.tau_global = global_params.get('tau_global', 0.6)
        
        # Instance cache config
        self.max_tracks = instance_params.get('max_tracks', 100)
        self.min_hits_to_confirm = instance_params.get('min_hits_to_confirm', 3)
        self.max_age = instance_params.get('max_age', 10)
        self.feature_smoothing = instance_params.get('feature_smoothing', 0.7)
        self.class_prob_smoothing = instance_params.get('class_prob_smoothing', 0.8)
        
        # Tracker config
        self.tracker_method = tracker_params.get('method', 'hungarian')
        self.iou_threshold = tracker_params.get('iou_threshold', 0.3)
        self.feature_weight = tracker_params.get('feature_weight', 0.3)
        self.tracker_high_threshold = tracker_params.get('high_threshold', 0.6)
        self.tracker_low_threshold = tracker_params.get('low_threshold', 0.1)
        
        # Fusion config
        self.fusion_mode = fusion_params.get('mode', 'entropy')
        self.vlm_weight = fusion_params.get('vlm_weight', 0.4)
        self.global_weight = fusion_params.get('global_weight', 0.3)
        self.instance_weight = fusion_params.get('instance_weight', 0.3)
        self.temperature = fusion_params.get('temperature', 1.0)
        
        # General config
        self.tau_update = params.get('tau_update', 0.6)
        self.update_after_nms = params.get('update_after_nms', True)
        self.alpha = params.get('alpha', 0.7)  # For detector
        
        # NMS from detector config
        self.nms_iou_threshold = config['detector'].get('iou_threshold', 0.7)
        
        # Components (initialized lazily)
        self.global_cache: Optional[GlobalCache] = None
        self.instance_cache: Optional[InstanceCache] = None
        self.associator: Optional[BaseAssociator] = None
        
        # State
        self.frame_count = 0
        self.image_size = None
        
        # Debug
        self.debug_mode = params.get('debug', False)
        self._last_stats = {}
    
    def adapt_and_detect(self, 
                        image: Image.Image,
                        target_classes: List[str],
                        threshold: float = 0.10):
        """
        Run detection with Global+Instance adaptation.
        
        Pipeline:
        1. Get raw detections from VLM
        2. Initialize components if needed
        3. Predict tracks (motion model)
        4. Fuse VLM + Global + Instance predictions
        5. Apply NMS
        6. Associate detections to tracks
        7. Update Global cache
        8. Update Instance cache
        
        Args:
            image: Input image
            target_classes: List of class names
            threshold: Detection threshold
            
        Returns:
            DetectionResult
        """
        # Update class info if changed
        if self.class_names != target_classes:
            self.class_names = target_classes
            self.num_classes = len(target_classes)
            self.reset()
        
        # Store image size
        self.image_size = image.size
        
        # Stage 1: Get raw detections from VLM
        result = self.detector.detect_with_features(
            image, target_classes, threshold, self.alpha
        )
        
        detection_data = self._result_to_dict(result)
        
        if len(detection_data['boxes']) == 0:
            self.frame_count += 1
            return self._to_detection_result(detection_data)
        
        # Stage 2: Initialize components if needed
        if self.feature_dim is None:
            self._initialize_components(detection_data)
        
        # Set image size for global cache
        if self.global_cache is not None:
            self.global_cache.set_image_size(*self.image_size)
        
        # Stage 3: Predict tracks (before association)
        if self.use_instance and self.instance_cache is not None:
            self.instance_cache.predict()
        
        # Stage 4: Fuse predictions
        adapted_data = self._fuse_predictions(detection_data)
        
        # Stage 5: Filter by threshold and apply NMS
        mask = adapted_data['scores'] >= threshold
        if not np.any(mask):
            self.frame_count += 1
            return self._to_detection_result({
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': []
            })
        
        filtered_data = self._filter_by_mask(adapted_data, mask)
        final_data = self._apply_nms(filtered_data)
        
        # Stage 6: Associate detections to tracks
        if self.use_instance and self.instance_cache is not None:
            self._associate_and_update_tracks(final_data)
        
        # Stage 7: Update Global cache
        if self.use_global and self.global_cache is not None:
            self._update_global_cache(final_data)
        
        self.frame_count += 1
        return self._to_detection_result(final_data)
    
    def _initialize_components(self, detection_data: Dict):
        """Initialize all components from first detection."""
        
        # Get feature dimension
        if 'features' in detection_data and len(detection_data['features']) > 0:
            self.feature_dim = detection_data['features'].shape[1]
        else:
            self.feature_dim = 256  # Default
        
        # Initialize global cache
        if self.use_global:
            self.global_cache = GlobalCache(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                max_clusters_per_class=self.max_clusters_per_class,
                prototype_momentum=self.prototype_momentum,
                scale_momentum=self.scale_momentum,
                use_instance_clusters=self.use_instance_clusters
            )
            
            # Initialize from text embeddings if available
            if 'text_embeddings' in detection_data and detection_data['text_embeddings'] is not None:
                self.global_cache.initialize_from_text_embeddings(
                    detection_data['text_embeddings']
                )
        
        # Initialize instance cache
        if self.use_instance:
            self.instance_cache = InstanceCache(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                max_tracks=self.max_tracks,
                min_hits_to_confirm=self.min_hits_to_confirm,
                max_age=self.max_age,
                feature_smoothing=self.feature_smoothing,
                class_prob_smoothing=self.class_prob_smoothing
            )
        
        # Initialize associator
        self._init_associator()
        
        print(f"Initialized Global+Instance adapter: "
              f"global={self.use_global}, instance={self.use_instance}, "
              f"tracker={self.tracker_method}, fusion={self.fusion_mode}")
    
    def _init_associator(self):
        """Initialize the data associator."""
        if self.tracker_method == 'hungarian':
            self.associator = create_associator(
                'hungarian',
                iou_threshold=self.iou_threshold,
                feature_weight=self.feature_weight
            )
        elif self.tracker_method == 'bytetrack':
            self.associator = create_associator(
                'bytetrack',
                high_threshold=self.tracker_high_threshold,
                low_threshold=self.tracker_low_threshold,
                iou_threshold_high=self.iou_threshold,
                feature_weight=self.feature_weight
            )
        elif self.tracker_method == 'jpda':
            self.associator = create_associator(
                'jpda',
                feature_weight=self.feature_weight
            )
        elif self.tracker_method == 'deepsort':
            self.associator = create_associator(
                'deepsort',
                iou_threshold=self.iou_threshold
            )
        else:
            self.associator = create_associator('hungarian')
    
    def _fuse_predictions(self, detection_data: Dict) -> Dict:
        """
        Fuse VLM predictions with Global and Instance caches.
        
        Fusion modes:
        - entropy: Weight by inverse entropy (confidence)
        - weighted: Fixed weights for each source
        - vlm_only: Only VLM predictions
        - global_only: Only global cache
        - instance_only: Only instance cache
        """
        vlm_probs = detection_data['class_probs']
        features = detection_data['features']
        boxes = detection_data['boxes']
        
        N = len(vlm_probs)
        
        # Get global predictions
        if self.use_global and self.global_cache is not None:
            global_probs = self.global_cache.predict_class_probs(
                features, temperature=self.temperature
            )
        else:
            global_probs = vlm_probs.copy()
        
        # Get instance-modified predictions
        if self.use_instance and self.instance_cache is not None:
            instance_probs = np.zeros_like(vlm_probs)
            for i in range(N):
                instance_probs[i] = self.instance_cache.compute_instance_prior(
                    features[i], boxes[i], vlm_probs[i]
                )
        else:
            instance_probs = vlm_probs.copy()
        
        # Fuse based on mode
        if self.fusion_mode == 'vlm_only':
            final_probs = vlm_probs
        elif self.fusion_mode == 'global_only':
            final_probs = global_probs
        elif self.fusion_mode == 'instance_only':
            final_probs = instance_probs
        elif self.fusion_mode == 'weighted':
            final_probs = (
                self.vlm_weight * vlm_probs +
                self.global_weight * global_probs +
                self.instance_weight * instance_probs
            )
        else:  # entropy (default)
            final_probs = self._entropy_fusion(
                vlm_probs, global_probs, instance_probs
            )
        
        # Compute final scores and labels
        final_scores = np.max(final_probs, axis=1)
        final_labels = [self.class_names[np.argmax(p)] for p in final_probs]
        
        # Build result
        result = detection_data.copy()
        result['scores'] = final_scores
        result['labels'] = final_labels
        result['class_probs'] = final_probs
        result['vlm_probs'] = vlm_probs
        result['global_probs'] = global_probs
        result['instance_probs'] = instance_probs
        
        return result
    
    def _entropy_fusion(self,
                        vlm_probs: np.ndarray,
                        global_probs: np.ndarray,
                        instance_probs: np.ndarray) -> np.ndarray:
        """
        Entropy-weighted fusion of multiple probability sources.
        
        Lower entropy = higher confidence = higher weight.
        """
        eps = 1e-10
        
        # Compute entropies
        H_vlm = -np.sum(vlm_probs * np.log(vlm_probs + eps), axis=1)
        H_global = -np.sum(global_probs * np.log(global_probs + eps), axis=1)
        H_instance = -np.sum(instance_probs * np.log(instance_probs + eps), axis=1)
        
        # Convert to weights (inverse entropy)
        w_vlm = np.exp(-H_vlm)[:, np.newaxis]
        w_global = np.exp(-H_global)[:, np.newaxis]
        w_instance = np.exp(-H_instance)[:, np.newaxis]
        
        # Apply base weights
        w_vlm *= self.vlm_weight
        w_global *= self.global_weight
        w_instance *= self.instance_weight
        
        # Normalize and fuse
        total_weight = w_vlm + w_global + w_instance + eps
        
        fused = (
            w_vlm * vlm_probs +
            w_global * global_probs +
            w_instance * instance_probs
        ) / total_weight
        
        return fused
    
    def _associate_and_update_tracks(self, detection_data: Dict):
        """Associate detections to tracks and update instance cache."""
        
        if len(detection_data['boxes']) == 0:
            return
        
        # Get active tracks
        tracks = self.instance_cache.get_active_tracks()
        
        # Run association
        assoc_result = self.associator.associate(
            tracks, detection_data, self.frame_count
        )
        
        # Update matched tracks
        for track_idx, det_idx in assoc_result.matches:
            self.instance_cache.update_track(
                track_idx,
                detection_data['boxes'][det_idx],
                detection_data['features'][det_idx],
                detection_data['class_probs'][det_idx],
                detection_data['scores'][det_idx]
            )
        
        # Create new tracks for unmatched detections (high confidence only)
        for det_idx in assoc_result.unmatched_detections:
            if detection_data['scores'][det_idx] >= self.tau_update:
                self.instance_cache.create_track(
                    detection_data['boxes'][det_idx],
                    detection_data['features'][det_idx],
                    detection_data['class_probs'][det_idx],
                    detection_data['scores'][det_idx]
                )
    
    def _update_global_cache(self, detection_data: Dict):
        """Update global cache with confident detections."""
        
        confidence_mask = detection_data['scores'] >= self.tau_global
        
        if not confidence_mask.any():
            return
        
        # Update class statistics
        self.global_cache.update(
            detection_data['features'],
            detection_data['boxes'],
            detection_data['class_probs'],
            confidence_mask
        )
        
        # Update instance clusters
        if self.use_instance_clusters:
            class_indices = np.argmax(detection_data['class_probs'], axis=1)
            self.global_cache.update_instance_clusters(
                detection_data['features'][confidence_mask],
                detection_data['boxes'][confidence_mask],
                class_indices[confidence_mask]
            )
    
    def _apply_nms(self, result: Dict) -> Dict:
        """Apply Non-Maximum Suppression."""
        if len(result['boxes']) == 0:
            return result
        
        boxes = result['boxes']
        scores = result['scores']
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute IoU with remaining
            remaining = []
            for idx in sorted_indices[1:]:
                iou = self._compute_iou(boxes[current], boxes[idx])
                if iou < self.nms_iou_threshold:
                    remaining.append(idx)
            
            sorted_indices = np.array(remaining)
        
        return self._filter_by_indices(result, keep)
    
    def _filter_by_mask(self, data: Dict, mask: np.ndarray) -> Dict:
        """Filter detection data by boolean mask."""
        result = {
            'boxes': data['boxes'][mask],
            'scores': data['scores'][mask],
            'labels': [data['labels'][i] for i in np.where(mask)[0]],
        }
        
        # Filter detection-level arrays (shape matches number of detections)
        n_detections = len(data['boxes'])
        for key in ['features', 'class_probs', 'vlm_probs', 
                    'global_probs', 'instance_probs']:
            if key in data and data[key] is not None:
                if isinstance(data[key], np.ndarray) and len(data[key]) == n_detections:
                    result[key] = data[key][mask]
        
        # Copy class-level arrays without filtering (e.g., text_embeddings)
        if 'text_embeddings' in data and data['text_embeddings'] is not None:
            result['text_embeddings'] = data['text_embeddings']
        
        return result
    
    def _filter_by_indices(self, data: Dict, indices: List[int]) -> Dict:
        """Filter detection data by indices."""
        result = {
            'boxes': data['boxes'][indices],
            'scores': data['scores'][indices],
            'labels': [data['labels'][i] for i in indices],
        }
        
        # Filter detection-level arrays
        n_detections = len(data['boxes'])
        for key in ['features', 'class_probs', 'vlm_probs', 
                    'global_probs', 'instance_probs']:
            if key in data and data[key] is not None:
                if isinstance(data[key], np.ndarray) and len(data[key]) == n_detections:
                    result[key] = data[key][indices]
        
        # Copy class-level arrays without filtering
        if 'text_embeddings' in data and data['text_embeddings'] is not None:
            result['text_embeddings'] = data['text_embeddings']
        
        return result
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-8)
    
    def _result_to_dict(self, result) -> Dict:
        """Convert DetectionResult to dict."""
        return {
            'boxes': np.array(result.boxes) if result.boxes else np.array([]),
            'scores': np.array(result.scores) if result.scores else np.array([]),
            'labels': result.labels if result.labels else [],
            'features': result.features if hasattr(result, 'features') else None,
            'class_probs': result.class_probs if hasattr(result, 'class_probs') else None,
            'text_embeddings': result.text_embeddings if hasattr(result, 'text_embeddings') else None,
        }
    
    def _to_detection_result(self, data: Dict):
        """Convert dict to DetectionResult."""
        from vlm_detector_system_new import DetectionResult
        return DetectionResult(
            boxes=data['boxes'].tolist() if len(data['boxes']) > 0 else [],
            scores=data['scores'].tolist() if len(data['scores']) > 0 else [],
            labels=data['labels'] if data['labels'] else [],
            image_path="",
            model_path=self.detector.model_path
        )
    
    def reset(self):
        """Reset for new video."""
        if self.global_cache is not None:
            self.global_cache.reset()
        if self.instance_cache is not None:
            self.instance_cache.reset()
        self.frame_count = 0
        self.feature_dim = None
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics for debugging."""
        stats = {
            'frame_count': self.frame_count,
            'use_global': self.use_global,
            'use_instance': self.use_instance,
            'fusion_mode': self.fusion_mode,
            'tracker_method': self.tracker_method
        }
        
        if self.global_cache is not None:
            stats['global'] = self.global_cache.get_stats_summary()
        
        if self.instance_cache is not None:
            stats['instance'] = self.instance_cache.get_stats_summary()
        
        return stats


# =============================================================================
# Ablation-friendly Adapter Variants
# =============================================================================

class GlobalOnlyAdapter(GlobalInstanceAdapter):
    """Adapter using only global cache (no instance tracking)."""
    
    def __init__(self, detector, config: dict):
        # Force instance off
        if 'adaptation' not in config:
            config['adaptation'] = {'params': {}}
        if 'params' not in config['adaptation']:
            config['adaptation']['params'] = {}
        
        config['adaptation']['params']['instance'] = {'enabled': False}
        config['adaptation']['params']['fusion'] = {'mode': 'global_only'}
        
        super().__init__(detector, config)


class InstanceOnlyAdapter(GlobalInstanceAdapter):
    """Adapter using only instance tracking (no global cache)."""
    
    def __init__(self, detector, config: dict):
        # Force global off
        if 'adaptation' not in config:
            config['adaptation'] = {'params': {}}
        if 'params' not in config['adaptation']:
            config['adaptation']['params'] = {}
        
        config['adaptation']['params']['global'] = {'enabled': False}
        config['adaptation']['params']['fusion'] = {'mode': 'instance_only'}
        
        super().__init__(detector, config)


class TrackingOnlyAdapter(GlobalInstanceAdapter):
    """Adapter using only tracking (no probability fusion)."""
    
    def __init__(self, detector, config: dict):
        # Disable fusion
        if 'adaptation' not in config:
            config['adaptation'] = {'params': {}}
        if 'params' not in config['adaptation']:
            config['adaptation']['params'] = {}
        
        config['adaptation']['params']['fusion'] = {'mode': 'vlm_only'}
        
        super().__init__(detector, config)


# =============================================================================
# Factory Function
# =============================================================================

def create_global_instance_adapter(detector, config: dict, variant: str = 'full'):
    """
    Factory function for creating adapter variants.
    
    Args:
        detector: VLM detector instance
        config: Configuration dict
        variant: 'full', 'global_only', 'instance_only', 'tracking_only'
        
    Returns:
        Adapter instance
    """
    if variant == 'global_only':
        return GlobalOnlyAdapter(detector, config)
    elif variant == 'instance_only':
        return InstanceOnlyAdapter(detector, config)
    elif variant == 'tracking_only':
        return TrackingOnlyAdapter(detector, config)
    else:
        return GlobalInstanceAdapter(detector, config)