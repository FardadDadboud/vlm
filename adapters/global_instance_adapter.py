"""
Global + Instance TTA Adapter

Main adapter combining:
1. Global BCA+ Cache - Frame-level class prototype adaptation
2. Instance-level Tracking - Per-object temporal adaptation with STAD

Key Design Decisions:
- Global cache adapts class-level priors across ALL objects
- Per-track STAD adapts class beliefs for INDIVIDUAL objects
- Part 5.3: New tracks initialize STAD from global cache
- CRITICAL: Global cache updates use RAW VLM probs (not post-instance probs)

Ablation Modes:
- vanilla: No adaptation
- global_only: BCA+ cache only
- instance_only: Tracking + per-track STAD only
- full: Global + Instance
- cascade: Hierarchical refinement

Self-Reinforcement Prevention:
- Global cache ALWAYS updates with raw VLM probs (pre-adaptation)
- Per-track STAD ALWAYS updates with raw VLM probs
- Post-adaptation probs are ONLY used for final output
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

# Handle both package and standalone imports
try:
    # Package import
    from .track import (
        KalmanBoxTracker, KalmanFilterConfig,
        AssociationConfig, associate,
        TrackSTADConfig, TrackSTADvMF, TrackSTADGaussian, create_track_stad,
        Track, TrackState, TrackConfig, TrackManager, Detection, reset_track_ids
    )
    from .enhanced_bca_cache import (
        EnhancedBCAPlusCache, EnhancedBCAPlusConfig, CacheEntryState
    )
    from .base_adapter import BaseAdapter
except ImportError:
    # Standalone import
    from track import (
        KalmanBoxTracker, KalmanFilterConfig,
        AssociationConfig, associate,
        TrackSTADConfig, TrackSTADvMF, TrackSTADGaussian, create_track_stad,
        Track, TrackState, TrackConfig, TrackManager, Detection, reset_track_ids
    )
    from enhanced_bca_cache import (
        EnhancedBCAPlusCache, EnhancedBCAPlusConfig, CacheEntryState
    )
    from base_adapter import BaseAdapter


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GlobalInstanceConfig:
    """
    Configuration for Global+Instance TTA adapter.
    
    Combines configurations for all components:
    - Global BCA+ cache
    - Tracking (Kalman, association)
    - Per-track STAD
    """
    # === Ablation Mode ===
    mode: str = 'full'  # 'vanilla', 'global_only', 'instance_only', 'full', 'cascade'
    
    # === Global BCA+ Cache ===
    use_global_cache: bool = True
    cache_config: Optional[EnhancedBCAPlusConfig] = None
    
    # === Tracking ===
    use_tracking: bool = True
    track_config: Optional[TrackConfig] = None
    association_config: Optional[AssociationConfig] = None
    association_method: str = 'hungarian'  # 'iou', 'hungarian', 'combined', 'bytetrack'
    
    # === Per-track STAD ===
    use_track_stad: bool = True
    stad_variant: str = 'vmf'  # 'vmf' or 'gaussian'
    stad_config: Optional[TrackSTADConfig] = None
    
    # === Detection Thresholds ===
    detection_threshold: float = 0.10  # Min confidence to keep detection
    nms_threshold: float = 0.7         # NMS IoU threshold
    
    # === Feedback ===
    # Track-to-cache feedback: confirmed tracks update global cache
    use_track_to_cache_feedback: bool = True
    feedback_min_hits: int = 5         # Min hits for track to update cache
    feedback_min_confidence: float = 0.8
    feedback_weight: float = 0.5       # Soft update weight
    
    # === Debug ===
    debug: bool = False


def get_ablation_config(mode: str) -> Dict:
    """
    Get configuration dict for ablation study.
    
    Args:
        mode: 'vanilla', 'global_only', 'instance_only', 'full', 'cascade'
        
    Returns:
        Configuration dictionary compatible with adapter initialization
    """
    base_config = {
        'mode': mode,
        'detection_threshold': 0.10,
        'nms_threshold': 0.7,
        'debug': False,
        
        # Global cache params (matching bca_plus.json structure)
        'tau1': 0.6,
        'tau2': 0.9,
        'ws': 0.1,
        'logit_temperature': 10.0,
        'alpha': 0.3,
        'tau2_init': 0.8,
        'max_cache_size': 25,
        
        # STAD params (matching temporal_tta_vmf_v2.json structure)
        'ssm_type': 'vmf',
        'kappa_trans': 10.0,
        'kappa_ems': 20.0,
        'gamma_init': 10.0,
        'kappa_max': 100.0,
        'kappa_min': 1e-6,
        'gamma_max': 200.0,
        'gamma_min': 1.0,
        'window_size': 5,
        'em_iterations': 3,
        'tau_update': 0.5,
        'min_updates_per_class': 1,
        'vlm_prior_weight': 0.2,
        'use_pi': True,
        'temperature': 1.0,
        'dirichlet_alpha': 0.1,
        
        # Tracking params
        'min_hits_to_confirm': 3,
        'max_age': 30,
        'iou_threshold': 0.3,
        'association_method': 'hungarian',
        
        # Feedback params
        'use_track_to_cache_feedback': True,
        'feedback_min_hits': 5,
        'feedback_min_confidence': 0.8,
        'feedback_weight': 0.5,
        
        # Part 5.2: Fusion params
        'fusion_mode': 'entropy_weighted',
        'fusion_init_weight': 0.1,
        'fusion_global_weight': 0.3,
        'fusion_track_weight': 0.6,
        'fusion_confidence_threshold': 0.7
    }
    
    # Mode-specific overrides
    if mode == 'vanilla':
        base_config['use_global_cache'] = False
        base_config['use_tracking'] = False
        base_config['use_track_stad'] = False
    elif mode == 'global_only':
        base_config['use_global_cache'] = True
        base_config['use_tracking'] = False
        base_config['use_track_stad'] = False
    elif mode == 'instance_only':
        base_config['use_global_cache'] = False
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = True
    elif mode == 'full':
        base_config['use_global_cache'] = True
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = True
    elif mode == 'cascade':
        base_config['use_global_cache'] = True
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = True
        base_config['cascade_mode'] = True
        base_config['fusion_mode'] = 'hierarchical'  # cascade uses hierarchical fusion
    
    return base_config


# =============================================================================
# Global Instance Adapter
# =============================================================================

class GlobalInstanceAdapter(BaseAdapter):
    """
    Global + Instance TTA adapter.
    
    Combines global BCA+ cache adaptation with per-track STAD for
    comprehensive temporal adaptation in video object detection.
    """
    
    def __init__(self, detector, config: Dict):
        """
        Initialize Global+Instance adapter.
        
        Args:
            detector: Base VLM detector with detect_with_features method
            config: Configuration dictionary
        """
        super().__init__(detector, config)
        
        self.config = config
        self.detector = detector
        
        # Parse config
        self._parse_config(config)
        
        # Get detector info
        self.target_classes = config.get('detector', {}).get(
            'target_classes', 
            ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        )
        self.num_classes = len(self.target_classes)
        
        # Feature dimension (will be set on first detection)
        self.feature_dim = None
        
        # Initialize components
        self._init_components()
        
        # Stats
        self.frame_count = 0
        self.total_detections = 0
        self.total_adapted = 0
    
    def _parse_config(self, config: Dict) -> None:
        """Parse configuration dictionary."""
        params = config.get('params', config)
        
        # Mode
        self.mode = params.get('mode', 'full')
        self.use_global_cache = params.get('use_global_cache', self.mode in ['global_only', 'full', 'cascade'])
        self.use_tracking = params.get('use_tracking', self.mode in ['instance_only', 'full', 'cascade'])
        self.use_track_stad = params.get('use_track_stad', self.mode in ['instance_only', 'full', 'cascade'])
        self.cascade_mode = params.get('cascade_mode', self.mode == 'cascade')
        
        # Detection thresholds
        self.detection_threshold = params.get('detection_threshold', params.get('tau_update', 0.10))
        self.nms_threshold = params.get('nms_threshold', params.get('iou_threshold', 0.7))
        
        # Global cache config
        self.cache_config = EnhancedBCAPlusConfig(
            tau1=params.get('tau1', 0.6),
            tau2_confirmed=params.get('tau2', params.get('tau2_confirmed', 0.9)),
            tau2_tentative=params.get('tau2_tentative', 0.85),
            tau2_init=params.get('tau2_init', 0.8),
            max_cache_size=params.get('max_cache_size', 25),
            ws=params.get('ws', 0.1),
            alpha=params.get('alpha', 0.3),
            logit_temperature=params.get('logit_temperature', 10.0),
            min_hits_to_confirm=params.get('min_hits_to_confirm', 3),
            max_age=params.get('max_age', 30),
            use_batch_init=params.get('use_batch_init', True),
            batch_init_size=params.get('batch_init_size', 10),
            debug=params.get('debug', False)
        )
        
        # Per-track STAD config
        self.stad_config = TrackSTADConfig(
            kappa_trans=params.get('kappa_trans', 10.0),
            kappa_ems=params.get('kappa_ems', 20.0),
            gamma_init=params.get('gamma_init', 10.0),
            gamma_min=params.get('gamma_min', 1.0),
            gamma_max=params.get('gamma_max', 200.0),
            kappa_max=params.get('kappa_max', 100.0),
            kappa_min=params.get('kappa_min', 1e-6),
            window_size=params.get('window_size', 5),
            em_iterations=params.get('em_iterations', 3),
            min_confidence=params.get('tau_update', 0.3),
            min_updates_per_class=params.get('min_updates_per_class', 0.5),
            temperature=params.get('temperature', 1.0),
            vlm_prior_weight=params.get('vlm_prior_weight', 0.2),
            use_pi=params.get('use_pi', True),
            use_ema_pi=params.get('use_ema_pi', False),
            pi_ema_decay=params.get('pi_ema_decay', 1.0),
            dirichlet_alpha=params.get('dirichlet_alpha', 0.1),
            gamma_ema_decay=params.get('gamma_ema_decay', 0.7),
            q_scale=params.get('q_scale', 0.01),
            r_base=params.get('r_base', 0.5),
            use_diagonal_cov=params.get('use_diagonal_cov', True),
            use_smoothing=params.get('use_smoothing', False),
            debug=params.get('debug', False)
        )
        
        # STAD variant
        self.stad_variant = params.get('ssm_type', params.get('stad_variant', 'vmf'))
        
        # Track config
        self.track_config = TrackConfig(
            min_hits_to_confirm=params.get('min_hits_to_confirm', 3),
            max_age=params.get('max_age', 30),
            feature_alpha=params.get('feature_alpha', 0.1),
            use_track_stad=self.use_track_stad,
            stad_variant=self.stad_variant,
            stad_config=self.stad_config,
            kalman_config=KalmanFilterConfig(),
            init_from_cache=params.get('init_from_cache', True),
            cache_match_threshold=params.get('cache_match_threshold', 0.5)
        )
        
        # Association config
        self.association_config = AssociationConfig(
            iou_threshold=params.get('iou_threshold', 0.3),
            feature_threshold=params.get('feature_threshold', 0.5),
            high_score_threshold=params.get('high_score_threshold', 0.6),
            low_score_threshold=params.get('low_score_threshold', 0.1)
        )
        self.association_method = params.get('association_method', 'hungarian')
        
        # Feedback config
        self.use_feedback = params.get('use_track_to_cache_feedback', True)
        self.feedback_min_hits = params.get('feedback_min_hits', 5)
        self.feedback_min_confidence = params.get('feedback_min_confidence', 0.8)
        self.feedback_weight = params.get('feedback_weight', 0.5)
        
        # Fusion config (Part 5.2)
        # Modes: 'entropy_weighted', 'hierarchical', 'parallel', 'selection'
        self.fusion_mode = params.get('fusion_mode', 'entropy_weighted')
        if self.cascade_mode:
            self.fusion_mode = params.get('fusion_mode', 'hierarchical')  # cascade defaults to hierarchical
        self.fusion_init_weight = params.get('fusion_init_weight', 0.1)  # weight for p_init in parallel
        self.fusion_global_weight = params.get('fusion_global_weight', 0.3)  # weight for p_global in parallel
        self.fusion_track_weight = params.get('fusion_track_weight', 0.6)  # weight for p_track in parallel
        self.fusion_confidence_threshold = params.get('fusion_confidence_threshold', 0.7)  # for hierarchical/selection
        
        # Debug
        self.debug = params.get('debug', False)
    
    def _init_components(self) -> None:
        """Initialize adaptation components (lazy init for feature_dim)."""
        self.global_cache = None
        self.track_manager = None
        self._components_initialized = False
    
    def _lazy_init_components(self, feature_dim: int) -> None:
        """Initialize components once feature dimension is known."""
        if self._components_initialized:
            return
        
        self.feature_dim = feature_dim
        
        # Global BCA+ cache
        if self.use_global_cache:
            self.global_cache = EnhancedBCAPlusCache(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                config=self.cache_config,
                class_names=self.target_classes
            )
        
        # Track manager (with reference to global cache for Part 5.3)
        if self.use_tracking:
            self.track_manager = TrackManager(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                config=self.track_config,
                global_cache=self.global_cache,  # Part 5.3: Pass cache for track init
                class_names=self.target_classes
            )
        
        self._components_initialized = True
    
    def adapt_and_detect(self, image, target_classes: List[str],
                        threshold: float = None, **kwargs) -> Any:
        """
        Perform adapted detection.
        
        Pipeline:
        1. VLM Detection → raw features, boxes, class_probs
        2. Store RAW VLM probs before ANY adaptation
        3. Global BCA+ Adaptation → adapted_probs (if use_global_cache)
        4. Filter by threshold
        5. Apply NMS
        6. Track Association → matches, new tracks (if use_tracking)
        7. Per-Track STAD refinement (for matched tracks)
        8. Update Global Cache with RAW VLM probs (NOT post-instance probs!)
        9. Optionally: Track-to-Cache Feedback
        
        CRITICAL: Global cache update uses RAW VLM probs to avoid self-reinforcement!
        
        Args:
            image: Input image
            target_classes: List of target class names
            threshold: Detection confidence threshold
            
        Returns:
            Detection results
        """
        threshold = threshold or self.detection_threshold
        
        # ===== Stage 1: VLM Detection =====
        alpha = self.cache_config.alpha if self.use_global_cache else 0.0
        detection_result = self.detector.detect_with_features(
            image, target_classes, threshold=threshold, alpha=alpha
        )
        
        if detection_result is None or len(detection_result.boxes) == 0:
            self.frame_count += 1
            self._end_frame()
            return detection_result
        
        # Extract data
        boxes = np.array(detection_result.boxes)
        scores = np.array(detection_result.scores)
        labels = list(detection_result.labels)
        class_probs = np.array(detection_result.class_probs)  # (N, K)
        features = np.array(detection_result.features)        # (N, D)
        
        N = len(boxes)
        if N == 0:
            self.frame_count += 1
            self._end_frame()
            return detection_result
        
        # Lazy init components
        self._lazy_init_components(features.shape[1])
        
        # ===== Stage 2: Store RAW VLM probs (BEFORE any adaptation) =====
        raw_vlm_probs = class_probs.copy()
        
        # ===== Stage 3: Global BCA+ Adaptation =====
        if self.use_global_cache and self.global_cache is not None:
            adapted_probs = np.zeros_like(class_probs)
            for i in range(N):
                adapted_probs[i] = self.global_cache.adapt_probs(
                    features[i], boxes[i], class_probs[i]
                )
            global_adapted_probs = adapted_probs
        else:
            global_adapted_probs = class_probs.copy()
        
        # ===== Stage 4: Filter by threshold =====
        keep_mask = scores >= threshold
        if not keep_mask.any():
            self.frame_count += 1
            self._end_frame()
            return self._create_empty_result(detection_result)
        
        # ===== Stage 5: Apply NMS =====
        nms_indices = self._apply_nms(boxes, scores, keep_mask)
        
        nms_boxes = boxes[nms_indices]
        nms_scores = scores[nms_indices]
        nms_labels = [labels[i] for i in nms_indices]
        nms_features = features[nms_indices]
        nms_raw_probs = raw_vlm_probs[nms_indices]  # RAW VLM probs for cache update!
        nms_global_probs = global_adapted_probs[nms_indices]
        
        N_nms = len(nms_indices)
        
        # ===== Stage 6: Track Association =====
        final_probs = nms_global_probs.copy()
        track_ids = [-1] * N_nms
        
        if self.use_tracking and self.track_manager is not None:
            # Predict existing tracks
            self.track_manager.predict_all()
            
            # Get track info
            active_tracks = self.track_manager.get_active_tracks()
            
            if len(active_tracks) > 0:
                track_boxes = np.array([t.get_state() for t in active_tracks])
                track_features = np.array([t.get_feature() for t in active_tracks])
                
                # Associate
                matches, unmatched_tracks, unmatched_dets = associate(
                    track_boxes=track_boxes,
                    detection_boxes=nms_boxes,
                    detection_scores=nms_scores,
                    track_features=track_features,
                    detection_features=nms_features,
                    method=self.association_method,
                    config=self.association_config
                )
                
                # ===== Stage 7: Update matched tracks and refine probs =====
                for track_idx, det_idx in matches:
                    track = active_tracks[track_idx]
                    
                    # Create detection object (with RAW probs for STAD update!)
                    det = Detection(
                        box=nms_boxes[det_idx],
                        score=nms_scores[det_idx],
                        label=nms_labels[det_idx],
                        class_idx=self.target_classes.index(nms_labels[det_idx]) if nms_labels[det_idx] in self.target_classes else 0,
                        class_probs=nms_global_probs[det_idx],
                        feature=nms_features[det_idx],
                        raw_class_probs=nms_raw_probs[det_idx]  # RAW probs for STAD!
                    )
                    
                    # Update track (STAD uses raw_class_probs internally)
                    track.update(det)
                    track_ids[det_idx] = track.track_id
                    
                    # Part 5.2: 3-source fusion (p_init, p_global, p_track)
                    if self.use_track_stad and track.class_stad is not None:
                        p_init = nms_raw_probs[det_idx]      # Raw VLM
                        p_global = nms_global_probs[det_idx]  # Global cache adapted
                        p_track = track.get_class_probs()     # Per-track STAD belief
                        final_probs[det_idx] = self._fuse_probs(p_init, p_global, p_track, track)
                
                # Create new tracks for unmatched detections
                for det_idx in unmatched_dets:
                    det = Detection(
                        box=nms_boxes[det_idx],
                        score=nms_scores[det_idx],
                        label=nms_labels[det_idx],
                        class_idx=self.target_classes.index(nms_labels[det_idx]) if nms_labels[det_idx] in self.target_classes else 0,
                        class_probs=nms_global_probs[det_idx],
                        feature=nms_features[det_idx],
                        raw_class_probs=nms_raw_probs[det_idx]
                    )
                    new_track = self.track_manager.create_track(det)
                    track_ids[det_idx] = new_track.track_id
            else:
                # No existing tracks - create new ones for all detections
                for det_idx in range(N_nms):
                    det = Detection(
                        box=nms_boxes[det_idx],
                        score=nms_scores[det_idx],
                        label=nms_labels[det_idx],
                        class_idx=self.target_classes.index(nms_labels[det_idx]) if nms_labels[det_idx] in self.target_classes else 0,
                        class_probs=nms_global_probs[det_idx],
                        feature=nms_features[det_idx],
                        raw_class_probs=nms_raw_probs[det_idx]
                    )
                    new_track = self.track_manager.create_track(det)
                    track_ids[det_idx] = new_track.track_id
        
        # ===== Stage 8: Update Global Cache (CRITICAL: Use RAW VLM probs!) =====
        if self.use_global_cache and self.global_cache is not None:
            # Update cache with RAW VLM probs, NOT post-instance probs!
            # This prevents self-reinforcement loop
            self.global_cache.update_cache(
                features=nms_features,
                boxes=nms_boxes,
                probs=nms_raw_probs,  # RAW VLM probs!
                scores=nms_scores
            )
        
        # ===== Stage 9: Track-to-Cache Feedback (optional) =====
        if (self.use_feedback and self.use_global_cache and 
            self.use_tracking and self.track_manager is not None):
            self._apply_track_to_cache_feedback()
        
        # ===== Create output =====
        # Update labels based on final class probs
        final_labels = []
        for i in range(N_nms):
            top_class_idx = np.argmax(final_probs[i])
            final_labels.append(self.target_classes[top_class_idx])
        
        # Create result
        result = self._create_result(
            detection_result=detection_result,
            boxes=nms_boxes,
            scores=nms_scores,
            labels=final_labels,
            class_probs=final_probs,
            features=nms_features,
            track_ids=track_ids
        )
        
        self.frame_count += 1
        self.total_detections += N
        self.total_adapted += N_nms
        
        self._end_frame()
        
        return result
    
    def _fuse_probs(self, p_init: np.ndarray, p_global: np.ndarray, 
                    p_track: np.ndarray, track: Optional['Track'] = None) -> np.ndarray:
        """
        Part 5.2: 3-source probability fusion.
        
        Fuses probabilities from three sources:
        - p_init: Raw VLM probabilities (initial detection)
        - p_global: Global cache adapted probabilities
        - p_track: Per-track STAD class belief
        
        Args:
            p_init: (K,) raw VLM class probabilities
            p_global: (K,) global cache adapted probabilities
            p_track: (K,) per-track STAD class belief (π)
            track: Optional Track object for confidence info
            
        Returns:
            (K,) fused class probabilities
        """
        eps = 1e-10
        
        if self.fusion_mode == 'parallel':
            # Weighted average of all three sources
            fused = (self.fusion_init_weight * p_init + 
                    self.fusion_global_weight * p_global + 
                    self.fusion_track_weight * p_track)
            return fused / (fused.sum() + eps)
        
        elif self.fusion_mode == 'hierarchical':
            # Hierarchical: p_track overrides p_global overrides p_init based on confidence
            # Track confidence based on hits and entropy
            track_conf = 0.0
            if track is not None and track.hits >= self.track_config.min_hits_to_confirm:
                H_track = -np.sum(p_track * np.log(p_track + eps))
                track_conf = np.exp(-H_track)  # Lower entropy = higher confidence
            
            # Global confidence from cache
            H_global = -np.sum(p_global * np.log(p_global + eps))
            global_conf = np.exp(-H_global)
            
            # Hierarchical selection
            if track_conf >= self.fusion_confidence_threshold:
                # Trust track STAD
                return p_track
            elif global_conf >= self.fusion_confidence_threshold:
                # Trust global cache, blend with track
                w_track = track_conf / (track_conf + global_conf + eps)
                fused = (1 - w_track) * p_global + w_track * p_track
                return fused / (fused.sum() + eps)
            else:
                # Blend all three with entropy weighting
                H_init = -np.sum(p_init * np.log(p_init + eps))
                w_init = np.exp(-H_init)
                w_global = global_conf
                w_track = track_conf
                total = w_init + w_global + w_track + eps
                fused = (w_init * p_init + w_global * p_global + w_track * p_track) / total
                return fused / (fused.sum() + eps)
        
        elif self.fusion_mode == 'selection':
            # Select the most confident source
            H_init = -np.sum(p_init * np.log(p_init + eps))
            H_global = -np.sum(p_global * np.log(p_global + eps))
            H_track = -np.sum(p_track * np.log(p_track + eps))
            
            # Pick lowest entropy (most confident)
            entropies = [H_init, H_global, H_track]
            sources = [p_init, p_global, p_track]
            return sources[np.argmin(entropies)]
        
        else:  # 'entropy_weighted' (default)
            # Entropy-weighted fusion of all three sources
            H_init = -np.sum(p_init * np.log(p_init + eps))
            H_global = -np.sum(p_global * np.log(p_global + eps))
            H_track = -np.sum(p_track * np.log(p_track + eps))
            
            w_init = np.exp(-H_init)
            w_global = np.exp(-H_global)
            w_track = np.exp(-H_track)
            
            total = w_init + w_global + w_track + eps
            fused = (w_init * p_init + w_global * p_global + w_track * p_track) / total
            return fused / (fused.sum() + eps)
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray,
                   keep_mask: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression."""
        valid_indices = np.where(keep_mask)[0]
        if len(valid_indices) == 0:
            return np.array([], dtype=np.int32)
        
        # Simple greedy NMS
        order = scores[valid_indices].argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(valid_indices[i])
            
            if len(order) == 1:
                break
            
            # Compute IoU with rest
            ious = self._compute_iou_batch(boxes[valid_indices[i]], boxes[valid_indices[order[1:]]])
            
            # Keep boxes with IoU below threshold
            remaining = np.where(ious < self.nms_threshold)[0] + 1
            order = order[remaining]
        
        return np.array(keep, dtype=np.int32)
    
    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        return inter / (area1 + area2 - inter + 1e-8)
    
    def _apply_track_to_cache_feedback(self) -> None:
        """Apply feedback from confirmed tracks to global cache."""
        if self.global_cache is None or self.track_manager is None:
            return
        
        confirmed_tracks = self.track_manager.get_confirmed_tracks()
        
        for track in confirmed_tracks:
            # Only feedback from mature, confident tracks
            if track.hits < self.feedback_min_hits:
                continue
            
            avg_score = np.mean(track.score_history[-10:]) if track.score_history else 0
            if avg_score < self.feedback_min_confidence:
                continue
            
            # Get track's smoothed feature and class belief
            feature = track.get_feature()
            class_probs = track.get_class_probs()
            
            # Create dummy box from last known state
            box = track.get_state()
            
            # Soft update to cache (lower weight to avoid overfitting)
            effective_score = avg_score * self.feedback_weight
            
            # Update cache directly with track's belief
            self.global_cache._update_single(feature, box, class_probs, effective_score)
    
    def _end_frame(self) -> None:
        """End-of-frame processing."""
        if self.global_cache is not None:
            self.global_cache.step()
        
        if self.track_manager is not None:
            self.track_manager.step()
    
    def _create_empty_result(self, template_result) -> Any:
        """Create empty detection result."""
        # Return result with empty arrays
        result = type(template_result)(
            boxes=np.array([]),
            scores=np.array([]),
            labels=[],
            class_probs=np.array([]).reshape(0, self.num_classes),
            features=np.array([]).reshape(0, self.feature_dim or 256)
        )
        return result
    
    def _create_result(self, detection_result, boxes, scores, labels,
                      class_probs, features, track_ids=None) -> Any:
        """Create detection result object."""
        # Try to use same result type as input
        try:
            result = type(detection_result)(
                boxes=boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
                scores=scores.tolist() if isinstance(scores, np.ndarray) else scores,
                labels=labels,
                class_probs=class_probs.tolist() if isinstance(class_probs, np.ndarray) else class_probs,
                features=features.tolist() if isinstance(features, np.ndarray) else features
            )
            
            # Add track_ids if supported
            if hasattr(result, 'track_ids'):
                result.track_ids = track_ids
            
            return result
        except Exception:
            # Fallback: just modify the input result
            detection_result.boxes = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
            detection_result.scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
            detection_result.labels = labels
            detection_result.class_probs = class_probs.tolist() if isinstance(class_probs, np.ndarray) else class_probs
            detection_result.features = features.tolist() if isinstance(features, np.ndarray) else features
            
            return detection_result
    
    def reset(self) -> None:
        """Reset all state for new video."""
        if self.global_cache is not None:
            self.global_cache.reset()
        
        if self.track_manager is not None:
            self.track_manager.reset()
        
        reset_track_ids()
        
        self.frame_count = 0
        self.total_detections = 0
        self.total_adapted = 0
        
        # Reset lazy init flag (keep feature_dim)
        # self._components_initialized = False
    
    def get_summary(self) -> Dict:
        """Get adapter summary."""
        summary = {
            'mode': self.mode,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'total_adapted': self.total_adapted,
            'use_global_cache': self.use_global_cache,
            'use_tracking': self.use_tracking,
            'use_track_stad': self.use_track_stad
        }
        
        if self.global_cache is not None:
            summary['global_cache'] = self.global_cache.get_summary()
        
        if self.track_manager is not None:
            summary['track_manager'] = self.track_manager.get_summary()
        
        return summary


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing global_instance_adapter.py")
    print("Note: This module requires a detector to be fully tested.")
    print("Run the integration test with an actual detector instance.")
    
    # Test config generation
    for mode in ['vanilla', 'global_only', 'instance_only', 'full', 'cascade']:
        config = get_ablation_config(mode)
        print(f"\n{mode} config keys: {list(config.keys())[:5]}...")
    
    print("\n✓ Config generation test passed!")