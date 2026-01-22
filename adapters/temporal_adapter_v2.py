"""
Temporal Adapter v2 for STAD-based Test-Time Adaptation

Fixes vs v1:
1. Uses renamed feature parameter (features, not raw_features)
2. Per-class update gating (not global count)
3. Debug logging for runaway behavior detection
4. Support for both vMF and Gaussian SSM variants
5. Cleaner fusion implementation

This adapter integrates temporal SSM predictions with VLM detector outputs.
"""

import numpy as np
from typing import List, Dict, Optional, Literal
from PIL import Image

# Import base adapter and SSM
from .base_adapter import BaseAdapter
from .temporal_ssm_v2 import (
    TemporalSSMvMF, 
    TemporalSSMGaussian, 
    create_temporal_ssm,
    VMFUtils,
    DEFAULT_DEBUG_EVERY
)


class TemporalTTAAdapterV2(BaseAdapter):
    """
    Temporal Test-Time Adaptation adapter using STAD.
    
    Key improvements over v1:
    - Proper soft-EM in SSM (not hard labels)
    - Mixing coefficients π
    - A_D usage for bounded predictions
    - Per-class update gating
    - Support for vMF and Gaussian variants
    """
    
    def __init__(self, detector, config: dict):
        super().__init__(detector, config)
        
        params = config.get('adaptation', {}).get('params', {})
        
        # SSM type selection
        self.ssm_type: Literal["vmf", "gaussian"] = params.get('ssm_type', 'vmf')
        
        # STAD parameters (vMF specific)
        self.kappa_trans_init = params.get('kappa_trans', 100.0)
        self.kappa_ems_init = params.get('kappa_ems', 100.0)
        self.gamma_init = params.get('gamma_init', 10.0)
        self.window_size = params.get('window_size', 5)
        self.em_iterations = params.get('em_iterations', 3)
        
        # STAD parameters (Gaussian specific)
        self.q_scale = params.get('q_scale', 0.01)
        self.r_base = params.get('r_base', 0.5)
        self.use_smoothing = params.get('use_smoothing', False)
        
        # Configurable bounds for kappa and gamma
        self.kappa_max = params.get('kappa_max', 500.0)
        self.kappa_min = params.get('kappa_min', 1e-6)
        self.gamma_max = params.get('gamma_max', 500.0)
        self.gamma_min = params.get('gamma_min', 1.0)
        
        # Confidence threshold for updates
        self.tau_update = params.get('tau_update', 0.8)
        
        # Fusion settings
        self.fusion_mode = params.get('fusion_mode', 'entropy')  # entropy, weighted, vlm_only, ssm_only
        self.fusion_weight = params.get('fusion_weight', 0.5)  # For weighted mode
        self.alpha = params.get('alpha', 0.7)  # Detector alpha
        
        # EMA settings
        self.use_pi = params.get('use_pi', False)
        self.use_ema_pi = params.get('use_ema_pi', True)
        self.pi_ema_decay = params.get('pi_ema_decay', 0.9)
        
        # Update settings - NOW PER-CLASS
        self.update_after_nms = params.get('update_after_nms', True)
        self.min_updates_per_class = params.get('min_updates_per_class', 2)
        
        # Whether to update global kappa (can be unstable)
        self.update_global_kappa = params.get('update_global_kappa', False)
        
        # VLM prior weight for responsibility computation
        self.vlm_prior_weight = params.get('vlm_prior_weight', 0.0)
        
        # Dirichlet alpha for pi prior
        self.dirichlet_alpha = params.get('dirichlet_alpha', 1e-2)
        
        # Class info
        self.num_classes = len(config['detector']['target_classes'])
        self.class_names = config['detector']['target_classes']
        
        # SSM instance
        self.ssm: Optional[TemporalSSMvMF | TemporalSSMGaussian] = None
        self.feature_dim: Optional[int] = None
        self.temperature = params.get('temperature', 1.0)
        
        # NMS
        self.iou_threshold = config['detector'].get('iou_threshold', 0.3)
        
        # Stats
        self.frame_count = 0
        self._last_stats = {}
        
        # Debug mode
        self.debug_mode = params.get('debug', False)
        self.debug_every = params.get('debug_every', DEFAULT_DEBUG_EVERY)
        self._adaptation_log = []
        
        # Adapter-level health counters
        self._num_updates_triggered = 0
        self._num_updates_skipped = 0
    
    def _dlog(self, msg: str, force: bool = False):
        """Print debug message if debug enabled and (forced or throttled)."""
        if not self.debug_mode:
            return
        if force or (self.frame_count % self.debug_every == 0):
            print(msg)
    
    def _dwarn(self, msg: str):
        """Print warning message (always prints if debug enabled)."""
        if self.debug_mode:
            print(msg)
    
    def _entropy(self, probs: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute entropy of probability distribution."""
        return -np.sum(probs * np.log(probs + 1e-10), axis=axis)
    
    def adapt_and_detect(self, image: Image.Image, target_classes: List[str],
                         threshold: float = 0.10):
        """
        Run detection with temporal adaptation.
        
        Args:
            image: Input image
            target_classes: List of class names to detect
            threshold: Detection confidence threshold
            
        Returns:
            DetectionResult with adapted predictions
        """
        # Handle class changes
        if self.class_names != target_classes:
            self.class_names = target_classes
            self.num_classes = len(target_classes)
            self.ssm = None  # Reset SSM
            self._dlog(f"[STAD][classes-change] resetting SSM for new classes", force=True)
        
        # Get detections with features from detector
        result = self.detector.detect_with_features(
            image, target_classes, threshold, self.alpha
        )
        
        if result.raw_features is None or result.raw_text_embeddings is None:
            raise RuntimeError("Detector did not return raw features")
        
        # Pack detection data
        detection_data = {
            'boxes': np.array(result.boxes) if result.boxes else np.array([]),
            'scores': np.array(result.scores) if result.scores else np.array([]),
            'labels': result.labels if result.labels else [],
            'features': result.features,  # Projected features (same space as text)
            'class_probs': result.class_probs,  # RAW VLM probs - keep reference
            'text_embeddings': result.text_embeddings,
            'raw_features': result.raw_features,  # Raw encoder features
            'raw_text_embeddings': result.raw_text_embeddings
        }
        
        # Store raw VLM class_probs for SSM updates (CRITICAL: avoid self-reinforcement)
        raw_vlm_class_probs = result.class_probs.copy() if result.class_probs is not None else None
        
        num_raw = len(detection_data['boxes'])
        
        # Track stats
        self._last_stats = {
            'frame': self.frame_count,
            'num_detections': num_raw,
        }
        
        # Handle empty detections
        if num_raw == 0:
            self._dlog(f"[STAD][frame={self.frame_count}] no detections")
            self.frame_count += 1
            return self._to_detection_result(detection_data)
        
        # Initialize SSM if needed
        if self.ssm is None:
            self._initialize_ssm(detection_data['text_embeddings'])
        
        # Apply temporal adaptation
        ssm_probs = None
        if self.ssm is not None:
            # Get SSM predictions using projected features
            features = detection_data['features']  # Use projected features
            ssm_probs = self.ssm.predict(features)
            
            # Fuse predictions
            adapted_data = self._fuse_predictions(detection_data, ssm_probs)
            
            # Debug logging for fusion
            if self.debug_mode:
                self._log_fusion_stats(detection_data, ssm_probs, adapted_data)
        else:
            adapted_data = detection_data.copy()
        
        # Filter by threshold
        mask = np.array(adapted_data['scores']) >= threshold
        num_after_thr = int(mask.sum())
        
        if not np.any(mask):
            self._dlog(
                f"[STAD][frame={self.frame_count}][{self.ssm_type}] det_raw={num_raw} "
                f"det_thr=0 tau={self.tau_update:.2f} thr={threshold:.2f}"
            )
            self.frame_count += 1
            return self._to_detection_result({
                'boxes': np.array([]), 
                'scores': np.array([]), 
                'labels': []
            })
        
        # Apply mask - keep both fused (for output) and raw VLM probs (for SSM update)
        filtered = {
            'boxes': adapted_data['boxes'][mask],
            'scores': adapted_data['scores'][mask],
            'labels': [adapted_data['labels'][i] for i in np.where(mask)[0]],
            'features': adapted_data['features'][mask],
            'class_probs': adapted_data['class_probs'][mask] if adapted_data.get('class_probs') is not None else None
        }
        
        # Keep raw VLM probs for SSM update (CRITICAL FIX: use raw, not fused)
        raw_vlm_probs_filtered = raw_vlm_class_probs[mask] if raw_vlm_class_probs is not None else None
        
        # Apply NMS
        final_result = self._apply_nms(filtered)
        num_after_nms = len(final_result.get('boxes', []))
        
        # Also apply NMS indices to raw VLM probs
        if hasattr(self, '_last_nms_keep_indices') and raw_vlm_probs_filtered is not None:
            raw_vlm_probs_for_update = raw_vlm_probs_filtered[self._last_nms_keep_indices]
        else:
            raw_vlm_probs_for_update = raw_vlm_probs_filtered
        
        # Debug: Frame header (throttled)
        self._dlog(
            f"[STAD][frame={self.frame_count}][{self.ssm_type}] det_raw={num_raw} "
            f"det_thr={num_after_thr} det_nms={num_after_nms} tau={self.tau_update:.2f} "
            f"thr={threshold:.2f} win={self.window_size}"
        )
        
        # Update SSM with confident detections using RAW VLM probs (not fused!)
        if self.update_after_nms and self.ssm is not None:
            if len(final_result.get('features', [])) > 0:
                self._update_ssm(
                    final_result['features'],  # Projected features
                    final_result['scores'],
                    raw_vlm_probs_for_update  # CRITICAL: Use raw VLM probs, not fused!
                )
        
        self.frame_count += 1
        return self._to_detection_result(final_result)
    
    def _initialize_ssm(self, text_embeddings: np.ndarray):
        """Initialize the appropriate SSM variant."""
        self.feature_dim = text_embeddings.shape[1]
        
        self._dlog(
            f"[STAD][adapter-init] Initializing {self.ssm_type.upper()} SSM: "
            f"{self.num_classes} classes, {self.feature_dim}D",
            force=True
        )
        
        if self.ssm_type == 'vmf':
            self.ssm = TemporalSSMvMF(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                kappa_trans_init=self.kappa_trans_init,
                kappa_ems_init=self.kappa_ems_init,
                gamma_init=self.gamma_init,
                window_size=self.window_size,
                em_iterations=self.em_iterations,
                temperature=self.temperature,
                use_pi=self.use_pi,
                use_ema_pi=self.use_ema_pi,
                pi_ema_decay=self.pi_ema_decay,
                min_updates_per_class=self.min_updates_per_class,
                update_global_kappa=self.update_global_kappa,
                vlm_prior_weight=self.vlm_prior_weight,
                dirichlet_alpha=self.dirichlet_alpha,
                kappa_max=self.kappa_max,
                kappa_min=self.kappa_min,
                gamma_max=self.gamma_max,
                gamma_min=self.gamma_min,
                class_names=self.class_names,
                debug=self.debug_mode,
                debug_every=self.debug_every
            )
        else:  # gaussian
            self.ssm = TemporalSSMGaussian(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                q_scale=self.q_scale,
                r_base=self.r_base,
                window_size=self.window_size,
                use_smoothing=self.use_smoothing,
                min_updates_per_class=self.min_updates_per_class,
                dirichlet_alpha=self.dirichlet_alpha,
                class_names=self.class_names,
                debug=self.debug_mode,
                debug_every=self.debug_every
            )
        
        self.ssm.initialize_from_text_embeddings(text_embeddings)
    
    def _update_ssm(self, features: np.ndarray, scores: np.ndarray,
                    class_probs: Optional[np.ndarray] = None):
        """
        Update SSM with confident detections.
        
        Args:
            features: (N, D) projected features (same space as text embeddings)
            scores: (N,) confidence scores
            class_probs: (N, K) soft class probabilities
        """
        if self.ssm is None:
            return
        
        # Create confidence mask
        confidence_mask = scores >= self.tau_update
        num_confident = int(confidence_mask.sum())
        
        # Note: Per-class gating is now handled inside the SSM
        # We just pass all confident detections and let SSM decide
        
        if num_confident == 0:
            self._num_updates_skipped += 1
            self._dlog(
                f"[STAD][update-skip] num_conf=0 < tau_update={self.tau_update:.2f}",
                force=True
            )
            return
        
        # Compute class distribution for logging
        if self.debug_mode and class_probs is not None:
            conf_probs = class_probs[confidence_mask]
            pred_classes = np.argmax(conf_probs, axis=1)
            class_counts = {}
            for c in pred_classes:
                name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
                class_counts[name] = class_counts.get(name, 0) + 1
            
            # Sort by count
            sorted_counts = sorted(class_counts.items(), key=lambda x: -x[1])
            top_classes_str = " ".join([f"{k}={v}" for k, v in sorted_counts[:4]])
            
            self._dlog(
                f"[STAD][update] num_conf={num_confident} | top_pred_classes: {top_classes_str} | "
                f"using RAW VLM probs (not fused)",
                force=True
            )
            
            # Log entropy stats of raw VLM probs being used for update
            conf_entropy = self._entropy(class_probs[confidence_mask], axis=1)
            self._dlog(
                f"[STAD][update-input] raw_VLM_probs H: mean={conf_entropy.mean():.2f} "
                f"min={conf_entropy.min():.2f} max={conf_entropy.max():.2f}",
                force=True
            )
        
        self._num_updates_triggered += 1
        
        # Call SSM update with RAW VLM probs (not fused!)
        self.ssm.update(features, confidence_mask, class_probs)
    
    def _fuse_predictions(self, detection_data: Dict, ssm_probs: np.ndarray) -> Dict:
        """
        Fuse VLM and SSM predictions.
        
        Modes:
        - 'entropy': Weight by inverse entropy (confident source gets more weight)
        - 'weighted': Fixed weighted average
        - 'vlm_only': Use only VLM predictions
        - 'ssm_only': Use only SSM predictions
        
        Args:
            detection_data: Dict with VLM detections
            ssm_probs: (N, K) SSM predicted probabilities
            
        Returns:
            Updated detection_data dict
        """
        vlm_probs = detection_data['class_probs']
        vlm_labels = detection_data['labels']
        
        if self.fusion_mode == 'ssm_only':
            final_probs = ssm_probs
            w_vlm_flat = np.zeros(len(vlm_probs))
            w_ssm_flat = np.ones(len(vlm_probs))
        elif self.fusion_mode == 'vlm_only':
            final_probs = vlm_probs
            w_vlm_flat = np.ones(len(vlm_probs))
            w_ssm_flat = np.zeros(len(vlm_probs))
        elif self.fusion_mode == 'weighted':
            # Fixed weighted average
            w = self.fusion_weight
            final_probs = w * ssm_probs + (1 - w) * vlm_probs
            w_vlm_flat = np.full(len(vlm_probs), 1 - w)
            w_ssm_flat = np.full(len(vlm_probs), w)
        else:  # entropy-based fusion
            eps = 1e-10
            
            # Compute entropies
            H_vlm = -np.sum(vlm_probs * np.log(vlm_probs + eps), axis=1)
            H_ssm = -np.sum(ssm_probs * np.log(ssm_probs + eps), axis=1)
            
            # Weights: lower entropy = higher weight
            w_vlm = np.exp(-H_vlm)[:, np.newaxis]
            w_ssm = np.exp(-H_ssm)[:, np.newaxis]
            
            # Store flattened weights for logging
            w_vlm_flat = w_vlm.flatten()
            w_ssm_flat = w_ssm.flatten()
            
            # Normalize and combine
            w_total = w_vlm + w_ssm + eps
            final_probs = (w_vlm * vlm_probs + w_ssm * ssm_probs) / w_total
            
            # Debug: Log fusion weights (throttled)
            if self.debug_mode:
                w_ratio = w_ssm_flat / (w_vlm_flat + eps)
                self._dlog(
                    f"[STAD][fusion-weights] w_vlm: min={w_vlm_flat.min():.3f} mean={w_vlm_flat.mean():.3f} "
                    f"max={w_vlm_flat.max():.3f} | w_ssm: min={w_ssm_flat.min():.3f} mean={w_ssm_flat.mean():.3f} "
                    f"max={w_ssm_flat.max():.3f} | ratio w_ssm/w_vlm: min={w_ratio.min():.2f} "
                    f"mean={w_ratio.mean():.2f} max={w_ratio.max():.2f}"
                )
                
                # Warn if SSM dominates fusion
                if w_ratio.mean() > 5.0:
                    self._dwarn(f"[STAD][WARN] SSM dominates fusion: mean(w_ssm/w_vlm)={w_ratio.mean():.1f}")
        
        # Get labels before and after fusion
        fused_labels = [self.class_names[np.argmax(final_probs[i])] 
                       for i in range(len(final_probs))]
        
        # Debug: Log top-1 class distribution pre/post fusion
        if self.debug_mode:
            from collections import Counter
            pre_counts = Counter(vlm_labels)
            post_counts = Counter(fused_labels)
            pre_str = " ".join([f"{k}={v}" for k, v in sorted(pre_counts.items(), key=lambda x: -x[1])[:4]])
            post_str = " ".join([f"{k}={v}" for k, v in sorted(post_counts.items(), key=lambda x: -x[1])[:4]])
            self._dlog(f"[STAD][fusion-classes] pre: {pre_str} | post: {post_str}")
        
        # Update detection data
        result = detection_data.copy()
        result['scores'] = np.max(final_probs, axis=1)
        result['labels'] = fused_labels
        result['class_probs'] = final_probs
        
        return result
    
    def _log_fusion_stats(self, detection_data: Dict, ssm_probs: np.ndarray, 
                          adapted_data: Dict):
        """Log detailed fusion statistics for debugging."""
        vlm_probs = detection_data['class_probs']
        vlm_scores = detection_data['scores']
        vlm_labels = detection_data['labels']
        fused_scores = adapted_data['scores']
        fused_labels = adapted_data['labels']
        
        eps = 1e-10
        H_vlm = -np.sum(vlm_probs * np.log(vlm_probs + eps), axis=1)
        H_ssm = -np.sum(ssm_probs * np.log(ssm_probs + eps), axis=1)
        
        # Count label changes
        num_changed = sum(1 for v, f in zip(vlm_labels, fused_labels) if v != f)
        
        stats = {
            'frame': self.frame_count,
            'n_detections': len(vlm_scores),
            'vlm_score_range': (vlm_scores.min(), vlm_scores.max()),
            'vlm_entropy_mean': H_vlm.mean(),
            'ssm_entropy_mean': H_ssm.mean(),
            'fused_score_range': (fused_scores.min(), fused_scores.max()),
            'score_increase_mean': (fused_scores - vlm_scores).mean(),
            'labels_changed': num_changed,
        }
        
        # Get SSM debug stats
        if hasattr(self.ssm, 'get_debug_stats'):
            stats.update(self.ssm.get_debug_stats())
        
        self._adaptation_log.append(stats)
        
        # Compact fusion log (throttled via _dlog)
        n = stats['n_detections']
        self._dlog(
            f"[STAD][fusion] score_pre mean={vlm_scores.mean():.2f} max={vlm_scores.max():.2f} | "
            f"score_post mean={fused_scores.mean():.2f} max={fused_scores.max():.2f} | "
            f"H_vlm mean={H_vlm.mean():.2f} | H_ssm mean={H_ssm.mean():.2f} | "
            f"changed={num_changed}/{n}"
        )
        
        # Warnings for suspicious behavior - multiple thresholds
        if stats['ssm_entropy_mean'] < 0.05:
            self._dwarn(f"[STAD][WARN] SSM entropy CRITICAL: {stats['ssm_entropy_mean']:.4f} (COLLAPSE DETECTED)")
        elif stats['ssm_entropy_mean'] < 0.1:
            self._dwarn(f"[STAD][WARN] SSM entropy very low: {stats['ssm_entropy_mean']:.3f} (approaching collapse)")
        
        score_change = fused_scores.mean() - vlm_scores.mean()
        if abs(score_change) > 0.3:
            self._dwarn(f"[STAD][WARN] Large score change: {score_change:+.3f}")
    
    def _apply_nms(self, result: Dict) -> Dict:
        """Apply Non-Maximum Suppression."""
        if len(result['boxes']) == 0:
            self._last_nms_keep_indices = np.array([], dtype=int)
            return result
        
        boxes, scores = result['boxes'], result['scores']
        
        def iou(b1, b2):
            x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / (area1 + area2 - inter + 1e-8)
        
        # Sort by score descending
        idx = np.argsort(scores)[::-1]
        keep = []
        
        while len(idx) > 0:
            keep.append(idx[0])
            if len(idx) == 1:
                break
            # Filter boxes with IoU >= threshold
            idx = np.array([i for i in idx[1:] 
                           if iou(boxes[idx[0]], boxes[i]) < self.iou_threshold])
        
        # Store keep indices for external use (e.g., mapping raw VLM probs)
        self._last_nms_keep_indices = np.array(keep)
        
        # Build output
        out = {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': [result['labels'][i] for i in keep],
        }
        
        # Copy optional fields
        for key in ['features', 'class_probs']:
            if key in result and result[key] is not None:
                out[key] = result[key][keep]
        
        return out
    
    def _to_detection_result(self, result: Dict):
        """Convert dict to DetectionResult."""
        from vlm_detector_system_new import DetectionResult
        return DetectionResult(
            boxes=result['boxes'].tolist() if len(result['boxes']) > 0 else [],
            scores=result['scores'].tolist() if len(result['scores']) > 0 else [],
            labels=result['labels'] if result['labels'] else [],
            image_path="",
            model_path=self.detector.model_path
        )
    
    def reset(self):
        """Reset adapter state for new video sequence."""
        self.ssm = None
        self.frame_count = 0
        self._adaptation_log = []
        self._num_updates_triggered = 0
        self._num_updates_skipped = 0
        self._dlog("[STAD][reset] SSM cleared for new sequence", force=True)
    
    def get_health_stats(self) -> Dict:
        """Get health statistics for monitoring."""
        stats = {
            'frame_count': self.frame_count,
            'adapter_updates_triggered': self._num_updates_triggered,
            'adapter_updates_skipped': self._num_updates_skipped,
        }
        
        if self.ssm is not None:
            stats['ssm_type'] = self.ssm_type
            if hasattr(self.ssm, 'num_updates_total'):
                stats['ssm_updates_total'] = self.ssm.num_updates_total
            if hasattr(self.ssm, 'num_updates_skipped'):
                stats['ssm_updates_skipped'] = self.ssm.num_updates_skipped
            if hasattr(self.ssm, 'num_updates_by_class') and self.ssm.num_updates_by_class is not None:
                stats['ssm_updates_by_class'] = dict(zip(self.class_names, 
                                                         self.ssm.num_updates_by_class.tolist()))
        
        return stats
    
    def get_adaptation_log(self) -> List[Dict]:
        """Get the adaptation log for analysis."""
        return self._adaptation_log.copy()
    
    def get_ssm_state_summary(self) -> Optional[Dict]:
        """Get summary of current SSM state."""
        if self.ssm is None:
            return None
        
        summary = {
            'ssm_type': self.ssm_type,
            'frame_count': self.frame_count,
        }
        
        if hasattr(self.ssm, 'get_mixing_coefficients'):
            pi = self.ssm.get_mixing_coefficients()
            if pi is not None:
                summary['pi'] = pi.tolist()
        
        if hasattr(self.ssm, 'get_concentrations'):
            gamma = self.ssm.get_concentrations()
            if gamma is not None:
                summary['gamma'] = gamma.tolist()
        
        if hasattr(self.ssm, 'state'):
            if hasattr(self.ssm.state, 'kappa_ems'):
                summary['kappa_ems'] = self.ssm.state.kappa_ems
            if hasattr(self.ssm.state, 'kappa_trans'):
                summary['kappa_trans'] = self.ssm.state.kappa_trans
            if hasattr(self.ssm.state, 'class_update_counts'):
                summary['class_update_counts'] = self.ssm.state.class_update_counts.tolist()
        
        # Add health stats
        summary['health'] = self.get_health_stats()
        
        return summary


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Alias for backward compatibility
TemporalTTAAdapter = TemporalTTAAdapterV2


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    print("Temporal Adapter v2 - Standalone test")
    print("Note: Full test requires detector instance")
    print("This module is designed to be imported and used with a detector.")