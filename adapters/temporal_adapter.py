"""
Diagnostic Temporal Adapter

Purpose: Investigate WHY fusion causes confidence to explode

Hypothesis: The issue might be one of:
1. SSM probs are very confident (due to high gamma/kappa)
2. SSM similarities increase after prototype updates
3. Entropy weighting gives too much weight to SSM
4. Raw features and text embeddings are in different subspaces
"""

import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from .base_adapter import BaseAdapter
from .temporal_ssm import TemporalSSM


class TemporalTTAAdapter(BaseAdapter):
    """Diagnostic version to investigate fusion behavior."""
    
    def __init__(self, detector, config: dict):
        super().__init__(detector, config)
        
        params = config.get('adaptation', {}).get('params', {})
        
        # STAD parameters
        self.kappa_trans_init = params.get('kappa_trans', 50.0)
        self.kappa_ems_init = params.get('kappa_ems', 10.0)
        self.window_size = params.get('window_size', 5)
        self.em_iterations = params.get('em_iterations', 3)
        self.tau_update = params.get('tau_update', 0.8)
        self.kappa_max = params.get('kappa_max', 30.0)
        self.kappa_min = params.get('kappa_min', 1e-6)
        
        # Fusion
        self.fusion_mode = params.get('fusion_mode', 'entropy')
        self.fusion_weight = params.get('fusion_weight', 0.5)
        self.alpha = params.get('alpha', 0.7)
        
        # EMA
        self.kappa_ema = params.get('kappa_ema', 0.9)
        self.use_ema_kappa = params.get('use_ema_kappa', True)
        
        # Update settings
        self.update_after_nms = params.get('update_after_nms', True)
        self.min_updates_per_class = params.get('min_updates_per_class', 2)
        
        # Class info
        self.num_classes = len(config['detector']['target_classes'])
        self.class_names = config['detector']['target_classes']
        
        # SSM
        self.ssm: Optional[TemporalSSM] = None
        self.feature_dim: Optional[int] = None
        self.temperature = params.get('temperature', 1.0)
        
        # NMS
        self.iou_threshold = config['detector'].get('iou_threshold', 0.3)
        
        # Stats
        self.frame_count = 0
        self._last_stats = {}
        
        # Debug - ALWAYS ON for diagnosis
        self.debug_mode = True
        self._adaptation_log = []
        self._prev_prototypes = None
    
    def adapt_and_detect(self, image: Image.Image, target_classes: List[str],
                         threshold: float = 0.10):
        """Run detection with detailed diagnostics."""
        
        if self.class_names != target_classes:
            self.class_names = target_classes
            self.num_classes = len(target_classes)
            self.ssm = None
        
        result = self.detector.detect_with_features(image, target_classes, threshold, self.alpha)
        
        if result.raw_features is None or result.raw_text_embeddings is None:
            raise RuntimeError("Detector did not return raw features")
        
        detection_data = {
            'boxes': np.array(result.boxes) if result.boxes else np.array([]),
            'scores': np.array(result.scores) if result.scores else np.array([]),
            'labels': result.labels if result.labels else [],
            'features': result.features,
            'class_probs': result.class_probs,
            'text_embeddings': result.text_embeddings,
            'raw_features': result.raw_features,
            'raw_text_embeddings': result.raw_text_embeddings
        }
        
        # Store original VLM scores
        vlm_scores_original = detection_data['scores'].copy()
        vlm_probs_original = detection_data['class_probs'].copy()
        
        self._last_stats = {
            'frame': self.frame_count,
            'num_detections': len(detection_data['boxes']),
        }
        
        if len(detection_data['boxes']) == 0:
            self.frame_count += 1
            return self._to_detection_result(detection_data)
            
        
        # Initialize SSM
        if self.ssm is None:
            self._initialize_ssm(detection_data['text_embeddings'])
        
        # =====================================================================
        # DIAGNOSTIC: Detailed fusion analysis
        # =====================================================================
        if self.ssm is not None and self.ssm.state.initialized:
            features = detection_data['features']
            # vlm_probs = detection_data['class_probs']
            
            # # Get raw similarities BEFORE softmax
            # features_norm = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-8)
            # raw_similarities = features_norm @ self.ssm.state.rho.T  # (N, K)
            
            # # Get SSM predictions
            proto_probs = self.ssm.predict(features)
            
            # # Compute fusion diagnostics
            # self._diagnose_fusion(
            #     vlm_probs=vlm_probs,
            #     vlm_scores=vlm_scores_original,
            #     proto_probs=proto_probs,
            #     raw_similarities=raw_similarities,
            #     gamma=self.ssm.get_concentrations()
            # )
            
            # Do actual fusion
            adapted_data = self._fuse_predictions(detection_data, proto_probs)
            # fused_scores = adapted_data['scores']
            
            # Compare scores
            # self._compare_scores(vlm_scores_original, fused_scores)
        else:
            adapted_data = detection_data.copy()
        
        # Filter and NMS
        mask = np.array(adapted_data['scores']) >= threshold
        if not np.any(mask):
            self.frame_count += 1
            return self._to_detection_result({
                'boxes': np.array([]), 'scores': np.array([]), 'labels': []
            })
        
        filtered = {
            'boxes': adapted_data['boxes'][mask],
            'scores': adapted_data['scores'][mask],
            'labels': [adapted_data['labels'][i] for i in np.where(mask)[0]],
            'features': adapted_data['features'][mask],
            'raw_features': adapted_data['raw_features'][mask],
            'class_probs': adapted_data['class_probs'][mask] if adapted_data.get('class_probs') is not None else None
        }
        
        final_result = self._apply_nms(filtered)
        
        # Update SSM
        if self.update_after_nms and self.ssm is not None:
            if len(final_result.get('features', [])) > 0:
                self._update_ssm(
                    final_result['features'],
                    final_result['scores'],
                    final_result.get('class_probs')
                )
        
        self.frame_count += 1
        return self._to_detection_result(final_result)
    
    # def _diagnose_fusion(self, vlm_probs, vlm_scores, proto_probs, raw_similarities, gamma):
    #     """Detailed diagnosis of fusion behavior."""
        
    #     print(f"\n{'='*70}")
    #     print(f"FRAME {self.frame_count} FUSION DIAGNOSIS")
    #     print(f"{'='*70}")
        
    #     N = len(vlm_probs)
        
    #     # 1. Raw similarities (before gamma weighting)
    #     print(f"\n1. RAW SIMILARITIES (query @ prototype.T):")
    #     print(f"   Shape: {raw_similarities.shape}")
    #     print(f"   Mean per class: {raw_similarities.mean(axis=0)}")
    #     print(f"   Overall range: [{raw_similarities.min():.3f}, {raw_similarities.max():.3f}]")
    #     print(f"   Overall mean: {raw_similarities.mean():.3f}")
        
    #     # 2. Gamma (concentration) values
    #     print(f"\n2. GAMMA (concentration) VALUES:")
    #     print(f"   {gamma}")
        
    #     # 3. SSM probs (after gamma weighting + softmax)
    #     proto_max = proto_probs.max(axis=1)
    #     proto_entropy = -np.sum(proto_probs * np.log(proto_probs + 1e-10), axis=1)
    #     print(f"\n3. SSM PROBS:")
    #     print(f"   Max prob range: [{proto_max.min():.3f}, {proto_max.max():.3f}]")
    #     print(f"   Max prob mean: {proto_max.mean():.3f}")
    #     print(f"   Entropy range: [{proto_entropy.min():.3f}, {proto_entropy.max():.3f}]")
    #     print(f"   Mean entropy: {proto_entropy.mean():.3f}")
    #     print(f"   (Lower entropy = more confident)")
        
    #     # 4. VLM probs
    #     vlm_max = vlm_probs.max(axis=1)
    #     vlm_entropy = -np.sum(vlm_probs * np.log(vlm_probs + 1e-10), axis=1)
    #     print(f"\n4. VLM PROBS:")
    #     print(f"   Max prob range: [{vlm_max.min():.3f}, {vlm_max.max():.3f}]")
    #     print(f"   Max prob mean: {vlm_max.mean():.3f}")
    #     print(f"   Entropy range: [{vlm_entropy.min():.3f}, {vlm_entropy.max():.3f}]")
    #     print(f"   Mean entropy: {vlm_entropy.mean():.3f}")
        
    #     # 5. Entropy-based weights
    #     w_vlm = np.exp(-vlm_entropy)
    #     w_proto = np.exp(-proto_entropy)
    #     weight_ratio = w_proto / (w_vlm + 1e-10)
    #     print(f"\n5. ENTROPY FUSION WEIGHTS:")
    #     print(f"   w_vlm range: [{w_vlm.min():.3f}, {w_vlm.max():.3f}], mean: {w_vlm.mean():.3f}")
    #     print(f"   w_proto range: [{w_proto.min():.3f}, {w_proto.max():.3f}], mean: {w_proto.mean():.3f}")
    #     print(f"   w_proto/w_vlm ratio range: [{weight_ratio.min():.3f}, {weight_ratio.max():.3f}]")
    #     print(f"   Mean ratio: {weight_ratio.mean():.3f}")
    #     print(f"   (Ratio > 1 means SSM dominates)")
        
    #     # 6. Compare class predictions
    #     vlm_pred = np.argmax(vlm_probs, axis=1)
    #     proto_pred = np.argmax(proto_probs, axis=1)
    #     agreement = (vlm_pred == proto_pred).mean()
    #     print(f"\n6. CLASS AGREEMENT:")
    #     print(f"   VLM and SSM agree on {agreement*100:.1f}% of detections")
        
    #     # 7. Score distribution analysis
    #     print(f"\n7. VLM SCORES (before fusion):")
    #     print(f"   Range: [{vlm_scores.min():.3f}, {vlm_scores.max():.3f}]")
    #     print(f"   Mean: {vlm_scores.mean():.3f}")
    #     print(f"   >0.5: {(vlm_scores > 0.5).sum()}")
    #     print(f"   >0.8: {(vlm_scores > 0.8).sum()}")
    #     print(f"   >0.9: {(vlm_scores > 0.9).sum()}")
        
    #     print(f"{'='*70}\n")
    
    # def _compare_scores(self, vlm_scores, fused_scores):
    #     """Compare VLM vs fused scores."""
        
    #     print(f"SCORE COMPARISON:")
    #     print(f"   VLM scores > {self.tau_update}: {(vlm_scores > self.tau_update).sum()}")
    #     print(f"   Fused scores > {self.tau_update}: {(fused_scores > self.tau_update).sum()}")
        
    #     score_increase = fused_scores - vlm_scores
    #     print(f"   Score increase range: [{score_increase.min():.3f}, {score_increase.max():.3f}]")
    #     print(f"   Mean increase: {score_increase.mean():.3f}")
    #     print(f"   Detections boosted above {self.tau_update}: {((vlm_scores <= self.tau_update) & (fused_scores > self.tau_update)).sum()}")
    
    def _initialize_ssm(self, text_embeddings: np.ndarray):
        """Initialize SSM."""
        self.feature_dim = text_embeddings.shape[1]
        
        print(f"SSM initialized: {self.num_classes} classes, {self.feature_dim}D")
        
        self.ssm = TemporalSSM(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            kappa_trans_init=self.kappa_trans_init,
            kappa_ems_init=self.kappa_ems_init,
            window_size=self.window_size,
            em_iterations=self.em_iterations,
            use_ema_kappa=self.use_ema_kappa,
            kappa_ema=self.kappa_ema,
            temperature=self.temperature,
            kappa_max=self.kappa_max,
            kappa_min=self.kappa_min
        )
        
        self.ssm.initialize_from_text_embeddings(text_embeddings)
        # self._prev_prototypes = text_embeddings.copy()
        
        # DIAGNOSTIC: Show initial prototype similarities
        # print(f"\nINITIAL PROTOTYPE ANALYSIS:")
        # proto = self.ssm.get_prototypes()
        # proto_sims = proto @ proto.T
        # print(f"   Inter-prototype similarities (diagonal=1):")
        # print(f"   {proto_sims}")
    
    def _update_ssm(self, raw_features: np.ndarray, scores: np.ndarray,
                    class_probs: Optional[np.ndarray] = None):
        """Update SSM with diagnostics."""
        
        confidence_mask = scores >= self.tau_update
        num_confident = int(confidence_mask.sum())
        
        # print(f"\nUPDATE: {num_confident} confident detections (threshold={self.tau_update})")
        
        if num_confident < self.min_updates_per_class:
            print(f"   Skipping (need at least {self.min_updates_per_class})")
            return
        
        # Store previous
        # old_proto = self.ssm.get_prototypes().copy()
        # old_gamma = self.ssm.get_concentrations().copy()
        
        # Update
        self.ssm.update(raw_features, confidence_mask, class_probs)
        
        # Compare
        # new_proto = self.ssm.get_prototypes()
        # new_gamma = self.ssm.get_concentrations()
        
        # Prototype change
        # proto_sim = np.sum(old_proto * new_proto, axis=1)
        # print(f"   Prototype cosine sim (old vs new): {proto_sim}")
        # print(f"   Gamma change: {old_gamma} -> {new_gamma}")
        
        # Check if prototypes are moving toward visual features
        # conf_features = raw_features[confidence_mask]
        # conf_features_norm = conf_features / (np.linalg.norm(conf_features, axis=1, keepdims=True) + 1e-8)
        
        # Mean similarity of confident features to OLD prototypes
        # old_sims = conf_features_norm @ old_proto.T
        # print(f"   Conf features to OLD prototypes: mean={old_sims.mean():.3f}")
        
        # Mean similarity of confident features to NEW prototypes  
        # new_sims = conf_features_norm @ new_proto.T
        # print(f"   Conf features to NEW prototypes: mean={new_sims.mean():.3f}")
        
        # self._prev_prototypes = new_proto.copy()
    
    def _fuse_predictions(self, detection_data: Dict, proto_probs: np.ndarray) -> Dict:
        """Fuse predictions."""
        vlm_probs = detection_data['class_probs']
        
        if self.fusion_mode == 'prototype_only':
            final_probs = proto_probs
        elif self.fusion_mode == 'vlm_only':
            final_probs = vlm_probs
        elif self.fusion_mode == 'weighted':
            final_probs = self.fusion_weight * proto_probs + (1 - self.fusion_weight) * vlm_probs
        else:  # entropy
            eps = 1e-10
            H_vlm = -np.sum(vlm_probs * np.log(vlm_probs + eps), axis=1)
            H_proto = -np.sum(proto_probs * np.log(proto_probs + eps), axis=1)
            w_vlm = np.exp(-H_vlm)[:, np.newaxis]
            w_proto = np.exp(-H_proto)[:, np.newaxis]
            final_probs = (w_vlm * vlm_probs + w_proto * proto_probs) / (w_vlm + w_proto + eps)
        
        result = detection_data.copy()
        result['scores'] = np.max(final_probs, axis=1)
        result['labels'] = [self.class_names[np.argmax(final_probs[i])] for i in range(len(final_probs))]
        result['class_probs'] = final_probs
        
        return result
    
    def _apply_nms(self, result: Dict) -> Dict:
        """Apply NMS."""
        if len(result['boxes']) == 0:
            return result
        
        boxes, scores = result['boxes'], result['scores']
        
        def iou(b1, b2):
            x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            return inter / (area1 + area2 - inter + 1e-8)
        
        idx = np.argsort(scores)[::-1]
        keep = []
        while len(idx) > 0:
            keep.append(idx[0])
            if len(idx) == 1:
                break
            idx = np.array([i for i in idx[1:] if iou(boxes[idx[0]], boxes[i]) < self.iou_threshold])
        
        out = {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': [result['labels'][i] for i in keep],
        }
        for key in ['features', 'raw_features', 'class_probs']:
            if key in result and result[key] is not None:
                out[key] = result[key][keep]
        return out
    
    def _to_detection_result(self, result: Dict):
        """Convert to DetectionResult."""
        from vlm_detector_system_new import DetectionResult
        return DetectionResult(
            boxes=result['boxes'].tolist() if len(result['boxes']) > 0 else [],
            scores=result['scores'].tolist() if len(result['scores']) > 0 else [],
            labels=result['labels'] if result['labels'] else [],
            image_path="",
            model_path=self.detector.model_path
        )
    
    def reset(self):
        """Reset for new video."""
        self.ssm = None
        self._prev_prototypes = None
        self.frame_count = 0