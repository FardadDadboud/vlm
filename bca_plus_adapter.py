import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from PIL import Image

class BCAPlusCache:
    """Dynamic cache for BCA+ adaptation"""
    def __init__(self, num_classes=6):
        self.F_cache = None  # Feature embeddings (d x M)
        self.B_cache = None  # Spatial scales (2 x M) - [w,h]
        self.V_cache = None  # Priors (K x M)
        self.C_cache = None  # Update counts (M,)
        self.num_classes = num_classes
        self.M = 0  # Number of cache entries
        
    def is_empty(self):
        return self.F_cache is None or self.M == 0

class BCAPlusAdapter:
    """Main BCA+ Test-Time Adaptation"""
    def __init__(self, detector, tau1=0.8, tau2=0.8, ws=0.2, num_classes=6):
        self.detector = detector
        self.cache = BCAPlusCache(num_classes=num_classes)
        self.tau1 = tau1  # Confidence threshold
        self.tau2 = tau2  # Similarity threshold  
        self.ws = ws      # Balance weight for scale vs feature
        self.num_classes = num_classes
        self.class_names = None  # Will be set during first detection
    
    @property
    def model_path(self):
        """Delegate model_path to wrapped detector"""
        return self.detector.model_path
    
    def adapt_and_detect(self, image, target_classes, threshold=0.10):
        """
        BCA+ Algorithm 1:
        1. Extract ALL queries (900) with features and scores
        2. Apply Bayesian inference to ALL 900 if cache exists
        3. Filter by threshold AFTER inference
        4. Apply NMS to remove overlapping detections
        5. Update cache with high-confidence detections
        """
        if self.class_names is None:
            self.class_names = target_classes
        
        # Stage 1: Get ALL 900 queries with proper scores
        all_queries_result = self._detect_with_features(image, target_classes, threshold)
        
        if len(all_queries_result['boxes']) == 0:
            return self._to_detection_result(all_queries_result)
        
        # Stage 2: Apply Bayesian inference to ALL queries if cache exists
        if not self.cache.is_empty():
            adapted_result = self._bayesian_inference_all_queries(all_queries_result)
        else:
            # First frame: no cache, use initial predictions
            adapted_result = all_queries_result
        
        # Stage 3: Filter by threshold AFTER inference
        mask = np.array(adapted_result['scores']) >= threshold
        if not np.any(mask):
            return self._to_detection_result({
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': [],
                'features': np.array([])
            })
        
        filtered_result = {
            'boxes': np.array(adapted_result['boxes'])[mask],
            'scores': np.array(adapted_result['scores'])[mask],
            'labels': [adapted_result['labels'][i] for i in np.where(mask)[0]],
            'features': adapted_result['features'][mask],
            'class_probs': adapted_result['class_probs'][mask] if 'class_probs' in adapted_result else None
        }
        
        # Stage 4: Apply NMS to remove overlapping detections
        nms_result = self._apply_nms(filtered_result, iou_threshold=0.3)
        
        # Stage 5: Update cache with high-confidence detections (from NMS result)
        high_conf_mask = np.array(nms_result['scores']) >= self.tau1
        if np.any(high_conf_mask):
            high_conf_result = {
                'boxes': nms_result['boxes'][high_conf_mask],
                'scores': nms_result['scores'][high_conf_mask],
                'labels': [nms_result['labels'][i] for i in np.where(high_conf_mask)[0]],
                'features': nms_result['features'][high_conf_mask],
                'class_probs': nms_result['class_probs'][high_conf_mask] if nms_result['class_probs'] is not None else None
            }
            self._update_cache(high_conf_result)
        
        return self._to_detection_result(nms_result)
    
    def _detect_with_features(self, image, target_classes, threshold):
        """Get ALL queries with features and scores from detector"""
        result = self.detector.detect_with_features(image, target_classes, threshold)
        
        # Convert to dict format
        result_dict = {
            'boxes': np.array(result.boxes),
            'scores': np.array(result.scores),
            'labels': result.labels,
            'features': result.features,
            'class_probs': result.class_probs
        }
        
        return result_dict
    
    def _bayesian_inference_all_queries(self, all_queries_result):
        """
        Apply Bayesian inference to ALL queries (Algorithm 1 Lines 4-8).
        
        For each of the ~900 queries:
        1. Compute P(U|x_ij) using cache
        2. Compute p_cache 
        3. Fuse p_init with p_cache
        """
        features = all_queries_result['features']  # (900, d)
        boxes = all_queries_result['boxes']  # (900, 4)
        init_probs = all_queries_result['class_probs']  # (900, K)
        
        N = len(features)
        final_probs = np.zeros_like(init_probs)
        
        # Process each query
        for i in range(N):
            # Compute similarity to cache
            P_U_given_x = self._compute_posterior_over_cache(features[i], boxes[i])
            
            # Cache-based prediction
            cache_probs_i = P_U_given_x @ self.cache.V_cache.T
            
            # Fuse initial and cache predictions
            final_probs[i] = self._uncertainty_fusion_single(init_probs[i], cache_probs_i)
        
        # Compute final scores and labels
        final_scores = np.max(final_probs, axis=1)
        final_label_indices = np.argmax(final_probs, axis=1)
        final_labels = [self.class_names[idx] for idx in final_label_indices]
        
        return {
            'boxes': boxes,
            'scores': final_scores,
            'labels': final_labels,
            'features': features,
            'class_probs': final_probs
        }
    
    def _uncertainty_fusion_single(self, init_prob, cache_prob):
        """
        Entropy-based uncertainty-guided fusion for a single query (Eq. 14)
        """
        # Compute entropy for each prediction
        E_init = self._entropy(init_prob)
        E_cache = self._entropy(cache_prob)
        
        # Weights based on confidence (lower entropy = higher weight)
        w_init = np.exp(-E_init)
        w_cache = np.exp(-E_cache)
        
        # Weighted fusion
        fused = (w_init * init_prob + w_cache * cache_prob) / (w_init + w_cache)
        
        return fused
    
    def _compute_posterior_over_cache(self, feature, box):
        """
        Compute P(U|x_ij) = Softmax(likelihood) (Eq. 11, 12)
        """
        # Feature similarity S_F (Eq. 7)
        S_F = self._compute_feature_similarity(feature)
        
        # Scale similarity S_B (Eq. 8)
        S_B = self._compute_scale_similarity(box)
        
        # Combined likelihood (Eq. 10 or 9)
        if self.cache.B_cache is not None:
            # Detection: use both feature and scale
            likelihood = self.ws * S_B + (1 - self.ws) * S_F
        else:
            # Recognition: feature only
            likelihood = S_F
        
        # Softmax to get posterior
        P_U_given_x = self._softmax(likelihood)
        
        return P_U_given_x
    
    def _compute_feature_similarity(self, feature):
        """Cosine similarity between current feature and cached features (Eq. 7)"""
        if self.cache.F_cache is None:
            return np.array([])
        
        # Normalize feature
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        
        # Compute cosine similarity with all cached features
        # F_cache is (d, M), feature_norm is (d,)
        # Correct: feature_norm @ F_cache = (d,) @ (d, M) = (M,)
        similarities = feature_norm @ self.cache.F_cache  # (M,)
        
        return similarities
    
    def _compute_scale_similarity(self, box):
        """L2-based scale similarity (Eq. 8)"""
        if self.cache.B_cache is None:
            return np.array([])
        
        # Extract width and height [w, h] from box [x, y, w, h]
        current_scale = box[2:4]
        
        # Compute L2 distance to each cached scale
        distances = np.linalg.norm(self.cache.B_cache - current_scale.reshape(2, 1), axis=0)
        
        # Convert to similarity (Eq. 8)
        similarities = 1 - distances / np.sqrt(2)
        
        return similarities
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _entropy(self, probs):
        """Shannon entropy: -sum(p * log(p))"""
        probs = probs + 1e-8  # Avoid log(0)
        return -np.sum(probs * np.log(probs))
    
    def _update_cache(self, final_result):
        """
        Stage 3: Update cache with high-confidence predictions (Eq. 15-17)
        """
        features = final_result['features']
        boxes = final_result['boxes']
        scores = final_result['scores']
        
        # Get class_probs, or create from scores and labels if None
        if final_result.get('class_probs') is not None:
            probs = final_result['class_probs']
        else:
            # Convert scores and labels to class_probs
            labels = final_result['labels']
            probs = np.zeros((len(scores), self.num_classes))
            for i, label in enumerate(labels):
                if label in self.class_names:
                    class_idx = self.class_names.index(label)
                    probs[i, class_idx] = scores[i]
            # Normalize
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            probs = probs / row_sums
        
        # Filter high-confidence detections (τ1 threshold)
        high_conf_mask = scores >= self.tau1
        
        if not np.any(high_conf_mask):
            return  # No high-confidence detections to add
        
        for i in np.where(high_conf_mask)[0]:
            feature = features[i]
            box = boxes[i]
            prob = probs[i]
            
            if self.cache.is_empty():
                # Initialize cache with first entry
                self._create_cache_entry(feature, box, prob)
            else:
                # Find best matching cache entry (Eq. 15)
                P_U_given_x = self._compute_posterior_over_cache(feature, box)
                m_star = np.argmax(P_U_given_x)
                max_similarity = P_U_given_x[m_star]
                
                if max_similarity < self.tau2:
                    # Create new cache entry (Eq. 16)
                    self._create_cache_entry(feature, box, prob)
                else:
                    # Update existing entry (Eq. 17)
                    self._update_cache_entry(m_star, feature, box, prob)
    
    def _create_cache_entry(self, feature, box, prob):
        """Add new entry to cache (Eq. 16)"""
        # Normalize feature
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        scale = box[2:4]  # [w, h]
        
        if self.cache.is_empty():
            self.cache.F_cache = feature_norm.reshape(-1, 1)  # (d, 1)
            self.cache.B_cache = scale.reshape(-1, 1)  # (2, 1)
            self.cache.V_cache = prob.reshape(-1, 1)  # (K, 1)
            self.cache.C_cache = np.array([1])
            self.cache.M = 1
        else:
            self.cache.F_cache = np.hstack([self.cache.F_cache, feature_norm.reshape(-1, 1)])
            self.cache.B_cache = np.hstack([self.cache.B_cache, scale.reshape(-1, 1)])
            self.cache.V_cache = np.hstack([self.cache.V_cache, prob.reshape(-1, 1)])
            self.cache.C_cache = np.append(self.cache.C_cache, 1)
            self.cache.M += 1
    
    def _update_cache_entry(self, m_star, feature, box, prob):
        """Update existing cache entry with count-based averaging (Eq. 17)"""
        c = self.cache.C_cache[m_star]
        
        # Update feature embedding
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        self.cache.F_cache[:, m_star] = (c * self.cache.F_cache[:, m_star] + feature_norm) / (c + 1)
        
        # Update spatial scale
        scale = box[2:4]
        self.cache.B_cache[:, m_star] = (c * self.cache.B_cache[:, m_star] + scale) / (c + 1)
        
        # Update prior
        self.cache.V_cache[:, m_star] = (c * self.cache.V_cache[:, m_star] + prob) / (c + 1)
        
        # Increment count
        self.cache.C_cache[m_star] += 1
    
    def _apply_nms(self, result, iou_threshold=0.3):
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            result: Dict with 'boxes', 'scores', 'labels', 'features', 'class_probs'
            iou_threshold: IoU threshold for NMS (default 0.3)
        
        Returns:
            Filtered result dict with NMS applied
        """
        if len(result['boxes']) == 0:
            return result
        
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        features = result['features']
        class_probs = result.get('class_probs', None)
        
        # Compute IoU between all pairs
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        while len(sorted_indices) > 0:
            # Keep the highest scoring box
            current = sorted_indices[0]
            keep_indices.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Remove boxes with high IoU with current box
            remaining_indices = []
            for idx in sorted_indices[1:]:
                iou = compute_iou(boxes[current], boxes[idx])
                if iou < iou_threshold:
                    remaining_indices.append(idx)
            
            sorted_indices = np.array(remaining_indices)
        
        # Filter results
        return {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'labels': [labels[i] for i in keep_indices],
            'features': features[keep_indices],
            'class_probs': class_probs[keep_indices] if class_probs is not None else None
        }
    
    def _to_detection_result(self, result):
        """Convert internal format back to DetectionResult"""
        from vlm_detector_system_new import DetectionResult
        return DetectionResult(
            boxes=result['boxes'].tolist() if len(result['boxes']) > 0 else [],
            scores=result['scores'].tolist() if len(result['scores']) > 0 else [],
            labels=result['labels'],
            image_path="",
            model_path=self.detector.model_path
        )