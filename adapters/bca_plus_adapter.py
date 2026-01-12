import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from PIL import Image
from .base_adapter import BaseAdapter

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

class BCAPlusAdapter(BaseAdapter):
    """Main BCA+ Test-Time Adaptation"""
    def __init__(self, detector, config: dict):
        super().__init__(detector, config)
        
        # Get adaptation parameters from config
        params = config['adaptation']['params']
        self.tau1 = params.get('tau1') or 0.7
        self.tau2 = params.get('tau2') or 0.8
        self.tau2_init = params.get('tau2_init') or 0.5
        self.max_cache_size = params.get('max_cache_size') or 50
        self.ws = params.get('ws') or 0.2
        self.logit_temperature = params.get('logit_temperature') or 1.0
        self.alpha = params.get('alpha') or 0.7
        # Get detector iou_threshold from config
        self.iou_threshold = config['detector'].get('iou_threshold') or 0.3
        
        # Get number of classes
        self.num_classes = len(config['detector']['target_classes'])
        
        # Initialize cache
        self.cache = BCAPlusCache(num_classes=self.num_classes)
        self.class_names = None
    
    
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
        
        # Store image size for box normalization (needed for scale similarity)
        self.image_size = image.size  # (width, height)
        
        # Stage 1: Get ALL 900 queries with proper scores
        all_queries_result = self._detect_with_features(image, target_classes, threshold, self.alpha)
        
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
            'class_probs': adapted_result['class_probs'][mask] if 'class_probs' in adapted_result else None,
            'P_U_given_x': adapted_result['P_U_given_x'][mask] if adapted_result.get('P_U_given_x') is not None else None,
            'raw_sims': adapted_result['raw_sims'][mask] if adapted_result.get('raw_sims') is not None else None
        }

        
        
        # Stage 4: Apply NMS to remove overlapping detections
        nms_result = self._apply_nms(filtered_result, iou_threshold=self.iou_threshold)
        
        # Stage 5: Update cache with high-confidence detections (from NMS result)
        high_conf_mask = np.array(nms_result['scores']) >= self.tau1
        if np.any(high_conf_mask):
            high_conf_result = {
                'boxes': nms_result['boxes'][high_conf_mask],
                'scores': nms_result['scores'][high_conf_mask],
                'labels': [nms_result['labels'][i] for i in np.where(high_conf_mask)[0]],
                'features': nms_result['features'][high_conf_mask],
                'class_probs': nms_result['class_probs'][high_conf_mask] if nms_result['class_probs'] is not None else None,
                'P_U_given_x': nms_result['P_U_given_x'][high_conf_mask] if nms_result.get('P_U_given_x') is not None else None,
                'raw_sims': nms_result['raw_sims'][high_conf_mask] if nms_result.get('raw_sims') is not None else None
            }
            self._update_cache(high_conf_result)

        
        return self._to_detection_result(nms_result)
    
    def _detect_with_features(self, image, target_classes, threshold, alpha):
        """Get ALL queries with features and scores from detector"""
        result = self.detector.detect_with_features(image, target_classes, threshold, alpha)
        
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
        M = self.cache.M  # Cache size at start of image (frozen)
        final_probs = np.zeros_like(init_probs)
        
        # CRITICAL: Store P(U|x) AND raw feature similarities for cache update
        P_U_given_x_all = np.zeros((N, M)) if M > 0 else None
        raw_sims_all = np.zeros((N, M)) if M > 0 else None  # NEW: For tau2 comparison
        
        # Process each query
        for i in range(N):
            # Compute similarity to cache (frozen at start of image)
            P_U_given_x = self._compute_posterior_over_cache(features[i], boxes[i])
            
            # Store for cache update (paper: use pre-computed P(U|x))
            if P_U_given_x_all is not None:
                P_U_given_x_all[i] = P_U_given_x
                # CRITICAL: Also store RAW feature and scale similarity for tau2 comparison
                raw_sims_all[i] = self.ws * self._compute_scale_similarity(boxes[i]) + (1 - self.ws) * self._compute_feature_similarity(features[i])
            
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
            'class_probs': final_probs,
            'P_U_given_x': P_U_given_x_all,  # Pre-computed matching distribution
            'raw_sims': raw_sims_all  # NEW: For tau2 comparison
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

        if np.isinf(w_init) or np.isinf(w_cache):
            print(f"w_init: {w_init}, w_cache: {w_cache}")
            print(f"E_init: {E_init}, E_cache: {E_cache}")
            print(f"init_prob: {init_prob}, cache_prob: {cache_prob}")
            exit()

        
        
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
        if len(likelihood) == 1:
            return likelihood
        likelihood = likelihood * self.logit_temperature
        P_U_given_x = self._softmax(likelihood)

        
        
        return P_U_given_x
    
    def _compute_feature_similarity(self, feature):
        """Cosine similarity between current feature and cached features (Eq. 7)"""
        if self.cache.F_cache is None:
            return np.array([])

        # In _detect_with_features or wherever you get features:
        
        # Normalize feature handle the inf elements
        feature_norm = feature.copy()
        feature_norm = feature_norm / (np.linalg.norm(feature_norm) + 1e-8)
        
        # Compute cosine similarity with all cached features
        # F_cache is (d, M), feature_norm is (d,)
        # Correct: feature_norm @ F_cache = (d,) @ (d, M) = (M,)
        similarities = feature_norm @ self.cache.F_cache  # (M,)

        return similarities
    
    def _compute_scale_similarity(self, box):
        """
        L2-based scale similarity (Eq. 8)
        
        Paper: S_B = 1 - ||b[2:] - b_cache|| / sqrt(2)
        where w, h are normalized to [0, 1]
        """
        if self.cache.B_cache is None:
            return np.array([])
        
        

        current_scale = box[2]-box[0], box[3]-box[1]
        
        # CRITICAL FIX: Normalize to [0, 1] as per paper
        # "ranges of w and h are constrained to [0, 1]"
        image_w, image_h = self.image_size
        normalized_scale = np.array([
            current_scale[0] / image_w,  # w normalized
            current_scale[1] / image_h   # h normalized
        ])
        
        # Compute L2 distance to each cached scale (already normalized)
        # B_cache shape: (2, M), normalized_scale shape: (2,)
        distances = np.linalg.norm(self.cache.B_cache - normalized_scale.reshape(2, 1), axis=0)
        
        # Convert to similarity (Eq. 8) - normalize by sqrt(2)
        # "maximum difference of sqrt(2) under perfect misalignment"
        similarities = 1 - distances / np.sqrt(2)
        
        return similarities
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        if x.shape[0] == 1:
            return x
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _entropy(self, probs):
        """Shannon entropy: -sum(p * log(p))"""
        probs = probs + 1e-8  # Avoid log(0)
        return -np.sum(probs * np.log(probs))
    
    def _update_cache(self, final_result):
        """
        Stage 3: Update cache using PRE-COMPUTED matching distributions (Paper-aligned)
        
        Key changes from old implementation:
        1. Use P(U|x) computed during prediction (frozen cache snapshot)
        2. Batch-init when cache is empty (clustering by class)
        3. No recomputation of P(U|x) during update
        """
        features = final_result['features']
        boxes = final_result['boxes']
        scores = final_result['scores']
        P_U_given_x_all = final_result.get('P_U_given_x', None)  # Pre-computed posteriors
        raw_sims_all = final_result.get('raw_sims', None)  # Raw feature similarities
        
        # Get class_probs
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
        
        # Paper: Filter by max_k p_final >= tau1 (already done in adapt_and_detect)
        # All proposals here are high-confidence
        
        if len(features) == 0:
            return
        
        # CASE 1: Cache is EMPTY - Batch-init with clustering
        if self.cache.is_empty():
            print(f"***********************BATCH INIT***********************")
            print(f"Initializing cache from {len(features)} high-confidence proposals")
            self._batch_init_cache(features, boxes, probs)
            print(f"Cache initialized with M={self.cache.M} entries")
            print(f"***********************BATCH INIT DONE***********************")
            return
        
        # CASE 2: Cache EXISTS - Use pre-computed P(U|x) for updates
        print(f"***********************CACHE UPDATE***********************")
        print(f"Updating cache (M={self.cache.M}) with {len(features)} proposals")
        
        if P_U_given_x_all is None:
            print(f"WARNING: No pre-computed P(U|x) available, skipping cache update")
            return
        
        # # DIAGNOSTIC: Check proposals vs cache similarities
        # print(f"\n{'='*60}")
        # print(f"CACHE UPDATE - PROPOSALS VS CACHE CROSS-SIMILARITY")
        # print(f"{'='*60}")
        # # Normalize proposals
        # proposal_norms = np.linalg.norm(features, axis=1, keepdims=True)
        # proposals_norm = features / (proposal_norms + 1e-8)
        # # Cache is already normalized
        # cross_sims = proposals_norm @ self.cache.F_cache  # (N_proposals, M_cache)
        # print(f"Cross-similarity matrix shape: {cross_sims.shape}")
        # print(f"Cross-similarities: min={cross_sims.min():.6f}, max={cross_sims.max():.6f}, mean={cross_sims.mean():.6f}")
        # print(f"  >0.95: {(cross_sims > 0.95).sum()} / {cross_sims.size} ({100*(cross_sims > 0.95).sum()/cross_sims.size:.1f}%)")
        # print(f"  >0.90: {(cross_sims > 0.90).sum()} / {cross_sims.size} ({100*(cross_sims > 0.90).sum()/cross_sims.size:.1f}%)")
        # print(f"  >0.80: {(cross_sims > 0.80).sum()} / {cross_sims.size} ({100*(cross_sims > 0.80).sum()/cross_sims.size:.1f}%)")
        # print(f"{'='*60}\n")
        
        # # DIAGNOSTIC: Check cache distinctiveness
        # print(f"\n=== CACHE ANALYSIS ===")
        # if self.cache.M > 1:
        #     # Compute pairwise similarities between cache entries
        #     cache_sims = self.cache.F_cache.T @ self.cache.F_cache  # (M, M)
        #     # Get upper triangle (excluding diagonal)
        #     triu_indices = np.triu_indices(self.cache.M, k=1)
        #     pairwise_sims = cache_sims[triu_indices]
            
        #     print(f"Overall inter-cache similarities (N={len(pairwise_sims)} pairs):")
        #     print(f"  Min:  {pairwise_sims.min():.3f}")
        #     print(f"  Max:  {pairwise_sims.max():.3f}")
        #     print(f"  Mean: {pairwise_sims.mean():.3f}")
        #     print(f"  >0.9: {(pairwise_sims > 0.9).sum()}/{len(pairwise_sims)} ({100*(pairwise_sims > 0.9).mean():.1f}%)")
        #     print(f"  >0.8: {(pairwise_sims > 0.8).sum()}/{len(pairwise_sims)} ({100*(pairwise_sims > 0.8).mean():.1f}%)")
            
        #     # CRITICAL: Analyze PER-CLASS similarities (this reveals the hidden problem!)
        #     print(f"\nPer-class analysis (may reveal hidden within-class similarity):")
        #     # Get predicted class for each cache entry
        #     cache_classes = np.argmax(self.cache.V_cache.T, axis=1)  # (M,)
            
        #     for class_idx in range(self.num_classes):
        #         class_mask = cache_classes == class_idx
        #         class_count = class_mask.sum()
                
        #         if class_count > 1:
        #             # Get indices of this class
        #             class_indices = np.where(class_mask)[0]
                    
        #             # Compute within-class similarities
        #             within_class_sims = []
        #             for i in range(len(class_indices)):
        #                 for j in range(i+1, len(class_indices)):
        #                     sim = cache_sims[class_indices[i], class_indices[j]]
        #                     within_class_sims.append(sim)
                    
        #             within_class_sims = np.array(within_class_sims)
        #             print(f"  {self.class_names[class_idx]} (n={class_count}):")
        #             print(f"    Within-class mean: {within_class_sims.mean():.3f}")
        #             print(f"    Within-class max:  {within_class_sims.max():.3f}")
        #             print(f"    Within-class >0.9: {(within_class_sims > 0.9).sum()}/{len(within_class_sims)} ({100*(within_class_sims > 0.9).mean():.1f}%)")
                    
        #             if within_class_sims.mean() > 0.90:
        #                 print(f"    ⚠️  CRITICAL: {self.class_names[class_idx]} entries TOO SIMILAR!")
        #                 print(f"        This causes uniform posteriors for {self.class_names[class_idx]} detections!")
            
        #     if pairwise_sims.mean() > 0.85:
        #         print(f"\n  ⚠️  WARNING: Cache entries are VERY similar (mean={pairwise_sims.mean():.3f})")
        #         print(f"      This causes uniform posteriors → always creates new entries!")
        # print(f"===================\n")
        
        # Process each proposal using PRE-COMPUTED P(U|x)
        for i in range(len(features)):
            feature = features[i]
            box = boxes[i]
            prob = probs[i]
            P_U_given_x = P_U_given_x_all[i]  # Posterior (paper-aligned)
            raw_sims = raw_sims_all[i] if raw_sims_all is not None else None
            
            # Find best matching cache entry using POSTERIOR (Eq. 15)
            m_star = np.argmax(P_U_given_x)
            s_star = P_U_given_x[m_star]  # Posterior probability (paper-aligned)
            raw_sim = raw_sims[m_star] if raw_sims is not None else None
            
            # Show both posterior and raw similarity for debugging
            raw_sim_info = f", raw_sim={raw_sim:.3f}" if raw_sim is not None else ""
            # print(f"Proposal {i}: m_star={m_star}, P(U|x)={s_star:.4f}{raw_sim_info}, tau2={self.tau2}, prob={prob}, P(U|x)={P_U_given_x}, raw_sims={raw_sims}, raw_sim_mean_sub={raw_sims - np.max(raw_sims)}, softmax_raw_sims={np.exp(raw_sims - np.max(raw_sims)) / np.sum(np.exp(raw_sims - np.max(raw_sims)))}")
            
            # Cache update decision based on posterior
            # Paper: Compare P(U|x) to tau2
            should_create_new = False
            reason = ""
            
            if s_star >= self.tau2:
                # Strong posterior match → UPDATE
                should_create_new = False
                reason = f"P(U|x)={s_star:.3f} >= tau2={self.tau2}"
            else:
                # Weak posterior → CREATE NEW
                should_create_new = True
                reason = f"P(U|x)={s_star:.3f} < tau2={self.tau2}"
            
            if should_create_new:
                # Check cache size limit (safety net against infinite growth)
                MAX_CACHE_SIZE = self.max_cache_size  # Reasonable limit for 6 classes
                
                if self.cache.M >= MAX_CACHE_SIZE:
                    # print(f"  -> CACHE FULL (M={self.cache.M}/{MAX_CACHE_SIZE}), UPDATING entry {m_star} instead")
                    self._update_cache_entry(m_star, feature, box, prob)
                else:
                    # print(f"  -> Creating NEW cache entry ({reason})")
                    self._create_cache_entry(feature, box, prob)
            else:
                # print(f"  -> UPDATING cache entry {m_star} ({reason})")
                self._update_cache_entry(m_star, feature, box, prob)
        
        print(f"Cache update done: M={self.cache.M}")
        print(f"***********************CACHE UPDATE DONE***********************")
    
    def _batch_init_cache(self, features, boxes, probs):
        """
        Batch initialize cache from first frame with clustering (Paper-aligned)
        
        Strategy:
        1. Group proposals by predicted class
        2. Within each class, cluster based on feature+scale similarity
        3. Create one cache entry per cluster
        
        This avoids creating near-duplicate entries from similar detections.
        
        CRITICAL: Use HIGHER threshold for clustering than tau2 to avoid
        over-aggressive merging that creates generic centroids.
        """
        # CRITICAL: Use stricter threshold for batch-init clustering
        # This prevents over-averaging many proposals into generic clusters
        TAU2_INIT = self.tau2_init  # Higher than self.tau2 (typically 0.5)
        MAX_CLUSTER_SIZE = 50  # Limit cluster size to prevent over-averaging
        
        print(f"Batch-init using TAU2_INIT={TAU2_INIT}, MAX_CLUSTER_SIZE={MAX_CLUSTER_SIZE}")
        
        # Get predicted class for each proposal
        print(f"probs shape: {probs.shape}")
        predicted_classes = np.argmax(probs, axis=1)
        
        # Process each class separately
        for class_idx in range(self.num_classes):
            # Get proposals for this class
            class_mask = predicted_classes == class_idx
            if not np.any(class_mask):
                continue
            
            class_features = features[class_mask]
            class_boxes = boxes[class_mask]
            class_probs = probs[class_mask]
            class_scores = np.max(class_probs, axis=1)
            
            # Sort by confidence (descending) for greedy clustering
            sorted_indices = np.argsort(class_scores)[::-1]
            
            # Greedy clustering
            clusters = []  # Each cluster: {'indices': [], 'centroid_feature': ..., 'centroid_scale': ...}
            
            for idx in sorted_indices:
                feature = class_features[idx]
                box = class_boxes[idx]
                prob = class_probs[idx]
                
                # Normalize feature handle the inf elements
                feature_norm = feature.copy()
                feature_norm = feature_norm / (np.linalg.norm(feature_norm) + 1e-8)
                
                # Get scale [w, h] normalized
                image_w, image_h = self.image_size
                w = box[2] - box[0]
                h = box[3] - box[1]
                scale = np.array([w / image_w, h / image_h])
                
                # Find best matching cluster
                best_cluster_idx = -1
                best_sim = -1.0
                
                for cluster_idx, cluster in enumerate(clusters):
                    # Feature similarity
                    feat_sim = feature_norm @ cluster['centroid_feature']
                    # feat_sim = feat_sim * self.logit_temperature
                    
                    # Scale similarity
                    scale_diff = np.linalg.norm(scale - cluster['centroid_scale'])
                    scale_sim = 1 - scale_diff / np.sqrt(2)
                    
                    # Combined similarity (simple average, can use ws if needed)
                    combined_sim = (1 - self.ws) * feat_sim + self.ws * scale_sim
                    
                    if combined_sim > best_sim:
                        best_sim = combined_sim
                        best_cluster_idx = cluster_idx
                
                # Decide: add to existing cluster or create new one
                # CRITICAL: Check cluster size BEFORE adding to enforce hard limit
                can_add_to_cluster = (best_sim >= TAU2_INIT and 
                                     best_cluster_idx >= 0 and 
                                     len(clusters[best_cluster_idx]['indices']) < MAX_CLUSTER_SIZE)
                
                if can_add_to_cluster:
                    # Add to existing cluster
                    cluster = clusters[best_cluster_idx]
                    cluster['indices'].append(idx)
                    
                    # Update centroid (running mean)
                    n = len(cluster['indices'])
                    cluster['centroid_feature'] = ((n-1) * cluster['centroid_feature'] + feature_norm) / n
                    cluster['centroid_feature'] /= (np.linalg.norm(cluster['centroid_feature']) + 1e-8)  # Renormalize
                    cluster['centroid_scale'] = ((n-1) * cluster['centroid_scale'] + scale) / n
                    cluster['total_prob'] += prob
                    
                    # print(f"    Proposal {idx}: Added to cluster {best_cluster_idx} (size={n}, feat_sim={feat_sim:.3f}, scale_sim={scale_sim:.3f}, combined_sim={best_sim:.3f}, prob={prob})")
                else:
                    # Create new cluster
                    clusters.append({
                        'indices': [idx],
                        'centroid_feature': feature_norm.copy(),
                        'centroid_scale': scale.copy(),
                        'total_prob': prob.copy()
                    })
                    
                    if best_cluster_idx >= 0:
                        reason = f"cluster full ({len(clusters[best_cluster_idx]['indices'])}/{MAX_CLUSTER_SIZE})" if best_sim >= TAU2_INIT else f"low sim ({best_sim:.3f} < {TAU2_INIT})"
                        # print(f"    Proposal {idx}: New cluster {len(clusters)-1} ({reason}), feat_sim={feat_sim:.3f}, scale_sim={scale_sim:.3f}, combined_sim={best_sim:.3f}, prob={prob}")
                    # else:
                        # print(f"    Proposal {idx}: First cluster, prob={prob}")
            
            # Create cache entries from clusters
            for cluster in clusters:
                n = len(cluster['indices'])
                mean_prob = cluster['total_prob'] / n
                
                # Create cache entry
                self._create_cache_entry(
                    feature=cluster['centroid_feature'] * np.linalg.norm(cluster['centroid_feature']),  # Denormalize for _create_cache_entry
                    box=np.array([0, 0, cluster['centroid_scale'][0] * self.image_size[0], cluster['centroid_scale'][1] * self.image_size[1]]),  # Dummy box with correct w,h
                    prob=mean_prob
                )
                
                # Set count to cluster size
                self.cache.C_cache[-1] = n
                
                # print(f"  Class {self.class_names[class_idx]}: Created cluster with {n} proposals (avg_prob={np.max(mean_prob):.3f})")
        
        # print(f"\nTotal clusters created: {self.cache.M}")
        # if self.cache.M > 0:
            # print(f"Cluster size distribution: min={min([c for c in self.cache.C_cache])}, max={max([c for c in self.cache.C_cache])}, mean={np.mean(self.cache.C_cache):.1f}")
        
        
        # DIAGNOSTIC: Check if created clusters are actually distinct
        # if self.cache.M > 1:
        #     print(f"\n=== CLUSTER DISTINCTIVENESS ANALYSIS ===")
        #     # Compute pairwise similarities between cluster centroids
        #     cluster_sims = self.cache.F_cache.T @ self.cache.F_cache  # (M, M)
        #     # Get upper triangle (excluding diagonal)
        #     triu_indices = np.triu_indices(self.cache.M, k=1)
        #     pairwise_sims = cluster_sims[triu_indices]
            
        #     print(f"Overall inter-cluster similarities (N={len(pairwise_sims)} pairs):")
        #     print(f"  Min:  {pairwise_sims.min():.3f}")
        #     print(f"  Max:  {pairwise_sims.max():.3f}")
        #     print(f"  Mean: {pairwise_sims.mean():.3f}")
        #     print(f"  >0.9: {(pairwise_sims > 0.9).sum()}/{len(pairwise_sims)} ({100*(pairwise_sims > 0.9).mean():.1f}%)")
        #     print(f"  >0.8: {(pairwise_sims > 0.8).sum()}/{len(pairwise_sims)} ({100*(pairwise_sims > 0.8).mean():.1f}%)")
        #     print(f"  >0.7: {(pairwise_sims > 0.7).sum()}/{len(pairwise_sims)} ({100*(pairwise_sims > 0.7).mean():.1f}%)")
            
        #     # CRITICAL: Analyze PER-CLASS similarities
        #     print(f"\nPer-class analysis (reveals within-class similarity):")
        #     cluster_classes = np.argmax(self.cache.V_cache.T, axis=1)  # (M,)
            
        #     for class_idx in range(self.num_classes):
        #         class_mask = cluster_classes == class_idx
        #         class_count = class_mask.sum()
                
        #         if class_count > 1:
        #             class_indices = np.where(class_mask)[0]
                    
        #             # Compute within-class similarities
        #             within_class_sims = []
        #             for i in range(len(class_indices)):
        #                 for j in range(i+1, len(class_indices)):
        #                     sim = cluster_sims[class_indices[i], class_indices[j]]
        #                     within_class_sims.append(sim)
                    
        #             within_class_sims = np.array(within_class_sims)
        #             print(f"  {self.class_names[class_idx]} (n={class_count}):")
        #             print(f"    Within-class mean: {within_class_sims.mean():.3f}")
        #             print(f"    Within-class >0.9: {(within_class_sims > 0.9).sum()}/{len(within_class_sims)} ({100*(within_class_sims > 0.9).mean():.1f}%)")
                    
        #             if within_class_sims.mean() > 0.90:
        #                 print(f"    ⚠️  WARNING: {self.class_names[class_idx]} clusters TOO SIMILAR!")
            
        #     if pairwise_sims.mean() > 0.85:
        #         print(f"\n  ⚠️  PROBLEM IDENTIFIED: Clusters are TOO SIMILAR (mean={pairwise_sims.mean():.3f})")
        #         print(f"      Cause: Same scene → similar backgrounds/features")
        #         print(f"      Effect: Uniform posteriors → always P(U|x) < tau2 → infinite growth")
        #         print(f"      Solution: Need LOWER tau2 or HIGHER TAU2_INIT")
        #     elif pairwise_sims.mean() > 0.70:
        #         print(f"\n  ⚠️  WARNING: Clusters are moderately similar (mean={pairwise_sims.mean():.3f})")
        #         print(f"      May cause some uniform posteriors when M grows large")
        #     else:
        #         print(f"\n  ✓ Clusters are sufficiently distinct (mean={pairwise_sims.mean():.3f})")
            
        #     print(f"========================================\n")
    
    def _create_cache_entry(self, feature, box, prob):
        """Add new entry to cache (Eq. 16)"""
        # Normalize feature handle the inf elements
        feature_norm = feature.copy()
        feature_norm = feature_norm / (np.linalg.norm(feature_norm) + 1e-8)
        # CRITICAL: Convert [x1, y1, x2, y2] to [w, h] then normalize to [0, 1]
        image_w, image_h = self.image_size
        w = box[2] - box[0]  # x2 - x1
        h = box[3] - box[1]  # y2 - y1
        scale = np.array([
            w / image_w,  # w normalized to [0, 1]
            h / image_h   # h normalized to [0, 1]
        ])
        
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
        feature_norm = feature.copy()
        feature_norm = feature_norm / (np.linalg.norm(feature_norm) + 1e-8)
        self.cache.F_cache[:, m_star] = (c * self.cache.F_cache[:, m_star] + feature_norm) / (c + 1)
        self.cache.F_cache[:, m_star] = self.cache.F_cache[:, m_star] / (np.linalg.norm(self.cache.F_cache[:, m_star]) + 1e-8)
        
        # Update spatial scale - CRITICAL: Convert [x1,y1,x2,y2] to [w,h] then normalize to [0,1]
        image_w, image_h = self.image_size
        w = box[2] - box[0]  # x2 - x1
        h = box[3] - box[1]  # y2 - y1
        scale = np.array([
            w / image_w,  # w normalized to [0, 1]
            h / image_h   # h normalized to [0, 1]
        ])
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
        # Ensure iou_threshold is not None
        if iou_threshold is None:
            iou_threshold = 0.3
        
        if len(result['boxes']) == 0:
            return result
        
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        features = result['features']
        class_probs = result.get('class_probs', None)
        P_U_given_x = result.get('P_U_given_x', None)  # Pre-computed matching distribution
        raw_sims = result.get('raw_sims', None)  # Raw feature similarities
        
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
            'class_probs': class_probs[keep_indices] if class_probs is not None else None,
            'P_U_given_x': P_U_given_x[keep_indices] if P_U_given_x is not None else None,
            'raw_sims': raw_sims[keep_indices] if raw_sims is not None else None
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

    def reset(self):
        """Reset for new video."""
        self.cache = BCAPlusCache(num_classes=self.num_classes)
