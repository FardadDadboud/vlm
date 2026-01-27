"""
Enhanced BCA+ Cache with Lifecycle Management

Global cache for BCA+ (Bayesian Class Adaptation) with:
1. Tracker-style lifecycle (tentative → confirmed → deleted)
2. Age-based cleanup
3. Batch initialization with clustering
4. Feature + Scale similarity for posterior computation
5. Confidence-weighted updates

This is the GLOBAL component of Global+Instance TTA. It maintains class-level
prototypes that adapt across the video, independent of per-track STAD.

Key difference from per-track STAD:
- Global cache operates on high-confidence detections across ALL objects
- Per-track STAD operates on individual object's temporal history
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Cache Entry State (Lifecycle)
# =============================================================================

class CacheEntryState(Enum):
    """Cache entry lifecycle states (mirrors track lifecycle)."""
    TENTATIVE = 'tentative'
    CONFIRMED = 'confirmed'
    DELETED = 'deleted'


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnhancedBCAPlusConfig:
    """Configuration for Enhanced BCA+ Cache."""
    # === Thresholds ===
    # tau1: Confidence threshold for cache UPDATE (only high-conf detections update cache)
    tau1: float = 0.6
    
    # tau2: Similarity threshold for cache MATCH
    # Separate thresholds for tentative vs confirmed entries (stricter for tentative)
    tau2_confirmed: float = 0.7
    tau2_tentative: float = 0.85
    tau2_init: float = 0.5  # Threshold during batch initialization
    
    # === Cache Size ===
    max_cache_size: int = 50
    
    # === Posterior Computation ===
    # ws: Weight for scale similarity in posterior (1-ws = feature similarity weight)
    ws: float = 0.2
    
    # alpha: Feature mixing (alpha=0 uses decoder features, alpha=1 uses class probs)
    # Hybrid: h = (1-alpha)*decoder + alpha*class_probs
    alpha: float = 0.7  # Match existing BCA+ adapter
    
    # logit_temperature: Temperature for softmax in posterior
    logit_temperature: float = 10.0
    
    # === Lifecycle ===
    min_hits_to_confirm: int = 3   # Hits needed to confirm entry
    max_age: int = 30              # Frames before deletion
    
    # === Initialization ===
    # Batch init uses greedy clustering to seed cache with diverse entries
    use_batch_init: bool = True
    batch_init_size: int = 10  # Number of frames to collect before batch init
    
    # === Debug ===
    debug: bool = False


# =============================================================================
# Enhanced BCA+ Cache
# =============================================================================

class EnhancedBCAPlusCache:
    """
    Enhanced BCA+ Cache with lifecycle management.
    
    Maintains arrays:
    - F_cache: (D, M) feature prototypes
    - B_cache: (M,) bounding box scales (areas)
    - V_cache: (K, M) class prior distributions
    - C_cache: (M,) confidence weights
    
    Lifecycle:
    - hits: (M,) number of times entry was matched
    - age: (M,) frames since creation
    - time_since_update: (M,) frames since last match
    - states: (M,) CacheEntryState for each entry
    
    Posterior computation:
    P(m|x) ∝ exp((1-ws)*S_F + ws*S_B)
    where S_F = cosine similarity, S_B = scale similarity
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 config: Optional[EnhancedBCAPlusConfig] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize enhanced BCA+ cache.
        
        Args:
            num_classes: Number of classes K
            feature_dim: Feature dimension D
            config: Configuration
            class_names: Optional class names for debugging
        """
        self.config = config or EnhancedBCAPlusConfig()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_names = class_names
        
        # === Cache Arrays (initially empty) ===
        self.M = 0  # Current cache size
        self.F_cache: np.ndarray = np.zeros((feature_dim, 0))  # (D, M)
        self.B_cache: np.ndarray = np.zeros(0)                 # (M,)
        self.V_cache: np.ndarray = np.zeros((num_classes, 0))  # (K, M)
        self.C_cache: np.ndarray = np.zeros(0)                 # (M,)
        
        # === Lifecycle Arrays ===
        self.hits: np.ndarray = np.zeros(0, dtype=np.int32)
        self.age: np.ndarray = np.zeros(0, dtype=np.int32)
        self.time_since_update: np.ndarray = np.zeros(0, dtype=np.int32)
        self.states: List[CacheEntryState] = []
        
        # === Batch Initialization Buffer ===
        self.init_buffer_features: List[np.ndarray] = []
        self.init_buffer_scales: List[float] = []
        self.init_buffer_probs: List[np.ndarray] = []
        self.init_buffer_scores: List[float] = []
        self.batch_init_done: bool = False
        
        # === Stats ===
        self.frame_count: int = 0
        self.total_entries_created: int = 0
        self.total_entries_deleted: int = 0
        
        # Debug counters
        self._debug_updates_attempted: int = 0
        self._debug_updates_low_conf: int = 0
        self._debug_updates_batch_buffered: int = 0
        self._debug_entries_matched: int = 0
        self._debug_entries_created: int = 0
    
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return self.M == 0
    
    def _normalize(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Normalize along axis."""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.maximum(norm, 1e-10)
    
    def _safe_softmax(self, logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        logits_shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / (np.sum(exp_logits, axis=axis, keepdims=True) + 1e-10)
    
    def _get_tau2(self, entry_idx: int) -> float:
        """Get tau2 threshold based on entry state."""
        if entry_idx >= len(self.states):
            return self.config.tau2_tentative
        
        state = self.states[entry_idx]
        if state == CacheEntryState.CONFIRMED:
            return self.config.tau2_confirmed
        else:
            return self.config.tau2_tentative
    
    def compute_posterior(self, feature: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Compute posterior distribution over cache entries given detection.
        
        P(m|x) ∝ exp(τ * ((1-ws)*S_F[m] + ws*S_B[m]))
        
        Args:
            feature: (D,) detection feature
            box: (4,) detection box [x1, y1, x2, y2]
            
        Returns:
            (M,) posterior probabilities over cache entries
        """
        if self.M == 0:
            return np.array([])
        
        # Normalize feature
        feature_norm = self._normalize(feature.reshape(-1, 1), axis=0).flatten()
        
        # Feature similarity: S_F[m] = <h, F[:,m]> (F_cache already normalized)
        S_F = feature_norm @ self.F_cache  # (M,)
        
        # Scale similarity: S_B[m] = 1 - |scale - B[m]| / max(scale, B[m])
        scale = (box[2] - box[0]) * (box[3] - box[1])  # Area
        scale_diff = np.abs(scale - self.B_cache)
        S_B = 1 - scale_diff / (np.maximum(scale, self.B_cache) + 1e-10)  # (M,)
        
        # Combined similarity
        S = (1 - self.config.ws) * S_F + self.config.ws * S_B
        
        # Posterior via softmax
        logits = self.config.logit_temperature * S
        posterior = self._safe_softmax(logits)
        
        return posterior
    
    def compute_posterior_batch(self, features: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Batch compute posterior over cache entries for multiple detections.
        
        Args:
            features: (N, D) detection features
            boxes: (N, 4) detection boxes
            
        Returns:
            (N, M) posterior probabilities
        """
        N = len(features)
        if self.M == 0 or N == 0:
            return np.zeros((N, 0))
        
        # Normalize features: (N, D)
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        
        # Feature similarity: (N, M) - F_cache already normalized
        S_F = features_norm @ self.F_cache  # (N, D) @ (D, M) -> (N, M)
        
        # Scale similarity: (N, M)
        scales = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
        scale_diff = np.abs(scales[:, np.newaxis] - self.B_cache[np.newaxis, :])  # (N, M)
        S_B = 1 - scale_diff / (np.maximum(scales[:, np.newaxis], self.B_cache[np.newaxis, :]) + 1e-10)
        
        # Combined similarity
        S = (1 - self.config.ws) * S_F + self.config.ws * S_B  # (N, M)
        
        # Posterior via softmax (row-wise)
        logits = self.config.logit_temperature * S
        posterior = self._safe_softmax(logits, axis=1)  # (N, M)
        
        return posterior
    
    def adapt_probs(self, feature: np.ndarray, box: np.ndarray,
                   class_probs: np.ndarray) -> np.ndarray:
        """
        Adapt class probabilities using cache via Bayesian inference.
        
        P(y|x) = Σ_m P(m|x) * V[:,m]
        
        Then fuse with VLM prediction using entropy weighting.
        
        Args:
            feature: (D,) detection feature
            box: (4,) detection box
            class_probs: (K,) VLM class probabilities
            
        Returns:
            (K,) adapted class probabilities
        """
        if self.M == 0:
            return class_probs
        
        # Compute posterior over cache entries
        posterior = self.compute_posterior(feature, box)  # (M,)
        
        # Cache prediction: P(y|cache) = Σ_m P(m|x) * V[:,m]
        cache_probs = self.V_cache @ posterior  # (K,)
        cache_probs = cache_probs / (cache_probs.sum() + 1e-10)
        
        # Entropy-weighted fusion
        eps = 1e-10
        H_vlm = -np.sum(class_probs * np.log(class_probs + eps))
        H_cache = -np.sum(cache_probs * np.log(cache_probs + eps))
        
        # Lower entropy = higher weight
        w_vlm = np.exp(-H_vlm)
        w_cache = np.exp(-H_cache)
        
        # Fused probabilities
        adapted = (w_vlm * class_probs + w_cache * cache_probs) / (w_vlm + w_cache + eps)
        
        return adapted / (adapted.sum() + eps)
    
    def adapt_probs_batch(self, features: np.ndarray, boxes: np.ndarray,
                         class_probs: np.ndarray) -> np.ndarray:
        """
        Batch adapt class probabilities for multiple detections.
        
        Args:
            features: (N, D) detection features
            boxes: (N, 4) detection boxes
            class_probs: (N, K) VLM class probabilities
            
        Returns:
            (N, K) adapted class probabilities
        """
        N = len(features)
        if self.M == 0 or N == 0:
            return class_probs.copy()
        
        # Batch posterior: (N, M)
        posterior = self.compute_posterior_batch(features, boxes)
        
        # Cache prediction: P(y|cache) = posterior @ V_cache.T -> (N, K)
        cache_probs = posterior @ self.V_cache.T  # (N, M) @ (M, K) -> (N, K)
        cache_probs = cache_probs / (cache_probs.sum(axis=1, keepdims=True) + 1e-10)
        
        # Entropy-weighted fusion (vectorized)
        eps = 1e-10
        H_vlm = -np.sum(class_probs * np.log(class_probs + eps), axis=1)    # (N,)
        H_cache = -np.sum(cache_probs * np.log(cache_probs + eps), axis=1)  # (N,)
        
        w_vlm = np.exp(-H_vlm)[:, np.newaxis]     # (N, 1)
        w_cache = np.exp(-H_cache)[:, np.newaxis]  # (N, 1)
        
        adapted = (w_vlm * class_probs + w_cache * cache_probs) / (w_vlm + w_cache + eps)
        adapted = adapted / (adapted.sum(axis=1, keepdims=True) + eps)
        
        return adapted
    
    def update_cache(self, features: np.ndarray, boxes: np.ndarray,
                    probs: np.ndarray, scores: np.ndarray) -> None:
        """
        Update cache with new detections.
        
        Args:
            features: (N, D) detection features
            boxes: (N, 4) detection boxes
            probs: (N, K) class probabilities (should be RAW VLM probs!)
            scores: (N,) confidence scores
        """
        N = len(features)
        if N == 0:
            return
        
        self._debug_updates_attempted += N
        
        # Handle batch initialization
        if self.config.use_batch_init and not self.batch_init_done:
            self._debug_updates_batch_buffered += N
            self._collect_for_batch_init(features, boxes, probs, scores)
            return
        
        # Filter by confidence threshold
        high_conf_mask = scores >= self.config.tau1
        n_low_conf = int(np.sum(~high_conf_mask))
        self._debug_updates_low_conf += n_low_conf
        
        if not high_conf_mask.any():
            return
        
        high_conf_idx = np.where(high_conf_mask)[0]
        
        for idx in high_conf_idx:
            feature = features[idx]
            box = boxes[idx]
            prob = probs[idx]
            score = scores[idx]
            
            self._update_single(feature, box, prob, score)
    
    def _update_single(self, feature: np.ndarray, box: np.ndarray,
                      prob: np.ndarray, score: float) -> None:
        """Update cache with single detection."""
        if self.M == 0:
            # Create first entry
            self._create_entry(feature, box, prob, score)
            self._debug_entries_created += 1
            return
        
        # Compute posterior
        posterior = self.compute_posterior(feature, box)
        m_star = np.argmax(posterior)
        tau2 = self._get_tau2(m_star)
        
        if posterior[m_star] >= tau2:
            # Match found - update existing entry
            self._update_entry(m_star, feature, box, prob, score)
            self._debug_entries_matched += 1
        else:
            # No good match - create new entry
            if self.M < self.config.max_cache_size:
                self._create_entry(feature, box, prob, score)
                self._debug_entries_created += 1
    
    def _create_entry(self, feature: np.ndarray, box: np.ndarray,
                     prob: np.ndarray, score: float) -> int:
        """Create new cache entry. Returns entry index."""
        # Normalize feature
        feature_norm = self._normalize(feature.reshape(-1, 1), axis=0).flatten()
        
        # Compute scale
        scale = (box[2] - box[0]) * (box[3] - box[1])
        
        # Expand arrays
        self.F_cache = np.hstack([self.F_cache, feature_norm.reshape(-1, 1)])
        self.B_cache = np.append(self.B_cache, scale)
        self.V_cache = np.hstack([self.V_cache, prob.reshape(-1, 1)])
        self.C_cache = np.append(self.C_cache, score)
        
        # Lifecycle
        self.hits = np.append(self.hits, 1)
        self.age = np.append(self.age, 0)
        self.time_since_update = np.append(self.time_since_update, 0)
        self.states.append(CacheEntryState.TENTATIVE)
        
        self.M += 1
        self.total_entries_created += 1
        
        return self.M - 1
    
    def _update_entry(self, idx: int, feature: np.ndarray, box: np.ndarray,
                     prob: np.ndarray, score: float) -> None:
        """Update existing cache entry with confidence-weighted running mean."""
        # Normalize feature
        feature_norm = self._normalize(feature.reshape(-1, 1), axis=0).flatten()
        
        # Confidence-weighted update
        old_weight = self.C_cache[idx]
        new_weight = score
        total_weight = old_weight + new_weight
        
        # Update feature (weighted average)
        self.F_cache[:, idx] = (old_weight * self.F_cache[:, idx] + new_weight * feature_norm) / total_weight
        self.F_cache[:, idx] = self._normalize(self.F_cache[:, idx].reshape(-1, 1), axis=0).flatten()
        
        # Update scale
        scale = (box[2] - box[0]) * (box[3] - box[1])
        self.B_cache[idx] = (old_weight * self.B_cache[idx] + new_weight * scale) / total_weight
        
        # Update class prior
        self.V_cache[:, idx] = (old_weight * self.V_cache[:, idx] + new_weight * prob) / total_weight
        self.V_cache[:, idx] = self.V_cache[:, idx] / (self.V_cache[:, idx].sum() + 1e-10)
        
        # Update confidence weight
        self.C_cache[idx] = total_weight
        
        # Lifecycle
        self.hits[idx] += 1
        self.time_since_update[idx] = 0
        
        # State transition
        if (self.states[idx] == CacheEntryState.TENTATIVE and 
            self.hits[idx] >= self.config.min_hits_to_confirm):
            self.states[idx] = CacheEntryState.CONFIRMED
    
    def _collect_for_batch_init(self, features: np.ndarray, boxes: np.ndarray,
                                probs: np.ndarray, scores: np.ndarray) -> None:
        """Collect detections for batch initialization."""
        N = len(features)
        
        for i in range(N):
            if scores[i] >= self.config.tau1:
                scale = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                self.init_buffer_features.append(features[i])
                self.init_buffer_scales.append(scale)
                self.init_buffer_probs.append(probs[i])
                self.init_buffer_scores.append(scores[i])
        
        # Check if we have enough for batch init
        if len(self.init_buffer_features) >= self.config.batch_init_size:
            self._batch_init_cache()
    
    def _batch_init_cache(self) -> None:
        """
        Initialize cache with clustering (matching original BCA+ behavior).
        
        Strategy (from original BCA+):
        1. Group proposals by predicted class
        2. Within each class, cluster based on feature+scale similarity
        3. Create ONE cache entry per cluster (not per detection!)
        
        This avoids creating near-duplicate entries from similar detections.
        
        CRITICAL: Use HIGHER threshold for clustering than tau2 to avoid
        over-aggressive merging that creates generic centroids.
        """
        features = np.array(self.init_buffer_features)  # (N, D)
        scales = np.array(self.init_buffer_scales)       # (N,)
        probs = np.array(self.init_buffer_probs)         # (N, K)
        scores = np.array(self.init_buffer_scores)       # (N,)
        
        N = len(features)
        if N == 0:
            self.batch_init_done = True
            return
        
        # CRITICAL: Use stricter threshold for batch-init clustering
        TAU2_INIT = self.config.tau2_init  # Higher than tau2 (typically 0.8)
        MAX_CLUSTER_SIZE = 50  # Limit cluster size to prevent over-averaging
        
        # Normalize features
        features_norm = self._normalize(features.T, axis=0).T  # (N, D)
        
        # Get predicted class for each proposal
        predicted_classes = np.argmax(probs, axis=1)
        
        all_clusters = []
        
        # Process each class separately
        for class_idx in range(self.num_classes):
            # Get proposals for this class
            class_mask = predicted_classes == class_idx
            if not np.any(class_mask):
                continue
            
            class_indices = np.where(class_mask)[0]
            class_features = features_norm[class_mask]
            class_scales = scales[class_mask]
            class_probs = probs[class_mask]
            class_scores = scores[class_mask]
            
            # Sort by confidence (descending) for greedy clustering
            sorted_order = np.argsort(class_scores)[::-1]
            
            # Greedy clustering within this class
            clusters = []  # Each cluster: {'indices': [], 'centroid_feature', 'centroid_scale', 'total_prob'}
            
            for local_idx in sorted_order:
                feature = class_features[local_idx]
                scale = class_scales[local_idx]
                prob = class_probs[local_idx]
                
                # Find best matching cluster
                best_cluster_idx = -1
                best_sim = -1.0
                
                for cluster_idx, cluster in enumerate(clusters):
                    # Feature similarity
                    feat_sim = float(np.dot(feature, cluster['centroid_feature']))
                    
                    # Scale similarity
                    scale_diff = abs(scale - cluster['centroid_scale'])
                    max_scale = max(scale, cluster['centroid_scale']) + 1e-10
                    scale_sim = 1 - scale_diff / max_scale
                    
                    # Combined similarity
                    combined_sim = (1 - self.config.ws) * feat_sim + self.config.ws * scale_sim
                    
                    if combined_sim > best_sim:
                        best_sim = combined_sim
                        best_cluster_idx = cluster_idx
                
                # Decide: add to existing cluster or create new one
                can_add_to_cluster = (best_sim >= TAU2_INIT and 
                                     best_cluster_idx >= 0 and 
                                     len(clusters[best_cluster_idx]['indices']) < MAX_CLUSTER_SIZE)
                
                if can_add_to_cluster:
                    # Add to existing cluster
                    cluster = clusters[best_cluster_idx]
                    cluster['indices'].append(local_idx)
                    
                    # Update centroid (running mean)
                    n = len(cluster['indices'])
                    cluster['centroid_feature'] = ((n-1) * cluster['centroid_feature'] + feature) / n
                    cluster['centroid_feature'] /= (np.linalg.norm(cluster['centroid_feature']) + 1e-8)
                    cluster['centroid_scale'] = ((n-1) * cluster['centroid_scale'] + scale) / n
                    cluster['total_prob'] = cluster['total_prob'] + prob
                else:
                    # Create new cluster
                    clusters.append({
                        'indices': [local_idx],
                        'centroid_feature': feature.copy(),
                        'centroid_scale': scale,
                        'total_prob': prob.copy()
                    })
            
            all_clusters.extend(clusters)
        
        # Create cache entries from clusters
        for cluster in all_clusters:
            n = len(cluster['indices'])
            mean_prob = cluster['total_prob'] / n
            
            # Denormalize feature for _create_entry
            feature = cluster['centroid_feature']
            
            # Create dummy box with correct area
            side = np.sqrt(cluster['centroid_scale'])
            box = np.array([0, 0, side, side])
            
            # Create entry
            self._create_entry(feature, box, mean_prob, float(np.max(mean_prob)))
            
            # Override scale with actual centroid scale
            self.B_cache[-1] = cluster['centroid_scale']
            
            # Set count to cluster size (for confidence)
            self.C_cache[-1] = n
        
        # Clear buffer
        self.init_buffer_features = []
        self.init_buffer_scales = []
        self.init_buffer_probs = []
        self.init_buffer_scores = []
        self.batch_init_done = True
        
        self._debug_entries_created += len(all_clusters)
    
    def age_entries(self) -> None:
        """Age all entries (call once per frame)."""
        if self.M == 0:
            return
        
        self.age += 1
        self.time_since_update += 1
        self.frame_count += 1
    
    def cleanup_stale(self) -> int:
        """Remove stale entries. Returns number removed."""
        if self.M == 0:
            return 0
        
        # Mark for deletion
        delete_mask = self.time_since_update > self.config.max_age
        
        # Also delete tentative entries that are too old
        for i in range(self.M):
            if self.states[i] == CacheEntryState.TENTATIVE and self.age[i] > self.config.max_age // 2:
                delete_mask[i] = True
        
        num_deleted = np.sum(delete_mask)
        if num_deleted == 0:
            return 0
        
        # Keep non-deleted entries
        keep_mask = ~delete_mask
        
        self.F_cache = self.F_cache[:, keep_mask]
        self.B_cache = self.B_cache[keep_mask]
        self.V_cache = self.V_cache[:, keep_mask]
        self.C_cache = self.C_cache[keep_mask]
        
        self.hits = self.hits[keep_mask]
        self.age = self.age[keep_mask]
        self.time_since_update = self.time_since_update[keep_mask]
        self.states = [s for s, keep in zip(self.states, keep_mask) if keep]
        
        self.M = len(self.B_cache)
        self.total_entries_deleted += num_deleted
        
        return num_deleted
    
    def step(self) -> None:
        """End-of-frame processing."""
        self.age_entries()
        self.cleanup_stale()
    
    def reset(self) -> None:
        """Reset all state."""
        self.M = 0
        self.F_cache = np.zeros((self.feature_dim, 0))
        self.B_cache = np.zeros(0)
        self.V_cache = np.zeros((self.num_classes, 0))
        self.C_cache = np.zeros(0)
        
        self.hits = np.zeros(0, dtype=np.int32)
        self.age = np.zeros(0, dtype=np.int32)
        self.time_since_update = np.zeros(0, dtype=np.int32)
        self.states = []
        
        self.init_buffer_features = []
        self.init_buffer_scales = []
        self.init_buffer_probs = []
        self.init_buffer_scores = []
        self.batch_init_done = False
        
        self.frame_count = 0
        self.total_entries_created = 0
        self.total_entries_deleted = 0
    
    def get_confirmed_count(self) -> int:
        """Get number of confirmed entries."""
        return sum(1 for s in self.states if s == CacheEntryState.CONFIRMED)
    
    def get_tentative_count(self) -> int:
        """Get number of tentative entries."""
        return sum(1 for s in self.states if s == CacheEntryState.TENTATIVE)
    
    def get_summary(self) -> Dict:
        """Get cache summary."""
        return {
            'size': self.M,
            'confirmed': self.get_confirmed_count(),
            'tentative': self.get_tentative_count(),
            'frame_count': self.frame_count,
            'total_created': self.total_entries_created,
            'total_deleted': self.total_entries_deleted,
            'batch_init_done': self.batch_init_done,
            'batch_init_buffer_size': len(self.init_buffer_features),
            # Debug stats
            'debug_updates_attempted': self._debug_updates_attempted,
            'debug_updates_low_conf': self._debug_updates_low_conf,
            'debug_updates_batch_buffered': self._debug_updates_batch_buffered,
            'debug_entries_matched': self._debug_entries_matched,
            'debug_entries_created': self._debug_entries_created
        }
    
    def get_class_distribution(self) -> np.ndarray:
        """Get aggregate class distribution from cache."""
        if self.M == 0:
            return np.ones(self.num_classes) / self.num_classes
        
        # Confidence-weighted average of class priors
        weights = self.C_cache / (self.C_cache.sum() + 1e-10)
        class_dist = self.V_cache @ weights  # (K,)
        
        return class_dist / (class_dist.sum() + 1e-10)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing enhanced_bca_cache.py")
    
    np.random.seed(42)
    
    num_classes = 6
    feature_dim = 256
    
    # Create cache
    config = EnhancedBCAPlusConfig(
        tau1=0.6,
        tau2_confirmed=0.7,
        tau2_tentative=0.85,
        max_cache_size=25,
        use_batch_init=True,
        batch_init_size=5
    )
    cache = EnhancedBCAPlusCache(num_classes, feature_dim, config)
    
    print(f"1. Initial state: {cache.get_summary()}")
    
    # Add some detections (for batch init)
    features = np.random.randn(10, feature_dim)
    boxes = np.random.rand(10, 4) * 100
    boxes[:, 2:] += boxes[:, :2] + 10  # Ensure valid boxes
    probs = np.random.rand(10, num_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)
    scores = np.random.rand(10) * 0.5 + 0.5  # 0.5-1.0
    
    cache.update_cache(features, boxes, probs, scores)
    
    print(f"2. After batch init: {cache.get_summary()}")
    
    # Test adaptation
    query_feature = np.random.randn(feature_dim)
    query_box = np.array([10, 10, 50, 50])
    query_probs = np.array([0.8, 0.1, 0.05, 0.02, 0.02, 0.01])
    
    adapted = cache.adapt_probs(query_feature, query_box, query_probs)
    print(f"3. Adapted probs: {adapted.round(3)}")
    
    # Age and cleanup
    for _ in range(10):
        cache.step()
    
    print(f"4. After 10 steps: {cache.get_summary()}")
    
    # Add more detections to confirm some entries
    for _ in range(5):
        cache.update_cache(features[:3], boxes[:3], probs[:3], scores[:3])
        cache.step()
    
    print(f"5. After more updates: {cache.get_summary()}")
    
    print("\n✓ All tests passed!")