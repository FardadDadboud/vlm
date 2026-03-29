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
    tau1_per_class: Optional[Dict[int, float]] = None  # {class_id: threshold}
    
    # tau2: Similarity threshold for cache MATCH (matching original BCA+ behavior)
    # Use SINGLE tau2 for all entries - lifecycle state doesn't affect matching threshold
    tau2: float = 0.5  # Original BCA+ default - lower = more aggressive merging
    tau2_init: float = 0.5  # Threshold during batch initialization clustering
    
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
    min_hits_to_confirm_cache: int = 1   # Hits needed to confirm entry
    max_age_cache: int = 999999              # Frames before deletion
    
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
    - B_cache: (2, M) normalized scales [w/image_w, h/image_h] - MATCHING ORIGINAL BCA+
    - V_cache: (K, M) class prior distributions
    - C_cache: (M,) update counts
    
    Lifecycle:
    - hits: (M,) number of times entry was matched
    - age: (M,) frames since creation
    - time_since_update: (M,) frames since last match
    - states: (M,) CacheEntryState for each entry
    
    Posterior computation (Eq. 11-12):
    P(m|x) ∝ exp(τ * ((1-ws)*S_F + ws*S_B))
    where S_F = cosine similarity, S_B = L2-based scale similarity (Eq. 8)
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 config: Optional[EnhancedBCAPlusConfig] = None,
                 class_names: Optional[List[str]] = None,
                 image_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize enhanced BCA+ cache.
        
        Args:
            num_classes: Number of classes K
            feature_dim: Feature dimension D
            config: Configuration
            class_names: Optional class names for debugging
            image_size: (width, height) for scale normalization (paper requirement)
        """
        self.config = config or EnhancedBCAPlusConfig()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_names = class_names
        self.image_size = image_size  # For scale normalization (paper Eq. 8)
        
        # === Cache Arrays (initially empty) ===
        self.M = 0  # Current cache size
        self.F_cache: np.ndarray = np.zeros((feature_dim, 0))  # (D, M)
        self.B_cache: np.ndarray = np.zeros((2, 0))            # (2, M) - [w/W, h/H] normalized
        self.V_cache: np.ndarray = np.zeros((num_classes, 0))  # (K, M)
        self.C_cache: np.ndarray = np.zeros(0)                 # (M,) update counts
        
        # === Lifecycle Arrays ===
        self.hits: np.ndarray = np.zeros(0, dtype=np.int32)
        self.age: np.ndarray = np.zeros(0, dtype=np.int32)
        self.time_since_update: np.ndarray = np.zeros(0, dtype=np.int32)
        self.states: List[CacheEntryState] = []
        
        # === Batch Initialization Buffer ===
        self.init_buffer_features: List[np.ndarray] = []
        self.init_buffer_boxes: List[np.ndarray] = []  # Store full boxes [x1,y1,x2,y2]
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
        self._debug_last_posterior_max: float = 0.0
        self._debug_last_tau2: float = 0.0
    
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
    
    def _get_tau2(self, entry_idx: int = None) -> float:
        """Get tau2 threshold for cache matching.
        
        Unlike the original enhanced version, we use a SINGLE tau2 for all entries
        (matching original BCA+ behavior). Lifecycle state doesn't affect matching.
        """
        return self.config.tau2
    
    def compute_posterior(self, feature: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Compute posterior distribution over cache entries given detection.
        
        P(m|x) ∝ exp(τ * ((1-ws)*S_F[m] + ws*S_B[m]))
        
        Scale similarity S_B (Eq. 8 from paper):
        S_B = 1 - ||[w,h] - [w_m,h_m]|| / sqrt(2)
        where w,h are normalized to [0,1] by image dimensions.
        
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
        
        # Scale similarity (Eq. 8): S_B = 1 - L2_dist / sqrt(2)
        # Convert box to normalized [w/W, h/H]
        image_w, image_h = self.image_size
        w = box[2] - box[0]
        h = box[3] - box[1]
        normalized_scale = np.array([w / image_w, h / image_h])  # (2,)
        
        # L2 distance to each cached scale
        # B_cache shape: (2, M), normalized_scale shape: (2,)
        distances = np.linalg.norm(self.B_cache - normalized_scale.reshape(2, 1), axis=0)  # (M,)
        
        # Convert to similarity (Eq. 8) - normalize by sqrt(2)
        # "maximum difference of sqrt(2) under perfect misalignment"
        S_B = 1 - distances / np.sqrt(2)  # (M,)
        
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
        
        # Scale similarity (Eq. 8): S_B = 1 - L2_dist / sqrt(2)
        # Convert boxes to normalized [w/W, h/H]
        image_w, image_h = self.image_size
        w = boxes[:, 2] - boxes[:, 0]  # (N,)
        h = boxes[:, 3] - boxes[:, 1]  # (N,)
        normalized_scales = np.stack([w / image_w, h / image_h], axis=1)  # (N, 2)
        
        # Compute L2 distances: (N, M)
        # For each detection n and cache entry m: ||scale_n - B[:,m]||
        # normalized_scales: (N, 2), B_cache: (2, M)
        # Expand to (N, 2, 1) - (1, 2, M) = (N, 2, M), then norm over axis=1
        scale_diff = normalized_scales[:, :, np.newaxis] - self.B_cache[np.newaxis, :, :]  # (N, 2, M)
        distances = np.linalg.norm(scale_diff, axis=1)  # (N, M)
        
        # Convert to similarity
        S_B = 1 - distances / np.sqrt(2)  # (N, M)
        
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
                         class_probs: np.ndarray,
                         return_posteriors: bool = False) -> np.ndarray:
        """
        Batch adapt class probabilities for multiple detections.
        
        Args:
            features: (N, D) detection features
            boxes: (N, 4) detection boxes
            class_probs: (N, K) VLM class probabilities
            return_posteriors: If True, also return the computed posteriors for cache update
            
        Returns:
            (N, K) adapted class probabilities
            OR tuple of ((N, K) adapted, (N, M) posteriors) if return_posteriors=True
        """
        N = len(features)
        if self.M == 0 or N == 0:
            if return_posteriors:
                return class_probs.copy(), None
            return class_probs.copy()
        
        # Batch posterior: (N, M) - FROZEN at current M!
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
        
        if return_posteriors:
            return adapted, posterior
        return adapted
    
    def update_cache(self, features: np.ndarray, boxes: np.ndarray,
                    probs: np.ndarray, scores: np.ndarray,
                    posteriors: np.ndarray = None) -> None:
        """
        Update cache with new detections.
        
        CRITICAL: Original BCA+ uses PRE-COMPUTED posteriors from a FROZEN cache snapshot.
        If posteriors are provided, they will be used for matching decisions.
        If not provided, posteriors are recomputed (which may give different results
        if cache is modified during the update loop - NOT recommended!).
        
        Args:
            features: (N, D) detection features
            boxes: (N, 4) detection boxes
            probs: (N, K) class probabilities
            scores: (N,) confidence scores
            posteriors: (N, M) PRE-COMPUTED posteriors from adapt_probs_batch (optional but recommended!)
        """
        N = len(features)
        if N == 0:
            return
        
        self._debug_updates_attempted += N
        
        # Track per-frame stats
        m_before = self.M
        matches_this_frame = 0
        creates_this_frame = 0
        
        # Handle batch initialization
        if self.config.use_batch_init and not self.batch_init_done:
            self._debug_updates_batch_buffered += N
            self._collect_for_batch_init(features, boxes, probs, scores)
            return
        
        # Filter by confidence threshold
        # high_conf_mask = scores >= self.config.tau1
        # Filter by confidence threshold — CLASS-AWARE
        if hasattr(self.config, 'tau1_per_class') and self.config.tau1_per_class is not None:
            # Per-class thresholds
            high_conf_mask = np.zeros(N, dtype=bool)
            for i in range(N):
                pred_class = int(np.argmax(probs[i]))
                class_tau1 = self.config.tau1_per_class.get(pred_class, self.config.tau1)
                high_conf_mask[i] = scores[i] >= class_tau1
        else:
            high_conf_mask = scores >= self.config.tau1
        n_low_conf = int(np.sum(~high_conf_mask))
        self._debug_updates_low_conf += n_low_conf
        
        if not high_conf_mask.any():
            return
        
        high_conf_idx = np.where(high_conf_mask)[0]
        
        # Track match stats before processing
        matched_before = self._debug_entries_matched
        created_before = self._debug_entries_created
        
        # CRITICAL: Use pre-computed posteriors if provided (matching original BCA+ frozen snapshot!)
        # The posteriors were computed against cache state at START of frame, before any updates
        for idx in high_conf_idx:
            feature = features[idx]
            box = boxes[idx]
            prob = probs[idx]
            score = scores[idx]
            
            # Get posterior for this detection
            if posteriors is not None and self.M > 0 and posteriors.shape[1] == m_before:
                # Use PRE-COMPUTED posterior (frozen snapshot - matches original BCA+!)
                posterior = posteriors[idx]
                self._update_single_with_posterior(feature, box, prob, score, posterior)
            else:
                # Fallback: recompute posterior (not ideal, cache may have changed)
                self._update_single(feature, box, prob, score)
        
        # Calculate per-frame stats
        matches_this_frame = self._debug_entries_matched - matched_before
        creates_this_frame = self._debug_entries_created - created_before
        
        # Debug summary for this frame
        if self.config.debug and (creates_this_frame > 0 or self.M != m_before):
            print(f"    [Cache Update Summary] M: {m_before}→{self.M}, "
                  f"high_conf={len(high_conf_idx)}, matched={matches_this_frame}, "
                  f"created={creates_this_frame}")
            
            # Show similarity stats if we have entries
            if self.M > 0 and m_before > 0 and len(high_conf_idx) > 0:
                # Sample a high-conf detection and show its similarity to cache
                sample_idx = high_conf_idx[0]
                sample_feature = features[sample_idx]
                sample_box = boxes[sample_idx]
                
                # Compute similarities (not posterior, raw similarities)
                feature_norm = sample_feature / (np.linalg.norm(sample_feature) + 1e-8)
                S_F = feature_norm @ self.F_cache[:, :m_before]  # Use pre-update cache
                
                image_w, image_h = self.image_size
                w = sample_box[2] - sample_box[0]
                h = sample_box[3] - sample_box[1]
                normalized_scale = np.array([w / image_w, h / image_h])
                distances = np.linalg.norm(self.B_cache[:, :m_before] - normalized_scale.reshape(2, 1), axis=0)
                S_B = 1 - distances / np.sqrt(2)
                
                S = (1 - self.config.ws) * S_F + self.config.ws * S_B
                
                print(f"      Sample det similarity to cache: S_F=[{S_F.min():.3f}, {S_F.max():.3f}], "
                      f"S_B=[{S_B.min():.3f}, {S_B.max():.3f}], "
                      f"S_combined=[{S.min():.3f}, {S.max():.3f}]")
                
                # Compute what posterior would be
                logits = self.config.logit_temperature * S
                posterior = np.exp(logits - logits.max()) / np.sum(np.exp(logits - logits.max()))
                print(f"      Posterior (temp={self.config.logit_temperature}): "
                      f"max={posterior.max():.3f} at entry {np.argmax(posterior)}, "
                      f"tau2={self.config.tau2}")
    
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
        
        # Debug: track posterior stats for diagnostics
        self._debug_last_posterior_max = float(posterior[m_star])
        self._debug_last_tau2 = tau2
        
        if posterior[m_star] >= tau2:
            # Match found - update existing entry
            self._update_entry(m_star, feature, box, prob, score)
            self._debug_entries_matched += 1
        else:
            # No good match - create new entry
            if self.M < self.config.max_cache_size:
                self._create_entry(feature, box, prob, score)
                self._debug_entries_created += 1
                
                # Debug: log why entry was created
                if self.config.debug:
                    state_str = 'CONFIRMED' if self.states[m_star] == CacheEntryState.CONFIRMED else 'TENTATIVE'
                    print(f"      [Cache] NEW entry: P(m*|x)={posterior[m_star]:.3f} < tau2={tau2:.3f} "
                          f"(m*={m_star} is {state_str}, M now {self.M})")
            else:
                # Cache full - force update best match
                self._update_entry(m_star, feature, box, prob, score)
                self._debug_entries_matched += 1
    
    def _update_single_with_posterior(self, feature: np.ndarray, box: np.ndarray,
                                      prob: np.ndarray, score: float,
                                      posterior: np.ndarray) -> None:
        """
        Update cache with single detection using PRE-COMPUTED posterior.
        
        CRITICAL: This matches original BCA+ behavior where posteriors are computed
        against a FROZEN cache snapshot at the start of each frame.
        
        Args:
            feature: (D,) detection feature
            box: (4,) detection box
            prob: (K,) class probabilities
            score: confidence score
            posterior: (M,) PRE-COMPUTED posterior over cache entries
        """
        if self.M == 0:
            # Create first entry
            self._create_entry(feature, box, prob, score)
            self._debug_entries_created += 1
            return
        
        # Use PRE-COMPUTED posterior (not recomputed!)
        m_star = np.argmax(posterior)
        tau2 = self._get_tau2(m_star)
        
        # Debug: track posterior stats for diagnostics
        self._debug_last_posterior_max = float(posterior[m_star])
        self._debug_last_tau2 = tau2
        
        if posterior[m_star] >= tau2:
            # Match found - update existing entry
            self._update_entry(m_star, feature, box, prob, score)
            self._debug_entries_matched += 1
        else:
            # No good match - create new entry
            if self.M < self.config.max_cache_size:
                self._create_entry(feature, box, prob, score)
                self._debug_entries_created += 1
                
                # Debug: log why entry was created
                if self.config.debug:
                    state_str = 'CONFIRMED' if self.states[m_star] == CacheEntryState.CONFIRMED else 'TENTATIVE'
                    print(f"      [Cache] NEW entry (frozen): P(m*|x)={posterior[m_star]:.3f} < tau2={tau2:.3f} "
                          f"(m*={m_star} is {state_str}, M now {self.M})")
            else:
                # Cache full - force update best match
                self._update_entry(m_star, feature, box, prob, score)
                self._debug_entries_matched += 1
    
    def _create_entry(self, feature: np.ndarray, box: np.ndarray,
                     prob: np.ndarray, score: float) -> int:
        """Create new cache entry (Eq. 16). Returns entry index."""
        # Normalize feature
        feature_norm = self._normalize(feature.reshape(-1, 1), axis=0).flatten()
        
        # Compute normalized scale [w/W, h/H] (matching original BCA+)
        image_w, image_h = self.image_size
        w = box[2] - box[0]
        h = box[3] - box[1]
        scale = np.array([w / image_w, h / image_h])  # (2,)
        
        if self.F_cache is None:
            self.F_cache = feature_norm.reshape(-1, 1)
            self.B_cache = scale.reshape(2, 1)
            self.V_cache = prob.reshape(-1, 1)
            self.C_cache = np.array([1])
        else:
            self.F_cache = np.hstack([self.F_cache, feature_norm.reshape(-1, 1)])
            self.B_cache = np.hstack([self.B_cache, scale.reshape(2, 1)])
            self.V_cache = np.hstack([self.V_cache, prob.reshape(-1, 1)])
            self.C_cache = np.append(self.C_cache, 1)
        
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
        """Update existing cache entry with count-based averaging (Eq. 17)."""
        # Normalize feature
        feature_norm = self._normalize(feature.reshape(-1, 1), axis=0).flatten()
        
        # Count-based update (matching original BCA+)
        c = self.C_cache[idx]
        
        # Update feature embedding
        self.F_cache[:, idx] = (c * self.F_cache[:, idx] + feature_norm) / (c + 1)
        self.F_cache[:, idx] = self._normalize(self.F_cache[:, idx].reshape(-1, 1), axis=0).flatten()
        
        # Update scale - normalized [w/W, h/H]
        image_w, image_h = self.image_size
        w = box[2] - box[0]
        h = box[3] - box[1]
        scale = np.array([w / image_w, h / image_h])
        self.B_cache[:, idx] = (c * self.B_cache[:, idx] + scale) / (c + 1)
        
        # Update class prior
        self.V_cache[:, idx] = (c * self.V_cache[:, idx] + prob) / (c + 1)
        
        # Increment count
        self.C_cache[idx] = c + 1
        
        # Lifecycle
        self.hits[idx] += 1
        self.time_since_update[idx] = 0
        
        # State transition
        if (self.states[idx] == CacheEntryState.TENTATIVE and 
            self.hits[idx] >= self.config.min_hits_to_confirm_cache):
            self.states[idx] = CacheEntryState.CONFIRMED
    
    def _collect_for_batch_init(self, features: np.ndarray, boxes: np.ndarray,
                                probs: np.ndarray, scores: np.ndarray) -> None:
        """Collect detections for batch initialization."""
        N = len(features)
        
        # for i in range(N):
        #     if scores[i] >= self.config.tau1:
        #         # Store boxes directly (original BCA+ style) - NOT scalar area
        #         self.init_buffer_features.append(features[i])
        #         self.init_buffer_boxes.append(boxes[i])  # Store full box [x1,y1,x2,y2]
        #         self.init_buffer_probs.append(probs[i])
        #         self.init_buffer_scores.append(scores[i])

        for i in range(N):
            # Use per-class tau1 if available (matching update_cache behavior)
            if hasattr(self.config, 'tau1_per_class') and self.config.tau1_per_class is not None:
                pred_class = int(np.argmax(probs[i]))
                effective_tau1 = self.config.tau1_per_class.get(pred_class, self.config.tau1)
            else:
                effective_tau1 = self.config.tau1
            if scores[i] >= effective_tau1:
                self.init_buffer_features.append(features[i])
                self.init_buffer_boxes.append(boxes[i])  # Store full box [x1,y1,x2,y2]
                self.init_buffer_probs.append(probs[i])
                self.init_buffer_scores.append(scores[i])
        
        # Check if we have enough for batch init
        if len(self.init_buffer_features) >= self.config.batch_init_size:
            self._batch_init_cache()
    
    def _batch_init_cache(self) -> None:
        """
        Batch initialize cache from first frame with clustering (Paper-aligned).
        
        MATCHING ORIGINAL BCA+ EXACTLY:
        1. Group proposals by predicted class
        2. Within each class, cluster based on feature+scale similarity
        3. Create ONE cache entry per cluster (not per detection!)
        
        Scale similarity uses Eq. 8: S_B = 1 - ||[w,h] - [w_m,h_m]|| / sqrt(2)
        where w,h are normalized to [0,1] by image dimensions.
        """
        features = np.array(self.init_buffer_features)  # (N, D)
        boxes = np.array(self.init_buffer_boxes)        # (N, 4)
        probs = np.array(self.init_buffer_probs)        # (N, K)
        scores = np.array(self.init_buffer_scores)      # (N,)
        
        N = len(features)
        if N == 0:
            self.batch_init_done = True
            return
        
        # CRITICAL: Use stricter threshold for batch-init clustering
        TAU2_INIT = self.config.tau2_init  # Typically 0.5
        MAX_CLUSTER_SIZE = 50  # Limit cluster size to prevent over-averaging
        
        # Get predicted class for each proposal
        predicted_classes = np.argmax(probs, axis=1)
        
        # Process each class separately (matching original BCA+)
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
            clusters = []  # Each cluster: {'indices': [], 'centroid_feature', 'centroid_scale', 'total_prob'}
            
            for idx in sorted_indices:
                feature = class_features[idx]
                box = class_boxes[idx]
                prob = class_probs[idx]
                
                # Normalize feature (matching original BCA+)
                feature_norm = feature.copy()
                feature_norm = feature_norm / (np.linalg.norm(feature_norm) + 1e-8)
                
                # Get scale [w, h] normalized to [0,1] (matching original BCA+ Eq. 8)
                image_w, image_h = self.image_size
                w = box[2] - box[0]
                h = box[3] - box[1]
                scale = np.array([w / image_w, h / image_h])  # 2D normalized scale
                
                # Find best matching cluster
                best_cluster_idx = -1
                best_sim = -1.0
                
                for cluster_idx, cluster in enumerate(clusters):
                    # Feature similarity (cosine)
                    feat_sim = feature_norm @ cluster['centroid_feature']
                    
                    # Scale similarity (Eq. 8): S_B = 1 - L2_dist / sqrt(2)
                    scale_diff = np.linalg.norm(scale - cluster['centroid_scale'])
                    scale_sim = 1 - scale_diff / np.sqrt(2)
                    
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
                    cluster['indices'].append(idx)
                    
                    # Update centroid (running mean)
                    n = len(cluster['indices'])
                    cluster['centroid_feature'] = ((n-1) * cluster['centroid_feature'] + feature_norm) / n
                    cluster['centroid_feature'] /= (np.linalg.norm(cluster['centroid_feature']) + 1e-8)  # Renormalize
                    cluster['centroid_scale'] = ((n-1) * cluster['centroid_scale'] + scale) / n  # 2D scale
                    cluster['total_prob'] += prob
                else:
                    # Create new cluster
                    clusters.append({
                        'indices': [idx],
                        'centroid_feature': feature_norm.copy(),
                        'centroid_scale': scale.copy(),  # 2D scale [w/W, h/H]
                        'total_prob': prob.copy()
                    })
            
            # Create cache entries from clusters
            for cluster in clusters:
                n = len(cluster['indices'])
                mean_prob = cluster['total_prob'] / n
                
                # Create cache entry using dummy box with correct w,h
                # (matching original BCA+ format)
                centroid_scale = cluster['centroid_scale']  # [w/W, h/H]
                dummy_box = np.array([
                    0, 0, 
                    centroid_scale[0] * self.image_size[0],  # w in pixels
                    centroid_scale[1] * self.image_size[1]   # h in pixels
                ])

                # Before creating a new cache entry:
                pred_class_idx = int(np.argmax(mean_prob))

                # Count existing entries for this class
                class_count = 0
                for j in range(self.M):
                    if int(np.argmax(self.V_cache[:, j])) == pred_class_idx:
                        class_count += 1

                max_per_class = max(2, self.config.max_cache_size // 6)  # e.g., 25//6 = 4
                if class_count >= max_per_class:
                    continue  # Skip — this class already has enough entries
                
                self._create_entry(
                    feature=cluster['centroid_feature'],
                    box=dummy_box,
                    prob=mean_prob,
                    score=float(np.max(mean_prob))
                )
                
                # Set count to cluster size (matching original BCA+)
                self.C_cache[-1] = n
        
        # Clear buffer
        self.init_buffer_features = []
        self.init_buffer_boxes = []
        self.init_buffer_probs = []
        self.init_buffer_scores = []
        self.batch_init_done = True
    
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
        delete_mask = self.time_since_update > self.config.max_age_cache
        
        # Also delete tentative entries that are too old
        for i in range(self.M):
            if self.states[i] == CacheEntryState.TENTATIVE and self.age[i] > self.config.max_age_cache // 2:
                delete_mask[i] = True
        
        num_deleted = np.sum(delete_mask)
        if num_deleted == 0:
            return 0
        
        # Keep non-deleted entries
        keep_mask = ~delete_mask
        
        self.F_cache = self.F_cache[:, keep_mask]
        self.B_cache = self.B_cache[:, keep_mask]  # FIXED: 2D array (2, M)
        self.V_cache = self.V_cache[:, keep_mask]
        self.C_cache = self.C_cache[keep_mask]
        
        self.hits = self.hits[keep_mask]
        self.age = self.age[keep_mask]
        self.time_since_update = self.time_since_update[keep_mask]
        self.states = [s for s, keep in zip(self.states, keep_mask) if keep]
        
        self.M = self.B_cache.shape[1]  # FIXED: shape[1] for 2D array
        self.total_entries_deleted += num_deleted
        
        return num_deleted
    
    def step(self) -> None:
        """End-of-frame processing."""
        self.age_entries()
        self.cleanup_stale()
    
    def reset(self) -> None:
        """Reset all state."""
        self.M = 0
        # self.F_cache = np.zeros((self.feature_dim, 0))
        # self.B_cache = np.zeros((2, 0))  # FIXED: 2D array (2, M)
        # self.V_cache = np.zeros((self.num_classes, 0))
        # self.C_cache = np.zeros(0)

        self.F_cache = None
        self.B_cache = None
        self.V_cache = None
        self.C_cache = None

        self.hits = np.zeros(0, dtype=np.int32)
        self.age = np.zeros(0, dtype=np.int32)
        self.time_since_update = np.zeros(0, dtype=np.int32)
        self.states = []
        
        self.init_buffer_features = []
        self.init_buffer_boxes = []  # FIXED: boxes not scales
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
        tau2=0.5,  # Single tau2 like original BCA+
        tau2_init=0.5,
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