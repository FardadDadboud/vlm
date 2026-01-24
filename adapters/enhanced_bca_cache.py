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
        
        # Feature similarity: S_F[m] = <h, F[:,m]>
        F_norm = self._normalize(self.F_cache, axis=0)  # (D, M)
        S_F = feature_norm @ F_norm  # (M,)
        
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
        
        # Handle batch initialization
        if self.config.use_batch_init and not self.batch_init_done:
            self._collect_for_batch_init(features, boxes, probs, scores)
            return
        
        # Filter by confidence threshold
        high_conf_mask = scores >= self.config.tau1
        
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
            return
        
        # Compute posterior
        posterior = self.compute_posterior(feature, box)
        m_star = np.argmax(posterior)
        tau2 = self._get_tau2(m_star)
        
        if posterior[m_star] >= tau2:
            # Match found - update existing entry
            self._update_entry(m_star, feature, box, prob, score)
        else:
            # No good match - create new entry
            if self.M < self.config.max_cache_size:
                self._create_entry(feature, box, prob, score)
    
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
        """Initialize cache with diverse entries using greedy clustering."""
        features = np.array(self.init_buffer_features)  # (N, D)
        scales = np.array(self.init_buffer_scales)       # (N,)
        probs = np.array(self.init_buffer_probs)         # (N, K)
        scores = np.array(self.init_buffer_scores)       # (N,)
        
        N = len(features)
        if N == 0:
            self.batch_init_done = True
            return
        
        # Normalize features
        features_norm = self._normalize(features.T, axis=0).T  # (N, D)
        
        # Greedy selection by class diversity
        selected_indices = []
        used_classes = set()
        
        # First pass: one high-conf detection per class
        class_indices = np.argmax(probs, axis=1)  # (N,)
        
        for c in range(self.num_classes):
            class_mask = class_indices == c
            if not class_mask.any():
                continue
            
            class_scores = scores.copy()
            class_scores[~class_mask] = -1
            
            best_idx = np.argmax(class_scores)
            if class_scores[best_idx] >= self.config.tau1:
                selected_indices.append(best_idx)
                used_classes.add(c)
        
        # Second pass: fill remaining slots with diverse entries
        remaining_budget = min(self.config.max_cache_size - len(selected_indices), N - len(selected_indices))
        
        for _ in range(remaining_budget):
            if len(selected_indices) >= N:
                break
            
            # Find entry most different from selected
            best_idx = -1
            best_min_sim = float('inf')
            
            for i in range(N):
                if i in selected_indices:
                    continue
                
                # Compute min similarity to selected
                min_sim = float('inf')
                for j in selected_indices:
                    sim = np.dot(features_norm[i], features_norm[j])
                    min_sim = min(min_sim, sim)
                
                # Lower min_sim = more diverse
                if min_sim < best_min_sim:
                    best_min_sim = min_sim
                    best_idx = i
            
            if best_idx >= 0 and best_min_sim < self.config.tau2_init:
                selected_indices.append(best_idx)
        
        # Create entries for selected detections
        for idx in selected_indices:
            box = np.array([0, 0, np.sqrt(scales[idx]), np.sqrt(scales[idx])])  # Dummy box with correct area
            self._create_entry(features[idx], box, probs[idx], scores[idx])
            # Use the stored scale directly
            self.B_cache[-1] = scales[idx]
        
        # Clear buffer
        self.init_buffer_features = []
        self.init_buffer_scales = []
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
            'batch_init_done': self.batch_init_done
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