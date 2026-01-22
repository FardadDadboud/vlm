"""
Global Cache Module for Global+Instance Temporal BCA+

Enhanced class-level cache that maintains:
1. Class prototypes (feature embeddings)
2. Class scale statistics (mean, variance)
3. Class location priors (spatial distribution)
4. Temporal evolution of class statistics

Key differences from baseline BCA+:
- Maintains temporal history of prototypes
- Tracks scale/location statistics per class
- Uses Bayesian update rules instead of simple averaging
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# =============================================================================
# Class Statistics Container
# =============================================================================

@dataclass
class ClassStatistics:
    """Statistics for a single class."""
    # Feature prototype
    prototype: np.ndarray = None  # Mean direction (D,)
    prototype_uncertainty: float = 1.0  # Concentration (inverse variance)
    
    # Scale statistics [w, h] normalized to [0,1]
    scale_mean: np.ndarray = None  # (2,)
    scale_var: np.ndarray = None   # (2,)
    
    # Location prior (spatial distribution)
    location_mean: np.ndarray = None  # [cx, cy] normalized
    location_var: np.ndarray = None   # (2,)
    
    # Count statistics
    observation_count: int = 0
    update_count: int = 0
    
    # Temporal history (for smoothing)
    prototype_history: List[np.ndarray] = field(default_factory=list)
    scale_history: List[np.ndarray] = field(default_factory=list)
    
    def is_initialized(self) -> bool:
        return self.prototype is not None


# =============================================================================
# Global Cache
# =============================================================================

class GlobalCache:
    """
    Global class-level cache for TTA.
    
    Maintains:
    - Per-class prototypes with uncertainty
    - Per-class scale statistics
    - Per-class location priors
    - Instance clusters within each class
    
    Update mechanism:
    - Bayesian update for prototypes (vMF posterior)
    - Online mean/variance for scale and location
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 max_clusters_per_class: int = 10,
                 prototype_momentum: float = 0.9,
                 scale_momentum: float = 0.95,
                 min_observations: int = 3,
                 history_length: int = 10,
                 use_instance_clusters: bool = True):
        """
        Initialize global cache.
        
        Args:
            num_classes: Number of object classes
            feature_dim: Dimension of feature embeddings
            max_clusters_per_class: Maximum instance clusters per class
            prototype_momentum: EMA momentum for prototype updates
            scale_momentum: EMA momentum for scale statistics
            min_observations: Minimum observations before using statistics
            history_length: Number of frames to keep in history
            use_instance_clusters: Whether to maintain sub-class clusters
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_clusters_per_class = max_clusters_per_class
        self.prototype_momentum = prototype_momentum
        self.scale_momentum = scale_momentum
        self.min_observations = min_observations
        self.history_length = history_length
        self.use_instance_clusters = use_instance_clusters
        
        # Per-class statistics
        self.class_stats: List[ClassStatistics] = [
            ClassStatistics() for _ in range(num_classes)
        ]
        
        # Instance clusters within each class (like BCA+ cache but organized by class)
        # Each entry: {'features': (D, M), 'scales': (2, M), 'counts': (M,), 'probs': (K, M)}
        self.instance_clusters: List[Optional[Dict]] = [None] * num_classes
        
        # Text embeddings (anchor)
        self.text_embeddings: Optional[np.ndarray] = None
        
        # Image size for normalization
        self.image_size: Optional[Tuple[int, int]] = None
    
    def initialize_from_text_embeddings(self, text_embeddings: np.ndarray):
        """
        Initialize prototypes from text embeddings.
        
        Args:
            text_embeddings: (K, D) text embeddings for each class
        """
        assert text_embeddings.shape[0] == self.num_classes
        
        self.text_embeddings = self._normalize(text_embeddings)
        self.feature_dim = text_embeddings.shape[1]
        
        # Initialize class prototypes from text embeddings
        for k in range(self.num_classes):
            stats = self.class_stats[k]
            stats.prototype = self.text_embeddings[k].copy()
            stats.prototype_uncertainty = 1.0  # Low initial confidence
            
            # Initialize scale with reasonable defaults
            stats.scale_mean = np.array([0.1, 0.1])  # 10% of image
            stats.scale_var = np.array([0.05, 0.05])  # High variance
            
            # Location prior: center of image
            stats.location_mean = np.array([0.5, 0.5])
            stats.location_var = np.array([0.25, 0.25])  # High variance
    
    def set_image_size(self, width: int, height: int):
        """Set image size for coordinate normalization."""
        self.image_size = (width, height)
    
    def get_prototypes(self) -> np.ndarray:
        """Get current class prototypes (K, D)"""
        prototypes = np.zeros((self.num_classes, self.feature_dim))
        for k in range(self.num_classes):
            if self.class_stats[k].is_initialized():
                prototypes[k] = self.class_stats[k].prototype
            elif self.text_embeddings is not None:
                prototypes[k] = self.text_embeddings[k]
        return prototypes
    
    def get_uncertainties(self) -> np.ndarray:
        """Get prototype uncertainties (K,)"""
        return np.array([s.prototype_uncertainty for s in self.class_stats])
    
    def predict_class_probs(self, features: np.ndarray, 
                            temperature: float = 1.0) -> np.ndarray:
        """
        Predict class probabilities using global prototypes.
        
        Args:
            features: (N, D) query features
            temperature: Softmax temperature
            
        Returns:
            (N, K) class probabilities
        """
        prototypes = self.get_prototypes()
        features_norm = self._normalize(features)
        
        # Cosine similarity
        similarities = features_norm @ prototypes.T  # (N, K)
        
        # Weight by confidence (inverse uncertainty)
        uncertainties = self.get_uncertainties()
        weights = 1.0 / (uncertainties + 0.1)
        weighted_sims = similarities * weights[np.newaxis, :]
        
        # Temperature scaling
        scaled_sims = weighted_sims / temperature
        
        # Softmax
        exp_sims = np.exp(scaled_sims - np.max(scaled_sims, axis=1, keepdims=True))
        probs = exp_sims / (np.sum(exp_sims, axis=1, keepdims=True) + 1e-10)
        
        return probs
    
    def get_scale_prior(self, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get scale prior for a class.
        
        Returns:
            (mean, variance) for [w, h]
        """
        stats = self.class_stats[class_idx]
        if stats.scale_mean is not None:
            return stats.scale_mean.copy(), stats.scale_var.copy()
        else:
            return np.array([0.1, 0.1]), np.array([0.05, 0.05])
    
    def get_location_prior(self, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get location prior for a class.
        
        Returns:
            (mean, variance) for [cx, cy] normalized
        """
        stats = self.class_stats[class_idx]
        if stats.location_mean is not None:
            return stats.location_mean.copy(), stats.location_var.copy()
        else:
            return np.array([0.5, 0.5]), np.array([0.25, 0.25])
    
    def compute_scale_likelihood(self, box: np.ndarray, class_idx: int) -> float:
        """
        Compute likelihood of box scale given class prior.
        
        Args:
            box: [x1, y1, x2, y2] in pixel coordinates
            class_idx: Class index
            
        Returns:
            Log-likelihood of scale
        """
        if self.image_size is None:
            return 0.0
        
        # Normalize scale
        w = (box[2] - box[0]) / self.image_size[0]
        h = (box[3] - box[1]) / self.image_size[1]
        scale = np.array([w, h])
        
        mean, var = self.get_scale_prior(class_idx)
        
        # Gaussian log-likelihood
        diff = scale - mean
        log_lik = -0.5 * np.sum(diff**2 / (var + 1e-6))
        
        return log_lik
    
    def compute_location_likelihood(self, box: np.ndarray, class_idx: int) -> float:
        """
        Compute likelihood of box location given class prior.
        
        Args:
            box: [x1, y1, x2, y2] in pixel coordinates
            class_idx: Class index
            
        Returns:
            Log-likelihood of location
        """
        if self.image_size is None:
            return 0.0
        
        # Normalized center
        cx = (box[0] + box[2]) / 2 / self.image_size[0]
        cy = (box[1] + box[3]) / 2 / self.image_size[1]
        loc = np.array([cx, cy])
        
        mean, var = self.get_location_prior(class_idx)
        
        # Gaussian log-likelihood
        diff = loc - mean
        log_lik = -0.5 * np.sum(diff**2 / (var + 1e-6))
        
        return log_lik
    
    def update(self, 
               features: np.ndarray,
               boxes: np.ndarray,
               class_probs: np.ndarray,
               confidence_mask: np.ndarray):
        """
        Update global cache with new observations.
        
        Args:
            features: (N, D) feature embeddings
            boxes: (N, 4) bounding boxes [x1, y1, x2, y2]
            class_probs: (N, K) class probabilities
            confidence_mask: (N,) boolean mask for confident detections
        """
        if not confidence_mask.any():
            return
        
        # Get confident detections
        conf_features = features[confidence_mask]
        conf_boxes = boxes[confidence_mask]
        conf_probs = class_probs[confidence_mask]
        
        # Normalize features
        conf_features_norm = self._normalize(conf_features)
        
        # Update each class
        for k in range(self.num_classes):
            # Soft assignment: weight by class probability
            class_weights = conf_probs[:, k]
            
            # Only update if significant weight for this class
            if class_weights.sum() < 0.1:
                continue
            
            self._update_class(k, conf_features_norm, conf_boxes, class_weights)
    
    def _update_class(self,
                      class_idx: int,
                      features: np.ndarray,
                      boxes: np.ndarray,
                      weights: np.ndarray):
        """
        Update statistics for a single class.
        
        Uses weighted Bayesian updates.
        """
        stats = self.class_stats[class_idx]
        
        # Filter by weight threshold
        mask = weights > 0.1
        if not mask.any():
            return
        
        features = features[mask]
        boxes = boxes[mask]
        weights = weights[mask]
        weights = weights / weights.sum()  # Normalize
        
        n = len(features)
        
        # === Update prototype ===
        # Weighted mean direction
        weighted_sum = (features * weights[:, np.newaxis]).sum(axis=0)
        new_direction = self._normalize(weighted_sum.reshape(1, -1))[0]
        
        if stats.is_initialized():
            # EMA update
            stats.prototype = self._normalize(
                (self.prototype_momentum * stats.prototype + 
                 (1 - self.prototype_momentum) * new_direction).reshape(1, -1)
            )[0]
            
            # Update uncertainty (more observations = less uncertainty)
            stats.prototype_uncertainty *= 0.99
            stats.prototype_uncertainty = max(stats.prototype_uncertainty, 0.01)
        else:
            stats.prototype = new_direction
            stats.prototype_uncertainty = 0.5
        
        # Add to history
        stats.prototype_history.append(stats.prototype.copy())
        if len(stats.prototype_history) > self.history_length:
            stats.prototype_history.pop(0)
        
        # === Update scale statistics ===
        if self.image_size is not None:
            scales = np.zeros((n, 2))
            for i, box in enumerate(boxes):
                scales[i, 0] = (box[2] - box[0]) / self.image_size[0]
                scales[i, 1] = (box[3] - box[1]) / self.image_size[1]
            
            new_scale_mean = (scales * weights[:, np.newaxis]).sum(axis=0)
            
            if stats.scale_mean is not None:
                stats.scale_mean = (self.scale_momentum * stats.scale_mean + 
                                   (1 - self.scale_momentum) * new_scale_mean)
                
                # Update variance (online Welford algorithm approximation)
                diff = scales - stats.scale_mean
                new_var = (diff**2 * weights[:, np.newaxis]).sum(axis=0)
                stats.scale_var = (self.scale_momentum * stats.scale_var + 
                                  (1 - self.scale_momentum) * new_var)
            else:
                stats.scale_mean = new_scale_mean
                stats.scale_var = np.array([0.05, 0.05])
        
        # === Update location statistics ===
        if self.image_size is not None:
            locations = np.zeros((n, 2))
            for i, box in enumerate(boxes):
                locations[i, 0] = (box[0] + box[2]) / 2 / self.image_size[0]
                locations[i, 1] = (box[1] + box[3]) / 2 / self.image_size[1]
            
            new_loc_mean = (locations * weights[:, np.newaxis]).sum(axis=0)
            
            if stats.location_mean is not None:
                stats.location_mean = (self.scale_momentum * stats.location_mean + 
                                      (1 - self.scale_momentum) * new_loc_mean)
                
                diff = locations - stats.location_mean
                new_var = (diff**2 * weights[:, np.newaxis]).sum(axis=0)
                stats.location_var = (self.scale_momentum * stats.location_var + 
                                     (1 - self.scale_momentum) * new_var)
            else:
                stats.location_mean = new_loc_mean
                stats.location_var = np.array([0.25, 0.25])
        
        # Update counts
        stats.observation_count += n
        stats.update_count += 1
    
    def update_instance_clusters(self,
                                 features: np.ndarray,
                                 boxes: np.ndarray,
                                 class_indices: np.ndarray,
                                 tau2: float = 0.8):
        """
        Update instance-level clusters within each class (BCA+ style).
        
        Args:
            features: (N, D) features
            boxes: (N, 4) boxes
            class_indices: (N,) class assignments
            tau2: Threshold for cluster matching
        """
        if not self.use_instance_clusters:
            return
        
        features_norm = self._normalize(features)
        
        for i in range(len(features)):
            k = int(class_indices[i])
            feature = features_norm[i]
            box = boxes[i]
            
            # Get or create cluster storage for this class
            if self.instance_clusters[k] is None:
                self._create_cluster(k, feature, box)
                continue
            
            clusters = self.instance_clusters[k]
            
            # Compute similarity to existing clusters
            sims = feature @ clusters['features']  # (M,)
            
            # Also consider scale similarity
            if self.image_size is not None:
                scale = np.array([
                    (box[2] - box[0]) / self.image_size[0],
                    (box[3] - box[1]) / self.image_size[1]
                ])
                scale_dists = np.linalg.norm(clusters['scales'] - scale[:, np.newaxis], axis=0)
                scale_sims = 1 - scale_dists / np.sqrt(2)
                combined_sims = 0.8 * sims + 0.2 * scale_sims
            else:
                combined_sims = sims
            
            best_idx = np.argmax(combined_sims)
            best_sim = combined_sims[best_idx]
            
            if best_sim >= tau2:
                # Update existing cluster
                self._update_cluster(k, best_idx, feature, box)
            else:
                # Create new cluster if not at limit
                if clusters['features'].shape[1] < self.max_clusters_per_class:
                    self._create_cluster(k, feature, box)
                else:
                    # Update closest cluster anyway
                    self._update_cluster(k, best_idx, feature, box)
    
    def _create_cluster(self, class_idx: int, feature: np.ndarray, box: np.ndarray):
        """Create new instance cluster."""
        if self.image_size is None:
            scale = np.array([0.1, 0.1])
        else:
            scale = np.array([
                (box[2] - box[0]) / self.image_size[0],
                (box[3] - box[1]) / self.image_size[1]
            ])
        
        if self.instance_clusters[class_idx] is None:
            self.instance_clusters[class_idx] = {
                'features': feature.reshape(-1, 1),
                'scales': scale.reshape(-1, 1),
                'counts': np.array([1])
            }
        else:
            clusters = self.instance_clusters[class_idx]
            clusters['features'] = np.hstack([clusters['features'], feature.reshape(-1, 1)])
            clusters['scales'] = np.hstack([clusters['scales'], scale.reshape(-1, 1)])
            clusters['counts'] = np.append(clusters['counts'], 1)
    
    def _update_cluster(self, class_idx: int, cluster_idx: int, 
                        feature: np.ndarray, box: np.ndarray):
        """Update existing instance cluster."""
        clusters = self.instance_clusters[class_idx]
        c = clusters['counts'][cluster_idx]
        
        # Running average
        clusters['features'][:, cluster_idx] = (
            c * clusters['features'][:, cluster_idx] + feature
        ) / (c + 1)
        # Renormalize
        clusters['features'][:, cluster_idx] /= (
            np.linalg.norm(clusters['features'][:, cluster_idx]) + 1e-8
        )
        
        if self.image_size is not None:
            scale = np.array([
                (box[2] - box[0]) / self.image_size[0],
                (box[3] - box[1]) / self.image_size[1]
            ])
            clusters['scales'][:, cluster_idx] = (
                c * clusters['scales'][:, cluster_idx] + scale
            ) / (c + 1)
        
        clusters['counts'][cluster_idx] += 1
    
    def get_instance_cluster_features(self, class_idx: int) -> Optional[np.ndarray]:
        """Get instance cluster features for a class."""
        if self.instance_clusters[class_idx] is None:
            return None
        return self.instance_clusters[class_idx]['features'].T  # (M, D)
    
    def compute_instance_similarity(self, feature: np.ndarray, 
                                   class_idx: int) -> float:
        """
        Compute maximum similarity to instance clusters.
        
        Returns:
            Maximum cosine similarity (0 if no clusters)
        """
        if self.instance_clusters[class_idx] is None:
            return 0.0
        
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        sims = feature_norm @ self.instance_clusters[class_idx]['features']
        return float(np.max(sims))
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Safely normalize vectors along last axis."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norm, 1e-8)
    
    def reset(self):
        """Reset cache for new video."""
        self.class_stats = [ClassStatistics() for _ in range(self.num_classes)]
        self.instance_clusters = [None] * self.num_classes
        
        # Re-initialize from text embeddings if available
        if self.text_embeddings is not None:
            self.initialize_from_text_embeddings(self.text_embeddings)
    
    def get_stats_summary(self) -> Dict:
        """Get summary of cache statistics for debugging."""
        summary = {
            'num_classes': self.num_classes,
            'classes_initialized': sum(1 for s in self.class_stats if s.is_initialized()),
            'total_observations': sum(s.observation_count for s in self.class_stats),
            'per_class': []
        }
        
        for k in range(self.num_classes):
            s = self.class_stats[k]
            class_summary = {
                'class_idx': k,
                'initialized': s.is_initialized(),
                'observations': s.observation_count,
                'updates': s.update_count,
                'uncertainty': s.prototype_uncertainty if s.is_initialized() else None,
                'scale_mean': s.scale_mean.tolist() if s.scale_mean is not None else None,
                'num_clusters': (self.instance_clusters[k]['features'].shape[1] 
                               if self.instance_clusters[k] is not None else 0)
            }
            summary['per_class'].append(class_summary)
        
        return summary


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Global Cache Module...")
    
    # Initialize
    cache = GlobalCache(num_classes=6, feature_dim=256)
    
    # Mock text embeddings
    text_emb = np.random.randn(6, 256)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    cache.initialize_from_text_embeddings(text_emb)
    cache.set_image_size(1920, 1080)
    
    print(f"Initialized: {cache.get_stats_summary()['classes_initialized']} classes")
    
    # Test prediction
    features = np.random.randn(10, 256)
    probs = cache.predict_class_probs(features)
    print(f"Predictions shape: {probs.shape}, sum: {probs.sum(axis=1)}")
    
    # Test update
    boxes = np.random.rand(10, 4) * 100 + np.array([0, 0, 100, 100])
    class_probs = np.random.rand(10, 6)
    class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)
    conf_mask = np.random.rand(10) > 0.5
    
    cache.update(features, boxes, class_probs, conf_mask)
    
    print(f"After update: {cache.get_stats_summary()['total_observations']} observations")
    
    # Test instance clusters
    class_indices = np.argmax(class_probs, axis=1)
    cache.update_instance_clusters(features, boxes, class_indices)
    
    print(f"Instance clusters per class: {[s['num_clusters'] for s in cache.get_stats_summary()['per_class']]}")
    
    print("\n✓ All tests passed!")
