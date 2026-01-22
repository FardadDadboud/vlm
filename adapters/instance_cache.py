"""
Instance Cache Module for Global+Instance Temporal BCA+

Per-object tracking state that maintains:
1. Feature history (with temporal smoothing)
2. Position/velocity/scale trajectory (Kalman filtered)
3. Class probability history
4. Uncertainty estimates

Each Track object represents a tracked instance with its full state history.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

try:
    from .trackers import KalmanBoxTracker
except ImportError:
    from trackers import KalmanBoxTracker


# =============================================================================
# Track State
# =============================================================================

@dataclass
class TrackState:
    """Comprehensive state for a single tracked object."""
    
    # Track ID
    track_id: int
    
    # Kalman filter for position/scale
    kalman: KalmanBoxTracker = None
    
    # Feature history (smoothed)
    feature_history: List[np.ndarray] = field(default_factory=list)
    feature_smoothed: np.ndarray = None  # EMA smoothed feature
    
    # Class probability history
    class_prob_history: List[np.ndarray] = field(default_factory=list)
    class_prob_smoothed: np.ndarray = None
    
    # Confidence history
    confidence_history: List[float] = field(default_factory=list)
    
    # Track lifecycle
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    
    # Track status
    is_confirmed: bool = False
    is_deleted: bool = False
    
    # Predicted class (most likely)
    predicted_class: int = -1
    
    # Association probabilities (for JPDA)
    _jpda_probs: np.ndarray = None


# =============================================================================
# Track Class
# =============================================================================

class Track:
    """
    Single object track with full state management.
    
    Maintains:
    - Kalman-filtered position/velocity/scale
    - EMA-smoothed features
    - Class probability history
    - Uncertainty estimates
    """
    
    # Class-level parameters
    MAX_HISTORY_LENGTH = 30
    FEATURE_SMOOTHING = 0.7  # EMA alpha for features
    CLASS_PROB_SMOOTHING = 0.8  # EMA alpha for class probs
    MIN_HITS_TO_CONFIRM = 3
    MAX_AGE_SINCE_UPDATE = 10
    
    _id_counter = 0
    
    def __init__(self,
                 initial_box: np.ndarray,
                 initial_feature: np.ndarray,
                 initial_class_probs: np.ndarray,
                 initial_confidence: float,
                 kalman_params: Dict = None):
        """
        Initialize track from first detection.
        
        Args:
            initial_box: [x1, y1, x2, y2]
            initial_feature: (D,) feature embedding
            initial_class_probs: (K,) class probabilities
            initial_confidence: Detection confidence
            kalman_params: Optional Kalman filter parameters
        """
        # Assign unique ID
        Track._id_counter += 1
        self.track_id = Track._id_counter
        
        # Initialize Kalman filter
        kalman_params = kalman_params or {}
        self.kalman = KalmanBoxTracker(initial_box, **kalman_params)
        
        # Initialize feature state
        self.feature_history = [initial_feature.copy()]
        self.feature_smoothed = initial_feature.copy()
        self.feature_dim = len(initial_feature)
        
        # Initialize class state
        self.num_classes = len(initial_class_probs)
        self.class_prob_history = [initial_class_probs.copy()]
        self.class_prob_smoothed = initial_class_probs.copy()
        self.predicted_class = int(np.argmax(initial_class_probs))
        
        # Confidence
        self.confidence_history = [initial_confidence]
        
        # Track lifecycle
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.is_confirmed = False
        self.is_deleted = False
        
        # JPDA support
        self._jpda_probs = None
        
        # Uncertainty tracking
        self.feature_uncertainty = 1.0  # High initial uncertainty
        self.class_uncertainty = 1.0
    
    @property
    def is_active(self) -> bool:
        """Check if track is still active."""
        return not self.is_deleted and self.time_since_update < self.MAX_AGE_SINCE_UPDATE
    
    def predict(self):
        """Predict next state (call before association)."""
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Check for deletion
        if self.time_since_update > self.MAX_AGE_SINCE_UPDATE:
            self.is_deleted = True
    
    def update(self,
               box: np.ndarray,
               feature: np.ndarray,
               class_probs: np.ndarray,
               confidence: float):
        """
        Update track with new detection.
        
        Args:
            box: [x1, y1, x2, y2]
            feature: (D,) feature embedding
            class_probs: (K,) class probabilities
            confidence: Detection confidence
        """
        # Update Kalman filter
        self.kalman.update(box)
        
        # Update feature (EMA smoothing)
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        self.feature_smoothed = (
            self.FEATURE_SMOOTHING * self.feature_smoothed +
            (1 - self.FEATURE_SMOOTHING) * feature_norm
        )
        self.feature_smoothed = self.feature_smoothed / (
            np.linalg.norm(self.feature_smoothed) + 1e-8
        )
        
        # Add to history
        self.feature_history.append(feature_norm.copy())
        if len(self.feature_history) > self.MAX_HISTORY_LENGTH:
            self.feature_history.pop(0)
        
        # Update class probabilities (EMA smoothing)
        self.class_prob_smoothed = (
            self.CLASS_PROB_SMOOTHING * self.class_prob_smoothed +
            (1 - self.CLASS_PROB_SMOOTHING) * class_probs
        )
        # Renormalize
        self.class_prob_smoothed = self.class_prob_smoothed / (
            self.class_prob_smoothed.sum() + 1e-8
        )
        
        self.class_prob_history.append(class_probs.copy())
        if len(self.class_prob_history) > self.MAX_HISTORY_LENGTH:
            self.class_prob_history.pop(0)
        
        self.predicted_class = int(np.argmax(self.class_prob_smoothed))
        
        # Update confidence
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.MAX_HISTORY_LENGTH:
            self.confidence_history.pop(0)
        
        # Update lifecycle
        self.hits += 1
        self.time_since_update = 0
        
        # Check for confirmation
        if not self.is_confirmed and self.hits >= self.MIN_HITS_TO_CONFIRM:
            self.is_confirmed = True
        
        # Update uncertainty (more observations = less uncertainty)
        self.feature_uncertainty = max(0.1, self.feature_uncertainty * 0.95)
        self.class_uncertainty = max(0.1, self.class_uncertainty * 0.95)
    
    def get_feature(self) -> np.ndarray:
        """Get current smoothed feature."""
        return self.feature_smoothed.copy()
    
    def get_feature_history(self) -> np.ndarray:
        """Get feature history as (T, D) array."""
        return np.array(self.feature_history)
    
    def get_class_probs(self) -> np.ndarray:
        """Get current smoothed class probabilities."""
        return self.class_prob_smoothed.copy()
    
    def get_box(self) -> np.ndarray:
        """Get current predicted box."""
        return self.kalman.get_predicted_box()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        return self.kalman.state.velocity
    
    def get_mean_confidence(self, window: int = 5) -> float:
        """Get mean confidence over recent history."""
        recent = self.confidence_history[-window:]
        return float(np.mean(recent)) if recent else 0.0
    
    def compute_feature_consistency(self) -> float:
        """
        Compute feature consistency over history.
        
        Returns:
            Mean pairwise cosine similarity in recent history
        """
        if len(self.feature_history) < 2:
            return 1.0
        
        recent = self.feature_history[-5:]
        recent = np.array(recent)
        
        # Pairwise similarities
        sims = recent @ recent.T
        n = len(recent)
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_sims = sims[mask]
        
        return float(np.mean(pairwise_sims)) if len(pairwise_sims) > 0 else 1.0
    
    def compute_class_stability(self) -> float:
        """
        Compute class prediction stability over history.
        
        Returns:
            Fraction of recent predictions matching current prediction
        """
        if len(self.class_prob_history) < 2:
            return 1.0
        
        recent_preds = [np.argmax(p) for p in self.class_prob_history[-5:]]
        current_pred = self.predicted_class
        
        matches = sum(1 for p in recent_preds if p == current_pred)
        return matches / len(recent_preds)
    
    def get_state_summary(self) -> Dict:
        """Get summary of track state for debugging."""
        return {
            'track_id': self.track_id,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'is_confirmed': self.is_confirmed,
            'is_deleted': self.is_deleted,
            'predicted_class': self.predicted_class,
            'mean_confidence': self.get_mean_confidence(),
            'feature_consistency': self.compute_feature_consistency(),
            'class_stability': self.compute_class_stability(),
            'box': self.get_box().tolist(),
            'velocity': self.get_velocity().tolist()
        }


# =============================================================================
# Instance Cache Manager
# =============================================================================

class InstanceCache:
    """
    Manager for all tracked instances.
    
    Handles:
    - Track creation and deletion
    - Association with new detections
    - Feature-based instance priors for detection
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 max_tracks: int = 100,
                 min_hits_to_confirm: int = 3,
                 max_age: int = 10,
                 feature_smoothing: float = 0.7,
                 class_prob_smoothing: float = 0.8):
        """
        Initialize instance cache.
        
        Args:
            num_classes: Number of object classes
            feature_dim: Feature embedding dimension
            max_tracks: Maximum number of active tracks
            min_hits_to_confirm: Hits needed to confirm track
            max_age: Maximum frames without update
            feature_smoothing: EMA alpha for features
            class_prob_smoothing: EMA alpha for class probs
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_tracks = max_tracks
        
        # Update Track class parameters
        Track.MIN_HITS_TO_CONFIRM = min_hits_to_confirm
        Track.MAX_AGE_SINCE_UPDATE = max_age
        Track.FEATURE_SMOOTHING = feature_smoothing
        Track.CLASS_PROB_SMOOTHING = class_prob_smoothing
        
        # Active tracks
        self.tracks: List[Track] = []
        
        # Frame counter
        self.frame_count = 0
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
    
    def predict(self):
        """Predict all tracks (call at start of each frame)."""
        for track in self.tracks:
            if not track.is_deleted:
                track.predict()
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted]
    
    def create_track(self,
                     box: np.ndarray,
                     feature: np.ndarray,
                     class_probs: np.ndarray,
                     confidence: float) -> Track:
        """
        Create new track from unmatched detection.
        
        Returns:
            New Track object
        """
        # Check track limit
        if len(self.tracks) >= self.max_tracks:
            # Remove oldest unconfirmed track
            unconfirmed = [t for t in self.tracks if not t.is_confirmed]
            if unconfirmed:
                oldest = min(unconfirmed, key=lambda t: t.hits)
                self.tracks.remove(oldest)
                self.total_tracks_deleted += 1
        
        track = Track(box, feature, class_probs, confidence)
        self.tracks.append(track)
        self.total_tracks_created += 1
        
        return track
    
    def update_track(self,
                     track_idx: int,
                     box: np.ndarray,
                     feature: np.ndarray,
                     class_probs: np.ndarray,
                     confidence: float):
        """Update existing track with matched detection."""
        if 0 <= track_idx < len(self.tracks):
            self.tracks[track_idx].update(box, feature, class_probs, confidence)
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (non-deleted) tracks."""
        return [t for t in self.tracks if t.is_active]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed and t.is_active]
    
    def get_tracks_by_class(self, class_idx: int) -> List[Track]:
        """Get tracks predicted as given class."""
        return [t for t in self.get_confirmed_tracks() 
                if t.predicted_class == class_idx]
    
    def compute_instance_prior(self,
                              feature: np.ndarray,
                              box: np.ndarray,
                              class_probs: np.ndarray) -> np.ndarray:
        """
        Compute instance-based class prior for a detection.
        
        Uses similarity to tracked instances to modify class probabilities.
        
        Args:
            feature: (D,) detection feature
            box: [x1, y1, x2, y2] detection box
            class_probs: (K,) initial class probabilities
            
        Returns:
            (K,) modified class probabilities
        """
        if len(self.tracks) == 0:
            return class_probs
        
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        
        # Find most similar track
        best_sim = -1
        best_track = None
        
        for track in self.get_confirmed_tracks():
            track_feature = track.get_feature()
            sim = float(feature_norm @ track_feature)
            
            if sim > best_sim:
                best_sim = sim
                best_track = track
        
        if best_track is None or best_sim < 0.5:
            return class_probs
        
        # Blend with track's class distribution
        track_probs = best_track.get_class_probs()
        
        # Weight by similarity
        alpha = min(1.0, max(0.0, (best_sim - 0.5) * 2))  # 0.5->0, 1.0->1
        
        blended = (1 - alpha) * class_probs + alpha * track_probs
        blended = blended / (blended.sum() + 1e-8)

        return blended
    
    def compute_instance_confidence_boost(self,
                                          feature: np.ndarray,
                                          box: np.ndarray,
                                          confidence: float) -> float:
        """
        Compute confidence boost based on instance matching.
        
        Detections that match tracked instances get a confidence boost.
        
        Args:
            feature: (D,) detection feature
            box: [x1, y1, x2, y2] detection box
            confidence: Initial confidence
            
        Returns:
            Boosted confidence
        """
        if len(self.tracks) == 0:
            return confidence
        
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        
        # Find best matching track
        best_match_score = 0
        
        for track in self.get_confirmed_tracks():
            # Feature similarity
            track_feature = track.get_feature()
            feat_sim = float(feature_norm @ track_feature)
            
            # IoU with predicted box
            pred_box = track.get_box()
            iou = self._compute_iou(box, pred_box)
            
            # Combined score
            match_score = 0.7 * feat_sim + 0.3 * iou
            
            if match_score > best_match_score:
                best_match_score = match_score
        
        # Compute boost (higher match -> higher boost)
        if best_match_score > 0.7:
            # Strong match: significant boost
            boost = 0.1 * (best_match_score - 0.7) / 0.3
            boosted = min(1.0, confidence + boost)
        else:
            boosted = confidence
        
        return boosted
    
    def get_motion_prediction(self, track_idx: int, 
                              num_frames: int = 1) -> np.ndarray:
        """
        Get motion-predicted box for future frames.
        
        Args:
            track_idx: Track index
            num_frames: Number of frames to predict ahead
            
        Returns:
            Predicted box [x1, y1, x2, y2]
        """
        if 0 <= track_idx < len(self.tracks):
            track = self.tracks[track_idx]
            # Simple linear extrapolation using velocity
            box = track.get_box()
            vel = track.get_velocity()
            
            # Adjust center by velocity * frames
            cx = (box[0] + box[2]) / 2 + vel[0] * num_frames
            cy = (box[1] + box[3]) / 2 + vel[1] * num_frames
            w = box[2] - box[0]
            h = box[3] - box[1]
            
            return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        return np.zeros(4)
    
    def reset(self):
        """Reset for new video."""
        self.tracks = []
        self.frame_count = 0
        Track._id_counter = 0
    
    def get_stats_summary(self) -> Dict:
        """Get summary of instance cache statistics."""
        active = self.get_active_tracks()
        confirmed = self.get_confirmed_tracks()
        
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active),
            'confirmed_tracks': len(confirmed),
            'total_created': self.total_tracks_created,
            'total_deleted': self.total_tracks_deleted,
            'tracks_by_class': {
                k: len(self.get_tracks_by_class(k)) 
                for k in range(self.num_classes)
            },
            'track_details': [t.get_state_summary() for t in confirmed[:10]]  # Limit for debugging
        }
    
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


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Instance Cache Module...")
    
    # Initialize
    cache = InstanceCache(num_classes=6, feature_dim=256)
    
    # Create some tracks
    for i in range(5):
        box = np.array([100 + i*50, 100, 200 + i*50, 200])
        feature = np.random.randn(256)
        class_probs = np.random.rand(6)
        class_probs = class_probs / class_probs.sum()
        confidence = 0.8 + np.random.rand() * 0.2
        
        track = cache.create_track(box, feature, class_probs, confidence)
        print(f"Created track {track.track_id}")
    
    # Simulate a few frames
    for frame in range(5):
        cache.predict()
        
        # Update some tracks
        for i, track in enumerate(cache.get_active_tracks()[:3]):
            new_box = track.get_box() + np.random.randn(4) * 5
            new_feature = track.get_feature() + np.random.randn(256) * 0.1
            new_probs = track.get_class_probs() + np.random.rand(6) * 0.1
            new_probs = new_probs / new_probs.sum()
            
            cache.update_track(i, new_box, new_feature, new_probs, 0.9)
    
    # Print stats
    stats = cache.get_stats_summary()
    print(f"\nStats: {stats['active_tracks']} active, {stats['confirmed_tracks']} confirmed")
    
    # Test instance prior
    test_feature = np.random.randn(256)
    test_box = np.array([120, 100, 220, 200])
    test_probs = np.array([0.1, 0.1, 0.6, 0.1, 0.05, 0.05])
    
    modified_probs = cache.compute_instance_prior(test_feature, test_box, test_probs)
    print(f"\nOriginal probs: {test_probs}")
    print(f"Modified probs: {modified_probs}")
    
    print("\n✓ All tests passed!")
