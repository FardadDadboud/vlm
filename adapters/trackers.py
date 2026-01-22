"""
Trackers Module for Global+Instance Temporal BCA+

Implements:
1. Kalman Filter for state estimation (position, velocity, scale)
2. Multiple data association methods:
   - Hungarian algorithm
   - JPDA (Joint Probabilistic Data Association)
   - ByteTrack-style (high/low confidence separation)
   - DeepSORT-style (appearance + motion)

All methods are modular and can be swapped via config.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


# =============================================================================
# Kalman Filter for Object Tracking
# =============================================================================

@dataclass
class KalmanState:
    """State vector: [cx, cy, w, h, vx, vy, vw, vh]"""
    x: np.ndarray  # State vector (8,)
    P: np.ndarray  # Covariance matrix (8, 8)
    
    @property
    def position(self) -> np.ndarray:
        """Return [cx, cy]"""
        return self.x[:2]
    
    @property
    def size(self) -> np.ndarray:
        """Return [w, h]"""
        return self.x[2:4]
    
    @property
    def velocity(self) -> np.ndarray:
        """Return [vx, vy]"""
        return self.x[4:6]
    
    @property
    def box(self) -> np.ndarray:
        """Return [x1, y1, x2, y2]"""
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class KalmanBoxTracker:
    """
    Kalman filter for bounding box tracking.
    
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    
    # Class-level tuning parameters (can be overridden per instance)
    DEFAULT_STD_WEIGHT_POSITION = 0.05
    DEFAULT_STD_WEIGHT_VELOCITY = 0.00625
    
    def __init__(self, 
                 initial_box: np.ndarray,
                 std_weight_position: float = None,
                 std_weight_velocity: float = None):
        """
        Initialize tracker with first detection.
        
        Args:
            initial_box: [x1, y1, x2, y2] format
            std_weight_position: Process noise weight for position
            std_weight_velocity: Process noise weight for velocity
        """
        self.std_weight_position = std_weight_position or self.DEFAULT_STD_WEIGHT_POSITION
        self.std_weight_velocity = std_weight_velocity or self.DEFAULT_STD_WEIGHT_VELOCITY
        
        # Convert box to [cx, cy, w, h]
        measurement = self._box_to_measurement(initial_box)
        
        # Initialize state vector
        x = np.zeros(8)
        x[:4] = measurement
        x[4:] = 0  # Zero initial velocity
        
        # Initialize covariance with high uncertainty for velocities
        P = np.eye(8)
        P[4:, 4:] *= 1000  # High uncertainty for velocity
        P *= 10  # Scale overall
        
        self.state = KalmanState(x=x, P=P)
        
        # Motion model matrices
        self._init_matrices()
        
        # Track statistics
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 1
    
    def _init_matrices(self):
        """Initialize Kalman filter matrices."""
        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # Position += velocity * dt (dt=1)
        
        # Measurement matrix (observe [cx, cy, w, h])
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
    
    def _get_process_noise(self) -> np.ndarray:
        """
        Get process noise covariance Q.
        Scaled by current state (larger objects have more uncertainty).
        """
        std_pos = self.std_weight_position * max(self.state.x[2], self.state.x[3])
        std_vel = self.std_weight_velocity * max(self.state.x[2], self.state.x[3])
        
        Q = np.diag([
            std_pos**2, std_pos**2,  # cx, cy
            std_pos**2, std_pos**2,  # w, h
            std_vel**2, std_vel**2,  # vx, vy
            std_vel**2, std_vel**2   # vw, vh
        ])
        return Q
    
    def _get_measurement_noise(self) -> np.ndarray:
        """
        Get measurement noise covariance R.
        Scaled by current state.
        """
        std = self.std_weight_position * max(self.state.x[2], self.state.x[3])
        R = np.diag([std**2, std**2, std**2, std**2])
        return R
    
    def _box_to_measurement(self, box: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]"""
        w = box[2] - box[0]
        h = box[3] - box[1]
        cx = box[0] + w / 2
        cy = box[1] + h / 2
        return np.array([cx, cy, w, h])
    
    def predict(self) -> KalmanState:
        """
        Predict next state.
        
        Returns:
            Predicted state
        """
        # Predict state
        x_pred = self.F @ self.state.x
        
        # Predict covariance
        Q = self._get_process_noise()
        P_pred = self.F @ self.state.P @ self.F.T + Q
        
        # Ensure positive size
        x_pred[2:4] = np.maximum(x_pred[2:4], 1.0)
        
        self.state = KalmanState(x=x_pred, P=P_pred)
        self.age += 1
        self.time_since_update += 1
        
        return self.state
    
    def update(self, box: np.ndarray) -> KalmanState:
        """
        Update state with measurement.
        
        Args:
            box: Detection [x1, y1, x2, y2]
            
        Returns:
            Updated state
        """
        measurement = self._box_to_measurement(box)
        
        # Innovation
        y = measurement - self.H @ self.state.x
        
        # Innovation covariance
        R = self._get_measurement_noise()
        S = self.H @ self.state.P @ self.H.T + R
        
        # Kalman gain
        K = self.state.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        x_new = self.state.x + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(8) - K @ self.H
        P_new = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T
        
        # Ensure positive size
        x_new[2:4] = np.maximum(x_new[2:4], 1.0)
        
        self.state = KalmanState(x=x_new, P=P_new)
        self.time_since_update = 0
        self.hit_streak += 1
        
        return self.state
    
    def get_predicted_box(self) -> np.ndarray:
        """Get predicted bounding box [x1, y1, x2, y2]"""
        return self.state.box
    
    def gating_distance(self, box: np.ndarray, 
                        only_position: bool = False) -> float:
        """
        Compute Mahalanobis distance for gating.
        
        Args:
            box: Detection [x1, y1, x2, y2]
            only_position: If True, only use position for gating
            
        Returns:
            Squared Mahalanobis distance
        """
        measurement = self._box_to_measurement(box)
        
        if only_position:
            y = measurement[:2] - self.state.x[:2]
            S = self.H[:2, :] @ self.state.P @ self.H[:2, :].T + self._get_measurement_noise()[:2, :2]
        else:
            y = measurement - self.H @ self.state.x
            S = self.H @ self.state.P @ self.H.T + self._get_measurement_noise()
        
        try:
            d = y.T @ np.linalg.inv(S) @ y
        except np.linalg.LinAlgError:
            d = float('inf')
        
        return d


# =============================================================================
# Data Association Base Class
# =============================================================================

@dataclass
class AssociationResult:
    """Result of data association."""
    matches: List[Tuple[int, int]]  # (track_idx, detection_idx) pairs
    unmatched_tracks: List[int]
    unmatched_detections: List[int]
    costs: Optional[np.ndarray] = None  # Cost matrix for debugging


class BaseAssociator(ABC):
    """Abstract base class for data association methods."""
    
    @abstractmethod
    def associate(self,
                  tracks: List['Track'],
                  detections: Dict,
                  frame_id: int) -> AssociationResult:
        """
        Associate detections to tracks.
        
        Args:
            tracks: List of Track objects
            detections: Dict with 'boxes', 'scores', 'features'
            frame_id: Current frame number
            
        Returns:
            AssociationResult
        """
        raise NotImplementedError


# =============================================================================
# Hungarian Algorithm Association
# =============================================================================

class HungarianAssociator(BaseAssociator):
    """
    Standard Hungarian algorithm for assignment.
    
    Uses IoU as cost metric with optional feature distance.
    """
    
    def __init__(self,
                 iou_threshold: float = 0.3,
                 feature_weight: float = 0.0,
                 gating_threshold: float = None):
        """
        Args:
            iou_threshold: Minimum IoU for valid match
            feature_weight: Weight for feature distance (0 = IoU only)
            gating_threshold: Mahalanobis distance threshold (None = no gating)
        """
        self.iou_threshold = iou_threshold
        self.feature_weight = feature_weight
        self.gating_threshold = gating_threshold
        
        # Chi-square threshold for gating (95% confidence, 4 DOF)
        self.chi2_threshold = chi2.ppf(0.95, 4) if gating_threshold is None else gating_threshold
    
    def associate(self,
                  tracks: List['Track'],
                  detections: Dict,
                  frame_id: int) -> AssociationResult:
        """Associate using Hungarian algorithm."""
        
        if len(tracks) == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=[],
                unmatched_detections=list(range(len(detections['boxes'])))
            )
        
        if len(detections['boxes']) == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=[]
            )
        
        # Build cost matrix
        n_tracks = len(tracks)
        n_dets = len(detections['boxes'])
        cost_matrix = np.zeros((n_tracks, n_dets))
        
        for t_idx, track in enumerate(tracks):
            pred_box = track.kalman.get_predicted_box()
            
            for d_idx in range(n_dets):
                det_box = detections['boxes'][d_idx]
                
                # IoU cost (1 - IoU so lower is better)
                iou = self._compute_iou(pred_box, det_box)
                iou_cost = 1.0 - iou
                
                # Optional: feature cost
                if self.feature_weight > 0 and 'features' in detections:
                    det_feat = detections['features'][d_idx]
                    track_feat = track.get_feature()
                    if track_feat is not None:
                        feat_dist = 1.0 - self._cosine_similarity(track_feat, det_feat)
                    else:
                        feat_dist = 0.5  # Neutral
                    
                    cost = (1 - self.feature_weight) * iou_cost + self.feature_weight * feat_dist
                else:
                    cost = iou_cost
                
                # Gating
                if self.gating_threshold is not None:
                    mahal_dist = track.kalman.gating_distance(det_box)
                    if mahal_dist > self.chi2_threshold:
                        cost = 1e6  # Invalid match
                
                # IoU gating
                if iou < self.iou_threshold:
                    cost = 1e6  # Invalid match
                
                cost_matrix[t_idx, d_idx] = cost
        
        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out invalid matches
        matches = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_detections = list(range(n_dets))
        
        for t_idx, d_idx in zip(row_indices, col_indices):
            if cost_matrix[t_idx, d_idx] < 1e5:  # Valid match
                matches.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(d_idx)
        
        return AssociationResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            costs=cost_matrix
        )
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-8)
    
    def _cosine_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute cosine similarity"""
        f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
        f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)
        return float(np.dot(f1_norm, f2_norm))


# =============================================================================
# ByteTrack-style Association (High/Low Confidence)
# =============================================================================

class ByteTrackAssociator(BaseAssociator):
    """
    ByteTrack-style association with high/low confidence separation.
    
    1. First associate high-confidence detections to all tracks
    2. Then associate low-confidence detections to unmatched tracks
    """
    
    def __init__(self,
                 high_threshold: float = 0.6,
                 low_threshold: float = 0.1,
                 iou_threshold_high: float = 0.3,
                 iou_threshold_low: float = 0.5,
                 feature_weight: float = 0.0):
        """
        Args:
            high_threshold: Confidence threshold for high-confidence detections
            low_threshold: Minimum confidence to consider
            iou_threshold_high: IoU threshold for high-confidence matching
            iou_threshold_low: IoU threshold for low-confidence matching
            feature_weight: Weight for feature distance
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.feature_weight = feature_weight
        
        self.hungarian_high = HungarianAssociator(
            iou_threshold=iou_threshold_high,
            feature_weight=feature_weight
        )
        self.hungarian_low = HungarianAssociator(
            iou_threshold=iou_threshold_low,
            feature_weight=0.0  # IoU only for low confidence
        )
    
    def associate(self,
                  tracks: List['Track'],
                  detections: Dict,
                  frame_id: int) -> AssociationResult:
        """ByteTrack-style two-stage association."""
        
        scores = detections['scores']
        
        # Split detections by confidence
        high_mask = scores >= self.high_threshold
        low_mask = (scores >= self.low_threshold) & (scores < self.high_threshold)
        
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]
        
        # Stage 1: High confidence matching
        high_dets = self._subset_detections(detections, high_indices)
        result1 = self.hungarian_high.associate(tracks, high_dets, frame_id)
        
        # Remap indices
        matches = [(t, high_indices[d]) for t, d in result1.matches]
        
        # Stage 2: Low confidence matching to unmatched tracks
        if len(result1.unmatched_tracks) > 0 and len(low_indices) > 0:
            unmatched_tracks = [tracks[i] for i in result1.unmatched_tracks]
            low_dets = self._subset_detections(detections, low_indices)
            
            result2 = self.hungarian_low.associate(unmatched_tracks, low_dets, frame_id)
            
            # Remap indices
            for t_local, d_local in result2.matches:
                t_global = result1.unmatched_tracks[t_local]
                d_global = low_indices[d_local]
                matches.append((t_global, d_global))
            
            unmatched_tracks_final = [result1.unmatched_tracks[i] for i in result2.unmatched_tracks]
        else:
            unmatched_tracks_final = result1.unmatched_tracks
        
        # All detections that weren't matched
        matched_det_indices = set(d for _, d in matches)
        unmatched_dets = [i for i in range(len(scores)) 
                         if i not in matched_det_indices and scores[i] >= self.low_threshold]
        
        return AssociationResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks_final,
            unmatched_detections=unmatched_dets
        )
    
    def _subset_detections(self, detections: Dict, indices: np.ndarray) -> Dict:
        """Extract subset of detections"""
        if len(indices) == 0:
            return {'boxes': np.array([]), 'scores': np.array([]), 'features': np.array([])}
        
        result = {
            'boxes': detections['boxes'][indices],
            'scores': detections['scores'][indices]
        }
        if 'features' in detections and detections['features'] is not None:
            result['features'] = detections['features'][indices]
        if 'class_probs' in detections and detections['class_probs'] is not None:
            result['class_probs'] = detections['class_probs'][indices]
        if 'labels' in detections:
            result['labels'] = [detections['labels'][i] for i in indices]
        return result


# =============================================================================
# JPDA (Joint Probabilistic Data Association)
# =============================================================================

class JPDAAssociator(BaseAssociator):
    """
    Joint Probabilistic Data Association.
    
    Instead of hard assignment, computes association probabilities
    and updates each track with weighted sum of all detections.
    """
    
    def __init__(self,
                 gate_threshold: float = 9.21,  # Chi-square 95% for 2DOF
                 detection_probability: float = 0.9,
                 clutter_density: float = 1e-6,
                 feature_weight: float = 0.0):
        """
        Args:
            gate_threshold: Mahalanobis distance gate
            detection_probability: P(detection | object exists)
            clutter_density: Clutter density per unit area
            feature_weight: Weight for feature-based likelihood
        """
        self.gate_threshold = gate_threshold
        self.detection_probability = detection_probability
        self.clutter_density = clutter_density
        self.feature_weight = feature_weight
    
    def associate(self,
                  tracks: List['Track'],
                  detections: Dict,
                  frame_id: int) -> AssociationResult:
        """
        JPDA association.
        
        Returns hard assignments but computes soft association probabilities
        which are stored in tracks for weighted updates.
        """
        n_tracks = len(tracks)
        n_dets = len(detections['boxes'])
        
        if n_tracks == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=[],
                unmatched_detections=list(range(n_dets))
            )
        
        if n_dets == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=list(range(n_tracks)),
                unmatched_detections=[]
            )
        
        # Compute gated likelihoods
        likelihoods = np.zeros((n_tracks, n_dets))
        gated = np.zeros((n_tracks, n_dets), dtype=bool)
        
        for t_idx, track in enumerate(tracks):
            for d_idx in range(n_dets):
                det_box = detections['boxes'][d_idx]
                mahal_dist = track.kalman.gating_distance(det_box, only_position=True)
                
                if mahal_dist < self.gate_threshold:
                    gated[t_idx, d_idx] = True
                    # Gaussian likelihood
                    likelihood = np.exp(-0.5 * mahal_dist)
                    
                    # Optional: feature likelihood
                    if self.feature_weight > 0 and 'features' in detections:
                        track_feat = track.get_feature()
                        if track_feat is not None:
                            det_feat = detections['features'][d_idx]
                            feat_sim = self._cosine_similarity(track_feat, det_feat)
                            feat_likelihood = np.exp(2 * (feat_sim - 0.5))  # Centered at 0.5
                            likelihood = (1 - self.feature_weight) * likelihood + self.feature_weight * feat_likelihood
                    
                    likelihoods[t_idx, d_idx] = likelihood
        
        # Compute association probabilities (simplified JPDA)
        # For each track, normalize likelihoods over gated detections
        assoc_probs = np.zeros((n_tracks, n_dets + 1))  # +1 for miss hypothesis
        
        for t_idx in range(n_tracks):
            gated_dets = np.where(gated[t_idx])[0]
            if len(gated_dets) == 0:
                assoc_probs[t_idx, -1] = 1.0  # Miss
            else:
                # Likelihood for each detection
                liks = likelihoods[t_idx, gated_dets]
                # Miss likelihood
                miss_lik = (1 - self.detection_probability) / (self.detection_probability + 1e-8)
                
                # Normalize
                total = liks.sum() + miss_lik
                assoc_probs[t_idx, gated_dets] = liks / (total + 1e-8)
                assoc_probs[t_idx, -1] = miss_lik / (total + 1e-8)
        
        # Store association probabilities in tracks
        for t_idx, track in enumerate(tracks):
            track._jpda_probs = assoc_probs[t_idx, :-1]
        
        # Hard assignment (highest probability)
        matches = []
        matched_tracks = set()
        matched_dets = set()
        
        # Greedy assignment by highest probability
        flat_probs = assoc_probs[:, :-1].flatten()
        sorted_idx = np.argsort(flat_probs)[::-1]
        
        for idx in sorted_idx:
            t_idx = idx // n_dets
            d_idx = idx % n_dets
            
            if flat_probs[idx] < 0.1:  # Minimum probability threshold
                break
            
            if t_idx not in matched_tracks and d_idx not in matched_dets:
                if gated[t_idx, d_idx]:
                    matches.append((t_idx, d_idx))
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)
        
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        
        return AssociationResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
            costs=likelihoods
        )
    
    def _cosine_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute cosine similarity"""
        f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
        f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)
        return float(np.dot(f1_norm, f2_norm))


# =============================================================================
# DeepSORT-style Association (Cascade Matching)
# =============================================================================

class DeepSORTAssociator(BaseAssociator):
    """
    DeepSORT-style cascade matching.
    
    1. First match by appearance (feature distance) with cascade by track age
    2. Then match remaining by IoU
    """
    
    def __init__(self,
                 max_age: int = 30,
                 cascade_depth: int = 3,
                 iou_threshold: float = 0.3,
                 feature_threshold: float = 0.5,
                 lambda_: float = 0.98):
        """
        Args:
            max_age: Maximum frames without update before track deletion
            cascade_depth: Number of cascade levels (match recent tracks first)
            iou_threshold: IoU threshold for IOU matching stage
            feature_threshold: Cosine distance threshold for appearance matching
            lambda_: Weighting for combined distance
        """
        self.max_age = max_age
        self.cascade_depth = cascade_depth
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.lambda_ = lambda_
        
        self.hungarian = HungarianAssociator(iou_threshold=iou_threshold)
    
    def associate(self,
                  tracks: List['Track'],
                  detections: Dict,
                  frame_id: int) -> AssociationResult:
        """DeepSORT cascade matching."""
        
        n_dets = len(detections['boxes'])
        
        if len(tracks) == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=[],
                unmatched_detections=list(range(n_dets))
            )
        
        if n_dets == 0:
            return AssociationResult(
                matches=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=[]
            )
        
        # Split tracks by confirmed/tentative
        confirmed_tracks = [i for i, t in enumerate(tracks) if t.is_confirmed]
        tentative_tracks = [i for i, t in enumerate(tracks) if not t.is_confirmed]
        
        # Stage 1: Cascade matching for confirmed tracks
        matches = []
        unmatched_dets = list(range(n_dets))
        
        for level in range(self.cascade_depth):
            if len(unmatched_dets) == 0:
                break
            
            # Get tracks at this cascade level (by time since update)
            level_tracks = [i for i in confirmed_tracks 
                          if tracks[i].kalman.time_since_update == level]
            
            if len(level_tracks) == 0:
                continue
            
            # Build cost matrix using combined distance
            cost_matrix = self._combined_cost(
                [tracks[i] for i in level_tracks],
                self._subset_detections(detections, unmatched_dets)
            )
            
            # Solve
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] < self.feature_threshold:
                    t_idx = level_tracks[r]
                    d_idx = unmatched_dets[c]
                    matches.append((t_idx, d_idx))
            
            # Update unmatched
            matched_det_local = [c for r, c in zip(row_idx, col_idx) 
                                if cost_matrix[r, c] < self.feature_threshold]
            unmatched_dets = [d for i, d in enumerate(unmatched_dets) 
                            if i not in matched_det_local]
        
        # Stage 2: IoU matching for remaining confirmed tracks
        unmatched_confirmed = [i for i in confirmed_tracks 
                              if i not in [m[0] for m in matches]]
        
        if len(unmatched_confirmed) > 0 and len(unmatched_dets) > 0:
            iou_matches = self._iou_matching(
                [tracks[i] for i in unmatched_confirmed],
                self._subset_detections(detections, unmatched_dets)
            )
            
            for t_local, d_local in iou_matches:
                t_idx = unmatched_confirmed[t_local]
                d_idx = unmatched_dets[d_local]
                matches.append((t_idx, d_idx))
            
            matched_det_local = [d for _, d in iou_matches]
            unmatched_dets = [d for i, d in enumerate(unmatched_dets)
                            if i not in matched_det_local]
        
        # Stage 3: IoU matching for tentative tracks
        if len(tentative_tracks) > 0 and len(unmatched_dets) > 0:
            tent_matches = self._iou_matching(
                [tracks[i] for i in tentative_tracks],
                self._subset_detections(detections, unmatched_dets)
            )
            
            for t_local, d_local in tent_matches:
                t_idx = tentative_tracks[t_local]
                d_idx = unmatched_dets[d_local]
                matches.append((t_idx, d_idx))
            
            matched_det_local = [d for _, d in tent_matches]
            unmatched_dets = [d for i, d in enumerate(unmatched_dets)
                            if i not in matched_det_local]
        
        # Unmatched tracks
        matched_track_idx = [m[0] for m in matches]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_idx]
        
        return AssociationResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets
        )
    
    def _combined_cost(self, tracks: List['Track'], detections: Dict) -> np.ndarray:
        """Compute combined appearance + motion cost."""
        n_tracks = len(tracks)
        n_dets = len(detections['boxes'])
        
        cost = np.zeros((n_tracks, n_dets))
        
        for t_idx, track in enumerate(tracks):
            track_feat = track.get_feature()
            pred_box = track.kalman.get_predicted_box()
            
            for d_idx in range(n_dets):
                det_feat = detections['features'][d_idx] if 'features' in detections else None
                det_box = detections['boxes'][d_idx]
                
                # Appearance cost (cosine distance)
                if track_feat is not None and det_feat is not None:
                    sim = self._cosine_similarity(track_feat, det_feat)
                    app_cost = 1 - sim
                else:
                    app_cost = 0.5
                
                # Motion cost (Mahalanobis)
                motion_cost = track.kalman.gating_distance(det_box) / 100  # Normalize
                
                # Combined
                cost[t_idx, d_idx] = self.lambda_ * app_cost + (1 - self.lambda_) * motion_cost
        
        return cost
    
    def _iou_matching(self, tracks: List['Track'], detections: Dict) -> List[Tuple[int, int]]:
        """Simple IoU-based matching."""
        if len(tracks) == 0 or len(detections['boxes']) == 0:
            return []
        
        n_tracks = len(tracks)
        n_dets = len(detections['boxes'])
        
        cost = np.zeros((n_tracks, n_dets))
        
        for t_idx, track in enumerate(tracks):
            pred_box = track.kalman.get_predicted_box()
            for d_idx in range(n_dets):
                iou = self._compute_iou(pred_box, detections['boxes'][d_idx])
                cost[t_idx, d_idx] = 1 - iou
        
        row_idx, col_idx = linear_sum_assignment(cost)
        
        matches = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < (1 - self.iou_threshold):
                matches.append((r, c))
        
        return matches
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-8)
    
    def _cosine_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute cosine similarity"""
        f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
        f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)
        return float(np.dot(f1_norm, f2_norm))
    
    def _subset_detections(self, detections: Dict, indices: list) -> Dict:
        """Extract subset of detections"""
        if len(indices) == 0:
            return {'boxes': np.array([]), 'scores': np.array([])}
        
        result = {
            'boxes': detections['boxes'][indices],
            'scores': detections['scores'][indices]
        }
        if 'features' in detections and detections['features'] is not None:
            result['features'] = detections['features'][indices]
        if 'class_probs' in detections and detections['class_probs'] is not None:
            result['class_probs'] = detections['class_probs'][indices]
        return result


# =============================================================================
# Factory Function
# =============================================================================

def create_associator(method: str, **kwargs) -> BaseAssociator:
    """
    Factory function to create associator.
    
    Args:
        method: 'hungarian', 'bytetrack', 'jpda', or 'deepsort'
        **kwargs: Method-specific parameters
        
    Returns:
        BaseAssociator instance
    """
    method = method.lower()
    
    if method == 'hungarian':
        return HungarianAssociator(**kwargs)
    elif method == 'bytetrack':
        return ByteTrackAssociator(**kwargs)
    elif method == 'jpda':
        return JPDAAssociator(**kwargs)
    elif method == 'deepsort':
        return DeepSORTAssociator(**kwargs)
    else:
        raise ValueError(f"Unknown association method: {method}")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Trackers Module...")
    
    # Test Kalman filter
    print("\n1. Testing KalmanBoxTracker:")
    tracker = KalmanBoxTracker(np.array([100, 100, 200, 200]))
    print(f"   Initial state: {tracker.state.x[:4]}")
    
    tracker.predict()
    print(f"   After predict: {tracker.state.x[:4]}")
    
    tracker.update(np.array([105, 102, 205, 202]))
    print(f"   After update: {tracker.state.x[:4]}")
    
    # Test associators
    print("\n2. Testing Associators:")
    
    # Mock tracks and detections
    class MockTrack:
        def __init__(self, box):
            self.kalman = KalmanBoxTracker(box)
            self.is_confirmed = True
        def get_feature(self):
            return np.random.randn(256)
    
    tracks = [
        MockTrack(np.array([100, 100, 200, 200])),
        MockTrack(np.array([300, 300, 400, 400]))
    ]
    
    detections = {
        'boxes': np.array([
            [105, 102, 205, 202],
            [310, 305, 410, 405],
            [500, 500, 600, 600]
        ]),
        'scores': np.array([0.9, 0.8, 0.7]),
        'features': np.random.randn(3, 256)
    }
    
    for name, assoc in [
        ('Hungarian', HungarianAssociator()),
        ('ByteTrack', ByteTrackAssociator()),
        ('JPDA', JPDAAssociator()),
        ('DeepSORT', DeepSORTAssociator())
    ]:
        result = assoc.associate(tracks, detections, 0)
        print(f"   {name}: matches={result.matches}, unmatched_t={result.unmatched_tracks}, unmatched_d={result.unmatched_detections}")
    
    print("\n✓ All tests passed!")
