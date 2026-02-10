"""
Track Module for Global+Instance TTA

Single file containing ALL tracking components (matching __init__.py imports):
1. Kalman Filter - Bounding box tracking (KalmanBoxTracker, KalmanFilterConfig)
2. Data Association - Hungarian, IoU, ByteTrack, Combined (AssociationConfig, associate_*)
3. Per-Track STAD - FULL vMF and Gaussian implementations (TrackSTADvMF, TrackSTADGaussian)
4. Track/TrackManager - Track lifecycle management (Track, TrackManager, Detection)

Key Design Decision:
Per-track STAD solves the global π bias problem. At track level, π reflects the 
individual object's class history, not dataset statistics.

CRITICAL: Per-track STAD uses the EXACT SAME algorithm as global STAD (temporal_ssm_v2.py),
just operating on a single track's history instead of all detections.

The windowed EM update in TrackSTADvMF._soft_em_update() matches temporal_ssm_v2.py line-by-line:
- E-step: Compute responsibilities with temperature scaling
- M-step: Update rho, gamma with transition prior, EMA, and stability fixes
- pi update with Dirichlet prior and optional EMA

References:
- SORT: Simple Online and Realtime Tracking (Bewley et al., 2016)
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al., 2022)
- STAD: Temporal Test-Time Adaptation with State-Space Models (Schirmer et al., TMLR 2025)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Literal, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment
from scipy.special import ive  # Exponentially scaled Bessel


# =============================================================================
# Constants (matching temporal_ssm_v2.py)
# =============================================================================
DEFAULT_KAPPA_MAX = 500.0
DEFAULT_KAPPA_MIN = 1e-6
DEFAULT_GAMMA_MAX = 500.0
DEFAULT_GAMMA_MIN = 1.0
DEFAULT_DEBUG_EVERY = 30


# =============================================================================
# Utilities (EXACT copy from temporal_ssm_v2.py VMFUtils)
# =============================================================================

def A_D(kappa: float, dim: int, 
        kappa_max: float = DEFAULT_KAPPA_MAX,
        kappa_min: float = DEFAULT_KAPPA_MIN) -> float:
    """
    Ratio A_D(κ) = I_{D/2}(κ) / I_{D/2-1}(κ)
    
    Expected value of dot product between vMF sample and mean direction.
    Bounded in (0, 1).
    
    Uses exponentially scaled Bessel functions to avoid overflow.
    """
    if kappa < kappa_min:
        return 0.0
    
    kappa = min(kappa, kappa_max)
    v = dim / 2.0
    
    # Use exponentially scaled Bessel to avoid overflow
    num = ive(v, kappa)
    denom = ive(v - 1, kappa)
    
    if denom < 1e-300:
        # Asymptotic expansion for large κ
        return 1.0 - (dim - 1) / (2 * kappa)
    
    return np.clip(num / denom, 0.0, 1.0 - 1e-10)


# LUT cache for A_D_vectorized (key: (dim, kappa_max, kappa_min, step))
_A_D_LUT_CACHE: Dict[Tuple[int, float, float, float], Tuple[np.ndarray, np.ndarray]] = {}


def _build_A_D_LUT(dim: int, kappa_max: float, kappa_min: float, step: float = 0.1
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Build LUT for A_D function using vectorized ive calls."""
    kappa_grid = np.arange(kappa_min, kappa_max + step, step)
    v = dim / 2.0
    
    # Vectorized ive calls
    num = ive(v, kappa_grid)
    denom = ive(v - 1, kappa_grid)
    
    # Handle asymptotic case where denom is tiny
    with np.errstate(divide='ignore', invalid='ignore'):
        values = num / denom
    
    # Fix asymptotic values
    asymptotic_mask = denom < 1e-300
    values[asymptotic_mask] = 1.0 - (dim - 1) / (2 * kappa_grid[asymptotic_mask])
    
    values = np.clip(values, 0.0, 1.0 - 1e-10)
    
    return kappa_grid, values


def A_D_vectorized(kappa: np.ndarray, dim: int,
                   kappa_max: float = DEFAULT_KAPPA_MAX,
                   kappa_min: float = DEFAULT_KAPPA_MIN,
                   step: float = 0.1) -> np.ndarray:
    """
    LUT-based vectorized A_D for arrays.
    
    Uses precomputed lookup table with linear interpolation for speed.
    """
    # Get or build LUT
    cache_key = (dim, kappa_max, kappa_min, step)
    if cache_key not in _A_D_LUT_CACHE:
        _A_D_LUT_CACHE[cache_key] = _build_A_D_LUT(dim, kappa_max, kappa_min, step)
    
    kappa_grid, values = _A_D_LUT_CACHE[cache_key]
    
    # Clip kappa to valid range and interpolate
    kappa_clipped = np.clip(kappa, kappa_min, kappa_max)
    return np.interp(kappa_clipped, kappa_grid, values)


def inv_A_D(r_bar: float, dim: int,
            kappa_max: float = DEFAULT_KAPPA_MAX,
            kappa_min: float = DEFAULT_KAPPA_MIN) -> float:
    """
    Inverse of A_D: estimate κ from mean resultant length using Banerjee approximation.
    
    κ ≈ r̄(D - r̄²) / (1 - r̄²)
    """
    r_bar = np.clip(r_bar, 1e-10, 1.0 - 1e-10)
    r_bar_sq = r_bar ** 2
    kappa = r_bar * (dim - r_bar_sq) / (1 - r_bar_sq)
    return np.clip(kappa, kappa_min, kappa_max)


def safe_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Safely normalize vectors to unit length."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-10)


def safe_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    logits_shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / (np.sum(exp_logits, axis=axis, keepdims=True) + 1e-10)


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute entropy of probability distribution."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=axis)


# =============================================================================
# Kalman Filter Configuration and Implementation
# =============================================================================

@dataclass
class KalmanFilterConfig:
    """Configuration for Kalman filter (SORT-style)."""
    # Process noise (uncertainty growth per step)
    q_xy: float = 1.0        # Position uncertainty growth
    q_s: float = 1.0         # Scale (area) uncertainty growth
    q_r: float = 1e-2        # Aspect ratio uncertainty growth
    q_v: float = 0.01        # Velocity uncertainty growth
    
    # Initial state uncertainty
    p_xy: float = 10.0       # Initial position uncertainty
    p_s: float = 10.0        # Initial scale uncertainty
    p_r: float = 1.0         # Initial aspect ratio uncertainty
    p_v: float = 1000.0      # Initial velocity uncertainty (high = unknown)
    
    # Measurement noise
    r_xy: float = 1.0        # Measurement position noise
    r_s: float = 1.0         # Measurement scale noise
    r_r: float = 1.0         # Measurement aspect ratio noise


class KalmanBoxTracker:
    """
    Kalman filter tracker for a single bounding box.
    
    State: [x, y, s, r, vx, vy, vs, vr]
    - x, y: center position
    - s: scale (area)
    - r: aspect ratio (w/h)
    - vx, vy, vs, vr: velocities
    
    Measurement: [x, y, s, r]
    """
    
    _count = 0
    
    def __init__(self, bbox: np.ndarray, config: Optional[KalmanFilterConfig] = None):
        """
        Initialize tracker from bounding box [x1, y1, x2, y2].
        """
        self.config = config or KalmanFilterConfig()
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        
        self.dim_x = 8  # state dimension
        self.dim_z = 4  # measurement dimension
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # s += vs
        self.F[3, 7] = 1  # r += vr
        
        # Measurement matrix (observe position/scale/ratio, not velocity)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1
        
        # Process noise covariance
        self.Q = np.diag([
            self.config.q_xy, self.config.q_xy, self.config.q_s, self.config.q_r,
            self.config.q_v, self.config.q_v, self.config.q_v, self.config.q_v
        ])
        
        # Measurement noise covariance
        self.R = np.diag([
            self.config.r_xy, self.config.r_xy, self.config.r_s, self.config.r_r
        ])
        
        # Initial state covariance
        self.P = np.diag([
            self.config.p_xy, self.config.p_xy, self.config.p_s, self.config.p_r,
            self.config.p_v, self.config.p_v, self.config.p_v, self.config.p_v
        ])
        
        # Initialize state from bbox
        self.x = np.zeros(self.dim_x)
        self._bbox_to_z(bbox, self.x[:4])
        
        # Track history
        self.history = []
        self.hits = 0
        self.time_since_update = 0
        self.age = 0
    
    def _bbox_to_z(self, bbox: np.ndarray, z: np.ndarray) -> None:
        """Convert [x1, y1, x2, y2] to [x_center, y_center, scale, aspect_ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z[0] = bbox[0] + w / 2  # center x
        z[1] = bbox[1] + h / 2  # center y
        z[2] = w * h            # scale (area)
        z[3] = w / (h + 1e-8)   # aspect ratio
    
    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [x_center, y_center, scale, aspect_ratio] to [x1, y1, x2, y2]."""
        w = np.sqrt(max(z[2] * z[3], 1e-8))
        h = max(z[2] / (w + 1e-8), 1e-8)
        return np.array([
            z[0] - w / 2, z[1] - h / 2,  # x1, y1
            z[0] + w / 2, z[1] + h / 2   # x2, y2
        ])
    
    def predict(self) -> np.ndarray:
        """Advance state estimate to next frame."""
        # Prevent negative scale
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        
        # Predict state and covariance
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.get_state())
        
        return self.get_state()
    
    def update(self, bbox: np.ndarray) -> np.ndarray:
        """Update state with matched detection."""
        # Convert bbox to measurement
        z = np.zeros(self.dim_z)
        self._bbox_to_z(bbox, z)
        
        # Kalman update equations
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        
        # Joseph form for numerical stability
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Ensure positive scale and ratio
        self.x[2] = max(self.x[2], 1e-8)
        self.x[3] = max(self.x[3], 1e-8)
        
        self.hits += 1
        self.time_since_update = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current bounding box estimate [x1, y1, x2, y2]."""
        return self._z_to_bbox(self.x[:4])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate (vx, vy) in pixels/frame."""
        return self.x[4], self.x[5]


def create_kalman_tracker(bbox: np.ndarray, tracker_type: str = 'standard',
                         config: Optional[KalmanFilterConfig] = None) -> KalmanBoxTracker:
    """Factory function to create Kalman tracker."""
    return KalmanBoxTracker(bbox, config)


def reset_kalman_id():
    """Reset the global tracker ID counter."""
    KalmanBoxTracker._count = 0


# =============================================================================
# Data Association
# =============================================================================

@dataclass
class AssociationConfig:
    """Configuration for data association."""
    iou_threshold: float = 0.3          # Min IoU for valid match
    feature_threshold: float = 0.5       # Max cosine distance for valid match
    iou_weight: float = 0.5              # Weight for IoU in combined matching
    feature_weight: float = 0.5          # Weight for feature distance
    high_score_threshold: float = 0.6    # ByteTrack: high confidence threshold
    low_score_threshold: float = 0.1     # ByteTrack: low confidence threshold
    second_match_threshold: float = 0.5  # ByteTrack: second round IoU threshold


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / (area1 + area2 - inter_area + 1e-8)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes."""
    N, M = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    return iou_matrix


def compute_cosine_distance_matrix(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix between two sets of features."""
    features1_norm = safe_normalize(features1, axis=1)
    features2_norm = safe_normalize(features2, axis=1)
    similarity = features1_norm @ features2_norm.T
    return 1 - similarity


def hungarian_assignment(cost_matrix: np.ndarray, threshold: float = 1.0
                        ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Optimal assignment using Hungarian algorithm.
    
    Returns:
        matches: List of (track_idx, detection_idx) pairs
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    N, M = cost_matrix.shape
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_rows = set(range(N))
    unmatched_cols = set(range(M))
    
    for row_idx, col_idx in zip(row_indices, col_indices):
        if cost_matrix[row_idx, col_idx] <= threshold:
            matches.append((row_idx, col_idx))
            unmatched_rows.discard(row_idx)
            unmatched_cols.discard(col_idx)
    
    return matches, list(unmatched_rows), list(unmatched_cols)


def associate_iou(track_boxes: np.ndarray, detection_boxes: np.ndarray,
                 iou_threshold: float = 0.3, use_hungarian: bool = True
                 ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Associate tracks with detections using IoU only."""
    if len(track_boxes) == 0:
        return [], [], list(range(len(detection_boxes)))
    if len(detection_boxes) == 0:
        return [], list(range(len(track_boxes))), []
    
    iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)
    cost_matrix = 1 - iou_matrix  # Cost = 1 - IoU
    
    return hungarian_assignment(cost_matrix, 1 - iou_threshold)


def associate_combined(track_boxes: np.ndarray, detection_boxes: np.ndarray,
                       track_features: Optional[np.ndarray] = None,
                       detection_features: Optional[np.ndarray] = None,
                       config: Optional[AssociationConfig] = None
                       ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Associate using combined IoU and feature distance (DeepSORT-style)."""
    config = config or AssociationConfig()
    
    if len(track_boxes) == 0:
        return [], [], list(range(len(detection_boxes)))
    if len(detection_boxes) == 0:
        return [], list(range(len(track_boxes))), []
    
    # Compute IoU cost
    iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)
    iou_cost = 1 - iou_matrix
    
    # Compute feature cost if available
    if (track_features is not None and detection_features is not None and
        len(track_features) > 0 and len(detection_features) > 0):
        feature_cost = compute_cosine_distance_matrix(track_features, detection_features)
        cost_matrix = config.iou_weight * iou_cost + config.feature_weight * feature_cost
    else:
        cost_matrix = iou_cost
    
    # Gate by IoU threshold
    cost_matrix[iou_matrix < config.iou_threshold] = 1e9
    
    return hungarian_assignment(cost_matrix, 1.0)


def associate_bytetrack(track_boxes: np.ndarray, track_scores: np.ndarray,
                        detection_boxes: np.ndarray, detection_scores: np.ndarray,
                        track_features: Optional[np.ndarray] = None,
                        detection_features: Optional[np.ndarray] = None,
                        config: Optional[AssociationConfig] = None
                        ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """ByteTrack-style cascade matching (high-conf → low-conf)."""
    config = config or AssociationConfig()
    
    if len(track_boxes) == 0:
        return [], [], list(range(len(detection_boxes)))
    if len(detection_boxes) == 0:
        return [], list(range(len(track_boxes))), []
    
    # Split detections by confidence
    high_conf_mask = detection_scores >= config.high_score_threshold
    low_conf_mask = (detection_scores >= config.low_score_threshold) & ~high_conf_mask
    
    high_conf_indices = np.where(high_conf_mask)[0]
    low_conf_indices = np.where(low_conf_mask)[0]
    
    all_track_indices = list(range(len(track_boxes)))
    
    # First round: match tracks with high-confidence detections
    if len(high_conf_indices) > 0:
        high_det_boxes = detection_boxes[high_conf_indices]
        high_det_features = detection_features[high_conf_indices] if detection_features is not None else None
        
        first_matches, unmatched_tracks, unmatched_high_det = associate_combined(
            track_boxes, high_det_boxes, track_features, high_det_features, config
        )
        
        # Map back to original indices
        matches = [(t, high_conf_indices[d]) for t, d in first_matches]
        unmatched_high_det = [high_conf_indices[d] for d in unmatched_high_det]
    else:
        matches = []
        unmatched_tracks = all_track_indices
        unmatched_high_det = []
    
    # Second round: match remaining tracks with low-confidence detections (IoU only)
    if len(low_conf_indices) > 0 and len(unmatched_tracks) > 0:
        unmatched_track_boxes = track_boxes[unmatched_tracks]
        low_det_boxes = detection_boxes[low_conf_indices]
        
        iou_matrix = compute_iou_matrix(unmatched_track_boxes, low_det_boxes)
        cost_matrix = 1 - iou_matrix
        
        second_matches, remaining_tracks, unmatched_low_det = hungarian_assignment(
            cost_matrix, 1 - config.second_match_threshold
        )
        
        # Map back to original indices
        for ut_idx, ld_idx in second_matches:
            matches.append((unmatched_tracks[ut_idx], low_conf_indices[ld_idx]))
        
        unmatched_tracks = [unmatched_tracks[i] for i in remaining_tracks]
        unmatched_low_det = [low_conf_indices[d] for d in unmatched_low_det]
    else:
        unmatched_low_det = list(low_conf_indices)
    
    # Combine unmatched detections
    unmatched_detections = unmatched_high_det + unmatched_low_det
    
    return matches, unmatched_tracks, unmatched_detections


def associate(track_boxes: np.ndarray, detection_boxes: np.ndarray,
              detection_scores: Optional[np.ndarray] = None,
              track_features: Optional[np.ndarray] = None,
              detection_features: Optional[np.ndarray] = None,
              track_scores: Optional[np.ndarray] = None,
              method: Literal['iou', 'hungarian', 'combined', 'bytetrack'] = 'hungarian',
              config: Optional[AssociationConfig] = None
              ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Main association function with strategy selection.
    
    Args:
        track_boxes: (N, 4) array of track boxes
        detection_boxes: (M, 4) array of detection boxes
        detection_scores: (M,) detection confidence scores
        track_features: (N, D) track appearance features
        detection_features: (M, D) detection appearance features
        track_scores: (N,) track confidence scores
        method: Association method ('iou', 'hungarian', 'combined', 'bytetrack')
        config: Association configuration
        
    Returns:
        matches: List of (track_idx, detection_idx) pairs
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    config = config or AssociationConfig()
    
    if method == 'iou':
        return associate_iou(track_boxes, detection_boxes, config.iou_threshold, 
                           use_hungarian=False)
    elif method == 'hungarian':
        return associate_iou(track_boxes, detection_boxes, config.iou_threshold,
                           use_hungarian=True)
    elif method == 'combined':
        return associate_combined(track_boxes, detection_boxes, 
                                track_features, detection_features, config)
    elif method == 'bytetrack':
        if detection_scores is None:
            raise ValueError("ByteTrack requires detection_scores")
        if track_scores is None:
            track_scores = np.ones(len(track_boxes))
        return associate_bytetrack(track_boxes, track_scores, detection_boxes, 
                                  detection_scores, track_features, detection_features, config)
    else:
        raise ValueError(f"Unknown association method: {method}")


# =============================================================================
# Per-Track STAD Configuration
# =============================================================================

@dataclass
class TrackSTADConfig:
    """
    Configuration for per-track STAD.
    
    EXACT SAME parameters as temporal_ssm_v2.py, but with per-track defaults.
    Per-track operates on smaller history, so some values differ.
    """
    # =========== vMF Parameters ===========
    # Global concentration parameters (matching temporal_ssm_v2.py)
    kappa_trans: float = 10.0     # Transition concentration (κ^trans) - inertia
    kappa_ems: float = 20.0       # Emission concentration (κ^ems) - scaling
    
    # Per-class variational concentration
    gamma_init: float = 10.0      # Initial γ_k
    gamma_min: float = 1.0
    gamma_max: float = 200.0
    
    # Bounds for kappa
    kappa_max: float = 100.0
    kappa_min: float = 1e-6
    
    # =========== EM Algorithm ===========
    # Window size for windowed EM (frames of history)
    window_size: int = 5          # Per-track uses smaller window
    
    # EM iterations per update
    em_iterations: int = 3
    
    # =========== Update Control ===========
    # Minimum confidence for update
    min_confidence: float = 0.3
    
    # Minimum effective samples per class for update (lower for single track)
    min_updates_per_class: float = 0.5
    
    # =========== Fusion ===========
    # Temperature for softmax (matching temporal_ssm_v2.py)
    temperature: float = 1.0
    
    # VLM prior weight for E-step (0=SSM only, 1=VLM only)
    vlm_prior_weight: float = 0.2
    
    # =========== Mixing Coefficients ===========
    # Whether to use mixing coefficients π (NOW MEANINGFUL at track level!)
    use_pi: bool = True
    
    # EMA for π updates
    use_ema_pi: bool = False
    pi_ema_decay: float = 1.0     # 1.0 = no EMA
    
    # Dirichlet prior for π
    dirichlet_alpha: float = 0.1  # Smaller for per-track (less regularization)
    
    # =========== Stability ===========
    # EMA decay for gamma updates
    gamma_ema_decay: float = 0.7
    
    # Max R_k fraction (stability)
    max_rk_fraction: float = 0.5
    
    # =========== Gaussian-specific ===========
    q_scale: float = 0.01         # Process noise scale
    r_base: float = 0.5           # Base emission noise
    use_diagonal_cov: bool = True
    use_smoothing: bool = False
    
    # =========== Debug ===========
    debug: bool = False
    debug_every: int = 30


# =============================================================================
# Per-Track STAD-vMF (FULL implementation matching temporal_ssm_v2.py)
# =============================================================================

class TrackSTADvMF:
    """
    Per-track vMF state-space model for class probability smoothing.
    
    This is the FULL implementation of STAD-vMF, with the EXACT SAME algorithm
    as TemporalSSMvMF in temporal_ssm_v2.py, just operating on a single track's
    history instead of all detections in a frame.
    
    Key Difference from Global STAD:
    - π (mixing coefficients) are now per-track, solving the class imbalance bias
    - History window is per-track, not across all detections in a frame
    - Updates only when this specific track gets matched
    
    The algorithm is LINE-BY-LINE IDENTICAL to temporal_ssm_v2.py.
    
    State:
    - rho: (K, D) mean directions for each class prototype
    - gamma: (K,) per-class concentrations
    - pi: (K,) mixing coefficients (meaningful at track level!)
    - kappa_trans, kappa_ems: global concentration parameters
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 initial_probs: np.ndarray,
                 initial_feature: Optional[np.ndarray] = None,
                 config: Optional[TrackSTADConfig] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize per-track STAD-vMF.
        
        Args:
            num_classes: Number of classes K
            feature_dim: Feature dimension D
            initial_probs: (K,) initial class probabilities from first detection
            initial_feature: Optional (D,) initial feature for prototype init
            config: Configuration
            class_names: Optional class names for debugging
        """
        self.config = config or TrackSTADConfig()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_names = class_names
        
        # === vMF State (matching STADvMFState in temporal_ssm_v2.py) ===
        
        # Mean directions ρ (K, D) - one prototype per class
        self.rho: np.ndarray = np.zeros((num_classes, feature_dim))
        
        # Initialize prototypes from initial feature
        if initial_feature is not None:
            feat_norm = safe_normalize(initial_feature)
            # Initialize all class prototypes to the initial feature direction
            # They will diverge as we accumulate class-specific evidence
            for k in range(num_classes):
                self.rho[k] = feat_norm
        else:
            # Random initialization on unit sphere (normalized)
            self.rho = safe_normalize(np.random.randn(num_classes, feature_dim), axis=1)
        
        # Per-class concentrations γ (K,)
        self.gamma: np.ndarray = np.full(num_classes, self.config.gamma_init, dtype=np.float64)
        
        # Per-track mixing coefficients π (K,) - NOW MEANINGFUL!
        # Initialize from detection's class probabilities
        self.pi: np.ndarray = initial_probs.copy().astype(np.float64)
        self.pi = self.pi / (self.pi.sum() + 1e-10)
        self.pi_ema: np.ndarray = self.pi.copy()  # For EMA updates
        
        # Global concentration parameters (per track, same semantics)
        self.kappa_ems: float = self.config.kappa_ems
        self.kappa_trans: float = self.config.kappa_trans
        
        # === History Buffers (windowed EM) ===
        self.feature_history: List[np.ndarray] = []  # List of (D,) features
        self.probs_history: List[np.ndarray] = []    # List of (K,) VLM probs
        self.confidence_history: List[float] = []    # List of confidence scores
        
        # === Stats (matching temporal_ssm_v2.py) ===
        self.num_updates_total: int = 0
        self.num_updates_skipped: int = 0
        self.num_updates_by_class: np.ndarray = np.zeros(num_classes, dtype=np.int64)
        self.class_update_counts: np.ndarray = np.zeros(num_classes, dtype=np.int32)
        
        # Debug
        self._debug_call_count: int = 0
    
    def _get_class_name(self, k: int) -> str:
        """Get class name for logging."""
        if self.class_names and k < len(self.class_names):
            return self.class_names[k]
        return f"cls{k}"
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using current prototypes.
        
        Uses the paper's formulation (Eq. 10) - EXACT copy from temporal_ssm_v2.py:
        p(c_{t,n,k}=1|h) ∝ π_k · C_D(κ^ems) · exp(κ^ems · w_k^T · h)
        
        With w_k = E_q[w_k] = A_D(γ_k) · ρ_k (expected mean direction)
        
        Args:
            features: (D,) or (N, D) query features
            
        Returns:
            (K,) or (N, K) class probabilities
        """
        # Handle single feature
        single_input = features.ndim == 1
        if single_input:
            features = features[np.newaxis, :]
        
        # Normalize features
        features_norm = safe_normalize(features, axis=1)
        N = features_norm.shape[0]
        K = self.num_classes
        D = self.feature_dim
        
        # Compute expected prototypes: m_k = A_D(γ_k) · ρ_k
        # This is the key vMF formulation - A_D bounds the effective magnitude
        a_d_gamma = A_D_vectorized(
            self.gamma, D, self.config.kappa_max, self.config.kappa_min
        )  # (K,)
        m_k = a_d_gamma[:, np.newaxis] * self.rho  # (K, D)
        
        # Compute logits: κ^ems · <h_n, m_k> / temperature + log(π_k)
        similarities = features_norm @ m_k.T  # (N, K)
        
        if self.config.use_pi:
            logits = (self.kappa_ems * similarities / self.config.temperature +
                     np.log(self.pi + 1e-10)[np.newaxis, :])
        else:
            logits = self.kappa_ems * similarities / self.config.temperature
        
        # Softmax
        probs = safe_softmax(logits, axis=1)
        
        if single_input:
            return probs[0]
        return probs
    
    def get_class_belief(self) -> np.ndarray:
        """
        Get track's current class belief (mixing coefficients π).
        
        This is the main output for fusion with detection predictions.
        At track level, π represents the accumulated class evidence for THIS object.
        
        Returns:
            (K,) class probabilities
        """
        return self.pi.copy()
    
    def update(self,
               feature: np.ndarray,
               vlm_probs: np.ndarray,
               confidence: float) -> None:
        """
        Update track state with new matched detection using windowed EM.
        
        CRITICAL: Use raw VLM probs, NOT post-fusion probs to avoid self-reinforcement!
        
        This implements the EXACT SAME algorithm as temporal_ssm_v2.py _soft_em_update(),
        just operating on this track's history instead of all detections.
        
        Args:
            feature: (D,) detection feature
            vlm_probs: (K,) raw VLM class probabilities (NOT post-adaptation!)
            confidence: Detection confidence score
        """
        # Skip low-confidence updates
        if confidence < self.config.min_confidence:
            self.num_updates_skipped += 1
            return
        
        # Add to history (store normalized feature and raw VLM probs)
        self.feature_history.append(safe_normalize(feature.copy()))
        self.probs_history.append(vlm_probs.copy())
        self.confidence_history.append(confidence)
        
        # Keep only last window_size entries
        while len(self.feature_history) > self.config.window_size:
            self.feature_history.pop(0)
            self.probs_history.pop(0)
            self.confidence_history.pop(0)
        
        # Run soft EM on window (FULL implementation from temporal_ssm_v2.py)
        self._soft_em_update()
        
        self.num_updates_total += 1
    
    def _soft_em_update(self) -> None:
        """
        Run soft variational EM on windowed history.
        
        This is the EXACT SAME algorithm as temporal_ssm_v2.py _soft_em_update(),
        with all the stability fixes:
        - Temperature scaling in E-step (matching predict())
        - r_bar_k clamping to prevent gamma explosion
        - EMA gamma updates for smooth adaptation
        - Per-class R_k capping to prevent single-class domination
        - Dirichlet prior for π regularization
        
        The code below is a LINE-BY-LINE port of temporal_ssm_v2.py lines 480-690.
        """
        if len(self.feature_history) < 1:
            return
        
        # Concatenate history
        features = np.array(self.feature_history)  # (T, D)
        vlm_probs = np.array(self.probs_history)   # (T, K)
        confs = np.array(self.confidence_history)  # (T,)
        
        n_frames = len(features)
        N = n_frames  # Number of samples in window
        K = self.num_classes
        D = self.feature_dim
        
        # Compute temporal + confidence weights (matching temporal_ssm_v2.py)
        # More recent frames and higher confidence = higher weight
        temporal_weights = np.linspace(0.5, 1.0, n_frames)
        weights = temporal_weights * confs
        weights = weights / (weights.sum() + 1e-10)
        
        # Normalize features
        features_norm = safe_normalize(features, axis=1)
        
        # Store previous prototypes for transition prior
        rho_prev = self.rho.copy()
        gamma_prev = self.gamma.copy()
        
        # EM iterations (matching temporal_ssm_v2.py)
        for em_iter in range(self.config.em_iterations):
            # ===== E-STEP: Compute soft responsibilities =====
            # λ_{n,k} ∝ π_k · exp(κ^ems · E_q[w_k]^T · h_n / temperature)
            
            # Compute expected prototypes
            a_d_gamma = A_D_vectorized(
                self.gamma, D, self.config.kappa_max, self.config.kappa_min
            )  # (K,)
            m_k = a_d_gamma[:, np.newaxis] * self.rho  # (K, D)
            
            # Compute SSM logits WITH TEMPERATURE (consistent with predict())
            similarities = features_norm @ m_k.T  # (N, K)
            ssm_logits = (self.kappa_ems * similarities / self.config.temperature +
                        np.log(self.pi + 1e-10)[np.newaxis, :])
            ssm_probs = safe_softmax(ssm_logits, axis=1)  # (N, K)
            
            # Optionally combine with VLM probs (matching temporal_ssm_v2.py)
            if self.config.vlm_prior_weight > 0:
                # λ = normalize(λ_vlm^η · λ_ssm^(1-η))
                # use entropy to weight the vlm and ssm probs
                w_vlm = np.exp(-entropy(vlm_probs, axis=1))
                w_ssm = np.exp(-entropy(ssm_probs, axis=1))
                w_total = w_vlm + w_ssm + 1e-10
                w_vlm = w_vlm / w_total
                w_ssm = w_ssm / w_total
                w_vlm = w_vlm[:, np.newaxis]
                w_ssm = w_ssm[:, np.newaxis]
                log_combined = (w_vlm * np.log(vlm_probs + 1e-10) +
                               w_ssm * np.log(ssm_probs + 1e-10))
                responsibilities = safe_softmax(log_combined, axis=1)
            else:
                responsibilities = ssm_probs
            
            # Apply temporal weights to responsibilities
            weighted_resp = responsibilities * weights[:, np.newaxis]  # (N, K)
            
            
            # ===== M-STEP: Update parameters =====
            
            # Count effective samples per class
            R_k = np.sum(weighted_resp, axis=0)  # (K,)
            
            # STABILITY FIX: Cap per-class R_k to prevent single-class domination
            max_Rk_per_class = N * self.config.max_rk_fraction
            R_k_capped = np.minimum(R_k, max_Rk_per_class)
            
            # Check per-class update threshold
            class_has_enough = R_k_capped >= self.config.min_updates_per_class
            
            
            # Update prototypes for classes with enough samples
            new_rho = self.rho.copy()
            new_gamma = self.gamma.copy()
            
            for k in range(K):
                if not class_has_enough[k]:
                    # Keep previous values
                    continue
                
                # Compute weighted sum of features (use capped weights)
                resp_scale = R_k_capped[k] / (R_k[k] + 1e-10) if R_k[k] > max_Rk_per_class else 1.0
                s_k = np.sum(weighted_resp[:, k:k+1] * resp_scale * features_norm, axis=0)  # (D,)
                
                # Add transition prior: pull toward previous prototype
                # β_k = κ^trans · E_q[w_{t-1,k}] + κ^ems · Σ_n λ_{n,k} · h_n
                # Reuse pre-computed a_d_gamma[k] instead of calling A_D per class
                m_prev_k = a_d_gamma[k] * rho_prev[k]
                
                s_k_combined = self.kappa_ems * s_k + self.kappa_trans * m_prev_k
                
                # New mean direction
                gamma_k_new_norm = np.linalg.norm(s_k_combined)
                if gamma_k_new_norm > 1e-10:
                    new_rho[k] = s_k_combined / gamma_k_new_norm
                    
                    # Estimate new concentration from mean resultant length
                    total_effective = self.kappa_ems * R_k_capped[k] + self.kappa_trans
                    r_bar_k = gamma_k_new_norm / total_effective
                    
                    # STABILITY FIX: Clamp r_bar_k to prevent gamma explosion
                    r_bar_k = np.clip(r_bar_k, 0.0, 0.95)
                    
                    gamma_k_from_r = inv_A_D(r_bar_k, D, self.config.kappa_max, self.config.kappa_min)
                    
                    # STABILITY FIX: EMA gamma update for smoother adaptation
                    new_gamma[k] = (self.config.gamma_ema_decay * gamma_prev[k] +
                                   (1 - self.config.gamma_ema_decay) * gamma_k_from_r)

                    # Clamp gamma
                    new_gamma[k] = np.clip(new_gamma[k], self.config.gamma_min, self.config.gamma_max)
                    
                    # Track per-class updates
                    self.num_updates_by_class[k] += 1
                    self.class_update_counts[k] += 1
            
            # Update state
            self.rho = safe_normalize(new_rho, axis=1)
            self.gamma = new_gamma
            
            # Update mixing coefficients π with Dirichlet prior
            # Use CAPPED R_k for pi update to prevent single-class domination
            new_pi = (R_k_capped + self.config.dirichlet_alpha) / (
                np.sum(R_k_capped) + K * self.config.dirichlet_alpha
            )
            
            if self.config.use_ema_pi:
                self.pi = self.config.pi_ema_decay * self.pi_ema + (1 - self.config.pi_ema_decay) * new_pi
                self.pi_ema = self.pi.copy()
            else:
                self.pi = new_pi
            
            # Ensure π sums to 1
            self.pi = self.pi / (np.sum(self.pi) + 1e-10)
    
    def predict_step(self) -> None:
        """
        Predict step when no observation is available (track lost/occluded).
        
        Increases uncertainty (reduces gamma) and preserves current belief.
        """
        # Reduce concentration (increase uncertainty)
        self.gamma = np.maximum(self.gamma * 0.95, self.config.gamma_min)
        # π stays the same (preserve class belief during occlusion)
    
    def get_refined_probs(self, detection_probs: np.ndarray) -> np.ndarray:
        """
        Fuse detection probabilities with track's class belief.
        
        Uses entropy-weighted fusion (same as temporal_adapter_v2.py).
        
        Args:
            detection_probs: (K,) detection class probabilities
            
        Returns:
            (K,) fused class probabilities
        """
        track_probs = self.pi
        
        # Compute entropy (lower = more confident)
        eps = 1e-10
        H_det = -np.sum(detection_probs * np.log(detection_probs + eps))
        H_track = -np.sum(track_probs * np.log(track_probs + eps))
        
        # Weights: lower entropy = higher weight
        w_det = np.exp(-H_det)
        w_track = np.exp(-H_track)
        
        # Weighted fusion
        fused = (w_det * detection_probs + w_track * track_probs) / (w_det + w_track + eps)
        
        return fused / (fused.sum() + eps)
    
    def initialize_from_cache(self, cache_feature: np.ndarray, cache_probs: np.ndarray,
                              cache_gamma: Optional[np.ndarray] = None) -> None:
        """
        Initialize track state from global cache entry (Part 5.3).
        
        This transfers learned state from global cache to per-track STAD.
        
        Args:
            cache_feature: (D,) prototype feature from cache entry
            cache_probs: (K,) class probabilities from cache entry
            cache_gamma: Optional (K,) concentrations from cache
        """
        # Initialize rho from cache feature
        feat_norm = safe_normalize(cache_feature)
        for k in range(self.num_classes):
            self.rho[k] = feat_norm
        
        # Initialize π from cache probs (key for class belief transfer)
        self.pi = cache_probs.copy()
        self.pi = self.pi / (self.pi.sum() + 1e-10)
        self.pi_ema = self.pi.copy()
        
        # Optionally use cache gamma
        if cache_gamma is not None:
            self.gamma = cache_gamma.copy()
    
    def get_state_summary(self) -> Dict:
        """Get summary of current track state (for debugging)."""
        top_class = np.argmax(self.pi)
        return {
            'gamma_mean': float(np.mean(self.gamma)),
            'gamma_range': (float(self.gamma.min()), float(self.gamma.max())),
            'pi': self.pi.tolist(),
            'top_class': int(top_class),
            'top_class_prob': float(self.pi[top_class]),
            'num_updates': self.num_updates_total,
            'history_len': len(self.feature_history)
        }


# =============================================================================
# Per-Track STAD-Gaussian (FULL implementation matching temporal_ssm_v2.py)
# =============================================================================

class TrackSTADGaussian:
    """
    Per-track Gaussian (Kalman filter) state-space model for class probability smoothing.
    
    This is the FULL implementation of STAD-Gaussian, with the SAME algorithm
    as TemporalSSMGaussian in temporal_ssm_v2.py, adapted for per-track operation.
    
    FIXED: Now uses proper windowed EM like TrackSTADvMF:
    - Maintains feature_history, probs_history, confidence_history
    - E-step: Computes SSM predictions using learned mu, combines with VLM
    - M-step: Uses soft responsibilities for weighted Kalman updates
    
    Uses Kalman filter updates instead of vMF EM:
    - Per-class mean μ_k and covariance P_k
    - Predict: P_pred = P_prev + Q (process noise)
    - Update: Kalman gain, innovation, covariance update
    
    Supports diagonal covariances for efficiency and optional RTS smoothing.
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 initial_probs: np.ndarray,
                 initial_feature: Optional[np.ndarray] = None,
                 config: Optional[TrackSTADConfig] = None,
                 class_names: Optional[List[str]] = None):
        """Initialize per-track STAD-Gaussian."""
        self.config = config or TrackSTADConfig()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_names = class_names
        
        # === Gaussian State (matching STADGaussState in temporal_ssm_v2.py) ===
        
        # Means μ (K, D)
        if initial_feature is not None:
            self.mu = np.tile(safe_normalize(initial_feature), (num_classes, 1))
        else:
            self.mu = safe_normalize(np.random.randn(num_classes, feature_dim), axis=1)
        
        # Covariances (diagonal for efficiency)
        if self.config.use_diagonal_cov:
            self.P = np.ones((num_classes, feature_dim)) * 0.1  # (K, D)
            self.Q = np.ones(feature_dim) * self.config.q_scale  # (D,) process noise
        else:
            self.P = np.tile(np.eye(feature_dim) * 0.1, (num_classes, 1, 1))  # (K, D, D)
            self.Q = np.eye(feature_dim) * self.config.q_scale  # (D, D)
        
        self.R_base = self.config.r_base
        
        # Mixing coefficients π
        self.pi = initial_probs.copy().astype(np.float64)
        self.pi = self.pi / (self.pi.sum() + 1e-10)
        
        # === History Buffers (windowed EM) - MATCHING TrackSTADvMF ===
        self.feature_history: List[np.ndarray] = []  # List of (D,) features
        self.probs_history: List[np.ndarray] = []    # List of (K,) VLM probs
        self.confidence_history: List[float] = []    # List of confidence scores
        
        # History for RTS smoothing (separate from EM history)
        self.mu_pred_history: List[np.ndarray] = []
        self.P_pred_history: List[np.ndarray] = []
        self.mu_filt_history: List[np.ndarray] = []
        self.P_filt_history: List[np.ndarray] = []
        
        # Stats
        self.num_updates_total: int = 0
        self.num_updates_skipped: int = 0
        self.num_updates_by_class: np.ndarray = np.zeros(num_classes, dtype=np.int64)
        self.class_update_counts: np.ndarray = np.zeros(num_classes, dtype=np.int32)
    
    def _get_class_name(self, k: int) -> str:
        """Get class name for logging."""
        if self.class_names and k < len(self.class_names):
            return self.class_names[k]
        return f"cls{k}"
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Gaussian mixture likelihood.
        
        p(k|h) ∝ π_k · N(h | μ_k, Σ^ems)
        
        For efficiency, uses diagonal Σ^ems = R_base · I.
        
        Args:
            features: (D,) or (N, D) query features
            
        Returns:
            (K,) or (N, K) class probabilities
        """
        single_input = features.ndim == 1
        if single_input:
            features = features[np.newaxis, :]
        
        features_norm = safe_normalize(features, axis=1)
        N = features_norm.shape[0]
        K = self.num_classes
        
        # Compute log-likelihoods under each Gaussian
        # log p(h|k) = -0.5 * ||h - μ_k||² / R_base + const
        log_probs = np.zeros((N, K))
        for k in range(K):
            diff = features_norm - self.mu[k]  # (N, D)
            sq_dist = np.sum(diff ** 2, axis=1)  # (N,)
            # Add temperature scaling for consistency with vMF
            if self.config.use_pi:
                log_probs[:, k] = (-0.5 * sq_dist / (self.R_base * self.config.temperature) +
                                   np.log(self.pi[k] + 1e-10))
            else:
                log_probs[:, k] = -0.5 * sq_dist / (self.R_base * self.config.temperature)
        
        probs = safe_softmax(log_probs, axis=1)
        
        if single_input:
            return probs[0]
        return probs
    
    def get_class_belief(self) -> np.ndarray:
        """
        Get track's current class belief (mixing coefficients π).
        
        This is the main output for fusion with detection predictions.
        At track level, π represents the accumulated class evidence for THIS object.
        
        Returns:
            (K,) class probabilities
        """
        return self.pi.copy()
    
    def update(self,
               feature: np.ndarray,
               vlm_probs: np.ndarray,
               confidence: float) -> None:
        """
        Update track state with new matched detection using windowed EM.
        
        CRITICAL: Use raw VLM probs, NOT post-fusion probs to avoid self-reinforcement!
        
        This now implements the SAME windowed EM pattern as TrackSTADvMF,
        with Kalman filter M-step instead of vMF M-step.
        
        Args:
            feature: (D,) detection feature
            vlm_probs: (K,) raw VLM class probabilities (NOT post-adaptation!)
            confidence: Detection confidence score
        """
        # Skip low-confidence updates
        if confidence < self.config.min_confidence:
            self.num_updates_skipped += 1
            return
        
        # Add to history (store normalized feature and raw VLM probs)
        self.feature_history.append(safe_normalize(feature.copy()))
        self.probs_history.append(vlm_probs.copy())
        self.confidence_history.append(confidence)
        
        # Keep only last window_size entries
        while len(self.feature_history) > self.config.window_size:
            self.feature_history.pop(0)
            self.probs_history.pop(0)
            self.confidence_history.pop(0)
        
        # Run soft EM on window (FIXED: now uses proper E-step with SSM predictions)
        self._soft_em_update()
        
        self.num_updates_total += 1
    
    def _soft_em_update(self) -> None:
        """
        Run soft EM on windowed history - FIXED to match TrackSTADvMF pattern.
        
        Key fix: E-step now uses SSM predictions (via predict()) combined with VLM probs,
        instead of just using VLM probs directly as responsibilities.
        
        This ensures the learned μ actually influences which classes get updated.
        """
        if len(self.feature_history) < 1:
            return
        
        # Concatenate history
        features = np.array(self.feature_history)  # (T, D)
        vlm_probs = np.array(self.probs_history)   # (T, K)
        confs = np.array(self.confidence_history)  # (T,)
        
        n_frames = len(features)
        N = n_frames
        K = self.num_classes
        D = self.feature_dim
        
        # Compute temporal + confidence weights (matching vMF)
        temporal_weights = np.linspace(0.5, 1.0, n_frames)
        weights = temporal_weights * confs
        weights = weights / (weights.sum() + 1e-10)
        
        # Normalize features
        features_norm = safe_normalize(features, axis=1)
        
        # Store previous state for potential smoothing
        mu_prev = self.mu.copy()
        P_prev = self.P.copy()
        
        # EM iterations
        for em_iter in range(self.config.em_iterations):
            # ===== E-STEP: Compute soft responsibilities =====
            # FIXED: Use SSM predictions, not just VLM probs!
            
            # Get SSM predictions using learned μ
            ssm_probs = self.predict(features_norm)  # (N, K)
            
            # Combine SSM + VLM using entropy weighting (matching vMF)
            if self.config.vlm_prior_weight > 0:
                # Compute entropies
                eps = 1e-10
                H_vlm = -np.sum(vlm_probs * np.log(vlm_probs + eps), axis=1)  # (N,)
                H_ssm = -np.sum(ssm_probs * np.log(ssm_probs + eps), axis=1)  # (N,)
                
                # Entropy-based weights (lower entropy = more confident = higher weight)
                w_vlm = np.exp(-H_vlm)
                w_ssm = np.exp(-H_ssm)
                w_total = w_vlm + w_ssm + eps
                w_vlm = (w_vlm / w_total)[:, np.newaxis]
                w_ssm = (w_ssm / w_total)[:, np.newaxis]
                
                # Combine in log space
                log_combined = (w_vlm * np.log(vlm_probs + eps) +
                               w_ssm * np.log(ssm_probs + eps))
                responsibilities = safe_softmax(log_combined, axis=1)
            else:
                responsibilities = ssm_probs
            
            # Apply temporal weights to responsibilities
            weighted_resp = responsibilities * weights[:, np.newaxis]  # (N, K)
            
            # ===== M-STEP: Kalman updates per class =====
            
            # Count effective samples per class
            R_k = np.sum(weighted_resp, axis=0)  # (K,)
            
            # Cap per-class R_k to prevent single-class domination
            max_Rk_per_class = N * self.config.max_rk_fraction
            R_k_capped = np.minimum(R_k, max_Rk_per_class)
            
            # Check per-class update threshold
            class_has_enough = R_k_capped >= self.config.min_updates_per_class
            
            # Per-class Kalman updates
            for k in range(K):
                if not class_has_enough[k]:
                    continue
                
                # === Kalman Predict ===
                if self.config.use_diagonal_cov:
                    P_pred = self.P[k] + self.Q
                else:
                    P_pred = self.P[k] + self.Q
                
                # === Compute class observation (weighted mean of features) ===
                resp_scale = R_k_capped[k] / (R_k[k] + 1e-10) if R_k[k] > max_Rk_per_class else 1.0
                y_k = np.sum(weighted_resp[:, k:k+1] * resp_scale * features_norm, axis=0) / (R_k_capped[k] + 1e-10)
                
                # Observation noise shrinks with more evidence
                R_obs = self.R_base / (R_k_capped[k] + 1e-10)
                
                # === Kalman Update ===
                if self.config.use_diagonal_cov:
                    # Diagonal Kalman gain: K = P_pred / (P_pred + R_obs)
                    K_gain = P_pred / (P_pred + R_obs + 1e-10)
                    
                    # Update mean
                    innovation = y_k - self.mu[k]
                    self.mu[k] = self.mu[k] + K_gain * innovation
                    
                    # Update covariance
                    self.P[k] = (1 - K_gain) * P_pred
                    
                    # Clamp covariance
                    self.P[k] = np.clip(self.P[k], 1e-6, 1.0)
                else:
                    # Full covariance (expensive)
                    S = P_pred + np.eye(D) * R_obs
                    K_gain = P_pred @ np.linalg.inv(S + np.eye(D) * 1e-10)
                    
                    innovation = y_k - self.mu[k]
                    self.mu[k] = self.mu[k] + K_gain @ innovation
                    self.P[k] = (np.eye(D) - K_gain) @ P_pred
                
                self.class_update_counts[k] += 1
                self.num_updates_by_class[k] += 1
            
            # Normalize means (keep on unit sphere for consistency)
            self.mu = safe_normalize(self.mu, axis=1)
            
            # Update mixing coefficients π with Dirichlet prior
            new_pi = (R_k_capped + self.config.dirichlet_alpha) / (
                np.sum(R_k_capped) + K * self.config.dirichlet_alpha
            )
            
            # EMA update for π (smoother adaptation)
            self.pi = 0.9 * self.pi + 0.1 * new_pi
            self.pi = self.pi / (np.sum(self.pi) + 1e-10)
        
        # Optional RTS smoothing
        if self.config.use_smoothing:
            self.mu_pred_history.append(mu_prev)
            self.P_pred_history.append(P_prev)
            self.mu_filt_history.append(self.mu.copy())
            self.P_filt_history.append(self.P.copy())
            
            # Trim history
            while len(self.mu_pred_history) > self.config.window_size:
                self.mu_pred_history.pop(0)
                self.P_pred_history.pop(0)
                self.mu_filt_history.pop(0)
                self.P_filt_history.pop(0)
            
            # Apply RTS smoothing
            if len(self.mu_filt_history) >= 2:
                self._rts_smooth()
    
    def _rts_smooth(self) -> None:
        """Apply Rauch-Tung-Striebel backward smoothing."""
        T = len(self.mu_filt_history)
        if T < 2:
            return
        
        K = self.num_classes
        D = self.feature_dim
        
        mu_smooth = self.mu_filt_history[-1].copy()
        P_smooth = self.P_filt_history[-1].copy()
        
        for t in range(T - 2, -1, -1):
            mu_filt = self.mu_filt_history[t]
            P_filt = self.P_filt_history[t]
            mu_pred_next = self.mu_pred_history[min(t + 1, T - 1)]
            P_pred_next = self.P_pred_history[min(t + 1, T - 1)]
            
            for k in range(K):
                if self.config.use_diagonal_cov:
                    # Diagonal RTS
                    J = P_filt[k] / (P_pred_next[k] + 1e-10)
                    mu_smooth[k] = mu_filt[k] + J * (mu_smooth[k] - mu_pred_next[k])
                    P_smooth[k] = P_filt[k] + J ** 2 * (P_smooth[k] - P_pred_next[k])
                else:
                    # Full RTS
                    J = P_filt[k] @ np.linalg.inv(P_pred_next[k] + np.eye(D) * 1e-10)
                    mu_smooth[k] = mu_filt[k] + J @ (mu_smooth[k] - mu_pred_next[k])
                    P_smooth[k] = P_filt[k] + J @ (P_smooth[k] - P_pred_next[k]) @ J.T
        
        self.mu = safe_normalize(mu_smooth, axis=1)
        self.P = np.clip(P_smooth, 1e-6, 1.0)
    
    def predict_step(self) -> None:
        """Predict step (no observation) - increase uncertainty."""
        if self.config.use_diagonal_cov:
            self.P = self.P + self.Q[np.newaxis, :]
        else:
            self.P = self.P + self.Q[np.newaxis, :, :]
    
    def get_refined_probs(self, detection_probs: np.ndarray) -> np.ndarray:
        """Fuse detection with track belief using uncertainty weighting."""
        track_probs = self.pi
        
        # Uncertainty-based weighting
        if self.config.use_diagonal_cov:
            track_uncertainty = np.mean(self.P)
        else:
            track_uncertainty = np.mean([np.trace(self.P[k]) for k in range(self.num_classes)])
        
        det_weight = track_uncertainty / (track_uncertainty + 1.0)
        
        fused = det_weight * detection_probs + (1 - det_weight) * track_probs
        return fused / (fused.sum() + 1e-10)
    
    def initialize_from_cache(self, cache_feature: np.ndarray, cache_probs: np.ndarray,
                              cache_P: Optional[np.ndarray] = None) -> None:
        """Initialize from global cache entry (Part 5.3)."""
        self.mu = np.tile(safe_normalize(cache_feature), (self.num_classes, 1))
        self.pi = cache_probs.copy()
        self.pi = self.pi / (self.pi.sum() + 1e-10)
        if cache_P is not None:
            self.P = cache_P.copy()
    
    def get_state_summary(self) -> Dict:
        """Get state summary for debugging."""
        top_class = np.argmax(self.pi)
        return {
            'pi': self.pi.tolist(),
            'top_class': int(top_class),
            'top_class_prob': float(self.pi[top_class]),
            'num_updates': self.num_updates_total,
            'P_mean': float(np.mean(self.P)) if self.config.use_diagonal_cov else float(np.trace(self.P.mean(axis=0))),
            'history_len': len(self.feature_history)
        }


def create_track_stad(num_classes: int,
                      feature_dim: int,
                      initial_probs: np.ndarray,
                      initial_feature: Optional[np.ndarray] = None,
                      variant: str = 'vmf',
                      config: Optional[TrackSTADConfig] = None,
                      class_names: Optional[List[str]] = None) -> Union[TrackSTADvMF, TrackSTADGaussian]:
    """
    Factory function to create track STAD.
    
    Args:
        variant: 'vmf' or 'gaussian'
    """
    if variant.lower() == 'gaussian':
        return TrackSTADGaussian(num_classes, feature_dim, initial_probs, 
                                 initial_feature, config, class_names)
    else:
        return TrackSTADvMF(num_classes, feature_dim, initial_probs,
                           initial_feature, config, class_names)


# =============================================================================
# Track State and Detection
# =============================================================================

class TrackState(Enum):
    """Track lifecycle states (DeepSORT-style)."""
    TENTATIVE = 'tentative'
    CONFIRMED = 'confirmed'
    DELETED = 'deleted'


@dataclass
class TrackConfig:
    """Configuration for Track."""
    # Lifecycle thresholds
    min_hits_to_confirm: int = 3     # Hits needed to confirm track
    max_age: int = 30                # Frames before deletion if unmatched
    
    # Feature EMA
    feature_alpha: float = 0.1       # Feature update rate (lower = smoother)
    
    # Per-track STAD
    use_track_stad: bool = True
    stad_variant: str = 'vmf'        # 'vmf' or 'gaussian'
    stad_config: Optional[TrackSTADConfig] = None
    
    # Kalman filter
    kalman_config: Optional[KalmanFilterConfig] = None
    
    # Part 5.3: Initialize track STAD from global cache
    init_from_cache: bool = True
    cache_match_threshold: float = 0.5


@dataclass
class Detection:
    """Detection data structure."""
    box: np.ndarray              # [x1, y1, x2, y2]
    score: float                 # Confidence score
    label: str                   # Class label string
    class_idx: int              # Class index
    class_probs: np.ndarray     # (K,) class probabilities (may be adapted)
    feature: np.ndarray         # (D,) feature vector
    raw_class_probs: Optional[np.ndarray] = None  # (K,) RAW VLM probs (for STAD update!)


# =============================================================================
# Track Class
# =============================================================================

class Track:
    """
    Track object maintaining state for a single tracked entity.
    
    Combines:
    - Kalman filter for box prediction/smoothing
    - Feature EMA for appearance smoothing
    - Per-track STAD for class probability adaptation
    
    Part 5.3: Optionally initializes STAD from global cache.
    """
    
    _id_counter = 0
    
    def __init__(self,
                 detection: Detection,
                 num_classes: int,
                 feature_dim: int,
                 config: Optional[TrackConfig] = None,
                 global_cache=None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize track from first detection.
        
        Args:
            detection: Initial detection
            num_classes: Number of classes
            feature_dim: Feature dimension
            config: Track configuration
            global_cache: Optional global cache for Part 5.3 initialization
            class_names: Optional class names for STAD debugging
        """
        self.config = config or TrackConfig()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_names = class_names
        
        # Assign unique ID
        self.track_id = Track._id_counter
        Track._id_counter += 1
        
        # Initial class info
        self.class_idx = detection.class_idx
        self.class_label = detection.label
        
        # Kalman filter for box tracking
        self.kalman = KalmanBoxTracker(detection.box, self.config.kalman_config)
        
        # Feature EMA for appearance
        self.feature_ema = detection.feature.copy()
        self.feature_history: List[np.ndarray] = [detection.feature.copy()]
        
        # Per-track STAD for class probability adaptation
        if self.config.use_track_stad:
            # Use raw VLM probs if available, else use adapted probs
            init_probs = detection.raw_class_probs if detection.raw_class_probs is not None else detection.class_probs
            
            self.class_stad = create_track_stad(
                num_classes=num_classes,
                feature_dim=feature_dim,
                initial_probs=init_probs.copy(),
                initial_feature=detection.feature.copy(),
                variant=self.config.stad_variant,
                config=self.config.stad_config,
                class_names=class_names
            )
            
            # Part 5.3: Initialize from global cache if available and enabled
            if self.config.init_from_cache and global_cache is not None:
                self._init_stad_from_cache(detection, global_cache)
        else:
            self.class_stad = None
        
        # Metadata
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        
        # History
        self.score_history: List[float] = [detection.score]
        self.class_history: List[int] = [detection.class_idx]
        
        # Stats
        self.total_stad_updates = 0
    
    def _init_stad_from_cache(self, detection: Detection, global_cache) -> None:
        """
        Part 5.3: Initialize track state from global cache.
        
        Find best matching cache entry and use its state to initialize track STAD.
        """
        if global_cache is None or not hasattr(global_cache, 'compute_posterior'):
            return
        
        if hasattr(global_cache, 'is_empty') and global_cache.is_empty():
            return
        
        try:
            # Compute posterior over cache entries
            posterior = global_cache.compute_posterior(detection.feature, detection.box)
            
            if posterior is None or len(posterior) == 0:
                return
            
            m_star = np.argmax(posterior)
            
            if posterior[m_star] >= self.config.cache_match_threshold:
                # Get cache entry state
                cache_feature = global_cache.F_cache[:, m_star]
                cache_probs = global_cache.V_cache[:, m_star]
                
                # Initialize STAD from cache
                if hasattr(self.class_stad, 'initialize_from_cache'):
                    self.class_stad.initialize_from_cache(cache_feature, cache_probs)
        except Exception:
            # Silently ignore cache init failures
            pass
    
    def predict(self) -> np.ndarray:
        """Predict step (no observation)."""
        pred_box = self.kalman.predict()
        
        if self.class_stad is not None:
            self.class_stad.predict_step()
        
        self.age += 1
        self.time_since_update += 1
        
        # Mark as deleted if too old
        if self.time_since_update > self.config.max_age:
            self.state = TrackState.DELETED
        
        return pred_box
    
    def update(self, detection: Detection) -> np.ndarray:
        """
        Update track with matched detection.
        
        CRITICAL: Uses raw_class_probs for STAD update to avoid self-reinforcement!
        """
        # Update Kalman filter
        updated_box = self.kalman.update(detection.box)
        
        # Update feature EMA
        self.feature_ema = ((1 - self.config.feature_alpha) * self.feature_ema +
                          self.config.feature_alpha * detection.feature)
        self.feature_ema = self.feature_ema / (np.linalg.norm(self.feature_ema) + 1e-8)
        self.feature_history.append(detection.feature.copy())
        if len(self.feature_history) > 10:
            self.feature_history.pop(0)
        
        # Update STAD (CRITICAL: use raw VLM probs, NOT adapted probs!)
        if self.class_stad is not None:
            raw_probs = detection.raw_class_probs if detection.raw_class_probs is not None else detection.class_probs
            self.class_stad.update(
                feature=detection.feature,
                vlm_probs=raw_probs,  # RAW probs to avoid self-reinforcement!
                confidence=detection.score
            )
            self.total_stad_updates += 1
        
        # Update class from STAD belief (if using STAD)
        new_class_idx = detection.class_idx
        if self.class_stad is not None and self.config.use_track_stad:
            stad_probs = self.class_stad.get_class_belief()
            new_class_idx = np.argmax(stad_probs)
        
        self.class_idx = new_class_idx
        self.class_history.append(detection.class_idx)
        self.score_history.append(detection.score)
        
        # Update metadata
        self.hits += 1
        self.time_since_update = 0
        
        # Lifecycle transition
        if self.state == TrackState.TENTATIVE and self.hits >= self.config.min_hits_to_confirm:
            self.state = TrackState.CONFIRMED
        
        return updated_box
    
    def get_state(self) -> np.ndarray:
        """Get current box estimate."""
        return self.kalman.get_state()
    
    def get_class_probs(self) -> np.ndarray:
        """Get current class probabilities from STAD."""
        if self.class_stad is not None:
            return self.class_stad.get_class_belief()
        return np.ones(self.num_classes) / self.num_classes
    
    def get_refined_probs(self, detection_probs: np.ndarray) -> np.ndarray:
        """Fuse detection probs with track's STAD belief."""
        if self.class_stad is not None:
            return self.class_stad.get_refined_probs(detection_probs)
        return detection_probs
    
    def get_feature(self) -> np.ndarray:
        """Get current smoothed feature."""
        return self.feature_ema.copy()
    
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
    
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE
    
    def mark_deleted(self) -> None:
        self.state = TrackState.DELETED
    
    def get_summary(self) -> Dict:
        """Get track summary for debugging."""
        summary = {
            'track_id': self.track_id,
            'state': self.state.value,
            'class_idx': self.class_idx,
            'class_label': self.class_label,
            'hits': self.hits,
            'age': self.age,
            'time_since_update': self.time_since_update,
            'box': self.get_state().tolist(),
            'avg_score': float(np.mean(self.score_history[-10:])) if self.score_history else 0,
        }
        
        if self.class_stad is not None:
            summary['class_probs'] = self.get_class_probs().tolist()
            summary['top_class_prob'] = float(self.get_class_probs().max())
            if hasattr(self.class_stad, 'get_state_summary'):
                summary['stad_state'] = self.class_stad.get_state_summary()
        
        return summary


# =============================================================================
# Track Manager
# =============================================================================

class TrackManager:
    """
    Manager for multiple tracks.
    
    Handles:
    - Track creation with optional global cache initialization (Part 5.3)
    - Track lifecycle management
    - Association with detections
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 config: Optional[TrackConfig] = None,
                 global_cache=None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize track manager.
        
        Args:
            num_classes: Number of classes
            feature_dim: Feature dimension
            config: Track configuration
            global_cache: Optional global cache for Part 5.3 initialization
            class_names: Optional class names for debugging
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.config = config or TrackConfig()
        self.global_cache = global_cache  # For Part 5.3
        self.class_names = class_names
        
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0
        
        # Stats
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
    
    def set_global_cache(self, global_cache) -> None:
        """Set/update global cache reference for Part 5.3."""
        self.global_cache = global_cache
    
    def create_track(self, detection: Detection) -> Track:
        """Create new track from detection (with Part 5.3 cache initialization)."""
        track = Track(
            detection=detection,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            config=self.config,
            global_cache=self.global_cache,  # Part 5.3
            class_names=self.class_names
        )
        self.tracks[track.track_id] = track
        self.total_tracks_created += 1
        return track
    
    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)
    
    def get_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if not t.is_deleted()]
    
    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.is_confirmed()]
    
    def get_tentative_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.is_tentative()]
    
    def predict_all(self) -> Dict[int, np.ndarray]:
        """Predict all tracks, returns dict of track_id -> predicted_box."""
        predictions = {}
        for track_id, track in list(self.tracks.items()):
            if not track.is_deleted():
                predictions[track_id] = track.predict()
        return predictions
    
    def cleanup_deleted(self) -> int:
        """Remove deleted tracks. Returns count removed."""
        to_delete = [tid for tid, t in self.tracks.items() if t.is_deleted()]
        for tid in to_delete:
            del self.tracks[tid]
        self.total_tracks_deleted += len(to_delete)
        return len(to_delete)
    
    def step(self) -> None:
        """End-of-frame processing."""
        self.frame_count += 1
        self.cleanup_deleted()
    
    def reset(self) -> None:
        """Reset all state."""
        self.tracks.clear()
        Track._id_counter = 0
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
    
    def get_summary(self) -> Dict:
        """Get manager summary."""
        return {
            'frame_count': self.frame_count,
            'tracks_by_state': {
                'tentative': len(self.get_tentative_tracks()),
                'confirmed': len(self.get_confirmed_tracks()),
                'total_active': len(self.get_active_tracks())
            },
            'total_created': self.total_tracks_created,
            'total_deleted': self.total_tracks_deleted,
            'track_ids': list(self.tracks.keys())
        }


def reset_track_ids():
    """Reset global track ID counters."""
    Track._id_counter = 0
    KalmanBoxTracker._count = 0


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing track.py module (FULL STAD implementations)")
    
    np.random.seed(42)
    reset_track_ids()
    
    num_classes = 6
    feature_dim = 256
    
    # Test 1: TrackSTADvMF with windowed EM
    print("\n1. Testing TrackSTADvMF (FULL windowed EM):")
    initial_probs = np.array([0.8, 0.1, 0.05, 0.02, 0.02, 0.01])
    config = TrackSTADConfig(
        kappa_trans=10.0,
        kappa_ems=20.0,
        gamma_init=10.0,
        window_size=5,
        em_iterations=3,
        use_pi=True,
        vlm_prior_weight=0.2
    )
    stad = TrackSTADvMF(num_classes, feature_dim, initial_probs, 
                        np.random.randn(feature_dim), config)
    
    # Simulate detections over time
    for i in range(10):
        # Gradually shift class distribution
        probs = np.array([0.8 - i*0.05, 0.1 + i*0.03, 0.05, 0.02, 0.02, 0.01])
        probs = probs / probs.sum()
        stad.update(np.random.randn(feature_dim), probs, confidence=0.8)
    
    print(f"   Final π: {stad.pi.round(3)}")
    print(f"   Gamma range: [{stad.gamma.min():.2f}, {stad.gamma.max():.2f}]")
    print(f"   Updates: {stad.num_updates_total}")
    
    # Test 2: TrackSTADGaussian with Kalman updates
    print("\n2. Testing TrackSTADGaussian (FULL Kalman):")
    stad_g = TrackSTADGaussian(num_classes, feature_dim, initial_probs,
                               np.random.randn(feature_dim), config)
    
    for i in range(10):
        probs = np.array([0.8 - i*0.05, 0.1 + i*0.03, 0.05, 0.02, 0.02, 0.01])
        probs = probs / probs.sum()
        stad_g.update(np.random.randn(feature_dim), probs, confidence=0.8)
    
    print(f"   Final π: {stad_g.pi.round(3)}")
    print(f"   P mean: {np.mean(stad_g.P):.4f}")
    
    # Test 3: Track with STAD
    print("\n3. Testing Track with per-track STAD:")
    track_config = TrackConfig(
        min_hits_to_confirm=3,
        use_track_stad=True,
        stad_variant='vmf',
        stad_config=config
    )
    
    det = Detection(
        box=np.array([100, 100, 150, 160]),
        score=0.85,
        label='car',
        class_idx=0,
        class_probs=initial_probs,
        feature=np.random.randn(feature_dim),
        raw_class_probs=initial_probs  # RAW probs for STAD!
    )
    
    track = Track(det, num_classes, feature_dim, track_config)
    print(f"   Created track {track.track_id}, state={track.state.value}")
    
    # Update with new detections
    for i in range(5):
        new_det = Detection(
            box=det.box + i*5,
            score=0.8,
            label='car',
            class_idx=0,
            class_probs=initial_probs,
            feature=np.random.randn(feature_dim),
            raw_class_probs=initial_probs  # Use raw probs!
        )
        track.update(new_det)
    
    print(f"   After 5 updates: state={track.state.value}, hits={track.hits}")
    print(f"   Class probs from STAD: {track.get_class_probs().round(3)}")
    
    # Test 4: TrackManager
    print("\n4. Testing TrackManager:")
    manager = TrackManager(num_classes, feature_dim, track_config)
    
    # Create some tracks
    for i in range(3):
        d = Detection(
            box=np.array([100+i*50, 100, 150+i*50, 160]),
            score=0.85,
            label=['car', 'person', 'truck'][i],
            class_idx=i,
            class_probs=initial_probs,
            feature=np.random.randn(feature_dim),
            raw_class_probs=initial_probs
        )
        manager.create_track(d)
    
    print(f"   Manager: {manager.get_summary()}")
    
    # Test 5: Data association
    print("\n5. Testing data association:")
    track_boxes = np.array([[100, 100, 150, 160], [200, 100, 250, 160]])
    det_boxes = np.array([[105, 105, 155, 165], [300, 100, 350, 160]])
    
    matches, unmatched_t, unmatched_d = associate(
        track_boxes, det_boxes, method='hungarian',
        config=AssociationConfig(iou_threshold=0.3)
    )
    print(f"   Matches: {matches}")
    print(f"   Unmatched tracks: {unmatched_t}")
    print(f"   Unmatched detections: {unmatched_d}")
    
    print("\n✓ All tests passed!")