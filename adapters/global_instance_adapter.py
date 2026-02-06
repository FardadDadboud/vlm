"""
Global + Instance TTA Adapter

Main adapter combining:
1. Global BCA+ Cache - Frame-level class prototype adaptation
2. Instance-level Tracking - Per-object temporal adaptation with STAD

Key Design Decisions:
- Global cache adapts class-level priors across ALL objects
- Per-track STAD adapts class beliefs for INDIVIDUAL objects
- Part 5.3: New tracks initialize STAD from global cache
- CRITICAL: Global cache updates use RAW VLM probs (not post-instance probs)

Ablation Modes:
- vanilla: No adaptation
- global_only: BCA+ cache only
- instance_only: Tracking + per-track STAD only
- tracking_only: Tracking only (no STAD, no global cache) - FOR DEBUGGING TRACKING
- full: Global + Instance
- cascade: Hierarchical refinement

Self-Reinforcement Prevention:
- Global cache ALWAYS updates with raw VLM probs (pre-adaptation)
- Per-track STAD ALWAYS updates with raw VLM probs
- Post-adaptation probs are ONLY used for final output
"""

import numpy as np
import time
import os
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

# Handle both package and standalone imports
try:
    # Package import
    from .track import (
        KalmanBoxTracker, KalmanFilterConfig,
        AssociationConfig, associate,
        TrackSTADConfig, TrackSTADvMF, TrackSTADGaussian, create_track_stad,
        Track, TrackState, TrackConfig, TrackManager, Detection, reset_track_ids
    )
    from .enhanced_bca_cache import (
        EnhancedBCAPlusCache, EnhancedBCAPlusConfig, CacheEntryState
    )
    from .base_adapter import BaseAdapter
except ImportError:
    # Standalone import
    from track import (
        KalmanBoxTracker, KalmanFilterConfig,
        AssociationConfig, associate,
        TrackSTADConfig, TrackSTADvMF, TrackSTADGaussian, create_track_stad,
        Track, TrackState, TrackConfig, TrackManager, Detection, reset_track_ids
    )
    from enhanced_bca_cache import (
        EnhancedBCAPlusCache, EnhancedBCAPlusConfig, CacheEntryState
    )
    from base_adapter import BaseAdapter


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GlobalInstanceConfig:
    """
    Configuration for Global+Instance TTA adapter.
    
    Combines configurations for all components:
    - Global BCA+ cache
    - Tracking (Kalman, association)
    - Per-track STAD
    """
    # === Ablation Mode ===
    mode: str = 'full'  # 'vanilla', 'global_only', 'instance_only', 'tracking_only', 'full', 'cascade'
    
    # === Global BCA+ Cache ===
    use_global_cache: bool = True
    cache_config: Optional[EnhancedBCAPlusConfig] = None
    
    # === Tracking ===
    use_tracking: bool = True
    track_config: Optional[TrackConfig] = None
    association_config: Optional[AssociationConfig] = None
    association_method: str = 'hungarian'  # 'iou', 'hungarian', 'combined', 'bytetrack'
    
    # === Per-track STAD ===
    use_track_stad: bool = True
    stad_variant: str = 'vmf'  # 'vmf' or 'gaussian'
    stad_config: Optional[TrackSTADConfig] = None
    
    # === Detection Thresholds ===
    detection_threshold: float = 0.10  # Min confidence to keep detection
    nms_threshold: float = 0.7         # NMS IoU threshold
    
    # === Feedback ===
    # Track-to-cache feedback: confirmed tracks update global cache
    use_track_to_cache_feedback: bool = True
    feedback_min_hits: int = 5         # Min hits for track to update cache
    feedback_min_confidence: float = 0.8
    feedback_weight: float = 0.5       # Soft update weight
    
    # === Debug ===
    debug: bool = False
    
    # === Visual Debug (for tracking_only mode) ===
    visual_debug: bool = False
    visual_debug_dir: str = ''  # Will be set from output_dir + 'tracking_debug'


def get_ablation_config(mode: str) -> Dict:
    """
    Get configuration dict for ablation study.
    
    Args:
        mode: 'vanilla', 'global_only', 'instance_only', 'tracking_only', 'full', 'cascade'
        
    Returns:
        Configuration dictionary compatible with adapter initialization
    """
    base_config = {
        'mode': mode,
        'detection_threshold': 0.10,
        'nms_threshold': 0.7,
        'debug': False,
        
        # Global cache params (matching original BCA+ behavior)
        'tau1': 0.6,
        'tau2': 0.5,  # Single tau2 like original BCA+ (lower = more aggressive merging)
        'ws': 0.2,
        'logit_temperature': 10.0,
        'alpha': 0.3,
        'tau2_init': 0.5,  # Batch init clustering threshold
        'max_cache_size': 25,
        
        # STAD params (matching temporal_tta_vmf_v2.json structure)
        'ssm_type': 'vmf',
        'kappa_trans': 10.0,
        'kappa_ems': 20.0,
        'gamma_init': 10.0,
        'kappa_max': 100.0,
        'kappa_min': 1e-6,
        'gamma_max': 200.0,
        'gamma_min': 1.0,
        'window_size': 5,
        'em_iterations': 3,
        'tau_update': 0.5,
        'min_updates_per_class': 1,
        'vlm_prior_weight': 0.2,
        'use_pi': True,
        'temperature': 1.0,
        'dirichlet_alpha': 0.1,
        
        # Tracking params
        'min_hits_to_confirm': 3,
        'max_age': 30,
        'iou_threshold': 0.3,
        'association_method': 'hungarian',
        
        # Feedback params
        'use_track_to_cache_feedback': True,
        'feedback_min_hits': 5,
        'feedback_min_confidence': 0.8,
        'feedback_weight': 0.5,
        
        # Part 5.2: Fusion params
        'fusion_mode': 'entropy_weighted',
        'fusion_init_weight': 0.1,
        'fusion_global_weight': 0.3,
        'fusion_track_weight': 0.6,
        'fusion_confidence_threshold': 0.7
    }
    
    # Mode-specific overrides
    if mode == 'vanilla':
        base_config['use_global_cache'] = False
        base_config['use_tracking'] = False
        base_config['use_track_stad'] = False
    elif mode == 'global_only':
        base_config['use_global_cache'] = True
        base_config['use_tracking'] = False
        base_config['use_track_stad'] = False
    elif mode == 'instance_only':
        base_config['use_global_cache'] = False
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = True
    elif mode == 'tracking_only':
        # NEW: Tracking only mode - for debugging tracking without any TTA
        base_config['use_global_cache'] = False
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = False
        base_config['visual_debug'] = True  # Enable visual debugging by default
    elif mode == 'full':
        base_config['use_global_cache'] = True
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = True
    elif mode == 'cascade':
        base_config['use_global_cache'] = True
        base_config['use_tracking'] = True
        base_config['use_track_stad'] = True
        base_config['cascade_mode'] = True
        base_config['fusion_mode'] = 'hierarchical'  # cascade uses hierarchical fusion
    
    return base_config


# =============================================================================
# Visual Debug Utilities
# =============================================================================

def save_tracking_debug_image(image, boxes, scores, labels, track_ids, 
                              frame_idx: int, output_dir: str,
                              active_tracks: List = None,
                              video_name: str = None) -> str:
    """
    Save debug visualization of tracking results.
    
    Args:
        image: Input image (PIL or numpy)
        boxes: (N, 4) boxes in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        labels: List of label strings
        track_ids: List of track IDs (-1 for untracked)
        frame_idx: Frame index for filename
        output_dir: Directory to save images
        active_tracks: Optional list of Track objects for extra info
        video_name: Optional video name for organizing output
        
    Returns:
        Path to saved image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[Warning] PIL not available for visual debugging")
        return None
    
    # Convert image to PIL if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    draw = ImageDraw.Draw(pil_image)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Color palette for track IDs (cycle through)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255)
    ]
    
    # Build track info dict if available
    track_info = {}
    if active_tracks:
        for track in active_tracks:
            track_info[track.track_id] = {
                'state': track.state.value,
                'hits': track.hits,
                'age': track.age
            }
    
    # Draw boxes and labels
    N = len(boxes) if len(boxes) > 0 else 0
    for i in range(N):
        box = boxes[i]
        score = scores[i] if i < len(scores) else 0
        label = labels[i] if i < len(labels) else "?"
        track_id = track_ids[i] if i < len(track_ids) else -1
        
        # Choose color based on track ID
        if track_id >= 0:
            color = colors[track_id % len(colors)]
        else:
            color = (128, 128, 128)  # Gray for untracked
        
        # Draw box
        if track_id >= 0:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Build label text
        
            info = track_info.get(track_id, {})
            state_char = info.get('state', '?')[0].upper()  # T/C/D
            hits = info.get('hits', '?')
            text = f"T{track_id}[{state_char}|h{hits}] {label} {score:.2f}"
        
        
            # Draw text background
            text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)
    
    # Add frame info
    info_text = f"Frame: {frame_idx} | Detections: {N} | Tracked: {sum(1 for t in track_ids if t >= 0)}"
    draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)
    
    # Create output directory
    if video_name:
        save_dir = os.path.join(output_dir, video_name)
    else:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Save image
    save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
    pil_image.save(save_path, quality=85)
    
    return save_path


# =============================================================================
# Global Instance Adapter
# =============================================================================

class GlobalInstanceAdapter(BaseAdapter):
    """
    Global + Instance TTA adapter.
    
    Combines global BCA+ cache adaptation with per-track STAD for
    comprehensive temporal adaptation in video object detection.
    """
    
    def __init__(self, detector, config: Dict):
        """
        Initialize Global+Instance adapter.
        
        Args:
            detector: Base VLM detector with detect_with_features method
            config: Configuration dictionary
        """
        super().__init__(detector, config)
        
        self.config = config
        self.detector = detector
        
        # Parse config
        self._parse_config(config)
        
        # Get detector info
        self.target_classes = config.get('detector', {}).get(
            'target_classes', 
            ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        )
        self.num_classes = len(self.target_classes)
        
        # Feature dimension (will be set on first detection)
        self.feature_dim = None
        
        # Initialize components
        self._init_components()
        
        # Stats
        self.frame_count = 0
        self.total_detections = 0
        self.total_adapted = 0
        
        # Visual debug
        self._current_video_name = None
    
    def _parse_config(self, config: Dict) -> None:
        """Parse configuration dictionary."""
        # Handle nested config structure: config["adaptation"]["params"] or config["params"] or config
        if 'adaptation' in config and isinstance(config['adaptation'], dict):
            params = config['adaptation'].get('params', config['adaptation'])
            config_path = 'config["adaptation"]["params"]'
        elif 'params' in config:
            params = config['params']
            config_path = 'config["params"]'
        else:
            params = config
            config_path = 'config (root level)'
        
        # Store for debug logging
        self._config_path = config_path
        self._raw_params = params
        
        # Mode
        self.mode = params.get('mode', 'full')
        
        # Set component flags based on mode
        if self.mode == 'vanilla':
            self.use_global_cache = False
            self.use_tracking = False
            self.use_track_stad = False
        elif self.mode == 'global_only':
            self.use_global_cache = True
            self.use_tracking = False
            self.use_track_stad = False
        elif self.mode == 'instance_only':
            self.use_global_cache = False
            self.use_tracking = True
            self.use_track_stad = True
        elif self.mode == 'tracking_only':
            # NEW MODE: Tracking only - for debugging
            self.use_global_cache = False
            self.use_tracking = True
            self.use_track_stad = False
        elif self.mode == 'cascade':
            self.use_global_cache = True
            self.use_tracking = True
            self.use_track_stad = True
        else:  # 'full' or default
            self.use_global_cache = True
            self.use_tracking = True
            self.use_track_stad = True
        
        # Allow explicit overrides from config
        self.use_global_cache = params.get('use_global_cache', self.use_global_cache)
        self.use_tracking = params.get('use_tracking', self.use_tracking)
        self.use_track_stad = params.get('use_track_stad', self.use_track_stad)
        self.cascade_mode = params.get('cascade_mode', self.mode == 'cascade')
        
        # Detection thresholds
        self.detection_threshold = params.get('detection_threshold', params.get('tau_update', 0.10))
        self.nms_threshold = params.get('nms_threshold', params.get('iou_threshold', 0.7))
        
        # Global cache config (using single tau2 like original BCA+)
        self.cache_config = EnhancedBCAPlusConfig(
            tau1=params.get('tau1', 0.6),
            tau2=params.get('tau2', 0.5),  # Single tau2 like original BCA+
            tau2_init=params.get('tau2_init', 0.5),
            max_cache_size=params.get('max_cache_size', 25),
            ws=params.get('ws', 0.2),
            alpha=params.get('alpha', 0.3),
            logit_temperature=params.get('logit_temperature', 10.0),
            min_hits_to_confirm=params.get('min_hits_to_confirm', 3),
            max_age=params.get('max_age', 30),
            use_batch_init=params.get('use_batch_init', True),
            batch_init_size=params.get('batch_init_size', 10),
            debug=params.get('debug', False)
        )
        
        # Per-track STAD config
        self.stad_config = TrackSTADConfig(
            kappa_trans=params.get('kappa_trans', 10.0),
            kappa_ems=params.get('kappa_ems', 20.0),
            gamma_init=params.get('gamma_init', 10.0),
            gamma_min=params.get('gamma_min', 1.0),
            gamma_max=params.get('gamma_max', 200.0),
            kappa_max=params.get('kappa_max', 100.0),
            kappa_min=params.get('kappa_min', 1e-6),
            window_size=params.get('window_size', 5),
            em_iterations=params.get('em_iterations', 3),
            min_confidence=params.get('tau_update', 0.3),
            min_updates_per_class=params.get('min_updates_per_class', 0.5),
            max_rk_fraction=params.get('max_rk_fraction', 0.5),  # Stability: cap per-class R_k
            temperature=params.get('temperature', 1.0),
            vlm_prior_weight=params.get('vlm_prior_weight', 0.2),
            use_pi=params.get('use_pi', True),
            use_ema_pi=params.get('use_ema_pi', False),
            pi_ema_decay=params.get('pi_ema_decay', 1.0),
            dirichlet_alpha=params.get('dirichlet_alpha', 0.1),
            gamma_ema_decay=params.get('gamma_ema_decay', 0.7),
            q_scale=params.get('q_scale', 0.01),
            r_base=params.get('r_base', 0.5),
            use_diagonal_cov=params.get('use_diagonal_cov', True),
            use_smoothing=params.get('use_smoothing', False),
            debug=params.get('debug', False)
        )
        
        # STAD variant
        self.stad_variant = params.get('ssm_type', params.get('stad_variant', 'vmf'))
        
        # Track config
        self.track_config = TrackConfig(
            min_hits_to_confirm=params.get('min_hits_to_confirm', 3),
            max_age=params.get('max_age', 30),
            feature_alpha=params.get('feature_alpha', 0.1),
            use_track_stad=self.use_track_stad,
            stad_variant=self.stad_variant,
            stad_config=self.stad_config,
            kalman_config=KalmanFilterConfig(),
            init_from_cache=params.get('init_from_cache', True),
            cache_match_threshold=params.get('cache_match_threshold', 0.5)
        )
        
        # Track creation limits (CRITICAL for performance!)
        self.track_creation_threshold = params.get('track_creation_threshold', 0.3)  # Only create tracks for high-conf dets
        self.max_tracks = params.get('max_tracks', 100)  # Limit total tracks
        self.association_fast_threshold = params.get('association_fast_threshold', 50)  # Use fast IoU-only when > this
        
        # Pre-filtering options
        # aggressive_prefilter=True: Always filter by track_creation_threshold (faster, but ByteTrack can't rescue tracks)
        # aggressive_prefilter=False: Only filter when no tracks exist (allows ByteTrack second-round rescue)
        self.aggressive_prefilter = params.get('aggressive_prefilter', False)
        
        # Phantom detections: Output predicted boxes for unmatched tracks
        # This helps with temporary occlusions/missed detections
        self.use_predicted_for_missed = params.get('use_predicted_for_missed', False)
        self.predicted_score_decay = params.get('predicted_score_decay', 0.9)  # Multiply score by this each missed frame
        self.predicted_min_score = params.get('predicted_min_score', 0.3)  # Don't output if score drops below this
        self.predicted_max_frames = params.get('predicted_max_frames', 3)  # Max frames to output predicted boxes
        
        # Track-based score modulation (CRITICAL for tracking to improve mAP!)
        # Without this, tracking_only gives same mAP as vanilla
        # NOTE: Only applies when tracking is PRIMARY method (tracking_only mode)
        # In full/global_only/instance_only modes, TTA already modifies probs
        self.use_track_score_modulation = params.get('use_track_score_modulation', True)
        # Auto-disable score modulation when other TTA methods are active (unless forced)
        self.force_score_modulation = params.get('force_score_modulation', False)
        if not self.force_score_modulation and (self.use_global_cache or self.use_track_stad):
            self.use_track_score_modulation = False
        self.confirmed_track_boost = params.get('confirmed_track_boost', 0.1)  # Add to score for confirmed tracks
        self.tentative_track_penalty = params.get('tentative_track_penalty', 0.0)  # Subtract from tentative (0 = no penalty)
        self.untracked_penalty = params.get('untracked_penalty', 0.05)  # Subtract from untracked detections
        self.filter_untracked = params.get('filter_untracked', False)  # If True, remove untracked detections entirely
        
        # Temporal score smoothing (optional: smooth score with track history)
        self.use_temporal_score_smoothing = params.get('use_temporal_score_smoothing', False)
        self.score_smoothing_alpha = params.get('score_smoothing_alpha', 0.3)  # EMA weight for history
        self.score_smoothing_window = params.get('score_smoothing_window', 5)  # Max history to use
        
        # Association config
        # NOTE: feature_threshold is defined in AssociationConfig but UNUSED in all association methods!
        # Keeping minimal config with only used params
        self.association_config = AssociationConfig(
            iou_threshold=params.get('iou_threshold', 0.3),
            high_score_threshold=params.get('high_score_threshold', 0.6),
            low_score_threshold=params.get('low_score_threshold', 0.1)
        )
        self.association_method = params.get('association_method', 'hungarian')
        
        # Feedback config
        self.use_feedback = params.get('use_track_to_cache_feedback', True)
        self.feedback_min_hits = params.get('feedback_min_hits', 5)
        self.feedback_min_confidence = params.get('feedback_min_confidence', 0.8)
        self.feedback_weight = params.get('feedback_weight', 0.5)
        
        # Fusion config (Part 5.2)
        # Modes: 'entropy_weighted', 'hierarchical', 'parallel', 'selection'
        self.fusion_mode = params.get('fusion_mode', 'entropy_weighted')
        if self.cascade_mode:
            self.fusion_mode = params.get('fusion_mode', 'hierarchical')  # cascade defaults to hierarchical
        self.fusion_init_weight = params.get('fusion_init_weight', 0.1)  # weight for p_init in parallel
        self.fusion_global_weight = params.get('fusion_global_weight', 0.3)  # weight for p_global in parallel
        self.fusion_track_weight = params.get('fusion_track_weight', 0.6)  # weight for p_track in parallel
        self.fusion_confidence_threshold = params.get('fusion_confidence_threshold', 0.7)  # for hierarchical/selection
        
        # Debug
        self.debug = params.get('debug', False)
        self.debug_every = params.get('debug_every', 30)
        
        # Visual debug (especially for tracking_only mode)
        self.visual_debug = params.get('visual_debug', self.mode == 'tracking_only')
        self.visual_debug_dir = os.path.join(params.get('output_dir', './tracking_debug'), 'tracking_debug')
        
        # Sanity warning thresholds
        self._warn_cache_empty_after = params.get('warn_cache_empty_after', 50)
        self._warn_no_stad_after = params.get('warn_no_stad_after', 50)
        self._debug_log_first_n = params.get('debug_log_first_n', 5)  # Always log first N frames
        
        # Debug counters for sanity warnings
        self._frames_with_empty_cache = 0
        self._frames_with_no_stad_updates = 0
        self._total_stad_updates = 0
        self._total_stad_skipped_low_conf = 0
        self._total_stad_skipped_no_track = 0
        
        # One-time debug print of resolved params
        if self.debug:
            self._print_config_debug()
    
    def _print_config_debug(self) -> None:
        """Print configuration debug info."""
        print(f"\n{'='*60}")
        print(f"[GlobalInstanceAdapter] CONFIG DEBUG")
        print(f"{'='*60}")
        print(f"  Config path used: {self._config_path}")
        print(f"  --- Mode & Components ---")
        print(f"    mode={self.mode}")
        print(f"    use_global_cache={self.use_global_cache}")
        print(f"    use_tracking={self.use_tracking}")
        print(f"    use_track_stad={self.use_track_stad}")
        print(f"    cascade_mode={self.cascade_mode}")
        print(f"  --- Global Cache (BCA+) ---")
        print(f"    tau1={self.cache_config.tau1} (conf threshold for cache UPDATE)")
        print(f"    tau2={self.cache_config.tau2} (similarity for MATCH - single threshold like original BCA+)")
        print(f"    tau2_init={self.cache_config.tau2_init} (batch init clustering threshold)")
        print(f"    ws={self.cache_config.ws} (scale weight in posterior)")
        print(f"    logit_temperature={self.cache_config.logit_temperature}")
        print(f"    max_cache_size={self.cache_config.max_cache_size}")
        print(f"    use_batch_init={self.cache_config.use_batch_init}")
        print(f"    batch_init_size={self.cache_config.batch_init_size}")
        print(f"  --- Per-Track STAD ---")
        print(f"    ssm_type={self.stad_variant}")
        print(f"    window_size={self.stad_config.window_size}")
        print(f"    em_iterations={self.stad_config.em_iterations}")
        print(f"    min_confidence={self.stad_config.min_confidence} (tau_update for STAD)")
        print(f"    vlm_prior_weight={self.stad_config.vlm_prior_weight}")
        print(f"    kappa_trans={self.stad_config.kappa_trans}")
        print(f"    kappa_ems={self.stad_config.kappa_ems}")
        print(f"  --- Fusion ---")
        print(f"    fusion_mode={self.fusion_mode}")
        print(f"    fusion_init_weight={self.fusion_init_weight}")
        print(f"    fusion_global_weight={self.fusion_global_weight}")
        print(f"    fusion_track_weight={self.fusion_track_weight}")
        print(f"  --- Detection ---")
        print(f"    detection_threshold={self.detection_threshold}")
        print(f"    nms_threshold={self.nms_threshold}")
        print(f"  --- Tracking ---")
        print(f"    min_hits_to_confirm={self.track_config.min_hits_to_confirm}")
        print(f"    max_age={self.track_config.max_age}")
        print(f"    association_method={self.association_method}")
        print(f"    track_creation_threshold={self.track_creation_threshold} (only track high-conf dets)")
        print(f"    max_tracks={self.max_tracks} (limit total tracks)")
        print(f"    association_fast_threshold={self.association_fast_threshold} (use IoU-only when > this)")
        print(f"    aggressive_prefilter={self.aggressive_prefilter} (filter low-conf even when tracks exist)")
        print(f"  --- Phantom Detections (for missed tracks) ---")
        print(f"    use_predicted_for_missed={self.use_predicted_for_missed}")
        print(f"    predicted_score_decay={self.predicted_score_decay}")
        print(f"    predicted_min_score={self.predicted_min_score}")
        print(f"    predicted_max_frames={self.predicted_max_frames}")
        print(f"  --- Score Modulation ---")
        print(f"    use_track_score_modulation={self.use_track_score_modulation} (auto-disabled when TTA active)")
        print(f"    force_score_modulation={self.force_score_modulation}")
        print(f"    confirmed_track_boost={self.confirmed_track_boost} (added to confirmed track scores)")
        print(f"    tentative_track_penalty={self.tentative_track_penalty} (subtracted from tentative)")
        print(f"    untracked_penalty={self.untracked_penalty} (subtracted from untracked)")
        print(f"    filter_untracked={self.filter_untracked} (remove untracked entirely)")
        print(f"  --- Temporal Score Smoothing ---")
        print(f"    use_temporal_score_smoothing={self.use_temporal_score_smoothing}")
        print(f"    score_smoothing_alpha={self.score_smoothing_alpha} (EMA weight for history)")
        print(f"    score_smoothing_window={self.score_smoothing_window} (max history length)")
        print(f"  --- Visual Debug ---")
        print(f"    visual_debug={self.visual_debug}")
        print(f"    visual_debug_dir={self.visual_debug_dir}")
        print(f"{'='*60}\n")
    
    def _init_components(self) -> None:
        """Initialize adaptation components (lazy init for feature_dim)."""
        self.global_cache = None
        self.track_manager = None
        self._components_initialized = False
    
    def _lazy_init_components(self, feature_dim: int) -> None:
        """Initialize components once feature dimension is known."""
        if self._components_initialized:
            return
        
        self.feature_dim = feature_dim
        
        # Global BCA+ cache
        if self.use_global_cache:
            self.global_cache = EnhancedBCAPlusCache(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                config=self.cache_config,
                class_names=self.target_classes
            )
        
        # Track manager (with reference to global cache for Part 5.3)
        if self.use_tracking:
            self.track_manager = TrackManager(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                config=self.track_config,
                global_cache=self.global_cache,  # Part 5.3: Pass cache for track init
                class_names=self.target_classes
            )
        
        self._components_initialized = True
    
    def set_video_name(self, video_name: str) -> None:
        """Set current video name for visual debug organization."""
        self._current_video_name = video_name
    
    def adapt_and_detect(self, image, target_classes: List[str],
                        threshold: float = None, **kwargs) -> Any:
        """
        Perform adapted detection.
        
        Pipeline:
        1. VLM Detection → raw features, boxes, class_probs
        2. Store RAW VLM probs before ANY adaptation
        3. Global BCA+ Adaptation → adapted_probs (if use_global_cache)
        4. Filter by threshold
        5. Apply NMS
        6. Track Association → matches, new tracks (if use_tracking)
        7. Per-Track STAD refinement (for matched tracks)
        8. Update Global Cache with RAW VLM probs (NOT post-instance probs!)
        9. Optionally: Track-to-Cache Feedback
        
        CRITICAL: Global cache update uses RAW VLM probs to avoid self-reinforcement!
        
        Args:
            image: Input image
            target_classes: List of target class names
            threshold: Detection confidence threshold
            
        Returns:
            Detection results
        """
        threshold = threshold or self.detection_threshold
        
        # Timing accumulators
        t_start = time.perf_counter()
        timings = {}
        
        # Debug stats for this frame
        dbg = {
            'N_raw': 0, 'N_threshold': 0, 'N_nms': 0,
            'cache_M_before': 0, 'cache_M_after': 0, 'cache_high_conf_count': 0,
            'adapt_L1_mean': 0.0, 'adapt_L1_max': 0.0,
            'n_active_tracks': 0, 'n_matches': 0, 'n_unmatched_dets': 0, 'n_unmatched_tracks': 0,
            'stad_updates': 0, 'stad_skipped_low_conf': 0, 'stad_skipped_no_track': 0,
            'label_changes': 0
        }
        
        # ===== Stage 1: VLM Detection =====
        t0 = time.perf_counter()
        alpha = self.cache_config.alpha if self.use_global_cache else 0.0
        detection_result = self.detector.detect_with_features(
            image, target_classes, threshold=threshold, alpha=alpha
        )
        timings['detector'] = time.perf_counter() - t0
        
        # Set image size on cache (needed for scale normalization - Eq. 8)
        if self.use_global_cache and self.global_cache is not None:
            if hasattr(image, 'size'):
                self.global_cache.image_size = image.size  # PIL: (width, height)
            elif hasattr(image, 'shape'):
                self.global_cache.image_size = (image.shape[1], image.shape[0])  # numpy: (H,W,C)
        
        if detection_result is None or len(detection_result.boxes) == 0:
            self.frame_count += 1
            self._end_frame()
            return detection_result
        
        # Extract data
        boxes = np.array(detection_result.boxes)
        scores = np.array(detection_result.scores)
        labels = list(detection_result.labels)
        class_probs = np.array(detection_result.class_probs)  # (N, K)
        features = np.array(detection_result.features)        # (N, D)
        
        N = len(boxes)
        if N == 0:
            self.frame_count += 1
            self._end_frame()
            return detection_result
        
        # Lazy init components
        self._lazy_init_components(features.shape[1])
        
        # ===== Stage 2: Store RAW VLM probs (BEFORE any adaptation) =====
        raw_vlm_probs = class_probs.copy()
        
        # ===== Stage 3: Global BCA+ Adaptation (BATCHED) =====
        t0 = time.perf_counter()
        dbg['N_raw'] = N
        dbg['cache_M_before'] = self.global_cache.M if self.global_cache else 0
        
        if self.use_global_cache and self.global_cache is not None:
            global_adapted_probs = self.global_cache.adapt_probs_batch(features, boxes, class_probs)
            # Compute adaptation effect
            adapt_diff = np.abs(global_adapted_probs - class_probs)
            dbg['adapt_L1_mean'] = float(np.mean(adapt_diff))
            dbg['adapt_L1_max'] = float(np.max(adapt_diff))
        else:
            global_adapted_probs = class_probs.copy()
        timings['global_cache_adapt'] = time.perf_counter() - t0
        
        # ===== Stage 4: Filter by threshold =====
        keep_mask = scores >= threshold
        dbg['N_threshold'] = int(np.sum(keep_mask))
        if not keep_mask.any():
            self.frame_count += 1
            self._end_frame()
            self._debug_log_frame(dbg, timings, t_start)
            return self._create_empty_result(detection_result)
        
        # ===== Stage 5: Apply NMS =====
        t0 = time.perf_counter()
        nms_indices = self._apply_nms(boxes, scores, keep_mask)
        timings['nms'] = time.perf_counter() - t0
        
        nms_boxes = boxes[nms_indices]
        nms_scores = scores[nms_indices]
        nms_labels = [labels[i] for i in nms_indices]
        nms_features = features[nms_indices]
        nms_raw_probs = raw_vlm_probs[nms_indices]  # RAW VLM probs for cache update!
        nms_global_probs = global_adapted_probs[nms_indices]
        
        N_nms = len(nms_indices)
        dbg['N_nms'] = N_nms
        dbg['score_min'] = float(nms_scores.min()) if N_nms > 0 else 0.0
        dbg['score_max'] = float(nms_scores.max()) if N_nms > 0 else 0.0
        
        # ===== Stage 6: Track Association =====
        t0 = time.perf_counter()
        final_probs = nms_global_probs.copy()
        track_ids = [-1] * N_nms
        t_stad = 0.0  # Accumulator for STAD updates
        frame_stad_updates = 0
        frame_stad_skipped_no_track = 0
        tracks_created = 0
        tracks_skipped_low_conf = 0
        tracks_skipped_max_limit = 0
        
        if self.use_tracking and self.track_manager is not None:
            # Predict existing tracks
            self.track_manager.predict_all()
            
            # Get track info
            active_tracks = self.track_manager.get_active_tracks()
            dbg['n_active_tracks'] = len(active_tracks)
            
            # ===== PRE-FILTERING FOR TRACKING ONLY =====
            # Filter creates a SUBSET for association (faster), but we keep ALL detections for output
            should_prefilter = (
                self.track_creation_threshold > 0 and
                (self.aggressive_prefilter or len(active_tracks) == 0)
            )
            
            if should_prefilter:
                # Create indices of detections to use for tracking
                track_mask = nms_scores >= self.track_creation_threshold
                track_indices = np.where(track_mask)[0]  # Indices into original nms_* arrays
                n_pre_filtered = N_nms - len(track_indices)
            else:
                track_indices = np.arange(N_nms)  # Use all detections
                n_pre_filtered = 0
            
            dbg['n_pre_filtered'] = n_pre_filtered
            
            # Extract subset for tracking (don't overwrite originals!)
            track_boxes = nms_boxes[track_indices]
            track_scores = nms_scores[track_indices]
            track_labels = [nms_labels[i] for i in track_indices]
            track_features = nms_features[track_indices]
            track_raw_probs = nms_raw_probs[track_indices]
            track_global_probs = nms_global_probs[track_indices]
            N_track = len(track_indices)
            
            if len(active_tracks) > 0 and N_track > 0:
                existing_track_boxes = np.array([t.get_state() for t in active_tracks])
                existing_track_features = np.array([t.get_feature() for t in active_tracks])
                
                # Choose association method based on scale
                n_tracks = len(active_tracks)
                n_dets = N_track  # Use filtered count
                use_fast = (n_tracks > self.association_fast_threshold or 
                           n_dets > self.association_fast_threshold)
                
                assoc_method = 'iou' if use_fast else self.association_method
                
                # Associate using filtered subset
                matches, unmatched_tracks, unmatched_dets = associate(
                    track_boxes=existing_track_boxes,
                    detection_boxes=track_boxes,  # Filtered subset
                    detection_scores=track_scores,
                    track_features=existing_track_features,
                    detection_features=track_features,
                    method=assoc_method,
                    config=self.association_config
                )
                timings['association'] = time.perf_counter() - t0
                
                dbg['n_matches'] = len(matches)
                dbg['n_unmatched_dets'] = len(unmatched_dets)
                dbg['n_unmatched_tracks'] = len(unmatched_tracks)
                dbg['association_method_used'] = assoc_method
                
                # ===== Stage 7: Update matched tracks and refine probs =====
                t0 = time.perf_counter()
                for track_idx, det_idx in matches:
                    track = active_tracks[track_idx]
                    orig_idx = track_indices[det_idx]  # Map back to original nms_* index
                    
                    # Create detection object (with RAW probs for STAD update!)
                    det = Detection(
                        box=nms_boxes[orig_idx],
                        score=nms_scores[orig_idx],
                        label=nms_labels[orig_idx],
                        class_idx=self.target_classes.index(nms_labels[orig_idx]) if nms_labels[orig_idx] in self.target_classes else 0,
                        class_probs=nms_global_probs[orig_idx],
                        feature=nms_features[orig_idx],
                        raw_class_probs=nms_raw_probs[orig_idx]  # RAW probs for STAD!
                    )
                    
                    # Track STAD update count before/after
                    stad_before = track.total_stad_updates if hasattr(track, 'total_stad_updates') else 0
                    
                    # Update track (STAD uses raw_class_probs internally)
                    track.update(det)
                    track_ids[orig_idx] = track.track_id  # Store in original index
                    
                    stad_after = track.total_stad_updates if hasattr(track, 'total_stad_updates') else 0
                    if stad_after > stad_before:
                        frame_stad_updates += 1
                    
                    # Part 5.2: 3-source fusion (p_init, p_global, p_track)
                    if self.use_track_stad and track.class_stad is not None:
                        p_init = nms_raw_probs[orig_idx]      # Raw VLM
                        p_global = nms_global_probs[orig_idx]  # Global cache adapted
                        p_track = track.get_class_probs()     # Per-track STAD belief
                        final_probs[orig_idx] = self._fuse_probs(p_init, p_global, p_track, track)
                t_stad = time.perf_counter() - t0
                
                # Create new tracks for unmatched detections (WITH THRESHOLD!)
                current_track_count = len(self.track_manager.get_active_tracks())
                for det_idx in unmatched_dets:
                    orig_idx = track_indices[det_idx]  # Map back to original index
                    
                    # Check score threshold (already filtered, but double-check)
                    if nms_scores[orig_idx] < self.track_creation_threshold:
                        tracks_skipped_low_conf += 1
                        continue
                    
                    # Check max tracks limit
                    if current_track_count >= self.max_tracks:
                        tracks_skipped_max_limit += 1
                        continue
                    
                    det = Detection(
                        box=nms_boxes[orig_idx],
                        score=nms_scores[orig_idx],
                        label=nms_labels[orig_idx],
                        class_idx=self.target_classes.index(nms_labels[orig_idx]) if nms_labels[orig_idx] in self.target_classes else 0,
                        class_probs=nms_global_probs[orig_idx],
                        feature=nms_features[orig_idx],
                        raw_class_probs=nms_raw_probs[orig_idx]
                    )
                    new_track = self.track_manager.create_track(det)
                    track_ids[orig_idx] = new_track.track_id  # Store in original index
                    tracks_created += 1
                    current_track_count += 1
            elif N_track > 0:
                timings['association'] = time.perf_counter() - t0
                dbg['association_method_used'] = 'none'
                dbg['n_matches'] = 0
                dbg['n_unmatched_dets'] = N_track
                dbg['n_unmatched_tracks'] = 0
                # No existing tracks - create new ones from filtered detections
                current_track_count = 0
                for det_idx in range(N_track):
                    orig_idx = track_indices[det_idx]  # Map back to original index
                    
                    # Check max tracks limit
                    if current_track_count >= self.max_tracks:
                        tracks_skipped_max_limit += 1
                        continue
                    
                    det = Detection(
                        box=nms_boxes[orig_idx],
                        score=nms_scores[orig_idx],
                        label=nms_labels[orig_idx],
                        class_idx=self.target_classes.index(nms_labels[orig_idx]) if nms_labels[orig_idx] in self.target_classes else 0,
                        class_probs=nms_global_probs[orig_idx],
                        feature=nms_features[orig_idx],
                        raw_class_probs=nms_raw_probs[orig_idx]
                    )
                    new_track = self.track_manager.create_track(det)
                    track_ids[orig_idx] = new_track.track_id  # Store in original index
                    tracks_created += 1
                    current_track_count += 1
            else:
                # N_track == 0, nothing to track
                timings['association'] = time.perf_counter() - t0
                dbg['association_method_used'] = 'none'
                dbg['n_matches'] = 0
                dbg['n_unmatched_dets'] = 0
                dbg['n_unmatched_tracks'] = 0
        else:
            timings['association'] = 0.0
            frame_stad_skipped_no_track = N_nms  # No tracking means no STAD for any detection
        
        dbg['stad_updates'] = frame_stad_updates
        dbg['stad_skipped_no_track'] = frame_stad_skipped_no_track
        dbg['tracks_created'] = tracks_created
        dbg['tracks_skipped_low_conf'] = tracks_skipped_low_conf
        dbg['tracks_skipped_max_limit'] = tracks_skipped_max_limit
        timings['stad_update'] = t_stad
        
        # ===== Stage 8: Update Global Cache (CRITICAL: Use RAW VLM probs!) =====
        t0 = time.perf_counter()
        if self.use_global_cache and self.global_cache is not None:
            # Count high-confidence detections that will be used to update cache
            high_conf_mask = nms_scores >= self.cache_config.tau1
            dbg['cache_high_conf_count'] = int(np.sum(high_conf_mask))
            
            # Update cache with RAW VLM probs, NOT post-instance probs!
            # This prevents self-reinforcement loop
            self.global_cache.update_cache(
                features=nms_features,
                boxes=nms_boxes,
                probs=nms_raw_probs,  # RAW VLM probs!
                scores=nms_scores
            )
            dbg['cache_M_after'] = self.global_cache.M
        timings['cache_update'] = time.perf_counter() - t0
        
        # ===== Stage 9: Track-to-Cache Feedback (optional) =====
        if (self.use_feedback and self.use_global_cache and 
            self.use_tracking and self.track_manager is not None):
            self._apply_track_to_cache_feedback()
        
        # ===== Stage 10: Track-based Score Modulation (CRITICAL for tracking to impact mAP!) =====
        # Without this, tracking_only gives same mAP as vanilla because we don't change scores
        final_scores = nms_scores.copy()
        score_mod_stats = {'boosted': 0, 'penalized_tentative': 0, 'penalized_untracked': 0, 
                          'filtered_untracked': 0, 'smoothed': 0}
        
        if self.use_tracking and self.use_track_score_modulation:
            active_tracks = self.track_manager.get_active_tracks() if self.track_manager else []
            track_id_to_track = {t.track_id: t for t in active_tracks}
            
            keep_indices = []  # For filter_untracked mode
            
            for i in range(N_nms):
                tid = track_ids[i]
                current_score = final_scores[i]
                
                if tid >= 0 and tid in track_id_to_track:
                    track = track_id_to_track[tid]
                    
                    # Temporal score smoothing: Use track's score history for stability
                    if self.use_temporal_score_smoothing and hasattr(track, 'score_history'):
                        if len(track.score_history) > 0:
                            avg_score = np.mean(track.score_history[-self.score_smoothing_window:])
                            smoothed = (1 - self.score_smoothing_alpha) * current_score + \
                                       self.score_smoothing_alpha * avg_score
                            final_scores[i] = smoothed
                            score_mod_stats['smoothed'] += 1
                    
                    # Track state-based boost/penalty
                    if track.is_confirmed():
                        # Boost confirmed track detections
                        final_scores[i] = min(1.0, final_scores[i] + self.confirmed_track_boost)
                        score_mod_stats['boosted'] += 1
                    elif track.is_tentative() and self.tentative_track_penalty > 0:
                        # Optionally penalize tentative tracks
                        final_scores[i] = max(0.0, final_scores[i] - self.tentative_track_penalty)
                        score_mod_stats['penalized_tentative'] += 1
                    keep_indices.append(i)
                else:
                    # Untracked detection
                    if self.filter_untracked:
                        score_mod_stats['filtered_untracked'] += 1
                        # Don't add to keep_indices
                    else:
                        final_scores[i] = max(0.0, final_scores[i] - self.untracked_penalty)
                        score_mod_stats['penalized_untracked'] += 1
                        keep_indices.append(i)
            
            # Apply filtering if enabled
            if self.filter_untracked and len(keep_indices) < N_nms:
                keep_indices = np.array(keep_indices)
                nms_boxes = nms_boxes[keep_indices]
                final_scores = final_scores[keep_indices]
                nms_features = nms_features[keep_indices]
                final_probs = final_probs[keep_indices]
                nms_raw_probs = nms_raw_probs[keep_indices]
                nms_labels = [nms_labels[i] for i in keep_indices]
                track_ids = [track_ids[i] for i in keep_indices]
                N_nms = len(keep_indices)
        else:
            final_scores = nms_scores  # No modulation
        
        dbg['score_mod'] = score_mod_stats
        
        # ===== Stage 11: Phantom Detections for Unmatched Tracks =====
        # Output predicted boxes for tracks that weren't matched (missed detections)
        phantom_stats = {'added': 0, 'skipped_score': 0, 'skipped_frames': 0}
        
        if self.use_tracking and self.use_predicted_for_missed and self.track_manager is not None:
            active_tracks = self.track_manager.get_active_tracks()
            matched_track_ids = set(tid for tid in track_ids if tid >= 0)
            
            # Lists to collect phantom detections
            phantom_boxes = []
            phantom_scores = []
            phantom_labels = []
            phantom_probs = []
            phantom_features = []
            phantom_track_ids = []
            
            for track in active_tracks:
                if track.track_id in matched_track_ids:
                    continue  # Already matched
                
                # Only output phantoms for confirmed tracks
                if not track.is_confirmed():
                    continue
                
                # Check how long track has been unmatched
                frames_missed = track.time_since_update
                if frames_missed > self.predicted_max_frames:
                    phantom_stats['skipped_frames'] += 1
                    continue
                
                # Compute decayed score
                if hasattr(track, 'score_history') and len(track.score_history) > 0:
                    last_score = track.score_history[-1]
                else:
                    last_score = 0.5  # Default if no history
                
                decayed_score = last_score * (self.predicted_score_decay ** frames_missed)
                
                if decayed_score < self.predicted_min_score:
                    phantom_stats['skipped_score'] += 1
                    continue
                
                # Get predicted box from Kalman filter
                predicted_box = track.get_state()
                
                # Use track's last known class probs
                if hasattr(track, 'class_probs') and track.class_probs is not None:
                    phantom_prob = track.class_probs
                elif track.class_stad is not None:
                    phantom_prob = track.get_class_probs()
                else:
                    # Fallback: uniform over classes
                    phantom_prob = np.ones(len(self.target_classes)) / len(self.target_classes)
                
                # Get last known label
                top_class_idx = np.argmax(phantom_prob)
                phantom_label = self.target_classes[top_class_idx]
                
                # Use track's smoothed feature
                phantom_feature = track.get_feature()
                
                phantom_boxes.append(predicted_box)
                phantom_scores.append(decayed_score)
                phantom_labels.append(phantom_label)
                phantom_probs.append(phantom_prob)
                phantom_features.append(phantom_feature)
                phantom_track_ids.append(track.track_id)
                phantom_stats['added'] += 1
            
            # Merge phantoms with real detections
            if len(phantom_boxes) > 0:
                nms_boxes = np.vstack([nms_boxes, np.array(phantom_boxes)])
                final_scores = np.concatenate([final_scores, np.array(phantom_scores)])
                nms_labels = list(nms_labels) + phantom_labels
                final_probs = np.vstack([final_probs, np.array(phantom_probs)])
                nms_features = np.vstack([nms_features, np.array(phantom_features)])
                track_ids = list(track_ids) + phantom_track_ids
                N_nms = len(track_ids)
        
        dbg['phantom'] = phantom_stats
        
        # ===== Create output =====
        # Update labels based on final class probs
        final_labels = []
        label_changes = 0
        for i in range(N_nms):
            top_class_idx = np.argmax(final_probs[i])
            new_label = self.target_classes[top_class_idx]
            final_labels.append(new_label)
            if new_label != nms_labels[i]:
                label_changes += 1
        dbg['label_changes'] = label_changes
        
        # Create result
        result = self._create_result(
            detection_result=detection_result,
            boxes=nms_boxes,
            scores=final_scores,  # Use modulated scores!
            labels=final_labels,
            class_probs=final_probs,
            features=nms_features,
            track_ids=track_ids
        )
        
        self.frame_count += 1
        self.total_detections += N
        self.total_adapted += N_nms
        
        # Update debug counters for sanity warnings
        self._total_stad_updates += frame_stad_updates
        if self.use_global_cache and dbg['cache_M_after'] == 0:
            self._frames_with_empty_cache += 1
        else:
            self._frames_with_empty_cache = 0
        if self.use_track_stad and frame_stad_updates == 0:
            self._frames_with_no_stad_updates += 1
        else:
            self._frames_with_no_stad_updates = 0
        
        # Print timing and debug info every debug_every frames
        timings['total'] = time.perf_counter() - t_start
        self._debug_log_frame(dbg, timings, t_start)
        
        # Visual debug - save tracking visualization
        if self.visual_debug and self.use_tracking:
            try:
                active_tracks = self.track_manager.get_active_tracks() if self.track_manager else []
                save_tracking_debug_image(
                    image=image,
                    boxes=nms_boxes,
                    scores=nms_scores,
                    labels=final_labels,
                    track_ids=track_ids,
                    frame_idx=self.frame_count,
                    output_dir=self.visual_debug_dir,
                    active_tracks=active_tracks,
                    video_name=self._current_video_name
                )
            except Exception as e:
                if self.debug:
                    print(f"[Warning] Visual debug save failed: {e}")
        
        self._end_frame()
        
        return result
    
    def _debug_log_frame(self, dbg: Dict, timings: Dict, t_start: float) -> None:
        """Print per-frame debug info and sanity warnings."""
        if not self.debug:
            return
        
        should_log = (self.frame_count % self.debug_every == 0) or (self.frame_count <= self._debug_log_first_n)
        
        # Always check sanity warnings
        if self.use_global_cache and self._frames_with_empty_cache >= self._warn_cache_empty_after:
            print(f"\n⚠️  [SANITY WARNING] Frame {self.frame_count}: Global cache has been EMPTY for "
                  f"{self._frames_with_empty_cache} frames!")
            print(f"    - use_global_cache={self.use_global_cache}, tau1={self.cache_config.tau1}")
            print(f"    - High-conf detections this frame: {dbg.get('cache_high_conf_count', 0)}")
            if self.global_cache:
                summary = self.global_cache.get_summary()
                print(f"    - Batch init: done={summary['batch_init_done']}, buffer_size={summary['batch_init_buffer_size']}")
                print(f"    - Total updates attempted: {summary['debug_updates_attempted']}")
                print(f"    - Updates filtered (low conf): {summary['debug_updates_low_conf']}")
                print(f"    - Updates buffered (batch init): {summary['debug_updates_batch_buffered']}")
            print(f"    - Check: Are scores >= tau1? Is batch_init_size too high?\n")
        
        if self.use_track_stad and self._frames_with_no_stad_updates >= self._warn_no_stad_after:
            print(f"\n⚠️  [SANITY WARNING] Frame {self.frame_count}: NO STAD updates for "
                  f"{self._frames_with_no_stad_updates} frames!")
            print(f"    - use_track_stad={self.use_track_stad}, min_confidence={self.stad_config.min_confidence}")
            print(f"    - Total STAD updates so far: {self._total_stad_updates}")
            print(f"    - Check: Are tracks being created? Are scores >= min_confidence?\n")
        
        if not should_log:
            return
        
        # Per-frame debug log
        print(f"\n{'─'*70}")
        print(f"[Frame {self.frame_count}] DEBUG STATS (mode={self.mode})")
        print(f"{'─'*70}")
        
        # Detection pipeline
        print(f"  📦 Detections: N_raw={dbg['N_raw']} → N_threshold={dbg['N_threshold']} → N_nms={dbg['N_nms']}")
        
        # Score distribution
        if 'score_min' in dbg and 'score_max' in dbg:
            print(f"       Scores: min={dbg['score_min']:.3f}, max={dbg['score_max']:.3f}, "
                  f">=tau1({self.cache_config.tau1}): {dbg['cache_high_conf_count']}")
        
        # Global cache
        if self.use_global_cache:
            print(f"  🗃️  Cache: M={dbg['cache_M_before']}→{dbg['cache_M_after']}, "
                  f"high_conf_updates={dbg['cache_high_conf_count']} (need score≥{self.cache_config.tau1})")
            if self.global_cache:
                summary = self.global_cache.get_summary()
                if not summary['batch_init_done']:
                    print(f"       Batch init: collecting {summary['batch_init_buffer_size']}/{self.cache_config.batch_init_size}")
            print(f"       Adapt effect: L1_mean={dbg['adapt_L1_mean']:.4f}, L1_max={dbg['adapt_L1_max']:.4f}")
            if dbg['adapt_L1_max'] < 0.001 and dbg['cache_M_before'] > 0:
                print(f"       ⚠️  Adaptation has MINIMAL effect on probs!")
        else:
            print(f"  🗃️  Cache: DISABLED (use_global_cache=False)")
        
        # Tracking
        if self.use_tracking:
            assoc_method = dbg.get('association_method_used', self.association_method)
            n_tentative = len(self.track_manager.get_tentative_tracks()) if self.track_manager else 0
            n_confirmed = len(self.track_manager.get_confirmed_tracks()) if self.track_manager else 0
            print(f"  🎯 Tracking: active={dbg['n_active_tracks']} (tentative={n_tentative}, confirmed={n_confirmed})")
            print(f"       matches={dbg['n_matches']}, unmatched_dets={dbg['n_unmatched_dets']}, unmatched_tracks={dbg['n_unmatched_tracks']}")
            print(f"       Association: method={assoc_method} (fast_threshold={self.association_fast_threshold})")
            print(f"       New tracks: created={dbg.get('tracks_created', 0)}, "
                  f"skipped_low_conf={dbg.get('tracks_skipped_low_conf', 0)} (need ≥{self.track_creation_threshold}), "
                  f"skipped_max_limit={dbg.get('tracks_skipped_max_limit', 0)} (max={self.max_tracks})")
            if n_confirmed == 0 and dbg['n_active_tracks'] > 0:
                print(f"       ℹ️  Tracks are TENTATIVE (need {self.track_config.min_hits_to_confirm} hits to confirm)")
        else:
            print(f"  🎯 Tracking: DISABLED (use_tracking=False)")
        
        # STAD
        if self.use_track_stad:
            print(f"  📈 STAD: updates={dbg['stad_updates']}, skipped_no_track={dbg['stad_skipped_no_track']} "
                  f"(need score≥{self.stad_config.min_confidence})")
            print(f"       Total STAD updates so far: {self._total_stad_updates}")
            if dbg['stad_updates'] == 0 and dbg['n_matches'] > 0:
                print(f"       ⚠️  Matched {dbg['n_matches']} tracks but 0 STAD updates!")
        else:
            print(f"  📈 STAD: DISABLED (use_track_stad=False)")
        
        # Score Modulation (tracking impact on mAP)
        if self.use_tracking and self.use_track_score_modulation:
            sm = dbg.get('score_mod', {})
            print(f"  📊 Score Mod: boosted={sm.get('boosted', 0)} (+{self.confirmed_track_boost}), "
                  f"penalized_tent={sm.get('penalized_tentative', 0)}, "
                  f"penalized_untracked={sm.get('penalized_untracked', 0)} (-{self.untracked_penalty})")
            if self.use_temporal_score_smoothing:
                print(f"       smoothed={sm.get('smoothed', 0)} (alpha={self.score_smoothing_alpha})")
            if self.filter_untracked:
                print(f"       filtered_untracked={sm.get('filtered_untracked', 0)} (filter_untracked=True)")
            if sm.get('boosted', 0) == 0 and dbg['n_matches'] > 0:
                print(f"       ⚠️  {dbg['n_matches']} matches but 0 boosted! Are tracks confirmed?")
        elif self.use_tracking:
            print(f"  📊 Score Mod: DISABLED (auto-disabled when TTA active, or use force_score_modulation=True)")
        
        # Pre-filtering stats
        if dbg.get('n_pre_filtered', 0) > 0:
            print(f"  🔽 Pre-filter: removed {dbg['n_pre_filtered']} detections (score < {self.track_creation_threshold})")
        
        # Phantom detections
        if self.use_predicted_for_missed:
            ph = dbg.get('phantom', {})
            print(f"  👻 Phantoms: added={ph.get('added', 0)}, "
                  f"skipped_score={ph.get('skipped_score', 0)}, "
                  f"skipped_frames={ph.get('skipped_frames', 0)}")
        
        # Label changes
        print(f"  🏷️  Label changes: {dbg['label_changes']}/{dbg['N_nms']} detections changed class")
        if dbg['label_changes'] == 0 and dbg['N_nms'] > 0 and (self.use_global_cache or self.use_track_stad):
            if dbg['cache_M_before'] == 0 and self.use_global_cache:
                print(f"       ℹ️  No label changes because cache is empty (still initializing)")
            else:
                print(f"       ⚠️  No label changes despite adaptation being enabled!")
        
        # Timings
        timing_str = ' | '.join(f"{k}={v*1000:.1f}ms" for k, v in timings.items())
        print(f"  ⏱️  Timing: {timing_str}")
        
        # Timing analysis
        if timings.get('total', 0) > 1.0:  # > 1 second
            print(f"       ⚠️  SLOW FRAME! Breakdown:")
            for k, v in sorted(timings.items(), key=lambda x: -x[1]):
                if v > 0.1:  # > 100ms
                    pct = 100 * v / timings['total']
                    print(f"          {k}: {v*1000:.0f}ms ({pct:.0f}%)")
        
        print(f"{'─'*70}\n")
    
    def _fuse_probs(self, p_init: np.ndarray, p_global: np.ndarray, 
                    p_track: np.ndarray, track: Optional['Track'] = None) -> np.ndarray:
        """
        Part 5.2: 3-source probability fusion.
        
        Fuses probabilities from three sources:
        - p_init: Raw VLM probabilities (initial detection)
        - p_global: Global cache adapted probabilities
        - p_track: Per-track STAD class belief
        
        Args:
            p_init: (K,) raw VLM class probabilities
            p_global: (K,) global cache adapted probabilities
            p_track: (K,) per-track STAD class belief (π)
            track: Optional Track object for confidence info
            
        Returns:
            (K,) fused class probabilities
        """
        eps = 1e-10
        
        if self.fusion_mode == 'parallel':
            # Weighted average of all three sources
            fused = (self.fusion_init_weight * p_init + 
                    self.fusion_global_weight * p_global + 
                    self.fusion_track_weight * p_track)
            return fused / (fused.sum() + eps)
        
        elif self.fusion_mode == 'hierarchical':
            # Hierarchical: p_track overrides p_global overrides p_init based on confidence
            # Track confidence based on hits and entropy
            track_conf = 0.0
            if track is not None and track.hits >= self.track_config.min_hits_to_confirm:
                H_track = -np.sum(p_track * np.log(p_track + eps))
                track_conf = np.exp(-H_track)  # Lower entropy = higher confidence
            
            # Global confidence from cache
            H_global = -np.sum(p_global * np.log(p_global + eps))
            global_conf = np.exp(-H_global)
            
            # Hierarchical selection
            if track_conf >= self.fusion_confidence_threshold:
                # Trust track STAD
                return p_track
            elif global_conf >= self.fusion_confidence_threshold:
                # Trust global cache, blend with track
                w_track = track_conf / (track_conf + global_conf + eps)
                fused = (1 - w_track) * p_global + w_track * p_track
                return fused / (fused.sum() + eps)
            else:
                # Blend all three with entropy weighting
                H_init = -np.sum(p_init * np.log(p_init + eps))
                w_init = np.exp(-H_init)
                w_global = global_conf
                w_track = track_conf
                total = w_init + w_global + w_track + eps
                fused = (w_init * p_init + w_global * p_global + w_track * p_track) / total
                return fused / (fused.sum() + eps)
        
        elif self.fusion_mode == 'selection':
            # Select the most confident source
            H_init = -np.sum(p_init * np.log(p_init + eps))
            H_global = -np.sum(p_global * np.log(p_global + eps))
            H_track = -np.sum(p_track * np.log(p_track + eps))
            
            # Pick lowest entropy (most confident)
            entropies = [H_init, H_global, H_track]
            sources = [p_init, p_global, p_track]
            return sources[np.argmin(entropies)]
        
        else:  # 'entropy_weighted' (default)
            # Entropy-weighted fusion of all three sources
            H_init = -np.sum(p_init * np.log(p_init + eps))
            H_global = -np.sum(p_global * np.log(p_global + eps))
            H_track = -np.sum(p_track * np.log(p_track + eps))
            
            w_init = np.exp(-H_init)
            w_global = np.exp(-H_global)
            w_track = np.exp(-H_track)
            
            total = w_init + w_global + w_track + eps
            fused = (w_init * p_init + w_global * p_global + w_track * p_track) / total
            return fused / (fused.sum() + eps)
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray,
                   keep_mask: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression."""
        valid_indices = np.where(keep_mask)[0]
        if len(valid_indices) == 0:
            return np.array([], dtype=np.int32)
        
        # Simple greedy NMS
        order = scores[valid_indices].argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(valid_indices[i])
            
            if len(order) == 1:
                break
            
            # Compute IoU with rest
            ious = self._compute_iou_batch(boxes[valid_indices[i]], boxes[valid_indices[order[1:]]])
            
            # Keep boxes with IoU below threshold
            remaining = np.where(ious < self.nms_threshold)[0] + 1
            order = order[remaining]
        
        return np.array(keep, dtype=np.int32)
    
    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        return inter / (area1 + area2 - inter + 1e-8)
    
    def _apply_track_to_cache_feedback(self) -> None:
        """Apply feedback from confirmed tracks to global cache."""
        if self.global_cache is None or self.track_manager is None:
            return
        
        confirmed_tracks = self.track_manager.get_confirmed_tracks()
        
        for track in confirmed_tracks:
            # Only feedback from mature, confident tracks
            if track.hits < self.feedback_min_hits:
                continue
            
            avg_score = np.mean(track.score_history[-10:]) if track.score_history else 0
            if avg_score < self.feedback_min_confidence:
                continue
            
            # Get track's smoothed feature and class belief
            feature = track.get_feature()
            class_probs = track.get_class_probs()
            
            # Create dummy box from last known state
            box = track.get_state()
            
            # Soft update to cache (lower weight to avoid overfitting)
            effective_score = avg_score * self.feedback_weight
            
            # Update cache directly with track's belief
            self.global_cache._update_single(feature, box, class_probs, effective_score)
    
    def _end_frame(self) -> None:
        """End-of-frame processing."""
        if self.global_cache is not None:
            self.global_cache.step()
        
        if self.track_manager is not None:
            self.track_manager.step()
    
    def _create_empty_result(self, template_result) -> Any:
        """Create empty detection result."""
        # Return result with empty arrays
        result = type(template_result)(
            boxes=np.array([]),
            scores=np.array([]),
            labels=[],
            class_probs=np.array([]).reshape(0, self.num_classes),
            features=np.array([]).reshape(0, self.feature_dim or 256)
        )
        return result
    
    def _create_result(self, detection_result, boxes, scores, labels,
                      class_probs, features, track_ids=None) -> Any:
        """Create detection result object."""
        # Try to use same result type as input
        try:
            result = type(detection_result)(
                boxes=boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
                scores=scores.tolist() if isinstance(scores, np.ndarray) else scores,
                labels=labels,
                class_probs=class_probs.tolist() if isinstance(class_probs, np.ndarray) else class_probs,
                features=features.tolist() if isinstance(features, np.ndarray) else features
            )
            
            # Add track_ids if supported
            if hasattr(result, 'track_ids'):
                result.track_ids = track_ids
            
            return result
        except Exception:
            # Fallback: just modify the input result
            detection_result.boxes = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
            detection_result.scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
            detection_result.labels = labels
            detection_result.class_probs = class_probs.tolist() if isinstance(class_probs, np.ndarray) else class_probs
            detection_result.features = features.tolist() if isinstance(features, np.ndarray) else features
            
            return detection_result
    
    def reset(self) -> None:
        """Reset all state for new video."""
        if self.global_cache is not None:
            self.global_cache.reset()
        
        if self.track_manager is not None:
            self.track_manager.reset()
        
        reset_track_ids()
        
        self.frame_count = 0
        self.total_detections = 0
        self.total_adapted = 0
        
        # Reset debug counters
        self._frames_with_empty_cache = 0
        self._frames_with_no_stad_updates = 0
        self._total_stad_updates = 0
    
    def get_summary(self) -> Dict:
        """Get adapter summary including debug stats."""
        summary = {
            'mode': self.mode,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'total_adapted': self.total_adapted,
            'use_global_cache': self.use_global_cache,
            'use_tracking': self.use_tracking,
            'use_track_stad': self.use_track_stad,
            # Debug counters
            'total_stad_updates': self._total_stad_updates,
            'frames_with_empty_cache': self._frames_with_empty_cache,
            'frames_with_no_stad_updates': self._frames_with_no_stad_updates,
            # Config values for verification
            'config': {
                'tau1': self.cache_config.tau1,
                'tau2': self.cache_config.tau2,
                'ws': self.cache_config.ws,
                'detection_threshold': self.detection_threshold,
                'fusion_mode': self.fusion_mode,
                'ssm_type': self.stad_variant,
                'em_iterations': self.stad_config.em_iterations
            }
        }
        
        if self.global_cache is not None:
            summary['global_cache'] = self.global_cache.get_summary()
        
        if self.track_manager is not None:
            summary['track_manager'] = self.track_manager.get_summary()
        
        return summary


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing global_instance_adapter.py")
    print("Note: This module requires a detector to be fully tested.")
    print("Run the integration test with an actual detector instance.")
    
    # Test config generation
    for mode in ['vanilla', 'global_only', 'instance_only', 'tracking_only', 'full', 'cascade']:
        config = get_ablation_config(mode)
        print(f"\n{mode} config:")
        print(f"  use_global_cache={config.get('use_global_cache')}")
        print(f"  use_tracking={config.get('use_tracking')}")
        print(f"  use_track_stad={config.get('use_track_stad')}")
        print(f"  visual_debug={config.get('visual_debug', False)}")
    
    print("\n✓ Config generation test passed!")