"""
Adapters module - Factory for TTA methods

Available adapters:
- vanilla: No adaptation, passthrough to detector
- bca_plus: BCA+ Bayesian Class Adaptation
- temporal: Temporal TTA with State-Space Models (STAD)
"""

from .base_adapter import BaseAdapter
from .vanilla_adapter import VanillaAdapter
from .bca_plus_adapter import BCAPlusAdapter
from .tent_adapter import TENTAdapter
from .base_adapter import BaseAdapter
# Import from track.py (single file with all tracking components)
from .track import (
    # Kalman filter
    KalmanBoxTracker,
    KalmanFilterConfig,
    create_kalman_tracker,
    
    # Data association
    AssociationConfig,
    associate,
    associate_iou,
    associate_combined,
    associate_bytetrack,
    compute_iou,
    compute_iou_matrix,
    
    # Per-track STAD (FULL implementations)
    TrackSTADConfig,
    TrackSTADvMF,
    TrackSTADGaussian,
    create_track_stad,
    
    # Track management
    Track,
    TrackState,
    TrackConfig,
    TrackManager,
    Detection,
    reset_track_ids
)

# Import enhanced BCA+ cache
from .enhanced_bca_cache import (
    EnhancedBCAPlusCache,
    EnhancedBCAPlusConfig,
    CacheEntryState
)

# Import main adapter
from .global_instance_adapter import (
    GlobalInstanceAdapter,
    GlobalInstanceConfig,
    get_ablation_config
)

# Try to import temporal adapter (may need additional dependencies)
try:
    from .temporal_adapter import TemporalTTAAdapter
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Temporal adapter not available: {e}")
    TEMPORAL_AVAILABLE = False

from .temporal_adapter_v2 import TemporalTTAAdapterV2


def create_adapter(adaptation_type: str, detector, config: dict) -> BaseAdapter:
    """
    Factory function to create TTA adapters.
    
    Args:
        adaptation_type: Type of adaptation ('none', 'vanilla', 'bca_plus', 'temporal')
        detector: Base VLM detector
        config: Configuration dictionary
        
    Returns:
        Adapter instance wrapping the detector
        
    Raises:
        ValueError: If unknown adaptation type is requested
    """
    adaptation_type = adaptation_type.lower().strip()
    
    if adaptation_type in ('none', 'vanilla'):
        return VanillaAdapter(detector, config)
    
    elif adaptation_type == 'bca_plus':
        return BCAPlusAdapter(detector, config)
    
    elif adaptation_type == 'temporal':
        if not TEMPORAL_AVAILABLE:
            raise ImportError("Temporal adapter not available. Check dependencies.")
        return TemporalTTAAdapter(detector, config)

    elif adaptation_type == 'temporal_v2':
        return TemporalTTAAdapterV2(detector, config)

    elif adaptation_type == 'global_instance':
        return GlobalInstanceAdapter(detector, config)

    elif adaptation_type == 'tent':
        return TENTAdapter(detector, config)
    else:
        raise ValueError(f"Unknown adaptation type: {adaptation_type}. "
                        f"Available: none, vanilla, bca_plus, temporal, global_instance, tent")


def list_adapters() -> dict:
    """List available adapters and their status"""
    adapters = {
        'none': {'available': True, 'description': 'No adaptation (alias for vanilla)'},
        'vanilla': {'available': True, 'description': 'No adaptation, passthrough'},
        'bca_plus': {'available': True, 'description': 'BCA+ Bayesian Class Adaptation'},
        'temporal': {'available': TEMPORAL_AVAILABLE, 'description': 'STAD temporal adaptation'},
        'global_instance': {'available': True, 'description': 'Global + Instance Temporal BCA+'},
        'tent': {'available': True, 'description': 'TENT (gradient-based TTA via LayerNorm entropy minimisation)'},
    }
    return adapters


__all__ = [
    'BaseAdapter',
    'VanillaAdapter',
    'BCAPlusAdapter',
    'TENTAdapter',
    'create_adapter',
    'list_adapters'
]

if TEMPORAL_AVAILABLE:
    __all__.append('TemporalTTAAdapter')

__all__.extend([
    # Main adapter
    'GlobalInstanceAdapter',
    'GlobalInstanceConfig',
    'get_ablation_config',
    
    # Global cache
    'EnhancedBCAPlusCache',
    'EnhancedBCAPlusConfig',
    'CacheEntryState',
    
    # Kalman filter
    'KalmanBoxTracker',
    'KalmanFilterConfig',
    'create_kalman_tracker',
    
    # Data association
    'AssociationConfig',
    'associate',
    'associate_iou',
    'associate_combined',
    'associate_bytetrack',
    'compute_iou',
    'compute_iou_matrix',
    
    # Per-track STAD
    'TrackSTADConfig',
    'TrackSTADvMF',
    'TrackSTADGaussian',
    'create_track_stad',
    
    # Track management
    'Track',
    'TrackState',
    'TrackConfig',
    'TrackManager',
    'Detection',
    'reset_track_ids'
])

__all__.append('TemporalTTAAdapterV2')