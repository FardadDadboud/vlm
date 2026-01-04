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

# Try to import temporal adapter (may need additional dependencies)
try:
    from .temporal_adapter import TemporalTTAAdapter
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Temporal adapter not available: {e}")
    TEMPORAL_AVAILABLE = False


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
    
    else:
        raise ValueError(f"Unknown adaptation type: {adaptation_type}. "
                        f"Available: none, vanilla, bca_plus, temporal")


def list_adapters() -> dict:
    """List available adapters and their status"""
    adapters = {
        'none': {'available': True, 'description': 'No adaptation (alias for vanilla)'},
        'vanilla': {'available': True, 'description': 'No adaptation, passthrough'},
        'bca_plus': {'available': True, 'description': 'BCA+ Bayesian Class Adaptation'},
        'temporal': {'available': TEMPORAL_AVAILABLE, 'description': 'STAD temporal adaptation'}
    }
    return adapters


__all__ = [
    'BaseAdapter',
    'VanillaAdapter', 
    'BCAPlusAdapter',
    'create_adapter',
    'list_adapters'
]

if TEMPORAL_AVAILABLE:
    __all__.append('TemporalTTAAdapter')