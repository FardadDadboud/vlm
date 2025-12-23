"""
Adapter factory for creating TTA adapters
"""

from .base_adapter import BaseAdapter
from .vanilla_adapter import VanillaAdapter
from .bca_plus_adapter import BCAPlusAdapter


def create_adapter(adaptation_type: str, detector, config: dict) -> BaseAdapter:
    """
    Factory function to create appropriate adapter
    
    Args:
        adaptation_type: Type of adaptation ('none', 'bca_plus', etc.)
        detector: Base detector instance
        config: Full configuration dict
        
    Returns:
        Adapter instance
    """
    if adaptation_type == 'none':
        return VanillaAdapter(detector, config)
    elif adaptation_type == 'bca_plus':
        return BCAPlusAdapter(detector, config)
    else:
        raise ValueError(f"Unknown adaptation type: {adaptation_type}")


__all__ = ['BaseAdapter', 'VanillaAdapter', 'BCAPlusAdapter', 'create_adapter']
