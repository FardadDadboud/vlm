"""
Base adapter interface for TTA methods
"""

from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class BaseAdapter(ABC):
    """Abstract base class for all adaptation methods"""
    
    def __init__(self, detector, config: dict):
        self.detector = detector
        self.config = config
        self.model_path = detector.model_path
    
    @abstractmethod
    def adapt_and_detect(self, image: Image.Image, target_classes: List[str], threshold: float):
        """
        Run detection with adaptation
        
        Args:
            image: PIL Image
            target_classes: List of class names
            threshold: Detection threshold
            
        Returns:
            DetectionResult object
        """
        raise NotImplementedError
