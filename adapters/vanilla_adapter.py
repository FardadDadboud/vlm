"""
Vanilla adapter - no adaptation, just passthrough to detector
"""

from PIL import Image
from typing import List
from .base_adapter import BaseAdapter


class VanillaAdapter(BaseAdapter):
    """No adaptation - just runs vanilla detector"""
    
    def __init__(self, detector, config: dict):
        super().__init__(detector, config)
        self.iou_threshold = config['detector'].get('iou_threshold') or 0.3
    
    def adapt_and_detect(self, image: Image.Image, target_classes: List[str], threshold: float):
        """
        Run vanilla detection without any adaptation
        """
        return self.detector.detect(image, target_classes, threshold, self.iou_threshold)
