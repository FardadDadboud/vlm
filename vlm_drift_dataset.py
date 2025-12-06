#!/usr/bin/env python3
"""
VLM DrIFT Dataset Loader
Integrates COCO dataset with filtering for VLM detection pipeline
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import pycocotools.mask as mask_util


class VLMDrIFTDataset:
    """
    DrIFT Dataset loader for VLM detection pipeline
    Supports filtering and background segmentation
    """
    
    def __init__(self, data_root: str, split: str = "val", filters: Optional[Dict] = None):
        self.data_root = Path(data_root)
        self.split = split
        self.filters = filters or {}
        
        # Dataset paths
        self.images_dir = self.data_root / "images"
        self.annotations_file = self.data_root / "annotations" / f"{split}.json"
        
        # Background mapping
        self.bg_mapping = {0: "sky", 1: "tree", 2: "ground"}
        
        # Load and filter dataset
        self._validate_dataset_structure()
        self.dataset_dict = self._load_dataset()
        self.filtered_samples = self._apply_filters()
        
        print(f"Loaded {self.split}: {len(self.filtered_samples)}/{len(self.dataset_dict.get('images', []))} samples")
    
    def _validate_dataset_structure(self):
        """Validate dataset structure"""
        if not self.data_root.exists():
            raise ValueError(f"Dataset root does not exist: {self.data_root}")
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_file.exists():
            raise ValueError(f"Annotations file not found: {self.annotations_file}")
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load COCO format annotations"""
        with open(self.annotations_file, 'r') as f:
            dataset_dict = json.load(f)
        
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in dataset_dict:
                raise ValueError(f"Missing required key: {key}")
        
        return dataset_dict
    
    def _apply_filters(self) -> List[Dict[str, Any]]:
        """Apply image and annotation filters"""
        if not self.filters:
            return self._process_all_images()
        
        image_filters = self.filters.get('image_filters', {})
        annotation_filters = self.filters.get('annotation_filters', {})
        
        # Create image ID to annotations mapping
        image_id_to_annos = {}
        for anno in self.dataset_dict.get('annotations', []):
            image_id = anno['image_id']
            if image_id not in image_id_to_annos:
                image_id_to_annos[image_id] = []
            image_id_to_annos[image_id].append(anno)
        
        filtered_samples = []
        for img_info in self.dataset_dict.get('images', []):
            # Apply image filters
            if not self._apply_image_filters(img_info, image_filters):
                continue
            
            # Get and filter annotations
            image_annotations = image_id_to_annos.get(img_info['id'], [])
            filtered_annotations = self._apply_annotation_filters(image_annotations, annotation_filters)
            
            if not filtered_annotations:
                continue
            
            # Create sample
            image_path = self._get_image_path(img_info['file_name'])
            if image_path.exists():
                sample = {
                    'image_info': img_info,
                    'annotations': filtered_annotations,
                    'image_path': image_path
                }
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _process_all_images(self) -> List[Dict[str, Any]]:
        """Process all images without filtering"""
        image_id_to_annos = {}
        for anno in self.dataset_dict.get('annotations', []):
            image_id = anno['image_id']
            if image_id not in image_id_to_annos:
                image_id_to_annos[image_id] = []
            image_id_to_annos[image_id].append(anno)
        
        samples = []
        for img_info in self.dataset_dict.get('images', []):
            image_annotations = image_id_to_annos.get(img_info['id'], [])
            image_path = self._get_image_path(img_info['file_name'])
            
            if image_path.exists():
                sample = {
                    'image_info': img_info,
                    'annotations': image_annotations,
                    'image_path': image_path
                }
                samples.append(sample)
        
        return samples
    
    def _apply_image_filters(self, img_info: Dict, filters: Dict) -> bool:
        """Apply image-level filters"""
        for filter_name, filter_values in filters.items():
            if filter_name in img_info:
                if img_info[filter_name] not in filter_values:
                    return False
        return True
    
    def _apply_annotation_filters(self, annotations: List[Dict], filters: Dict) -> List[Dict]:
        """Apply annotation-level filters"""
        if not filters:
            return annotations
        
        filtered_annotations = []
        for anno in annotations:
            include_annotation = True
            for filter_name, filter_values in filters.items():
                if filter_name in anno:
                    if anno[filter_name] not in filter_values:
                        include_annotation = False
                        break
            if include_annotation:
                filtered_annotations.append(anno)
        
        return filtered_annotations
    
    def _get_image_path(self, file_name: str) -> Path:
        """Get full image path with fallback options"""
        # Try direct path
        img_path = self.images_dir / file_name
        if img_path.exists():
            return img_path
        
        # Try just filename
        img_path = self.images_dir / Path(file_name).name
        if img_path.exists():
            return img_path
        
        # Try parent directory
        img_path = self.images_dir.parent / file_name
        if img_path.exists():
            return img_path
        
        return self.images_dir / file_name
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample by index"""
        if index >= len(self.filtered_samples):
            raise IndexError(f"Index {index} out of range")
        return self.filtered_samples[index]
    
    def create_background_masks(self, sample: Dict) -> Dict[str, np.ndarray]:
        """Create background masks for sample"""
        image_info = sample['image_info']
        bg_segments = image_info.get('background_segmentation', [])
        
        if not bg_segments:
            return {}
        
        height = image_info['height']
        width = image_info['width']
        bg_masks = {}
        
        for segment in bg_segments:
            cat_id = segment.get('category_id', 0)
            bg_name = self.bg_mapping.get(cat_id, "sky")
            
            mask = self._decode_segmentation(segment.get('segmentation'), height, width)
            if mask is not None:
                if bg_name not in bg_masks:
                    bg_masks[bg_name] = np.zeros((height, width), dtype=np.uint8)
                bg_masks[bg_name] = np.logical_or(bg_masks[bg_name], mask).astype(np.uint8)
        
        return bg_masks
    
    def _decode_segmentation(self, segmentation: Any, height: int, width: int) -> Optional[np.ndarray]:
        """Decode segmentation mask"""
        try:
            if isinstance(segmentation, dict):
                return mask_util.decode(segmentation)
            elif isinstance(segmentation, list):
                rle = mask_util.frPyObjects(segmentation, height, width)
                if isinstance(rle, list):
                    rle = mask_util.merge(rle)
                return mask_util.decode(rle)
        except Exception as e:
            print(f"Error decoding segmentation: {e}")
        return None
    
    def __len__(self):
        return len(self.filtered_samples)
    
    def __getitem__(self, index):
        return self.get_sample(index)


def create_vlm_drift_dataset(data_root: str, split: str = "val", filters: Optional[Dict] = None) -> VLMDrIFTDataset:
    """Factory function to create VLM DrIFT dataset"""
    return VLMDrIFTDataset(data_root=data_root, split=split, filters=filters)
