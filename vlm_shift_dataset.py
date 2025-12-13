#!/usr/bin/env python3
"""
VLM SHIFT Dataset Loader
Loads SHIFT continuous video dataset for VLM detection with temporal support
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from collections import defaultdict


class VLMSHIFTDataset:
    """
    SHIFT Dataset loader for VLM detection pipeline with temporal support
    Handles continuous video sequences with domain shift metadata
    """
    
    def __init__(self, data_root: str, split: str = "val", filters: Optional[Dict] = None):
        """
        Args:
            data_root: Root path to SHIFT dataset (e.g., /scratch/fardadd/SHIFT/data)
            split: One of 'train', 'val', 'test'
            filters: Optional filters for domain selection
                {
                    'weather': ['clear', 'rainy', 'overcast'],
                    'time': ['daytime', 'night'],
                    'video_sequences': ['028b-3dfe', '07f5-abab'],  # Specific videos
                    'max_frames_per_video': 100  # Limit frames per sequence
                }
        """
        self.data_root = Path(data_root)
        self.split = split
        self.filters = filters or {}
        
        # SHIFT-specific paths (continuous videos at 10fps)
        self.videos_dir = self.data_root / "continuous" / "videos" / "1x" / split / "front"
        self.annotations_file = self.videos_dir / "det_2d.json"
        
        # Category mapping (SHIFT has 6 classes)
        self.categories = [
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "truck"},
            {"id": 4, "name": "bus"},
            {"id": 5, "name": "motorcycle"},
            {"id": 6, "name": "bicycle"}
        ]
        self.cat_name_to_id = {cat["name"]: cat["id"] for cat in self.categories}
        
        # Validate and load dataset
        self._validate_dataset_structure()
        self.dataset_dict = self._load_dataset()
        
        # Group by video sequences
        self.video_sequences = self._group_by_video()
        
        # Apply filters
        self.filtered_samples = self._apply_filters()
        
        print(f"Loaded SHIFT {self.split}: {len(self.filtered_samples)} frames from {len(self.video_sequences)} video sequences")
        if self.filters:
            print(f"Applied filters: {self.filters}")
    
    def _validate_dataset_structure(self):
        """Validate SHIFT dataset structure"""
        if not self.data_root.exists():
            raise ValueError(f"Dataset root does not exist: {self.data_root}")
        if not self.videos_dir.exists():
            raise ValueError(f"Videos directory not found: {self.videos_dir}")
        if not self.annotations_file.exists():
            raise ValueError(f"Annotations file not found: {self.annotations_file}")
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load SHIFT annotations (nested frame format)"""
        with open(self.annotations_file, 'r') as f:
            dataset_dict = json.load(f)
        
        if 'frames' not in dataset_dict:
            raise ValueError("Missing 'frames' key in SHIFT annotations")
        
        print(f"Loaded {len(dataset_dict['frames'])} frames from SHIFT {self.split} set")
        return dataset_dict
    
    def _group_by_video(self) -> Dict[str, List[Dict]]:
        """Group frames by video sequence for temporal processing"""
        video_sequences = defaultdict(list)
        
        for frame in self.dataset_dict['frames']:
            video_name = frame['videoName']
            video_sequences[video_name].append(frame)
        
        # Sort frames within each video by frameIndex
        for video_name in video_sequences:
            video_sequences[video_name].sort(key=lambda x: x['frameIndex'])
        
        print(f"Grouped into {len(video_sequences)} video sequences")
        return dict(video_sequences)
    
    def _apply_filters(self) -> List[Dict[str, Any]]:
        """Apply domain and temporal filters"""
        filtered_samples = []
        
        # Filter by specific video sequences if specified
        target_videos = self.filters.get('video_sequences', None)
        if target_videos:
            video_sequences = {k: v for k, v in self.video_sequences.items() if k in target_videos}
        else:
            video_sequences = self.video_sequences
        
        for video_name, frames in video_sequences.items():
            # Limit frames per video if specified
            max_frames = self.filters.get('max_frames_per_video', None)
            if max_frames:
                frames = frames[:max_frames]
            
            for frame in frames:
                # Apply weather filter
                if 'weather' in self.filters:
                    weather_coarse = frame['attributes'].get('weather_coarse', '')
                    if weather_coarse not in self.filters['weather']:
                        continue
                
                # Apply time of day filter
                if 'time' in self.filters:
                    time_coarse = frame['attributes'].get('timeofday_coarse', '')
                    if time_coarse not in self.filters['time']:
                        continue
                
                # Convert frame to sample format
                sample = self._convert_frame_to_sample(frame, video_name)
                if sample:
                    filtered_samples.append(sample)
        
        return filtered_samples
    
    def _convert_frame_to_sample(self, frame: Dict, video_name: str) -> Optional[Dict[str, Any]]:
        """Convert SHIFT frame format to standard sample format"""
        # Get image path
        image_filename = frame['name']
        image_path = self.videos_dir / video_name / image_filename
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return None
        
        # Convert labels to COCO-like annotations
        annotations = []
        for label in frame.get('labels', []):
            bbox_dict = label['box2d']
            bbox = [bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']]
            
            # Calculate bbox width and height
            width = bbox_dict['x2'] - bbox_dict['x1']
            height = bbox_dict['y2'] - bbox_dict['y1']
            area = width * height
            
            annotation = {
                'id': label['id'],  # Track ID (useful for temporal tracking)
                'category_id': self.cat_name_to_id.get(label['category'], 0),
                'category_name': label['category'],
                'bbox': bbox,  # [x1, y1, x2, y2] format
                'area': area,
                'iscrowd': 0
            }
            annotations.append(annotation)
        
        # Create image info
        image_info = {
            'id': f"{video_name}_{frame['frameIndex']:06d}",
            'file_name': image_filename,
            'video_name': video_name,
            'frame_index': frame['frameIndex'],
            'width': 1280,  # SHIFT image size
            'height': 800,
            # Domain shift metadata
            'weather_coarse': frame['attributes'].get('weather_coarse', 'unknown'),
            'weather_fine': frame['attributes'].get('weather_fine', 'unknown'),
            'timeofday_coarse': frame['attributes'].get('timeofday_coarse', 'unknown'),
            'timeofday_fine': frame['attributes'].get('timeofday_fine', 'unknown'),
            'shift_type': frame['attributes'].get('shift_type', 'unknown'),
        }
        
        return {
            'image_info': image_info,
            'annotations': annotations,
            'image_path': image_path
        }
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample by index"""
        if index >= len(self.filtered_samples):
            raise IndexError(f"Index {index} out of range")
        return self.filtered_samples[index]
    
    def get_video_sequence(self, video_name: str) -> List[Dict[str, Any]]:
        """Get all samples from a specific video sequence (for temporal processing)"""
        return [s for s in self.filtered_samples if s['image_info']['video_name'] == video_name]
    
    def get_video_names(self) -> List[str]:
        """Get list of all video sequence names"""
        unique_videos = sorted(set(s['image_info']['video_name'] for s in self.filtered_samples))
        return unique_videos
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about domain distribution"""
        stats = {
            'total_frames': len(self.filtered_samples),
            'total_videos': len(self.get_video_names()),
            'weather_distribution': defaultdict(int),
            'time_distribution': defaultdict(int),
            'category_distribution': defaultdict(int)
        }
        
        for sample in self.filtered_samples:
            weather = sample['image_info']['weather_coarse']
            time = sample['image_info']['timeofday_coarse']
            stats['weather_distribution'][weather] += 1
            stats['time_distribution'][time] += 1
            
            for anno in sample['annotations']:
                cat_name = anno['category_name']
                stats['category_distribution'][cat_name] += 1
        
        return dict(stats)
    
    def __len__(self):
        return len(self.filtered_samples)
    
    def __getitem__(self, index):
        return self.get_sample(index)


def create_vlm_shift_dataset(data_root: str, split: str = "val", filters: Optional[Dict] = None) -> VLMSHIFTDataset:
    """Factory function to create VLM SHIFT dataset"""
    return VLMSHIFTDataset(data_root=data_root, split=split, filters=filters)


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset loader
    dataset = create_vlm_shift_dataset(
        data_root="/scratch/fardadd/SHIFT/data",
        split="val",
        filters={
            'weather': ['overcast', 'clear'],  # Filter specific weather
            'max_frames_per_video': 50  # Limit for quick testing
        }
    )
    
    print(f"\n=== SHIFT Dataset Statistics ===")
    stats = dataset.get_domain_statistics()
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total videos: {stats['total_videos']}")
    print(f"\nWeather distribution:")
    for weather, count in stats['weather_distribution'].items():
        print(f"  {weather}: {count}")
    print(f"\nTime distribution:")
    for time, count in stats['time_distribution'].items():
        print(f"  {time}: {count}")
    print(f"\nCategory distribution:")
    for cat, count in stats['category_distribution'].items():
        print(f"  {cat}: {count}")
    
    # Test getting a sample
    print(f"\n=== Sample Frame ===")
    sample = dataset[0]
    print(f"Video: {sample['image_info']['video_name']}")
    print(f"Frame: {sample['image_info']['frame_index']}")
    print(f"Weather: {sample['image_info']['weather_coarse']}")
    print(f"Time: {sample['image_info']['timeofday_coarse']}")
    print(f"Annotations: {len(sample['annotations'])} objects")
    
    # Test video sequence retrieval
    print(f"\n=== Video Sequences ===")
    video_names = dataset.get_video_names()
    print(f"Available videos: {video_names[:5]}...")  # Show first 5
    
    first_video = video_names[0]
    video_frames = dataset.get_video_sequence(first_video)
    print(f"\nVideo '{first_video}': {len(video_frames)} frames")