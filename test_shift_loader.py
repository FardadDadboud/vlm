#!/usr/bin/env python3
"""
Test script for SHIFT dataset loader
Run this to verify the dataset is loading correctly
"""

import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from vlm_shift_dataset import create_vlm_shift_dataset


def test_shift_loader():
    """Test SHIFT dataset loader with various configurations"""
    
    # Update this path to your SHIFT data location
    DATA_ROOT = "/media/fardad/T7 Touch/SHIFT/shift-dev/data"
    
    print("=" * 80)
    print("SHIFT Dataset Loader Test")
    print("=" * 80)
    
    # Test 1: Load without filters
    print("\n[Test 1] Loading validation set without filters...")
    try:
        dataset_full = create_vlm_shift_dataset(
            data_root=DATA_ROOT,
            split="val"
        )
        print(f"✓ Loaded {len(dataset_full)} total frames")
        
        # Get statistics
        stats = dataset_full.get_domain_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Weather: {dict(stats['weather_distribution'])}")
        print(f"  Time: {dict(stats['time_distribution'])}")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Test 2: Load with weather filter
    print("\n[Test 2] Loading with weather filter (clear + overcast)...")
    try:
        dataset_weather = create_vlm_shift_dataset(
            data_root=DATA_ROOT,
            split="val",
            filters={'weather': ['clear', 'overcast']}
        )
        print(f"✓ Filtered to {len(dataset_weather)} frames")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: Load subset for quick testing
    print("\n[Test 3] Loading small subset (50 frames per video)...")
    try:
        dataset_small = create_vlm_shift_dataset(
            data_root=DATA_ROOT,
            split="val",
            filters={'max_frames_per_video': 50}
        )
        print(f"✓ Loaded {len(dataset_small)} frames")
        print(f"  Videos: {len(dataset_small.get_video_names())}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 4: Access individual samples
    print("\n[Test 4] Accessing individual samples...")
    try:
        sample = dataset_small[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        print(f"  Image path: {sample['image_path']}")
        print(f"  Video: {sample['image_info']['video_name']}")
        print(f"  Frame index: {sample['image_info']['frame_index']}")
        print(f"  Annotations: {len(sample['annotations'])} objects")
        
        if len(sample['annotations']) > 0:
            anno = sample['annotations'][0]
            print(f"  First object: {anno['category_name']} @ {anno['bbox']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 5: Video sequence retrieval
    print("\n[Test 5] Testing video sequence retrieval...")
    try:
        video_names = dataset_small.get_video_names()
        first_video = video_names[0]
        video_frames = dataset_small.get_video_sequence(first_video)
        print(f"✓ Video '{first_video}': {len(video_frames)} frames")
        
        # Check temporal ordering
        frame_indices = [f['image_info']['frame_index'] for f in video_frames]
        is_sorted = all(frame_indices[i] <= frame_indices[i+1] for i in range(len(frame_indices)-1))
        print(f"  Frames sorted by index: {is_sorted}")
        print(f"  Frame range: {min(frame_indices)} to {max(frame_indices)}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: Check image files exist
    print("\n[Test 6] Verifying image files exist...")
    try:
        sample = dataset_small[0]
        img_path = sample['image_path']
        exists = img_path.exists()
        print(f"✓ Image exists: {exists}")
        if exists:
            from PIL import Image
            img = Image.open(img_path)
            print(f"  Image size: {img.size}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    test_shift_loader()