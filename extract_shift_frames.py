#!/usr/bin/env python3
"""
Utility to extract frames from SHIFT video files (.mp4)
Use this if img.tar is corrupted or if you prefer extracting from videos
"""

import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path: Path, output_dir: Path, prefix: str = "img_front"):
    """
    Extract all frames from a video file
    
    Args:
        video_path: Path to .mp4 file
        output_dir: Directory to save extracted frames
        prefix: Prefix for frame filenames (default: "img_front")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting {frame_count} frames from {video_path.name}...")
    
    for frame_idx in tqdm(range(frame_count), desc=video_path.stem):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame with SHIFT naming convention
        frame_name = f"{frame_idx:08d}_{prefix}.jpg"
        frame_path = output_dir / frame_name
        cv2.imwrite(str(frame_path), frame)
    
    cap.release()
    print(f"✓ Extracted {frame_idx + 1} frames to {output_dir}")


def extract_all_videos_in_directory(videos_dir: Path):
    """
    Extract frames from all .mp4 videos in a directory
    
    Args:
        videos_dir: Directory containing .mp4 files
    """
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files in {videos_dir}")
    
    for video_path in video_files:
        # Create output directory with video name (without .mp4 extension)
        output_dir = videos_dir / video_path.stem
        
        # Skip if already extracted
        if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) > 0:
            print(f"⊘ Skipping {video_path.name} (already extracted)")
            continue
        
        extract_frames_from_video(video_path, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python extract_shift_frames.py <path_to_front_directory>")
        print("\nExample:")
        print('  python extract_shift_frames.py "/media/fardad/T7 Touch/SHIFT/shift-dev/data/continuous/videos/1x/val/front"')
        sys.exit(1)
    
    videos_dir = Path(sys.argv[1])
    
    if not videos_dir.exists():
        print(f"Error: Directory not found: {videos_dir}")
        sys.exit(1)
    
    extract_all_videos_in_directory(videos_dir)
    print("\n✓ All videos processed!")