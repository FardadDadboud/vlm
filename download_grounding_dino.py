#!/usr/bin/env python3
"""
Script to download GroundingDINO model for offline use.
Run this script on a machine with internet access, then copy the downloaded
model to your cluster.

Usage:
    python download_grounding_dino.py --output_dir /path/to/models/grounding-dino-base
"""

import argparse
import os
from pathlib import Path
from transformers import pipeline

def download_model(model_name: str = "IDEA-Research/grounding-dino-base", output_dir: str = None):
    """
    Download GroundingDINO model and save it locally.
    
    Args:
        model_name: Hugging Face model identifier
        output_dir: Directory to save the model (default: ./models/{model_name})
    """
    if output_dir is None:
        output_dir = f"./models/{model_name.replace('/', '_')}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name} to {output_path}...")
    print("This may take a few minutes...")
    
    # Download the pipeline (this will download all necessary files)
    pipe = pipeline(
        "zero-shot-object-detection",
        model=model_name,
        device=-1  # CPU for downloading
    )
    
    # Save the model
    print(f"\nSaving model to {output_path}...")
    pipe.model.save_pretrained(str(output_path))
    pipe.image_processor.save_pretrained(str(output_path))
    pipe.tokenizer.save_pretrained(str(output_path))
    
    print(f"\n✓ Model downloaded successfully to: {output_path}")
    print(f"\nTo use this model offline:")
    print(f"  1. Copy the directory '{output_path}' to your cluster")
    print(f"  2. Set model_path='{output_path}' when initializing GroundingDINODetector")
    print(f"  3. Or set environment variable: export TRANSFORMERS_OFFLINE=1")
    
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GroundingDINO model for offline use")
    parser.add_argument(
        "--model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face model identifier (default: IDEA-Research/grounding-dino-base)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the model (default: ./models/{model_name})"
    )
    
    args = parser.parse_args()
    
    download_model(args.model, args.output_dir)

