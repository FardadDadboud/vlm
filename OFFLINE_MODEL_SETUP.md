# Offline Model Setup for GroundingDINO

This guide explains how to use GroundingDINO models on cluster nodes without internet access.

## Method 1: Download Model to Local Directory (Recommended)

### Step 1: Download the Model

On a machine **with internet access**, run:

```bash
python download_grounding_dino.py --output_dir /path/to/models/grounding-dino-base
```

Or specify a different model:

```bash
python download_grounding_dino.py --model IDEA-Research/grounding-dino-tiny --output_dir /path/to/models/grounding-dino-tiny
```

### Step 2: Copy to Cluster

Copy the downloaded model directory to your cluster:

```bash
# Example: using scp
scp -r /path/to/models/grounding-dino-base user@cluster:/path/on/cluster/models/

# Or using rsync
rsync -avz /path/to/models/grounding-dino-base user@cluster:/path/on/cluster/models/
```

### Step 3: Use in Your Code

When initializing the detector, use the local path:

```python
from vlm_detector_system_new import GroundingDINODetector

# Use local path
detector = GroundingDINODetector(
    model_path="/path/on/cluster/models/grounding-dino-base",
    device="cuda"
)
```

## Method 2: Use Hugging Face Cache Directory

### Step 1: Download Model (with internet)

On a machine with internet, run Python to download and cache:

```python
from transformers import pipeline

# This will download to ~/.cache/huggingface/hub/
pipe = pipeline(
    "zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-base"
)
```

### Step 2: Copy Cache Directory

Copy the Hugging Face cache to your cluster:

```bash
# Copy the entire cache directory
scp -r ~/.cache/huggingface user@cluster:/path/on/cluster/.cache/
```

### Step 3: Set Environment Variables

On your cluster, set:

```bash
export HF_HOME=/path/on/cluster/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
```

Then use the model name as usual:

```python
detector = GroundingDINODetector(
    model_path="IDEA-Research/grounding-dino-base",
    device="cuda"
)
```

## Method 3: Environment Variable for Offline Mode

If you've already cached the model on the cluster (e.g., from a previous run with internet), you can force offline mode:

```bash
export TRANSFORMERS_OFFLINE=1
```

Then run your code normally. The code will automatically use `local_files_only=True` when this environment variable is set.

## Troubleshooting

### Error: "We couldn't connect to 'https://huggingface.co'"

This means the model files are not available locally. Solutions:

1. **Use Method 1**: Download the model to a local directory and use that path
2. **Use Method 2**: Copy the Hugging Face cache directory to your cluster
3. **Check cache location**: Verify the model is in the expected cache directory

### Check if Model is Cached

```python
from transformers import file_utils

# Check cache location
cache_dir = file_utils.default_cache_path
print(f"Cache directory: {cache_dir}")

# List cached models
import os
cache_path = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_path):
    print("Cached models:")
    for item in os.listdir(cache_path):
        print(f"  - {item}")
```

## Notes

- The code automatically detects if `model_path` is a local directory
- When using a local directory, `local_files_only=True` is automatically set
- Setting `TRANSFORMERS_OFFLINE=1` also forces offline mode
- Model files are typically several GB in size, so ensure you have enough disk space

