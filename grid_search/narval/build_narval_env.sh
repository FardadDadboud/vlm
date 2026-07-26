#!/usr/bin/env bash
# build_narval_env.sh  —  ONE-TIME setup on a Narval LOGIN node (has internet).
# Builds the PINNED transformers==4.57.6 venv and pre-caches the GroundingDINO
# model so the offline compute nodes can load it. Run this BEFORE sbatch.
#
# CRITICAL: the whole point of the pinned run is transformers==4.57.6 (the slow
# image processor that reproduces the paper's ~800 dets/frame). A 5.x env is
# worthless. Everything else (torch/numpy/scipy) may float on Narval's wheels.
#
# >>> VERIFY the three module lines against `module spider` on current Narval. <<<
set -euo pipefail

# ---- paths (EDIT to your allocation) ---------------------------------------
PROJECT=~/projects/def-mbolic/$USER/trust        # persistent /project location of the repo
VENV=$PROJECT/pinned_venv_457
export HF_HOME=$PROJECT/hf_cache                  # model cache lives on /project (offline nodes read it)

# ---- modules (Narval; verify names/versions) -------------------------------
module load StdEnv/2023 gcc/12.3 python/3.11 opencv/4.11.0 || {
  echo "!! module load failed — run 'module spider python opencv' and edit this line"; exit 2; }

# ---- venv ------------------------------------------------------------------
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# Compiled scientific stack from the Compute Canada wheelhouse (matches Narval CUDA):
pip install --no-index torch numpy scipy pillow pycocotools matplotlib tqdm

# The PINNED transformers from PyPI (login node has internet). pip auto-selects a
# compatible huggingface_hub (<1.0), tokenizers, safetensors into the venv.
pip install "transformers==4.57.6"

echo "=== installed versions ==="
python - <<'PY'
import torch, transformers, numpy, huggingface_hub
print("transformers", transformers.__version__, "(MUST be 4.57.6)")
print("torch", torch.__version__, "| numpy", numpy.__version__, "| hub", huggingface_hub.__version__)
from transformers import AutoProcessor, GroundingDinoForObjectDetection  # import smoke test
print("GroundingDINO classes import OK")
PY

# ---- pre-cache the model on the login node (compute nodes are OFFLINE) ------
echo "=== caching IDEA-Research/grounding-dino-base into $HF_HOME ==="
python - <<'PY'
from transformers import AutoProcessor, GroundingDinoForObjectDetection
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
print("model cached")
PY

echo
echo "DONE. venv=$VENV   HF_HOME=$HF_HOME"
echo "Next: edit paths in narval_grid_search.sbatch to match, then: sbatch narval_grid_search.sbatch"
