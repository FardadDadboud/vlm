#!/usr/bin/env bash
#SBATCH --job-name=trust_yolo_grid
#SBATCH --account=def-mbolic_gpu
#SBATCH --array=0-2                     # = length of chunk axis (score_blend_lambda has 3 values); 18 configs/chunk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=249G
#SBATCH --time=05:59:00                 # YOLO-World is fast (real-time detector)
#SBATCH --output=%x_chunk%a_%A.out
set -euo pipefail

# ================== paths (EDIT to your allocation) ==================
PROJECT=~/projects/def-mbolic/$USER/trust
VLM=$PROJECT/code/VLM
VENV=$PROJECT/pinned_venv_457            # MUST also have: pip install ultralytics
export HF_HOME=$PROJECT/hf_cache
GRID=$VLM/grid_search/narval/grid_config_yolo_world_pooled13.json
RUNNER=$VLM/grid_search/narval/grid_search_runner_multidet.py
SCRATCH_OUT=$SCRATCH/trust_yolo_grid
PROJECT_OUT=$PROJECT/grid_results/yolo_world
NARVAL_DATA_ROOT=$PROJECT/data/SHIFT/shift-dev/data
MODEL_PATH=$PROJECT/models/yolov8s-world.pt              # <-- upload the .pt to /project

module load StdEnv/2023 gcc/12.3 python/3.11 opencv/4.11.0
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import transformers, ultralytics; assert transformers.__version__=='4.57.6', transformers.__version__; print('transformers', transformers.__version__, '| ultralytics', ultralytics.__version__)"

K=${SLURM_ARRAY_TASK_ID}
CHUNK_OUT="$SCRATCH_OUT/chunk_$(printf '%02d' "$K")"
mkdir -p "$CHUNK_OUT"

CHUNK_CFG="$SLURM_TMPDIR/yolo_chunk_${K}.json"
python - "$GRID" "$CHUNK_CFG" "$K" "$CHUNK_OUT" "$VLM" "$NARVAL_DATA_ROOT" "$MODEL_PATH" "$SLURM_TMPDIR" "${SLURM_ARRAY_TASK_COUNT:-3}" <<'PY'
import json, sys, os
grid_p, out_p, k, chunk_out, vlm, data_root, model_path, tmp, ntasks = sys.argv[1:10]
k = int(k); ntasks = int(ntasks)
g = json.load(open(grid_p))
base = json.load(open(os.path.join(vlm, g["base_config_path"])))
base["data_root"] = data_root
base["detector"]["model_path"] = model_path
base_patched = os.path.join(tmp, "base_patched.json")
json.dump(base, open(base_patched, "w"), indent=2)
g["base_config_path"] = base_patched
gp = g["grid_params"]
chunk_key = next(kk for kk, vv in gp.items() if len(vv) > 1)
vals = gp[chunk_key]
assert len(vals) == ntasks, f"array size {ntasks} != chunk-axis '{chunk_key}' length {len(vals)}; fix #SBATCH --array"
gp[chunk_key] = [vals[k]]
g["output_dir"] = chunk_out
json.dump(g, open(out_p, "w"), indent=2)
n = 1
for v in gp.values(): n *= len(v)
print(f"chunk {k}: {chunk_key}={vals[k]} -> {n} configs; model={model_path}")
PY

cd "$VLM"
python "$RUNNER" --config "$CHUNK_CFG"

mkdir -p "$PROJECT_OUT"
rsync -a "$CHUNK_OUT" "$PROJECT_OUT/"
echo "yolo chunk $K done; rsync'd $CHUNK_OUT -> $PROJECT_OUT/"
