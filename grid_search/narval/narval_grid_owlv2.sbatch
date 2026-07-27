#!/usr/bin/env bash
#SBATCH --job-name=trust_owlv2_grid
#SBATCH --account=def-mbolic_gpu
#SBATCH --array=0-2                     # = length of the chunk axis (tau1 has 3 values); 18 configs/chunk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=249G
#SBATCH --time=11:59:00
#SBATCH --output=%x_chunk%a_%A.out
set -euo pipefail

# ================== paths (EDIT to your allocation) ==================
PROJECT=~/projects/def-mbolic/$USER/trust
VLM=$PROJECT/code/VLM
VENV=$PROJECT/pinned_venv_457
export HF_HOME=$PROJECT/hf_cache
GRID=$VLM/grid_search/narval/grid_config_owlv2_pooled13.json
RUNNER=$VLM/grid_search/narval/grid_search_runner_multidet.py
SCRATCH_OUT=$SCRATCH/trust_owlv2_grid
PROJECT_OUT=$PROJECT/grid_results/owlv2
# injected at TOP LEVEL into the (copied) base config:
NARVAL_DATA_ROOT=$PROJECT/data/SHIFT/shift-dev/data      # <-- SHIFT on /project
MODEL_PATH=google/owlv2-base-patch16-ensemble            # cached in $HF_HOME (offline)

module load StdEnv/2023 gcc/12.3 python/3.11 opencv/4.11.0
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import transformers; assert transformers.__version__=='4.57.6', transformers.__version__; print('transformers', transformers.__version__)"

K=${SLURM_ARRAY_TASK_ID}
CHUNK_OUT="$SCRATCH_OUT/chunk_$(printf '%02d' "$K")"
mkdir -p "$CHUNK_OUT"

# ---- derive chunk config: patch base (data_root/model_path at TOP LEVEL),
#      pin the FIRST swept param (len>1) for this array index --------------
CHUNK_CFG="$SLURM_TMPDIR/owlv2_chunk_${K}.json"
python - "$GRID" "$CHUNK_CFG" "$K" "$CHUNK_OUT" "$VLM" "$NARVAL_DATA_ROOT" "$MODEL_PATH" "$SLURM_TMPDIR" "${SLURM_ARRAY_TASK_COUNT:-3}" <<'PY'
import json, sys, os
grid_p, out_p, k, chunk_out, vlm, data_root, model_path, tmp, ntasks = sys.argv[1:10]
k = int(k); ntasks = int(ntasks)
g = json.load(open(grid_p))

# patch the base config at TOP LEVEL (not dotted grid_params)
base = json.load(open(os.path.join(vlm, g["base_config_path"])))
base["data_root"] = data_root
base["detector"]["model_path"] = model_path
base_patched = os.path.join(tmp, "base_patched.json")
json.dump(base, open(base_patched, "w"), indent=2)
g["base_config_path"] = base_patched

# chunk axis = first grid_param whose value list has length > 1 (robust; no
# hardcoded dimension -> avoids the length-1 assert bug)
gp = g["grid_params"]
chunk_key = next(kk for kk, vv in gp.items() if len(vv) > 1)
vals = gp[chunk_key]
assert len(vals) == ntasks, f"array size {ntasks} != chunk-axis '{chunk_key}' length {len(vals)}; fix #SBATCH --array"
gp[chunk_key] = [vals[k]]
g["output_dir"] = chunk_out
json.dump(g, open(out_p, "w"), indent=2)
n = 1
for v in gp.values():
    n *= len(v)
print(f"chunk {k}: {chunk_key}={vals[k]} -> {n} configs; base data_root={data_root} model={model_path}")
PY

cd "$VLM"
python "$RUNNER" --config "$CHUNK_CFG"

mkdir -p "$PROJECT_OUT"
rsync -a "$CHUNK_OUT" "$PROJECT_OUT/"
echo "owlv2 chunk $K done; rsync'd $CHUNK_OUT -> $PROJECT_OUT/"
