#!/usr/bin/env bash
#SBATCH --job-name=trust_pooled13_grid
#SBATCH --account=def-mbolic_gpu
#SBATCH --array=0-8                     # 9 chunks = the 3x3 (tau1 x alpha) combos; 64 configs/chunk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4               # a full Narval A100 node (4x A100-SXM4-40GB)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=249G
#SBATCH --time=23:59:00                 # <24h; each chunk (~13h) fits with margin
#SBATCH --output=%x_chunk%a_%A.out
#SBATCH --signal=B:USR1@120             # (optional) advance warning before preempt/timeout
#
# TRUST single-config JOINT grid search on Narval, PINNED transformers 4.57.6.
# Each array task processes one 64-config chunk on its own 4-A100 node, with an
# INDEPENDENT output_dir + checkpoint.json (so preemption/timeout is safe:
# resubmitting the array resumes each chunk from its checkpoint). Results are
# written to $SCRATCH then rsync'd to /project (scratch purges).
set -euo pipefail

# ================== paths (EDIT to match build_narval_env.sh) ==================
PROJECT=~/projects/def-mbolic/$USER/trust
VLM=$PROJECT/code/VLM                              # repo checkout on /project
VENV=$PROJECT/pinned_venv_457
export HF_HOME=$PROJECT/hf_cache
GRID=$VLM/grid_search/narval/grid_config_pooled13.json
SCRATCH_OUT=$SCRATCH/trust_grid                    # working outputs (scratch)
PROJECT_OUT=$PROJECT/grid_results                  # durable copy (project)

# ================== environment (offline compute node) =========================
module load StdEnv/2023 gcc/12.3 python/3.11 opencv/4.11.0
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # nodes have no internet; use cached model
python -c "import transformers; assert transformers.__version__=='4.57.6', transformers.__version__; print('transformers', transformers.__version__)"

K=${SLURM_ARRAY_TASK_ID}
CHUNK_OUT="$SCRATCH_OUT/chunk_$(printf '%02d' "$K")"
mkdir -p "$CHUNK_OUT"

# ============ derive this chunk's grid_config (pin tau1,alpha; sweep the rest) ==
# Chunk k selects tau1 = tau1_vals[k//3], alpha = alpha_vals[k%3] from the full grid.
CHUNK_CFG="$SLURM_TMPDIR/chunk_${K}.json"
python - "$GRID" "$CHUNK_CFG" "$K" "$CHUNK_OUT" <<'PY'
import json, sys
grid_path, out_path, k, chunk_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
g = json.load(open(grid_path))
gp = g["grid_params"]
tau1 = gp["adaptation.params.tau1"]
alpha = gp["adaptation.params.alpha"]
assert len(tau1) == 3 and len(alpha) == 3, "chunking assumes 3x3 tau1 x alpha"
gp["adaptation.params.tau1"]  = [tau1[k // 3]]     # pin
gp["adaptation.params.alpha"] = [alpha[k % 3]]     # pin
g["output_dir"] = chunk_out                        # independent output + checkpoint per chunk
json.dump(g, open(out_path, "w"), indent=2)
n = 1
for v in gp.values():
    n *= len(v)
print(f"chunk {k}: tau1={tau1[k//3]} alpha={alpha[k%3]}  -> {n} configs  out={chunk_out}")
PY

# ================== run (resumes from CHUNK_OUT/checkpoint.json) ================
cd "$VLM"                                           # base_config_path is relative to VLM
python grid_search/grid_search_runner.py --config "$CHUNK_CFG"

# ================== persist to /project IMMEDIATELY (scratch purges) ============
mkdir -p "$PROJECT_OUT"
rsync -a --info=progress2 "$CHUNK_OUT" "$PROJECT_OUT/"
echo "chunk $K done; rsync'd $CHUNK_OUT -> $PROJECT_OUT/"
