#!/bin/bash
# One array task = one (point, dataset, seed, model) cell = one GPU.
#
# Resource directives (--array/--partition/--time/...) are supplied by the generated submit.sh,
# so this file stays site-agnostic. Everything else arrives through SWEEP_* env vars.
#SBATCH --job-name=opssm-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

set -uo pipefail

: "${SWEEP_ROOT:?set by submit.sh}"
: "${SWEEP_REPO:?set by submit.sh}"
SWEEP_PYTHON="${SWEEP_PYTHON:-python}"
SWEEP_VAL_EVERY="${SWEEP_VAL_EVERY:-100}"
SWEEP_SAMPLES="${SWEEP_SAMPLES:-128}"
SWEEP_SOLVER_DT="${SWEEP_SOLVER_DT:-0.01}"
SWEEP_MONITOR="${SWEEP_MONITOR:-val_forecast_rmse}"
SWEEP_GT_DIAG="${SWEEP_GT_DIAG:-1}"

[ -n "${SWEEP_SETUP:-}" ] && eval "${SWEEP_SETUP}"

# Read this task's cell from the manifest (line 1 is the header, so offset by one).
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SWEEP_ROOT}/manifest.tsv")
[ -z "${LINE}" ] && { echo "No manifest row for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }
IFS=$'\t' read -r IDX POINT DATASET SEED MODEL STEPS <<< "${LINE}"

OUT="${SWEEP_ROOT}/results/${POINT}"
LOG="${SWEEP_ROOT}/logs/${POINT}_${DATASET}_seed${SEED}_${MODEL}.log"
CONFIG="${SWEEP_ROOT}/points/${POINT}.json"

# Idempotent: a requeued task must resume the sweep, not abort on its own partial output.
if [ -f "${OUT}/${DATASET}/seed_${SEED}/${MODEL}/result.json" ]; then
  echo "[$(date -u +%FT%TZ)] task ${IDX} already complete: ${POINT}/${DATASET}/seed_${SEED}/${MODEL}"
  exit 0
fi

export JAX_PLATFORMS="${SWEEP_PLATFORM:-cuda}"   # cpu is useful for a dry run
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SWEEP_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
# Deliberately NOT setting XLA_PYTHON_CLIENT_PREALLOCATE=false: disabling preallocation is what
# produced the em_lorenz OOM. Leave XLA's default allocator alone.
unset XLA_PYTHON_CLIENT_PREALLOCATE
# An inherited CPU binding from an enclosing allocation is applied to this job step and makes
# srun reject it as outside the allocation.
unset SLURM_CPU_BIND

ARGS=(--datasets "${DATASET}" --models "${MODEL}" --seeds "${SEED}"
      --steps "${STEPS}" --val-every "${SWEEP_VAL_EVERY}" --samples "${SWEEP_SAMPLES}"
      --solver-dt "${SWEEP_SOLVER_DT}" --out "${OUT}" --skip-existing
      --opssm-monitor "${SWEEP_MONITOR}")
[ "${SWEEP_GT_DIAG}" = "1" ] && ARGS+=(--opssm-gt-diagnostics)
# An empty overrides file means the unmodified preset; passing it would be a no-op either way.
[ -s "${CONFIG}" ] && [ "$(tr -d '[:space:]' < "${CONFIG}")" != "{}" ] && ARGS+=(--dataset-config "${CONFIG}")

echo "[$(date -u +%FT%TZ)] task ${IDX}: ${POINT} ${DATASET} seed=${SEED} ${MODEL} steps=${STEPS} node=$(hostname)" | tee -a "${LOG}"
cd "${SWEEP_REPO}"
"${SWEEP_PYTHON}" -u -m opssm.benchmarks.runner "${ARGS[@]}" >> "${LOG}" 2>&1
CODE=$?
echo "[$(date -u +%FT%TZ)] task ${IDX} exit=${CODE}" | tee -a "${LOG}"
exit ${CODE}
