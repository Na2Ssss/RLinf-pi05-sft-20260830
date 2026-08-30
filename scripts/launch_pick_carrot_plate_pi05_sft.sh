#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

MODE="${1:-train}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "train" ]]; then
    echo "Usage: $0 [smoke|train]" >&2
    exit 2
fi

REPO_ROOT="/data0/zhibo/storage/code/RLinf-pi05-sft-20260830"
VENV_ROOT="${REPO_ROOT}/.venv"
EXPERIMENT_ROOT="/data0/zhibo/storage/experiments/rlinf_pi05_pick_carrot_plate_absquat_20eps_20260830"
DATASET_ROOT="/data0/zhibo/storage/datasets/pick_carrot_plate_rlinf_pi05_absolute_quat_20eps_seed20260830"
MODEL_ROOT="/data0/zhibo/storage/models/pi05_base_openpi_rlinf_fp32_20260830"
NORM_STATS="${EXPERIMENT_ROOT}/assets/pick_carrot_plate_absolute_quat/norm_stats.json"
CONFIG_NAME="pick_carrot_plate_pi05_absolute_quat"
TRAIN_ENTRY="${REPO_ROOT}/examples/sft/train_vla_sft.py"
RUNS_ROOT="${EXPERIMENT_ROOT}/runs"
STAMP="$(date +'%Y%m%d-%H%M%S')"
LAUNCH_LOG_DIR="${EXPERIMENT_ROOT}/launcher_logs/${STAMP}-${MODE}"
LOG_FILE="${LAUNCH_LOG_DIR}/train.log"

for required_file in \
    "${VENV_ROOT}/bin/python" \
    "${DATASET_ROOT}/meta/info.json" \
    "${MODEL_ROOT}/model.safetensors" \
    "${MODEL_ROOT}/config.json" \
    "${NORM_STATS}" \
    "${TRAIN_ENTRY}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if pgrep -u "$(id -u)" -f "${TRAIN_ENTRY}.*${CONFIG_NAME}" >/dev/null; then
    echo "A matching carrot-plate SFT process is already running; refusing a duplicate launch." >&2
    exit 1
fi

# Refuse to collide with another workload. Small driver/monitor contexts are allowed.
for physical_gpu in 1 2; do
    IFS=, read -r used_mib utilization < <(
        nvidia-smi --id="${physical_gpu}" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' '
    )
    if (( used_mib >= 2048 || utilization > 5 )); then
        echo "Physical GPU ${physical_gpu} is not idle: ${used_mib} MiB, ${utilization}% utilization." >&2
        exit 1
    fi
done

mkdir -p "${LAUNCH_LOG_DIR}" "${RUNS_ROOT}"
cd "${REPO_ROOT}"

# Physical GPU isolation was verified with Ray workers: logical workers map back
# to physical IDs 1 and 2, respectively.
export CUDA_VISIBLE_DEVICES="1,2"
export RAY_ADDRESS="local"
export RAY_TMPDIR="/data0/zhibo/rpi05-${MODE}-$$"
export RAY_OVERRIDE_RESOURCES='{"CPU": 8, "GPU": 2, "object_store_memory": 21474836480}'
export RAY_DEDUP_LOGS="0"
export RAY_DISABLE_DOCKER_CPU_WARNING="1"
# Avoid starving fresh workers on this heavily loaded shared host.
export RAY_worker_niceness="0"
export EMBODIED_PATH="${REPO_ROOT}/examples/sft"
export OPENPI_DATA_HOME="${EXPERIMENT_ROOT}/cache/openpi"
export JAX_PLATFORMS="cpu"
export LD_LIBRARY_PATH="${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="8"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"

overrides=("runner.logger.log_path=${RUNS_ROOT}")
if [[ "${MODE}" == "smoke" ]]; then
    overrides+=(
        "runner.max_epochs=-1"
        "runner.max_steps=1"
        "runner.save_interval=-1"
        "runner.logger.experiment_name=pi05_absquat_h32_fsdp2_gpu1_2_smoke"
        "data.num_workers=0"
    )
fi

{
    echo "mode=${MODE}"
    echo "physical_gpus=${CUDA_VISIBLE_DEVICES}"
    echo "ray_tmpdir=${RAY_TMPDIR}"
    echo "model=${MODEL_ROOT}"
    echo "dataset=${DATASET_ROOT}"
    echo "norm_stats=${NORM_STATS}"
    nvidia-smi --id=1,2 \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
} | tee "${LOG_FILE}"

"${VENV_ROOT}/bin/python" "${TRAIN_ENTRY}" \
    --config-path "${REPO_ROOT}/examples/sft/config" \
    --config-name "${CONFIG_NAME}" \
    "${overrides[@]}" 2>&1 | tee -a "${LOG_FILE}"

if grep -Eq \
    '^Error executing job|^Traceback \(most recent call last\):|Exiting main process due to a failure' \
    "${LOG_FILE}"; then
    echo "RLinf reported an error; see ${LOG_FILE}" >&2
    exit 1
fi

echo "Completed ${MODE}; log=${LOG_FILE}" | tee -a "${LOG_FILE}"
