#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${RLINF_CARROT_VENV:-${REPO_ROOT}/.venv}"
UV_CACHE_DIR="${RLINF_CARROT_UV_CACHE:-/data0/zhibo/storage/cache/uv-rlinf-pi05-pytorch211}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data0/zhibo/storage/experiments/rlinf_pi05_pick_carrot_plate_absquat_20eps_20260830/cache/openpi}"
OVERRIDES="${REPO_ROOT}/requirements/embodied/models/openpi_pytorch211_overrides.txt"

export UV_CACHE_DIR OPENPI_DATA_HOME

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    uv venv --python 3.11.15 "${VENV_DIR}"
fi

uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --torch-backend auto \
    --overrides "${OVERRIDES}" \
    -e "${REPO_ROOT}[embodied]" \
    "rlinf-openpi==0.1.1" \
    "torchaudio==2.11.0" \
    "cuda-toolkit[npp,nvjpeg]==13.0.2" \
    -r "${REPO_ROOT}/requirements/embodied/models/openpi.txt"

SITE_PACKAGES="$("${VENV_DIR}/bin/python" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"

# RLinf's OpenPI fork ships patched SigLIP/Gemma implementations that must
# replace the stock modules after dependency resolution.
cp -a \
    "${SITE_PACKAGES}/openpi/models_pytorch/transformers_replace/." \
    "${SITE_PACKAGES}/transformers/"

uv pip uninstall --python "${VENV_DIR}/bin/python" pynvml >/dev/null 2>&1 || true

# Populate the cache path used by openpi.models.tokenizer.  Reuse the already
# downloaded official tokenizer when available; otherwise OpenPI downloads it.
TOKENIZER="${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
if [[ ! -f "${TOKENIZER}" ]]; then
    mkdir -p "$(dirname -- "${TOKENIZER}")"
    EXISTING_TOKENIZER="/data0/zhibo/yuanxingball_pi05_rlinf_20260721/cache/openpi/big_vision/paligemma_tokenizer.model"
    if [[ -f "${EXISTING_TOKENIZER}" ]]; then
        cp -a "${EXISTING_TOKENIZER}" "${TOKENIZER}"
    else
        "${VENV_DIR}/bin/python" - <<'PY'
from openpi.shared.download import maybe_download

print(maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
PY
    fi
fi

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch

for package in ("torch", "torchvision", "torchaudio", "lerobot", "rlinf-openpi", "ray"):
    print(f"{package}={metadata.version(package)}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
