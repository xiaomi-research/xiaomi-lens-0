#!/bin/bash
# Copyright 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fine-tune a released Xiaomi-Lens-0 checkpoint on a custom JSONL manifest.
set -euo pipefail

usage() {
    echo "Usage: DATA_CONFIG=path/to/data.yaml bash scripts/finetune.sh {full|lora} [train.py arguments]"
    echo "Optional environment variables: MODEL_PATH, OUTPUT_DIR, NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, DEEPSPEED_CONFIG (full mode only)."
}

MODE="${1:-}"
case "${MODE}" in
    full|lora) shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_HOME}"

: "${DATA_CONFIG:?Set DATA_CONFIG to a YAML data manifest (see examples/train/README.md)}"
MODEL_PATH="${MODEL_PATH:-xiaomi-research/Xiaomi-Lens-0}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lens-${MODE}}"

if [[ "${MODE}" == "full" ]]; then
    MODE_ARGS=(
        --deepspeed "${DEEPSPEED_CONFIG:-src/xiaomi_lens/train/configs/zero2.json}"
        --unfreeze_all_parameters true
        --learning_rate 5e-6
    )
else
    MODE_ARGS=(--use_lora true --learning_rate 1e-4)
fi

NNODES="${NNODES:-1}"
if [[ "${NNODES}" -gt 1 ]]; then
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
fi
LAUNCHER_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE:-1}" --nnodes "${NNODES}"
    --node_rank "${NODE_RANK:-0}" --master_addr "${MASTER_ADDR:-127.0.0.1}"
    --master_port "${MASTER_PORT:-29500}"
)

exec python -m torch.distributed.run "${LAUNCHER_ARGS[@]}" train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_config "${DATA_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 --attn_implementation sdpa \
    --per_device_train_batch_size 1 --max_pixels 4194304 \
    --num_train_epochs 1 \
    "${MODE_ARGS[@]}" \
    "$@"
