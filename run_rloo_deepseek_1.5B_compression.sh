#!/usr/bin/env bash

# Non-Slurm version of run_rloo_1.5B.sh with these requested changes:
# 16 responses per prompt; one optimizer update per RL step; LR 1e-6;
# KL coefficient 0; alpha 0.1; save every 20 RL steps.
# Keep the paper's 32 prompts/step: 32 * 16 = 512 responses/update.
# The current reward server uses MathVerify 0.9.0 for correctness.
# Gold comes directly from each sample's extracted field.
#
# Preview without downloading data, starting services, or using GPUs:
#   DRY_RUN=1 bash run_rloo_deepseek_1.5B_compression.sh
# Launch with automatic upload/verification/deletion of local checkpoints:
#   HF_REPO_PREFIX=YOUR_USER/er-r1-distill-1.5b-compression-n16-extracted \
#     bash run_rloo_deepseek_1.5B_compression.sh
# Archives use public model repos named ${HF_REPO_PREFIX}-step_<N>.
# Set ARCHIVE_CHECKPOINTS=0 only when keeping checkpoints locally is intended.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DATASET=datasets/compression_dataset
DATASET_REPO=daman1209arora/compression_dataset
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
TRAIN_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
INPUT_TEMPLATE=$'<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{{}}. Question: {}<｜Assistant｜>'

# Only operational settings are environment overrides. Training hyperparameters
# below are explicit so exports from earlier experiments cannot change them.
CONDA_ENV=${CONDA_ENV:-efficient_reasoning}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_NAME=${RUN_NAME:-rloo_r1_distill_1.5b_compression_n16_b512_lr1e-6_kl0_alpha0.1_seed42_extracted}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs}
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
CKPT_PATH=${RUN_DIR}/checkpoints
SAVE_PATH=${RUN_DIR}/final_model
LOG_DIR=${RUN_DIR}/logs
RM_HOST=127.0.0.1
RM_PORT=${RM_PORT:-24373}
RM_URL=http://${RM_HOST}:${RM_PORT}/query
VERIFIER_WORKERS=${VERIFIER_WORKERS:-16}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-0}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-efficient_reasoning_compression}
ARCHIVE_CHECKPOINTS=${ARCHIVE_CHECKPOINTS:-1}
HF_REPO_PREFIX=${HF_REPO_PREFIX:-}
ARCHIVER=${REPO_ROOT}/archive_openrlhf_checkpoints_to_hf.sh

for option in DRY_RUN RESUME USE_WANDB ARCHIVE_CHECKPOINTS; do
    [[ "${!option}" == "0" || "${!option}" == "1" ]] || {
        echo "${option} must be 0 or 1." >&2
        exit 2
    }
done
[[ "${GPU_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || {
    echo "GPU_IDS must list four physical GPU IDs, for example 0,1,2,3." >&2
    exit 2
}
[[ "${RM_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] && (( RM_PORT <= 65535 )) || {
    echo "RM_PORT must be a valid TCP port." >&2
    exit 2
}

REWARD_CMD=(
    python -m reward_server.math_server
    --address "${RM_HOST}:${RM_PORT}"
    --dataset "${DATASET}"
    --tokenizer "${MODEL}"
    --reward_type sigmoid
    --alpha 0.1
    --check_eos
    --verifier_workers "${VERIFIER_WORKERS}"
)
TRAIN_CMD=(
    python -m openrlhf.cli.train_ppo_ray
    --pretrain "${MODEL}"
    --advantage_estimator rloo
    --num_episodes 1
    --max_epochs 1
    --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
    --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --micro_train_batch_size 1
    --micro_rollout_batch_size 1
    --prompt_data "${DATASET}"
    --prompt_data_probs 1.0
    --prompt_split train
    --input_key problem
    --input_template "${INPUT_TEMPLATE}"
    --max_samples 3200
    --prompt_max_len 512
    --generate_max_len 32000
    --temperature 1.0
    --top_p 1.0
    --min_p 0
    --actor_learning_rate 1e-6
    --init_kl_coef 0
    --scheduler_type warmup_with_constant_lr
    --lr_warmup_ratio 0.03
    --adam_betas 0.9 0.95
    --l2 0
    --max_norm 1.0
    --eps_clip 0.2
    --zero_stage 2
    --bf16
    --flash_attn
    --gradient_checkpointing
    --seed 42
    --ref_num_nodes 1
    --ref_num_gpus_per_node 2
    --actor_num_nodes 1
    --actor_num_gpus_per_node 2
    --colocate_actor_ref
    --colocate_critic_reward
    --vllm_num_engines 2
    --vllm_tensor_parallel_size 1
    --remote_rm_url "${RM_URL}"
    --save_path "${SAVE_PATH}"
    --ckpt_path "${CKPT_PATH}"
    --save_steps 20
    --max_ckpt_num 10
    --wandb_project "${WANDB_PROJECT}"
    --wandb_run_name "${RUN_NAME}"
)
if [[ "${USE_WANDB}" == "1" ]]; then
    # Trainer reads cached/environment credentials; never put a key in argv.
    TRAIN_CMD+=(--use_wandb enabled)
    if [[ -n "${WANDB_ORG:-}" ]]; then
        TRAIN_CMD+=(--wandb_org "${WANDB_ORG}")
    fi
else
    TRAIN_CMD+=(--use_tensorboard "${RUN_DIR}/tensorboard")
fi
if [[ "${RESUME}" == "1" ]]; then
    TRAIN_CMD+=(--load_checkpoint)
fi

echo "32 prompts x 16 responses = 512 responses; batch 512 = 1 optimizer update/RL step."
echo "1 episode over 3200 prompts = 100 RL steps; checkpoints at 20, 40, 60, 80, 100."
echo "Output cap: 32000 tokens; prompt cap: 512; vLLM context: 32512."
echo "LR: 1e-6 with original warmup; KL: 0; sigmoid alpha: 0.1."
echo "Gold: sample.extracted; no solution parsing or question-text lookup."
if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Reward command:\n'
    printf '  %q' "${REWARD_CMD[@]}"
    printf '\nTraining command:\n'
    printf '  %q' "${TRAIN_CMD[@]}"
    printf '\nArchive checkpoints: %s; HF prefix: %s\n' \
        "${ARCHIVE_CHECKPOINTS}" "${HF_REPO_PREFIX:-set HF_REPO_PREFIX before launching}"
    exit 0
fi

if [[ "${ARCHIVE_CHECKPOINTS}" == "1" && ! "${HF_REPO_PREFIX}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Set HF_REPO_PREFIX=owner/name for checkpoint uploads, or ARCHIVE_CHECKPOINTS=0 for local storage." >&2
    exit 2
fi
if [[ "${RESUME}" != "1" && -e "${RUN_DIR}" ]]; then
    echo "Run directory exists: ${RUN_DIR}. Choose another RUN_NAME or set RESUME=1." >&2
    exit 1
fi
if [[ "${RESUME}" == "1" ]]; then
    [[ -s "${CKPT_PATH}/_actor/latest" ]] && \
        [[ -d "${CKPT_PATH}/_actor/$(tr -d '[:space:]' <"${CKPT_PATH}/_actor/latest")" ]] || {
        echo "No local checkpoint to resume. Restore an archived checkpoint first." >&2
        exit 1
    }
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_ADDRESS=local
# Keep Ray's socket paths below the AF_UNIX length limit.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/er_r1_s42}
export TMPDIR=${TMPDIR:-${RUN_DIR}/tmp}
unset TRANSFORMERS_CACHE

IFS=',' read -r -a selected_gpus <<<"${GPU_IDS}"
for gpu_id in "${selected_gpus[@]}"; do
    busy=$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "${busy}" ]]; then
        echo "GPU ${gpu_id} is busy (PIDs: ${busy}); wait or select four idle GPUs with GPU_IDS." >&2
        exit 1
    fi
done
if [[ "${ARCHIVE_CHECKPOINTS}" == "1" ]]; then
    command -v hf >/dev/null
    command -v flock >/dev/null
    [[ -f "${ARCHIVER}" ]]
    python -c 'from huggingface_hub import HfApi; print("HF upload account:", HfApi().whoami()["name"])'
fi
command -v setsid >/dev/null

# Match the README download route; do not rewrite or regenerate the dataset.
python - "${DATASET}" "${DATASET_REPO}" "${RM_HOST}" "${RM_PORT}" "${USE_WANDB}" <<'PY'
import socket
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from huggingface_hub import snapshot_download

dataset_path, dataset_repo, host, port, use_wandb = sys.argv[1:]
if version("math-verify") != "0.9.0":
    raise SystemExit("This experiment requires math-verify==0.9.0")
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("Expected four distinct, visible CUDA GPUs")
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(4)])
with socket.socket() as probe:
    probe.bind((host, int(port)))
if use_wandb == "1":
    import wandb

    if not wandb.api.api_key:
        raise SystemExit("Configure W&B credentials, or set USE_WANDB=0 for TensorBoard")
dataset_dir = Path(dataset_path)
if not (dataset_dir / "state.json").is_file() and not (dataset_dir / "dataset_dict.json").is_file():
    snapshot_download(repo_id=dataset_repo, repo_type="dataset", local_dir=dataset_path)
data = load_from_disk(dataset_path)
train = data["train"] if isinstance(data, DatasetDict) else data
if not {"problem", "extracted"}.issubset(train.column_names) or len(train) < 3200:
    raise SystemExit("Expected compression_dataset with problem/extracted columns and at least 3200 rows")
if any(row["extracted"] is None or not str(row["extracted"]).strip() for row in train.select(range(3200))):
    raise SystemExit("Every selected training sample must have a non-empty extracted gold answer")
print(f"Dataset ready: {len(train)} rows; training selects 3200")
PY

mkdir -p "${LOG_DIR}" "${CKPT_PATH}" "${SAVE_PATH}" "${RAY_TMPDIR}" "${TMPDIR}"
TRAIN_PID=""
RM_PID=""
ARCHIVE_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    # Both groups were started by this launcher with setsid.
    for pid in "${TRAIN_PID}" "${RM_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done
    # The watcher can finish uploading after the trainer exits.
    if [[ -n "${ARCHIVE_PID}" ]]; then
        echo "Checkpoint watcher PID ${ARCHIVE_PID}: ${LOG_DIR}/checkpoint_archiver.log"
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "${REWARD_CMD[@]}" >"${LOG_DIR}/reward_server.log" 2>&1 &
RM_PID=$!
ready=0
for _ in {1..300}; do
    kill -0 "${RM_PID}" 2>/dev/null || {
        echo "Reward server exited; see ${LOG_DIR}/reward_server.log." >&2
        exit 1
    }
    if python - "${RM_HOST}" "${RM_PORT}" <<'PY' 2>/dev/null
import socket
import sys

with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
    pass
PY
    then
        ready=1
        break
    fi
    sleep 1
done
[[ "${ready}" == "1" ]] || { echo "Reward server startup timed out." >&2; exit 1; }

setsid "${TRAIN_CMD[@]}" > >(tee "${LOG_DIR}/training.log") 2>&1 &
TRAIN_PID=$!
echo "Training PID: ${TRAIN_PID}; logs: ${LOG_DIR}"
if [[ "${ARCHIVE_CHECKPOINTS}" == "1" ]]; then
    PYTHON_BIN=$(command -v python) HF_ARCHIVE_EXPECTED_WORLD_SIZE=2 \
        HF_ARCHIVE_UPLOAD_LOCK="${OUTPUT_ROOT}/.hf_checkpoint_upload.lock" \
        bash "${ARCHIVER}" "${CKPT_PATH}" "${HF_REPO_PREFIX}" "${TRAIN_PID}" \
        >"${LOG_DIR}/checkpoint_archiver.log" 2>&1 &
    ARCHIVE_PID=$!
fi
status=0
wait "${TRAIN_PID}" || status=$?
TRAIN_PID=""
if [[ -n "${ARCHIVE_PID}" ]]; then
    wait "${ARCHIVE_PID}" || status=1
    ARCHIVE_PID=""
fi
exit "${status}"
