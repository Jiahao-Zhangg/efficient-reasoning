#!/usr/bin/env bash

# Four-GPU Efficient-Reasoning/RLOO run using Qwen3-1.7B and Math12K.
# It uses the model, data, and core rollout hyperparameters from MaxRL's
# Qwen3-1.7B Math12K experiment.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-efficient_reasoning}
MODEL=${MODEL:-Qwen/Qwen3-1.7B-Base}
DATASET=${DATASET:-hiyouga/math12k}
GPU_IDS=${GPU_IDS:-0,1,2,3}

RUN_NAME=${RUN_NAME:-rloo_qwen3_1.7b_math12k_len4096_seed79}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs}
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
SAVE_PATH=${RUN_DIR}/final_model
CKPT_PATH=${RUN_DIR}/checkpoints
LOG_DIR=${RUN_DIR}/logs

RM_HOST=${RM_HOST:-127.0.0.1}
RM_PORT=${RM_PORT:-24372}
RM_URL=http://${RM_HOST}:${RM_PORT}/query
VERIFIER_WORKERS=${VERIFIER_WORKERS:-16}

# Requested comparison settings. This repository uses RLOO rather than MaxRL
# as its advantage estimator.
NUM_EPISODES=${NUM_EPISODES:-5}
MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-}
# MaxRL batch configuration: 256 prompts x 16 responses, global PPO batch 256,
# and per-GPU micro-batch 4.
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
PROMPT_MAX_LEN=${PROMPT_MAX_LEN:-1024}
GENERATE_MAX_LEN=${GENERATE_MAX_LEN:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-12000}
ACTOR_LEARNING_RATE=${ACTOR_LEARNING_RATE:-1e-6}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-79}

# Efficient-Reasoning reward settings. Set REWARD_TYPE=linear for a plain
# binary MathVerify reward without the relative-length adjustment.
REWARD_TYPE=${REWARD_TYPE:-sigmoid}
ALPHA=${ALPHA:-0.1}
CHECK_EOS=${CHECK_EOS:-1}

SAVE_STEPS=${SAVE_STEPS:-50}
MAX_CKPT_NUM=${MAX_CKPT_NUM:-1}
RESUME=${RESUME:-0}
DRY_RUN=${DRY_RUN:-0}
ALLOW_BUSY_GPUS=${ALLOW_BUSY_GPUS:-0}

INPUT_TEMPLATE=$'{}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'

if [[ "${GPU_IDS}" != "0,1,2,3" ]]; then
    echo "GPU_IDS must be 0,1,2,3 for this four-GPU launcher; got ${GPU_IDS}." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    export PATH="/work2/jiahaoz4/miniconda3/condabin:${PATH}"
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
export HF_HOME=${HF_HOME:-/work2/jiahaoz4/.cache/huggingface}
export RAY_ADDRESS=local
# Ray appends long session/socket names and Linux limits AF_UNIX paths to 107
# bytes, so keep its temporary root deliberately short.
export RAY_TMPDIR=${RAY_TMPDIR:-/work2/jiahaoz4/.ray/q3_s${SEED}}
export TMPDIR=${TMPDIR:-${RUN_DIR}/tmp}
unset TRANSFORMERS_CACHE

REWARD_CMD=(
    python -m reward_server.math_server
    --address "${RM_HOST}:${RM_PORT}"
    --dataset "${DATASET}"
    --tokenizer "${MODEL}"
    --reward_type "${REWARD_TYPE}"
    --alpha "${ALPHA}"
    --verifier_workers "${VERIFIER_WORKERS}"
)
if [[ "${CHECK_EOS}" == "1" ]]; then
    REWARD_CMD+=(--check_eos)
else
    REWARD_CMD+=(--no-check_eos)
fi

TRAIN_CMD=(
    python -m openrlhf.cli.train_ppo_ray
    --advantage_estimator rloo
    --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}"
    --num_episodes "${NUM_EPISODES}"
    --max_epochs 1
    --remote_rm_url "${RM_URL}"
    --ref_num_nodes 1
    --ref_num_gpus_per_node 2
    --actor_num_nodes 1
    --actor_num_gpus_per_node 2
    --colocate_actor_ref
    --vllm_num_engines 2
    --vllm_tensor_parallel_size 1
    --pretrain "${MODEL}"
    --save_path "${SAVE_PATH}"
    --ckpt_path "${CKPT_PATH}"
    --save_steps "${SAVE_STEPS}"
    --max_ckpt_num "${MAX_CKPT_NUM}"
    --prompt_data "${DATASET}"
    --prompt_data_probs 1.0
    --prompt_split train
    --input_key problem
    --input_template "${INPUT_TEMPLATE}"
    --apply_chat_template
    --max_samples "${MAX_SAMPLES}"
    --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
    --micro_rollout_batch_size "${MICRO_BATCH_SIZE}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --micro_train_batch_size "${MICRO_BATCH_SIZE}"
    --prompt_max_len "${PROMPT_MAX_LEN}"
    --generate_max_len "${GENERATE_MAX_LEN}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --min_p 0
    --actor_learning_rate "${ACTOR_LEARNING_RATE}"
    --scheduler_type warmup_with_constant_lr
    --lr_warmup_ratio 0
    --l2 0.01
    --adam_betas 0.9 0.999
    --max_norm 0.3
    --init_kl_coef 0.0
    --zero_stage 2
    --bf16
    --flash_attn
    --gradient_checkpointing
    --seed "${SEED}"
    --wandb_project qwen3_math12k_comparison
    --wandb_run_name "${RUN_NAME}"
)

if [[ -n "${MAX_ROLLOUT_STEPS}" ]]; then
    if [[ ! "${MAX_ROLLOUT_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_ROLLOUT_STEPS must be a positive integer; got ${MAX_ROLLOUT_STEPS}." >&2
        exit 2
    fi
    TRAIN_CMD+=(--max_rollout_steps "${MAX_ROLLOUT_STEPS}")
fi
if [[ -n "${WANDB_ORG:-}" ]]; then
    TRAIN_CMD+=(--wandb_org "${WANDB_ORG}")
fi

if [[ "${RESUME}" == "1" ]]; then
    TRAIN_CMD+=(--load_checkpoint)
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    # wandb reads WANDB_API_KEY directly; never persist the credential in
    # process arguments or train_config.json.
    TRAIN_CMD+=(--use_wandb enabled)
else
    TRAIN_CMD+=(--use_tensorboard "${RUN_DIR}/tensorboard")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Reward server command:"
    printf '  %q' "${REWARD_CMD[@]}"
    printf '\nTraining command:\n'
    printf '  %q' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ "${RESUME}" != "1" && -e "${CKPT_PATH}/train_config.json" ]]; then
    echo "Run state already exists at ${RUN_DIR}. Set RESUME=1 or choose another RUN_NAME." >&2
    exit 1
fi

if [[ "${ALLOW_BUSY_GPUS}" != "1" ]]; then
    for gpu_id in 0 1 2 3; do
        busy_processes=$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
        if [[ -n "${busy_processes}" ]]; then
            echo "GPU ${gpu_id} is busy:" >&2
            echo "${busy_processes}" >&2
            echo "Wait for GPUs 0-3 to become idle, or explicitly set ALLOW_BUSY_GPUS=1." >&2
            exit 1
        fi
    done
fi

mkdir -p "${LOG_DIR}" "${CKPT_PATH}" "${SAVE_PATH}" "${RAY_TMPDIR}" "${TMPDIR}"

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected four visible CUDA GPUs, found {torch.cuda.device_count()}")
print("Visible GPUs:", [torch.cuda.get_device_name(index) for index in range(4)])
PY

RM_PID=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${RM_PID}" ]] && kill -0 "${RM_PID}" 2>/dev/null; then
        kill "${RM_PID}"
        wait "${RM_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "Starting MathVerify reward server at ${RM_URL}"
"${REWARD_CMD[@]}" >"${LOG_DIR}/reward_server.log" 2>&1 &
RM_PID=$!

reward_ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "${RM_PID}" 2>/dev/null; then
        echo "Reward server exited during startup. See ${LOG_DIR}/reward_server.log." >&2
        exit 1
    fi
    if python -c "import socket; socket.create_connection(('${RM_HOST}', ${RM_PORT}), timeout=1).close()" 2>/dev/null; then
        reward_ready=1
        break
    fi
    sleep 1
done
if [[ "${reward_ready}" != "1" ]]; then
    echo "Timed out waiting for the reward server. See ${LOG_DIR}/reward_server.log." >&2
    exit 1
fi

echo "Launching ${RUN_NAME} on physical GPUs ${GPU_IDS}"
echo "Outputs: ${RUN_DIR}"
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/training.log"
