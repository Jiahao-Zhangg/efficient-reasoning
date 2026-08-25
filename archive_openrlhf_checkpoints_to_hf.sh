#!/usr/bin/env bash

# Archive completed OpenRLHF/DeepSpeed actor checkpoints to Hugging Face.
# A local checkpoint is deleted only after every uploaded file is verified by
# remote path and byte size.
set -uo pipefail

usage() {
    echo "Usage: $0 CHECKPOINT_ROOT HF_REPO_PREFIX TRAINING_PID" >&2
    exit 2
}

[[ $# -eq 3 ]] || usage

CHECKPOINT_ROOT=$(realpath -m -- "$1")
ACTOR_DIR=${CHECKPOINT_ROOT}/_actor
HF_REPO_PREFIX=$2
TRAINING_PID=$3
POLL_SECONDS=${HF_ARCHIVE_POLL_SECONDS:-30}
UPLOAD_WORKERS=${HF_ARCHIVE_UPLOAD_WORKERS:-4}
EXPECTED_WORLD_SIZE=${HF_ARCHIVE_EXPECTED_WORLD_SIZE:-2}
UPLOAD_LOCK=${HF_ARCHIVE_UPLOAD_LOCK:-/work2/jiahaoz4/efficient-reasoning/outputs/.hf_checkpoint_upload.lock}
PYTHON_BIN=${PYTHON_BIN:-/work2/jiahaoz4/miniconda3/envs/efficient_reasoning/bin/python}

[[ "${HF_REPO_PREFIX}" =~ ^[^/]+/[^/]+$ ]] || {
    echo "HF_REPO_PREFIX must have the form owner/name" >&2
    exit 2
}
[[ "${TRAINING_PID}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAINING_PID must be a positive integer" >&2
    exit 2
}
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_ARCHIVE_POLL_SECONDS must be a positive integer" >&2
    exit 2
}
[[ "${UPLOAD_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_ARCHIVE_UPLOAD_WORKERS must be a positive integer" >&2
    exit 2
}
[[ "${EXPECTED_WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_ARCHIVE_EXPECTED_WORLD_SIZE must be a positive integer" >&2
    exit 2
}

command -v hf >/dev/null 2>&1 || {
    echo "The Hugging Face 'hf' CLI is required" >&2
    exit 1
}
command -v flock >/dev/null 2>&1 || {
    echo "flock is required" >&2
    exit 1
}
[[ -x "${PYTHON_BIN}" ]] || {
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
}
"${PYTHON_BIN}" -c 'import huggingface_hub' >/dev/null 2>&1 || {
    echo "huggingface_hub is unavailable in ${PYTHON_BIN}" >&2
    exit 1
}

mkdir -p "${ACTOR_DIR}" "$(dirname -- "${UPLOAD_LOCK}")"

log() {
    printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*"
}

training_is_running() {
    local state
    state=$(ps -o stat= -p "${TRAINING_PID}" 2>/dev/null) || return 1
    [[ "${state//[[:space:]]/}" != Z* ]]
}

list_checkpoints() {
    local path name
    while IFS= read -r path; do
        name=${path##*/}
        [[ "${name}" =~ ^global_step[0-9]+$ ]] && printf '%s\n' "${path}"
    done < <(find "${ACTOR_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step*' -print)
}

checkpoint_is_complete() {
    local checkpoint=$1
    local checkpoint_name checkpoint_step latest_tag latest_step
    local model_count optimizer_count empty_count

    checkpoint_name=${checkpoint##*/}
    [[ "${checkpoint_name}" =~ ^global_step[0-9]+$ ]] || return 1
    checkpoint_step=${checkpoint_name#global_step}
    [[ -s "${ACTOR_DIR}/latest" ]] || return 1
    latest_tag=$(tr -d '[:space:]' <"${ACTOR_DIR}/latest")
    [[ "${latest_tag}" =~ ^global_step[0-9]+$ ]] || return 1
    latest_step=${latest_tag#global_step}
    (( checkpoint_step <= latest_step )) || return 1

    model_count=$(find "${checkpoint}" -maxdepth 1 -type f -size +0c -name '*_model_states.pt' | wc -l)
    optimizer_count=$(find "${checkpoint}" -maxdepth 1 -type f -size +0c -name '*_optim_states.pt' | wc -l)
    empty_count=$(find "${checkpoint}" -maxdepth 1 -type f -size 0c | wc -l)
    [[ "${model_count}" -ge 1 && "${optimizer_count}" -eq "${EXPECTED_WORLD_SIZE}" && "${empty_count}" -eq 0 ]]
}

make_public_repo() {
    local repo_id=$1
    "${PYTHON_BIN}" - "${repo_id}" <<'PY'
import sys

from huggingface_hub import HfApi

repo_id = sys.argv[1]
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
info = api.repo_info(repo_id=repo_id, repo_type="model")
if info.private:
    api.update_repo_settings(repo_id=repo_id, private=False)
PY
}

upload_train_config() {
    local repo_id=$1
    [[ -s "${CHECKPOINT_ROOT}/train_config.json" ]] || return 0
    "${PYTHON_BIN}" - "${CHECKPOINT_ROOT}/train_config.json" "${repo_id}" <<'PY'
import sys

from huggingface_hub import HfApi

config_path, repo_id = sys.argv[1:]
HfApi().upload_file(
    path_or_fileobj=config_path,
    path_in_repo="train_config.json",
    repo_id=repo_id,
    repo_type="model",
)
PY
}

verify_upload() {
    local checkpoint=$1
    local repo_id=$2
    "${PYTHON_BIN}" - "${checkpoint}" "${repo_id}" "${CHECKPOINT_ROOT}/train_config.json" <<'PY'
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

checkpoint = Path(sys.argv[1]).resolve()
repo_id = sys.argv[2]
config_path = Path(sys.argv[3])
prefix = checkpoint.name + "/"
local_files = {
    prefix + path.relative_to(checkpoint).as_posix(): path.stat().st_size
    for path in checkpoint.rglob("*")
    if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
}
if config_path.is_file():
    local_files["train_config.json"] = config_path.stat().st_size
if not local_files:
    raise SystemExit("checkpoint has no files to verify")

api = HfApi()
last_error = "repository metadata unavailable"
for attempt in range(6):
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
        if info.private:
            raise RuntimeError("repository unexpectedly remained private")
        remote_files = {item.rfilename: item.size for item in info.siblings}
        missing = sorted(set(local_files) - set(remote_files))
        wrong_size = sorted(
            path for path, size in local_files.items() if remote_files.get(path) != size
        )
        if not missing and not wrong_size:
            print(f"Verified {len(local_files)} files ({sum(local_files.values())} bytes)")
            raise SystemExit(0)
        last_error = f"missing={missing[:3]}, wrong_size={wrong_size[:3]}"
    except Exception as error:
        last_error = str(error)
    if attempt < 5:
        time.sleep(10)
raise SystemExit(f"remote verification failed: {last_error}")
PY
}

delete_verified_checkpoint() {
    local checkpoint=$1
    local checkpoint_name resolved expected

    checkpoint_name=${checkpoint##*/}
    [[ "${checkpoint_name}" =~ ^global_step[0-9]+$ ]] || return 1
    resolved=$(realpath -- "${checkpoint}") || return 1
    expected=${ACTOR_DIR}/${checkpoint_name}
    [[ "${resolved}" == "${expected}" && -d "${resolved}" ]] || return 1

    find "${resolved}" -depth -delete
    [[ ! -e "${resolved}" ]]
}

clear_upload_metadata() {
    local metadata_dir=${ACTOR_DIR}/.cache/huggingface
    if [[ -d "${metadata_dir}" ]]; then
        find "${metadata_dir}" -depth -delete
        rmdir "${ACTOR_DIR}/.cache" 2>/dev/null || true
    fi
}

archive_checkpoint() {
    local checkpoint=$1
    local checkpoint_name step repo_id lock_fd

    checkpoint_name=${checkpoint##*/}
    step=${checkpoint_name#global_step}
    repo_id=${HF_REPO_PREFIX}-step_${step}

    checkpoint_is_complete "${checkpoint}" || {
        log "Deferring incomplete checkpoint ${checkpoint_name}"
        return 1
    }

    exec {lock_fd}>"${UPLOAD_LOCK}"
    log "Waiting for the Hugging Face upload lock for ${checkpoint_name}"
    flock "${lock_fd}"

    if [[ ! -d "${checkpoint}" ]]; then
        log "${checkpoint_name} was already removed"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 0
    fi
    if ! checkpoint_is_complete "${checkpoint}"; then
        log "${checkpoint_name} changed or is incomplete; retaining it"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi

    log "Uploading ${checkpoint_name} to https://huggingface.co/${repo_id}"
    if ! make_public_repo "${repo_id}"; then
        log "Repository creation failed; retaining ${checkpoint_name}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! env -u PYTHONNOUSERSITE HF_XET_HIGH_PERFORMANCE=1 hf upload-large-folder \
        "${repo_id}" "${ACTOR_DIR}" \
        --repo-type model \
        --include "${checkpoint_name}/**" \
        --exclude "${checkpoint_name}/.cache/**" \
        --num-workers "${UPLOAD_WORKERS}"; then
        log "Checkpoint upload failed; retaining ${checkpoint_name} for retry"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! upload_train_config "${repo_id}"; then
        log "train_config.json upload failed; retaining ${checkpoint_name} for retry"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! verify_upload "${checkpoint}" "${repo_id}"; then
        log "Remote verification failed; retaining ${checkpoint_name} for retry"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! delete_verified_checkpoint "${checkpoint}"; then
        log "Safety checks prevented deletion of ${checkpoint_name}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    clear_upload_metadata
    log "Verified, archived, and deleted ${checkpoint_name}"

    flock -u "${lock_fd}"
    exec {lock_fd}>&-
}

log "Watching ${ACTOR_DIR} for checkpoints from training PID ${TRAINING_PID}"
log "Hugging Face repository prefix: ${HF_REPO_PREFIX}"

while true; do
    mapfile -t checkpoints < <(list_checkpoints | sort -V)
    for checkpoint in "${checkpoints[@]}"; do
        if checkpoint_is_complete "${checkpoint}"; then
            archive_checkpoint "${checkpoint}" || true
        fi
    done

    if ! training_is_running; then
        mapfile -t remaining < <(list_checkpoints | sort -V)
        if (( ${#remaining[@]} == 0 )); then
            log "Training stopped and all completed checkpoints are archived"
            exit 0
        fi

        incomplete=0
        for checkpoint in "${remaining[@]}"; do
            if checkpoint_is_complete "${checkpoint}"; then
                archive_checkpoint "${checkpoint}" || true
            else
                incomplete=1
                log "Retaining incomplete checkpoint ${checkpoint##*/}"
            fi
        done
        mapfile -t remaining < <(list_checkpoints | sort -V)
        if (( ${#remaining[@]} == 0 )); then
            log "Training stopped and all completed checkpoints are archived"
            exit 0
        fi
        if (( incomplete == 1 )); then
            exit 1
        fi
    fi

    sleep "${POLL_SECONDS}"
done
