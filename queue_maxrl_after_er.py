"""Launch the requested MaxRL script once ER finishes and GPUs 0-3 are free.

This queue never stops jobs or changes either repository's training parameters.
The existing ER supervisor owns recovery/cleanup; the MaxRL launcher owns its
checkpoint uploader. A STOP file beside this queue's state cancels waiting only.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from watch_rloo_training import HF_PREFIX, checkpoint_manifest, digest, gpu_blockers, processes, same_process

ROOT = Path(__file__).resolve().parent
MAXRL_ROOT = Path("/work2/jiahaoz4/maxrl")
LAUNCHER = MAXRL_ROOT / "qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_offset_marginrl.sh"
PYTHON = Path("/work2/jiahaoz4/miniconda3/envs/maxrl/bin/python")
PREDECESSOR = ROOT / "outputs/r1_compression_watchdog/state.json"
DEFAULT_STATE = ROOT / "outputs/queue_offset_after_er/state.json"
OUTPUT_ROOT = Path("/home/jiahaoz4/maxrl_training_outputs/offset256_after_er")
EXPERIMENT = "fixed_n_rb_offset_cost_aware_marginrl_Qwen3-1.7B-Base_math12k_offset256_token_mean"
PROJECT = "Qwen3_MaxRL_Experiments"
ENTITY = "jiahaozhangg-carnegie-mellon-university"
TARGET_HF_PREFIX = "zjhhhh/" + EXPERIMENT.replace("_", "-").lower()
FROZEN_FILES = (
    LAUNCHER,
    MAXRL_ROOT / "qwen3_experiments/run_qwen3_1_7b_math12k.sh",
    MAXRL_ROOT / "qwen3_experiments/run_with_checkpoint_upload.sh",
    MAXRL_ROOT / "qwen3_experiments/archive_checkpoints_to_hf.sh",
    MAXRL_ROOT / "verl/trainer/config/ppo_trainer.yaml",
    MAXRL_ROOT / "verl/trainer/ppo/core_algos.py",
    MAXRL_ROOT / "verl/trainer/ppo/ray_trainer.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def initialize(path):
    if path.exists():
        raise RuntimeError("Queue state already exists; refusing to overwrite it")
    predecessor = json.loads(PREDECESSOR.read_text())
    run_id = uuid.uuid4().hex[:8]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        path,
        {
            "created_utc": utc_now(),
            "phase": "waiting",
            "predecessor": str(PREDECESSOR),
            "predecessor_created_utc": predecessor["created_utc"],
            "launcher": str(LAUNCHER),
            "output_root": str(OUTPUT_ROOT),
            "ray_dir": f"/home/jiahaoz4/mro_{run_id}",
            "run_id": run_id,
            "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
            "hf_prefix": TARGET_HF_PREFIX,
            "code_hashes": {str(file): digest(file) for file in FROZEN_FILES},
        },
    )


def launch_environment(state, inherited=None):
    environment = dict(os.environ if inherited is None else inherited)
    # Retain credentials, but discard previous experiments' operational/training overrides.
    for key in list(environment):
        if key.startswith(("MAXRL_", "WANDB_", "VLLM_", "RAY_")) and key != "WANDB_API_KEY":
            environment.pop(key)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "ip_head",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "LD_LIBRARY_PATH",
    ):
        environment.pop(key, None)
    output = Path(state["output_root"])
    environment.update(
        {
            "PATH": f"{PYTHON.parent}:/usr/local/cuda-12.4/bin:/usr/local/bin:/usr/bin:/bin",
            "CONDA_PREFIX": str(PYTHON.parent.parent),
            "CONDA_DEFAULT_ENV": "maxrl",
            "PYTHONNOUSERSITE": "1",
            "CUDA_HOME": "/usr/local/cuda-12.4",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "MAXRL_SKIP_ENV_SETUP": "1",
            "MAXRL_OUTPUT_DIR": str(output),
            "MAXRL_RAY_DIR": state["ray_dir"],
            "RAY_ADDRESS": "local",
            "RAY_TMPDIR": state["ray_dir"],
            "TMPDIR": str(output / "tmp"),
            "HF_HOME": "/work2/jiahaoz4/.cache/huggingface",
            "WANDB_RUN_ID": state["run_id"],
            "WANDB_RESUME": "never",
            "WANDB_MODE": "online",
            "WANDB_ENTITY": ENTITY,
            "WANDB_DIR": str(output / "wandb"),
        }
    )
    return environment


class Queue:
    def __init__(self, path, api=None):
        self.path = path
        self.state = json.loads(path.read_text())
        self.api = api if api is not None else HfApi()
        self.idle_confirmations = 0

    def record(self, status, message, **values):
        changed = (self.state.get("status"), self.state.get("message")) != (status, message)
        self.state.update(status=status, message=message, last_check_utc=utc_now(), **values)
        atomic_json(self.path, self.state)
        if changed:
            entry = {"time": utc_now(), "status": status, "message": message}
            with self.path.with_name("events.jsonl").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)

    def preflight(self):
        for name, expected in self.state["code_hashes"].items():
            if digest(name) != expected:
                raise RuntimeError(f"Queued source changed; review before launch: {name}")
        if not PYTHON.is_file():
            raise RuntimeError("The existing MaxRL Python environment is missing")
        output = Path(self.state["output_root"])
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Refusing to overwrite an existing output directory: {output}")
        if Path(self.state["ray_dir"]).exists():
            raise RuntimeError("The queued Ray directory already exists; refusing to reuse another Ray session")
        for name in ("math12k/train.parquet", "aime25/test.parquet", "math500/test.parquet"):
            path = MAXRL_ROOT / "data" / name
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Prepared dataset is missing: {path}")
        if self.api.whoami()["name"] != "zjhhhh":
            raise RuntimeError("Expected the configured zjhhhh Hugging Face account")
        for step in (50, 100, 150):
            repo = f"{self.state['hf_prefix']}-step_{step}"
            try:
                self.api.model_info(repo)
            except RepositoryNotFoundError:
                continue
            raise RuntimeError(f"Refusing to reuse an existing checkpoint repository: {repo}")

    def readiness(self):
        if self.path.with_name("STOP").exists():
            return False, "stopped", "Queue cancelled; current training is untouched"
        predecessor = json.loads(Path(self.state["predecessor"]).read_text())
        if predecessor.get("created_utc") != self.state["predecessor_created_utc"]:
            raise RuntimeError("The predecessor supervisor was replaced; manual review is required")
        if predecessor.get("status") != "complete":
            return False, "waiting_for_er", "Waiting for ER step 100, verified HF archival, and supervisor completion"
        snapshot = processes()
        for role in ("trainer", "launcher", "archiver"):
            identity = predecessor.get(role) or {}
            if same_process(identity.get("pid"), identity.get("start"), snapshot):
                return False, "waiting_for_er_exit", f"Waiting for the completed ER {role} to exit"
        run_dirs = [Path(value) for value in predecessor["run_dirs"]]
        if any((path / "checkpoints/_actor/global_step100").exists() for path in run_dirs):
            return False, "waiting_for_archive", "Waiting for the final ER checkpoint's verified local cleanup"
        markers = []
        for path in run_dirs:
            log = path / "logs/checkpoint_archiver.log"
            if log.is_file():
                markers.append("Verified, archived, and deleted global_step100" in log.read_text(errors="replace"))
        if not any(markers):
            return False, "waiting_for_archive", "Waiting for the final ER upload-and-verification receipt"
        info = self.api.model_info(f"{HF_PREFIX}-step_100", files_metadata=True)
        manifest = checkpoint_manifest(info, 100)
        if manifest is None:
            return False, "waiting_for_archive", "The final HF checkpoint does not yet have all verified shards"
        blocked = gpu_blockers()
        if blocked:
            return False, "waiting_for_gpus", f"Waiting for GPU 0-3; busy devices: {blocked}"
        # Outputs and Ray spill/log files go on /home, away from the nearly full /work2 disk.
        if (
            shutil.disk_usage(OUTPUT_ROOT.parent if OUTPUT_ROOT.parent.exists() else OUTPUT_ROOT.parent.parent).free
            < 50 * 2**30
        ):
            return False, "waiting_for_disk", "Waiting for at least 50 GiB free on the output filesystem"
        if shutil.disk_usage(MAXRL_ROOT).free < 8 * 2**30:
            return False, "waiting_for_disk", "Waiting for at least 8 GiB free for the existing model/data cache"
        return True, "ready", "ER is complete and archived; GPU 0-3 and output disk are available"

    def tick(self):
        if self.state["phase"] != "waiting":
            return False
        ready, status, message = self.readiness()
        if not ready:
            self.idle_confirmations = 0
            self.record(status, message)
            if status == "stopped":
                self.state["phase"] = "cancelled"
                self.record(status, message)
                return False
            return True
        self.preflight()
        self.idle_confirmations += 1
        if self.idle_confirmations < 2:
            self.record("confirming_idle", "First idle check passed; confirming again in 30 seconds")
            return True
        # Recheck after network preflight: never launch if the predecessor/GPU state changed.
        if not self.readiness()[0]:
            self.idle_confirmations = 0
            return True
        self.launch()
        return False

    def launch(self):
        # Persist the one-shot claim BEFORE starting the process. An ambiguous interrupted
        # launch requires review, not an automatic duplicate run.
        self.state["phase"] = "launching"
        self.record("launching", "Launching the requested MaxRL script exactly once")
        output = Path(self.state["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        for directory in (output / "tmp", output / "wandb"):
            directory.mkdir(exist_ok=True)
        log_path = self.path.with_name("maxrl_launch.log")
        with log_path.open("x") as log:
            child = subprocess.Popen(
                ["bash", str(LAUNCHER)],
                cwd=MAXRL_ROOT,
                env=launch_environment(self.state),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.state["phase"] = "launched"
        self.record(
            "launched",
            f"MaxRL started; W&B: {self.state['wandb_url']}",
            launcher_pid=child.pid,
            launched_utc=utc_now(),
            launch_log=str(log_path),
        )
        while child.poll() is None:
            time.sleep(30)
        self.state["phase"] = "finished"
        status = "complete" if child.returncode == 0 else "failed"
        self.record(
            status, f"MaxRL launcher/uploader exited with status {child.returncode}", exit_code=child.returncode
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--check", action="store_true", help="Read-only checks; never launches training")
    args = parser.parse_args()
    if args.initialize:
        initialize(args.state)
    queue = Queue(args.state)
    if args.check:
        queue.preflight()
        ready, status, message = queue.readiness()
        print(json.dumps({"ready": ready, "status": status, "message": message, "launcher": str(LAUNCHER)}))
        return
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if not queue.tick():
                    break
            except Exception as error:
                queue.idle_confirmations = 0
                queue.record("attention", f"{type(error).__name__}: {str(error)[:400]}")
                if queue.state["phase"] != "waiting":
                    break
            time.sleep(30)


if __name__ == "__main__":
    main()
