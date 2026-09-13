"""Supervise this four-A100 compression experiment without changing its parameters.

Only an exited trainer triggers recovery. W&B state and slow generation never
trigger a kill. Recovery waits for idle GPUs, restores a verified HF checkpoint,
and gives the restarted trainer a fresh W&B ID. A STOP file disables recovery.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent
PYTHON = Path("/work2/jiahaoz4/miniconda3/envs/efficient_reasoning/bin/python")
LAUNCHER = ROOT / "run_rloo_deepseek_1.5B_compression.sh"
ARCHIVER = ROOT / "archive_openrlhf_checkpoints_to_hf.sh"
DEFAULT_NAME = "rloo_r1_distill_1.5b_compression_n16_b512_lr1e-6_kl0_alpha0.1_seed42_extracted"
HF_PREFIX = "zjhhhh/er-r1-distill-1.5b-compression-n16-extracted"
ENTITY = "jiahaozhangg-carnegie-mellon-university"
PROJECT = "efficient_reasoning_compression"
READY = "Checkpoint weights synchronized to all vLLM engines before generation."
STATE_FILES = (
    "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
    "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt",
    "mp_rank_00_model_states.pt",
)
OPERATIONAL_KEYS = {"ckpt_path", "save_path", "load_checkpoint", "wandb_run_name", "wandb_group", "use_tensorboard"}
FROZEN_FILES = (
    "run_rloo_deepseek_1.5B_compression.sh",
    "openrlhf/trainer/ray/ppo_actor.py",
    "reward_server/math_server.py",
    "utils/utils.py",
)


class Defer(Exception):
    """A safe retry needs external state to change, not changes to training."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def processes():
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] in ("Z", "X"):
                continue
            result[int(entry.name)] = {"start": fields[19], "ppid": int(fields[1]), "sid": int(fields[3])}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def same_process(pid, identity, snapshot):
    return pid is not None and identity is not None and snapshot.get(int(pid), {}).get("start") == identity


def descendants(snapshot, roots):
    owned = set(roots) & snapshot.keys()
    while True:
        added = {pid for pid, info in snapshot.items() if info["ppid"] in owned} - owned
        if not added:
            return owned
        owned.update(added)


def gpu_blockers():
    blocked = []
    for gpu in range(4):
        applications = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=15
        ).strip()
        memory = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        ).strip()
        total, free = [int(value.strip()) for value in memory.split(",")]
        if applications or total < 79000 or free < total - 1024:
            blocked.append(gpu)
    return blocked


def checkpoint_manifest(info, step):
    siblings = {item.rfilename: item for item in info.siblings}
    if "train_config.json" not in siblings:
        return None
    manifest = {}
    for name in STATE_FILES:
        key = f"global_step{step}/{name}"
        item = siblings.get(key)
        sha = getattr(getattr(item, "lfs", None), "sha256", None)
        if item is None or not item.size or not re.fullmatch("[0-9a-f]{64}", sha or ""):
            return None
        manifest[key] = {"size": item.size, "sha256": sha}
    return {"step": step, "repo": info.id, "revision": info.sha, "files": manifest}


def latest_archive(api):
    owner, prefix = HF_PREFIX.split("/")
    candidates = []
    for model in api.list_models(author=owner, search=prefix + "-step_"):
        match = re.fullmatch(re.escape(HF_PREFIX) + r"-step_(\d+)", model.id)
        if match and 0 < int(match[1]) <= 100:
            candidates.append((int(match[1]), model.id))
    for step, repo in sorted(candidates, reverse=True):
        info = api.model_info(repo, files_metadata=True)
        manifest = checkpoint_manifest(info, step)
        if manifest is not None:
            return manifest
    raise Defer("No complete, hash-verifiable HF checkpoint is available")


def latest_local(run_dirs):
    newest = 0
    for run_dir in run_dirs:
        actor = Path(run_dir) / "checkpoints/_actor"
        latest = actor / "latest"
        if not latest.is_file():
            continue
        match = re.fullmatch(r"global_step(\d+)", latest.read_text().strip())
        if not match:
            continue
        for checkpoint in actor.glob("global_step*"):
            tag = re.fullmatch(r"global_step(\d+)", checkpoint.name)
            if tag and int(tag[1]) <= int(match[1]):
                if all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size for name in STATE_FILES):
                    newest = max(newest, int(tag[1]))
    return newest


def verify_files(actor, manifest):
    for relative, expected in manifest["files"].items():
        path = actor / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
            raise Defer(f"Checkpoint file missing or wrong size: {relative}")
        if digest(path) != expected["sha256"]:
            raise Defer(f"Checkpoint checksum mismatch: {relative}")


def launch_environment(run_name, run_id):
    env = os.environ.copy()
    for key in (
        "WANDB_FORK_FROM",
        "WANDB_RESUME_FROM",
        "WANDB_SWEEP_ID",
        "WANDB_SERVICE",
        "_WANDB_SERVICE",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        "WANDB_DIR",
        "WANDB_DISABLED",
        "TRANSFORMERS_CACHE",
    ):
        env.pop(key, None)
    env.update(
        {
            "PATH": str(PYTHON.parent)
            + os.pathsep
            + "/work2/jiahaoz4/miniconda3/bin"
            + os.pathsep
            + env.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "GPU_IDS": "0,1,2,3",
            "CONDA_ENV": "efficient_reasoning",
            "CONDA_EXE": "/work2/jiahaoz4/miniconda3/bin/conda",
            "RUN_NAME": run_name,
            "OUTPUT_ROOT": str(ROOT / "outputs"),
            "RESUME": "1",
            "DRY_RUN": "0",
            "USE_WANDB": "1",
            "ARCHIVE_CHECKPOINTS": "0",
            "HF_REPO_PREFIX": HF_PREFIX,
            "HF_HOME": "/work2/jiahaoz4/.cache/huggingface",
            "WANDB_ORG": ENTITY,
            "WANDB_PROJECT": PROJECT,
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "never",
            "WANDB_MODE": "online",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "RAY_TMPDIR": f"/tmp/er_auto_{run_id}",
            "TMPDIR": f"/tmp/er_auto_{run_id}",
            "RM_PORT": "24373",
            "VERIFIER_WORKERS": "16",
            "CUDA_HOME": "/usr/local/cuda-12.4",
            "PYTHON_BIN": str(PYTHON),
            "HF_ARCHIVE_EXPECTED_WORLD_SIZE": "2",
            "HF_ARCHIVE_UPLOAD_LOCK": str(ROOT / "outputs/.hf_checkpoint_upload.lock"),
        }
    )
    return env


class Supervisor:
    def __init__(self, state_path):
        self.path = state_path
        self.state = json.loads(state_path.read_text())
        self.api = HfApi()

    def save(self):
        self.state["last_check_utc"] = utc_now()
        atomic_json(self.path, self.state)

    def event(self, status, message):
        changed = (self.state.get("status"), self.state.get("message")) != (status, message)
        self.state.update(status=status, message=message)
        self.save()
        if changed:
            entry = {"time": utc_now(), "status": status, "message": message}
            with self.path.with_name("events.jsonl").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)

    def remember_owned(self, snapshot):
        roots = [pid for pid, identity in self.state.get("owned", {}).items() if same_process(pid, identity, snapshot)]
        for key in ("trainer", "launcher"):
            record = self.state.get(key, {})
            if same_process(record.get("pid"), record.get("start"), snapshot):
                roots.append(record["pid"])
        for pid in descendants(snapshot, [int(pid) for pid in roots]):
            self.state["owned"][str(pid)] = snapshot[pid]["start"]

    def cleanup_failed_processes(self):
        # Individual, fingerprint-checked PIDs only: never ray stop or a GPU-wide kill.
        self.remember_owned(processes())
        owned = self.state["owned"].copy()
        for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 2)):
            snapshot = processes()
            targets = [int(pid) for pid, start in owned.items() if same_process(pid, start, snapshot)]
            for pid in targets:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if targets:
                time.sleep(grace)

    def ensure_archiver(self):
        archive = self.state.get("archiver") or {}
        if same_process(archive.get("pid"), archive.get("start"), processes()):
            return
        if not self.state.get("trainer", {}).get("pid"):
            return
        run_dir = Path(self.state["run_dir"])
        log = run_dir / "logs/checkpoint_archiver.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as output:
            child = subprocess.Popen(
                ["bash", str(ARCHIVER), str(run_dir / "checkpoints"), HF_PREFIX, str(self.state["trainer"]["pid"])],
                cwd=ROOT,
                env=launch_environment(run_dir.name, "archive"),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.state["archiver"] = {"pid": child.pid, "start": processes()[child.pid]["start"]}
        self.save()

    def remove_restored_copy(self):
        restored = self.state.get("restored")
        if not restored:
            return
        run_dir = Path(self.state["run_dir"]).resolve()
        actor = run_dir / "checkpoints/_actor"
        target = actor / f"global_step{restored['step']}"
        if target.exists():
            if (
                target.is_symlink()
                or actor.resolve() != actor
                or target.resolve().parent != actor.resolve()
                or run_dir.parent != ROOT / "outputs"
            ):
                raise Defer("Refusing to remove a checkpoint outside this supervisor attempt")
            if {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()} != set(
                STATE_FILES
            ):
                raise Defer("Unexpected files in the restored checkpoint; retaining it for review")
            info = self.api.model_info(restored["repo"], revision=restored["revision"], files_metadata=True)
            if checkpoint_manifest(info, restored["step"]) != restored:
                raise Defer("Archived checkpoint manifest changed; keeping the local copy")
            verify_files(actor, restored)
            shutil.rmtree(target)
            self.event(
                "checkpoint_cleanup", f"Released restored step {restored['step']} locally; verified HF copy retained"
            )
        self.state["restored"] = None
        self.save()

    def prepare(self, manifest, run_dir):
        for name, expected in self.state["code_hashes"].items():
            if digest(ROOT / name) != expected:
                raise Defer(f"Recovery paused: {name} changed after supervision was enabled")
        needed = sum(item["size"] for item in manifest["files"].values())
        actor = run_dir / "checkpoints/_actor"
        # Partial HF downloads are resumable and already consume disk space.
        allocated = sum(path.stat().st_size for path in actor.rglob("*") if path.is_file() and not path.is_symlink())
        if shutil.disk_usage(ROOT).free < max(0, needed - allocated) + 8 * 1024**3:
            raise Defer("Insufficient disk: need checkpoint download plus 8 GiB free headroom")
        actor.mkdir(parents=True, exist_ok=True)
        config_path = hf_hub_download(
            manifest["repo"], "train_config.json", revision=manifest["revision"], local_dir=actor
        )
        archived_config = json.loads(Path(config_path).read_text())
        differences = [
            key
            for key, value in self.state["training_config"].items()
            if key not in OPERATIONAL_KEYS and archived_config.get(key) != value
        ]
        if differences:
            raise Defer("Archived training configuration mismatch: " + ", ".join(differences))
        for relative in manifest["files"]:
            path = actor / relative
            bad_existing_file = path.exists() and (
                path.stat().st_size != manifest["files"][relative]["size"]
                or digest(path) != manifest["files"][relative]["sha256"]
            )
            hf_hub_download(
                manifest["repo"],
                relative,
                revision=manifest["revision"],
                local_dir=actor,
                force_download=bad_existing_file,
            )
        verify_files(actor, manifest)
        (actor / "latest").write_text(f"global_step{manifest['step']}\n")

    def restart(self, manifest):
        if self.path.with_name("STOP").exists():
            raise Defer("STOP requested; no restart will be launched")
        blocked = gpu_blockers()
        if blocked:
            raise Defer(f"Waiting for GPUs {blocked}; other processes will not be interrupted")
        pending = self.state.get("pending")
        if pending is None:
            run_id = uuid.uuid4().hex[:8]
            name = f"{DEFAULT_NAME}_resume{manifest['step']}_{run_id}"
            pending = {"run_id": run_id, "name": name, "manifest": manifest}
            self.state["pending"] = pending
            self.save()
        if pending["manifest"] != manifest:
            raise Defer("A newer archive appeared during restore; review the pending download before switching")
        run_id, name = pending["run_id"], pending["name"]
        run_dir = ROOT / "outputs" / name
        self.event("restoring", f"Restoring latest checkpoint step {manifest['step']} from {manifest['repo']}")
        self.prepare(manifest, run_dir)
        if self.path.with_name("STOP").exists() or gpu_blockers():
            raise Defer("Launch deferred after download: STOP requested or GPUs no longer idle")
        log = run_dir / "launch.log"
        with log.open("ab") as output:
            child = subprocess.Popen(
                ["bash", str(LAUNCHER)],
                cwd=ROOT,
                env=launch_environment(name, run_id),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.state.update(
            launcher={"pid": child.pid, "start": processes()[child.pid]["start"]},
            trainer={},
            owned={},
            archiver=None,
            launch_log=str(log),
            run_id=run_id,
            launched_at=time.time(),
            phase="starting",
            last_checkpoint=manifest["step"],
            attempts=self.state.get("attempts", 0) + 1,
            failure_seen_at=None,
            run_dir=str(run_dir),
            restored=manifest,
            pending=None,
        )
        self.state["run_dirs"].append(str(run_dir))
        self.state["owned"][str(child.pid)] = self.state["launcher"]["start"]
        self.event(
            "starting",
            f"Restart from step {manifest['step']}; new W&B: https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
        )

    def tick(self):
        if self.path.with_name("STOP").exists():
            self.event("stopped", "Supervision disabled; existing training and checkpoint archiver left untouched")
            return False
        snapshot = processes()
        self.remember_owned(snapshot)
        trainer = self.state.get("trainer") or {}
        log_path = Path(self.state["launch_log"])
        log = ""
        if log_path.exists():
            with log_path.open("rb") as handle:
                log = handle.read(2 * 1024 * 1024).decode(errors="replace")
        if self.state["phase"] == "starting" and not trainer:
            match = re.search(r"Training PID: (\d+);", log)
            if match and int(match[1]) in snapshot:
                pid = int(match[1])
                self.state["trainer"] = trainer = {"pid": pid, "start": snapshot[pid]["start"]}
                self.remember_owned(snapshot)
            elif same_process(self.state["launcher"]["pid"], self.state["launcher"]["start"], snapshot):
                self.event("starting", "Launcher is preparing services; no duplicate launch will be started")
                return True
        if same_process(trainer.get("pid"), trainer.get("start"), snapshot):
            if self.state.get("restored"):
                if READY not in log:
                    self.event("loading", "Waiting for checkpoint load and vLLM synchronization before local cleanup")
                    return True
                self.remove_restored_copy()
            self.state["phase"] = "running"
            self.state["failure_seen_at"] = None
            self.ensure_archiver()
            if time.time() - log_path.stat().st_mtime > 3 * 3600:
                self.event(
                    "attention", "Trainer is alive but logs are stale >3 hours; not killing a possibly live job"
                )
            else:
                self.event("running", "Trainer alive; checkpoint archiving supervised; W&B state ignored for recovery")
            return True
        if self.state.get("failure_seen_at") is None:
            self.state["failure_seen_at"] = time.time()
            self.event("failure_detected", "Trainer exited; confirming before cleanup and checkpoint recovery")
            return True
        if time.time() - self.state["failure_seen_at"] < 60:
            return True
        self.cleanup_failed_processes()
        if self.state.get("restored"):
            self.remove_restored_copy()
        self.ensure_archiver()
        manifest = latest_archive(self.api)
        local_step = latest_local(self.state["run_dirs"])
        if local_step > manifest["step"]:
            raise Defer(f"Waiting for latest local step {local_step} to finish archiving before recovery")
        if manifest["step"] >= 100:
            self.event("complete", "Training step 100 checkpoint is verified on HF; no further restart is needed")
            return False
        if time.time() < self.state.get("retry_after", 0):
            return True
        self.state["retry_after"] = time.time() + 300
        self.restart(manifest)
        return True


def initialize(args):
    if args.state.exists():
        raise SystemExit("State already exists; refusing to overwrite it")
    snapshot = processes()
    if args.trainer_pid not in snapshot or args.launcher_pid not in snapshot:
        raise SystemExit("Initial trainer and launcher must both be alive")
    run_dir = ROOT / "outputs" / DEFAULT_NAME
    config = json.loads((run_dir / "checkpoints/train_config.json").read_text())
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "created_utc": utc_now(),
        "phase": "running",
        "attempts": 0,
        "last_checkpoint": 20,
        "run_dir": str(run_dir),
        "run_dirs": [str(run_dir)],
        "run_id": "26cb73oo",
        "launch_log": str(ROOT / "outputs/launch_logs/r1_compression_n16_extracted_resume20_sync_20260910.log"),
        "trainer": {"pid": args.trainer_pid, "start": snapshot[args.trainer_pid]["start"]},
        "launcher": {"pid": args.launcher_pid, "start": snapshot[args.launcher_pid]["start"]},
        "archiver": {"pid": args.archiver_pid, "start": snapshot[args.archiver_pid]["start"]},
        "owned": {},
        "failure_seen_at": None,
        "restored": None,
        "training_config": config,
        "code_hashes": {name: digest(ROOT / name) for name in FROZEN_FILES},
    }
    atomic_json(args.state, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "outputs/r1_compression_watchdog/state.json")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--trainer-pid", type=int, default=4190051)
    parser.add_argument("--launcher-pid", type=int, default=4189313)
    parser.add_argument("--archiver-pid", type=int, default=16584)
    parser.add_argument("--check", action="store_true", help="Read-only preflight; no process or checkpoint changes")
    args = parser.parse_args()
    if args.initialize:
        initialize(args)
    if args.check:
        runner = Supervisor(args.state)
        manifest = latest_archive(runner.api)
        print(
            json.dumps(
                {
                    "trainer_alive": same_process(
                        runner.state["trainer"]["pid"], runner.state["trainer"]["start"], processes()
                    ),
                    "latest_hf_checkpoint": manifest,
                    "free_disk_bytes": shutil.disk_usage(ROOT).free,
                },
                indent=2,
            )
        )
        return
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Supervisor(args.state)
        while True:
            try:
                if not runner.tick():
                    break
            except Defer as error:
                runner.event("waiting", str(error))
            except Exception as error:
                # Keep supervising after network or filesystem errors; never improvise a fix.
                runner.event("attention", f"Check failed ({type(error).__name__}): {str(error)[:300]}")
            time.sleep(60)


if __name__ == "__main__":
    main()
