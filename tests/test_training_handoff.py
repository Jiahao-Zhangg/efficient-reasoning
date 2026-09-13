"""CPU-only safety checks for the ER-to-MaxRL one-shot queue."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import queue_maxrl_after_er as handoff


@pytest.fixture
def queue(tmp_path, monkeypatch):
    predecessor = tmp_path / "predecessor.json"
    run_dir = tmp_path / "er"
    log = run_dir / "logs/checkpoint_archiver.log"
    log.parent.mkdir(parents=True)
    log.write_text("Verified, archived, and deleted global_step100\n")
    predecessor.write_text(
        json.dumps(
            {
                "created_utc": "original",
                "status": "complete",
                "run_dirs": [str(run_dir)],
                "trainer": {"pid": 11, "start": "11"},
                "launcher": {},
                "archiver": {},
            }
        )
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "phase": "waiting",
                "predecessor": str(predecessor),
                "predecessor_created_utc": "original",
                "output_root": str(tmp_path / "output"),
                "ray_dir": str(tmp_path / "ray"),
                "run_id": "newrun",
                "code_hashes": {},
                "hf_prefix": handoff.TARGET_HF_PREFIX,
                "wandb_url": "https://wandb.ai/test/newrun",
            }
        )
    )
    monkeypatch.setattr(handoff, "processes", lambda: {})
    monkeypatch.setattr(handoff, "gpu_blockers", lambda: [])
    monkeypatch.setattr(handoff, "checkpoint_manifest", lambda info, step: {"step": step})
    monkeypatch.setattr(handoff.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    api = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace())
    result = handoff.Queue(state, api=api)
    monkeypatch.setattr(result, "preflight", lambda: None)
    return result


def predecessor(queue, **updates):
    path = Path(queue.state["predecessor"])
    state = json.loads(path.read_text())
    state.update(updates)
    path.write_text(json.dumps(state))
    return state


@pytest.mark.parametrize("status", ["running", "failure_detected", "starting", "waiting", "attention", "stopped"])
def test_never_starts_while_er_is_running_or_recovering(queue, status):
    predecessor(queue, status=status)
    assert queue.readiness()[:2] == (False, "waiting_for_er")


def test_replaced_predecessor_is_not_accepted(queue):
    predecessor(queue, created_utc="different")
    with pytest.raises(RuntimeError, match="replaced"):
        queue.readiness()


@pytest.mark.parametrize("role", ["trainer", "launcher", "archiver"])
def test_waits_for_exact_predecessor_processes_to_exit(queue, monkeypatch, role):
    predecessor(queue, **{role: {"pid": 12, "start": "identity"}})
    monkeypatch.setattr(handoff, "processes", lambda: {12: {"start": "identity"}})
    assert queue.readiness()[1] == "waiting_for_er_exit"


def test_reused_process_id_does_not_block(queue, monkeypatch):
    monkeypatch.setattr(handoff, "processes", lambda: {11: {"start": "different"}})
    assert queue.readiness()[0]


def test_waits_for_final_checkpoint_cleanup(queue):
    run_dir = Path(predecessor(queue)["run_dirs"][0])
    (run_dir / "checkpoints/_actor/global_step100").mkdir(parents=True)
    assert queue.readiness()[1] == "waiting_for_archive"


def test_requires_successful_archival_receipt(queue):
    run_dir = Path(predecessor(queue)["run_dirs"][0])
    (run_dir / "logs/checkpoint_archiver.log").write_text("upload failed\n")
    assert queue.readiness()[1] == "waiting_for_archive"


def test_requires_complete_remote_checkpoint(queue, monkeypatch):
    monkeypatch.setattr(handoff, "checkpoint_manifest", lambda *args: None)
    assert queue.readiness()[1] == "waiting_for_archive"


def test_busy_gpu_and_disk_defers_without_stopping_anything(queue, monkeypatch):
    monkeypatch.setattr(handoff, "gpu_blockers", lambda: [2])
    assert queue.readiness()[1] == "waiting_for_gpus"
    monkeypatch.setattr(handoff, "gpu_blockers", lambda: [])
    monkeypatch.setattr(handoff.shutil, "disk_usage", lambda path: SimpleNamespace(free=3 * 2**30))
    assert queue.readiness()[1] == "waiting_for_disk"


def test_stop_cancels_waiting_without_launch(queue):
    queue.path.with_name("STOP").touch()
    assert queue.tick() is False
    assert queue.state["phase"] == "cancelled"


def test_requires_two_consecutive_idle_checks(queue, monkeypatch):
    launches = []
    monkeypatch.setattr(queue, "launch", lambda: launches.append(True))
    assert queue.tick() is True
    monkeypatch.setattr(handoff, "gpu_blockers", lambda: [0])
    assert queue.tick() is True
    assert queue.idle_confirmations == 0
    monkeypatch.setattr(handoff, "gpu_blockers", lambda: [])
    assert queue.tick() is True
    assert not launches
    assert queue.tick() is False
    assert launches == [True]


def test_gpu_taken_during_preflight_prevents_launch(queue, monkeypatch):
    queue.idle_confirmations = 1
    monkeypatch.setattr(queue, "preflight", lambda: monkeypatch.setattr(handoff, "gpu_blockers", lambda: [0]))
    assert queue.tick() is True
    assert queue.state["phase"] == "waiting"


def test_one_shot_launch_is_not_repeated(queue, monkeypatch):
    calls = []

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(handoff.subprocess, "Popen", start)
    assert queue.tick() is True
    assert queue.tick() is False
    assert queue.tick() is False
    assert len(calls) == 1
    assert calls[0][0] == ["bash", str(handoff.LAUNCHER)]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert calls[0][1]["start_new_session"]
    assert queue.state["status"] == "complete"


def test_ambiguous_claim_never_duplicates_a_launch(queue):
    queue.state["phase"] = "launching"
    assert queue.tick() is False


def test_environment_drops_old_run_and_hyperparameter_overrides(queue):
    environment = handoff.launch_environment(
        queue.state,
        {
            "MAXRL_TOTAL_TRAINING_STEPS": "230",
            "MAXRL_COST_OFFSET_TOKENS": "999",
            "MAXRL_UPLOAD_CHECKPOINTS": "0",
            "WANDB_RUN_ID": "old",
            "WANDB_RESUME": "must",
            "WANDB_API_KEY": "test-secret",
            "RAY_ADDRESS": "existing-cluster",
            "VLLM_USE_V1": "1",
            "PYTHONPATH": "/another/repository",
            "HF_TOKEN": "test-hf-secret",
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        },
    )
    assert environment["WANDB_RUN_ID"] == "newrun"
    assert environment["WANDB_RESUME"] == "never"
    assert environment["WANDB_API_KEY"] == "test-secret"
    assert environment["HF_TOKEN"] == "test-hf-secret"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["MAXRL_SKIP_ENV_SETUP"] == "1"
    assert environment["RAY_ADDRESS"] == "local"
    assert not any(
        key in environment
        for key in (
            "MAXRL_TOTAL_TRAINING_STEPS",
            "MAXRL_COST_OFFSET_TOKENS",
            "MAXRL_UPLOAD_CHECKPOINTS",
            "PYTHONPATH",
        )
    )


def test_source_changes_block_launch(queue, tmp_path):
    source = tmp_path / "launcher.sh"
    source.write_text("changed")
    queue.state["code_hashes"] = {str(source): "old-hash"}
    with pytest.raises(RuntimeError, match="source changed"):
        handoff.Queue.preflight(queue)


def test_preflight_preserves_existing_outputs(queue, tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "PYTHON", Path(sys.executable))
    output = Path(queue.state["output_root"])
    output.mkdir()
    (output / "existing.log").write_text("keep")
    with pytest.raises(RuntimeError, match="overwrite"):
        handoff.Queue.preflight(queue)
    assert (output / "existing.log").read_text() == "keep"
