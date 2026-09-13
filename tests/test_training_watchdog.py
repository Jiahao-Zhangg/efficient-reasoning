"""Exercise recovery decisions without stopping processes or starting GPU work."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import watch_rloo_training as watch


def runner(tmp_path, monkeypatch, alive=True):
    state_path = tmp_path / "state.json"
    log = tmp_path / "launch.log"
    log.write_text("training in progress\n")
    state = {
        "phase": "running",
        "trainer": {"pid": 10, "start": "100"},
        "launcher": {"pid": 9, "start": "90"},
        "archiver": {"pid": 11, "start": "110"},
        "owned": {},
        "launch_log": str(log),
        "run_dir": str(tmp_path),
        "run_dirs": [str(tmp_path)],
        "restored": None,
        "failure_seen_at": None,
        "attempts": 0,
    }
    state_path.write_text(json.dumps(state))
    snapshot = {
        9: {"start": "90", "ppid": 1, "sid": 9},
        10: {"start": "100", "ppid": 9, "sid": 10},
        11: {"start": "110", "ppid": 1, "sid": 11},
        12: {"start": "120", "ppid": 10, "sid": 10},
        99: {"start": "990", "ppid": 1, "sid": 99},
    }
    if not alive:
        snapshot.pop(10)
    monkeypatch.setattr(watch, "processes", lambda: snapshot)
    monkeypatch.setattr(watch, "HfApi", Mock)
    result = watch.Supervisor(state_path)
    result.ensure_archiver = Mock()
    result.cleanup_failed_processes = Mock()
    result.restart = Mock()
    return result, snapshot


def manifest(step=20):
    return {"step": step, "repo": f"{watch.HF_PREFIX}-step_{step}", "revision": "abc", "files": {}}


def test_live_training_is_never_restarted(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(watch, "latest_archive", Mock(side_effect=AssertionError("No cloud check needed")))
    assert supervisor.tick()
    supervisor.restart.assert_not_called()
    supervisor.cleanup_failed_processes.assert_not_called()
    assert supervisor.state["status"] == "running"
    assert "99" not in supervisor.state["owned"]
    assert "12" in supervisor.state["owned"]


def test_only_real_process_identity_counts():
    snapshot = {10: {"start": "new"}}
    assert not watch.same_process(10, "old", snapshot)
    assert not watch.same_process(11, None, snapshot)
    assert not watch.same_process(None, None, snapshot)
    assert watch.same_process(10, "new", snapshot)


def test_failure_requires_confirmation_then_latest_checkpoint(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    monkeypatch.setattr(watch, "latest_archive", lambda _: manifest(40))
    monkeypatch.setattr(watch, "latest_local", lambda _: 40)
    assert supervisor.tick()
    supervisor.restart.assert_not_called()
    supervisor.state["failure_seen_at"] = time.time() - 61
    assert supervisor.tick()
    supervisor.restart.assert_called_once_with(manifest(40))


def test_does_not_fall_back_while_newer_local_checkpoint_is_uploading(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    supervisor.state["failure_seen_at"] = time.time() - 61
    monkeypatch.setattr(watch, "latest_archive", lambda _: manifest(20))
    monkeypatch.setattr(watch, "latest_local", lambda _: 40)
    with pytest.raises(watch.Defer, match="latest local step 40"):
        supervisor.tick()
    supervisor.restart.assert_not_called()


def test_verified_step100_stops_watchdog_without_restarting(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    supervisor.state["failure_seen_at"] = time.time() - 61
    monkeypatch.setattr(watch, "latest_archive", lambda _: manifest(100))
    monkeypatch.setattr(watch, "latest_local", lambda _: 0)
    assert not supervisor.tick()
    assert supervisor.state["status"] == "complete"
    supervisor.restart.assert_not_called()


def test_stop_file_leaves_running_training_untouched(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch)
    supervisor.path.with_name("STOP").touch()
    assert not supervisor.tick()
    supervisor.cleanup_failed_processes.assert_not_called()
    supervisor.restart.assert_not_called()


def test_incomplete_download_is_kept_as_pending_not_as_live_checkpoint(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    supervisor.restart = watch.Supervisor.restart.__get__(supervisor)
    monkeypatch.setattr(watch, "gpu_blockers", lambda: [])
    supervisor.prepare = Mock(side_effect=OSError("download interrupted"))
    old_dir = supervisor.state["run_dir"]
    with pytest.raises(OSError):
        supervisor.restart(manifest())
    assert supervisor.state["run_dir"] == old_dir
    assert supervisor.state["restored"] is None
    first_id = supervisor.state["pending"]["run_id"]
    with pytest.raises(OSError):
        supervisor.restart(manifest())
    assert supervisor.state["pending"]["run_id"] == first_id


def test_busy_gpus_do_not_download_or_launch(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    supervisor.restart = watch.Supervisor.restart.__get__(supervisor)
    monkeypatch.setattr(watch, "gpu_blockers", lambda: [0])
    supervisor.prepare = Mock()
    with pytest.raises(watch.Defer, match="Waiting for GPUs"):
        supervisor.restart(manifest())
    supervisor.prepare.assert_not_called()


def test_manifest_requires_all_optimizer_shards_and_hashes():
    siblings = [SimpleNamespace(rfilename="train_config.json")]
    info = SimpleNamespace(siblings=siblings, id=manifest()["repo"], sha="abc")
    assert watch.checkpoint_manifest(info, 20) is None
    for name in watch.STATE_FILES:
        siblings.append(
            SimpleNamespace(rfilename=f"global_step20/{name}", size=123, lfs=SimpleNamespace(sha256="a" * 64))
        )
    assert len(watch.checkpoint_manifest(info, 20)["files"]) == 3
    siblings[-1].lfs.sha256 = None
    assert watch.checkpoint_manifest(info, 20) is None


def test_latest_archive_uses_numeric_step_and_ignores_partial_repo():
    api = Mock()
    api.list_models.return_value = [SimpleNamespace(id=f"{watch.HF_PREFIX}-step_{step}") for step in (20, 100, 40)]

    def info(repo, files_metadata):
        step = int(repo.rsplit("_", 1)[1])
        siblings = [SimpleNamespace(rfilename="train_config.json")]
        if step != 100:
            siblings.extend(
                SimpleNamespace(rfilename=f"global_step{step}/{name}", size=123, lfs=SimpleNamespace(sha256="a" * 64))
                for name in watch.STATE_FILES
            )
        return SimpleNamespace(id=repo, sha="abc", siblings=siblings)

    api.model_info.side_effect = info
    assert watch.latest_archive(api)["step"] == 40


def test_restore_cleanup_waits_for_vllm_synchronization(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch)
    supervisor.state["restored"] = manifest()
    supervisor.remove_restored_copy = Mock()
    assert supervisor.tick()
    supervisor.remove_restored_copy.assert_not_called()
    supervisor.ensure_archiver.assert_not_called()
    Path(supervisor.state["launch_log"]).write_text(watch.READY)
    assert supervisor.tick()
    supervisor.remove_restored_copy.assert_called_once()
    supervisor.ensure_archiver.assert_called_once()


def test_restart_environment_always_selects_new_online_run(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "old-run")
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_FORK_FROM", "old-run?_step=20")
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
    env = watch.launch_environment("new-attempt", "fresh-id")
    assert env["WANDB_RUN_ID"] == "fresh-id"
    assert env["WANDB_RESUME"] == "never"
    assert env["WANDB_MODE"] == "online"
    assert env["RESUME"] == "1"
    assert env["GPU_IDS"] == "0,1,2,3"
    assert env["ARCHIVE_CHECKPOINTS"] == "0"
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert "WANDB_FORK_FROM" not in env
    assert "TORCH_FORCE_WEIGHTS_ONLY_LOAD" not in env


def test_cleanup_signals_only_tracked_processes_with_matching_identity(tmp_path, monkeypatch):
    supervisor, snapshot = runner(tmp_path, monkeypatch, alive=False)
    supervisor.state["owned"] = {"12": "120", "99": "old-identity"}
    supervisor.state["launcher"] = {}
    supervisor.cleanup_failed_processes = watch.Supervisor.cleanup_failed_processes.__get__(supervisor)
    signals = []
    monkeypatch.setattr(watch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    supervisor.cleanup_failed_processes()
    assert [pid for pid, _ in signals] == [12, 12]
    assert snapshot[99]["start"] == "990"


def test_launcher_startup_never_creates_a_duplicate_training_job(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    supervisor.state["phase"] = "starting"
    supervisor.state["trainer"] = {}
    assert supervisor.tick()
    supervisor.restart.assert_not_called()
    supervisor.cleanup_failed_processes.assert_not_called()


def test_failed_launch_without_training_pid_does_not_start_an_archiver(tmp_path, monkeypatch):
    supervisor, _ = runner(tmp_path, monkeypatch, alive=False)
    supervisor.state["trainer"] = {}
    supervisor.state["archiver"] = None
    supervisor.ensure_archiver = watch.Supervisor.ensure_archiver.__get__(supervisor)
    popen = Mock(side_effect=AssertionError("No archiver should start"))
    monkeypatch.setattr(watch.subprocess, "Popen", popen)
    supervisor.ensure_archiver()
    popen.assert_not_called()
