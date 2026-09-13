"""Read-only stream recovery and history isolation for the live W&B mirror."""

import json

import pytest

import mirror_wandb_run as mirror
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def history(step, **values):
    record = wandb_internal_pb2.Record()
    for key, value in {"train/global_step": step, **values}.items():
        record.history.item.add(key=key, value_json=json.dumps(value))
    return record


def write_stream(path, records):
    store = DataStore()
    store.open_for_write(str(path))
    offsets = [store.write(record)[:2] for record in records]
    store.close()
    return offsets


def test_reader_preserves_source_and_exact_metrics(tmp_path):
    path = tmp_path / "run.wandb"
    run = wandb_internal_pb2.Record()
    run.run.run_id = "original"
    run.run.entity = "owner"
    run.run.project = "project"
    run.run.config.update.add(key="actor_learning_rate", value_json="0.000001")
    write_stream(path, [run, history(21, **{"train/reward": 0.4402315738843754, "_step": 24})])
    before = path.read_bytes()
    snapshot = mirror.read_snapshot(path)
    assert snapshot.run["id"] == "original"
    assert snapshot.config["actor_learning_rate"] == 1e-6
    assert snapshot.history == [{"train/global_step": 21, "train/reward": 0.4402315738843754, "_step": 24}]
    assert not snapshot.incomplete_tail
    assert path.read_bytes() == before


@pytest.mark.parametrize("cut", [1, 5, 10, 100, 32768])
def test_incomplete_fragment_retries_without_losing_previous_rows(tmp_path, cut):
    path = tmp_path / "run.wandb"
    offsets = write_stream(path, [history(21), history(22, **{"train/text": "x" * 70000})])
    complete = path.read_bytes()
    path.write_bytes(complete[:-cut])
    partial = mirror.read_snapshot(path)
    assert partial.history == [{"train/global_step": 21}]
    assert partial.incomplete_tail
    path.write_bytes(complete)
    assert [mirror.training_step(row) for row in mirror.read_snapshot(path).history] == [21, 22]
    assert offsets[1][1] == len(complete)


def test_corrupt_complete_record_is_not_silently_skipped(tmp_path):
    path = tmp_path / "run.wandb"
    write_stream(path, [history(21), history(22)])
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(AssertionError, match="checksum"):
        mirror.read_snapshot(path)


def snapshots():
    identity = {"id": "original", "entity": "owner", "project": "project"}
    baseline = mirror.Snapshot(
        run=identity, history=[{"train/global_step": step, "train/reward": -1} for step in range(1, 25)]
    )
    resumed = mirror.Snapshot(run=identity, history=[{"train/global_step": 21, "train/reward": 0.4}])
    return resumed, baseline


def test_old_steps_after_checkpoint_are_excluded():
    source, baseline = snapshots()
    assert mirror.select_rows(source, baseline, 20) == [
        {"train/global_step": 20, "train/reward": -1},
        {"train/global_step": 21, "train/reward": 0.4},
    ]


def test_baseline_cannot_come_from_another_run():
    source, baseline = snapshots()
    baseline.run = {**baseline.run, "id": "other"}
    with pytest.raises(ValueError, match="same original run"):
        mirror.select_rows(source, baseline, 20)


def test_duplicate_source_steps_are_rejected():
    source, baseline = snapshots()
    source.history *= 2
    with pytest.raises(ValueError, match="strictly increasing"):
        mirror.select_rows(source, baseline, 20)


def test_internal_source_clocks_do_not_overwrite_destination_clocks():
    payload = mirror.metric_payload({"train/global_step": 21, "_step": 24, "_timestamp": 123, "_runtime": 456})
    assert payload == {
        "train/global_step": 21,
        "mirror/source_wandb_step": 24,
        "mirror/source_timestamp": 123,
        "mirror/source_runtime_seconds": 456,
    }


def test_credentials_are_not_copied_to_new_run():
    assert mirror.safe_config(
        {
            "actor_learning_rate": 1e-6,
            "use_wandb": "private-value",
            "WANDB_API_KEY": "private-value",
            "secret": "private-value",
            "input_key": "problem",
            "_wandb": {},
        }
    ) == {"actor_learning_rate": 1e-6, "use_wandb": True, "input_key": "problem"}


def test_repeated_polls_and_resume_do_not_duplicate_steps():
    class FakeRun:
        def __init__(self):
            self.rows = []

        def log(self, payload, step, commit):
            assert commit is True
            self.rows.append((step, payload))

    run = FakeRun()
    rows = [{"train/global_step": step} for step in (20, 21, 22)]
    last = mirror.upload_new_rows(run, rows, 20)
    last = mirror.upload_new_rows(run, rows, last)
    assert last == 22
    assert [step for step, _ in run.rows] == [21, 22]


def test_clean_exit_is_retained(tmp_path):
    path = tmp_path / "run.wandb"
    exit_record = wandb_internal_pb2.Record()
    exit_record.exit.exit_code = 0
    write_stream(path, [history(100), exit_record])
    assert mirror.read_snapshot(path).exit_code == 0


def test_remote_fallback_uses_resume_boundary_not_just_training_step():
    source, baseline = snapshots()
    source.history[0]["_step"] = 24
    rows = mirror.select_rows(source, baseline, 20)
    remote = [
        {"train/global_step": 21, "train/reward": -1, "_step": 20},
        {"train/global_step": 21, "train/reward": 0.4, "_step": 24},
        {"train/global_step": 22, "train/reward": 0.5, "_step": 25},
    ]
    combined = mirror.merge_remote_rows(source, rows, remote, 20)
    assert [row["train/global_step"] for row in combined] == [20, 21, 22]
    assert [row["train/reward"] for row in combined] == [-1, 0.4, 0.5]
    assert mirror.merge_remote_rows(source, combined, remote, 20) == combined


def test_remote_fallback_rejects_wrong_anchor_or_another_resume():
    source, baseline = snapshots()
    source.history[0]["_step"] = 24
    rows = mirror.select_rows(source, baseline, 20)
    with pytest.raises(ValueError, match="boundary"):
        mirror.merge_remote_rows(source, rows, [{"train/global_step": 21, "train/reward": -1, "_step": 24}], 20)
    with pytest.raises(ValueError, match="mapping"):
        mirror.merge_remote_rows(
            source, rows, [source.history[0], {"train/global_step": 21, "train/reward": 0.4, "_step": 25}], 20
        )


def test_remote_fallback_requires_a_local_anchor():
    source, baseline = snapshots()
    source.history = []
    rows = mirror.select_rows(source, baseline, 20)
    assert mirror.merge_remote_rows(source, rows, [{"train/global_step": 21, "_step": 24}], 20) == rows
