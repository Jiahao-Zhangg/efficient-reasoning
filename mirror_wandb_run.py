"""Mirror a live, local W&B stream into a separate run without touching training.

The binary reader uses the installed W&B SDK's framing/CRC implementation, but
only on an in-memory, read-only snapshot. Tested with wandb 0.28.0. Incomplete
tail records are retried on the next poll; corrupt complete records are errors.
"""

import argparse
import io
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import LEVELDBLOG_BLOCK_LEN, LEVELDBLOG_HEADER_LEN, DataStore


class IncompleteTail(Exception):
    """The writer has not flushed the rest of a record yet."""


class SnapshotStore(DataStore):
    def __init__(self, data):
        super().__init__()
        self._fname = "<read-only snapshot>"
        self._fp = io.BytesIO(data)
        self._size_bytes = len(data)
        self._opened_for_scan = True
        if len(data) < LEVELDBLOG_HEADER_LEN:
            raise IncompleteTail()
        self._read_header()

    def scan_record(self):
        offset = self.get_offset()
        remaining = self._size_bytes - offset
        if remaining == 0:
            return None
        if remaining < LEVELDBLOG_HEADER_LEN:
            raise IncompleteTail()
        header = bytes(self._fp.getbuffer()[offset : offset + LEVELDBLOG_HEADER_LEN])
        _, length, _ = struct.unpack("<IHB", header)
        if remaining < LEVELDBLOG_HEADER_LEN + length:
            raise IncompleteTail()
        return super().scan_record()


@dataclass
class Snapshot:
    run: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    exit_code: int | None = None
    incomplete_tail: bool = False
    size: int = 0


def decode_items(items):
    result = {}
    for item in items:
        keys = [item.key] if item.key else list(item.nested_key)
        if not keys:
            raise ValueError("W&B item has no key")
        target = result
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = json.loads(item.value_json)
    return result


def read_snapshot(path):
    # Bound the read to the current size; the trainer may continue appending.
    with Path(path).open("rb") as source:
        data = source.read(os.fstat(source.fileno()).st_size)
    result = Snapshot(size=len(data))
    store = SnapshotStore(data)
    try:
        while True:
            offset = store.get_offset()
            padding = LEVELDBLOG_BLOCK_LEN - offset % LEVELDBLOG_BLOCK_LEN
            try:
                if padding < LEVELDBLOG_HEADER_LEN and len(data) - offset < padding:
                    raise IncompleteTail()
                raw = store.scan_data()
            except IncompleteTail:
                result.incomplete_tail = True
                break
            if raw is None:
                result.incomplete_tail = offset < len(data)
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                result.run = {
                    "id": record.run.run_id,
                    "entity": record.run.entity,
                    "project": record.run.project,
                    "name": record.run.display_name,
                }
                result.config.update(decode_items(record.run.config.update))
            elif kind == "config":
                result.config.update(decode_items(record.config.update))
            elif kind == "history":
                result.history.append(decode_items(record.history.item))
            elif kind == "exit":
                result.exit_code = record.exit.exit_code
    finally:
        store.close()
    return result


def training_step(row):
    value = row["train/global_step"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 0:
        raise ValueError(f"Invalid training step: {value!r}")
    return int(value)


def select_rows(source, baseline, checkpoint_step):
    rows = []
    if baseline is not None:
        for key in ("id", "entity", "project"):
            if source.run[key] != baseline.run[key]:
                raise ValueError("Baseline and resumed stream must belong to the same original run")
        matches = [row for row in baseline.history if row.get("train/global_step") == checkpoint_step]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one baseline row at step {checkpoint_step}")
        rows.extend(matches)
    rows.extend(row for row in source.history if "train/global_step" in row and training_step(row) > checkpoint_step)
    previous = -1
    for row in rows:
        step = training_step(row)
        if step <= previous:
            raise ValueError("Source steps must be strictly increasing; refusing to mix or duplicate history")
        previous = step
    return rows


def metric_payload(row):
    # Keep training values exact; the destination owns its own internal clock.
    result = {key: value for key, value in row.items() if not key.startswith("_")}
    for key, name in (("_timestamp", "timestamp"), ("_runtime", "runtime_seconds"), ("_step", "wandb_step")):
        if key in row:
            result[f"mirror/source_{name}"] = row[key]
    return result


def training_values(row):
    return {key: value for key, value in row.items() if key.startswith(("train/", "perf/"))}


def merge_remote_rows(source, rows, remote_rows, checkpoint_step):
    """Use cloud rows only after proving their boundary against this local session.

    A resumed run has monotonic internal `_step` values even when its custom
    train/global_step restarts. Matching the first local resumed row prevents
    importing pre-crash rows with the same training step.
    """
    local_rows = [row for row in source.history if row.get("train/global_step", -1) > checkpoint_step]
    if not local_rows or not remote_rows:
        return rows
    anchor = local_rows[0]
    start = anchor["_step"]
    matches = [row for row in remote_rows if row.get("_step") == start]
    if len(matches) != 1 or training_values(matches[0]) != training_values(anchor):
        raise ValueError("Remote resume boundary does not match the local resumed session")
    merged = {training_step(row): row for row in rows}
    for row in remote_rows:
        if row.get("_step", -1) < start or "train/global_step" not in row:
            continue
        step = training_step(row)
        if step != training_step(anchor) + row["_step"] - start:
            raise ValueError("Remote internal-step mapping changed; refusing to mix training sessions")
        if step in merged and training_values(merged[step]) != training_values(row):
            raise ValueError(f"Local and remote metrics disagree at training step {step}")
        merged.setdefault(step, row)
    return [merged[step] for step in sorted(merged)]


def safe_config(config):
    result = {}
    for key, value in config.items():
        if key.startswith("_") or any(
            word in key.lower() for word in ("api_key", "password", "secret", "access_token")
        ):
            continue
        # OpenRLHF may store an API key in this otherwise boolean CLI option.
        result[key] = bool(value) if key == "use_wandb" else value
    return result


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in ("Z", "X") else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def upload_new_rows(run, rows, last_step):
    for row in rows:
        step = training_step(row)
        if step <= last_step:
            continue
        run.log(metric_payload(row), step=step, commit=True)
        last_step = step
        print(f"Mirrored train/global_step={step}", flush=True)
    return last_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument("--run-id", required=True, help="A NEW W&B ID, never the source run ID")
    parser.add_argument("--name", required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="Separate directory for the mirror's own logs")
    parser.add_argument("--source-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--drain-seconds", type=float, default=120)
    parser.add_argument("--resume-mirror", action="store_true")
    parser.add_argument(
        "--read-remote", action="store_true", help="Also read verified resumed rows from the source API"
    )
    parser.add_argument("--inspect", action="store_true", help="Read local metrics only; do not contact W&B")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.drain_seconds < 0 or args.checkpoint_step < 0:
        parser.error("Invalid poll, drain, or checkpoint value")
    args.source = args.source.resolve()
    source = read_snapshot(args.source)
    baseline = read_snapshot(args.baseline) if args.baseline else None
    if not source.run or args.run_id == source.run["id"]:
        parser.error("A distinct destination run and a valid source run are required")
    if args.read_remote and source.config.get("logging_steps") != 1:
        parser.error("Remote fallback requires one history row per RL step (logging_steps=1)")
    rows = select_rows(source, baseline, args.checkpoint_step)
    metadata = {
        "source_run": "/".join(source.run[key] for key in ("entity", "project", "id")),
        "source_file": str(args.source),
        "checkpoint_step": args.checkpoint_step,
        "baseline_file": str(args.baseline.resolve()) if args.baseline else None,
        "read_only": True,
    }
    if args.inspect:
        print(json.dumps({"source": metadata, "steps": [training_step(row) for row in rows], "rows": rows}, indent=2))
        return
    identity = process_identity(args.source_pid)
    if identity is None:
        parser.error("Source trainer is not alive; refusing to start a live mirror")
    work_dir = args.work_dir.resolve()
    if work_dir in args.source.parents or args.source.parent in work_dir.parents:
        parser.error("Use a separate work directory, not the original W&B directory")

    # Clear inherited run/session selection, but retain credentials and network settings.
    for key in (
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_RESUME_FROM",
        "WANDB_FORK_FROM",
        "WANDB_SWEEP_ID",
        "WANDB_DIR",
        "WANDB_SERVICE",
        "_WANDB_SERVICE",
        "WANDB_SETTINGS_PATH",
    ):
        os.environ.pop(key, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import wandb

    last_step = -1
    if args.resume_mirror:
        existing = wandb.Api(timeout=30).run(f"{source.run['entity']}/{source.run['project']}/{args.run_id}")
        if existing.config.get("mirror") != metadata:
            raise ValueError("Existing destination does not match this source; refusing to resume")
        last_step = max(
            (int(row["train/global_step"]) for row in existing.scan_history(keys=["train/global_step"])), default=-1
        )
    remote_source = None
    if args.read_remote:
        remote_source = wandb.Api(timeout=20).run(metadata["source_run"])
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=source.run["entity"],
        project=source.run["project"],
        id=args.run_id,
        name=args.name,
        resume="must" if args.resume_mirror else "never",
        mode="online",
        dir=str(args.work_dir.resolve()),
        config={**safe_config(source.config), "mirror": metadata},
        job_type="metrics-mirror",
        tags=["metrics-mirror", f"resume-step-{args.checkpoint_step}"],
        notes=(
            "Read-only live metrics mirror, not a separate training job. Only the checkpoint baseline and "
            "this resumed session's metrics are included. The original run is not modified. "
            f"Source: https://wandb.ai/{source.run['entity']}/{source.run['project']}/runs/{source.run['id']}"
        ),
        settings=wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            x_disable_stats=True,
            x_disable_meta=True,
        ),
    )
    print(f"Mirror URL: {run.url}", flush=True)
    run.define_metric("train/global_step")
    for prefix in ("train/*", "perf/*"):
        run.define_metric(prefix, step_metric="train/global_step", step_sync=True)
    run.define_metric("mirror/*", hidden=True)
    dead_since = None
    last_size = source.size
    last_change = time.monotonic()
    try:
        while True:
            local_rows = [row for row in source.history if row.get("train/global_step", -1) > args.checkpoint_step]
            if remote_source is not None and local_rows:
                try:
                    # Refresh the cached last-history bound before each incremental scan.
                    remote_source.load(force=True)
                    remote_rows = list(remote_source.scan_history(min_step=int(local_rows[0]["_step"])))
                except (wandb.Error, OSError) as exc:
                    print(f"Source API unavailable ({type(exc).__name__}); continuing from local logs", flush=True)
                else:
                    rows = merge_remote_rows(source, rows, remote_rows, args.checkpoint_step)
            last_step = upload_new_rows(run, rows, last_step)
            alive = process_identity(args.source_pid) == identity
            if source.size != last_size:
                last_size, last_change = source.size, time.monotonic()
            if source.exit_code is not None and not source.incomplete_tail:
                run.summary["mirror/source_status"] = "logger_exited"
                run.finish(exit_code=source.exit_code)
                return
            if not alive:
                if dead_since is None:
                    dead_since = time.monotonic()
                    run.summary["mirror/source_status"] = "trainer_stopped_draining_logs"
                if time.monotonic() - max(dead_since, last_change) >= args.drain_seconds:
                    run.summary["mirror/source_status"] = "trainer_stopped_without_clean_logger_exit"
                    run.finish(exit_code=1)
                    return
            time.sleep(args.poll_seconds)
            source = read_snapshot(args.source)
            rows = select_rows(source, baseline, args.checkpoint_step)
    except BaseException:
        # This affects ONLY the separate mirror, never the original run/trainer.
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
