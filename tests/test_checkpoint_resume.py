"""Exercise the resume synchronization branch without starting CUDA or Ray."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _resume_branch():
    source = Path(__file__).parents[1] / "openrlhf/trainer/ray/ppo_actor.py"
    module = ast.parse(source.read_text())
    actor = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ActorModelRayActor")
    fit = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    branch = next(
        node for node in fit.body if isinstance(node, ast.If) and "args.load_checkpoint" in ast.unparse(node.test)
    )
    return compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec")


@pytest.mark.parametrize(
    ("load_checkpoint", "checkpoint_exists", "engines_present", "should_sync"),
    [(True, True, True, True), (False, True, True, False), (True, False, True, False), (True, True, False, False)],
)
def test_resume_waits_for_broadcast_before_generation(load_checkpoint, checkpoint_exists, engines_present, should_sync):
    events = []
    namespace = {
        "args": SimpleNamespace(load_checkpoint=load_checkpoint),
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda _: checkpoint_exists)),
        "ckpt_path": "checkpoints/_actor",
        "vllm_engines": [object()] if engines_present else None,
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: events.append("barrier"))),
        "trainer": SimpleNamespace(_broadcast_to_vllm=lambda: events.append("broadcast")),
        "strategy": SimpleNamespace(print=lambda _: events.append("ready")),
    }
    exec(_resume_branch(), namespace)
    events.append("generate")
    assert events == (["barrier", "broadcast", "barrier", "ready", "generate"] if should_sync else ["generate"])
