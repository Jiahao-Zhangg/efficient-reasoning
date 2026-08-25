# Repository Guidelines

## Project Structure & Module Organization

`openrlhf/` contains the training framework: CLI entry points live in `openrlhf/cli/`, trainers in `openrlhf/trainer/`, and model, dataset, and distributed helpers in their matching subpackages. `reward_server/math_server.py` provides the Flask reward endpoint used by RLOO jobs. Repository-level grading and parsing helpers are in `utils/`; `utils/latex2sympy/` is a separately installable parser with its own tests and generated ANTLR files. Top-level `run_rloo_*.sh` files are the paper's Slurm recipes, while `examples/scripts/` contains upstream training examples. Documentation and website assets live under `docs/` and `static/`; local datasets belong in `datasets/`.

## Setup, Test, and Development Commands

- `pip install -e utils/latex2sympy && pip install -e .` installs the parser and project in editable mode (Python 3.10 is the tested baseline).
- `(cd utils/latex2sympy && pytest -c setup.cfg tests)` runs the existing parser unit suite.
- `black --check . && isort --check-only . && ruff check .` checks the formatting and lint settings defined in `pyproject.toml`.
- `sbatch run_rloo_1.5B.sh` launches the four-GPU Slurm/Ray training recipe; use `run_rloo_7B.sh` for the two-node variant.
- `python evaluate_model.py --model_path <checkpoint> --dataset openai/gsm8k --scale 1.5B` evaluates a checkpoint with vLLM.

Training and evaluation require CUDA-capable GPUs; the published setup used GH200 GPUs, CUDA 12.6, and Python 3.10.15.

## Coding Style & Naming Conventions

Use four-space indentation and keep Python lines at or below 119 characters. Black, isort (Black profile), and Ruff are the configured tools. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Keep CLI flags descriptive and consistent with existing underscore-separated options. Do not hand-edit generated files under `utils/latex2sympy/gen/`.

## Testing Guidelines

Use pytest. Existing parser files follow `*_test.py`; new project-level tests should go in `tests/` as `test_<feature>.py`, matching `pyproject.toml`. Add focused unit tests for grading, parsing, or reward changes and mark broader tests with the configured `integration` or `system` markers. No coverage threshold is enforced, but changed behavior should be exercised.

## Commit & Pull Request Guidelines

History favors short summaries such as `Update README.md` and `gitignore fixed`; make these more specific, imperative, and scoped (for example, `Fix reward length normalization`). Pull requests should explain the motivation, summarize behavior changes, list commands run, and link relevant issues. Include GPU/CUDA details for training changes and screenshots only for website or visualization updates. Never commit W&B keys, model checkpoints, logs, or generated outputs; these paths are ignored for local use.
