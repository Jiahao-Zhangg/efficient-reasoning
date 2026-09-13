# Reproduce the ER Conda Environment

This is the **2026-09-13 snapshot of `efficient_reasoning`**, used for both
Qwen3/Math12K and DeepSeek/compression ER experiments. It is not the separate
`maxrl` environment or the paper's unmodified dependency set. The source revision
at export was `ccaaf3f116e7f7692b81851359a107c3b6d69760`.

## What is included

- `conda-linux-64.explicit.txt`: 30 exact Conda artifacts, including Python
  3.10.15, build identifiers, download URLs, and MD5 checksums.
- `requirements.lock.txt`: all 218 non-editable Python distributions, including
  transitive dependencies and CUDA runtime packages. PyTorch is `2.6.0+cu124`;
  Transformers `4.51.3`, vLLM `0.8.4`, and MathVerify `0.9.0`.
- `build-constraints.txt`: the exported versions of pip's isolated build tools.
- `snapshot.json`: all 220 unique Python package versions, Conda builds, source
  revision, and reference host details. The two additional Python packages are
  this checkout's editable `openrlhf` and modified `latex2sympy2`.
- `check_environment.py`: a read-only package/build/origin comparison. It does
  not allocate GPU memory or change the environment.

The original FlashAttention installation referenced a machine-local temporary
wheel. Its replacement is the official CUDA 12 / Torch 2.6 / Python 3.10 /
`cxx11abiFALSE` release wheel, with the **same verified SHA256**. Conda's local
pip build path and the editable checkout paths are not copied to the lock file.

## Host requirements

Use **Linux x86_64** with an NVIDIA GPU supported by this older CUDA stack.
The reference machine runs Ubuntu 22.04.5, glibc 2.35, A100 80GB GPUs, driver
550.144.03, GCC 11.4.0, and CUDA Toolkit 12.4.131. Match these where practical;
the explicit artifacts are not for macOS, Windows, or ARM/GH200 hosts.

Install the NVIDIA driver and CUDA 12.4 toolkit separately. Pip provides CUDA
runtime libraries, **not `nvcc` or the host driver**. DeepSpeed compiles its
optimizer kernels on first use, so a working C++ compiler and matching toolkit
are required. Set `CUDA_HOME` to the toolkit location; do not accidentally use
an older system `nvcc`. Provide disk space for the environment, downloads,
compiler cache, datasets, and checkpoints.

## Restore on another machine

Install Conda first (exporter version: 26.1.1). Clone this repository and use the
commit containing this snapshot, not an arbitrary future `main`. Run the
following **from the repository root**, using a new environment name if
`efficient_reasoning` already exists. Do not update an environment used by an
active training job.

```bash
conda create --name efficient_reasoning --no-default-packages \
  --file environments/efficient_reasoning/conda-linux-64.explicit.txt
conda activate efficient_reasoning
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"

DS_BUILD_OPS=0 python -m pip --isolated install --no-cache-dir --no-deps \
  --build-constraint environments/efficient_reasoning/build-constraints.txt \
  -r environments/efficient_reasoning/requirements.lock.txt

python -m pip --isolated install --no-deps --no-build-isolation \
  -e ./utils/latex2sympy -e .

python -m pip check
python environments/efficient_reasoning/check_environment.py
```

`--no-deps` keeps the complete exported package set unchanged rather than
resolving newer dependencies. Keep build isolation enabled for the dependency
installation: the original DeepSpeed wheel was built without Torch and without
precompiled ops. Repository editables are installed last without re-resolving
dependencies. Do not follow this with `pip install -U` or `conda update` if you
want to retain the snapshot.

When the destination GPUs are idle, check the compiled imports and CUDA runtime:

```bash
python -c 'import torch, flash_attn, vllm, deepspeed, math_verify; assert torch.version.cuda == "12.4"; assert not torch._C._GLIBCXX_USE_CXX11_ABI; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

W&B/Hugging Face credentials, datasets, checkpoints, and logs are deliberately
excluded. Authenticate separately on the new machine. This export recreates
package **versions** and exact Conda/FlashAttention artifacts; it is not a
container or a fully hash-locked/offline archive of every pip wheel. OS, drivers,
GPU architecture, and runtime-compiled kernels can still differ, and identical
training results are not guaranteed. A fresh destination GPU installation has
not been tested as part of this export.

References: [Conda explicit specifications](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#explicit-spec-files),
[pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/),
and the [official FlashAttention release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1).
