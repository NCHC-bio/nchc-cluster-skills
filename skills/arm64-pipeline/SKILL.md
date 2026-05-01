---
name: arm64-pipeline
description: >
  Use when setting up, porting, or debugging a job on any NCHC/TWCC partition whose
  compute nodes are aarch64 / ARM64 while the login node is x86_64. The current
  example is GB200 (gb200-rack1, gb200-rack2, gb200-dev, gb200-full; Grace CPU),
  and this skill will apply to any future ARM-based partition the cluster adds.
  Scope is strictly aarch64/arch-mismatch issues. Trigger on: user wants to run
  on an ARM / aarch64 / Grace / Grace Hopper node, "Exec format error",
  "cannot execute binary file", torch CUDA False on ARM, torchcodec on ARM,
  libavutil missing, FFmpeg on ARM, uv on ARM node, conda/virtualenv leakage
  to ARM node, "my pipeline works on login but fails on the ARM node".
  Do NOT trigger for: generic SLURM submission (use slurm-submission),
  non-interactive auth / HF token / wandb in batch (use slurm-submission),
  partition specs, preemption, QoS (use cluster-info),
  tmpfs / cgroup OOM / SIGKILL / hangs (use slurm-debug).
---

# ARM64 (aarch64) Pipeline Setup Guide

**Core principle:** When the **login node is x86_64** but a **compute partition
is aarch64** (ARM64), **nothing installed in `$HOME` on the login node is
guaranteed to run on the compute node**. Every binary, Python interpreter,
and shared library used by the job must be aarch64-native and accessible
from the compute node (typically staged under `/work`).

**Current ARM partitions on NCHC/TWCC:** `gb200-rack1`, `gb200-rack2`,
`gb200-dev`, `gb200-full` (NVIDIA Grace CPU). If the cluster adds new ARM
partitions in the future, the same rules apply — check `uname -m` on an
interactive session to confirm the ISA.

**REQUIRED:** Use cluster-info skill first for partition specs, QoS, and
preemption rules of the specific ARM partition you're targeting.

---

## Quick Reference: Symptom → Section

| Symptom on an ARM compute node | Root cause | Section |
|--------------------------------|-----------|---------|
| `cannot execute binary file: Exec format error` | x86 binary on aarch64 node | §1 Arch mismatch |
| `Failed to inspect Python interpreter ... Exec format error (os error 8)` | `CONDA_PREFIX` / `VIRTUAL_ENV` leaked from login shell | §2 Env leakage |
| `torch.cuda.is_available() == False` after install succeeds | Got aarch64 CPU wheel from PyPI, not CUDA wheel | §3 PyTorch CUDA |
| `libavutil.so.56: cannot open shared object file` | No system FFmpeg on the ARM node; torchcodec dlopens unversioned | §4 FFmpeg + torchcodec |
| `torchvision.io.VideoReader` AttributeError | VideoReader removed in torchvision ≥0.26 (the only aarch64 CUDA wheels) | §4 |

> This skill covers **only aarch64/arch-mismatch** issues. For other ARM-job
> pain points, see:
>
> - **Non-interactive auth** (HuggingFace token, wandb API key, `hf auth
>   whoami` 401 in batch) → `slurm-submission` skill.
> - **Preemption / requeue / QoS priority, `--time` not guaranteed** →
>   `cluster-info` skill.
> - **SIGKILL / OOM / tmpfs-cgroup pressure / hangs** → `slurm-debug` skill.

---

## 1. Architecture mismatch (x86 login, aarch64 compute)

| Location | ISA | What runs here |
|----------|-----|----------------|
| Login node | `x86_64` | `sbatch`, git, editors |
| ARM compute node (e.g. `gb200-*`) | `aarch64` | The actual job |

Anything you install interactively on the login node (`pip install --user`, `uv` in
`~/.local/bin`, conda envs) is x86 and **will fail** when the job runs.

### Fix: stage aarch64 binaries under `/work` from inside an aarch64 job

Example for `uv`:

```bash
UV_AARCH64="/work/$USER/.local/bin/uv-aarch64"
if [ ! -x "$UV_AARCH64" ]; then
    mkdir -p "$(dirname "$UV_AARCH64")"
    curl -fsSL \
      "https://github.com/astral-sh/uv/releases/download/0.10.0/uv-aarch64-unknown-linux-gnu.tar.gz" \
      -o /tmp/uv.tgz
    tar -xz -f /tmp/uv.tgz -C "$(dirname "$UV_AARCH64")" \
      --strip-components=1 uv-aarch64-unknown-linux-gnu/uv
    chmod +x "$UV_AARCH64"
fi
UV="$UV_AARCH64"
```

Same pattern for any CLI you expect in `$HOME`: install the aarch64 build to
`/work/...` so every ARM compute node can read and execute it.

---

## 2. Conda / virtualenv leakage from login shell

`uv` (and plain `python`) inherit `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `VIRTUAL_ENV`
from the login environment. Those point to **x86** interpreters and crash on the
aarch64 compute node with `Exec format error (os error 8)`.

### Fix: unset at the top of every sbatch script

```bash
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE VIRTUAL_ENV
```

Then let `uv` download a native aarch64 CPython (don't trust any system Python
on the compute node):

```bash
"$UV" python install 3.12
"$UV" sync --locked --python 3.12 --extra <your-extras>
"$UV" run --python 3.12 <command>
```

Cached under `~/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/`.

---

## 3. PyTorch on aarch64: default wheel is CPU-only

The aarch64 torch wheel on PyPI is CPU-only. `torch.cuda.is_available()`
returns `False` on ARM compute even with GPUs present.

**Index name follows the node's CUDA driver.** Run `nvidia-smi` on the
compute node and use `cu<major><minor>`. GB200 currently ships CUDA 13.0 →
`cu130`. Do not hardcode `cu128`.

### Fix A — solo project (you own `uv.lock`)

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch       = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" }]
torchvision = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" }]
```

Then **on an aarch64 compute node**:

```bash
"$UV" lock --upgrade-package torch --upgrade-package torchvision   # required if uv.lock exists
"$UV" sync --python 3.12
```

Skipping the relock leaves the CPU wheel pinned and `uv sync` honours it.
Widen `[project]` bounds if the cu1XX wheel jumps a major (e.g.
`torch>=2.7,<2.13`).

### Fix B — shared `uv.lock` (x86 collaborators)

Don't mutate the lock. Override the installed wheel instead:

```bash
"$UV" sync --python 3.12
"$UV" pip install --python "$VENV/bin/python" \
    --torch-backend=auto --upgrade torch torchvision
```

Lock keeps `torch==X.Y` from PyPI; venv shows `X.Y+cu1XX`. By design.

### 3a. uv subcommand gotchas

- `uv sync --torch-backend=auto` — flag does **not** exist on `sync` (only
  on `uv pip install` / `uv add`).
- `UV_TORCH_BACKEND=auto uv sync` — ignored when `uv.lock` exists (sync is
  lock-driven).
- `uv pip install` ignores `UV_PROJECT_ENVIRONMENT` and defaults to
  `./.venv`. **Always pass `--python <aarch64 venv>/bin/python`** or it
  may install into a stale x86 venv → `Exec format error`.
- `uv run` auto-syncs and silently undoes any Fix B override. Use
  `uv run --no-sync` or call `"$VENV/bin/python"` directly when the
  override matters (see §5).

---

## 4. Video stack: torchcodec + FFmpeg on aarch64

Three back-to-back problems when your code decodes video.

### 4a. torchcodec aarch64 wheels exist only for ≥0.11

Package markers like `torchcodec<0.11; platform_machine != 'aarch64'` exclude
every available aarch64 wheel. **Bump the floor AND drop `aarch64` from the
exclusion list** (keep `arm64` for macOS and `armv7l`):

```
"torchcodec>=0.11.0,<0.12.0; ... platform_machine != 'arm64' and platform_machine != 'armv7l' ..."
```

### 4b. `torchvision.io.VideoReader` is gone

Torchvision CUDA aarch64 wheels ≥0.26 removed `VideoReader` in favor of
torchcodec. Any code calling `torchvision.io.VideoReader` (regardless of a
`video_backend="pyav"` flag) only works once torchcodec is working.

### 4c. No system FFmpeg on the ARM compute node

torchcodec dlopens `libav*.so.N` by unversioned name. PyAV's bundled ffmpeg
uses hash-suffixed filenames (`libavutil-2a27f4d8.so.59.39.100`), so the
dlopen fails with e.g. `libavutil.so.56: cannot open shared object file`.

### Fix: ship a BtbN prebuilt FFmpeg on `LD_LIBRARY_PATH`

```bash
FFMPEG_DIR="/work/$USER/.local/ffmpeg-linuxarm64"
if [ ! -d "$FFMPEG_DIR/lib" ]; then
    curl -fsSL \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linuxarm64-gpl-shared-7.1.tar.xz" \
      -o /tmp/ffmpeg.tar.xz
    mkdir -p "$FFMPEG_DIR"
    tar -xJf /tmp/ffmpeg.tar.xz -C "$FFMPEG_DIR" --strip-components=1
fi
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}"
```

The FFmpeg major version (`n7.1`) must match the ABI of one of
`libtorchcodec_core{4,5,6,7}.so` shipped in your torchcodec install.

---

## 5. Arch-level pre-flight check

Verify directly via the venv's python — `uv run` auto-syncs and may
reinstall the lock-pinned CPU wheel on top of a Fix B override:

```bash
uname -m                                    # must print aarch64
"$VENV/bin/python" -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
# expect: 2.X.Y+cu1XX  True
```

`False` → revisit §3 (CPU wheel installed, or `uv run`/`uv sync` clobbered
the override).

### Known harmless warnings

- **unsloth → vLLM CUDA mismatch.** `import unsloth` on a CUDA-13 node
  prints `vLLM was built for CUDA 12 but this system has CUDA 13.0`.
  Benign unless your code actually calls `vllm.LLM` — vLLM is a
  transitive import-time dep most QLoRA paths don't invoke.

---

## aarch64-specific sbatch additions

These are the lines the **aarch64 ISA** forces into your sbatch script for any
ARM compute partition (currently `gb200-*`), on top of whatever `slurm-submission`
already gave you:

```bash
# 1. Kill x86 env leakage from login shell (§2)
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE VIRTUAL_ENV

# 2. aarch64 FFmpeg for torchcodec (§4)
export LD_LIBRARY_PATH="/work/$USER/.local/ffmpeg-linuxarm64/lib:${LD_LIBRARY_PATH:-}"

# 3. Arch sanity check (§5) — fail fast before wasting GPU time
uname -m                       # expect: aarch64
```

Everything else in the sbatch script (partition flags, `--mem`, `--time`,
TMPDIR, tokens, utilization tracker) comes from `slurm-submission`, not here.

### If `sbatch` rejects: `Requested node configuration is not available`

`--cpus-per-gpu × --gpus-per-node` exceeds post-reservation capacity.
Re-invoke `cluster-info` for current per-node spec — do not trust cached
values (in `~/.claude/cluster_info.md` or earlier in this session); per-node
reservation overhead changes on cluster reconfig. Resubmit with reduced
`--cpus-per-gpu`.

---

## Known-good versions (as of 2026-05-01)

CUDA driver and torch wheel index are **dynamic** — detect on the node.

| Component   | Version                                       |
|-------------|-----------------------------------------------|
| uv          | 0.10.0 (aarch64)                              |
| CPython     | 3.12.12 (aarch64)                             |
| torch       | `X.Y+cu<NN>` from `nvidia-smi` major.minor    |
| torchvision | matched to torch from same index              |
| torchcodec  | 0.11.x                                        |
| FFmpeg      | n7.1 (BtbN linuxarm64)                        |
| CUDA driver | `nvidia-smi`; GB200 today: 13.0 → `cu130`     |
