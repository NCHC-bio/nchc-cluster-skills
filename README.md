# NCHC Cluster Skills

Claude Code skills plugin for working with TWCC / NCHC (National Center for High-performance Computing) HPC clusters.

## What's Included

| Skill | Triggers on |
|-------|-------------|
| `nchc-cluster` | SLURM, sbatch, sinfo, GPU jobs, SU billing, TWCC, NCHC, DDP, multi-GPU, partitions, H200, GB200 |
| `slurm-debug` | Job hang, timeout, CUDA error, OOM, segfault, NCCL timeout, exit code, node drain, GPU utilization 0%, deadlock, job cancelled, slow training |

## Installation

### Via marketplace

Open Claude Code by running `claude` in your terminal, then run the following commands:

```bash
/plugin marketplace add whats2000/nchc-marketplace
/plugin install nchc-cluster-skills@nchc-marketplace
```

### Local development / testing

```bash
claude --plugin-dir /path/to/nchc-cluster-skills
```

### Reload after changes

```bash
/reload-plugins
```

## Skills

### nchc-cluster

Guides Claude through NCHC cluster usage with:

- **Partition info caching** -- queries `sinfo`/`scontrol` once, stores in memory, refreshes when stale
- **Live pricing** -- fetches current GPU-hour rates from TWCC docs instead of hardcoding
- **GPU allocation philosophy** -- request only what your code actually uses; anti-patterns that risk access revocation
- **SLURM job template** -- with built-in GPU utilization tracker
- **Post-job review** -- `seff`, utilization log analysis, actionable next-step table
- **Architecture awareness** -- ARM (GB200) vs x86 (H100/H200) compatibility notes

### slurm-debug

Guides Claude through a structured debug flow when SLURM jobs fail or hang:

- **Failure routing** -- classifies by `sacct` State + ExitCode, then follows the matching branch
- **OOM diagnosis** -- distinguishes GPU OOM (CUDA) vs CPU/RAM OOM (signal 9) with targeted fixes
- **Hang / deadlock triage** -- environment checks (`/tmp`, `/dev/shm`, `dmesg`) before code investigation
- **Bad node detection** -- tracks failing nodes across jobs, manages `--exclude` lists
- **PENDING diagnosis** -- maps `squeue` reasons to QoS limits and partition constraints
- **Resubmit checklist** -- prevents blind retries by verifying root cause, cleanup, and checkpointing
