# NCHC Cluster Skills

Claude Code skills plugin for working with TWCC / NCHC (National Center for High-performance Computing) HPC clusters.

## What's Included

| Skill | Triggers on |
|-------|-------------|
| `cluster-info` | sinfo, scontrol, partition specs, QoS limits, MinGPU, SU billing, pricing, ARM vs x86, cluster identification |
| `slurm-submission` | sbatch, job submission, GPU allocation, DDP, multi-GPU, wall time, job template, post-job review |
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

### cluster-info

Shared data layer for cluster specs, used by both slurm-submission and slurm-debug:

- **Cluster identification** -- determines current cluster before doing anything
- **Partition info caching** -- queries `sinfo`/`scontrol`/`sacctmgr` once, stores in memory, refreshes when stale
- **QoS limits** -- MinGPU, MaxGPU per user/job, MaxJobs — mapped from partition → QoS
- **Live pricing** -- fetches current GPU-hour rates from TWCC docs instead of hardcoding
- **Architecture awareness** -- ARM (GB200) vs x86 (H100/H200) compatibility notes

### slurm-submission

Guides Claude through job submission (depends on cluster-info for cached specs):

- **GPU allocation philosophy** -- request only what your code actually uses; anti-patterns that risk access revocation
- **SLURM job template** -- with built-in GPU utilization tracker
- **Post-job review** -- `seff`, utilization log analysis, actionable next-step table

### slurm-debug

Guides Claude through a structured debug flow when SLURM jobs fail or hang:

- **Failure routing** -- classifies by `sacct` State + ExitCode, then follows the matching branch
- **OOM diagnosis** -- distinguishes GPU OOM (CUDA) vs CPU/RAM OOM (signal 9) with targeted fixes
- **Hang / deadlock triage** -- environment checks (`/tmp`, `/dev/shm`, `dmesg`) before code investigation
- **Bad node detection** -- tracks failing nodes across jobs, manages `--exclude` lists
- **PENDING diagnosis** -- maps `squeue` reasons to QoS limits and partition constraints
- **Resubmit checklist** -- prevents blind retries by verifying root cause, cleanup, and checkpointing
