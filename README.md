# NCHC Cluster Skills

Claude Code skills plugin for working with TWCC / NCHC (National Center for High-performance Computing) HPC clusters.

## What's Included

| Skill | Triggers on |
|-------|-------------|
| `nchc-cluster` | SLURM, sbatch, sinfo, GPU jobs, SU billing, TWCC, NCHC, DDP, multi-GPU, partitions, H200, GB200 |

## Installation

### Via marketplace

```bash
/plugin marketplace add whats2000/nchc-marketplace
/plugin install nchc-cluster-skills@nchc-marketplace
```

### Local development / testing

```bash
claude --plugin-dir /path/to/NCHC-cluster-skills
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
