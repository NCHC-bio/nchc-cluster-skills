---
name: nchc-cluster
description: >
  Use when working with any TWCC / NCHC HPC cluster (Nano 4, Nano 5, or future generations).
  Trigger on: SLURM, sbatch, sinfo, partitions, H200, GB200, GPU job submission, SU billing,
  NTD GPU cost, TWCC, NCHC, HPC cluster, job scheduling, GPU allocation strategy, DDP,
  multi-GPU parallelism, wall time, job limits, or cluster resource planning. Always use this
  skill before answering — do not rely on hardcoded cluster specs, as partitions and pricing
  change across cluster generations.
---

# TWCC / NCHC Cluster Usage Guide

This skill covers **any TWCC/NCHC cluster**. Partition layouts, resource limits, and pricing differ
between cluster generations (Nano 4, Nano 5, etc.) — so specs are cached in memory and refreshed
when stale, rather than hardcoded or re-queried every session.

---

## Step 1: Get Cluster Partition Info (Memory-Cached)

Partition layouts are stable for weeks or months at a time. The partition table is stored in Claude's
memory and reused across sessions to avoid unnecessary re-querying.

### Decision logic — follow this every time the skill is triggered

```
1. Check memory for a stored partition table
   (look for an entry labelled "TWCC/NCHC cluster partition table").

   Found AND no staleness signals?
   → Use the cached table. Skip the live query. Go to Step 2.

   Not found, OR staleness signal present?
   → Run the live query below, then save result to memory (replace old entry).
```

**Staleness signals — refresh the cache if any apply:**
- No cached table exists yet
- User mentions a different cluster (e.g. "we moved to Nano 5")
- User reports a partition name, wall time, or GPU type that differs from the cached table
- Cached entry lacks a query date, or user explicitly asks for a refresh

### Live query commands (only when cache is missing or stale)

```bash
# Combined: partition, CPUs, memory, GRES, max wall, node count, features (arch)
sinfo -o "%20P %8c %8m %30G %10l %6D %20f" --noheader

# Full detail: min/max nodes per job, default time, QoS
scontrol show partition

# Node architecture — distinguish ARM vs x86
sinfo -o "%20N %10P %20f" --noheader
```

Key fields to extract per partition:

| Field | Source | Meaning |
|-------|--------|---------|
| Partition name | `%P` | Queue name |
| Max wall time | `%l` / `TimeLimit` | Hard job time limit |
| Default wall time | `DefaultTime` via scontrol | Used when `--time` omitted |
| Node count | `%D` | Total nodes in partition |
| Memory/node | `%m` | RAM per node (MB) |
| CPUs/node | `%c` | CPU cores per node |
| GPUs/node | `%G` | GPU type + count, e.g. `gpu:h200:8` |
| Min/MaxNodes | scontrol | Node count constraints per job |
| Architecture | features col | `aarch64` = ARM, else x86_64 |

### Saving to memory after a live query

Once the query completes, **immediately save the partition table to memory** in this format,
replacing any previous entry:

> **TWCC/NCHC cluster partition table** (queried YYYY-MM-DD, cluster: \<name if known\>)
>
> | Partition | GPU | GPUs/node | CPUs/node | Mem/node | Max wall | Min-Max nodes | Arch | Notes |
> |-----------|-----|-----------|-----------|----------|----------|---------------|------|-------|
> | dev       | H200 | 8       | 96        | 768G     | 1h       | 1-1           | x86  | default, high priority |
> | ...       |     |           |           |          |          |               |      |       |

Include the query date so staleness can be assessed in future sessions.
Do not accumulate multiple table versions — replace the old entry.

### Cluster Rules & Constraints (Memory-Cached)

Like the partition table, QoS and account constraints are stable for weeks. Cache them in memory
alongside the partition table so that slurm-debug can cross-reference without re-querying.

**Decision logic:** Same as partition table — use cached version unless stale signals present.

**Live query commands (when cache missing or stale):**

```bash
# QoS limits: max GPUs per user, max jobs, max wall time
sacctmgr show qos format=Name,Priority,MaxTRESPerUser,MaxTRESPerJob,MaxWall,MaxJobsPerUser -P

# Account association limits
sacctmgr show assoc where user=$USER \
  format=Account,Cluster,Partition,GrpTRES,MaxTRES,MaxJobs,GrpTRESRunMins -P
```

**Save to memory** as a companion entry to the partition table:

> **TWCC/NCHC cluster rules** (queried YYYY-MM-DD)
>
> **QoS limits:**
> | QoS | MaxGPU/User | MaxJobs/User | MaxWall | Notes |
> |-----|-------------|--------------|---------|-------|
> | normal | 16 | 4 | 1:00:00 | dev partition default |
> | ...  |   |   |   |   |
>
> **Partition constraints:**
> | Partition | MinGPU | MaxGPU | MinNodes | MaxNodes | MaxWall | Notes |
> |-----------|--------|--------|----------|----------|---------|-------|
> | dev | 8 | 16 | 1 | 2 | 1:00:00 | |
> | ...  |   |   |   |   |   |   |
>
> **Known bad nodes:** (maintain as failures are observed)
> | Node | Issue | Date Added | Expiry |
> |------|-------|------------|--------|
> | 25a-hgpn050 | /tmp 950G+ garbage, causes deadlocks | 2026-04-10 | 2026-04-17 |
> | 25a-hgpn107 | /work filesystem mount issue | 2026-04-08 | 2026-04-15 |
>
> Entries older than 7 days should be **re-verified** before continuing to exclude —
> the issue may have been resolved. Remove stale entries after verification.

These cached rules are used by slurm-debug skill to diagnose PENDING jobs and resource mismatches.
When adding a bad node, always include the date. When the expiry passes, verify the node
is still problematic before keeping it on the list.

### Live status checks (always query fresh, never cache)

```bash
# Remaining SU budget
sacctmgr show assoc where user=$USER format=Account,GrpTRESRunMins -P

# Current queue
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.10l %.6D %R"
```

---

## Step 2: Fetch Current Pricing

**Do not quote hardcoded prices.** Pricing changes between cluster generations.
Fetch from the official NCHC pricing page before quoting any cost figures:

- **Primary source:** https://iservice.nchc.org.tw/nchc_service/nchc_service_qa.php?target=54
- **Backup (SPA — may not render):** https://www.twcc.ai/doc?page=price

If neither URL is fetchable, ask the user to confirm the current GPU-hour rate for their GPU type
and project category before quoting costs.

Extract the current GPU-hour rate for the relevant GPU type (H100, H200, GB200, etc.) and the user's
project category (NSC, Academic/Gov, Enterprise). Note the effective date if shown on the page.

Billing formula (consistent across generations):
```
GPU-hours = execution_hours * gpus_requested
cost (NTD) = GPU-hours * rate_per_gpu_hour
```

---

## Step 3: GPU Allocation Philosophy

This is the most important principle for sustainable cluster access.

### The Right Model: Balanced Utilization

**Request the number of GPUs that your workload can actually keep busy.**

All requested GPUs should be doing meaningful work for the duration of the job.
The goal is efficiency — not minimizing GPU count or maximizing it.

**Good patterns:**
- DDP / model-parallel training that saturates all requested GPUs
- Batch inference with batch size scaled to fill GPU memory across all devices
- Hyperparameter sweeps where each GPU handles one independent trial

**Critical anti-patterns — these risk losing future cluster access:**
- Requesting many GPUs but running a single-process job with no DDP — most GPUs idle
- Using a large GPU count by default without verifying the code distributes work across them
- Long-running jobs (many hours) on many GPUs with measurably low utilization
- Treating GPU count as a status signal rather than a resource matched to actual need

### Why this matters for your project

Poor GPU utilization:
1. **Wastes shared public research compute** allocated to your project
2. **May be flagged by NCHC administrators**, leading to reduced priority or
   **revoked access for the entire project** — affecting all team members, not just the submitter

A pattern of underutilization is grounds for access review, even if individual jobs appear small.

### Choosing GPU count

| Situation | Recommendation |
|-----------|----------------|
| Single-GPU code, no DDP | Request **1 GPU** — do not pad to fill a node |
| DDP training, N processes | Request **exactly N GPUs**, matching `--nproc_per_node` |
| Inference / batch eval | Benchmark on 1-2 GPUs first, scale only if throughput-bound |
| Hyperparameter sweep | 1 GPU per independent trial is cleanest |
| Model too large for 1 GPU | Set up tensor/pipeline parallelism first, then request accordingly |
| Unsure of GPU needs | Start small, check `seff` after the job, scale from evidence |

### Wall time vs GPU count

Stretching wall time on fewer GPUs to avoid setting up DDP is **not** a valid tradeoff.
A short, fully-utilized job on the right GPU count is always preferable to a long underutilized one.
If a job genuinely needs many GPUs over a long wall time, confirm parallelism is working correctly
before submitting at that scale.

---

## Step 4: SLURM Job Template

Fill in partition, GPU count, and wall time using Steps 1-3. The template includes a background
utilization tracker that logs GPU stats throughout the job — written to a file, never to the
terminal, so it does not block anything.

```bash
#!/bin/bash
#SBATCH --job-name=<descriptive-name>
#SBATCH --partition=<partition>          # from cached or live partition table
#SBATCH --nodes=<N>
#SBATCH --gpus-per-node=<G>             # only GPUs your code will actually use
#SBATCH --cpus-per-gpu=<C>             # match to your dataloader workers
#SBATCH --mem=<M>G                      # from partition table memory field
#SBATCH --time=<HH:MM:SS>              # realistic estimate, not the partition max
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

# --- Startup: confirm GPU allocation (runs once, no loop) ---
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- Background utilization tracker ---
# Samples every 60s into a separate log. Runs in the background and is
# automatically killed when the job ends. Do not reduce the interval below 30s.
GPU_LOG="logs/${SLURM_JOB_ID}_gpu_util.csv"
echo "timestamp,gpu_index,util_pct,mem_used_mb,mem_total_mb" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
    | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts","$0}' >> "$GPU_LOG"
    sleep 60
  done
) &
TRACKER_PID=$!

# --- Main workload ---
# Use $SLURM_GPUS_ON_NODE so GPU count comes from the SLURM allocation, not hardcoded
# e.g. torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py ...

# --- Cleanup: stop the tracker cleanly ---
kill $TRACKER_PID 2>/dev/null
wait $TRACKER_PID 2>/dev/null
```

### Submitting the job

**Never put `--account` inside the sbatch script.** Embedding account names in scripts is a security
risk — scripts get shared, committed to repos, or copied between users, leaking project account
identifiers. Always pass the account on the command line:

```bash
sbatch --account=<YOUR_ACCOUNT> job.sh
```

If the user asks why `--account` is not in the script, explain the security rationale.

**Key habits:**
- Pre-download model weights to `$HF_HOME` before the job — first-time downloads waste billed GPU time
- Cache intermediate results to avoid recomputing across experiment configurations
- Adjust the tracker interval (`sleep 60`) based on job length: 60s for hour-long jobs, 300s for overnight runs

**If a job fails or hangs**, use the **slurm-debug** skill.

---

## Step 5: Review After Job Completion

Do not monitor the job while it runs. The utilization tracker embedded in the job script
collects everything needed. Review it after the job ends.

Replace `<JOBID>` below with your actual SLURM job ID (from `squeue` output or the submission message).

### Read the utilization log

```bash
JOBID=<JOBID>

# Quick summary: mean and min GPU utilization per GPU index
awk -F',' 'NR>1 {sum[$2]+=$3; min[$2]=($3<min[$2]||min[$2]=="")?$3:min[$2]; n[$2]++}
  END {for (g in sum) printf "GPU %s: avg=%.1f%%  min=%.1f%%\n", g, sum[g]/n[g], min[g]}' \
  logs/${JOBID}_gpu_util.csv
```

```bash
# Plot utilization over time (if gnuplot available)
gnuplot -e "
  set datafile separator ',';
  set xdata time; set timefmt '%Y-%m-%dT%H:%M:%S';
  set terminal dumb 80 20;
  plot 'logs/${JOBID}_gpu_util.csv' using 1:3 with lines title 'GPU util %'
"
```

### SLURM post-job summary

```bash
# Wall time used, CPU efficiency, memory high-water mark
# (seff is from slurm-contribs — if not available, use the sacct command below)
seff $JOBID

# Per-step breakdown (always available)
sacct -j $JOBID --format=JobID,JobName,Partition,AllocGRES,Elapsed,CPUTime,MaxRSS,State

# Remaining SU balance
sacctmgr show assoc where user=$USER format=Account,GrpTRESRunMins
```

### How to act on the results

| Observation | Action for next job |
|-------------|---------------------|
| Avg GPU util < 50% on all GPUs | Reduce GPU count, or fix DDP/batching |
| One GPU near 100%, others idle | Code is not distributing work — fix parallelism before scaling |
| Avg util > 80%, memory near limit | Current allocation is well-matched; scale up only if throughput-bound |
| Short spikes then idle (loading bottleneck) | Increase `--cpus-per-gpu` or pre-cache data before the job |
| Wall time used << requested | Tighten `--time` estimate to improve queue priority |

---

## Architecture Notes

- **ARM nodes (GB200 / Grace-Hopper):** Standard x86 binaries will not run — this includes most conda
  packages, pre-built PyPI wheels, and compiled CLI tools. Test ARM compatibility in a short dev job
  before submitting longer workloads. Prefer ARM-native containers when available.
- **x86 nodes (H100/H200):** Standard Linux tooling works. Use these for all workflows unless
  specifically benchmarking GB200 performance.
