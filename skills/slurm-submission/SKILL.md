---
name: slurm-submission
description: >
  Use when submitting SLURM jobs, writing sbatch scripts, choosing GPU count, or reviewing
  completed jobs on TWCC/NCHC clusters. Trigger on: sbatch, job submission, GPU allocation,
  DDP, multi-GPU, wall time, job template, seff, post-job review, HuggingFace token in batch
  job, wandb login in batch, HF 401 in job, non-interactive auth.
  Do NOT trigger for: partition queries or pricing (use cluster-info),
  or job failures/hangs (use slurm-debug).
  If the target partition has aarch64/ARM compute nodes (currently any gb200-*, and
  any future ARM partition), ALSO load arm64-pipeline for aarch64-specific setup.
---

# SLURM Job Submission Guide

**REQUIRED:** Use cluster-info skill first to ensure partition specs and QoS limits are cached
in memory. If no cached cluster info exists, invoke cluster-info before proceeding.

**If the user is targeting an ARM / aarch64 compute partition** — currently
any `gb200-*` (Grace CPU), and any future ARM partition the cluster adds:
the compute nodes are `aarch64` while the login node is `x86_64`, which
breaks most pre-installed environments. **Also load the `arm64-pipeline` skill**
for the aarch64-specific setup (uv, PyTorch CUDA wheel, torchcodec + FFmpeg).
Check `uname -m` on an interactive session of the target partition if you're
unsure whether it's ARM.

---

## GPU Allocation Philosophy

This is the most important principle for sustainable cluster access.

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

---

## SLURM Job Template

Fill in partition, GPU count, and wall time from cached cluster info. The template includes a
background utilization tracker that logs GPU stats throughout the job.

```bash
#!/bin/bash
#SBATCH --job-name=<descriptive-name>
#SBATCH --partition=<partition>          # from cached cluster info
#SBATCH --nodes=<N>
#SBATCH --gpus-per-node=<G>             # must be >= MinGPU from cluster info
#SBATCH --cpus-per-gpu=<C>             # match to your dataloader workers
#SBATCH --mem=<M>G                      # from cluster info memory field
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

## Non-interactive auth in batch jobs

Batch jobs have no TTY, so any interactive login flow (`hf auth login`,
`wandb login`, `gh auth login`, `gcloud auth login`, docker registry login, …)
**cannot run inside the job**. Agents cannot run these either — they
require a human to paste a token or complete a browser OAuth step.

**Agent workflow:**

1. **Before writing the sbatch script**, verify the credential file exists —
   do NOT read, print, or echo its contents (treat the file as a secret):
   ```bash
   test -s ~/.cache/huggingface/token && echo "HF token present" || echo "HF token missing"
   ```
   Use `test -s` (non-empty), `ls -l`, or `stat`. Never `cat`, `head`, `tr`
   the contents into output the user or logs will see.
2. **If the file is missing**, stop and ask the user to run the login command
   on the login node first, and only then re-run your task. Give them the
   exact command — e.g. for HuggingFace: `hf auth login`.
3. **Inside the sbatch script**, read the credential file and `export` the
   token as the env var the client library looks for. Verify with the tool's
   `whoami`-style command **before** loading your ML framework or allocating
   GPU memory — failing fast beats failing 10 minutes into training.
4. **Strip whitespace** when reading the credential file — a trailing `\n`
   silently makes most services return 401.

Example sbatch block (HuggingFace; same shape for W&B, GitHub CLI, gcloud,
docker, etc.):

```bash
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # different clients read different vars
hf auth whoami                              # fail-fast check before GPU work
```

**Don't put tokens in `.bashrc`** — they leak into every shell and are easy
to accidentally commit. Read from the credential file the official CLI
already manages for you.

---

## Review After Job Completion

Do not monitor the job while it runs. The utilization tracker collects everything needed.
Review after the job ends.

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
