---
name: slurm-debug
description: >
  Use when debugging SLURM job failures, hangs, crashes, or unexpected behavior on TWCC/NCHC HPC
  clusters. Trigger on: job hang, timeout, CUDA error, OOM, segfault, NCCL timeout, srun error,
  exit code, node drain, GPU utilization 0%, deadlock, job cancelled, ImportError in container,
  slow training, or any "my job isn't working" question.
  Do NOT trigger for: job submission, resource planning, pricing, or partition queries (use nchc-cluster).
---

# SLURM Job Debug Flow

**Core principle:** When a job fails or hangs, **do not guess**. Route by `sacct` State + ExitCode first, then follow the matching branch.

**REQUIRED:** Use nchc-cluster skill for partition specs, QoS limits, and known bad nodes data.

---

## Quick Reference

| Symptom | Likely cause | Section |
|---------|-------------|---------|
| Traceback in `.err` log | Code / container / path error | Application Error |
| `signal 9` (kill -9) | CPU/RAM OOM | OOM |
| `CUDA out of memory` in `.err` | GPU OOM | OOM |
| No output, no error, no progress | Hang / deadlock | Hang or Slow |
| `NODE_FAIL` state | Hardware / node issue | Bad Node |
| Job stays `PENDING` forever | QoS / resource mismatch | PENDING |

---

## Step 1: Determine Failure Category

Get the job's **State** and **ExitCode**:

```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList -P
```

| State | ExitCode | Go to |
|-------|----------|-------|
| FAILED | 1:0 or 2:0 | → Application Error |
| FAILED | 137:0 (signal 9) | → OOM |
| TIMEOUT / CANCELLED (signal 15) | | → Hang or Slow |
| OUT_OF_MEMORY | | → OOM (SLURM memory limit) |
| NODE_FAIL | | → Bad Node |
| PENDING (never starts) | | → PENDING |

---

## Application Error (exit 1-2)

Read `.err` log for the last traceback. Check in order:

1. **Code interface changed?** — check `git log` since last working run
2. **Path missing inside container?** — verify `--bind` / `--mount` covers all needed paths
3. **Architecture mismatch?** — ARM vs x86 (check node arch from nchc-cluster cached partition table)
4. **Python/library version mismatch?** — container image may differ from dev environment

---

## OOM (signal 9 or CUDA out of memory)

First distinguish **GPU OOM** (CUDA error in `.err`) vs **CPU/RAM OOM** (signal 9, no CUDA error).

**GPU OOM fixes:** reduce batch size, enable gradient checkpointing, use mixed precision, offload optimizer states.

**CPU/RAM OOM** — check these in order:
1. **HuggingFace model cache duplication** — multiple processes on same node each loading the same
   model independently. Fix: shared `HF_HOME`, pre-download, `TRANSFORMERS_OFFLINE=1`
2. **Result accumulation** — collecting all batch outputs in lists then concatenating.
   Multiplied by concurrent processes per node
3. **DataLoader `num_workers`** — each worker forks and may duplicate dataset memory

---

## Hang / Timeout (no error, no progress)

This is the hardest category. Follow in order:

### 1. Confirm it's actually stuck

Check `voluntary_ctxt_switches` in `/proc/<PID>/status` twice, 30 seconds apart.
- **Not changing** = deadlock → continue to step 2
- **Changing** = alive but slow → add logging to find bottleneck

### 2. If deadlock, check environment first — not code

- **`/tmp` usage on the node** — NCHC cluster has **no SLURM Prolog/Epilog cleanup**;
  `/tmp` garbage accumulates from all users indefinitely. Usage >50% causes silent
  deadlocks. Workaround: add `/tmp` cleanup of your own cache patterns at job start.
  If another user's garbage fills it, exclude the node.
- **`/dev/shm`** — DataLoader shared memory can fill up
- **`dmesg`** — check for hardware errors (ECC, GPU Xid)

### 3. If environment is clean, investigate code

- `fork()` after JAX/CUDA init → use `spawn` multiprocessing start method
- Multiple processes writing same file → filesystem contention
- Distributed training waiting for crashed rank → check **all** ranks' logs

**Rule of thumb:** Same code works on some nodes but hangs on others → it's the node, not the code.

---

## Bad Node

- Track which nodes fail across recent jobs — look for patterns in `sacct` NodeList
- `drained`/`down` in `sinfo` = known issue, already excluded by scheduler
- `mixed`/`idle` doesn't mean healthy — `/tmp` garbage, GPU errors can exist on SLURM-healthy nodes
- Check **known bad nodes** data via nchc-cluster skill
- When confirmed bad: add to `--exclude` and update known bad nodes record with date and symptom

---

## Job Stuck in PENDING

Check the `(Reason)` from `squeue -j <JOBID>`, then cross-reference against QoS limits
and partition constraints from nchc-cluster skill:

| Reason | Check |
|--------|-------|
| `QOSMinGRESNotSatisfied` | Requested GPUs below QoS MinGPU/Job — check cached QoS limits |
| `QOSMaxGRESPerUser` | Total GPUs across running jobs exceeds QoS MaxGPU/User limit |
| `QOSMaxJobsPerUserLimit` | Too many concurrent jobs |
| `Resources` | Request doesn't fit partition (GPUs, nodes, wall time) |

**Common mistakes:**
- Fewer GPUs than QoS minimum (e.g. requesting 1 GPU when MinGPU/Job is 8)
- `--time` exceeds partition `MaxWall`
- `--nodes` exceeds partition `MaxNodes`

---

## Before Resubmitting Checklist

- [ ] Root cause identified? (don't retry blindly)
- [ ] `/tmp` cleanup added to job script?
- [ ] `--exclude` updated for bad nodes?
- [ ] `--time` / `--mem` adjusted if limits were hit?
- [ ] Checkpointing in place so completed work isn't repeated?
- [ ] If node-specific issue, targeting different nodes?
