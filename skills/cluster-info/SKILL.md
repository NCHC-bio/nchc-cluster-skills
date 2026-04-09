---
name: cluster-info
description: >
  Use when needing TWCC/NCHC cluster specs, partition info, QoS limits, pricing, or architecture
  details. Trigger on: sinfo, scontrol, partition, QoS, MinGPU, GPU type, SU billing, NTD cost,
  TWCC pricing, cluster identification, ARM vs x86, GB200 compatibility.
  This is the shared data layer — slurm-submission and slurm-debug both depend on it.
---

# TWCC / NCHC Cluster Info

Queries, caches, and serves cluster specs for other skills. Partition layouts, QoS limits, and
pricing differ between cluster generations — so specs are cached in memory and refreshed when stale.

---

## Step 1: Identify Current Cluster — NEVER Guess

**Do not assume which cluster the user is on.** Clusters share similar setups but have different
partitions, GPUs, QoS limits, and node names. Guessing wrong leads to failed jobs or wrong advice,
especially when users transfer data between clusters.

**Always determine the cluster first:**

```bash
# Check which cluster this SSH session connects to
echo $SSH_CONNECTION | awk '{print $3}'
```

Cross-reference the IP against known TWCC login node IPs. If the IP is not recognized or
`SSH_CONNECTION` is empty, **ask the user** which cluster they are on.

**Never proceed to Step 2 without knowing the cluster.** The cached partition table is per-cluster —
using the wrong cluster's cache is worse than having no cache.

---

## Step 2: Get Cluster Partition Info (Memory-Cached)

Partition layouts are stable for weeks or months at a time. The partition table is stored in Claude's
memory and reused across sessions to avoid unnecessary re-querying.

### Decision logic — follow this every time

```
1. Check memory for a stored cluster info entry
   (look for an entry labelled "TWCC/NCHC cluster info").

   Found AND matches current cluster AND no staleness signals?
   → Use the cached table. Done.

   Not found, OR wrong cluster, OR staleness signal present?
   → Run the live query below, then save result to memory (replace old entry).
```

**Staleness signals — refresh the cache if any apply:**
- No cached table exists yet
- Cached entry is for a different cluster than current
- User mentions a different cluster (e.g. "we moved to Nano 5")
- User reports a partition name, wall time, or GPU type that differs from the cached table
- Cached entry lacks a query date, or user explicitly asks for a refresh

### Live query commands (only when cache is missing or stale)

Run **all four** commands. Each provides fields that the others do not.

```bash
# 1. Hardware layout: partition, CPUs, memory, GRES, max wall, node count, features (arch)
sinfo -o "%20P %8c %8m %30G %10l %6D %20f" --noheader

# 2. Partition detail: min/max nodes, default time, QoS name, preempt mode
#    NOTE: scontrol does NOT show MinTRES — that's only in sacctmgr (command 3).
#    Use the QoS field here to map each partition → its QoS → the QoS limits.
scontrol show partition

# 3. QoS limits: min/max GPUs per job, max jobs per user, max wall time
#    **This is the ONLY source for MinTRES (minimum GPU requirement).**
#    MinTRES column shows e.g. "gres/gpu=64" meaning min 64 GPUs per job.
#    Cross-reference: partition → QoS name (from command 2) → this table.
sacctmgr show qos format=Name,Priority,MinTRESPerJob,MaxTRESPerUser,MaxTRESPerJob,MaxWall,MaxJobsPerUser -P

# 4. Account association limits (user-specific)
sacctmgr show assoc where user=$USER \
  format=Account,Cluster,Partition,GrpTRES,MaxTRES,MaxJobs,GrpTRESRunMins -P
```

**How to map MinGPU to partitions:** Each partition has a QoS (from `scontrol show partition`).
Look up that QoS in the `sacctmgr show qos` output to get MinTRES. If MinTRES is empty, there
is no minimum GPU requirement. Example: partition `normal` → QoS `p_normal` → MinTRES `gres/gpu=64`
→ MinGPU = 64.

Key fields to extract per partition:

| Field | Source | Meaning |
|-------|--------|---------|
| Partition name | `%P` from sinfo | Queue name |
| GPU type + count | `%G` from sinfo | e.g. `gpu:h200:8` |
| **MinGPU/job** | `MinTRESPerJob` from **sacctmgr QoS** (mapped via partition→QoS) | **Minimum GPUs required to submit** |
| **MaxGPU/job** | `MaxTRESPerJob` from sacctmgr QoS | Maximum GPUs per job (empty = no limit) |
| **MaxGPU/user** | `MaxTRESPerUser` from sacctmgr QoS | Maximum total GPUs across all running jobs |
| Max wall time | `%l` from sinfo / `MaxTime` from scontrol | Hard job time limit |
| MaxJobs/user | sacctmgr QoS | Maximum concurrent jobs per user |
| Node count | `%D` from sinfo | Total nodes in partition |
| Memory/node | `%m` from sinfo | RAM per node (MB) |
| CPUs/node | `%c` from sinfo | CPU cores per node |
| Min/MaxNodes | scontrol | Node count constraints per job |
| Architecture | features col from sinfo | `aarch64` = ARM, else x86_64 |
| Preempt mode | scontrol | e.g. REQUEUE |

### Saving to memory after a live query

Save a **single consolidated table** that includes both hardware specs AND scheduling rules.
Replace any previous entry — one memory file per cluster.

> **TWCC/NCHC cluster info** (queried YYYY-MM-DD, cluster: \<name\>)
>
> | Partition | GPU | GPUs/node | MinGPU | MaxGPU/job | MaxGPU/user | CPUs/node | Mem/node | Max wall | MaxJobs/user | Nodes | Arch | Notes |
> |-----------|-----|-----------|--------|------------|-------------|-----------|----------|----------|--------------|-------|------|-------|
> | dev       | ... | ...       | —      | —          | 16          | ...       | ...      | 1:00:00  | —            | ...   | x86  | example row |
> | normal    | ... | ...       | 64     | —          | 64          | ...       | ...      | 12:00:00 | —            | ...   | x86  | MinGPU from QoS |
>
> `—` = no limit or not set. `...` = fill from query results.
>
> **Node ranges:**
> - \<GPU type\>: `<hostname-prefix>[range]`
>
> **User accounts:** (from sacctmgr assoc query)
>
> **Known bad nodes:** (maintain as failures are observed)
> | Node | Issue | Date Added | Expiry |
> |------|-------|------------|--------|
>
> Entries older than 7 days should be **re-verified** before continuing to exclude.

Include the query date so staleness can be assessed in future sessions.
Do not accumulate multiple versions — replace the old entry.

### Live status checks (always query fresh, never cache)

```bash
# Remaining SU budget
sacctmgr show assoc where user=$USER format=Account,GrpTRESRunMins -P

# Current queue
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.10l %.6D %R"
```

---

## Step 3: Fetch Current Pricing

**Do not quote hardcoded prices.** Pricing changes between cluster generations.
Fetch from the official NCHC pricing page before quoting any cost figures:

- **Primary source:** https://iservice.nchc.org.tw/nchc_service/nchc_service_qa.php?target=54
- **Backup (SPA — may not render):** https://www.twcc.ai/doc?page=price

If neither URL is fetchable, ask the user to confirm the current GPU-hour rate for their GPU type
and project category before quoting costs.

Billing formula (consistent across generations):
```
GPU-hours = execution_hours * gpus_requested
cost (NTD) = GPU-hours * rate_per_gpu_hour
```

---

## Architecture Notes

- **ARM nodes (GB200 / Grace-Hopper):** Standard x86 binaries will not run — this includes most conda
  packages, pre-built PyPI wheels, and compiled CLI tools. Test ARM compatibility in a short dev job
  before submitting longer workloads. Prefer ARM-native containers when available.
- **x86 nodes (H100/H200):** Standard Linux tooling works. Use these for all workflows unless
  specifically benchmarking GB200 performance.
