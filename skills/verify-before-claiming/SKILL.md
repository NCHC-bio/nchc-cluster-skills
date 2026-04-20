---
name: verify-before-claiming
description: Use when the user explicitly asks to verify, reproduce, benchmark, or compare behavior by running code — "can you verify this?", "run it and check", "compare these approaches", "does this actually work?", "prove it". Do NOT trigger for general claims made during code review or analysis.
---

# Verify Before Claiming

## When to Use

- User asks to verify, reproduce, benchmark, or compare a behavioral claim by execution.
- The comparison involves two or more code variants against the HEAD baseline.
- Output needs to be auditable later (someone else must be able to re-read the log).

**When NOT to use:**
- Routine end-of-task "does it compile / do tests pass" — use `superpowers:verification-before-completion`.
- Claims the user is not asking you to prove.
- Situations where you have no environment to run the code — hand off reproduction steps instead (see below).

## The Only Rule

**Run it. Capture the real output. Show it.**

Everything else — reading source code, logical reasoning, pattern matching, "it clearly does X" — is inference, not evidence. Inference is where hallucination lives.

- If you ran it: quote the actual output (file path + log line). That is your evidence.
- If you did not run it: say so. Do not reason toward a conclusion instead.

---

## The Verification Artifact

When you can run it, produce a **persistent artifact set**. Ephemeral runs (`python -c`, bash heredocs, one-off tool calls) do **not** count — they leave no audit trail, no re-runnable log, no drift detection.

Add `.verify_*/` to `.gitignore` — these directories contain run logs and scratch state, not source.

```
.verify_<slug>/
├── switch_<file>.py            # one switcher per subject; declares variants as literal substitutions
├── run_<file>_Nvariants.sh     # sequences variants; trap restores baseline on ANY exit
├── parse_<metric>.py           # extracts measured values only; no PASS/FAIL labels
└── logs/<run_id>/
    ├── stdout_<name1>.log      # one log per variant
    ├── stdout_<name2>.log
    └── stdout_<name3>.log
```

### Switcher skeleton — `switch_<file>.py`

Variants are whatever you need to compare. One must be the HEAD no-op (baseline). Name them anything meaningful — labels, versions, approach names. Add or remove variants freely; the structure stays the same.

```python
"""Switch <description> between variants.

    usage: python switch_<file>.py {<variant_name> ...}

    <name1>   <what it does>
    <name2>   HEAD (baseline — no modification)
    <name3>   <what it does>
    ...

Behavior:
    1. Resets the file to HEAD with `git checkout` — always starts from known state.
    2. For the baseline variant, stops — file is already HEAD.
    3. For all others, locates the EXACT HEAD block and replaces it.
       Aborts loudly if not found (detects upstream drift).
    4. Prints a unified diff for verification.
"""
import difflib
import subprocess
import sys
from pathlib import Path

REPO   = Path("/path/to/repo")
TARGET = REPO / "path/to/file.py"

# --- Canonical HEAD block — must appear verbatim in the file ---
HEAD_BLOCK = (
    '        # exact line 1 from HEAD\n'
    '        # exact line 2 from HEAD\n'
)

# --- Variants: None = HEAD baseline (no modification) ---
VARIANTS = {
    "<name1>": (
        '        # replacement line 1\n'
        '        # replacement line 2\n'
    ),
    "<name2>": None,   # HEAD baseline
    "<name3>": (
        '        # replacement line 1\n'
        '        # replacement line 2\n'
    ),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print(f"usage: {sys.argv[0]} {{{'|'.join(VARIANTS)}}}", file=sys.stderr)
        sys.exit(2)
    variant = sys.argv[1]

    # Step 1: reset to HEAD
    subprocess.check_call(
        ["git", "-C", str(REPO), "checkout", "HEAD", "--", str(TARGET.relative_to(REPO))]
    )

    original = TARGET.read_text()
    if HEAD_BLOCK not in original:
        sys.stderr.write(
            "FATAL: the expected HEAD block was not found in the file.\n"
            "Upstream has drifted. Not modifying the file.\n\n"
            f"Expected block:\n{HEAD_BLOCK}"
        )
        sys.exit(3)

    if VARIANTS[variant] is None:
        print(f"[switch] variant {variant!r} = HEAD, no modification applied")
        return

    modified = original.replace(HEAD_BLOCK, VARIANTS[variant], 1)
    if modified == original:
        sys.stderr.write("FATAL: replacement produced no change.\n")
        sys.exit(4)

    TARGET.write_text(modified)

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"{TARGET.name} (HEAD)",
        tofile=f"{TARGET.name} (variant {variant})",
        n=1,
    )
    print(f"[switch] applied variant {variant!r}. Diff:")
    sys.stdout.writelines(diff)


if __name__ == "__main__":
    main()
```

### Runner skeleton — `run_<file>_Nvariants.sh`

The runner must invoke the **real codebase entry point** — the same command a user or CI would run. Not a mock, not a reduced stub, not an isolated fragment. If the claim is about a CLI, run the CLI. If it is about a training loop, run the training loop (even for a few steps with a small dataset).

**For cluster submission, follow the `slurm-submission` skill** — use its SBATCH header template, GPU-count guidance, and the background utilization tracker. Do not hand-roll SBATCH headers here. For local runs, drop the SBATCH block and keep `set -euo pipefail`.

```bash
#!/bin/bash
# --- Cluster: SBATCH headers go here. Fill in from slurm-submission skill
#     (partition, --gpus-per-node, --cpus-per-gpu, --mem, --time from cached
#     cluster-info). Submit with: sbatch --account=<ACCT> run_<file>_Nvariants.sh
#     Never embed --account in the script. ---

set -euo pipefail

# --- Environment (adjust to your setup) ---
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
# export HF_HOME=/work/.../hf_cache
# export LD_LIBRARY_PATH=...

REPO=/path/to/repo
SCRATCH=$REPO/.verify_<slug>
RUN_DIR=$SCRATCH/logs/<slug>_${SLURM_JOB_ID:-$$}

mkdir -p "$SCRATCH/logs" "$RUN_DIR"

# Move SLURM logs into run dir (cluster only, harmless on local)
mv "$REPO/slurm-${SLURM_JOB_ID:-}.out" "$RUN_DIR/slurm.out" 2>/dev/null || true
mv "$REPO/slurm-${SLURM_JOB_ID:-}.err" "$RUN_DIR/slurm.err" 2>/dev/null || true

cd "$REPO"

# --- Always restore HEAD on ANY exit ---
trap 'echo "[trap] restoring HEAD"; git -C "$REPO" checkout HEAD -- path/to/file.py || true' EXIT

run_variant() {
    local variant=$1
    echo "============================================================"
    echo "VARIANT $variant"
    echo "============================================================"

    # Apply variant (prints diff)
    uv run python "$SCRATCH/switch_<file>.py" "$variant"

    # Sanity-check: print the active block from the file
    echo "--- active block ---"
    sed -n '<start>,<end>p' path/to/file.py
    echo "--- (run starts) ---"

    # Real entry point — same seed, config, env across ALL variants
    <real_command> \
        --seed=1000 \
        ... \
        2>&1 | tee "$RUN_DIR/stdout_${variant}.log"

    echo "--- completed variant $variant ---"
}

run_variant <name1>
run_variant <name2>
run_variant <name3>

echo "============================================================"
echo "PARSED COMPARISON"
echo "============================================================"
uv run python "$SCRATCH/parse_<metric>.py" "$RUN_DIR"

echo "ALL_DONE"
```

### Parser skeleton — `parse_<metric>.py`

```python
"""Extract measured values from each variant's stdout log and print a neutral table.

    usage: python parse_<metric>.py <run_dir>
"""
import re, sys
from pathlib import Path

RUN_DIR  = Path(sys.argv[1])
VARIANTS = ("<name1>", "<name2>", "<name3>")   # must match switcher + runner
PATTERN  = re.compile(r"<regex for the quantity you measured>")

def parse(path):
    return [m.groups() for line in path.read_text().splitlines()
            if (m := PATTERN.search(line))]

rows = {v: parse(RUN_DIR / f"stdout_{v}.log") for v in VARIANTS}
# Physical labels only: loss, exit_code, latency_ms
# NEVER: expected, correct, PASS, FAIL
print(rows)
```

### Non-negotiable invariants

| Invariant | Rule |
|---|---|
| **Reversibility** | Any exit (success, crash, signal) restores the subject to its original state. |
| **Verbatim substitution** | Each variant is a literal exact-string replacement — no regex, no `sed`. Readable side-by-side. |
| **Drift detection** | If the subject changed since the artifact was written, fail loudly with a clear message. Never silently patch the wrong region. |
| **Input invariance** | Seed, config, env, fixtures — identical across all variants. The only difference is the one substitution under test. |
| **Baseline is a variant** | One variant (B) is an identity no-op. Running it must leave the subject unchanged. |
| **Audit trail before execution** | Print the diff/substitution before any real work runs. |
| **Output isolation** | Each variant's output lands in its own log. No shared mutable state between variants. |
| **Real-path execution** | Run through the same path the actual user/CI hits. Mocked internals, isolated fragments, and out-of-context stubs do not count. |
| **Neutral extraction** | Parser reports physical values only (`exit_code`, `latency_ms`). No `PASS`/`FAIL`, no `expected`. |
| **Outcome-independent report** | Swap the numbers and no sentence in the report becomes wrong. If swapping makes it false, you encoded prediction, not observation. |

---

## What Does NOT Count as Running It

- `grep` / regex / AST search of source code
- `assert pattern in source` scripts — static text-matching with a test wrapper is still static text-matching
- "I read the code carefully and it clearly does X"
- `python -c`, bash heredocs, inline one-shot tool calls — run code but leave no artifact
- "The tests would catch this" — without running the tests
- "CI is green" — CI proves only what CI exercises
- A subagent reading the source — same static analysis, same problem

---

## When You Cannot Run It

**If you can run it yourself — run it. Do not ask the user first.**

If you genuinely cannot (no environment, no credentials, no hardware):

1. Say you could not run it. Do not reason toward a conclusion instead.
2. Give the user **concrete, runnable steps** to verify themselves — not "investigate further."

> *"I wasn't able to run this. To verify: [exact command or steps]."*

| ❌ Never | ✅ Instead |
|---|---|
| "This silently loses data." (not run) | "I wasn't able to run this. To verify: trigger a save failure and check stdout for warnings." |
| "Removing this will break CI." (not run) | "I wasn't able to run CI. To verify: remove the import and run `pytest tests/`." |
| "Approach A is faster." (not benchmarked) | "I haven't benchmarked this. To verify: run both with `time` under your real workload." |

---

## Common Mistakes

| Mistake | Why it fails | Fix |
|---|---|---|
| `python -c "..."` or bash heredoc | No artifact, no audit trail, no re-run | Write switcher + runner + logs into `.verify_<slug>/` |
| `grep`, `assert pattern in source`, AST inspection | Static text-matching is not execution | Run the real entry point |
| Different seed/config per variant | Output differences are confounded | Identity inputs across variants — only substitution differs |
| Skipping the baseline variant | No control, cannot detect environment drift | Always include HEAD no-op as a variant |
| Parser labels `PASS/FAIL` / `expected` | Encodes prediction, not observation | Emit physical values only; conclusions go in prose |
| Writing conclusions before logs exist | Outcome-dependent reporting | Run first, read log, then write |

## Red Flags — Stop and Restart

- "I read the code carefully and it clearly does X"
- "The tests would catch this" (without running them)
- "CI is green" (CI only proves what CI exercises)
- A subagent read the source and reported back — same static analysis
- "I'll just confirm with `python -c`" — ephemeral, no audit trail
- Writing the summary paragraph before any log has been read

**All of these mean: you have not run it. Produce the artifact, or explicitly hand off.**

---

## Iron Rule

A claim counts as verified only if **all three** hold:

1. The condition was **triggered** — not just described.
2. The output was **captured** in a persistent per-variant log.
3. The log is **cited** (path + line) so someone else can check it.

If any step is missing: say you could not verify it and hand off the steps to reproduce.
