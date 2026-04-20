---
name: verify-before-claiming
description: Use when the user asks to prove a bug claim by running code — "can you verify this?", "is that actually a bug?", "prove it", "run it and check". Agents often hallucinate bugs; this skill forces the claim through a falsification test — propose a patch, run HEAD and patch against the real entry point, compare outputs. No behavior change = claim invalidated. Do NOT trigger for general discussion or analysis where no bug claim is being tested.
---

# Verify Before Claiming

## When to Use

- User asks to prove a bug claim, or you are about to assert one.
- The claim can be paired with a concrete patch — if you cannot describe the fix, you do not yet have a testable claim.
- Output needs to be auditable later (someone else must be able to re-read and reproduce the log).

**When NOT to use:**
- Routine end-of-task "does it compile / do tests pass" — use `superpowers:verification-before-completion`.
- Claims the user is not asking you to prove and that you are not asserting.
- Situations where you have no environment to run the code — hand off reproduction steps instead (see below).

## The Rule

**One issue per artifact set.** Never batch multiple claims into a single switcher, runner, or run. Each claim gets its own `switch_<subject>.py` and `run_<subject>_Nvariants.*` so any output difference can only be attributed to that one claim. A parser may be shared across subjects that emit the same log format.

For every bug claim:

1. **Define the variants.** Minimum is `HEAD` (claimed-buggy, control) + `patch` (proposed fix). Add more only when the claim specifically needs them — e.g. a pre-patch variant to show HEAD introduced a regression, or multiple candidate patches to compare.
2. **Run every variant against the real entry point with identical input** — same seed, same config, same env, same fixtures. The only change between runs is the one substitution from the switcher. Each variant's stdout goes to its own log.
3. **Parse the logs mechanically, then read the parser's table.** Do not read raw stdout and eyeball the numbers — that is where LLM hallucination lives. The parser extracts physical values (loss, grad norm, exit code, latency) via regex; the agent only reports what the table says.
4. **Apply the verdict from the parsed table:**

| Parsed observation | Verdict | What to do |
|---|---|---|
| Variants differ on the measured quantity | **Bug confirmed, patch addresses it** | Quote the parser's rows for both variants (path + line) as evidence. |
| Variants are identical on the measured quantity | **Claim invalidated** | The bug is not in that code path, the patch misses the root cause, or the path you blamed is not the one that runs. Retract. Do not ship the patch. |
| Parser found no records in one or more logs | **Unknown** | The run crashed or emitted a different format before the metric. Fix the runner or the regex; do not guess from partial output. |

Inference — reading source, pattern matching, "it clearly does X", eyeballing raw logs — does not substitute for a parsed comparison. If you did not run it, say so and hand off the reproduction steps. Do not reason toward a conclusion instead.

A claim counts as verified only if **all three** hold: the condition was **triggered** (not just described), every variant's output was **captured** in its own log, and the parser's row for each variant is **cited** (path + line) so someone else can check it. If any step is missing, the claim is not verified.

---

## The Verification Artifact

When you can run it, produce a **persistent artifact set**. Ephemeral runs (`python -c`, bash heredocs, one-off tool calls) do **not** count — they leave no audit trail, no re-runnable log, no drift detection.

Add `.verify_*/` to `.gitignore` — these directories contain run logs and scratch state, not source.

One `.verify_<task_slug>/` per task (e.g. one per code review). Inside it, **one switcher and one runner per subject**, plus a parser that may be shared across subjects when they emit the same log format. Logs are grouped per-run; variants within one run live side-by-side under that run's directory.

```
.verify_<task_slug>/
├── switch_<subject_a>.py              # one switcher per subject
├── switch_<subject_b>.py
├── run_<subject_a>_Nvariants.sbatch   # one runner per subject; trap restores HEAD on ANY exit
├── run_<subject_b>_Nvariants.sbatch
├── parse_<metric>.py                  # mechanical extractor — may be shared if log format matches
└── logs/
    ├── <subject_a>_<run_id>/          # one dir per run
    │   ├── stdout_HEAD.log            # one log per variant (minimum: HEAD + patch)
    │   ├── stdout_patch.log
    │   ├── slurm.out                  # cluster stdout (if applicable)
    │   └── slurm.err
    └── <subject_b>_<run_id>/
        ├── stdout_HEAD.log
        └── stdout_patch.log
```

**Do not simplify or merge the artifact set.** Do not:

- Share one switcher across multiple subjects — substitutions collide and drift detection breaks.
- Share one runner — variant logs from different subjects end up in the same place and you lose attribution.
- Skip the runner and call the entry point "by hand" — no re-runnable log, no trap to restore HEAD.
- Skip the parser and eyeball the logs — extracting numbers by reading stdout is where hallucination enters.
- Dump multiple subjects' logs into the same `logs/<run_id>/` directory — you lose which variant produced which output.

If it feels like too many files for a small claim: that is the correct amount. The artifact is what makes the claim reproducible by someone who is not you.

### Switcher skeleton — `switch_<subject>.py`

Default is two variants: `HEAD` (no-op, control) and `patch` (proposed fix). Only add more when the claim needs them — then short labels like `A` / `B` / `C` are fine. Exactly one variant must be the HEAD no-op.

```python
"""Switch <subject description> between variants.

    usage: python switch_<subject>.py {<variant> ...}

    HEAD    claimed-buggy, no modification (control)
    patch   proposed fix applied as a verbatim substitution
    <more>  additional candidate patches or prior states, as needed

Behavior:
    1. Resets the file to HEAD with `git checkout` — always starts from a known state.
    2. For the HEAD variant, stops — file is already at HEAD.
    3. For any other variant, locates the EXACT HEAD block and replaces it verbatim.
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

# --- Variants: None = HEAD (control); strings = verbatim replacements ---
VARIANTS = {
    "HEAD":  None,
    "patch": (
        '        # proposed fix line 1\n'
        '        # proposed fix line 2\n'
    ),
    # Add a third variant if the claim needs a pre-patch reference, e.g.:
    # "prepatch": (
    #     '        # original line 1 before HEAD introduced the regression\n'
    #     '        # original line 2\n'
    # ),
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

### Runner skeleton — `run_<issue>.sh`

The runner must invoke the **real codebase entry point** — the same command a user or CI would run. Not a mock, not a reduced stub, not an isolated fragment. If the claim is about a CLI, run the CLI. If it is about a training loop, run the training loop (even for a few steps with a small dataset).

**For cluster submission, follow the `slurm-submission` skill** — use its SBATCH header template, GPU-count guidance, and the background utilization tracker. Do not hand-roll SBATCH headers here. For local runs, drop the SBATCH block and keep `set -euo pipefail`.

```bash
#!/bin/bash
# --- Cluster: SBATCH headers go here. Fill in from slurm-submission skill
#     (partition, --gpus-per-node, --cpus-per-gpu, --mem, --time from cached
#     cluster-info). Submit with: sbatch --account=<ACCT> run_<issue>.sh
#     Never embed --account in the script. ---

set -euo pipefail

# --- Environment (adjust to your setup) ---
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
# export HF_HOME=/work/.../hf_cache
# export LD_LIBRARY_PATH=...

REPO=/path/to/repo
SUBJECT=<subject>                 # matches switch_<subject>.py and run_<subject>_Nvariants.sh
TASK_SLUG=<task_slug>             # shared across subjects in the same task
SCRATCH=$REPO/.verify_${TASK_SLUG}
RUN_DIR=$SCRATCH/logs/${SUBJECT}_${SLURM_JOB_ID:-$$}

mkdir -p "$RUN_DIR"

# Move SLURM logs into run dir (cluster only, harmless on local)
mv "$REPO/slurm-${SLURM_JOB_ID:-}.out" "$RUN_DIR/slurm.out" 2>/dev/null || true
mv "$REPO/slurm-${SLURM_JOB_ID:-}.err" "$RUN_DIR/slurm.err" 2>/dev/null || true

cd "$REPO"

# --- Always restore HEAD on ANY exit ---
trap 'echo "[trap] restoring HEAD"; git -C "$REPO" checkout HEAD -- path/to/file.py || true' EXIT

run_variant() {
    local variant=$1
    echo "============================================================"
    echo "SUBJECT=$SUBJECT  VARIANT=$variant"
    echo "============================================================"

    # Apply variant (prints diff)
    uv run python "$SCRATCH/switch_${SUBJECT}.py" "$variant"

    # Sanity-check: print the active block from the file
    echo "--- active block ---"
    sed -n '<start>,<end>p' path/to/file.py
    echo "--- (run starts) ---"

    # Real entry point — identical seed, config, env for every variant
    <real_command> \
        --seed=1000 \
        ... \
        2>&1 | tee "$RUN_DIR/stdout_${variant}.log"

    echo "--- completed variant $variant ---"
}

# Declare variants here; keep order stable. Must match keys in switch_<subject>.py.
for v in HEAD patch; do
    run_variant "$v"
done

echo "============================================================"
echo "PARSED COMPARISON"
echo "============================================================"
uv run python "$SCRATCH/parse_<metric>.py" "$RUN_DIR"

echo "ALL_DONE  logs at $RUN_DIR"
```

After the runner finishes, do **not** read the raw stdout logs to form the verdict. Read the parser's tabulated output and cite the rows. Applying the outcome table from **The Rule** against eyeballed log lines is the hallucination path the parser exists to close.

### Parser skeleton — `parse_<metric>.py`

The parser turns N raw stdout logs into a single neutral table. It reports only what the regex captured — no `PASS`/`FAIL`, no "expected", no verdict. The agent reads the table and maps it to The Rule's outcome table.

```python
"""Extract the measured quantity from each variant's stdout log and tabulate.

    usage: python parse_<metric>.py <run_dir>

    Reads every stdout_<variant>.log in <run_dir>, extracts the measured
    quantity per step (or per call), and prints one row per step with one
    column per variant. Prints raw physical values only — no labels,
    no thresholds, no PASS/FAIL.
"""
import re
import statistics
import sys
from pathlib import Path

RUN_DIR = Path(sys.argv[1])

# Regex matching one record from the entry point's stdout. Name every group
# you want to tabulate. Keep the pattern strict — if it drifts, the parser
# will correctly report zero records rather than silently extract the wrong thing.
PATTERN = re.compile(
    r"step:(?P<step>\d+)\s+.*?\s+loss:(?P<loss>[\d.]+)\s+grdn:(?P<grdn>[\d.]+)"
)

def parse(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if m := PATTERN.search(line):
            rows.append((int(m["step"]), float(m["loss"]), float(m["grdn"])))
    return rows

def main():
    # Discover variants from whatever stdout_*.log files are present
    variants = sorted(p.stem.removeprefix("stdout_")
                      for p in RUN_DIR.glob("stdout_*.log"))
    if not variants:
        raise SystemExit(f"no stdout_*.log files in {RUN_DIR}")

    data = {v: {s: (l, g) for s, l, g in parse(RUN_DIR / f"stdout_{v}.log")}
            for v in variants}

    counts = {v: len(data[v]) for v in variants}
    print("Records per variant:", counts)
    if any(c == 0 for c in counts.values()):
        print("WARNING: at least one variant produced no records. Verdict = UNKNOWN.")

    steps = sorted(set.intersection(*(set(data[v]) for v in variants)))
    header = f"{'step':>5}  " + "  ".join(f"{'loss_'+v:>10}" for v in variants) \
                             + "  " + "  ".join(f"{'grdn_'+v:>10}" for v in variants)
    print(header)
    print("-" * len(header))
    for s in steps:
        losses = [data[v][s][0] for v in variants]
        grdns  = [data[v][s][1] for v in variants]
        print(f"{s:>5}  " + "  ".join(f"{x:>10.4f}" for x in losses)
              + "  "    + "  ".join(f"{x:>10.3f}" for x in grdns))

    # Pairwise means/stdevs vs the first variant (convention: first is HEAD).
    # Ratios are just raw numbers; they are NOT thresholded here.
    if len(variants) >= 2 and steps:
        ref = variants[0]
        print()
        for v in variants[1:]:
            ratios = [data[v][s][0] / data[ref][s][0] for s in steps]
            print(f"mean  loss_{v} / loss_{ref} = {statistics.mean(ratios):.4f}"
                  f"    stdev={statistics.stdev(ratios):.4f}"
                  if len(ratios) > 1 else
                  f"loss_{v} / loss_{ref} = {ratios[0]:.4f}")

if __name__ == "__main__":
    main()
```

One parser file can serve every subject whose runner emits the same log format. When a new subject emits a different format, add a second `parse_<other_metric>.py` — do not branch an existing parser with subject-specific logic.

### Non-negotiable invariants

| Invariant | Rule |
|---|---|
| **Reversibility** | Any exit (success, crash, signal) restores the subject to its original state. |
| **Verbatim substitution** | Each variant is a literal exact-string replacement — no regex, no `sed`. Readable side-by-side. |
| **Drift detection** | If the subject changed since the artifact was written, fail loudly with a clear message. Never silently patch the wrong region. |
| **Input invariance** | Seed, config, env, fixtures — identical across all variants. The only difference is the one substitution under test. |
| **HEAD is a variant** | The `HEAD` variant is an identity no-op. Running it must leave the subject unchanged. |
| **Audit trail before execution** | Print the diff/substitution before any real work runs. |
| **Output isolation** | Each variant's output lands in its own log. No shared mutable state between variants. |
| **Real-path execution** | Run through the same path the actual user/CI hits. Mocked internals, isolated fragments, and out-of-context stubs do not count. |
| **Mechanical extraction** | The parser emits physical values only (loss, grad norm, exit code, latency) via regex. No `PASS`/`FAIL`, no `expected`, no thresholds, no verdict. The agent reads the parser's table — never the raw stdout — to form the verdict. |
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
| `python -c "..."` or bash heredoc | No artifact, no audit trail, no re-run | Write switcher + runner + logs into `.verify_<task_slug>/` |
| `grep`, `assert pattern in source`, AST inspection | Static text-matching is not execution | Run the real entry point |
| Different seed/config between HEAD and patch | Output differences are confounded | Identity inputs across variants — only the substitution differs |
| Skipping the HEAD variant | No control, cannot detect environment drift | Always include `HEAD` (no modification) alongside `patch` |
| Eyeballing raw stdout to decide bug/no-bug | Reading logs to extract numbers is where LLMs hallucinate values | Run the parser; cite its rows |
| Parser labels `PASS/FAIL` / `expected` / thresholds | Encodes prediction into the extractor — parser now hides the answer | Parser emits raw physical values only; verdict is applied by the agent from The Rule |
| Writing conclusions before parser has run | Outcome-dependent reporting | Run → parse → read table → write |

## Red Flags — Stop and Restart

- "I read the code carefully and it clearly does X"
- "The tests would catch this" (without running them)
- "CI is green" (CI only proves what CI exercises)
- A subagent read the source and reported back — same static analysis
- "I'll just confirm with `python -c`" — ephemeral, no audit trail
- Writing the summary paragraph before any log has been read

**All of these mean: you have not run it. Produce the artifact, or explicitly hand off.**
