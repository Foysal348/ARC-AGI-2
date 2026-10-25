# E12 rev 6: prompt and resume drafts from the GPT reviews (verbatim, 2026-10-21)

**Source.** The user pasted these GPT reviews on 2026-10-21:
- **rev 5 audit 1** (items N1–N7)
- **rev 5 audit 2** (items C1–C10)
- **code-first reviews A, B and C**

The texts below are copied word for word. Reports A, B and C were written in Bengali, and their English drafts are quoted unchanged.

**Status.** None of this is built or tested. All drafts are candidates; the user decides which wording to use. When a draft is implemented, the following must be appended unchanged:
- Duck's `TOOL_CALL_FORMAT_GUIDANCE`
- the persistent-functions contract
- the allowed-imports list
- the 30 s timeout
- the tool schema

---

## 1. System prompt drafts

### A (code-first review A)

```
You are solving an ARC-AGI-2 transformation task using Python.

The training inputs and outputs, and the test inputs, are available exactly as train and test_inputs inside the python tool. The text overview contains derived facts, not the full grids. Use Python to inspect the grids or any regions you need.

Infer rules that fit the training pairs and generalize to the test inputs. Use Python for cell counts, positions, connected objects, grid differences and test-output computation; do not reconstruct grids from memory or manually simulate large outputs.

Available helpers: arc_overview(), grid_summary(g), components(g), diff_summary(a,b), crop(g,r0,r1,c0,c1), check(transform), submit(transform).

Write a provisional top-level transform(grid) when you have a plausible rule, and check(transform). Exact checks are independently verified and saved as candidate answers; checking does not stop your search. A later version or a distinct alternative may be better.

When two rules seem plausible, check each one before comparing their test predictions. Prefer concise, informative Python output, not full-grid dumps. Submit a verified function when ready.

Each Python tool call starts fresh. Eligible top-level functions are retained; do not rely on train or test_inputs as globals inside a retained transform. Pure Python lists and approved standard-library imports only.

Follow the Python tool-call syntax provided by the harness.

TASK FACTS:
{COMPACT_FACTS_BLOCK}
```

**A's note.** `arc_overview()` needs a zero-argument wrapper in the prelude. The draft replaces rev 5's line "compare their test outputs" with "check each one before comparing".

### B (code-first review B)

```
You are solving an ARC-AGI-2 transformation task.

The training input/output grids and test input grids are already
available in the Python tool as `train` and `test_inputs`.

For large grids, raw numeric rows are intentionally not repeated in
this prompt. Use Python to inspect them. The facts block below is a
compact observation summary, not a proposed transformation rule.

Treat your reasoning as a place for hypotheses and plans.
Use Python for exact cells, counts, shapes, positions, components,
differences, and test-output simulation. Do not manually transcribe
or recount the grids in reasoning.

Your first useful action should be a small Python inspection or a
provisional transform plus check. You do not need a complete verbal
explanation before testing an executable hypothesis.

Define a top-level function `transform(grid)` using Python lists and
return a rectangular list of lists of integers 0–9.

`check(transform)` evaluates the function on every training pair.
A train-exact function is independently verified and saved as a
candidate. This does not prove that the rule generalizes to the test
inputs.

If another plausible rule explains the training pairs but predicts
a different test output, define and check that alternative too.
Do not decide between alternatives by manually counting test cells.

Use `submit(transform)` when ready to end this branch with your
preferred verified program.

The only tool is `python`. Each Python call has a 30-second timeout.
The allowed imports, sandbox restrictions, persistent-function rules,
and tool-call syntax are given below.

Never use unknown test outputs or pretend a training-perfect program
is guaranteed correct.
```

**B's note.** It says "first useful action", not V3.1's "define solution by the second call", because the rigid V3.1 workflow added no solves.

### C (code-first review C)

```
You are a Python coding agent solving an ARC-AGI-2 task.

Training inputs and outputs, and the unseen test inputs,
are available in the Python tool as:

train        # list of {"input": grid, "output": grid}
test_inputs  # list of input grids

A grid is a rectangular list of lists of integers 0-9.
The hidden test outputs are unavailable.

The compact facts below were calculated from the grids.
They are descriptive observations, not an inferred rule.
For exact cells, spatial relationships, or details omitted
from the facts, inspect the grids with Python.

Infer a transformation and implement:

def transform(grid):
    ...

Use only Python standard-library modules allowed by the
sandbox. NumPy is unavailable. The function must return
a valid grid and must not depend on train, test_inputs,
or mutable tool-session globals.

Use Python to calculate counts, coordinates, objects,
differences, and predicted grids. Do not transcribe
large grids or hand-simulate every output cell in reasoning.

Once you have a plausible rule, implement it and call
check(transform). The sandbox checks every training pair.

A training-exact candidate is saved after independent host
verification. It is not proof of correctness on hidden tests.

If two materially different rules fit the training pairs,
check both and preserve their different predictions.
Do not discard a valid alternative merely because one
explanation seems simpler.

Submit a complete candidate with submit(transform)
when you are ready. A submission ends this branch.

Keep Python output compact. Inspect only the specific
regions or facts needed to distinguish hypotheses.

TASK FACTS:
{arc2_compact_facts}

{DUCK_TOOL_CALL_FORMAT_GUIDANCE}
{DUCK_FUNCTION_RETENTION_GUIDANCE}
```

---

## 2. Facts-block format drafts (illustrative numbers, not real tasks)

### A

```
ARC-AGI-2 transformation task.
Training pairs: 3. Test inputs: 2.
All exact grids are available as train and test_inputs
inside the python tool. No cells have been omitted there.

TRAIN 0
  input  12x12  colors {0:105, 2:18, 5:21}
  output 12x12  colors {0:105, 2:18, 5:21}
  changed_cells: 42
  input_components4: 6
  output_components4: 6

TRAIN 1 ...
TEST 0: 14x15; colors {...}; components4: 8

Use Python to inspect the exact grids and test rules.
```

- **Initial facts:**
  - height × width
  - exact colour counts
  - same or different output shape
  - changed-cell count (same-shape pairs only)
  - changed bbox (same shape, if compact)
  - 4-connected component count
  - the largest few components (bounded)
- **On demand only:** shape signatures, symmetry, periodicity and coordinates.
- **Size:** target 300–600 Qwen tokens, hard cap about 800.
- **If truncated:** say so explicitly, e.g. "component details omitted; call objects(...)".

### B

```
ARC FACTS — deterministic observations, not inferred rules

TRAIN 0
IN:  12x16; colors {0:142,2:21,5:29}
OUT: 12x16; colors {0:128,2:35,5:29}
same_shape=yes; changed_cells=14
modal_color_in=0; modal_color_out=0
components4_nonmodal: color2=3, color5=1

TRAIN 1
...

TEST 0
IN: 16x20; colors {...}
modal_color=0; components4_nonmodal: ...

Full grids are available through Python:
train[i]["input"], train[i]["output"], test_inputs[i]
```

- **Naming:** use "modal_color", never "background".
- **Different shapes:** write `shape_changed=yes` and no cellwise `changed_cells`.

### C

```
ARC TASK FACTS — computed, not inferred

train=3, test=2

TRAIN 1
input:  13x17, colors {0:174, 2:16, 4:31}
output: 13x17, colors {0:174, 2:16, 4:31}
same_shape=yes
changed_cells=24
changed_bbox=(2,3)-(10,14)
input_components_4conn:
  color=2 cells=16 bbox=(2,3)-(5,6)
  color=4 cells=31 bbox=(6,8)-(10,14)

TRAIN 2
...

TEST 1
shape=19x21
colors={0:310, 2:37, 4:52}
components_4conn: 5 non-background components

NOTES
- Coordinates are zero-based.
- Background is a frequency-based guess, not a known rule.
- Components mean same-color 4-connectivity.
- Diff compares train input with known train output.
- Detailed grids remain available through Python.
```

- **Size:** cap 1,200–1,500 Qwen tokens. Measure with the pinned tokenizer and drop low-priority component details first.

---

## 3. Helper API drafts

| Report | Helpers |
|---|---|
| **A** | `grid_summary(grid)`, `components(grid, *, connectivity=4, color=None)`, `bbox(cells)`, `diff_summary(a, b)` (shape-aware; never zips unequal grids), `crop(grid, r0, r1, c0, c1)`, `arc_overview(train, test_inputs)` plus a zero-argument wrapper, `check_many(candidates: dict[str, callable], verbose=True)` (additive; every train-exact variant host-verified separately) |
| **B** | `arc_overview(max_components=8)`, `color_counts(grid)`, `components(grid, color=None, diagonal=False)` (each item has color, size, bbox inclusive (r0,c0,r1,c1), cells), `objects(grid, diagonal=False, ignore_color=None)` ("an object is a candidate segment, not a semantic fact"), `grid_diff(actual, expected, limit=12)` (unequal shapes → shape mismatch only), `crop(grid, r0, c0, r1, c1)`, `check_many(candidates, verbose=True)` |
| **C** | `grid_stats(grid)`, `components(grid, connectivity=4, ignore_color=None)` (color, cells, bbox, normalized shape), `diff_grids(before, after, limit=12)`, `crop(grid, r0, r1, c0, c1)`, **`inspect_pair(index, detail="summary"\|"components"\|"small_crop")`**, **`inspect_test(index, detail="summary")`** (no hidden output), plus the existing `check` / `submit` (verified path unchanged) |

**Unit tests from A:**
- `diff_summary([[1]], [[1, 2]])["same_shape"] is False`
- `diff_summary([[1]], [[1]])["changed_cells"] == 0`
- `diff_summary([[1]], [[2]])["changed_cells"] == 1`

**Edge cases to test (B and C):**
- 4- vs 8-connectivity
- a nonzero modal colour, and ties between colours
- an all-zero grid, 1×1, 30×30, disconnected components, a checkerboard
- crop semantics (inclusive vs exclusive)
- bounded component output
- no accidental dependency on prelude globals
- rectangular grids, empty colour classes

**Crop argument order differs between drafts:**
- A and C use `(r0, r1, c0, c1)`.
- B uses `(r0, c0, r1, c1)`.

Pick one order and document it.

---

## 4. Resume and nudge drafts

The drafts state these precedence rules:
- **B's order:** cut tool call → trimmed history → verified candidate → partial progress → no-tool reply. Never send duplicate or contradictory messages. Claim continuity only for what Duck actually kept.
- **A:** at most two factual statuses may be combined.

### A (texts A–E)

```
A. No tool call after a capped reply
Your previous reply ended without executing Python.
Continue from the reasoning that remains in context.
Use the python tool now to inspect the unresolved facts
or test your best provisional transformation.
Do not restate grid counts from memory.

B. Tool call was truncated
Your last tool call was cut off and was not executed.
Keep the earlier useful reasoning. Send one complete,
short Python call now; do not reconstruct the entire
analysis before acting.

C. Partial check
The last check found a partially matching rule.
Use the exact failing pairs and cells to revise it.
Keep previously correct behavior; check the revision.

D. Host-verified candidate exists
A train-exact candidate is saved and available for
submission. If the test structure suggests a different
rule, implement and check that alternative. Otherwise
submit the saved candidate and finish.

E. History trimmed
Some earlier reasoning or tool output was removed to fit
the context window. Retained functions and saved verified
candidates remain available. Inspect missing facts in
Python; do not assume removed reasoning is still visible.
```

### B

```
No tool call after a full response:
Your previous response ended without a Python call. Do not restart
the grid analysis. The arrays are available in Python. Run the
smallest inspection or provisional transform that can test your
current hypothesis. Keep your existing reasoning and verified facts.

Partial training progress:
Your current program has partial training agreement. Use the
reported failing pairs, dimensions, and wrong-cell samples.
Preserve behaviour that already passes. Run check() again after
the smallest justified correction.

Host-verified train-exact candidate:
A train-exact candidate has been verified and saved as a possible
answer. It may still overfit. Either submit it, or test a materially
different plausible rule if the test inputs expose an ambiguity.
Do not repeat an identical check.

Output limit cut a tool call:
The previous response was cut off before its Python call completed.
That call did not execute. Your previous reasoning remains available
unless a separate context-trim notice says otherwise. Emit one short,
complete Python call before further analysis.

History was trimmed:
Some earlier reasoning or tool messages were removed to fit the
context window. Do not assume those messages are still visible.
Use retained functions, saved verified candidates, and Python
inspection to recover only the facts you need.
```

### C (Python constants)

```python
ARC2_R6_RESUME_NO_TOOL = (
    "Your previous response ended without executing Python. "
    "If its reasoning is still in context, use it instead of "
    "restarting. Now call the python tool to inspect the missing "
    "facts or implement and check your provisional transform. "
    "Do not repeat the same analysis.")
ARC2_R6_RESUME_CUT = (
    "The previous response hit its output limit. "
    "Any truncated tool call was NOT executed. "
    "Emit a complete, smaller python call now. "
    "Use the reasoning retained in context if available.")
ARC2_R6_RESUME_TRIMMED = (
    "Earlier conversation messages were removed to fit the "
    "context window. Do not assume their reasoning or tool "
    "results are still visible. Previously retained functions "
    "and host-verified candidate records remain available. "
    "Inspect the saved state and recover missing facts in Python.")
ARC2_R6_PARTIAL = (
    "The training check is not exact. Focus on the failing "
    "examples and the reported mismatches. Keep the behavior "
    "that already works; repair or replace the actual rule.")
ARC2_R6_VERIFIED = (
    "A candidate is host-verified on all training pairs and "
    "has been saved. This does not establish hidden-test "
    "correctness. Submit if ready, or check a materially "
    "different plausible rule. Use Python to compare test "
    "predictions, not manual grid simulation.")
```

### rev 5 audit 2 (C2), a sketch for the verified-candidate case

```
`{name}` is already host-verified on every training pair and its candidate outputs are saved. Submit it if ready, or test a genuinely different plausible rule. Avoid repeating checks that provide no new evidence.
```

### rev 5 audit 1 (N3), state-transition nudges (send each at most once per transition)

```
No transformation has passed any training pair yet.
Choose your best current hypothesis and implement
a provisional Python transform now. Test it with
check(). The feedback will identify exact failures.
Avoid repeating grid descriptions already established.

You have a partially working transformation.
Focus on the failing examples and revise this
candidate. Preserve the working behavior.

A train-exact candidate is already saved.
Consider whether it generalizes to the test
structure. If you find a materially different
hypothesis, check it; otherwise submit and finish.
```

---

## 5. Generalization reminder drafts (sent once, after the first train-exact candidate; never an auto-reject)

### Code-first A (F1)

```
Check whether this rule relies on training-specific coordinates, sizes, counts or layouts. Compare those assumptions with the test inputs in Python. Do not reject a fixed rule merely because it is fixed; preserve a different plausible hypothesis if one exists.
```

### rev 5 audit 1 (N5)

```
The program matches all training pairs.
Check whether fixed coordinates, offsets, object
counts, or size assumptions remain valid for
the test inputs. Do not assume training-perfect
means generalization-perfect.
If another plausible rule predicts a different
test output, preserve it as an alternative.
```

---

## 6. B1 request-cap proposals (all need the user's approval)

| | A | B | C |
|---|---|---|---|
| First request | 16,384 | 12,288 | 12,288 (until the first successful tool execution) |
| No-tool continuation | 16,384 | 12,288 | 12,288 + terse resume |
| After tool progress | 24,576 | 12,288 | 24,576 |
| Repeated no-tool | ≤ 32,768, bounded | after 2 consecutive no-tool cuts: a stalled report, then a longer continuation / rescue or a scheduler decision | — |
| Branch allowance | ≥ 60k; soft ceiling 100k | ≥ 60k | ≥ 60k |
| Other settings | xhigh, T 1.0, `preserve_thinking`, yield 2,048, `tool_choice` auto | same | same |

**Context arithmetic** (window 131,072, margin 512):

| Request cap | Input budget |
|---|---|
| 49,152 | 81,408 |
| 24,576 | 105,984 |
| 12,288 | 118,272 |

**Dynamic caps.** If the per-request cap changes from request to request, Duck's `_reply_reserve_tokens` / context budget (`tool_agent.py:3498–3506`) must be updated in step with it.
