# Puzzle ordering — NVARC 32.22-LB reference notebooks (analysed 2026-10-12)

## Sources

| Notebook | Version on disk | Ordering code |
|---|---|---|
| `foysalemonshanto/arc-2026d-32-22lb-updated` (link: scriptVersionId 330020564) | latest = version 5 (`arc-2026d-32-22lb-updated.ipynb`, pulled 2026-10-12) | `starter.py`: `get_puzzle_priority` + `sorted(...)` |
| `foysalemonshanto/arc-2026d-32-22lb-updated-v2` (link: scriptVersionId 331102752, the LB-32.64 reference run) | latest, byte-identical to `resources/Reference Notebooks/LB_compare/…-v2.ipynb` | same |

**Version check**
- The Kaggle API refuses versioned pulls (403), so only the latest version of each notebook could be pulled.
- **This does not matter for the ordering.** The ordering code is identical in both notebooks, character for character.
- The only `starter.py` difference between v1 and v2 is 12 extra IDs in `DEBUG_PUZZLES`, which is used only in the commit run.

## The algorithm (starter.py)

**1. Load the file.** In a rerun this is `arc-agi_test_challenges.json` (every hidden task); in the commit run it is `arc-agi_evaluation_challenges.json`. The file is loaded with `json.load`, which keeps the file's key order.

**2. Score every task with `get_puzzle_priority(key, data)`.** It looks at **training pairs only**: no test inputs, no labels.

| Quantity | Definition |
|---|---|
| `n_train` | number of training pairs |
| `avg_input_size` | mean of H×W over the training inputs |
| `size_ratio` | per pair: max(in cells, out cells) / max(min(in cells, out cells), 1) |
| `max_ratio`, `avg_ratio` | max and mean of the size ratios |
| `n_colors` | number of distinct colours over all training inputs and outputs |
| `is_dynamic` | true if any training input shape ≠ its output shape |

**3. Assign a tier.** The first matching rule wins:
1. `is_dynamic and max_ratio > 30` → **tier 3** (extreme)
2. `is_dynamic and max_ratio > 5` → **tier 2**
3. `avg_input_size > 400` → **tier 2**
4. otherwise → **tier 1** (easy)

**4. Compute the fine score:**

```
fine = avg_input_size/900*0.30 + (avg_ratio-1)/10*0.25 + n_colors/10*0.20 + (0.25 if is_dynamic else 0)
```

- If `n_train >= 6 and tier < 3`, then `fine *= 0.7`.

**5. Sort.** `keys_sorted = sorted(data.keys(), key=lambda k: get_puzzle_priority(k, data))`.
- The sort is ascending on `(tier, fine)`, so the **easiest tasks come first** and the extreme ones last.
- Python's sort is stable, so ties keep the JSON file order (not alphabetical order).

**6. Queue the tasks.** Every task goes into **one shared queue** in that order, followed by 4 `None` sentinels.
- In the commit run, only `DEBUG_PUZZLES` are queued, still in sorted order.

**7. Workers pull from the queue.** There are 4 workers, one per GPU.
- Each worker pulls the next task whenever it is free, until `end_time` (12 h from the starter's start).
- A task runs as a whole: train, then every test input. There is a 1,600 s cap per task.

**What the ordering achieves:** when time runs out, the tasks never reached are the hardest ones (the large or extreme-scaling tasks), and the easy tasks are guaranteed to be finished.

## Our implementation: `puzzle_order.py`

- `get_puzzle_priority` is copied **verbatim**. `test_puzzle_order.py` asserts it is byte-identical to both reference `starter.py` files.
- `order_task_ids(data)` is the reference's own `sorted(...)` call.
- `order_units(data)` gives our per-test-input units (`task#q`) in that task order. A task's test inputs stay together, in ascending q.
- **Rule:** pass the dict exactly as `json.load` returned it. Do not sort the IDs first; v1.4.4 uses `sorted(CHALLENGES)`, which would change the tie order.

**Test runs:**

| Data | Tasks / units | Easy | Hard | Extreme |
|---|---|---|---|---|
| Evaluation file | 120 / 172 | 65 | 51 | 4 |
| Commit-run placeholder test file | 240 / 259 | 178 | 52 | 10 |

**Where the EVAL12 tasks fall in the 120-task evaluation order:**
- 8 are tier 1, at ranks 15–59.
- 2 are tier 2: 6e4f6532 at 80 and 898e7135 at 100.
- 2 are tier 3 and come last: 20a9e565 at 118 and 4c7dc4dd at 120.

## How to apply it in our notebooks (not yet applied)

**Q27-XHIGH DIRECT (v1.4.4 → next version)**
- Today, `UNITS` is built from `TASK_IDS = sorted(CHALLENGES)`, an alphabetical order.
- Build the scheduler queue from `order_units(test_challenges)` instead.
- With admission until 11:34, the queries left unadmitted would then be the hardest ones, not an arbitrary alphabetical tail.

**Q27-AGENT synthesis (v1.9.x)**
- Submit tasks to the thread pool in `order_task_ids(...)` order. The pool starts work in submission order.

**Both**
- Keep the reference's split: the rerun sorts **all** hidden tasks; the commit run sorts the selected subset.
