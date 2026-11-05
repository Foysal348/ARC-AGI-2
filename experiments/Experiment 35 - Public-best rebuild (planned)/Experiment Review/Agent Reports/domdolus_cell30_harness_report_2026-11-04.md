# Domdolus notebook, cell 30: harness source files (full inventory)

Source: `Build/domdolus_cells/cell30.txt`. The file is 2,033 lines. PowerShell `Measure-Object` reported 1,796 lines because some string literals contain embedded line breaks. All 2,033 lines were read.
Date: 2026-11-04. Read-only analysis.

**Scope note.** Cell 30 holds only the *per-sample* harness: the client loop, the REPL aids, the prompts and the config. The *scheduling* across the 120 tasks is set by later cells through environment variables: how many runs per task, rounds, the token budget per round, the deadline, rerun notes, and how check_attempts.json becomes attempt_1 and attempt_2. A grep of the sibling cells shows this:
- `cell46.txt:450-456` sets the run env. It **overrides several cell-30 defaults**: `ARC_THINK_CAP=0`, `ARC_BUDGET_REMINDERS=0`, `ARC_FINAL_REVIEW=0`, `ARC_SALVAGE_AT_DEADLINE=0`, `ARC_SAMPLE_SECONDS=0`, `ARC_DEADLINE_EPOCH=STOP_AT`, `ARC_REASONING_FIELD=reasoning`, `ARC_REASONING_EFFORT=REASONING_EFFORT`, `ARC_RERUN_NOTES=SETTINGS["notes_file"]`, `ARC_RUN_LIMITS=SETTINGS["limits_file"]`. It also sets the window, trim and recovery values from notebook variables.
- `cell46.txt:64,89` describes the BUDGET (TOTAL_TOKEN_BUDGET, round-1 runs, per-round limits).
- `cell33.txt:40-63` patches `_rendered_prompt` again ("the model is told its real budget") and calls `setdefault("ARC_MAX_GEN_TOKENS","100000")`.
These cells must be read for the scheduling, voting and time-budget parts. Section 2 marks which items cell 30 decides and which are external.

---

## 1. Files written and their roles

The cell asserts that `REPO/src/arcagi2` exists ("Run the pivotools cell first"), that `sandbox/local_sandbox.py` exists (from an earlier local-sandbox cell), and that `REPO`, `ARC_PY` and `ARC_ENV` come from earlier cells.

### Files written in full (the `FILES` dict, written at lines 1911-1916, each with a sha256 print)

| File (under `src/arcagi2/`) | Lines in cell | Role |
|---|---|---|
| `utils/grid_image.py` | 8-209 | Stdlib-only PNG renderer for ARC grids (palette, gridlines, side-by-side layout). Builds the first-message content (text plus one image per train pair and per test input). Estimates image tokens. `render_part_png` draws a crop with row and column number labels. Its source is also exec'd inside the REPL so the model can call `show()` there. |
| `api/clients_v3.py` | 210-1674 | The core harness: `AsyncChatCompletionsAPIClientV3`, the multi-turn tool loop with trimming at a 32K window, the thinking cap, budget reminders, recovery of bad final answers, final review, salvage, `check()` records saved to `check_attempts.json`, and a live `transcript.md` plus images. It also defines the REPL-side code: `show`/`show_part`, `check`, the grid helpers, describe, salvage and verify. |
| `solver/config/qwen3_8_flash_next_free.py` | 1675-1813 | `QWEN3_8_FLASH_NEXT_FREE_AGENTIC_CODING_CONFIG`: model name, sampling, tool, and every knob read from `ARC_*` env vars. Renders the prompt templates with the run's numbers. |
| `solver/prompts/system_prompt_free.txt` | 1814-1827 | System prompt (python tool description). |
| `solver/prompts/agentic_coding_solver_free.txt` | 1828-1908 | User prompt template (ARC background, puzzle, task aids, working style, output format). |

### In-place edits of existing pivotools files (`edit()`, idempotent through marker strings, lines 1918-2017)

- `utils/puzzle_utils.py`:
  - `grid_to_text(grid, compact=False)`: compact mode writes one digit per cell with no spaces. The comment says this is "about half the tokens".
  - `puzzle_to_text(..., compact=False)` passes `compact` through to all 4 grid_to_text calls.
  - With `ARC_TEXT_GRIDS=0`, a grid is replaced by `"(a HxW grid: see its image below; the values are in `puzzle`)"`.
- `solver/turn.py`: `run()` gets a `content_builder` argument, so the first user message can be a list of text and image parts.
- `solver/solver.py`: imports grid_image and passes `compact=` and `content_builder=build_first_message_content(...)` when `first_message_images` is set. `initial_code=[f"puzzle={repr(get_copy_without_solutions(puzzle_json))}"]` is unchanged.
- `solver/config/__init__.py`: registers `"qwen3_8_flash_next_free"` in `SOLVER_CONFIGS`.

### Self-test (lines 2019-2033)

The cell runs a subprocess with `ARC_PY`. It imports the config, renders a 2x2 toy puzzle, and prints the window, trim trigger, trim target, tools, recovery max, chat_template_kwargs and trim_note. It also checks that the compact text contains `"01\n23"` and counts the image parts. The cell asserts that the return code is 0.

---

## 2. Harness algorithm

### 2.1 Prompts, quoted in full

**System prompt** (`system_prompt_free.txt`, lines 1814-1827):

```
# Tools

## python

Use this tool to execute Python code in your chain of thought. 

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. The environment supports common third party libraries like numpy, scipy, shapely, networkx, scikit-image, more-itertools, pillow, python-constraint, ortools and z3-solver. python will respond with the output of the execution or time out after 120.0 seconds. 

### Usage guideline

1. Use the variables and functions already loaded in the Jupyter notebook instead of defining them yourself. The user message will provide a complete list of such variables and functions.
2. Prefer to write reusable functions. These will persist in the stateful Jupyter notebook and can be referenced in later code snippets without having to define them again.
3. Long tool output is cut off in the middle.
```

**User prompt template** (`agentic_coding_solver_free.txt`, lines 1828-1908). The model sees this as the first (pinned) user message:

```
You are an expert ARC AGI puzzle solver. Please solve the puzzle given below.

# Background information about ARC AGI puzzles

Each task consists of a small number of demonstration examples (3.3 on average), and a small number of test examples (1 - 3).

Each example consists of an “input grid” and an “output grid”. Each “grid” is a literal grid of colored squares There are 10 unique colors. A grid can be any height or width between 1x1 and 30x30, inclusive.

The color code is as follows:

0 - black
1 - navy blue
2 - red
3 - green
4 - yellow
5 - gray
6 - pink
7 - orange
8 - sky blue
9 - maroon

When solving an evaluation task, a test-taker has access to the training examples for the task (both the input and output grids), as well as the input grid of the test examples for the task. The test-taker must construct on its own the output grid corresponding to the input grid of each test example. “Constructing the output grid” is done entirely from scratch, meaning that the test-taker must decide what the height and width of the output grid should be, what colors they should place on the grid, and where. The task is successfully solved if the test-taker can produce the exact correct answer on all test examples for the task (binary measure of success).

The mapping from integers (0-9) to color names is arbitrary and should be treated as categorical labels; the transformation rule never relies on the specific integer–color pairing.

ARC AGI puzzle solutions feel "natural" to humans. Natural rules tend to build on visual priors and small-number counting. They are visually immediate or prominent for humans. That is why humans from all educational backgrounds are able to solve these puzzles. Rules that depend on non-visual considerations (e.g. mathematics that's more advanced than small number counting) or inelegant mechanics (e.g. tie-breakers, complex conditional rules, special exceptions) are considered "unnatural."  

This doesn't mean puzzles have to be simple or easy. Common patterns that puzzles use to increase difficulty are:  

- **Rule composition:** multiple simple, natural rules are applied in sequence, with one transformation setting up the next.
- **Contextual application:** a simple if/else based on a visually clear feature.
- **In-grid symbol definition:** certain shapes or patterns act as symbols whose meaning is established using the training examples and which control some aspect of the transformation. The meaning can even vary across training examples.
- **Iterated steps:** a simple action is applied repeatedly (as in assembling pieces of a jigsaw), where each step constrains the next. Repetition is restricted to a small number.
- **Approximate features**: Features that are easily detected by humans, but hard to detect using naive algorithms. Examples are slightly broken symmetry, slightly broken continuity, off-center symmetry axes, objects defined via 8-connectivity etc.
- **Non-locality:** the solution requires applying changes to one region, with the changes depending on features in a different, distant region.

# Puzzle to solve

Grids below are written one row per line, as digits without spaces (for example `0120` is a row of four cells).

{{puzzle}}

# Task aids

You may execute code in the Jupyter notebook environment.

Functions and variables available in the Jupyter notebook:

- `puzzle`: A Python `dict` containing the training and test grids. The key "train" contains the training examples in a list. Similarly, the key "test" contains the test examples as a list. Each training/test example is a dict containing keys "input" and "output". The values are the grids expressed as list of lists of integers (0 - 9). Access: `puzzle["train"][i]["input"]`, `puzzle["train"][i]["output"]`, `puzzle["test"][j]["input"]`.

- `show(*grids, labels=None)`: draws one or more grids as an image and shows it to you in the next message (grids are drawn side by side). You can call it whenever you want to look at grids as images, as often as you like. Example: `show(puzzle["train"][0]["input"], puzzle["train"][0]["output"])`, or `show(candidate_output)` to look at your own result. Use images for overall shape, symmetry and layout, and use code or the text above for exact cell values.

- `show_part(grid, rows=None, cols=None, labels=None)`: like `show`, but draws only a part of one grid, with the row and column numbers of the original grid written on the borders, so you can see exactly which row and which column each cell is in. `rows=(start, stop)` and `cols=(start, stop)` are 0-based and the stop is not included; a single number selects one row or column. Example: `show_part(puzzle["train"][0]["input"], rows=(2, 8), cols=(10, 16))`.{{CHECK_AID}}{{HELPERS_AID}}

Your very first code execution should be `puzzle.keys()` to check data access. 

Don't print the `puzzle` dict, because it prints out the grids that are already present in this user message and consumes valuable token budget.

# Working style

- The conversation window is about {{WINDOW}} tokens. When the conversation reaches it, the oldest messages are removed automatically until about {{TRIM_TARGET}} tokens remain, so do not rely on remembering earlier steps.
- One thing is never lost: everything you define in the Jupyter notebook. Write reusable functions and keep your best `solution` draft defined in the notebook.
- Reason through code, not in your head. Whenever you want to know something about the grids (is this a transposed or flipped copy of that? how many cells have color 3? which cells differ between two grids? is the output symmetric?), write a few lines of code and run them instead of working it out by hand in your thinking. Do not simulate a transformation cell by cell yourself: write the function and compare its output with the expected grid.{{CHECK_STYLE}}
- Do not think for long before acting. As soon as you have a hypothesis, however rough, test it with code right away and read the result. If the first idea is wrong, the output will teach you more than more thinking would. Keep each thinking step short and move on to a tool call.
- Keep working until you have a `solution` that passes every training example, then give it as your final answer.

# Your task

Please solve the puzzle. Solving the puzzle means constructing a function `solution(grid: list[list[int]])` that implements the transformation from input grid to output grid shown in the training examples. The solution must work on all training examples.

```
for idx, example in enumerate(puzzle["train"]):
    assert solution(example["input"]) == example["output"], f"Train example {idx}: mismatch"
```

The solution must generalize to the test input(s). Solutions that cheat by hardcoding the training example outputs are not acceptable.

# Output format (STRICT)

Your response must contain a single Python code block fenced by ```python ... ```. Must include a `solution` function. Must be self-contained i.e. include all necessary imports and helper functions.
```

**Template substitutions** (`_rendered_prompt`, lines 1734-1752; cell 33 patches this function again):
- `{{WINDOW}}` is `ARC_CTX_WINDOW // 1000 * 1000` with a thousands separator. The default renders as "32,000".
- `{{TRIM_TARGET}}` is `ARC_TRIM_TARGET`, rounded the same way. The default renders as "16,000".
- `{{CHECK_AID}}` / `{{CHECK_STYLE}}` are filled when `ARC_CHECK_HELPER` is on (default on, and cell 46 sets "1"):
  - CHECK_AID (line 1707): "\n\n- `check(fn=None)`: runs a function (default: your current `solution`) on every training example and every test input and prints the result: for each training example OK, or the wrong shape, or how many cells are wrong and which ones as (row,col) got/expected; for each test input the size and colours of the output (and any error, with its line). Example: `check()` or `check(my_idea)`. It is exact and costs far fewer tokens than checking cells in your thinking."
  - CHECK_STYLE (line 1711): " To test a hypothesis, write it as a function and run `check(f)`; never verify a rule cell by cell in your thinking."
- `{{HELPERS_AID}}` is filled when `ARC_GRID_HELPERS` is on (line 1713): "\n\n- Grid helpers (already defined; each takes a grid, a list of lists or a numpy array, never changes it and returns plain lists of ints): `rot90(g, k=1)` (clockwise, k times), `flip_lr(g)`, `flip_ud(g)`, `transpose(g)`, `dihedral(g)` (a dict of all 8 rotations/flips by name), `symmetries_between(a, b)` (the names of the rotations/flips that turn a into b), `crop(g, rows, cols)` (the same (start, stop) as `show_part`), `bbox(g, bg=0)` (rows and cols of the part that is not background, so `crop(g, *bbox(g))` cuts it out), `components(g, bg=0, diagonal=False, multicolor=False)` (the objects: a list of dicts with color, colors, cells, size, rows, cols, shape). They exist only in this notebook: if your final `solution` uses one, put its code into the final code block too (`helper_source(\"components\", \"crop\")` prints the code of the ones you name, with what they need)."
- `ARC_TEXT_GRIDS=0` (TEXT_GRIDS_OFF, lines 1721-1731) swaps three sentences. Each swap is asserted to match exactly once. This mode needs `ARC_IMAGES=1` and `ARC_CHECK_HELPER=1`:
  1. "Grids below are written one row per line, ..." becomes "The grids are NOT written out as text in this message: below, each grid is given by its size, and after the puzzle text every grid is drawn as an image. The exact cell values are in the `puzzle` variable in the Jupyter notebook: get anything exact (colours, counts, positions, differences, objects) with code."
  2. "Use images for overall shape, symmetry and layout, and use code or the text above for exact cell values." becomes "Use images for overall shape, symmetry and layout, and use code for exact cell values."
  3. "Don't print the `puzzle` dict, ..." becomes "Don't print whole grids to read them cell by cell: compute what you need with code, and test a rule by writing it as a function and running `check()` on it."
- The rendered file is cached in the temp dir as `arcagi2_free_<sha1[:12]>_<name>`.

**First user message content** (`build_first_message_content`, lines 99-117). The parts, in order:
1. The prompt text.
2. A text block "\n\n# Images of the grids\n\nThe same grids as above, drawn as images. Each square is one cell and thin lines separate cells. Use the images for overall shape, symmetry and layout, and the text or code for exact values." With `ARC_TEXT_GRIDS=0` the variant reads "...(they are not written out as text)...".
3. For each training pair, the text `"\nTraining example {i}: input (left) and output (right)"` and one PNG with input and output side by side.
4. For each test input, the text `"\nTest input {j}"` and one PNG.

The images use scale 16 px per cell, with gridlines when scale >= 4. The gridline colour is (85,85,85) and the gap is 10 px of fill colour (32,32,32).

**Puzzle text**: `puzzle_to_text(markdown_level=2, separator="\n\n", include_test=True, include_all_tests=True, include_solutions=False, compact=True)`. Compact mode writes one digit per cell with no spaces (`ARC_COMPACT` default 1, and cell 46 sets "1").

### 2.2 Harness-injected messages, quoted in full

The thinking-cap and budget messages are on by default in cell 30 but **off in cell 46**.
- THINK_CAP_NOTE (265): "[Note from the harness] Your reply was stopped after {k}K tokens of thinking without running any code. Stop checking by hand: write the idea you are working on as code and run it with the python tool right now (check(f) tests a function on every training example)."
- BUDGET_MID_NONE (267): "[Note from the harness] You have used half of this task's token budget ({used}K of {limit}K) and have not run check() on a solution yet. Write your current best idea as `solution` now and run check(), even if it is incomplete. If it is far off, drop that idea and try another hypothesis."
- BUDGET_MID_LOW (269): "[Note from the harness] You have used half of this task's token budget ({used}K of {limit}K) and your best check() so far gets only {ok}/{n} training examples right. Stop refining this idea in your thinking: test a different hypothesis with the python tool now."
- BUDGET_LATE (271): "[Note from the harness] About {left}K tokens are left for this task. If a function passes check() on every training example, reply with the final answer now (no more visual checks). Otherwise write your best idea as `solution` and run check() right away."
- REVIEW_NOTE (273-282): "[Note from the harness] Your final answer passes every training example. Before it is submitted, check it once more against the test input(s): the training examples may not show everything a test input contains. Look at each test input for something new: a colour, size or shape that no training input has, objects touching each other or the border, more or fewer objects, overlaps, a different layout. Run your solution on the test input(s), look at the outputs with show(), and decide whether each one is what the rule really asks for.\n- If they are right, reply with your final answer again (the same code block).\n- If a test input is an edge case that your code gets wrong, fix the code, run check() on the new version, and reply with the new code as your final answer.\nYour current answer is kept as the first attempt; a different final answer is submitted as the second attempt. Keep the review short: about {k}K tokens."
- RECOVERY_NO_CODE (286-291). This one is always active. "Your reply was not accepted as a final answer: {why}\nThe final answer must be ONE reply that contains a ```python code block that defines `solution(grid)`; the LAST code block of the reply is the one that is used. The program must be self-contained: all imports and helper functions inside that block (it is run in a fresh Python session where nothing from your notebook exists). Try again: you may use the python tool again, then reply with the final code block."
  - The `why` variants (1245-1257): "it was empty: it had no text at all, only thinking." / "it contains no ```python code block." / "its last ```python code block has a syntax error: {msg} (line {n}: {line})." / "...could not be parsed: ..." / "its last ```python code block does not define a function named `solution`."
- RECOVERY_CHECK_FAILED (292-297): "Your final program was not accepted. It was run in a fresh Python session (only what is inside the code block exists there) on the training examples and on the test inputs, and these problems came up:\n{details}\n`solution(grid)` must return a list of lists of ints and must reproduce every training output; the program must be self-contained (all imports and helper functions inside the block). Fix the problem and reply with the final code block again."
  - The harness appends this to the recovery feedback (1557): "\n\nYour Python session is unchanged. Currently defined in it: {functions: ... | variables: ...}."
- Trim note (1200-1202): "[Context note] Earlier steps of this conversation were removed to save space. Your Python session is unchanged." followed by " Currently defined in it: functions: f(sig); ... | variables: ...". The functions part is cut at 700 characters and the variables part at 400 (643-644).
- Image message (1106): "Image(s) from your show() call(s):" followed by `[label]` and an image for each picture.
- Second pass (session 15, `_with_rerun_note`, 1325-1351): the per-puzzle note from the `ARC_RERUN_NOTES` JSON is appended to the end of the first user message. A note of `None` means "round 1, no note" (1339). A missing note raises a RuntimeError (1341-1342).

### 2.3 Loop (`_run_tool_loop`, lines 1410-1670)

1. **Setup** (`call`, 984-1015): the system prompt replaces the developer message. A temp `show_dir` is created. The REPL initial code is `puzzle=...` (from solver.py), then the show setup (grid_image source plus `show`/`show_part`), then `_CHECK_CODE`, then `_GRID_HELPERS_CODE`. The loop then runs through `ensure_repl_and_run_tool_loop`.
2. **Head**: the system message and the pinned first user message (plus the rerun note). `entries` holds everything after the head. The per-run limit is `gen_limit = _run_limit()`: the per-puzzle value from the `ARC_RUN_LIMITS` JSON (a missing value raises an error), otherwise `max_generated_tokens`.
3. **Each iteration**:
   - Check the deadline: `_remaining()` returns the earlier of `deadline_epoch` and `t_start + sample_seconds`. If it is <= 0, set the deadline flag and break.
   - `_maybe_trim(force=window_hit)`.
   - Send the request: `messages = head + entries` with `tools=[python]` and **no max_tokens** (1422), except under the thinking cap (below). The call is wrapped in `asyncio.wait_for(timeout=max(1, remaining))`, so a reply in flight is cancelled at the deadline (1449-1453).
   - Errors:
     - BadRequest with a context-length error: trim deeper, up to level 3 (1455-1461). If nothing can be dropped, `base_tokens += 2000`.
     - Other BadRequest: retry up to `max_retries`.
     - InternalServerError: sleep 2 s and retry, up to 6 per task (1467-1474).
     - Malformed tool calls with a finish other than "length": retry up to `max_retries` (1475-1481).
   - Count tokens: `generated += completion_tokens`, using the usage numbers from the server or an estimate of 3 characters per token.
     - `over = generated >= gen_limit`.
     - During a review, `over` is also set once the review's own budget is used up.
     - The estimate is calibrated against the real prompt_tokens (k_text, the characters per token, starts at 2.5 and is clamped to 1-4).
   - Branches:
     a. **Thinking-cap cut** (finish=length, no final code, cap sent, completion >= cap-16): keep a tail of the reply (the last 4,000 characters of reasoning and the last 1,500 of content), add THINK_CAP_NOTE, continue. Off in cell 46.
     b. **Window cut** (finish=length, no final code): keep the same short tail and set `window_hit`, so the next request is preceded by a forced trim. The model is told nothing (1517-1527).
     c. **Review end** (no tool calls while a review is running): the result is kept, changed or no_code. A changed answer becomes a check record (attempt-2 candidate). Break.
     d. **No tool calls = final answer** (1545-1579). If `not over and recoveries < recovery_max (5)`, the harness looks for a problem with `_final_problem`:
        - empty reply, no code block, syntax error, or no `solution` defined;
        - otherwise `_verify_final`: `_VERIFY_CODE` runs the final code in a **fresh namespace inside the live REPL** that contains only `puzzle`. Each train example must equal its output exactly, and the test inputs must run without error and return ints. The timeout is `verify_timeout=90` s.

        If a problem is found, the recovery runs: `head = base_head + [failed final answer (text only, at most 12,000 characters, thinking dropped), feedback]`, **`entries.clear()`** (every working step is dropped, the REPL state is kept), `rebase=True`, continue. A newer failure replaces the older pair.

        If there is no problem and `final_review` is on and at least `review_min_left` (8,000) tokens are left, one review round starts. Otherwise break with this answer.
     e. **Tool calls**: append the assistant entry and run every tool call in order. Top-level import lines are recorded for salvage. Tool output is capped at 4,500 characters (the first 2/3 and the last 1/3 are kept, the middle is cut). Each `check:` line updates `best_check`. If a check line appeared, `check_attempts.json` is rewritten. If `over`, break after running the tools (the model never sees their output, but the last check() is recorded). A budget note can be appended. The tool messages are appended, then any new show() PNGs are appended as one user message with images.
4. **After the loop** (1643-1670):
   - If a review ran, the **first** final answer is the response.
   - On a deadline with no review, `salvage_at_deadline` builds a program from every non-baseline function in the REPL plus the default imports and the recorded imports (only if it defines `solution`). This is off in cell 46.
   - If `check_attempts` and `check_helper` are on, `_ck_final_run()` checks the current `solution` silently, then `_ck_final_dump(top=6)` writes `check_attempts.json` `{token_limit, deadline, records}`.
   - The transcript end and the V3_STATS log line are written.

### 2.4 Trimming

- `trigger = context_window - safety_margin` = 32768 - 1024 = 31,744 estimated tokens (1180).
- `target = max(trim_target_tokens=16384, base_tokens + 2500)` (1186). On deeper levels, the trigger is multiplied by (1 - 0.1*level) and the target by (1 - 0.25*level), with a floor of base + 1500.
- The oldest entries are dropped in whole assistant-turn blocks. The last exchange is always kept. The head (system message plus the puzzle message plus any recovery pair) is never trimmed.

### 2.5 Tools

- Exactly one tool: `REPLToolWithProtection(name="python", timeout=120, protected_variables=["puzzle"], include_strict=False)` (1780). It runs in `LocalKernelSandbox`, a stateful Jupyter kernel.
- Inside the REPL: `puzzle`, `show`, `show_part`, `check(fn=None, max_cells=12)`, and the grid helpers `rot90`, `flip_lr`, `flip_ud`, `transpose`, `dihedral`, `symmetries_between`, `crop`, `bbox`, `components`, `helper_source`. A hidden layer holds `_ck_record`, `_ck_final_run`, `_ck_final_dump`, `_ck_check_code`, `_pv_verify`, and `_pivot_baseline` (which excludes harness names from describe and salvage).
- `check()` prints the following, then records the function's test outputs in `_CHECK_RECORDS` (keyed by the JSON of the test outputs, so one record per distinct output; it keeps the best ok/pct and the code, at most 6,000 characters):
  - for each training example: OK, the wrong shape, an error with its line, or up to 12 wrong cells as (row,col) got/expected, with an x/. map of the wrong cells when there are more than 12;
  - for each test input: its size and colours;
  - a summary line `check: k/n training examples right | p% of training cells right`.

### 2.6 Thinking budget, max tokens, sampling

- `chat_template_kwargs = {"enable_thinking": True, "reasoning_effort": ARC_REASONING_EFFORT}`. The **default is "low"** (1701). Allowed values: xhigh, medium, low. Cell 46 sets the actual value through `REASONING_EFFORT`.
- No max_tokens per reply. Each reply is limited only by what is left of the 32K window. The optional thinking cap (`ARC_THINK_CAP`, default 12,000 in cell 30, **0 in cell 46**) applies only to replies after the first, and only when the window has room for it.
- Per-sample decoded-token limit: `ARC_MAX_GEN_TOKENS`, default 200,000 in cell 30. Cell 33 calls `setdefault` with 100,000. The per-puzzle value comes from `ARC_RUN_LIMITS` (session 17 Kaggle rounds), where 0 means deadline only.
- Sampling (1769-1777): `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`.
- Client HTTP timeout: 3600 s (`client_kwargs={"timeout": 3600}`).
- Model name: `ARC_MODEL_NAME`, default "Qwen/Qwen3.8-Flash-Next".
- Reasoning field: `ARC_REASONING_FIELD`. The default is reasoning_content, and cell 46 sets "reasoning" for vLLM.

### 2.7 Attempts, scheduling, voting, early stop

- **Per sample (cell 30)**: one sample is one multi-turn agent conversation. It ends in one of four ways:
  - a final answer is accepted, which is the natural early stop (no forced answer);
  - the token limit is reached;
  - the deadline is reached;
  - after a review.
- **Inputs to answer selection** (cell 30 provides these, cell 46 or later decides):
  - The submitted final answer comes from the response content (pivotools reads the last ```python block).
  - `check_attempts.json` holds the top 6 records, sorted by `(ok/n, pct, seq)` in descending order (463), so the most recent wins a tie. The code comment says: "the best ones fill the two attempts of the submission when the task ends (with no final answer: attempt 1 and 2; with a final answer: attempt 2 when it differs)" (421-422).
  - With the review on, the first final answer is attempt 1 and a changed answer is a record for attempt 2.
- **Across the 120 tasks**: not in this cell. The env names point to the external design:
  - `ARC_RUN_LIMITS` gives per-round token limits;
  - `ARC_RERUN_NOTES` adds a second-pass note per puzzle, written from the first pass by a "stage manager" (session 16);
  - `ARC_DEADLINE_EPOCH=STOP_AT` sets one global stop time;
  - `cell46` mentions `TOTAL_TOKEN_BUDGET` and round-1 runs.

  How many concurrent samples run per task and how votes are combined across runs must be read from cell 46 and later cells.
- **Retries**: `max_retries=2` in the call config for malformed tool calls and bad requests. The AgenticCodingConfig sets `max_retries=0` at the turn level. Up to 6 server errors are allowed, with 3 shrink levels for context errors and up to 5 recoveries.
- **Concurrency**: none inside this cell. A sample runs its tool calls one after another; nothing runs in parallel within a sample. Running samples in parallel is up to the runner.
- **Timeouts**:
  - python tool: 120 s per execution;
  - initial code: 120 s;
  - final verification: 90 s;
  - describe: 15 s;
  - salvage: 20 s;
  - `_ck_final_run`: 60 s;
  - `_ck_final_dump`: 30 s;
  - `_record_final`: 60 s;
  - the request wait is capped by the time left before the deadline.

### 2.8 Image use

- The first message carries one PNG per training pair (input and output side by side) and one per test input, at 16 px per cell with gridlines (`ARC_IMAGES` default 1).
- `show()` and `show_part()` inside the REPL write PNGs to `show_dir`. After each batch of tool calls, the new PNGs are sent as one user message (1636-1640). `show_part` labels rows and columns with a 3x5 bitmap font, at scale >= 24 by default.
- The token estimate per image is `max(64, ceil(w/32)*ceil(h/32)+2)` (83).
- The saved trace replaces image payloads with `<omitted>`. The transcript saves every image to `images/`.

---

## 3. Constants and defaults

| Name | Value | Line |
|---|---|---|
| PALETTE | ARC standard 10 colours (0,0,0), (0,116,217), (255,65,54), (46,204,64), (255,220,0), (170,170,170), (240,18,190), (255,133,27), (127,219,255), (135,12,37) | 17-20 |
| GRIDLINE / FILL / _LABEL_COLOR | (85,85,85) / (32,32,32) / (200,200,200) | 21-22, 131 |
| render_png defaults | scale=16, gridlines=True, gap=10 | 43 |
| render_part_png | scale >= 8 (show_part default max(scale,24)); font size 2 when scale >= 14 and (digits < 2 or scale >= 20), else 1 | 185, 192, 337 |
| image_tokens | max(64, ceil(w/32)*ceil(h/32)+2); fallback 256 | 83, 90, 96 |
| _MSG_OVERHEAD | 8 tokens per message | 262 |
| _CHECK_LINE_RE | `check: (\d+)/(\d+) training examples right \| ([\d.]+)%` | 283 |
| check max_cells | 12 | 354 |
| check record code | 6,000 characters | 435 |
| _ck_final_dump top | 6 | 460 |
| DESCRIBE caps | functions 700 characters, variables 400 characters | 643-644 |
| _DEFAULT_IMPORTS | numpy as np, copy, math, itertools, collections, Counter/deque/defaultdict | 732-735 |
| k_text initial | 2.5 characters per token (calibrated, clamped to 1-4) | 781, 1149 |
| usage fallback | 3 characters per token; prompt +1,200 | 1127-1128 |
| context_window | 32,768 (ARC_CTX_WINDOW) | 951, 1787 |
| trim_target_tokens | 16,384 (ARC_TRIM_TARGET) | 952, 1788 |
| safety_margin | 1,024 (ARC_TRIM_MARGIN) | 953, 1789 |
| trim_note | True | 954, 1790 |
| max_generated_tokens | 200,000 (ARC_MAX_GEN_TOKENS; cell 33 setdefault 100,000) | 955, 1791 |
| recovery_max | 5 | 956, 1792 |
| recovery_final_chars | 12,000 | 957 |
| verify_timeout | 90 s | 959 |
| deadline_epoch / sample_seconds | 0 / 0 (env) | 960-961 |
| salvage_at_deadline | True (cell 46: 0) | 962 |
| tool_output_max_chars | 4,500 (first 2/3 and last 1/3 kept) | 964, 1057 |
| cutoff_tail_chars | 4,000 of reasoning; content tail 1,500 | 965, 1076 |
| image_scale | 16 | 967 |
| think_cap_tokens | 0 in class; 12,000 config default; cell 46 sets 0 | 973, 1803 |
| budget_late_left | 15,000 | 975 |
| review_max_tokens / review_min_left | 20,000 / 8,000 | 977-978 |
| final_review | False (cell 46 also 0) | 1806 |
| trim deeper levels | trigger*(1-0.1L), target*(1-0.25L), floor base+1500; up to 3 levels; +2000 base if nothing to drop | 1182-1188, 1455-1460 |
| target floor | base_tokens + 2,500 | 1186 |
| server errors | up to 6, 2 s sleep | 1471-1472 |
| thinking-cap detection | completion >= cap - 16 | 1504 |
| sampling | T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0 | 1770-1775 |
| reasoning_effort | "low" default | 1701 |
| client timeout | 3,600 s | 1768 |
| python tool timeout | 120 s; initial code 120 s | 1780, 1784 |
| max_retries | 2 (call), 0 (AgenticCodingConfig) | 1762, 1782 |

---

## 4. Differences from a "one long thinking call per task" harness, and what looks improvable

### Differences

1. **The model works through many short turns with code inside a 32K window, not one long thinking call.** There is no max_tokens per reply (1422). Context is managed by trimming old turns down to about 16K (1174-1210). The REPL state persists and is described after each trim. The model is told about the window in the prompt (1888-1889).
2. **Default reasoning_effort is "low"** (1701). Thinking is meant to stay short ("Do not think for long before acting", 1891). Tokens go to code/verify cycles instead.
3. **Exact verification tool `check()`** with cell-level diffs (354-418). Every check() call is recorded with its test outputs as an attempt candidate (`_CHECK_RECORDS`, 421-445). A failed or unfinished run still produces scored candidates for attempts 1 and 2.
4. **Final answers are checked by the harness**: a fresh-namespace check on train and test (663-729). A failed answer is sent back with the reason, and the working steps are wiped (`entries.clear()`, 1560) while the REPL state stays. Up to 5 times per sample.
5. **Vision**: the first message has images, and `show`/`show_part` give images on request in the middle of a run.
6. **Compact grid text** (one digit per cell, no spaces, 1928-1931) roughly halves the puzzle tokens.
7. **Hooks for a second pass**: per-puzzle rerun notes (1325-1351) and per-round token limits (1305-1323) give a staged, multi-round schedule driven by cell 46.

### What looks improvable (observations with line evidence)

1. **Default effort is low, and the "real" value is external.** If cell 46 keeps "low", the model's xhigh mode (ARC Prize: 42.4 % vs 22.8 % for low) is unused. A/B medium or xhigh inside the same REPL loop, with a larger window. Their 32K window (951) is a KV/speed choice for 2-bit on L4.
2. **The window is only 32K; trimming drops reasoning.** After a trim only the REPL is kept. Thinking and earlier tool outputs are lost (1189-1197). A larger window (64K+ on our KV budget) or a model-written summary note at trim time could keep hypotheses. Currently the note lists only function signatures (630-645).
3. **A window cut is silent.** On `finish == "length"` the model is told nothing (1517-1527) and gets only a 4,000-character tail. A short "you hit the window, run code now" note could save tokens. The thinking cap that does this (1504-1515) is turned off in cell 46.
4. **Recovery wipes all working entries** (`entries.clear()`, 1560), even when the failure is only a missing import or a not-self-contained helper. A lighter recovery that keeps the last N entries could converge faster.
5. **Candidate ranking is train-only.** `_ck_final_dump` sorts by `(ok/n, pct, seq)` (463). Records with the same score are ordered by recency only. There is no use of how many separate runs or functions agree on the same test output (`calls` is counted at 442 but not used in sorting). Cross-sample voting on identical test outputs, weighted by calls or agreement, would be a cheap gain for attempt 2.
6. **The test-input sanity check only verifies that the code runs** (720-725 check only that `_norm` succeeds). No size or colour plausibility checks are run against the training outputs. Heuristic filters (output size consistent with the train ratio, palette subset) could reject bad attempt-2 candidates.
7. **Salvage is turned off** in cell 46 (`ARC_SALVAGE_AT_DEADLINE=0`), so a run cut at the deadline relies only on check records. That is fine if records exist. A run that never called check() gets nothing, although `_ck_final_run` (448-457) does try `solution` at the end.
8. **Final review is off** (1806, cell 46). The comment says it was not measured. The review prompt (273-282) targets test-input edge cases, a known ARC-AGI-2 failure mode. It is a candidate arm.
9. **The image token estimate is fixed** at a 32x32 merge (83). It is used for trimming estimates only, so it is low risk, but at 16 px per cell a 30x30 pair image is about 960x480 px, about 452 tokens per image, repeated in every request until trimmed. That cost is high in a 32K window. Lowering the scale or dropping first-message images after turn 1 is possible, but the head is pinned and is never trimmed (1438).
10. **Tool output is capped at 4,500 characters by cutting the middle** (1053-1059). check() maps of wrong cells for a 30x30 grid fit, but several prints in one call may lose the middle.
11. **The python tool runs one call at a time inside a sample** (1591-1620). That is fine. Throughput comes from running many samples at once on the server, which is decided outside this cell.
12. **The system prompt says "time out after 120.0 seconds"** and matches the tool timeout (1780), so the two are consistent. The prompt claims libraries (ortools, z3, ...) whose presence in the Kaggle image is not verified in this cell.
