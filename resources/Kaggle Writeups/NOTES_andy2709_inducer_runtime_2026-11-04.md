# andy2709 "ARC Flash Next W4A16 Inducer" (public 74.31): runtime notes (read in full, 2026-11-04)

## Sources and how they were read
- **Kernel script:** `andy2709_arc-flash-next-w4a16-inducer.py` (1,470 lines), abbreviated **K** below.
  - It is byte-identical to `andy2709_inducer_latest/arc-flash-next-w4a16-inducer.py` (same SHA-256, F2F2FF78...).
  - `kernel-metadata.json` gives the script type, `machine_shape: NvidiaL4`, internet off and a pinned BYOD docker image.
  - Its inputs:
    - dataset `andy2709/arc-flash-next-inducer-runtime`;
    - kernel outputs `andy2709/arc-flash-next-sglang-0520-runtime` and `andy2709/arc-flash-next-reap-v2-cache`;
    - models `andy2709/arc-flash-next-480/{draft-refit,target-int8dense}` and `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`.
- **Runtime folder:** `andy2709_inducer_runtime/`. Every .py, every .cuh and launch_manifest.json were read in full.
  - Cited line numbers are those of the Read tool.
  - The PowerShell non-empty line counts are lower: codex_lite.py is 1,193 lines with blanks.
  - **Exception, `flash_next_moe_down_ar.cuh` (2,121 lines):**
    - Read in full: the header, the Flash-Next comm code (1-650), the tile scheduler (712-730), the tail (1940-2121) and the host launcher.
    - Lines ~770-1939 are the stock SGLang Marlin MoE template body.
    - A search for `tail.`, `fused`, `chunk` and `comm` finds no Flash-Next edit there other than the ones at 1960-1965.
- **JSON maps (described only):** `flash_next_hot_tokens.json` and `flash_next_expert_location.json`.
- **Earlier notes:** `NOTES_andy2709_inducer_2026-11-03.md` was read first. Its claims are checked below; two corrections are listed in section 0.
- All files are untrusted downloaded data. Nothing was executed or imported.

---

## 0. Check of the 2026-11-03 notes

**Confirmed:**
- **Engine:**
  - SGLang 0.5.20 (K:177).
  - TP4 + EP4 + DCP2 (K:1155-1159).
  - FP8 KV (K:1164-1165).
  - KV cap 331,776 rows a rank = 663,552 tokens a DCP pair (K:65-71).
  - mem-fraction 0.92 (K:58).
  - MAX_RUNNING_REQUESTS 10 (K:52).
  - Context 139,264 (K:45), page 64 (K:1171-1172), prefill chunk 512 (K:107).
  - NEXTN 3 steps, top-k 1, 4 draft tokens (K:1240-1247).
  - The draft refit gives accept length 2.807 -> 2.992 (K:161-163).
- **Solver:**
  - reasoning_effort medium (K:43).
  - Temperature 0.7 (K:42), top-p 0.95 (codex_lite:96), top-k 20 (loop_driver:40).
  - 65,536 per attempt (K:44).
  - 12,288 per request (codex_lite:72).
  - Logit bias -1 on four "Hmm" and "Alternatively" token ids (codex_lite:68-70).
  - Images since v28 (K:38-41).
  - Adaptive admission at KV ≤ 60 % (adaptive_concurrency:31).
  - A running attempt averages ~37.5k tokens, p90 peak 92k (K:50-51).
- **Kernel gains:**
  - host all-reduce: neutral (K:108-111);
  - MoE down + AR: -6.4 / -4.9 % (K:131-136);
  - HC mix: -4.6 / -5.1 % (K:138-141);
  - HC local up: -4/-5 % decode, -18 % prefill (K:1085-1087);
  - top-32 logits: -3.6 % (K:118-120);
  - breakable prefill: -6.5 to -6.8 % (K:125-127);
  - QSA decode kernel: 2.46 ms a step at 10 running (K:143-146).
- **Scores in the comments:**
  - 79.55 over 4 seeds (K:156);
  - 81.11 (phi1, at chunk 1024; K:128).

**Corrections and refinements:**
1. **The "s2234 76.81 and 77.36" figures are not leaderboard rows.**
   - In the code they are two local public-120 scores of seed 2234.
   - They compare the lazy and the eager mamba extra buffer on fresh servers (K:59-62: "fresh servers scored lazy and eager alike (s2234 76.81 and 77.36)").
   - The earlier notes' link to LB rows 11 and 9 is not supported by the code.
2. **"Only 10 running" is also the attempt ceiling, not only the server's.**
   - The runner's adaptive limiter starts at 4 and caps at 10 (K:853-859).
   - It has exactly 10 task workers (adaptive_concurrency:519).
3. **"Second attempts go first to tasks whose first attempt did not verify" is right, with two details.**
   - Every task still gets a second attempt, verified or not; verified tasks are only ordered later (kaggle_runner:466-472).
   - The carry happens only after an unverified first attempt (codex_lite:1166).
4. **The 65,536 cap counts every generated token of the attempt.**
   - That is all turns: reasoning, replies and tool-call text (codex_lite:1081-1083, `completion_tokens_used += accounted`).
   - It is not a thinking-only cap.
5. **The speed (~400 tok/s) is my estimate, not a number in the code.**
   - Decode-step numbers in the code: 66.9-77.9 ms at 8-10 running across versions.
     - K:133-141: 68.5 / 76.3 ms after the MoE fusion, 66.9 / 73.9 ms after the HC mix.
     - flash_next_serving:37: "85 ms at 10 running" before HC local up.
     - flash_next_serving:69: "78 ms step at 10 running" before top-k logits.
   - With accept length ≈ 3: 10 × 3 / 0.074 s ≈ 405 tok/s. This is my arithmetic.

---

## 1. Solver and harness, exactly

### 1.1 Policy in use
- **Mode:** `ATTEMPT_MODE = "codex_lite_cap12k_images"` (K:38). It maps to two identical policies, both `CODEX_LITE_IMAGES_POLICY` (kaggle_runner:46).
- **Policy name:** `codex_lite_compact_v2_medium_alt_tools_carry_nofiller_cap12k_images` (launch_manifest:3-6; built at codex_lite:42-60).
- **Lineage, from the comments:**
  - v14-v20 `codex_lite`: second attempts fresh.
  - 2026-10-19 alternates + arc_tools (codex_lite:40-42).
  - v21 carry (codex_lite:43-45).
  - v22 filler bias: "cut generated tokens 4.5 %" (codex_lite:46-49).
  - v27 request cap 24,576 -> 12,288: "6.9 % fewer tokens, +1.1 to +2.2 under Kaggle-sized budgets" (codex_lite:50-53).
  - v28 images (codex_lite:57-60).
  - Retired experiments listed in the comments: "filler2, phase caps, the 8k cap, the 'act' carry note, low/xhigh effort, dropped reasoning" (codex_lite:62-64).
  - A candidate not used on Kaggle: `carrycap` (codex_lite:54-56), a 4,096 cap on a carried attempt's first request, "saved ~4 % of their tokens at equal credit".

### 1.2 Messages sent to the model (codex_lite:985-992)
The conversation is `[system, user(environment), user(task)]`.

**System prompt** (codex_lite_interface:11), verbatim:
> You solve one ARC grid-transformation task through the Codex execution loop.
>
> Use only exec_command for local bash/Python and write_stdin to poll a running command. Python, NumPy and SciPy are installed. Network access is disabled. Edit files with bash or Python. A description of a command does not execute it; issue an actual tool call whenever computation or a file change is needed.
>
> Infer one general transformation from all visible training pairs, and work in code. After a brief first look, write solution.py with transform(grid) for your current best hypothesis, returning a rectangular list of lists of integer colors 0 through 9. Execute it on every training input, compare with the supplied outputs, and repair mismatches; revise the program rather than re-deriving the rule in prose. Compute grids, objects and coordinates with Python instead of transcribing them in your reasoning, and keep each reasoning phase short before the next tool call. Hidden test answers are unavailable.
>
> Keep tool observations concise: print dimensions, object summaries, per-example pass/fail, mismatch counts, and a few mismatching coordinates. Avoid reprinting grids already in the prompt. Save large diagnostics to local files and inspect selected rows or regions when needed; preserve useful error messages.
>
> Write predictions.json containing one predicted output grid per test input in the original order. Execute a final check that the file parses, contains the required number of grids, and uses rectangular integer grids with colors 0 through 9. Continue working until the required files exist and have been checked. If the rule remains uncertain, save your best candidate before finishing. Send a brief final reply only after the files are saved; an intention to investigate or write code is not completion. For a preflight, produce the specific files requested by that preflight instead.
>
> For this ARC run the only callable tools are exec_command and write_stdin. Edit files by running bash or Python through exec_command. The apply_patch function tool is unavailable; this tool profile overrides earlier generic editing-tool instructions. Complete the user's requested files before sending a final answer: solution.py defining transform(grid), and predictions.json containing every test output in order. Validate the saved predictions.json structure after writing it. For a preflight task, create the exact files that preflight requests instead. Use only local computation; network access is disabled.<permissions instructions>
> Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is `workspace-write`: The sandbox permits reading files, and editing files in `cwd` and `writable_roots`. Editing files in other directories requires approval. Network access is restricted.
> Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, commands will be rejected.
>  The writable roots are `/tmp`, `/workspace/task`.
> </permissions instructions>

- **Source of the prompt:** generated from a recorded Codex CLI 0.162.1 attempt (`arc_compact_v2` profile), "so the light loop shows Flash-Next exactly what Codex showed it" (codex_lite_interface:1-8).

**Environment message** (codex_lite_interface:13):
- An `<environment_context>` block: cwd `/workspace/task`, shell bash, `current_date` (today's UTC date, codex_lite:986) and timezone Etc/UTC.
- A filesystem permission profile: read root; write `/workspace/task`, `:slash_tmp` and `:tmpdir`; read `.git`, `.agents`, `.codex` and `.aws` under the task directory.

**Task message.** It is TASK_PREFIX + " " + ALTERNATE_TASK_NOTE + " " + TOOLS_TASK_NOTE + "\n\n" + the grid rendering (codex_lite:987-988). All Kaggle policies have alternates and tools on (codex_lite:79-81).
- TASK_PREFIX (codex_lite_interface:15):
  > Solve this ARC task using the Codex coding harness. Work independently to completion. Only bash/code execution and local file editing tools are available. Network access is disabled. Python, numpy and scipy are installed. The same task is in puzzle.json. Infer a single general rule from all visible training pairs, write solution.py with transform(grid), and test it on every training example. Then write predictions.json as a JSON list of predicted output grids, one per test input in the given order. Do not put training outputs in predictions.json. Hidden test answers are unavailable. Keep useful code and results in this directory; finish with a brief completion message.
- ALTERNATE_TASK_NOTE (codex_lite:88-91):
  > If several programs or rules reproduce every training pair but give different test outputs, keep the one that treats every object uniformly with the fewest special cases in predictions.json, and save the strongest different candidate as predictions_alt.json in the same format.
- TOOLS_TASK_NOTE (codex_lite:83-87):
  > arc_tools.py in this directory has tested grid helpers (load, show, diff, check, compare, write_predictions, objects, conn_diff, fragments, bbox, crop, lines, split_by_lines, ray, background_report, dihedral, shape_rules); python3 -c 'import arc_tools; print(arc_tools.__doc__)' lists their signatures. Import them in your scripts and solution.py instead of re-implementing grid utilities.

### 1.3 How a grid is shown: text and images
**Text** (loop_driver:43-75, "NVIDIA-style layout"):
- The rendering starts with `Please solve this ARC-AGI problem:` and `**Allowed Tools:** ['exec_command', 'write_stdin']`.
- Then for each pair: `Train Example i:`, blank, `Input:`, the rows, blank, `Output:`, the rows, blank.
- Then `Test Input:` (or `Test Input k:` when there are several tests).
- Rows are digits separated by spaces, one row per line.

**Images, since v28** (codex_lite:874-912):
- The task message becomes a content list: the text part, then `IMAGE_NOTE` ("The same grids as images (10 pixels per cell; colors 0-9 are black, blue, red, green, yellow, grey, magenta, orange, azure and maroon):"), then a labelled PNG for each train input, train output and test input.
- PNG format:
  - 10 px per cell, a one-pixel dark line (85,85,85) on the far edges of each cell;
  - the standard ARC palette (codex_lite:875-878);
  - written as pure-Python PNG with zlib, no imaging library (codex_lite:883-896).
- The server loads the vision tower: there is no `language_model_only` override in image mode (K:1212-1213).
- Server flags `--mm-feature-transport cpu` and `--image-processor-backend pil` (K:1214-1217).
- The local renderer expands each image pad to the Qwen2-VL resize token count (loop_driver:176-200), so the prompt-ID check still holds.

### 1.4 Tools and sandbox
**Tools.** Two Codex-schema function tools, `exec_command` and `write_stdin` (codex_lite_interface:17-117).
- `exec_command` parameters: `cmd` (required), `justification`, `login`, `max_output_tokens`, `prefix_rule`, `sandbox_permissions`, `shell`, `tty`, `workdir` and `yield_time_ms`.
- Only `cmd`, `workdir`, `yield_time_ms` and `max_output_tokens` are used by the implementation (codex_lite:548-560).
- Requests set `tool_choice="auto"` and `parallel_tool_calls=True` (codex_lite:1065).

**exec_command** (codex_lite:548-560):
- Runs `bash -c <cmd>` in the per-attempt work directory, after path rewriting.
- `yield_time_ms` defaults to 10,000 and is clamped to 250-30,000.
- `max_output_tokens` defaults to 10,000 and is clamped to 0-10,000 (OUTPUT_TOKENS, codex_lite:107).
- If the command is still running at the yield, the reply is "Process running with session ID N" (codex_lite:538-546).

**write_stdin** (codex_lite:562-579):
- Polls or writes to a session.
- Yield: 250 ms default with chars, 5,000 ms without; maximum 30,000 or 60,000 ms.

**Observation format** (codex_lite:522-529):
- `Chunk ID`, `Wall time`, the status (`Process exited with code N` or the running session id), `Original token count` (chars/4) and the output.
- Above the token limit the middle is cut: `max_tokens*2` characters are kept at each end, with "…N tokens truncated…".

**Path views** (codex_lite:410-473):
- The model sees `/workspace/task` and its own `/tmp`. Both are rewritten inward in commands and outward in outputs.
- Reason given: on a shared /tmp, attempts imported each other's `/tmp/solution.py` (codex_lite:413-418).
- The work directory holds `puzzle.json` (train plus test inputs) and `arc_tools.py` (codex_lite:427-429).

**Isolation:**
- Under root, each attempt gets a private uid from 70,000 + (pid % 50) × 1,024 (codex_lite:123-125, 446-450).
- Without root, all of an attempt's commands share private user, network, mount and PID namespaces with their own /proc (codex_lite:232-266).
- Network is disabled where the host allows a private network namespace (codex_lite:316-319).
- The launcher preflights the sandbox before loading weights: under the attempt uid it must import numpy and scipy within 300 s, or the notebook fails (K:100-102, 404-431, 1118-1132).

**Limits per command** (codex_lite:116, 121-122):
- RLIMIT_AS 8 GiB, CPU 600 s, FSIZE 64 MiB, NOFILE 512.
- `COMMAND_WALL_SECONDS = 900`: a command still running past 900 s is SIGKILLed the next time it is collected.
- Env: OMP/OPENBLAS/MKL threads = 1; PYTHONPATH limited to the numpy/scipy site directories (codex_lite:130-131, 467-473).

**arc_tools.py** (343 non-empty lines, "tested grid helpers"):
- `load`, `show`, `diff`, `check(transform)` (runs every train pair, prints ok or diffs, reports test shapes).
- `compare({name: fn})`: which train pairs each candidate passes, and where the passing ones disagree on tests.
- `write_predictions(transform)`: validates and writes predictions.json.
- `background`, `background_report`, `objects(grid, conn, bg, by_color)`, `conn_diff`, `fragments`, `bbox`, `crop`, `lines`, `split_by_lines`, `ray`, `dihedral` / `find_dihedral`.
- `shape_rules`: fits output H and W as a·feature + b over about 20 grid features, with test predictions (arc_tools:1-21, 373-406).

### 1.5 Attempt loop and budgets (codex_lite:954-1193)
**Budgets:**
- `cap = completion_token_cap` = 65,536 on Kaggle. It comes from `--teacher-token-cap` (K:831-832) and may only lower `CODEX_LITE_TOKEN_CAP = 131_072` (codex_lite:93, 975-977).
- `STEERING` (codex_lite:97-98): `request_cap` 24,576, replaced by 12,288 for the cap12k and images policies (codex_lite:72, 980); `reminder_after` 32,768; `finalize_reserve` 8,192; `truncation_continuations` 6.

**Per-request limit** (`request_limits`, codex_lite:806-811):
- If `remaining ≤ 8,192 + 4,096` (12,288): limit = remaining and **finalizing** (thinking off).
- Otherwise limit = `min(remaining - 8,192, 12,288)`.
- Then `limit = min(limit, context_window - prompt_len - 256)` (CONTEXT_MARGIN, codex_lite:101, 1057).
- If the limit is ≤ 0: end_reason `context_guard` (codex_lite:1058-1060).

**Notes injected as user messages** (codex_lite:1037-1052), each once:
- `finalize` on entering the reserve:
  > Harness note: only {cap - used} generated tokens remain. Do not analyze further. Immediately write solution.py with your best current transform(grid) and predictions.json for every test input, then check both files.
- `reminder` once ≥ 32,768 tokens are used and no solution.py exists:
  > Harness note: {used} of {cap} generated tokens are spent and solution.py does not exist yet. Stop analyzing in prose. Write solution.py with transform(grid) for your current best hypothesis now, run it on every training pair, and refine it from the mismatches.

**Request** (codex_lite:1061-1068):
- `chat.completions.create(model, messages, tools, tool_choice="auto", parallel_tool_calls=True, temperature, top_p, presence_penalty=0.0, seed, max_tokens=limit, timeout=min(request_timeout, deadline-now), stream=False, logit_bias)`.
- `extra_body` = `{top_k: 20, min_p: 0.0, reasoning_effort: "medium", chat_template_kwargs: {enable_thinking: not finalizing, preserve_thinking: True}, return_prompt_token_ids: True}` (loop_driver:358-367).

**Prompt contract:**
- Every request is rendered locally with the model's chat template (`preserve_thinking=True`, loop_driver:270-278).
- If the server's prompt token ids differ, a `PromptContractError` is raised. The only allowed difference is image-pad positions (loop_driver:127-156; codex_lite:1069-1074).
- Usage `completion_tokens > max_tokens` also raises (loop_driver:328-331).
- The reasoning of earlier turns is kept in the messages (`reasoning_content`, codex_lite:1086-1088) and rendered back (`preserve_thinking`).

**Truncation gate** (codex_lite:1077-1080, 1098-1099):
- A reply with `finish_reason == "length"` never executes its tool calls (Codex behaviour).
- Only native tool calls count; calls written as text are never recovered (loop_driver:337-355).

**Tool turn** (codex_lite:1101-1111):
- Each call runs and its observation is appended as a `tool` message.
- Then `provisional.refresh()` runs, and the loop continues.

**Reply without a tool call** = "Codex's Stop event". It triggers `inspect_completion` (codex_lite:773-803), which needs all of:
- solution.py and predictions.json present;
- predictions.json with exactly n_test valid grids (1-30 × 1-30, ints 0-9);
- a replay of `transform` on every train input passing all of them;
- the replay's test outputs equal to predictions.json.

What happens next (codex_lite:1112-1141):
- If the check passes: end_reason `complete`.
- If the reply was truncated and fewer than 6 truncation nudges were used, this nudge is appended (as a `<hook_prompt ...>` user message, codex_lite:106, 1141):
  > Your reasoning reached the {limit}-token limit for one response and was cut off. Stop analyzing in prose. With exec_command, write solution.py now with your current best transform(grid), run it on every training pair, and refine it from the mismatches; then write and validate predictions.json. You have at most {remaining} generated tokens left in the original budget.
- Otherwise, up to 3 completion continuations (COMPLETION_CONTINUATIONS, codex_lite:102) send:
  > Completion check k/3 failed. This is visible-example and saved-file feedback only; no hidden answers were used.
  > - {issue lines}
  > Continue using exec_command and write_stdin. Repair the program, run visible training checks, and write and validate predictions.json before finishing. You have at most {remaining} generated tokens left in the original budget. If uncertain, save your best candidate now.
- After that: end_reason `continuation_limit`.

**Other ends:**
- `completion_token_budget` when used ≥ cap.
- `deadline` when the per-attempt wall deadline passes (codex_lite:1027-1033).

**Wall-clock per attempt:**
- `deadline = started + time_budget`, with started taken after the admission permit is acquired (loop_driver:528-532).
- The Kaggle launcher passes `--time-budget 14400` and `--request-timeout 14400`, i.e. 4 h each (K:847-850). The default `REQ_TIMEOUT` 900 s (loop_driver:37) is overridden.

### 1.6 Which answer an attempt returns (codex_lite:1142-1164)
1. `final = inspect_completion(...)`; `outputs` = predictions.json if structurally valid.
2. If the replayed test outputs are all valid and (predictions.json is missing, or **the program passes every train pair and its replay differs from predictions.json**), the replay outputs replace them (source `replay`).
3. Else, if there are still no outputs, take the last provisional answer (`provisional_*`).
4. Else take a partial replay, with `None` where the program raised (`replay_partial`).

**Provisional answers** (codex_lite:642-713):
- While the attempt runs, each changed solution.py is replayed, at most once every 60 s with a 20 s timeout (codex_lite:119-120).
- The current published answer is:
  - the replay of the latest solution.py when it passes all train pairs and differs from the saved predictions.json;
  - else the latest valid predictions.json;
  - else the best-train-passing replay.
- It is published to the runner at once (`on_outputs` -> `persist`), so a hard stop keeps it (codex_lite:118; kaggle_runner:427-431).

**Alternates** (codex_lite:716-770):
- In the thinking-off finalize window, each changed solution.py that passes every train pair has its test outputs stored (`finalize_turn`, codex_lite:1053-1054).
- For each test the alternate is the newest stored passing output that differs from the final answer, else `predictions_alt.json` when valid and different.
- Comment (codex_lite:720-722): "On the public tasks (2026-10-19) 9 of 91 alternates were right where the answer was wrong, 8 of them from predictions_alt.json."

### 1.7 Verification rule ("verified", used for scheduling, not for acceptance)
- `primary_verified` (kaggle_runner:94-108): true when an attempt has no error, `best_n_passed ≥ n_train` (the final program passes every visible train pair) and every test output is non-None.
- Comment: "On the public tasks such an attempt was correct 83 times in 100 (73 in 102 with REAP v2). With REAP v2 a second attempt after an unverified one gave 4.6 solves per million tokens, and after a verified one 2.6."
- Answers are always submitted, verified or not. "Verified" decides only:
  - (a) the order of second attempts;
  - (b) whether a carry is written (codex_lite:1166: `carry_out ... and not (fits and not missing)`).

### 1.8 Attempt structure, carry and "secondary unverified first"
**Structure:**
- Two attempts per task, `primary` (sample 0, seed 1234) and `secondary` (sample 1, seed 1235) (kaggle_runner:84-91; K:103, 1327; loop_driver:536).
- Each phase calls `solve_puzzle` with one sample (`sample_indices=[phase.sample_index]`, kaggle_runner:427).
- The **same seed** is passed on every request of an attempt (codex_lite:1066, `seed=seed`).

**Order** (kaggle_runner:456-472):
- All first attempts in **sorted task-id order** (`order_task_keys` sorts, rollout_runner:59-66).
- Then second attempts as workers free up. There is no barrier: a second attempt may start while the last first attempts still run.
- `CoverageFirstAdmission` makes every waiting second attempt wait until every first attempt has been admitted or ended (kaggle_runner:111-157).
- With `--secondary-unverified-first` (K:861), the ready list (first attempt finished) is filtered to tasks whose first attempt was not verified. If none, any ready task is taken, in sorted id order. So **every task gets a second attempt**; verified ones come last.

**Carry, when the first attempt ends unverified** (codex_lite:1166-1181):
- The first attempt sends one more request, thinking off, max 512 tokens (RECAP_TOKENS). It goes as a `<hook_prompt>` user message, tools still offered (codex_lite:110-113, 820-865):
  > Harness note: pause here. Do not call tools and do not analyze further. In at most 150 words, recap your work so far: each transformation rule you considered, which training pairs its program passed or failed and how, and the idea you would try next.
- It then saves its messages, the work directory and /tmp (files up to 16 MiB, codex_lite:127), the recap, the train pairs passed, the tests without output and the end reason.
- The second attempt (sample 1, `carry_in`, loop_driver:545) starts from those files. Its context (`carried_history`, codex_lite:868-871) is:
  - the first three messages;
  - every assistant tool call and tool result of the first attempt, **with the reasoning and plain replies removed**;
  - one carry note (codex_lite:927-951):
    > This is a second attempt at the same puzzle. The workspace keeps every file of the first attempt, and the tool calls and outputs above are that attempt's work; its reasoning was removed. The first attempt {ran out of its token budget|ended} {with solution.py failing training pairs [..] (zero-based) | with solution.py passing every training pair but giving no valid output for test inputs [..]}. Its own recap:\n{recap}\n\n[Training check of the saved solution.py:\ntrain i: ok | shape HxW vs HxW | N cells differ: (r,c) a->b, ... (first 12)]\n\nThese failures may come from a bug or from a misread rule. Decide whether to repair solution.py or replace its rule, then write and validate solution.py and predictions.json. You have {cap} generated tokens.
- The per-pair diff block appears only when ≥ 50 % of the train pairs passed (CARRY_NEAR_MISS 0.5, codex_lite:114, 940).
- The carried attempt gets a **fresh full cap** of 65,536.
- A broken carry falls back to a fresh start (codex_lite:1010-1017).
- A second attempt after a verified first starts fresh, because no carry state exists.

### 1.9 How attempt_1 and attempt_2 are filled (kaggle_runner:194-264; loop_driver:374-403)
**Candidates from one attempt** (`submission_candidates`):
- The attempt's output, plus its own alternate when the alternate differs.
- `_vote_top2` over a single sample gives one grid; the second place is filled only from the alternate.

**Initial state:** both slots are `[[0]]` for every test, and are written before setup (K:938-944; kaggle_runner:68, 168-175, 377-378).

**Primary result** (attempt_index 1, `mirror_missing=True`):
- `attempt_1` = the answer.
- `attempt_2` = the attempt's alternate if it differs, else a copy of the answer.
- The tests it filled become "protected", and its alternates are remembered (kaggle_runner:255-264).

**Secondary result** (attempt_index 2):
- `attempt_2` = its answer.
- If that equals the protected `attempt_1`, `attempt_2` takes the first differing grid from the primary's alternates, then the secondary's own alternates (kaggle_runner:232-234).
- If the primary left a test unfilled (not protected), the secondary also mirrors into `attempt_1` (kaggle_runner:236-237).

**When the slots change:**
- **Provisional outputs during a run** go through the same `persist` with a single candidate (kaggle_runner:429-431).
- So a running secondary overwrites `attempt_2` as soon as it has a provisional answer. The secondary's provisional answer replaces the primary's alternate in `attempt_2`.
- Every update rewrites `submission.json` atomically (tmp + `os.replace`, kaggle_runner:160-165, 406-410).

### 1.10 Sampling, logit bias and effort (summary)
- **Sampling:** temperature 0.7, top_p 0.95, top_k 20, min_p 0, presence_penalty 0 (codex_lite:95-96; loop_driver:40-41, 358-367; codex_lite:1066).
- **Effort:** `reasoning_effort = "medium"` for every Kaggle policy (codex_lite:75-76, 94; K:43). Low and xhigh were tried and retired (codex_lite:62-64).
- **Thinking:** on, except in the finalize window and the recap (codex_lite:1056, 1061, 834-845).
- **Logit bias** `{77264 "Hmm", 85152 " Hmm", 88842 "Alternatively", 37201 " Alternatively"}: -1.0` (codex_lite:68-70).
  - "acting as -1/T" (codex_lite:46-48).
  - Each id is checked to decode to its text before use (loop_driver:406-414).
  - Applied on every request, also verifying drafts (rollout_runner:26).
- **Server side:** the reasoning parser `qwen3`, the tool parser `qwen3_coder` and `--default-chat-template-kwargs {"preserve_thinking":true}` (K:1218-1223).

### 1.11 The 12 h deadline
- **Budget:** `KERNEL_BUDGET_SECONDS = 11*3600 + 55*60` = 42,900 s from the start of the script (K:32-33).
- **Server start:** startup has no fixed limit ("on Kaggle's GCSFUSE dataset mounts loading can take hours, and a late start still leaves inference time", K:1301-1304). Startup past the budget raises `TimeoutError`.
- **Runner deadline:** `kernel_started + 42,900` (K:1349). On reaching it the launcher logs "wall budget reached; retaining incremental submission", SIGTERMs the runner's process group (30 s grace, then SIGKILL) and stops the server (K:1361-1368, 608-616, 1379-1380).
  - There is **no drain or no-new-admission phase before the deadline**: attempts in flight are killed.
  - What they leave is their provisional answers, already in `submission.json`.
- **After the stop:**
  - `validate_submission` checks the schema (K:717-737, 1456).
  - On the competition rerun (`KAGGLE_IS_COMPETITION_RERUN`) the test challenges are used and there is no smoke or post-verify (K:894-910, 1436-1438).
- **Failure paths:**
  - If SGLang dies during evaluation, the runner is terminated and a RuntimeError is raised (K:1355-1360). The notebook then ends in an error with the incremental submission on disk.
  - My inference: Kaggle may count that run as failed.
  - A runner exiting non-zero also raises (K:1376-1378).
- **Ordinary (non-rerun) saves** run a 20-minute smoke: 16 easy public tasks, first attempts only, and at least one finished attempt is required (K:78-90, 1330-1335, 1403-1433).

---

## 2. Scheduler and concurrency

### 2.1 Runner settings from the launcher (K:851-863)
- `--concurrency 10 --adaptive-concurrency --adaptive-min-inflight 1 --adaptive-initial-inflight 4 --adaptive-max-inflight 10 --adaptive-poll-interval 5 --secondary-unverified-first`.
- `local_adaptive_runtime` builds the controller with `min_burst=1` and a `LoadMonitor` polling SGLang every 5 s. `task_workers = max(10, 10) = 10` (adaptive_concurrency:491-523).
- One permit wraps one whole attempt, all turns and tool runs included, not one HTTP request (adaptive_concurrency:81-87; loop_driver:528-570).

### 2.2 The controller, `capacity_admission_v3` (adaptive_concurrency)
- **Origin:** adapted from prime-rl's `orchestrator/concurrency.py` (adaptive_concurrency:9-11).
- **Constants** (25-41):
  - GROWTH_CLEAR_POLLS 2;
  - GROWTH_STEP_DIVISOR 8;
  - GROWTH_BETWEEN_TURNS_DIVISOR 20;
  - KV_USAGE_GROW 0.6;
  - KV_USAGE_SOFT_CAP 0.8;
  - KV_USAGE_HARD_CAP 0.9;
  - KV_USAGE_TARGET 0.7;
  - KV_TRIM_COOLDOWN_POLLS 6;
  - QUEUE_RATIO 0.5;
  - QUEUE_PERSISTENCE_POLLS 6;
  - QUEUE_CUT_FRACTION 0.9;
  - PREEMPTION_CUT_FRACTION 0.8;
  - ADMISSION_WINDOW_S 5.0.
- **Load source** (`SGLangLoadSource`, 400-470):
  - It reads `/v1/loads?include=core,queues`: `token_usage` (KV use), running, waiting, `max_total_num_tokens`, `max_running_requests`, the retraction count and a timestamp.
  - A retraction delta > 0 counts as a preemption.
  - The engine ceiling is SGLang's effective `max_running_requests`, after the Mamba slots (195-202).
- **Signal** (204-231):
  - "hard" on any retraction, or on queue overload (waiting > 0.5 × running for 6 consecutive polls).
  - "soft" when waiting > 0 on two consecutive polls, or when **KV use > 0.6**.
  - Otherwise "clear".
- **Shrink:**
  - A preemption cuts the limit to 0.8 × in-flight; a queue overload to 0.9 × (246-256).
  - KV > 0.8 trims to `inflight × 0.7 / usage` ("hard" label above 0.9), then waits a 6-poll cooldown (258-262).
  - In-flight attempts are never killed; the limit drains as they finish (`draining`, 159, 242-244).
- **Grow** (264-283), only when all of these hold:
  - the signal is clear (so KV ≤ 0.6) and nothing is waiting;
  - no cooldown;
  - in-flight == current limit;
  - running ≥ in-flight − in-flight//20 (all admitted attempts visibly running: below 20 in-flight, every one must be on the server, not between tool turns);
  - the limit is below the engine ceiling.
- **Growth step:** after 2 consecutive fresh clear polls (≥ 0.9 × poll interval apart), +max(1, limit//8). That is +1 at Kaggle sizes.
- **Burst cap** (311-318): at most `max(min_burst=1, limit//10)` = 1 new admission per 5 s window. A finished attempt refunds its admission (154-156).
- **Start:** initial limit 4 (bootstrapped, so there is no capacity/max_len initial sizing, 237-240). Growth to 10 needs ≥ 6 grow steps of ≥ 2 clear polls (10 s) each.

### 2.3 What this gives on Kaggle (numbers from the comments)
- **Server profile:** `--max-running-requests 10`, decode CUDA graphs for batch 1-10, 40 Mamba slots (4 per request) (K:878-887, 1173-1182; launch_manifest:18-33).
- **Why 10:** at 12 running, free device memory fell to 0.05 GiB. At 10 the graphs for 11-12 and 8 Mamba slots go, and v28's save run kept 0.38 GiB (K:46-51).
- **Running context:** "a running attempt averages ~37.5k tokens (p90 peak 92k)" (K:50-51).
  - My arithmetic: 10 × 37.5k = 375k tokens ≈ 57 % of the 663,552-token pool, so the 60 % growth rule and the 10 ceiling bind at about the same point.
- **Graph ladder** (K:880-886): every size 1-16 has its own graph. With only 1, 2, 4, 8, 16 graphs, "9-15 running all paid the bs-16 step on L4, ~100 ms".
- **Earlier profile:** 8 running through v27 (K:46). v21's 8-running profile without DCP had 484,032 KV rows (K:70).
- **On H200 (local only):** up to 160 running under the same controller. KV peaked at 10 % at 128 running (K:72-76).

---

## 3. Engine patches (SGLang 0.5.20) and measured numbers

### 3.0 Patch mechanics (flash_next_serving:1478-1530)
- **Pinning:**
  - Every patched stock file is pinned by SHA-256 (PINS, 129-153).
  - Every anchor must occur the exact number of times, so any other SGLang fails closed.
  - The stock file is kept as `.py.before_serving`, and the patched source is compiled before writing.
- **Installed beside SGLang:** `flash_next_host_ar.py`, `flash_next_moe_down_ar.py/.cuh` (device_communicators); `flash_next_qsa_dcp.py` (as `qsa/flash_next_dcp.py`); `flash_next_hc_local_mix.cuh` (srt/layers); `flash_next_qsa_index_decode.cuh` (qsa) (115-128).
- **Environment flags** (K:1083-1104):
  - `SGLANG_FLASH_NEXT_HC_TP=1`, `HC_LOCAL_UP=1`, `NCCL_GRAPH_MIXING_SUPPORT=0`;
  - `DRAFT_SHARED_VOCAB=1`, `HOST_EMBED=1`, `HOST_AR=1`, `MOE_DOWN_AR=1`;
  - `HC_MIX_CUDA=1`, `QSA_DECODE=1`, `QSA_DCP=1`;
  - `TOPK_LOGITS=32`, `PREFILL_GRAPH=1`.

### 3.1 Always-on fixes (flash_next_serving:8-17, 249-309, 344-374)
- **Marlin MoE under EP** (gptq_kernels.py and awq_kernels.py): the runner now receives the dispatcher's `local_expert_mapping`, "so each EP rank runs only its own experts".
- **BF16 expert scales** (gptq_moe.py): `dtype=torch.half` -> `params_dtype`. Upstream allocated FP16 scales for BF16 activations.
- **PLE loader:** the self-referencing warning flag that kept a parameter snapshot alive is fixed.
- **PLE offload:** the table is built on the meta device. Stock 0.5.20 allocates each offloaded table on the GPU first, "23.84 GiB per rank at TP4", which exceeds an L4.
- **Expert routing with `--init-expert-location`** (qwen2_moe.py):
  - Stock remaps router ids only on the DeepEP path, "so the normal path ran tokens through the wrong experts".
  - The patch remaps on exactly the layers the loader placed (`_flash_next_expert_dispatch`).

### 3.2 Expert placement and draft hot tokens
**`flash_next_expert_location.json`:**
- 48 layers × 480 physical slots. "Physical slot g*120+i holds the listed logical expert", i.e. 120 experts per GPU at EP4.
- Fitted on 8,165 decode passes and checked on 4,962 held-out passes. The optimizer is pairwise swaps minimising the per-pass maximum of activated experts per GPU (40,000 proposals a layer).
- Held-out max activated per GPU: identity 37.582 -> placed 35.33 (mean 32.12, balanced bound 32.495).
- Checked against the target's REAP export hash (flash_next_serving:1533-1550).

**`flash_next_hot_tokens.json`:**
- 27,982 token ids: every id generated in 4 fit-trace runs (60,389,514 generated tokens), by frequency, plus all added tokens.
- Held-out coverage 0.999581. "The target still verifies over the full vocabulary."

**TP hot-token draft head** (flash_next_serving:22-27, 311-342):
- Upstream indexes the local vocab shard with global hot ids, valid only at TP1.
- The patch fills the owned rows, all-reduces and gives each rank an equal slice in rank-major order.

### 3.3 HC (hyper-connection) sharding, local INT8 up projection and CUDA mix
**HC_TP** (flash_next_serving:19-21, 155-247):
- Each rank keeps a lowrank slice of the mix down-projection and a hidden slice of the up-projection.
- It all-gathers `[T, lowrank]`, computes sigmoid weights and the branch mix for its hidden slice, then all-gathers `[T, hidden]`, instead of all-reducing `[T, hc*hidden]`.

**HC_LOCAL_UP** (flash_next_serving:34-39, 1131-1274):
- Every rank keeps the whole up-projection in INT8: one fp16 scale per row and 32 lowrank columns, scales stored ×1024, quantized on the CPU at load.
- This drops the `[T, hidden]` all-gather: "97 a decode step, 7.3 of 85 ms at 10 running on L4".
- Launcher comment (K:1085-1087): "Modal 4x L4: decode step -4/-5% at 8/10 running, prefill -18% a batch".
- Memory: +0.18 GB per L4 (flash_next_serving:35), but "0.25 GB a rank" in K:67-69. The two figures differ; the KV rows were cut by 11,584 to compensate.

**HC_MIX_CUDA** (`flash_next_hc_local_mix.cuh`; K:138-141):
- ForgeKernel's `hc_local_mix`, for 1-64 tokens at hc 4, hidden 2560, lowrank 320.
- Design: 80 CTAs × 128 threads, a cp.async ring, mma.sync m16n8k16 bf16, built with --use_fast_math.
- Bit-identical to the Triton kernel for t = 1..64.
- Cold-L2 L4 timing: 18.5 / 19.8 / 20.7 / 23.6 µs at t = 16/32/40/64, against Triton's 23.8 / 45.2 / 46.0 / 48.4 (cuh:1-13).
- End to end: "HC mix 3.9 -> 2.1 ms a step; decode step 68.5 -> 66.9 / 76.5 -> 73.9 ms at 8/10 running, both kernels against v30 before them -4.6 / -5.1%".

### 3.4 Host-memory all-reduce (`flash_next_host_ar.py`)
**Why** (host_ar:1-22):
- "Kaggle's 4xL4 G2 VMs have no usable peer path (0.46-0.9 GB/s per GPU when all four talk), so NCCL runs its SHM transport, whose protocol costs dominate at decode sizes."

**Design:**
- Each rank's kernel exchanges its bf16 partial through one pinned `/dev/shm` buffer that all four processes map and register (`cuMemHostRegister`).
- Synchronisation uses system-scope acquire/release flags. An epoch per GPU persists across CUDA-graph replays, and the two buffer halves alternate by epoch parity.
- A wait longer than about 9 minutes (2^40 cycles) traps.
- Kernels are compiled with NVRTC and launched through the driver API on torch's stream.
- Setup is agreed across ranks: if any rank fails, all stay on NCCL.

**Sizes** (host_ar:37-41):
- One-shot ≤ 10 KiB (two 2560-wide rows).
- Two-shot ≤ 320 KiB (64 rows): each rank sums its quarter.
- Larger, non-bf16 or misaligned tensors stay on NCCL.
- Every output element is the fp32 sum in rank order 0..3, rounded once, so all ranks hold the same bits.

**Measured** (µs per op in a graph chain, against shipped NCCL):

| Size | Host AR | NCCL |
|---|---|---|
| 1×2560 | 9.5 | 19.3 |
| 8×2560 | 28.5 | 35.6 |
| 32×2560 | 74.5 | 101.5 |
| 64 rows (two-shot) | 132.7 | 150.4 |
| 2.5 MiB (prefill) | 1005 | 831 (NCCL wins) |

- End to end: "neutral on Kaggle's L4s, decode p25 71.2 against 71.1 ms" (K:108-111). It is kept on Kaggle because the MoE-down fusion needs it (K:131-137, 1028).
- `NCCL_GRAPH_MIXING_SUPPORT=0`: "graph-mixing support only adds 17-60 us per captured collective" (K:1088-1090).

### 3.5 Fused MoE down projection + all-reduce (`flash_next_moe_down_ar.py` / `.cuh`)
**What it replaces:**
- For decode MoE layers (m ≤ 64 tokens), one launch does the routed experts' second Marlin GEMM, `moe_sum_reduce`, `fused_gate_sigmoid_mul_add` (the gated shared-expert add) and the TP4 all-reduce (moe_down_ar.py:1-17; flash_next_serving:58-65, 547-654).

**Kernel design** (cuh:1-26, 220-452, 712-730):
- Marlin's tiles without split-K: one CTA computes all of K for one 8-row MoE block × 128 output columns.
- Tiles are numbered column-chunk-major, so output chunks complete in order.
- The last `comm_ctas` CTAs (40 by default, in 20 groups) wait for each chunk, build this rank's partial and two-shot it through pinned host memory while the GEMM works on later chunks.
- They stage rows with cp.async, because "each dependent round of loads costs ~7-8 us on L4".
- v1 (one chunk at a time) "paid 22-36 us of latency per chunk".

**Exactness:** "Its tiles are Marlin's without split K (1.9e-5 of elements differ from stock Marlin in the last bits); everything after them is bit-identical to the stock tail" (flash_next_serving:63-65).

**Streams in decode-graph capture:** the shared expert and its gate on the current stream, the router and routed experts on the alt stream, with an event wait (flash_next_serving:572-627).

**Measured:**
- Kaggle 4xL4, 48 random-weight layers: 271 -> 221 µs a layer at 40 tokens, 349 -> 283 at 64.
- Modal 4xL4 paired A/B with the shared expert serialized: decode step 70.1 -> 68.5 and 77.9 -> 76.3 ms at 8 and 10 running.
- With the shared expert beside the routed experts again: "-6.4 / -4.9% at 8/10 running" against v30 (K:131-137).

**When it applies** (moe_down_ar.py:231-264): EP experts, uint4b8 weights with bf16 group-128 scales, 8-row blocks, SiLU gated, no bias, zeros or act-order, hidden 2560, top_k ≤ 16. Otherwise the stock path finishes the layer.

### 3.6 QSA decode context parallelism (`flash_next_qsa_dcp.py`)
**Problem** (qsa_dcp:1-23; flash_next_serving:91-97, 656-757):
- At TP4 the target's 2 KV heads are each stored on two ranks.
- Stock QSA has no DCP path. Every rank kept every token at its logical slot, and "the first slot past the physical rows failed store_kvcache".

**Fix:**
- **Ownership:** a logical slot s lives on DCP rank s % 2, at row s // 2. Tokens owned by the other rank are written to padding row 0.
- **Attention:** decode, target verify and chunked prefill all-gather the DCP group's query heads, attend to the selected tokens the rank owns and merge the partials through their natural-log LSE.
  - Only all-gathers are used: "the DCP group's custom all-reduce fails CUDA-graph capture on Kaggle's L4s, which have no usable peer path".
  - At size 2 each rank sends its partner only the partner's heads (qsa_dcp:265-277).
  - The kernel is QSA's chunk-prefill Triton kernel with an LSE output, a per-row key count and NaN-safe online softmax. FA4 does not run on L4.
- **Index widths:** int64 token offsets fix a held-server corruption: int32 offsets wrapped past row 2^31 / 256 on H200 (qsa_dcp:163-165; K:59-62, "fixed in v30").
- **Indexer cache:** it stays replicated over the logical slots and is counted once per DCP rank in the pool cost. Hence "about 1.8x the tokens in the same memory" rather than 2x (K:46-48, 113-117).

### 3.7 QSA indexer decode kernel (`flash_next_qsa_index_decode.cuh`)
**What it does:**
- In speculative verify and draft-extend CUDA graphs, the indexer is scored by ForgeKernel's `qsa_index_decode`.
- One CTA scores one request's 4 verify rows × 4 heads against a 64-key tile; dead tiles write -inf without loads. This replaces TileLang's per-row CTAs over the whole context limit.
- Same math and order, with fast math, bit-identical in 64 serving-shaped cases.

**Measured:**
- Cold-L2 L4 at 40 rows: 49.9 / 82.8 / 122.8 / 239.4 µs at 16k / 32k / 48k / 96k mean context, against TileLang's 234 / 326 / 435 / 742 (cuh:1-24).
- "The TileLang kernel costs 2.46 ms per decode step at 10 running on L4 (13 calls)" (K:143-147).

**Scope:** chosen at capture only when rows == 4 × batch with exact shapes; draft decode and eager batches keep TileLang (flash_next_serving:99-106, 1343-1446).

### 3.8 Top-k logits exchange (TOPK_LOGITS = 32)
**Change** (flash_next_serving:67-73, 759-799; K:118-121):
- The target's vocab-parallel logits all-gather (0.5 MB a token in bf16, "3.4 ms of a 78 ms step at 10 running on L4") becomes each rank's top-32 values and ids, scattered into a -inf row.
- "Exact when every request samples with top_k <= K - n", with n the number of negatively biased tokens: 20 ≤ 32 − 4. No logprobs are requested.

**Gain:** -3.6 % decode step at 10 running.

### 3.9 Prefill CUDA graph (PREFILL_GRAPH, backend `breakable`, one 512 bucket)
**Design** (flash_next_serving:75-89, 801-1129, 1276-1339; K:122-130, 1183-1188):
- Prefill chunks replay a captured graph.
- GDN, QSA attention, the QSA indexer and PLE run eagerly between captured segments. The HC norm and HC combine become mutating custom ops.
- The MTP draft's prefill stays eager. The HC state is made an explicit captured output.

**Measured:**
- tc_piecewise (v29-v30): a 512-token batch went 585 -> 516 ms for 0.15 GB a GPU.
- Breakable (v31): -6.5 to -6.8 % against tc_piecewise on vast.ai 4xL4 at 512 chunks (cold prompts -10 %, cached continuations -0.5 to -2 %).
- A full bucket ladder cost 0.8-1.1 GB a GPU and "raised teacher-forced NLL on phi1 (+0.017 nats over 35 tasks)", so there is one bucket.
- The 1024 chunk was rejected for Kaggle: prefill -5 % a token, but lowest free memory 0.37 -> 0.17 GiB, and 0.12 GiB within 84 minutes on a vast.ai L4 (K:104-107).

### 3.10 Memory reclaim and the KV pool
- **Draft shared vocab:** the MTP draft builds its embedding and lm_head on the meta device; the target's replace them (0.64 GB per L4 at TP4) (flash_next_serving:42-47, 421-498).
- **Host embedding:** the target's `embed_tokens` shard is served from pinned host memory like the PLE tables. "On L4 only the host embedding is a net saving" (K:1091-1096).
- **INT8 dense target (v29):** GPTQ W8A16 on in_proj_qkvz, attention q/k/v, shared-expert gate/up and linear-attention out_proj on layers 4-43. "It frees 0.23 GiB per L4 at the same speed; public 120 on phi1 79.55 over 4 seeds" (K:152-158).
- **Mamba:** lazy extra buffer, 4 slots a request, 13.9 MB a slot per L4 (K:59-64). Also `--mamba-ssm-dtype bfloat16 --mamba-track-interval 64 --mamba-backend triton --enable-linear-replayssm-spec` (K:1189-1205).
- **Pool:** `--max-total-tokens 331776` (rows a rank before DCP doubling) with `--mem-fraction-static 0.92` (K:53-58, 65-71). Reported lowest free memory: 0.37-0.38 GiB.

### 3.11 Weight loading and staging (`flash_next_loading.py`, `draft_view_adapter.py`; K:323-362, 488-539)
**Target:**
- REAP v2: 480 of 512 experts, mass-ranked, in every layer and the draft (K:152).
- The target is a symlink view: changed shards come from the private Kaggle Model, all other files (the 95 GiB PLE table included) from the dfranzen attachment (K:331-362).
- Kaggle Models are mounted over NFS while Datasets are GCSFUSE (K:157-158; loading:21-24).

**Staging:**
- Rank 0 stages at most 2 GPU shards at a time (≤ 12 GiB window), each copied by 16 parallel 16 MiB range readers (loading:25-27, 53-57, 94-143).
- Each TP rank fills its pinned PLE rows directly with 8 parallel readers, 32 streams in all (loading:27, 183-259).
- Measured: v12 copied single-stream, 65.7 GB in 1,290 s; four PLE readers reached ~254 MB/s (loading:20-24).
- The draft (refit, compressed-tensors INT4 g32 experts) is copied in full with 8 workers (K:77, 990-992).
- `draft_view_adapter` builds a metadata-only config view of the draft, "from Daniel Franzen's pinned solution" (github da-fr/arc-agi-3-solution).

**Compile cache:**
- A separate cache kernel ("warmup" mode) runs the same 16-task smoke and exports its compile cache. Submissions restore and verify it by SHA-256 and profile (K:167-176, 370-401, 1382-1402).
- Version 15 recorded: "16/16, no errors, 0 compiles outside the cache root, 0.37 GiB lowest free".
- "every run's first real prefill stalls ~115 s" (K:85-89).

### 3.12 Server command (K:1139-1264), all flags in one place
- `--tensor-parallel-size 4 --expert-parallel-size 4 --decode-context-parallel-size 2`
- `--dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.92 --max-total-tokens 331776`
- `--context-length 139264 --page-size 64 --max-running-requests 10`
- `--chunked-prefill-size 512 --max-prefill-tokens 512`
- `--cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1..10`
- `--cuda-graph-backend-prefill breakable --cuda-graph-tc-compiler eager --cuda-graph-bs-prefill 512`
- Mamba: `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 40 --mamba-radix-cache-strategy extra_buffer_lazy --mamba-track-interval 64 --mamba-backend triton`
- Linear attention: decode, prefill and verify backends triton; `--enable-linear-replayssm-spec`
- `--attention-backend flashinfer --moe-runner-backend auto --ple-offload-embedding`
- `--reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs {"preserve_thinking":true}`
- `--watchdog-timeout 1800 --schedule-policy lpm --disable-flashinfer-autotune`
- Speculative: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-token-map <hot ids> --init-expert-location <map>`
- Draft: `--speculative-draft-model-quantization compressed-tensors --speculative-draft-kv-cache-dtype fp8_e4m3 --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0`

---

## 4. kaggle_runner and loop_driver: run flow

### 4.1 Launcher (K `main`, 890-1466)
1. **Mode:** rerun if `KAGGLE_IS_COMPETITION_RERUN`. The challenges are then the test file; otherwise the evaluation file and a smoke (K:894-911).
2. **Integrity:** every runtime source is checked against `launch_manifest.json` SHA-256s, and the SGLang bundle manifest and its version 0.5.20 are checked (K:913-934).
3. **Fallback submission:** `submission.json` with `[[0]]` in both attempts for every test is written before anything expensive (K:937-944).
4. **Runtime:** the bundle is restored to `/kaggle/temp`, with a gzip CRC check instead of a slow SHA pass; the compile cache is restored and verified (K:301-321, 370-401, 958-965).
5. **Patches and weights:** `flash_next_serving.patch_sglang`, the REAP target view, the loading plan and loader hooks, draft staging and the draft view (K:966-994).
6. **Environment:** CUDA, Triton and linker setup; then the sandbox preflight (K:999-1132).
7. **Maps:** `prepare_serving` writes the hot-token and expert maps as .pt files (K:1134-1138; flash_next_serving:1553-1574).
8. **Server:** started in its own session, and the launcher waits for `/health` with no fixed limit beyond the notebook budget. Heartbeats go to `resources.jsonl` (K:1265-1311, 664-714).
9. **Runner:** `kaggle_runner.py` runs with CUDA hidden and workspaces under `/kaggle/temp/.../codex_lite` (K:1326-1348). The launcher then loops: if the server died, terminate and raise; at the deadline, terminate; it samples cgroup counters every 30 s (K:1349-1378).
10. **Shutdown:** the server stops in `finally`. In smoke or warmup the launcher packs the cache or runs the save-run checks; post-verify runs only when solutions exist (save runs). It then validates the submission (K:1379-1460).

### 4.2 kaggle_runner (`main`, kaggle_runner:332-537)
- **CLI:** narrow (267-312). Defaults that the launcher overrides: time-budget 600, context 262,144, concurrency 8, attempt mode `codex_lite_carry`.
- **Setup:**
  - `RolloutConfig`: model, seed_base, policies, time_budget, tokenizer, context, timeout, temperature, top_p, logit_bias, teacher_token_cap (rollout_runner:14-49).
  - The append-only trace store `attempts.jsonl` carries a run fingerprint over the rollout config, the challenges hash and the harness file hashes; a torn tail is repaired (rollout_runner:69-138; trace_store:34-132).
- **Loop:**
  - A ThreadPoolExecutor with 10 workers; `fill()` submits `next_attempt()` while workers are free (478-532).
  - Each finished attempt: `persist` the slot update, then `inducer_progress.json` (partial flag, per-phase counts, the controller snapshot, per-task results) and a status line (493-531).
- **Error handling:** a task error is caught. The attempt's outputs are lost, but the run continues (440-446).
  - If an attempt throws (for example a PromptContractError), `solve_puzzle` returns its error. With a single sample, "all attempts failed" raises, which `run_one` catches (loop_driver:558-567, 659-667).

### 4.3 loop_driver (`solve_puzzle`, 417-671)
- **Prompt renderer:** one locally cached renderer for the tokenizer (loop_driver:203-314).
- **Per sample:**
  - acquire the permit, then set the deadline;
  - carry path: sample 0 writes, sample 1 reads, only for carry policies (534, 545);
  - the policy's logit bias, effort, alternates, tools, request cap and images.
- **Results:**
  - `[live]` lines with best_train, selected and gen_tok (634-647).
  - The full trace (schema 3) with every turn's reasoning, content, tool calls, results, token accounting and prompt-identity hashes; the last turn carries the full request messages (codex_lite:1092-1100, 1183-1190).
  - `post_verify_traces.py` (save runs only) scores each attempt against solutions and can project solved traces to SFT rows (post_verify_traces:1-238; trace_store:149-206). The SFT output is not used on Kaggle.

---

## 5. Differences vs our pipeline that could matter
Our side, as given by the caller: Flash-Next on Merlin (vLLM 0.32 fork) TP2×PP2, klee100 3bpw, 24 streams, MTP or DFlash2, xhigh effort with a 60k thinking floor and caps 104k/120k, Duck/Poetiq/Direct harnesses, E28 K100 eval 107/172 (task-norm 65.0) at 419.5 tok/s, best LB 54+.

Labels: **[F]** = fact from their code (file:line); **[I]** = my inference.

### 5.1 Token budget and effort
1. **Effort.** [F] `reasoning_effort = "medium"` on every request (K:43; codex_lite:75-76). Low and xhigh were tried and retired (codex_lite:62-64). Ours is xhigh.
2. **Budget per attempt.** [F] The total generated-token budget per attempt is 65,536, across all turns (codex_lite:975, 1081-1083; K:44).
   - Two attempts give at most ~131k tokens per task, plus a ≤ 512-token recap.
   - Ours: a 60k thinking floor with 104k/120k caps on one attempt's thinking.
   - [I] Their whole two-attempt budget for a task is about one of our attempts.
3. **Per-request cap with nudges.** [F]
   - One response is capped at 12,288 tokens (codex_lite:72).
   - When thinking hits it, the reply's tool calls are dropped and a "write solution.py now" nudge follows (codex_lite:1122-1128), up to 6 times.
   - The last 8,192-12,288 tokens run with thinking off as a finalize window (codex_lite:806-811, 1039-1043).
   - Measured by them: 24,576 -> 12,288 gave "6.9 % fewer tokens, +1.1 to +2.2" (codex_lite:50-53).
   - [I] Their reasoning is many short think-then-run turns, not one long think.
4. **Anti-hedging logit bias.** [F] -1 on "Hmm" and "Alternatively" (4 ids); "cut generated tokens 4.5 %" (codex_lite:46-49, 68-70).

### 5.2 Harness and prompt
5. **Harness shape.** [F]
   - A Codex-CLI clone: two tools, `exec_command` (bash, 10 s yield, 10k-token output cap, 900 s command limit) and `write_stdin`.
   - The required artefacts are `solution.py` with `transform(grid)` and `predictions.json` (codex_lite_interface:11-15).
   - Acceptance of a "finish" = a replay of transform on all train pairs passes and equals predictions.json. Otherwise up to 3 "Completion check failed" feedbacks (codex_lite:773-803, 1112-1141).
   - [I] Our Duck, Poetiq and Direct harnesses differ in shape. I cannot compare details of ours that are not in this brief.
6. **Prompt cache.** [F] `preserve_thinking=True` keeps earlier turns' reasoning in later prompts. Every request's prompt ids are verified identical to a local rendering (loop_driver:127-156, 270-278; K:1222-1223).
   - [I] Every turn is a pure prefix extension, so prefix caching should hit across turns.
7. **Images.** [F] Every grid is also shown as a 10 px/cell PNG with grid lines, after the text grids (codex_lite:874-912; vision tower loaded, K:1213).
8. **Helper library.** [F] `arc_tools.py` in the workspace: `objects`, `compare` (which candidate rules disagree on tests), `shape_rules` (output-size rules) and others (arc_tools:1-21). The prompt tells the model to import it (codex_lite:83-87).
9. **Second candidate inside one attempt.** [F] The prompt asks for `predictions_alt.json` when rules disagree, and passing programs from the finalize window are kept. These fill `attempt_2` when the attempt is the primary (codex_lite:88-91, 716-770; kaggle_runner:236-237). Their stat: 9 of 91 alternates were right where the answer was wrong (codex_lite:720-722).
10. **Carry.** [F] A second attempt after an unverified first:
    - reuses the first attempt's files and its tool-call history without reasoning;
    - adds a 150-word recap and per-pair cell diffs when ≥ 50 % of train pairs passed;
    - gets a fresh 65,536 budget (codex_lite:110-115, 868-951, 1166-1181).
    - Their yield: 4.6 solves per million tokens after an unverified first, against 2.6 after a verified one (kaggle_runner:99-100).
11. **Sampling.** [F] Temperature 0.7, top-p 0.95, top-k 20, min-p 0. The **same seed** goes on every request of an attempt: 1234 for the first attempt, 1235 for the second (codex_lite:1066; loop_driver:536).
12. **Text grid format.** [F] Digits separated by spaces, NVIDIA-style labels (loop_driver:43-75).

### 5.3 Scheduling and deadline
13. **Scheduling.** [F]
    - All first attempts first, in sorted task-id order (no difficulty ordering).
    - Then second attempts for every task: unverified-first tasks before verified ones, no barrier (kaggle_runner:456-472; rollout_runner:59-66).
    - There is no re-run of train-fail attempts: each task has exactly two attempts.
    - Context for comparison: our Exp 28 scheduler re-runs train-fails before attempt 2. This comes from our project memory, not from the brief.
14. **Concurrency.** [F] At most 10 running attempts.
    - Admission starts at 4 and grows by 1 only while KV ≤ 60 %, nothing waits and all admitted attempts are running.
    - It trims at KV > 80 % and cuts on any retraction (adaptive_concurrency:25-41, 204-283). The server ceiling is 10 (K:52).
    - Ours: 24 streams.
    - [I] They choose fewer streams with long contexts (avg 37.5k, p90 92k, K:50-51) and zero preemption, over more streams.
15. **Deadline.** [F] Hard stop at 11:55 after the kernel start, with no drain phase. In-flight attempts are killed, and their provisional answers, published continuously, remain in `submission.json` (K:32-33, 1349-1368; codex_lite:642-713; kaggle_runner:429-431).
    - Each attempt also has a 4 h wall limit (K:847-850).

### 5.4 Engine
16. **Engine and layout.** [F] SGLang 0.5.20 with 23 pinned file patches; TP4 + EP4 + DCP2; FP8 KV; 663,552 KV tokens a DCP pair (K:65-71, 1155-1168). Ours: Merlin vLLM 0.32 TP2×PP2.
    - [I] TP4 at EP4 forces the custom collectives: host-memory all-reduce, fused MoE down + AR, top-k logits, local HC up.
    - [F] Kaggle's L4s "have no usable peer path (0.46-0.9 GB/s per GPU when all four talk)" (host_ar:3-4).
17. **Quantization and pruning.** [F] W4A16 AutoRound target; REAP 480/512 experts in the target and the MTP draft; GPTQ INT8 dense layers 4-43 (K:149-160). Ours: klee100 3bpw.
18. **Speculative decoding.** [F] NEXTN MTP: 3 steps, top-k 1, 4 draft tokens, accept thresholds 1.0.
    - The draft was **refit to the pruned target** (accept length 2.807 -> 2.992 on 1,939 requests).
    - A 27,982-token hot vocabulary drives the draft head (K:161-164, 1240-1263; hot_tokens.json).
    - Ours: MTP or DFlash2.
19. **Expert placement.** [F] Balanced per-GPU expert placement from recorded decode routing: held-out max activated per GPU 37.58 -> 35.33 (expert_location.json).
    - [F] Without their routing fix, SGLang's `--init-expert-location` "ran tokens through the wrong experts" (flash_next_serving:28-32).
20. **Speed.** [I] About the same as ours: decode step ~74 ms at 10 running with accept length ~3 gives ≈ 400 tok/s, against our 419.5 (my arithmetic from K:133-141).
    - [I] On the evidence in the code, the gap to their ~80 public-120 scores comes from the solver side (items 1-13), not throughput.
    - [F] Their public-120 scores are 79.55 over 4 seeds (K:156) and 81.11 for the v31 rehearsal on phi1, run at chunk 1024 (K:128).
21. **Startup.** [F]
    - A pre-built compile cache is restored and SHA-checked, so a save run compiles nothing.
    - Weights are staged from Kaggle Models (NFS) with 16-way range reads; the 95 GiB PLE table stays on the attachment.
    - There is no fixed startup timeout (K:167-176, 334, 1301-1304; loading:20-27).
22. **Server crash.** [F] If SGLang dies mid-run, the runner is killed and the script raises (K:1355-1360). There is no restart or fallback server. [I] Such a run probably scores as an error, despite the incremental submission on disk.

---

## 6. Notebook inputs fetched (2026-11-04, `andy2709_inputs/`)
- **Inputs of `andy2709/arc-flash-next-w4a16-inducer`:**
  - dataset `andy2709/arc-flash-next-inducer-runtime` (the 21 runtime files above);
  - models `andy2709/arc-flash-next-480`: "480-expert INT8-qkvz target shards and re-fit MTP draft for Flash-Next W4A16";
  - model `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`;
  - kernel outputs `andy2709/arc-flash-next-sglang-0520-runtime` and `andy2709/arc-flash-next-reap-v2-cache`.
- **`arc-flash-next-sglang-0520-runtime`.** Fetched: `runtime_manifest.json`, `runtime_portability.json` and the run log. `flash_next_runtime.tar.gz` (the SGLang bundle) was not downloaded.
- **`arc-flash-next-reap-v2-cache`.** The warmup / compile-cache run (16 first attempts).
  - All files fetched except `flash_next_compile_cache.tar.gz`.
  - Its `warmup_report.json` gives **measured** numbers, which replace the estimate in section 0 item 5:
    - server ready after 1,491.6 s;
    - KV pool 331,776 rows;
    - 16 attempts, 181,142 generated tokens in 771.6 s;
    - decode 283.7 tok/s mean over a 639 s window at mean 5.94 running, mean accept length 3.013.
  - Median decode tok/s by running requests:

    | running | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | tok/s | 123.4 | 164.2 | 223.5 | 265.1 | 330.1 | 345.5 | 344.0 | 353.7 | 395.3 | **402.5** |

  - `attempts.jsonl` (2.0 MB) holds the full traces of those 16 attempts. `sglang_server.log` holds per-step decode lines (accept len about 2.5-3.5).
