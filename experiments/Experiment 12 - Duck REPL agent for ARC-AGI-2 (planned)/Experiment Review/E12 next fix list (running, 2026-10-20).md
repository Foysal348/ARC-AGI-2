# E12 Duck agent: next fix list (running)

> **Status 2026-10-22 (final): rev 6g BUILT, the LAST code version before the GPU run (user). Notebooks: EVAL12R6 `74a220de…`, EVAL12R6G4B_REBUILD `5d2bc39f…`, PARITYR6 `82b257a5…`. Kaggle-run rev 6d G4B `1bdadd82…` unchanged. CPU tests all green; not pushed. Next: the G4B GPU run. See "Rev 6g" at the end.**
> - **EVAL6R5 xhigh** (sha `f4fed1ac…`): 4/6 tasks, 5/8 inputs.
> - **EVAL12R5_LOW** (sha `26fff41e…`, the EVAL12R5 profile + low): 7/12 tasks, 10/19 inputs. It was submitted as ref 57456562 (pending).
> - **EVAL30R5** (sha `415a68f2…`): not run.
> - **Rev 6 design:** the sections from "Rev 6 candidates" to the end, the verbatim drafts in `E12 rev 6 prompt and resume drafts from GPT (verbatim, 2026-10-21).md`, and the consolidated gap check "Rev 6 gap check" at the very end. The gap check has the **rev 6 ID mapping (R6-xx)** to use from now on.

**Rule (user, 2026-10-20):**
- Read every GPT report in full and validate every claim.
- Add the validated items here.
- Change no code until the user says so.

**Sources:**
- **W**: my weakness analysis (`Experiment Results/E12 agent weakness analysis (2026-10-20).md`, zip `E12_AGENT_WEAKNESS_ANALYSIS_FOR_GPT_2026-10-20.zip`).
- **G1**: GPT "Deep Root-Cause Analysis".
- **G2**: GPT "দুর্বলতা, Root Cause ও উন্নয়ন পরিকল্পনা" (Weakness, Root Cause and Improvement Plan).
- **G3**: GPT "Deep Weakness Analysis & ARC-AGI-2 Adaptation".

All three were read in full on 2026-10-20.

## Where the three agree

- **No redesign:** keep Duck's transport, sandbox, retention, gate and transcript.
- **Order of work:**
  1. candidate harvest + two-attempt selection;
  2. context retention + truthful resume;
  3. reasoning efficiency (tool use, sampling, output size, loops);
  4. coverage/routing (Duck as a complement to Poetiq, not a replacement).
- **All three ask for isolated, measured steps.** This conflicts with the user's "bundle changes, big jumps" rule, so the order of testing is the **user's decision**.

## Fix list

| ID | Problem | Fix (validated choice) | Sources | Validation / evidence | Priority |
|---|---|---|---|---|---|
| **F1** | Train-exact (and test-correct) functions are lost when not submitted | **Runtime `check()` hook** (G1): when `check()` finds every pair exact inside the sandbox, the helper sends an action with the function name. The host re-verifies with the existing clean `_verify_submission`, stores the verdict, and **does not stop the branch**. An explicit `submit()` keeps today's end. Dedupe by source and test outputs; bounded verification cost | W1, G1 §4-5, G2 §3, G3 W1 | **Verified:** `select_attempts` reads only `submissions` (arc2_duck.py:556). Lost: 62593bfd_b1, b5ca7ac4_b0 (both test-correct by replay). **No auto-stop** (G1, verified): EVAL30 247ef758_b0 first exact (L1760) was test-wrong; the model fixed it and submitted a correct one (L2190). **Method chosen by data:** G2's passive harvest of retained `transform*/solver*` functions was replayed over all 43 branches. It found **no** exact function beyond `check()` and **missed 2** (135a2760_b1, 247ef758_b0: exact functions under other names). G3's AST scan of `check(name)` in the code would also verify non-exact checks (more cost) and can see unreached calls. The runtime hook sees only executed, exact checks; a forged signal only triggers a clean host verify | **P0** |
| F1-a | Proof of F1 | Validate with the real Duck sandbox and adapter (re-run the 62593bfd_b1 / b5ca7ac4_b0 / 247ef758_b0 sources), never with `replay_all.py` | G1, G2 §11B, G3 | **True:** `replay_all.py` runs host `exec` with every callable kept (no sandbox, no retention rules). Its 7/8 is a retrospective oracle estimate, not a production number | with F1 |
| **F2** | The reply just generated is dropped from the next prompt; the resume says "your tool results above are still valid" | (a) Correct the token estimate for reasoning text: the pinned Qwen tokenizer + chat template (G2 preferred) or the server's `completion_tokens` for assistant messages. Validate the estimate against the server's `usage.prompt_tokens`. (b) **Truthful resume:** the resume text is the adapter's own (arc2_duck.py:330-332); Duck sets `_context_was_trimmed` / `_has_evicted` in `_note_history_evicted` (tool_agent.py:6307), so the adapter can say "earlier reasoning was trimmed; reuse retained functions…". (c) **[rejected in the rev 4 audit (#5): trims every turn kill the prefix cache]** `ARC3_CONTEXT_DRAIN_FLOOR` only as an optional test (the donor keeps it off because it costs cache hits, tool_agent.py:221-224) | W2, G1 §6, G2 §4, G3 W2 | **Verified, mechanism read in code.** After a tool-less reply, history = [opener, reply, resume]. Over budget, `_drop_oldest_history_block` (tool_agent.py:6222-6258) pops the opener and everything up to the next user message, so the reply goes. The estimate uses one chars/token ratio from the served prompt (`_calibrate_from_usage`, :5962-6008): 1.40-1.72 on grid prompts, later 2.26 / 2.43 (G2, true). 11 of 84 EVAL6 later requests (426,781 tokens) were not carried; 5 were full 60k replies reset to a system-only prompt (G3's independent count of 5 matches: 135a2760_b0, 62593bfd_b0 x2, 62593bfd_b1, 898e7135_b1). G2/G3: the share due to calibration vs the real 70,560 limit is unproven until a tokenizer check (accepted) | **P0** |
| F2-a | Capacity trade-off | 60k reserve → 70,560 input budget; a 60k reasoning + 9k prompt barely fits even when estimated exactly | G1, G3 | True (arithmetic) | decide with F3 |
| **F3** | Repetition loops in long thinking | Separate arms, not all at once (G2/G3): (a) **temperature: 0.7 (Duck) vs 1.0 (official Qwen3.8 thinking)**; (b) shorter per-request output (G1 32-40k, G2 24-32k) — **needs the user's approval (60k floor rule)**; (c) presence penalty only if loops persist; (d) better loop detection later (early cancel needs streaming, a big change, deferred) | W3, G1 §7, G2 §6, G3 W3 | **Verified on HF model card** (huggingface.co/Qwen/Qwen3.8-27B): thinking = temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0; presence 0-2 reduces repetition but risks language mixing. **My "0.6 Qwen-like" was wrong**; our earlier DIRECT pipelines used 0.6, a separate data point. Duck's detector threshold is 0.01 zlib (tool_agent.py:1643-1657); our long blocks are 0.037-0.19, so 0 detections. Repeat share EVAL6 29.4 % (G3 independently 29.2 %), EVAL30 5.2 %; this is characters, not tokens (G2, accepted). **SGLang #28180** (verified): repetition penalty ineffective under DFlash; fix PRs not confirmed merged → check 0.5.21 before any penalty arm | **P1**; user decision on (b) |
| **F4** | Thinking in the head instead of testing in code; hand misreads; refusing a lookup that fits | Short wording changes only, no multi-stage template (G1/G2/G3, supported by Johan Land's abstract: prescriptive templates reduce diversity; verified on arXiv 2606.31543). Replace "Investigate until your code has a clear plan." (arc2_duck.py:266) and soften "derive positions, sizes, colors and counts from the input" with: use Python for exact grid values; implement and check a plausible rule early; an input-derived lookup / case split is valid; do not drop a train-exact candidate for lack of an elegant explanation; keep alternatives when ambiguous. Duck's `ARC3_PREFER_TOOL_CALLS` exists (tool_agent.py:1879) but **does not reach our prompt** (the adapter builds its own `arc2_system_prompt`); its wording (prompts.py:285) can be reused | W4, G1 §8, G2 §5, G3 W4 | Prompt lines verified. Failed-branch pattern measured (22/24 EVAL30, 5/8 EVAL6 without `check()`). G2: "median 100 % of tokens before the first check" is censored (most never check) and is not a timing estimate (accepted). G3 failure taxonomy (no candidate / partial / exact-no-submit / exact-wrong / correct-unselected) adopted for reporting | **P1** |
| **F5** | attempt_2 wasted; overfit accepts | Portfolio selection over submissions + F1 candidates. Two distinct train-exact predictions → both attempts, without letting a weak majority drop a minority. Identical grids deduplicated. Train-exact > partial > fallback. No invented confidence. Later, possibly a Poetiq candidate when Duck has none (G3) | W5, G1 §9, G2 §7, G3 W5 | **Verified:** attempt_2 = the test input in 56/56 slots (EVAL6 8 + EVAL30 48); 4/11 accepts test-wrong. Kaggle scoring: either attempt exact counts. b5ca7ac4 (EVAL6) would be rescued with b0's function as attempt_2 | **P0** with F1 |
| **F6** | `train` dict-unpack trap; retention NameErrors; a tool call cut at the cap executed as a stub | G3 prefers a **documentation fix first**: show `for p in train: inp, out = p["input"], p["output"]` in the tool guidance (smaller surface than a new global). G1/G2 propose a `train_pairs` alias. Small inspection helpers optional. Stub execution: audit the parser condition first (G3) | W6, G1 §10, G2 §8, G3 W6 | Dict trap verified (3 of 4 EVAL30 tasks read, 62593bfd_b0); `train` is a list of dicts (arc2_duck.py:127-131). Stub execution is reported by reader C (EVAL30 136b0064 L3550), not yet checked in code | P2 |
| **F7** | Coverage: about 10.4 min/task → 240 tasks ≈ 41 h vs about 11 h | **[Superseded 2026-10-21: the user's rule is that Duck runs on every task (full coverage). This is now the rev 6 scheduler, R6-S1..S6. "Poetiq complement" is a later, separate question.]** Original text: treat Duck as a **complement to Poetiq**. Measure Duck's unique solves and opportunity cost first. First-pass budget about 2.75 min/task (G2 arithmetic) if Duck ran alone | W7, G1 §11, G2 §12, G3 §5 | Arithmetic true. **G3's complementarity signal verified** in our ledger: 247ef758 query 1 was solved by Duck EVAL30 and by **0 of 12** other runs (`Puzzle Capability Ledger/ledger_units.csv`) | P0 for a submission, after F1-F5 |
| F8 | Branch sharing | **Not now.** Merge candidate pools, not reasoning contexts (diversity) | G3 W8, G1 | opinion, consistent with the paper's diversity point | — |
| F9 | Reviewer/judge (Athanor / Johan Land style) | Later, only when F1/F5 produce conflicting portfolios; frontier-API results are not transferable | G1, G2, G3 | Paper abstract verified (holistic judging recovers minority hypotheses). G1's numbers (144/167 vs 128; Table 4) are not visible in the abstract → unverified | later |
| F10 | Trusted serializer (json.dumps spoof) | **[Done in rev 5 as R5-1.]** Revisit before hidden submission with harvesting. Decided 2026-10-19 not to fix (the score is truth-checked, never observed, a lookup is equivalent); with harvesting, a forged candidate could take an attempt slot in a hidden run. Still never observed in 43 transcripts | G2 §11A | Source fact verified (`_grid_call_code` unchanged) | P2, decision |
| F11 | Analysis hygiene | No truth in production paths; replay claims re-checked in the real sandbox; report characters vs tokens and censored statistics properly | G1, G2, G3 | accepted | always |
| F12 | Do not copy wholesale | No 29-candidate ensemble, no multi-agent API orchestration, no rigid templates, no generic augmentation, no streaming rewrite now. Imbue Darwinian Evolver is **AGPL-3.0** (verified) → ideas only | G1, G2, G3 | AGPL verified | note |

## Metrics for every future run (merged G1 §13 + G3 §8)

- Correct outputs and fully solved tasks.
- Any train-exact program found.
- Host-verified candidates (harvested vs submitted).
- Correct candidate generated but not selected.
- Distinct predictions per test input.
- Tokens before the first `check()` (censored).
- Repeat share (characters).
- Reply tokens dropped / newest reply retained rate.
- Prompt-cache %.
- Train-exact / test-wrong rate.
- Unique correct outputs per 4xL4 wall-hour.
- Duck-only solves vs Poetiq.

## Corrections to my own earlier statements (found through these reviews)

1. "No failed branch got even one train pair exact" (weakness README W4): **wrong** (G1). 62593bfd_b1 2/2, b5ca7ac4_b0 3/3, EVAL30 135a2760_b0 1/2. Fixed in the doc.
2. "The newest block always stays" (pre-run validation): **wrong** for tool-less replies (code read above).
3. "Sampling closer to the Qwen recommendation (temperature 0.6)": **wrong**. The official thinking setting is 1.0.

## Claim-by-claim tables

### G1

| Claim | Verdict |
|---|---|
| README W4 wrong | True |
| Selection only from submissions | True |
| 247ef758 → no auto-stop | True |
| `replay_all.py` is not the sandbox | True |
| Calibration at :5962-6008 | True |
| Calibration share is a hypothesis | Partly resolved in code (the estimate exceeds the budget only with the grid ratio); tokenizer proof pending |
| 70,560 trade-off | True |
| Degenerate threshold 0.01 misses loops | True |
| Non-streaming, so no mid-request recovery | True |
| SGLang #28180 | True; fix status unknown |
| Prompt line :266 | True |
| 4/11 wrong accepts | True |
| Poetiq 16/48 | True |
| 10.4 min/task | True |
| Paper numbers 144/167, 128, Table 4 | Unverified (not in the abstract; PDF not available to me) |
| AGPL | True |
| Athanor and other donors | Ideas, not checked |
| `check()` → `action()` feasible | True; note verification sandboxes refuse actions |

### G2

| Claim | Verdict |
|---|---|
| `_run_python_tool` appends only pending submits (arc2_duck.py:361-474) | True |
| Passive harvest via `keepable_functions` is feasible | True (name/source exist, tool_agent.py:4614-4627), but **replay shows it misses 2 of 12 and adds 0** → not chosen |
| Calibration 1.40 → 2.26 / 2.43 | True |
| Exact cause unproven without tokenization | Accepted |
| Tokenizer counting preferred; constant bump rejected | Accepted |
| "Prefer the simplest rule … derive … from the input" | True (system prompt) |
| Johan Land §8.6: prescriptive prompts hurt | Abstract confirms the general claim; section number not checked |
| Qwen3.8 thinking: temperature 1.0, top_p 0.95, top_k 20, presence 0, repetition 1.0 | **True** (HF card) |
| Presence penalty caveat | True |
| 56/56 attempt_2 = input | True |
| Majority vs holistic, 7 extra instances | Abstract confirms the principle; "7" unverified |
| `train` dict trap + alias | True |
| Serializer guard absent | True (decided earlier, revisit = F10) |
| `replay_all.py:17-27` keeps all callables | True |
| 62.4 min / 6 = 10.4; 240 tasks = 41.6 h; 2.75 min/task target | True (arithmetic) |
| Censored median, characters vs tokens | Accepted |
| Vendored tree not byte-compared to upstream (by G2) | Note; our earlier manifest check showed 23/23 hash match to our vendored copy |

### G3

| Claim | Verdict |
|---|---|
| 12 branches / 96 requests / 1,217,904 tokens; soft cap stopped most failures | True |
| 62593bfd_b1 / b5ca7ac4_b0 lost | True |
| Do not auto-submit every check | Agrees with G1 |
| AST-based `checked_function_names` sketch | Feasible; the runtime hook chosen instead (only executed + exact checks; the AST also catches non-executed calls, G3 notes it) |
| At least 5 system-only resets | True (5 identified) |
| Mechanism (1.40 ratio, 126k chars ≈ 90k) | True (code read) |
| Donor knobs `ARC3_CONTEXT_DRAIN_FLOOR`, `ARC3_STUB_DEAD_REASONING`, `ARC3_PRUNE_CONTROL_CONTEXT`, `ARC3_SUMMARY_REPLACES_HISTORY` | Exist in the vendored tool_agent.py; none set by the adapter |
| `ARC3_PREFER_TOOL_CALLS` | Exists (:1879), but only enters Duck's own prompt; our adapter prompt does not use it |
| Resume text at arc2_duck.py:330-332 | True (adapter text) |
| Repeat share 29.2 % / 5.2 %, 11 of 96 blocks | True (matches) |
| Qwen temperature 1.0 | True |
| Degenerate stubbing at :6945-6979 | True; post-hoc only |
| Early cancel needs streaming | True (`stream=False`) |
| `train` = list of dicts (:127-131); documentation-first fix | True; recorded as F6 option |
| 247ef758 Duck-only signal | **True** (ledger: query 1 solved 0/12 by other runs) |
| 62 / 6 x 240 = 41.3 h; Kaggle 12 h / 4xL4 quota cost | True |
| b0 0/6 vs b1 4/6 not a branch effect | Agreed |
| Johan Land pp. 32-33 negative results | Abstract confirms the principle; pages not checked |
| Not-now list (judge, multi-agent, templates, augmentation, check-stdout trust, sharing, streaming, extra judges) | Accepted as F12/F8/F9 |

## Deep re-validation (2026-10-21, all 43 branches; user: one bundled run, evidence-based changes only)

| Check | Method | Result |
|---|---|---|
| **A. F1 in the real sandbox** | `e12_smoke/replay_real_sandbox.py`: every recorded tool call of all 43 branches through the REAL `Arc2ToolAgent._run_python_tool` (Duck sandbox, prelude, retention); at each exact `check()`, the adapter's `_verify_submission` | 62593bfd_b1: host-accepted, both tests correct. b5ca7ac4_b0 `transform2`: host-accepted, test correct. 247ef758_b0: call 9 host-accepted but test half-wrong; calls 12 and 14 correct. **New: 136b0064_b1 call 4**: the model's `check()` printed exact but the host verify **rejected** it (`patmap` not retained), and the model fixed it later. So the host verify is required, and a failed harvest verify must be reported back to the model. 3dc255db_b0 (EVAL30) found too (wrong on test) |
| **Simulated F1 + F5** | latest host-accepted function per branch first, then earlier ones; two attempts; scored with truth | **EVAL6: 5/6 tasks, 7/8 inputs** (actual 3/6, 4/8). **EVAL30: unchanged 4 tasks, 7/48** (branch 1 never ran there, so no second candidate) |
| **B. F2 with the real tokenizer** | Qwen INT4 `tokenizer.json`; reply tokens vs Duck estimate vs budget 70,560 | Reply = exactly 60,000 tokens (= server). The 4 system-only resets would really have been 65,203-69,461 tokens (**fit**), but Duck estimated them at 77,269-111,375 (over). Kept case (ratio 2.10): estimate 70,042, just under |
| **All 11 drops classified** | prev prompt + reply vs budget (tool result ≤ 3.3k) | **5 avoidable** (the system-only resets: 135a2760_b0, 62593bfd_b0 x2, 62593bfd_b1, 898e7135_b1; about 279k reply tokens), caused only by the estimate. **6 real overflows** mid-run (prev + reply = 75.5-84.2k > 70,560). At a 49,152 output cap the budget is 81,408, so 4 of the 6 would fit (75.5k, 76.3k, 79.4k, 80.8k + tool) |
| **C. Output-cap evidence** | completion by finish reason (EVAL6) | Tool-call replies (87): p50 2.3k, p90 31.6k, p95 35.6k, max 58.2k; **> 40,960: 3 (one in a solved branch: 4c3d4a41_b1 first reply 44,779); > 49,152: 1** (898e7135_b0, failed). The 7 `length` replies (all at 60,000) had **no** tool call. No solved-branch reply exceeded 44,779 → **a 49,152 cap cuts no productive reply seen; 40k would have cut one solved reply** |
| **Loop position** | repeated-line share per sixth of each >60k-char reply | Loops start early (by about 1/6-2/6 of the reply) and then fill the rest at ~100 %. One solved branch (4c3d4a41_b1) escaped a mid-reply loop. A cap trims only the tail; a loop stop or penalty is needed to save the bulk |
| **SGLang 0.5.21 DFlash penalties (static read)** | `sgl_src` 0.5.21 | EAGLE accumulates penalty output tokens each decode step (`eagle_utils.py:1056-1058` → `batch.cumulate_penalty_output_tokens()`); the DFlash path (`spec_utils.spec_prepare_for_decode` → `dflash_info_v2.prepare_for_decode`) never calls it. `dflash_utils.py:255-285` applies only linear penalties. So **presence/frequency/repetition penalties are effectively inactive with DFlash2 in 0.5.21** (= SGLang #28180). A penalty arm needs a 2-line SGLang patch mirroring EAGLE |
| **E. Truncated tool call executed** | finish `length` with a tool call | EVAL30: 5 cases executed (e.g. 136b0064_b0 L2895: `length`, arguments `def block_pattern(...): H=len(inp)` executed and retained); EVAL6: 0. Duck flags truncation only when the arguments fail to parse (tool_agent.py:6887); SGLang's parser returns closed JSON, so it runs. Relevant again with a 49k cap |
| **D. Sweep for missed issues** | all tool results / statuses / server log | Tool errors EVAL6: NameError 4 (3x `patmap` retention, 1x `collections`), ValueError 4, ImportError 1 (`inspect`), IndexError 1. **EVAL30 request timeouts: 40** in 21 of 31 branches (28 full 900 s + 12 deadline-bounded; about 31,000 stream-seconds). I had written "4", which was **wrong**; fixed by the 1,800 s timeout (EVAL6: 0). Server log: no OOM / errors (benign `wrapt` sitecustomize, 1 early abort). Decode log: 184 of 1,154 lines with 1 running, 358 with < 6 → an EVAL-tail effect (12 branches / 6 live), small for 240 tasks |

## Bundle for the next run (evidence-based; for the user's decision)

| # | Change | Evidence | Scope |
|---|---|---|---|
| B1 | **F1** runtime `check()`-exact hook → host clean verify → candidate stored; branch continues; explicit `submit()` keeps today's end. The verify result goes back to the model (accepted / why rejected) | 2 lost solves (real sandbox), 247ef758 (no auto-stop), 136b0064_b1 (host reject) | adapter |
| B2 | **F5** two attempts from distinct host-verified candidates, latest per branch first, then earlier ones, then partial, then fallback | b5ca7ac4 rescue; 56/56 wasted attempt_2; 247ef758 (latest = correct) | adapter |
| B3 | **F2a** real token accounting for history: the server's usage (`prompt_tokens` + `completion_tokens`) or the tokenizer, instead of one grid-calibrated ratio | 5 avoidable resets, about 279k tokens, tokenizer-proven | adapter override of the estimate (vendor untouched if possible) |
| B4 | **F2b** truthful resume when history was trimmed (Duck flags `_context_was_trimmed`) | restarts after trims + the false "still valid" text | adapter |
| B5 | **F3** output cap 60,000 → **49,152** (input budget 70,560 → 81,408) | no productive reply > 44,779; 7 dead 60k replies; 4 of 6 real overflows would fit. The user approved evidence-based cuts (2026-10-21) | config |
| B6 | **F6-b** a `length`-cut tool call is not executed; "cut off" feedback instead | 5 executed stubs in EVAL30; more likely at 49k | small vendor/adapter change |
| B7 | **F4** prompt wording: replace "Investigate until your code has a clear plan."; soften "derive …"; code-first; an exact lookup is valid; keep alternatives | 22/24 + 5/8 failed branches without `check()`; misreads; lookup refusal (136b0064) | adapter prompt |
| B8 | **F6** `train` usage example in the tool guidance | dict trap in ≥ 4 branches | adapter prompt |
| B9 | **Loops: user decision.** (a) temperature 0.7 → **1.0** (official Qwen3.8 thinking setting; no A/B data of ours). (b) presence penalty **only with** the 2-line SGLang DFlash patch; HF warns of language mixing; no data of ours. Loops are the largest remaining waste (29 % of thinking characters) | loops measured; remedies unmeasured on our stack | config / SGLang patch |
| — | Not in the bundle | F7 coverage/routing (needs this run's metrics), F8-F12 | — |

## Rev 5 implementation (2026-10-21; user: B1-B8 + B9(a) temperature 1.0)

All in `duck_arc2/arc2_duck.py` (the vendored Duck files are unchanged) and the builder. Every change sits behind `E12_ARC2_REV5=1`, which only the R5 builds set, so SMOKE1 / EVAL30 / EVAL6 keep their exact behaviour (regression below).

| # | How | Where |
|---|---|---|
| B1 | The prelude `check()` sends `CHECKED:<name>` through the same action channel as `submit()` when every pair is exact (try/except: refused inside verification). After retention is updated, the host runs the existing `_verify_submission(name)`. The verdict goes to `arc2_checked` (not `submissions`, so the opener still reports only real submissions) and a short note goes back to the model (kept / why not). Dedupe on (name, sha of all retained sources); cap 24 verifications per branch (`E12_CHECK_VERIFY_CAP`). The branch is **not** stopped; an explicit `submit()` still ends it | `_HELPERS_SRC_R5`, `_handle_action`, `_arc2_harvest_checks`, `run_branch` record (`checked`, `rev5_events`) |
| B2 | `select_attempts_r5`: per branch, its latest accepted submission, else its latest accepted check; grouped by output (more branches > explicit submit > latest); then other accepted (latest first); then partial; then the test input. attempt_2 = the next distinct grid. Invalid grids are never selected | `select_attempts_r5`, `build_submission` |
| B3 | `_estimate_request_input_tokens`: assistant reasoning (≥ 2,000 chars) is counted with the served model's `tokenizer.json` (`E12_TOKENIZER_JSON`; fallback 1.9 chars/token); `_calibrate_from_usage` then removes the reasoning from the ratio measurement, so the ratio describes the non-reasoning text it is applied to | `reasoning_tokens`, overrides |
| B4 | Resume text without the false "tool results still valid". When Duck's trimmer removed messages, the resume opener it keeps says older messages (possibly the latest reasoning / tool results) were removed and asks to recover facts with Python | `ARC2_RESUME_R5`, `ARC2_RESUME_TRIMMED`, `_trim_messages_for_context` |
| B5 | `E12_DUCK_MAX_OUTPUT` 49,152 (input budget 81,408) | builder |
| B6 | `_chat_completion`: on `finish_reason=length`, tool-call arguments are made invalid, so Duck's own path drops them. **Found by the harness:** with the 2,048-token yield, Duck's "cut off" note is never sent (the turn yields first), so the next resume opener now says the call was cut and nothing ran | `_chat_completion`, `ARC2_RESUME_CUT` |
| B7 | Four exact-text prompt edits: the simplest-rule line softened (+ lookup valid + compare test outputs when rules disagree); "read exact values with Python"; "check() keeps candidates"; "write and check a candidate early" instead of "Investigate until your code has a clear plan." | `_R5_EDITS` |
| B8 | `train` usage example in the tool guidance | `_R5_EDITS` |
| B9(a) | `E12_DUCK_TEMPERATURE` 1.0 → `LOCAL_ANALYZER_TEMPERATURE` | `duck_env`, builder |
| — | Report: per-branch `checked_verified` / `checked_kept` / `rev5_events`; per query `harvested_programs(_correct)` | CELL 7 |
| — | Profiles: `EVAL6R5` (6 tasks, 90 min solving, 125 min budget) and `EVAL30R5` (the same 30 seed tasks as EVAL30, 330 min solving, 6 h budget). The rev 5 knobs apply to every profile of an R5 build, including the hidden rerun | builder |

### Local evidence (rev 5)

- **Rev 5 unit checks (`smoke_rev5.py`, real Duck sandbox): 40/40.**
  - Harvest: exact / dup / changed source / `verbose=False` / non-exact / not-retained / forged signal / cap / check+submit.
  - Selection: two distinct / latest first / dedupe / partial / fallback / invalid grid.
  - B3: the recorded 60,000-token 62593bfd_b1 reply, at the EVAL6 budget of 70,560 and ratio 1.40. Old estimate over budget, so the reply is dropped (the bug is reproduced); rev 5 estimate within budget, so the reply is kept. Calibration stays at the grid ratio.
  - B4: trimmed and untrimmed.
  - B6: invalid args; normal call untouched; the next resume says "cut", once; trimmed + cut.
  - B9 env.
- **End-to-end real-sandbox replay with the rev 5 hook** (`replay_rev5.py`, all recorded tool calls of both branches, then `build_submission`):
  - **EVAL6: 5/6 tasks, 7/8 inputs** (recorded run 3/6, 4/8);
  - **EVAL30: 4 tasks, 7/48** (unchanged: branch 1 never ran there).
- **Notebook harness (real cells + fake SGLang):**
  - **`r5` on EVAL6R5: 30/30.** Covered: knobs; every request 49,152 / temperature 1.0; rev 5 prompt; ≤ 6 in flight; no branch submits yet every branch keeps ≥ 1 verified candidate; the cut call never ran and the model was told; rev 5 resume; every query hit with one right + one test-wrong candidate per task; harvested counts in the report.
  - **`rerun` on EVAL6R5 (hidden mode with rev 5): 21/21.** `launchfail`: 6/6.
- **Regression:**
  - `eval6` on the rebuilt rev 4 EVAL6: 25/25.
  - `smoke` on SMOKE1_REV4: 22/22.
  - Adapter unit checks with rev 5 off: 41/41.
  - The Kaggle-run notebooks are untouched: EVAL6 `53a66856…`, SMOKE1 `0fea4d0c…`, EVAL30 `b65fb9dc…`. The builder now never overwrites EVAL6 (`_REBUILD`).
- **Check-only on hosen42:** both R5 notebooks passed on the previous build (account, inputs, quota 30.8 of 60 h used). It will be re-run with the final shas at push time; `--sha` is mandatory.

**Validation before GPU:** extend the CPU harness and unit checks:
- harvest (exact / non-exact / rejected-by-host / duplicate / forged signal);
- selection (two distinct, identical, latest-first);
- estimate vs the recorded usage of the 11 drop cases;
- truthful resume;
- cut tool call not executed;
- `replay_real_sandbox.py` on all 43 branches as a regression.

## Rev 5 audit, GPT 1 (2026-10-21): GO for EVAL6R5. Validated

| GPT 1 claim | Validation | Verdict / action |
|---|---|---|
| Notebook shas match; only `arc2_duck.py` changed in the embedded package (1/23); both notebooks compile | Agrees with our build and manifest | true |
| B1-B9(a) correctly placed; legacy smoke with rev 5 off passes | Agrees with our tests (rev 5 unit 40/40, flag-off 41/41, harness r5 30/30, rerun 21/21, launchfail 6/6) | true |
| **N1:** redefining the same name after an exact `check()` in one tool call loses the exact version (retention keeps only the final definition) | True by design: the hook sends only the name, and retention happens after the call. **Frequency in our data: 0** of 405 recorded tool calls (84 EVAL6 + 321 EVAL30) redefine a name, or redefine it after checking it (AST scan) | real edge, never observed. Options: one prompt sentence ("give each version its own name, e.g. transform_v1/_v2"), or nothing. **Doc wording fixed:** "exact checks of the retained (final) definition are kept", not "every exact check" |
| **R1:** verification worst case 24 x 60 s per branch | Measured in the real sandbox on recorded candidates (7 branches, 16 verifications): **0.75-2.71 s each, ≤ 3 per branch** (Windows) | worst case real, observed cost negligible; keep cap 24; report `checks_verified` |
| **R2:** tokenizer fallback (1.9 chars/token) if `tokenizer.json`/`tokenizers` missing | The Kaggle EVAL6 run loaded exactly this file with `tokenizers` in CELL 2 (`tokenizer_check ok: True`, `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/...`). The E12 block reads `SGL_MODEL_PATH` after `_e11_discover` (same cell, appended later) | low risk. Optional: a report field "tokenizer loaded / fallback" (today only a log warning) |
| **R3:** B6 could drop a complete, valid call cut at the limit | All **5** EVAL30 length-cut tool calls were incomplete: 3 parse (stubs: `H=len(inp)`, an import line, a loop body), 2 syntax errors. EVAL6: 0. No complete call was ever cut | no evidence of loss; keep B6 |
| B2 cannot select an invalid grid; replay 5/6 is retrospective, not a prediction | True (`valid_grid` filter; tested) | agreed; keep that wording in the report |
| B3 does not keep all reasoning forever (budget 81,408 + hysteresis) | True | agreed |
| Interpretation table (5-6 / 3-4 / 0-2 of 6; exposure) and "bundle, so no per-fix attribution" | Consistent with our plan | adopted for the run report |
| EVAL30R5: late tasks may get less compute; run EVAL6R5 first | Consistent with the capacity check | adopted |

## Rev 5 audit, GPT 2 (2026-10-21): NO-GO until 1 fix (P0) + tokenizer preflight. Validated

| GPT 2 claim | Validation | Verdict / action |
|---|---|---|
| **P0:** a wrong function that patches `json.dumps` is host-accepted, now also through a forged `CHECKED:` | **Reproduced on the current rev 5 code** (`Analysis scripts/weakness/validate_spoof_r5.py`, real sandbox): forged CHECKED accepted=True **and** the same spoof via `submit()` accepted=True. A correct function that uses `json.dumps` is kept; a wrong identity function is rejected. Not new: the same path was found on 2026-10-19 (submit) and left open (truth-scored eval, never observed: 0 in 43 transcripts). The CHECKED path does not widen it (equal to submit) | **GPT's guard tested as a prototype:** forged CHECKED → rejected; spoof submit → rejected; correct json-using function → still kept; identity → rejected. Adapter-only, a few lines in `_grid_call_code`; it covers `json.dumps` reassignment, not every possible tamper (GPT says so too). **Recommend taking it** (cheap, no false rejects seen), behind the rev 5 flag so old profiles stay identical |
| **HIGH:** B3 silently falls back to 1.9 chars/token if the tokenizer is missing | True (a log warning only). On Kaggle the same file loaded with `tokenizers` in the EVAL6 run (`tokenizer_check ok`) | **Recommend a preflight that records + prints** (`rev5_tokenizer`: loaded / path / probe tokens) in CONFIG and the report. GPT proposes an `assert`; that would turn a fallback (the run still works) into a dead solver, so record rather than assert. User decision |
| B1 limits: same-name redefinition (= GPT 1 N1); the cap also counts rejected verifications | Both true (keys are added before verifying). N1: 0 of 405 recorded calls | accepted as known limits |
| B4: the rewritten resume adds tokens after the trim computed its budget | True: +216 characters (≈ 50-110 tokens) over `ARC2_RESUME_R5`. Duck window 131,072 vs server context 155,648 leaves about 24k of margin | negligible; no change |
| B6 invalid args never reach history | Agrees (tool_agent.py:6862-6887 drops all calls of the reply; harness: the cut code never ran) | true |
| EVAL30R5's 330-min window ≠ EVAL30's 4 h | True | report the runtime difference when comparing with 7/48 |
| Hidden 240-task coverage not solved | True | known (F7) |

## Rev 5 audit, GPT 3 (2026-10-21): CONDITIONAL NO-GO (1 fix). Validated

| GPT 3 claim | Validation | Verdict / action |
|---|---|---|
| Integrity: shas, embedded adapter, vendor byte-identical to rev 4; flag-off 41/41 rerun on Linux | Agrees | true |
| **B6 bypass:** on a `length` reply with **no** parsed `tool_calls`, Duck recovers a call from `<tool_call>` markup in content/reasoning (tool_agent.py:6842-6849) and executes it; the rev 5 override only touches parsed calls | **Reproduced** (`Analysis scripts/weakness/validate_b6_markup.py`, Duck's exact sequence): length + markup in content → EXECUTES; length + markup in reasoning → EXECUTES; length + parsed call → dropped; stop + markup → recovered as designed. **In the recorded runs: 0** responses had markup in text (EVAL6 + EVAL30) | real gap in B6's invariant. **Fix:** on `length` with no parsed calls but markup present, insert one placeholder call with invalid arguments, so Duck's malformed path drops it and skips recovery. Adapter-only |
| **B4 edge:** `_force_reduce_messages` (used only when a server rejection was not fixed by the forced trim, tool_agent.py:6809-6818) bypasses the resume rewrite | True (code read) | **Fix:** same rewrite in an override of `_force_reduce_messages` (tiny) |
| B3 is a hybrid (reasoning by tokenizer, the rest by the calibrated ratio), not exact chat-template accounting | True | accepted. **Add a metric:** estimate vs the server's `prompt_tokens` per request, to validate B3 on GPU |
| B1 verification worst case 24 x 60 s | Same as GPT 1 R1; measured 0.75-2.71 s per verification, ≤ 3 per branch | keep |
| B2 deliberate trade-off (a branch's submission ranks over its checks) | True; not a bug | keep |
| B5 / B7 / B8 / B9(a) OK; bundle means no per-fix attribution | Agrees | report wording |

## Consolidated pre-run change list from the three rev 5 audits (awaiting the user's approval)

| # | Change | From | Evidence | Recommendation |
|---|---|---|---|---|
| R5-1 | Serializer guard in `_grid_call_code` (`json.dumps` reassignment rejected), behind the rev 5 flag | GPT 2 (P0) | reproduced: forged CHECKED and spoof submit both accepted today; guard rejects both and keeps a correct json-using function | **take** |
| R5-2 | B6: block markup recovery on `length` replies | GPT 3 | reproduced (content and reasoning markup execute today) | **take** |
| R5-3 | B4: resume rewrite also in `_force_reduce_messages` | GPT 3 | code read | **take** (tiny) |
| R5-4 | Record + print the tokenizer state (loaded / path / probe tokens); no assert | GPT 1 R2, GPT 2 | fallback is silent today; it loaded on Kaggle before | **take** (record, do not assert) |
| R5-5 | Estimate-vs-actual prompt tokens per request in `rev5_events` | GPT 3 | needed to validate B3 on GPU | **take** (measurement only) |
| R5-6 | One prompt sentence: distinct names for versions tested in one call | GPT 1 N1, GPT 2 | 0 of 405 recorded calls | optional (no evidence of loss) |

Regression to add: spoof via CHECKED / submit / correct json-user; length + markup (content / reasoning / incomplete); stop + markup still recovered; forced-reduce resume; tokenizer record (present / missing); estimate metric present. Then rebuild both notebooks and rerun every suite (unit rev 5 + flag-off, harness r5 / rerun / launchfail / eval6 / smoke, e2e replay).


## R5-1..R5-5 applied (user 2026-10-21: "take them, validate again"); R5-6 not taken (no evidence)

| # | Applied as | Test |
|---|---|---|
| R5-1 | `_grid_call_code(..., guard=self._arc2_rev5)`: restores `json.dumps` after the call and fails verification if it was reassigned | unit: forged CHECKED spoof rejected; submit spoof rejected; a correct json-using function still kept; control with rev 5 off unchanged. `validate_spoof_r5.py` now: all three False/False/True/False |
| R5-2 | `_chat_completion`: on `length` with no parsed call but `<tool_call>` markup, a placeholder call with invalid arguments → Duck's malformed path drops it and skips markup recovery | unit: complete markup in content / reasoning dropped; incomplete markup never executes; plain reasoning = no call; stop + markup still recovered. `validate_b6_markup.py` now: DROPPED / DROPPED / DROPPED / EXECUTES (stop) |
| R5-3 | `_force_reduce_messages` override → the same resume rewrite | unit |
| R5-4 | `tokenizer_status()` in CELL 3 → `CONFIG["e12"]["rev5_tokenizer"]` + print; never fatal | unit: present → loaded; missing (subprocess) → not loaded + reason + fallback count; harness: recorded and loaded in the report |
| R5-5 | `rev5_events["prompt_estimates"]`: [estimate, server prompt_tokens] per request (≤ 400) | unit; harness: present for every branch |

**Final validation (2026-10-21):**
- **Unit checks:**
  - rev 5: 53/53;
  - rev 5 off: 41/41.
- **Notebook harness:**
  - r5: 32/32;
  - rerun: 21/21;
  - launchfail: 6/6;
  - rebuilt rev 4 eval6: 25/25;
  - SMOKE1_REV4: 22/22.
- **End-to-end real-sandbox replay:** EVAL6 5/6 tasks, 7/8 inputs; EVAL30 4 tasks, 7/48.
- **Package and originals:**
  - pyflakes clean;
  - embedded `arc2_duck.py` == source in both R5 notebooks;
  - 0 vendor files changed vs the Kaggle-run rev 4 notebook (23 files);
  - Kaggle-run originals untouched (EVAL6 `53a66856…`, SMOKE1 `0fea4d0c…`, EVAL30 `b65fb9dc…`).
- **Check-only on hosen42:** OK for EVAL6R5 `f4fed1ac5ee332ad3a69eda841a8d7679d39b9a96f68e0b6777e2a6eb87d32a1` and EVAL30R5 `415a68f27ed9108024206d03cf4124ae3952aa1e8acee04ad88a6490ce7facd9`. Quota used 30.79 of 60 h.

## Low-effort variant (user 2026-10-21: "the same notebook, change only the xhigh agent"; chose low)

- **Evidence offered for the choice:**
  - ARC Prize Qwen3.8 DIRECT v2 semi-private: xhigh 42.4 / low 22.8 / medium 13.2 %;
  - our PIV probe (2026-09-15): low reached a tool call in 87.5 % of tasks, medium in 12.5 %;
  - the HF card warns that lower effort does not always reduce agentic task time.
- **Change:**
  - adapter `_harness_template_kwargs` adds `reasoning_effort` when `E12_REASONING_EFFORT` is set (xhigh / medium / low only; Duck's ladder keeps precedence);
  - builder `--low` bakes `low` into `Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL6R5_LOW.ipynb` (sha `cce9cb8f2abe02f02a9f06ac768b9bd005c5168146aa9ae2fb57de2785213ca0`). **[Not the one that ran: the GPU run and the submission used `…EVAL12R5_LOW.ipynb`, sha `26fff41e`, the 12-task profile; user 2026-10-21.]**
  - push script `--low` → kernel `…-eval6r5-low`.
  - Template effect (INT4 `chat_template.jinja`): the system prompt gets "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration." instead of the xhigh instruction.
- **Versus the running EVAL6R5 (`f4fed1ac`):** only the effort lines in CELL 1, the header text and the adapter override. Package: only `arc2_duck.py` differs. The embedded adapter equals the source.
- **Tests:**
  - unit 56/56 (+3: unset → not sent; low → sent with `preserve_thinking`; unsupported "high" → not sent);
  - flag-off 41/41;
  - harness on LOW: r5 33/33 (+1: every request carries `reasoning_effort` low), rerun 21/21, launchfail 6/6;
  - regression: eval6 25/25, smoke 22/22.
- **Check-only:** hosen42 OK (quota 31.29 / 60 used while EVAL6R5 runs).
- **Design:** one variable changed against EVAL6R5 (effort), so the comparison is clean.

## Rev 6 candidates: bring the first check() from ~55k to ~15-25k tokens (2026-10-21; for the user's decision, nothing built)

**Sources:**
- Two transcript readers: xhigh EVAL6R5 (`Analysis scripts/precheck/reader_xhigh.md`) and low EVAL12R5_LOW (`reader_low.md`).
- A cross-run script over all 36 branches: `Analysis scripts/precheck/precheck_behaviours.py` (+ `_output.txt`).
- Every cited line was spot-checked in the transcripts.

**Correction:** the reader caught one labelling error. 62593bfd_b0 (xhigh) *did* check, 2/2 exact at 58k. The branch that never solved is 135a2760_b1. Failing checks are not in branch_result `checked`.

### Measured (script, all branches)

| Measure | xhigh (12) | low (24) |
|---|---|---|
| First check, median tokens | 55.1k | 53.7k |
| Never checked | 1 | 3 |
| Pre-check tokens generated **after** the response's first "use code / in Python / stop guessing" phrase | **273k of 742k (37 %)** | 279k of 1,394k (20 %) |
| Branches with a self-detected misread before check ("miscount", "my manual count was wrong", "completely different from my visual reading" …) | **12/12** | **23/24** |
| Responses hitting the 49,152 cap before check | 2 branches | 2 branches |

- The intent regex is approximate (it can match plans that still need thinking). The misread regex is broad ("recount" can be harmless). Both count *frequency*, not exact waste.
- **Context-loss mechanism (validated, low 4c3d4a41_b0):**
  - The prompt was 66,189 with a 16,365 reply.
  - The next request would need 66k + 16k + a 49,152 output reserve > 131,072, so the history was trimmed and the prompt dropped to 19,856.
  - The next request accepted a hard-coded (overfit) transform.
  - The 49k output reserve forces drops at about 82k of context.

### Behaviours (both readers agree)

1. **Hand reading / counting instead of computing.**
   - Misreads in every slow branch: about 25k (62593bfd), 45k (4c3d4a41), 31k (135a2760), 13k (65b59efc low).
   - The model builds wrong rules on wrong counts and then debates them.
2. **Stated intent to use code, then more thinking.**
   - 62593bfd_b0 xhigh: 43 intent phrases before acting (first at L289, about 3k; rule at L1247, about 37k).
   - 2 × 49k replies with zero calls (low 4c3d4a41, 62593bfd).
   - Evidence that acting helps: the 4c3d4a41_b0 xhigh resume (L1241) "stop hand-simulating" led to 3 count calls, and the rule followed in about 7k.
3. **Rule found, check delayed:**
   - test-output simulation / alternatives about 18k (62593bfd), 6k (136b0064_b1), 5.7k (4c3d4a41);
   - about 26k in 65b59efc low (L3581: "let me first implement the pure as-is rule and see the check results", then it did not).
4. **Own comparison against `p['output']` and hand-settling of variants** instead of `check()`: cbebaa4b low about 24k; 135a2760 xhigh about 25k, refuted by one check.
5. **Debug loops on errors:** disallowed `sys` / `traceback` imports and IndexErrors (135a2760_b1, 898e7135_b0: 6 of 15 calls failed).
6. **Fast branches (136b0064, bf45cf4b, 247ef758, b5ca7ac4):**
   - the first real call after a think is transform + check, even if imperfect;
   - check settles the open details;
   - no own comparison.

### Candidates

| ID | Change | Kind | Evidence | Approval needed |
|---|---|---|---|---|
| R6-1 | **Facts block** in the first user message, computed by the harness. Per grid: dims, background, colour counts. Per component: colour, cell count, bbox, shape string. Per pair: diff count. Plus the prompt line "use only numbers printed by Python or this block; never count cells by hand" | prompt + harness | behaviour 1 (35/36 branches) | normal; adds prefill tokens; measure size |
| R6-2 | Prompt: "When one rule explains every training pair, call check(transform) in the same reply. Compare test outputs only after check is exact, and do it in Python." This replaces the current "compare their test outputs" wording | prompt | behaviour 3 | normal |
| R6-3 | Prompt: "Never compare against p['output'] yourself; to test a partial idea, write a transform for just that part and check it. If two variants fit, write both and check both in one call." Harness: `check({'A': fA, 'B': fB})` prints one line per variant | prompt + helper | behaviours 3-4 | normal |
| R6-4 | check() errors print the failing line and source line inside transform. The prompt lists the allowed imports | helper + prompt | behaviour 5 | normal |
| R6-5 | **Code-intent stop:** after about 3k reasoning tokens, a code-intent phrase ends the stream. Resume with "your reasoning is kept; emit the python call you planned". Alternative: cap only the first request at about 12k | harness | behaviour 2; 37 % / 20 % of pre-check tokens come after the intent | **yes: changes per-request length (60k floor rule).** The earlier "pause-resume negative" result argues for a measured test |
| R6-6 | Output reserve vs context: keep the 49k reserve but count it against the KV, not the window. Or lower the reserve only when the history is above about 80k, so tool results are not dropped (4c3d4a41 low overfit) | harness | validated trim at 66k + 16k + 49k > 131k | **yes if any output cap changes** |

**R6-1..R6-4** are prompt / helper only. They can be bundled and measured on the same EVAL12R5 set with the metric "tokens to first check" (target ≤ 25k median) plus accuracy.

## GPT audit of the rev 5 GPU outputs (2026-10-21, user-pasted): validated, added as N1–N7

**Scope of the audit:** the results ZIP README, the 3 GPU runs, branch JSON, transcripts, candidates, decode logs, the 12 labelled tasks, the donor code and external research.

**GPT's verdict:**
- Keep rev 5. No catastrophic adapter bug was found.
- The next priority is hypothesis generation and compute allocation, not selection.
- Strongest items: N1 + N2 + N3.

### Claim validation

Scripts are inline in the session. The numbers come from `branch_result.json` `checked` / `submissions` and `e12_generated_attempts.json`, compared against `eval_solutions_12.json`.

| # | GPT claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Rev 5 selection lost no test-correct host-verified candidate: oracle = final, xhigh 5/8 and low 10/19 | **Confirmed** | My recount: xhigh final 5/8, oracle 5/8; low final 10/19, oracle 10/19. Selection is not the lever on these tasks |
| 2 | Effective tokens per task under full coverage: about 40–45k at today's 244–265 tok/s; 55–60k or more with higher throughput | **Confirmed (arithmetic)** | 11.3 h × 3,600 × 255 / 240 ≈ 43k. Our earlier 55–72k assumed 330–430 tok/s |
| 3 | 127 requests / 117 python calls (xhigh) and 449 / 437 (low), so about 92–97 % of requests carry a tool call. GPT concludes "the problem is not tool use but what Python is used for" | **Counts confirmed; conclusion only partly.** Tool calls do happen, but the tokens sit in long thinking before each call | xhigh: the 28 of 127 requests with ≥ 10k tokens carry **80 %** of all tokens (the 15 requests ≥ 20k carry 60 %). low: 45 of 449 carry 58 %. Pre-check tokens are about 100 % reasoning, and 35/36 branches show hand-count misreads. Both problems are real: long thinking before calls, and unproductive calls (cbebaa4b_b0 low: 65 requests, 100.6k tokens, no correct candidate) |
| 4 | Late corrections: 135a2760_b1 (low) first train-exact wrong at 71,306, correct at 97,972; 898e7135_b0 (low) first accepted at about 56.9k, correct at about 73.5k | **Confirmed**, plus one more case | 135a2760_b1: 71,306 → 97,972. 898e7135_b0: 56,868 → 73,523. **898e7135_b1: 77,859 → 79,662** (new). Stopping at the first train-exact candidate would lose these |
| 5 | The outer scheduler (arc2_duck.py:962–1033) is task-major, with a ThreadPoolExecutor of n_live = 6, so the next task starts only when a branch finishes | **Confirmed** | `arc2_duck.py:973` (task-major when `live_branches` is set), `:975` (attempt-major otherwise), `:1020` (the pool). There is no budget redistribution and no coverage admission |
| 6 | 4c3d4a41 overfit: "out[r][c + 10] = 5"; both branches in both runs are train-exact but test-wrong | **Confirmed, and worse** | xhigh 4c3d4a41_b0 L1706: the model itself wrote "**+10 is specific to the training layout** … A more generic version …" and still accepted the fixed offset |
| 7 | SGLang issues: #38009 (DFlash2 and target-only outputs diverge in thinking mode), #36548 (concurrent state contamination), #37475 (draft `target_layer_ids` fallback), #28180 (penalties under DFlash) | #28180 was verified earlier (fix list F3). **#38009, #36548 and #37475 are not verified by me (C)** | Before going to 10–12 streams, check those three issues and our drafter config (`target_layer_ids` in the DFlash2 checkpoint) |
| 8 | Qwen model card: lower effort can raise retries and total latency in multi-turn agentic tasks | Not re-checked (C), but consistent with our data | Low: 18.7 requests per branch vs 10.6 for xhigh; 72.7k tokens per branch vs 70.5k |
| 9 | Franzen WRITEUP: prefix reuse 93.43 %, 609 tok/s harness-side, p90 decode 836.6 tok/s, prefill fraction 15.25 % | Not re-checked (C; measured on ARC-3 with an RTX PRO 6000) | Our P1 ARC-3 notes measured 590–617 tok/s for his build, the same order |
| 10 | Mixed effort is optional research, not a fix | Agree | — |

### New fix items (GPT N1–N7), merged with R6

| ID | Change | Priority | Pre-GPU check | Relation to R6 / earlier items |
|---|---|---|---|---|
| **N1** | **Progress-aware branch accounting with bounded continuation.** Per-branch state: generated tokens, best train pairs correct, host-verified candidates, distinct test predictions, recent improvement, trimmed, warm prefix. Classes: unproductive / partial_progress / candidate_available / multiple_candidates | P0 | Replay: 135a2760_b1 and 898e7135_b0/b1 must still reach their late correct candidates; cbebaa4b_b0 must stay bounded | New. **A budget cut for "unproductive" branches lowers thinking below the current soft cap, so its numbers need the user's approval (60k floor)** |
| **N2** | **Coverage-first admission, with second branches allocated separately.** Every task gets its first opportunity before second branches take compute. Promising branches get extra compute without losing their warm context or retained functions. At the deadline, unfinished tasks are submitted from their current verified candidates (reuse `build_submission` and the atomic fallback) | P0 | A 240-task CPU simulator: no starvation, deadline and fallback correct, no leaked gate slots | E4/E5 in the GPT research bundle. The attempt-major mode already exists (`:975`) |
| **N3** | **Short state-aware nudges, sent once per state transition.** No train pair passes yet: "implement a provisional transform now, check()". Partial: "focus on the failing examples, preserve the working behaviour". Train-exact saved: "consider generalization to the test structure; check a materially different hypothesis or submit" | P1 | No duplicate nudges; the tool protocol is unchanged | Overlaps R6-2 / B2 (reason-specific resume texts) |
| **N4** | **Concurrency up to 10, capacity-qualified** (KV and mamba headroom, an admission reserve, a rollback path), not a hard-coded 12 | P1 | KV/mamba headroom; a per-request isolation test for #36548 (a distinct marker in each prompt) | E1/E2; the 11-of-12 bug |
| **N5** | **Generalization feedback after the first train-exact candidate:** "check fixed coordinates, offsets, counts and sizes against the test inputs; keep a different plausible rule as an alternative". Never auto-reject: a fixed offset can be the real rule | P1 | 4c3d4a41 replay; valid fixed-coordinate rules are not rejected | F1 |
| **N6** | **Speculative-serving correctness preflight:** a static check of the DFlash2 `target_layer_ids`, and a bounded comparison of target-only vs DFlash2 output with thinking on and off | P1 | Inside the same GPU session, bounded | New (the C issues) |
| **N7** | **More logging:** solver tokens, prefill, accepted drafts, tokens to the first exact and first useful candidate, the train-exact vs test-correct gap | P1 | — | Metrics list |
| KEEP | Rev 5 harvest, pass@2, tokenizer estimator, serializer guard, cut-call handling | Mandatory | All earlier regressions pass | — |

### Offline regressions GPT requires before GPU

| Case | Expected |
|---|---|
| 135a2760_b1 | A correct candidate is still reachable after the first wrong train-exact candidate |
| 898e7135_b0 | The ~57k → 74k refinement is not stopped early |
| cbebaa4b_b0 | Bounded budget; valid artifacts kept |
| 4c3d4a41 | Train pass, no invented confidence, alternative search allowed |
| 240-task simulation | No starvation; coverage accounting correct |
| 6 → 10 streams | No deadlock, no leaked slots, no forgotten branch |
| Security | Forged CHECKED / submit calls still rejected |
| Selection | The same archives still give 5/8 and 10/19 |
| Cut calls / trims | No partial execution, no false continuity |
| Deadline | One valid attempt pair per test input |

Offline ground truth is never a runtime input.

### Success criteria for the next run (GPT)

- **Primary:** correct outputs per GPU-hour.
- **Coverage:** count genuine model attempts, not fallback JSON.
- **Efficiency:** tokens to the first useful candidate.
- **Overfit:** the gap between train-exact and test-correct candidates.

Fewer tokens per task with fewer correct outputs per GPU-hour is not an improvement.

### My notes

- **The two plans are complementary.** GPT's audit covers compute allocation (N1/N2). The code-first research bundle covers input representation and early action: A1 (no grid text), A2 (facts block), B1/B2 (short requests with continuation), B4 (auto-check).
- **Claim 3 does not remove the hand-reading lever.** Most tokens sit in long pre-call thinking, which is what A and B target.
- **Proposed rev 6 bundle:** N1 + N2 + N3 + N5, plus A1/A2/B2/B4, plus the N4/N6 diagnostics in the same GPU session.
- **Approval needed for:** any budget cut in N1, B1, and R6-5.

## GPT audit 2 of the rev 5 GPU outputs (2026-10-21, user-pasted): validated, items C1–C10

This audit used the same results ZIP, the donor code, the Qwen Hugging Face documentation, GitHub repositories and papers.

**GPT's verdict:** GO for one consolidated efficiency + coverage revision. The order it recommends:
1. new-task exposure first;
2. then rescue tasks that have no candidate;
3. then a second hypothesis on ambiguous verified tasks.

### Claim validation

I re-ran every check myself. The selection replay imports the real `arc2_duck.select_attempts_r5`; the ground truth is `eval_solutions_12.json`.

| # | GPT claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Token positions of the first correct candidate: xhigh 32k, 35k, 54k, 58k, 62k; low 14k–98k | **Confirmed** | xhigh [32, 35, 54, 58, 62]k. low (minimum per branch) [14, 19, 29, 42, 44, 47, 56, 74, 80, 98]k; GPT lists 11 events because it counts events, not branches |
| 2 | **Selective branch 1** (use branch 1 only when branch 0 has no host-accepted candidate) | **Confirmed exactly** | **xhigh: 5/8 at 498,253 tokens vs 5/8 at 846,080 (−41.1 %)**; branch 1 needed only for 898e7135. **low: 9/19 at 1,094,372 vs 10/19 at 1,743,889 (−37.2 %)**; branch 1 needed for 136b0064, 62593bfd, 65b59efc. The lost output is 135a2760: branch 0 was train-exact but test-wrong, and branch 1 found the correct rule. This is a retrospective replay; in a live run branch 1 order, KV residency and sampling would differ |
| 3 | The branch stop condition (arc2_duck.py:820–824) looks only at `arc2_accepted`, so harvested check() candidates (`arc2_checked`) are not progress for the driver | **Confirmed** | `should_stop()` returns on `agent.arc2_accepted` or the deadline only. Not stopping at the first exact candidate is intended (late corrections); the gap is that the scheduler cannot see the progress |
| 4 | The opener (arc2_duck.py:503–508) looks only at `arc2_submissions` | **Confirmed, and wider** | Every resume after a yield sends `ARC2_RESUME_R5` = "**Nothing has been submitted yet.** …", even after a candidate was harvested. The model sees the one-time tool note "Kept `transform` as a candidate answer …" and then a resume that ignores it |
| 5 | low 135a2760_b0: first host-verified candidate at 68,423 tokens, branch total 114,538 (+46,115 tokens after it, no new candidate) | **Confirmed** | `checked` = [('transform', 68423, True)], generated 114,538 |
| 6 | 4c3d4a41 xhigh b1: the model wrote `transform_alt()` and diffed the two candidates' test predictions, but never ran `check(transform_alt)`, so the alternative never entered the portfolio | **Confirmed** | transcript L1003 / L1577 define `transform_alt`, L1597 compares, no `check(transform_alt)`. GPT: the alternative would fix some cells but not solve the task (not re-checked) |
| 7 | 4c3d4a41 was 2/2 in rev 4 xhigh and 0/2 in rev 5; the cause cannot be separated (several changes at once) | **Confirmed** | Rev 5 runs analysis, per-task table |
| 8 | Median completion per request 1,115 (xhigh) / 1,090 (low); median prompt 51.9k / 41.6k; cached fraction 85.4 % / 91.0 %; uncached about 888k / 1.577M | **Confirmed exactly** | Recomputed from `usage` |
| 9 | 898e7135 xhigh: both branches about 216k, no accepted candidate; solved in low | **Confirmed** | 107,202 + 108,897 |
| 10 | f931b4a8 low: branch 0 2/2 correct; branch 1 spent about 97k more and added no new correct output | **Confirmed** | b0 candidate at 47,466 (total 50,713); b1 candidate only at 96,765 (total 96,854) |
| 11 | bf45cf4b low: both branches accepted at about 14k / 19k, so the work is duplicated | **Confirmed** (rev 5 runs analysis) | — |
| 12 | `run_branch()` creates the agent inside the function, so a later stage that calls it again starts a fresh agent (history and retained functions lost); pause/resume needs a state holder | **Confirmed** | arc2_duck.py:815-817 `agent = agent_cls(...)` inside `run_branch` |
| 13 | Qwen card / HF discussion #173: xhigh and low inject instructions, medium injects none; lower effort can mean more retries in multi-turn agents | Consistent with what we verified earlier (template: low injects "Keep your thinking brief…"); #173 not re-checked (C) | — |
| 14 | Franzen WRITEUP: removing all reasoning from history was a large regression; a limited admission count keeps prefix reuse; 10 active streams with a cache-aware priority | Consistent with the vendored code (C for the write-up numbers) | `preserve_thinking`, drain hysteresis, priority gate |
| 15 | SGLang `schedule_policy.py` LPM: verify on 0.5.21 before relying on it | Agree | E3 in the research bundle |

### Items (GPT audit 2), merged with N1–N7 and A–F

| ID | Change | Priority | Merges with |
|---|---|---|---|
| **C1** | **First-exposure-first scheduler; branch 1 conditional.** Stage 1: branch 0 of every task. Stage 2 runs on live evidence: no verified candidate → rescue; conflicting predictions → ambiguity; otherwise defer branch 1 | P0 | N2, E4. **Replay evidence: −37 to −41 % tokens, xhigh 5/8 kept, low 10 → 9** |
| **C2** | **Expose harvested candidate progress** to the driver, scheduler and model. Record the token count of the last verified candidate, the verified count and its name. The resume says "`X` is already host-verified on every training pair and saved. Submit it if ready, or test a genuinely different rule." Not an auto-stop | P0 | N1, N3. Also fixes the wrong "Nothing has been submitted yet" resume |
| **C3** | **Global compute-budget accounting** with safe branch checkpoints and escalation. Budget is enforced only at a request boundary; never by a small `max_tokens` that cuts a tool call (B6 would drop it) | P0 | N1, E5. **Any budget below the current cap needs the user's approval (60k floor)** |
| **C4** | **No Duck state reset between phases.** If a branch pauses, keep its history, retained functions, candidates, token counters and trim state (a state holder, or Duck's own serialize/restore). A fresh, independent branch 1 is fine | P0 | Required by C1/C3 |
| **C5** | **Short, consistent ARC-2 prompt:** "single rule" → executable candidate rules (alternatives allowed); remove "Investigate until your code has a clear plan" (already replaced in rev 5); explain that verified candidates are kept and alternatives are valuable; "if two rules seem train-perfect, check() both — don't only print their test diff". No long procedural lists (Johan Land §8.6); do not duplicate the template's effort text | P1 | B6, C1, N5 |
| **C6** | **Small deterministic grid helpers** (colour counts, bboxes, components with 4- vs 8-connectivity, compact diffs) as an optional library; no hard-coded ARC heuristics | P1 | A2/A3 |
| **C7** | **Prefill/cache-aware admission, initial target 10 streams**; record evictions, queue backlog, speculative acceptance and effective tok/s | P1 | N4, E1–E3 |
| **C8** | **Candidate-disagreement and overfit telemetry** | P1 | N7, F1 |
| **C9** | **Per-task tokens, wall time and candidate yield accounting** | Mandatory | N7 |
| C10 | A full judge ensemble, a new model or a broad runtime rewrite | Not now | — |
| (CSV) | CSV grid format: a hypothesis to test on GPU; check formatting and token count on CPU | P2 | A5 |

### GPT's "avoid" list

- dropping Qwen3.8-27B or adopting a new framework;
- deleting reasoning history indiscriminately;
- a 30k hard cap on all branches (xhigh recorded coverage would fall from 5/8 to 1/8);
- a universal auto-stop at the first train-exact candidate;
- copying the ARC-AGI-3 level-priority formula verbatim;
- merging the contexts of the two branches;
- using eval truth for routing;
- a full multi-model judge.

### Offline validation required before GPU (GPT audit 2)

This adds to the N-list regressions:
- a 240-task admission simulation;
- the scheduler at 1 / 6 / 10 slots, with no deadlock, starvation or duplicate run;
- pause/resume invariants: conversation, retained functions, candidates and counters unchanged;
- budget boundary: the token count is correct and a cut call never executes;
- the first verified candidate is visible to the scheduler;
- after a wrong train-exact candidate, a second hypothesis is still possible;
- 4c3d4a41: `check(transform_alt)` yields two distinct verified predictions (run the recorded code in the real sandbox);
- rev 5 safety regressions still pass;
- context/prefix consistency;
- the submission is valid and distinct;
- the recorded EVAL6R5 / EVAL12R5 candidates are not lost.

Routing is never tuned on known truth.

### My notes

- **The two GPT audits agree on compute allocation.** C1 ≈ N2, and C2/C3 ≈ N1.
- **The strongest new evidence is the selective branch-1 replay:** −37 to −41 % tokens at almost the same score.
- **Combined with the code-first items** (A1 no grid text, A2/C6 facts and helpers, B2/C2 state-aware resume, B4 auto-check), the rev 6 bundle attacks both the tokens per branch and the number of branches.
- **Needing the user's approval:** any budget cut in N1/C3, B1 (short requests), and R6-5.
- **Nothing built yet.**

## GPT reviews of the code-first research bundle (2026-10-21, two reports, user-pasted): validated, rev 6 design

**Sources:**
- Report A ("REV6 — Code-First Research ও Consolidated Design Review"), received in full.
- Report B ("Rev6 — Code-First Research & Deep Audit"), **truncated in the paste after §9** (serving risks). Everything after that point is missing; ask the user for the rest if it matters.

**Verdict (both reports):**
- GO for rev 6 *design development*. Keep Qwen3.8-27B + INT4 + SGLang 0.5.21 + Duck.
- The lever is **what the model sees, when it can run Python, and how compute is allocated**, not more instructions.
- GPU run only after an implementation audit.

### Claim validation (mine)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | 273k (xhigh) / 279k (low) pre-check tokens come after the first code intent; first check at a median of 55.1k / 53.7k; 1/12 and 3/24 branches never checked | **Confirmed** | `precheck_behaviours_output.txt` |
| 2 | All raw grids are in the system prompt (arc2_duck.py:315–330) | **Confirmed** | `arc2_system_prompt` loops over train and test |
| 3 | Raw grid text per task: median 3,737 characters; prototype facts median 875; facts shorter on 12/12 (B, CPU) | **Raw part confirmed**: my median 3,729.5, range 1,914–8,036 characters (digits plus spaces). Facts length not re-checked (B's prototype is not in the paste) | — |
| 4 | Small-grid exception (≤ 64 cells shown as text) (B) | Measured: **only 10 of 87 grids** in the 12 tasks are ≤ 64 cells, so the exception rarely fires | — |
| 5 | Token budget per task: 41–45k at 244–265 tok/s; 56–73k at 330–430. Two branches × 60k × 240 = 28.8M tokens = 18.6 h at 430 tok/s (B). 240 × 60k = 14.4M = 15.1 h at 265 and about 10 h at 400 (A) | **Arithmetic confirmed** | Full coverage needs branches to finish naturally well below 60k, and only conditional second branches |
| 6 | Context arithmetic: input budget 131,072 − 49,152 − 512 = 81,408; with 12,288 it becomes 118,272 (+36.9k) | **Confirmed** | Matches the rev 5 settings (input budget 81,408). A bigger input budget can raise live KV per stream; measure with 10 streams |
| 7 | **Scheduler trap S1:** with `live_branches` unset, the job order becomes attempt-major but `n_live = len(jobs)`, i.e. all 480 branches live (the EVAL30 KV-thrash class). Queue position is also tied to the same switch | **Confirmed** | arc2_duck.py:976 `n_live = min(len(jobs), int(live_branches)) if live_branches else len(jobs)`; :984 `qpos = (i * branches + b) if live_branches else (b * 100000 + i)`. **Job order, worker count and queue priority must be independent settings** |
| 8 | `diff_summary` must not zip unequal shapes; report shape_changed instead (A, B) | Agree (design rule) | ARC-2 output size often differs |
| 9 | Say "modal_color", never "background"; report 4- and 8-connected components separately (B) | Agree | — |
| 10 | SGLang #31954 (LPM starves cold requests), #39087 (quantized DFlash2 drafter acceptance collapse; our drafter is BF16), #43514 (required tool_choice poisons DFlash batches), z-lab/dflash #166 (multi-turn prefix-cache regression, vLLM), plus #36548 / #38009 / #37475 | **Not verified by me (C)** | Actions: an isolation smoke, a `target_layer_ids` check, acceptance monitoring, never named/required tool_choice (also our own Exp 4 lesson), outer coverage fairness independent of LPM |
| 11 | Qwen `chat_template.jinja`: `preserve_thinking` rebuilds earlier reasoning but only for the history that is kept | Consistent with Duck (trims drop blocks) | — |

### Rev 6 design: where A and B agree

| ID | Change | Priority | Approval |
|---|---|---|---|
| **A1** | **Hybrid grid view.** No large raw grids in the system prompt; full exact grids in `train` / `test_inputs`. Requirements: byte-equal globals, verification on the original pairs, the model can always print a crop or a full grid on demand. B adds an optional raw view for small grids (≤ 64 cells; experimental) | P0, the highest-impact hypothesis | Design approval (no thinking cut) |
| **A2** | **Bounded facts block** (A: 300–600 Qwen tokens, hard cap about 800 with an explicit "details omitted, call objects()" notice). Contents: dims, exact colour counts, modal colour, same/different shape, changed cells and changed bbox (same shape only), 4-connected component counts by colour, a few largest components. Symmetry, periodicity, shape strings and coordinates on demand only. **Measure the p50/p90/p99 size with the Qwen tokenizer on the 120 eval tasks** | P0 | — |
| **A3** | **Helpers** in the trusted prelude: `arc_overview()` (zero-argument wrapper), `grid_summary`, `color_counts`, `components(grid, color=None, connectivity=4 \| diagonal)`, `objects`, `bbox`, `diff_summary` / `grid_diff` (shape-aware), `crop`, `check_many({name: fn})` (additive; every train-exact variant host-verified separately). Reuse the donor `segment_layer` algorithm through an ARC-2 wrapper; no fake `current_frame` | P0 | — |
| **B2** | **State-aware resume texts** with precedence cut call → trimmed history → verified candidate → partial progress → no-tool reply. Never duplicate or contradict; claim continuity only for what Duck kept. Both reports give verbatim drafts (A: A–E, B: five texts). This replaces the current always-sent "Nothing has been submitted yet" (validated as wrong after a harvest) | P0 | — |
| **B4/B5** | **Auto-check** when a call defines or changes `transform`. It runs after the snippet executes (no host execution of model code; AST only for detection); deduplicated by source; never auto-submit or auto-stop; rev 5 host verification unchanged. Plus `check_many` | P1 (A) / P0 (B) | — |
| **C1** | **Short operational prompt** (both give verbatim drafts): the grids live in Python; reasoning holds hypotheses and plans; Python does counts, positions, components, diffs and test outputs; "first useful action = a small inspection or a provisional transform + check"; "if two rules fit, check each before comparing test predictions"; "train-exact ≠ generalizes". Keep the Duck tool-call guidance, persistent-functions contract, allowed imports and timeout verbatim. **No rigid stages** (V3.1's "by the second call define solution" is deliberately dropped) | P1 | — |
| **D1** | **The context reserve equals the real per-request `max_tokens`** (otherwise trims drop history, or the server overflows) | P0 with B1 | with B1 |
| **E4/E5** | **First exposure for every task:** branch 0 of all tasks first, then a conditional branch 1. Priority order: no candidate → one train-exact with a generalization concern → two distinct verified predictions (lowest). Fair-share accounting from remaining wall time, observed useful tok/s, unfinished tasks and the submission margin; a deadline-safe fallback. **A branch stays coherent** (no mid-branch preemption in the first rev 6); `run_branch` builds a fresh agent, so there is no pause/resume across phases | P0 | Yes, if any budget falls below the 60k floor |
| **E1–E3** | **10 live streams, qualified** (KV / mamba / isolation first; fall back to 6/8). `lpm`, `--page-size 64` and `--mamba-track-interval 64` only after a support and memory check | P1 | — |
| **F1** | **One bounded generalization reminder** after the first train-exact candidate (both give text); never auto-reject fixed-coordinate rules | P1 | — |
| **N6 / R6-I** | **Serving diagnostics:** isolation, DFlash `target_layer_ids`, acceptance, a target-only vs DFlash parity check at T 0 (diagnostic only) | P1 | — |
| KEEP | Rev 5 safety: atomic submission, tokenizer estimate, candidate archive, cut-call drop, serializer guard, forged-CHECKED rejection | Mandatory | — |

**Not in the first rev 6 (A):**
- A4 CPU rule library
- A6 image input
- B3 text-triggered stream cancel
- B8 effort ladder
- an LLM judge

### Where they differ: B1 short requests (user decision; the 60k floor rule)

| | Report A | Report B |
|---|---|---|
| First request `max_tokens` | **16,384** | **12,288** (Duck default) |
| No-tool continuation | 16,384 | 12,288 |
| After meaningful tool progress | 24,576 | 12,288 |
| Repeated no-tool rescue | up to 32,768, bounded | after 2 consecutive no-tool cuts: a stalled-trajectory report, then a longer continuation / rescue or a scheduler decision |
| Branch allowance | ≥ 60k kept; normal soft ceiling 100k | cumulative ≥ 60k kept |
| Thinking | preserved (`preserve_thinking`), xhigh, T 1.0, yield 2,048, `tool_choice` auto | same |

**Both reports say this needs the user's explicit approval**, and both say A1 + A2 (+ B2) must ship with it. Short requests alone failed in EVAL30 (29/30 first requests cut, with full grids in the prompt).

### Offline checks and GPU gates (both reports)

**Offline:**
- grid omission (no unwanted rows in the prompt; globals exact);
- facts correctness, including shape mismatch;
- components: 4/8 connectivity, holes, touching colours, 1×1, 30×30, checkerboard;
- crop index semantics;
- auto-check dedupe and no false accept;
- the rev 5 security regressions;
- short-request resume: no lost call, no false continuity;
- the context reserve equals the request cap;
- candidate state visible to the scheduler;
- a 240-task admission simulation, including the S1 trap;
- the deadline fallback;
- the rev 5 archives still give 5/8 and 10/19.

**GPU gates, one session:**
1. **Serving:** health, DFlash config, KV/mamba, real concurrency.
2. **Code-first:** Python really runs in the first requests; reasoning and retained functions carry over.
3. **Early verification:** median tokens to first check ≤ 25k on EVAL12 (a stretch target, not a hard gate).
4. **Accuracy vs rev 5:** xhigh 4/6, low 7/12, plus oracle coverage.
5. **Full coverage:** projected completion before the deadline, logged end-to-end tok/s.

**Success metric:** correct outputs per GPU-hour, counting meaningful (non-fallback) attempts. An early check with no useful transforms is not an improvement (the V3.1 warning).

### Open risks (both reports)

1. **A1 may remove visual information** the model needs (the largest accuracy risk).
2. **Short turns may fragment reasoning.**
3. **Auto-check sandbox overhead.**
4. **10 streams may be slower** than 6 if KV thrashes or DFlash acceptance drops.
5. **Coverage vs the 60k floor:** full coverage requires early natural completion.
6. **Train-exact ≠ test-correct** remains the main accuracy limit.
7. **Exposed vs meaningfully attempted tasks** must be counted separately.

### My notes

- **Both reports converge on the same bundle:** A1 + A2 + A3 + B2 + B4/B5 + C1 + D1 + E4/E5 + F1 + qualified 10 streams + serving diagnostics + rev 5 KEEP. This also covers the earlier N1–N7 and C1–C10 items (C1 ≈ E4, C2 ≈ B2 + state, C6 ≈ A3).
- **New, validated:** the S1 scheduler trap (`n_live = len(jobs)` and `qpos` both tied to `live_branches`). It must be fixed before any attempt-major run.
- **Decision needed from the user:** the B1 numbers, i.e. 12,288 (Duck default) vs 16,384 / 24,576 / 32,768 (A), or no short requests. Everything else can be built without changing thinking length.
- **Nothing built.**

### Addendum: report B's missing tail and a third report C ("Rev 6 — Code-First Duck Agent: Deep Research & Proposed Design")

**What the user sent:**
- Report A again, unchanged (already recorded above).
- Report B in full, including the part that was cut at §9.
- A third review, report C.

#### Report B, the part that was missing (§9 end to §12)

**Serving-issue limits:**
- Do not assume our 0.5.21 build is affected by any issue.
- The T 0 parity test is a diagnostic only; production stays at T 1.0.
- No upstream kernel cherry-pick until the failure is reproduced.

**REFRAIN early exit** (ACL 2026, 20–55 % token savings reported): **do not port now.**
- It is not validated on Qwen3.8, ARC or Duck.
- It conflicts with our measured negatives on early stopping.

**GPU smoke gates:**
- Qwen template: xhigh, preserved thinking, qwen3_coder parses tool calls.
- First tool call within a median of ≤ 12k tokens on the pilot tasks.
- First check() within a median of ≤ 25k tokens.
- Auto-check gives feedback and never a false accept.
- Full reasoning is saved, and live requests never overflow.
- DFlash target-only parity matches.
- Concurrency isolation holds.
- 10-stream admission works.
- No KV thrash.
- Selection never discards a correct candidate.
- Rev 5 security checks still pass.
- The 240-task simulator passes.

**Accuracy is compared on the same task sets:** EVAL6R5 xhigh 5/8, EVAL12R5 low 10/19. A rev 6 xhigh run on EVAL12 is not a causal comparison against the low baseline.

**Do not add now:**
- a new model, LoRA or RL;
- a full judge;
- a large DSL;
- an unverified vision path;
- a streaming rewrite;
- untested speculative kernel patches.

#### Report C: validation (mine)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| C-1 | Capacity at 11.3 h for 240 tasks: 255 tok/s → 43.2k tokens per task, 350 → 59.3k, 416 → 70.5k | **Confirmed** (arithmetic) | — |
| C-2 | Duck sets `_reply_reserve_tokens = _max_output_tokens`, so changing only the HTTP `max_tokens` leaves the context budget unchanged | **Confirmed, with a refinement** | `tool_agent.py:3498` `self._reply_reserve_tokens = self._max_output_tokens or 512` (budget at :3506). A **static** cap set through `LOCAL_ANALYZER_MAX_OUTPUT` keeps the reserve in step automatically. A **dynamic** per-request cap (12k → 24k) breaks the sync unless both values are updated together. So D1 matters only for dynamic caps |
| C-3 | Auto-check belongs in the adapter's `_run_python_tool()` layer | **Confirmed** | Duck `tool_agent.py:5717`; the adapter already overrides it at `arc2_duck.py:633` |
| C-4 | Small-grid text exception: each grid ≤ 8×8 and ≤ 256 displayed cells in total | **Measured: 0 of 12 tasks qualify** | So in practice C is a pure facts view on eval-like tasks. Report B's per-grid ≤ 64 cells matches only 10 of 87 grids |
| C-5 | Qwen3.8 makes `<think>` before a tool call, but nothing forces it to finish solving first; the model *can* call early but usually does not | **Accepted as a correction of my wording** ("must finish a full solve", quoted from the PIV notes). It is a tendency, not a template rule | — |
| C-6 | The Franzen yield is checked between requests, not as a hard cut | **Confirmed** | Validated earlier (`tool_agent.py` yield check between requests) |
| C-7 | The current resume says "Nothing has been submitted yet" even after a harvest | **Confirmed** | GPT audit 2, claim 4 |
| C-8 | `run_branch()` builds a fresh agent (arc2_duck.py:805–817), so a scheduler that pauses a branch must keep the same agent object | **Confirmed** | — |

#### Where report C differs from A and B (the user decides)

| Topic | A | B | C |
|---|---|---|---|
| B1 request caps | 16,384 first / no-tool; 24,576 after tool progress; ≤ 32,768 rescue | 12,288 everywhere; after 2 no-tool cuts, a stalled report and a longer rescue | **12,288 until the first successful tool execution, then 24,576** |
| Facts budget | 300–600 Qwen tokens, hard cap about 800 | not set (≈ 875 characters in its prototype) | **1,200–1,500 Qwen tokens**, measured with the pinned tokenizer |
| Raw grids in text | none (optional for very small tasks) | grids ≤ 64 cells (10/87) | tasks with all grids ≤ 8×8 and ≤ 256 cells in total (0/12), as CSV |
| Scheduler | branches stay coherent; no mid-branch pause in the first rev 6 | phases; branch 1 conditional | **pause and resume the same agent object** after a bounded first exposure (25–40k per task), then progress-based compute. **This needs approval**: it pauses a branch below the 60k floor |
| Prompt | "Submit a verified function when ready" | "Use submit when ready to end this branch" | "A submission ends this branch" plus "preserve different predictions; do not discard a valid alternative because another explanation is simpler" |

#### Approvals C asks for explicitly

- **A.** 12,288 / 24,576 per-request caps, with the accumulated thinking preserved.
- **B.** In coverage-first scheduling, pause or defer a branch after a 25–40k first opportunity per task, with late refinement allowed on selected tasks.

#### C's gates

| Gate | Check | Stop if it fails |
|---|---|---|
| G1 | Facts are exact, nothing leaks, the token budget holds | Yes |
| G2 | First Python call within a median of ≤ 12k tokens | — |
| G3 | First check() within ≤ 25k (stretch ≤ 15k) | — |
| G4 | Auto-check works and forged results are rejected | Yes |
| G5 | Cut and resume are truthful | Yes |
| G6 | The 240-task simulation keeps the same agent state after a pause | Yes |
| G7 | Serving is stable at 10 streams | — |
| G8 | The candidate portfolio is not lost | Yes |
| G9 | Accuracy matches rev 5 | — |
| G10 | Correct outputs per GPU-hour improve | — |

**CPU regressions:** use the recorded sources of 135a2760, 898e7135 and 4c3d4a41.

#### Consolidated after A + B + C

- **All three agree on the code-first bundle.** A1 facts view (grids exact in Python) + A2 facts block + A3 helpers (shape-aware diff, 4/8 connectivity, "modal colour") + B2 truthful state-aware resumes (fixing "Nothing has been submitted yet") + B4 auto-check in `_run_python_tool` after execution (dedupe, never submit or stop) + B5 `check_many` + C1 short prompt without rigid stages + first-exposure scheduling with a conditional branch 1 (fixing the S1 trap) + 10 qualified streams + serving diagnostics + rev 5 KEEP.
- **The user must decide three things:**
  1. **B1 caps:** none / 12,288 / 12,288 → 24,576 (C) / 16,384 → 24,576 → 32,768 (A).
  2. **First-exposure budget:** branches run to completion (A/B, no cut), or a pause/defer after 25–40k (C, needs approval and a state holder).
  3. **Facts-block size:** about 600–800 (A) vs 1,200–1,500 (C). I will measure both on the 120 eval tasks with the Qwen tokenizer before the build.
- **Nothing built.**

## Rev 6 gap check (2026-10-21, user: "re-read every GPT and agent recommendation, something important may be missing")

**Method:**
- I re-read the five GPT reports myself: rev 5 audits 1 and 2, and code-first reviews A, B and C.
- Three agents each read a share of the 12 research notes, the reader reports, the rev 5 analysis, the research brief and this whole fix list, and listed what is missing or contradicted. Their reports are in `_scratch/session_1c6db7aa/gapcheck/gaps_set1..3.md`. I read all three in full.
- I checked every code claim marked "code" below myself in `arc2_duck.py` and the vendored Duck.
- The verbatim prompt and resume drafts are in `E12 rev 6 prompt and resume drafts from GPT (verbatim, 2026-10-21).md`. Until today they existed only in the chat.

### 1. Rev 6 ID mapping (use these IDs from now on; the old IDs B1/B2/C1/F1 collided)

| Rev 6 ID | Item | Former IDs |
|---|---|---|
| **R6-V1** | Facts view: no large raw grids in the prompt; exact grids in `train`/`test_inputs` | A1, R6-A |
| **R6-V2** | Facts block, **in the system prompt** (see T2) | A2, R6-1, R6-B |
| **R6-V3** | Helpers, solution-usable and exploration-only (see T1) | A3, C6 |
| **R6-V4** | Print helper and grid-print guard (see T4) | B7, A5 (re-scoped) |
| **R6-P1** | Short operational prompt, no rigid stages | C1 (prompt), C5, R6-2, R6-3, F4 |
| **R6-P2** | State-aware resumes built on Duck's `_resume_reason` | B2, N3, C2 (resume), R6-D |
| **R6-P3** | One generalization note after the first verified candidate | F1 (reminder), N5 |
| **R6-K1** | Auto-check on any newly defined or changed grid function (see T3) | B4, R6-C |
| **R6-K2** | `check_many`, verdict-first output, failing line inside transform, infra errors labelled | B5, R6-3, R6-4 |
| **R6-K3** | Verification dedupe and cap fixed for auto-check (see T5) | new |
| **R6-S1** | First exposure for every task in ORDER2 order; job order, workers and queue priority as independent settings (the S1 trap) | C1 (scheduler), E4, N2 |
| **R6-S2** | Conditional branch 1, stage-2 order (see T7) | C1, E4, N2 |
| **R6-S3** | Fair-share accounting, overshoot modelled, last-admission cutoff | E5, C3, N1 |
| **R6-S4** | Progress signal to scheduler and model (single-function, identity-relative) | N1, C2 |
| **R6-S5** | 10 streams, qualified | E1–E3, N4, C7 |
| **R6-S6** | Deadline safety and fallback | N2, E7 |
| **R6-Q1** | Request caps (user decision) | B1, R6-E, R6-5 (alternative) |
| **R6-Q2** | First-exposure budget / pause (user decision) | report C approval B, R6-G |
| **R6-D1** | Context reserve equals the real cap; trims at about 82k | D1, R6-6 |
| **R6-D2** | Prefix-cache invariants | D3, new |
| **R6-D3** | Trim order keeps tool results | D2 (brief) |
| **R6-X1** | Serving diagnostics | N6, R6-I |
| **R6-M1** | One report contract (metrics) | N7, C8, C9, "Metrics for every future run" |
| KEEP | Rev 5 safety and selection | R6-H, R6-J |

### 2. P0 traps (code-verified by me): the code-first bundle would silently fail without these

| ID | Trap | Code evidence (checked) | Required fix |
|---|---|---|---|
| **T1** (→ R6-V3) | **A transform that calls a prelude helper is dropped by retention and rejected by the host verify.** The more the model follows the code-first design, the more exact candidates are lost (the 136b0064_b1 `patmap` class, made systematic) | Retention's available names = `runtime_globals` + builtins at restore time (`python_tool_sandbox.py:651`); the prelude is prepended to the snippet later. `prepare_functions` drops functions with "unavailable dependencies" (`retained_functions.py:125-136`). The adapter removes prelude names from `keepable_functions` (`arc2_duck.py:669-674`). `_grid_call_code` runs only the retained sources plus one grid (`:216-234`, `:768-774`) | Split helpers into **solution-usable** (pure grid functions) and **exploration-only** (read `train`/`test_inputs`: `arc_overview`, facts views), as V3.1 did. Make solution-usable helpers visible to retention, and inject the same trusted source into every verification call and into the test-prediction call. Keep the R5-1 serializer guard working with the helpers present. **Regressions:** a transform that calls each helper is retained, host-verified, harvested and selected; one that reads `train` is rejected with a clear note |
| **T2** (→ R6-V2) | **The facts block must be in the system prompt.** If it is in the first user message, the first trim drops it, and with R6-V1 the model then has neither grids nor facts | `_drop_oldest_history_block` pops the opener and everything up to the next user message (F2). The adapter's design rule: "the task lives in the system prompt, which is never trimmed" (`arc2_duck.py:492`) | Facts block in the system prompt (cache-stable per task); the opener stays short. **Regression:** after a forced trim the facts are still in the request |
| **T3** (→ R6-K1) | **Auto-check keyed on the name `transform` misses real exact candidates** | Exact functions under other names: `transform2` (b5ca7ac4_b0), 135a2760_b1, 247ef758_b0 (F1 row: the passive name-pattern harvest missed 2 of 12); 4c3d4a41 xhigh b1 `transform_alt` was never checked | Auto-check every top-level function defined or changed in the call that takes one grid and returns a valid grid on the train inputs (at least `transform*`, `solve*`, `*_alt`). Run it **after** the snippet executes. One summary line per function. **Regression:** the recorded `transform2` / `transform_alt` sources are auto-checked and harvested |
| **T4** (→ R6-V4) | **With grids hidden, the model will print them, and Duck's board guard does not recognise ARC-2 rows**, so hand-reading comes back through tool output. The 3,072-token cap with middle truncation hides rows silently. A1's line "can always print a full grid on demand" contradicts every donor (Tufa "Never print or echo full board frames", PIV AST guard, dfranzen guard) | `_board_like` needs ≥ 24 characters and `stripped.isalnum()` (`tool_agent.py:1592-1598`), so `0 1 2` or `[0, 1, 2]` rows never match | Change the A1 sentence to "a crop or a single small grid on demand". ARC-2 guard: collapse runs of ≥ 6 grid-like rows (digits with or without separators) and add a note ("character counting is unreliable for you; compute in Python"). `show(g, r0, r1, c0, c1)` prints an R×C header, row indices and a `|` every 4–5 columns, and says when output was truncated. Report `grid_rows_suppressed` and stdout tokens per call |
| **T5** (→ R6-K3) | **Auto-check collides with the rev 5 verify cap and dedupe.** Any new helper changes the dedupe key, so an unchanged transform is verified again. Every verification uses a cap slot. The cap note says "submit your best function", which ends the branch | Key = (name, sha of **all** retained sources); the key is added before verification; cap 24; note text (`arc2_duck.py:716-729`). Correction to one agent's claim: harvest runs only for functions `check()` reported exact, so non-exact checks do not use the cap | Dedupe on (name, source of the function and its transitive retained dependencies) **and** on the output signature (hash of train + test outputs). A duplicate signature costs no slot and gets a one-line note. Count only distinct verifications, or raise the cap and report `checks_skipped_cap`. Replace "submit your best function" with a neutral note. Run auto-check as its own sandbox call with its own time budget. **Regression:** 30 auto-checked redefinitions followed by a new correct function, and the last one is still verified |
| **T6** (→ R6-P2) | Duck's generic nudge still says "**You have not acted yet. Investigate first.**", which contradicts R6-P1 | `tool_agent.py:6983`; the adapter keeps the prefix | Replace it with the R6-P2 no-tool text; count nudges per branch |
| **T7** (→ R6-S2) | **A task with one verified candidate has no rank in stage 2.** About a third of accepts are test-wrong (rev 5 xhigh ≥ 3/9, low 6/17, rev 4 4/11); this is why the selective replay lost 135a2760 | Fix list C1 / E4 order | Stage-2 order: (1) no candidate, with partial-progress branches before nothing; (2) one verified candidate, any kind, flagged generalization concern first; (3) two distinct verified predictions. Do not hard-code the order on n = 5: log branch-1 yield per M tokens by class. Cross-method priors: same-budget rescue 3.2 %, fresh attempt after an overfit pass 22 %, near-miss feedback 27 % |
| **T8** (→ R6-S3/S6) | **[Corrected 2026-10-21: the overrun was 679 s, not 1,412 s, and the root cause is found. See "GPT review of the consolidated rev 6 plan, 2 of 2" → T8 root cause.]** Original text: **EVAL30 ran 1,412 s past its deadline, still unexplained.** Request timeouts are already clamped to the time left (`arc2_duck.py:838`), so the cause is something else. A hidden run that overruns scores 0 | exp12 docs | Explain the overrun from the EVAL30 logs before any hidden run. Add a last-admission cutoff (no new branch when the time left < p90 branch duration). The simulator asserts finish ≤ deadline − margin |

### 3. Design rules missing from the bundle (evidence in the gap reports)

| Rev 6 ID | Rule | Evidence |
|---|---|---|
| R6-S1 | **Stage-1 order = ORDER2 + cells tie-break** (already in the builder, `build_e12_duck.py` L508). Clock from kernel start. The simulator asserts admission order == ORDER2. Never the NVARC `fine` score | ORDER2 +4–7 vs random; cells +1; NVARC fine vs Qwen AUC 0.435 |
| R6-S4 | **Progress = the best single function in a single check, measured against the identity baseline.** Pairs correct, then changed-cell accuracy. Never a union across functions or revisions; never raw cell accuracy | V3.1 near misses were identity-level; Noemon / Handoff |
| R6-D2 | **Prefix invariants:** the system prompt and facts are deterministic (sorted keys, no set or hash order) and byte-identical across a task's branches. State text goes only as new messages at the tail. Same tools list and template kwargs on every request; no tool-free finalizer request. Any abort waits for the server stream to end. Offline check: facts identical under 3 PYTHONHASHSEEDs, and request n+1's prefix equals request n's prompt + reply except after a trim | Stable order raised the cache from 11–18 % to 87 % (E11-06). An abort without waiting gives 0 % vs 52–99.7 % with waiting. No tools: 15 % vs 99 % shared prefix. Template conditional: 75 → 45 %. `hash()` order changed an LB (hosen42). Uncached prefill already exceeds generated tokens (0.97M vs 0.85M xhigh) |
| R6-S5 | **KV cap for 10 streams:** the sum over live streams of (resident context + max_tokens) ≤ 0.9 × the KV pool, i.e. about 79k per stream at W1's 881k. If R6-Q1 lowers the cap, do not let the history budget grow to 118k (10 × 118k = 1.18M > pool); drain near 70–80k. Server readback before the run: max-running = 10 (+1 workaround), CUDA-graph batch list covers 1..10 including 9 and 10, mamba slots ≥ the donor ratio (60 per 10 streams), keep fp16 SSM, real running count from the decode log | Decode at 40–60k context is about 25 % slower (323 vs 429 tok/s); DFlash acceptance 4.43 → 3.77 over context; EVAL30 re-prefilled 8.4M; 6 → 12 gave +17 % in one measurement (+30–60 % from the short sweep) |
| R6-X1 | **Parity diagnostic calibrated:** run a same-config control first. Judge on task-level outputs (tool call parsed, same check verdict) and batch-size-1 prefixes, not token equality under batching. Reuse the E11-04 marker-leak isolation test at 10 streams. Target-only is a separate bounded launch, never alongside production | Spec vs plain shared only 152–166 of 512 tokens under batching (E11-03); Deep Audit: an unchanged repeat scored 0.1875 vs a 0.50 threshold |
| R6-P2 | **Resume states, complete:** cut call → length cut in reasoning with no call (new, the most common under R6-Q1: "your reasoning is kept; call python with check(transform) now") → trimmed → verified candidate → partial → no-tool reply → timeout → `yield_tools` (neutral continuation, no scolding). Built from Duck's `_resume_reason` plus adapter flags. Borrowed donor texts must say `check()` where the donor said submit: **our `submit()` ends the branch** | `tool_agent.py` `_resume_reason` ∈ {yield_bare, yield_tools, timeout, tool_steps}; madarshbb/Poetiq "submit is free" |
| R6-P1/P2/P3 | **Facts, not advice; terse orders only at cut or yield boundaries.** Every "test / compare / consider" instruction names the Python action. No periodic nudges. Measure the next-request tool-call rate per message type and drop types with no effect | Fable: advice blocks 6.42 → 4.16, own history +8.0. Exp 4: cut + terse order → 24/24 next-turn tool calls (17 also stepped to medium); periodic nudges → 0 voluntary submits; madarshbb "nudge read and ignored". The Poetiq "test your code" prompt sent 10/12 to the cap. Prompt-only rules failed twice (Exp 4 run 4, E11-06) |
| R6-D3 | **Trim order keeps tool results:** drop old reasoning-only blocks before tool-result blocks (or keep the last K tool results) | Readers: the low 4c3d4a41 trim lost tool results, then an overfit accept |
| R6-D1 | **If R6-Q1 is declined, R6-6 is not solved by anything.** Options: (a) Duck window → about 150k (server context 155,648) with a KV check; the rev 4 window change was reverted, so record why first; (b) reserve = the real cap; (c) accept trims, with T2 + R6-P2 covering facts and truthfulness | rev 5 low 4c3d4a41_b0 |
| R6-S3 | **Overshoot modelled:** with 49,152 per request, a request-boundary budget B can reach B + 49k. The 240-task simulator models it. A hard clamp stays rejected unless R6-Q1 is approved | rev 5 analysis: soft caps do not bound tokens; overshoot to 159k at the 100k cap |
| R6-V2 | Facts content: also the top-4 (in → out) colour transitions for same-shape pairs, and the change in component counts. Start from the archived, tested V3.1 `arc_overview()`. No mandated first call that repeats the facts | V3.1 (`puzzle.keys()` was 65/173 calls as a wasted first call) |
| R6-V3 | Helpers also: an object-level pair diff (donor `compute_frame_diff` via a wrapper), rot90/flip/transpose, `inspect_pair(i, detail)` / `inspect_test(i, detail)` (report C), `bbox`, `crop` with **one documented argument order** (the drafts differ: A/C `(r0, r1, c0, c1)`, B `(r0, c0, r1, c1)`). All listed in the tool description, which is identical across tasks | reports A/B/C, brief A3 |
| R6-K2 | check output: all verdict lines first (pairs, identity flag, shape mismatch, test inputs ok/error), then details capped per variant. Infrastructure errors labelled as such. The failing line inside transform in errors (old R6-4, was dropped) | PIV feedback, V3.1 test-input run |
| R6-M1 | **One report contract (CELL 7):** tokens to the first python call and the first `check()` (one definition: cumulative branch tokens at the request that *executes* it; censored branches at their total; the 55.1k vs 56.6k ambiguity resolved this way); hand-reading share (digit-token and grid-paragraph via `sim_tokens.py`, rev 5 baselines computed offline); facts tokens p50/p90/p99; auto-check and check_many counts; `verify_sec` / verify timeouts (pending since rev 2/3); nudges and resumes by type; uncached prefill / generated; real running count; DFlash acceptance; degeneracy count; redefinition share; estimate vs `prompt_tokens` re-validated (R6-V1 changes the text mix); exposed vs meaningfully attempted tasks; label-free overfit flags per candidate (train-output literal, fixed offset tied to train dims, test prediction == test input, shape/palette outside the train pattern, source length: train-perfect-wrong median 11.1k vs 6.5k characters, cross-branch agreement); branch-1 yield by stage-2 class | gap reports 1–3 |
| Gates | **EVAL6/EVAL12 accuracy is a sanity check only** (identical code gave 2, 2, 3, 1 of 22). Decision metrics are tokens to first check, hand-reading share, correct outputs per GPU-hour and oracle coverage. Add **gate 4b:** one eval profile at the hidden-equivalent compute share (about 43k tokens per task at 255 tok/s); its score vs the unconstrained score predicts LB/eval. If it lowers thinking below 60k it is a diagnostic needing the user's approval | LB/eval 1.06 / 0.62 / 0.44 |
| Regressions | (a) tasks with 2–3 test inputs, so facts and globals cover every test input; (b) **an offline corpus of earlier successful trajectories** (e.g. rev 4 4c3d4a41 2/2, which rev 5 lost) to check that working tool-call and verification paths still work; (c) the 2-exact + 1-timeout rejection test (pending since rev 2/3) | GPT audit 2 §5.2; exp12 docs |

### 4. Evidence to add to the user's decisions

**User correction (2026-10-21):** "Those were different experiments. That something did not work there does not mean it will not work here."

So the "against" items below are **context notes, not verdicts**. Each comes from a different setup:
- V3.1 / EVO12: low or medium effort, a 12k total budget per task, a different harness.
- PIV forced close: a stop sentence at 3,072 tokens, not a cap with the reasoning kept.
- The pause/resume negative: DIRECT single long think, state not kept.

None of them tested rev 6's combination: xhigh, ≥ 60k per branch, thinking kept in history, facts view, auto-check, state-aware resume.

They are used **only to choose safeguards and what to measure** (e.g. "the cut call is never executed", "tool-call quality after a cut"), **never as a reason not to try**. The decision is made by the rev 6 GPU gates.

**R6-Q1 (request caps).**

*For a structural cut (measured):*
- Exp 4: cut + terse order → 24/24 next-turn tool calls (17 also stepped to medium), about 1k tokens per tool turn.
- madarshbb Qwen3.8-27B probes: solves 2–3k tokens per turn, failures cut at 16k; but its published config runs direct mode (third party, unverified).
- E10: code by 30k → 77 % vs 30 % solved.
- dfranzen 12,288: truncation about 4 % of replies (ARC-3, no grid text).
- Rev 5 xhigh's first reply is p50 33.6k before any tool call, so the prompt changes act only after it.

*Against (measured):*
- V3.1: 12/12 tool calls but solves 1/12 (low, 12k total).
- EVO12: 0/12.
- PIV forced close at 3,072: 6/9 dirty calls and +37.5 % tokens, so prefer cap + continuation over a forced stop sentence.

*Approval history:* the settings doc records "12,288 = approved exception to the 60k floor, user 2026-10-19" for the milestone port (rev 1 / EVAL30). That approval was for that run, so rev 6 needs a new decision.

**R6-Q2 (pause/defer after 25–40k, report C).**
- The earlier pause/resume was negative at every budget: straight 31.7 vs 25–27 at 70 %. That was DIRECT, with state not kept, so it may not transfer to a Duck pause that keeps the same agent object.
- budget_sim gives 0–1/6 at 25–35k with the rev 5 agent.
- 240 × 25–40k = 6.0–9.6M tokens = 46–92 % of the run.
- Only consider it if the pilot meets "median first check ≤ 25k".

**R6-E (effort).** Record xhigh as the rev 6 default. Low on the conditional branch 1 is a separate, labelled option, not in the first build. Low solved 898e7135 (xhigh did not); union on the 6 shared tasks 5/6.

**Report B's representation hedge (open, for the user).** Branch 1 (rescue) could use the rev 5 text grids or CSV instead of the facts view. Johan: a poor format costs about 10 %, and compact sparse formats were much worse; exclusive solves by view: text 2, image 6, code 7. CoreThink: give the hint to one branch only. This hedges R6-V1's biggest risk and measures solves by view inside one run.

### 5. Explicitly not in rev 6 (so it is not re-proposed)

- **Reasoning and history:**
  - rolling summaries / compaction (Franzen: removing reasoning was a large regression);
  - `ARC3_PRUNE_*`, `ARC3_STUB_DEAD_REASONING`, opener/step hints, degeneracy-ratio change (loops now 0–8 %);
  - stripping reasoning from history (madarshbb `visible_reply`).
- **Forcing calls:**
  - a forced or required `tool_choice` (Exp 4 HTTP 500 + 59 empty submits; SGLang #43514);
  - a PIV no-tool JSON commit (prefix drops to about 15 %; strict JSON underperformed).
- **Early stopping and thinking budgets:**
  - REFRAIN and other text early exits;
  - a hopeless-thinking stop;
  - LOOPSTOP at request boundaries is deferred (CPU-validated: 5/5 loops, 0 false positives over 13.1M tokens) because loops are now rare;
  - relying on a server thinking budget or penalties under DFlash (Exp 5; #28180).
- **Different pipelines:**
  - NVARC as the attempt_2 filler (+6.1 eval, but about 9.5 h per 240 tasks);
  - a one-shot code stage;
  - hiding the test input (Noemon).
- **Inputs:**
  - images;
  - the CPU rule library (hosen42: about 30 ms per task, 5.3–5.7 % on real held-out, false-fire bugs; if revived, train-exact only, host-verified, for slots that would otherwise be the test input);
  - `ARC3_PREFER_TOOL_CALLS` as if proven (it was off in dfranzen's winning config; outcome evidence none; its cost argument still holds).
- **Pending decisions, not built:**
  - `ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS`: test with T1 first (the adapter passes it to the tool but `repair_hints=False` to verify);
  - the yield value (2,048) is a recorded decision;
  - Franzen patch #4 + `--watchdog-timeout 1800` only if the prefix-hit rate at 10 streams falls below the rev 5 level;
  - a JIT cache dataset (−4 to −5 min) is outside the agent bundle.
- **Corrections:**
  - the old PIV wording "the model must finish a full solve before a tool call" is wrong (report C); keep it out of rev 6.
  - The "stop at the first train pass" advice in the Poetiq-era notes is overruled by our measured late corrections (135a2760_b1, 898e7135_b0/b1).

**Status:** nothing built. The user decides R6-Q1, R6-Q2, R6-E and the representation hedge. The facts-block size (600–800 vs 1,200–1,500 tokens) will be measured on the 120 eval tasks with the Qwen tokenizer before the build.

## GPT review of the consolidated rev 6 plan, 1 of 2 (2026-10-21, user-pasted): validated, new ideas only

**Verdict:** GO for the rev 6 implementation design. The GPU run must wait for the source audit and CPU regressions. **This is design approval, not notebook approval.**

**Strongest parts of the plan, in GPT's view:**
- keeping Qwen3.8-27B + Duck;
- full coverage as a hard goal;
- Python facts and verification;
- leaving rev 5 harvest and pass@2 intact;
- the T1 / T2 / T3 / S1 finds.

**Asks for no new big features.**

### New items and corrections (validated)

| # | GPT point | Verdict | Evidence / effect on the plan |
|---|---|---|---|
| P1 | **T5 correction:** a function's output signature is only known after it runs, so "a duplicate signature costs no slot" cannot skip the execution. Order of the fix: (1) dedupe on name + source + real transitive dependencies; (2) if new, a bounded sandbox run; (3) after the clean verify, build the output signature; (4) an already-accepted identical prediction does not enter the archive again. **Avoiding duplicate predictions ≠ avoiding duplicate execution cost** | **Accepted; T5 is corrected.** The train outputs of every exact function equal the truth, so only the test predictions differ, and they need an execution | R6-K3 rewritten as GPT's steps 1–4. The signature saves archive slots and selection noise, not sandbox time |
| P2 | **Exhausting the auto-check cap must never disable explicit `check()` and `submit()`** | **Partly true today** | Code: `submit()` goes through its own pending path and is not capped. Explicit `check()` harvest shares the cap of 24 (`_arc2_harvest_checks`, `arc2_duck.py:724-729`). **Fix:** separate budgets, auto-check cap vs explicit-check cap. **Mandatory test:** after 30 auto-checked redefinitions, a new correct function is still accepted through explicit `check()` and through `submit()` |
| P3 | **Auto-check must not run every arbitrary one-argument helper.** Only new or changed plausible transformation functions, bounded. Auto-check success is never trusted acceptance; the rev 5 clean verifier stays the authority | Accepted | R6-K1 scope: functions named `transform*` / `solve*` / `*_alt` / `*_v<n>`, or the ones `check_many` is given. T3's "any function returning a grid" is narrowed to this list. Every acceptance still goes through `_verify_submission` |
| P4 | **Deadline safety (T8) is P0, and a last-admission cutoff alone is not enough.** `solve_all()` waits on every future; request timeouts are clamped to the time left; but `_verify_submission`'s budget is not bounded by the global deadline | **Confirmed in code** | `_verify_submission`: `budget_end = now + 2 × python_timeout` (60 s), not tied to the run deadline (`arc2_duck.py:761`). `solve_all`: `for f in futures: f.result()` inside `with ThreadPoolExecutor(...)`, which also waits for every worker on exit (`:1020-1032`). **Fix (R6-S6):** one shared absolute deadline minus a fixed submission-write margin. Every request, tool call, verification and retry/backoff respects the time left. No new verifier or rescue after the deadline. Bounded worker cleanup: `shutdown(wait=False)` alone is not enough; workers must observe the deadline. **Mandatory tests:** a hung model request; a verifier timeout; a worker exception; 10 active branches right before the deadline. In each, a valid submission is written on time and no harvested candidate is lost. GPT notes this is a set of possible paths; the EVAL30 1,412 s overrun is still not explained from its logs |
| P5 | **R6-D3: tool calls and their results are paired**, so dropping one of a pair can make the conversation invalid or give false continuity. Keep or remove coherent blocks (call + result). If space runs out, a compact factual summary of the last tool results may be kept, consistent with the retained sources and verified candidates | Accepted | R6-D3 rewritten: block-level, never message-level deletion. Duck's own `_drop_oldest_history_block` already drops whole blocks up to the next user message, so the change is in *which* blocks go first. **Test:** after repeated trims the tool-call IDs and result pairing are valid, the server accepts the request, and the latest retained function is still usable |
| P6 | **Host-side state kept ≠ GPU KV still resident.** Paused stage-1 branches keep their Python objects and history, but their prefixes may be evicted, so a stage-2 resume can pay a full re-prefill. Admission should be driven by runtime KV occupancy and the actual decode tok/s, with fallback 10 → 8 → 6 streams as a controlled capacity step, not a failure | Accepted | R6-S5 + R6-D2 telemetry: re-prefill tokens per resume, per stage. Cold prefill measured about 1,400–1,650 tok/s for the engine (E11), i.e. about 30–35 s per 50k resume |
| P7 | **Capacity arithmetic for report C's pause:** a 12,288 cap with a 25k soft first exposure → up to about 37k per branch (one-request overshoot). 240 × 37k ≈ 8.9M tokens ≈ 9.3–10.1 h at 244–265 tok/s, leaving little room for stage 2. So 25k is **not a hard interrupt**; judge the pause policy on the pilot's first-check timing | **Arithmetic confirmed** | Recorded under R6-Q2 |
| P8 | **Gate classes.** Mandatory (blockers): helper retention and clean verify; auto-check dedupe / cap exhaustion / late correct candidate; facts exactness, no truth leak, multiple test inputs; rev 5 security regressions; dynamic reserve/context consistency; 240-task admission, pause/resume, deadline-safe submission. Performance targets (not blockers): first python ≤ 12k, first check ≤ 25k, 10-stream gain (fallback allowed). Outcome: accuracy vs rev 5 (no strict equivalence on small samples) | Accepted | Replaces the mixed gate lists. Matches the gap-check note "EVAL6/12 accuracy is a sanity check" |
| P9 | **The regression corpus of old correct programs** (rev 4 4c3d4a41, rev 5 135a2760 / 898e7135 late corrections) must keep working with the new helpers and scheduler (source compatibility) | Already in the gap check (Regressions b); GPT confirms | — |
| P10 | **Representation hedge for branch 1:** accept it, but it is not a reason to always run branch 1; branch 1 runs only when the scheduler budget says it is worth it | Accepted | Recorded as GPT's support for the open hedge item |

### GPT's numeric preferences (recommendations; the user decides)

| Setting | GPT's preference | Notes |
|---|---|---|
| R6-Q1 request cap | 12,288 first; up to 24,576 after verified tool progress; a larger cap in rescue when needed | 12,288 is a per-request maximum, not a branch ceiling; a branch can still use 60k+ over several requests |
| R6-Q2 first exposure | A safe request-boundary pause after 25–30k generated tokens; resume the same history, retained functions and candidates | Not a permanent stop; judged on the pilot (P7) |
| Facts block | Initial target 600–800 Qwen tokens; if larger, move details on demand rather than truncate | Agrees with report A |

**Status:** design accepted by this GPT. Nothing built. The user's decisions R6-Q1 / R6-Q2 / R6-E / hedge are still open. A second GPT review of the plan is coming.

## GPT review of the consolidated rev 6 plan, 2 of 2 (2026-10-21, user-pasted): validated; T8 root cause found

**Verdict:** REV6 DEVELOPMENT GO. This is approval to implement, not to run on GPU. GPT will audit the diff, safety, the simulator and the source before a GPU GO / NO-GO.

**Review 2's verdicts, by group:**
- **TAKE:** R6-V, P, K, S1–S4, D, X, M.
- **Conditional GO:** S5 (10 streams).
- **Mandatory:** S6 (deadline safety).
- **TAKE WITH SAFEGUARDS:** Q1 / Q2.

### New items (validated)

| # | Point | Verdict | Evidence / change |
|---|---|---|---|
| Q-1 | **KV arithmetic conflict in R6-S5.** 0.9 × 881k = 793k → 79.3k per stream at 10 streams. **Minus a 24.6k output reserve, that leaves about 54.7k of history**, so "drain near 70–80k" in the gap check contradicts the per-stream budget | **Accepted; R6-S5 corrected** | Drain target = per-stream allowance − real reserve: about 55k at a 24,576 cap, about 67k at 12,288. Franzen drains at 58k, which agrees. Admission uses projected KV: Σ (resident + reserved generation) + new prompt + new reserve ≤ 0.9 × usable KV, counted conservatively. Server occupancy and retraction behaviour are taken into account. Validate at 6 / 8 / 10 streams with long-context outliers, mamba headroom, fallback and no deadlock |
| Q-2 | **The R6-K3 trust boundary.** An output signature from the untrusted sandbox must never skip the host verify (the serializer-spoof class). Sequence: (1) source key = hash of the function + its transitive dependency sources; (2) cache hit → reuse that *trusted* verdict; (3) else clean host verify, then cache it; (4) only an accepted verdict yields an output key, used only for dedupe after verification. Also: invalidate on any dependency change, record rejected-verify timeouts and errors separately, and never lose a valid candidate silently when the cap is reached | **Accepted** (same as review 1 P1, now with the cache) | R6-K3 final form. The same trust rule applies to R6-K1 |
| Q-3 | **Auto-check eligibility:** not every one-grid function (`rotate`, `components` are helpers). Name rule plus an eligibility check, bounded, with clean verification. **T1 still applies:** a trusted helper that a solution uses must exist in the host verify. **Regressions:** `transform2`, `transform_alt`, a helper-dependent valid transform, a transform using an unavailable global, and 30 redefinitions followed by a correct one | Accepted (= review 1 P3) | — |
| Q-4 | **Representation hedge:** support it. One deterministic alternative configuration for branch 1 (rev 5-style grids or careful CSV), **not** several generated views, so that prompts and prefixes stay stable | Accepted | Hedge spec: exactly one alternative view |
| Q-5 | **Facts size:** measure p50 / p90 / p99 on the 120 eval tasks first. Use a compact default with an explicit truncation notice. Allow 1,200–1,500 tokens only if 600–800 misses important facts | Accepted | Matches the gap check |
| Q-6 | **Capacity check for Q2:** 25k × 240 = 6.0M tokens ≈ 6.5 h at about 255 tok/s; 40k × 240 = 9.6M ≈ 10.5 h. So start at 25–30k, but only if the pilot shows that python and check really come early | **Arithmetic confirmed** | — |

### GPT's preferred configuration (review 2; the user decides)

| Setting | Value |
|---|---|
| Effort | xhigh |
| Request cap, first / no-tool | 12,288 |
| Request cap, after meaningful tool progress | 24,576 |
| Request cap, repeated no-tool rescue | up to 32,768 |
| First-exposure cumulative budget | about 25–30k, soft |
| Overall branch soft cap | the rev 5 100k, kept at the start |
| Thinking | preserved; `tool_choice` auto |
| Pause | only at a safe request or yield boundary |
| Resume | the same live agent object and retained state |

**Both plan reviews agree** on: xhigh; 12,288 then 24,576; a soft 25–30k first exposure; the same agent on resume; 600–800 facts tokens first.

### T8 root cause (found while validating this review; read from the EVAL30 logs and code)

**1. The overrun was 679 s, not 1,412 s. The 1,412 figure was my error.**
- In the report, `deadline_in_sec` = 13,367 is relative to the solve start (`build_e12_duck.py:594`: `_end − now`), and `started_sec` = 733.
- So the absolute solve deadline was about **14,100 s** after notebook start (= HARD_STOP − 180).
- `finished_sec` = 14,779, so solve ended **679 s** past its deadline, and 379 s past the 4 h total budget of 14,400.
- I had subtracted the relative 13,367 from the absolute 14,779.

**2. The mechanism (code-verified):**
- `run_branch` computes `request_timeout_seconds = min(analyzer_timeout, remaining)` **once per `analyze()` call, i.e. per turn** (`arc2_duck.py:835-838`).
- Duck reuses that same value for **every request inside the turn** (`tool_agent.py:6763-6764` → `:5275`, used as the HTTP read timeout).
- `should_stop()` is checked before each request in the turn (`tool_agent.py:6719-6739`), so no new request starts after the deadline. But a request that starts just *before* the deadline, late in a long turn, carries a timeout computed at the turn's start.
- **Log evidence:**
  - At 14,113 s (deadline) about 16 requests time out together. Those were correctly clamped.
  - At 14,436 s a request times out with "read timeout = 419.6 s", i.e. it started at about 14,017 s, before the deadline, with a turn-start timeout that ran it 336 s past the deadline. Likewise 469.7 s at 14,492.
  - At 14,247 s: `num_running_reqs` 11, `num_queue_reqs` 8, `gen_throughput` 0.0, so the engine was saturated with these late requests.
  - The last branch finished at about 14,7xx; the run report was written at 14,792.

**3. Risk for the hidden run:**
- The maximum overrun is about one analyzer timeout.
- In rev 5 that timeout is **1,800 s**. A long turn starting with plenty of time left can therefore run up to about 30 min past the solve deadline.
- That eats the 480 s drain and the hard margin, so this is a real 12 h-limit risk.

**4. Fix (adapter-only, small): R6-S6-a.**
- Re-clamp the read timeout **per request** to the absolute deadline: an override of the request call (or `_chat_completion`) in `Arc2ToolAgent` that uses `min(given, deadline − now)`. The branch deadline is stored on the agent.
- The same rule applies to `_verify_submission`: budget = min(2 × python timeout, deadline − now − margin) (review 1 P4).
- No new verifier or tool call after the deadline. A worker that times out still writes its record (the existing `finally`).
- **Regression:** a fake server that holds a request; a turn starting at T − 400 s with a 1,800 s timeout; its second request at T − 10 s must end by T plus a small margin. Also the four deadline tests from review 1 P4.

**5. Not yet explained:** why `gen_throughput` read 0 with 11 running at 14,247. It is likely KV thrash (EVAL30 had all 60 branches admitted and re-prefilled 8.4M tokens). It is not needed for the fix.

**Status:** nothing built. Open user decisions: R6-Q1, R6-Q2, R6-E, the hedge. Both plan reviews recommend the same values.

## Rev 6 implementation (built 2026-10-21, after the user accepted the plan)

**Files**
- `duck_arc2/arc2_duck.py`: 2,162 lines, LF, pyflakes clean, sha `53f09866…`. Backup `arc2_duck.py.pre_rev6_backup` (sha `172371f1…`). All rev 6 code sits behind `E12_ARC2_REV6=1`. `_rev5()` is true under rev 6, so every rev 5 fix stays on.
- `Experiment Notebooks/Build/build_e12_duck.py`: `--eval12r6` / `--eval30r6`. REV6-only `once()` patches cover CELL 6 (preflight, KV window, `solve_all_r6`) and CELL 7 (`report["rev6"]` metrics). Rev 5 notebooks are not touched by these patches. Backup `.pre_rev6_backup`.
- **Notebook `Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL12R6.ipynb`, sha `cef11ad3…`.**
  - Profile: the 12 EVAL12R5 tasks, 150 min solve, 10 streams.
  - Caps: 12,288 first, 24,576 after a successful Python call, 32,768 rescue.
  - First exposure 28,000; b1 view = grids; admit gate 240 s; check / auto caps 24/24; 3 auto-checks per call; KV fraction 0.9.
  - Not pushed, not run.
- Kaggle-run notebooks are unchanged (sha checked after every build): 26fff41e EVAL12R5_LOW, f4fed1ac EVAL6R5, 415a68f2 EVAL30R5, 53a66856 EVAL6, b65fb9dc EVAL30, 0fea4d0c SMOKE1, 216c59ed S12, cce9cb8f EVAL6R5_LOW.

**Built (ID → what)**
- R6-V1–V4:
  - deterministic facts block in the system prompt (bounded to 3,600 characters, says so when lists are dropped);
  - b1 grids view;
  - 12 trusted helpers restored as retained sources (T1) and hidden from the model's kept list;
  - ARC-2 print guard, with an explicit note.
- R6-P1–P3:
  - merged short prompt;
  - state-aware openers (cut > length > timeout > verified/partial > notool > continue, plus **rejected**, see below);
  - generalization note sent once.
- R6-K1–K3:
  - auto-check of `transform*` / `solve*` / `*_alt` / `*_vN` functions that take one grid argument;
  - verdict-first `check`, plus `check_many`;
  - closure-hash verify cache, output-signature dedupe, separate caps with neutral notes.
- R6-S1–S6:
  - `BranchSession` with pause/resume on the same agent;
  - stage 1 = every task's b0 first; stage 2 classes;
  - capacity probe (queue → workers −2, floor 6);
  - deadline clamp per request (T8 root cause) and per verify.
- R6-Q1/Q2: per-request caps, re-derived reserve/budget (R6-D1); first exposure 28k.
- R6-D2/D3: only the newest opener's sentinel is rewritten after a trim.
- R6-X1: drafter tap-layer config read + n-request isolation smoke. Workers drop to 6 on a leak.
- R6-M1: `report["rev6"]["metrics"]`:
  - exposed / meaningful;
  - first tool / first check (censored);
  - autocheck;
  - suppressed rows;
  - no-call cuts;
  - clamps;
  - caps per request;
  - resumes;
  - verify p50/p90.

**Three defects found while testing the notebook (all fixed, each now has a test)**
1. **Regex hang (critical).** `_GRID_ROW_RE` had overlapping separators (`\s*,?\s+\d+ | \s*,\s*\d+`), so a long `print(grid)` nested-list line backtracked exponentially. The smoke hung for more than 11 minutes in `guard_grid_prints`, and on Kaggle this would freeze an agent thread. Fix:
   - disjoint separators `(?:\s*,\s*|\s+)\d+`;
   - rows longer than 600 characters skip the regex (the nested-list cut handles them).
   - Check: `regex_check.py` gives the same verdicts as the old pattern on 14 row forms and runs in under 6 ms on 10k-character lines. The smoke has a linear-time test.
2. **Wrong opener after a rejected submit.** A rejected `submit` is an executed step, so Duck opens a new turn without a resume flag, and rev 6 sent the first-turn text ("Solve the task…"). Rev 5 named the rejection. Fix: a new state `rejected` = "Your submission `name` was not accepted: {summary} Continue from your work above: fix the rule or the code, check() it, then submit again." Covered by smoke P2 and by a harness rerun check (first == branches, rejected ≥ 1).
3. **Caps metric counted trims, not requests.** The trimmer runs several times per request. The count moved to `_chat_completion` (one per request, using the real `max_tokens`). The harness checks that it equals the server's max_tokens histogram.
- Also: my Python edits had turned `arc2_duck.py` into CRLF. It is restored to LF; the content is identical.

**Tests (all on the final files)**
- `smoke_rev6.py` **68/68**:
  - facts, prompt, T1 helpers, K1–K3, S4, V4 (+ linear time), P2 (+ rejected), T6, Q1/D1, T8/S6 clamps;
  - scheduler sim: 240 fake tasks, 12,000 requests, ORDER2 tolerance, stage order, pause/resume, deadline, capacity fallback.
- `smoke_rev5.py` 56/56, `smoke_adapter.py` 41/41 (flag off = old behaviour).
- Notebook harness on `EVAL12R6` (real cells, fake SGLang): **r6 37/37, rerun 22/22, launchfail 6/6**.
  - r6 checks:
    - profile and flags; window 79,320 = 0.9 × 881,336 / 10; preflight 10/10 own markers; 10 workers;
    - caps used only {12,288, 24,576}; both views; cut / length / verified resumes; cut code never ran;
    - in-flight ≤ 10; every b0 before any b1;
    - every one of the 24 branches has an auto-check candidate without any submit;
    - b0 paused then resumed; 19/19 eval hits; metrics and scheduler summary present.
  - rerun needed `E12_R6_MIN_ADMIT_SEC=5`, because the fake window is 25 s and the 240 s admit gate correctly admitted nothing.
- Replay of the recorded rev 5 tool calls through the rev 6 agent:
  - xhigh 5/8, the same as rev 5;
  - low **10/19**, the same as rev 5;
  - kept_r6 ≥ kept_r5 on every branch (9385bd28_b0 +1 via auto-check).
  - The replay does not depend on printed output, so the regex fix cannot change it.
- Harness Python: `_scratch/e13_venv` (transformers 4.57.6). The C: user-site transformers is gone.

**Capacity (pre-run check, EVAL12R6)**
- Rev 5 low ran at 1.744M generated tokens in 6,570 s at 6 streams, about 265 tok/s.
- Rev 6 needs about 24 × 70k ≈ 1.7M tokens. The worst case is 24 × 100k = 2.4M; the solve window is 9,000 s.
- At 265 tok/s the worst case needs about 9,000 s. At 10 streams it should be faster, but that is not measured.
- Resident history: 10 × 79k ≤ 0.9 × KV.

**Not done / open**
- GPU run: needs the user's go.
- EVAL30R6: builder variant present, not built.
- GPT audit bundle for rev 6: see the zip next to this file.

## Rev 6b: every row of the problem → fix map checked against the code (2026-10-21, user: "check every fix was taken, update, validate, then zip")

**Method:**
- I read `solve_all_r6`, `BranchSession`, the whole agent class, the facts, prompt and helpers, Duck's trimmer, and the builder's CELL 6/7 patches.
- Each row of the map ("Rev 6-এর প্রতিটা fix কোন মাপা সমস্যার সমাধান করবে", i.e. "which measured problem each rev 6 fix solves") was marked built, missing or partial.
- Everything missing was built and tested. Notebook `Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL12R6.ipynb` is now sha **`fe51d1d4…`**; this supersedes `cef11ad3`.

### Map rows → status

| Map row | Fix | Before 6b | Now |
|---|---|---|---|
| ক hand reading, first check ~55k | R6-V1 facts view, R6-V2 facts in the system prompt | built | **R6-V2 size fixed (Q-5).** Measured on 120 eval tasks with the Qwen tokenizer, the block was p50 1,237 / p90 1,752 tokens, but both plan reviews agreed on 600–800 first. Now it is token-bounded with 4 levels and an explicit omission note: **p50 736 / p90 837 / max 1,396**, smaller than the raw grids on 120/120, and identical under 3 PYTHONHASHSEEDs |
| ক helpers | R6-V3 | built (12 helpers) | **+ `object_diff(a, b)`** (kept / recolored / moved with offset / removed / added; the donor `compute_frame_diff` idea). It is a trusted helper and `inspect_pair(i, 'objects')` uses it |
| ক grid printing | R6-V4 show / guard | built; regex hang fixed | — |
| ক "I'll write code" then more thinking | R6-Q1 caps, R6-P2 terse order | built | — |
| ক rule found, check late | R6-P1, R6-K1 auto-check | built | — |
| ক variants by hand, error loops | R6-K2 check_many, verdict first, error line | built, but **infrastructure errors were not labelled** | **A sandbox time or memory stop is now labelled "harness limit … says nothing about whether the rule is right"**, and counted (`infra_errors`) |
| খ transform_alt / transform2 never checked | R6-K1 | built | — |
| খ verification cap ran out | R6-K3 | built | — |
| খ late corrections | KEEP | built | — |
| খ trim loses tool results | R6-D1 reserve = cap; **R6-D3 pairs + trim order**; R6-S5 55–67k | D1 and S5 built; **D3 missing** (Duck always drops the oldest block) | **R6-D3 built:** the oldest complete block with no tool result (a reasoning-only reply) is dropped first. Blocks are removed whole and the newest block is never a candidate. Test: call/result pairing stays valid after repeated trims |
| গ "Nothing has been submitted yet" | R6-P2 | built (+ `rejected`) | — |
| গ "Investigate first" | R6-P2 / T6 | built | — |
| গ several nudges at once | R6-P1/P2/P3 | built | — |
| ঘ 4c3d4a41 +10 offset | R6-P3 once | built | — |
| ঘ both branches make the same mistake | R6-S2 + b1 grids view | built; **the class-2 "generalization concern first" order was missing** | **T7 built:** inside class 2, tasks whose verified candidate carries an overfit sign get branch 1 first |
| ঘ no way to tell overfit | R6-M1 label-free signs | **missing** | **`overfit_flags()` on every host-verified candidate**: train-output row literal, fixed offset, offset tied to the train dimensions, test output equal to its input, test shape or palette outside the train pattern, source > 9,000 characters. Plus cross-branch agreement in the report. Never a rejection |
| ঙ both branches of a task run together | R6-S1 | built | — |
| ঙ duplicated work | R6-S2 | built | — |
| ঙ no budget accounting (100k soft cap → 159k) | **R6-S3 fair share**, overshoot accounted; R6-S4; R6-Q2 | S4 and Q2 built; **S3 missing** (stage-2 runs went to 100k while others waited) | **R6-S3 built:** a stage-2 run pauses softly after max(24,576, share) tokens, where share = measured aggregate tok/s × usable time left / items still to serve. It is re-queued **round robin** (fewer stage-2 rounds first, then class, flag, ORDER2). A pause is not a cut: the same agent resumes while time remains, up to the 100k per-branch cap. Overshoot is recorded per run |
| ঙ scheduler trap (480 live) | R6-S1 | built | — |
| চ 6 of 12 streams | R6-S5 10, then 8, then 6 | built (probe −2, floor 6) | report: **real running max** from the decode log, and acceptance length |
| চ prefill > generated | R6-D2 | deterministic facts built; **no invariant test** | **Tests:** facts and prelude identical under 3 hash seeds. In the fake server, request n+1 must equal request n + its reply in every conversation: **84 ok, 0 breaks**, 12 expected changes right after a cut reply (only the broken call is dropped, rev 5 B6), and the system prompt never changes |
| চ DFlash correctness, isolation | R6-X1 | isolation + drafter config built; **parity control missing** | **Parity control built:** the same request twice at T 0, batch size 1, recorded in the preflight. A target-only comparison needs its own server launch and stays out of production (GPT: "never alongside production") |
| ছ deadline | R6-S6 | built | **+ the P4 mandatory tests:** a request hanging at the server ends at the deadline with its record written; a worker exception ends only its branch; paused branches keep their candidates in the final snapshot |
| জ T1 / T2 / spoof / same agent | — | built | — |
| ঝ measurement | R6-M1 | partial | **Added:** facts tokens p50/p90/max; hand-reading (grid-row) / digit / repeat shares of reasoning; redefinition share; infrastructure errors; reasoning-only trims; accepted candidates + overfit signs; cross-branch agreement; **branch-1 yield by stage-2 class**; overshoot p50/max; **re-prefill of each stage-2 resume** (P6); estimate/actual prompt-token ratio; uncached prompt tokens vs generated; real running max; acceptance; last fair share. Per-branch report rows carry `stage2_class` and `overfit_flags` |

### Two defects found while building 6b (both mine, both fixed before any test passed)

1. **Two workers could drive the same agent.** A paused branch was re-queued while it was still running. This is a real data race: one Duck agent, two threads. A negative control without the fix gives **24 concurrent entries**; with the fix, 0. The smoke test now asserts it, and so does the fake-server prefix check (a shared agent breaks the prefix).
2. **Starvation in the first fair-share draft:** paused items came back with their ORDER2 rank, so the first tasks took every round. Fixed with round robin. Test: stage-2 tokens per task have max ≤ 4 × median, and 30/30 tasks get stage-2 work (24k–72k each) vs 3–4 without fair share, where one task took 792k.

### Not built (reasons; the user decides)

- **Gate 4b** (an eval profile at the hidden-equivalent compute, about 43k tokens per task): it would cut thinking below the 60k floor, so it needs the user's approval. Not built.
- **A p90 last-admission cutoff:** not built. Every request and verification is now clamped to the deadline, so a late admission cannot overrun, while a p90 cutoff would drop late coverage. The fixed 240 s admission gate stays.
- **Target-only DFlash parity:** needs a second server launch inside the 12 h; only the same-config control is built.
- **Engine details, unchanged (E11 cell):**
  - mamba cache 56 slots for 10 streams (donor ratio 60);
  - CUDA-graph batch list 1–8, 10, 12, so 9 pads to 10, which is SGLang's own behaviour.
  - The real running count is now in the report to verify it on GPU.

### Validation (all on the final files)

- `smoke_rev6.py` **91/91** (+23 since rev 6):
  - Q-5 bound and levels;
  - object_diff;
  - overfit signs (unit + on a host-verified candidate);
  - events;
  - infra label;
  - Q-3 unavailable global;
  - 2 exact + 1 timeout → rejected;
  - D3 ×2;
  - S3: more tasks, rounds, no double drive, round robin, overshoot bound;
  - T7 flagged first + class recorded;
  - final snapshot keeps paused candidates;
  - P4 worker exception;
  - P4 hanging server (ends at the deadline with the 1,800 s timeout clamped).
- `smoke_rev5.py` 56/56; `smoke_adapter.py` (flag off) 41/41; `regex_check.py` ALL OK; `facts_size.py` (numbers above).
- **Notebook harness on `fe51d1d4`:** r6 **41/41** (+4: parity control, M1 additions, class/flags per branch, prefix invariant), rerun 22/22, launchfail 6/6.
- **Regression corpus through the rev 6b agent:**
  - rev 4 EVAL6 (incl. 4c3d4a41 2/2): **7/8**;
  - rev 5 xhigh **5/8**; rev 5 low **10/19**, both equal to the rev 6 replay;
  - kept_r6 ≥ kept_r5 on every branch.
- **Kaggle-run notebooks:** unchanged (26fff41e, f4fed1ac, 415a68f2, 53a66856, b65fb9dc, 0fea4d0c, 216c59ed, cce9cb8f).
- **Backups:** `build_e12_duck.py.pre_rev6b_backup`, `fake_sgl_duck_server.py.pre_r6b`.

## Rev 6c: the user's approvals built (2026-10-21, user: "Gate 4b: permission granted. Take 12 streams. Permission for the rest too")

**Notebooks** (all rev 6; not pushed, not run):

| Notebook | sha | What |
|---|---|---|
| `Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL12R6.ipynb` | **`7d69e502…`** (supersedes `fe51d1d4`) | 12 tasks, 150 min solving, **12 workers** |
| `Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL12R6G4B.ipynb` | **`f70c3228…`** | **Gate 4b**: the same 12 tasks at the hidden run's compute per task |
| `Q27-XHIGH-E12_DUCK_REPL_SGLANG_PARITYR6.ipynb` | **`f3876b40…`** | **R6-X1 target-only parity** on 4 tasks, no solving |

All three are dual mode: in a competition rerun every hidden task is solved with the normal rev 6 settings. The gate-4b window and the parity diagnostic apply to commit mode only.

### What was built

- **12 streams:**
  - rev 6 builds use 12 workers (`LIVE = 12`; the `_S12` file suffix stays only for the rev 5 `--s12` build);
  - per-stream window = **1.0** × KV / 12 = 73,444 at the W1 KV of 881,336;
  - history budget 48.4k at the 24,576 cap and 60.6k at 12,288.
  - **Why 1.0 and not 0.9:** 0.9 gives 66.1k, i.e. 41.0k of history at the 24,576 cap, below the rev 5 median prompts of 41.6–51.9k, so trims would be frequent. D66 ran 12 slots at context 155,648 with a KV peak of 90 % and 0 preemptions.
  - The queue probe still steps 12 → 10 → 8 → 6 on server queueing.
  - The engine already serves 12: `SGL_MAX_RUNNING` 12, server 13 (the 11-of-12 workaround), CUDA graph up to 12, mamba slots 4 per request × 12 = 48 ≤ 56.
- **Gate 4b** (profile `EVAL12R6G4B`; it is a time share, not a token cap, so the scheduler, caps and fair share behave as in the hidden run):
  - CELL 6 sets the solve window to `n_tasks × (hidden solve end − this run's measured solve start) / 240`;
  - the hidden solve end comes from RERUN_ALL: 43,020 − 120 − 180 = 42,720 s;
  - at the usual start of about 800 s this is about 175 s per task, i.e. about 35 min for 12 tasks;
  - recorded in `report["rev6"]["g4b"]`.
  - **Comparison** of its score with EVAL12R6 (unconstrained): it predicts the LB/eval ratio (gap check "gate 4b").
- **R6-X1 target-only parity** (profile `PARITYR6`, 4 tasks):
  - For each task, the rev 6 system prompt + first opener, with tools, T 0, top_p 1, one request at a time, max 2,048 tokens, with thinking on and off.
  - Two rounds on the DFlash2 server (the same-config control), then the server is killed and relaunched from the **same command without the four `--speculative-*` flags**, and the same 8 probes run again.
  - Compared: exact equality, common prefix characters, tool-call presence and arguments, finish, tokens.
  - `report["rev6"]["parity"]`; raw replies in `duck/parity_*.json`; the target-only server log is bundled.
- **Stage-2 p90 cutoff:** `solve_all_r6(stage2_p90_cutoff=...)`, knob `E12_R6_P90_CUTOFF`, **built and tested but off by default.**
  - When on, no stage-2 item starts when the time left is below the p90 of the stage-2 run durations (≥ 5 runs, capped at 900 s). Stage 1 is exempt, so every task still gets its first exposure. Skips are counted.
  - **Why off by default:**
    - every request and verification is already clamped to the deadline, so a late admission cannot overrun;
    - a fair-share chunk of 24,576 tokens at about 25–30 tok/s per stream takes about 14–16 min, so the cutoff would idle up to 12 workers for the last ~15 min;
    - auto-check keeps a candidate found mid-run.
  - Switch it on with `E12_R6_P90_CUTOFF=1`.
- **Engine, not changed (permission used, but the numbers show no need):**
  - mamba cache 56: this engine needs 4 slots per request, so 12 streams need 48;
  - the CUDA-graph list 1–8, 10, 12: 9 and 11 pad up, which is SGLang's normal behaviour and avoids capture memory.
  - The real running count and acceptance are in the report, so this can be checked on GPU.
- **Header fix:** the rev 6 notebooks' markdown described rev 5 ("EVAL30R5 … 49,152 … 6 live"). Each variant now has its own rev 6 text.

### Validation

- `smoke_rev6.py` **93/93**. The 2 new tests cover the p90 cutoff: skips near the deadline with every task still getting branch 0, and off by default.
- `smoke_rev5.py` 56/56; `smoke_adapter.py` 41/41.
- **Notebook harness:**
  - EVAL12R6: r6 **43/43** (12 workers, window 73,444, 12 isolation markers, ≤ 12 in flight, p90 off, no g4b), rerun 22/22, launchfail 6/6;
  - **EVAL12R6G4B g4b 44/44**: window = 12 × (42,720 − measured start) / 240 (177.9 s/task at the harness start of 14 s), and the deadline = the smaller window;
  - **PARITYR6 parity 19/19**:
    - all 4 spec flags removed and the relaunch is ready;
    - control identical 8/8;
    - the planted target-only difference on one task (2 probes) is found: 6/8 identical;
    - tool-call presence and finish agree 8/8;
    - the relaunched server saw 8 probes without speculation;
    - its log is written;
    - the fallback submission stays.
- Replays are not re-run: rev 6c changes only the scheduler, the builder and the knobs, not the agent.
- The Kaggle-run notebooks are unchanged.

### Capacity (pre-run check)

- **EVAL12R6:** 24 branches, 12 workers, ≤ 2.4M tokens in a 9,000 s window.
  - Rev 5 measured 265 tok/s at 6 streams; 12 streams should be higher, but that is not measured on this stack.
  - Resident: 12 × 73.4k = 881k = the whole KV pool in the worst case. Real occupancy is lower (D66: peak 90 %), and the probe falls back.
- **EVAL12R6G4B:** about 35 min solving + about 13 min load, in an 80 min budget. Tokens per task = measured tok/s × 175 s (e.g. 46k at 265 tok/s).
- **PARITYR6:** 2 launches (about 10–13 min each) + 24 probes × ≤ 2,048 tokens at batch size 1, in a 75 min budget.

## GPT final pre-GPU audit of rev 6c (2026-10-21, user-pasted): validated

**GPT's verdict:**
- EVAL12R6G4B: GO.
- EVAL12R6: GO, after the G4B review.
- PARITYR6: NO-GO until its deadline is fixed.
- Claude did not drift from the fix list.
- GPT re-verified independently: notebook shas, 23/23 package files identical in all three notebooks, the embedded adapter equal to the source, and the bundled 93/56/41 test logs.

| # | GPT finding | Verdict (my check) | Evidence | Proposed fix (not built; the user decides) |
|---|---|---|---|---|
| C6-1 | **PARITY deadline bypass** (BLOCKER for PARITY only) | **Confirmed** | `build_e12_duck.py:852` `urlopen(req, timeout=900)` with no deadline. There are 24 probes, so a hung server could hold the session for up to 6 h. The relaunch wait (`:874`) is bounded by HARD_STOP − 600 but has `max(60, …)` | Pass one absolute deadline (`_end`) to every probe and to the relaunch. No probe starts once the time left is under 60 s. The partial comparison is still written. **Test:** a fake server that hangs every request; the routine exits within the deadline |
| C6-2 | **The isolation preflight fails open** (HIGH) | **Confirmed** | `:940` `isolation_ok = not leaks`, so 0/12 own markers or empty replies still PASS. `:942`: an exception gives `None`, and `:972` keeps 12 workers on `None` because the test is `is not False` | Three states: PASS only when own == n and there are no leaks. Workers stay at 12 only on PASS; FAIL or INCONCLUSIVE → 6 plus a warning. **Test:** 12/12 own, 1 foreign, 0/12 own, HTTP exception |
| C6-3 | PARITY compares only the tool arguments, not the tool name | **Confirmed** (`:855`). There is only one tool (`python`), so a name difference is unlikely, but it is a real gap | Record `{name, arguments}` and compare raw and parsed arguments. **Test:** same arguments with a different name → FAIL |
| C6-4 | **The scheduler's final join has no shared bound** (HIGH for the 12 h run) | **Confirmed** | `arc2_duck.py:2421`: each `t.join(timeout=max(0, deadline − now) + 120)`. After the deadline each blocked thread can add 120 s, i.e. 12 × 120 s = 24 min worst case. The normal paths are deadline-clamped, so this is an exceptional path. The margin after the solve end is only `E12_STOP_BEFORE_HARD_SEC` 180 s | One absolute cutoff `join_until = deadline + 60` for every join. Then the final snapshot of every session (candidates kept) is published. **Test:** ≥ 2 deliberately blocked workers; the call returns by the cutoff and the submission is valid |
| Q1 | KV fraction 1.0 is acceptable for G4B, but the capacity fallback depends only on the server queue | **Confirmed, with a concrete fix found.** SGLang 0.5.21 exports `sglang:num_retracted_requests_total` (`observability/metrics_collector.py:474-476`), but our `metrics_snapshot` name filter does not keep it, so the probe cannot see retractions | Keep the retraction counter in the filter. The probe also steps down (12 → 10 → 8 → 6) when that counter rises in 2 consecutive probes, or when KV `token_usage` stays ≥ 0.98 for 3 probes. Report both |
| Q2 / C6-3 | PARITY probes of 2,048 tokens rarely reach a tool call | **Confirmed.** rev 5 xhigh first replies had a p50 of 33.6k tokens before any tool call; rev 6 caps the first request at 12,288 | Keep the 8 short probes and add 2 thinking-on probes at 12,288 (the real first-request cap), with name and arguments compared |
| Q3 | G4B uses the right formula, but it is a limited proxy, not a validated LB predictor: with 12 tasks, stage 2 starts much earlier than with 240 | **Confirmed. My wording was too strong.** In "Rev 6c" I wrote "it predicts the LB/eval ratio" | **Wording corrected (below):** G4B = a test of the same average time share, a throughput and accuracy proxy |
| C6-5 | The 800-token facts target is soft | **Confirmed and already reported** (18/120 over 800, max 1,396; the smallest level is returned when nothing fits) | Add a `facts_over_target` flag per branch plus a count in the report. Never truncate the facts |
| C6-6 | "Conditional branch 1" is really a prioritised branch 1 (class 2 still gets branch 1) | **Confirmed. This is by design** (T7: about a third of accepts are test-wrong; 135a2760 low). My wording "conditional" is inaccurate | Wording corrected to "prioritised by evidence class". The marginal value is already measured (`branch1_yield_by_stage2_class`: branches, new distinct verified, tokens) |
| Doc | The common notebook header still describes the donor defaults (`build_e12_duck.py:1097`: "128 Ki context … T 0.7") next to the rev 6 settings | **Confirmed** (a documentation contradiction, not code) | Split it into "Duck donor defaults" and "active rev 6 settings" |

**Correction to "Rev 6c" above:** G4B does not "predict the LB/eval ratio". It is the same average time share per task, and a limited throughput and accuracy proxy. Its queue (12 tasks) differs from the hidden run (240), so the stage-2 timing, cache distribution and late-task pressure differ.

**Proposed patch set before GPU** (all small, adapter + builder, each with the test above):
- **Before G4B:** C6-2 (isolation, three states), C6-4 (shared join cutoff), Q1 (retraction / token-usage step-down), C6-5 flag, the header wording.
- **Only if PARITY is run:** C6-1, C6-3 and the 2 long probes (Q2).
- **Then:** rebuild the three notebooks, check the new shas, check embedded adapter == source, re-run every suite.

**Status:** validated. No code changed yet (rule: no code change until the user says so).

### GPT audits 2 and 3 of rev 6c (same day, user-pasted): validated, merged with audit 1

**The three verdicts agree:**
- EVAL12R6G4B: GO, run first.
- EVAL12R6: GO after the G4B evidence.
- PARITYR6: optional, NO-GO / conditional GO until its deadline is fixed.
- No fix-list item is missing.
- No donor regression; the 22 vendor files are unchanged.
- Rev 6c is not yet qualified as a 240-task submission.

**New points in audits 2 and 3:**

| # | Point | Verdict (my check) | Proposed fix |
|---|---|---|---|
| A2-1 | Facts bound: when no level fits, the last level is returned (18/120 over 800, max 1,396; audit 2 also reproduced 955 on a synthetic task). Proposes 2 extra compact levels | **Confirmed** (= C6-5) | Level 4: per-pair shapes + test shapes + an inspection notice. Level 5: counts + an inspection notice. Each is used only if the previous one is over the limit. **Test:** 120 eval + synthetic multi-pair tasks stay within both limits; every TEST is listed; deterministic; no truth leak |
| A2-2 | KV: 1.0 is fine for the G4B pilot. Audit 2 prefers 0.95 as the hidden-run default (44.7k history); audits 1 and 3 accept 1.0 | Arithmetic **confirmed**: 0.95 → 69,772 per stream, 44,684 input at the 24,576 cap | Keep 1.0 for G4B (capacity qualification). Decide the hidden default from G4B's real running count, KV peak, retractions and queue |
| A2-3 / A3 | **The test bundle is not self-contained:** `smoke_rev6.py` imports `smoke_adapter.py`, which is not in the zip; the fixtures and Windows paths are missing | **Confirmed** (zip listing: no `smoke_adapter.py`, no fixtures) | The next zip includes `smoke_adapter.py`, the fixture loader and the 12 + 120 task fixture files; paths become relative |
| A3-F3 | The p90 index `d2[int(0.9 × (n − 1))]` is not the nearest-rank p90: [100, 200, 300, 400, 800] gives 400 instead of 800 | **Confirmed** (computed) | `d2[ceil(0.9 × n) − 1]` (the cutoff is off by default) |
| A3-F4 | The first exposure (28k) is soft, so 240-task coverage is not guaranteed; check `stage1_left` / `tasks_exposed` | **Confirmed.** This is the known trade-off of the approved design; `stage1_left` is in `SCHED_LAST` | Report `stage1_left` at the deadline and `tasks_exposed` in the run summary line; full coverage is claimed only when `stage1_left == 0` |
| A3-Q3 | G4B's 12 tasks were chosen from earlier solver results (not a random hidden sample), so its score cannot forecast the LB | **Confirmed** (EVAL12R5 = EVAL6 + 6 informative EVAL30 tasks) | Interpretation: "a selected 12-task screening at the hidden budget", never an LB projection |
| A3-Q2 | PARITY covers only the first request: no multi-turn preserved-thinking / tool-result continuation | **Confirmed** | If PARITY is run: one probe with a recorded tool-result continuation, as well as the long probe (Q2) |
| A3-F2 | Isolation: `own_marker = 0` with `isolation_ok = True` is really inconclusive | **Confirmed** (= C6-2) | Covered by the three-state fix |

**Consolidated patch set (from all three audits; small; adapter + builder + tests; waiting for the user's go):**
1. **C6-2:** isolation in three states (PASS only with own == n and no leak); 12 workers only on PASS.
2. **C6-4:** one absolute join cutoff, deadline + 60, then the final snapshots.
3. **Q1:** the capacity step-down also reacts to `sglang:num_retracted_requests_total` rising, or to KV `token_usage` ≥ 0.98; both reported.
4. **A2-1 / C6-5:** facts levels 4 and 5 plus a `facts_over_target` flag.
5. **A3-F3:** the nearest-rank p90.
6. **A3-F4:** `stage1_left` and `tasks_exposed` in the run summary.
7. **Docs:** the header splits the donor defaults from the active rev 6 settings; G4B is worded as a screening test.
8. **PARITY only:**
   - deadline clamp on every probe and on the relaunch (C6-1);
   - tool name + arguments (C6-3);
   - 2 long thinking-on probes at 12,288 + 1 tool-result continuation (Q2).
9. **Bundle:** a self-contained test set.

Then: rebuild the 3 notebooks, record the new shas, check embedded adapter == source, re-run smoke rev6 / rev5 / adapter, the 5 harness scenarios and the 3 replays.

## Rev 6d: items 1–5 of the consolidated audit patch set (2026-10-21, user: "OK, do 1 to 5")

| # | Item | Built as | Test |
|---|---|---|---|
| 1 | **C6-2:** isolation in three states | `isolation_verdict(marks, replies)`: PASS only when every reply holds its own marker and no foreign one; any foreign marker → FAIL; missing markers without a leak → INCONCLUSIVE. `workers_for_isolation`: 12 only on PASS, otherwise 6, with a printed warning. The preflight uses both | smoke: PASS / FAIL / INCONCLUSIVE and the 12 / 6 / 6 worker rule; harness: PASS → 12 workers |
| 2 | **C6-4:** one absolute join cutoff | `join_until = deadline + join_grace_sec` (60) for every join, then a final snapshot. **Found while testing:** published records shared nested lists (`runs`, `checked`) with the live session, so a stuck worker could still change them, even while CELL 7 serialized them. Records are now **deep-copied** on publish, and nothing is published after shutdown (`closed`) | smoke: 3 workers stuck for 4 s with a 0.5 s grace → the call returns in < 2.5 s, stragglers are counted, and the records and callbacks are unchanged after the stuck workers finish |
| 3 | **Q1:** capacity step-down on retractions and KV usage | The probe steps 12 → 10 → 8 → 6 on queue > 0 for 3 probes, **the retraction counter rising in 2 consecutive probes**, or **KV usage ≥ 0.98 for 3 probes**; the reason is recorded. `_e12_capacity_probe` reads `num_queue_reqs`, `num_retracted_requests_total` and `e11:kv_cache_usage` (= SGLang `full_token_usage`) | smoke (fast clock): rising retractions → 6 (reason "retract"); KV 0.99 → "kv_usage"; a healthy server keeps 12; harness: no events on the fake server |
| 4 | **C6-5 / A2-1:** facts levels 4–6 + `facts_over_target` | Level 4 = shapes + colour counts + changed-cell counts per pair, and shapes + colours per test input. Level 5 = shapes only. Level 6 = counts only. Each says where the details are in Python. Nothing is cut mid-line | 120 eval tasks: **0 over 800 tokens (max 798, p50 699, p90 779)**. 16 tasks at level 4 and 2 at level 5 (none at 6). Every TEST is listed. Deterministic under 3 hash seeds. An impossible bound → level 6 + `facts_over_target` |
| 5 | **A3-F3 / A3-F4 / docs** | Nearest-rank p90 via `stage2_cutoff_hit()` (a pure function; the cutoff stays off). Report and summary line `E12 rev 6 coverage: tasks exposed / stage-1 starts left / workers / capacity events / stragglers`. Header: "Duck donor defaults … rev 6 overrides them"; G4B worded as "a screening test, not an LB projection" | smoke: deterministic cutoff rule ([100, 200, 300, 400, 800] → 800, cap 900, ≥ 5 runs); harness: 12/12 exposed, 0 left, 0 stragglers |

**Correction of my audit note (Q1):** I wrote that "our `metrics_snapshot` filter drops `num_retracted_requests_total`". That was the **vLLM-variant** filter in the builder. The SGLang notebooks use E11's `_E11_SNAP_KEYS`, which already keeps `retract` and `token_usage`. The real gap was only that the probe did not use them, and that is fixed in item 3.

**Test hygiene:**
- Three scheduler tests were timing-sensitive on this Windows machine. The fake 240-task sim took 3–18 s for the old and the new code alike, so the slowdown was machine noise.
- Fixes:
  - the p90 cutoff check is now a deterministic unit test, and the sim checks only coverage;
  - the S1 order tolerance is 30 ranks (10 workers + jitter);
  - stage-2-after-stage-1 allows 0.5 s of first-request jitter (admission itself is ordered);
  - the fair-share sim runs 10 s.
- Result: `smoke_rev6.py` **104/104 on 3 consecutive runs**.
- The fake server now also recognises the compact facts levels. Without that, 3 tasks were unidentified in the harness; this was a fixture gap, not a production one.

**Validation:**
- smoke rev6 104/104 (×3); rev5 56/56; adapter 41/41; regex ALL OK; facts_size as above.
- Harness:
  - EVAL12R6: r6 **45/45**, rerun 22/22, launchfail 6/6;
  - EVAL12R6G4B: g4b **46/46**;
  - PARITYR6: parity 19/19.
- Replays through the 6d agent, unchanged: rev 4 EVAL6 **7/8**, rev 5 xhigh **5/8**, rev 5 low **10/19**; kept_r6 ≥ kept_r5 on every branch.
- Kaggle-run notebooks unchanged.
- Adapter sha `7c60454c…`, embedded in all three notebooks.
- Backups: `arc2_duck.py.pre_rev6d_backup`, `build_e12_duck.py.pre_rev6d_backup`.

**Not taken (the user chose 1–5):**
- 6: the PARITY fixes (C6-1 deadline, C6-3 tool name, Q2 long probes). PARITYR6 stays NO-GO.
- 7: a self-contained test bundle.

## GPU run log

| When | Notebook | Account / kernel | What |
|---|---|---|---|
| 2026-10-21 | EVAL12R6G4B `1bdadd82…` | `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b` v1 | User: "run EVAL12R6G4B on the Foysal account". The check-only passed (account, 3 models, 2 datasets, sha); GPU quota used 21.88 of 30 h. Commit run (SAVE_AND_RUN_ALL) queued. Expected about 50 min (13 min load + about 35 min gate-4b solve + report). EVAL12R6 comes after the G4B review |
| 2026-10-21 | EVAL12R6G4B v1 **result** | same | COMPLETE in 3,528 s. **2/19, task-norm 16.67 %** (bf45cf4b, b5ca7ac4). 12/12 tasks exposed, 47.7k tokens per task. **Bug: the capacity probe stepped 12 → 6 on queue = 1 at KV 13–31 %**, which is the server's steady state with 12 in flight (11 running + 1 queued). Facts view: first call after 46 tokens, but 7/12 b0 branches never wrote a transform. One or two long thinks per branch remain (p90 25.6k characters per request). **Print-guard leak:** `print(r, row)` rows bypass `_GRID_ROW_RE` (173 rows in 3 branches). Load 1,197 s (slow cold start). **User decision:** replace the probe with "no new task while KV usage ≥ 0.95" (not built yet). Review bundle for GPT: `E12_REV6D_G4B_RESULTS_FOR_GPT_2026-10-21.zip`. Full analysis: `Experiment Results/E12 rev 6d G4B GPU run analysis (2026-10-21).md` |
| 2026-10-22 | EVAL12R6G4B_REBUILD `5d2bc39f…` (rev 6g) | `esfersami50/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b` v1 | User: "run the G4B commit run on Sami bhai's account". Check-only passed (account, 3 models, 2 datasets, sha); GPU quota used 26.87 of 30 h. The first save hit a connection reset (not saved; verified), the retry saved. Commit run (SAVE_AND_RUN_ALL) queued. Expected about 55-62 min |

## GPT audit of the G4B results (2026-10-21, user-pasted): validated, items G4-1..G4-9

**Verdicts:**
- unmodified EVAL12R6: NO-GO (the probe bug would recur);
- a consolidated corrected run: CONDITIONAL GO, provided the KV-only admission rule and its regressions pass.
- Main lesson: "an earlier Python call ≠ an earlier useful hypothesis".

**GPT's own tests:** 81 rev 6 tests passed, then the slow scheduler tests timed out in its environment; 4 rev 5 tests failed because `tokenizers` is missing there. These are environment limits. Our bundle run: 104 / 56 / 41, harness 46 / 45.

Validation script: `_scratch/session_1c6db7aa/validate_gpt_g4b.py` (+ `_out.txt`).

| # | GPT claim | Verdict (my check) | Evidence |
|---|---|---|---|
| 1 | Result numbers (2/19, 572,035 tokens, 180 requests, 75.0 % cached, 16 branches, 2 done / 14 paused, 0 stragglers, stage-1 left 0) | **Confirmed** | report |
| 2 | Probe at `arc2_duck.py:2397-2425` steps down on 3 queue probes; KV 13.3 / 26.0 / 31.5 % at the 3 events | **Confirmed** | code read; events |
| 3 | `_e12_capacity_probe` picks by substring; prefer the exact full-attention KV metric; `token_usage` is a max over pools | **Partly.** SGLang 0.5.21 `metrics_collector.py:88-92`: `token_usage` = max(full, swa, mamba); `full_token_usage` = the full-attention KV pool. **Our probe already read the full-attention value**: the first pick is `e11:kv_cache_usage(sglang:full_token_usage)`, which the E11 snapshot gives that name (`build_e11_05_s49_sglang.py:668-677`). The event values (0.13-0.31, below the mamba 0.62-0.70) confirm it. Only the last fallback `pick("token_usage")` could return the max | use the exact name; no fallback to `token_usage` |
| 4 | Fail-open with a warning when the metric is missing; mamba monitored only (max 0.70) | Agree | mamba max 0.71 in all 7 logs |
| 5 | **M1-A:** `accepted_candidates` counts check + submit of the same program twice (`build_e12_duck.py:1037`) | **Confirmed** | 4 events, 2 distinct programs and predictions; `fixed_offset` 2 = both from b5ca7ac4_b0 (check + submit) |
| 6 | **M1-B:** observed first-check median 29,384.5 (6/16 checked), b0 26,228; the reported 33,690.5 is a censored proxy | **Confirmed exactly** | — |
| 7 | **M1-C:** rev 5 aggregate cache is 84.3 % (xhigh) / 88.7 % (low), not 85.4 / 91.0 | **Confirmed.** My README used request-level numbers | the G4B deficit vs the same denominator: −9.3 / −13.7 points |
| 8 | **P1-1:** all 4 b1 first requests were cut at 12,288 before any call (first call at 12.4-18.0k) | **Confirmed** | every b1: 1 no-call cut, then the second request (also 12,288) made the call. The 32,768 rescue was never used |
| 9 | **P1-2:** indexed rows (`0 [0, 0, 5, …]`) bypass the guard; the proposed `_INDEXED_GRID_ROW_RE` | **Confirmed.** The proposed regex matches **173 of 173** leaked rows and does not match `show()` rows. **It misses** `0: [..]`, `row 3: [..]` and tuples, so it should be widened | regex test in the validation script |
| 10 | Transcript refs 136b0064_b0 :150-201 (guard hides rows) and :541-669 (hypothetical rules, no executable transform) | **Confirmed** | 6 guard notes in 150-201; no `def transform` in 541-669 |
| 11 | Cache misses: 15/180 later requests < 5 % while KV ≤ 0.44; trims explain only 2 branches; SGLang #22935 and #36935 support a mamba-checkpoint hypothesis | **Confirmed, and stronger than GPT says.** #22935 (open): the mamba radix split leaves the parent without `mamba_value`, giving a 0-hit. #36935 (open, seen on 0.5.17, "persists in latest"): branch reuse collapses when kv_pool_tokens / chunked_prefill_size > max_mamba_cache_size. **Ours: 866,744 / 4,096 = 211.6 vs a mamba cache of 56**, deep in that regime. Still a hypothesis for 0.5.21 + `extra_buffer_lazy` until replayed | GPT's bounded replay diagnostic is the right test |
| 12 | **P0-2 nudge:** after 2-3 read-only calls with no transform, one "write a provisional transform" nudge | **Threshold not supported by our data.** The two solved branches defined `transform` only at call 15 (bf45cf4b_b0) and call 18 (b5ca7ac4_b0); the 7 no-transform branches had 8-14 calls. A call-count trigger fires in the solved branches too and does not separate them | trigger on generated tokens without a transform; replay it on bf45cf4b_b0 / b5ca7ac4_b0 so they are not disturbed |
| 13 | Q5: 12 running slower (194-258 vs 370-406 at 11), only 29 lines, so do not drop to 11 | Agree | — |

### Items (for the user's decision; nothing built)

| ID | Change | Priority | CPU gate |
|---|---|---|---|
| **G4-1** | Probe replaced by the user's rule: 12 workers fixed; no new task or branch while full-attention KV usage ≥ 0.95; poll about 5 s; exact metric name; bounded retry, then fail-open with a warning and a counter | P0 | queue 1 / KV 0.31 → 12 unchanged; KV 0.97 → hold; 0.94 → resume; metric missing → warning, no deadlock; deadline while held; 240-task sim; **replay the 7 recorded logs: no hold except E12 EVAL30 rev 1** |
| G4-2 | b1 first request cap 24,576 (b0 stays 12,288) | P1 | payload capture; a cut call never runs |
| G4-3 | Guard: indexed rows (`N [..]`, `N: [..]`, `row N: [..]`, tuples) | P1 | 173/173 recorded leaks hidden; `show()` and summary dicts unchanged |
| G4-4 | Bounded indexed whole-grid `show()` (≤ 30×30, token-checked) | P1 | tool-output budget with the Qwen tokenizer |
| G4-5 | `arc_overview()` does not repeat the system-prompt facts | P1 | — |
| G4-6 | One provisional-transform nudge, **token-triggered** (threshold from a replay, not a call count) | P1 | the solved branches are not nudged before their transform; sent once per state |
| G4-7 | Report: accepted events / unique programs / distinct predictions; first check observed / censored / no-check count; aggregate cache denominator | P2 | check + submit = 2 events, 1 program |
| G4-8 | Cache diagnostic (exact-prompt replay: warm → replay → other prefixes → replay; cached tokens, TTFT, hit rate, KV, mamba) in the same GPU session | P1 telemetry | — |
| G4-9 | Not now: `--disable-radix-cache`, a mamba-strategy switch, kernel patches (GPT). Note for later: per #36935, a larger `--max-mamba-cache-size` or `--chunked-prefill-size` changes the 211.6 vs 56 ratio; measure with G4-8 first | — | — |

**Kept:**
- rev 5 security;
- harvest and the host verifier;
- context preservation;
- stage-1 queue, pass@2 and the deadline clamp.

### GPT audit 2 of the G4B results (same day, user-pasted): validated, new points only

**Verdicts:**
- G4B results usable as diagnostic evidence: GO.
- Next GPU push as is: NO-GO.
- Mandatory: the KV-only gate, the indexed guard and the scheduler tests.
- It agrees with audit 1 on everything already listed above.

Validation script: `_scratch/session_1c6db7aa/validate_gpt2_g4b.py` (+ `_out.txt`).

| # | Claim | Verdict (my check) | Evidence |
|---|---|---|---|
| 1 | 14 of the 15 low-cache later requests have `cached` = null, so a real 0-hit "is not confirmed" | **Null confirmed (20 of 180 requests, 254,890 prompt tokens; 14 later ones), but the conclusion is wrong.** SGLang 0.5.21 `usage_processor.py:13-15` returns `prompt_tokens_details` only when cached > 0 ("keeps JSON slim"), and our server runs `--enable-cache-report`. So null = exactly 0 cached tokens: **the 0-hits are real.** Null share: G4B 20/180 (11 %) vs rev 5 xhigh 6/127, low 23/449 (5 %) | The cache regression is real. Item G4-7 still records "null = 0 (SGLang omits it)" in the report |
| 2 | Exactly 12 running: 29 lines, early phase only, not a causal slowdown; some rows ~380-394 | **Confirmed, and my claim is withdrawn.** All 29 lines fall in 11:59-12:12 (before the first step-down). They alternate between about 42-250 tok/s and 382-497 tok/s, which fits admission prefill interfering with decode, not a 12-stream ceiling | No change to 12 workers |
| 3 | Mamba usage max about 0.70, so "56 slots full" cannot be claimed | **Accepted.** It weakens the #36935 reading I added above (the 211.6 vs 56 ratio is the issue's criterion, but sampled mamba usage never reached full). The cause stays unproven | G4-8 (prefix instrumentation) decides |
| 4 | FP8 KV "no scaling factors, defaulting to 1.0" in the server log | **Confirmed, not new.** The same 8 warnings appear in rev 4 EVAL6 and both rev 5 runs | not a blocker; note |
| 5 | Facts 510-798 tokens in the 16 branches | **Confirmed** | — |
| 6 | Transcript refs: 4c3d4a41_b0 :155-159 (`print(r, row)` leak), :987 (per-column reasoning); f931b4a8_b0 :888 (hand tabulation); 136b0064_b0 :1444 (geometric narration); 247ef758_b0 :1930 / :2175 / :2876 (checks 1/3, 0/3, 1/3) | **Confirmed** (all lines match) | — |
| 7 | Widened indexed regex (`row`, optional colon, case-insensitive) | **Confirmed:** catches 173/173 leaks plus `0: [..]` and `row 3: [..]`; leaves `show()` rows, dicts and short lists alone. **It misses tuples** `0 (0, 0, 5, 2)` (none were seen) | use it for G4-3, plus a tuple case |
| 8 | b1 = 21.6 % of generated tokens with 0 verified candidates | **Confirmed** (123,560 / 572,035) | — |
| 9 | 240 × 60k = 14.4M tokens = 10.81 h at 370 tok/s vs an 11.46 h window | **Arithmetic confirmed** | — |

**New proposals (for the user; nothing built):**

| ID | Change | Note |
|---|---|---|
| **G4-10** | **Candidate-first prompt** (verbatim text in the audit): the first call looks at evidence not already in the facts; test an executable hypothesis before a full verbal explanation; after several inspections with no candidate, test the simplest current hypothesis even if it may fail; no handwritten grid reconstruction | A prompt rule, not a trigger. Our data: both solved branches wrote `transform` only at call 15 / 18, so "within the first 2-3 interactions" must not become a hard rule. Prompt-only rules failed before (Exp 4 run 4, E11-06); pair it with G4-6 |
| **G4-11** | **b1 representation:** a compact indexed grid + concise differences instead of the raw grids view; small grids exact, large grids as a bounded digest | Overlaps G4-4 (indexed `show`). Measure: distinct verified candidates per 100k b1 tokens |
| **G4-12** | **Stage-2 allocation:** before starting a new b1, prefer refining a b0 with productive partial progress (247ef758_b0: 1/3 pairs, cell accuracy 0.771) | An allocation hypothesis, not proven (GPT says so too). It changes the T7 class order; needs the 240-task simulator |
| G4-13 | Patch E dedupe key = (task, test outputs); branch agreement keeps the branch identity | = G4-7, more precise |

### Re-read of both G4B audits (user 2026-10-21: "re-read the reviews, I think you missed something")

**What I had missed:**
- **Main gap:** the fix-by-fix verdicts (worked / not / why / what would make it work) were the user's main question to GPT. I had validated GPT's claims, but I had not recorded the verdicts. They are below.
- **Smaller gaps:** six proposals and two corrections were also missing.

#### Consolidated fix-by-fix verdict (audit 1 = A1, audit 2 = A2; both agree unless noted)

| Fix | Verdict | Why (measured) | What would make it work |
|---|---|---|---|
| R6-V1 facts view | **Partly** | first call after 46 tokens, but in most branches it is a redundant `arc_overview()`; 7/12 b0 wrote no transform | candidate-first steering (G4-6, G4-10); A2: compact indexed grid / structural diff alongside the facts |
| R6-V2 facts block | **Worked** (size 510-798) | usefulness unproven | drop the redundant `arc_overview()` echo (G4-5) |
| R6-V3 helpers | **Underused** | `inspect_pair` 7, `components` 3, the rest 0 | A1: structured observations, **no more helpers**. A2: **2-3 task-relevant usage examples** of the key helpers in the prompt (G4-16) |
| R6-V3 T1 | **Not exercised** | no accepted transform used a helper | keep the CPU test (it exists in `smoke_rev6`) |
| R6-V4 guard | **Partly; bug** | 432 hidden, 173 leaked | G4-3 + G4-4 (A1: page the rows if a whole grid exceeds the tool-output budget) |
| R6-P1 prompt | **Did not meet its goal** | 7/12 b0 had no transform; long manual reasoning | G4-10 + G4-6 |
| R6-P2 resumes | **Worked mechanically** | first 16, continue 45, length 5, partial 5, verified 1 | A1: a more targeted candidate-state text; A2: CPU-test the states not reached |
| R6-P3 generalization | **Not testable** | sent only in the 2 solved branches | judge after runs with several train-exact candidates |
| T6 nudge | **Not exercised** | `nudges_replaced` 0 | A2: a stage-specific nudge when no first candidate is written (= G4-6) |
| R6-K1 auto-check | **Worked when triggered** | 2 calls | more transforms first; no change to the mechanism |
| R6-K2 `check_many` | **Not used** | 0 calls | A2: one prompt example for two hypotheses (G4-16) |
| R6-K3 verify | **Worked** | 2 verified, 1 duplicate skipped, 0.86 s | keep |
| Rev 5 B1 / B3 | **Worked** | 2 harvested; estimate ratio 0.998 | keep |
| Rev 5 B2 / B6 | **Not exercised** | ≤ 1 candidate per task; 0 cut calls | B2 needs candidate diversity first; keep the B6 tests |
| R6-S1 coverage | **Worked** | 12/12, stage-1 left 0 | A2: a 240-task capacity test (simulator) |
| R6-Q2 soft pause | **Worked** | overshoot p50 5.1k, max 15.8k | keep; never call 28k a hard cap |
| R6-S2 b1 | **No yield** | 4 b1, 123,560 tokens (21.6 %), 0 candidates; every b1 first request cut | G4-2 (+ G4-11 / G4-12); keep b1, but not unconditionally |
| **R6-S3 fair share** | **Did nothing in this run (A2: "weak allocation")** | **verified: every computed share (7,544 → 3,758) was below the 24,576 minimum chunk**, so all 6 stage-2 runs got ≥ 24.6k (25-38k) regardless of the share | **G4-17** (new) |
| R6-S4 progress | **Worked once** | 247ef758_b0: 1/3 pairs, accuracy 0.771, 5 partial resumes, no solve | A1: a concise correction for a stalled partial program; A2: **track improvement of the same transform separately from a distinct hypothesis (G4-18)** |
| R6-S5 capacity | **Bug** | 12 → 6 at KV 0.13-0.31 | G4-1 |
| R6-S6 deadline | **Worked** | 91 clamps, 0 stragglers, finished 9 s early | keep |
| R6-Q1 caps | **Mixed** | 7 no-call cuts, all 4 b1 first requests | G4-2 (A1: 24,576 still does not guarantee a completed call) |
| R6-D1 window | **Conservative, works** | KV ≤ 0.44, 1 reasoning-only trim | no change now |
| R6-D2 prefix | **Regression, cause unknown** | 75.0 % vs 84.3 / 88.7 %; 20/180 0-hit requests vs 5 % in rev 5 | G4-8 + G4-15 |
| R6-X1 serving | **Limited PASS** | isolation 12/12, control parity 2/2; target-only never run | never claim target-only parity |
| R6-M1 metrics | **Mostly; counting bugs** | double count; censored median; prose hand-reading not detected | G4-7 + G4-14 |

#### Missed proposals (new IDs)

| ID | Change | From | Note |
|---|---|---|---|
| **G4-14** | Metrics: tokens to the first `transform` definition; share of branches with a candidate within 20k; train pairs passed by the best candidate; train-exact candidate yield per task; distinct verified candidates per 100k b1 tokens | A1 P0-2, A2 Patch C/D | measurement only |
| **G4-15** | **Per-request production instrumentation:** exact input-prefix match length vs the branch's previous request, cached tokens (null = 0), trimmed flag, mamba usage, **idle time since the branch's last request** (paused branches); per-concurrency telemetry (running count, prefill in the batch, acceptance length) for the 12-running question | A2 Q4, A1 Q5 | separates "the prefix changed" from "the cache state was lost", in the real run. G4-8 is the controlled replay |
| **G4-16** | Prompt: 2-3 short usage examples of the key helpers, and one `check_many` example for two hypotheses | A2 (V3, K2) | short; prompt-only rules have failed before, so measure helper use |
| **G4-17** | **Fair share:** the 24,576 minimum chunk overrides every computed share, so the fair-share logic is inert. At 240 tasks the computed share will be smaller still. Options: lower the minimum chunk to about 12,288, or allocate by the remaining candidates (A2) | A2 S3, verified | **the user decides;** a smaller chunk = more, shorter stage-2 rounds (more re-prefill), so it needs the 240-task simulator |
| **G4-18** | Progress: record "same transform improved" separately from "a new, distinct hypothesis" | A2 S4 | feeds G4-12 and the stage-2 class |
| G4-4+ | Whole-grid `show()`: page the rows when a grid exceeds the tool-output budget | A1 P0-2 | part of G4-4 |

#### Corrections to my own numbers

1. **Probe throughput loss:**
   - I wrote "about 15-20 %". At the same context bin (20-40k) the loss is 370.9 → 321.4 = **13.3 %**, so A2's "13-16 %" is right.
   - My 20 % mixed the < 10k bin (406.3) with the 20-40k bin.
2. **#36935:** downgraded in audit 2 row 3 (mamba usage never full). The cause is still open.

## Rev 6e: G4-1..G4-18 built (2026-10-21, user: "take all of them")

| ID | Built as | Test |
|---|---|---|
| **G4-1** | `solve_all_r6`: the queue / retraction / KV-0.98 step-down is **removed**; workers stay fixed (12). The probe (`probe_once`, every `E12_R6_KV_POLL` 5 s, one synchronous sample before the first admission) only feeds the gate: **no new task or branch while full-attention KV usage ≥ `E12_R6_KV_ADMIT` 0.95**; running requests untouched. A missing / NaN / out-of-range / stale (> 30 s) sample admits (counted, one warning per outage); with nothing running a hold is released (no deadlock). Builder probe: exact names via `e11_prom()` (`sglang:full_token_usage`, `sglang:mamba_usage`, `num_running_reqs`, `num_queue_reqs`; max over TP ranks; never `sglang:token_usage`) | smoke: G4B steady state (queue 1, KV 0.31) → no hold, 12 concurrent; KV 0.97 → held (only idle releases, 1 at a time); release below 0.95; missing / invalid / exception / stale → fail-open; deadline while held → clean return. **Replay of 7 recorded logs (`kv_gate_replay.py`): holds only in E12 EVAL30 rev 1 (219 samples, 18.2 min); 0 in G4B, E11-05, E11-06, rev 4, rev 5 ×2.** Harness: the fake server now reports **queue 1 always** (the real steady state) → 12 workers, 0 holds |
| **G4-2** | Branch 1 (any grids view) first cap `E12_R6_CAP_FIRST_B1` 24,576; branch 0 keeps 12,288 | smoke; harness: no b1 request at 12,288 |
| **G4-3** | `_GRID_ROW_RE` + an indexed alternative: `N [..]`, `N: [..]`, `row N: [..]`, tuples, case-insensitive; linear time | smoke (print(i, row) ×12, row-tuples, dicts and short lists untouched); regex_check ALL OK |
| **G4-4** | `show(g)` prints a whole grid up to `E12_R6_SHOW_MAX_CELLS` 900 cells; per python call at most `E12_R6_SHOW_CALL_CELLS` 1,000 shown cells; beyond that it prints the rows that fit and says `call show(g, r0=…)` next call (paging). Measured: a 30×30 show = 2,033 Qwen tokens (tool cap 3,072). The guard note and prompt point to `show(grid)` | smoke: whole 30×30; paging; > 900 refused; crop keeps indices; a full call < 3,072 tokens (Qwen tokenizer) |
| **G4-5** | `arc_overview()` prints only the facts lines the bounded system-prompt block left out (with their TRAIN/TEST header), else a one-line pointer | smoke |
| **G4-6** | One provisional-transform resume (verbatim GPT text) when a branch has generated ≥ `E12_R6_NUDGE_TOKENS` **27,000** tokens without defining any solution-named function and without any check(). Threshold from the G4B replay (`g4_6_threshold.py`): the 2 solved branches defined `transform` at 18,383 / 26,228; the 10 branches that never defined one all passed 29k. Precedence: cut > length > timeout > provisional > verified / partial | smoke: not below 27k; once at 27.5k; suppressed by a transform; cut keeps precedence |
| **G4-7 / G4-13** | Report: `accepted_events`, `unique_verified_programs` (= `accepted_candidates`), `distinct_test_predictions`, per-task programs; overfit flags counted on unique programs; first check: observed median, observed b0 median, checked branches, censored proxy kept | harness |
| **G4-8** | `_e12_cache_diag` (CELL 6, commit mode only, ≤ 240 s, before solving): about 9k-token prompt A warm → exact replay → 2 divergent suffixes → 12 distinct long prompts at once (churn) → replay + divergent suffix; cached / prompt / time / KV / mamba per step; `report.rev6.cache_diag` | harness: every step recorded, warm 0 cached, replay > 0 |
| **G4-9** | Not built (by design): no radix-cache / mamba-strategy / kernel change until G4-8 / G4-15 show the cause | — |
| **G4-10** | Prompt: first call for evidence the facts lack; executable hypothesis before a verbal explanation; after several inspections, test the simplest hypothesis; no hand-reconstruction | smoke (prompt text) |
| **G4-11** | Branch-1 view `indexed` (default `E12_R6_B1_VIEW`): facts + every grid in the `show()` layout; bounded by `E12_R6_B1_GRID_MAX_TOKENS` 12,000 (grids > 400 cells become a `show(train[i]['input'])` pointer, largest first, every grid still listed) | smoke; harness (facts + indexed views) |
| **G4-12** | Stage 2: a task whose branch 0 has partial progress (class 0) gets no fresh branch 1 while each stage-2 round of branch 0 improves its best single-function progress; branch 1 is queued after a no-gain round (`BranchSession.stalled`) | smoke: improving b0 → never b1 (deterministic); stalled b0 → b1 |
| **G4-14** | `first_transform_tokens` per branch; report: observed / censored median, branches with a transform within 20k, best single-function pairs histogram, b1 distinct verified per 100k tokens, nudges | smoke; harness |
| **G4-15** | Per request (`branch_result.json` `requests`, ≤ 600): message-level prefix kept vs the branch's previous request, trimmed flag, idle seconds, prompt / cached (null = 0, flagged) / completion / finish, server KV / mamba / running / queue at the start. Report: zero-hit later requests with intact prefix / trimmed / idle / mamba; concurrency telemetry from the server log (decode tok/s by running count, split by "a prefill batch just before", accept length, mamba); scheduler gauge series every 30 s | smoke (3 requests); harness |
| **G4-16** | Prompt: helper examples (`components(g, ignore=modal_color(g))`, `object_diff`, `diff_summary`) and `check_many` for two hypotheses | smoke |
| **G4-17** | Minimum stage-2 chunk 12,288 (`E12_R6_MIN_CHUNK`; adapter default too). G4B: every computed share 3.8-7.5k was below 24,576 | smoke (default); harness (knob) |
| **G4-18** | Progress events: `progress_same_improved` vs `progress_new_hypothesis`; each run records best before / after and `improved`; report: stage-2 runs improved | smoke |

**Defects found while building (mine, fixed before the final tests):**
1. `BranchSession.write` read `agent.arc2_requests` directly, so a minimal agent broke the record → now `getattr(…, [])`.
2. The first KV sample could arrive after the first admissions (race) → one synchronous `probe_once()` before the workers start.
3. Two scheduler tests are timing-sensitive on this Windows machine (S3 round robin; my first G4-12 assertion). Both passed on re-runs. The G4-12 test now asserts the deterministic half strictly and the other half as "at least one stalled task got branch 1".
4. A parallel replay run collided in the Duck sandbox's temp-dir cleanup (Windows file lock). This is a test-environment issue; re-run sequentially.

**Fake server and harness follow real data now:**
- `/metrics` queue 1 always, mamba 0.62, KV from `FAKE_KV`;
- `prompt_tokens_details` omitted when 0 cached;
- G4-8 diagnostic requests handled;
- indexed view recognised.

**Validation (final files):**
- **Unit and regression:**
  - `smoke_rev6.py` **139/139**: 104 → 139; the old step-down tests were replaced; earlier ×3 at 138 before the last G4-12 split;
  - `smoke_rev5.py` 56/56; `smoke_adapter.py` 41/41;
  - `regex_check` ALL OK;
  - `facts_size` unchanged (p50 699 / p90 779 / max 798; deterministic under 3 hash seeds);
  - `kv_gate_replay` 7/7.
- **Harness on the final notebooks:**
  - r6 **51/51**, rerun 22/22, launchfail 6/6;
  - **g4b 52/52** (`_REBUILD`), parity 19/19.
- **Replay corpus through the rev 6e agent:**
  - rev 4 EVAL6 **7/8**, rev 5 xhigh **5/8**, rev 5 low **10/19**;
  - each identical to rev 6d per task, kept_r6 equal.
- **Package and originals:**
  - embedded adapter == source in all 3 notebooks; only `arc2_duck.py` differs from the Kaggle-run G4B package (22 vendor files identical);
  - all 9 Kaggle-run notebooks present and unchanged (26fff41e, f4fed1ac, 415a68f2, 53a66856, b65fb9dc, 0fea4d0c, 216c59ed, cce9cb8f, 1bdadd82).
  - The push script maps `--eval12r6g4b` to `_REBUILD` (`--rev6d` for the old file).
- **Backups:** `arc2_duck.py.pre_rev6e_backup`, `build_e12_duck.py.pre_rev6e_backup`, `fake_sgl_duck_server.py.pre_r6e`, `harness_e12_rev4_nb.py.pre_r6e`, `smoke_rev6.py.pre_r6e`.

**Not done:** GPU run (needs the user's go and a pre-run capacity check).


## Rev 6e self-audit (2026-10-22, user: "audit it yourself against the fix list, read today's whole conversation, update if needed, then the zip for GPT")

**Method:**
- I read the code of every G4 item against its fix-list row and its CPU gate.
- I re-read all 83 user messages of 2026-10-21 for requests that were never done.
- Three new checks:
  - a guard-leak scan over all 149 recorded transcripts (`_scratch/session_1c6db7aa/self_audit/guard_leak_scan.py`, `guard_leak_compare.py`);
  - a 240-task capacity simulation of the real scheduler (`self_audit/sim240.py`, output `sim240_out.txt`);
  - the gate-4b clock read in the built notebook.

### Found and fixed (2 defects, both mine)

| # | Defect | Evidence | Fix | Test |
|---|---|---|---|---|
| SA-1 | **G4-8 ate the G4B window.** CELL 6 fixes `_end` (the gate-4b window, 2,063 s in G4B) **before** the preflight and the cache diagnostic, which can run up to 240 s, about 12 % of the 12-task window. The hidden rerun never runs the diagnostic, so the comparison was biased. The builder comment claimed the opposite ("the window is computed from the later solve start") | code read: `_end` at the CELL 6 head, `_e12_cache_diag()` later in the same cell | For `hidden_equiv` only: the end is shifted by the time the preflight and diagnostic took (`started_sec` → solve start), still bounded by `HARD_STOP`. Recorded as `g4b.shift_sec` and `g4b.window_start_sec`. The per-task compute is unchanged | harness g4b: new check (the full window is left at the solve start, the shift is recorded) → **53/53** |
| SA-2 | **G4-3 was incomplete.** Rows printed as a row index plus joined digits (`print(r, ''.join(map(str, row)))` → `0 0000000000…`) also reached the model. **Corrected count:** G4B leaked **646 grid rows in 8 branches**, not 173 in 3. The 173/173 check covered only the bracket form. Rev 5 also leaked `r 0: [..]` | scan of the 149 transcripts with the real guard semantics (runs of ≥ 6) | `_GRID_ROW_RE` + 2 alternatives: an index (optionally `row` / `r`, `:` / `=`) followed by ≥ 8 joined digits, or `N: 0 0 5 2` with a colon. `|` is never a separator, so `show()` rows stay exempt; the match is still linear time | G4B rows reaching the model: rev 6d 646 → rev 6e 480 → **now 1**. All 149 transcripts: 1,564 → 438. The 438 left are narrow rows (≤ 7 joined digits), kept on purpose: an 8-digit floor avoids hiding counts. Smoke: 4 new tests (joined digits, `r N:`, `N: 0 0 5`, `show()` untouched) → **143/143**. `regex_check` (now imports the pattern) ALL OK; the adversarial-line timing test passes |

### G4 items: code vs fix list

| ID | Code matches the row? | Note |
|---|---|---|
| G4-1 | Yes | Fixed workers; gate on `sglang:full_token_usage` ≥ 0.95; fail-open on missing / invalid / stale; idle release; one synchronous sample first. 240-task sim: 0 holds, KV max 0.47-0.53 (assumed context model). Replay of 7 logs: 7/7 |
| G4-2 | Yes | `first_b1` for any non-facts view |
| G4-3 | **Fixed (SA-2)** | — |
| G4-4 / G4-5 | Yes | `show()` budget resets per call (prelude re-run) |
| G4-6 | Yes | Sent only in a resume opener (Duck's resume path), at most once |
| G4-7 / G4-13 | Yes | `unique_verified_programs` key = (task, branch, name, outputs); `distinct_test_predictions` = (task, outputs) |
| G4-8 | Yes; **SA-1 fixed** | Commit mode only (never in the hidden rerun) |
| G4-10 / G4-11 / G4-16 | Yes (prompt text) | Prompt-only rules failed before: judge them by `first_transform_tokens` and helper use |
| G4-12 | Yes | Sim: with 12,288 chunks, branch 1 started in 177-181 of 240 tasks |
| G4-14 / G4-15 | Yes | **Limitation:** G4-15 prefix match is message-level (JSON per message), not token-level |
| G4-17 | Yes | **Open (sim, below):** at 240 tasks the computed share falls below the minimum chunk in every one of the last 200 stage-2 runs, so the minimum chunk is the real allocation rule |
| G4-18 | Yes | **Limitation:** "same transform" = the same function name; a new hypothesis under the same name counts as "improved" |

### 240-task simulation (the CPU gate of G4-1, G4-12, G4-17 and A2 R6-S1; it was never run for rev 6e)

**Setup:** the real `solve_all_r6` + `BranchSession` with virtual time.
- **Measured:** 371 tok/s at 12 running (321 at 6), about 3.2k tokens per request, a 41,257 s window, 12 workers, 28k first exposure, 100k soft cap.
- **Assumed:** which branches solve and when (35 % of tasks by b0 at 15-90k, 10 % only by b1), partial progress, and the context → KV model.
- **So:** coverage and allocation numbers are scheduler facts; solve counts only show a direction.

| | min chunk 12,288 (rev 6e) | 24,576 (rev 6d) |
|---|---|---|
| Tasks exposed / stage-1 starts left | 240 / 0 (both seeds) | 240 / 0 |
| Last branch-0 start | 19,757 / 20,018 s of 41,257 | 19,728 / 20,018 s |
| Branch 1 started | 177 / 181 | 103 / 103 |
| Tokens per task p10 / p50 / p90 | 49.6k / 62.8k / 72.1k | 30.5k / 62.2k / 91.2k |
| Tasks with < 60k tokens in total | 89 / 90 | 101 / 110 |
| Stage-2 runs | 433 / 442 | 276 / 273 |
| Queue items never served at the deadline | 389 / 390 | 383 / 382 |
| Solved under the assumed model | 30 / 35 | 37 / 36 |
| KV holds | 0 | 0 |

**Reading:**
- **Coverage holds:** stage 1 ends at about 48 % of the window.
- **Budget:** 15.0-15.3M tokens / 240 = about 63k per task on average whichever chunk is used. About 385 stage-2 items are never served.
- **Chunk size:** 12,288 spreads tokens more evenly and starts more branch 1s. 24,576 lets fewer branches go deeper; under this solve model it solved equal or more.
- **Not modelled:** the re-prefill cost of the extra resumes (about +160 at 12,288).
- **Open for GPT and the user:** which allocation rule, given that fair share is inert at 240 tasks. Unchanged: the user approved G4-17.

### Today's conversation (83 user messages, 2026-10-21) vs what was done

- **Done:** every request from #1 to #83, except one.
- **Open:** #63 "make a commit on Hamid bhai's ID". It was overtaken by #64 (timings) and #65 (run G4B on the Foysal account). EVAL12R6 never ran on hosen42. It now has a new sha (`c036215e…`) and needs the user's go and a pre-run capacity check.
- **Still not taken (user: "OK, do 1 to 5"):** PARITY fixes 6-8.

### Validation (final files, 2026-10-22)

- **Unit tests:**
  - smoke rev6 **143/143**;
  - rev5 56/56; adapter 41/41;
  - `regex_check` ALL OK; `kv_gate_replay` 7/7;
  - facts size unchanged (p50 699 / p90 779 / max 798; deterministic).
- **Harness on the final notebooks:** g4b **53/53** (`_REBUILD`), r6 **51/51**, rerun 22/22, launchfail 6/6, parity 19/19 (`e12_harness/h15_*.txt`).
- **Replay corpus through the final agent:** rev 4 7/8, rev 5 xhigh 5/8, rev 5 low 10/19. Identical to rev 6e per task (hits and kept candidates); grid rows suppressed in low 1,051 → 1,161.
- **Environment note:** the harness needs `transformers`. It had disappeared from Python 3.14 (C: clean-up); reinstalled 5.20.2 (pip cache on D:).
- **Packaging:**
  - only the 3 rev 6e notebooks changed; the embedded adapter == source in all 3;
  - all 9 Kaggle-run notebooks are present and unchanged.
- **Backups:** `_scratch/session_1c6db7aa/self_audit/` (`arc2_duck.py.pre_selfaudit`, `build_e12_duck.py.pre_selfaudit` (reconstructed: the current file minus the SA-1 block), `fixlist.pre_selfaudit.md`, `harness_e12_rev4_nb.py.pre_selfaudit`); `smoke_rev6.py.pre_selfaudit`, `regex_check.py.pre_selfaudit` in `e12_smoke`.

## GPT audits of the rev 6e bundle (2026-10-22, three reports, user-pasted; the third was cut off at its Q6): validated, items H6-1..H6-13

**Verdicts:**
- Audit 1: G4B / EVAL12R6 NO-GO as is; PARITY GO (diagnostic only).
- Audit 2: G4B GO, EVAL12R6 GO after G4B; PARITY NO-GO; hidden submission not yet qualified.
- Audit 3: no explicit verdict in the visible part; HIGH-1 / HIGH-2 before G4B.
- All three confirmed: the notebook shas, the embedded package (23/23, vendor 22/22 unchanged), the regex / KV-replay tests and SA-2 (865 → 210 on the 64 bundled transcripts). None could run the full smoke suite: `tokenizers` / `transformers` are missing in their environment.

Validation scripts: `self_audit/g4_11_bound_check.py` (+ `_out`), `self_audit/sim240.py` with `SIM_RATE` (`sim240_rate_out.txt`), plus code reads.

| ID | Claim (audit) | My check | Verdict | Fix (proposed; nothing built) |
|---|---|---|---|---|
| **H6-1** | G4-6 nudge rides a new opener, so the 28k stage-1 pause usually comes first; a branch that is never resumed never gets it (A1 HIGH) | **Code:** the nudge is checked only in the resume branch of `_arc2_r6_opener`. Duck builds an opener once per turn, and a turn spans many requests. The stage-1 pause (`stop_requested`) ends the turn with `_resume_after_yield = True`, so the nudge goes out at the **first stage-2 resume**, not "never". **Sim:** at 371 tok/s, 232 of 240 branch 0 are resumed at least once; **at 277 tok/s (the G4B end-to-end rate), only 151 of 240** (105 with a 24,576 chunk) | **Confirmed, and it matters at the realistic rate** | Two options, **the user decides** (a threshold change): (a) keep 27k and accept delivery at the first resume; (b) deliver it in the next **tool result** after the threshold (mid-turn), with the threshold below the pause (for example 24k). Under (b), 1 of the 2 G4B solved branches (b5ca7ac4_b0, transform at 26,228) would get it about 2k tokens early. Test: two requests 12,288 → 36,864 with no transform → the nudge is in the history before the pause |
| **H6-2** | G4-6 / G4-14: any solution-named kept function, even `transform(a, b)` or a helper `solve_row(r, i)`, suppresses the nudge and sets `first_transform_tokens` (A2 P1) | **Code:** the regex only (`_run_python_tool`); auto-check also requires `_r6_single_grid_arg` | **Confirmed** | Use the same `_r6_single_grid_arg` predicate. Test: `transform(a, b)` → nudge still at 27k; `transform(g)` → none |
| **H6-3** | The cache diagnostic runs **before** solving, so G4B starts from a churned radix / mamba cache, unlike the hidden rerun; SA-1 fixed only the time (all three) | **Code:** CELL 6 order is preflight → diag → solve | **Confirmed** (benchmark confound; no evidence it hurts the score) | Run the diagnostic **after** solving (commit mode, inside the profile's hard stop; skip it if under 300 s remain). Then SA-1's shift only covers the preflight. Test: the fake server's request order shows no `E12-CACHE-DIAG` before the first solver request |
| **H6-4** | A late straggler still writes `branch_result.json` after shutdown (A1, A2 reproduced) | **Code:** `BranchSession.run()` ends with `self.write()`; `publish(s.write())` writes before `publish` checks `closed` | **Confirmed** (artifact integrity). Submission and report use the in-memory records; the output bundle copies `branch_result.json` from disk. G4B had 0 stragglers | A freeze flag on every session at shutdown; `write()` returns the snapshot without touching disk once frozen. Test: stuck worker → file hash unchanged after the late return |
| **H6-5** | `smoke_rev5.py` / `smoke_adapter.py` exit 0 on a FAIL (A1) | **Code:** both end with only `print(...)`. `smoke_rev6.py` has `sys.exit`. My own runs were judged by the "ALL OK" line, so no result was misreported | **Confirmed** | `sys.exit(0 if ok else 1)` in both |
| **H6-6** | SA-1 shifts only gate 4b. EVAL12R6's 150-min solve window still loses the preflight + diagnostic time (A2) | **Code:** the shift is under `hidden_equiv` only; EVAL12R6 `solve_window_sec` = 9,000 | **Confirmed** (≤ 240 s of 9,000, 2.7 %; gone with H6-3 except the preflight) | Apply the shift to every commit profile with a solve window, still bounded by `HARD_STOP` |
| **H6-7** | G4-11: `... or len(g)*len(g[0]) <= 400: break` means the 12,000 bound is not enforced when every grid is ≤ 400 cells (A2, A3) | **Code: confirmed.** **Measured** on the 120 eval tasks with the Qwen tokenizer: p50 5,782 / p90 10,125 / **max 11,719, 0 over the bound**; 5 tasks hide a grid | **Confirmed defect, no observed case** | Drop the size condition: hide the largest grids until within the bound. Test: 10 train + 3 test 20×20 → ≤ 12,000 |
| **H6-8** | G4-12: a branch 1 already queued (class 1 at stage-1 end) still starts after branch 0 later improves (A3 MEDIUM) | **Code:** `defer_b1` is checked only when items are queued; the pop re-ranks but never re-checks it. For class 1 the order is `start1` before `resume0` | **Confirmed** (narrow: b0 must improve while b1 waits) | Re-check `defer_b1` when a `start1` is popped (re-queue after the next b0 round). Test: no progress → b1 queued → b0 improves → b1 deferred → b0 stalls → b1 runs |
| **H6-9** | `unique_verified_programs` key (task, branch, name, outputs) is not source identity; `verified_programs_per_task` counts distinct predictions (A2, A3) | **Code: confirmed both** | **Confirmed** (metrics only) | Add a closure-source hash (`_arc2_closure_key` exists) to the verdict; report programs and predictions separately |
| **H6-10** | G4-15 prefix match is message-level, not token-level (all three) | Known limitation (stated in the self-audit) | Agreed | Telemetry: add a hash of the canonical request payload (messages + tools + template kwargs), and keep "cached reported" separate. Token-level only for zero-hit samples, if at all |
| **H6-11** | G4-18 "same transform" = same name (all three) | Known | Agreed: no scheduler change | Log the source hash with the name (with H6-9) |
| **H6-12** | KV idle override is an exception to "no new task while KV ≥ 95 %" (A1, A3) | **Code:** `running == 0` → admit. `full_token_usage` excludes the evictable cache, so with 0 running it is about 0 on real data; the override fires only on a stuck or wrong metric | **Policy, the user decides** | Keep it (liveness) and report `kv_idle_overrides`, or go strict |
| **H6-13** | G4-5: `arc2_facts_extra` drops a line whose text already appears under another section (A2) | **Code:** `have = set(bounded lines)` ignores the section | **Confirmed** (low) | Compare (header, line) pairs |

**Also raised:**
- **Sim rate (A2):** the sim used 371 tok/s; G4B end-to-end was 277 (with the 12 → 6 bug).
  - **Re-run at 277:** still 240/240 exposed. Stage 1 now ends at 26.3k s (64 % of the window); 47.6k tokens per task on average; 169-171 tasks get < 60k in total.
  - So the 371 figure is optimistic, and coverage holds either way.
- **SA-1 test (A3):** the harness's 300 s cap dominates the 2,063 s window. Add a unit test at the real window.
- **Narrow labelled rows (A1, A3):** guard only `r N: 388883`-style rows with a row label (A1), or 6+ consecutive indexed rows (A3). Optional.
- **Isolation INCONCLUSIVE → 6 workers (A3):** the rev 6c C6-2 fail-safe. A bounded retry would be new; the user decides.
- **PARITY timeout / tool-name defects (A2):** still deferred (user: "1 to 5").

**Consolidated proposal (one release, for the user's decision):**
- **Must** before the G4B run: H6-1, H6-2, H6-3, H6-4, H6-5, H6-6.
- **Should** (cheap, tested): H6-7, H6-8, H6-9 + H6-11 (source hash), H6-13.
- **Telemetry:** H6-10.
- **Policy:** H6-12.
- **Not proposed:** a new allocation scheduler (all three: decide after G4B, from per-chunk marginal progress).

### Re-read of audits 1 and 2 (user re-pasted each one in full, 2026-10-22): additions to H6

| ID | From | Point | My check | Proposal |
|---|---|---|---|---|
| H6-1 (note) | A1 | "a pending-nudge flag, delivered at the next resume" | This is already how it works: the condition (≥ 27k, no transform, no check) persists until an opener is built. The real gap is a branch that is never resumed (sim at 277 tok/s: 89 of 240) | unchanged: options (a) / (b) |
| H6-3 (test) | A1, A2 | The rerun must never run the diagnostic. If the diagnostic moves after solving, its time comes from the hard-stop margin; the solve deadline must not grow | **Harness gap:** no check asserts `cache_diag` is absent in the rerun (the code has `not RERUN`) | add the rerun assertion + the request-order check |
| H6-4 (test) | A2 | 3 stuck workers; after 5 s the file hash is unchanged, the JSON is valid, the submission and the callback count are unchanged | — | use this as the H6-4 regression |
| **H6-14** | A1 Q1, A2 Q1 | The allocation decision needs **new host-verified distinct test predictions per 100k generated tokens, for all branches** (we report it for branch 1 only), plus the correct outputs per GPU-hour (analysis) and the per-run `first_uncached` | **Code:** only `b1_new_distinct_verified_per_100k_tokens` exists | report metric (overall, b0, b1; stage 1 vs stage 2) |
| **H6-15** | A1 Q2 (yes), A2 / A3 Q2 (optional) | Narrow rows: guard only **labelled** rows (`r2: 388883`, `row 2: 388883`) in runs of ≥ 6, or ≥ 6 consecutive increasing indices of the same width; bare numbers never | residual 438 rows (149 transcripts): 'r9: 9' 96, '9 9' 201, '9' 127 | optional; the labelled form is low-risk |

**Verdict differences:**
- **PARITY:** A1 says GO for diagnostics only; A2 says NO-GO (old timeout / tool-name defects).
- **EVAL12R6:** A1 also wants real 12-worker KV and throughput evidence first, which the G4B run gives.
- **Audit 3:** its text was cut off at Q6. It is validated up to there; the rest is still awaited.

### Deep re-check of all three audits (user 2026-10-22: "look at the GPT fixes deeply once more, add anything left out, verify the truth")

**Method:**
- Every reproducible claim was re-run, not only read: `e12_smoke/verify_h6.py` (output `self_audit/verify_h6_out.txt`) and `self_audit/h6_1_openers_g4b_out.txt` (the real G4B transcripts).
- Numbers were recomputed.
- **Result:** two of my earlier readings were wrong (H6-8, H6-13; corrected below). The audits' numbers that could be checked are right.

#### Reproductions

| ID | Claim | Reproduction | Result |
|---|---|---|---|
| **H6-1** | The 27k nudge is not delivered in stage 1 | G4B rev 6d transcripts: the generated tokens at every new turn (opener = `[USER PROMPT]`). Openers per branch p50 4. **0 of 12 branch-0 runs had any opener at ≥ 27k before their stage-1 pause.** The last stage-1 openers: 135a2760 16,039; 136b0064 17,785; 4c3d4a41 17,890; 9385bd28 20,575; 62593bfd 24,419; 898e7135 25,244; 65b59efc 26,131; b5ca7ac4 26,609; 247ef758 26,962 | **Confirmed with real data.** An opener-based nudge at 27k reaches the model only at a stage-2 resume. A lower threshold that still rides an opener (for example 24k) would reach 5 of 12, by chance. **Only mid-turn delivery (inside a tool result) is reliable**: tool results come every few k tokens |
| **H6-2** | `transform(a, b)` / a helper `solve_row(r, i)` suppress the nudge | Real agent and sandbox: both set `first_transform_tokens`; at 27.5k the opener is "Continue…", not the nudge; `_r6_single_grid_arg` = False | **Reproduced** |
| **H6-4** | A straggler rewrites `branch_result.json` after the scheduler returned | Real `solve_all_r6`, a hanging agent: stragglers 1; file 399 bytes / 1,000 tokens at return → 544 bytes / 2,000 tokens later. The published record stayed at 1,000 | **Reproduced** (A2's 395 → 543 bytes) |
| **H6-7** | The 12,000-token indexed-view bound fails when every grid is ≤ 400 cells | 20×20 grids: **10 train + 3 test = 21,963 tokens (Qwen tokenizer) / 12,073 (fallback, = A3 exactly); 12 + 1 = 23,879 / 13,136 (= A2 exactly)**, 0 grids hidden. The 120 eval tasks: max 11,719, none over | **Reproduced.** The real overflow is 1.8× the bound, larger than the audits' fallback numbers. Never seen on eval |
| **H6-8** | A queued branch 1 starts after branch 0 already improved | Real scheduler, 6 fake tasks (no progress in stage 1, progress in stage 2): 1 and 2 workers: none; **4 workers: 1 of 6 (t0)** | **Reproduced (rare).** **Correction:** from the queue order (b1's `start1` sorts before b0's `resume0`) I had argued it could not happen; with several workers it does |
| **H6-13** | `arc_overview` extras drop a line whose text appears under another header | 120 eval tasks: every (header, line) of the full facts is in the bounded block or the extras: **0 lost lines** | **Not reproduced.** The code pattern exists but never fires on eval, so it is **downgraded to "no change"** |
| H6-5 | The smoke scripts exit 0 on a FAIL | Code: `smoke_rev5.py` and `smoke_adapter.py` end without `sys.exit` | Confirmed (a code fact) |
| — | A3: adapter sha `02ac03f2eeaa4b09…`; rev 6d `7c60454c…`; vendor 22/22 unchanged | sha256 of the file | **Confirmed** |
| — | A2: 572,035 / 2,063 = 277.3 tok/s → 47.7k per task over 41,257 s; 371 → 63.8k | arithmetic | **Confirmed** |
| — | A1: cache 84.26 / 88.70 / 74.99 %, zero-cache 6/127, 23/449, 20/180 | = our G4B validation | **Confirmed** |

#### Items that were left out of H6 (added now)

| ID | From | Point | Check | Proposal |
|---|---|---|---|---|
| **H6-16** | A3 §4 | The SA-1 harness test is dominated by the harness's 300 s cap, so the real 2,063 s gate-4b window is never exercised | Confirmed (the check uses `min(300, window)`) | A unit test of the CELL 6 window logic at the real values (hidden end 42,720, start 1,463, 12 tasks, a mocked 240 s diagnostic): solve deadline = start + 2,063 s after the preflight, bounded by `HARD_STOP` |
| **H6-17** | A3 §4 | An isolation INCONCLUSIVE (a missing marker, no leak) drops the run to 6 workers; a bounded retry would avoid a needless drop | Code: `workers_for_isolation` (rev 6c C6-2). G4B: PASS | Optional, **the user decides**: one retry of the isolation preflight on INCONCLUSIVE; FAIL still drops to 6 |
| **H6-18** | A2 Q3, A3 Q3, A1 Q3 | Nudge efficacy cannot be judged: the report has only a nudge count | Code: `provisional_nudges` only | Per branch: the nudge tokens, the first one-grid transform after it, the first check after it, tokens from nudge to first check, accepted. Report: medians for nudged vs not-nudged branches |
| **H6-19** | all three (none could reproduce the suites) | `smoke_rev6.py` imports `tokenizers` at module level in the G4-4 block, so without the package every later test is lost (A2: 36 passed, then stop). The bundle has no dependency list | Confirmed (`from tokenizers import Tokenizer` at the G4-4 test) | The tokenizer-dependent tests print an explicit `SKIP (tokenizers missing)` and the rest run; the final line shows passed / failed / skipped. Add `tests/requirements.txt` (tokenizers 0.23.2, transformers 5.20.2). A missing dependency must never read as ALL OK |
| H6-14 (more) | A3 Q1, A2 Q1 | Per stage-2 run: the new distinct verified predictions it produced (the runs already record tokens, best before / after, improved, `first_uncached`) | Code: missing in `runs` | add `new_verified` per run |
| H6-12 (more) | A1 E | Report `kv_idle_overrides` separately | Code: in `SCHED_LAST`, not in the coverage print | add it to the coverage line |
| H6-2 (test) | A2 | Redefining an unchanged function must not reset the first-transform timestamp; the metric means "the first retained one-grid solution-named function", not the first useful candidate (A3 G4-14) | — | the test + the definition in the report |

#### Updated proposal (for the user's decision)

- **Must, before G4B:**
  - H6-1: option (b), **mid-turn delivery** in the next tool result. The G4B data show an opener-based nudge never reaches stage 1. The threshold stays the user's choice: 27k still mostly lands after the 28k pause, so (b) needs a threshold below it.
  - H6-2, H6-3, H6-4, H6-5, H6-6, H6-19.
- **Should:** H6-7, H6-8, H6-9 + H6-11, H6-14 (+ per-run `new_verified`), H6-16, H6-18.
- **Telemetry:** H6-10.
- **Policy, the user decides:** H6-12 (+ print), H6-17.
- **No change:** H6-13 (not reproduced). H6-15 stays optional.

## Rev 6f: the H6 set built (2026-10-22, user: "Ok, update start")

**Taken as recommended:**
- H6-1 delivered in the tool result at 24,000;
- H6-12 kept and printed;
- H6-17 one retry;
- all must / should / telemetry items;
- H6-13 no change; H6-15 not built (optional).

| ID | Built as | Test |
|---|---|---|
| **H6-1** | The provisional nudge goes into the **next tool result** (`harness_note`) once a branch has generated ≥ `E12_R6_NUDGE_TOKENS` **24,000** with no one-grid solution function and no check; else the next opener, at most once; `provisional_nudge_via` = tool / opener. **G4B replay** (`self_audit/h6_1_toolresult_g4b_out.txt`), branch 0, 12 runs: 7 would get it inside stage 1, of which 4 are 3.1-9.9k tokens before the pause (247ef758, 62593bfd, 65b59efc, b5ca7ac4) and 3 at the pause request. 3 had a transform first. 2 (136b0064, f931b4a8) had no tool call between 24k and the pause, one long think, so nothing mid-turn reaches them. The opener path reached 0 of 12 | smoke: no note at 23k; note at 24.5k (via tool); once; the 36,864 request of audit 1's timeline carries it; a one-grid transform suppresses it |
| **H6-2** | `first_transform_tokens` and the nudge use the auto-check predicate `_r6_single_grid_arg` | smoke: `transform(a, b)` and `solve_row(r, i)` do not count, so the nudge still goes out; an unchanged redefinition keeps the timestamp |
| **H6-3** | The cache diagnostic runs **after solving** (commit mode; its time comes from what is left before the output margin; skipped under 90 s; `cache_diag.ran = "after solving"`) | harness: every diagnostic request comes after the last solver request; the rerun never runs it |
| **H6-4** | `BranchSession`: an I/O lock, an atomic write (temp + replace), `freeze()` after the scheduler's final snapshot; later writes by a straggler are no-ops (record and file) | smoke: 3 stuck workers, 5 s after the return every `branch_result.json` hash is unchanged, valid JSON, and the published records and callback count are unchanged |
| **H6-5** | `smoke_adapter.py` / `smoke_rev5.py` / `smoke_rev6.py` share `finish()`: FAIL → exit 1, skipped → exit 2 ("INCOMPLETE"), else exit 0 | the runs below |
| **H6-6** | Every commit profile with a solve window (and gate 4b) starts its window after the preflight (`_e12_shifted_end`, bounded by the hard stop; `e12_run.window_shift`); the hidden rerun keeps its absolute end | harness r6 / g4b: shift recorded, the full window left at the solve start |
| **H6-7** | `arc2_indexed_grids`: grids are hidden largest first until within the bound; small grids are no longer exempt | smoke: 10 train + 3 test 20×20 → ≤ 12,000 tokens, every grid listed; a small task is unchanged. The 120 eval tasks are unchanged (max 11,719) |
| **H6-8** | `g412_defer_b1()`, one predicate used when items are queued and again when a queued branch-1 start is popped (only while branch 0 is queued or running, so branch 1 is never starved); `b1_deferred_at_pop` in the scheduler report | smoke: the predicate (improving → defer; stalled / no progress / done / verified → start); the scheduler calls it at both places; the G4-12 sims still pass |
| **H6-9 / H6-11** | Every verdict carries `src_key` (source + transitive dependencies). Report: `verified_programs_by_source`, `verified_programs_per_task` (by source), `distinct_predictions_per_task` (the old field's meaning); `progress_same_name_new_source`; the best progress records its `src` | smoke; harness |
| **H6-10** | Per request: `payload_sha`, `chars`, `prefix_chars` (serialized characters shared with the previous request), `tools_same`, `template_same`. Report: zero-hit later requests with a full character prefix and with tools / template changed; `cached_not_reported_requests` | smoke (3 requests, a tools change flagged); harness |
| **H6-12** | `kv_idle_overrides` in the coverage line | harness |
| **H6-14** | Each run records `new_verified`. Report `verified_yield` per 100k generated tokens: all / b0 / b1 / stage 1 / stage 2 | smoke; harness |
| **H6-16** | `_e12_g4b_window` and `_e12_shifted_end` are functions; `e12_smoke/window_check.py` reads them from the built notebook and checks the real values: 42,720 / 1,463 / 12 → 171.9 s per task and a 2,063 s window; the full window after a 35 s or 240 s setup; never past the hard stop; EVAL12R6 9,000 s | 7/7 |
| **H6-17** | Isolation INCONCLUSIVE → one retry of the preflight before the 6-stream fail-safe (`retry_of` recorded); a FAIL never retries | code; harness (PASS path) |
| **H6-18** | Per branch: `first_transform_after_nudge`, `first_check_after_nudge`. Report `nudge`: branches, via, tokens p50, transform / check after, and tokens to each (p50), accepted | smoke; harness |
| **H6-19** | Tokenizer-dependent checks use `expect_tok` and print `SKIP … (tokenizers not installed)`; the rest still run. **Simulated without the package** (a blocking `tokenizers` on `PYTHONPATH`): smoke_rev6 161 passed + 1 skipped, exit 2; smoke_rev5 52 + 4 skipped, exit 2 (2 B3 estimate checks found to need the tokenizer were converted too); smoke_adapter 41, exit 0. The bundle gets `tests/requirements.txt` | as stated |

**Validation (final files, 2026-10-22):**
- **Unit and regression:**
  - smoke_rev6 **162/162** (143 → 162);
  - rev5 56/56; adapter 41/41;
  - `regex_check` ALL OK; `kv_gate_replay` 7/7; `window_check` 7/7;
  - facts size unchanged (p50 699 / p90 779 / max 798).
- **Harness on the final notebooks:** g4b **57/57**, r6 **55/55**, rerun 23/23, launchfail 6/6, parity 19/19 (`e12_harness/h16_*.txt`).
  - My first harness version read `requests` / `checked` from the report's branch rows, which do not hold them, giving 2 FAILs. Fixed to read the raw `branch_result.json` files.
- **Replays through the rev 6f agent:** rev 4 7/8, rev 5 xhigh 5/8, low 10/19, identical per task to the self-audit replays.
- **Notebooks:**
  - EVAL12R6 `344a516c…`, EVAL12R6G4B_REBUILD `b98b020f…`, PARITYR6 `530bbc93…`;
  - the embedded adapter == source in all 3 (23 files);
  - all 9 Kaggle-run notebooks present and unchanged.
- **Backups:** `_scratch/session_1c6db7aa/rev6f_backup/` (adapter, builder, smoke ×3, harness, fake server before rev 6f; the patch scripts).

**Not done:** GPU run (needs the user's go and a pre-run capacity check).

### Correction (2026-10-22, while validating rev 6f): H6-8 was not reproduced

- **What I claimed:** in the deep re-check I marked H6-8 "Reproduced (rare, 1 of 6 at 4 workers)".
- **Why it was wrong:** the measurement compared the request-log order. A branch 1 popped **before** branch 0 improved logs its first request 3-73 ms after the pop (thread start-up), so the comparison gave false positives.
- **The correct measurement:** the real pop time. A branch-1 session is created under the scheduler lock when its start is popped.
  - `verify_h6.py` (now pop-time, 7 scheduler runs) and `verify_h6_8_precise.py` (4 more runs × 5 configurations): **0 cases on rev 6e and 0 on rev 6f**.
  - The rev 6f pop-time deferral never triggered (`b1_deferred_at_pop` 0).
- **Status:** H6-8 = **not reproduced; the fix stays as a defensive change.** The code path is possible in principle (branch 0 must improve while its branch 1 waits behind other items); the G4-12 tests and both sims still pass.

**`verify_h6.py` on both versions** (`self_audit/verify_h6_rev6e_out.txt`, `verify_h6_rev6f_out.txt`):

| | H6-2 | H6-4 | H6-7 | H6-8 | H6-13 |
|---|---|---|---|---|---|
| rev 6e | reproduced | reproduced | reproduced | not reproduced | not reproduced |
| rev 6f | not reproduced | not reproduced | not reproduced | not reproduced | not reproduced |

## Rev 6f self-audit (2026-10-22, user: "audit it yourself: does every fix work, is every fix done, has any earlier bug come back; run every kind of test")

### 1. Every fix-list ID vs the tests

- **Scale:** 126 IDs in the fix list; 44 are not named in any test. Each was checked against its row.
- **Aliases** of a tested item:
  - N1–N7 and C1–C9 were merged into R6-*;
  - R6-A..J were proposal letters mapped to R6-V/P/K/Q/X;
  - T3 → K1, T4 → V4, T5 → K3 tests;
  - G4-13 → the harness G4-7/13 check;
  - SA-1 → window_check + harness; SA-2 → G4-3+.
- **Not built by decision:** C10, R5-6, G4-9, H6-15, and the PARITY items C6-1 / C6-3 (deferred, user "1 to 5"). C6-6 was a wording correction only.
- **Real gaps, now tested:**
  - **R6-P3** (the generalization note once): no test existed. Added. My first version used a second candidate with identical outputs, a duplicate that never gets the note, so the mutation run showed it could not fail. Rewritten with a second distinct candidate.
  - **H6-12** (idle overrides printed): the harness now captures CELL 7's printed lines and checks the coverage line.
  - **H6-17** (isolation retry): the fake server can drop one preflight reply once (`FAKE_PREFLIGHT_DROP_ONCE`). The harness r6 run sees INCONCLUSIVE → retry → PASS → 12 workers, `retry_of` recorded.

### 2. Static analysis

- **pyflakes:**
  - adapter: clean;
  - builder, harness, fake server, window_check: no undefined names;
  - the code of all three built notebooks (2,023 lines each): no undefined names, 2 unused imports;
  - smoke scripts: every undefined name is defined by the shared preamble they exec.
- **nbformat:** all three notebooks validate; every code cell compiles.
  - Warning: cells have no `id` field (nbformat 5.11 "will become a hard error in future versions"). Kaggle accepts this; the rev 6d G4B ran with the same format.
- **Stale text in the notebooks** (fixed):
  - the markdown header said "rev 6e (G4-1..G4-18)";
  - a CELL 3 comment said "provisional nudge at 27,000";
  - two adapter comments still said H6-8 was "reproduced with 4 workers" (it was not; see the H6-8 correction).

### 3. Mutation testing (does each guard test really fail when its fix is undone?)

**Adapter** (`audit6f/mutation_run.py`, a copy of the adapter per mutation; output `audit6f/mutation_out.txt`): **15 of 15 mutations caught** (R6-P3 after its test was rewritten).

| Mutation (the old bug put back) | Guard that failed |
|---|---|
| KV gate removed (G4-1) | G4-1 KV 0.97 held; release |
| Stale KV sample treated as fresh (G4-1) | G4-1 stale sample |
| Print guard reverted to rev 6d (G4-3 / SA-2) | G4-3 print(i, row), G4-3+ joined digits |
| Tool-result nudge removed (H6-1) | H6-1 past 24,000 / once / 36,864 timeline |
| Nudge threshold back to 27k (G4-6) | H6-1 + G4-6 |
| One-grid predicate removed (H6-2) | H6-2 (3 checks) |
| Freeze ignored (H6-4) | H6-4 three stuck workers |
| Small grids exempt again (H6-7) | H6-7 |
| Pop-time check removed (H6-8) | H6-8 scheduler structure |
| G4-12 deferral removed | G4-12 improving branch 0; H6-8 predicate |
| Per-request deadline clamp removed (T8) | T8 (2 checks) |
| Branch-1 first cap back to 12,288 (G4-2) | G4-2 |
| src_key removed (H6-9) | H6-9, H6-11 |
| Cut call executed again (R5-2) | R5-2 (2 checks) |
| Generalization note every time (R6-P3) | R6-P3 (rewritten test) |

**Builder** (`audit6f/mutation_nb_run.py`, mutated copies of the built EVAL12R6 notebook, harness r6): the clean copy passes 57/57, and all 3 mutations were caught.
- **Diagnostic before solving (H6-3):** caught by H6-3 and G4-8.
- **No window shift (H6-6):** caught by H6-6. The check had been too weak: the fake preflight was instant, so the shift was about 0. The fake preflight now takes about 2 s, and the check needs a shift ≥ 3 s and the full window ±1 s.
- **No isolation retry (H6-17):** caught by H6-17, R6-X1, C6-2 and the 12-worker check.

### 4. Earlier bugs: none came back

Each has a regression test that passes on the final files:

| Earlier bug | Test |
|---|---|
| Regex hang | `regex_check` linear time; V4 near-row line |
| Rejected-submit opener | P2 |
| A branch driven by two workers | S3 |
| Starvation | S3 fair share / round robin |
| Facts size | Q-5, `facts_size` |
| T8 deadline overrun | T8 / S6 / P4 |
| Cut call executed | R5-2 |
| rev 6d probe step-down | G4-1 (no step-down code, the steady state with queue 1) |
| Guard leaks | G4-3 / G4-3+ |
| Double-counted accepts | harness G4-7/13 |
| null cache = 0 | G4-15 |
| Straggler rewrite | C6-4 + H6-4 |
| Rebuilt Kaggle-run notebooks | the 9 shas unchanged |

### 5. Full run on the final files

- **Notebooks:** EVAL12R6 `ac7a46a8…`, EVAL12R6G4B_REBUILD `913aef97…`, PARITYR6 `7d35d72b…`.
- **Harness:** g4b **58/58**, r6 **57/57**, rerun 23/23, launchfail 6/6, parity 19/19.
- **Smoke with tokenizers:** rev6 **163/163**, rev5 56/56, adapter 41/41.
- **Smoke without tokenizers:**
  - rev6 162 + 1 skipped (exit 2);
  - rev5 52 + 4 skipped (exit 2);
  - adapter 41 (exit 0).
  - **A timing flake:** one of three no-tokenizer runs failed "G4-1: KV 0.97 … dH < 3.0" with 3.22 s; the other two passed. The bound guards against a deadlock, which would take ≥ 60 s, so it was widened to 2 + 2 s.
- **Other suites:**
  - `regex_check` ALL OK; `kv_gate_replay` 7/7; `window_check` 7/7;
  - facts size unchanged (p50 699 / p90 779 / max 798);
  - `verify_h6`: 6 of 6 NOT REPRODUCED;
  - replays 7/8, 5/8, 10/19, identical per task.
- **240-task sim** (rev 6f code):
  - 371 tok/s: 240/240 exposed, stage 1 ends at 19,990 s, KV max 0.50, 0 holds, 232/240 branch 0 resumed;
  - 277 tok/s: 240/240 exposed, stage 1 ends at 26,761 s, KV max 0.40, 0 holds.
- **Packaging:** the embedded adapter == source in all 3 notebooks; all 9 Kaggle-run notebooks are unchanged.

## GPT audits of the rev 6f bundle (2026-10-22, user-pasted; more coming): validated, items H7-1..

### Audit 1

**Verdicts:**
- G4B / EVAL12R6: NO-GO as is, GO after the H6-4 callback fix; PARITY GO (diagnostic only).
- **Confirmed by the auditor:** the shas 3/3; the packages 23/23; KV replay 7/7; window_check 7/7; regex; smoke_adapter; rev5 52 + 4 skipped (exit 2); rev6 162 + 1 skipped (exit 2); historical gold (rev 5 5/8 and 10/19, rev 6d 2/19).
- **Not run by the auditor:** the full suites and the harness (no network for `pip`).

**Validation:**
- reproductions in `e12_smoke/verify_h7.py` (output `self_audit/verify_h7_out.txt`);
- code reads.

| ID | Claim | My check | Verdict | Fix (proposed; nothing built) |
|---|---|---|---|---|
| **H7-1** | **H6-4 callback race (audit: HIGH, the only P0):** `publish()` runs `on_branch_done` outside the lock, so an older callback can finish after the final one, and the notebook's `_e12_branch_done_r6` replaces by arrival order. Their run: older 1,000 tokens after the final 2,000 → the outside record = 1,000 | **Ordering: reproduced.** A slow callback (the worker becomes a straggler inside it) finishes after the scheduler's final callback (`arrivals [1000, 1000]`, stragglers 1). **Content: not reproduced in the production flows.** (a) The late record equals the final one: same tokens, status, runs, candidates, progress. (b) A straggler run that returns during finalization leaves no older record (outside 2,000 = scheduler 2,000). **Why:** a branch is resumed only after its worker's `publish()` and callback return (`stage2_items` runs after `publish`), and the final snapshot is taken from an idle agent. So the two snapshots that can race carry the same progress. Their 1,000 vs 2,000 needs the agent to generate between the two, which the production flow does not allow (their fake setup is not in the bundle) | **Ordering race confirmed; stale data not reproduced** (the submission cannot lose a candidate this way) | **Cheap and defensive, recommended:** a monotonic `publish_seq` set under the lock; `_e12_branch_done_r6` drops a record whose seq is ≤ the stored one. Test: hold an older callback, let the final one finish, release it → the stored record and `submission.json` are unchanged |
| **H7-2** | **H6-11:** after a worse same-name redefinition, the best score stays but `best["src"]` is overwritten with the worse source | **Reproduced** (real agent): v1 1/2 (src `e18df2bc…`) → worse v2 → best still 1/2 but src `58a4015f…` (v2) | **Confirmed** (diagnostics only; selection uses host verdicts) | Set `src` only when this call changed the best (a new best dict from `_arc2_note_progress`). Test: better → worse → better under one name: score and src follow v1, then v3 |
| **H7-3** | **H6-14:** `new_verified` is branch-local, so b0 and b1 with the same output count twice in `verified_yield.all` | **Code-confirmed** (`verified_signatures()` is per branch); the task-global count exists separately (`distinct_predictions_per_task`) | **Confirmed** (metric meaning) | Rename to `new_in_branch`; add a task-global `new_for_task` (first discovery of each (task, outputs) signature). Test: identical b0 / b1 outputs → branch-local 2, task-global 1 |
| **H7-4** | **H6-3 edge:** the post-solve diagnostic does not check stragglers, so a straggler's request could overlap it | **Code-confirmed.** G4B had 0 stragglers | **Confirmed** (low; it only affects the diagnostic's validity) | Skip the diagnostic when `stragglers > 0` (recorded). Test: stragglers 1 → no diagnostic request |
| **H7-5** | **H6-9:** `unique_verified_programs` is still name / output based next to the new `verified_programs_by_source` | **Code-confirmed** (kept for continuity with rev 6e) | **Confirmed** (naming) | Rename the old field (`accepted_name_output_pairs`) or document it in the report; keep `verified_programs_by_source` as the program count |
| **H7-6** | `verify_h6.py` stops at the `tokenizers` import (auditor's run) | **Reproduced:** with the package blocked, an `ImportError` at line 57 (the H6-7 part), so the remaining reproductions do not run | **Confirmed** (an H6-19 gap I missed in the self-audit: only the smoke suites were made skip-aware) | Skip the tokenizer half of H6-7 visibly (the fallback half still runs); the final line says INCOMPLETE |
| — | Q1: a mid-request nudge is impossible in the non-streaming setup; keep the nudge as is; judge it by first transform / check after the nudge, not by the count | Agree (Duck sends complete requests) | — | none |
| — | Q2: H6-8 possible interleaving (b0 runs again before b1 is admitted); no starvation path | Agree; matches our pop-time measurements (0 cases) | keep H6-8 | none |
| — | Q4 decision order for the G4B results: correct outputs → distinct verified predictions per task → first transform / check timing → stage-2 progress per extra 12k → cache / throughput / KV. One run cannot give a causal 12k vs 24k winner | Agree. With H7-3, use the task-global count | — | none |
| — | All three auditors so far ran without `tokenizers` / `transformers` (no network) | — | — | **Option (user):** ship an offline wheel for `tokenizers` (abi3 Linux, a few MB) in the bundle so reviewers can run the full smoke suites; `transformers` (harness only) is heavier |

**Audit 1 consolidated (for the user's decision; nothing built):**
- **H7-1:** `publish_seq`. The audit's P0; my rating is defensive and recommended, since no data loss was reproduced.
- **H7-2, H7-3, H7-6** (should).
- **H7-4, H7-5** (low).

### Audit 2

**Verdicts:**
- G4B: GO (with real-request coverage logged).
- EVAL12R6: GO after the G4B review.
- PARITY: NO-GO (the old deferred bugs).
- Hidden rerun: not yet GO.
- **No new mandatory blocker;** the callback guard is optional hardening.

**Confirmed by the auditor:**
- the shas, the 23-file packages and the embedded adapter;
- 8/8 cells compile; window 7/7;
- the nudge replay (7/12, 4/12);
- 101 smoke checks (then a tool timeout in their environment).

**New or different points vs audit 1** (all checked):

| ID | Claim | My check | Verdict | Fix (proposed; nothing built) |
|---|---|---|---|---|
| = H7-3 | F6-1: `verified_yield` over-counts task-level novelty (b0 and b1 with the same output) | = H7-3, code-confirmed | Confirmed | as H7-3; for "first occurrence" across concurrent branches use the host-verification timestamp (`v["t"]`) |
| **H7-7** | F6-2: "tasks exposed" counts initialized sessions, not model requests: `SCHED_LAST.tasks_exposed` = branch 0 with an agent object, created in `_ensure()` before any request | **Reproduced:** a `BranchSession` past its deadline creates its agent, makes 0 analyze calls and generates 0 tokens, yet the scheduler's rule counts it. **But the report already has the response-based count:** `rev6.metrics.tasks_exposed` = branch 0 with generated tokens > 0, and `tasks_meaningful` = a first tool call. The two fields share a name with different meanings, and the printed coverage line uses the scheduler's. In practice an admission needs ≥ `min_admit_sec` (240 s) left, so a zero-request exposure needs a failing request | **Confirmed** (naming / accounting, not a run failure) | Scheduler: `tasks_initialized` + `tasks_with_request_sent` (a request started: `arc2_requests` non-empty). Report: `tasks_with_response` (= today's metric). The coverage line prints all three, and "240/240" is claimed on the request-sent or response count. Test: a deadline-expired session → initialized 1, sent 0 |
| — | F6-2 nuance: `stage1_left` falls at pop time, so the first stage-2 items can start while the last stage-1 runs are still running | **Code-confirmed** (`state["stage1_left"] -= 1` at the pop). It is the intended S1 rule (tested: "no stage-2 work before the last task's first exposure **starts**") | By design | none (document it) |
| = H7-1 | F6-3: callback ordering guard missing; the auditor's counterexample feeds the callback in isolation (200 / accepted, then 100 / not accepted → 100 kept) and agrees that the production flow (one driver per branch; the next run only after the callback) does not lose a candidate | Matches my H7-1 measurement (ordering reproduced, content identical in the production flows) | Same verdict as H7-1 | as H7-1 (optional hardening) |
| **H7-8** | H6-7 edge: when hiding every grid still leaves the view over the bound, nothing says so | **Code-confirmed:** the loop ends without a flag. It needs about 600 grids (a pointer line is about 20 tokens), not seen in eval | Confirmed (negligible) | Record `view_over_target` in the agent events, as the facts block does (`facts_over_target`) |
| **H7-9** | Q2: add a deterministic injected-state regression for H6-8 (pop-time defer → branch 0 finishes → branch 1 re-queued → eventually admitted) | **Gap confirmed.** We have the predicate unit test and the structural test; my attempt at a deterministic scheduler scenario failed because the fake fair share produces large chunks (see the H6-8 correction) | Confirmed test gap | A scheduler test with `fair_share=False` and an injected pop: the item is deferred while branch 0 is queued, then re-queued by `stage2_items` after branch 0's run, then admitted |
| = known | F6-4: PARITY `urlopen(timeout=900)` × 24 probes and the tool-name omission | = C6-1 / C6-3, deferred (user "1 to 5") | Known | only if PARITY is to run |

**Numbers re-checked:**
- 42,720 − 1,463 = 41,257 → × 12 / 240 = 2,062.85 s;
- 171.9 s per task;
- nudge replay 7/12 and 4/12;
- sim 277 → about 47.6k per task, 371 → about 63k.

All match.

**After audits 1 + 2:**
- **H7-1:** two audits rate it optional or defensive (audit 1 rated it P0 from a flow production does not allow). My recommendation: build it, because it is cheap.
- **Should:** H7-2, H7-3, H7-6, H7-7.
- **Low:** H7-4, H7-5, H7-8.
- **Test gap:** H7-9.

### Audit 3

**Verdicts:**
- G4B: Conditional GO (add the nudge-delivery metric and the freeze test; rerun the full suites with dependencies).
- EVAL12R6: Conditional GO after G4B.
- PARITY: GO (diagnostic only).
- **Confirmed by the auditor:** the shas 3/3; the packages 23/23 (manifest entries matched); the vendor unchanged; KV replay 7/7; window 7/7; regex; smoke_adapter ALL OK; rev5 / rev6 exit 2 with skips.
- **Stopped in their environment:** `facts_size.py` (tokenizers) and the harness (transformers).

| ID | Claim | My check | Verdict | Fix (proposed; nothing built) |
|---|---|---|---|---|
| **H7-10** | **Finding 1: "emitted" is not "seen".** In 3 of the 7 G4B replays (4c3d4a41_b0 33,107; 9385bd28_b0 39,353; cbebaa4b_b0 38,432) the note lands in the tool result of the request that triggers the pause, so the model sees it only if the branch is resumed. The report measures emission, not delivery | **Confirmed:** the three tasks and token counts equal our replay's "0 tokens before the pause" rows (`h6_1_g4b_nudge_delivery_out.txt`). The code has no delivered / seen field (`provisional_nudge_via` is set at emission). A trim before the resume could also drop that tool result | **Confirmed** (measurement; not a correctness bug) | `provisional_nudge_seen_request`: in `_chat_completion`, the first outbound request whose messages contain the nudge text records its index and tokens. Report: emitted / seen / effective (a transform or check after it). Test: a note emitted at the pause boundary → seen None; after the resume, the first request records it |
| **H7-11** | **Finding 2: `freeze()` is not atomic.** `write()` (lock) and then a deep copy (lock again) are two critical sections, so a straggler that mutates in between makes the disk file and `_final` differ | **Reproduced:** (1) deterministic interleaving: disk usage 1 vs `_final` 2. (2) A threaded live agent: **60 of 60 freezes** gave a disk / `_final` mismatch; 0 write errors. **A second cause found while checking:** `write()` stores **references** to the agent's live lists and dicts (`usage`, `checked`, `rev6_events` …), so a running agent also changes `self.record` itself. It needs a straggler that is still running at finalization (G4B: 0 stragglers). The published final (`_final`, deep-copied) stays stable once frozen; the mismatch is disk vs published | **Confirmed** (artifact consistency; the submission uses the published record) | `freeze()`: under one lock, build a deep-copied snapshot from the agent, write that same snapshot to disk, set `_final`. `write()` copies the agent's lists (`list(...)`, `copy.deepcopy`) instead of referencing them. Test: a barrier between snapshot and write, plus a mutating agent → disk == `_final` == the callback record |
| = H7-5 | Finding 3: two program-count fields with different keys | = H7-5 | Confirmed | as H7-5 (document `unique_verified_programs` as name / output candidates) |
| = H6-10 | Finding 4: the character prefix is not the token prefix; report it as `serialized_char_prefix`, never as proof of cache preservation or eviction | Known (the H6-10 row says "proxy") | Agree | reporting wording only: rename `prefix_chars` → `serialized_char_prefix` in the report and the README |
| = H7-6 (wider) | `facts_size.py` (and the harness) stop without `tokenizers` / `transformers` | **Reproduced:** `facts_size.py` with the package blocked → `ImportError` | Confirmed | as H7-6: `verify_h6.py` **and** `facts_size.py` say INCOMPLETE instead of crashing; the harness prints a clear "needs transformers" exit |
| — | Q1: the nudge must not tell the model to drop a hypothesis | Checked the text (`R6_TEXT["provisional"]`): "write one provisional `transform(grid)` now … use `check(transform)` to locate what fails …". It does not ask to abandon anything | Already satisfied | none |
| — | Q4: never use verified-candidate counts as a score substitute (train-exact but test-wrong rises too) | Agree | — | — |
| — | Rev 5 / R6 core unchanged (verifier, harvest, pass@2, caps, 28k, 12,288, deadline clamp, vendor) | Agree | — | — |

**After audits 1 + 2 + 3 (for the user's decision; nothing built):**
- **Consistency fixes** (cheap, recommended before G4B):
  - H7-1 `publish_seq` (callback order);
  - H7-11 atomic freeze with copied agent state;
  - H7-10 nudge seen-by-model.
- **Metrics and naming:** H7-2, H7-3, H7-5, H7-7, H7-8; the H6-10 wording.
- **Tests:** H7-6 (skip-aware `verify_h6` / `facts_size`, a clear harness message), H7-9 (deterministic H6-8 scenario).
- **Low:** H7-4.
- **Option:** an offline `tokenizers` wheel in the bundle (all three auditors had no network).

### Re-read of all three rev 6f audits (user 2026-10-22: "read the GPT texts again, check nothing was left out, validate again, then organize the fix list")

**Method:**
- I re-read each audit claim by claim and mapped every finding, recommended test and decision rule to an H7 item.
- I re-ran the reproductions on the current code (unchanged since rev 6f):
  - `verify_h7.py`: H6-4 ordering yes, content no (a) and (b); H6-11 yes; H6-14 yes;
  - `verify_h7_freeze.py`: (1) disk 1 vs `_final` 2; (2) 60 / 60;
  - `verify_h6.py` and `facts_size.py` with `tokenizers` blocked: `ImportError` (exit 1).
- Every result matches the earlier run.

**Left out before (now added):**
- **H7-12 (A2 Q4):** "actual correct outputs per GPU-hour" and task-level yield per GPU-hour. The report has the parts (eval hits, start / finish seconds) but no derived field. Code-confirmed: no such field.
- **H7-13 (A3 Finding 4 test):** a per-request classification of zero-hit requests. "Serialized prefix full + tools and template unchanged + not trimmed" → `unexplained_cache_miss`; else `prompt_changed` (with the reason). Code-confirmed: today there are only separate counters.
- **The H7-5 test (A3):** same name + new source + same outputs; same source in two branches. Both expected counts asserted.
- **The H7-1 schema detail (A1):**
  - `publish_seq` goes into every published record: the scheduler's `records`, the callback copy and `branch_result.json`;
  - the notebook's `E12_RECORDS` keeps the higher one;
  - the final snapshot always has the highest;
  - the test also asserts that `submission.json` is unchanged after a late older callback.

#### Consolidated H7 list (all validated; nothing built)

| ID | From | What | Validation | Priority (mine) | Fix | Test |
|---|---|---|---|---|---|---|
| **H7-1** | A1 P0, A2 F6-3 optional, A3 — | Callbacks can finish out of order; the notebook replaces by arrival | Ordering **reproduced**; stale content **not reproduced** in the production flows (equal snapshots) | P1, defensive (cheap) | `publish_seq` set under the scheduler lock; in every published record and `branch_result.json`; `_e12_branch_done_r6` drops a seq ≤ the stored one and skips the submission write | hold an older callback, finish the final one, release it → the stored record and `submission.json` are unchanged |
| **H7-11** | A3 F2 | `freeze()` is two critical sections, and the record references the live agent's lists / dicts | **Reproduced** (deterministic 1 vs 2; stress 60 / 60) | P1 (needs a running straggler; G4B had 0) | One locked snapshot: deep-copy from the agent, write that copy, set `_final`; `write()` copies the agent state | a barrier and a mutating agent → disk == `_final` == the callback record |
| **H7-10** | A3 F1 | The nudge is "emitted", not "seen": 3 of 7 land in the pause request | **Confirmed** (the same 3 tasks and token counts; no seen field) | P1 (makes the G4B nudge result interpretable) | `provisional_nudge_seen_request` / tokens set in `_chat_completion` when an outbound request contains the text; report emitted / seen / effective | emitted at the pause → seen None; after the resume → recorded |
| **H7-2** | A1 | A worse same-name redefinition overwrites the best progress's `src` | **Reproduced** | P2 (diagnostics) | Set `src` only when this call changed the best | better → worse → better under one name |
| **H7-3** | A1, A2 F6-1 | `new_verified` is branch-local; the "all" yield over-counts | **Code-confirmed** | P2 (decision metric) | Rename `new_in_branch`; add `new_for_task` (first discovery of a (task, outputs) signature by the host-verification time) | same output in b0 and b1 → 2 branch-local, 1 task-global |
| **H7-7** | A2 F6-2 | "Tasks exposed" counts initialized sessions; the report's same-name field counts responses | **Reproduced** (0 requests counted) | P2 (coverage claim) | Scheduler: `tasks_initialized`, `tasks_with_request_sent`; report: `tasks_with_response`; the coverage line prints all three; 240/240 is claimed on "request sent" | an expired session → initialized 1, sent 0 |
| **H7-5** | A1, A3 F3 | `unique_verified_programs` (name / output) next to `verified_programs_by_source` | **Code-confirmed** | P3 (naming) | Rename the old one `unique_name_output_candidates` (or document it) | the two cases above |
| **H7-12** | A2 Q4 | No derived "correct outputs / GPU-hour" or task yield per GPU-hour | **Code-confirmed** | P3 (report) | Report: `correct_inputs_per_gpu_hour`, `tasks_solved_per_gpu_hour`, `new_for_task_per_gpu_hour` (solve-window seconds × 4 GPUs) | the harness checks the fields |
| **H7-13** | A3 F4 | No per-request classification of zero-hit requests | **Code-confirmed** | P3 (telemetry) | Classify every later zero-hit request: `unexplained_cache_miss` (serialized prefix full, tools and template same, not trimmed) vs `prompt_changed:<reason>`; report counts; rename `prefix_chars` → `serialized_char_prefix` (H6-10 wording) | template change → `prompt_changed:template`; full prefix + 0 cached → `unexplained_cache_miss` |
| **H7-6** | A1, A3 | `verify_h6.py` and `facts_size.py` crash without `tokenizers`; the harness without `transformers` | **Reproduced** (both `ImportError`, exit 1) | P2 (reviewers could not run them) | Visible SKIP / INCOMPLETE (exit 2) in both; the harness exits with "needs transformers" (exit 2) | run with the package blocked |
| **H7-9** | A2 Q2 | No deterministic test of H6-8 (defer → branch 0 finishes → branch 1 re-queued → admitted) | **Gap confirmed** (my fake-scheduler attempt failed on the chunk sizes) | P3 (test) | A test with `fair_share=False` and an injected state | as described |
| **H7-8** | A2 | The indexed view can still be over the bound after hiding everything, with no flag | **Code-confirmed** (needs about 600 grids) | P3 | `view_over_target` event | an impossible bound → flagged |
| **H7-4** | A1 | The post-solve diagnostic does not check stragglers | **Code-confirmed** (G4B 0 stragglers) | P3 | Skip it when `stragglers > 0` | stragglers 1 → no diagnostic request |
| option | A1, A2, A3 | No reviewer could install `tokenizers` / `transformers` | All three reported it | user decides | Ship an offline `tokenizers` wheel (abi3 Linux) in the bundle; `transformers` is heavier | — |

#### Process gates the audits asked for (not code)

- **Before the GPU run (A1, A3):**
  - the final notebook shas re-recorded after any patch;
  - the full smoke and harness suites plus the historical replays with no new failure;
  - a pre-run capacity check (our rule).
- **Reading the G4B results (A1 Q4, A2 Q4, A3 Q4):**
  - correct outputs first;
  - then distinct verified predictions per task (task-global, H7-3);
  - then first transform / check timing and the nudge seen → effective (H7-10);
  - then stage-2 progress per extra 12k (`runs[].improved`, `new_for_task`);
  - then cache / throughput / KV.
  - Verified-candidate counts are never a score substitute.
  - One G4B run gives evidence, not a causal 12k vs 24k winner.
- **Hidden 240-task rerun (A2):** not qualified until a run shows real-request coverage on all 240 tasks and a deadline-safe submission.

#### No change (agreed by the audits and by me)

- The nudge wording (it does not ask the model to drop a hypothesis).
- One nudge only; no mid-request intervention (non-streaming).
- H6-8 stays defensive (no starvation path).
- H6-12 idle override (user decision).
- H6-13 (not reproduced).
- H6-15 (optional).
- PARITY fixes C6-1 / C6-3 (deferred by the user).
- No scheduler redesign before G4B.

## Rev 6g: the H7 set built, the LAST code version before the GPU run (2026-10-22)

**Decision:**
- The user said "make the last update", then: "two or three days in this loop and we lose; this is the last version".
- Rev 6g builds H7-1..H7-13 (the H7-9 analysis below).
- No further audit rounds: the next step is the G4B GPU run. Only a confirmed score-affecting bug would reopen the code.

| ID | Built |
|---|---|
| H7-1 | `publish_seq` on every published record (under the scheduler lock); the notebook callback drops a record whose seq is ≤ the stored one and skips its submission write (`e12_run.stale_callbacks_dropped`) |
| H7-11 | `freeze()` = one critical section: a snapshot of **copied** agent state, the same snapshot written to disk, `_final` set; `write()` copies the agent's lists and dicts (retried if a dict grows mid-copy) |
| H7-10 | `provisional_nudge_seen_request` / `_tokens`: the first outbound request whose messages carry the nudge text; report `nudge.seen`, `nudge.emitted_not_seen` |
| H7-2 | The best progress gets a new `src` only when this call created a new best |
| H7-3 | `runs[].new_in_branch` + `runs[].new_for_task` (the scheduler, in run-completion order); `verified_yield.*` reports both (`per_100k_for_task`) |
| H7-7 | Scheduler `tasks_initialized`, `tasks_with_request_sent` (`requests_started`); report `tasks_with_response`; the coverage line prints all three |
| H7-5 | `unique_name_output_candidates` (documented) next to `verified_programs_by_source` |
| H7-12 | `rev6.per_gpu_hour`: GPU-hours (solve window × 4), correct inputs, tasks solved, new-for-task predictions per GPU-hour |
| H7-13 | `later_zero_hit_class`: `unexplained_cache_miss` vs `prompt_changed:{trimmed, tools, template, prefix}`; per-request `prefix_chars` renamed `serialized_char_prefix` |
| H7-8 | `indexed_view_tokens`, `view_over_target` events; report `indexed_view_over_target` |
| H7-4 | The post-solve diagnostic is skipped when the scheduler reports stragglers |
| H7-6 | `verify_h6.py` and `facts_size.py` end INCOMPLETE (exit 2) without `tokenizers`; the harness exits 2 with "needs transformers" |
| **H7-9** | **Not buildable as a real scenario: analysed instead.** `rank()` orders stage-2 items by (rounds of this branch, class, …, sub). A queued branch-1 start has 0 rounds. For class 1 it sorts before its own branch 0's resume (same rounds, `start1` sub 0); after a stall, branch 0 has ≥ 1 round. So a queued branch 1 is always popped before its branch 0 can run again and improve. The pop-time check of H6-8 cannot trigger under the current order: it is dead defensive code, consistent with `b1_deferred_at_pop` = 0 in every run. The predicate's unit test remains |

**Validation (final files):**
- **Smoke:**
  - rev6 **171/171** (163 → 171, two tests updated to the new semantics);
  - rev5 56/56; adapter 41/41.
- **Without `tokenizers`:**
  - rev6 170 + 1 skipped (exit 2);
  - `verify_h6` and `facts_size` exit 2 INCOMPLETE.
- **Harness:** g4b **60/60**, r6 **59/59**, rerun 23/23, launchfail 6/6, parity 19/19. My first g4b / r6 run had two harness-code errors (a check used `m` before it was defined, and read report rows that do not hold `publish_seq`); fixed.
- **Other suites:**
  - regex ALL OK; KV replay 7/7; window 7/7;
  - facts size unchanged (p50 699 / max 798);
  - `verify_h6` 6/6 not reproduced;
  - replays 7/8, 5/8, 10/19, identical per task;
  - sim at 277 tok/s: 240/240 exposed, KV max 0.40, 0 holds.
- **pyflakes:** the adapter and builder are clean.
- **Not re-run:** the mutation suite. Two of its anchors moved with H7-11; the guarded behaviours are covered by the passing tests above.
- **Notebooks:**
  - EVAL12R6 `74a220de…`, **EVAL12R6G4B_REBUILD `5d2bc39f…`**, PARITYR6 `82b257a5…`;
  - the embedded adapter == source in all 3;
  - all 9 Kaggle-run notebooks unchanged.
- **Backups:** `_scratch/session_1c6db7aa/rev6g_backup/`.

**Next:** the G4B GPU run (user's go + account; pre-run capacity check). After the run, read the results in this order: correct outputs → task-global verified predictions → nudge seen / effective → stage-2 progress → cache / throughput.
