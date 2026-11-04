# E31 design: Flash-Next DIRECT on 30 EVAL tasks (Kaggle 4xL4, klee100 on Merlin)

Status: design, in build (2026-11-01). Nothing has run on GPU.

## Goal
Measure two things for Qwen3.8-Flash-Next klee100 3bpw-MTP with the DIRECT method (no tools: think at xhigh, then write the output grid as text):
- how strong it is;
- how well it complements our Duck-based K100 run.

The answer decides whether a DIRECT attempt goes into the production notebook. E29 would fill idle Duck workers with direct attempts, and a direct grid would become attempt_2 when Duck has no train pass.

Evidence for the complement comes from Qwen3.8-27B (`EVAL120_K100_W4_Direct_Duck_per_task_2026-11-01.md`):
- K100 and 27B-Direct overlap on 43 tasks.
- 39 tasks are solved only by K100.
- 9 tasks are solved only by 27B-Direct. Five of them (446ef5d2, 7b5033c1, a47bf94d, a6f40cea, cbebaa4b) are solved by no other run.
- Flash-Next DIRECT has never been measured.

## Base pieces (all proven; copied, not re-invented)
- **Engine boot:** E28 K100 EVAL120 6H. This ran 6 h, the result was VALID, and task-norm was 65.0 %.
  - Stack: Merlin vLLM-0.32 fork, klee100, TP2xPP2, MTP K3, Q-defer, I8 int8 draft head, 24 seqs, pin 7,933,526,016, FP8 QSA KV, max-model-len 131072.
  - Boot took about 15 min.
  - CELL 0 is the E18 R2 identity submission, verbatim. The E15B submission writer, CELL 7 scoring and the E28 report are unchanged.
  - Report: `E28_K100_notebook_hookpoints_agent_report_2026-11-01.md`.
- **DIRECT method:** v1.4.4 DIRECT / E10 ORDER2 86k (27B, 58/172, task-norm 36.94 %).
  - The russcore prompt and the chat template with xhigh. The Flash-Next template is byte-identical to the 27B one, and the special ids are equal.
  - Phase 1 is one request. If thinking does not close, phase 2 is the forced-close STOP text, then the answer.
  - `parse_grid` and `make_attempts`: attempt_1 is the model grid; attempt_2 is the test input or its mirror.
  - ORDER2 queue order.
  - Report: `DIRECT_solver_spec_E10_ORDER2_agent_report_2026-11-01.md`.
- **Client:** the E18 vLLM-0.32 shim, used as is. It streams `/v1/completions` with prompt token ids, returns token ids, and aborts by closing the stream.

## Design
1. **Notebook shape.** The E28 K100 notebook with two edits:
   - CELL 1 gets the new profile and config.
   - One new cell rebinds `E12A.solve_all_r6` to the direct solver. It keeps the same signature, return value and record contract, so `e21_cell6`, `e21_cell7`, the E28 report and the writer stay unchanged.
   - `e28_sched` is off: Duck stays original and is not used.
   - The notebook must stay under 400 kB, so the unused E28 scheduler literal is dropped or the solver is packed.
2. **Editable line at the top of CELL 1.** It sits outside the packed code, so it can be changed without a rebuild:
   `E31_CFG = {'think': 104000, 'answer': 8192, 'window_min': 170, 'workers': 24, 'temperature': 0.6}`
   - Phase-1 cap = min(think, 131072 - prompt - answer*N - 256). The floor is 60k (user rule) and is recorded.
   - 104k matches E28's T1 attempt budget.
   - Temperature 0.6 is the DIRECT method's value. Duck used 1.0.
3. **Tasks.** 30 EVAL tasks, concurrency 24 (user):
   - **Main, 16 tasks:** K100 failed them, but W4, 27B-Direct or 27B-Duck solved them: 2d0172a1 35ab12c3 409aa875 446ef5d2 4c3d4a41 4c416de3 5545f144 7491f3cf 7b5033c1 800d221b 89565ca0 9385bd28 a47bf94d a6f40cea b6f77b65 cbebaa4b.
   - **Partial, 8 tasks:** K100 got some test inputs right: 142ca369 247ef758 36a08778 581f7754 71e489b6 88e364bc a32d8b75 f931b4a8.
   - **Control, 6 tasks:** K100 solved them.
     - Also solved by 27B-Direct: 135a2760, 136b0064, 1818057f.
     - Solved by K100 only: 0934a4d8, 13e47133, 16b78196.
4. **One unit per TASK (user change).** A task with 2 or 3 test inputs gets one prompt that holds all of them, one thinking pass, and one answer with one grid per test input. ORDER2 instead thinks separately for every test input, which wastes tokens and time.
   - **Single-test task:** the prompt is byte-identical to ORDER2.
   - **Multi-test task:** the test inputs are listed as `FINAL INPUT 1..N`. The closing instruction asks for one labeled fenced block per input, `grid 1` .. `grid N`, in order, followed by "Output nothing after the last block." Every other sentence is ORDER2's.
   - **Parser:**
     - Labeled blocks map to test K; the last occurrence of a label wins.
     - If labels are missing, the last N valid fenced blocks are taken in order.
     - A test with no grid falls back to ORDER2 (test input or mirror) for that test only.
   - **Answer budget:** answer x N, capped by the room left in the context.
   - This prompt variant is new and untested on 27B. The report shows multi-test tasks separately.
5. **Deadlines.** At the end of the window, live phase-1 streams are aborted. The tokens already streamed are kept, and phase 2 runs with the forced close, as in the ORDER2 drain. No quality or label heuristic ever stops a request (user rule: gates only on real failures).
6. **Outputs:**
   - The E28 record contract, so CELL 7 scores it.
   - `q27x/e31_direct_records.jsonl`: per task and per test. Fields: mode (natural / continue / forced_close), tokens, finishes, has_grid, hit, group.
   - `q27x/e31_direct_report.json`: hits and task-norm per group, mean tokens, tok/s, modes, and a per-task comparison against K100, W4, 27B-Direct and 27B-Duck.
   - The full thinking of every request is saved.
7. **Account.** esfersami50. The kernel-metadata sources are E28 K100's. The slug is `q27-xhigh-e31-fn-direct30`.

## Timeline estimate
- **Measured in E28:** 419.5 tok/s with 23.4 running streams, about 18 tok/s per stream.
- **Per task:** a 104k trace takes about 90 min.
- **Waves:** 30 requests on 24 workers make one wave plus 6 more.
- **Total:** boot 15 min + window up to 170 min + reserve. That is about 3 h at most, and less if traces close early.
- **Shorter option:** think 77824 with window 130 gives about 2.5 h.

## Kits (before any GPU commit)
- **units:** tests for:
  - the prompt bytes against ORDER2;
  - the multi-test prompt;
  - STOP_IDS (25 ids, `</think>` = 248069);
  - the caps;
  - a token-exact phase-2 prefix;
  - the drain;
  - `parse_grid`, the label parser and `make_attempts`;
  - the record contract;
  - the `E31_CFG` line.
- **real:** E28's real kit, retargeted. The E28V scheduler check is dropped because `e28_sched` is off. The kit calls the notebook's own pre-launch sequence verbatim.
- **e2e-commit:** a fake server that answers streaming token-id `/v1/completions`.

## Questions for the reviewer (blockers / severe only)
1. Does the multi-test prompt or parser risk wrong test-to-grid mapping, or a grid lost to truncation?
2. Is the token-exact phase-2 continuation correct on Merlin, given prefix caching in mamba "align" mode, block 4096 and prefix-match-unit 64?
3. Can the solver rebind break anything in e21_cell6 / e21_cell7 / the E28 report / the submission writer?
4. Are the timeline and the 24-worker, 104k setting consistent with KV capacity? Measured streams that fit: 120k → 13, 90k → 17, 131k → 12. With 24 running, expect preemptions; E28 ran 24 with 438 preemptions.
