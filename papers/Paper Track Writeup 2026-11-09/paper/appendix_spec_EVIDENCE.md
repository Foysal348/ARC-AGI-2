# Evidence for appendix_spec.tex (submission 57851995)

## Sources and short names

- **NB6** = `experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build/nb6_code_export.py`. This is the code export written by `build_e25_nb6.py`, from the same build as the notebook `Q27-XHIGH-E25_NB6_SUBMIT.ipynb`. The hidden rerun runs `PROFILE = "RERUN_PROD"` (NB6:52).
- **B6** = `build_e25_nb6.py` in the same folder. It lists the NB6 edits E1–E7 and P1–P6 (B6:1-18).
- **LIB** = `nb5_lib.py` in the same folder. Its sha256 is `18efcbc1…`, the same as the source embedded at NB6:363 (checked by decoding the embedded blob).
- **DUCK** = `arc2_duck.py`, and **TA** / **SBX** = `vendor/inference/agent/tool_agent.py` / `python_tool_sandbox.py`, all from the payload.
  - The payload is `experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload/q27x_e15b_duck_9e4b89285136.bin`. Its sha256 `9e4b8928…` matches NB6:389.
  - It was extracted for this check, and the file hashes match the manifest at NB6:393-417 (arc2_duck `b9f67010…`, tool_agent `f7454201…`, sandbox `d29d807e…`, prompts `14965a99…`).
- **K100** = `experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build/e28k100_code_export.py` (sha256 `ec8b94f3…`).
  - Notebook `Q27-XHIGH-E28_K100_EVAL120_6H.ipynb` has sha `d3d132da…`, which equals `e28k100_build_facts.json`.
- **SCHED** = `E28_SCHED_API.md` in the same folder (the API document of `e28_sched_patch.py`, sha `108bb75c…`).

## 0. Agent origin
- Duck fork: da-fr/arc-agi-3-solution @ 10882e332d, Apache-2.0. Two patches are switched off by default; the other 18 files are byte-identical (NB6:382-386). The commit is also recorded at NB6:321.
- Served name `q27`: NB6:324 and NB6:1162.
- Model path `qwen3-8-flash-next-w4a16-merlin-p1`: NB6:224.

## 1. Prompt and grids
- The prompt builder is DUCK:1038-1118 (`arc2_system_prompt_r6`); the excerpt is from DUCK:1042-1050 and 1084-1085.
- B-mode edits:
  - budget sentence: DUCK:1121-1139;
  - inserted with `budget = b_attempt_budget(branch)`: DUCK:1449-1450 and 1142-1145.
- The B mode is on through env E12_B_MODE=1 (NB6:299-304), E12_ARC2_REV6=1 (NB6:279-283) and E12_ARC2_REV5 (NB6:253-255).
- View `facts` for both attempts:
  - attempt 1 view: DUCK:3263 (`"facts" if b == 0`);
  - attempt 2 view (`b1_view`): `E12_R6_B1_VIEW: "facts"` at NB6:262, passed at NB6:3468.
- Facts block:
  - content: DUCK:846-907;
  - bound of 800 tokens / 3,600 chars: DUCK:932-943, with the values at NB6:267 and 270.
- The facts view has no raw grids: DUCK:1056-1059 (the `else` branch). Grids are printed only for `view in ("grids","indexed")` (DUCK:1088-1094).
- `show()` limits:
  - the code: DUCK:770-797;
  - 900 per view and 1,000 per call: NB6:267 and 276, and DUCK:975-976.
- Grid-print guard: DUCK:997-1030.
- Opener texts: DUCK:1202-1226 and 1351-1356. The first opener adds the budget at DUCK:1663-1666.

## 2. Tools
- One tool `python`: DUCK:1101-1102, plus the B edit at DUCK:1131-1133. The tool description is at DUCK:458-463.
- `check`:
  - source: DUCK:642-735 (`_arc2_eval_pair` with 6 wrong-cell samples, the identity flag, test-input status, the verdict line, details);
  - host report via `PROGRESS:` / `CHECKED:`: DUCK:695-704.
- `submit`: DUCK:231-238. It is spliced into the R6 prelude at DUCK:970-994.
- Host verification:
  - one clean sandbox per grid, structured `result`, json.dumps guard: DUCK:267-289 and 2500-2570;
  - acceptance requires all train pairs exact and every test output non-None: DUCK:2561-2565;
  - acceptance ends the attempt (`self.arc2_accepted = verdict`): DUCK:2203-2211, with the loop end at DUCK:2975.
- Helpers: DUCK:493-628 and 737-797. The list given to the model is at DUCK:1067-1076.
- Auto-check:
  - name regex (B mode): DUCK:1233-1238;
  - caps of 24 / 24 / 3 per call: NB6:263 and DUCK:1513-1516.
- submit / check called as tool names are run as python: DUCK:1575-1621.

## 3. Sampling and limits
- Temperature, top_p, top_k:
  - DUCK_ENV 0.7 / 0.95 / 20: DUCK:38-61;
  - the temperature is overridden to 1.0 by E12_DUCK_TEMPERATURE: NB6:248, DUCK:69-71, written to env at NB6:293-296.
- Seed:
  - `LOCAL_ANALYZER_SEED` default -1: TA:204;
  - passed to the payload: TA:5250;
  - sent only if >= 0: `vendor/inference/utils/openai_compat.py:95`;
  - RERUN_PROD sets no seed: NB6:95-100 (nb5_env has no LOCAL_ANALYZER_SEED; compare NB5QUALITY at NB6:116).
- reasoning_effort xhigh:
  - per request: NB6:249-252 and DUCK:1562-1567;
  - server default: XHIGH at NB6:1155, passed at NB6:1162.
  - Thinking is on: DUCK_ENV `LOCAL_ANALYZER_ENABLE_THINKING` at DUCK:42.
- Budget 104,000 for both attempts:
  - NB6:95 and 202-203;
  - E15B dict at NB6:299-304;
  - 60k floor asserts: NB6:221-222 and 320-321.
- Request cap:
  - remaining minus reserve 4,096: DUCK:1787-1805 (line 1800), reserve at NB6:204;
  - context guard `room = peak - prompt - 2048`: LIB:874-898, called per request at NB6:546-600;
  - peak = min(131072, 8192 + budget + 8192): LIB:116-118 and NB6:207, used as `max_len` at NB6:540-543 (`_nb5_peak_len`).
- Commit zone, remaining <= reserve + 1,024:
  - DUCK:1693-1695, with the text at DUCK:1352-1355;
  - variant A = Duck's normal request path: NB6:663-677 (C only when `v == "C"`);
  - mode "A" from the phase spec: NB6:100 and 2458.
- Context:
  - MAX_LEN 131072: NB6:777;
  - `--max-model-len` in the server extra: NB6:1161-1162;
  - window = min(131072, 131072 - 512): NB6:206 and 3326-3328;
  - raw reasoning in history: NB6:230 and 433.
- Request timeout 3,600 s: NB6:292. It is clamped to the deadline at DUCK:1915-1925 and 3255.
- Tool steps: `LOCAL_ANALYZER_TOOL_STEPS "0"` (DUCK:43) means `None`, i.e. no cap (TA:3497).
- Sandbox:
  - timeout 30 s: DUCK_ENV at DUCK:44 and TA:3498;
  - RLIMIT_CPU timeout+1, FSIZE 1,000,000, NOFILE 32: SBX:319-330;
  - RLIMIT_AS 4,096 MB per child with read-back: DUCK:296-347, value at NB6:316, installed at NB6:466;
  - fail closed: NB6:508-514.
- Tool output 3,072 tokens: DUCK:50 and TA:3507-3508.
- Turn yield 2,048 tokens and ARC3_YIELD_AFTER_BATCH: DUCK:46 and 78-80. This is a setting only; it is not in the appendix table.

## 4. Attempt policy and selection
- E15B settings:
  - skip 0, extra 0, b1 mode 0: NB6:96;
  - applied through setdefault: NB6:159-160, then E15B at NB6:299-304;
  - `{0: ""}` mapping (NB6 E4): NB6:3478 and B6:91-92;
  - WEAK_COMMIT stays 1: NB6:301.
- Scheduler:
  - stage 1 pushes attempt 1 of every task first: DUCK:3317-3318;
  - attempt 2 is queued when an attempt ends: DUCK:3573, via `stage2_items` at DUCK:3320-3345;
  - rank (1, rounds, class, …): DUCK:3293-3302;
  - classes: DUCK:3197-3209;
  - strong_only = weak_commit: DUCK:3298;
  - `first_exposure_tokens` 0 means no pause (NB6:262 and 3467). Fair share is off (NB6:271 and 3471).
  - min_admit_sec 300: NB6:262, DUCK:3454.
  - The worker count is fixed: DUCK:3220-3249.
- End-of-attempt harvest and partial test:
  - DUCK:3046-3065;
  - the functions: DUCK:2319-2335 and 2383-2393;
  - grace 30 s: DUCK:3049;
  - join grace 90 s: NB6:3474.
- Selection: `select_attempts_r5` (DUCK:2772-2835).
  - sort key: DUCK:2815-2819;
  - rule library: DUCK:2748-2767 and 2797-2803;
  - partial tier: DUCK:2789-2790 and 2820-2823.
  - Used because REV5 is on (DUCK:2840-2841).
- Fallback identity:
  - CELL 0 writes it: NB6:39-43;
  - selection-error fallback: DUCK:2846-2848;
  - empty-grid fallback: DUCK:2829-2832.
- Several test inputs: the per-`qi` loop at DUCK:2809-2833. Rows replace a task only if their count matches (NB6:2912).

## 5. ORDER2
NB6:2864-2875: score = [train >= 3] + [test == 1], sort by (-score, total cells, index in sorted TASK_IDS). TASK_IDS are sorted ids in the rerun (NB6:759 and 2450).

## 6. Engine
- Merlin install:
  - payload path: NB6:1237;
  - offline install script, 5,400-s timeout: NB6:1109-1143;
  - the plan `ML.build("flash-next", "fn-quad-24g-sm89", …)`: NB6:1154 and 1181;
  - LD_PRELOAD shim: NB6:1191.
- Merlin public source:
  - `experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/prep/P1_merlin_checkpoint/README.md:180-187`: tag merlin-v3.0.0 = commit ad42e37f, GitHub halt95/merlin-engine;
  - the wheel name is also at `prep/P2_merlin_env/build_p2_kernel.py:26`;
  - commit ad42e37f is also stated in the #60490 header at NB6:1884.
- Checkpoint:
  - 5 parts + FP8 PLE overlay: NB6:1226-1236;
  - merge with size checks and `ple_embedding_dtype` check: NB6:1033-1078;
  - no sha gate in the rerun: NB6:4038-4039 (NB6 P6).
- Layout, block size, batched tokens, GPU utilisation, prefix caching, mamba align, prefix-match-unit, replayssm, tool and reasoning parsers, generation override, MTP K3 probabilistic: the RC2 reference argv `NB5_RC2_FP8_ARGV_TAIL` at NB6:1857.
  - Caveat: this is the recorded Merlin plan of the RC2 boot. The production argv is produced by Merlin's `build()` at boot time and is diffed against it (NB6:2691).
  - The boot read-back verifies only the KV dtype, max_num_seqs, the KV bytes and the speculative config (LIB:1052-1082).
  - In the rerun a read-back mismatch is not fatal (NB6:2700).
- KV FP8 + scales: NB6:180 and 1179-1180.
- Pin ladder:
  - steps: LIB:1146-1153;
  - base pin: NB6:184;
  - start at 0 if #60490 is not verified: NB6:2784-2788;
  - lever drop: NB6:2818-2828.
- #60490: diff text at NB6:1884-1885, applier `nb5_apply_60490` at NB6:1928, applied before the first boot at NB6:4042-4048.
- MTP readback (method mtp, K = 3): LIB:1064-1076.
- CAPS16: NB6:83 and 210-212. These are trailing args, so they win over the plan's `[1..32]`.
- MERLIN_SHARE_DRAFT_WORKSPACE:
  - set: NB6:98;
  - dropped on an error line: NB6:2775-2776.
  - Its meaning ("MTP draft shares the target workspace") is from `PLAN_1500.md:321`, a plan document, not code.
- Measured F: LIB:107-112 and NB6:2034-2041.
- Workers rule: LIB:1122-1137 and NB6:2042-2053.
  - SEQS_TABLE fp8 {1.0: 9, 0.75: 8, 0.5: 8, 0.0: 6}: LIB:1133.
  - Workers = CONCURRENCY: NB6:2711-2724.
  - streams = min(CONCURRENCY, E12_STREAMS): NB6:3302.
  - Isolation fallback to 6: DUCK:3177-3179 and NB6:3338.
  - "pool 214, 9 workers" is the comment for the NB5QUALITY_R2 run at NB6:88. The pool of the hidden run itself is not in the code.
- Slots admission:
  - NB6:699-735 (`acquire(key, 1)`, need > pool cap means no start);
  - Reservation sized to CONCURRENCY: NB6:2482;
  - KV gate = NB5_GATE = 1.0: NB6:95, 201 and 274, passed at NB6:3470.

## 7. Run safety
- Clock anchor: NB6:7-15.
- Budget 12 h - 3 min: NB6:141. Margins: NB6:54-55.
- Deadlines: NB6:166-175.
  - HARD_STOP = budget - 120 s.
  - Solve end = HARD_STOP - 180 s: NB6:310 and 3304.
- `DRAIN_DEADLINE` / `ADMIT_DEADLINE` are used only in their definition and assert (grep: lines 170-175 only).
- Engine watchdog at HARD_STOP_EPOCH = T0 + budget: NB6:775 and 1260.
- Boot ladder:
  - <= 3 tries, 30 / 15 min: NB6:2802-2810 (B6 P3 at B6:173-185);
  - no try with < 25 min left: NB6:2793-2795;
  - ready / stall rule: NB6:964-990.
- Solve watchdog:
  - poll 10 s, stall 600 s, fail 180 s, min left 20 min: NB6:3368-3373;
  - logic: NB6:3378-3460;
  - <= 2 relaunches: NB6:3431.
- Relaunch:
  - NB6:2629-2663: needs >= 30 min after subtracting 5 min (NB6:2634-2636), ready wait <= 20 min (NB6:2655), sampled check (NB6:2656-2657);
  - the relaunch boots with `NB5_KV_BYTES[kv]`, the base pin, not the ladder step (NB6:2652);
  - branches wait while down: DUCK:3014-3020.
- Submission writes:
  - CELL 0 at NB6:43;
  - per attempt at NB6:2941, stopping at the solve end (NB6:2937-2939);
  - final writes at NB6:3591 and 3985;
  - atomic writer: NB6:2890-2903.

## 8. klee100 (K100)
- `RERUN_PROD` = NB6 levers + E28_LEVERS: K100:213 and 221.
- Levers: K100:185-205.
  - T1 104,000, B2 120,000, B2 timeout 9,000: K100:191-195.
  - I8 env: K100:197.
  - top_rung_seqs 24: K100:205.
- Checkpoint path: K100:306.
- Quantization layout: `k100load_lib.py:506` and `build_e25_klee100_load.py:1-2` (both in the E25 Build folder).
- Pins and seqs: K100:173-178. Extra flags: K100:176 and 293.
- Reused W4 scales: K100:1574-1575.
- Workers rule and top rung: K100:2801-2815 and 3592-3593.
- caps16 False / k100_caps True come from K100_BOOT_LEVERS (K100:179) and override NB6_LEVERS caps16 in the merge (K100:221).
- Q-defer:
  - library sha 8dadf076: K100:507-513;
  - apply and gate (the rerun continues without it if unverified): K100:1747-1762 and 1810-1845;
  - env only on a verified patch: K100:1877-1879;
  - mechanism: E25 Build `nb4_ppack.py:1-38`.
- I8:
  - on from the first try: K100:4857;
  - trusted only on `I8_ENGAGED`: K100:1954-1960 and 3483.
- Request timeout 7,200 s: K100:375.
- Plain order: K100:3684.
- A / B / C scheduler and answer slots: SCHED section 3 (lines 52-104).
- Patch applied and env set in CELL 3: K100:590-601.
- C12 is a W4-notebook lever (E28 README section 1 table). Nothing in the K100 or NB6 code sets it.

## Could not find or could not verify
1. **MTP acceptance rule.** The code sets `draft_sample_method: "probabilistic"` (NB6:1857) and checks method / K (LIB:1064-1076). The vLLM/Merlin rejection-sampler source was not inspected (a grep of `_scratch` timed out), so the appendix does not describe the accept/reject arithmetic.
2. **MERLIN_SHARE_DRAFT_WORKSPACE.** The exact effect is not visible in our code. The description comes from PLAN_1500.md:321.
3. **Production server argv.** The exact argv of the rerun boot is created at run time by Merlin's `build()`. The values quoted (TP2×PP2+EP, block 4096, batched tokens 1024, gpu util 0.96, prefix caching, generation override) come from the RC2 reference argv embedded in the notebook, not from a read-back of the hidden run.
4. **Pool size and actual worker count in the hidden run** are runtime values and are not in the code. "Up to 9" follows from SEQS_TABLE at the top pin step.
5. **The pushed version of the klee100 notebook** (esfersami50/q27-xhigh-e28-k100-eval120-6h) was not matched against Kaggle. The local notebook sha d3d132da is the E28 build-facts sha.
6. **Merlin's public status** is from the E23 prep README (GitHub tag and release), not from the notebook code.
