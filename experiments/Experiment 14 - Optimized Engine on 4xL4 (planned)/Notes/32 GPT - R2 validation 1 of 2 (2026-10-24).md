<!-- Pasted by the user on 2026-10-24 (GPT validation of the R2 zip GPT_VALIDATE_E14A_R2_2026-10-24.zip). Kept VERBATIM. Claude's validation: Notes/34. -->

E14-A R2 — GPU ফলাফল, ফিক্স যাচাই ও পরবর্তী সিদ্ধান্ত
ZIP-এর 00_GPT_PROMPT.md প্রথমে পড়েছি। তারপর notes/30, actual GPU JSON, server logs, notebook, builder ও E15 পরিকল্পনার source মিলিয়ে যাচাই করেছি। নিচে প্রতিটি প্রশ্নের উত্তর এবং কোন ফিক্স কার্যকর হয়েছে, কোনটি হয়নি, কোনটি এখনো অনিশ্চিত—আলাদা করে দিচ্ছি।
প্রথমেই একটি গুরুত্বপূর্ণ submission finding
বর্তমান notebook-এ পরিকল্পিত width6 + graphs13 rerun configuration নেই

Source-confirmed: code/Q27-XHIGH-E14A_SLIM_R2_OPTIMIZED_ENGINE_SGLANG.ipynb, CELL 5 e11_flags()-এ defaults এখনো:
- --speculative-num-draft-tokens 8
- --cuda-graph-max-bs-decode 12
- --max-running-requests 13
- --mamba-ssm-dtype float16
code/build_e14.py:133 অনুযায়ী RERUN=True হলে measurement phases বন্ধ। ফলে commit run-এ width6 কিংবা graphs13 adopt হলেও সেই decisions hidden rerun-এ স্বয়ংক্রিয়ভাবে বহাল থাকবে না।

সিদ্ধান্ত: W1 baseline-এ rerun করার জন্য এটি bug নয়। কিন্তু ঘোষিত width6 + graphs13 rerun plan-এর জন্য NO-GO, যতক্ষণ না exact rerun flags profile-এ সেট করে rebuild ও CPU-harness validation করা হয়।

যেসব R2 ফল raw output দিয়ে নিশ্চিত হয়েছে
Duck correct outputs
6/19
12 evaluation tasks

Prompt cache
82.0%
1,266,937 / 1,545,883

DFlash2 W6
531.43
tok/s, short-context arm


notes/30-এর Duck score, width6/4 raw throughput, target-only result, 12-stream baseline, nodelay crash, int8 Mamba OOM এবং NCCL baseline-এর প্রধান সংখ্যাগুলো archived e14_results.json ও e12_run_report.json-এর সঙ্গে মেলে।

তবে 6/19 হলো 25-minute selected-eval result; 240-task hidden leaderboard score নয়। আর raw-reasoning 115 kept / zero mismatch-কে 115টি সম্পূর্ণ cache-reusable turns বলে ব্যাখ্যা করা যাবে না।
A. R2 ফলাফল: README-এর সাতটি প্রশ্ন
A1. notes/30-এর সংখ্যাগুলো ঠিক?
Verdict: GO — প্রধান সংখ্যাগুলো মিলে। তবে কয়েকটির denominator ও interpretation সংশোধন দরকার।
Result	Raw evidence	Verdict
W1 short 505.0 tok/s, 114.02 ms, accept 4.86	e14_results.json → phases.A1-2_baseline_W1.short	Confirmed
W1 long 449.84 tok/s, 123.07 ms, accept 4.62	একই phase .long	Confirmed
#44401 508.14 short, 455.15 long	A1-5_p44401	Confirmed
nodelay 490.32 short; 480.96 long	A1-5b_nodelay	Confirmed
graphs13 500.14 short; 436.89 long	A1-6_graphs13	Confirmed
target-only 339.55	A2-3_target_only.short	Confirmed
Width6 531.43 / 96.39 ms / accept 4.24	A2-4_width6.short	Confirmed
Width4 526.81 / 74.50 ms / accept 3.27	A2-5_width4.short	Confirmed
int8 Mamba OOM, 192 exceptions	A2-2_int8_mamba.server_log_signs	Confirmed
Duck 6/19, 4/12 tasks	q27x/e12_run_report.json → eval_query_hits, eval_task_norm_pct	Confirmed from saved evaluation report
17 started, 10 done, 3 accepted branches	e14_results.json → duck_run.e12_run	Confirmed
Raw reasoning 115 kept, 0 mismatch	duck_run.raw_reasoning[]	Event counts confirmed
JIT-cache IndexError	e14_results.json → jit_cache.error	Confirmed

তিনটি সংখ্যাগত clarification:

প্রথমত, width6-এর +6.3% এবং width4-এর +5.3% এসেছে adopted graphs13 arm-এর 500.14 tok/s-এর তুলনায়, original W1-এর 505.0-এর তুলনায় নয়।
Comparison	Width6	Width4
Against graphs13 (500.14)	+6.26%	+5.33%
Against original W1 (505.0)	+5.23%	+4.32%

দ্বিতীয়ত, graphs13 adopt হয়েছে accept-normalized steps/sec criterion দিয়ে। কিন্তু raw committed tok/s আসলে কমেছে: short −0.96%, long −2.88%। তাই “graphs13 কাজ করেছে” বলা ঠিক; “graphs13 speed বাড়িয়েছে” বলা প্রমাণিত নয়।

তৃতীয়ত, raw_reasoning-এর kept count 115 হলেও A1-3_replay_E14P.summary-তে মাত্র তিনটি ছোট conversations পরীক্ষা হয়েছে; raw এবং normalized উভয়েই 2/3 full reply reuse। অর্থাৎ এই run raw-reasoning correctness regression দেখায়নি, কিন্তু cache-speed benefit establish করেনি।

আরেকটি limitation: ZIP-এ evaluation report আছে, কিন্তু সব labelled solution পুনরায় oracle-score করার পৃথক ground-truth file নেই। তাই 6/19 saved evaluator output-এর সঙ্গে যাচাই হয়েছে, আমার পৃথক truth-scoring নয়।
A2. Width6/4 gain কি noise-এর ওপরে? Correctness gate যথেষ্ট?
Verdict: UNSURE — throughput gain measured, robust improvement unproven।
- Width6: 75-second arm; 20 decode-log records।
- Width4: 75-second arm; 26 records।
- Width8 control: ভিন্ন server launch।
- Per-arm repeat অথবা width8→width6→width8 A/B/A নেই।
- একই session-এর A/A test-এ stochastic divergence হয়েছে।
A2-0_reference-এ serial 6টির মাত্র 2টি output identical; concurrent 12টির মধ্যে 5টি identical। Width6 gate PASS হলেও এটি মূলত zero large-margin flips-এর test, long ARC correctness proof নয়।

Correct fix/test: W8→W6→W8 অথবা randomized order, একই long-context input set-এ repeated benchmark। Minimum 30–40k context, আর অন্তত কিছু 60–100k-prefix sessions। Whole-response correctness ও verified ARC outputs মাপতে হবে।

বর্তমান তথ্য দিয়ে আমার ranking: W8 = proven baseline; W6 = promising experimental arm; W4 = weaker practical choice।
A3. Nodelay crash-এর root cause ঠিক? কীভাবে fix?
Verdict: GO for the root-cause diagnosis; NO-GO for untested nodelay deployment।

run_outputs/bundle/logs/sglang_server_e14_nodelay.log-এ চারটি TP rank-এর traceback একই:

dflash_worker_v2.py:278 → stage_sampling_params, tensor dimension 12 vs 13।

এটি static sampling buffer-এর capacity 12, কিন্তু running batch 13 হওয়ায় ঘটেছে। বর্তমান SGLang source-ও captured sampler buffers সর্বোচ্চ graph batch size দিয়ে allocate করার mechanism দেখায়। 

GitHub




দুটি সম্ভাব্য fix:
- nodelay + --cuda-graph-bs-decode 1..13: buffer size 13 হওয়ার সম্ভাবনা; কিন্তু CPU source reasoning দিয়ে production safety প্রমাণ হয় না।
- nodelay + --max-running-requests 12: 13-এর incompatible shape আটকাবে; তবে admission latch থাকলে 12 running পাওয়া যাবে এমন নিশ্চয়তা নেই।
আমার সিদ্ধান্ত: Tonight's submission-এ nodelay নয়। আগে bounded 13-request correctness + stability test, 30–100k context এবং no retractions নিশ্চিত করতে হবে।
A4. 11-running latch কি সত্যি? Duck-এ খরচ কত?
Verdict: GO — test-এ latch reproduced। কিন্তু exact causal attribution অনিশ্চিত।

e14_results.json:
- W1 churn: 114/114 samples-এ running=11; queued 97.4%।
- #44401: 114/114 running=11; queued 95.6%।
- nodelay: 83 samples-এর মধ্যে 30টি running=12; তারপর 13-এ crash।
এটি artificial churn workload-এ real observed admission bottleneck। তবে #44401 failed-to-fix মানেই upstream PR অর্থহীন নয়; আমাদের pinned config-এ অন্য delayer/capacity mechanism প্রভাব ফেলছে। 

GitHub




Duck-এর আসল 25-minute decode summary-তে:
Running	Clean median gen tok/s	Decode-log records
10	407.55	21
11	404.21	216
12	420.34	26

একই context ও acceptance-controlled A/B নয়। তাই এই run থেকে 11→12 করলে ~4% instantaneous gain দেখা যায়, কিন্তু whole-run gain নিশ্চিত নয়। আগে বলা +8–9% অন্য context-matched analysis থেকে এসেছে।

Recommendation: এই 3–8%-এর সম্ভাব্য occupancy gain-এর জন্য crash-risk নেওয়া নয়। Cache এবং first useful candidate-এর সমস্যা বেশি গুরুত্বপূর্ণ।
A5. Profiler: communication 59/116 ms সত্যি?
Verdict: GO — strong profiler evidence।

e14_results.json → phases.A1-2_baseline_W1.profile.during.summary:
Category	TP rank 0	Across four ranks (reported)
NCCL all-reduce	50.92 ms	50.92–51.27 ms
NCCL all-gather	8.11 ms	8.10–8.11 ms
Marlin INT4 GEMM	25.26 ms	প্রায় 25 ms
Other GEMM	14.50 ms	প্রায় 14.5 ms
GDN Triton	7.95 ms	প্রায় 8 ms
Kernel-duration sum	116.09 ms	115.96–116.09 ms
GPU busy-time union	112.96 ms	প্রায় 112.85–112.96 ms
Wall time	116.66 ms	প্রায় 116.7 ms

TP0-এর profiled kernel time-এর ভাগ


All-gather

All-reduce

GDN

Marlin

Other GEMM

Other kernels
















গুরুত্বপূর্ণ qualification: ZIP-এ raw compressed trace রয়েছে শুধু TP0-এর; বাকি তিন rank-এর summary archived JSON-এ আছে। TP0-এর raw trace 107,212 events ধারণ করে। চার rank-এর raw traces আবার independently parse করেছি—এমন দাবি করছি না।

GPU busy union ও wall-এর পার্থক্য প্রায় 3.7 ms/step, অর্থাৎ সম্পূর্ণ wall time-এর বিশাল অজানা CPU idle bubble পাওয়া যায়নি। Kernel-duration sum-এ প্রায় 3.1 ms overlap রয়েছে; তাই যোগফলকে perfectly additive bottleneck হিসেবে ধরবে না।

Conclusion: Communication bottleneck সত্যিই বড়। GDN-এর 8 ms দেখে 5–10 ms GDN-only saving আশা করা এখন যুক্তিযুক্ত নয়। Profiler-এ আরও দেখার মতো বিষয়: NCCL launch-level synchronization, other GEMM-এর drafter/lm-head breakdown এবং prefill interleaving।
A6. JIT-cache tar-এর IndexError কোথা থেকে?
Verdict: GO — probable source-level root cause পাওয়া গেছে।

code/build_e14.py:813–818, generated notebook CELL 7b line 17:
_q = subprocess.run([...], capture_output=True, text=True, ...)_v = json.loads(_q.stdout.strip().splitlines()[-1])



_q.stdout empty হলে .splitlines()[-1] সরাসরি IndexError('list index out of range') দেয়।

একই block-এর nvcc ও driver lines [-1:]/[:1] ব্যবহার করে—সেগুলো empty list-এ IndexError দেয় না। তাই এটি সবচেয়ে নির্দিষ্ট candidate root cause।

কিন্তু command কেন empty stdout দিয়েছে, সেটা জানা যায়নি; কারণ subprocess return code ও stderr report-এ রাখা হয়নি। কোনো pinned Python distribution name unavailable হওয়া অথবা import failure একটি সম্ভাবনা, নিশ্চিত কারণ নয়।

Minimal fix:
proc = subprocess.run(    command,    capture_output=True,    text=True,    timeout=180,    env=env,)if proc.returncode != 0 or not proc.stdout.strip():    raise RuntimeError(        f"JIT metadata failed rc={proc.returncode}: "        f"{proc.stderr[-1000:]}"    )metadata = json.loads(proc.stdout.strip().splitlines()[-1])



Test: Empty stdout, nonzero return code, stderr-only output এবং valid JSON success। Cache tar saved হলে তারপর আলাদা restore + version/hash compatibility + known-answer smoke test।

Priority: P2। Compile-cache speed improvement সম্ভাব্য, কিন্তু এখনো কার্যকর হয়নি।
A7. অন্য unnoticed bug, warning বা stall?
Verdict: GO for R2 diagnostic completeness, with residual risks।
- nodelay: four-rank runtime crash — confirmed, do not adopt।
- int8 Mamba: CUDA OOM ও 192 exception records — reject।
- SHM CUDA memcpy NCCL: benchmark arm timeout — reject।
- Profiler: no large unaccounted GPU-idle interval in measured window।
- Duck: infra_errors=0, stragglers=0, all 12 tasks responded।
- Cache: 5/102 later requests had zero-hit; four classified as prompt-prefix change, one unexplained।
- Partial hypotheses: 7 paused branches; 12/17 branches never checked a function।
- Submission output: completed, 6/19 reported; no evidence of invalid file in the collected run。
Notebook kernel log-এ MissingIDFieldWarning, Python SyntaxWarning-ও আছে; এগুলো বর্তমান failed server arms-এর মূল কারণ নয়।

অতিরিক্ত finding: int8_mamba server launch সফল হলেও execution OOM করেছে। তাই ভবিষ্যতে startup READY-কে serving stability-এর প্রমাণ হিসেবে ধরে কোনো memory-saving variant adopt করবে না।
B. Tonight's 240-task submission — চারটি প্রশ্ন
B1. আজকের notebook hidden rerun-এ ঠিক কী চালাবে?
Verdict: NO-GO for the stated W6+graphs13 plan; GO for the original W8 baseline configuration.

Source path:
1. CELL 0, line 17: KAGGLE_IS_COMPETITION_RERUN থেকে RERUN নির্ধারণ।
2. CELL 1, lines 2, 81–84: PROFILE="RERUN_ALL", challenge source hidden test।
3. CELL 1b: model/drafter staging ও E12_RAW_REASONING_HISTORY=1।
4. CELL 5, lines 162–183: SGLang W1-এর serving flags।
5. CELL 5b: E14_DO_MEASURE=False হলে engine measurement ও adoption arms চলে না।
6. CELL 6, lines 404–413: solve_all_r6() সব hidden tasks, coverage-first scheduler, two possible branches ও deadline-aware stopping।
7. CELL 0 / CELL 6: fallback এবং continuously updated submission.json।
Actual engine flags:
Flag	Currently baked into rerun	Proposed plan
TP	4	4
INT4 W4A16	Yes	Yes
DFlash2	Yes	Yes
Draft width	8	6
CUDA graphs	--cuda-graph-max-bs-decode 12	--cuda-graph-bs-decode 1..13
Maximum server running	13	13
Duck active streams	12	12
GDN/SSM	FP16	FP16
KV	FP8 E4M3	FP8 E4M3
Raw reasoning preservation	On	On
Nodelay	Off	Off
#44401	Off	Off
int8 Mamba	Off	Off

Minimal correction: যদি তোমরা W6+graphs13-ই বেছে নাও, তাহলে code/build_e14.py-এর generated CELL 5 e11_flags()-এ rerun-only flags apply করতে হবে। Duplicate flags append না করে existing values replace করবে।

Mandatory test: RERUN=True mock launch capture-এ exact server argv assert; RERUN=False commit configuration অপরিবর্তিত। Notebook rebuild-এর পরে SHA ও hidden-mode CPU harness PASS।
B2. Long-context-এ width6 নিরাপদ? W8 না W6 না W4?
Verdict: W8 — GO; W6 — CONDITIONAL; W4 — research arm only।

বর্তমান measured gain:
- W6: +5.23% বনাম W1 short-context।
- W4: +4.32% বনাম W1 short-context।
- Verification context প্রায় 5.9k/request।
- তোমাদের actual Duck history অনেক সময় 40–100k tokens।
Width6 trained width8 checkpoint-এর supported runtime override হলেও drafter-এর block-local bidirectional attention বদলায়। এটি শুধু শেষ দুটি speculative positions বাদ দেওয়া নয়।

SGLang-এ DFlash2 concurrency-related wrong-output এবং thinking-enabled divergence-এর public reports আছে। এগুলো exact RedHat/L4 stack-এ reproduced নয়, কিন্তু regression risk বাস্তব। 

GitHub
+1




আমার সিদ্ধান্ত: deadline-sensitive competition rerun-এর জন্য W8 baseline বেশি defensible। W6-এর 5% short-context gain-এর জন্য unmeasured long-context quality risk নেওয়া বাধ্যতামূলক নয়।

যদি W6 ব্যবহার করতেই চাও, minimum validation হবে একই session-এ W8 বনাম W6, 40–60k retained prefix, অন্তত 12-way concurrency এবং target-only paired output checks। Passing short 256-token greedy gate long reasoning equivalence প্রমাণ করে না।
B3. 240 task কি 12 ঘণ্টায় fit করবে? Zero-score paths আছে?
Verdict: CONDITIONAL GO — deadline architecture ভালো, কিন্তু actual coverage/accuracy unproven।

Clock: RERUN_ALL budget 12 ঘণ্টা থেকে 3 মিনিট বাদ। তার পরে hard-stop margin 2 মিনিট এবং solver stop-before-hard আরও 3 মিনিট।

অর্থাৎ solver-এর theoretical deadline প্রায় kernel start-এর 11 ঘণ্টা 52 মিনিট পরে। Startup, wheel install ও load এই সময়ের মধ্যেই পড়ে।

তোমাদের 502-second cold startup হলে actual solve সময় প্রায় 11 ঘণ্টা 44 মিনিট—অন্যান্য setup overhead বাদে।
Compute capacity
28k-token soft first-exposure target ধরে:
\[
240\times 28{,}000=6.72\text{ million tokens}
\]

Sustained effective rate	240×28k token generate করতে
180 tok/s	~10.37 ঘণ্টা
220 tok/s	~8.48 ঘণ্টা
280 tok/s	~6.67 ঘণ্টা
350 tok/s	~5.33 ঘণ্টা

এগুলো model generation-এর arithmetic, সম্পূর্ণ scheduler wall-time নয়। Request-boundary overshoot, prefill, verification, startup এবং task-length variation অতিরিক্ত খরচ।

R2 Duck-এর actual 430,062 tokens/1,500-second solve window ≈ 287 tok/s। যদি full hidden run-এ এটি বজায় থাকে, 240-task first exposure সম্ভব। কিন্তু বেশি live context, cache misses ও long-tail task দিয়ে throughput কমতে পারে।
Zero-score বা invalid-output failure paths
Failure	Current protection	Residual risk
Solver exception, model server failure after CELL 0	Identity submission.json exists	Valid but near-zero score
INT4/drafter mount missing	Model discovery এবং load-path fallback	No useful solve if startup never READY
Model load stalls	Launch give-up / watchdog	Remaining solve time shrinks
SGLang smoke/capacity failure	Server rejected; fallback file retained	Duck does not run
Python error before CELL 0's fallback write	No fallback yet	Missing/invalid submission possible
Tool/branch errors during solving	Host verification, saved candidates, deadline	Partial task coverage
Hard-stop	Early stopping and atomic submission	A blocked exceptional path still needs runtime monitoring

বিশেষ করে source-এর CELL 5:638–675-এ SGLang failed হলে ENGINE_READY=False হয়। CELL 6:340–435 তখন Duck solve skip করে।

এটি safety fallback, কিন্তু successful competitive fallback নয়। Notebook-এর legacy E11_ALLOW_VLLM_FALLBACK setting থাকলেও বর্তমান Duck hidden path-এ স্বয়ংক্রিয় full vLLM solver recovery প্রমাণিত হয়নি।

Launch preflight: exact datasets mounted, model config.json, model_mtp.safetensors যেখানে প্রয়োজন, staged target/draft আলাদা, SGLang READY, full capacity smoke, 240 challenge IDs, submission fallback schema এবং watchdog—সব যাচাই।
B4. এক দিনে এর চেয়ে বড় leaderboard improvement কোথায়?
Verdict: একদিনে সবচেয়ে বড় proven lever নতুন 5% speculation speed নয়; working model থেকে more useful hypotheses পাওয়া।

R2 Duck-এর actual task breakdown:
Solved task	Correct test outputs	Source
247ef758	2	Submitted verified program
65b59efc	2	Harvested check() candidate, no submit
b5ca7ac4	1	Harvested candidate
bf45cf4b	1	Verified submissions
Total	6/19	eval_unit_records

এখানে Rev5-এর সবচেয়ে গুরুত্বপূর্ণ fix—submit না-করা train-exact candidate harvest করা—সরাসরি correct outputs উদ্ধার করেছে।

কিন্তু একই run-এ:
- 17 branches-এর 12টিতে কখনো check() আসেনি।
- মাত্র 5 branches transform/check পর্যন্ত পৌঁছেছে।
- Branch-1 থেকে 0টি task-level নতুন distinct verified prediction।
- 8টি provisional nudges emit হয়েছে, 5টি model দেখেছে, এবং পরে মাত্র একটি accepted candidate এসেছে।
- 12/12 genuine first exposure হয়েছে।
অর্থাৎ সবচেয়ে বড় remaining bottleneck হচ্ছে earlier useful transform generation, শুধু serving TPS নয়।

আমার একদিনের priority হবে existing approved code-first mechanism-এ 12-task transcript-driven ছোট correction, নতুন orchestration নয়। তবে এমন পরিবর্তন GPU যাচাই ছাড়া hidden rerun-এ লাগানোও ঝুঁকিপূর্ণ।

আরেকটি গুরুত্বপূর্ণ decision: তোমাদের best recorded LB 34.58, কিন্তু আগের Duck Rev5 LB 17.50। R2-এর 6/19 নির্বাচিত evaluation-এ উন্নতি আশাব্যঞ্জক হলেও Duck Rev6g hidden leaderboard 34.58 ছাড়াবে—এমন verified evidence নেই। তাই exploration submission ও final best-scoring choice আলাদা রাখতে হবে।
C. E15 speed experiment — তিনটি প্রশ্ন
C1. Arm order সঠিক? কিছু বাদ বা অসম্ভব?
Verdict: GO for E15 design, with a small priority reorder।

তোমাদের proposed sequence:

NCCL sweep → DFlash2 baseline → MTP 3/1/4 → MTP + ReplaySSM → 14/16 running → chunk2048 → conditional 2×TP2

বড় architecture change ছাড়াই testing-এর জন্য গ্রহণযোগ্য। তবে তোমাদের recent profiler অনুযায়ী NCCL SHM baseline সেরা; P2P 5.5× ধীর। তাই বড় NCCL parameter sweep-এর চেয়ে MTP comparison-এর GPU budget বেশি মূল্যবান।

আমার preferred sequence:
1. DFlash2-W8 reference: same-session short ও long context, throughput, accept length, KV/Mamba এবং target-only correctness snapshot।
2. MTP 3/1/4: একই workload, identical sampling, exact per-request capacity metrics; baseline-এর চেয়ে ভালো হলে এগোবে।
3. MTP 2/1/3: যদি MTP3 short ও long-এ mixed result দেয়; lower verification cost-এর arm।
4. ReplaySSM, then C12/C14/C16: শুধু MTP load/correctness/memory preflight PASS করলে।
5. Chunked prefill 2048: সবচেয়ে ভালো stable arm-এ, effective tok/s ও TTFT দুটো মেপে।
6. Optional NCCL tuning এবং 2×TP2 feasibility: সময় থাকলে; TP2 launch আগে KV/Mamba capacity test।
কী বাদ দেব: নতুন model, PP4 migration, unsupported adaptive speculation অথবা unverified IPC/custom-all-reduce patches।

Public SGLang documentation MTP ও DFlash2 দুটোর launch recipe দেয়; ReplaySSM-এর memory benefitও documented। কিন্তু official main-branch recipe pinned v0.5.21-এর complete GPU compatibility guarantee নয়। 

GitHub
+1



C2. MTP 3/1/4 flags সঠিক? RedHat INT4 checkpoint-এ head আছে?
Verdict: GO for GPU preflight; actual MTP serving correctness UNSURE।

Official recipe:
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4


এটির জন্য পৃথক DFlash2 draft-model path দেওয়া হয় না। SGLang 0.5.21-এ qwen3_5_mtp.py রয়েছে। 

GitHub
+1




https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/qwen3_5_mtp.py

RedHat model evidence: bundled speed review অনুযায়ী target checkpoint-এর সঙ্গে পৃথক প্রায় 0.85 GB model_mtp.safetensors রয়েছে। RedHat INT4 config-এর quantization ignore list-এ re:^mtp.* আছে; অর্থাৎ MTP head intentionally BF16 রাখা হয়েছে বলে local research reports জানিয়েছে।

এটি MTP feasibility-এর পক্ষে শক্তিশালী evidence। কিন্তু E14-A R2 ZIP-এ পুরো model weight files নেই; তাই tensor-by-tensor load স্বাধীনভাবে যাচাই করতে পারিনি।

GPU preflight-এর requirements: checkpoint-এ actual MTP tensors present, missing/unexpected weights নেই, EAGLE3/1/4 initialization successful, normal 12-running, acceptance reported, no OOM, no retractions এবং target-only correctness gate।

আরেকটি relevant warning: vLLM issue #55533-এ hybrid GDN + MTP high-batch serving-এ concurrency collapse রিপোর্ট হয়েছে। এটি SGLang 0.5.21-এর confirmed bug নয়, কিন্তু C12/14/16 test কেন জরুরি তার ভালো কারণ। 

GitHub




https://github.com/vllm-project/vllm/issues/55533
C3. 4×L4-এ realistically কত tok/s পাওয়া সম্ভব?
Verdict: measured baseline clear; নতুন arms-এর exact gains এখনো unknown।
Configuration	Short context, C12	Long context	Evidence
DFLASH W8 TP4	505 tok/s	450 tok/s at ~8.7k	Measured
DFLASH W6 TP4	531 tok/s	পাওয়া যায়নি	Measured short
DFLASH W4 TP4	527 tok/s	পাওয়া যায়নি	Measured short
Target-only TP4	340 tok/s	পাওয়া যায়নি	Measured short
MTP 3/1/4 TP4	C12 measurement পাওয়া যায়নি	পাওয়া যায়নি	Next experiment
MTP + ReplaySSM	পাওয়া যায়নি	পাওয়া যায়নি	Next experiment
MTP C14/C16	পাওয়া যায়নি	পাওয়া যায়নি	Next experiment
2×TP2 + MTP	পাওয়া যায়নি	পাওয়া যায়নি	Capacity/throughput unknown

একটি conditional throughput model সাহায্য করবে:
\[
T=\frac{\text{running requests}\times\text{mean accepted length}}{\text{step time}}
\]

C12-এ MTP যদি mean acceptance 3.0 এবং step time 75 ms পায়, তাহলে ~480 tok/s। Acceptance 3.5 এবং step time 75 ms হলে ~560 tok/s। 600 tok/s পেতে C12-এ 75 ms step-এর জন্য acceptance 3.75 দরকার।

এগুলো outcome scenarios, predictive benchmark নয়।

600–800 tok/s goal: MTP C14/16 আর substantial communication reduction ছাড়া C12-এ 800 পাওয়া কঠিন। কিন্তু current L4 memory ও GDN-state constraints-এর কারণে C16 সম্ভব হবে কি না GPU preflight ছাড়া বলা যাবে না।

D. প্রতিটি গুরুত্বপূর্ণ fix — কাজ করেছে, ব্যর্থ, নাকি এখনো অজানা?
এই matrix-ই তোমাদের প্রশ্নের সবচেয়ে ব্যবহারযোগ্য অংশ।
Fix / experiment	Verdict	Measured result	পরের কাজ
E14 parallel staging	Worked for reliability	23.32 GB staged, validated, separate target/draft; 124s	Keep; startup wall speed gain দাবি নয়
Startup watchdog / fallback	Worked in successful run	W1 ready ~502s, relaunch ~61–76s	Mount-failure test; identity fallback warning
JIT-cache export	Failed	IndexError, no tar	Fix empty metadata stdout; CPU test
NCCL baseline benchmarking	Worked	Graph AR 343.9 µs; all-gather 8.02 ms	Keep evidence
NCCL forced P2P/SYS	Rejected	1,880.5 µs, ~5.47× slower	Do not use
SHM CUDA memcpy	Failed	Timed out	Stop this arm for now
Profiler	Worked	~59 ms/step communication	Optimize based on measured split
#44401 patch	No useful gain	508 vs 505; still running=11 under churn	Do not adopt alone
Nodelay	Unsafe	Running 13 triggers buffer mismatch crash	No production deployment
CUDA graphs 1..13	Mechanically worked, no TPS gain	500 vs 505 short, 437 vs 450 long	Optional; not justified for LB yet
DFlash2 width6	Short-context improvement	531 vs 500 adopted control	Long-context gate before production
DFlash2 width4	Smaller conditional improvement	527 vs 500 control	Lower priority
Target-only comparison	Diagnostic worked	340 vs W8 505	Keep for correctness
int8 Mamba	Failed	OOM, 192 exceptions	Reject at mem 0.83
Raw reasoning preservation	Mechanically worked	115 kept; 0 reported mismatch	Benefit to cache/score still unproven
Cache replay diagnostic	Worked	67/120 reply reused; 26 lost at prefill end; 27 partial	Next: token-prefix + Mamba checkpoint analysis
Host-clean check() harvesting	Demonstrably useful	3/6 correct outputs came from check-only branches	Preserve
Pass@2 candidate selection	No incremental gain here	6/19 all attempt_1; attempt_2 added 0	Need distinct hypotheses, not remove selector
Facts + indexed grid helpers	Tool access improved	Median first tool 80 tokens	Accuracy effect not isolated
Provisional nudge	Weak evidence	Seen in 5 branches; 1 later accepted	No stronger nudges without tests
First-exposure scheduler	Worked	12/12 received model responses	Keep
Conditional b1 / stage 2	No task-level benefit here	b1 10,976 tokens, 0 new predictions	Reassess allocation before hidden
KV-only admission	No false step-down	12 workers; no KV holds needed	Keep
Deadline / atomic fallback	Worked this run	Zero stragglers, valid reported output	Keep + failure-path preflight
R6 candidate reporting / telemetry	Worked	5 source-verified programs, 4 distinct predictions	Keep; don't confuse programs with predictions

সবচেয়ে বড় accuracy lever বাস্তবে কোনটি? R2-তে check-only harvesting তিনটি correct outputs উদ্ধার করেছে। অর্থাৎ এটি confirmed useful fix। অন্যদিকে CUDA graphs 13-এর মতো infrastructure change implementation success হলেও measurable score বা raw throughput লাভ দেখায়নি।

আর R2 6/19 বনাম Rev6d G4B 2/19: observed improvement যথেষ্ট বড়, কিন্তু experiment এক-variable A/B নয়। Different solver revision, run window, sampling ও serving conditions থাকায় এটিকে কোনো এক fix-এর causal gain বলা যাবে না।

E. পরবর্তী কাজের অগ্রাধিকার
Priority	কাজ	প্রত্যাশিত মূল্য	GPU cost
P0	Hidden rerun-এর exact profile/flags freeze ও CPU-harness test	Wrong arm push আটকাবে	CPU only
P0	Decide W8 stable বনাম W6 unproven; preserve last good LB submission	Competition risk কমাবে	CPU decision
P1	MTP 3/1/4 vs W8, C12, 40k+ contexts	Potentially largest engine gain	E15 GPU
P1	Token-prefix/Mamba cache-miss diagnosis	Agent-effective TPS বাড়ার সুযোগ	CPU first + bounded GPU
P1	Earlier transform/check, second-branch diversity	Accuracy per token বাড়ানো	CPU replay + small GPU
P2	JIT-cache tar fix	Cold startup saving	CPU first
P2	MTP ReplaySSM, 14/16 running	MTP successful হলে next-step gain	E15 GPU
P3	2×TP2, NCCL tuning, broad kernel changes	High risk, uncertain benefit	Later

Final GO / NO-GO
GO — E15 development and bounded GPU measurement

MTP vs DFlash2 পরীক্ষা যথেষ্ট evidence-backed। Official SGLang recipe supported, model MTP weights থাকার evidence আছে; actual long-context correctness ও memory fit পরীক্ষা বাধ্যতামূলক।

NO-GO — Declared W6 + graphs13 hidden rerun with the current exact notebook

কারণ current rerun code W8 + graph-max12 চালাবে। Intentional configuration mismatch।

CONDITIONAL GO — W8 baseline hidden rerun

Exact notebook ও Kaggle dataset mounts verified থাকলে current flags সঙ্গতিপূর্ণ। তবে 240-task full accuracy, stable long-context decode এবং 34.58 best LB অতিক্রম করা এখনও প্রমাণিত নয়।

আমার চূড়ান্ত সিদ্ধান্ত: এখন সবচেয়ে বেশি কাজ দেবে verified program generation + cache efficiency, তার পরে MTP high-concurrency optimization। বর্তমান W1 engine-এর 59 ms communication bottleneck বাস্তব, কিন্তু unsafe P2P/nodelay দিয়ে সেটা সমাধানের চেষ্টা করা উচিত নয়। আর tonight's competition run-এ পরীক্ষামূলক 5% short-context speed gain-এর চেয়ে valid 12-hour inference এবং correct task coverage বেশি মূল্যবান।
External sources verified
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/qwen3_5_mtp.py

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/dflash_worker_v2.py

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/kv_cache_configurator.py

https://github.com/sgl-project/sglang/pull/44401

https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx

https://github.com/sgl-project/sglang/issues/36548

https://github.com/sgl-project/sglang/issues/38009

https://github.com/vllm-project/vllm/issues/55533
