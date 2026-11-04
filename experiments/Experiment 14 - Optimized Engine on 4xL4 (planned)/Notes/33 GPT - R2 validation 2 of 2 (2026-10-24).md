<!-- Pasted by the user on 2026-10-24 (GPT validation of the R2 zip GPT_VALIDATE_E14A_R2_2026-10-24.zip). Kept VERBATIM. Claude's validation: Notes/34. -->

E14-A R2 Independent Validation — ফলাফল, Fix Audit ও পরবর্তী কাজ
প্রথমে পড়েছি: 00_GPT_PROMPT.md। এরপর notes/30, notes/25, notes/29, R2-এর raw JSON, server logs, notebook ও Python source মিলিয়েছি। বাইরের SGLang documentation-ও যাচাই করছি।
সবচেয়ে জরুরি ফল: আজকের 240-task rerun plan-এ একটি বাস্তব configuration blocker
NO-GO — পরিকল্পিত width6 + graphs13 rerun as-is
কারণ: আপলোড করা notebook-এর commit-mode-এ width6 এবং graphs13 পরীক্ষা ও adoption থাকলেও competition rerun mode-এ সেগুলো স্বয়ংক্রিয়ভাবে কার্যকর হচ্ছে না।

নির্দিষ্ট code evidence:
- code/build_e14.py:133 — E14_DO_MEASURE = bool(P.get("e14")) and not RERUN
- একই ফাইলের lines 492–685 — graphs13 পরীক্ষা/adoption শুধু E14_DO_MEASURE branch-এ।
- Lines 760–773 — width6/4 adoption শুধু A2 measurement section-এ।
- Notebook CELL 5, e11_flags() — default --speculative-num-draft-tokens 8, --cuda-graph-max-bs-decode 12।
- Notebook CELL 1 — RERUN_ALL profile নির্বাচন করা হলেও engine defaults override হচ্ছে না।
ফলে এই exact notebook rerun দিলে পরিকল্পিত width6 + graphs13 নয়, W1 width8 + graphs12 চালু হওয়ার কথা।

এটি fallback-এর কারণে submission শূন্য হয়ে যাবে—এমন bug নয়; কিন্তু যে configuration-এর জন্য অনুমোদন চাওয়া হচ্ছে, সেই configuration বাস্তবে চলছে না। আগের audit-গুলোর অভিজ্ঞতা বিবেচনায় এটিকে pre-submission blocker হিসেবে ধরছি।

Minimal fix: শুধু RERUN_ALL-এর engine flag construction-এ explicit width ও graph list বসাতে হবে। Measurement/adoption runtime state থেকে কোনো flag উত্তরাধিকার হিসেবে নেওয়া যাবে না। তারপর generated notebook-এর fresh SHA এবং fake-server rerun test-এ actual launch argv যাচাই করতে হবে।

অন্যদিকে, width6 আদৌ production-এর জন্য সঠিক পছন্দ কি না, সেটি raw evidence অনুযায়ী এখনও চূড়ান্ত নয়। সেই সিদ্ধান্ত নিচে আলাদাভাবে বিচার করছি।
A. R2-এর সাতটি প্রশ্নের উত্তর
A1. notes/30-এর সংখ্যাগুলো সঠিক?
মূল সংখ্যাগুলো মেলে
run_outputs/e14_results.json, bundle/e14/e14_report.md, q27x/e12_run_report.json এবং সংশ্লিষ্ট server logs থেকে:
Configuration	Short tok/s	Step ms	Accept length	Verdict
W1 width8	505.00	114.02	4.86	Confirmed
#44401	508.14	114.25	4.84	Confirmed
nodelay	490.32	114.38	4.63	Confirmed
graphs13	500.14	114.69	4.79	Confirmed
Target-only	339.55	35.34	—	Confirmed
DFlash width6	531.43	96.39	4.24	Confirmed
DFlash width4	526.81	74.50	3.27	Confirmed

কিন্তু চারটি গুরুত্বপূর্ণ interpretation correction দরকার।

প্রথমত, width6-এর +6.3% হলো graphs13 baseline 500.14 tok/s-এর তুলনায়। Original W1 505 tok/s-এর তুলনায় gain 5.23%। Width4-এর +5.3% একই graphs13 baseline-এর তুলনায়; original W1-এর তুলনায় প্রায় 4.32%।

দ্বিতীয়ত, long benchmark বলতে 40k context বোঝাচ্ছে না। Raw ctx_per_req_p50 W1-এ 8,761 tokens; width6/4-এর জন্য কোনো long-window test-ই হয়নি।

তৃতীয়ত, profiler-এর numerical summary চার TP rank-এর জন্য আছে; কিন্তু ZIP-এ শুধু TP0-এর raw trace। তাই TP0 সরাসরি verified, বাকি তিনটির summary internally consistent হলেও raw cross-rank validation অসম্পূর্ণ।

চতুর্থত, 6/19 score e12_run_report.json → eval_unit_records[].hit থেকে মেলে। ZIP-এ independent gold-label adjudication সম্পূর্ণ নেই, তাই এটি recorded evaluation score-এর verification; নতুন করে official gold re-grading নয়।
A2. Width6 এবং width4-এর gain কি noise-এর উপরে?
Verdict: UNSURE — promising কিন্তু conclusive নয়।

Same session-এ W1 505, #44401 508, graphs13 500 tok/s। এই তিনটির spread প্রায় 1.6%। Width6-এর ~6% improvement তার চেয়ে বড়।

তাই এটিকে নিছক numerical rounding বা একদম trivial noise বলা যাবে না।

কিন্তু প্রতিটি width একবার cold/relaunch configuration-এ মাপা হয়েছে। Repeated A/B/A নেই। Accept length boot ও input distribution-এ বদলায়। তাই 6% gain production-এ reproduce হবে—এমন confidence এখন নেই।

আর correctness gate-এ:
- A2 concurrency A/A: মাত্র 5/12 exact-identical।
- Width6: 4/12 exact-identical।
- Width4: 3/12 exact-identical।
- Large-margin flips: 0।
এটি high-confidence disagreement না-পাওয়ার evidence, long-reasoning correctness-এর proof নয়।

Public SGLang issue #36548-এ concurrency 8-এ DFlash2-এর 111/304 affected outputs ভুল হওয়ার measured reproduction রয়েছে। Width4-এও 37/304 ভুল ছিল। আপনার model/quantization আলাদা—সমস্যাটি আপনার stack-এ ঘটছে প্রমাণিত নয়, কিন্তু ছোট parity gate দিয়ে বাদ দেওয়া যায় না। 

GitHub




Fix/test: Width8 বনাম 6-এর matched 12-stream, 40k-context benchmark এবং reasoning-on 1,024-token target-only comparison। 12/12 requests complete, OOM নেই, no high-confidence divergence, এবং measured gain বজায় থাকলে width6 গ্রহণযোগ্য।
A3. nodelay crash-এর root cause কী?
Confirmed crash; root-cause hypothesis strongly supported
sglang_server_e14_nodelay.log:870–1000-এ:
- Sampling buffer-এর shape 12।
- Actual batch 13।
- dflash_worker_v2.py:273-এর warning।
- stage_sampling_params, line 278-এ 12 বনাম 13 tensor mismatch।
- চার TP rank-এ RuntimeError।
এটি 12-size allocation এবং 13 running admission-এর mismatch।

--cuda-graph-max-bs-decode 12 probable allocation trigger, কিন্তু কেবল 13 capture যুক্ত করলেই সব DFlash sampling buffers ঠিকভাবে 13-size হবে—সেটি এখনও measured নয়।

দুটি test candidate:
Candidate	Safety judgement
nodelay + --max-running-requests 12	তুলনামূলকভাবে সহজ; server কখনো 13 admit করবে না
nodelay + graphs13	সম্ভাব্য fix; combined sampling/verify graph test অপরিহার্য

প্রথমটির negative test: ১৪ arrivals দিয়ে sustained churn, running_max <=12, no mismatch, no OOM এবং running=12 ধরে রাখার হার।

দ্বিতীয়টির negative test: 13 simultaneous decodes দিয়ে অন্তত কয়েকশো steps, তারপর mixed prefill + decode, no invalid buffer shape। শুধু graph capture success যথেষ্ট নয়।

আজকের submission-এ nodelay বাদ রাখার সিদ্ধান্ত সঠিক।
A4. Delayer সত্যিই ১১-তে আটকে আছে?
Verdict: YES, in this controlled churn workload।

e14_results.json → phases.*.latch:
Arm	11-running samples	12-running samples	Result
W1	114	0	Stuck at 11
#44401	114	0	একই আচরণ
nodelay	53	30	12-তে উঠেছে, কিন্তু পরে crash

এটি robust evidence যে #44401 backport আপনার measured workload-এ occupancy improvement দেয়নি।

তবে Duck-এর প্রকৃত ক্ষতি আলাদা মাপতে হবে। Steady-state benchmark-এ ১২টি request একই সময়ে শুরু হলে server ঠিক ১২টিতেই চলে।

Churn-এ 11 বনাম 12 হলে concurrency slot loss সর্বোচ্চ প্রায় 8.3%, কিন্তু aggregate tok/s loss সরলভাবে 8.3% হবে না—batch efficiency বদলায়।

আমার estimate: প্রায় 0–10% agent throughput opportunity, workload-dependent। এটি measured performance loss নয়।
A5. Profiler: communication 59/116 ms সত্য?
Verdict: YES for TP0, independently reproduced।

আমি archived TP0 trace-এর actual kernel events যোগ করেছি।
Category	Calls/step	TP0 ms/step
NCCL all-reduce	140	50.919
NCCL all-gather	3	8.107
Total communication	143	59.027
Marlin GEMM	512	25.257
All kernel durations	—	116.036

এর সঙ্গে supplied summary-এর gpu_busy_ms_per_step = 112.96, wall_ms_per_step = 116.66 মেলে।

সুতরাং বড় unexplained 20–30ms idle gap পাওয়া যায়নি।

একটি গুরুত্বপূর্ণ interpretation: GPU profiler-এর NCCL kernel duration-এর মধ্যে synchronization wait থাকতে পারে। 59ms মানে communication-related GPU time; pure bytes-on-wire duration নয়।

Kernel profile অনুযায়ী next priorities: NCCL/all-gather, Marlin এবং other GEMM। GDN প্রায় 8ms; এখন GDN rewrite বড় lever নয়।
A6. JIT cache tar-এর IndexError কেন?
Verdict: Source bug location confirmed; underlying subprocess failure unknown।

code/build_e14.py:813–818:
_q = subprocess.run(    [...],    capture_output=True,    text=True,    timeout=180,    env=E14_VENV["env"],)_v = json.loads(_q.stdout.strip().splitlines()[-1])



stdout empty হলে splitlines()[-1] ঠিক সেই IndexError('list index out of range') দেয়।

এটি confirmed unsafe parsing। কিন্তু child process-এর stdout খালি থাকার মূল কারণ—failed package metadata, torch import, CUDA initialization, অথবা অন্য exception—ZIP-এ stderr সংরক্ষিত না হওয়ায় নিশ্চিত করা যায়নি।

Minimal patch:
if _q.returncode != 0:    raise RuntimeError(        f"JIT metadata probe failed rc={_q.returncode}: "        f"{_q.stderr[-1000:]}"    )lines = _q.stdout.strip().splitlines()if not lines:    raise RuntimeError(        "JIT metadata probe returned empty stdout; "        f"stderr={_q.stderr[-1000:]}"    )_v = json.loads(lines[-1])



Validation: zero stdout, nonzero exit, invalid JSON, successful JSON এবং missing metadata package—পাঁচটি cases। মূল failure reason না জেনে install/package পরিবর্তন করবেন না।

এটি non-blocking: compile-cache artifact তৈরি হয়নি, কিন্তু solver evaluation valid।
A7. Logs-এ আরও কী পাওয়া গেল?
দুটি উল্লেখযোগ্য observation।

sglang_server_e14_width6.log:215,232–234-এ Triton kernels serving শুরু হওয়ার পরে device-loaded হয়েছে। Log-এর warning অনুযায়ী available memory প্রায় 0.89 GiB। Late kernel loading OOM risk তৈরি করতে পারে।

আর width6/width4 এবং অন্যান্য launches-এ FP8 KV scaling factors absent হওয়ায় 1.0 defaults ব্যবহারের warning আছে। এটি নতুন bug নয়, কিন্তু numerically calibrated FP8 KV নয়।

অন্যদিকে সব Traceback string runtime failure নয়। প্রায় সব server log-এ Error in sitecustomize; set PYTHONVERBOSE for traceback warning আছে। এটিকে actual application exception গণনা করা ভুল হবে।

সত্যিকারের exception:
- nodelay: 13-running tensor mismatch।
- int8 Mamba: CUDA OOM, distributed worker teardown-এর ফলে বহু subsequent exceptions।
- Ordinary width6/width4: supplied phase-এর real windows-এ নতুন fatal exception পাওয়া যায়নি।

Fix-by-fix audit: কোনটা কাজ করেছে?
এই table-এ E14-এর original 16 fixes, Build 2 corrections এবং Rev 6g-এর গুরুত্বপূর্ণ behavioral fixes একসঙ্গে দিলাম। Mechanism কাজ করা আর score/speed বাড়া আলাদা।
Fix	Mechanism	R2 outcome	Verdict
HTTP deadline clamp	Strict timeout	Phase boundaries respected; no fatal overrun	Worked
Late parity request abort	Cleanup	Windows-এ alive_threads=0	Worked
wait_level plateau	Actual occupancy check	12-running windows achieved	Worked
A/A noise-floor gate	Divergence comparison	Incomplete/crashed arm rejected; long correctness unknown	Partial
Generated-cell syntax guard	Notebook compilation	Notebook executed to completion	Worked
Profiler lightweight settings	Profiling	Four-rank summary recorded; TP0 raw verified	Worked
NCCL microbench gate	Reject slow transport	Baseline best; p2p_sys 5.47× slower	Worked as rejection gate
Nodelay adoption	Higher occupancy	Reached 12 but crashed at 13; not adopted	Unsafe arm correctly rejected
/start_profile plain text handling	Endpoint parse	Profiler worked	Worked
Server-log counters	Detect errors	nodelay 4 rank errors, int8 OOM recorded	Worked with warning-filter limitations
/flush_cache response handling	Clean benchmark	Flush success text recorded	Worked for tested path
Graphs13 capture validation	Capture sizes	12/13 captured; only 12 actually reached	Mechanically worked, speed not improved
CR/build guard	Valid notebook	R2 ran	Worked
Watchdog/staging diagnostics	Startup/relaunch	502s cold, ~61–76s relaunch	Worked; startup still slow
Fake-server flush tests	Failure simulation	CPU-only evidence, GPU success response aligned	Test coverage worked
Platform timing robustness	Stable unit tests	Recorded 125/125 repeated; not GPU benchmark	CPU-only validation
Parity incomplete fail-closed	Reject insufficient comparisons	Int8 0/12 rejected	Worked
A1 Duck reserve	Protect solver time	duck_window.status=full, 1,500s	Worked
12-stream latch metric	Detect 11-running issue	Reproduced 114/114	Worked
Cross-arm int8 parity	Reject unsafe arm	Int8 crashed before full parity	Safe rejection; quality untested
Rejected-arm rollback	Restore last adopted	Duck ran on graphs13, not crashed nodelay	Worked
Raw reasoning E14-P	Retain reply text	115 kept, 0 mismatch	Worked mechanically
Rev 6g KV-only gate	No false worker reduction	Workers=12, KV holds=0	Worked under observed load
Rev 6g coverage priority	Task exposure	12/12 task b0 started	Worked on G4B
Rev 6g tool-result nudge	Candidate-first intervention	8 sent, 5 seen, 1 subsequent transform/check	Partially effective
Rev 6g code-first/compact view	Earlier hypothesis	Only 5/17 branches reached check()	Behavioral target largely unmet
Rev 5 candidate harvest/pass@2	Preserve verified programs	5 unique programs, 4 distinct predictions; correct check-only candidates retained	Worked
Rev 6g progress-aware stage2	Additional refinement	10 stage2 runs, 3 improved	Some productive allocation
Rev 6g cache instrumentation	Diagnose lost reuse	Cache 82%, 5 later zero-hit requests	Worked as diagnostic
JIT-cache packaging	Save reusable compile cache	IndexError; no tar	Failed
Width6 adoption	Faster short-context engine	+6.3% vs graphs13	Promising; long contexts unverified
Width4 adoption	Faster short-context engine	+5.3% vs graphs13	Promising; long contexts unverified

সবচেয়ে গুরুত্বপূর্ণ solver-side finding
Correct outputs
6/19
Recorded evaluation

Never checked
12/17
Branches

Tool-result nudge seen
5/8
Of emitted nudges


R2-এর 6/19 হলো Rev 6d G4B-এর 2/19 থেকে উন্নতি, কিন্তু একটি stochastic run-এর ফল। Rev 5 low-এর 10/19-এর নিচেও রয়েছে। তাই Rev 6g নিশ্চিতভাবে আগের সেরা solver হয়েছে—এমন সিদ্ধান্ত দেওয়া যায় না।

Raw rev6.metrics থেকে first-transform median (observed) 20,654 tokens, কিন্তু সেটি transform লেখা branch-গুলোর ওপর selection-biased median। ১২টি branch কখনো check করেনি—এই তথ্যটি বেশি গুরুত্বপূর্ণ।

Cache-এর 82% ভালো হলেও A2-1_agent_replay_default-এ 120 follow-ups-এর 67টি full reply reused, 26টি prefill-boundary-তে cache reuse হারিয়েছে, 27টি partial। এগুলো cache-token reuse classes; Python reasoning text সম্পূর্ণ হারিয়ে যাওয়ার সমান নয়।

B. আজকের competition submission-এর চারটি প্রশ্ন
B1. Exact rerun code path
Verdict: NO-GO for the stated width6+graphs13 plan.

বর্তমান path:
CELL 0: KAGGLE_IS_COMPETITION_RERUN = true
    ↓
CELL 1: PROFILE = RERUN_ALL
    ↓
CELL 1b: E14_DO_MEASURE = false
    ↓
CELL 5: e11_flags() uses W1 defaults
        DFLASH width8
        cuda-graph-max-bs-decode 12
        max-running-requests 13
    ↓
CELL 5b: All adoption arms skipped
    ↓
CELL 6: Rev 6g Duck over 240 hidden tasks
    ↓
CELL 7: Build submission from verified candidates


Exact fix: In code/build_e14.py, regenerate the notebook with rerun-specific engine flags applied before the initial SGLang launch—not in CELL 5b adoption logic। Preserve commit-mode measurements unchanged।

For intended width6+graphs13:
# RERUN_ALL SGLang engine branch only# Replace:"--speculative-num-draft-tokens", "8"# With:"--speculative-num-draft-tokens", "6"# Replace:"--cuda-graph-max-bs-decode", "12"# With:"--cuda-graph-bs-decode","1", "2", "3", "4", "5", "6", "7","8", "9", "10", "11", "12", "13"



এটি conceptual splice; final builder-এ argument-list parser দিয়ে duplicate/conflicting graph flags বাদ দিতে হবে।

Required negative test: Fake competition rerun-এ E14_DO_MEASURE=false, তারপর actual SGLang launch argv assert:
assert draft_tokens == 6assert graph_batches == list(range(1, 14))assert "--cuda-graph-max-bs-decode" not in argvassert max_running_requests == 13



পাশাপাশি commit-mode এখনও default W1 থেকে শুরু করছে—এটিও test।
B2. Width6 কি 40k–100k-এ নিরাপদ?
Verdict: UNSURE।

পক্ষে evidence: short context-এ 531.43 tok/s, gate PASS, no application error।

বিপক্ষে: matching long-context benchmark নেই; combined width6+graphs13 run নেই; speculative reasoning/state correctness issue public-এ documented। 

GitHub
+1




আমার সিদ্ধান্ত: একমাত্র high-stakes hidden submission হলে W1 width8 + graphs12 হলো বেশি পরীক্ষিত baseline। Width6 গ্রহণযোগ্য করতে অন্তত একটি small long-context/concurrency gate দরকার। Graphs13 speed বাড়িয়েছে এমন evidence নেই; 12-running long window-এ বরং 449.84→436.89 tok/s হয়েছে, যদিও এটি controlled repeated comparison নয়।

Width4-কে width6-এর চেয়ে safer বলা যাচ্ছে না; public issue-তে width4-ও failure পুরোপুরি দূর করেনি।
B3. 240 tasks কি 12 ঘণ্টায় fit করবে? Submission empty হওয়ার risk?
Verdict: Coverage plausible; 60k tokens per task guaranteed নয়।

R2 Duck 1,500-second window-এ 430,062 generated tokens দিয়েছে:
\[
430062 / 1500 \approx 287\ \text{agent-effective tok/s}
\]

প্রথম server ready হতে প্রায় 1,074 seconds notebook wall-time গেছে (install, staging, initial launchসহ)।

সুতরাং hidden run-এ solver-এর জন্য আনুমানিক 11½ ঘণ্টা পাওয়া গেলে:
\[
287\times41{,}000 \approx11.77\text{M tokens}
\]

২৪০ task-এ গড়ে 49k tokens/task। সকলকে 60k দিতে 14.4M tokens দরকার; একই timeframe-এ প্রায় 351 effective tok/s লাগবে।

এটি scenario calculation, hidden-run forecast নয়। Full task exposure, stage-2 allocation এবং actual context mix outcome বদলাবে।

Submission integrity:
- Notebook CELL 0 lines 31–45 প্রথমেই identity fallback submission.json লেখে।
- CELL 6-এ error হলে সেই fallback/partial submission রেখে দেওয়ার intention আছে।
- CELL 7 final verified attempts write করে।
তবে challenge JSON খুলতে ব্যর্থ হলে CELL 0 fallback-ও লেখা হবে না। Missing model/payload mount-এর ফলে later exception হলে Kaggle notebook failure status submission গ্রহণে বাধা দিতে পারে। তাই fallback file আছে মানেই guaranteed valid competition submission—এমন নিশ্চয়তা নয়।

Pre-push test: hidden-mode fake run with 240 synthetic tasks, missing model, missing draft, wrong payload SHA, launch timeout, zero solver output এবং normal partial candidates। প্রতিটি ক্ষেত্রে valid 240-task submission.json এবং accepted notebook completion status নিশ্চিত করতে হবে।
B4. এক দিনে সবচেয়ে বড় সম্ভাব্য LB lever কী?
Current best LB 34.58 plain reasoning, আর previous Duck Rev 5 LB 17.50—এটি গুরুত্বপূর্ণ বাস্তব evidence। G4B 6/19 দিয়ে Duck-কে নিশ্চিত superior বলা যায় না।

এক দিনে সবচেয়ে বড় practical lever হবে:

Existing best submission অক্ষত রেখে, independent Duck candidate diversity এবং verified pass@2-এর complementary value মাপা।

তার সঙ্গে low-risk engine change ও existing known-good 240-task scheduler validation।

নতুন architecture rewrite বা speculative kernel patch-এর চেয়ে এটি কম ঝুঁকির। কিন্তু কত leaderboard gain হবে—পাওয়া যায়নি।

C. E15 speed plan-এর তিনটি প্রশ্ন
C1. Arm order সঠিক? কোনো unsupported arm?
Verdict: GO for an engineering experiment, with revised priority।

E15-এর সবচেয়ে মূল্যবান experiment হলো built-in MTP-কে একই 12-stream 40k context-এ DFlash2-এর সঙ্গে তুলনা করা।

Official SGLang Qwen3.8 cookbook এই native MTP recipe দেয়:
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4


NEXTN ও EAGLE একই built-in MTP path-এর alias হিসেবে documented। 

GitHub




আমার preferred E15 order:
Order	Experiment	কেন
1	DFlash2 width8 same-session 12-stream reference	R2-এর সঙ্গে comparable baseline
2	MTP steps3/draft4, 12 streams	Highest-value unknown
3	MTP steps2/draft3	Verify overhead/acceptance trade-off
4	MTP + ReplaySSM, if supported	SSM memory savings
5	Best MTP at 14/16 actual running	Higher capacity path
6	Chunked prefill 4096→2048	Agent-effective improvement
7	NCCL protocol microbenchmark	Already measured baseline best; cheap optional arm
8	2×TP2 + MTP	Only after memory fit and TP4 MTP success

NCCL microbenchmark আগে CPU/GPU setup phase-এ parallelভাবে চালাতে পারলে fine, কিন্তু এর জন্য MTP benchmark বাদ দেওয়া উচিত নয়।

Unsupported/risky: SGLang 0.5.21-এ DFlash + PP route drop-in নয়। MTP + ReplaySSM flag অবশ্যই exact release-এর support ও runtime logs দিয়ে যাচাই করতে হবে। SGLang cookbook --enable-linear-replayssm-spec ব্যবহার করে, কিন্তু সব hybrid paths এই option সমানভাবে support করে না। 

GitHub



C2. RedHat INT4 checkpoint-এ MTP head আছে?
Verdict: UNSURE — critical E15 preflight।

Official Qwen3.8 architecture-এ built-in MTP রয়েছে। কিন্তু আপনার local RedHat compressed-tensors artifact-এ সেই head সত্যিই সংরক্ষিত এবং loader তা ব্যবহার করতে পারবে কি না, ZIP থেকে জানা যায়নি। Kaggle model dataset-এর actual config.json ও safetensors index এখানে নেই।

অন্য public quantized checkpoints-এ MTP আলাদা file-এ থাকতে দেখা গেছে; কিছু quantized exports MTP বাদও দেয়। 

Hugging Face
+1




GPU quota খরচের আগে notebook-এর offline metadata check:
import jsonfrom pathlib import Pathmodel_dir = Path("/path/to/redhat/int4")cfg = json.loads((model_dir / "config.json").read_text())print("MTP config:", {    k: v for k, v in cfg.items()    if "mtp" in k.lower() or "nextn" in k.lower()})index = model_dir / "model.safetensors.index.json"if index.exists():    weights = json.loads(index.read_text())["weight_map"]    mtp = [        name for name in weights        if "mtp" in name.lower() or "nextn" in name.lower()    ]    print("MTP indexed tensors:", len(mtp))    print(mtp[:20])



num_mtp_heads বা equivalent nested configuration-ও inspect করতে হবে। Index না থাকলে safetensors metadata inspect করুন, tensor data পুরো GPU-তে load নয়।

Acceptance: MTP tensors সত্যিই আছে; expected head weights load হয়েছে; server ready; short greedy sanity check; memory/KV capacity reported; no missing-weight warning।

No MTP tensors হলে E15-M1-এর জন্য original BF16 head আলাদাভাবে attach করা বা compatible checkpoint লাগবে। শুধু EAGLE flag দিয়ে missing trained head তৈরি হবে না।
C3. 600–800 tok/s বাস্তবসম্মত?
Verdict: Goal, not forecast।

আপনার R2 baseline 505 tok/s। TP0 profiling অনুযায়ী communication 59/116ms ≈51%।

Amdahl-style sensitivity:
Hypothetical reduction in communication time	Resulting throughput estimate
25%	~578 tok/s
40%	~635 tok/s
60%	~728 tok/s
70%	~783 tok/s

এগুলো pure hypothetical estimates, কারণ MTP communication count, GEMM cost, acceptance length এবং overlap একসঙ্গে বদলায়।

NCCL baseline already graph collective 343.9µs/call। p2p_sys 1,880.5µs, 5.47× ধীর। shm_memcpy arm timeout হয়েছে। তাই শুধু environment flag tune করে 800 tok/s আশা করা বাস্তবসম্মত নয়।
E15 arm	Responsible expectation
Current DFlash2 width8	Measured ~505 short, ~450 at 8.8k
MTP steps3/draft4	Could beat or lose to DFlash2; exact range পাওয়া যায়নি
MTP steps2/draft3	Could help if verification overhead dominates
MTP at 14/16	Higher aggregate possible; OOM/acceptance constraints unknown
2×TP2	High uncertainty, memory fit blocker possible
Chunked prefill 2048	May improve agent-effective tok/s even without large decode gain

একটি শক্ত বাইরের risk reference: SGLang #36548 ও #38009 concurrency/thinking-related DFlash correctness anomalies দেখায়। MTP-তে switch শুধু speed experiment নয়, potentially useful correctness comparison-ও। কিন্তু MTP-কে automatically correct বলা যাবে না। 

GitHub
+1




পরবর্তী কাজ: Priority ranking
Priority	Action	GO criterion
P0 — আগে	RERUN_ALL engine flag mismatch ঠিক করা	Generated notebook argv exactly expected; fresh SHA
P0 — আগে	Hidden 240-task failure-path test	Missing mount, launch timeout, partial completion: valid submission
P0 — আগে	Decide width8 proven baseline বনাম width6 experimental	Width6 long-context gate পাস না করলে width8
P1	Keep known-good 34.58 baseline safe	Existing good submission unchanged
P1	Fix JIT metadata probe empty-stdout crash	CPU error-injection tests, no GPU needed
P1	Improve candidate-first behavior	More branches with first check, not merely nudges emitted
E15-1	Check RedHat MTP head locally	Tensor/loader presence
E15-2	12-stream MTP versus DFlash2 at 8k/40k	Actual tok/s, accept, parity, KV, no errors
E15-3	MTP steps2/3, ReplaySSM	Matched comparison, no accuracy regression
E15-4	14/16 running, chunked prefill	More agent-effective throughput without OOM/retraction
E15-5	2×TP2 only after memory fit	Two replicas successfully admit required long contexts

Overall verdict
GO — R2 results as evidence
NO-GO — intended rerun flags as-is

R2 measurement সফল: communication bottleneck independently confirmed; #44401 কার্যকর নয়, nodelay unsafe, int8 Mamba OOM, raw reasoning working, width6/4 small short-context gains, এবং Duck 6/19 করেছে।

কিন্তু engine speed-এর পাশাপাশি agent reasoning behavior-এ বড় দুর্বলতা রয়ে গেছে: 17-এর 12 branch কোনো check() করেনি। 25-minute window-এ effective generation ~287 tok/s; এই rate-এ 240 task-এর প্রত্যেককে পূর্ণ 60k reasoning দেওয়া সম্ভব নয়।

তাই next submission-এ প্রথম কাজ নতুন model বানানো নয়। Exact rerun profile সত্যিই intended engine চালাচ্ছে কি না, submission নিরাপদ কি না, এবং candidate-generation efficiency কত—এই তিনটি আগে নিশ্চিত করা।

Public verification URLs:

https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx

https://github.com/sgl-project/sglang/issues/36548

https://github.com/sgl-project/sglang/issues/38009

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/dflash_worker_v2.py

https://github.com/sgl-project/sglang/releases

https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin

কোনো push, commit বা code change করিনি।
