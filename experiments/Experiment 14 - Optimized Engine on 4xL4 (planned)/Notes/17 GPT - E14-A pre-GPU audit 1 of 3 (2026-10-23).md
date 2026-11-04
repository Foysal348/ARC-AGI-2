# GPT audit 1 of 3 of E14-A notebook a9dad9fc (user-pasted 2026-10-23, verbatim)

Validation: Notes/20.

---

E14-A — Independent Pre-GPU Audit
Target notebook SHA-256: a9dad9fcbe0003867d34f205732a23106dbfed30380535ef4531ba78806de494

প্রথমে ZIP-এর README_FOR_GPT.md সম্পূর্ণ পড়েছি। তারপর embedded notebook cells, পাঁচটি E14 module, SGLang patches, tests, harness logs এবং source excerpts পরীক্ষা করেছি। কোনো push, commit বা source modification করিনি।
১. Artifact integrity
ZIP integrity
Verified

SHA-256: 44cb351807d14decb9a1da36d51f0ad2f131aab906b58f80a9361441281e01ec

Notebook SHA-256
Exact match

README-তে দেওয়া a9dad9fc… fingerprint হুবহু মিলে গেছে।

Recorded Windows unit tests
125/125 × 5

পাঁচটি test-log-এ PASS লেখা আছে। এগুলো recorded evidence, আমার Linux machine-এ একই ফল নিশ্চিত হওয়ার প্রমাণ নয়।
Fake-server notebook harness
26/26 recorded

Run 2-এর 23/24 failure ও final run 5-এর 26/26 outcome-এর log আছে।
আমার Linux rerun-এ প্রথম ২৮টি test পাস করেছে, এরপর test_mounts_longest_prefix ব্যর্থ হয়েছে। Test-টি os.name = "nt" দিয়ে Windows path behavior simulate করছে; Linux-এ এই artificial platform change নিয়ে test expectation portable নয়। এটি Kaggle-এর real Linux runtime bug-এর প্রমাণ নয়, তবে 125/125 independently revalidated বলারও সুযোগ নেই।
২. প্রথম গুরুত্বপূর্ণ সিদ্ধান্ত
E14-A-এর architecture অর্থবহ: measurement → controlled adoption → 25-minute Duck evaluation → remaining diagnostic arms। GPU server endpoints-এর আগের কিছু ভুল এবার source excerpts দিয়ে যাচাই করা হয়েছে।

কিন্তু adoption system-এর correctness gate ও latch measurement-এ এমন কিছু দুর্বলতা পেয়েছি যেগুলো ভালো arm reject করতে পারে, অথবা যথেষ্ট evidence ছাড়াই কোনো arm accept করতে পারে। এগুলো measurement notebook-এর উদ্দেশ্যকেই প্রভাবিত করে। নিচে প্রতিটির exact source location, counterexample এবং প্রয়োজনীয় fix দিচ্ছি।
৩. Final GO / NO-GO
NO-GO — exact SHA as-is
চারটি সীমিত, evidence-backed correction প্রয়োজন।

এই verdict-এর অর্থ E14-A architecture বাতিল নয়। Engine profiling, staging, NCCL benchmark এবং Duck integration-এর অধিকাংশ অংশ রাখতে বলছি।

কিন্তু notebook-এর উদ্দেশ্য শুধু তথ্য সংগ্রহ নয়; মাপা ফলের ভিত্তিতে স্বয়ংক্রিয়ভাবে engine configuration গ্রহণ করে Duck চালানো। সেই উদ্দেশ্যে নিচের সমস্যাগুলো প্রাসঙ্গিক।
Severity	Finding	Consequence
HIGH / must-fix	Correctness gate A/A noise বেশি হলে অর্থহীন PASS দিতে পারে	Unsafe engine adoption
HIGH / must-fix	A1 phases 25-minute Duck run-এর সময় reserve করে না	Solver window প্রত্যাশার চেয়ে অনেক কম হতে পারে
HIGH / must-fix	HTTP phase deadline পেরিয়েও request শুরু হতে পারে	Runtime ও quota accounting drift
HIGH / must-fix	/flush_cache failure success থেকে আলাদা করা হচ্ছে না	Duck screening cold-cache-equivalent না-ও হতে পারে
MEDIUM	Nodelay latch metric 13-কে primary target করেছে, Duck needs 12	Useful arm rejection
MEDIUM	Int8 cache parity baseline-vs-int8 correctness যাচাই করে না	A2 adoption decision misleading
MEDIUM	Window speed comparisons running/context equality enforce করে না	Apples-to-oranges performance comparison
LOW / operational	Staging daemon fallback-এর পরও copy চালাতে পারে	Disk contention / wasted startup time

প্রথম চারটি এক consolidated patch set-এ নেওয়া উচিত। বাকি পরিবর্তনগুলোও ছোট; বিশেষত latch এবং measurement validity ঠিক করলে পরীক্ষার মূল্য বাড়বে।

৪. Confirmed findings, reproduction ও patches
HIGH-1 — Correctness gate false PASS
Location: src/e14_measure.py:435–451, 560–567

বর্তমান decision:
return {    "ok": (        cmp_arm["identical"] >=        cmp_aa["identical"] - allowance        and cmp_arm["large_margin_flips"] == 0    )}



আমি source function দিয়েই দুইটি counterexample চালিয়েছি।

Counterexample A:
cmp_arm = {    "compared": 12,    "identical": 0,    "large_margin_flips": 0}cmp_aa = {    "compared": 12,    "identical": 0,    "large_margin_flips": 0}parity_ok(cmp_arm, cmp_aa, allowance=2)



Actual output: ok = True
অর্থাৎ A/A control-এর একটি prompt-ও পুরো 512-token sequence-এ identical না হলে একটি সম্পূর্ণ divergent arm-ও PASS করতে পারে, যদি large-margin-flip detector সেটিকে না ধরে।

আপনার আগের E11-03 evidence-এ batching-এর কারণে A/A sequences মাত্র 152–166/512 token পর্যন্ত একই ছিল। ফলে whole-sequence identical count শূন্য হওয়ার আশঙ্কা বাস্তব, শুধু synthetic edge নয়।

Counterexample B: একটি মাত্র successful comparison থাকলেও function compared=1 নিয়ে PASS করতে পারে।

আরেকটি সমস্যা compare_passes()-এ:
m = ra["margins"][k] if ... else None



Margin না পাওয়া গেলে large_margin_flips বাড়ে না। ফলে missing logprob evidence নীরবে zero-risk হিসেবে বিবেচিত হচ্ছে।

Concrete fix:
- Expected prompt count এবং valid compared count দুটোই যাচাই।
- Missing top-two logprobs আলাদাভাবে count।
- Baseline ও candidate উভয়ের margin inspect।
- A/A control অস্থিতিশীল হলে INCONCLUSIVE, স্বয়ংক্রিয় PASS নয়।
- Greedy output একই prefix-এ প্রথম significant disagreement হলে সেই অবস্থানের top-logprob evidence ব্যবহার।
- 12/12 comparison না হলে incomplete gate হিসেবে record; সীমিত sample দিয়ে adoption নয়।
প্রস্তাবিত decision structure:
def parity_decision(arm, aa, expected):    if arm["compared"] != expected:        return {"ok": False, "why": "incomplete arm"}    if aa["compared"] != expected:        return {"ok": False, "why": "incomplete control"}    if arm.get("missing_margins", 0):        return {"ok": False, "why": "margin evidence missing"}    if aa["identical"] == 0:        return {            "ok": False,            "why": "A/A whole-sequence control inconclusive"        }    # Then apply the calibrated divergence rule.



এটি সম্পূর্ণ algorithm নয়; final function-এ actual per-position margins দরকার।

Validation tests: stable A/A, unstable A/A 0/12, missing margins, missing requests, large-margin flip এবং low-margin benign divergence। শুধু successful fake-server response দিয়ে test করা যথেষ্ট নয়।

আরও গুরুত্বপূর্ণ: CELL 5b-এর #44401, nodelay ও graphs13 adoption-এ এখন speed/latch ব্যবহৃত হয়, arm-specific parity gate নয়। README-এর “speed rule and correctness gate” contract অনুযায়ী retained configuration-এর বিরুদ্ধে ছোট, controlled parity check যুক্ত করা প্রয়োজন।
HIGH-2 — A1 Duck-এর 25 মিনিট reserve করে না
Location: notebook CELL 5b, e14_phase():81–107; CELL 6-এর shifted solver deadline।

বর্তমানে phase budget:
left = HARD_STOP - time.perf_counter() - 8 * 60



এটি শুধু final hard stop-এর জন্য ৮ মিনিট রাখে। পরের Duck run-এর ২৫ মিনিট রাখে না।

ধরা যাক:
- HARD_STOP পর্যন্ত ১৮ মিনিট বাকি।
- A1-এর নতুন phase শুরু হলো।
- left = 10 min।
- সেই phase পুরো ১০ মিনিট খরচ করল।
এরপর Duck শুরু করলে solver cutoff HARD_STOP - 180 s পর্যন্ত সময় থাকবে মাত্র প্রায় ৫ মিনিট।

অর্থাৎ notebook 25-minute Duck benchmark দেওয়ার কথা বললেও late A1 measurement সেই opportunity কমিয়ে ফেলতে পারে।

Minimal fix:

A1-এর জন্য:
DUCK_RESERVE_SEC = (    25 * 60    + E12_STOP_BEFORE_HARD_SEC    + 120)



নতুন A1 phase admission:
available = (    HARD_STOP    - time.perf_counter()    - DUCK_RESERVE_SEC)if available < MIN_PHASE_USEFUL_SEC:    return {"skipped": "Duck reserve"}phase_cap = min(cap_sec, available)



A2-তে Duck reserve লাগবে না, কারণ Duck ইতিমধ্যে শেষ।

e14_launch()-এর ক্ষেত্রেও A1 সময় শেষ হওয়ার দিকে relaunch skip করতে হবে—শুধু HARD_STOP - 15 min check যথেষ্ট নয়।

Test: fake monotonic clock দিয়ে early, medium ও late startup scenarios। Late startup-এ optional A1 arms skip হবে; Duck-এর 25-minute opportunity বজায় থাকবে, যদি hard stop-এর আগে আদৌ এত সময় অবশিষ্ট থাকে।
HIGH-3 — Deadline clamp আসলে strict নয়
Location: src/e14_measure.py:58–70

বর্তমান:
timeout = max(    5.0,    min(float(timeout), ctx.left() + 5.0))



আমি ctx.left() = -2 দিয়ে সরাসরি পরীক্ষা করেছি।
Controlled reproduction

Deadline 2 seconds আগে শেষ

তবু নতুন HTTP request শুরু হয়েছে; timeout = 5.0 seconds

এর অর্থ post_json() phase deadline পার হওয়ার পরও request পাঠাতে পারে।

একাধিক abort request serially চালালে overrun আরও বাড়তে পারে। parity_pass()-এ unfinished generation threads abort করার পরে bounded global cleanup completion-ও কঠোরভাবে enforce করা হয়নি।

Minimal fix:
left = ctx.left()if left <= 0:    raise TimeoutError("phase deadline passed")timeout = min(float(timeout), left)



Cleanup request-এর জন্য আলাদা bounded grace policy রাখুন। কারণ phase deadline শেষ হওয়ার সঙ্গে সঙ্গে সব abort request নিষিদ্ধ করলে outstanding server work থেকে যাওয়ার ঝুঁকি থাকবে।

প্রস্তাব: abort()-এ concurrent bounded cancellation, তারপর active-request count check; unresolved হলে controlled server termination। সমস্ত cleanup-এর জন্য একটি global deadline, প্রতিটি request-এর নতুন ৫ বা ৩০ সেকেন্ড নয়।

Tests: deadline before call, deadline during response, 12 outstanding requests, abort error, server shutdown fallback, এবং final hard-stop boundary।
HIGH-4 — /flush_cache failure success থেকে আলাদা করা হচ্ছে না
Location: notebook CELL 5b:287–290 এবং src/e14_measure.py:65–70।

SGLang 0.5.21-এর /flush_cache endpoint success ও failure দুটোতেই plain text ফেরত দিতে পারে। ZIP-এর source excerpt-এ এটি রয়েছে।

আমি একটি response simulation করেছি:
Flush cache failed.


বর্তমান post_json() result:
{"text": "Flush cache failed.\n"}



কোনো exception হয় না। Notebook সেটি duck_flush_cache-এ record করে Duck চালায়।

এই report recording ভুল নয়; কিন্তু cold-cache-equivalent Duck comparison তখন নিশ্চিত নয়।

Fix:
reply = E14M.post_json(    E14_CTX, "/flush_cache", {}, timeout=60)msg = (reply.get("text") or "").strip()flushed = msg.startswith("Cache flushed.")E14_RESULTS["duck_flush_cache"] = {    "ok": flushed,    "reply": msg}



Flush-এর আগে metrics দিয়ে running ও queued requests zero হওয়ার জন্য bounded wait যুক্ত করুন।

Flush failure হলে:
- সময় থাকলে last adopted arm একবার clean relaunch করা যায়।
- সময় না থাকলে Duck চালানো যাবে, কিন্তু cold-cache comparison invalid হিসেবে report করতে হবে।
- শুধু flush failure-এর কারণে পুরো notebook fail করা প্রয়োজন নেই।
SGLang-এ active/paused requests থাকলে flush সমস্যা হতে পারে—এ ধরনের documented issue রয়েছে। 

GitHub




https://github.com/sgl-project/sglang/issues/33783

তবে extra_buffer_lazy-এর ক্ষেত্রে সফল /flush_cache সব Mamba state পরিষ্কার করে কি না, সেটি supplied snippets দিয়ে সম্পূর্ণভাবে প্রতিষ্ঠিত হয়নি। সেটি GPU/server-state diagnostic হিসেবে রাখতে হবে।

৫. Nodelay ও #44401 latch rule — README Q2
Location: src/e14_measure.py:455–487, notebook CELL 5b:243–267।

বর্তমান latch test:
decodes = 11arrivals = 3server_cap = 13want = min(14, 13)  # 13



তারপর মাপা হচ্ছে:
- running ≥13;
- running =12 এবং queue >0।
কিন্তু production Duck client-এর target 12 streams, 13 নয়।

আর আগের সমস্যাটি ছিল 11 running + 1 queued।

তাই W1-এ 11 running + queue থাকলেও বর্তমান share_stuck_one_below সেটি ধরবে না। যদি নতুন arm 12 running + queue অবস্থায় উন্নতি করে, তার stuck-one-below count উল্টো বেড়ে যেতে পারে।

এটি একটি confirmed metric-design defect, যদিও real engine-এ nodelay ভালো/খারাপ এখনও অমাপা।

প্রস্তাবিত primary metric:
\[
\mathrm{share\_productive}
=
P(\mathrm{running}\ge12)
\]

এবং old pathology:
\[
\mathrm{share\_stuck11}
=
P(\mathrm{running}=11,\ \mathrm{queue}\ge1)
\]

running ≥13 আলাদা over-admission/headroom diagnostic হতে পারে; primary goal নয়।

আপনার প্রস্তাবিত “running ≥12 and queue ≥1” একটি useful pressure-specific metric, কিন্তু একে একমাত্র metric করবেন না। Queue zero হয়েও 12 active streams থাকলে সেটি productive state।

Adoption rule: 12-stream share বাড়ছে, 11+queue share কমছে, error/retraction বাড়ছে না এবং meaningful matched-context throughput নষ্ট হচ্ছে না—এই চারটি দেখুন।

এটিই nodelay-এর জন্য অধিক যুক্তিসংগত।

৬. আরও দুটি measurement defect
6.1 Speed comparison context ও running level যাচাই করে না
e14_measure.adopt_speed() কেবল tok_s_p50 ব্যবহার করে।

আমি synthetic inputs-এ চালিয়ে দেখেছি:
Baseline	Arm	Current result
11 running, 350 tok/s	12 running, 354 tok/s	Adopt
40k context, 350 tok/s	8k context, 380 tok/s	Adopt

দ্বিতীয়টি clearly invalid comparison। প্রথমটি nodelay-এর ক্ষেত্রে স্বাভাবিক improvement হতে পারে, কিন্তু generic #44401 বা graphs test-এর ক্ষেত্রে comparison context স্পষ্ট করতে হবে।

Fix: Per-arm report-এ measured_level, ctx_per_req_p50, window_sec, real_running_share সব mandatory। Context mismatch excessive হলে comparative score INCONCLUSIVE। Nodelay-এর ক্ষেত্রে intentional 11→12 transition আলাদাভাবে হিসাব করা যাবে।
6.2 Int8 Mamba gate baseline-vs-int8 output parity যাচাই করছে না
CELL 6b:11–34-এ _cache_hit_parity() একই configuration-এর fresh বনাম cached prompts তুলনা করে।

ধরা যাক:
- Default cold = A, default cached = A।
- Int8 cold = B, int8 cached = B।
দুটি within-arm test-ই exact। তাই বর্তমান gate PASS করতে পারে, যদিও int8 target output সম্পূর্ণ বদলে দিয়েছে।

Fix: Default-vs-int8 parity আলাদাভাবে test করুন; একই input IDs, same greedy settings, একই prompt stage। Within-arm cache parity রাখতে হবে, কিন্তু সেটি cross-arm correctness-এর বিকল্প নয়।

এটি Duck-এর আগের 25-minute evaluation-কে সরাসরি ক্ষতিগ্রস্ত করে না, কারণ int8 test A2-তে চলে। তবে ভবিষ্যৎ engine selection-এর জন্য প্রয়োজনীয় correction।

৭. সব ১৬টি fix-এর verdict
#	Fix	Verdict
1	HTTP request timeout clamp	Partial; HIGH-3
2	Late parity requests aborted	Implemented; global cleanup deadline অসম্পূর্ণ
3	wait_level plateau	Implemented; comparative adoption-এ level checking নেই
4	A/A noise-floor parity	Partial; HIGH-1
5	Generated-cell compile/CR check	Verified in builder design; recorded harness PASS
6	Profiler with_stack=false, record_shapes=false	Source verified
7	NCCL arm ≥20% microbench gain	Code present; real workload gain separately measured
8	Nodelay adoption	Flag correct; latch metric সংশোধন প্রয়োজন
9	Plain-text /start_profile response	Fixed, earlier crash class eliminated
10	Server-log sign counts	Code verified; substring counts diagnostic, structured error classifier নয়
11	Pre-Duck /flush_cache	Implemented; failure detection incomplete
12	Graphs13 captured list check	Code present; unparsed list হলে permissive fallback
13	Builder CR guard	Corrective change documented; final notebook parses
14	Watchdog peak statistics	Code present; no new behavioral bug found
15	Fake /flush_cache harness	Added; failure semantic test দরকার
16	Platform timing test robustness	Recorded 125/125 ×5; Linux portability caveat

Important: ১৬টির সবগুলোকে defective বলছি না। চারটি HIGH correction-এর বাইরে অধিকাংশ fix সঠিক উদ্দেশ্যে এবং সীমিত surface-এ করা হয়েছে।
৮. README-এর ১০টি no-change decision
Item	Verdict
Token-id fields for E14-P	Bundled v0.5.21 protocol excerpts support the fields; non-streaming restriction যথাযথ
Null cached_tokens = 0	Bundled usage processor evidence supports current interpretation, provided cache reporting enabled
KV pool recalculated after relaunch	Code path present
Separate JIT cache + fingerprint	Sensible; restored cache still needs known-answer validation
E11 vs Duck KV gates	Separate paths retained
Duck window start after A1	Start-time shift present; 25-minute reservation missing
No vLLM fallback	Supported by notebook flow; fallback submission retained
Stale pyc after patch	Patch tool removes pyc
Delayer behavior	Core formula supported; latch interpretation needs repair
Relaunch skip near HARD_STOP	Code present; Duck reserve not protected
No unsupported penalties/ReplaySSM/PCIe-IPC	No contrary evidence in implemented E14 arms

শেষের দুটি কাছাকাছি statement আমি আলাদাভাবে দেখিয়েছি, কারণ launch guard ও reserved solver window পৃথক বিষয়।
৯. ইচ্ছাকৃতভাবে বাদ দেওয়া ৫টি change
Omission	My decision
Long 12,288-token thinking parity	Accept omission for E14-A; expensive এবং current goal-এর জন্য বাধ্যতামূলক নয়
Full 60k context band per arm	Accept omission; 8k/40k bands থেকে useful evidence পাওয়া সম্ভব
Cosmetic legacy DIRECT fields	Accept omission; Duck reporting semantics আলাদা রাখা উচিত
Frozen Rev 6g metric fixes / publish_seq	Accept freeze for this experiment; তবে final record-integrity regression ও SHA check রাখতে হবে
Mamba cache 64 / mrr14	Accept omission; memory evidence ছাড়া concurrency পরিবর্তন নয়

এই পাঁচটির কোনোটি E14-A GPU run আটকানোর কারণ নয়।

১০. README-এর ছয়টি প্রশ্নের সরাসরি উত্তর
Q1. Phase invalid বা budget exceed হতে পারে?
হ্যাঁ। চারটি প্রধান জায়গা:
- post_json() deadline-এর পরও request করতে পারে;
- A1 Duck budget reserve করে না;
- incomplete parity যথেষ্ট evidence ছাড়াই gate PASS করতে পারে;
- /flush_cache failure cold-start assumption ভেঙে দিতে পারে।
এগুলো source-level এবং reproduction-supported। GPU চালিয়ে জানার প্রয়োজন নেই।
Q2. Latch-test adoption rule ঠিক?
বর্তমানভাবে পুরোপুরি নয়। 13 target-এর বদলে client target 12-কে primary করুন। 11 running + queue আলাদা metric চাই। Queue present অবস্থায় 12 streams কতক্ষণ ধরে থাকে, সেটি throughput-এর সঙ্গে দেখুন।
Q3. /flush_cache safe?
Endpoint রয়েছে এবং idle server-এ ব্যবহার যুক্তিসংগত। কিন্তু failure plain-text response হতে পারে, আর Mamba/hybrid cache সম্পূর্ণ পরিষ্কার হওয়ার নিশ্চয়তা supplied evidence দেয় না।

Success response + zero running/queued + পরবর্তী cold-prefix cache observation—এই তিনটি একসঙ্গে log করুন। Blindly success ধরে নেবেন না।
Q4. Greedy parity allowance 2 + zero large-margin flips sound?
এখনকার implementation-এ নয়। A/A full-sequence identity শূন্য হলে acceptance criterion দুর্বল হয়ে যায়। Missing logprobs-ও guard bypass করে। Correctness gate-এ completeness, confidence evidence এবং baseline instability আলাদাভাবে বিচার করতে হবে।
Q5. Width 6/4-এ flag ছাড়া কিছু লাগবে?
SGLang DFlash path width override সমর্থন করে, কিন্তু checkpoint-এর trained block size 8। Width পরিবর্তন করলে internal draft indexing ও acceptance distribution বদলাতে পারে।

তাই শুধু flag syntactically accepted হলেই যথেষ্ট নয়। প্রয়োজন:
- Server startup এবং effective resolved width check;
- first draft-position behavior ও acceptance distribution;
- target-only এবং width8 comparison;
- tool-call/correctness parity;
- 40k context-এ memory ও step-time check।
তবে এখনই নতুন kernel বা architecture rewrite বাধ্যতামূলক নয়। DFlash layout সম্পর্কিত public issue-ও একই caution সমর্থন করে। 

GitHub




https://github.com/sgl-project/sglang/issues/40144
Q6. Quota কম থাকলে A2-এর কোন arm বাদ দেব?
আমার resource-priority order:
1. রাখবেন: baseline engine measurements, nodelay test, relevant profiler, 25-minute Duck evaluation।
2. রাখার চেষ্টা করবেন: target-only parity এবং width6/4 throughput/acceptance।
3. প্রথমে বাদ: আলাদা graphs13 experiment, যদি startup বা capture বেশি সময় নেয় এবং expected gain ছোট।
4. পরেরটি বাদ: int8 Mamba checkpoint arm—Mamba eviction এখনো মূল bottleneck হিসেবে নিশ্চিত নয়।
5. তারপর বাদ: NCCL environment relaunch, যদি microbenchmark already weak বা uncertain হয়।
#44401/#41872 patch CPU evidence থাকা সত্ত্বেও GPU impact নিশ্চিত নয়; patch evaluation সংক্ষিপ্ত রেখে nodelay/solver budget বেশি গুরুত্বপূর্ণ।

১১. GPU-only আটটি প্রশ্ন: বর্তমান অবস্থা
Question	Before-GPU verdict
Nodelay-এ real 12/13 running	Unknown
Backport x86_64 wheel-এ apply	Source-hash guarded; actual wheel unknown
Raw reasoning-এর token-prefix restoration	Fake-server evidence strong; real GPU unknown
NCCL communication split	Unknown, profiler দরকার
Width6/4 acceptance	Unknown
Int8 Mamba eviction effect	Unknown
/dev/shm, copy throughput, startup	Kaggle session-dependent
Hybrid /flush_cache behavior	Exact success/complete state unknown

এই unknown-গুলোকে bug হিসেবে চিহ্নিত করছি না। এগুলো experiment চালানোর কারণ।
১২. Quota এবং worst-case plan
README-এর last reported quota:
hosen42 — last reported, not live verified
8.63 h
Normal charge
4.9–5.5 h

Conservative worst
~7.0 h



যদি 8.63 h এখনও থাকে, worst-case-এর পরে আনুমানিক 1.63 h থাকবে। তবে quota ২২ অক্টোবরের; push-এর আগে বর্তমান balance দেখা অপরিহার্য।

200-minute code budget এবং আনুমানিক 210-minute platform envelope README-তে পৃথক করা হয়েছে—ভালো। কিন্তু external 215-minute stop rule platform stall-এর জন্য; এটি notebook-এর internal phase-deadline correctness-এর বিকল্প নয়।

একটি সতর্কতা: storage slow হলে E14_STAGING thread fail-open করার পরে এখনও background copy চালাতে পারে। src/e14_loadpath.py:658–668 এবং CELL 1b staging join path-এ explicit cancellation নেই। যদি fallback original GCSFUSE path থেকে server load হয়, সেই একই storage-এ background reader contention তৈরি হতে পারে। এটি plausible operational risk, confirmed failure নয়। Optional improvement: stop event দিয়ে staging বন্ধ করে bounded join-এর পর fallback launch।

১৩. Claude-এর জন্য consolidated patch plan
এটি আলাদা আলাদা GPU run নয়। একটি local correction build, তারপর final checks।
Priority	Change	Acceptance test
Must	Fail-closed parity completeness + meaningful A/A baseline + margin availability	0/12 A/A, 1/12 incomplete, no margins → no adoption
Must	A1-এর আগে 25-minute Duck reserve	Late startup simulated; optional arms skipped, solver opportunity retained
Must	Strict HTTP deadline + bounded global abort cleanup	Negative deadline never sends request; 12 slow requests cannot extend phase arbitrarily
Must	Flush response classification + idle check	Plain-text failure → ok=false, cold-equivalence not claimed
Recommended same patch	12-stream latch and 11+queue metrics	W1=11 stuck, nodelay=12 productive scenario
Recommended same patch	Cross-arm int8 parity and window metadata gating	Int8 A/A exact but different from baseline → rejected
Validation	SHA, patch/revert, code compile, fake-server failure injections	Final exact notebook matches report and all tests

যা বদলাব না: SGLang version, TP4, RedHat INT4, DFlash2 architecture, Rev 6g solver, host verifier, raw-reasoning switch, sandbox, model loading strategy, output-token caps এবং existing source-hash-guarded backports।
Final audit statement
E14-A আগের experiments-এর তুলনায় অনেক বেশি instrumented। বিশেষ করে storage slowdown, profiler responses, patch rollback, NCCL benchmark এবং raw-reasoning history-এর জন্য কাজগুলো বাস্তব সমস্যাকে লক্ষ্য করে করা হয়েছে।

কিন্তু 125/125 tests × 5 মানেই deployment decision নিরাপদ নয়। এখানে কয়েকটি গুরুত্বপূর্ণ test oracle—বিশেষ করে parity PASS এবং nodelay latch interpretation—বাস্তব experiment objective-এর সঙ্গে যথেষ্ট aligned নয়। এগুলো CPU-level fixes; GPU quota খরচ করে আবিষ্কার করার প্রয়োজন নেই।

Exact notebook SHA a9dad9fc…-এর জন্য NO-GO। উপরের চারটি must-fix এবং সংশ্লিষ্ট negative tests পাস হলে সংশোধিত SHA-তে নতুন architecture redesign ছাড়াই GPU measurement চালানো যুক্তিসংগত হবে।
