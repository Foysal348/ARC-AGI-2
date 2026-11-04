# GPT audit 3 of 3 of E14-A notebook a9dad9fc (user-pasted 2026-10-23, verbatim)

Validation: Notes/20.

---

ARC-AGI-2 — E14-A Implementation Audit
বর্তমান রায়: NO-GO for the exact notebook SHA
Notebook: Q27-XHIGH-E14A_OPTIMIZED_ENGINE_SGLANG.ipynb
SHA-256: a9dad9fcbe0003867d34f205732a23106dbfed30380535ef4531ba78806de494

এটি E14-A-এর GPU push-এর আগের code audit। কোনো ফাইল পরিবর্তন, commit, Kaggle push বা GPU run করিনি।
১. Integrity ও baseline
যাচাই	ফলাফল
ZIP SHA-256	44cb351807d14dec… — আপনার দেওয়া prefix মেলে
ZIP integrity	40/40 entries, CRC test PASS
Notebook SHA-256	README-এর সঙ্গে exact match
Notebook cell count	13
Included unit-test logs	125/125 × 5 PASS বলে রিপোর্ট করা হয়েছে
Fake-server harness	Run 5-এ 26/26 PASS রিপোর্ট আছে
Actual Kaggle 4×L4 GPU	এই build-এ এখনো পরীক্ষা হয়নি

README-তে লেখা ১৬টি fix এবং তাদের implementation location মিলিয়ে দেখেছি। বেশির ভাগ fix বাস্তবায়িত, কিন্তু গুরুত্বপূর্ণ পার্থক্য হলো: CPU harness সফল মানেই GPU-তে config adoption ও deadline handling সঠিক নয়।

দুটি বিশেষ সমস্যা পাওয়া গেছে:
1. #44401, nodelay ও CUDA graphs13 arm correctness gate ছাড়া Duck run-এর জন্য adopt হতে পারে।
2. Abort-এর পরে thread joins phase deadline অতিক্রম করতে পারে।
দ্বিতীয়টির একাধিক code path রয়েছে। এগুলো README-এর ১–২ নম্বর fix যে সময়সীমা-সমস্যা সমাধান করতে চেয়েছে, তারই অবশিষ্ট অংশ। নির্দিষ্ট fixes নিচে দিচ্ছি।
২. Must-fix findings
P0-A — Candidate arm-এর correctness যাচাই না করেই Duck-এ ব্যবহার
File: Notebook CELL 5b
Lines: 211–239, 251–286

p44401, nodelay, graphs13—এই তিনটি arm adopt করার সিদ্ধান্ত নেওয়া হচ্ছে মূলত throughput এবং latch metrics দেখে।

উদাহরণ:
_keep = bool(_ok and _pp and _speed["adopt"] and _better)



এখানে candidate arm-এর output correctness gate নেই। A1-4_parity_D কেবল baseline DFlash-এর serial/concurrent behavior পরীক্ষা করে। Candidate relaunch-এর পরে একই parity test আবার করা হয় না।

এটি বিশেষ গুরুত্বপূর্ণ কারণ high-concurrency DFlash2 state corruption-এর public report রয়েছে। এর অর্থ আপনার stack-এ corruption নিশ্চিত নয়, কিন্তু শুধু speed দেখে নতুন configuration নির্বাচন করার ঝুঁকি বাস্তব। 

GitHub




Fix: প্রতিটি candidate arm adopt করার আগে একই fixed prompts-এ একটি bounded greedy parity পরীক্ষা করুন।
- Baseline: concurrent-12 A/A।
- Candidate: concurrent-12 output।
- সব planned prompts-এর usable response বাধ্যতামূলক।
- Missing output, timeout অথবা unmeasured margin-কে safe result গণনা করবেন না।
- Gate fail করলে পূর্ববর্তী verified arm-এ ফিরবেন।
প্রতিটি arm-এর জন্য দীর্ঘ 512-token × 12 × multiple-repeat benchmark চালানো প্রয়োজন নেই। GPU quota বিবেচনায় একটি ছোট shared probe suite যথেষ্ট প্রাথমিক gate হতে পারে; বড় correctness comparison পরে করা যাবে।

Regression: fake server-এ arm-এর speed 10% বাড়িয়ে একটি large-margin output পরিবর্তন করান। প্রত্যাশিত ফল adopt=False।

P0-B — Deadline শেষ হওয়ার পরেও thread joins বহু মিনিট চলতে পারে
File: src/e14_measure.py
Function	Location	সমস্যা
steady_window()	287–288	প্রতি generator-এ join(60)
latch_test()	476–478	১৪টি generator পর্যন্ত join(60)
parity_pass()	409–420	একাধিক request-এর sequential waiting
agent_replay()	534–537	Shared deadline নয়, প্রতি thread-এ join timeout
abort()	208–213	প্রতি request-এর HTTP abort sequential

একটি ১৪-generator latch test-এ worst-case sequential joins:
\[
14\times60=840\text{ seconds}
\]

অর্থাৎ ১৪ মিনিট—এর সঙ্গে abort overhead যোগ হতে পারে।

এটি theoretical worst case, recorded 26/26 harness failure নয়। কিন্তু একটি GPU hang ঘটলে এই code path phase budget ভাঙতে পারে।

Fix: প্রতিটি phase-এর জন্য একটি shared absolute shutdown deadline ব্যবহার করুন। abort() calls bounded করুন এবং সব thread-এর জন্য মোট 10–20 সেকেন্ডের join budget রাখুন, প্রতি thread-এর জন্য আলাদা 60 সেকেন্ড নয়।

Abort-এর পরে server-এ requests এখনো চললে পরবর্তী arm measurement শুরু করবেন না; server relaunch বা phase invalidation করুন।

Validation:
def test_hung_workers_respect_shared_deadline():    # 14 generators never return normally.    # Abort may fail or time out.    # All cleanup shares one bounded deadline.    assert phase_cleanup_sec <= 20    assert next_arm_not_started_with_live_requests



GO impact: Must-fix। এটি আপনার README-এর প্রথম দুটি timeout fix-এর উদ্দেশ্যের সঙ্গে সরাসরি সম্পর্কিত।

P0-C — Nodelay latch metric আসল 11→12 improvement ধরছে না
File: src/e14_measure.py:455–487

বর্তমান code:
offered = decodes + arrivalswant = min(offered, server_cap)



আপনার test-এ decodes=11, arrivals=3, server_cap=13। তাই want=13।

কিন্তু মূল সমস্যা হলো 11 running + queued থেকে 12 running-এ পৌঁছানো।

বর্তমান share_stuck_one_below শুধু running 12 + queued গণনা করে, running 11 + queued নয়। ফলে সবচেয়ে গুরুত্বপূর্ণ failure mode metric-এ ধরা পড়ে না।

এর বাস্তব প্রমাণ fake-server run 5-এ আছে:
Arm	Speed ratio	share_at_target	Decision
#44401	1.000	0	Adopt
Nodelay	1.000	0	Adopt
Graphs13	1.000	—	Adopt

এই fake-server result বাস্তব GPU লাভের প্রমাণ নয়; বরং test decision-এর সীমাবদ্ধতা দেখায়।

Fix: দুটি metric যোগ করুন:
share_running_ge_12 = sum(    r["running"] >= 12 for r in ok) / len(ok)share_stuck_11_queued = sum(    r["running"] == 11 and (r.get("queue") or 0) >= 1    for r in ok) / len(ok)



share_at_target_13 diagnostic হিসেবে রেখে দিন। Practical 12-stream decision-এ 11→12 improvement, queue pressure এবং total tokens/s ব্যবহার করুন।

Validation: Synthetic records-এ baseline 11-running/1-queued এবং candidate 12-running দেখিয়ে candidate-কে improvement হিসেবে শনাক্ত করতে হবে। Baseline ও candidate একই হলে শুধু noise tolerance-এর কারণে improvement দাবি করা যাবে না।

HIGH-D — Speed/parity helper-এর fail-open acceptance
File: src/e14_measure.py:550–567

আমি সরাসরি function import করে তিনটি reproduction চালিয়েছি।
Reproduction 1: মাত্র একটি response, তবুও correctness PASS
parity_ok(    {"compared": 1, "identical": 0, "large_margin_flips": 0},    {"compared": 12, "identical": 0},    allowance=2)["ok"]# True



Reproduction 2: সব ছয়টি output-ই আলাদা, তবুও PASS
parity_ok(    {"compared": 6, "identical": 0, "large_margin_flips": 0},    {"compared": 6, "identical": 0})["ok"]# True



Reproduction 3: ভিন্ন context/concurrency, তবুও speed adopt
adopt_speed(    {"tok_s_p50": 370, "measured_level": 11,     "lines": 1, "ctx_per_req_p50": 8000},    {"tok_s_p50": 365, "measured_level": 12,     "lines": 0, "ctx_per_req_p50": 40000})["adopt"]# True




তিনটির সবগুলোই independently reproduced।

এখানে দ্বিতীয় result শুধু তখনই acceptable হতে পারে যখন A/A control-ও সমানভাবে unstable এবং আলাদা equivalence criterion পূরণ হয়েছে। বর্তমান function সেটি প্রমাণ করে না।

Fix: parity_ok()-তে expected comparison count, missing margins, timeout/error count এবং sample completeness বাধ্যতামূলক করুন। Speed decision-এ minimum valid samples, actual measured concurrency ও context band check করুন।

11→12 running-এর deliberate comparison অনুমোদিত হতে পারে; সেটি explicit occupancy-gain experiment হিসেবে label দিতে হবে। অন্যথায় context ও sample mismatch থেকে speed winner নির্বাচন করবেন না।

Tests: Empty, incomplete, no-margin, 1/12, 0/6 identical এবং mismatched-context cases—সব fail closed। Valid noisy A/A case অকারণে fail করা যাবে না।

HIGH-E — ২৫ মিনিট Duck solver-এর জন্য সময় সংরক্ষিত নয়
Notebook: CELL 5b ও CELL 6

A1 measurements একাধিক relaunch, profiling ও parity phase চালায়। প্রতিটি phase নিজের deadline মানার চেষ্টা করে, কিন্তু পরবর্তী 25-minute Duck run-এর জন্য আলাদা সময় reserve করে না।

CELL 6-এ solver end নির্ধারণ হয়:
_end = min(    HARD_STOP - E12_STOP_BEFORE_HARD_SEC,    time.perf_counter() + E12_SOLVE_WINDOW_SEC)



A1 বেশি সময় নিলে solver তার সম্পূর্ণ 25 মিনিট পাবে না। Worst-case stall-এ এটি প্রায় শূন্যও হতে পারে।

Fix: A1-এর প্রতি নতুন relaunch-এর আগে Duck-এর 25-minute window, solver shutdown margin এবং final artifact margin সংরক্ষণ করে remaining-time gate চালান।

উদাহরণ:
reserve = (    E12_SOLVE_WINDOW_SEC    + E12_STOP_BEFORE_HARD_SEC    + 10 * 60)if time.perf_counter() + predicted_phase_cost > HARD_STOP - reserve:    skip_optional_arm()



এখানে predicted cost-এর fallback conservative হতে হবে; warm restart-এর median দিয়ে worst-case guarantee দাবি করা যাবে না।

Test: Simulated slow-loading notebook-এ A1 optional phases skip হয়ে Duck-এর reserved window বজায় থাকে।

এটি solver logic change নয়—measurement scheduler-এর budget protection।

৩. /flush_cache — README প্রশ্ন ৩
File: Notebook CELL 5b:287–290

Source excerpt অনুযায়ী SGLang 0.5.21-এর /flush_cache plain text দেয়। Request চলতে থাকলে operation প্রত্যাখ্যান করতে পারে।

বর্তমান code response save করে, কিন্তু failure text বা flush success পরীক্ষা করে না।

এখানে দুটি বিষয় আলাদা:
- Endpoint call করা নিরাপদ কি না?
- Cache সত্যিই খালি হয়েছে কি না?
প্রথমটি স্বাভাবিক server API ব্যবহারের মধ্যে পড়ে। দ্বিতীয়টি বর্তমান implementation নিশ্চিত করে না।

Minimal fix: response text-এ successful flush শনাক্ত করুন; ব্যর্থ হলে diagnostic-এ cache_not_cleared record করুন এবং Duck result-কে clean-cache comparison বলে উপস্থাপন করবেন না।

Mamba-radix state সম্পূর্ণ পরিষ্কার হয়েছে কি না বোঝার জন্য server-side cache state ও একটি cold-prefix probe ব্যবহার করুন।

GPU correctness blocker? একা নয়। কিন্তু clean-cache baseline দাবি করার blocker।
৪. Width 6/4 — README প্রশ্ন ৫
--speculative-num-draft-tokens বদলানো এই checkpoint-এর supported runtime path। Source অনুযায়ী set_block_size() ব্যবহার হয়।

কিন্তু is_causal=False হওয়ায় এটি width-8 output truncate করার সমান নয়। Width পরিবর্তনের ফলে drafter-এর candidate distribution পরিবর্তিত হতে পারে।

বর্তমান A2-তে serial parity ও throughput আছে। উন্নতির জায়গা:
- Width 6/4-এর জন্য sampled acceptance distribution আলাদা record করুন।
- Baseline ও candidate একই concurrency-তে পরীক্ষা করুন।
- Greedy serial parity-এর পাশাপাশি ছোট concurrent parity রাখুন।
- কোনো candidate output error বা missing margin থাকলে adopt=False।
- Width 4-এর maximum acceptance length 4-এর বেশি হলে instrumentation/configuration failure হিসেবে flag করুন।
আরেকটি সমস্যা: A2-এর int8-Mamba correctness gate baseline-এর cold-versus-cached consistency এবং নতুন arm-এর cold-versus-cached consistency তুলনা করে। এটি সরাসরি baseline output বনাম int8 output equivalence পরীক্ষা নয়।

Fix: identical prompts-এ direct cross-arm comparison যোগ করুন। এটি ভবিষ্যতের int8 adoption decision-এর জন্য গুরুত্বপূর্ণ; বর্তমান Duck run যেহেতু A2-এর আগে শেষ হয়, তাই immediate solver blocker নয়।
৫. #44401 ও nodelay নিয়ে চূড়ান্ত মূল্যায়ন
Public PR #44401 সত্যিই merged হয়েছে এবং continuing chunked-prefill request-এর held req_to_token row double-count সমস্যাটি সংশোধন করে। এটি আপনার 11-running/queued observation-এর সঙ্গে মেলে। 

GitHub




https://github.com/sgl-project/sglang/pull/44401

তবে #44401 এবং --min-free-slots-delay 1 একই সমস্যা সমাধান করে—এমন নয়।
- #44401: allocator bookkeeping ও admission-latch correction।
- Nodelay: deliberate request admission delay disable করে।
- Graphs13: CUDA-graph padding/capture coverage পরিবর্তন করে।
এই তিনটি intervention আলাদা। তাই cumulative arm testing গ্রহণযোগ্য, কিন্তু প্রতিটির measured contribution ও adoption reason পৃথক থাকতে হবে।

আমার পছন্দ হলো initial 11→12 occupancy improvement, overall tok/s, cache cost এবং correctness—চারটি একসঙ্গে বিবেচনা করা। শুধুমাত্র share_at_target=13 অথবা short-window ratio যথেষ্ট নয়।
৬. GPU quota বাঁচাতে A2-এর কোন অংশ আগে বাদ দেব?
বর্তমান গবেষণার উদ্দেশ্য অনুযায়ী priority:
A2 component	Priority	সিদ্ধান্ত
Target-only correctness	Highest	রাখুন; সম্ভব হলে ছোট pre-Duck gate
Agent-like cache replay	High	রাখুন; re-prefill root cause গুরুত্বপূর্ণ
NCCL winning env	High, conditional	Bench-এ বড় লাভ + transport safety evidence থাকলে রাখুন
Width 6	Medium	সময় থাকলে
Width 4	Lower	Width 6 খারাপ হলে বা profiler row-cost বেশি দেখালে
Int8 Mamba checkpoint	Lowest now	আগে বাদ দিন; output equivalence ও memory benefit আলাদা যাচাই দরকার

যদি quota ৭ ঘণ্টার নিচে থাকে, আমি প্রথমে int8 Mamba arm এবং width 4 বাদ দেব। এরপর NCCL env arm-এর decision benchmark evidence অনুযায়ী হবে।
৭. Test reproducibility: কী independently করেছি
Notebook-এর সব code cell compile() করে 12/12 PASS পেয়েছি।

CELL 1b থেকে embedded পাঁচটি E14 Python module decode করে source-এর সঙ্গে byte comparison করেছি:
Module	Match
e14_loadpath.py	PASS
e14_measure.py	PASS
e14_sglang_patches.py	PASS
e14_nccl_bench.py	PASS
e14_trace_summary.py	PASS

তবে এই Linux environment-এ আপনার README-র full pytest suite independently PASS করাতে পারিনি। দুটি কারণ:
1. Test command-এ module path resolution-এর জন্য PYTHONPATH=src প্রয়োজন হয়েছে।
2. কিছু tests সরাসরি D:/ARC AGI 2/... Windows path-এর external fixtures এবং original SGLang source tree ব্যবহার করে। সেই reference tree ZIP-এ নেই।
সুতরাং 125/125 × 5-কে provided logs-এ recorded PASS হিসেবে গ্রহণ করছি, independently reproduced বলছি না।

এটি build-এর বিরুদ্ধে P0 নয়, কিন্তু final audit evidence আরও portable করতে external fixture paths configurable করা ভালো।
৮. Must-fix বনাম nice-to-have
Priority	Item	GPU push-এর আগে
P0	Candidate arms-এর correctness gate	Fix
P0	Shared deadline ও bounded abort/join cleanup	Fix
P0	Nodelay-এর 11→12 latch decision	Fix
HIGH	parity_ok এবং adopt_speed completeness/validity	Fix
HIGH	Duck 25-minute budget reservation	Fix
MEDIUM	/flush_cache success verification	একই patch set-এ নিন
MEDIUM	A2 int8 direct baseline parity	পরবর্তী adoption-এর আগে
LOW	Linux-portable tests/fixtures	Nice-to-have
LOW	Longer 60k-context band	এখন নয়
LOW	নতুন scheduler, HiCache, TP2, GPU kernel redesign	এখন নয়

৯. চূড়ান্ত GO/NO-GO
NO-GO — SHA a9dad9fc…
কারণ: Notebook packaging ঠিক আছে, ১৬টি fix-এর অধিকাংশ implement হয়েছে, কিন্তু candidate selection, measurement completeness এবং deadline guarantees-এ code-confirmed gaps রয়েছে।

এটি E14-A architecture বাতিল করার রায় নয়। পাঁচটি focused fix একসঙ্গে করলে আবার GPU-ready হওয়া সম্ভব; অপ্রয়োজনীয় feature বা serving-engine redesign দরকার নেই।

Claude-কে আমার consolidated নির্দেশ হবে: P0-A, P0-B, P0-C, HIGH-D এবং HIGH-E একসঙ্গে সংশোধন, /flush_cache success validation যোগ, তারপর আগের 125 tests + নতুন targeted failure-injection tests + notebook harness পুনরায় চালানো।

বিশেষ করে fake-server harness-এ তিনটি নতুন scenario চাই: একটি faster-but-wrong candidate, একটি hung abort with 14 active requests, এবং একটি 11-running/queued baseline থেকে 12-running nodelay candidate। এই তিনটিতেই সিদ্ধান্ত ঠিক হলে পরবর্তী GPU run-এর evidence অনেক বেশি বিশ্বাসযোগ্য হবে।

GPU run এখনো অনুমোদিত নয়। কোনো কোড পরিবর্তনও করিনি।
