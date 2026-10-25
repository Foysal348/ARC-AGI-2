<!-- Pasted by the user on 2026-10-24 (GPT audit of E15_MTP_ENGINE_AUDIT_FOR_GPT_2026-10-24.zip). Kept VERBATIM. Claude's validation: Notes/41. -->

E15 MTP Engine — Independent Pre-GPU Audit
Notebook: Q27-XHIGH-E15_MTP_ENGINE_DUCK_SGLANG.ipynb
CONDITIONAL GO

E15-এর GPU measurements চালানোর পরিকল্পনা গ্রহণযোগ্য। MTP launch-এর মৌলিক incompatibility বা নিশ্চিত catastrophic startup bug পাইনি।

তবে বর্তমান exact notebook-এ automatic arm selection-এর দুটি reproducible defect এবং Duck-এর fresh launch verification-এর একটি গুরুত্বপূর্ণ gap আছে।

এগুলো সংশোধন করে একই experiment চালানোর পক্ষে আমি। নতুন architecture বা অতিরিক্ত engine experiment দরকার নেই।

SHA256 — independently matched

2007cba65cf058091d4c13eb216853957cdbfa58e1b05ea36cc53a98dfeb4177

README, E15_cell_5c.py, actual notebook CELL 5c, builder, serving helpers, previous GPU evidence ও CPU harness logs পরীক্ষা করেছি।
1. Notebook শেষ পর্যন্ত চলবে কি?
Verdict: GO for deadline architecture; HIGH risk in final Duck relaunch verification।
1.1 যা ঠিক আছে
Component	Finding
180-minute budget	Correct
Hard stop at 178 minutes	Correct
E15 measurement reserve	Correctly configured
Each arm's relaunch	Budget-aware
Each measurement phase	Separate deadline
Failed optional arm	Later arms can continue
15-minute Duck window	Reserved before engine testing
Experiment report	Incrementally saved

E15_cell_5c.py:14–17 অনুযায়ী:
Reserve	Seconds
Duck solving	900
Stop-before-hard	180
Preflight	300
Artifacts	600
Duck relaunch	400
Total	2,380

e14_phase() অতিরিক্ত 480-second safety margin রাখে। তাই measurement phases স্বাভাবিক deadline মেনে চললে Duck-এর সময় আগেই খরচ হওয়ার কথা নয়।

তবে ১৫–১৬টি model relaunch-এর মধ্যে কিছু বাদ পড়তে পারে। CPU harness-এর compressed-time 22/22 PASS real GPU-তে সব arms শেষ হবে—তার প্রমাণ নয়।
1.2 HIGH — Final Duck launch-এর ফল উপেক্ষা করা হচ্ছে
File: notebook/E15_cell_5c.py:187–198

বর্তমান code:
e14_launch(    "duck_" + _duck,    set_=_da.get("set") or {},    drop=tuple(_da.get("drop") or ()),    force=True,)



Return value পরীক্ষা করা হচ্ছে না।

যদি selected MTP/W6 arm-এর fresh launch ব্যর্থ হয়, e14_launch() আগের E14_KEPT arm restore করার চেষ্টা করে। কিন্তু E15 এরপরও _duck-কে selected arm হিসেবে রিপোর্ট করে।

আর restored server-ও ব্যর্থ হলে Duck solve করতে পারবে না।

Minimal fix:
ok, rec = e14_launch(    "duck_" + _duck,    set_=_da.get("set") or {},    drop=tuple(_da.get("drop") or ()),    force=True,)E15R["duck_launch"] = {    "requested": _duck,    "ok": ok,    "restored": rec.get("relaunched_kept"),    "actual_arm": E14_ARM["name"] if E11_SERVER and E11_SERVER.alive() else None,}



এরপর actual running arm-এর capacity/smoke verification করতে হবে। Requested arm ব্যর্থ হলে successful restored arm দিয়ে Duck চলতে পারে, কিন্তু কোন arm সত্যিকার চলেছে তা রিপোর্ট করতে হবে।

Test: MTP launch FAIL → W6 restore SUCCESS। Duck actual arm W6 হবে; report MTP দাবি করবে না।

এটি আমার মতে GPU-এর আগে নেওয়ার মতো সবচেয়ে গুরুত্বপূর্ণ safety correction।

2. MTP ও high-concurrency flags সঠিক?
Verdict: GO for flags; memory/SSM compatibility partly GPU-unverified।
2.1 MTP launch
E15_cell_5c.py:33–38
--speculative-algorithm EAGLE
--speculative-draft-model-path <RedHat INT4 directory>
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4


পুরোনো DFlash flags _SPEC দিয়ে সরানো হচ্ছে।

Source-level verdict: Correct.

আর একটি গুরুত্বপূর্ণ evidence আছে।

evidence/35 Claude agent - E15 MTP preflight অনুযায়ী E11-03-তে একই RedHat INT4 checkpoint ও SGLang 0.5.21-এ MTP load হয়েছিল।
MTP acceptance
3.05
Mean reported

Maximum running
10
Mamba-capped

Running=10 TPS
374
Prior GPU log


আগের run-এ MTP-এর SSM ছিল FP32, extra_buffer, mamba cache 50। নতুন E15-এ W6 থেকে inherited FP16 ও extra_buffer_lazy ব্যবহার হচ্ছে।

সুতরাং নতুন combination-এর performance ও stability GPU-only unknown।
2.2 Every arm — configuration audit
Arms	Flags / intent	Verdict
Reference W6 graphs13	Existing submitted arm	Correct
Target-only	All speculative flags removed	Correct
W8	Width 6→8	Correct
W6 MRR14	MRR14, graphs 1–14	Correct
W6 nodelay MRR12	--min-free-slots-delay 1, max running12	Correct
W6 at14	MRR17, mamba68, graphs1–17	Correct
W6 at16	MRR20, mamba80, graphs1–20	Correct
W6 at20	MRR24, mamba96, graphs1–24	Correct but high memory risk
MTP 3/1/4	EAGLE + explicit draft path	Correct
MTP mem0.86	Same + mem fraction0.86	Higher OOM risk
MTP 2/1/3	Two steps, three draft tokens	Correct
MTP ReplaySSM FP32	ReplaySSM + FP32 state	Source-supported; GPU-unverified
MTP ReplaySSM FP16	ReplaySSM + inherited FP16 state	Source-supported; accuracy unknown
Best MTP at14/16	Increased MRR/mamba/graph sizes	Correct construction; capacity unknown

e14_edit_flags() correctly removes old flags with their values, including list-valued CUDA graph sizes। Bare None for --enable-linear-replayssm-spec correctly generates a flag without a value।
2.3 New public caution: MTP KV starvation
SGLang issue #44145 describes duplicate MTP embedding/head allocations during memory profiling, which can reduce the reported KV pool even after temporary buffers are freed. This was reported on October 2026 source.

https://github.com/sgl-project/sglang/issues/44145


GitHub


এটি আমাদের precise checkpoint-এ ঘটেছে—এমন প্রমাণ নেই। কিন্তু E15-তে KV tokens, Mamba cache, available memory এবং graph allocation log রাখা অত্যন্ত জরুরি।

Recommendation: MTP startup সফল হলেও KV pool অত্যধিক ছোট হলে arm reject/diagnostic-only করো। কেবল READY status যথেষ্ট নয়।

3. Measurement validity — দুটি independently reproduced bug
এখানেই সবচেয়ে গুরুত্বপূর্ণ findings।
P1 — Probe অনুপস্থিত থাকলেও _gated() PASS
File: notebook/E15_cell_5c.py:101–105

বর্তমান condition:
(_pf is None or _pf <= _aa + 1)



অর্থাৎ long probe একেবারেই না থাকলেও arm accepted হিসেবে বিবেচিত হতে পারে।

আমি ZIP-এর actual AST function আলাদাভাবে execute করে পরীক্ষা করেছি।

Input:
{    "ok": True,    "gate": "PASS",    "log_errors": {},    "probe_flips": None}



Actual _gated() result: True।

কিন্তু README-তে arm selection-এর requirement হিসেবে long-probe large-margin flips উল্লেখ আছে।

এটি especially important কারণ নিচের arms-এ probe_n নেই:
- W8
- W6 MRR14
- W6 nodelay MRR12
- MTP mem0.86
- MTP 2/1/3
- MTP ReplaySSM FP32/FP16
এদের কেউ 256-token gate পাস করে সবচেয়ে দ্রুত হলে Duck-এ নির্বাচিত হতে পারে।
Minimal fix
Automatic adoption-এর জন্য long-probe status explicit করো।
def _gated(a, require_probe=False):    if not a or not a.get("ok"):        return False    if a.get("gate") not in ("PASS", "reference"):        return False    if a.get("log_errors"):        return False    if require_probe:        pf = a.get("probe_flips")        aa = (E15R.get("probe_ref_aa") or {}).get(            "large_margin_flips"        )        if pf is None or aa is None:            return False        if pf > aa + 1:            return False    return True



Efficient implementation: সব arm-এ 1,024-token probe চালানোর দরকার নেই। Short+long speed ও 256-token gate দিয়ে প্রথমে fastest candidate shortlist করো। তারপর শুধু selected candidate-কে long probe দাও। FAIL/INCOMPLETE হলে reference W6।

Test: Missing probe → measurement allowed, automatic Duck adoption disallowed। Probe PASS → eligible।

Priority: Must-fix for automatic best-arm selection।
P1 — 20-stream probe-এর শেষ 8টি responses উপেক্ষিত
File: notebook/E15_cell_5c.py:88–91
return E14M.compare_passes(    E15_PROBE_REF[0],    rows[:len(E15_PROBE_REF[0])],    name,)



Reference probe 12টি prompts ব্যবহার করে।

কিন্তু 14/16/20 stream-এর arm বেশি prompts পাঠায়। Model-এর শেষ অতিরিক্ত responses compare-ই হয় না।

Independent reproduction:
- Reference: 12টি correct rows।
- Arm: প্রথম 12 correct, পরের 8 corrupted।
- _probe_vs_ref() reports:
Metric	Actual result
Expected compared	12
Compared	12
Errors	0
Large-margin flips	0

অর্থাৎ 20-stream probe-এ 8টি ভুল response থাকলেও result নিখুঁত দেখায়।
Fix
Reference-এর 12 prompt indices repeated হয়ে থাকলে corresponding comparison:
ref = E15_PROBE_REF[0]expanded = [ref[i % len(ref)] for i in range(len(rows))]cmp = E14M.compare_passes(expanded, rows, name)



কিন্তু এটিও high-concurrency correctness-এর পূর্ণ proof নয়, কারণ reference C12-তে measured, arm C20-তে। A/A numerical variation-এর distribution আলাদা হতে পারে।

সুতরাং extra rows-এর errors ও divergence আলাদা record করবে:
{    "reference_level": 12,    "arm_level": len(rows),    "all_rows_checked": ...,    "extra_rows_errors": ...,    "cross_level_comparison": True}



Test: C20-এর 13–20 নম্বর response corrupt/timeout করলে report zero errors বলতে পারবে না।

Priority: Must-fix for trustworthy 14/16/20 correctness results।
P2 — Latch test high-concurrency target-এও 12 মাপে
File: notebook/E15_cell_5c.py:63–66

High-concurrency arms-এ:
decodes=SGL_MAX_RUNNING - 1level=SGL_MAX_RUNNING



অর্থাৎ সর্বত্র 11 decodes এবং level12।

এতে 20-stream server-এর latch test-ও মাত্র 14 offered requests পর্যন্ত যায়। 20 running সম্ভব হলো কি না, সেটি এই latch test বলতে পারবে না।

অন্যদিকে steady_window(level=20) সত্যিকার 20 request শুরু করে—তাই high-concurrency throughput measurement পুরোপুরি missing নয়।

Fix: Latch-এর decodes, arrivals, level target concurrency অনুযায়ী করো:
target = leveldecodes = target - 1arrivals = 3



আর measurement report-এ steady_window.real_running_reached, measured_level, running_hist অবশ্যই রাখবে।

Test: At20 arm-এর latch offered ≥20; successful result-এ running_max ≥20।

Priority: Nice-to-have for this run, required before claiming latch success at 20।

4. Duck correctness — selected arm কি নিরাপদ?
Verdict: CONDITIONAL।
যেগুলো ভালো
- প্রতিটি candidate arm-এর আগে 256-token serial/concurrent correctness gate আছে।
- Reference-এর A/A variability বিবেচনা করা হয়।
- Server log-এ OOM/Traceback markers পরীক্ষা হয়।
- Short ও long speed windows নেওয়া হয়।
- Chosen arm fresh launch করা হয়।
- Duck 12 streams-এ সীমাবদ্ধ।
- Existing host-clean verification, candidate harvesting, pass@2 এবং deadline logic অপরিবর্তিত।
তিনটি remaining gap
Gap	Impact	Fix
Missing long probe can pass _gated()	Weakly tested arm adopted	Selected finalist-এর long probe বাধ্যতামূলক
Fresh Duck relaunch unchecked	Wrong arm attribution / unavailable server	Check launch result, actual arm, capacity/smoke
Nested measurement failure ignored	A window can contain errors while top-level phase has no error	Validate short.errors, long.errors, cleanup and meaningful decode samples

শেষের issue-এ E15_cell_5c.py:70–93-এর _arm():
"ok": bool(_ok and _r and not _r.get("error"))



এটি top-level phase error দেখে। কিন্তু steady_window() nested result-এ request errors বা invalid/partial windows রাখতে পারে।

Minimal fix: ok-এর বদলে separate measurement_valid metric। Automatic selection-এ complete, error-free short ও long windows চাইবে। GPU testing-এর জন্য rejected/partial arm-এর data অবশ্যই report-এ রাখবে; নষ্ট data মুছে ফেলবে না।

5. E15-তে আর কী যোগ করা উচিত?
আমার recommendation: খুব বেশি কিছু নয়।

README-তে already 15-এর বেশি serving arms আছে। 180 মিনিটে এগুলোর সবগুলো guaranteed চলবে না।

বর্তমান priorities:
Rank	Addition	Cost	Decision
1	Selected Duck finalist-এর 1,024-token long probe	সাধারণত কয়েক মিনিট	Take
2	Fresh selected-arm capacity/smoke check	অল্প	Take
3	Correct measurement-level recording for C14/C16/C20	CPU + existing GPU windows	Take
4	Chunked prefill 2048	আরেকটি launch	Defer
5	DFlash W6 + FP32 SSM at C12	আরেকটি launch	Defer
6	Mixed-chunk	Correctness risk	Do not add now
7	Cache-restore parity	Valuable for E17	Separate experiment

SGLang-এর hybrid Mamba checkpoint correctness নিয়ে ইতিমধ্যে known issues আছে। তাই throughput experiment-এ untested mixed-chunk বা Mamba patch যোগ করে ফলাফল confound করার প্রয়োজন নেই।
External source corroboration
Official SGLang documentation MTP/NEXTN-এর জন্য EAGLE3/1/4-style flags সমর্থন করে। তবে current main branch pinned 0.5.21-এর সমতুল্য নয়। 

GitHub




https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx

ReplaySSM ও hybrid attention implementation:

https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py

6. অন্য কোনো catastrophic bug, notebook error বা missing artifact?
6.1 Exact integrity
Check	Result
Notebook SHA256	MATCH
Notebook code compilation	13/13 PASS
E15 source vs notebook CELL 5c	Matching executable code
Difference in loose E15 file	Only trailing CELL 6 comment
E14 payload archive	Valid ZIP
Payload module comparison	4/5 current source files equal
e14_sglang_patches.py differs	README documented older payload
GPU execution	Not performed

একটি গুরুত্বপূর্ণ limitation: README-তে Duck payload-এর নাম আছে, কিন্তু এই audit ZIP-এ পৃথক Duck payload file নেই। Notebook external Kaggle payload ব্যবহার করে। CPU harness-এ Duck চলে, কিন্তু ZIP থেকেই deployed Duck payload-এর byte-level audit সম্পূর্ণ করা যায়নি।
6.2 Test results
Bundled evidence:
Suite	Result
Kaggle CPU pytest	167/167 PASS
E15 harness	22/22 PASS
Hidden rerun harness	14/14 PASS
CPU harness package/pins	PASS

আমি notebook compilation, source equality এবং দুইটি false-PASS defect independently reproduced করেছি।

পূর্ণ 167-test suite locally শেষ হয়নি; initial tests PASS হওয়ার পরে দীর্ঘ-running execution timeout হয়েছে। তাই 167/167 bundled validation, independently reproduced নয়।
6.3 Memory & OOM
At20-এর 96 Mamba slots, graphs1–24 এবং W6 verification buffers combined memory footprint বাড়ায়। MTP mem0.86-ও additional pressure তৈরি করে।

তবে প্রতিটি configuration আলাদা relaunch এবং deadline-bounded arm। কোনো experimental arm OOM করলে সেটি পুরো notebook-এর অবশ্যম্ভাবী failure নয়।

Recommendation: OOM-এ rejected arm রাখবে, memory lowering বা retry-loop-এ GPU quota খরচ করবে না।

Final fix priority
GPU-এর আগে নেওয়া উচিত
#	Fix	Classification
M1	Missing long probe-কে automatic adoption PASS না ধরা	Must-fix
M2	14/16/20 probe-এর সব rows validate/report	Must-fix for C20 conclusion
M3	Fresh Duck relaunch-এর success ও actual arm verify; capacity/smoke	Must-fix
M4	Selected arm-এর short/long nested errors ও valid windows gate	Must-fix for selection

এগুলো সবই small, targeted code fixes। Existing architecture, 180-minute budget, arms, Duck agent, model weights ও SGLang version অপরিবর্তিত থাকবে।
Optional — এই run-এর জন্য blocker নয়
High-level latch target সংশোধন, larger 65k context band, chunked prefill 2048, FP32 SSM DFlash, cache replay, additional profiling, 2×TP2 feasibility এবং MTP tuning।

চূড়ান্ত GO/NO-GO
E15-এর verdict
GO — মূল engine benchmark পরিকল্পনা

E15-তে MTP model loading-এর আগের GPU evidence আছে, relevant flags সঠিক এবং 180-minute measurement + 15-minute Duck reserve যথাযথভাবে পরিকল্পিত।

NO-GO — বর্তমান exact SHA-তে automatic fastest-arm Duck adoption

Unprobed arm accepted হওয়া, high-C response বাদ যাওয়া এবং unchecked final relaunch-এর কারণে selected arm-এর speed/correctness conclusion ভুল হতে পারে। M1–M4-এর ছোট patches নিয়ে rebuild করলেই পুনরায় GO যাচাই করা যাবে।

সারকথা: E15 নতুন কিছু গবেষণা করার জন্য যথেষ্ট ভালো ভিত্তি। এখানে MTP weights অনুপস্থিত, SGLang flag invalid বা experiment নিশ্চিতভাবেই crash করবে—এমন কোনো প্রমাণ পাইনি। কিন্তু তিন ঘণ্টার GPU খরচ করে ভুল arm-কে winner ঘোষণা করার ঝুঁকি যথেষ্ট বাস্তব। সেটি দূর করতে অল্প কয়েকটি fix-ই প্রয়োজন; experiment redesign নয়।
