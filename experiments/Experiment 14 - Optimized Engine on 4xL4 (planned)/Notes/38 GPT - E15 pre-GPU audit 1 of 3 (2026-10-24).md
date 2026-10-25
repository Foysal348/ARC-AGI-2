<!-- Pasted by the user on 2026-10-24 (GPT audit of E15_MTP_ENGINE_AUDIT_FOR_GPT_2026-10-24.zip). Kept VERBATIM. Claude's validation: Notes/41. -->

ARC-AGI-2 — E15 MTP Engine Final Pre-GPU Audit
NO-GO — exact notebook SHA, as built
কারণ: তিনটি গুরুত্বপূর্ণ gap পাওয়া গেছে, যার মধ্যে দুটি independently reproduced। এগুলো MTP engine-এর architecture সমস্যা নয়; মূলত ভুল engine arm নির্বাচন এবং 14–20-stream measurement-এর নির্ভরযোগ্যতা নিয়ে। সীমিত code changes-এ ঠিক করা সম্ভব।

আপনার আগের নির্দেশ মেনে অপ্রয়োজনীয় redesign, নতুন feature বা purely theoretical risk দিয়ে NO-GO দিচ্ছি না।
অডিট পরিচিতি
ZIP: E15_MTP_ENGINE_AUDIT_FOR_GPT_2026-10-24(1).zip

Notebook: Q27-XHIGH-E15_MTP_ENGINE_DUCK_SGLANG.ipynb

Verified SHA-256:
2007cba65cf058091d4c13eb216853957cdbfa58e1b05ea36cc53a98dfeb4177


যাচাই	ফল
ZIP integrity	40/40 entries PASS
Notebook SHA	README-এর সঙ্গে exact match
Notebook code cells	13/13 compile PASS
E15 cell বনাম embedded notebook	Logic identical
E14 payload modules	5টির মধ্যে 4টি current workspace-এর সঙ্গে identical
e14_sglang_patches.py	Workspace copy ভিন্ন; README-তে documented #43567 addition
Provided CPU pytest	167/167 PASS
Provided E15 harness	22/22 PASS
Provided rerun harness	14/14 PASS
Independent focused tests	14/14 PASS
Real E15 GPU execution	এখনো হয়নি

সম্পূর্ণ pytest suite এই পরিবেশে স্বাধীনভাবে PASS করাতে পারিনি; environment-dependent subprocess tests-এ failure/hang হয়েছে। তাই provided CPU test results এবং আমার independent checks আলাদা রাখছি।

১. Will it run to the end?
Verdict: CONDITIONAL GO

মূল execution design সঠিক:
Initial W6 + graphs13 launch
          ↓
Reference A/A + speed windows
          ↓
Target-only, DFlash, MTP arms
          ↓
Select candidate
          ↓
Fresh forced server launch
          ↓
15-minute Duck run
          ↓
Report + bundle


যা সঠিক পেয়েছি
notebook_cells.py:1537–1587, 1608–1647
- Measurement launches Duck reserve বিবেচনা করে।
- প্রতিটি phase-এর নিজস্ব deadline আছে।
- Normal launch failure-এ error record হয়।
- Failed candidate স্বয়ংক্রিয়ভাবে adopted arm হয় না।
- Measurement phase-এর exception report-এ record হয়।
- Duck-এর আগে fresh launch রাখা হয়েছে।
- End-of-run report ও ZIP bundling path আছে।
Time budget
E15-এর nominal budget 180 মিনিট; HARD_STOP 178 মিনিটে।

E15_cell_5c.py:14–16-এ reserved time:
Component	Seconds
Duck solving	900
Stop margin	180
Preflight	300
Artifacts	600
Duck relaunch	400
Total reserve	2,380 s (39m 40s)

Measurement phases-এর অতিরিক্ত 8-minute margin-ও আছে। Normal execution-এর জন্য এটি যুক্তিসঙ্গত।

কিন্তু 16 launches শেষ হবে—এমন guarantee নেই। Slow CUDA-graph capture, server failures বা long cleanup হলে শেষের MTP arms skip হবে। এটি README-এর skip-from-end policy অনুযায়ী acceptable।
HIGH-1 — Final Duck launch success যাচাই করা হয়নি
E15_cell_5c.py:187–189

বর্তমান code:
E14_RESERVE = 0.0_da = E15R["arms"].get(_duck) or {}e14_launch(    "duck_" + _duck,    set_=_da.get("set") or {},    drop=tuple(_da.get("drop") or ()),    force=True,)



এখানে e14_launch()-এর returned success ignored।

যদি নির্বাচিত MTP arm fresh launch-এ ব্যর্থ হয়, function kept-arm restore চেষ্টা করে। কিন্তু restore-ও ব্যর্থ হলে পরের CELL 6 পুরোনো ENGINE_READY=True দেখে Duck চালানোর চেষ্টা করতে পারে।

Fix: Return status record করুন এবং effective running arm যাচাই করুন।
_okd, _rd = e14_launch(    "duck_" + _duck,    set_=_da.get("set") or {},    drop=tuple(_da.get("drop") or ()),    force=True,)_restored = bool(    (_rd.get("relaunched_kept") or {}).get("ok"))E15R["duck_launch"] = {    "requested": _duck,    "requested_ok": bool(_okd),    "restored_kept": _restored,    "effective_arm": E14_ARM.get("name"),}E15R["duck_server_ready"] = bool(    (_okd or _restored) and    E14M.server_idle(E14_CTX, wait=5))



Duck CELL 6-এর আগে duck_server_ready check করতে হবে। False হলে safe fallback/report path ব্যবহার করতে হবে; dead server-এ 15 মিনিট inference চালানো যাবে না।

Test: Selected MTP launch FAIL, kept-arm restore PASS → solver restored arm-এ চলে। উভয় launch FAIL → solver inference skip, valid fallback এবং diagnostic report সংরক্ষিত।

২. SGLang 0.5.21 flags audit
Verdict: GO — মূল flags সঠিক
Arm	Verdict	মন্তব্য
ref_w6g13	PASS	Width 6, graphs 1..13
target_only	PASS	Speculative flags removed
w8	PASS	Draft width 8
w6_mrr14	PASS	MRR 14, graphs 1..14
w6_nodelay_mrr12	PASS	Delay 1, MRR 12; আগের 13-running crash এড়ানোর design
w6_at14	PASS for flags	MRR 17, Mamba 68, graphs 1..17
w6_at16	PASS for flags	MRR 20, Mamba 80, graphs 1..20
w6_at20	PASS for flags	MRR 24, Mamba 96, graphs 1..24
mtp_3_1_4	PASS	EAGLE 3/1/4, explicit INT4 path
mtp_3_1_4_mem86	PASS	Static memory 0.86
mtp_2_1_3	PASS	EAGLE 2/1/3
mtp_replay_fp32	PASS for configuration	ReplaySSM with FP32
mtp_replay_fp16	PASS for configuration	ReplaySSM with FP16

e14_edit_flags() correctly supports set_ values as lists and None for boolean flags.
MTP flags
--speculative-algorithm EAGLE
--speculative-draft-model-path <INT4_MODEL_DIR>
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4


এই combination আপনার পূর্ববর্তী E11-03 GPU experiment-এ MTP head load ও acceptance দেখিয়েছে। Bundle-এর Notes/35 অনুযায়ী mean acceptance প্রায় 3.05।
MTP memory risk
এখানে গুরুত্বপূর্ণ external evidence আছে।

SGLang v0.5.21-এর issue #44145-এ deguzguy, ১৩ অক্টোবর ২০২৬-এ দেখিয়েছেন যে MTP embedding/head sharing KV pool allocation-এর পরে হওয়ায় available KV capacity ভুলভাবে ছোট হতে পারে। তাঁর RTX 5090-এ reported KV pool 15,363 থেকে patch-এর পরে 169,603 tokens হয়েছিল। 

GitHub




https://github.com/sgl-project/sglang/issues/44145

আপনার পূর্ববর্তী MTP run-এও pre-sharing memory overhead-এর evidence আছে, কিন্তু সেটি exact একই severity হবে—এমন নয়।

Recommendation: E15-তে MTP arm অবশ্যই রাখুন। তবে expected ~1.2M KV pool-কে verified fact ধরবেন না। Real startup থেকে actual KV/Mamba pool record করুন।

Mandatory before adoption: নির্বাচিত MTP arm-এর actual KV, Mamba slots, free VRAM ও CUDA graph capture completeness যাচাই করতে হবে। Startup successful হলেই capacity sufficient—এমন নয়।
OOM risk
w6_at20-এ MRR 24, Mamba 96 এবং graphs 1..24 aggressive। OOM হতে পারে, কিন্তু সেটি E15-এর উদ্দেশ্যেই পরীক্ষিত upper limit।

Launch failure safely recorded হলে এই arm-কে শুধু OOM risk-এর জন্য বাদ দেওয়ার প্রয়োজন নেই।

৩. Measurement validity
NO-GO until two fixes
এখানে সবচেয়ে গুরুত্বপূর্ণ bugs পাওয়া গেছে।
P0-1 — Incomplete/failed long probe correctness PASS হিসেবে গণ্য হতে পারে
Location: E15_cell_5c.py:71–74, 107–111

বর্তমান code:
def _gated(a):    _aa = (        E15R.get("probe_ref_aa") or {}    ).get("large_margin_flips") or 0    _pf = (a or {}).get("probe_flips")    return bool(        a and a.get("ok")        and a.get("gate") in ("PASS", "reference")        and not a.get("log_errors")        and (_pf is None or _pf <= _aa + 1)    )



দুটি সমস্যা:
1. _pf is None হলে probe না চললেও candidate PASS।
2. Probe-এ সব request fail করলেও large_margin_flips=0 হতে পারে।
Independent reproduction:
from e14_measure import compare_passesbad = [    {"ids": [], "error": "request failed", "margins": []}    for _ in range(12)]cmp = compare_passes(bad, bad, "failed probe")



ফল:
expected = 12
compared = 0
errors = 12
large_margin_flips = 0


অর্থাৎ ১২টি request-ই ব্যর্থ, তবু flip count শূন্য।

এটি _gated()-এ একটি plausible pass তৈরি করতে পারে, কারণ function errors, compared, expected, missing_margins যাচাই করে না।

আর mtp_3_1_4_mem86, mtp_2_1_3, mtp_replay_fp32, mtp_replay_fp16-এ long probe নেই। ফলে এগুলো 256-token short gate এবং speed-এর ভিত্তিতেই Duck candidate হতে পারে।

Fix: Missing probe-কে auto-adoption-এর অনুমতি দেবেন না। Candidate-এর probe completeness আলাদা field-এ record করুন।
def _probe_valid(cmp):    return bool(        cmp        and cmp.get("expected", 0) >= 12        and cmp.get("compared") == cmp["expected"]        and cmp.get("errors") == 0        and cmp.get("missing_margins") == 0    )



MTP candidate Duck-এর জন্য eligible হবে শুধু valid long probe থাকলে। Probe না চালানো MTP variants diagnostic হিসেবে থাকবে, কিন্তু auto-adopt হবে না।

অন্য বিকল্প: সব MTP candidate-এ bounded 1,024-token probe চালানো। এটি বেশি GPU খরচ করবে; minimal fix হলো unprobed arms-কে শুধু adoption থেকে বাদ দেওয়া।

Validation tests:
- 12/12 probe failed → reject।
- No reference probe → reject।
- Candidate probe missing → reject।
- 12/12 compared এবং acceptable flips → allow।
- Probe pass কিন্তু short gate fail → reject।
Severity: Must-fix, কারণ ভুল MTP arm Duck run-এ পৌঁছাতে পারে।
P0-2 — Latch-fix override long-context throughput gate bypass করে
Location: E15_cell_5c.py:175–185

Normal _cand selection-এর জন্য condition:
long_tok_s >= 0.98 * reference_long_tok_s



কিন্তু _fix override-এ শুধু short throughput এবং latch metric দেখা হয়।

Independent reproduction: নিচের synthetic case-এ current logic latch fix গ্রহণ করে:
Metric	Reference	Latch candidate
Short throughput	500	495
Long throughput	450	170
Latch ≥12 share	0.0	0.6
Short gate	Reference	PASS

Candidate-এর long throughput reference-এর মাত্র 37.8%, তবু _fix তাকে বেছে নিতে পারে।

এটি README-এর “long tok/s ≥ 0.98 × reference” adoption policy-এর সরাসরি লঙ্ঘন।

Minimal fix: _fix filter-এ একই long throughput condition যুক্ত করুন।
def _long_ok(a, ref):    base = ref.get("long_tok_s")    value = (a or {}).get("long_tok_s")    return bool(        isinstance(base, (int, float))        and base > 0        and isinstance(value, (int, float))        and value >= 0.98 * base    )



তারপর _fix list comprehension-এ:
and _long_ok(E15R["arms"][k], _ref)



Validation: Short + latch ভালো কিন্তু long 170 বনাম 450 → reject। Short + latch ভালো এবং long ≥441 বনাম 450 → অন্য gates PASS হলে accept।

Severity: Must-fix। এটি নিশ্চিত code-level decision bug, শুধু theoretical risk নয়।
HIGH-3 — 14/16/20-stream latch test আসলে ওই levels পরীক্ষা করছে না
Location: E15_cell_5c.py:55–57

বর্তমান call:
E14M.latch_test(    E14_CTX, _S, _L,    decodes=SGL_MAX_RUNNING - 1,    arrivals=3,    level=SGL_MAX_RUNNING,    server_cap=latch_cap,)



SGL_MAX_RUNNING=12, তাই:
initial decodes = 11
arrivals        = 3
total offered   = 14
tested level    = 12


এটি w6_at16 বা w6_at20 arm-এর ক্ষেত্রেও একই।

ফলে latch_test দিয়ে 16 বা 20 active request হয়েছে কি না কখনো জানা যাবে না, কারণ মাত্র 14 request পাঠানো হচ্ছে।

এটি E15-এর অন্যতম প্রধান গবেষণা প্রশ্নের measurement defect।

Fix:
_test_level = level or SGL_MAX_RUNNINGout["latch"] = E14M.latch_test(    E14_CTX, _S, _L,    decodes=_test_level - 1,    arrivals=3,    secs=_ts(60),    server_cap=latch_cap,    level=_test_level,)



এখানে আগের 12-stream latch behaviour অপরিবর্তিত থাকবে।

Tests: level 12→14 offered; level 14→16; level 16→18; level 20→22। প্রত্যেক ক্ষেত্রে expected share_running_ge_level নতুন target-এর জন্য গণনা হবে।

steady_window() নিজেও actual measured_level দেখে, তাই পুরো high-concurrency experiment অকার্যকর নয়। কিন্তু latch test-এর এই bug ঠিক করা উচিত।
Measurement-এর অন্যান্য সীমা
- Duplicate 12 prompts cycle করে 14/16/20 stream তৈরি করা হয়। Radix-cache sharing latency ও throughput-কে প্রভাবিত করতে পারে। এটি controlled-engine workload হিসেবে গ্রহণযোগ্য, independent 20-task workload-এর সমতুল্য নয়।
- 14/16/20 concurrency probe-এর প্রথম 12 output reference c12-এর সঙ্গে index-aligned, কিন্তু অতিরিক্ত 2/4/8 output correctness comparison-এর বাইরে থাকে।
- BF16 logits-এর ties এবং ছোট margin-এর কারণে greedy divergence নিজে corruption প্রমাণ করে না। Reference A/A control রাখা সঠিক সিদ্ধান্ত।
উপরের শেষ তিনটি বিষয় experimental limitations, mandatory blockers নয়।

৪. Duck run-এর correctness ও safety
Verdict: NO-GO until adoption and fresh-launch checks fixed

Current selection path:
Candidate gate
      ↓
Short + long speed
      ↓
Optional probe check
      ↓
Latch override
      ↓
Fresh selected-arm launch
      ↓
Duck


এখানে Optional probe check এবং latch override-ই সমস্যার জায়গা।

এছাড়াও e14_launch() selected arm-এর জন্য actual KV/Mamba capacity gate চালায় না। প্রথম server-এর initial e11_capacity_gate() এবং প্রতিটি subsequent measurement-arm launch এক জিনিস নয়।

Minimal safeguard: Duck-এর জন্য নির্বাচিত arm fresh launch হওয়ার পরে:
- server reachable ও idle;
- kv_tokens logged এবং configured 12 streams-এর memory minimum পূরণ করে;
- Mamba cache cap কমপক্ষে 12-running অনুমোদন করে;
- target verify CUDA graph coverage আছে;
- free GPU headroom minimum পূরণ করে;
- effective arm selected arm অথবা explicitly restored verified reference।
এগুলো report করুন। Gate ব্যর্থ হলে reference restore করুন; কোনো অনিরীক্ষিত MTP server দিয়ে Duck চালাবেন না।
GPU memory বিষয়ে
mem=0.86 এবং ReplaySSM-এর predicted KV gain আকর্ষণীয়। কিন্তু এগুলো local memory model, measured E15 result নয়।

#44145-এর মতো duplicate MTP allocation problem থাকলে predicted 1.2M KV pool কমে যেতে পারে। তাই memory estimates-কে adoption gate-এর পরিবর্তে ব্যবহার করা যাবে না। 

GitHub




৫. এই run-এ আর কী যোগ করা উচিত?
Verdict: নতুন বড় experiment যোগ করার দরকার নেই।

আপনার E15 ইতোমধ্যে প্রায় 16 launches পরীক্ষা করতে চায়। এখন আরও পরিবর্তন যোগ করলে GPU quota ও attribution দুটোই ক্ষতিগ্রস্ত হতে পারে।

Priority অনুযায়ী:
Rank	Idea	সিদ্ধান্ত
1	Candidate probe completeness + final-arm KV/smoke	Must-fix existing gates
2	High-concurrency latch correction	Must-fix measurement
3	DFlash W6 + FP32 SSM	Optional, E15-এর পরে
4	Chunked prefill 2048	Later; isolated agent-like replay প্রয়োজন
5	--enable-mixed-chunk	এখন নয়; hybrid state risk ও speculative decode degradation
6	Cache-restore parity	E17 correctness analysis-এর জন্য রাখুন

বিশেষ করে --enable-linear-replayssm-spec MTP-এর সঙ্গে পরীক্ষা করা যুক্তিসঙ্গত, কিন্তু DFlash GDN path-এর জন্য এই bundle-এর source analysis অনুযায়ী supported নয়। তাই DFlash+ReplaySSM arm না যোগ করার সিদ্ধান্ত সঠিক।

৬. Source, variable scope ও reporting audit
Verdict: MOSTLY GO, কিন্তু উপরের gates সংশোধন দরকার।
বিষয়	ফল
E15_REF via globals()	Notebook top-level execution-এ কাজ করার মতো
E15_PROBE_REF via globals()	একইভাবে গ্রহণযোগ্য
_tok() key tok_s_p50	Correct
share_running_ge_level	Correct key; ভুল target level পাঠানো হচ্ছে
_probe() index order	প্রথম 12 prompts aligned
_probe_vs_ref()	Missing/error completeness validation নেই
_gated()	Incomplete probe PASS করতে পারে
_cand	Long throughput gate আছে
latch_fix	Long throughput gate bypass করে
Final e14_launch()	Return value ignored
e14_save()	Arm/decision records save করে
End-of-run bundle	Code path আছে; CPU harness PASS
E14_DO_MEASURE in E15	False, intended
E15 MTP flags	Correct source-level configuration

Report accuracy
E15R["arms"]-এ high-concurrency arm-এর requested level এবং actual measured level আলাদা করে সংরক্ষণ করা ভালো। বর্তমান level_short/level_long আছে, কিন্তু report-এ নাম দেখে “at20 successful” বলা যাবে না।

একটি অতিরিক্ত boolean:
"reached_requested_short": (    (_r or {}).get("short", {}).get("measured_level")    == (level or SGL_MAX_RUNNING))



এর সমতুল্য long field রাখলে reporting আরও স্পষ্ট হবে। এটি convenience fix, blocker নয়।

৭. Fix-before-GPU ranking
1. Fail-closed probe / MTP adoption
BLOCKER

12/12 failed বা missing long probe-কে successful evidence ধরা যাবে না।
2. Latch override long-speed check
BLOCKER

Long-context performance reference-এর 98%-এর নিচে হলে Duck arm override করা যাবে না।
3. High-concurrency latch offered load
HIGH

16/20-stream test-এ যথেষ্ট request পাঠিয়ে actual level যাচাই করতে হবে।
4. Final forced launch + capacity verification
HIGH

Selected engine আর reference fallback—দুটোই fail করলে dead server-এ Duck চালানো বন্ধ করতে হবে।
এই চারটি targeted পরিবর্তন ছাড়া নতুন feature প্রয়োজন নেই।
৮. Optional items — GO আটকাবে না
- Larger multi-prompt correctness sample এবং extra concurrency probe rows।
- FP32 SSM on DFlash W6।
- CUDA graph 1..24 memory/capture-time comparison।
- 8k/40k prompt duplication bias analysis।
- E17-তে cache-restore parity এবং #43567 test।
- Cleaner results table with requested-vs-observed concurrency।
এগুলো E15-এর exploratory value বাড়াবে, কিন্তু বর্তমান mandatory fixes-এর বিকল্প নয়।
চূড়ান্ত সিদ্ধান্ত
Scope	Verdict
E15 notebook packaging, exact SHA	GO
EAGLE/MTP experimental flags	GO
ReplaySSM experimental arms	GO for testing
Time-budget architecture	Conditional GO
14/16/20-stream current latch metrics	NO-GO as valid evidence
Automatic Duck arm selection	NO-GO
Exact notebook SHA 2007cba6… GPU push	NO-GO pending four focused fixes

শেষ কথা: E15-এর experiment design বাতিল করছি না। MTP ও high-concurrency পরীক্ষা করাটা প্রয়োজনীয়। তবে বর্তমান notebook ভুলভাবে probe failure-কে acceptable ধরে, long-context speed gate bypass করে, এবং 20-stream latch পরীক্ষা করার জন্য মাত্র 14টি request পাঠায়। এগুলো বাস্তব, পরীক্ষাযোগ্য ভুল।

চারটি ছোট fix ও targeted regression tests PASS করলে GO দেওয়া উচিত, আর অপ্রয়োজনীয় নতুন enhancement দিয়ে GPU run আটকে রাখা উচিত নয়।

কোনো source পরিবর্তন, GPU run, notebook push বা commit করিনি।
