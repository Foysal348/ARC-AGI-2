<!-- Pasted by the user on 2026-10-24 (GPT audit of E15_MTP_ENGINE_AUDIT_FOR_GPT_2026-10-24.zip). Kept VERBATIM. Claude's validation: Notes/41. -->

E15 MTP Engine — Final Pre-GPU Independent Audit
Target: E15_MTP_ENGINE_AUDIT_FOR_GPT_2026-10-24(2).zip

Audit scope: README-এর ৬টি প্রশ্ন, SGLang 0.5.21 flags, MTP compatibility, ১৬টি engine arms, 180-minute budget, correctness gates, Duck integration এবং historical evidence।
Notebook integrity
PASS

SHA-256: 2007cba65cf058091d4c13eb216853957cdbfa58e1b05ea36cc53a98dfeb4177

Bundled Kaggle CPU validation
167/167

Recorded pytest 167/167; E15 harness 22/22; rerun harness 14/14।
My independent measurement tests
34/34

test_e14_measure.py independently PASS। পূর্ণ 167-test rerun timeout হয়েছে; independent full PASS দাবি করছি না।
Exact notebook, as-is
NO-GO

তিনটি ছোট কিন্তু গুরুত্বপূর্ণ correctness/measurement fix প্রয়োজন। E15 experiment বাতিল বা redesign করার প্রয়োজন নেই।

১. সবচেয়ে গুরুত্বপূর্ণ নতুন finding
একটি failed long probe-ও arm adoption আটকাতে পারছে না।

notebook/E15_cell_5c.py:107–111:
def _gated(a):    _aa = (E15R.get("probe_ref_aa") or {}).get("large_margin_flips") or 0    _pf = (a or {}).get("probe_flips")    return bool(        a and a.get("ok")        and a.get("gate") in ("PASS", "reference")        and not a.get("log_errors")        and (_pf is None or _pf <= _aa + 1)    )



এখানে _pf is None হলে arm PASS করতে পারে। অথচ সেটির অর্থ long probe পরীক্ষাই হয়নি।

আরও খারাপ: e14_measure.compare_passes()-এ ১২টি request-ই timeout হলে:
expected = 12
compared = 0
errors = 12
large_margin_flips = 0


আমি bundled source দিয়ে এটি independently reproduce করেছি। _gated()-এ probe_flips=0 দিলেও arm eligible হয়ে যায়, কারণ failed comparisons zero flips হিসেবে গণনা হচ্ছে।

এটি real GPU run-এ ঘটেছে বলে দাবি করছি না। কিন্তু exact source-এ এমন failure path আছে, যা পরীক্ষার primary selection decision ভুল করতে পারে।

Bundled fake-server run-ও দুর্বলতাটি দেখায়: Duck-এর জন্য w8 নির্বাচন হয়েছে, যদিও w8-এর দীর্ঘ 1,024-token probe নেই।

এই কারণে শুধু 22/22 harness PASS দেখে GPU launch-এর GO দেওয়া ঠিক হবে না।
২. প্রশ্ন ১ — 180-minute run কি শেষ হবে?
Verdict: CONDITIONAL GO for scheduling; NO-GO for unchecked final launch।

Source: notebook/notebook_cells.py:1537–1647, E15_cell_5c.py:113–201।
Budget যাচাই
Total notebook budget
180 min

Internal hard stop
178 min

E15 downstream reserve
39m 40s

Duck solve window
15 min



Reserve-এর breakdown:
Component	Reserved
Duck solve window	900 s
Stop-before-hard	180 s
Preflight	300 s
Artifacts/report	600 s
Final engine relaunch	400 s
Total	2,380 s

e14_phase() এবং launch_allowed() এই reserve বিবেচনা করে নতুন experimental arm শুরু করে। ফলে সময় কমলে arm skip হওয়ার ব্যবস্থা বাস্তবে আছে।

তবে README-এর “সব ~16 launches, তারপর Duck, একই সঙ্গে reserve guaranteed” কথাটি একটু বেশি শক্তিশালী। Cold relaunch 400 s-এর বেশি হলে বা startup খুব ধীর হলে সব arms শেষ হবে না। Skip করা এখানে সঠিক fail-safe।
HIGH-1 — Final Duck launch result যাচাই হচ্ছে না
E15_cell_5c.py:187–199:
E14_RESERVE = 0.0_da = E15R["arms"].get(_duck) or {}e14_launch(    "duck_" + _duck,    set_=_da.get("set") or {},    drop=tuple(_da.get("drop") or ()),    force=True)E14_STATE.enter("DUCK", arm=_duck)



এখানে e14_launch()-এর (ok, record) return value উপেক্ষা করা হয়েছে।

e14_launch() নিজে launch failure হলে E14_KEPT restore করার চেষ্টা করে। কিন্তু restore-ও ব্যর্থ হতে পারে। তারপরেও notebook CELL 6 চালানোর চেষ্টা করবে।

বিপদ: Dead server থাকলেও solver শুরু হতে পারে। আবার fallback server সফল হলেও report-এ নির্বাচিত MTP arm-এর নাম থেকে যেতে পারে, অথচ Duck অন্য engine-এ চলছে।

Minimal fix:
ok, launch_rec = e14_launch(    "duck_" + _duck,    set_=_da.get("set") or {},    drop=tuple(_da.get("drop") or ()),    force=True,)restored = bool(    (launch_rec.get("relaunched_kept") or {}).get("ok"))E15R["duck_launch"] = {    "selected_arm": _duck,    "selected_launch_ok": ok,    "restored": restored,    "effective_arm": E14_ARM["name"] if (ok or restored) else None,}if not (ok or restored):    E15R["duck_launch"]["error"] = "No live SGLang server"    e14_save()    # Enter existing failure/fallback reporting path;    # do not declare Duck successful or launch solver on a dead endpoint.



এখানে নতুন exception ছুড়ে পুরো notebook বন্ধ না করে existing fallback/report path ব্যবহার করতে হবে।

সফল final launch-এর পর একটি bounded smoke request এবং effective Mamba/KV capacity check প্রয়োজন। কারণ আগের arm-এর gate PASS করলেও relaunch-এর resolved KV allocation বা serving state একই থাকবে—এমন নিশ্চয়তা নেই।

Validation test: তিনটি injected scenarios—selected arm launches successfully, selected arm fails but kept arm restores, এবং উভয় launches fail। শেষ scenario-তে dead server-এ Duck execution নয়; report ও fallback artifact অক্ষত থাকবে।

৩. প্রশ্ন ২ — SGLang flags, MTP, ReplaySSM ও memory
Verdict: Core flags GO. নতুন নিশ্চিত flag-parser blocker পাইনি।
সবচেয়ে মূল্যবান প্রমাণ: MTP head আগেই load হয়েছিল
evidence/35 Claude agent - E15 MTP preflight (2026-10-24).md-এ পূর্ববর্তী E11-03 Rev 3b GPU run-এর নির্দিষ্ট লগ উদ্ধৃত আছে।

সেখানে একই RedHat INT4 checkpoint, SGLang 0.5.21, 4×L4 এবং built-in MTP ব্যবহার হয়েছিল।

Reported:
Metric	E11-03 MTP
MTP model type	Qwen3_5ForCausalLMMTP
Draft head loading	Completed
Mean accept length	3.05
Mean accept rate	~0.68
Concurrent running cap	10
Decode throughput at 10 running	~374 tok/s

এটি bundled historical-log analysis, নতুন GPU reproduction নয়। ওই raw E11-03 server log এই ZIP-এ নেই।

তবুও এটি আগের অনিশ্চয়তা অনেক কমায়: MTP head আছে কি না—সেটি আর সম্পূর্ণ অজানা নয়।

E11-03-তে MTP ধীর হওয়ার অন্যতম কারণ ছিল FP32 SSM এবং 10-request cap, যেখানে DFlash configuration 12–13 streams ব্যবহার করেছে। তাই আগের performance comparison unfair।
Flag-by-flag audit
Arm	Static verdict	Evidence / risk
target_only	PASS	spec=False সব --speculative-* সরায়
w8	PASS	Draft width 8; existing W6-এর অন্য flags retained
w6_mrr14	PASS	max-running=14, graph batches 1–14; 12th admission সম্ভব
w6_nodelay_mrr12	PASS	Delay=1, max-running=12; আগের 13-running crash path বন্ধ
w6_at14	PASS — runtime unverified	MRR17, Mamba68, graphs1–17
w6_at16	PASS — runtime unverified	MRR20, Mamba80, graphs1–20
w6_at20	PASS — higher risk	MRR24, Mamba96, graphs1–24; capacity/graph memory unknown
mtp_3_1_4	PASS	EAGLE, explicit INT4 draft path, steps3/topk1/draft4
mtp_3_1_4_mem86	PASS — memory risk	Static memory 0.86; post-capture headroom জানা নেই
mtp_2_1_3	PASS	Steps2/draft3
mtp_replay_fp32	PASS — untested combination	ReplaySSM enabled, explicit SSM FP32
mtp_replay_fp16	PASS — numerical risk	ReplaySSM enabled, SSM FP16
Best MTP at 14/16	PASS mechanically	Selected configuration-এর source flags retained

e14_edit_flags()-এর list-valued graph arguments এবং None-valued boolean flags handling সঠিক। Bundled fake-server harness-ও generated argv যাচাই করেছে।

SGLang-এর public speculative-decoding documentation EAGLE/NEXTN alias, draft path, steps/topk/draft width ব্যবহারের সমর্থন দেয়। তবে latest documentation-এর সব behavior v0.5.21-এ আছে, এমন ধরে নেওয়া হয়নি। 

GitHub



Memory/OOM সম্পর্কে সিদ্ধান্ত
আপনার E15-এর higher-concurrency arms plausible। কিন্তু KV usage 18–41% হলেই MTP/ReplaySSM + graphs1–24 নিরাপদ প্রমাণিত হয় না।

বিশেষ করে:
- Mamba pool slots বাড়ালে usable KV pool ছোট হতে পারে।
- MTP draft head ও graphs-এর জন্য আলাদা memory প্রয়োজন।
- ReplaySSM recurrent-state allocation বদলায়।
- mem-fraction-static=0.86 late Triton/JIT loading-এর headroom কমাতে পারে।
সঠিক fail-safe হচ্ছে arm launch failure → record → next arm, memory দেখে blind retry নয়।

Pre-run fix প্রয়োজন নেই। তবে প্রতিটি arm-এ caps, peak memory, late OOM এবং actual running level সংরক্ষণ বাধ্যতামূলক।

৪. প্রশ্ন ৩ — Measurements ও adoption decision কি বিশ্বাসযোগ্য?
Verdict: PARTIAL. দুটি confirmed measurement defects।
HIGH-2 — 14/16/20-stream latch এখনও 12-stream test
E15_cell_5c.py:44–60:
out["latch"] = E14M.latch_test(    E14_CTX, _S, _L,    decodes=SGL_MAX_RUNNING - 1,    arrivals=3,    secs=_ts(60),    server_cap=latch_cap,    level=SGL_MAX_RUNNING,)



SGL_MAX_RUNNING = 12।

তাই w6_at20 arm-এও latch test ১১টি initial decode + ৩টি arrivals দিচ্ছে। সর্বোচ্চ ১৪টি offered requests দিয়ে 20-running latch পরীক্ষা করা অসম্ভব।

Fake harness-এও w6_at20-এর field latch_ge12 হয়েছে 0.917। এটি 20-running saturation-এর measurement নয়।

Fix:
out["latch"] = E14M.latch_test(    E14_CTX, _S, _L,    decodes=level - 1,    arrivals=3,    secs=_ts(60),    server_cap=latch_cap,    level=level,)



Report key আলাদা রাখুন:
"latch_level": level,"latch_ge_target": latch.get("share_running_ge_level"),"latch_stuck_below": latch.get("share_stuck_below_queued"),



w6_mrr14 ও w6_nodelay_mrr12-এর level=12 আগের মতোই থাকবে।

Validation test:
Arm	Initial decodes	Arrivals	Expected latch level
W6 baseline	11	3	12
W6 at14	13	3	14
W6 at16	15	3	16
W6 at20	19	3	20

বর্তমান code-এ test fail করবে; patch-এর পরে pass করতে হবে।
HIGH-3 — 14/16/20 probe-এর অতিরিক্ত rows তুলনা হচ্ছে না
E15_cell_5c.py:66–74:
def _probe_vs_ref(rows, name):    if not E15_PROBE_REF:        return {"skipped": "no reference probe"}    return E14M.compare_passes(        E15_PROBE_REF[0],        rows[:len(E15_PROBE_REF[0])],        name    )



Reference-এ 12 rows। ফলে:
Candidate concurrency	Generated probe rows	Compared
12	12	12
14	14	12
16	16	12
20	20	12

এতে 20-stream configuration-এর শেষ আটটি outputs verification থেকে বাদ পড়ে।

আর compare_passes() failed requests-কে error হিসেবে ধরে, কিন্তু _arm() কেবল probe_flips সংরক্ষণ করছে। compared, errors, missing_margins বাদ যাচ্ছে।

Minimal fix: All-N probe records অক্ষত সংরক্ষণ এবং explicit coverage reporting:
"probe_expected": probe.get("expected"),"probe_compared": probe.get("compared"),"probe_errors": probe.get("errors"),"probe_missing_margins": probe.get("missing_margins"),



N>12-এর ক্ষেত্রে matching task/prompt indices অনুযায়ী reference comparison করতে হবে। Reference c12-এর rows cycle করা একটি diagnostic comparison হতে পারে, কিন্তু N-concurrency matched reference নয়। রিপোর্টে এ পার্থক্য লিখতে হবে।

Validation: 20 rows-এর মধ্যে 19টি success ও 1টি error → compared=19, expected=20, probe_valid=False।
Shared-prefix benchmarking-এর সীমা
e14_measure.py:321–345-এ steady_window() একই 12 prompts cycle করে। Long prompts-এর প্রায় 36k common prefix আছে এবং warm request দিয়ে cache prefill করা হয়।

এটি shared-prefix-friendly throughput মাপে, ২০টি independent 41k-context ARC tasks-এর memory চাপ নয়।

এই benchmark অকার্যকর নয়। কিন্তু 20-stream high throughput থেকে hidden workload-এর একই throughput অনুমান করা যাবে না।

সঠিক report label: warm_shared_prefix_throughput। পাশাপাশি KV allocated, KV occupied, actual contexts এবং measured running level দেখান।

৫. প্রশ্ন ৪ — Duck কি ভুল engine গ্রহণ করতে পারে?
Verdict: YES. এটিই সবচেয়ে গুরুত্বপূর্ণ pre-GPU blocker।
Problem A: Incomplete probe → PASS
Section 1-এর reproduction।
Problem B: _fix নির্বাচনে long-context qualification নেই
E15_cell_5c.py:180–185-এ latch fix নির্বাচনের শর্ত:
- Short gate passes;
- Latch gain;
- Short tok/s ≥98% reference।
কিন্তু long tok/s ≥98% reference শর্ত নেই।

তাই long-context regression থাকা arm-ও latch_fix হয়ে _duck override করতে পারে।
Problem C: Window নিজেই unhealthy হলেও ok=True হতে পারে
_arm()-এর:
"ok": bool(    _ok and _r and not _r.get("error"))



যদি outer phase সফল হয় কিন্তু short/long window-এর ভেতরে request errors, incomplete cleanup বা insufficient running থাকে, তাহলে arm ok=True থাকতে পারে।
Recommended minimal correction
আমি আলাদা architecture চাইছি না। শুধু measurement validity এবং selection eligibility পৃথক করুন।
def _healthy_window(w, required_level):    return bool(        w        and not w.get("error")        and not w.get("errors")        and (w.get("cleanup") or {}).get("server_idle") is True        and w.get("measured_level") == required_level        and (w.get("tok_s_p50") or 0) > 0    )def _valid_probe(p, expected):    return bool(        p        and p.get("expected") == expected        and p.get("compared") == expected        and p.get("errors") == 0        and p.get("missing_margins") == 0    )



এই functions-এর result _arm()-এ record করতে হবে; short, long এবং probe-এর raw verdict সংরক্ষণ করতে হবে।

Duck adoption-এর জন্য আলাদা final gate রাখুন: shortlist তৈরি হওয়ার পর সবচেয়ে দ্রুত candidate-এর final 1,024-token probe নিন। সেটি fail করলে next eligible candidate পরীক্ষা করুন, সর্বোচ্চ ২টি। উভয় ব্যর্থ হলে tested reference।

এতে সব 12-stream variant-এর জন্য অপ্রয়োজনীয় full probe চালানোর খরচ এড়ানো যায়। তবে final selected arm verified হবে।

সবশেষে fresh launch-এর actual engine identity, capacity ও smoke নিশ্চিত করুন (HIGH-1)।
Exact negative tests
Scenario	Expected
Probe absent	Cannot be finally adopted
Probe 0/12, flips=0	Reject
Probe 11/12, flips=0	Reject
Probe 12/12, missing margins 1	Reject / inconclusive
Short gate PASS, long throughput −15%	Do not adopt
Short/long window contains API errors	Do not adopt
Latch fix improves occupancy but long performance fails	Do not override winner
Final relaunch fails, fallback works	Record actual fallback arm
Both launches fail	No dead-server Duck execution

এই tests বর্তমান 22/22 harness-এ নেই।

৬. প্রশ্ন ৫ — আর কী এই E15 run-এ যোগ করা উচিত?
তিনটি mandatory correction-এর বাইরে আমি খুব বেশি change যোগ করব না।
Priority	Change	Reason	Decision
1	Final chosen-arm 1,024-token probe + smoke	Duck correctness and effective launch	Must
2	Correct N-stream latch/probe accounting	E15-এর primary coverage question	Must
3	--chunked-prefill-size 2048 arm	Agent-effective throughput, prefill interference	Optional
4	W6 + SSM FP32 at 12	Native dtype accuracy sensitivity	Optional; E17 more suitable
5	ReplaySSM FP16/FP32 1,024-token probe	State-replay correctness	Useful if budget permits
6	DFlash width4 at 41k context	Previous short-context speed result	Optional

যা এখন যোগ করব না: Pipeline parallel, 2×TP2 redesign, new quantization, experimental lossy acceptance thresholds, mixed-chunk under unresolved hybrid issues এবং speculative kernel rewrite।

SGLang-এর PP/speculative-decoding issue tracking এখনও model-specific limitations দেখাচ্ছে। 

GitHub



MTP expected performance
আপনার historical E11-03 MTP mean acceptance 3.05, DFlash mean around 4.65; কিন্তু সেই runs concurrency- ও memory-matched ছিল না।

তাই E15-এর সবচেয়ে মূল্যবান নতুন measurement হলো same TP4, same SSM dtype, same concurrency, same 8k/41k contexts-এ MTP বনাম DFlash2।

600–800 tok/s gain অনুমান হিসেবে রাখা যেতে পারে, কিন্তু এই ZIP-এ তার benchmark evidence নেই। MTP adoption কেবল speed-এর ওপর নয়, long-context correctness ও actual cache headroom-এর ওপর নির্ভর করবে।

৭. প্রশ্ন ৬ — Variable scope, report keys ও storage bug
Check	Verdict	Evidence
E15_REF global assignment	PASS	Notebook global namespace-এ globals()["E15_REF"]; harness runs
E15_PROBE_REF assignment	PASS	Same mechanism
e14_edit_flags() lists / None	PASS	Source + fake launch argv
tok_s_p50 key	PASS	e14_measure.window_stats()-এর key
share_running_ge_level key	PASS, wrong level in at-N arms	Correct field, incorrect level=12
Per-arm E15R saving	PASS normal path	_arm() শেষে e14_save()
probe_flips	Incomplete	Zero flips does not prove valid comparisons
Reference fallback	Mechanism exists	Final effective-arm reporting missing
Error-sign filtering	Partial	Per-request errors and window validity not enforced
Full 180-minute completion	Not guaranteed	Deadline skip logic is reasonable; all arms may not run

আরেকটি positive finding: Main notebook-এর সব Python code cells compile হয়েছে, embedded E14 module payload ZIP valid, এবং target file SHA exact match। Source workspace-এ #43567 patch থাকা সত্ত্বেও embedded E14 module bundle পুরোনো—এটি README-তে স্পষ্টভাবে documented, accidental replacement নয়।

৮. Consolidated GO/NO-GO এবং next actions
GO
MTP experiment design ও static serving flags

RedHat INT4 MTP head-এর historical load evidence আছে; EAGLE/ReplaySSM flag construction সঙ্গতিপূর্ণ। নতুন kernel rewrite প্রয়োজন নেই।



NO-GO
Exact SHA 2007cba6…-এর automatic Duck-arm adoption

Failed/absent probes pass করতে পারে; final relaunch outcome unchecked। এই দুটি সরাসরি decision-integrity risk।



GO after fixes
One 180-minute engineering GPU commit

নিচের ছোট patch set এবং negative tests পাস হলে নতুন exploratory redesign ছাড়াই run যুক্তিসংগত।



Fix before GPU run — ranked
1. P0: Incomplete/failed/absent probe-কে final adoption থেকে বাদ দিন। Final shortlist winner-এর 1,024-token probe এবং explicit compared/expected/missing_margins validation যোগ করুন।
2. P0: Forced Duck launch-এর ok পরীক্ষা, restored engine identity, bounded smoke এবং capacity check যুক্ত করুন।
3. HIGH: 14/16/20 latch-এর actual target level ব্যবহার করুন; probe-এর N rows না পরীক্ষা করে N-concurrency correctness দাবি করবেন না।
4. HIGH: Window health, full comparison ও long-performance qualification দিয়ে _cand এবং _fix selection ঠিক করুন।
5. Validation: Fake-server negative tests, exact notebook SHA, 167 unit tests, E15 harness ও rerun harness পুনরায় চালান।
Optional — not blockers
Chunked-prefill 2048, W6+FP32 SSM, ReplaySSM long probe, DFlash width4, JIT tar error handling, full 240-task failure-path simulation। এগুলোর কোনোটি এই engine experiment-এর mandatory scope বাড়ানোর কারণ নয়।
Final answer to the README
এই ZIP-এ catastrophic GPU kernel bug নিশ্চিত পাইনি। MTP flags, memory variants, launch framework ও core measurement tools মোটের ওপর যুক্তিসংগত।

কিন্তু পূর্ববর্তী audit-এ যেমন test-oracle ভুলের কারণে GPU quota নষ্ট হয়েছিল, এবারও success criteria-র মধ্যে নির্দিষ্ট ভুল আছে। সব request fail করলেও zero large-margin flips দেখা যেতে পারে—এটিকে success হিসেবে গ্রহণ করা যাবে না।

সবচেয়ে ইতিবাচক বিষয়: এই ভুলগুলো GPU ছাড়া reproduce হয়েছে, এবং প্রত্যেকটির fix ছোট ও localized। একবার consolidated correction করে tests পাস করলেই GO দেওয়া যাবে; E15-এর মূল experiment বদলানোর দরকার নেই।

কোনো code, notebook, commit বা push পরিবর্তন করিনি।

Public verification URLs:

https://github.com/sgl-project/sglang/releases/tag/v0.5.21

https://github.com/sgl-project/sglang/issues/39634

https://github.com/sgl-project/sglang/blob/v0.5.21/docs/advanced_features/speculative_decoding.md
