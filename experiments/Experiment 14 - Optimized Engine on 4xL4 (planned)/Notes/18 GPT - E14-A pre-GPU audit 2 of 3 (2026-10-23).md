# GPT audit 2 of 3 of E14-A notebook a9dad9fc (user-pasted 2026-10-23, verbatim)

Validation: Notes/20.

---

E14-A — Independent Pre-GPU Audit
Notebook audited: Q27-XHIGH-E14A_OPTIMIZED_ENGINE_SGLANG.ipynb
SHA256 verification
MATCH

a9dad9fcbe0003867d34f205732a23106dbfed30380535ef4531ba78806de494

ZIP integrity: 40/40 entries readable। Notebook-এর 12টি code cells compile হয়। README_FOR_GPT.md সম্পূর্ণ পড়ে actual notebook, embedded modules ও tests-এর সঙ্গে যাচাই শুরু করেছি।

প্রাথমিক গুরুত্বপূর্ণ সিদ্ধান্ত
এই exact SHA-কে এখনই GPU GO দেওয়া উচিত নয়। অন্তত একটি গুরুত্বপূর্ণ run-budget defect পাওয়া গেছে: ২৫ মিনিটের Duck solver run-এর সময় আগে থেকে reserve করা নেই। Engine measurements ও relaunches সময় নিতে থাকলে notebook solver-এ পৌঁছানোর আগেই 200-minute budget প্রায় শেষ করতে পারে।

এর পাশাপাশি multi-request cleanup-এর deadline risk এবং correctness gate-এর অসম্পূর্ণ-sample acceptance পাওয়া গেছে। এগুলো বড় architecture পরিবর্তন ছাড়াই ঠিক করা সম্ভব।

এই মুহূর্তে কোনো code বা notebook পরিবর্তন করিনি।
1. P0 — Model staging-এ নিশ্চিত path collision
এটি সবচেয়ে গুরুতর নতুন finding। Load-path reliability ঠিক করার জন্য বানানো code-ই default Kaggle paths-এ target model overwrite করতে পারে।

Files: src/e14_loadpath.py:545–548, 627–634
Notebook: CELL 1-এর SGL_MODEL_PATH, SGL_DRAFT_PATH; CELL 1b-এর staging।

দুটি default path:
.../qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1
.../qwen3-8-27b-dflash2/pytorch/bf16/1


উভয়ের basename 1।

কিন্তু parallel_copy() destination তৈরি করে:
final = Path(dst_root) / src.namework = Path(dst_root) / (src.name + ".partial")



এরপর stage_models() target ও draft একই destination root-এ পর্যায়ক্রমে copy করে।

ফল: প্রথম model সফলভাবে .../e14-weights/1-এ copy হলে দ্বিতীয় model-এর copy শুরুর সময় সেই directory মুছে যায়। শেষে target ও draft-এর staged paths একই directory নির্দেশ করতে পারে।

এটি synthetic issue নয়—README-তে দেওয়া default paths-এই collision রয়েছে।

Required fix: destination name-এ unique role ও source-identity রাখতে হবে। যেমন target-<source-hash> এবং draft-<source-hash>। একই সঙ্গে stage_models()-এ assert করতে হবে যে distinct source directories কখনো একই destination resolve করছে না।

Validation: একই basename 1-এর দুটি আলাদা fake model copy করে নিশ্চিত করবে যে দুই final directories আলাদা, file contents ও checksum সঠিক, এবং দ্বিতীয় copy প্রথমটিকে পরিবর্তন করেনি।

এই fix ছাড়া exact notebook SHA-এর জন্য NO-GO।
2. আরও তিনটি must-fix
P0 — A1 phases Duck-এর 25-minute window খেয়ে ফেলতে পারে
Files: notebook CELL 5b e14_phase() lines 81–90; CELL 6 lines 349–359; E14A config CELL 1 lines 69–76।

e14_phase() শুধু HARD_STOP − 8 minutes দেখে phases চালায়। কিন্তু Duck solver-এর জন্য 25 minutes + preflight + finalization reserve রাখে না।

200-minute budget-এ HARD_STOP প্রায় 198 মিনিটে। A1 চলতে পারে প্রায় 190 মিনিট পর্যন্ত, অথচ solver-এর branch deadline তারও আগে (hard stop থেকে 180 seconds দূরে)। ফলে A1 সময় বেশি নিলে 25-minute solver কার্যত 5 মিনিট বা তারও কম পেতে পারে।

Fix: A1-এর আলাদা absolute cutoff:
\[
\text{A1 cutoff} \leq
\text{HARD\_STOP}
-\text{solver stop margin}
-1500\text{ s}
-\text{preflight allowance}
\]

A1 phase শুরুর আগে সময় না থাকলে সেটি skip করবে; Duck বাদ দিয়ে optional measurement শেষ করবে না।

Validation: Simulated slow startup ও relaunch-এর পর নিশ্চিত করতে হবে Duck সত্যিকার 1,500-second solve allowance পায়, hard stop permitting। সময়ই না থাকলে explicit solver_window_unavailable status দিতে হবে; 25-minute run হিসেবে report করা যাবে না।
P0 — NCCL benchmark ভুল CUDA-graph result adopt করতে পারে
File: src/e14_nccl_bench.py:437–469

build_summary()-এর candidate filter দেখে:
v["graph_per_call_us"] is not None and v["correct"] is True



কিন্তু graph_correct সত্য কিনা যাচাই করে না।

আমি function-এর বাস্তব code চালিয়ে reproduce করেছি:
- baseline: 300 µs, eager correct, graph correct;
- P2P arm: 100 µs, eager correct, graph incorrect;
- function-এর ফল: P2P arm adopt_candidate!
এটি বিপজ্জনক কারণ আমাদের প্রধান workload CUDA graphs ব্যবহার করে।

Fix:
if (    v["status"] == "ok"    and v["correct"] is True    and v["graph_correct"] is True    and v["graph_per_call_us"] is not None)



Test: Eager correct কিন্তু captured graph wrong হলে 90% দ্রুত হলেও arm rejected হবে। Missing-rank, timeout এবং missing graph-correctness-ও reject করবে।
P0 — Faster nodelay/graphs arm-এর আগে actual parity gate নেই
Notebook CELL 5b lines 230–285

#44401, nodelay এবং graphs13-এর adoption প্রধানত speed/latch দিয়ে হয়। নতুন engine configuration-এর output baseline-এর মতো আছে কি না, সেটি adoption condition-এ নেই।

nodelay-এর মাধ্যমে বেশি concurrent requests চালানোই লক্ষ্য; তাই concurrency-sensitive correctness regression বিশেষভাবে প্রাসঙ্গিক।

Fix: প্রতিটি adopted arm-এর জন্য same-workload bounded greedy/parity gate, অথবা known-answer correctness gate রাখতে হবে। Gate চালানোর সময় না থাকলে risky arm adopt নয়—baseline-এ ফিরে যাবে।

Test: Fake server-এ faster throughput কিন্তু incorrect output injected করলে arm অবশ্যই rejected হবে।

এটি README-এর “speed rule and correctness gate” দাবি বাস্তবে পূরণ করার জন্য দরকার।
3. বাকি findings: deadline, parity ও measurement integrity
ID	Severity	File : line	Finding	Minimal fix ও validation
E14-4	HIGH — must-fix	src/e14_measure.py:58–64	Deadline পেরিয়ে গেলেও post_json() HTTP request পাঠায়; timeout minimum 5 seconds	Remaining time শেষ হলে request না পাঠিয়ে TimeoutError; expired-deadline test
E14-5	HIGH — must-fix	src/e14_measure.py:251–296,403–420,455–479	abort() এবং প্রতি thread-এর join(60) দিয়ে cumulative timeout phase deadline অতিক্রম করতে পারে	সব cleanup-এর জন্য shared absolute cutoff; 12 stuck requests test
E14-6	HIGH — must-fix	src/e14_measure.py:560–566	Partial parity samples দিয়েও parity_ok() PASS দিতে পারে	Expected comparison count enforce; missing/error samples reject
E14-7	HIGH — must-fix	Notebook CELL 6b 11–34	Int8 Mamba correctness default বনাম int8 তুলনা করে না; শুধু cold বনাম cached response দেখে	Same-input default/int8 cross-arm parity ও A/A control
E14-8	HIGH — must-fix for A2	Notebook CELL 6b 38–68	Width 6/4 এবং target-only-এর control আসে পুরোনো W1 run থেকে, কিন্তু arm-এ patched/adopted engine settings থাকতে পারে	Final adopted width-8 arm-এর matched control
E14-9	MEDIUM	Notebook CELL 5b 230–285	Nodelay কোনো gain না দিলেও adopted হতে পারে; graphs13 capture list অনুপস্থিত হলেও speed-only adoption সম্ভব	Positive admission improvement এবং confirmed graph capture
E14-10	MEDIUM	Notebook CELL 5b 287–290	/flush_cache failure log হয়, কিন্তু subsequent Duck benchmark-কে guaranteed cold-start বলা যায় না	Idle check, flush result ও cache state আলাদা record
E14-11	LOW	src/e14_measure.py:371–376	Last-token divergence-ও full_reply_text_match=True হতে পারে	পুরো expected token sequence prefix হিসেবে match করতে হবে
E14-12	LOW	src/test_e14_sglang_patches.py:25	Tests অন্য machine-এ unavailable Windows absolute source path ব্যবহার করে	Bundled source fixture বা configurable reference path

E14-4: Deadline guard-এর independent reproduction
বর্তমান code:
timeout = max(5.0, min(float(timeout), ctx.left() + 5.0))



আমি ctx.left() = -100 দিয়ে function চালিয়েছি।

Actual result: deadline 100 seconds আগে শেষ হলেও HTTP call পাঠানো হয়েছে, timeout 5 seconds।

Proposed logic:
remaining = ctx.left()if remaining <= 0:    raise TimeoutError("E14 phase deadline reached")timeout = min(float(timeout), remaining)



এটি minimum 5 seconds বজায় রাখার চেয়ে নিরাপদ। Request শুরু করার মতো সময় না থাকলে request skip করতে হবে।

আর শুধু এই function ঠিক করলেই হবে না। steady_window()-এ 12টি stuck requests-এর প্রতিটির জন্য join(60) করলে সর্বোচ্চ 720 seconds cumulative wait তৈরি হতে পারে।

একটি shared cleanup cutoff ব্যবহার করতে হবে। Deadline-এর পরে alive requests পরবর্তী phase-এ carry over করছে কি না, তাও পরীক্ষা করতে হবে।
E14-6: Parity gate-এর independent reproduction
বর্তমান parity_ok()-এর বাস্তব code দিয়ে দিয়েছি:
arm = {    "compared": 1,    "identical": 1,    "large_margin_flips": 0,}aa = {    "compared": 12,    "identical": 3,    "large_margin_flips": 0,}



allowance=2 দিলে ok=True আসে।

অর্থাৎ ১২টির মধ্যে মাত্র একটি prompt সফলভাবে compare হওয়া সত্ত্বেও arm correctness PASS পেতে পারে।

Required: একই prompt indices, একই sample count, সব required responses valid, zero unresolved errors—তারপর identical/large-margin rule চালাবে।

আর margin-এর অর্থও স্পষ্ট রাখা দরকার। Large-margin divergence সন্দেহজনক হতে পারে, কিন্তু দুইটি A/A run-এ observed numerical variation থাকলে সেটিকে automatic corruption বলতে হবে না।

4. README-এর ছয়টি প্রশ্নের উত্তর
Q1. কোন defect phase invalid বা budget overrun করতে পারে?
চারটি গুরুত্বপূর্ণ defect পেয়েছি:
- Model-copy basename collision: independently reproduced; successful staging-এর পরে target model draft model দিয়ে replace হতে পারে।
- Duck-এর 25-minute window reserved নয়: A1 engine phases প্রায় পুরো available budget খরচ করতে পারে।
- Deadline-expired HTTP calls ও cumulative joins: phase cap hard নয়।
- Incorrect/partial correctness gate: কিছু দ্রুত arm পর্যাপ্ত output validation ছাড়াই adopted হতে পারে।
এগুলোর মধ্যে প্রথমটি সবচেয়ে স্পষ্ট startup blocker।
Q2. Nodelay ও #44401-এর latch-test adoption rule ঠিক?
বর্তমান rule যথেষ্ট নয়।

Notebook CELL 5b 230–265-এ:
share_at_target_arm >= share_at_target_baselineand share_stuck_arm <= share_stuck_baseline



এর সঙ্গে speed rule আছে, যার zero-minimum-gain setting-এ প্রায় 2% slowdown পর্যন্ত গ্রহণযোগ্য।

আমি independently verify করেছি, যদি baseline ও nodelay দুটোরই latch metrics শূন্য এবং TPS সমান হয়, nodelay adopted হয়।

আর share of samples with running ≥12 and queue ≥1 একা যথেষ্ট নয়। সফল admission-এর পরে queue খালি হয়ে গেলে সেই metric-ও শূন্য হয়ে যেতে পারে।

Better metrics: offered load একই রেখে running≥12-তে কাটানো সময়, maximum simultaneous running, queue-drain latency, request retractions এবং actual committed tok/s। #44401-এর held-row correctness test আলাদা রাখবে। Nodelay-এর জন্য real admission improvement ছাড়া adoption নয়।
Q3. /flush_cache safe?
Endpoint আছে, কিন্তু hybrid radix + Mamba-এর জন্য এটিকে unconditional correctness proof বলা যাবে না।

বর্তমান code /flush_cache exception ধরলেও Duck run চালিয়ে যায়। এটি solver-এর জন্য reasonable fallback হতে পারে, কিন্তু তখন cold-cache baseline নিশ্চিত নয়।

আমি চাই:
1. Request queue/running count শূন্য কিনা যাচাই।
2. Flush acknowledgement ও errors আলাদা record।
3. Flush-এর পরে একটি bounded cache-state check।
4. Failure হলে Duck run বাতিল না করে cold_start_unverified flag।
প্রয়োজনে server restart clean-state guarantee দিতে পারে, কিন্তু এর startup cost বড়; তাই শুধু diagnostic-এর জন্য automatic costly restart করব না।
Q4. Greedy-parity allowance sound?
না—বর্তমান implementation-এ correctness gate হিসেবে দুর্বল।

কারণ partial samples PASS করতে পারে এবং nodelay/graphs-এর জন্য parity adoption condition-এ নেই।

আর ১২টি prompt-এর মধ্যে সর্বোচ্চ ২টি divergence allowance কোনো statistical equivalence guarantee নয়। এটি exploratory screening rule হিসেবে ব্যবহারযোগ্য, কিন্তু exact serving correctness-এর শক্ত প্রমাণ নয়।

Recommendation: Full valid sample count, matched A/A baseline, zero large-margin unexplained regression, known-answer test ও error-free serving—এইগুলোকে admission condition করো। Sample size ছোট হলে result INCONCLUSIVE, PASS নয়।
Q5. Width 6/4-এর জন্য flag ছাড়া কিছু দরকার?
Additional model-weight conversion দরকার এমন প্রমাণ নেই। Source অনুযায়ী runtime width override supported। কিন্তু trained width 8, bidirectional draft-block attention ও grouped-convolution indexing বদলায়—তাই width 4/6-কে simple truncation মনে করা যাবে না।

প্রতি width-এ checkpoint/config mismatch warning record, draft acceptance distribution, target-only parity, KV/graph capacity, errors এবং committed output tok/s মাপা প্রয়োজন।

সবচেয়ে গুরুত্বপূর্ণ হলো matched width-8 control, যাতে earlier patch/nodelay/graph differences width-এর ফলের মধ্যে না ঢোকে।
Q6. Quota কম হলে কী বাদ দেব?
প্রথমে বাদ দেব A2-এর speculative follow-up arms: width 4, width 6, int8 Mamba, অতিরিক্ত NCCL env relaunch।

এরপর expensive repeated profiling বা long-context arms কমানো যায়।

যেগুলো রাখব: startup reliability, bounded NCCL baseline, ছোট serving/profile diagnostic, necessary parity এবং reserved 25-minute Duck run।

শেষ quota reading 8.63 hours পুরোনো। তাই নতুন quota না দেখে 200-minute wall budget চালানোর অনুমোদন দেব না। Notebook-এর internal hard stop এবং external Kaggle platform stall—দুটিও আলাদা ঝুঁকি।

5. ১৬টি fix-list-এর reconciliation
Fix IDs	Audit result
1–2 Deadline handling	Implemented, কিন্তু expired-call ও cumulative cleanup gaps রয়েছে
3 Real running-level plateau	Implemented; sample validity ও level comparability গুরুত্বপূর্ণ
4 A/A contamination rule	Implemented, কিন্তু parity gate incomplete
5 Code compilation/CR	Verified: all notebook code cells compile
6 Profiler flags	Present
7 NCCL adoption	Implemented, কিন্তু graph_correct gate missing
8 Nodelay arm	Present; no-benefit adoption সম্ভব
9 Plain-text profiler response	Fixed
10 Server-log error signs	Implemented
11 Pre-Duck flush	Implemented; clean-state guarantee conditional
12 Graph capture list	Implemented, তবে unparsed list-এ fail-open
13 Builder CR guard	Fixed
14 Watchdog peak metrics	Implemented
15 Fake-server flush tests	Present; bundled harness 26/26
16 Timing-test changes	Present; bundled suite five times 125/125

সিদ্ধান্ত: Claude ১৬টির কোনোটি সম্পূর্ণ বাদ দেয়নি। কিন্তু কয়েকটি fix-এর acceptance invariant পূর্ণ হয়নি; বিশেষত 1, 2, 4, 7, 8, 11 ও 12। উপরন্তু নতুন model-staging collision এসেছে, যা fix list-এ ছিল না।
Independent test status
- ZIP: 40 files, archive integrity PASS।
- Notebook SHA: exact match।
- Five embedded modules: source bytes ও README hashes match।
- Notebook code compilation: PASS।
- Bundled harness: 26/26 reported PASS।
- Bundled tests: five × 125/125 reported PASS।
- আমার independent execution: loadpath-এর 28 checks PASS-এর পরে environment-specific mount fixture failed; NCCL/trace tests 33 PASS; SGLang-patch tests source fixture-এর hardcoded D:/ARC AGI 2/... path অনুপস্থিত থাকায় পুনরায় চালানো যায়নি।
শেষের বিষয়টি implementation bug-এর প্রমাণ নয়। তবে পূর্ণ 125/125-কে independently reproduced বলতে পারছি না।

6. Must-fix বনাম nice-to-have
Must-fix before this SHA is pushed
#	Fix	Acceptance test
M1	Model/drafter staged paths unique	Two source dirs ending /1 remain separate, correct hashes
M2	Reserve Duck's 25-minute window before A1 measurements	Slow-start simulation never silently truncates solver allowance
M3	Shared absolute deadline for HTTP, abort and joins	Stuck 12-request phase exits within bounded cleanup margin
M4	NCCL CUDA-graph correctness required for candidate adoption	Eager correct + graph wrong ⇒ reject
M5	Correctness gate for adopted #44401/nodelay/graphs	Faster corrupted arm ⇒ reject; incomplete samples ⇒ inconclusive
M6	Matched W8/int8/target-only A2 controls, if A2 runs	An injected cross-arm divergence cannot pass via cold/hot self-parity

Nice-to-have
No-benefit nodelay adoption rule আরও discriminating করা; /flush_cache-এর cold-state diagnostic; graph-list parse fail-closed; E14-P full_reply_text_match off-by-one; portable test fixtures; profiler-window duration semantics।

এগুলো গুরুত্বপূর্ণ, কিন্তু M1–M6-এর মতো একই মাত্রার blocking failure নয়।

Final GO / NO-GO
NO-GO — Exact SHA a9dad9fc…

কারণ: Model-copy path collision independently reproduced, solver time reservation absent, এবং measurement/adoption correctness gates incomplete।

GO for targeted corrections and CPU revalidation. Existing rev6g Duck solver, model weights, task-selection logic ও competition-specific agent architecture বদলানোর প্রয়োজন নেই।

GPU run: সংশোধিত notebook-এর নতুন SHA, integrated test logs ও launch/quota plan পুনরায় audit করার পর সিদ্ধান্ত নেওয়া উচিত।

আমাদের মূল কাজ থেকে Claude সরে যায়নি। E14-A এখনো engine measurement এবং Rev6g Duck-এর controlled 25-minute evaluation-এর পরিকল্পনার মধ্যেই আছে। সমস্যা feature scope নয়; বরং বহু optional phase এক notebook-এ যোগ করার ফলে কিছু deadline ও adoption invariant বাদ পড়েছে। ভালো দিক হলো, প্রধান সমস্যাগুলো solver redesign ছাড়াই ছোট, নির্দিষ্ট patches দিয়ে ঠিক করা সম্ভব।
