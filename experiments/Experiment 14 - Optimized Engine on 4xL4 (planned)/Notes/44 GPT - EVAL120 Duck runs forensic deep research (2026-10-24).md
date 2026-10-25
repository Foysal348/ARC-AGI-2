<!-- Pasted by the user on 2026-10-24 (GPT deep research of EVAL120_DUCK_RUNS_DEEP_RESEARCH_FOR_GPT_2026-10-24.zip, incl. the follow-up on the 55 no-progress LOW tasks). Kept VERBATIM. Merged into Notes/43. -->

EVAL120 — তিনটি Duck Run-এর Forensic Audit ও Deep Research
ফাইল: EVAL120_DUCK_RUNS_DEEP_RESEARCH_FOR_GPT_2026-10-24(2).zip
পদ্ধতি: আগে README.md ও 00_GPT_PROMPT.md পড়েছি; তারপর তিনটি run-এর ৬০৪টি branch transcript ও JSON record, Poetiq R5 ledger, source code এবং public research পরীক্ষা করছি।
১. কেন Duck-এর solve rate এত কম?
120 public tasks · 172 test queries · তিনটি প্রায় 6.5-hour GPU run
XHIGH + E16
46/172
30.00% task-normalized

XHIGH base
41/172
25.83% task-normalized

LOW + E16
31/172
18.89% task-normalized


প্রধান ফল: Duck-এর সবচেয়ে বড় সমস্যা output selection নয়, executable candidate তৈরি করতে না পারা।

Task-level classification করেছি; একই task-এর b0 ও b1 একসঙ্গে ধরে প্রতিটি task একটি primary category-তে রেখেছি।
Primary outcome	XHIGH+E16	XHIGH base	LOW+E16
অন্তত একটি test query correct	37	32	25
Train-exact program আছে, test ভুল	10	14	16
Executable transform বা measurable progress নেই	64	67	55
Check/progress হয়েছে, কিন্তু train-exact নয়	9	7	24
মোট task	120	120	120

Source: প্রতিটি runs/<run>/Q27X_E12_DUCK_BUNDLE.zip-এর duck/*/branch_result.json, e12_run_report.json → eval_unit_records। এগুলো primary outcome categories; timeout, loops ও tool errors overlapping secondary causes।Generated tokens by task outcome
Millions of reported generated tokens, including both branches of each task. Deadline-cut requests may be missing from this accounting.



No transform/progress

Train-exact, test-wrong

Other failures

Correct task


0246XHIGH + E16XHIGH baseLOW + E16














XHIGH base-এ প্রায় 4.61 million generated tokens এমন ৬৭টি task-এ গেছে, যেখানে কোনো recorded transform বা measurable progress নেই। XHIGH+E16-তেও একই সমস্যা প্রায় 4.54 million tokens।

এটি মোট generated tokens-এর প্রায় 63–65%। এই সংখ্যা একাই দেখাচ্ছে কেন শুধু engine 5–10% দ্রুত করলেই Duck-এর score সমস্যার সমাধান হবে না।
২. Agent বনাম Model বনাম Engine — প্রকৃত সমস্যার উৎস
2.1 Branch-level evidence
Metric	XHIGH+E16	XHIGH base	LOW+E16
Total branches	202	199	203
First transform recorded	56	55	65
First check recorded	56	55	69
Host-verified train-exact branch	47	46	41
কোনো tool call হয়নি	60	55	0
Tool ছাড়া length-এ request শেষ	98	84	12
Total tool calls	1,591	1,558	2,506
Reported infra errors	2	4	3
Generated tokens	7.14M	7.09M	6.61M
Prompt cache hit	78.2%	76.8%	82.4%

Source: branch_result.json → rev6_events, requests, e12_run_report.json → usage_all। Tool calls এবং requests একে অপরের সমান হওয়ার প্রয়োজন নেই।এখানে LOW-এর একটি শিক্ষণীয় ফল রয়েছে। LOW প্রায় 60% বেশি requests করেছে, অনেক বেশি Python tool call করেছে, কম length-cut হয়েছে এবং cache hit-ও ভালো। তবু LOW-এর accuracy সবচেয়ে খারাপ।

অর্থাৎ tool calls বাড়ালেই solve rate বাড়বে না। Model-এর hypothesis quality, reasoning depth এবং Python দিয়ে কী পরীক্ষা করছে, সেগুলো বেশি গুরুত্বপূর্ণ।
2.2 Transcript থেকে সরাসরি উদাহরণ
Case A — 28a6681f: ভাবছে, code লিখতে বিলম্ব করছে

runs/XHIGH_E16/.../duck/28a6681f_b1/transcript.txt, line 250:
Now let me write the code.

কিন্তু ওই branch-এর first_transform_tokens = null। পুরো task-এর দুই branch মিলে 103,304 generated tokens, measurable progress নেই।


Case B — faa9f03d: manual reconstruction loop

runs/XHIGH_E16/.../duck/faa9f03d_b1/transcript.txt, line 373:
This is getting complicated. Let me think about pair 1 more.

Transcript-এর আগে বারবার হাত দিয়ে গ্রিডের রঙ, path ও position গণনা করার চেষ্টা রয়েছে। ফলাফল: কোনো accepted candidate নেই।


Case C — 78332cb0: একই reasoning পুনরাবৃত্তি

runs/XHIGH_E16/.../duck/78332cb0_b0/transcript.txt, lines 1935 এবং 2002-এ একই বাক্য আবার এসেছে:
Let me reconsider the whole thing. The rule might be simpler than I think.

এটি repetition-এর concrete example। অবশ্য সব দীর্ঘ reasoning-কে loop বলা যাবে না।


Case D — 0934a4d8: train-exact, কিন্তু test-wrong

runs/XHIGH_E16/.../duck/0934a4d8_b0/branch_result.json-এ solve_from_block-এর চারটি training pair exact। Host verification accepted।

কিন্তু test output gold-এর সঙ্গে মেলেনি। এটি overfitting/generalization failure, missing candidate নয়।


এই চারটি উদাহরণে আলাদা ধরনের সমস্যা আছে। সবার জন্য একই prompt nudge বা একই token cut সঠিক সমাধান নয়।
2.3 Root-cause ranking
1
Hypothesis-generation এবং tool-use behavior

সবচেয়ে বেশি evidence রয়েছে। ৬৪–৬৭টি task-এ meaningful candidate নেই; দীর্ঘ analysis ও inspection কার্যকর code-এ রূপান্তরিত হচ্ছে না।



2
Compute allocation এবং branch scheduling

প্রায় ১.৭–১.৮৫M tokens branch-1-এ খরচ হয়েছে, অথচ নতুন correct output আসেনি। সরাসরি measured opportunity cost।



3
Program generalization

XHIGH+E16-তে অন্তত ১০টি task-এ train-exact solution থাকলেও test ভুল। এগুলো candidate-first fix দিয়ে একা উদ্ধার হবে না।



4
Potential engine numerical-quality issue

গুরুত্বপূর্ণ এবং public evidence আছে, কিন্তু আপনার 120-task runs-এ কত accuracy ক্ষতি হয়েছে তা মাপা হয়নি।



5
Tool parser, EOS ও infrastructure failures

কিছু error আছে, কিন্তু observed counts দেখে এগুলো প্রধান ব্যাখ্যা নয়। ১,৫০০-এর বেশি requests-এর মধ্যে অধিকাংশই valid tool-call finish করেছে।



2.4 Model / DFlash2 numerical quality — public research
এখানে তিনটি source গুরুত্বপূর্ণ।

SGLang #36548 — opened 26 August 2026। Qwen3.8-27B+DFlash2-এ concurrency 8-এ একটি reporter 111/304 incorrect outputs পেয়েছিলেন, serial run-এ 0/100। BF32 SSM, disabling decode CUDA graph অথবা draft width কমালে ভুল কমলেও নির্মূল হয়নি। BF16 dense lm_head-এ 1/304 হয়েছিল। এটি অন্য checkpoint/hardware-এর controlled reproduction; আমাদের stack-এ একই error rate ধরে নেওয়া যাবে না। 

GitHub




SGLang #38009 — opened 4 September 2026। একই target model-এ thinking enabled অবস্থায় DFlash decoding target-only greedy sequence থেকে token 30-এ diverge করেছে, যদিও thinking-disabled control মিলেছে। 

GitHub




SGLang #35723 — opened 20 August 2026। Qwen3.5-architecture model-এ একটি runtime update-এর পর repetition loops এবং tool call না হওয়ার bug report আছে। 

GitHub




এসব আপনার observed no-candidate behavior-এর সঙ্গে কিছুটা সাদৃশ্যপূর্ণ। কিন্তু সাদৃশ্য মানেই একই root cause নয়।

বিশেষ করে আপনার XHIGH base run-এ প্রায় ১,৫৬৯ requests-এর মাত্র ০টি finish_reason=stop, ৮৪টি length এবং প্রায় সব বাকি tool-call finish। তাই public Qwen issue #216-এর empty-content EOS problem-কে এখানে primary culprit বলা যাবে না। 

QwenLM/Qwen3.8




সঠিক পরীক্ষা: একই ১২–২৪টি representative ARC prompts, একই RedHat INT4 target, একই sampling, target-only বনাম DFlash2 width6; short এবং long reasoning trajectories compare করতে হবে। Greedy token parity দিয়ে numerical-path divergence খুঁজবেন; sampled pass@2 দিয়ে task-quality difference মাপবেন। দুটো measurement এক নয়।
2.5 কেন Poetiq / uninterrupted reasoning বেশি কার্যকর?
Bundled reference-এ একই ১২০ task-এ ফলাফল:
Solver	Correct queries	Task-normalized
E10 ORDER2 direct 86k	58/172	আলাদা report metric প্রয়োজন
E10 Poetiq R5	78/172	48.61%
Duck XHIGH+E16	46/172	30.00%
Duck XHIGH base	41/172	25.83%
Duck LOW+E16	31/172	18.89%

Source: reference/poetiq_r5_eval120/ledger/summary.json, ledger_units.csv এবং Duck eval_unit_records।

এই comparison থেকে তিনটি সম্ভাব্য ব্যাখ্যা আসে:
- Poetiq-এর direct/program-synthesis loop-এ hypothesis তৈরির আগে বারবার Python inspection ও context switching করতে হয় না।
- দীর্ঘ uninterrupted reasoning একটি coherent transformation rule তৈরিতে সাহায্য করতে পারে; Duck-এর request caps এবং repeated resumes সেই process বদলে দেয়।
- Poetiq-এর input representation, program verification ও prompting ARC-AGI-2 static transformation task-এর সঙ্গে বেশি aligned হতে পারে।
তবে একটি সীমা অত্যন্ত গুরুত্বপূর্ণ: Poetiq runs-এ vLLM FP8+MTP এবং ভিন্ন token/prompt budgets রয়েছে; Duck uses INT4+DFlash2। এই comparison একা harness effect বনাম model-serving effect আলাদা করতে পারে না।

আর Tufa Labs-এর public Duck harness মূলত ARC-AGI-3-এর stateful interactive games-এর জন্য, যেখানে action নিয়ে environment থেকে নতুন observations আসে। ARC-AGI-2-এ static train/test examples দেওয়া আছে; অতিরিক্ত inspection কোনো নতুন ground truth তৈরি করে না। তাই ARC3 loop সরাসরি ARC2-তে স্থানান্তর করলেই optimal হবে, এমন প্রত্যাশা ঠিক নয়। 

GitHub




ARC Prize-এর নিজস্ব test policy-তেও ARC-AGI-2 stateless direct input/output benchmark হিসেবে বর্ণিত, যদিও agent harness গবেষণা উৎসাহিত করা হয়। 

ARC Prize



৩. Attempt-2 কেন প্রায় শূন্য?
3.1 Raw result
Metric	XHIGH+E16	XHIGH base	LOW+E16
Attempt-1 correct	45	41	31
Attempt-2 অতিরিক্ত correct	1	0	0
Branch-1 চালু হয়েছে	82	79	83
Branch-1 generated tokens	1.847M	1.730M	1.462M
Branch-1 train-exact candidates	1	0	0
Branch-1 test-correct candidates	0	0	0

Source: e12_generated_attempts.json, branch_result.json → checked, submissions, runs এবং gold data/arc-agi_evaluation_solutions.json।এখানে গুরুত্বপূর্ণ বিষয়: Attempt-2 selection algorithm প্রধানত নষ্ট নয়; alternative candidate pool-ই তৈরি হচ্ছে না।

তিনটি run-এর eval_unit_records যাচাই করে এমন কোনো test-correct host-verified candidate পাইনি, যেটি final attempts থেকে বাদ পড়েছে।

অর্থাৎ বর্তমানে ranking algorithm আরও sophisticated করলে rescue করার মতো correct candidate খুব কম।
3.2 Branch-1 ঠিক কতটা অপচয় করছে?
XHIGH+E16 b1 budget share
25.9%
1.847M / 7.144M tokens

Additional b1 correct outputs
0
Across all three runs



এটি next-action ranking-এর সবচেয়ে শক্ত evidence।

code/arc2_duck.py:2664–2697-এ stage-2 class অনুসারে branch-1 শুরু হয়। stage2_items() lines 2774–2791-এ start1 অনেক সময় paused b0 continuation-এর সঙ্গে compete করে। কিন্তু b1 নতুন indexed-grid representation দিয়ে আবার শুরু করে; b0-এর established hypothesis code সরাসরি inherit করে না।

আমার সুপারিশ: পরবর্তী ছোট controlled evaluation-এ branches=1 arm রাখুন। এতে b1-তে যাওয়া compute paused b0 refinements-এ যাবে। Production-এর সব b1 চিরতরে বাদ দেওয়ার সুপারিশ নয়—আগে দেখতে হবে reallocating time actually improves solves কি না।

Minimal configuration change:
solve_all_r6(    tasks=tasks,    order=order,    # Existing arguments unchanged    branches=1,)



CPU validation: 120 synthetic tasks-এ stage-1 coverage 120/120, branch-1 creation zero, paused b0 resumes correctly, late accepted candidate remains and final submission valid।

GPU validation: Same 24–48 representative tasks with branches=1 versus branches=2, identical per-task clock and engine settings। Correct queries, first checks, total verified candidates ও stage-2 improvement per million tokens তুলনা।
3.3 Better attempt-2 strategy
দুটি verified programs থাকলে তাদের আলাদা test predictions রাখা উচিত। কিন্তু একটি Duck program থাকলে দ্বিতীয় slot-এ একই output বা identity fallback না দিয়ে independent Poetiq/NVARC candidate ব্যবহার করা অনেক বেশি সম্ভাবনাময়।

এই strategy-তে অবশ্য candidate-এর train verification, valid grid, model/input isolation ও deterministic selection gate বজায় থাকতে হবে।
def select_two(candidates, test_input):    valid = [        c for c in candidates        if c["host_verified"] and c["valid_grid"]    ]    # Prefer independent sources and distinct test predictions.    distinct = []    seen = set()    for c in valid:        key = serialize_grid(c["test_output"])        if key not in seen:            distinct.append(c)            seen.add(key)    return distinct[:2]



এটি conceptual selection sketch; current select_attempts_r5()-এর shape/source-aware logic নষ্ট করে replace করা উচিত নয়। Poetiq ও Duck-এর branch/source identities একই candidate schema-তে merge করতে হবে।

৪. E16 early nudge: কাজ করেছে, নাকি noise?
Verdict: Mechanically worked; statistical gain এখনো inconclusive।
4.1 Mechanism
E16 code diff-এ পরিবর্তনগুলো localized:
- দুইটি successful inspection-only Python call-এর পরে early nudge eligibility।
- Threshold প্রায় 12k generated tokens।
- আগের 24k fallback অক্ষত।
- Tool-call markup না থাকলে raw assistant content preservation।
- No change to host verifier, pass@2, sandbox বা total token caps।
Source: code/arc2_duck_e16.py, _arc2_wants_nudge(), _e16_raw_content_message() এবং _run_python_tool()।
4.2 Actual measurements
Metric	XHIGH+E16	XHIGH base	LOW+E16
Nudges emitted	111	Existing 24k policy	177
Nudge subsequently seen	99	78	123
First-check median among checking tasks	30,658	27,822	24,951
Raw content restorations	0	—	7
Score	46/172	41/172	31/172

E16-এর প্রথম-check median বরং base-এর চেয়ে বেশি। কিন্তু median শুধু check করা tasks-এর মধ্যে, তাই এটি E16 delay ঘটিয়েছে—এমন causal evidence নয়।

Raw-content change XHIGH+E16-তে একবারও effective হয়নি। ফলে XHIGH score পার্থক্যকে raw-content preservation-এর ফল বলা যাবে না।
4.3 Paired comparison
একই ১৭২টি query-তে:
20
E16 solved, base failed

15
Base solved, E16 failed

26
Both solved


McNemar-এর exact discordant-pair binomial calculation:
\[
p \approx 0.50
\]

Task-level paired bootstrap-এ E16 minus base task-normalized improvement +4.17 percentage points, আনুমানিক 95% interval −4.17 থেকে +12.50 points।

এটি statistically conclusive improvement নয়।
4.4 E16 কি কিছু task ক্ষতিগ্রস্ত করেছে?
Observed: হ্যাঁ, ১৫টি previously-correct query এবার ভুল হয়েছে। কিন্তু সেগুলো nudge-এর কারণেই ভুল—এটি প্রমাণিত নয়। Sampling variation, request timing, scheduling ও branch divergence-ও সম্ভাব্য কারণ।

উদাহরণ: 247ef758, b5ca7ac4, de809cff base run-এ correct ছিল, E16 run-এ correct নয়। অপরদিকে 20270e3b, 898e7135, eee78d87 E16-তে correct হয়েছে, base-এ হয়নি।

এখন আমার সিদ্ধান্ত:
- Keep E16 as an experimental option.
- Production default হিসেবে E16 নিশ্চিত superiority দাবি করবেন না।
- আর শুধু nudge threshold 12k→6k নামানোর মতো আরেকটি arbitrary adjustment করবেন না।
- পরের experiment-এ nudge content নয়, actionable hypothesis generation মাপতে হবে।

৫. Duck বনাম Poetiq — Task-level Complementarity
এটি সম্ভবত পুরো ZIP-এর দ্বিতীয় সবচেয়ে মূল্যবান ফল।
5.1 Paired query outcomes
78/172
Poetiq R5 alone

88/172
Poetiq + XHIGH E16 oracle union


Paired outcome	XHIGH+E16 বনাম Poetiq	XHIGH base বনাম Poetiq
Both correct	36	34
Duck-only correct	10	7
Poetiq-only correct	42	44
Neither correct	84	87
Oracle union correct	88/172	85/172
Oracle union task-normalized	54.44%	53.61%

Source: reference/poetiq_r5_eval120/ledger/ledger_units.csv বনাম প্রত্যেক Duck run-এর eval_unit_records।Poetiq 48.61%-এর সঙ্গে XHIGH+E16-এর oracle union 54.44%: +5.83 percentage points।

কিন্তু এখানে oracle কথাটি অত্যন্ত গুরুত্বপূর্ণ। Test truth জেনে প্রতি query-তে Poetiq অথবা Duck-এর correct output নির্বাচন করেছি। Hidden evaluation-এ সেই truth পাওয়া যাবে না।
5.2 শুধুমাত্র Duck stage-1 ব্যবহার করলে কী হয়?
Stage-1-এর b0 record থেকে accepted candidate-এর tokens field ও first-exposure accounting ব্যবহার করে gold-check করেছি।
Combination	Correct queries	Task-normalized
Poetiq alone	78	48.61%
Poetiq + XHIGH E16 stage-1 oracle	83	51.11%
Poetiq + XHIGH base stage-1 oracle	79	49.44%
Poetiq + LOW E16 stage-1 oracle	80	50.28%
Poetiq + full XHIGH E16 oracle	88	54.44%

বাস্তব পর্যবেক্ষণ: Duck stage-1-এর প্রায় 3.86M generated tokens ব্যবহার করে Poetiq-এর বাইরে অতিরিক্ত মাত্র ৫টি correct outputs এসেছে।

এটি hybrid-এর জন্য গুরুত্বপূর্ণ warning। অতিরিক্ত solve পাওয়া যাচ্ছে, কিন্তু compute cost খুব বেশি।

আর Poetiq R5 ও Duck দুটি সম্পূর্ণ run একসঙ্গে যোগ করলে GPU budget দ্বিগুণের কাছাকাছি হবে। Hidden 12-hour competition-এ এই oracle combination সরাসরি বাস্তবায়ন করা যাবে না।
5.3 Hidden LB extrapolation কেন বিপজ্জনক?
আপনার supplied reference/notes/lb_history.md অনুযায়ী:
- Poetiq R5 public evaluation: 48.61%
- Poetiq R5 hidden leaderboard: 21.39
- S1-60K Poetiq hidden leaderboard: 34.58
- Duck Rev 6g hidden leaderboard: 22.78
অর্থাৎ same family-এর public score ও hidden competition score অনেক ভিন্ন হতে পারে। কারণটি এই ZIP দিয়ে এককভাবে নিশ্চিত নয়—runtime scale, task allocation, order, stochastic quality ও solver formulation সব বিবেচ্য।

ARC Prize public/private eval sets-কে difficulty-calibrated হিসেবে বর্ণনা করে, কিন্তু সেটি একটি নির্দিষ্ট Kaggle solver-এর identical-score guarantee নয়। 

ARC Prize




এই কারণে 54.44% oracle public union দেখে hidden leaderboard 50+ হবে বলা ভুল।
5.4 Realistic hybrid design
প্রস্তাবিত architecture:
All tasks
    │
    ▼
Cheap deterministic / NVARC candidate (when available)
    │
    ▼
Primary Poetiq-style long-reasoning solver
    │
    ├── verified candidate ─────────────┐
    │                                   │
    └── uncertain / incomplete cases    │
                │                       │
                ▼                       │
       Targeted Duck candidate          │
                │                       │
                └──────────┬────────────┘
                           ▼
             Host verify + diversity
                           ▼
                  attempt_1 / attempt_2


এটি proposed architecture, tested run নয়। সবচেয়ে গুরুত্বপূর্ণ শর্ত: Duck-কে সব task-এ full extra 28k tokens দেওয়া যাবে না। Routing decision নিতে হবে train-derived difficulty, incomplete/uncertain candidate state, available compute এবং predicted complementarity দিয়ে—gold test result দেখে নয়।

Source research হিসেবে ARCANA paper perception → hypothesis generation → execution → reflection এবং scheduler separation বর্ণনা করে। কিন্তু সেটির published abstract থেকে আপনার exact Kaggle stack-এ performance gain প্রমাণ করা যায় না। 

arXiv



৬. Concrete fixes — expected impact, risks, tests এবং next actions
6.1 কোন fix বাস্তবে কাজ করেছে?
Existing fix / feature	Measured verdict	Next decision
Stage-1 full coverage	Worked: 120/120 tasks started in all runs	Keep
KV-only worker admission	Stable 12-worker configuration; no observed false step-down	Keep
Rev 5 host verification	Worked: correct verified candidates final selection-এ হারায়নি	Keep
Pass@2 candidate selection	Mechanically correct, candidate scarcity	Keep; change candidate sources
Prompt/code-first instructions	Behavioral goal largely failed	Redesign how hypotheses are proposed
Facts/indexed grid views	Operationally worked, insufficient candidate yield	Compare against direct/full-grid view
28k stage-1 soft pause	Coverage worked, reasoning quality trade-off unresolved	Controlled no-pause arm
Stage-2 b0 refinement	Productive: XHIGH+E16 final 46 vs stage-1 20 correct queries	Retain productive continuations
Branch-1 independent view	Very poor yield: 0 additional correct across 3 runs	Disable in one experimental arm
E16 early nudge	Delivered, score gain uncertain	Experimental only
E16 raw content preservation	XHIGH: 0 effective restorations	Not a demonstrated lever
XHIGH effort	Outperformed LOW in these runs	Prefer for current solver
Output-cut/resume handling	98/84 tool-less length endings in XHIGH	Investigate request-cap/behavior interaction
DFlash2 width6 engine	Served all runs	Throughput operational; numerical equivalence still unproven
Cache tracking	Useful measured cache rates	Keep as diagnostic

6.2 Ranked action plan
P0

Preserve the known-good Poetiq 34.58 baseline
Mandatory
Hidden leaderboard evidence বর্তমানে Poetiq S1-60K-এর পক্ষে। Duck-only-এর 22.78-কে primary replacement করবেন না, যতক্ষণ না same-condition validation সেটিকে ছাড়িয়ে যায়।

Test: Last known-good notebook SHA, model paths, complete 240-task submission schema, fallback, wall-clock budget এবং offline compatibility।



P1

Branch-1 disabled versus current scheduler

Evidence: 1.46–1.85M tokens/run, zero extra correct output

Minimal change: branches=1 experimental flag। No new architecture needed।

Expected benefit: অনেক compute reallocate করা যাবে। Accuracy gain অমাপা, তাই নির্দিষ্ট +LB points দাবি করছি না।

Test: first-stage coverage, b0 continuation, stage2 new verified candidates, late test-correct discoveries এবং final valid submission।



P2

Direct long-think Poetiq-style solver versus Duck — same engine A/B

উদ্দেশ্য: harness effect ও engine effect আলাদা করা

একই SGLang INT4+DFlash2 server, একই 24–48 tasks, same time allocation দিয়ে direct program synthesis বনাম REPL compare করুন। Full task grids, stable output format, train verifier এবং one uninterrupted hypothesis-construction phase ব্যবহার করুন।

Success: candidate rate, pass@2 accuracy ও effective tok/s—তিনটিই report করতে হবে।



P3

Candidate-first prompt with an executable milestone

৬৪–৬৭টি no-progress tasks প্রধান লক্ষ্য

শুধু “code early” বাক্য নয়, response protocol-এ একটি minimal transform hypothesis ও check() milestone রাখুন। Tool-call-এর আগে long verbal explanation নিষিদ্ধ করার পরিবর্তে model-কে concise hypothesis → code → failure feedback loop-এ guide করুন।

A/B: First accepted program per generated-token band, no-transform task count, wrong-train cases এবং correct queries। Prescriptive template generalization ক্ষতিগ্রস্ত করলে rollback।



P4

Train-exact generalization review এবং second independent hypothesis

XHIGH+E16-তে ১০টি train-exact/test-wrong tasks

Fixed positions, accidental color/size matches, test input structure এবং multiple plausible rules review করুন। Existing accepted candidate discard করবেন না।

Test: Historical 0934a4d8, 78332cb0, 1ae2feb7-এ review একটি materially different second program তৈরি করতে পারে কি না; correct fixed-coordinate rules যাতে incorrectly rejected না হয়।



P5

Target-only বনাম DFlash2 numerical-quality isolation

External bug reports আছে; আমাদের causality unknown

Same target weights, prompt tokens, seed, concurrency, FP8 KV ও FP16 SSM। Greedy parity এবং independently sampled ARC solve comparison আলাদা রাখুন।

Test: short, 30k/60k context, concurrency 1 এবং 12; complete tool-call integrity এবং accepted-candidate differences। DFlash2 disable করার সিদ্ধান্ত কেবল এই ফলের পরে।



6.3 Token allocation সম্পর্কে নতুন সিদ্ধান্ত
Stage totals:
Run	Stage-1 generated tokens	Stage-2 generated tokens	Stage-2 improved runs
XHIGH+E16	3.861M	3.283M	30/170
XHIGH base	3.909M	3.183M	30/166
LOW+E16	3.654M	2.953M	33/174

বর্তমান allocation-এ stage-2 পুরোপুরি অকার্যকর নয়। কিন্তু stage-2-এর একটি বড় অংশ unproductive b1-এ যাচ্ছে।

আমার প্রস্তাবিত experimental policy:
Stage 1
    → All tasks, first exposure
    → Preserve verified candidates and partial programs

Stage 2
    → Resume b0 with promising partial progress
    → Also give bounded continuation to no-candidate b0
    → Do not start b1 in experimental arm
    → Keep deadline/fallback rules unchanged

Selection
    → Retain all host-verified candidates
    → Merge independent solver predictions
    → Select two distinct valid outputs


এই change-এ কোনো new GPU kernel, quantization, speculative decoding অথবা Duck sandbox rewrite দরকার নেই।
6.4 Four-day strategy, accounting for the 2 November deadline
আগামী controlled comparison-এ সবচেয়ে বেশি শেখা যাবে তিনটি experiments থেকে: branch-1 elimination, Poetiq-style uninterrupted hypothesis generation, এবং same-engine numerical parity। তিনটিকে আলাদা hypothesis হিসেবে রাখবেন, যাতে gain কোন পরিবর্তন থেকে এসেছে বোঝা যায়।

Public dataset-এর task IDs দিয়ে retrospective error taxonomy, paired evaluation ও benchmark balancing করা যায়। কিন্তু সেই labels দিয়ে hidden task routing logic তৈরি করা যাবে না।

Competition-এর official rules অনুযায়ী final notebook offline, 12-hour runtime limit এবং valid submission.json আবশ্যক। 

Kaggle




৭. Complete unsolved-task taxonomy
এটি 00_GPT_PROMPT.md-এর “classify every unsolved task” অনুরোধ পূরণের জন্য। প্রতিটি run-এর সব ১২০ task-কে class দিয়েছি; নিচে শুধু failed tasks। IDs source branch_result.json এবং eval_unit_records থেকে।

সীমা: no_transform_or_progress মানে কোনো recorded recognized transform বা measurable progress নেই। এটি task-এ একটি helper function-ও লেখা হয়নি, এমন দাবি নয়। Transcript-গুলো programmatically full-text scan করেছি, এবং representative passages manually inspected; প্রতিটি দীর্ঘ reasoning বাক্য semanticভাবে আলাদা hand-label করিনি।
XHIGH + E16 — 83 unsuccessful tasks
Train-exact but test-wrong — 10

0934a4d8, 1ae2feb7, 4c3d4a41, 53fb4810, 6e453dd6, 78332cb0, 7b5033c1, 89565ca0, dfadab01, e376de54

Checked/progressed but not train-exact — 9

271d71e2, 71e489b6, 97d7923e, a6f40cea, d35bdbdc, d8e07eb2, edb79dae, f560132c, f931b4a8

No recorded transform/progress — 64

13e47133, 142ca369, 16b78196, 195c6913, 20a9e565, 21897d95, 247ef758, 269e22fb, 28a6681f, 291dc1e1, 2b83f449, 2c181942, 2d0172a1, 332f06d7, 35ab12c3, 3a25b0d8, 3dc255db, 446ef5d2, 4c416de3, 4c7dc4dd, 4e34c42c, 5545f144, 581f7754, 62593bfd, 64efde09, 65b59efc, 6e4f6532, 6ffbe589, 7666fa5d, 7b0280bc, 7b3084d4, 7b80bb43, 7c66cb00, 800d221b, 8698868d, 88bcf3b4, 88e364bc, 8b7bacbf, 8e5c0c38, 9385bd28, 9bbf930d, a25697e4, a32d8b75, a47bf94d, abc82100, b0039139, b10624e5, b5ca7ac4, b6f77b65, b99e7126, b9e38dc0, c7f57c3e, cbebaa4b, d59b0160, da515329, db0c5428, dbff022c, de809cff, e12f9a14, e3721c99, e8686506, e87109e9, faa9f03d, fc7cae8d
XHIGH base — 88 unsuccessful tasks
Train-exact but test-wrong — 14

221dfab4, 31f7f899, 58490d8a, 5961cc34, 71e489b6, 78332cb0, 7b5033c1, 7b80bb43, 80a900e0, d35bdbdc, dbff022c, dfadab01, e376de54, fc7cae8d

Checked/progressed but not train-exact — 7

0934a4d8, 53fb4810, 62593bfd, 64efde09, 7b0280bc, 7b3084d4, e8686506

No recorded transform/progress — 67

136b0064, 13e47133, 142ca369, 16b78196, 195c6913, 20270e3b, 20a9e565, 21897d95, 271d71e2, 28a6681f, 291dc1e1, 2b83f449, 2c181942, 2d0172a1, 35ab12c3, 36a08778, 3a25b0d8, 3dc255db, 409aa875, 446ef5d2, 4c3d4a41, 4c416de3, 4c7dc4dd, 4e34c42c, 5545f144, 581f7754, 5dbc8537, 65b59efc, 6e4f6532, 6ffbe589, 7491f3cf, 7666fa5d, 800d221b, 8698868d, 88bcf3b4, 88e364bc, 89565ca0, 898e7135, 8b7bacbf, 8e5c0c38, 8f3a5a89, 9385bd28, 97d7923e, 9bbf930d, a251c730, a25697e4, a32d8b75, a47bf94d, a6f40cea, abc82100, b10624e5, b6f77b65, b99e7126, b9e38dc0, c7f57c3e, cb2d8a2c, cbebaa4b, d59b0160, d8e07eb2, da515329, e12f9a14, e3721c99, e87109e9, eee78d87, f560132c, f931b4a8, faa9f03d
LOW + E16 — 95 unsuccessful tasks
Train-exact but test-wrong — 16

0934a4d8, 221dfab4, 31f7f899, 3dc255db, 4c3d4a41, 53fb4810, 67e490f4, 7491f3cf, 7b5033c1, 80a900e0, 97d7923e, a6f40cea, b5ca7ac4, d35bdbdc, db695cfb, e376de54

Checked/progressed but not train-exact — 24

135a2760, 13e47133, 28a6681f, 291dc1e1, 2b83f449, 35ab12c3, 409aa875, 581f7754, 5961cc34, 5dbc8537, 71e489b6, 7b3084d4, 7b80bb43, 800d221b, 8b7bacbf, 981571dc, a25697e4, d59b0160, dbff022c, de809cff, dfadab01, f560132c, faa9f03d, fc7cae8d

No recorded transform/progress — 55

136b0064, 142ca369, 16b78196, 195c6913, 20270e3b, 20a9e565, 21897d95, 271d71e2, 2c181942, 2d0172a1, 332f06d7, 3a25b0d8, 446ef5d2, 4c416de3, 4c7dc4dd, 4e34c42c, 5545f144, 62593bfd, 64efde09, 65b59efc, 6e4f6532, 6ffbe589, 7666fa5d, 78332cb0, 7b0280bc, 8698868d, 88bcf3b4, 88e364bc, 89565ca0, 898e7135, 8b9c3697, 8e5c0c38, 9385bd28, 9bbf930d, a251c730, a32d8b75, a47bf94d, abc82100, b0039139, b6f77b65, b99e7126, b9e38dc0, c7f57c3e, cb2d8a2c, cbebaa4b, d8e07eb2, da515329, db0c5428, e12f9a14, e3721c99, e8686506, e87109e9, edb79dae, eee78d87, f931b4a8

৮. Final research verdict
NO-GO — Duck-only as primary
GO — Controlled solver experiments

এই তিনটি বড় run-এর evidence থেকে সিদ্ধান্ত:

প্রথম: Qwen3.8-27B মডেল পুরোপুরি অক্ষম নয়—Poetiq এবং direct baselines একই 120-task dataset-এ বেশি solve করেছে। কিন্তু engine ও prompt ভিন্ন, তাই INT4+DFlash2 numerical contribution এখনও আলাদা পরীক্ষা দরকার।

দ্বিতীয়: বর্তমান Duck ARC-AGI-2 adaptation-এ executable hypothesis creation খুব কম। প্রায় 60%+ generation wrong/no-progress tasks-এ ব্যয় হচ্ছে। Input representation, branch policy এবং thinking-to-code transition প্রধান engineering problem।

তৃতীয়: b1-তে ১.৫–১.৮৫M tokens দিয়ে correct output শূন্য—এটি এখন নতুন stage-2 allocation experiment চালানোর যথেষ্ট evidence।

চতুর্থ: E16-এর +৫ correct queries একটি encouraging run result, কিন্তু paired significance ও flips বলছে এটিকে confirmed improvement বলা যাবে না।

পঞ্চম: Poetiq+Duck-এর 88/172 oracle union সম্ভাব্য complementarity দেখায়, কিন্তু বাস্তব hybrid gain ছোট এবং GPU cost বড় হতে পারে। Blind ensemble নয়—verified independent predictions এবং compute-aware routing দরকার।
Public sources most relevant to the next work
- SGLang #36548 — DFlash2 concurrent state corruption
- SGLang #38009 — DFlash versus target-only thinking divergence
- SGLang #35723 — reasoning repetition regression
- Qwen #216 — empty content at xhigh effort
- Tufa Labs Duck original prompting and Python workflow
- ARCANA — Reflective Multi-Agent Program Synthesis (10 July 2026)
- ARC Prize — ARC-AGI-2 dataset and evaluation design
আমার prioritization: নতুন GPU time থাকলে আগে branches=1 বনাম current scheduler এবং same-engine direct Poetiq-style reasoning comparison। এরপর bounded target-only/DFlash quality isolation। আরও nudge wording বা micro-optimizations এখন high-priority নয়।

কোনো source, notebook, GitHub commit বা Kaggle submission পরিবর্তন করিনি।
















No recorded transform/progress — 55 এটার কারণ ও সমাধান দিও












Worked for 5m 32s







LOW + E16: ৫৫টি task-এ Progress না হওয়ার কারণ ও সমাধান
ZIP-এর README.md, 00_GPT_PROMPT.md, LOW+E16-এর branch_result.json, transcripts এবং code/arc2_duck_e16.py মিলিয়ে ৫৫টি task-এর আলাদা বিশ্লেষণ করেছি।
১. সবচেয়ে গুরুত্বপূর্ণ আবিষ্কার
৫৫টি task-এ model কোনো Python code লেখেনি—এটি সঠিক ব্যাখ্যা নয়। বরং প্রচুর code লিখেছে, কিন্তু কার্যকর transform(grid) তৈরি ও যাচাই করার পর্যায়ে পৌঁছাতে পারেনি।
Progress ছাড়া task
55/120
45.8% of public evaluation tasks

Generated tokens
3.66M
55টি task-এর সব branches মিলিয়ে

Python tool calls
1,287
55/55 task-এ Python ব্যবহার হয়েছে

কোনো Python function সংজ্ঞায়িত
53/55
অধিকাংশই inspection/analysis helpers


Source: runs/LOW_E16/Q27X_E12_DUCK_BUNDLE.zip/duck/*/branch_result.json এবং transcript.txt। Function counts executed tool-call blocks-এর source থেকে গণনা; এগুলো retained বা correct function-এর count নয়।

আরও কয়েকটি গুরুত্বপূর্ণ ফল:
পর্যবেক্ষণ	৫৫টির মধ্যে
অন্তত ১০টি Python tool call করেছে	54
অন্তত ২০টি Python tool call করেছে	41
অন্তত একটি context-trimmed request	37
অন্তত একটি finish_reason=length	8
Early/provisional nudge পাঠানো হয়েছে	55
অন্তত একটি nudge পরবর্তী request-এ পৌঁছেছে	55
Harness infrastructure error recorded	2

এর অর্থ, প্রধান bottleneck Python tool অনুপস্থিতি নয়; exploration থেকে executable solution-এ রূপান্তর না হওয়া।

একই সঙ্গে নোট করুন: No recorded transform/progress হলো harness-এর নির্দিষ্ট metric। কোনো task-এ trans, solve, transform বা অন্য নামে আংশিক code থাকলেও সেটি retained, valid one-grid solver হিসেবে শনাক্ত না হলে এই category-তে থাকতে পারে। তাই এই ৫৫টি task-এর সবগুলোকে একই ধরনের reasoning failure বলা যাবে না।
২. Root cause: ৫৫টি task-এর ভেতরে আসলে কী ঘটেছে?
এই তালিকার কারণগুলো পরস্পর overlap করে। যেমন একই task-এ অতিরিক্ত inspection, context trimming ও output-limit—তিনটিই থাকতে পারে। তাই এগুলো যোগ করে ৫৫ বানানো যাবে না।
কারণ / evidence	আক্রান্ত task	গুরুত্ব
Python দিয়ে তথ্য বিশ্লেষণ করেছে, কিন্তু recorded check() result নেই	55	প্রধান bottleneck
Python functions লিখেছে, কিন্তু usable verified solver হয়নি	53	অত্যন্ত গুরুত্বপূর্ণ
Context trimming হয়েছে	37	সম্ভাব্য secondary cause
Request output limit hit করেছে	8	দীর্ঘ reasoning interruption
Harness infrastructure timeout/error	2	নির্দিষ্ট code execution failure
Nudge পৌঁছানোর পরও candidate progress নেই	55	বর্তমান intervention যথেষ্ট নয়
Branch-1 অতিরিক্ত computation নিয়েছে	53	Allocation inefficiency

Cause A — Analysis paralysis: model পরীক্ষা করেই যাচ্ছে
142ca369 task-এর ফল:
- 71,597 generated tokens।
- 28 Python tool calls।
- Repeated object/beam analysis।
- কোনো recorded check() result নেই।
- Nudge পেয়েছে, কিন্তু verified transform হয়নি।
b9e38dc0-তে 35টি Python calls এবং 64,462 generated tokens। অসংখ্য flood-fill, boundary, enclosure helper লিখেছে, কিন্তু একটি পূর্ণ input→output function যাচাই করেনি।

এমন আচরণের pattern:
Inspect grids → Extract objects → Hypothesize


Inspect again → Change hypothesis → Inspect again


Executable transform + check পর্যায়ে পৌঁছায় না


এটি সবচেয়ে বেশি tokens নষ্ট করছে।

সমাধান: Inspection-কে উদ্দেশ্যহীন exploration না রেখে candidate-driven করতে হবে। দুই-তিনটি inspection-এর পর একটি executable hypothesis চাই, সেই hypothesis ভুল হলেও। তারপর Python-এর actual failing cells দিয়ে সংশোধন।

কিন্তু বর্তমান E16 already early nudge ব্যবহার করে। তাই শুধু আরও কড়া nudge লিখে সমস্যার সমাধান হবে, এমন প্রমাণ নেই। আমাদের response workflow-ই বদলাতে হবে।
Cause B — Code আছে, কিন্তু transform(grid) নেই
cbebaa4b task-এ:
- 58,015 tokens।
- 19 Python calls।
- center, frames, transform_point ইত্যাদি helper।
- কোনো পূর্ণ train-checked transform(grid) নেই।
এখানে model transformation-এর অংশগুলো তৈরি করেছে, কিন্তু সেগুলোকে একটি পূর্ণ grid-to-grid function-এ assemble করেনি।

Code-level solution: Helper completion-এর পরে final composition step স্পষ্ট করুন। উদাহরণস্বরূপ, model যে trans(g) লিখেছে তা সত্যিই grid-to-grid function হলে:
def transform(grid):    return trans(grid)check(transform)



এটি কেবল তখনই কাজ করবে যখন trans(grid) প্রয়োজনীয় output ফেরায়। শুধু wrapper লিখে কোনো ভুল algorithm সঠিক হয় না।

code/arc2_duck_e16.py-এর _R6_AUTOCHECK_RE মূলত transform*, solve*, solution*, predict* এবং version-suffixed names শনাক্ত করে। trans-এর মতো নাম শনাক্ত করে না।

Patch option: Explicit candidate names সমর্থন করুন:
_R6_AUTOCHECK_RE = re.compile(    r"^(transform|solve|solution|predict|candidate|trans)\w*$"    r"|^\w+_(alt\w*|v\d+)$")



তবে এই পরিবর্তন সব helper-এর জন্য blindly apply করা উচিত নয়। Existing _r6_single_grid_arg() guard এবং 30-second execution budget বজায় থাকবে।

Validation: trans(grid) grid ফেরালে autocheck; trans(x,y) autocheck নয়; non-grid output reject; wrong-train program candidate হিসেবে retain নয়।
Cause C — Actual solver তৈরি করলেও sandbox execution ব্যর্থ
এটি classification-এর সবচেয়ে গুরুত্বপূর্ণ exception।

e8686506_b0-তে বাস্তবে def transform(grid) লেখা হয়েছিল। পরে print(check(transform))-ও চেষ্টা করেছে।

কিন্তু transcript-এ:
- একটি call ImportError: Module 'sys' is not allowed দিয়ে ব্যর্থ।
- পরের দুটি check(transform) attempt-এ Tool timed out after 30s।
- ফলে সফল execution, retained transform এবং progress signal পাওয়া যায়নি।
অর্থাৎ এই task-কে নিছক “model code লেখেনি” বলা ভুল।

Fix: ব্যয়বহুল search/backtracking-এর মধ্যে bounded node counter যুক্ত করতে হবে।
def bounded_search(initial_state, max_nodes=20000):    nodes = 0    def dfs(state):        nonlocal nodes        nodes += 1        if nodes > max_nodes:            return None        if is_solution(state):            return state        for next_state in successors(state):            result = dfs(next_state)            if result is not None:                return result        return None    return dfs(initial_state)



এটি illustrative control structure, e8686506-এর ready-to-paste solver নয়। Actual packing code-এ dfs()-এর state, candidate placement ও rollback অক্ষত রেখে counter যুক্ত করতে হবে।

আর check()-এর আগে একটি training input-এ candidate চালিয়ে return shape ও runtime যাচাই করা উচিত।

Test: Hard unsolvable packing → node budget-এ terminate; easy packing → valid solution; same deterministic result under repeated calls; কোনো sandbox timeout নয়।
Cause D — Context loss ও output-limit interruption
৫৫টি task-এর ৩৭টিতে context trimming হয়েছে। আটটিতে request length limit hit হয়েছে।

db0c5428-তে 82,848 generated tokens, দুটি length-cut request, কিন্তু tool-executed complete candidate নেই। Transcript-এ def transform(grid)-এর text reasoning অংশে পাওয়া যায়, executed retained function হিসেবে নয়।

এর কারণ হতে পারে model একটি rule ভেবেছে, কিন্তু সেই code Python tool call-এ পৌঁছানোর আগে reply শেষ হয়েছে।

Fix: শুধু reasoning transcript-এ code আছে কি না নয়—tool executed কি না তা মাপতে হবে। Length-cut-এর পর state-aware continuation আরও কার্যকরভাবে Python call-এ নিয়ে যেতে হবে।

Current R6_TEXT["length"] already instructs the model to write transform and check()। তাই একই message পুনরাবৃত্তি নয়; short experimental candidate-construction turn পরীক্ষা করুন।

এখানে নতুন streaming-engine rewrite অবিলম্বে সুপারিশ করছি না।

৩. এই ৫৫টি task কি মডেলের পক্ষে খুব কঠিন?
এই প্রশ্নের উত্তর দিতে একই task-এ অন্য solver-এর ফল মিলিয়েছি।
7/55
XHIGH+E16 অন্তত একটি query correct করেছে

5/55
XHIGH base solve করেছে

22/55
Poetiq R5 অন্তত একটি query correct করেছে


Poetiq ৫৫টির মধ্যে ২২ task-এ মোট ২৭টি correct queries পেয়েছে।

উদাহরণ:
Task	LOW+E16	অন্য run-এর evidence
136b0064	No progress	XHIGH+E16 এবং Poetiq correct
8b9c3697	No progress	XHIGH-এর দুই variant-এ correct
cb2d8a2c	No progress	XHIGH+E16 এবং Poetiq correct
332f06d7	No progress	XHIGH base ও Poetiq correct
db0c5428	No progress	XHIGH base ও Poetiq correct
62593bfd	No progress	Poetiq correct
e8686506	Timeout after transform attempt	Poetiq correct

এই evidence দেখায় যে অন্তত কিছু ব্যর্থতা solver/workflow-dependent। সব ৫৫টি task অসম্ভব বা intrinsically too hard নয়।

তবে XHIGH বনাম LOW comparison একই Duck harness-এর, আর Poetiq-তে engine, prompt, thinking budget ও workflow আলাদা। তাই Poetiq-এর বাড়তি ২২ solve-এর পুরো credit শুধু agent design-কে দেওয়া যাবে না।

৪. কী কী fix করব — নির্দিষ্ট অগ্রাধিকার
Priority	Fix	কেন	Validation metric
P0	LOW-এর বদলে XHIGH primary রাখুন	একই dataset-এ XHIGH বেশি correct, LOW বেশি inspection-heavy	Paired correct queries
P0	No-candidate task-এ hypothesis→code→check workflow	1,287 tool calls থেকেও recorded progress নেই	First valid candidate, first check
P1	Branch-1 বন্ধ রেখে একটি controlled arm	৫৫টির ৫৩টিতে b1 চালু, প্রায় 949,800 tokens ব্যয়	Correct outputs per million tokens
P1	Retained helper থেকে পূর্ণ candidate assembly	53 task-এ functions আছে, usable solver নেই	Candidates checked per task
P1	Timeout-resistant search/packing	e8686506-এর actual check failure	Sandbox runtime, verified programs
P1	Better diagnostic classification	Written-but-not-executed code বর্তমানে no-progress বলে ধরা পড়ে	No-code / candidate-attempt / timeout counts
P2	Facts view বনাম compact actual grid view A/B	অনেক task-এ spatial rule reconstruction কঠিন	Candidate rate ও accuracy
P2	DFlash2 বনাম target-only quality comparison	Numerical contribution এখনো অজানা	Same-prompt parity ও ARC quality

Diagnostic metrics patch
বর্তমান first_transform_tokens-এর ওপর একা নির্ভর না করে code/arc2_duck_e16.py-এর _run_python_tool()-এ AST parse-এর পরে আলাদা source-attempt counters রাখা যায়।
def classify_candidate_source(code):    import ast    try:        tree = ast.parse(code)    except SyntaxError:        return {            "candidate_written": False,            "syntax_ok": False,        }    defs = [        node.name        for node in tree.body        if isinstance(node, ast.FunctionDef)    ]    return {        "syntax_ok": True,        "defined_functions": defs,        "candidate_written": any(            re.match(                r"^(transform|solve|solution|predict|candidate|trans)",                name            )            for name in defs        ),    }



এটি source attempt মাপবে, successful execution বা verified candidate নয়। আলাদা fields হওয়া উচিত:
- candidate_source_written
- candidate_retained
- check_attempted
- check_executed
- check_timeout
- train_pairs_exact
- host_verified
এই পার্থক্য না রাখলে e8686506-এর মতো failure-কে আবার ভুলভাবে “model never coded” হিসেবে রিপোর্ট করা হবে।

Validation test: Valid candidate, unsupported import, timed-out check, reasoning-only code এবং successful host verification—প্রতিটি case আলাদা category-তে পড়তে হবে।

৫. পরবর্তী GPU experiment কী হওয়া উচিত?
আমার প্রস্তাব একটি ছোট paired experiment—সব ১২০ task নিয়ে আবার তিনটি দীর্ঘ run নয়।

প্রথমে এই ৫৫টি থেকে stratified ২৪টি task নিন। কিছু XHIGH-solved, কিছু Poetiq-only, কিছু no-solver-solved এবং কিছু timeout/length-cut cases রাখুন।

একই GPU server, একই context এবং একই wall-clock budget-এ তুলনা করুন:
Arm	কী পরীক্ষা করব
A — Current XHIGH Duck	Baseline
B — XHIGH, branches=1	b1 budget b0 refinement-এ দিলে লাভ?
C — XHIGH, hypothesis-first workflow	Inspection loop কমিয়ে candidate বাড়ে?
D — Direct Poetiq-style solver	Tool-agent architecture-এর opportunity cost কত?

প্রাথমিক success criteria:
- No-recorded-candidate task count কমে কি না;
- 20k–30k tokens-এর মধ্যে প্রথম executable candidate আসে কি না;
- First check() rate বাড়ে কি না;
- Train-exact ও test-correct outputs বাড়ে কি না;
- প্রতি এক মিলিয়ন generated tokens-এ কত correct output পাওয়া যায়;
- Context loss, timeout ও failed execution-এর আলাদা সংখ্যা।
কেবল check বেশি হলেই GO নয়। Correct queries per GPU-hour উন্নত হতে হবে।
৬. আমার চূড়ান্ত সিদ্ধান্ত
NO-GO
LOW+E16-কে primary solver রাখা

GO
XHIGH ও Poetiq-based controlled redesign

৫৫টি task-এর root cause মূলত তিন স্তরে:

প্রথম: Model Python ব্যবহার করছে, কিন্তু executable ARC rule তৈরি করছে না। এটি সবচেয়ে বড় সমস্যা।

দ্বিতীয়: Code তৈরি হলেও কখনো retained/recognized solver নয়, কখনো sandbox timeout-এর কারণে check অসম্পূর্ণ।

তৃতীয়: Context trimming, repeated inspection এবং branch-1 allocation এই দুর্বলতাকে আরও ব্যয়বহুল করছে।

এখানে সবচেয়ে বড় সম্ভাব্য উন্নতি নতুন 5–10% engine speed নয়। বরং বর্তমানে ব্যর্থ ৫৫টি task-এর মধ্যে যতগুলোতে সম্ভব একটি যাচাইযোগ্য transformation candidate তৈরি করানো।

XHIGH ইতিমধ্যে এই একই ৫৫টির কয়েকটি solve করেছে, আর Poetiq ২২টিতে সফল হয়েছে। তাই সমস্যার অন্তত একটি উল্লেখযোগ্য অংশে উন্নতির বাস্তব evidence আছে। তবে কোনো নির্দিষ্ট fix কতগুলো task উদ্ধার করবে, সেটি এখনও মাপা হয়নি।

কোনো code, notebook, commit বা push পরিবর্তন করিনি।
