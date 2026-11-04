<!-- Pasted by the user on 2026-10-24: a second GPT deep-research report on EVAL120_DUCK_RUNS_DEEP_RESEARCH_FOR_GPT_2026-10-24.zip. Kept VERBATIM. Merged into Notes/43. -->

EVAL120 Duck — Deep Forensic Research
২৪ অক্টোবর ২০২৬ | ১২০ ARC-AGI-2 tasks, ১৭২ test queries

Bundle-এর তিনটি nested run archive থেকে ৬০৪টি branch transcript (মোট প্রায় ৮৮ MB text), branch results, generated attempts, evaluation ground truth এবং Poetiq reference ledger পরীক্ষা করেছি। README-র ছয়টি প্রশ্ন অনুযায়ী বিশ্লেষণ করছি।
১. প্রথম বড় ফলাফল: সমস্যা শুধু engine speed নয়
XHIGH + E16
46/172
30.00% task-normalized

XHIGH, no E16
41/172
25.83% task-normalized

LOW + E16
31/172
18.89% task-normalized


এগুলো কেবল report থেকে কপি করিনি—e12_generated_attempts.json-এর প্রতিটি attempt_1/attempt_2-কে supplied arc-agi_evaluation_solutions.json-এর সঙ্গে স্বাধীনভাবে তুলনা করেছি। তিনটি score-ই মিলে গেছে।
১২০ task-এর প্রথম failure taxonomy
প্রতিটি task-এর সব branch একসঙ্গে ধরে শ্রেণিবদ্ধ করেছি। এগুলো mutually exclusive primary categories; tool errors, repetition, timeouts আলাদা overlapping কারণ হিসেবে পরে বিশ্লেষণ করছি।
Task-level outcome	XHIGH + E16	XHIGH no E16	LOW + E16
সম্পূর্ণ solved	35	30	20
Train-exact program পাওয়া, কিন্তু task সম্পূর্ণ solve হয়নি	12	16	21
check() হয়েছে, train-exact হয়নি	9	7	24
Usable transform/check কোনোটিই নেই	64	67	55
মোট	120	120	120

প্রধান আবিষ্কার: XHIGH+E16-এ ১২০টির মধ্যে ৬৪টি task কোনো retained, usable transform candidate পর্যন্ত পৌঁছায়নি। এটি engine-এর ৫–১০% speed gain দিয়ে একা সমাধান হওয়ার মতো সমস্যা নয়।

LOW+E16 আরও বেশি task-এ transform/check পর্যন্ত পৌঁছেছে, কিন্তু train-exact ও test-correct কম পেয়েছে। তাই আগে code লেখা গুরুত্বপূর্ণ, কিন্তু সঠিক hypothesis তৈরি করা আরও গুরুত্বপূর্ণ।
Poetiq-এর সঙ্গে স্বাধীনভাবে মাপা complementarity
reference/poetiq_r5_eval120/ledger/ledger_units.csv-এর Poetiq R5 solved-by records দিয়েও 78/172 score পুনর্গণনা করেছি।
Combination	Correct queries	Task-normalized
Poetiq R5 alone	78/172	48.61%
Duck XHIGH+E16 alone	46/172	30.00%
Poetiq ∪ Duck E16, oracle selection	88/172	54.44%
Poetiq ∪ Duck no-E16, oracle selection	85/172	53.61%
Poetiq ∪ LOW+E16, oracle selection	82/172	51.11%

Poetiq ও Duck E16 দুটোই সঠিক হয়েছে 36টি query-তে। শুধু Duck E16 সঠিক হয়েছে 10টি, শুধু Poetiq 42টি।

অত্যন্ত গুরুত্বপূর্ণ: 88/172 হলো দুই run-এর পূর্বে উৎপন্ন prediction থেকে ground-truth oracle union। এটি বাস্তবে ১২ ঘণ্টার মধ্যে চালানো hybrid score নয়। Gold-free selector, দুই solver-এর time allocation এবং scheduling যাচাই না করলে 54.44% অর্জিত result বলা যাবে না।
২. সবচেয়ে বড় confirmed waste: Branch-1 দ্বিতীয় hypothesis তৈরি করছে না
আগের audit-গুলোতে branch-1-এর utility নিয়ে সন্দেহ ছিল। এবার পূর্ণ ১২০-task run থেকে স্পষ্ট হিসাব পাওয়া গেছে।
Branch-1 metric	XHIGH + E16	XHIGH no E16	LOW + E16
Branch-1 শুরু	82	79	83
একবারও Python tool call করেনি	60	55	0
Recorded check() পর্যন্ত গেছে	1	3	9
Host-verified train-exact candidate	1	0	0
নতুন test-correct output	0	0	0
Generated tokens	1.847M	1.730M	1.462M

Branch-1-এর ব্যয় বনাম ফল
XHIGH + E16, ১২০ tasks; branch-1 generated-token budget-এর প্রায় ২৫.৯% পেয়েছে।


01.5M3M4.5M6MBranch-0Branch-1














XHIGH+E16-তে 1.847 million tokens—মোট generated tokens-এর প্রায় 25.9%—branch-1-এ খরচ হয়েছে, একটিও নতুন correct query ছাড়া। XHIGH no-E16 ও LOW+E16-তেও একই zero-gain pattern।

runs/*/Q27X_E12_DUCK_BUNDLE.zip → duck/*_b1/branch_result.json-এর actual result, rev6_events.tool_calls, first_check_tokens, checked, submissions এবং ground truth দিয়ে যাচাই করেছি।
Candidate selector কি correct program হারিয়েছে?
না—এই তিনটি run-এ তা হয়নি।

প্রতিটি branch-এর সব host-accepted checked[] ও submissions[] candidate-এর test_outputs সরাসরি ground truth দিয়ে পরীক্ষা করেছি।
Metric	XHIGH+E16	XHIGH no E16	LOW+E16
যেসব task-এ host-accepted program আছে	47	46	41
যেকোনো accepted program থেকে oracle-correct queries	46	41	31
Final two-attempt output-এর correct queries	46	41	31
Selector-এর কারণে হারানো correct query	0	0	0

এটি একটি শক্তিশালী finding: pass@2 selector ভেঙে নেই; দ্বিতীয়, সত্যিকার ভিন্ন এবং correct hypothesis তৈরি হচ্ছে না।

আমার প্রাথমিক P0 recommendation: বর্তমান 28k-first-exposure/24,576-cap configuration-এ b1-এ blanket compute দেওয়ার বদলে অন্য method—বিশেষত Poetiq-style long-think program synthesis—পরীক্ষা করো। তবে আগে ১২-task bounded replay ও same-wall comparison চাই; ১.৮৫M tokens অন্যত্র দিলে স্বয়ংক্রিয়ভাবে কত score বাড়বে, সেটি এখনো জানা নেই।
৩. Failure-এর প্রকৃত কারণ: agent, model, নাকি engine?
3.1 হারানো ১২৬টি query কোথায়?
XHIGH+E16-এর 46/172 correct মানে 126টি missed query। এগুলোকে task-এর সর্বোচ্চ অগ্রগতি অনুযায়ী ভাগ করলে:
Primary failure class	Tasks	Missed queries	মোট missed queries-এর ভাগ
কোনো recorded usable transform হয়নি	64	95	75.4%
Check হয়েছিল, train-exact হয়নি	9	15	11.9%
Train-exact পাওয়া, test-এ ভুল	12	16	12.7%
মোট unsolved	85	126	100%

একইভাবে XHIGH no-E16-এর 131টি missed query-এর মধ্যে 102টি এবং LOW+E16-এর 141টির মধ্যে 84টি এমন task-এ, যেখানে কোনো recorded usable transform হয়নি।

এই শ্রেণিবিভাগ runs/*/Q27X_E12_DUCK_BUNDLE.zip → e12_run_report.json → branches[], rev6_events, checked[] ও eval_unit_records থেকে তৈরি। এটি task-level primary classification; একই task-এ cache trim, token cap ও ভুল hypothesis একসঙ্গে থাকতে পারে।

সবচেয়ে বড় score-loss class পরিষ্কার: program synthesis শুরুই না হওয়া।

তবে “no transform” মানে model কোনো ভাবনাই করেনি নয়। উদাহরণ:
13e47133_b0 — XHIGH+E16

duck/13e47133_b0/transcript.txt, শেষের reasoning section

Model বিভিন্ন frame/distance hypothesis নিয়ে দীর্ঘ আলোচনা করেছে। এক পর্যায়ে লিখেছে:
“Ugh. So no. ... what the actual rule could be.”

কিন্তু পুরো branch-এ 60,719 generated tokens ও 23 tool calls-এর পরও কোনো usable transform/check নেই।

b5ca7ac4_b0 — E16 বনাম no-E16

E16-তে 65,104 tokens, 15 tool calls, কোনো transform নেই। অথচ no-E16 run-এ 54,012 tokens-এর কাছে transform/check হয়েছে এবং output correct।

এটি দেখায়, additional steering থাকলেই candidate quality উন্নত হবে—এমন নিশ্চয়তা নেই।

3.2 Tool ব্যবহার হচ্ছে, কিন্তু code তৈরি হচ্ছে না
Metric	XHIGH+E16	XHIGH no-E16	LOW+E16
Total Python tool calls, telemetry	1,591	1,558	2,506
E16 inspection-classified calls	1,266	—	2,085
Branches with no tool call	60	55	0
Branches with recorded first check	56	55	69
Median first check, observed	30,658 tokens	28,469	24,951

E16-এর inspection classification heuristic; সব 1,266 call যে নিখুঁত read-only ছিল তা নয়। কিন্তু tool usage-এর বড় অংশ inspection-oriented—এটি source telemetry দিয়ে সমর্থিত।

LOW-এর গুরুত্বপূর্ণ শিক্ষা: 2,506 tool calls এবং earlier median check সত্ত্বেও মাত্র 31 correct queries। তাই model-কে দ্রুত Python call করানোই যথেষ্ট নয়। ভুল hypothesis দ্রুত implement করলেও লাভ কম।
3.3 Token caps, loops, EOS এবং tool failures
Overlapping diagnostic	XHIGH+E16	XHIGH no-E16	LOW+E16
Requests ending finish_reason=length	98	84	12
Tool-call-free length cuts, telemetry	98	82	10
finish_reason=stop requests	1	0	8
Recorded adapter infrastructure errors	2	4	3
Branches with substantial repetition, threshold 1k characters	13	18	23
Branches with reasoning trim	66	78	86

Source: duck/*/branch_result.json → requests[], rev6_events, rev5_events।

XHIGH+E16-এর 98টি length request প্রতিটিই প্রায় 24,576-token cap-এ পৌঁছেছে। আনুমানিক 2.41M generated tokens এই capped requests-এ ব্যয় হয়েছে—মোট 7.14M-এর প্রায় 34%।

কিন্তু এই 2.41M পুরোপুরি অপচয় বলা যাবে না। Reasoning পরের request-এ retained থাকলে তা উপকারী হতে পারে। সমস্যাটি হলো, কিছু branch বারবার cap-এ পৌঁছেও Python implementation তৈরি করেনি।

Premature EOS/empty content: XHIGH+E16-এ মাত্র একটি stop request, LOW-তে আটটি; LOW-এর সাতটি একই task 58490d8a-তে। তাই এই bundle-এ premature EOS প্রধান failure class হওয়ার প্রমাণ পাইনি। শুধু finish_reason=stop দিয়েও empty-content corruption প্রতিষ্ঠিত হয় না।

Tool/parser bugs: Adapter infrastructure errors তুলনামূলক অল্প। Transcripts-এ NameError, IndexError ও invalid assumptions রয়েছে, কিন্তু এগুলোকে সব missed query-এর root cause বলা যাবে না। Host verifier-এর silent systematic rejection বা accepted-but-lost correct program পাওয়া যায়নি।

Repetition: XHIGH+E16-এর telemetry-তে repeated reasoning characters মোট reasoning characters-এর প্রায় 1.0%। এটি detector-defined measure; মানুষের চোখে দীর্ঘ circular hypothesis discussion অনেক বেশি হতে পারে। 13e47133-এর মতো repeated reconsideration detector-এ পুরোপুরি ধরা পড়ে না।
3.4 28k pause কি ভুল জায়গায়?
এই বিষয়ে বাস্তব সন্দেহ আছে।

XHIGH+E16-এর 56টি branch-এ first check এসেছে। তার মধ্যে:
- 23টি check হয়েছে 28k-এর মধ্যে।
- 33টি check হয়েছে 28k-এর পরে।
অর্থাৎ model যখন ব্যবহারযোগ্য code/check-এর দিকে এগোচ্ছিল, তখন বহু task-এ প্রথম exposure pause boundary অতিক্রম হয়েছে।

কিন্তু সব task-এর first exposure 28k→40k করলে 240-task hidden run-এ অতিরিক্ত প্রায় 2.88M tokens প্রয়োজন হতে পারে। 280 effective tok/s ধরলে প্রায় 2.9 ঘণ্টা অতিরিক্ত। Full coverage ক্ষতিগ্রস্ত হতে পারে।

Recommendation: Global first-exposure cap বাড়ানোর বদলে progress-aware continuation পরীক্ষা করা উচিত। Python-এ concrete candidate তৈরির পর্যায়ে পৌঁছানো branch-কে অল্প বাড়তি সুযোগ দেওয়া, কিন্তু hypothesis-less branch-কে অনির্দিষ্টভাবে চালিয়ে না যাওয়া।

এটি এখনো proposal, measured improvement নয়।

৪. কেন Poetiq এত এগিয়ে? Agent বনাম model বনাম engine
বিষয়	Duck XHIGH+E16	Poetiq R5
Public evaluation	46/172	78/172
Task-normalized	30.00%	48.61%
Generated tokens	~7.14M	~11.65M
Overall evaluation wall time	~6.5 ঘণ্টা	~11.5 ঘণ্টা
Serving	SGLang INT4 + DFlash2 W6	vLLM FP8 + MTP
Reasoning	বহু tool turns, request caps 12,288/24,576/32,768	Long reasoning, 77,824 thinking + 8,192 answer per iteration
Verification	Host-clean check() + harvested candidates	Train execution + feedback
Follow-up	Paused branches, branch-1	Failed task-এর iteration 2

প্রথম সতর্কতা: এটি controlled one-variable comparison নয়। Poetiq বেশি total tokens ও wall time পেয়েছে, এবং engine/quantization-ও আলাদা। তাই 18.61 task-normalized-point gap পুরোটাই Duck prompting-এর কারণে—এমন দাবি ভুল হবে।

তবু দুটি source-based observation গুরুত্বপূর্ণ।
A. Poetiq-এর computation সরাসরি program তৈরির দিকে যাচ্ছে
Poetiq-এর first iteration-এ model দীর্ঘ reasoning করে একটি executable program দেয়। সেই program training inputs-এ চালিয়ে feedback পাওয়া যায়।

Duck-এর বহু branch 40–60k tokens ব্যবহার করেছে, কিন্তু কোনো usable transform-ই তৈরি করেনি। এটি harness strategy ও model behaviour-এর mismatch-এর পক্ষে evidence।

Official Poetiq repository-ও executable program generation এবং iterative correction-কে মূল method হিসেবে ব্যবহার করে। 

GitHub




https://github.com/poetiq-ai/poetiq-arc-agi-solver
B. Duck-এর original design ARC-AGI-3 থেকে এসেছে
Tufa Labs-এর Duck মূলত interactive ARC-AGI-3 games-এর জন্য—সেখানে observation, action ও environment feedback ধারাবাহিকভাবে আসে। 

Tufa Labs
+1




https://tufalabs.ai/research/duck-harness/

ARC-AGI-2-তে সমস্যা ভিন্ন: fixed training examples থেকে একবার একটি generalizing transformation বের করতে হবে।

তাই Duck-এর strong interactive-inspection behaviour ARC2-তে unproductive হতে পারে। এটি আমাদের measured results-এর সঙ্গে সামঞ্জস্যপূর্ণ hypothesis, original Duck harness ভুল—এমন সিদ্ধান্ত নয়।
C. Engine/model-quality issue কি বাদ দেওয়া যায়?
না। কিন্তু এই bundle তা প্রতিষ্ঠা করে না।

External SGLang reports:
Public issue	Evidence	আমাদের relevance
#36548, ২৬–২৭ Aug 2026	DFlash2 high concurrency-তে target-only থেকে wrong outputs; selected affected subset-এ 111/304	Serious serving-correctness risk
#38009, ৪ Sep 2026	Thinking-enabled DFlash2 greedy output token 30-এ target-only থেকে diverge	XHIGH-এর জন্য directly relevant
#43956, ১৩ Oct 2026	SGLang 0.5.21-এ batch-shape-dependent numerical decisions, FP8/BF16	Batch-dependent behaviour সম্ভব
Qwen issue #216, Aug 2026	Thinking-mode premature stop/loop reports	আমাদের stop counts দিয়ে প্রধান cause প্রমাণিত নয়



GitHub
+3


Raw URLs:

https://github.com/sgl-project/sglang/issues/36548

https://github.com/sgl-project/sglang/issues/38009

https://github.com/sgl-project/sglang/issues/43956

https://github.com/QwenLM/Qwen3.8/issues/216

RedHat INT4 checkpoint-এর published generic evaluation-এ BF16-এর তুলনায় মোটামুটি ভালো recovery আছে, কিন্তু তা ARC-specific reasoning equivalence প্রমাণ করে না। তোমাদের bundled reference/notes/kernel_issue_redhat_readme.md-এর provenance:

https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4
সবচেয়ে সস্তা definitive experiment
একই ১২-task subset-এ same prompt ও token budget দিয়ে:
1. SGLang INT4 + DFlash2, Poetiq-style single-program generation।
2. SGLang INT4 target-only অথবা MTP, একই Poetiq prompt।
3. সম্ভব হলে existing vLLM FP8/MTP baseline-এর সঙ্গে তুলনা।
একই engine-এ Poetiq prompt যথেষ্ট উন্নতি করলে agent-loop hypothesis শক্তিশালী হবে। Same prompt-এ DFlash2 থেকে target-only/MTP অনেক বেশি correct হলে serving-quality hypothesis শক্তিশালী হবে।

এই experiment ছাড়া numerical corruption-কে 25.83%-এর root cause বলা যাবে না।

৫. E16 সত্যিই উন্নতি করেছে? আর Duck+Poetiq hybrid কতদূর যেতে পারে?
5.1 E16-এর gains এবং losses দুটোই বাস্তব
একই ১৭২ query-তে XHIGH+E16 বনাম no-E16:
Paired outcome	Queries
দুটো run-ই correct	26
শুধু E16 correct	20
শুধু no-E16 correct	15
দুটোই wrong	111
Net	+5 queries

এটি কেবল নতুন পাঁচটি answer নয়। E16 ২০টি answer উদ্ধার করেছে, কিন্তু no-E16-এর ১৫টি correct answer হারিয়েছে।

উদাহরণ:
Task	E16	No-E16	কী ঘটেছে
136b0064	Correct	Wrong	E16 transform/check পেয়েছে; no-E16 50k tokens-এও পায়নি
5dbc8537	Correct	Wrong	E16 candidate পেয়েছে; no-E16-এ নেই
1ae2feb7	Wrong	Correct on 3 queries	No-E16 প্রায় 17.6k-এ transform পায়; E16 b0-তে পায়নি
b5ca7ac4	Wrong	Correct	E16 65k tokens-এও no transform; no-E16 54k-এ correct
269e22fb	Wrong	Correct on 2 queries	no-E16 better hypothesis তৈরি করেছে
221dfab4	Correct on 2	Wrong	E16 verified program পেয়েছে

5.2 E16-এর লাভ কি statistically established?
না।

Single-run paired query comparison-এ discordant outcomes 20 বনাম 15। Exact two-sided binomial/McNemar-style test-এ p ≈ 0.50।

Task-level bootstrap-এ E16−no-E16 task-normalized difference:
+4.17 percentage points
95% task-bootstrap interval: approximately −4.17 to +12.08 points

Interval-এ zero রয়েছে। তাই E16-এর observed score improvement promising, কিন্তু একটি run দিয়ে reliable causal gain প্রমাণিত নয়।

আর E16 যে first check earlier আনবে—সেটিও data-তে প্রতিষ্ঠিত হয়নি:
Metric	E16	No-E16
Median first-check tokens, observed	30.7k	28.5k
Branches reaching check	56	55
Tasks with any usable host-accepted program	47	46

Raw-content preservation-এর cache gain? XHIGH+E16-এ e16_raw_content_kept=0। অর্থাৎ নতুন raw-content restoration কোনো eligible event-এ কাজই করেনি। LOW run-এ সাতটি event ছিল।

XHIGH cache share 78.25% বনাম no-E16 76.76%, কিন্তু এই 1.49-point difference raw-content fix-এর প্রমাণ নয়। Prompt length, task path ও cached state বদলেছে।

E16 verdict: Keep as a research candidate, not an established production improvement। একে সরাসরি 240-task hidden LB gain ধরে নেওয়া যাবে না।
5.3 Hybrid-এর নির্ভরযোগ্য upper bound
Poetiq R5 ও Duck E16-এর independently verified outputs:
Oracle union
88/172
54.44% task-normalized

Duck exact → otherwise Poetiq
79/172
49.86% task-normalized, retrospective


দ্বিতীয় সংখ্যাটি বিশেষ গুরুত্বপূর্ণ:

যেসব task-এ Duck কোনো host-accepted train-exact candidate পায়নি, শুধু সেগুলোতে full historical Poetiq R5 result ব্যবহার করলে 79/172 পাওয়া যেত।

এটি hindsight replay, কিন্তু switch criterion gold-free: Duck train-exact আছে কি নেই।

তবুও ওই 73টি Duck-unverified task-এ full Poetiq reasoning/iteration budget চালাতে যত সময় লাগবে, তার হিসাব এই counterfactual-এ নেই। Original Poetiq R5-এর পুরো run প্রায় 11.5 ঘণ্টা নিয়েছিল। তাই 79/172-কে 12-hour feasible hybrid benchmark বলা যাবে না।

কেন hybrid promising: Duck E16-এর Poetiq-missed 10টি correct query আছে। অর্থাৎ Duck পুরোপুরি redundant নয়। কিন্তু Poetiq আগে থেকেই 78টি correct দেয়; তাই equal-time/head-to-head benchmark না করে Duck-কে primary solver বানানোর যুক্তি দুর্বল।

৬. Ranked fixes — আগামী কাজ কী?
পরবর্তী কাজ score gain, implementation complexity এবং Kaggle-এর ২ নভেম্বর deadline মাথায় রেখে সাজিয়েছি। Gain estimate-গুলো hypothesis, measured leaderboard gain নয়।
Rank	Concrete action	Evidence	Potential value	Cheapest decisive test
P0-1	Branch-1 allocation বন্ধ করে Poetiq-style fallback পরীক্ষা	1.847M b1 tokens, 0 new correct	সবচেয়ে বড় recoverable compute pool	12-task equal-wall comparison
P0-2	Poetiq first-pass বনাম Duck stage-1 head-to-head	Poetiq 48.61%, Duck 30%; compute confounded	Primary solver decision	Same engine, same time, EVAL12
P0-3	Long-thinking থেকে executable program-এ পৌঁছানোর নতুন completion path	95 missed queries in no-transform class	High accuracy upside, magnitude unknown	12 no-transform tasks
P1-1	DFlash2 vs target-only/MTP correctness gate	Published concurrency/thinking divergence	Avoid silent wrong programs	12 paired prompts, C1/C12, long context
P1-2	Progress-aware stage-2 scheduling	Median check after 28k; many partial branches	Better useful tokens per GPU-hour	CPU scheduler replay then EVAL12
P1-3	Distinct second-method candidate as attempt_2	Existing selector loses 0; candidate diversity absent	Small but plausible extra pass@2	Mock verified-program portfolio + EVAL12
P2-1	Better object-centric/grid views on hard tasks	Transcripts show spatial misinterpretation	Task-specific; unknown	A/B on identified 12 failures
P2-2	Raw token-prefix/Mamba cache diagnostic	XHIGH E16 raw-content kept 0; cache improvement unproven	Possible serving gain, not primary accuracy lever	CPU token diff, bounded GPU
Later	DSL search, bounded program mutation, third-party ensembles	ARC research supports some value	Higher engineering risk	Separate research branch

P0-1: Branch-1 বাদ দেওয়ার minimal implementation
বর্তমান notebook source code/E14S_notebook_cells.py:335:
E12_BRANCHES = int(os.environ.get("E12_BRANCHES", 2))



আর solver call-এ branches=E12_BRANCHES ইতিমধ্যে আছে (:2305)।

অর্থাৎ CPU-level first experiment-এর জন্য E12_BRANCHES=1 যথেষ্ট। Scheduler rewrite লাগবে না।

তবে বাঁচানো সময় কীভাবে ব্যবহার হবে সেটিই মূল বিষয়। শুধু b1 বন্ধ করলে score স্বয়ংক্রিয়ভাবে বাড়বে না।

Required tests: 120/120 first exposure, zero b1 launches, candidate archive unchanged, submission-এ দুটি valid attempts, deadline/fallback unchanged। এরপর same-wall 12-task GPU run-এ freed compute দিয়ে Poetiq fallback-এর genuinely new correct outputs গণনা।
P0-2: Poetiq-style program generation-কে primary alternative হিসেবে নাও
দুটি candidate protocol:

Arm A — Existing Duck: 28k first exposure, Python inspections, host verification, stage-2 continuation।

Arm B — Single long-think program synthesis: model একই training examples থেকে hypothesis reasoning করে একটি transform(grid) তৈরি করবে; output sandbox-এ verify করবে; failure হলে bounded second iteration।

দুটোতেই same engine, same target checkpoint, same temperature/effort, same wall limit রাখবে। Poetiq iteration budget কমিয়ে 24k বানাবে না; existing 60k/77.8k options-এর মধ্যেই test করবে।

Decision rule: correct queries এবং train-verified programs প্রতি GPU-hour। শুধু generated tokens/sec বা number of tool calls নয়।
P0-3: Model code না লিখলে কী করবে?
বর্তমান E16 nudge শুধু advisory text। 99 branches-এ nudge দেখা গেছে, তবু বহু task-এ কোনো program আসেনি।

আরও একই রকম nudge যোগ করাকে আমি highest priority মনে করি না।

এর বদলে একটি bounded program-commit path পরীক্ষা করো: model দীর্ঘ analysis চালানোর পরে আরেকটি inspection cycle না করে বর্তমান best hypothesis-এর executable candidate লিখবে; incomplete হলে partial candidate হিসেবে check করবে এবং feedback পাবে।

এতে code quality কমতে পারে—LOW run-এর data সেই সতর্কতা দেয়। তাই existing baseline-এর সঙ্গে A/B ছাড়া production-এ নয়।

Safety: Code generation model করবে; host verifier, sandbox এবং test-output selection পুরোনোর মতো fail-closed থাকবে। No hidden ground truth।
P1-1: Engine correctness before major redesign
External issues-তে DFlash2 numerical divergence রয়েছে, কিন্তু আমাদের 604 transcripts-এ তার প্রত্যক্ষ fingerprint নেই।

তাই engine বদলানোর আগে same-input, same-context paired generation test প্রয়োজন। Greedy mismatch, numerical margin, tool-call validity ও ARC code correctness আলাদা মাপবে।

Only if MTP/target-only reliably improves correct program generation, then switching the serving path becomes justified. Otherwise এই সমস্যার মূল agent hypothesis-generation strategy।
P1-2: Selective continuation, not universal 40k
বর্তমান 28k first exposure full coverage-এর জন্য গুরুত্বপূর্ণ।

Potential policy: task-এ genuine program progress থাকলে stage-2 priority; শুধু আরও grid পড়া হচ্ছে এমন branch-এ কম priority; verified but test-ambiguous candidate থাকলে independent Poetiq/DIRECT hypothesis পরীক্ষা।

Regression: 135a2760, 898e7135-এর late-correct candidates যেন prematurely cut না হয়। cbebaa4b-এর মতো low-yield branch bounded থাকবে। Existing clean verification ও fallback preserved।
P1-3: Second attempt-এর বাস্তব ব্যবহার
বর্তমান selector ঠিক কাজ করছে—এটি পরিবর্তন করার প্রয়োজন নেই।

যেখানে Duck ও Poetiq দুটো ভিন্ন train-exact program দেয়, তাদের different test predictions থাকলে attempt_1/attempt_2-তে রাখবে।

Poetiq verified নয় কিন্তু Duck verified হলে Poetiq-এর plausible alternative second attempt হতে পারে, তবে invalid shape বা non-grid output reject করতে হবে।

Test: Same prediction deduplicate; conflicting verified programs দুটোই retain; only one program হলে fallback valid। কোনো candidate-এর correctness নির্ধারণে public evaluation solution ব্যবহার করা যাবে না।

৭. বর্তমান fixes-এর final evidence verdict
Fix / design	Audit verdict
Human-readable facts ও grid inspection helpers	Mechanically work; solve-rate gain isolated নয়
E16 earlier nudge	Mixed: +20 gained, −15 lost; causal benefit unproven
E16 raw-content cache preservation	XHIGH run-এ কার্যকর restoration 0; gain unproven
Host-clean check() harvesting	Works, correct programs selection-এ হারায়নি
Pass@2 selector	Works as intended; candidate diversity নেই
Branch-1 separate grid view	Fails to produce incremental correct outputs in all three runs
Coverage-first scheduler	120-task evaluation-এ coverage achieved
28k pause	Useful for coverage, potentially interrupts hypothesis formation
XHIGH reasoning	LOW-এর চেয়ে noticeably better accuracy; same engine details still matter
Short request cap	Many length cuts; possible major efficiency problem
Cache/prefix optimization	Token efficiency improvement এখনো score-proven নয়
INT4+DFlash2 serving accuracy	Not proven equivalent to FP8/MTP or target-only
Poetiq long-think program synthesis	Highest observed public-eval score among these supplied runs

৮. Final research verdict
GO — Poetiq/Duck method comparison এবং hybrid prototype

Strongest evidence: Duck-এর প্রধান ব্যর্থতা হলো 95টি missed query-সহ no-transform tasks এবং 1.847M branch-1 tokens থেকে zero incremental correct output।

NO-GO — E16-কে already-proven gain ধরে final production-এ promote করা। Observed +5 queries আছে, কিন্তু gains/losses বড় এবং statistical uncertainty zero improvement-এর সঙ্গে সামঞ্জস্যপূর্ণ।

NO-GO — শুধু আরও engine TPS tuning করে 22.78→60+ আশা করা। Faster engine সাহায্য করতে পারে; কিন্তু Python hypothesis quality না বদলালে 95টি no-transform misses থেকে যথেষ্ট উদ্ধার পাওয়া কঠিন।

আমার recommendation: আগের best 34.58 submission অক্ষত রেখে, প্রথমে E12_BRANCHES=1 + Poetiq-style fallback-এর same-wall ছোট পরীক্ষা। তারপর evidence দিয়ে primary/hybrid method বেছে নাও। এখনই notebook rewrite, new model বা broader kernel patch করব না।
External research links
https://github.com/poetiq-ai/poetiq-arc-agi-solver

https://github.com/Tufalabs/duck-harness

https://tufalabs.ai/research/duck-harness/

https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md

https://huggingface.co/Qwen/Qwen3.8-27B

https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4

https://github.com/sgl-project/sglang/issues/36548

https://github.com/sgl-project/sglang/issues/38009

https://github.com/sgl-project/sglang/issues/43956

https://github.com/QwenLM/Qwen3.8/issues/216

https://arxiv.org/abs/2609.27242

https://arxiv.org/abs/2607.09059

Source limitation: Poetiq reference ledger ও analysis bundle-এ রয়েছে, কিন্তু Poetiq-এর সব 172 original model-thinking/program artifacts এই ZIP-এ নেই। ফলে Poetiq score ও per-task complementarity independently যাচাই করা গেছে; Poetiq-এর প্রতিটি individual reasoning trace-এর causal explanation নয়।
