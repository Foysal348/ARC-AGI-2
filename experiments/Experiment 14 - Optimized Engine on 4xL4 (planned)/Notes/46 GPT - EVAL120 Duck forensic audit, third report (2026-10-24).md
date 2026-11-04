<!-- Pasted by the user on 2026-10-24: a third GPT deep-research report on EVAL120_DUCK_RUNS_DEEP_RESEARCH_FOR_GPT_2026-10-24.zip. Kept VERBATIM. Merged into Notes/43 section 8. -->

ARC-AGI-2 — EVAL120 Duck Runs: Forensic Audit ও Deep Research
২৪ অক্টোবর ২০২৬ · তিনটি full EVAL120 run · 120 tasks / 172 test queries প্রতি run
১. সবচেয়ে গুরুত্বপূর্ণ আবিষ্কার: সমস্যার কেন্দ্রবিন্দু GPU speed নয়, executable solution তৈরি করতে না পারা
তিনটি Duck run-এর 604টি branch record ও transcript এবং সংশ্লিষ্ট evaluation records থেকে task-level failure taxonomy পুনর্গণনা করেছি।
ফলাফল	XHIGH + E16	XHIGH (submitted)	LOW + E16
সঠিক test queries	46/172	41/172	31/172
Task-normalized score	30.00%	25.83%	18.89%
সব test input সঠিক এমন task	35	30	20
আংশিক সঠিক task	2	2	5
কোনো executable transform শনাক্ত হয়নি	64	67	57
Transform লিখেছে, train-exact হয়নি	9	7	22
Train-exact পাওয়া গেছে, কিন্তু test-এ সব ভুল	10	14	16
সম্পূর্ণ ভুল task	83	88	95

শ্রেণিগুলো task-প্রতি exclusive। “Transform শনাক্ত হয়নি” মানে adapter-এর first_transform_tokens event ও retained-check evidence অনুযায়ী শনাক্ত হয়নি; transcript-এ যেকোনো Python expression না-থাকা বোঝায় না। আংশিক সঠিক task-এও একটি train-exact program থাকতে পারে।

XHIGH + E16

XHIGH

LOW + E16


020406080No transformNo train-exactTrain-exact, test wrong














এটাই সবচেয়ে গুরুত্বপূর্ণ finding: Submitted XHIGH Duck-এর 120টির মধ্যে 67টি task-এ কোনো retained executable transform পর্যন্ত পৌঁছানো যায়নি। অর্থাৎ 55.8% task-এ generalization যাচাই করার পর্যায়ই আসেনি।

এটি শুধু “train-exact program overfit করছে” সমস্যা নয়। তার আগের ধাপেই বিপুল compute খরচ হচ্ছে—grid inspection, hypothesis analysis, reasoning এবং tool calls চললেও executable candidate তৈরি হচ্ছে না।

XHIGH-এর প্রথম transform তৈরি হওয়া 53টি task-এর মধ্যে:
- 7টিতে প্রথম transform এসেছে 12k generated tokens-এর মধ্যে।
- 20টিতে 12k–28k-এর মধ্যে।
- 26টিতে 28k-এর পরে।
অর্থাৎ প্রথম useful code লিখতে model অনেক দেরি করছে। তবে দেরিতে পাওয়া correct programs-এর গুরুত্বও আছে; তাই চিন্তার budget হঠাৎ 12k বা 28k-তে কেটে দেওয়া সমাধান নয়।
২. সবচেয়ে ব্যয়বহুল failure: reasoning চলেছে, কিন্তু code আসেনি
Submitted XHIGH run-এ:
No-transform task-এ ব্যয়
4.61M
7.09M generated tokens-এর প্রায় 65%

Branch 1: tool call ছাড়াই ব্যয়
1.28M
55টি branch; 79টির মধ্যে প্রায় 70%


এই দ্বিতীয় সংখ্যাটি বিশেষভাবে গুরুত্বপূর্ণ। XHIGH-এর 79টি second branch-এর মধ্যে 55টিতে একটি Python tool call-ও নেই। অথচ LOW+E16-এর 83টি second branch-এ অন্তত একটি tool call হয়েছে।
Branch-1 metric	XHIGH+E16	XHIGH	LOW+E16
Started	82	79	83
কোনো Python tool call নেই	60	55	0
Train-exact branch	1	0	0
Generated tokens	1.85M	1.73M	1.46M
Tool-call-free tokens	1.39M	1.28M	0

Branch 1 বর্তমানে প্রায় সম্পূর্ণ ব্যর্থ একটি compute investment। XHIGH submitted run-এ প্রায় 1.73 million tokens দ্বিতীয় branch-এ খরচ করেও নতুন train-exact program পাওয়া যায়নি।

এখানে একটি গুরুত্বপূর্ণ ব্যাখ্যা: LOW বেশি দ্রুত Python call করেছে, কিন্তু তার overall score কম। তাই কেবল দ্রুত tool call করানো যথেষ্ট নয়; দ্রুত candidate তৈরি এবং গভীর XHIGH reasoning—দুটোকে একসঙ্গে রাখতে হবে।
Transcript-এর সরাসরি প্রমাণ
runs/XHIGH_noE16/Q27X_E12_DUCK_BUNDLE.zip
- duck/136b0064_b1/transcript.txt: দীর্ঘ reasoning-এ character/shape নিয়ে চিন্তা চলে, কিন্তু প্রথম 24,576-token response শেষ হওয়ার আগেও executable transform আসে না।
- duck/0934a4d8_b1/transcript.txt: Python exploration হয়েছে, কিন্তু NameError: name 'f1' is not defined—আগের function/context state সম্পর্কে ভুল assumption।
- duck/135a2760_b0/transcript.txt: model submit-কে আলাদা tool হিসেবে ডাকতে চেষ্টা করেছে, ফল Unknown tool: submit। বাস্তবে exposed tool শুধু python।
এই শেষ সমস্যাটি শুধু isolated নয়:
Transcript issue	XHIGH+E16	XHIGH	LOW+E16
Unknown tool: submit পাওয়া branch	22	27	23
NameError পাওয়া branch	19	21	23
Traceback পাওয়া branch	53	59	69
Adapter-reported infrastructure errors	2	4	3

এগুলো overlapping counts। সব NameError বা Unknown tool score নষ্ট করেছে—এমন causal proof নেই। কিন্তু tool-interface confusion ও context-retention ambiguity বাস্তব।

প্রাথমিক diagnosis: XHIGH branch-1-এর long internal reasoning এবং tool-interface confusion এই harness-এর দুটি বড় design weakness। এগুলোকে শুধু GPU kernel বা INT4 quality দিয়ে ব্যাখ্যা করা যায় না।
৩. Agent, model, নাকি engine—কোনটি মূল দায়ী?
সম্ভাব্য কারণ	প্রমাণ	রায়
Agent output discipline	67/120 task-এ transform শনাক্ত হয়নি; 55টি branch-1 tool-free	প্রধান demonstrated bottleneck
28k pause ও stage-2 allocation	প্রথম transform প্রায়ই 28k-এর পরে; second branch-এ বিপুল ব্যয়	গুরুত্বপূর্ণ scheduling/design issue
Tool interface ও retained state	Unknown tool: submit, NameError, context-dependent functions	বাস্তব robustness issue
Train-exact কিন্তু test-wrong	Submitted XHIGH-এ 14 task	দ্বিতীয় স্তরের generalization issue
INT4 quantization quality loss	FP8 বনাম INT4-এর matched same-harness test নেই	প্রমাণ অপর্যাপ্ত
DFlash2/GDN numerical divergence	Relevant public bugs রয়েছে	সম্ভাব্য risk; এই ZIP-এ root cause প্রমাণিত নয়
GPU throughput	Duck-এ 7.09M tokens তৈরি হয়েছে	একক প্রধান কারণ নয়
EOS mid-thinking	এই records দিয়ে নির্ভরযোগ্য per-task confirmation নেই	প্রমাণিত নয়

কেন Poetiq বেশি solve করেছে?
Poetiq R5-এর saved summary ও ledger পুনর্গণনায় 78/172 queries, 48.61% task-normalized পাওয়া যায়। Submitted Duck-এর 41/172, 25.83%-এর তুলনায় অনেক বেশি।

কিন্তু দুটির budget-ও আলাদা:
	Submitted Duck XHIGH	Poetiq R5
Generated tokens	7.09M	11.65M
Solver window	প্রায় 350 মিনিট	691 মিনিট
Train-exact task	46	71
Correct queries	41	78
Correct queries / million tokens	5.78	6.70
Correct queries / solver minute	0.117	0.113

অর্থাৎ Poetiq মোট score-এ অনেক এগিয়েছে, কিন্তু প্রতি solver minute-এ স্পষ্টভাবে এগিয়েছে—এমন নয়। তার প্রায় দ্বিগুণ সময় এবং 64% বেশি generation budget ছিল।

তবে Poetiq-এর design-এর একটি সুবিধা যথেষ্ট স্পষ্ট: প্রতিটি iteration-এর উদ্দেশ্য একটি executable transform(grid) তৈরি করা। Duck-এ বহু Python inspection-এর পরেও function লেখা হচ্ছে না।

Train-exact program পাওয়া গেলে generalization-এর ক্ষেত্রেও দুটি method পুরোপুরি আলাদা নয়:
- Duck XHIGH: 46 train-exact task-এর মধ্যে 30 সম্পূর্ণ সঠিক, 2 আংশিক, 14 ভুল।
- Poetiq R5: 71 train-exact task-এর মধ্যে 49 সম্পূর্ণ সঠিক, 7 আংশিক, 15 ভুল।
অতএব আগে আরও বেশি valid candidate তৈরি করা প্রয়োজন; শুধু train-exact হওয়ার পরে selection উন্নত করলে 67টি no-transform task উদ্ধার হবে না।
Public research কী বলছে?
Qwen3.8-এর model card-এ xhigh, medium, low reasoning effort-এর পার্থক্য documented। Qwen নিজেই সতর্ক করেছে যে low effort একেকটি response দ্রুত দিলেও multi-turn agent-এ retry বেড়ে মোট কাজ খারাপ হতে পারে। আপনার LOW+E16 ফলের সঙ্গে এটি সামঞ্জস্যপূর্ণ। 

Hugging Face




Tufa Labs-এর ১ জুলাই ২০২৬ প্রকাশিত Duck harness মূলত ARC-AGI-3-এর interactive game agent। সেখানে REPL ব্যবহার করে game state দেখা ও action নেওয়া হয়। ARC-AGI-2-তে একই mechanism দিয়ে একবারে একটি সম্পূর্ণ grid-transformation program আবিষ্কার করতে হয়। এই task mismatch একটি শক্তিশালী design hypothesis—মূল Duck donor-এর সাফল্য ARC2-তে স্বয়ংক্রিয়ভাবে স্থানান্তরিত হয় না। 

Tufa Labs
+1




SGLang-এর সেপ্টেম্বরের #38009-এ Qwen3.8+DFlash2 thinking-mode greedy divergence এবং আগস্টের #36548-এ concurrent-state corruption রিপোর্ট করা হয়েছে। এগুলো numerical quality test করার যুক্তি দেয়, কিন্তু আপনার বর্তমান 67টি no-transform task-এর কারণ DFlash2—এমন প্রমাণ নয়। 

GitHub
+1




সুতরাং বর্তমান evidence-এ agent design-ই প্রথমে সংশোধনের জায়গা; INT4 বা DFlash2 পরিবর্তন পরে controlled comparison।
৪. E16 early nudge — উন্নতি, নাকি run-to-run variability?
Same task order এবং একই seed-এ comparison
দুটি XHIGH run-এর task order identical এবং seed 20261005। তাই comparison মূল্যবান। তবে stochastic decoding ও GPU scheduling-এর কারণে এগুলো identical execution নয়।
+5
Net correct queries

20
E16-only correct queries

15
Baseline-only correct queries


Task-level-এ E16 জিতেছে 16টি, হেরেছে 12টি, অপরিবর্তিত 92টি। একটি simple paired sign test-এ \(p\approx0.57\)। Query-level discordance ধরে \(p\approx0.50\)। এগুলো improvement statistically নিশ্চিত করে না; query-গুলো একই task-এর ভেতরে correlated হওয়ায় task-level comparison বেশি উপযোগী।
E16-এর পৃথক উপাদান
E16 mechanism	Recorded evidence	রায়
Early inspection-based nudge	XHIGH+E16-এ 90 branch, LOW+E16-এ 168 branch	সক্রিয় হয়েছে
আগের 24k fallback nudge	XHIGH+E16-এ 21 branch	সক্রিয়
Actual subsequent request-এ nudge উপস্থিত	XHIGH+E16-এ 99 branch; baseline-এ 78	Delivery বেড়েছে
Raw-content whitespace preservation	XHIGH+E16-এ 0 kept	কার্যকর optimization হয়নি
Train-exact task	46 → 47	+1, অনিশ্চিত
Task-normalized score	25.83% → 30.00%	+4.17 percentage points, অনিশ্চিত

আরেকটি অস্বস্তিকর তথ্য: প্রথম transform-এর observed median XHIGH baseline-এ প্রায় 28.5k, E16-এ প্রায় 30.7k tokens। অর্থাৎ early nudge প্রথম code লেখার সময় নিশ্চিতভাবে কমিয়েছে—এমন প্রমাণ নেই।

E16-এর নতুন 20টি correct query-এর মধ্যে 136b0064, 20270e3b, 36a08778, 409aa875, cb2d8a2c আছে। বিপরীতে 1ae2feb7, 269e22fb, b5ca7ac4, db0c5428-এর মতো task-এ baseline ভালো করেছে।

Verdict: E16 = provisional positive; production improvement proven নয়। Raw-content component-কে gain-এর কারণ বলা যাবে না। Early nudge আরও একবার matched EVAL12/EVAL24 comparison পাওয়ার যোগ্য, কিন্তু একে বড় leaderboard jump-এর সমাধান ধরে নেওয়া ঠিক হবে না।
৫. Attempt 2 প্রায় শূন্য কেন এবং Poetiq hybrid-এর আসল ceiling
Attempt 2 সমস্যা
Submitted XHIGH-এর result:
- 79টি branch 1 শুরু।
- Branch 1 থেকে 0টি train-exact program।
- সমস্ত branch মিলিয়ে মাত্র 52টি retained train-exact candidate।
- মাত্র 6টি test query-র জন্য একাধিক distinct train-exact prediction ছিল।
- attempt_2 থেকে 0টি নতুন correct query।
এটি মূলত selection-ranking bug নয়। দ্বিতীয় slot-এ দেওয়ার মতো নতুন, যথাযথ candidate-ই তৈরি হয়নি।

আরও ভালো second attempt হতে পারে সম্পূর্ণ ভিন্ন reasoning pathway—যেমন Poetiq-style একবারে transform লেখা, independent spatial representation, কিংবা previous failure feedback-ভিত্তিক fresh hypothesis। কিন্তু দ্বিতীয় xhigh Duck branch আগের মতো পুনরায় চালালে লাভ পাওয়ার প্রমাণ নেই।
Duck বনাম Poetiq — per-query complementarity
reference/poetiq_r5_eval120/ledger/ledger_units.csv-এর Poetiq outcome এবং তিনটি Duck run-এর 172টি evaluation records exact unit ID দিয়ে join করেছি।
Combination	Correct queries (oracle union)	Task-normalized
Poetiq R5 একা	78/172	48.61%
Poetiq + submitted Duck XHIGH	85/172	53.61%
Poetiq + XHIGH E16	88/172	54.44%
Poetiq + LOW E16	82/172	51.11%

এগুলো oracle unions—যে কোনো run-এর যেকোনো সঠিক output থাকলে সেটি গণনা করা। তাই এগুলো বাস্তবে gold-free selection, শুধু দুইটি allowed attempts, অথবা একই 12-hour GPU budget দিয়ে অর্জনযোগ্য score নয়।
বিশেষ প্রশ্ন: Duck stage 1 + Poetiq হলে?
প্রথম stage-এর শেষে বাস্তবে পাওয়া train-exact candidates-গুলো আলাদা করে পুনর্গণনা করেছি:
Combination	Stage-1 Duck correct queries	Poetiqসহ oracle union	Task-normalized
Poetiq + XHIGH stage 1	18	79/172	49.44%
Poetiq + XHIGH E16 stage 1	20	83/172	51.11%
Poetiq + LOW E16 stage 1	19	80/172	50.28%

এখানে stage 1 বলতে প্রথম pause/first-exposure run-এ completed verification বোঝানো হয়েছে—response overshoot থাকলে 28k-এর পরে হলেও একই first exposure-এর অংশ।

এটি hybrid design-এর জন্য বড় সিদ্ধান্ত: Poetiq-এর আগে পুরো Duck stage 1 চালিয়ে লাভ খুব সীমিত, বিশেষ করে submitted XHIGH-এর ক্ষেত্রে মাত্র +1 correct query-এর oracle opportunity। অথচ এতে প্রচুর GPU সময় খরচ হয়।

Poetiq একা যেখানে 78টি query সঠিক করেছে, submitted Duck-এর 41টি correct query-এর মধ্যে 34টি Poetiq-ও সঠিক করেছে; Duck-only মাত্র 7টি। তাই hybrid-এর মূল উদ্দেশ্য হওয়া উচিত Poetiq-এর ব্যর্থ বা অনিশ্চিত task-এ সত্যিকার নতুন hypothesis তৈরি করা, সবার ওপর একই Duck stage চালানো নয়।

Poetiq-এর public repository-তে iterative executable program ও verification-based refinement-এর design পাওয়া যায়। এটি দ্বিতীয় পদ্ধতি হিসেবে গবেষণার উপযোগী; তবে আমাদের এখানে পাওয়া score-গুলো আপনার নিজস্ব runs থেকে, Poetiq authors-এর official leaderboard থেকে নয়। 

GitHub



৬. চূড়ান্ত root causes, fixes, validation tests ও পরের কাজ
৬.১ Failure taxonomy — কত test query হারাচ্ছি?
Submitted XHIGH-এর 131টি missed query-র exclusive breakdown:
Root-cause category	Missed queries	উদাহরণ
No retained executable transform	102	136b0064, 13e47133, 20270e3b
Transform তৈরি, train-exact নয়	8	0934a4d8, 53fb4810, 62593bfd
Train-exact, কিন্তু test-wrong	19	221dfab4, 78332cb0, dfadab01
আংশিক সঠিক task-এর অবশিষ্ট ভুল query	2	247ef758, b0039139
মোট	131	172 − 41

এই breakdown থেকে একটি জিনিস নিশ্চিত: প্রথম সমাধানের সুযোগ হলো 102টি no-transform misses; কেবল second-attempt ranking উন্নত করলে এগুলো সমাধান হবে না।

তবে এই 102টি query-এর সবগুলো উদ্ধারযোগ্য নয়। একই 67টি no-transform task-এ Poetiq 30টি query সঠিক করেছে—অর্থাৎ অন্তত ওই 30টি আপনার ব্যবহৃত model family দিয়ে অন্য পদ্ধতিতে সমাধানের recorded evidence আছে।
৬.২ Train-exact ভুলগুলোর প্রকৃতি
branch_result.json → checked[].test_outputs এবং evaluation ground truth তুলনায় কয়েকটি concrete failure pattern পেয়েছি:
- 78332cb0: Output orientation ভুল—একটি query-তে 23×5 বনাম প্রয়োজনীয় 5×23, অন্যটিতে উল্টো। এটি shape/rotation hypothesis failure।
- 247ef758 query 1: একই output shape, মাত্র একটি cell ভুল। এটি high-value near miss; কিন্তু train-exact variant থাকলেই gold-free পদ্ধতিতে ভুল cell শনাক্ত করা সহজ নয়।
- 221dfab4: বহু cell ভুল—এটি simple one-cell repair-এর সমস্যা নয়।
- b0039139 query 1: prediction-এর width 24, expected 29। Output-size rule generalize করেনি।
অর্থাৎ সব train-exact failure এক ধরনের overfitting নয়। কোনোটি shape-rule ভুল, কোনোটি orientation, কোনোটি সূক্ষ্ম recoloring, আবার কোনোটি সম্পূর্ণ ভুল abstraction।

Fix: Train-exact candidate-এর জন্য একটি soft sanity score যোগ করা যেতে পারে—shape-rule consistency, input/output palette relation, object counts এবং স্বাধীন দ্বিতীয় program-এর agreement। কিন্তু এগুলোর কোনো একটিকে universal hard rejection rule করবেন না। ARC-তে unusual transformations-ই অনেক সময় সঠিক।

৬.৩ Ranked action plan
এখানে expected gain-কে তিনভাবে বোঝাচ্ছি: measured, oracle opportunity, অথবা unmeasured hypothesis। Hypothesis-কে নিশ্চিত leaderboard improvement বলছি না।
Rank	সমস্যা	প্রস্তাবিত fix	Gain evidence	সবচেয়ে সস্তা test	Risk
P0-1	67 task-এ code নেই	Poetiq-style program-first Duck prompt, সর্বোচ্চ দুটি exploratory call-এর পরে provisional transform check করার স্পষ্ট নির্দেশ	102 missed queries-এর failure class; gain অমাপা	EVAL12-এ 12টি no-transform task, same tokens/time	Medium
P0-2	Branch 1-এ 1.28M tool-free tokens	একই Duck branch পুনরায় না চালিয়ে একটি genuinely different program-synthesis second branch পরীক্ষা	79 branch থেকে 0 train-exact; gain অমাপা	12–24 failed task, same token budget	Medium
P0-3	Tool schema confusion	check() ও submit() যে কেবল python tool-এর ভিতরের functions, তা explicit করে prompt সংশোধন	27টি branch-এ Unknown tool: submit	Saved malformed tool-call regression + EVAL12	Low
P1-1	28k pause usefulness অনিশ্চিত	First-exposure length-এর controlled A/B; শুধু promising branches-এ long continuation	53টি transform task-এর 26টিতে first transform 28k-এর পরে	EVAL12 matched tasks	Medium
P1-2	E16 +5 query net, 15 loss	Early nudge একই workload-এ repeat এবং task-clustered comparison	এক run-এ +4.17 pp, significance নেই	Same 12/24 task, matched seeds	Low
P1-3	Train-exact wrong-test	Structural sanity + candidate diversity tie-break	Submitted XHIGH-এ 19 missed queries	14 train-exact/test-wrong task-এর offline retrospective evaluation	Medium
P1-4	Model বনাম engine confounded	Same prompts, same Qwen weights, target-only/INT4/DFlash comparison	External bug reports আছে; local quality cause অজানা	12 selected tasks, short correctness preflight	GPU cost
P2	Context/state confusion	Retained-functions inventory, fresh Python-call contract, prompt re-render diagnosis	NameError/traceback evidence	CPU replay + EVAL12	Low
P2	Selection rule	Distinct candidate outputs সংরক্ষণ ও deterministic ordering	Base-এ মাত্র 6 units-এ 2+ distinct predictions	Offline generated-attempts test	Low
P3	Larger architecture	Poetiq + Duck mixed allocation	Full-run oracle union 53.61–54.44%, কিন্তু GPU budget দ্বিগুণের বেশি	Offline scheduler simulation, পরে bounded GPU	High

Fix 1 — Prompt-এ executable contract আরও স্পষ্ট করুন
code/arc2_duck.py, arc2_r6_system_prompt()-এর Tool session rules অংশে এই নির্দেশ পরীক্ষামূলকভাবে যুক্ত করা যায়:
"- There is exactly one external tool: `python`. ""`check(transform)` and `submit(transform)` are Python ""functions inside it, NOT separate tools. Never emit ""`<function=check>` or `<function=submit>`.\n""- After no more than two inspection-only Python calls, ""implement your best current provisional transform(grid) ""and call check(transform). It is acceptable to fail a ""training pair; use the returned mismatch to revise it. ""Continue deep reasoning after obtaining executable feedback.\n"



এটি একটি experimental patch, validated fix নয়। কঠিন task-এ provisional implementation অতিরিক্ত সময় খেতে পারে।

Validation: আগের no-transform tasks-এ first-check latency, number of train-exact programs, actual correct queries এবং generated tokens সব মাপতে হবে। শুধু tool-call count বাড়লে promotion দেবেন না।
Fix 2 — Branch 1-কে সত্যিকারের alternative method বানান
এখন branch 1-এ view আলাদা হলেও একই XHIGH multi-turn tool-agent pattern চলছে। সেটি 79 বার চেষ্টা করে শূন্য verified program দিয়েছে।

প্রস্তাবিত A/B:
- Control: বর্তমান branch-1 Duck।
- Candidate: একবারে executable transform লেখার জন্য Poetiq-style long-thinking request, তারপর train verification; failure feedback থাকলে bounded দ্বিতীয় iteration।
Poetiq-এর 77,824 thinking-token ceiling অকারণে কমাবেন না। কাজটি করতে একই GPU-token budget-এর মধ্যে কত request চালানো যায়, সেটি মেপে তুলনা করতে হবে।

Promotion gate হিসেবে অন্তত নতুন train-exact tasks, নতুন correct test queries, এবং tokens per new solve—তিনটি metric ব্যবহার করুন। Gold শুধু offline EVAL12 analysis-এ থাকবে, runtime selection-এ নয়।
Fix 3 — submit confusion regression
Prompt-only fix-এর পাশাপাশি একটি test যোগ করুন:
def test_submit_is_not_a_separate_tool():    tools = build_arc2_tools(...)    names = {t["function"]["name"] for t in tools}    assert names == {"python"}    prompt = build_arc2_prompt(...)    assert "NOT separate tools" in prompt    assert "submit(transform)" in prompt



এটি schema/prompt contract যাচাই করে; model আর ভুল করবে না—তার guarantee নয়। সেই জন্য EVAL12-তে Unknown tool: submit count-ও মাপতে হবে।
Fix 4 — Failure reporting-এ measured effort সঠিক রাখুন
LOW+E16 run-এর top-level configuration.REASONING_EFFORT এখনো xhigh দেখায়। কিন্তু effective Duck override হলো:
configuration.e12.E12_REASONING_EFFORT = "low"


Adapter code/arc2_duck_e16.py-এ request kwargs-এ সেটি inject করে।

Reporting-এ top-level field দেখে LOW run-কে XHIGH বলে ভুল করবেন না। Effective request effort, source এবং default-versus-override status আলাদা field-এ record করুন।

৭. এখন কোন পরীক্ষা করলে সবচেয়ে বেশি শেখা যাবে?
আমি নতুন 120-task full run এখনই সুপারিশ করছি না। আপনার কাছে ইতোমধ্যে তিনটি 120-task run-এর যথেষ্ট diagnostic evidence আছে।

পরের experiment-এর উদ্দেশ্য শুধু score বাড়ানো নয়—67টি no-transform task-এর কতগুলোকে executable candidate stage-এ আনা যায় তা নির্ধারণ।

একটি matched EVAL12/EVAL24 suite নিন, যেখানে থাকবে no-transform, transform-but-not-exact, train-exact-but-test-wrong এবং কয়েকটি already-solved control task। Existing fixed seed ও order বজায় রাখুন। নতুন prompt যেন পুরোনো solved task নষ্ট না করে, সেটিও দেখতে হবে।

প্রথম experiment-এ engine, INT4, DFlash width, KV ও SSM একদম অপরিবর্তিত রাখুন। তাহলে পরিবর্তনের কারণ সত্যিই agent prompt/branch design কি না বোঝা যাবে।

GO gate হবে শুধু দ্রুত first-check নয়; correct queries, train-exact yield, token efficiency এবং solved-task regression—সব মিলিয়ে উন্নতি।
৮. Public research—সরাসরি প্রাসঙ্গিক উৎস
উৎস	তারিখ / লেখক	প্রমাণ	আপনার জন্য অর্থ
Tufa Labs Duck Harness	১ জুলাই ২০২৬; Bessis et al.	ARC-AGI-3 REPL architecture, author report	ARC3-to-ARC2 adaptation পুনর্বিবেচনা
Poetiq Solver	Public repository	Iterative program-synthesis implementation	Alternative branch/harness design
ARC Prize Qwen3.8 result	১৪ আগস্ট ২০২৬	XHigh 42.4%, Low 22.8% semi-private result	XHIGH-quality reasoning অক্ষত রাখার যুক্তি; আপনার EVAL120 score-এর সরাসরি comparator নয়
Official model card	২০২৬, Qwen team	Reasoning-effort and thinking-history documentation	LOW faster ≠ better agent completion
SGLang #38009	সেপ্টেম্বর ২০২৬; forlayo	Greedy DFlash thinking divergence reproduction	Numerical correctness control প্রয়োজন
SGLang #36548	আগস্ট ২০২৬; L3tum	Concurrent-state inconsistency report	12-stream parity verification
SGLang #40959	২৩ সেপ্টেম্বর ২০২৬	Qwen3.8 reasoning-history rendering problem, Anthropic endpoint	History representation audit-এর প্রাসঙ্গিক lead, Duck-এর exact API path-এর প্রমাণ নয়
Qwen #236	৭ সেপ্টেম্বর ২০২৬	Long-prompt non-thinking tool-call failure report	Thinking সম্পূর্ণ বন্ধ করার আগে সতর্কতা



Tufa Labs
+7


চূড়ান্ত রায়
Engine serving
Keep baseline

এই dataset-এ engine speed-ই প্রধান ব্যর্থতার কারণ—এমন প্রমাণ নেই।
Duck branch 1
NO-GO unchanged

Submitted run-এ 1.73M generated tokens ব্যয় করে শূন্য নতুন train-exact program।
E16 early nudge
Inconclusive

+5 queries net, কিন্তু 20 gains-এর বিপরীতে 15 losses; পুনরাবৃত্তি প্রয়োজন।
Poetiq-style program-first experiment
GO for offline development

সবচেয়ে গুরুত্বপূর্ণ root cause—executable candidate না তৈরি হওয়া—সরাসরি পরীক্ষা করবে।
আমার গবেষণা-ভিত্তিক সিদ্ধান্ত: এই মুহূর্তে শুধু decoding speed আরও 10–20% বাড়ানোর চেয়ে Duck-এর program-production rate উন্নত করা বেশি গুরুত্বপূর্ণ। তবে Poetiq-এর বেশি total score-এর একটি বড় অংশ বেশি runtime থেকেও এসেছে—এটি ভুলে যাওয়া যাবে না।

প্রথমে code-first prompt এবং alternative branch-1 design-এর একটি ছোট controlled test করুন। তার ফল না দেখে আরেকটি 6.5-hour Duck run বা বড় engine migration-এ GPU quota খরচ করা যুক্তিযুক্ত নয়।

কোনো source code পরিবর্তন, GPU run, Kaggle push বা commit করিনি।
