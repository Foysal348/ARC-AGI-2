""

<pasted_content id="1891">
Q27 DIRECT বনাম AGENT — Deep Technical Audit
১১ অক্টোবর ২০২৬ | ARC Prize 2026 – ARC-AGI-2 | Qwen3.8-27B-FP8
ZIP-এর README এবং ফাইল-ইনভেন্টরি পড়েছি। অডিটে তিনটি বিষয় আলাদা রাখছি: আপনাদের বাস্তবে মাপা ফলাফল, কোড থেকে নিশ্চিত হওয়া সমস্যা, এবং বাইরের গবেষণা থেকে পাওয়া সম্ভাব্য উন্নতির পথ।
১. প্রাথমিক সিদ্ধান্ত: কোন পথে এগোনো উচিত?
আমার অগ্রাধিকার হলো DIRECT-কে baseline রেখে একটি verified, budget-aware HYBRID solver তৈরি করা। তবে AGENT-এর সত্যিকারের GPU performance মাপার আগে DIRECT বাদ দেওয়া ঠিক হবে না।
DIRECT public LB
18.75%
জমা দেওয়া v1.4
DIRECT EVAL12
3/22
v1.4.4, 13.64%
AGENT GPU score
অজানা
v1.4 রান ব্যর্থ


গুরুত্বপূর্ণ পর্যবেক্ষণ: AGENT-এর প্রথম GPU রান reasoning-quality পরীক্ষাই করতে পারেনি। HTTP 400-এর কারণে কার্যত কোনো agent inference হয়নি। ফলে AGENT ভালো বা খারাপ—এখনই কোনোটির প্রমাণ নেই।

DIRECT-এর দুটি EVAL12 লগে ৪৯ হাজার thinking-token সীমার কাছে পৌঁছে জোরপূর্বক উত্তর বের করার ঘটনা যথাক্রমে ১৮/২২ এবং ১৯/২২। এটি বাস্তব bottleneck। কিন্তু শুধু token budget বাড়ানোই সমাধান নয়; সময়ের মধ্যে কতগুলো স্বতন্ত্র hypothesis তৈরি, পরীক্ষা ও সংশোধন করা যায়, সেটাই বেশি গুরুত্বপূর্ণ হতে পারে।
৬০% লক্ষ্যকে অসম্ভব বলার ভিত্তি নেই; আবার বর্তমান Qwen3.8-27B implementation দিয়ে সেখানে পৌঁছানো যাবে—এমন প্রত্যক্ষ প্রমাণও নেই। লক্ষ্য পূরণ করতে হলে কার্যকরী accuracy ও ১২ ঘণ্টার coverage দুটোই একসঙ্গে বাড়াতে হবে।
২. DIRECT-এর বাস্তব ফল: গুরুত্বপূর্ণ সংশোধন
দুটি records.jsonl-এর ৪৪টি query record পুনরায় গণনা করেছি।
ফলাফল	DIRECT v1.4, Run 4	DIRECT v1.4.4, Run 7
মোট test outputs	22	22
সঠিক উত্তর	2	3
Natural completion	4	3
Natural completion-এ সঠিক	2	2
Forced completion	18	19
Forced completion-এ সঠিক	0	1
মোট throughput	248.9 tok/s	274.6 tok/s
Prefix-cache hit	—	57.2%
GPU preemption	0	0
সিদ্ধান্ত: Forced completion একেবারে অকেজো নয়—Run 7-এ ১টি সঠিক উত্তর দিয়েছে। তবে দুই রান মিলিয়ে ৩৭টি forced completion-এর মাত্র ১টি সঠিক।
এর অর্থ token limit একটি শক্তিশালী সন্দেহভাজন bottleneck। কিন্তু সরাসরি ৪৯k থেকে ৮০k বা ১০০k করলেই accuracy বাড়বে, সেটা এই ফলাফল প্রমাণ করে না। হয়তো কঠিন task-গুলিই বেশি token নেয়। অর্থাৎ এখানে selection bias আছে।
আরেকটি বিষয়: v1.4 থেকে v1.4.4-এ throughput প্রায় ১০.৩% বেড়েছে। কিন্তু এটির পুরোটাই prefix caching-এর প্রভাব বলা যাবে না, কারণ রান দুটির startup, configuration ও scheduling-এ আরও পার্থক্য রয়েছে।
vLLM-এর নিজস্ব documentation-এও স্পষ্ট: “does not reduce the time of generating new tokens”—অর্থাৎ prefix caching prefill বাঁচায়, দীর্ঘ decoding-এর মূল খরচ নয়। 

vLLM



৩. গবেষণা থেকে দুটি বিশেষ গুরুত্বপূর্ণ আবিষ্কার
A. Tufa Labs-এর আসল Duck Harness পাওয়া গেছে
README-তে এই harness-এর public link পাওয়া যায়নি বলা হয়েছিল। এখন আসল repository পাওয়া গেছে:
GitHub — Tufalabs/duck-harnessএর ভেতরে ARC3-Inference/, framework, example runs এবং Kaggle notebook আছে। Tufa Labs-এর ব্যাখ্যায় REPL, Python variables, automatic context eviction এবং tool-driven reasoning ব্যবহারের প্রমাণ পাওয়া যায়। 

GitHub
+1



সীমাবদ্ধতা: এটি ARC-AGI-3 game-playing harness; Qwen3.8-27B দিয়ে ARC-AGI-2-তে ৬০% পাওয়ার প্রমাণ নয়। কিন্তু আপনাদের AGENT-এর REPL, context, retained state এবং tool-loop design তুলনার জন্য এটি সরাসরি প্রাসঙ্গিক।
B. একই মডেলে DIRECT থেকে agentic reasoning-এর উল্লেখযোগ্য উন্নতির প্রকাশিত প্রমাণ আছে
Hugging Face — Agentic Coding Publication, full tracesএখানে GPT-OSS-120B-High মডেলের ARC-AGI-2 public evaluation-এ:
Plain reasoning6.11%


Agentic coding26.39%



একই গবেষণার প্রকাশিত ফল; পার্থক্য +২০.২৮ percentage points। গবেষকেরা সম্পূর্ণ prompts, model responses এবং tool calls-ও প্রকাশ করেছেন। 

Datasets at Hugging Face



এটি গুরুত্বপূর্ণ method evidence, Qwen3.8-27B-তে একই লাভের নিশ্চয়তা নয়।
অন্যদিকে ARC Prize-এর প্রকাশিত Qwen3.8-27B xhigh ফল ARC-AGI-2 semi-private-এ ৪২.৪%। সেটি আপনাদের Kaggle hardware ও token budget-এর অধীনে মাপা ফল নয়। ARC Prize-ও বলছে যে reasoning effort বদলালে Qwen-এর chat-template instructions বদলায়, কেবল token budget নয়। 

ARC-AGI Results



৪. সব দুর্বলতার Priority Ranking
নিচের impact ranking অনুমানভিত্তিক; এগুলো কোনো মাপা leaderboard gain নয়। একই সমস্যার একাধিক নাম থাকলে বাস্তবে একই fix দিয়ে সমাধান হতে পারে।
Rank	সমস্যা	Verdict	সম্ভাব্য গুরুত্ব
P0	AGENT real GPU-তে unmeasured	Confirmed	BLOCKER
P0	Evaluation-code filesystem isolation নেই	Reproduced	BLOCKER
1	Coverage ও দুই attempt-এর budget accounting	Confirmed	Very high
2	AGENT failure-এর পরে DIRECT fallback নেই	Confirmed	Very high
3	Train-fit হলেও test ভুল হতে পারে	Confirmed	Very high
4	DIRECT-এ অতিরিক্ত reasoning-token ব্যবহার	Measured	Very high
5	DIRECT-এর দ্বিতীয় attempt প্রায় অর্থহীন	Confirmed	High
6	Agent-এ hypothesis diversity ও independent validation নেই	Confirmed	High
7	DIRECT-এ executable verification নেই	Confirmed	High
8	Context trimming-এ গুরুত্বপূর্ণ তথ্য হারানো	Plausible	Medium–high
9	Per-turn thinking ও forced submit policy	Unmeasured	Medium–high
10	Retained function-এর nonportable global references	Reproduced	Medium
11	Smoke test আর actual request-এর mismatch	Confirmed	High reliability
12	Grid representation ও parsing	Needs A/B	Medium–low
13	Test-time training নেই	Confirmed	Unknown, costly
P0 মানে প্রথমে সমাধান করতে হবে। Ranking-এর বাকিগুলো সম্ভাব্য leaderboard impact অনুযায়ী।
৫. বিস্তারিত দুর্বলতা, সমাধান ও পরীক্ষা
P0-A. AGENT-এর real GPU performance অজানা
প্রমাণ — measured: runs/agent_run1_v1_4/-এ server startup ও tool smoke সফল। কিন্তু প্রত্যেক task request-এ HTTP 400 পাওয়া গেছে:
Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.

v1.5-এ template_kwargs.reasoning_effort="xhigh" ব্যবহার করা হয়েছে, এবং smoke-এর agent-parameter coverage বাড়ানো হয়েছে। কিন্তু v1.5-এর real inference log ZIP-এ নেই।
নতুন অডিট পর্যবেক্ষণ: submit_main.py-এর smoke()-তে actual agent-parameter test তৈরি করা হয়েছে max_tokens=32 দিয়ে, যদিও বাস্তব request limit 8192। ফলে comment-এ “EXACT request shape” বললেও token cap পুরোপুরি সমান নয়। আরও একটি পার্থক্য হলো মূল thinking smoke-তে low effort ব্যবহৃত হয়।
Fix: Smoke test-কে দুই ভাগ করুন। প্রথমটি ছোট এবং দ্রুত parameter-acceptance smoke। দ্বিতীয়টি actual xhigh + 8192 max_tokens + 6144 thinking budget + normal tool choice দিয়ে একটি bounded end-to-end inference। দ্বিতীয় test-এ অন্তত একটি real tool response এবং verify ফলাফল পেতে হবে।
এই GPU smoke-এর cost পরিকল্পনা হিসেবে ৩–৮k generated tokens ধরুন; এটি measurement নয়।
EVAL12 validation: HTTP 400 = 0, thinking_budget_dropped_at_runtime=false, অন্তত একটি পূর্ণ python → submit cycle এবং ১২টি task-এর সকলটির attempted status থাকতে হবে। এর আগে AGENT বনাম DIRECT score comparison অর্থহীন।
P0-B. Program execution-এ solution-file access সম্ভব
প্রমাণ — reproduced locally: AGENT-এর sandbox.py-র run_one()-তে user-generated Python একটি স্বতন্ত্র temporary file পড়তে সক্ষম হয়েছে। সীমিত execution time ও memory ব্যবহার করা হলেও filesystem আলাদা করা হয়নি।
repl.py-এর run_snippet()-এও একই শ্রেণির সমস্যা আছে। বর্তমানে run_eval.py regex দিয়ে সন্দেহজনক file access শনাক্ত করে, কিন্তু regex execution আটকায় না।
এটি গুরুতর, কারণ EVAL12-এর public solution JSON inference environment-এ উপস্থিত।
Fix: Actual inference subprocess থেকে evaluation solutions-এ filesystem access বন্ধ করুন। সবচেয়ে পরিষ্কার নকশা হলো isolated container/process filesystem এবং scoring process আলাদা রাখা। Linux filesystem namespace, sandboxed container অথবা equivalent isolation-এর কাজ করা ব্যবস্থা না থাকলে fail closed করুন। কেবল open() blacklist বা Python builtins বদলানোকে নিরাপদ sandbox ধরে নেওয়া যাবে না।
স্বয়ংক্রিয় file-access detector রাখুন, তবে সেটিকে security boundary নয়, diagnostic layer হিসেবে গণ্য করুন।
Athanor-এর public repository-ও সতর্ক করেছে: “Model-generated Python execution is unsafe”—এবং isolated environment ব্যবহার করতে বলেছে। 

GitHub



Cost: inference tokens শূন্য; process/container launch overhead আলাদাভাবে benchmark করতে হবে।
Validation: ইচ্ছাকৃত একটি harmless sentinel file পড়ার program sandbox-এ চালান। Expected result: access denied। এরপর normal NumPy transform-এর correctness ও execution overhead পরীক্ষা করুন।
এটি scoring-integrity blocker; আগে ঠিক করা উচিত।
1. Budget ও coverage: বর্তমান AGENT-এ সবচেয়ে বড় scaling risk
প্রমাণ — confirmed from code: run_submit.py:130–132-এ:
supply = cfg.get("tok_per_s", 140) * availcap = int(min(max_cap, max(min_cap,              supply / max(n, 1) * cap_factor)))



সমস্যা হলো cap হিসাব করা হচ্ছে task প্রতি, কিন্তু কোনো task ব্যর্থ হলে দ্বিতীয় attempt-ও সম্পূর্ণ cap ব্যবহার করতে পারে। solve_repl() প্রথমে সব task-এর first attempt queue করে—এটি ভালো fairness design—কিন্তু দ্বিতীয় attempt-এর জন্য আলাদা aggregate reserve নেই।
বর্তমান config-এ tok_per_s=300; অথচ বাস্তবে AGENT-এর aggregate throughput এখনো মাপা হয়নি।
আনুমানিক capacity হিসাব:
Scenario	Aggregate output token budget	গড়ে প্রতি ২৪০ task
10.5 h × 250 tok/s	9.45M	39.4k
10.5 h × 275 tok/s	10.40M	43.3k
10.5 h × 300 tok/s	11.34M	47.3k
এগুলোতে prompt processing, tool execution এবং বাকি runtime overhead অন্তর্ভুক্ত নয়।
একটি ৫০k first attempt-এর পর আরেকটি ৫০k attempt প্রত্যেক task-এ চালানো সম্ভব নয়—এই রেঞ্জে coverage ভেঙে পড়বে।
Fix: প্রথম pass-এ সব task-এর জন্য সীমিত scout budget দিন; পরের pass-এ failure ও uncertainty অনুযায়ী অবশিষ্ট budget বিতরণ করুন।
প্রাথমিক পরীক্ষার জন্য first-pass per-task cap ১২–১৮k generated tokens, তারপর rescue/deep pass। এটি tune করার starting range, tested optimum নয়।
EVAL12 validation: time_to_first_attempt_coverage, percentage tasks attempted, percentage verified, generated tokens/task, prompt tokens/task, tool time এবং projected hidden-run completion হিসাব করুন।
GO শর্ত: কেবল ১২/১২ task attempted নয়; measured runtime দিয়ে ২৪০-task hidden run-এর first-pass completion যুক্তিসংগতভাবে feasible হতে হবে।
2. AGENT fallback দুর্বল: HYBRID সবচেয়ে প্রয়োজনীয় পরিবর্তন
প্রমাণ — confirmed: run_submit.py:157–161-এ কোনো program candidate না থাকলে input grid-ই attempt_1/attempt_2 হয়ে যায়। AGENT-এর run_direct.py ইতিমধ্যে আছে, কিন্তু mode="repl" execution path-এ সেটি rescue হিসেবে চালু নয়।
Fix: AGENT ও DIRECT-এর আলাদা answer pool রাখতে হবে। প্রথমে verified program candidates, তারপর direct candidates, তারপর valid unverified candidates বিবেচনা করুন। এই order-ও validation-set outcome দিয়ে tune করতে হবে; সব verified program-কে অন্ধভাবে সঠিক ধরে নেওয়া যাবে না।
প্রথম version-এর জন্য run_submit.py-এ এই candidate-merging function ব্যবহার করা যায়:
def merge_candidates(agent_grids, direct_grids, inp):    out = []    for g in [*agent_grids, *direct_grids, inp]:        if g is None:            continue        if g not in out:            out.append(g)        if len(out) == 2:            break    while len(out) < 2:        out.append(inp)    return {        "attempt_1": out[0],        "attempt_2": out[1]    }



এটি কেবল valid, already-ranked grids-এর জন্য। Integration-এ grid validator, per-task persistent result pool এবং thread lock অপরিহার্য। agent_grids-এ verified ও unverified ranking আলাদা রাখা ভালো।
Cost: উদাহরণস্বরূপ ১০০টি unresolved task-এ প্রত্যেকটি ১৬k-token DIRECT rescue দিলে ১.৬M tokens; ২৭৫ tok/s ধরে প্রায় ৯৭ মিনিটের generation-equivalent cost।
EVAL12 validation: AGENT-only, DIRECT-only এবং merged hybrid—তিনটির একই ২২ test output-এ paired score মাপুন। বিশেষ করে agent_failed → direct_correct কতবার ঘটছে, সেটিই প্রধান metric।
3. Train verification যথেষ্ট নয়
প্রমাণ — confirmed: repl_agent.py:95–116 প্রতিটি train example-এ program চালিয়ে n_ok গণনা করে। n_ok == n_train হলে program accepted হয়, কিন্তু held-out test rule-টি ঠিক থাকবে এমন নিশ্চয়তা নেই।
আপনাদের local test-এও verified program থাকা সত্ত্বেও wrong test output-এর scenario পরীক্ষা করা হয়েছে। তবে সেটি mock test, real Qwen model-এর overfitting rate নয়।
Fix: শুধু train-fit নয়, candidate-এর generalization পরীক্ষা করুন। তিনটি কার্যকর স্তর:
- Leave-one-out train validation: সম্ভব হলে একটি train pair আড়াল রেখে বাকি pair দিয়ে hypothesis তৈরি এবং আড়াল করা pair-এ পরীক্ষা।
- Transformation invariance: relevant color permutation, translation বা rotation-এর ক্ষেত্রে rule-consistency পরীক্ষা; সব ARC task-এ সব augmentation বৈধ নয়।
- Independent critic: fresh Qwen context-এ hypothesis/code/test output দেখিয়ে ambiguity, hardcoding ও counterexample খুঁজতে বলুন।
Athanor সরাসরি independent artifact-only review ব্যবহার করে এবং train-fit solution-ও generalization grounds-এ reject করতে পারে বলে বর্ণনা দিয়েছে। 

GitHub



এখানে hidden answer জানা লাগে না; reviewer কেবল program, examples এবং proposed outputs নিয়ে কাজ করে।
Cost: static checks প্রায় বিনা generated-token খরচে; independent critic আনুমানিক ১.৫–৪k tokens/candidate।
EVAL12: verified_count, verified_and_correct_count, verified_but_wrong_count আলাদা করুন। Critic-এর precision, false rejection এবং net score gain মাপুন। Train verification বাড়লেও exact score না বাড়লে improvement গ্রহণযোগ্য নয়।
4. DIRECT-এর দীর্ঘ reasoning-token budget
প্রমাণ — measured: আগে দেখানো ৩৭টি forced query-তে ১টি সঠিক। ARC Prize-এর Qwen xhigh result ৪২.৪%, কিন্তু সেখানে dedicated inference environment ও পৃথক token economics ছিল। 

ARC-AGI Results



Kaggle আলোচনা #745787-এর সংরক্ষিত লেখায় NguyenThanhNhan জানিয়েছেন তিনি প্রায় ৪৯k token পরে reasoning force-close করতেন; তাঁর বক্তব্যেও pure reasoning-কে token-inefficient বলা হয়েছে। এটি অংশগ্রহণকারীর দাবি, independently reproduced measurement নয়।
Fix: সবার জন্য ৪৯k নয়। কম token-এ confident answer পাওয়া গেলে early commit; বাকি budget hard cases-এ অথবা দ্বিতীয় hypothesis তৈরিতে দিন। একটি DIRECT candidate ৫০k পর্যন্ত নেওয়ার চেয়ে ১২k + ১২k independent candidates কখন বেশি exact answer দেয়, সেটি পরীক্ষা করা দরকার।
Cost: দুটি ১২k candidate প্রায় ২৪k tokens—৪৯k single candidate-এর অর্ধেকের কাছাকাছি। কিন্তু quality improvement অনিশ্চিত।
EVAL12: 8k/16k/32k/49k cap-এ matched-budget experiment করুন। শুধু score নয়, answer validity, natural completion, forced completion এবং marginal hits per million tokens মাপুন।
5. DIRECT-এর attempt_2 নষ্ট হচ্ছে
প্রমাণ — confirmed: old_cell5.py-এর make_attempts() model answer-এর পরে test input এবং তার horizontal mirror দিয়ে দ্বিতীয় candidate তৈরি করে।
এটি কিছু identity/mirror task-এ সাহায্য করতে পারে, তবে programmatically grounded না হলে বেশিরভাগ task-এ দ্বিতীয় সুযোগ নষ্ট হয়।
Fix: দ্বিতীয় attempt-এ প্রথমটির থেকে ভিন্ন, independently generated rule অথবা verified program-এর output ব্যবহার করুন। প্রথম grid-এর সামান্য random perturbation নয়।
উদাহরণ: candidate A direct xhigh answer, candidate B ছোট independent code-synthesis/critic pass-এর result।
Cost: ৫–১৫k extra tokens per selected task; সব task-এ চালালে অতিরিক্ত ১.২–৩.৬M tokens।
EVAL12: attempt_1_hits, attempt_2_only_hits, duplicate_rate, per_million_tokens_gain রিপোর্ট করুন। দ্বিতীয় attempt-এর সাফল্য হলো প্রথমটি ভুল কিন্তু দ্বিতীয়টি সঠিক—শুধু আলাদা grid তৈরি হওয়া নয়।
6. Candidate diversity নেই
প্রমাণ — confirmed: AGENT-এর temperature 0.6 থাকলেও sample count বাড়ালেই meaningful hypothesis diversity পাওয়া যায় না। একই prompt, একই tools, প্রায় একই reasoning path থেকে near-duplicate programs তৈরি হতে পারে।
Imbue-এর প্রকাশিত Darwinian Evolver একটি population ধরে mutation ও evaluation করে। তাদের ARC-AGI-2 ফলাফলে Kimi K2.5-এর ১২% থেকে ৩৪% উন্নতি রিপোর্ট করা হয়েছে, তবে এটি Qwen3.8-27B-এর result নয়। 

GitHub
+1



Fix: Candidate archive রাখুন; program hash এবং output hash দিয়ে duplicate বাদ দিন। দ্বিতীয় attempt-এ নতুন seed-এর পাশাপাশি পৃথক hypothesis strategy ব্যবহার করুন—যেমন object decomposition বনাম geometry/pattern reasoning। পূর্ণ evolutionary system আনার আগে lightweight best-of-2 পরীক্ষা করুন।
Cost: নতুন full candidate-এর খরচ ৬–১৮k tokens; archive/dedup-এর generation-token খরচ নেই।
EVAL12: unique executable programs, unique test predictions, verified diversity এবং second-attempt-only hits মাপুন। Duplicate rate বেশি হলে আরও attempts কেনা লাভজনক নয়।
7. DIRECT-এ verification নেই
DIRECT এখন final text থেকে grid parse করে জমা দেয়। Prompt-এ examples যাচাই করতে বলা হলেও model বাস্তবে executable transform() দিয়ে তা পরীক্ষা করে না।
Fix: DIRECT answer দেওয়ার আগে একটি optional rule/code artifact তৈরির short pass, অথবা answer দেওয়ার পরে limited self-verification pass পরীক্ষা করুন। সব task-এ পূর্ণ tool agent চালানো নয়।
Noemon-এর public solver একটি Reasoner, Validator, final solver এবং judge ব্যবহার করে। তাদের published architecture এই decomposition সমর্থন করে। 

noemon/noemon-arc-agi · GitHub



Cost: ২–৬k tokens/task হলে ২৪০ task-এর জন্য ০.৪৮–১.৪৪M tokens।
EVAL12: কত ভুল answer verifier শনাক্ত করে, কত সঠিক answer ভুলভাবে reject করে, এবং সংশোধনের পরে কত exact hit ফিরে আসে—এই তিনটি মাপুন।
8. Context trimming অতি সরল
_est_tokens() বর্তমানে JSON character count থেকে approximate token count বের করে:
return len(json.dumps(messages)) // 3



এটি tokenizer-accurate নয়। trim() পুরোনো turn সরিয়ে একটি count note রাখে; কোনো structured hypothesis memory নেই। ফলে previously tested rules, rejected candidates এবং important tool observations হারাতে পারে।
Fix: Trimming-এর আগে সংক্ষিপ্ত checkpoint তৈরি করুন: task hypothesis, best verified/unverified program, failed checks এবং next action। সম্ভব হলে actual tokenizer দিয়ে prompt length মাপুন। Repeated reasoning text নয়, compact artifacts ধরে রাখুন।
Tufa Duck Harness context eviction ব্যবহার করে, আর Athanor reasoning-chain-এর পরিবর্তে artifact exchange ও checkpointing বর্ণনা করেছে। 

Tufa Labs
+1



Cost: প্রতি checkpoint ১৫০–৫০০ tokens, plus compression overhead যদি LLM summarization করা হয়।
EVAL12: context overflow count, evictions, recovery-success rate, per-turn prefill tokens এবং score compare করুন। Prefix cache hit-ও track করুন।
9. Per-turn cap ও forced-submit policy
v1.5-এ thinking budget 6144, turn max 8192 এবং attempt cap 60k। চার-পাঁচটি দীর্ঘ turn অনেক বাজেট শেষ করতে পারে, কিন্তু বাস্তবে কত tool action হচ্ছে তা অজানা।
আরেকটি সূক্ষ্ম সমস্যা: force_submit_after=5 হলো Python-call count ভিত্তিক। Model যদি বারবার thinking শেষে কোনো tool call না দেয়, এই trigger কার্যকর হবে না।
Fix: শুধু Python-call count নয়, generated-token count ও consecutive cut count দিয়ে intervention করুন। পরীক্ষার জন্য 4096 বনাম 6144 thinking cap এবং forced-submit trigger ৩ বনাম ৫ Python calls তুলনা করুন। Tool-less consecutive cut দুবার হলে targeted action prompt দিন; forced submit সবসময় সঠিক হবে না।
Cost: কম cap নিজে token বাঁচাতে পারে, কিন্তু বাড়তি turns ও prompt prefill লাগে। প্রকৃত cost real GPU measurement ছাড়া নির্ধারিত নয়।
EVAL12: tool_calls / 1000 generated tokens, first-submit latency, forced-submit validity, cuts/turn এবং solve rate তুলনা করুন।
10. Retained functions-এর portability bug
প্রমাণ — নিজে reproduce করেছি।
AGENT-এর retained-function validator train_inputs-কে available global ধরে। ফলে এই function retain হয়:
def transform(grid):    return train_inputs[0].tolist()



কিন্তু submit() sandbox-এ ওই global নেই। পরীক্ষায় ফল:
NameError: name 'train_inputs' is not defined


Fix: Retained functions-এ train_inputs, train_outputs, test_inputs ও train global reference থাকলে portable বলে accept করবেন না। Tool reply-এ স্পষ্ট নির্দেশ দিন: grid input parameter থেকে rule compute করতে হবে।
with_kept()-এ শুধু required helper preamble prepend করাই যথেষ্ট নয়। Missing-global validation দরকার।
Cost: negligible; compilation/static validation।
EVAL12: retained-function submit-এর NameError শূন্য হতে হবে। একই সঙ্গে valid numpy ও helper-based functions যেন reject না হয়, সেটা regression test-এ নিশ্চিত করুন।
11. Smoke validation-এর completeness
আগের reasoning_effort="high" failure বন্ধ করার জন্য v1.5-এর পরিবর্তন উপকারী। কিন্তু actual request-এর বড় generation window, repeated turns, prefix reuse এবং forced submit path এখনও end-to-end GPU test পায়নি।
Fix: smoke-এর পরে একটি tiny actual agent task চালান, actual config ব্যবহার করে; তার trace থেকে ২–৩ turn, parse, tool call, verify ও deadline দেখুন।
EVAL12: smoke success-কে readiness হিসেবে ধরুন, accuracy proof নয়। Real mini-task failure হলে full EVAL12 শুরু করবেন না।
12. Grid layout
DIRECT space-separated rows ব্যবহার করে; AGENT-এর show() compact digit rows ব্যবহার করে। কোনোটিকে এই Qwen3.8 model-এর জন্য universal optimum বলা যায় না।
Fix: তিনটি format তুলনা করুন: spaced digits, compact digits এবং coordinate/object summaries। Grid data যেন lossless থাকে।
Cost: মূলত prompt length; extra generated tokens প্রয়োজন নেই।
EVAL12: prompt token count, parse success এবং identical model-budget accuracy compare করুন। শুধু format পরিবর্তনে ১/২২ score gain হলেও একটি sample-এ সেটিকে definitive improvement বলা যাবে না।
13. Test-time training ও অন্য model-family combination
Confirmed: বর্তমান AGENT-এ task-specific gradient update বা 4B NVARC ensemble নেই।
কিন্তু এখানে তিনটি আলাদা প্রশ্ন রয়েছে: inference-time candidate search, gradient-based test-time training এবং multi-model ensemble। এগুলো এক জিনিস নয়।
আপনার নির্ধারিত constraints অনুযায়ী scoring run-এ Qwen3.8-27B-ই থাকবে। তাই 4B NVARC বা অন্য model আনছি না। Qwen-generated programs-এর population refinement করা সম্ভব এবং সেটি একই মডেলের inference-time computation।
Verdict: Full gradient-based TTT এখন lower priority। ৪×L4-তে 27B model serving-এর পাশাপাশি per-task fine-tuning memory, throughput ও checkpoint overhead বাড়াতে পারে। এই সীমায় measurable benefit না পাওয়া পর্যন্ত এটি implementation roadmap-এর প্রথম ধাপে রাখা উচিত নয়।
EVAL12: আগে cheap inference-time self-revision বনাম independent candidate generation পরীক্ষা করুন। Gradient TTT করতে হলে আলাদা GPU feasibility ও accuracy experiment প্রয়োজন।
৬. সবচেয়ে জরুরি Code Patches
নিচের তিনটি পরিবর্তন প্রথম বাস্তব GPU test-এর আগে করা উচিত। এগুলো implementation proposal; স্থানীয়ভাবে পাওয়া সমস্যাগুলোর ভিত্তিতে লেখা, কিন্তু modified notebook-এর পূর্ণ GPU regression test করা হয়নি।
Patch A — Nonportable retained function আগেই reject করা
repl_agent.py-এর verify()-তে run_one()-এর আগে:
import astNONPORTABLE_GLOBALS = {    "train_inputs",    "train_outputs",    "test_inputs",    "train",}def detect_nonportable_names(code):    try:        tree = ast.parse(code)    except SyntaxError:        return set()    # Conservative guard: reject these names anywhere    # in submitted source, even if shadowed locally.    used = {        node.id for node in ast.walk(tree)        if isinstance(node, ast.Name)    }    return used & NONPORTABLE_GLOBALS



তারপর:
def verify(code, task, timeout=10.0):    bad = detect_nonportable_names(code)    if bad:        feedback = (            "Nonportable globals: "            + ", ".join(sorted(bad))            + ". Rewrite transform(grid) so it depends "              "only on its input and portable helpers."        )        return 0, [None] * len(task["test"]), feedback    # Existing verify implementation continues here.



এটি fail-fast guard, সম্পূর্ণ sandbox security নয়।
Regression test: train_inputs-dependent function rejected; standalone np.rot90 transform accepted; valid retained helper dependency কাজ করে।
Patch B — AGENT parameter smoke সত্যিকার configuration দিয়ে চালানো
submit_main.py-তে agent_params smoke শেষে actual request পরীক্ষার জন্য:
# Additional end-to-end GPU probe after fast parameter smokeprobe_client = ChatClient(    "http://127.0.0.1:1234/v1",    "qwen",    temperature=cfg["temperature"],    top_p=cfg["top_p"],    top_k=cfg["top_k"],    max_tokens=cfg["turn_max_tokens"],    thinking=True,    thinking_token_budget=cfg["thinking_token_budget"],    template_kwargs=cfg["template_kwargs"],)probe = probe_client.chat(    [        {"role": "system", "content":            "Use Python tools to check simple grid rules."},        {"role": "user", "content":            "Use the python tool to print the shape "            "of train_inputs[0]."}    ],    tools=make_tools(cfg.get("retain_functions", False)),    timeout=90,)assert "error" not in probe, probe.get("error")assert probe.get("tool_calls"), "No real tool call"



এতে বাস্তব xhigh path পরীক্ষা শুরু হবে। তবে এটিকে পূর্ণ roundtrip validation করতে প্রাপ্ত tool code execute করে result পরের chat request-এ ফেরত দিতে হবে। বর্তমান smoke()-এর roundtrip logic reuse করাই ভালো।
Patch C — Global token ledger
run_submit.py-তে প্রতি task-এর cap ছাড়াও shared budget accounting দরকার।
import threadingclass TokenLedger:    def __init__(self, total_budget):        self.remaining = int(total_budget)        self.lock = threading.Lock()    def reserve(self, requested):        requested = max(0, int(requested))        with self.lock:            allowed = min(requested, self.remaining)            self.remaining -= allowed        return allowed    def reconcile(self, reserved, actual, success):        # Unknown server consumption on timeout:        # conservatively do not refund.        if not success:            return        refund = max(0, reserved - int(actual))        with self.lock:            self.remaining += refund



ChatClient.chat()-এর চারপাশে এই ledger বসাতে হবে, যাতে প্রত্যেক generation request আগে token reserve করে। Success হলে completion_tokens অনুযায়ী refund, timeout/unknown server state হলে conservatively debit রাখতে হবে। Concurrent requests-ও এতে global budget-এর বাইরে যেতে পারবে না।
এটি generated-token accounting; prefill ও CPU overhead-এর জন্য wall-clock reserve আলাদা রাখতে হবে।

৭. ১২ ঘণ্টার মধ্যে HYBRID-এর বাস্তব compute plan
এখানে কোনো নতুন model বা engine দরকার নেই। Qwen3.8-27B-FP8, vLLM 0.19, 4×L4 অপরিবর্তিত থাকবে।
প্রথম AGENT GPU measurement পেলে scheduling dynamically adjust করতে হবে। তার আগে একটি illustrative budget scenario:
Illustrative generated-token allocation
ধরা হয়েছে AGENT প্রায় ২৭৫ aggregate generated tok/s ধরে রাখতে পারে। এটি এখনো মাপা হয়নি।Agent scout · 240 × 16k
3.84M




Direct rescue · 150 × 20k
3.00M




Candidate refinement · 80 × 10k
0.80M




Critic · 100 × 3k
0.30M





Planned output tokens
7.94M

২৭৫ tok/s হলে শুধু decoding-equivalent সময় প্রায় ৮.০২ ঘণ্টা। Server loading, prefill, Python tools, failed calls এবং scheduling overhead যোগ হবে।
এই plan কেন গুরুত্বপূর্ণ? কারণ ২৪০ task-এর সবগুলোতে ৫০k-token agent attempt চালালে শুধু first pass-ই ১২M tokens। সেটি ২৭৫ tok/s-এ প্রায় ১২.১ ঘণ্টার generation time—startup বা tool overhead ছাড়াই। অথচ ৬০%-এর জন্য শুধু কিছু কঠিন task অতিরিক্ত গভীরভাবে solve করাই যথেষ্ট নয়, সম্পূর্ণ task coverage-এর ওপর উচ্চ solve rate দরকার।
AGENT যদি ১৫০ tok/s-এর কাছাকাছি চলে, তাহলে উপরোক্ত allocation কমাতে হবে। তাই measured throughput ছাড়া fixed 60k per-attempt cap ব্যবহার বিপজ্জনক।
৮. EVAL12: ২.৫ ঘণ্টার সিদ্ধান্ত-পরীক্ষা
EVAL12-কে final accuracy estimate নয়, একটি diagnostic gate হিসেবে ব্যবহার করার পরামর্শ দিচ্ছি। ২২টি test output-এ ১টি অতিরিক্ত correct মানে ৪.৫৫ percentage-point পরিবর্তন—খুবই noisy signal।
Experiment	কী পরিবর্তন	প্রধান GO condition
E1 — Agent baseline	v1.5, existing config	400 error নেই, real throughput ও coverage জানা
E2 — Short-turn	4096 বনাম 6144 thinking budget	বেশি meaningful tool calls per token
E3 — Early-submit	প্রথম submit তাড়াতাড়ি	valid submit ও train-fit efficiency বাড়ে
E4 — Hybrid	agent + direct rescue	exact hits-এর net gain
E5 — Generalization check	critic + candidate filtering	verified-but-wrong কমে
E6 — Diversity	independent second candidate	attempt_2-only hits বাড়ে
এগুলো সব আলাদা full ২.৫ ঘণ্টার GPU run করতে হবে এমন নয়। প্রথম infrastructure gate-এর পর selected tasks-এ paired A/B চালিয়ে total budget নিয়ন্ত্রণ করা সম্ভব।
একটি গুরুত্বপূর্ণ পরিবর্তন: EVAL12-এর GPU allocation hidden run-এর per-task allocation-এর সঙ্গে normalize করতে হবে। ১২টি evaluation task-এ ৬০k × ২ attempt খরচ করে ভালো score পেলেও সেটি ২৪০-task hidden run-এর feasible policy প্রমাণ করবে না।
প্রতিটি experiment থেকে একটি নির্দিষ্ট record schema রাখা উচিত:

{
  "task_id": "...",
  "attempt_id": 0,
  "strategy": "agent_scout",
  "status": "verified",
  "generated_tokens": 0,
  "prompt_tokens": 0,
  "wall_seconds": 0,
  "python_calls": 0,
  "submit_calls": 0,
  "train_pairs_correct": 0,
  "train_pairs_total": 0,
  "test_grid_valid": true,
  "independent_candidate": false,
  "file_access_flag": false
}



Hidden scoring-এ answer unavailable থাকবে; evaluation-এর পরে scorer আলাদাভাবে exact correctness যুক্ত করবে। Inference process-এ solution labels expose করা হবে না।
আরও অন্তত একটি পৃথক holdout evaluation set রাখুন। EVAL12-তে বহু A/B এবং prompt tuning করার পর সেটিকে model-selection validation হিসেবে আর নিরপেক্ষ ধরা যাবে না।

৯. ৬০%-এর পথে কী প্রমাণ করতে হবে?
আপনাদের 18.75 public LB baseline থেকে 60 লক্ষ্য মানে +41.25 percentage points। যদি পুরোনো correct answers সব রক্ষা করা যায়, তাহলে পুরোনো ভুলগুলোর প্রায় ৫০.৮% নতুনভাবে ঠিক করতে হবে।
এটি সহজ নয়, কিন্তু engineering target হিসেবে পরিমাপযোগ্য।
Milestone gates — লক্ষ্য, predicted score নয়
1. Infrastructure-valid AGENT — real tool loop, verified programs, token efficiency এবং coverage নিশ্চিত করতে হবে। AGENT-এর প্রকৃত baseline এখানেই প্রথম জানা যাবে।
2. Complementarity proof — AGENT কি DIRECT-এর ভুল task solve করছে? একই task-এ দুজনই ভুল হলে hybrid score gain আসবে না।
3. Independent second-attempt proof — attempt_2 শুধু আলাদা নয়, attempt_1 ভুল থাকলে নতুন correct answer দিতে হবে।
4. Generalization proof — train-fit verification বনাম true held-out correctness-এর gap কমাতে হবে।
5. Scale proof — ১২ ঘণ্টায় প্রায় সব ২৪০ hidden task-এর meaningful candidate output দিতে পারতে হবে।
6. Leaderboard progression — একাধিক holdout ও actual submission-এ ধারাবাহিক improvement দেখেই ৩০, ৪০, ৫০ এবং তারপর ৬০ লক্ষ্য অনুযায়ী budget আরও গভীরে নিতে হবে।

কেন শীর্ষ দলগুলোর score দেখে সরাসরি architecture অনুমান করা উচিত নয়?
ZIP-এর ১১ অক্টোবরের leaderboard snapshot-এ সাতটি দল ৬০-এর ওপরে। এটি আপনাদের দেওয়া snapshot-এর প্রমাণ; প্রত্যেক দলের algorithm বা model নিশ্চিত করার প্রমাণ নয়। আমার web retrieval-এ leaderboard-এর cached snapshot-ও আলাদা সময়ের দেখা যাচ্ছে, তাই ওই সংখ্যাগুলোকে live, exhaustive standings বলছি না। 

Kaggle



Kaggle discussion #748082-এর সংরক্ষিত ১০ অক্টোবরের বক্তব্যে একজন অংশগ্রহণকারী Qwen3.8-27B-এর local public evaluation-এ ৮০% এবং Kaggle LB-তে ২৬.৮১ বলেছেন। এই বিশাল gap মনে করিয়ে দেয় যে local public-set performance সরাসরি hidden LB-তে translate হয় না।
আর #746149-এর “juicing qwen 27b” সম্পর্কে মন্তব্যটি engineering-based improvement-এর hypothesis, verified top-team disclosure নয়। এই দুটি আলোচনা আমি ZIP-এর সংরক্ষিত text থেকে পড়েছি; আলাদা Kaggle discussion page সরাসরি web দিয়ে খোলা যায়নি।

১০. সবচেয়ে প্রাসঙ্গিক গবেষণা ও কোড — Evidence Summary
উৎস	সরাসরি প্রমাণ/ব্যবহার	আপনার pipeline-এ গ্রহণযোগ্যতা
Tufa Duck Harness	Public REPL harness, context handling, traces	High; ARC-AGI-3 থেকে adaptation দরকার
gutfeeling Agentic Coding	Stateful REPL, same-model baseline comparison	Very high
Agentic Coding Traces	Full model/tool traces; 6.11 → 26.39 GPT-OSS	Very high
Darwinian Evolver	Candidate population, mutation, scoring	High, but compute-limited adaptation
ARCgentica	Sub-agents, program synthesis, published results	Medium; API-model budget differs
Noemon	Reasoner–validator–judge separation	High for critic design
Athanor	Independent reviewer, artifact exchange	High; score claims not Qwen evidence
Confluence	Multi-model reflective/program synthesis design	Concepts only
beetree	Multiple solvers and judge/council	Concepts only
vLLM Prefix Caching	Actual cache benefit and limitations	Directly relevant
এই repository-গুলোর প্রকাশিত numerical results-কে সাধারণভাবে author-reported public-eval results ধরে ব্যবহার করতে হবে। ARC Prize-এর testing policy বলছে public-set community scores self-reported, unless specifically verified. 

Community Leaderboard



তাই অন্য model ও API budget-এর ৯০%+ score থেকে আপনার 4×L4 Qwen3.8 run-এর সম্ভাব্য LB gain অনুমান করিনি।

১১. Final audit verdict
DIRECT v1.4
VALID BASELINE

বাস্তব public LB 18.75; local run-এ high token wastage ও দুর্বল second attempt পরিষ্কার। এটি ফেলে দেওয়ার কারণ নেই।

DIRECT v1.4.4
LB PENDING

EVAL12-এ 3/22; performance gain দেখা গেছে, কিন্তু prefix caching-কে একমাত্র কারণ বলা যাবে না।

AGENT v1.5
GPU GO PENDING

CPU mock tests ও নতুন parameter fixes ইতিবাচক। test_v15_local.py আমি চালিয়ে PASS পেয়েছি। তবে অন্য দুটি test পুনরায় চালানোর যৌথ command শেষ হয়নি, তাই সেগুলোর archived PASS output-কে fresh independent verification বলছি না। Real inference-quality evidence এখনো অনুপস্থিত।

HYBRID
DESIGN CANDIDATE

সবচেয়ে promising engineering direction, কিন্তু বাস্তব GPU run ও paired score ছাড়া এখনই submission-ready নয়।

আমার চূড়ান্ত সুপারিশ: প্রথম পরিবর্তন হবে না আরও বড় model, পূর্ণ evolutionary framework, কিংবা ১০০k reasoning। প্রথম কাজ হলো AGENT v1.5-এর GPU feasibility নিশ্চিত করা, filesystem isolation ঠিক করা এবং তারপরে AGENT ও DIRECT-এর complementary correct answers মাপা।
এরপর global token ledger, adaptive scheduler, independent second attempt ও generalization critic বসাতে হবে। ৬০%-এর দিকে এটিই সবচেয়ে যুক্তিসংগত পরীক্ষাযোগ্য পথ—কিন্তু কোনো নির্দিষ্ট score improvement এখনো নিশ্চিত নয়।
সবচেয়ে মূল্যবান পরের সিদ্ধান্ত নির্ধারণ করবে তিনটি মাপা সংখ্যা: AGENT-এর real tok/s, verified-but-correct rate, এবং DIRECT-wrong/AGENT-correct task count। এগুলো ছাড়া AGENT-এ বড় বিনিয়োগ করা অন্ধ অনুমানের ওপর নির্ভর করবে।
</pasted_content id="1891">

"