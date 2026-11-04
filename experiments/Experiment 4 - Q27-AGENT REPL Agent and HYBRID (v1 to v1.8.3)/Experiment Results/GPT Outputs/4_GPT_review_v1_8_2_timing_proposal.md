"

<pasted_content id="1891">
Q27 v1.8.2 — প্রস্তাবিত পরিবর্তনের Deep Review
আমার সিদ্ধান্ত: চারটির মধ্যে তিনটি পরিবর্তন গ্রহণ করব, তবে ৪ নম্বরের deadline-aware budget algorithm বর্তমান অবস্থায় গ্রহণ করব না। ধারণাটি ভালো, কিন্তু কয়েকটি অনুমান ভুল হলে শেষ ৩০–৬০ মিনিটে অনেক request অসম্পূর্ণ থেকে যেতে পারে।
আমি আগের v1.8.1 ZIP-এর source, DIRECT v1.4.4-এর actual run report এবং vLLM 0.19.1 documentation মিলিয়ে দেখেছি। v1.8.2-এর নতুন notebook এখনো পাইনি, তাই এটি proposed design audit—বাস্তব v1.8.2 implementation verification নয়।
১. চারটি সিদ্ধান্তের Verdict
পরিবর্তন	লাভ	ঝুঁকি	সিদ্ধান্ত
KV metrics যোগ	প্রকৃত memory pressure ও preemption জানা যাবে	শুধু final snapshot নিলে peak হারাবে	GO, সংশোধনসহ
MTP3	আগের সফল engine configuration বজায় থাকবে	AGENT-এ একই গতি নিশ্চিত নয়	GO
Admission ১১:৩৪ পর্যন্ত	GPU utilisation ও query coverage বাড়তে পারে	শেষের request hard stop-এ কাটা পড়তে পারে	Conditional GO
Remaining-time thinking budget	Deadline অনুযায়ী token বণ্টন	prediction ভুল হলে output-ই পাওয়া যাবে না	Revision required
E2E ১০→৫ মিনিট	Diagnostic startup-এ সর্বোচ্চ ৫ মিনিট বাঁচতে পারে	Slow but healthy AGENT ভুলভাবে reject হতে পারে	Conditional GO
২. KV ১.৩৫M এবং zero preemption
আগের DIRECT v1.4.4-এর run_report.json-এ সত্যিই আছে:
KV capacity1,353,600
tokens


Concurrency16
requests


Preemptions0
recorded



১৬ × ৫৭,০০০ = ৯১২,০০০ tokens। কিন্তু এই হিসাব থেকে zero preemption নিশ্চিত করা যায় না।
কারণ KV requirement-এ input prompt, generated tokens, speculative lookahead, retained prefix-cache blocks এবং engine-specific memory allocation বিবেচনা করতে হয়। AGENT-এর tool turns ও DIRECT-এর long-context pattern-ও আলাদা।
vLLM scheduler-এর source স্পষ্টভাবে দেখায় যে নতুন KV blocks allocate করা না গেলে running request preempt হতে পারে। 

vLLM



সিদ্ধান্ত: KV pool প্রায় একই হওয়ার যুক্তি গ্রহণযোগ্য, কিন্তু preemption=0 একটি যাচাইযোগ্য hypothesis, guarantee নয়। Metrics যোগ করা অবশ্যই প্রয়োজন।
৩. MTP3 — আমার সমর্থন আছে, কিন্তু গুরুত্বপূর্ণ শর্ত
আগের DIRECT v1.4.4-এর মাপা ফল:
- MTP draft tokens: 1,065,303
- Accepted tokens: 704,499
- Acceptance rate: 66.1%
- Aggregate generation throughput: 274.6 tokens/s
- Preemptions: 0
এই engine configuration-এ MTP3 কার্যকরভাবে চলেছিল। তাই নতুন HYBRID server-এ এটিই প্রথম পছন্দ হওয়া যুক্তিসংগত।
তবে ৬৬.১% acceptance মানেই ৬৬.১% speedup নয়। MTP ছাড়া একই workload-এর paired throughput measurement না থাকলে প্রকৃত speedup দাবি করা যায় না। vLLM speculative decoding-এর counters দিয়ে acceptance rate আলাদাভাবে হিসাব করা যায়। 

vLLM



আমি যেভাবে করব: আগে {"method":"mtp","num_speculative_tokens":3} দিয়ে চেষ্টা করব। ব্যর্থ হলে MTP2, তারপর MTP ছাড়া পূর্বনির্ধারিত fallback। কোন configuration বাস্তবে load হলো এবং তার KV capacity কত হলো, সেটা রিপোর্টে স্পষ্ট রাখব।
আর একটি বিষয়: v1.8.1-এ config-এ max_num_seqs=16 ও max_model_len=81920 আছে, যা DIRECT-এর সঙ্গে মেলে। কিন্তু AGENT একটি OpenAI-serving process, আর DIRECT একটি offline LLM engine ব্যবহার করেছিল। তাই একই config হলেও measured KV capacity আবার যাচাই করতে হবে।
৪. Deadline-aware budgeting — মূল সমস্যা এখানেই
প্রস্তাবিত formula:
\[
B=\min(49152,\; (T_{\text{end}}-T_{\text{now}}-T_{\text{answer}})\times r)
\]

এটি ভালো starting formula, কিন্তু safe completion guarantee নয়।
সমস্যা A — ১০:৫০-এর উদাহরণটি পুরোপুরি মেলে না
ধরি ১৬টি concurrent request চলার সময় প্রতি request-এর effective speed ১৭.২ token/s। এটি আগের aggregate throughput 274.6 ÷ 16 থেকে নেওয়া illustrative estimate, কোনো guaranteed per-stream speed নয়।
Output deadline ১১:৪৩ এবং answer reserve ৮,১৯২ token হলে:
Request শুরু	অবশিষ্ট সময়	আনুমানিক সর্বোচ্চ thinking budget
১০:৫০	৫৩ মিনিট	~46.5k
১১:২০	২৩ মিনিট	~15.5k
১১:৩৩	১০ মিনিট	~2.1k
১১:৩৪	৯ মিনিট	~1.1k
এখানে prefill, queueing, request startup বা নিরাপত্তার margin ধরা হয়নি।
১০:৫০-এ পুরো ৪৯,১৫২ thinking tokens এবং ৮,১৯২ answer tokens দুটোই ১১:৪৩-এর মধ্যে শেষ করতে এই হারে প্রায় ৫৫.৬ মিনিট প্রয়োজন। কিন্তু হাতে ৫৩ মিনিট।
অর্থাৎ প্রস্তাবিত সময়সূচির প্রথম উদাহরণে full budget কিছুটা বেশি আশাবাদী।
সমস্যা B — গতির estimator ভুল হতে পারে
বর্তমান proposal-এ শেষ হওয়া request-এর tokens / wall_seconds দিয়ে ভবিষ্যৎ request-এর গতি অনুমান করা হবে।
এখানে পাঁচটি সমস্যা:
1. আগে শেষ হওয়া request সহজ বা ছোট হলে speed estimate biased হতে পারে।
2. Concurrency ১৬ থেকে ৮ বা ৪ হলে প্রতি stream-এর speed বদলায়।
3. Tool-agent stage থেকে DIRECT stage-এ switching-এর পর throughput আলাদা হতে পারে।
4. Prefill, prefix cache, context length ও MTP acceptance পরিবর্তিত হয়।
5. Slow request এখনো চলমান থাকলে completed-request average-এ তার ধীরগতি দেখা যাবে না।
Fix: Rolling per-stream throughput-এর lower percentile ব্যবহার করুন। Completed requests-এর পাশাপাশি active requests-এর progress থাকলে সেটিও বিবেচনা করুন। শুরুতে historical rate conservative করে ব্যবহার করুন, তারপর measured DIRECT-stage data দিয়ে update করুন।
প্রস্তাবিত validation criterion: প্রতিটি request-এর forecasted completion time বনাম actual completion time-এর error সংরক্ষণ করুন। শুধু mean error নয়, late completion rate মাপুন।
সমস্যা C — সবচেয়ে গুরুত্বপূর্ণ: বর্তমান ChatClient dynamic thinking budget সমর্থন করে না
আমি v1.8.1-এর client.py পরীক্ষা করেছি। সেখানে ChatClient.chat()-এ per-request max_tokens override আছে, কিন্তু per-request thinking_token_budget override নেই।
এখন constructor-এ একবার:
self.thinking_token_budget = thinking_token_budget



তারপর প্রত্যেক call-এ:
if self.thinking_token_budget:    payload["thinking_token_budget"] = (        self.thinking_token_budget    )



পাঠানো হয়।
ফলে আপনি শুধু max_tokens কমিয়ে ১৫k করলে server-এর thinking budget ৪৯,১৫২-ই থাকবে। Model ১৫k generation limit-এ পৌঁছে যেতে পারে, অথচ </think>-এ পৌঁছানোর আগেই response কেটে যাবে।
এটি proposed feature implement করার সময় অবশ্যই ঠিক করতে হবে।
সঠিক patch-এর মূল কাঠামো:
_UNSET = object()# Add to ChatClient.chat() parameters:# thinking_token_budget=_UNSETbudget = (    self.thinking_token_budget    if thinking_token_budget is _UNSET    else thinking_token_budget)if budget is not None:    payload["thinking_token_budget"] = int(budget)



তারপর DIRECT request-এ দুটোই পাঠাতে হবে:
r = client.chat(    messages,    max_tokens=think_budget + answer_cap,    thinking_token_budget=think_budget,    timeout=request_timeout,)



ChatClient.chat()-এর HTTP-400 retry path-এও একই per-request budget বহাল রাখতে হবে। Server budget reject করলে silently unlimited-thinking retry করা যাবে না—deadline scheduler-এর জন্য এটি correctness ও timing failure।
সমস্যা D — </think> জোর করে বন্ধ করা মানেই উত্তর পাওয়া নয়
vLLM 0.19.1 documentation নিশ্চিত করে, configured thinking_token_budget-এ পৌঁছালে reasoning-ending string inject করে reasoning section বন্ধ করা যায়। 

vLLM



কিন্তু documentation এটি বলে না যে তারপর অবশ্যই valid ARC grid তৈরি হবে।
সম্ভাব্য failure:
- Model ভুল grid তৈরি করে।
- Output malformed হয়।
- Grid সম্পূর্ণ হওয়ার আগেই answer cap শেষ হয়।
- Prompt + thinking + answer মিলে context limit ছাড়ায়।
- HTTP timeout বা server preemption/queue delay ঘটে।
- Commit fallback চালানোর মতো সময় অবশিষ্ট থাকে না।
v1.8.1-এর direct_answer()-এ no-grid হলে দ্বিতীয় no-thinking commit আছে, কিন্তু সেটি চালাতে deadline-এর অন্তত ১২০ সেকেন্ড বাকি থাকতে হয়। তাই শেষ মুহূর্তে cut-free answer নিশ্চিত নয়।
এই বক্তব্যটি সংশোধন করা প্রয়োজন: “শেষ মুহূর্তের request-ও একটি উত্তর দেবে” নয়; “পর্যাপ্ত budget ও server throughput থাকলে উত্তর দেওয়ার সম্ভাবনা বাড়বে।”
৫. Deadline-এর আরেকটি HIGH bug হওয়ার সম্ভাবনা
v1.8.1-এর launch cell-এ:
cfg["deadline_epoch"] = START + 12 * 3600 - 25 * 60



আর main process চালানোর সময়:
timeout=max(    60,    cfg["deadline_epoch"] + 15 * 60 - time.time())



বর্তমানে এটি ১১:৩৫ + ১৫ মিনিট = ১১:৫০-এ process timeout দেয়।
কিন্তু v1.8.2-এ যদি শুধু deadline_epoch-কে ১১:৪৩ করা হয়, তাহলে একই calculation থেকে process timeout হবে ১১:৫৮। আপনার পরিকল্পিত ১১:৫০ নয়।
এছাড়া AGENT-এর deadline-ও এই shared deadline_epoch থেকে নির্ধারিত হয়। একটাই variable পরিবর্তন করলে AGENT/DIRECT split-ও অনিচ্ছাকৃতভাবে বদলাতে পারে।
Fix: চারটি deadline আলাদা করুন
# All deadlines are absolute Unix timestamps.run_start = STARTagent_stage_end = (    run_start + agent_stage_budget_seconds)rescue_admit_end = run_start + (    11 * 3600 + 34 * 60)rescue_output_end = run_start + (    11 * 3600 + 43 * 60)process_hard_stop = run_start + (    11 * 3600 + 50 * 60)assert (    agent_stage_end    < rescue_admit_end    < rescue_output_end    < process_hard_stop)



agent_stage_budget_seconds বর্তমান policy অনুযায়ী নির্ধারিত হবে; ১১:৪৩ deadline বসানোর কারণে সেটি নিজে বদলাবে না।
Main subprocess timeout হিসাব করবেন process_hard_stop থেকে। DIRECT request-এর timeout হিসাব করবেন rescue_output_end থেকে। Independent timeout-এর জন্য safety margin থাকবে।
আরেকটি আবশ্যক বিষয়: worker timeout হলেও submission.json-এ আগের completed outputs যেন সংরক্ষিত থাকে। বর্তমান incremental writing mechanism এই জায়গায় সহায়ক।

৬. আমি যে Adaptive Budget Algorithm নেব
বর্তমান proposal বাদ না দিয়ে এটিকে একটু উন্নত করব।
তিনটি মূল নিয়ম:
1. Early stage: 49,152 thinking tokens বজায় থাকবে, যদি predicted deadline-এ শেষ করার পর্যাপ্ত সময় থাকে।
2. Late stage: Remaining-time-aware smaller budget, কিন্তু full grid লেখার জন্য পৃথক answer reserve।
3. Emergency stage: Valid grid তৈরির ন্যূনতম সময়ও না থাকলে নতুন long-thinking request শুরু নয়; চাইলে আলাদা পরীক্ষিত short/no-thinking mode।
একটি reference budget calculator:
def choose_direct_budget(    now,    output_deadline,    stream_tok_s,    prompt_overhead_s,    answer_reserve_tokens,    safety_seconds=60,    max_think=49152,    min_think=512,):    remaining = (        output_deadline        - now        - prompt_overhead_s        - safety_seconds    )    if remaining <= 0:        return None    safe_rate = max(0.0, float(stream_tok_s))    if safe_rate == 0:        return None    token_capacity = int(remaining * safe_rate)    # Keep enough tokens for the final output.    usable_think = token_capacity - answer_reserve_tokens    if usable_think < min_think:        return None    think = min(max_think, usable_think)



এটি scheduler-এর budget calculation; complete production implementation নয়। এখানে stream_tok_s অবশ্যই conservative measured estimate হবে।
গুরুত্বপূর্ণ: answer_reserve_tokens কেবল ৮,১৯২ স্থির রাখব কি না, সেটা EVAL records থেকে নির্ধারণ করব। বড় ARC grid-এর জন্য বেশি output room, ছোট grid-এর জন্য কম reserve সম্ভব। তবে prompt-এ output shape অজানা থাকলে অতিরিক্ত confidence দেখানো যাবে না।
এই function None দিলে scheduler পরবর্তী query শুরু না করে অন্য কাজ করতে পারবে।
৭. Metrics কীভাবে যোগ করবেন
শুধু রানের শেষে /metrics নিয়ে report বানাবেন না। কারণ request শেষ হয়ে গেলে kv_cache_usage_perc কমে যেতে পারে, এবং peak memory pressure হারিয়ে যাবে।
vLLM 0.19.1-এ প্রয়োজনীয় Prometheus metrics পাওয়া যায়। 

vLLM
+2



Required telemetry
Metric	কী সংরক্ষণ করবেন
GPU KV cache capacity	Startup log থেকে resolved value
vllm:kv_cache_usage_perc	প্রতি ৩০–৬০ সেকেন্ডে snapshot, peak ও p95
vllm:num_preemptions_total	Run start/end counter delta
vllm:num_requests_running	Active concurrency
vllm:num_requests_waiting	Queue pressure
vllm:spec_decode_num_draft_tokens_total	MTP proposed tokens
vllm:spec_decode_num_accepted_tokens_total	MTP accepted tokens
Generation throughput	Tokens per second, time series
Prefix cache	Queries, hits, hit rate

Metrics-এর exact exported names একবার real /metrics response দিয়ে যাচাই করুন, বিশেষ করে labels ও suffix। Startup KV-capacity message log থেকে parse করবেন; /metrics-এর percentage gauge-কে মোট token capacity ধরে নেবেন না।
Validation: Fake metrics server-এ ২০%, ৮০%, ৪০%, ০% usage ফিরিয়ে দিন। Report-এ peak ৮০% হতে হবে, final ০% নয়। Counter ২ থেকে ৫ হলে preemption delta ৩। Server restart হলে পুরোনো counter-এর সঙ্গে নতুন counter সরাসরি বিয়োগ করবেন না।
Metrics সংগ্রহ ব্যর্থ হলেও solver চলতে পারে, কিন্তু report-এ metrics_unavailable লিখতে হবে। Zero preemption বানিয়ে দেখানো যাবে না।

৮. E2E ১০ মিনিট থেকে ৫ মিনিট
এখানেও আমি পুরোপুরি একমত নই।
v1.8.1-এর agent_e2e()-তে Python-call ও submit-call, দুটি successive inference দরকার। আবার xhigh ব্যর্থ হলে medium এবং low চেষ্টা করার ব্যবস্থা আছে।
তাই ৫ মিনিটে তিনটি reasoning-effort configuration-এর কার্যকারিতা নির্ভরযোগ্যভাবে যাচাই করা কঠিন হতে পারে।
আমার প্রস্তাব: ৫ মিনিটকে initial soft budget করুন। এই সময়ের মধ্যে tool call সফল হলে কিন্তু submit roundtrip অসম্পূর্ণ থাকলে, সর্বোচ্চ আগের ১০ মিনিট পর্যন্ত bounded extension দিন—শুধু sufficient overall runtime reserve থাকলে।
অন্যদিকে HTTP 400, invalid parameter বা deterministic failure হলে ৫ মিনিটের আগেই terminate করুন। এমন failure-এ অপেক্ষা করার লাভ নেই।
এতে healthy কিন্তু slow AGENT ভুলভাবে reject হওয়ার ঝুঁকি কমবে এবং দ্রুত failure-এর ক্ষেত্রে GPU সময় বাঁচবে।

৯. পরিবর্তনের ফলে বাস্তবে কত লাভ হতে পারে?
এখানে expected engineering gain ও leaderboard gain আলাদা করা জরুরি।
পরিবর্তন	Engineering লাভ	LB-এর প্রমাণ
KV telemetry	Capacity planning, preemption debugging	সরাসরি score gain নেই
MTP3	আগের প্রমাণিত throughput configuration পুনর্ব্যবহার	নতুন server-এ gain অজানা
Late admission	Previously idle GPU slots ব্যবহার	Additional correct outputs হলে gain
Adaptive budget	Hard-stop waste কমার সম্ভাবনা	Short reasoning accuracy unmeasured
E2E soft limit	Startup overhead কমতে পারে	Scoring-path gain পরোক্ষ
একটি মূল ব্যাপার মনে রাখতে হবে: DIRECT v1.4.4-এর EVAL12-তে মোট তিনটি correct test output ছিল। এর মধ্যে দুটির natural thinking completion প্রায় ২৬k ও ৩৭k tokens-এ ঘটেছে। বাকি একটি দীর্ঘ reasoning-এর পর forced-close থেকে এসেছে।
তাই ১১:৩৩-এর ২–৩k reasoning budget দিয়ে অনেক নতুন correct answer পাওয়া যাবে, এমন কোনো evidence নেই।
শেষ মুহূর্তে কয়েকটি extra grid তৈরি হওয়া আর leaderboard বাড়া এক নয়। কম budget-এ generated grids প্রায়ই ভুল হলে এই optimization throughput-এ লাভ দিলেও accuracy-তে খুব সামান্য লাভ দিতে পারে।
অপরদিকে যদি late request যোগ করার ফলে আগে চলমান বড় reasoning requests-এর speed কমে যায় বা preemption শুরু হয়, তাহলে মোট score কমারও সম্ভাবনা আছে।
এজন্য late admission শুধু ফাঁকা slot থাকলেই নয়, predicted completion এবং KV pressure দুটো বিবেচনা করে করতে হবে।

১০. নতুন DRYRUN-এ যে regression tests বাধ্যতামূলক
আমি এই আটটি test যোগ করতাম।
Test	Input scenario	Expected
T1	Dynamic 15k thinking budget	HTTP request-এ budget 15k, 49,152 নয়
T2	Thinking budget server reject করে	Safe failure/fallback; silent unlimited retry নয়
T3	Time ১১:৩৪, slow stream	Unsafe request reject
T4	Time ১০:৫০, fast stream	Budget maximum 49,152, context-safe
T5	Throughput মাঝপথে ৩০% পড়ে	Forecast adjust; output deadline রক্ষা
T6	KV usage ৯৫%, preemption বাড়ছে	Late admission throttle
T7	Rescue deadline ১১:৪৩	Process timeout এখনো ১১:৫০
T8	Interrupted DIRECT request	আগে সম্পন্ন output submission-এ অক্ষত
এগুলোর সঙ্গে আরও দুটি integration condition থাকবে: EVAL12-এর original AGENT scoring path বদলাবে না, এবং hidden rerun fallback-এ AGENT failure হলে DIRECT একই xhigh settings দিয়ে চালু থাকবে।
শুধু fake-server DRYRUN দিয়ে GPU-specific performance প্রমাণ করা যাবে না। DRYRUN control flow যাচাই করবে। MTP3 compatibility, actual KV capacity, deadline forecast accuracy ও valid-grid rate real vLLM run-এ মাপতে হবে।

১১. আমার চূড়ান্ত সিদ্ধান্ত
Metrics + MTP3
GO

আগের proven DIRECT configuration reuse করুন; peak KV, preemption ও accepted tokens মাপুন।

Late admission
CONDITIONAL GO

১১:৩৪ upper bound রাখুন, কিন্তু শুধু ঘড়ি দেখে নতুন request নেবেন না। Completion forecast ও memory pressure বিবেচনা করুন।

Dynamic thinking budget
FIX FIRST

Per-request thinking-token override, output reserve, deadline separation এবং failure handling ছাড়া implement করবেন না।

E2E ৫ মিনিট
SOFT LIMIT

দ্রুত failure-এ exit, genuine progress থাকলে bounded extension।

আমার পছন্দের final strategy
DIRECT v1.4.4-এর 26.94 baseline যতটা সম্ভব অক্ষত রাখুন। MTP3, fp8 KV, ১৬ concurrency, prefix cache, proven prompt/parser—এসব পরিবর্তন করার দরকার নেই।
তার ওপর নতুন দুটি জিনিস বসান: পর্যবেক্ষণযোগ্য runtime telemetry এবং deadline-aware admission control। তারপর dynamic thinking budget আলাদাভাবে test করুন। সব একসঙ্গে বদলালে LB বেড়েছে বা কমেছে কেন, সেটা বোঝা কঠিন হবে।
সবচেয়ে গুরুত্বপূর্ণভাবে, v1.4.4-এর global drain ও v1.8.2-এর per-request budget মোটেই সমতুল্য নয়। প্রথমটি চলমান inference-এর actual progress দেখে force-close করত; দ্বিতীয়টি শুরুতেই ভবিষ্যৎ speed অনুমান করে। দ্বিতীয়টির ঝুঁকি বেশি, যদিও ঠিকভাবে implement করলে tail utilisation ভালো হতে পারে।
আমার রায়: পরিকল্পনাটি engineering-এর দিক থেকে লাভজনক, কিন্তু সংশোধন ছাড়া ১২ ঘণ্টার submission-ready নয়। উপরের সংশোধন ও tests পাস করলে এগোনো যুক্তিসংগত; ৬০% লক্ষ্য অর্জনের জন্য শুধু এই scheduling পরিবর্তন যথেষ্ট হবে না—AGENT-এর বাস্তব correct-answer gain-ও প্রমাণ করতে হবে।
</pasted_content id="1891">

"

জিপিটি বললো