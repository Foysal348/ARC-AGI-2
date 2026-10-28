# GPT audit 3 of PLAN_E17 (pasted by the user 2026-10-26, verbatim, Bengali original)

Note: the same message also re-pasted audit 1 unchanged (identical to `GPT_E17_PLAN_AUDIT_1_2026-10-26.md`), so it is not saved again.

E17 Pre-GPU Research Audit — ৮টি প্রশ্নের উত্তর
পর্যালোচনার ভিত্তি: E17 plan: GPU tests for the SGLang silent-accuracy suspects, SGLang v0.5.21 source, ২০২৬ সালের অক্টোবরের upstream PR ও regression tests।
E17 পরিকল্পনা সংশোধন প্রয়োজন

## প্রথমে চারটি গুরুত্বপূর্ণ সিদ্ধান্ত
1. বর্তমান T1-কে deterministic reproduction বলা যাবে না। শুধু SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1, একটি seed request এবং warm repeat দিলেই stale checkpoint দান ও পুনরুদ্ধার ঘটেছে—এটি নিশ্চিত হয় না। Upstream-এর working reproducer-এ নির্দিষ্ট prefix topology, checkpoint length, overlap timing এবং memory-slot pressure রয়েছে।
2. Upstream-এ প্রায় হুবহু ব্যবহারযোগ্য একটি regression recipe আছে। PR #43567-এর TestMambaLazyInplaceKL একই prompt-এর 255-token distribution cold/warm অবস্থায় তুলনা করে। Buggy run-এ KL 0.649448, patched run-এ 0, এবং cached-prefix length যথাক্রমে 192 বনাম 128।
3. পূর্ণ vocabulary logprobs পেতে CUDA hook বাধ্যতামূলক নয়। SGLang-এর upstream test নিজেই /generate-এ token_ids_logprob=list(range(vocab_size)) দিয়ে full-vocabulary KL গণনা করে। তবে 248k-token vocabulary-তে এটি শুধু অল্প কয়েকটি scoring position-এর জন্য ব্যবহার করা উচিত, long generation-এর প্রতিটি token-এ নয়।
4. আরেকটি প্রাসঙ্গিক fix আছে: #43559। এটি #43567 থেকে আলাদা—consumed Mamba prefix lock release করে donation allocation retry ঠিক করে। ১৮ অক্টোবর merge হয়েছে। আপনার source-এ crash দেখা না গেলেও E17-এর memory-pressure পরীক্ষায় দুটিকে আলাদা রাখা জরুরি।
সুপারিশ: প্রথমে upstream-এর ছোট deterministic test নিজের 0.5.21 stack-এ adapt করুন। তার পরে DFlash speculative path, তারপর natural 16-stream frequency পরীক্ষা। এই ক্রমে bug reproduction, production exposure এবং leaderboard impact একে অপরের সঙ্গে গুলিয়ে যাবে না।

## প্রশ্ন ১ — T1 কি সত্যিই #43567 trigger করবে?
রায়: বর্তমান T1 যথেষ্ট নয়। Upstream-এর regression sequence দিয়ে T1 বদলাতে হবে।
PR #43567-এর bug trigger হওয়ার জন্য চারটি ঘটনা একসঙ্গে প্রয়োজন:
1. একটি valid cached Mamba checkpoint থাকে, যার সঙ্গে পুরোনো mamba_last_track_seqlen যুক্ত।
2. Decode অথবা verify নতুন tracking boundary-তে পৌঁছায়; optional pending slot allocate করা যায় না।
3. ফলে নতুন recurrent state পুরোনো keep slot-এই লেখা হয়, কিন্তু unpatched code পুরোনো checkpoint length valid রেখে দেয়।
4. Overlap scheduler-এর deferred prefill result ওই slot পুরোনো length-এর অধীনে radix cache-এ donate করে। পরে request ওই prefix match করে ভুল state restore করে।
শুধু allocation fail হওয়া bug reproduction নয়। পূর্ববর্তী checkpoint না থাকলে, donation skip হলে, অথবা পরের request নিরাপদ ছোট prefix match করলে KL প্রায় শূন্য থাকতেই পারে।

Upstream-এর verified reproduction — TestMambaLazyInplaceKL-এর মূল recipe:
```
def warm_prefixes(generate, flush):
    flush()
    generate([44] * 128, max_new_tokens=1)
    generate([42] * 128, max_new_tokens=1)
prompt = [42] * 128 + [43] * 127  # 255 tokens
# Reference
warm_prefixes(generate, flush)
reference = generate(prompt, max_new_tokens=1, score_full_vocab=True)
assert reference["meta_info"]["cached_tokens"] == 128
# Seed the corruption opportunity
warm_prefixes(generate, flush)
seed = generate(prompt, max_new_tokens=32)
assert seed["meta_info"]["cached_tokens"] == 128
# Same next-token position as reference
reused = generate(prompt, max_new_tokens=1, score_full_vocab=True)
assert reused["meta_info"]["cached_tokens"] >= 128
compare_full_vocab_kl(reference, reused)
```
এখানে generate() ও flush() upstream test-এর wrappers; নতুন notebook-এ HTTP /generate ও /flush_cache-এর সঙ্গে bind করতে হবে।
Source-এর exact settings: mamba strategy: extra_buffer_lazy; track interval: 128; max Mamba slots: 4; max running requests: 1; overlap scheduling: enabled; decode CUDA graph: enabled; temperature: 0; ignore_eos: true.
Buggy upstream result: Reference cached prefix: 128 tokens। Warm reused prefix: 192 tokens, ভুল checkpointসহ। Full-vocabulary KL: 0.649448।
Patched: Safe reused prefix: 128 tokens। KL: 0।

আপনার Qwen3.8-27B-তে কী পরিবর্তন? প্রথম test-এ production 16 streams রাখার দরকার নেই। বরং 1 running / minimum feasible Mamba slots দিয়ে upstream reproduction নকল করুন। 4-slot configuration আপনার model-এর জন্য launch না হলে minimum supported slots নিন এবং actual available/evictable slots log করুন।
আমার সংশোধিত T1:
| Arm | Cache strategy | Patch | Failure mode |
|---|---|---|---|
| A | lazy | Unpatched | Natural slot pressure |
| B | lazy | #43567 | Same natural pressure |
| C | lazy | Unpatched | Forced allocation failure, targeted diagnostic |
| D, optional | lazy | #43567 | Forced allocation failure |
A/B primary। C/D diagnostic। Upstream-এর নিজস্ব reproducer natural pressure ব্যবহার করেছে; forced failure বাধ্যতামূলক নয়।
Forced failure দিলে slot-এর reuse topology বদলে যেতে পারে। ফলে negative test ambiguous হবে।
GO gate: Allocation failure confirmed, unsafe checkpoint donation confirmed, subsequent cache hit confirmed—এর পরে distribution comparison। তিনটি event-এর কোনোটিই না থাকলে ফল হবে INCONCLUSIVE, PASS নয়।

## প্রশ্ন ২ — DFlash speculative path-এর deterministic test
এখানে ordinary decode ও speculative verification-এর মধ্যে একটি মৌলিক পার্থক্য রয়েছে। Ordinary decode প্রতি step-এ একটি token এগোয়। DFlash W6 এক verify step-এ একাধিক accepted token commit করতে পারে। তাই boundary-crossing event-এর হিসাব আলাদা।
SGLang PR #30437-এর speculative lazy mechanism অনুযায়ী সম্ভাব্য crossing আগেই আন্দাজ করা হয়: floor(L/I) ≠ floor((L+2D)/I), যেখানে L হলো committed length, I tracking interval এবং D সর্বোচ্চ speculative draft tokens। এই widened window overlap scheduler-এর এক-step lag বিবেচনা করে। DFlash lazy support পরে PR #34763-এ যোগ হয়েছে।
DFlash W6-এর জন্য প্রস্তাবিত test — প্রথমে: mamba-track-interval = 64; DFlash width = 6; max-running-requests = 1; extra_buffer_lazy; overlap = ON; temperature = 0; ignore_eos = true. এই 64 একটি diagnostic setting, production-এর default নয়।
Committed length boundary-এর কাছাকাছি আনুন, উদাহরণস্বরূপ 64-এর গুণিতকের 1–12 tokens আগে। তবে client-side prompt length দিয়ে অনুমান না করে scheduler-এর actual kv_committed_len log করতে হবে।
Test sequence:
1. Safe prefix warm করুন।
2. Seed request-কে 128–192 output tokens generate করতে দিন, যাতে একাধিক crossing-এর সুযোগ হয়।
3. Targeted failure branch-এ সত্যিই new_slot is None হয়েছে কি না দেখুন।
4. Verify শেষে crossing committed হয়েছে কি না দেখুন।
5. Request active থাকা অবস্থায় বা finishing-এর সময়ে cache donation path inspect করুন।
6. একই exact prompt/continuation warm এবং cold অবস্থায় score করুন।
এখানে fixed token count-ই সবচেয়ে গুরুত্বপূর্ণ নয়; actual boundary crossing এবং donation event-ই decisive।
DFlash-specific instrumentation — নিচের fields log করুন:
```
{"request_id": rid, "path": "mamba_lazy_spec_prepare", "kv_committed_len": committed_len, "track_interval": interval, "max_draft_tokens": draft_width, "in_window": in_window, "pending_alloc_failed": failed, "keep_slot": keep_slot, "planned_write_slot": planned_slot, "accepted_count": accepted_count, "crossed_confirmed": crossed, "finished": finished, "donation_skipped": donation_skipped, "donated_checkpoint_len": donated_len}
```
এই fields-এর সবগুলো কোনো একটি public API সরাসরি দেয় না। Scheduler এবং result-processing code-এর নির্দিষ্ট জায়গায় debug-only instrumentation করতে হবে।
আরেকটি সতর্কতা: পুরোনো upstream NEXTN test-এ SGLANG_TEST_SKIP_CACHE_HIT_ASSERT=1 ব্যবহার করা হয়েছিল। এটি negative-control stress-এর জন্য উপযোগী হতে পারে, কিন্তু cache hit সত্যিই ঘটেছে কি না যাচাইয়ের বিকল্প নয়।
DFlash W6 + exact #43567 + L4-এর deterministic GPU reproduction: UNVERIFIED। উপরোক্ত procedure হলো source-grounded adaptation।

## প্রশ্ন ৩ — Wrong Mamba state restore ধরার সবচেয়ে ভালো signal
আমি তিন-স্তরের evidence ব্যবহার করব।
| Evidence | কী প্রমাণ করে | গুরুত্ব |
|---|---|---|
| State provenance / slot generation ID | ভুল logical checkpoint কোন physical slot থেকে restore হয়েছে | সবচেয়ে শক্ত mechanistic evidence |
| Full-vocabulary next-token KL | একই prefix-এ model distribution বদলেছে | সবচেয়ে শক্ত practical numerical evidence |
| Generated text / ARC accuracy | downstream outcome বদলেছে | সবচেয়ে গুরুত্বপূর্ণ competition evidence, কিন্তু root-cause isolation কম |
A. Full-distribution KL — Reference হলো cold বা safely cached একই token position। Warm request-এর distribution-এর সঙ্গে তুলনা: D_KL(P_ref || P_warm) = Σ_i P_ref(i) log(P_ref(i)/P_warm(i)). এটি upstream-এর official method।
আপনার ক্ষেত্রে threshold অন্ধভাবে 1e-4 করবেন না। FP8 KV, FP16 SSM, INT4 Marlin ও TP4-তে A/A numerical variation মাপতে হবে। Upstream-এর আরেক speculative truncation regression-এ valid cache reuse-এর KL প্রায় 0.000235–0.000942 ছিল এবং threshold 0.005 ব্যবহার করা হয়েছিল। অর্থাৎ test-specific baseline ছাড়া universal KL threshold সঠিক নয়।
B. State provenance — প্রতিটি Mamba slot-এর জন্য debug-only generation counter রাখা যায়:
```
# Diagnostic concept: not production SGLang API.
slot_version[slot_id] += 1
checkpoint = {"slot_id": slot_id, "slot_version": slot_version[slot_id], "logical_length": committed_length, "request_id": request_id}
```
Donation-এর সময় slot ID/version/length record করুন। Restore-এর সময় cached metadata-র slot generation-এর সঙ্গে actual slot generation মিলিয়ে দেখুন।
এটি একটি নতুন instrumentation design, upstream-এ built-in validator আছে—এমন প্রমাণ পাইনি। বিশেষ করে CUDA asynchronous writes-এর সঙ্গে generation counter-এর ordering ঠিক না রাখলে false results আসতে পারে। তাই প্রথমে length validity এবং donation provenance log করুন; GPU state tensor hash তুলনা পরে, শুধুমাত্র isolated diagnostic build-এ।
C. Actual state tensor comparison — সম্ভব, তবে আপনার প্রথম E17 session-এর জন্য প্রয়োজন নেই। Recurrent state বহু layer এবং convolution/SSM buffers জুড়ে থাকে; state layout, timing ও synchronization না মেলালে raw tensor comparison-ও বিভ্রান্তিকর হতে পারে।
সুপারিশ: প্রথমে cached length + exact failure/donation event + full-vocab KL। এই তিনটি মিলে root cause শক্তভাবে আলাদা করতে পারে।

## প্রশ্ন ৪ — SGLang 0.5.21-এ full-vocabulary logits কীভাবে পাবেন?
সবচেয়ে উপযোগী উত্তর: token_ids_logprob। Upstream-এর test_mamba_cache_kl.py নিজেই ব্যবহার করে:
```
payload = {"input_ids": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}, "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": list(range(vocab_size))}
```
Response: `rows = result["meta_info"]["output_token_ids_logprobs"]`
প্রতিটি returned row-এ requested token IDs-এর logprob ও IDs পাওয়া যায়। Source test ordered vocabulary, finite values এবং probability sum যাচাই করে।
| পদ্ধতি | সুবিধা | অসুবিধা | রায় |
|---|---|---|---|
| return_logprob only | খুব কম overhead | শুধু generated token probability; full KL নয় | যথেষ্ট নয় |
| top_logprobs_num=20/100 | দ্রুত shortlist, top-1 changes | Missing probability mass; exact KL নয় | Screening |
| top_logprobs_num=vocab_size | Full distribution theoretically | বড় topk, sorting/gather ও serialization overhead | এড়িয়ে চলুন |
| token_ids_logprob=all IDs | Exact distribution test; upstream-verified | বড় request/response JSON | Best for 1-token probes |
| Server-side logits hook | CPU transfer কমানো যায়, custom KL | Inference internals patch এবং GPU synchronization risk | Later |
248,320 vocabulary হলে একটি FP32 vector প্রায় 0.95 MiB। IDs, Python objects ও JSON encoding যোগ হলে HTTP payload কয়েক MiB হবে; exact size serializer অনুযায়ী বদলাবে।
Full KL computation:
```
import numpy as np
def full_logprobs(result, vocab_size):
    rows = result["meta_info"]["output_token_ids_logprobs"]
    assert len(rows) == 1
    pairs = rows[0]
    assert len(pairs) == vocab_size
    ids = [int(x[1]) for x in pairs]
    assert ids == list(range(vocab_size))
    lp = np.asarray([float(x[0]) for x in pairs], dtype=np.float64)
    assert np.isfinite(lp).all()
    lp -= np.logaddexp.reduce(lp)
    return lp
def kl_reference_to_warm(ref_lp, warm_lp):
    p = np.exp(ref_lp)
    return float(np.sum(p * (ref_lp - warm_lp)))
```
প্রয়োগ: প্রতি server-এ 8 prompt × 1 scoring position। দীর্ঘ continuation-এর সব 256 position-এ full-vocabulary JSON ফেরত নেওয়ার দরকার নেই। Long-context test-এ প্রথমে 8k bucket-প্রতি 1–2 selected position score করুন। তারপর suspicious position-এ প্রয়োজনে বিস্তারিত trace।
আর top-k union-কে প্রতিটি distribution-এর ওপর নতুন করে normalize করে পাওয়া KL full-vocab KL নয়; missing tail mass ভিন্ন হলে সেটি misleading হতে পারে।

## প্রশ্ন ৫ — Natural allocation failure frequency কীভাবে মাপবেন?
বর্তমান পরিকল্পনার wrapper যথেষ্ট নয়। ScheduleBatch.mamba_lazy_prealloc_at_boundary() call count আর new_slot is None count এক জিনিস নয়। শুধু function wrapper দিয়ে internal local new_slot দেখা যায় না। তাই দুই স্তরে instrumentation প্রয়োজন:
1. Allocation site-এর ভেতরে actual success/failure।
2. Cache donation এবং later match/restore-এর provenance।
Recommended event chain: boundary_reachable → optional_alloc_attempt → optional_alloc_failed → keep_slot_written_in_place → checkpoint_donation_attempt → checkpoint_donated(length, slot) → later_prefix_match(length) → state_restored(slot, length) → KL compared against cold
এগুলো আলাদা counters রাখতে হবে। একটি failure harmless-ও হতে পারে।
```
events = {"boundary_windows": 0, "optional_alloc_attempts": 0, "optional_alloc_failed": 0, "inplace_tracking": 0, "checkpoint_invalidated": 0, "donation_attempted": 0, "donation_skipped": 0, "checkpoint_donated": 0, "later_restored": 0, "stale_length_candidate": 0}
```
প্রতিটি event-এ request ID, slot ID, intended/committed length, cache node identity এবং monotonically increasing local event index দিন।
Critical: forced failure environment variable internal allocation call bypass করতে পারে। তাই mamba_allocator.alloc wrapper-এ failure count শূন্য পেলেও failure হয়নি—এমন বলা যাবে না। Allocation branch-এর new_slot is None outcome-ই primary counter।
কেন mamba_usage যথেষ্ট নয়? আপনার নথির source-derived হিসাব: mamba_usage = (size − available − evictable)/size. সুতরাং usage 0.70 হলেও available=0 সম্ভব, যদি অনেক cached checkpoints evictable থাকে।
Natural-pressure detection-এর জন্য অবশ্যই log করুন: available_slots, evictable_slots, locked_slots, live_request_slots, optional_alloc_attempted, optional_alloc_failed.
আরেকটি observation: upstream lazy-MTP ছোট-pool test-এ 68 slots/16 running থাকলেও cache-hit KL পরীক্ষা সম্ভব হয়নি, কারণ repeat-এর আগে cached entries evicted হয়েছিল। অর্থাৎ সবচেয়ে বেশি allocation pressure হলেই stale cached state অবশ্যই restore হবে না।
O2-এর সঠিক উত্তর হবে দুটি সংখ্যা: প্রতি 1,000 relevant boundary attempts-এ কত failure, এবং প্রতি 1,000 repeated-prefix opportunities-তে কত unsafe donation/restore candidate। শুধু failure count দিয়ে corruption frequency হিসাব করবেন না।

## প্রশ্ন ৬ — #43567-only patch নিরাপদ? 0.5.22 নাকি backport?
#43567-এর দুই-hunk patch — আপনার ZIP-এর backport-এ দুটি site-এ `else: req.kv.mamba_last_track_seqlen = None`. এটি upstream fix-এর মূল logic। Source compilation, idempotent apply এবং exact-match test আপনারা আগেই করেছেন। এটি minimum-risk candidate, কারণ version, CUDA kernels, DFlash2 এবং quantization stack অপরিবর্তিত থাকে। কিন্তু এর পূর্ণ GPU safety আপনার Qwen3.8-27B/L4-এ এখনো UNVERIFIED।
Finished request-এ length-0 insert — এটি স্বয়ংক্রিয় corruption নয়। Invalidation-এর উদ্দেশ্যই হলো পুরোনো checkpoint unsafe হলে সেটি আর valid length হিসেবে cache-এ না দেওয়া। আপনার নথিতে prepare_for_caching_req-এর None path existing behavior হিসেবে চিহ্নিত। অর্থাৎ এই fallback নতুন code path নয়। তবে length-0 বা no-donation outcome হলে নিচের invariants প্রয়োজন:
- Released Mamba slot পরে অন্য request-এর ব্যবহারে যাওয়ার আগে dangling radix reference থাকবে না।
- Unsafe checkpoint donation skip হবে।
- Prefix hit valid shorter checkpoint-এ fallback করবে।
- Cache miss হলেও fresh recomputation correct হবে।
এই চারটি actual E17 test-এ যাচাই না করে শুধু compile PASS-কে production safety বলব না।
#43559 কেন আলাদা? ১৮ অক্টোবর merged PR #43559, #43567-এর ওপর নির্ভর করে। Failed lazy donation-এ prefix lock release করে eviction/retry-এর পথ খুলেছে। Upstream-এর negative control-এ #43567 retained রেখেও #43559 সরালে Can not alloc mamba cache assertion ঘটেছে। এটি মূলত availability/crash risk, #43567-এর stale-checkpoint accuracy bug নয়। আপনাদের 62 server log-এ ওই crash দেখা যায়নি, তাই এখন দুটো fix একসঙ্গে backport করার প্রয়োজনীয়তা প্রমাণিত নয়।
| Branch | Purpose | Decision |
|---|---|---|
| SGLang 0.5.21 original | Historical baseline | Required |
| 0.5.21 + exact #43567 | Isolated correctness fix | Primary candidate |
| 0.5.21 + #43567 + #43559 | Extra donation-lock robustness | Only if targeted crash/pressure requires |
| SGLang 0.5.22 | Integration compatibility reference | Optional; not initial production choice |
0.5.22-তে অন্যান্য scheduler, speculative, kernel এবং cache changes রয়েছে। তাই 0.5.22-তে KL ভালো হলে improvement কোন নির্দিষ্ট fix থেকে এসেছে বোঝা যাবে না। আমার সিদ্ধান্ত: আগে 0.5.21 + #43567। Full version migration এখন নয়।

## প্রশ্ন ৭ — KL difference থেকে ARC solve-rate gain সবচেয়ে কম খরচে কীভাবে যাচাই করবেন?
প্রথমে তিনটি আলাদা প্রশ্নের উত্তর দরকার: Bug trigger হয় কি? Mechanistic test। বাস্তব load-এ trigger হয় কতবার? Exposure test। Fix করলে ARC solve-rate বাড়ে কি? Paired task evaluation। একটি small KL smoke-এ তৃতীয় প্রশ্নের উত্তর পাওয়া যাবে না।
প্রথম পর্যায়: 24–32 task, mechanism-sensitive subset। Duck-এর multi-turn, high-prefix-reuse এবং পূর্বে train-exact বা near-correct cases বেশি রাখুন। একই model, tokenizer, prompt, task order, sampling policy ও budget। Unpatched এবং patched arms-এর মধ্যে seed matched রাখুন। প্রতিটি task-এ: task_id, seed, unpatched_result, patched_result, first_executable_program_at, train_exact, host_verified, test_queries_correct_offline, tokens_used, cache_restore_events, optional_alloc_failures. এই ছোট test screening, score claim নয়।
দ্বিতীয় পর্যায়: EVAL60 বা EVAL120 — যদি প্রথমে meaningful numerical recovery পাওয়া যায়, তখন broader stratified set-এ score পরীক্ষা করুন।
| Sample | কী জানা সম্ভব |
|---|---|
| 12–24 tasks | Catastrophic corruption, obvious regressions |
| 32 tasks | Large paired gains detect করা সম্ভব; moderate gain অনিশ্চিত |
| 60 tasks | Moderate-to-large directional evidence |
| 120 tasks | বেশি স্থিতিশীল score comparison, তবু stochastic variation থাকবে |
উদাহরণ: 5টি task patched-only correct, 0টি unpatched-only হলে two-sided paired sign test p=0.0625। 10টি patched-only, 0টি unpatched-only হলে p≈0.00195। তাই ছোট positive changes দেখে নিশ্চিত improvement ঘোষণা করা যাবে না।
LB standard deviation 1.5 percentage points দিয়ে EVAL20/EVAL60-এর task-normalized sampling uncertainty সরাসরি নির্ধারণ করা যায় না। Paired disagreement, task difficulty এবং seed variance আলাদাভাবে মাপতে হবে।
কোন harness? প্রথমে Duck, কারণ long multi-turn prefixes বেশি। তারপর Poetiq S60-এ verify করুন—বিশেষ করে যদি production adoption Poetiq notebook-এ হয়।
একটি গুরুত্বপূর্ণ causal control: Same prompts, no-cache/target-only reference। Same prompts, lazy cache unpatched। Same prompts, lazy cache patched। এতে program-selection randomness আর cached state corruption কিছুটা আলাদা হবে।
Adoption criterion: Reproducible numerical correctness recovery, no new crash/throughput catastrophe, এবং paired task evaluation-এ কোনো material regression নয়। পজিটিভ LB gain থাকলে ভালো, কিন্তু সংখ্যাগত correctness fix-কে শুধুই ছোট noisy score test দিয়ে বাতিল করা উচিত নয়।

## প্রশ্ন ৮ — আর কোনো silent corruption mechanism বাদ পড়ছে?
একটি নতুন গুরুত্বপূর্ণ upstream lead পেয়েছি: PR #43387 — Speculative truncation checkpoint mismatch। ১৭ অক্টোবর merged। এটি no_buffer speculative path-এ target verify live state এগিয়ে যাওয়ার পরে shorter finished-cache key-এর অধীনে state donate করার সমস্যা ঠিক করে।
| Scenario | Before KL | After KL |
|---|---|---|
| NEXTN count, short finish | 0.876971 | 0.000235 |
| NEXTN story, short finish | 4.107031 | 0.000942 |
আপনার exact configuration: extra_buffer_lazy, no_buffer নয়। তাই #43387 সরাসরি production blocker নয়। তবে finished request, accepted speculative tokens এবং cache key length-এর invariant যাচাই করার মূল্যবান reference।
বাকি suspects-এর priority:
| Mechanism | Relevance | Cheapest decisive test |
|---|---|---|
| #36548 DFlash concurrent state drift | HIGH; DFlash W6/C16 | Serial বনাম 16-way, target-only/DFlash, frozen prompts |
| #43567 lazy checkpoint | HIGH; exact affected source path | T1/T2 revised |
| FP16 SSM accumulation | HIGH uncertainty; runtime FP16 | Same long continuation FP16 vs FP32 |
| FP8 KV scales | Medium; calibrated INT4 stack | Runtime loaded scales ও actual saturation/quant state |
| CUDA graph verify path | Medium; 1..17 graphs | Graph ON/OFF same prompts |
| Mid-thinking EOS/parser | Medium; agent score loss possible | CPU saved-token and stop-reason replay |
| #43559 donation lock assertion | Lower observed frequency; not silent output corruption | Natural slot-pressure + crash counters |
| #43387 no_buffer truncation | Not exact production mode | Do not add to initial E17 |
DFlash #36548-এ reporter একই configuration-এ serial 0/100 wrong বনাম concurrent DFlash 111/304 affected-subset wrong পেয়েছেন; no-spec ছিল 0/304। কিন্তু সেই report-এ SSM verify-kernel এবং quantized output head-এর ভিন্ন environment ছিল—তাই আপনার L4-তে ওই rate ব্যবহার করা যাবে না।
আর SGLang #38009-এ single-request thinking-mode DFlash ও target-only greedy divergence রিপোর্ট করা হয়েছে। এর অর্থ শুধু concurrency নয়, verify path নিজেই আলাদা numerical behavior তৈরি করতে পারে।
এ জন্য E17-এর DFlash arm-এ দুটি controls প্রয়োজন: Serial target-only বনাম serial DFlash। Concurrent target-only বনাম concurrent DFlash। শুধু concurrency-16 test থেকে error-এর কারণ concurrency নাকি speculative verification—আলাদা হবে না।

## সংশোধিত E17 GPU plan
বর্তমান 60–90-minute পরিকল্পনায় তিনটি engine mode (target-only ও DFlashসহ), তিনটি patch state এবং multiple long-load sweeps রাখলে startup/capture time-ই বড় অংশ খেয়ে ফেলতে পারে। আপনার historical first READY প্রায় 8–13 মিনিট। তাই সব পরীক্ষা একটি session-এ শেষ করা guaranteed নয়। আমি plan-টি এইভাবে ছোট করব।
- Stage A — CPU-only source gate (GO): Exact 0.5.21 source SHA, #43567 hunk presence, actual DFlash lazy code path, probe implementation এবং full-vocabulary API payload test। GPU ছাড়াই fake request/slot lifecycle test।
- Stage B — Mechanistic GPU test (First GPU priority): Upstream 128/255/32-token scenario। Original ও #43567 patched server; actual cached length, failure/donation events এবং full-vocabulary KL। প্রথমে ordinary decode দিয়ে known regression replicate; পরে DFlash W6।
- Stage C — Production exposure (Conditional): যদি Stage B bug দেখায়, তাহলে 16-way DFlash, 30k–60k context, 10–15-minute load। Natural alloc failure, stale-donation candidates, prefix restores ও correctness তুলনা।
- Stage D — Accuracy validation (Separate run): Stage C-তে material divergence পাওয়া গেলে 24–32 selected-task paired Duck screening, তারপর প্রয়োজনে EVAL60/120।
E17-এর required machine-readable result:
```
{"version": "0.5.21", "source_sha256": "<actual sha>", "patch_43567": "absent|present", "mode": "ordinary|dflash6", "forced_alloc_failure": false, "boundary_windows": 0, "optional_alloc_failed": 0, "inplace_tracking": 0, "unsafe_donation_candidates": 0, "later_restores": 0, "cached_tokens_reference": null, "cached_tokens_warm": null, "full_vocab_kl": null, "aa_kl": null, "gpu_crash": false, "verdict": "PASS|FAIL|INCONCLUSIVE"}
```
এটি schema example; unsafe_donation_candidates মানে instrumented condition, confirmed incorrect-state restore নয়। Real evidence না থাকলে field null অথবা INCONCLUSIVE রাখবেন।

## চূড়ান্ত GO/NO-GO
| Scope | Verdict |
|---|---|
| #43567 source defect ও exact backport | Confirmed |
| Original T1 seed/warm test যথেষ্ট | NO-GO as deterministic reproduction |
| Revised upstream-style T1 | GO for GPU test |
| DFlash speculative branch T1 | GO with boundary/donation instrumentation |
| Full-vocabulary KL through API | GO — upstream demonstrated |
| Natural frequency from usage metric alone | NO-GO |
| 0.5.21 + exact #43567 experimental branch | GO for validation |
| Immediate 0.5.22 production migration | NO-GO without compatibility tests |
| Immediate leaderboard uplift estimate | UNVERIFIED |
সবচেয়ে গুরুত্বপূর্ণ correction: E17-তে শুধু KL > 0 পেলেই bug ঘোষণা করবেন না। প্রমাণ করতে হবে একই token position, matched model and sampling configuration, cache hit, failure/donation provenance এবং A/A noise baseline। বিপরীতে KL≈0 পেলেও যদি unsafe donation path আদৌ trigger না হয়, পরীক্ষাটি clean PASS নয়—INCONCLUSIVE।
Verified public URLs:
- https://github.com/sgl-project/sglang/pull/43567
- https://github.com/sgl-project/sglang/pull/43559
- https://github.com/sgl-project/sglang/pull/43387
- https://github.com/sgl-project/sglang/pull/30437
- https://github.com/sgl-project/sglang/pull/34763
- https://github.com/sgl-project/sglang/blob/v0.5.22/test/registered/radix_cache/test_mamba_cache_kl.py
- https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/schedule_batch.py
- https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/layers/logprob_processor.py
- https://github.com/sgl-project/sglang/issues/36548
- https://github.com/sgl-project/sglang/issues/38009
- https://github.com/sgl-project/sglang/releases/tag/v0.5.22
কোনো notebook, patch বা production file পরিবর্তন করা হয়নি; GPU execution-ও করা হয়নি।
