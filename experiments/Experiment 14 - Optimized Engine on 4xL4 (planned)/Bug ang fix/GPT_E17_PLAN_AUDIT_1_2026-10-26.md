# GPT audit 1 of PLAN_E17 (pasted by the user 2026-10-26, verbatim, Bengali original)

E17 GPU Plan — Independent Pre-Run Audit
২৬ অক্টোবর ২০২৬ | SGLang 0.5.21 | Qwen3.8-27B INT4 + DFlash2 | 4× L4
NO-GO: T1 as currently designed
E17 গবেষণার দিক সঠিক, কিন্তু T1/T2-তে এমন measurement gaps আছে, যার কারণে #43567 bug উপস্থিত থাকলেও পরীক্ষাটি ভুলভাবে PASS করতে পারে। তাই পুরো GPU investigation বাতিল করার দরকার নেই; trigger এবং verification logic সংশোধন করে চালানো উচিত।
সবচেয়ে গুরুত্বপূর্ণ source-level discovery
SGLang PR #43567-এর author-এর মূল reproduction আপনার T1-এর চেয়ে অনেক বেশি নির্দিষ্ট:
- একটি 128-token cached prefix;
- ওই prefix থেকে 255-token prompt-এ extend;
- পরবর্তী 32 generated tokens;
- overlap scheduling এবং decode CUDA graphs;
- মাত্র 4টি Mamba slots;
- পুরোনো 192-token checkpoint ভুলভাবে restore হওয়ার ঘটনা।
Unpatched code-এ full-vocabulary KL divergence ছিল 0.6494482615; patched code-এ 0। অর্থাৎ কেবল allocation failure নয়, একটি ভুল 192-token checkpoint cache-এ ঢুকে পরে পুনরায় ব্যবহৃত হয়েছে—এটাই প্রমাণিত failure।

মূল প্রমাণ:
https://github.com/sgl-project/sglang/pull/43567
SGLang 0.5.21 source:
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/schedule_batch.py

আপনার T1-এ বর্তমানে 2k–8k prompt, max_new=1, তারপর vaguely defined seed এবং warm repeat আছে। এভাবে stale checkpoint তৈরি, radix node-এ publish এবং পরবর্তী request-এ restore—এই তিনটি ঘটনা ঘটেছে কি না নিশ্চিত করা যায় না।

আরেকটি বড় সিদ্ধান্ত: #43567-এর upstream author নিজেই বলেছেন, GPU end-to-end test ordinary decode path covered করেছিল; speculative verification path covered করেনি। ফলে DFlash-এর জন্য আলাদা deterministic reproduction প্রয়োজন।

## প্রশ্ন ১ — T1 কি সত্যিকারের stale-donation path trigger করবে?
উত্তর: বর্তমান T1-এ নিশ্চয়তা নেই।

v0.5.21 source অনুযায়ী bug-এর জন্য একটি নির্দিষ্ট event chain প্রয়োজন:
1. Request-এর জন্য valid Mamba checkpoint তৈরি হয়েছে, যার সঙ্গে একটি prefix length যুক্ত আছে।
2. Scheduler overlap-এর মধ্যে পুরোনো checkpoint buffer এখনও referenced থাকা অবস্থায় নতুন decode/verify শুরু হয়।
3. Lazy allocation-এর new_slot is None branch নেওয়া হয়।
4. Unpatched version পুরোনো keep slot overwrite করে, কিন্তু mamba_last_track_seqlen পুরোনো valid length-ই থাকে।
5. cache_unfinished_req → prepare_for_caching_req ওই পুরোনো length দিয়ে checkpoint publish/donate করে।
6. পরবর্তী matching request radix cache থেকে সেই checkpoint restore করে।

বিশেষ করে v0.5.21-এর cache_unfinished_req() বিভিন্ন component-এর returned cache length-এর minimum নিয়ে radix key বানায়। ফলে Mamba checkpoint length ভুল হলে cached prefix-এর logical depth ভুল হতে পারে।
Sources:
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/unified_radix_cache.py
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/unified_cache/components/mamba.py

সংশোধিত T1 protocol
| Step | কাজ | বাধ্যতামূলক evidence |
|---|---|---|
| 1 | Fresh unpatched/patch server, cache state reset | SHA256, patch status, model hash |
| 2 | একটি 128-token prefix warm করে valid checkpoint তৈরি | Real mamba_checkpoint_len=128, valid pool slot |
| 3 | একই prefix-সহ 255-token input process; 32-token generation | Overlap সক্রিয়, real decode progress |
| 4 | Targeted allocation failure | alloc_failed, request ID, old checkpoint length, keep slot ID |
| 5 | Overlap-এর মধ্যে donation | Published radix key length, Mamba slot ID এবং checkpoint validity |
| 6 | Identical 255-token prompt আবার query | Actual matching Mamba checkpoint depth, not just KV cache hit |
| 7 | Same checkpoint, cold recompute oracle | Full-vector logits difference / KL |

এখানে 128/255/32 সংখ্যাগুলো arbitrary নয়—upstream-এর demonstrated regression। তবে v0.5.21-এর আপনার active mamba_track_interval আলাদা হলে corresponding boundary-aligned lengths ব্যবহার করতে হবে।

নতুন গুরুত্বপূর্ণ সংশোধন: global forced-fail যথেষ্ট নয়
SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1 শুরু থেকেই চালু করলে checkpoint seeding-এর প্রয়োজনীয় normal allocation-ও ব্যর্থ হতে পারে। ফলে পুরোনো valid checkpoint কখনো তৈরি না-ও হতে পারে।
আর max_new=1 দিয়ে baseline নেওয়ার ফলে contaminated prefix cache তৈরি হতে পারে; যেটিকে পরে ভুল করে clean oracle ধরে নেওয়া হবে।
তাই আমার প্রস্তাব:
- আগে normal allocation দিয়ে valid 128-token checkpoint seed করুন।
- তারপর শুধু নির্বাচিত request-এর নির্বাচিত boundary crossing-এ একবার allocation failure inject করুন।
- তার পরের donation এবং restore record করুন।
- Clean reference সম্পূর্ণ পৃথক fresh-cache instance-এ তৈরি করুন।
এই request-scoped/one-shot fault injector হবে ছোট local diagnostic patch, production patch নয়। Process-wide environment flag নিয়েই তুলনামূলক forced-fail smoke রাখা যায়, কিন্তু সেটি negative result দিলে bug নেই বলে সিদ্ধান্ত দেওয়া যাবে না।
Confidence: Event chain সম্পর্কে high; 128/255/32 sequence আপনার Qwen3.8 INT4-এ একই ফল দেবে কি না UNVERIFIED।

## প্রশ্ন ২ — DFlash/EAGLE speculative path কীভাবে trigger করবেন?
এই প্রশ্নে সবচেয়ে গুরুত্বপূর্ণ source-level detail হলো:
```
def mamba_lazy_spec_in_window(    req, mamba_track_interval, max_draft_tokens):    seq_len = req.kv.kv_committed_len    window = 2 * max_draft_tokens    return (        seq_len // mamba_track_interval        != (seq_len + window) // mamba_track_interval    )
```
এটি SGLang 0.5.21-এর actual implementation। Source:
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/schedule_batch.py
আপনার DFlash width6-এ max_draft_tokens=6 হলে potential look-ahead window 12 committed tokens। কিন্তু এই window-এর মধ্যে boundary reachable হলেই boundary actually crossed হয়েছে—এমন নয়।
এটাই আপনার পরীক্ষার একটি বড় missing condition।
Correct speculative reproduction
1. Seed একটি valid tracked Mamba checkpoint।
2. DFlash W6 চালু রেখে committed length এমন জায়গায় আনুন, যেখানে next track boundary 1–12 tokens-এর মধ্যে।
3. Pending slot allocation failure inject করুন।
4. Verify করুন mamba_lazy_spec_prepare() আসলেই has_pending=False-সহ keep slot বেছে নিয়েছে।
5. commit_mamba_states_after_verify()-এ accepted length, crossed boundary, track step এবং commit target record করুন।
6. Actual state overwrite হয়েছে কি না, এরপর সেই state cache-এ donate হয়েছে কি না, এবং পরবর্তী request restore করেছে কি না যাচাই করুন।
Spec verification-এ persistent state প্রতি proposed draft token-এ সরাসরি advance হয় না; accepted states পরে commit হয়। তাই শুধু mamba_lazy_spec_prepare call count বা allocation failure count দিয়ে #43567-এর speculative corruption প্রমাণ করা যাবে না।
Source:
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/spec_utils.py
কত tokens লাগবে? Fixed কোনো universal token count নেই। একটি valid checkpoint, একটি reachable এবং actually committed boundary crossing, একটি overlapping donation এবং একটি cache restore দরকার। Test-only ছোট track interval ব্যবহার করে path coverage বাড়ানো যায়, কিন্তু সেটি production behavior-এর সমতুল্য নয়।
Recommended controls: DFlash W6, same-target non-speculative, এবং সম্ভব হলে EAGLE MTP—প্রতিটিতে same prefix/seed, একই state-depth ও matched sampling settings রাখুন। এগুলো তিনটি পৃথক paths; একটি পাস করলে অন্যটি পাস করেছে ধরে নেওয়া যাবে না।

## প্রশ্ন ৩ — Stale Mamba state restore-এর সবচেয়ে নির্ভরযোগ্য signal কী?
আমার সুপারিশ: তিন স্তরের evidence, শুধু logits KL নয়।
| Evidence | কী প্রমাণ করে | Reliability |
|---|---|---|
| A. Checkpoint lifecycle trace | পুরোনো checkpoint invalid হওয়া উচিত ছিল, কিন্তু old length নিয়ে donate/restore হয়েছে | Root-cause evidence — সবচেয়ে গুরুত্বপূর্ণ |
| B. Restored vs recomputed Mamba state | একই logical token depth-এ recurrent tensors আলাদা | Strong state-corruption evidence |
| C. Full logits comparison | ভুল restore model prediction পর্যন্ত প্রভাব ফেলেছে | Strong output-impact evidence |
A+B+C একসঙ্গে মিললে সবচেয়ে শক্তিশালী case হবে।
শুধু logits KL-এর সীমাবদ্ধতা
- KL শূন্যের কাছাকাছি হলেও সব internal states identical প্রমাণ হয় না।
- KL বড় হলেও সেটি #43567-এর জন্যই হয়েছে প্রমাণ হয় না; FP8 KV, SSM precision বা DFlash verification path-ও কারণ হতে পারে।
- top-k union-এর probability renormalize করে গণনা করা KL full-vocabulary KL নয়। বাদ পড়া vocabulary mass-এর কারণে ফল misleading হতে পারে।
State-level instrumentation
MambaComponent-এর cache metadata-তে stored mamba_value মূলত physical state-slot reference। শুধু slot index তুললে contents identical প্রমাণ হয় না। Underlying recurrent state ও convolution state একই logical depth-এ compare করতে হবে।
Recommended trace record:
```
{    "rid": rid,    "event": event_name,    "kv_committed_len": committed_len,    "checkpoint_len": checkpoint_len,    "mamba_keep_slot": keep_slot,    "mamba_pending_slot": pending_slot,    "alloc_failed": alloc_failed,    "inplace_write": inplace_write,    "donated_len": donated_len,    "donated_slot": donated_slot,    "restored_len": restored_len,    "restored_slot": restored_slot,    "cache_node_id": node_id,    "is_finished": is_finished,}
```
এটি proposed telemetry schema, কোনো বিদ্যমান upstream API নয়। প্রতিটি field সংশ্লিষ্ট call site-এ পাওয়া যায় কি না builder-এ যাচাই করতে হবে।
Strong test assertion: কোনো checkpoint যে logical prefix length-এর নামে restore হচ্ছে, তার recurrent state ওই prefix recompute করে পাওয়া state-এর সঙ্গে নির্দিষ্ট precision tolerance-এর মধ্যে মিলতে হবে।
State copy প্রয়োজন হলে scheduler metadata এবং TP model-worker-এর physical state extraction আলাদাভাবে করতে হবে। Scheduler থেকে কেবল slot index নিয়ে GPU tensor contents জানা সম্ভব নয়।
SGLang 0.5.21-এ production-ready built-in checkpoint-consistency assertion পেয়েছি—UNVERIFIED। Upstream #43567-এর dedicated KL regression আছে, কিন্তু সেটিকে production internal consistency assertion বলা যাবে না।

## প্রশ্ন ৪ — 248k vocabulary-এর full logits সবচেয়ে সস্তায় কীভাবে নেবেন?
সেরা পদ্ধতি: GPU-side, server-internal one-position logits capture। আপনার পরীক্ষায় প্রতিটি request-এর জন্য পুরো reasoning trace-এর full logits লাগবে না। প্রথম কয়েকটি critical restore positions যথেষ্ট।
খরচের তুলনা
| Method | সুবিধা | সমস্যা | Decision |
|---|---|---|---|
| return_logprob=True, top_logprobs_num=20/50 | সবচেয়ে সহজ, public API | Full distribution পাওয়া যায় না | Diagnostic |
| top_logprobs_num=248000 | ধারণাগতভাবে full top-k | Massive top-k, serialization, implementation limits; 0.5.21-এ tested নয় | NO-GO |
| token_ids_logprob over all IDs | Selected token logprobs API-supported | 248k IDs request/response; overhead ও limits | NO-GO |
| return_flat_raw_top_logprobs | Array/binary-shaped top-k transport path v0.5.21 source-এ আছে | Full-vocab raw logits guarantee নয় | Optional |
| Server-side raw logits capture | Full vector; sampler-এর আগের distribution | Small diagnostic source hook দরকার | GO |
v0.5.21-এর logprob input fields এখানে রয়েছে:
https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/io_struct.py
Sampling API documentation:
https://github.com/sgl-project/sglang/blob/main/docs/docs/basic_usage/sampling_params.mdx
Actual memory arithmetic (248,000 vocabulary ধরে):
| Capture | Approx size |
|---|---|
| 1 position, FP32 | 0.99 MB |
| 1 position, BF16 | 0.50 MB |
| 8 prompts × 1 position, FP32 | 7.94 MB |
| 16 requests × 256 positions, FP32 | 4.06 GB |
এগুলো শুধু dense numeric vectors-এর raw byte sizes; transfer, buffering, intermediate tensors ও metadata অন্তর্ভুক্ত নয়।
আপনার জন্য এক position × 8 prompts—খুবই manageable। 256 positions × 16 streams-এ full capture করা অপ্রয়োজনীয়।
Diagnostic implementation
Logits hook-এ target model-এর final LM-head logits, sampling transform-এর আগে, capture করুন।
```
import torch@torch.no_grad()def capture_logits(logits, path):    # logits: exactly one selected target-token position    x = logits.detach().float()    assert x.ndim == 1    assert torch.isfinite(x).all()    torch.save(x.cpu(), path)def compare_logits(a, b):    a = a.float()    b = b.float()    assert a.shape == b.shape    assert torch.isfinite(a).all()    assert torch.isfinite(b).all()    logp = torch.log_softmax(a, dim=-1)    logq = torch.log_softmax(b, dim=-1)    kl = torch.sum(logp.exp() * (logp - logq))    return {        "kl_p_q": float(kl),        "max_logit_diff": float((a - b).abs().max()),        "top1_changed": int(a.argmax() != b.argmax()),    }
```
এটি standalone comparator। Hook-টি exact SGLang output structure অনুযায়ী বসাতে হবে; full target logits ও DFlash draft logits যেন মিশে না যায়।
একটি বিশেষ সতর্কতা: TP4-এ প্রতিটি TP worker-এ একই result capture করলে duplicate বা partial tensor পাওয়ার ঝুঁকি আছে। Capture করার আগে final target logits সম্পূর্ণ vocabulary-এর কি না যাচাই করুন; প্রয়োজনে শুধু TP rank 0-এ write করুন।

## প্রশ্ন ৫ — Natural failure frequency কীভাবে মাপবেন?
আপনার proposed hook points সঠিক subsystem-এ আছে, কিন্তু শুধু function wrapper দিয়ে new_slot is None local variable নির্ভরযোগ্যভাবে গণনা সম্ভব নয়।
কারণ:
```
if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():    new_slot = Noneelse:    new_slot = pool.mamba_allocator.alloc(1)
```
Forced failure branch-এ allocator call হয়ই না। আবার function-level wrapper return value থেকেও failure বোঝা যাবে না, কারণ ওই functions সাধারণত failure result return করে না।
Exact instrumentation locations
1. mamba_lazy_prealloc_at_boundary: eligible boundary, skip due to existing pending, attempted allocation, failed allocation, invalidation।
2. mamba_lazy_spec_prepare: reachable boundary, pending present/absent, fallback selected।
3. Target verification commit: actual accepted length, real boundary crossing, state commit।
4. prepare_for_caching_req: valid checkpoint length বনাম invalid (None), requested donation।
5. cache_unfinished_req: effective radix insertion length।
6. Prefix matcher/load: Mamba checkpoint depth used on next request।
7. Allocator: available slots, evictable slots, protected slots, failed eviction/retry।
গুরুত্বপূর্ণ: available_size()==0 যথেষ্ট evidence নয়। Available zero হওয়ার পরও evictable slots থাকতে পারে; optional lazy allocation deliberately evict করে না। তাই সাধারণ availability এবং optional-allocation failure আলাদা counters চাই।
Suggested event counters
```
METRICS = {    "eligible_boundary": 0,    "spec_window_reachable": 0,    "optional_alloc_attempts": 0,    "optional_alloc_failed": 0,    "inplace_fallbacks": 0,    "actual_boundary_crossings": 0,    "donations_after_fallback": 0,    "stale_length_donations": 0,    "stale_checkpoint_restores": 0,    "cache_recompute_fallbacks": 0,}
```
এই counters request-scoped event IDs দিয়ে link করুন। Aggregate counts দিয়ে একই request-এর fail→donate→restore chain establish করা যায় না।
সবচেয়ে গুরুত্বপূর্ণ চারটি derived measurements
r1 = optional alloc failures / optional alloc attempts
r2 = stale donations / in-place fallbacks
r3 = stale restores / stale donations
r4 = affected outputs / completed scored outputs
r1 বড় হলেও r3=0 হতে পারে—তাহলে failures harmless recompute-এ শেষ হতে পারে। Production accuracy risk-এর জন্য r3 এবং r4 সবচেয়ে গুরুত্বপূর্ণ।
Telemetry প্রতি 10 সেকেন্ডে buffered JSONL flush করুন, কিন্তু individual CUDA decode step-এ blocking disk write, forced synchronization বা heavy state copy করবেন না। এতে scheduler overlap বদলে bug লুকিয়ে যেতে পারে।

## প্রশ্ন ৬ — শুধু #43567-এর দুইটি else hunk কি safe?
আমার verdict: isolated correctness test-এর জন্য GO; production-safe প্রমাণিত নয়।
Upstream PR #43567 স্পষ্ট বলেছে, patchটি allocation strategy বা capacity formula পরিবর্তন করে না; শুধু in-place fallback-এর আগে পুরোনো checkpoint validity invalidate করে। অন্যদিকে #43559 আলাদা lock-release/eviction retry সমস্যার সমাধান।
Sources:
https://github.com/sgl-project/sglang/pull/43567
https://github.com/sgl-project/sglang/pull/43559
cache_len=0 নিয়ে আপনার সংশয়
v0.5.21-এর MambaComponent.prepare_for_caching_req():
- Unfinished request-এ checkpoint length None হলে caching skip করতে পারে।
- Finished request-এ None হলে cache_len=0 হয়।
- Finished branch-এ keep-slot reference clone করার পথ আছে।
এর অর্থ None path আগে থেকেই supported, এবং #43567 নতুন ধরনের invalidity marker তৈরি করছে না। তবে length-zero insertion থেকে downstream radix-node cleanup, ownership এবং checkpoint restore-এর প্রতিটি edge case আপনার exact build-এ verified নয়।
বিশেষত cache_len=0 অবস্থায় একটি physical state slot ধরে রাখা হচ্ছে কি না এবং সেটি পরে সঠিকভাবে release হচ্ছে কি না, এই test যোগ করুন।
Patch safety test
```
def test_checkpoint_invalidation(event):    if event["optional_alloc_failed"] and event["inplace_fallback"]:        assert event["checkpoint_len_after_prepare"] is Nonedef test_no_stale_donation(event):    if event["checkpoint_len_after_prepare"] is None:        assert not event["donated_old_checkpoint"]def test_finished_cleanup(event):    if event["is_finished"] and event["checkpoint_len"] == 0:        assert not event["reusable_nonzero_mamba_checkpoint"]        assert event["slot_cleanup_ok"]
```
এগুলো proposed test assertions; upstream-এর সরাসরি copy নয়।
আরও পরীক্ষা করতে হবে: request abort, ignore_eos শেষ হওয়া, normal EOS, cancellation, repeated prefix reuse এবং no-cache fallback। #43559 না থাকলে আলাদা donation-time allocation assertion এখনও সম্ভব, তাই #43567 success-কে সমস্ত Mamba cache bugs fixed হিসেবে রিপোর্ট করবেন না।
0.5.21 patched বনাম 0.5.22
| Arm | কী প্রমাণ করবে | Recommendation |
|---|---|---|
| 0.5.21 unpatched | Existing production reference | Mandatory |
| 0.5.21 + only #43567 | Isolated causal fix | Primary experiment |
| 0.5.22 stock | Upstream integrated behavior | Secondary verification |
| 0.5.22 + further fixes if necessary | Future migration candidate | Later |
SGLang v0.5.22-এ #43029-এর committed-length fix এবং অন্যান্য serving changes-ও রয়েছে। তাই শুধু 0.5.21 vs 0.5.22 তুলনা করলে #43567-এর isolated effect জানা যাবে না।
Release notes:
https://github.com/sgl-project/sglang/releases/tag/v0.5.22
Patch adoption rule: Correctness improved + no new crash + no slot leak + no severe throughput regression—সবগুলো প্রমাণ চাই।

## প্রশ্ন ৭ — কতগুলো ARC tasks দিয়ে actual solve-rate impact বোঝা যাবে?
এখানে full ARC EVAL120 চালানোর আগে একটি paired, exposure-enriched subset ব্যবহার করব।
দুটি sample category প্রয়োজন:
- High-exposure tasks: Duck/Poetiq-style বহু turn, frequent prefix restoration, repeated model reasoning, যেখানে suspected bug path বাস্তবেই ঘটার সুযোগ বেশি।
- Control tasks: কম turn, কম cache reuse, এবং target-only/no-cache conditions।
Practical experiment
- Stage 1 — Signal discovery, 16–24 tasks: Same tasks unpatched বনাম patched। 8–12টি confirmed high-exposure এবং বাকিগুলো control। এটি শুধু strong effect খোঁজার জন্য; final statistical claim নয়।
- Stage 2 — Paired validation, 48–64 tasks: Stage 1-এ exposure ও output differences পাওয়া গেলে নতুন fixed task set-এ paired comparison। Task-level solve rates, valid programs, failure counts এবং coverage তুলুন।
- Stage 3 — Final confidence, ~120 tasks: Meaningful positive signal থাকলে full EVAL120, ideally repeated paired seeds এবং separate holdout।
এগুলো operational starting sizes, formally powered sample sizes নয়। Task-normalised variance ও expected effect size ছাড়া exact required sample count নির্ধারণ করা সম্ভব নয়। আপনার LB SD ≈1.5 সরাসরি paired EVAL120 variance নয়।
পরিসংখ্যানের জন্য কী মাপবেন? একই task-এ: Unpatched fail / patched pass = Fix wins; Unpatched pass / patched fail = Fix regressions; Both pass = Concordant success; Both fail = Concordant failure.
Paired comparison-এ McNemar/exact sign test এবং task-level bootstrap confidence interval ব্যবহার করুন। একই prompt, same task, matched seed, same max time এবং একই completion policy রাখতে হবে।
একটি বড় methodological correction: Temperature 1.0-এ patched ও unpatched output আলাদা হওয়া মাত্রই bug-এর evidence নয়। Sampling trajectory বদলাতে পারে। তাই first stage-এ greedy/teacher-forced diagnostics দিয়ে root cause establish করুন; তারপর production temperature 1.0-এ multiple seeds দিয়ে task-level gain মাপুন।
KL > 0 থেকে ARC score improved যাওয়ার evidentiary chain: Observed stale restore → Restored state differs from recomputation → Target logits materially change → Real model generation changes → ARC program/output validity changes → Task solve rate changes → Leaderboard-relevant benefit. যেকোনো stage-এ chain ভেঙে যেতে পারে। শুধু KL বড় মানে leaderboard benefit নয়।

## প্রশ্ন ৮ — E17-তে আর কোন silent-accuracy mechanisms পরীক্ষা করা উচিত?
| Priority | Suspect | Cheap test | Evidence/applicability |
|---|---|---|---|
| P0 | DFlash TARGET_VERIFY বনাম ordinary decode intrinsic mismatch | Single request, width-one forced reject, greedy next-token + state comparison | #36548, একই Qwen3.8 family; exact INT4/L4 result UNVERIFIED |
| P0 | Speculative Mamba committed-length/track boundary mismatch | Boundary crossing with acceptance lengths 1–6; no cache reuse control | #43029, 0.5.22 fix |
| P1 | GDN state numerical error amplified by fp16 | Same recurrent path, fp16/fp32, aligned long-token logits and ARC outputs | #36548 fp32 mitigation; exact L4 improvement UNVERIFIED |
| P1 | FP8 KV prefill-versus-decode numerical mismatch | FP8 vs BF16 KV, prefill/decode equivalence with same tokens | #25790, H100 attention backend; L4 UNVERIFIED |
| P1 | Invalid/missing runtime KV scales | Inspect every attention layer's loaded K/V scales and checkpoint values | #31224; warning alone unreliable |
| P2 | KDA padding sentinel write to live SSM slot | Check effective backend, padding slot mapping -1, batch padding variation | #36880, #36885; fixed-status on exact code must be checked |
বিশেষ গুরুত্বপূর্ণ: T3-এর teacher-forced NLL যথেষ্ট নয়। এটি prefill/logprob path পরীক্ষা করতে পারে, কিন্তু necessarily DFlash TARGET_VERIFY এবং speculative state commit path execute করে না। অতএব O4 ধরার জন্য T3-কে দুটি অংশে ভাগ করুন:
- T3a — Teacher-forced numerical drift: 8k position buckets-এ fp16 vs fp32 SSM, FP8 KV vs BF16 KV।
- T3b — Speculative verification parity: identical committed prefixes, forced-reject verification, target-only reference, per-layer recurrent state এবং next-token logits।
এই দুটি পরীক্ষার ফল আলাদা report করতে হবে।

## E17-এর সংশোধিত final experiment plan
| Phase | Test | কী প্রমাণ করবে | Priority |
|---|---|---|---|
| 0 | Source/hash/patch provenance, FP8 scales, telemetry smoke | Experiment integrity | Required |
| 1 | Upstream-style 128/255/32 no-spec repro, targeted fault | #43567 causal path | First GPU test |
| 2 | Same with DFlash W6, committed-boundary instrumentation | Speculative variant | Required |
| 3 | Production 16-stream natural pressure, state lifecycle trace | Natural exposure frequency | Required |
| 4 | Target-only vs DFlash forced-reject and greedy correctness | Independent #36548-class issue | Required |
| 5 | FP16/FP32 SSM and FP8 KV diagnostic | Other numerical issues | Only if remaining GPU budget |
| 6 | Paired ARC subset | Real solve-rate benefit | Separate follow-up after positive signal |
Required fail-closed gates:
```
def e17_adoptable(result):    return all([        result.get("patch_hash_verified") is True,        result.get("clean_oracle_verified") is True,        result.get("trigger_path_proven") is True,        result.get("no_stale_restore") is True,        result.get("no_new_crashes") is True,        result.get("no_state_slot_leaks") is True,        result.get("arc_accuracy_nonregression") is True,    ])
```
এই code-এ trigger_path_proven মানে failure→in-place overwrite→donation→restore সম্পূর্ণ chain observed। শুধু forced failure count positive হলেই true করা যাবে না।
arc_accuracy_nonregression E17 engine-only session-এ জানা সম্ভব নয়। তাই ওই session-এর verdict সর্বোচ্চ ENGINE_CORRECTNESS_VALIDATED; production adoption করতে পরবর্তী ARC task-level test প্রয়োজন।

## Final decision
CONDITIONAL GO — after T1/T2 corrections. Blocking experiment-design issues:
1. T1-এ valid checkpoint seeding এবং actual stale donation/restore guarantee নেই।
2. Global forced-fail initialization নিজেই seed তৈরি বন্ধ করতে পারে।
3. DFlash speculative window reachable হওয়া আর actual committed crossing এক নয়।
4. top-k union KL দিয়ে full-vocabulary KL দাবি করা যাবে না।
5. T3-এর teacher-forced scoring দিয়ে DFlash target-verify correctness প্রতিষ্ঠা করা যাবে না।
6. Function-entry/exit wrappers দিয়ে local new_slot is None failure reliably count করা যায় না।
এই ছয়টি সংশোধন করা হলে E17 একটি যুক্তিযুক্ত targeted GPU investigation। #43567 বাস্তবে আপনার score নষ্ট করছে কি না জানা এখনও বাকি; ২৬+ LB uplift-এর কোনো credible numeric projection নেই।
সবচেয়ে কার্যকর পন্থা হবে: আগে upstream-এর ছোট deterministic cache-state corruption reproduce, তারপর DFlash variant, তারপর production workload, সবশেষে actual ARC solve-rate।
কোনো Kaggle GPU run বা patch deployment আমি করিনি।
