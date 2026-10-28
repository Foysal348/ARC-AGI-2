# GPT audit 2 of PLAN_E17 (pasted by the user 2026-10-26, verbatim, Bengali original)

Note: the paste contained a rendered Mermaid flowchart. Its CSS blob was dropped, and the chart is given as the text of the attached image.

E17 Pre-GPU Audit — Silent-Accuracy Investigation
২৬ অক্টোবর ২০২৬ | SGLang 0.5.21 | Qwen3.8-27B INT4 + DFlash W6 | 4× L4

## প্রাথমিক সিদ্ধান্ত
E17 গবেষণার দিক সঠিক, কিন্তু T1 ও T2 পরীক্ষার বর্তমান নকশা অপরিবর্তিত রেখে GPU চালানো উচিত নয়। বিশেষ করে একটি alloc(1) failure দেখা এবং পরবর্তী cache hit-এ stale checkpoint সত্যিই restore হওয়া—দুটি আলাদা ঘটনা।
আমি তিনটি পরিবর্তনকে অত্যাবশ্যক মনে করছি:
1. T1-এ actual stale checkpoint donation ও restoration নিশ্চিত করতে হবে। শুধু forced allocation failure এবং cold/warm KL যথেষ্ট নয়।
2. T2-এ প্রতিটি allocation failure-এর lifecycle track করতে হবে। নইলে harmless failure ও silent corruption আলাদা হবে না।
3. DFlash/TARGET_VERIFY-এর আলাদা correctness control চাই। কারণ cache বন্ধ রেখেও speculative verification-এ target-only থেকে output বদলানোর upstream evidence আছে।
এখানে সবচেয়ে শক্ত প্রমাণ SGLang-এর নিজস্ব PR:
https://github.com/sgl-project/sglang/pull/43567
এই PR-এ upstream পূর্ণ-vocabulary পরীক্ষায় KL = 0.6494482615 পেয়েছে, আর fix-এর পরে KL = 0। কিন্তু পরীক্ষাটি Qwen3.5-0.8B/GB300-তে ordinary decode দিয়ে করা; আপনার DFlash W6/L4 path-এর জন্য সেই ফল সরাসরি প্রযোজ্য নয়।
চূড়ান্ত GPU verdict হবে CONDITIONAL GO: নিচের instrumentation ও test-design corrections যুক্ত হলে E17 চালানো যুক্তিসংগত। আপনার production-এ #43567 সত্যিই accuracy কমাচ্ছে—এটি এখনও UNVERIFIED।

## ১. প্রশ্ন ১ — T1 কি সত্যিকারের #43567 bug ধরবে?
বর্তমান T1 দিয়ে নিশ্চিতভাবে নয়। সবচেয়ে ভালো সংশোধন হলো upstream-এর আসল regression sequence-কে test-এর প্রথম control হিসেবে ব্যবহার করা।
আমি PR #43567-এর Files changed থেকে upstream-এর test source যাচাই করেছি। এতে গুরুত্বপূর্ণ একটি তথ্য পাওয়া গেছে: upstream deterministic repro SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0 রেখেই স্বাভাবিক চার-slot pressure তৈরি করেছে। অর্থাৎ জোর করে allocation fail করানো আবশ্যক নয়।
Source: https://github.com/sgl-project/sglang/pull/43567/files

Upstream-এর যাচাইকৃত reproduction sequence (Qwen3.5-0.8B-এর জন্য):
```
# Minimal reproduction structure from upstream
# Requires the upstream's small-slot server configuration.
def warm_prefixes():
    flush_cache()
    generate([44] * 128, max_new_tokens=1)
    generate([42] * 128, max_new_tokens=1)
P = [42] * 128 + [43] * 127  # 255 tokens
warm_prefixes()
reference = generate(P, max_new_tokens=1, score=True)
warm_prefixes()
seed = generate(P, max_new_tokens=32)
reused = generate(P, max_new_tokens=1, score=True)
```
Upstream server configuration:
```
--mamba-radix-cache-strategy extra_buffer_lazy \
--mamba-track-interval 128 \
--max-mamba-cache-size 4 \
--max-running-requests 1 \
--chunked-prefill-size 2048 \
--cuda-graph-max-bs-decode 1
```
সঙ্গে: `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0`, `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0`

| Signal | Broken | Patched |
|---|---|---|
| Warm prefix hit | Stale 192-token checkpoint | Safe 128-token checkpoint |
| Full-vocab KL | 0.6494482615 | 0 |
| HTTP response | Normal | Normal |

উপরের exact numbers upstream Qwen3.5-0.8B/GB300 run-এর, আপনার Qwen3.8-27B-এর নয়।

Stale checkpoint তৈরির জন্য প্রয়োজনীয় causal sequence (flowchart, from the attached image):
Request matches cached prefix → Request owns live state and a valid keep checkpoint → Next decode or verify needs an optional tracking slot → alloc(1) succeeds?
- Yes → Writes to a distinct pending slot.
- No → Falls back to in-place write into keep slot → Old checkpoint length invalidated?
  - Yes, patched → Do not publish stale checkpoint.
  - No, unpatched → Overlapping prefill-result processing may donate overwritten state under old length → Later radix prefix hit restores stale state → Same input IDs may give wrong logits or output.

উল্লেখযোগ্য বিষয়: request শেষ হওয়া বাধ্যতামূলক নয়। Upstream #43567-এর critical timing হলো prefill-result processing ও পরবর্তী decode/verify write-এর overlap। অতএব seed-এর পরে একটি finished request দেখা যথেষ্ট নয়; actual donation-এর ordering দেখতে হবে।

আপনার ৬৮-slot production-এর জন্য সংশোধন
চার-slot test সরাসরি আপনার production flags নয়। তাই দুটি পৃথক ফল রাখতে হবে:
- Small-slot reproduction: --max-mamba-cache-size 4, concurrency 1, mamba-track-interval 128—bug mechanism-এর জন্য। Qwen3.8-27B/L4-তে এই configuration server successfully load করে কি না UNVERIFIED।
- Production exposure: ৬৮ slots, ১৬ running, actual track interval, DFlash W6—বাস্তবে একই event ঘটে কি না।
SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1-এর test আলাদা রাখুন। Forced failure খুব দ্রুত ও ভিন্ন সময়ে ঘটতে পারে, ফলে প্রয়োজনীয় valid old checkpoint তৈরি হওয়ার আগেই invalidation/overwrite শুরু হতে পারে। এটি negative branch unit test, কিন্তু complete stale-restore E2E-এর বিকল্প নয়।
T1-এর নতুন PASS condition: allocation failed → same keep slot overwritten → stale length remained valid → donation published → later restore consumed that node—এই chain observed হলে তবেই BUG_EXPOSED। Chain অসম্পূর্ণ হলে TRIGGER_NOT_PROVEN, KL মিলে গেলেও bug নেই বলা যাবে না।

## ২. প্রশ্ন ২ — DFlash / EAGLE-এর speculative path কীভাবে পরীক্ষা করবেন?
উত্তর: mamba_lazy_spec_prepare()-এর failure ঘটালেই bug reproduction সম্পূর্ণ হয় না। Speculative verification-এ state in-place update এবং তার পরে stale checkpoint donation/restore—দুটোই ঘটতে হবে।
Upstream #43567-এর patch-এ speculative failure branch-টি সরাসরি রয়েছে:
```
if new_slot is not None:
    pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
    has_pending = True
else:
    req.kv.mamba_last_track_seqlen = None
```
Patch ছাড়া else branch-এ পুরোনো mamba_last_track_seqlen valid থাকার সম্ভাবনাই সমস্যার মূল।
Proposed deterministic procedure:
1. প্রথমে ordinary target-only test দিয়ে আপনার model-এ চার-slot অথবা ছোট-slot cache scenario কাজ করে কি না যাচাই করুন।
2. একই prompt family দিয়ে DFlash W6 চালান।
3. mamba_lazy_spec_prepare()-এ অন্তত একটি allocation failure এবং has_pending=False record করুন।
4. সেই iteration-এ TARGET_VERIFY বাস্তবে হয়েছে কি না, committed token count এবং track position কী ছিল, তা record করুন।
5. সংশ্লিষ্ট request-এর checkpoint donation এবং পরবর্তী reuse একই node/slot-এর সঙ্গে মেলে কি না পরীক্ষা করুন।
6. শেষে একই fixed input_ids দিয়ে cold/warm next-token distribution তুলুন।
কত generated tokens লাগবে—একটি নির্দিষ্ট সংখ্যা দেওয়া যাচ্ছে না। সেটি runtime mamba_track_interval, current tracked position, accepted draft length এবং boundary crossing-এর ওপর নির্ভর করে। Test budget হিসেবে:
```
max_new_tokens = (2 * actual_mamba_track_interval + 2 * speculative_num_draft_tokens + 16)
```
এটি আমার প্রস্তাবিত initial test budget, upstream-verified minimum নয়। Boundary event না ঘটলে generated-token budget আরও বাড়ানো নয়, আগে instrumentation ও start position ঠিক করুন।
বিশেষ করে DFlash W6 ও EAGLE NEXTN-কে একই correctness result হিসেবে ধরবেন না। একটি path-এর PASS দিয়ে অন্যটি certify হবে না।
একটি আরও গুরুত্বপূর্ণ caution: #36548-এর concurrency corruption-এর একটি reproduction-এ cache ছাড়াই forced-reject TARGET_VERIFY বনাম ordinary decode mismatch পাওয়া গেছে। তাই DFlash divergence দেখা গেলেই সেটি #43567 থেকে এসেছে বলা যাবে না।
Required controls: target-only/radix on, target-only/radix off, DFlash/radix on, DFlash/radix off। প্রথমে একক request, তারপর ১৬ concurrent।

## ৩. প্রশ্ন ৩ — stale state restored হওয়ার সবচেয়ে ভালো signal কোনটি?
সবচেয়ে শক্তিশালী পদ্ধতি হলো event provenance + identical-input logprob comparison।
| Signal | কী প্রমাণ করবে | Limitation |
|---|---|---|
| alloc_failed | Optional tracking slot পাওয়া যায়নি | একা corruption নয় |
| inplace_write + prior valid length | পুরোনো checkpoint overwrite হওয়ার ঝুঁকি | Donation হয়েছে কি না অজানা |
| checkpoint_donated | Cache-এ একটি checkpoint প্রকাশিত হয়েছে | Stale কি না আলাদা যাচাই চাই |
| checkpoint_restored | পরবর্তী request cache node ব্যবহার করেছে | Numeric correctness এখনও অজানা |
| cached_tokens | Reuse length-এর external confirmation | Mamba state সঠিক কি না বলে না |
| Same-input full-vocab KL | Actual model distribution বদলেছে কি না | কোন kernel দায়ী তা একা জানায় না |
| Selected Mamba-state tensor comparison | State divergence localize করতে সাহায্য | Intrusive; synchronization/copy overhead |
Recommended instrumentation: প্রতিটি allocation attempt-এ request ID, scheduler iteration, track length, slot identity এবং free/evictable counts সংরক্ষণ করুন। একই সঙ্গে donation ও restore path-এও instrumentation যুক্ত করতে হবে। শুধু আপনার প্রস্তাবিত দুটি ScheduleBatch method wrap করলে lifecycle সম্পূর্ণ পাওয়া যাবে না।
Suggested JSONL event schema:
```
{"event": "lazy_alloc_failed", "request_id": "redacted-request-id", "scheduler_step": 123, "phase": "TARGET_VERIFY", "tracked_seqlen_before": 8192, "track_slot_before": 17, "available_before": 0, "evictable_before": 8, "pending_slot_allocated": false}
```
এখানে numeric values কেবল schema example—observed run নয়।
Donating/restoring সময়ে একই event identity-র সঙ্গে radix_node_id, cache_length, slot_id এবং slot reuse generation যোগ করুন। Slot ID একা যথেষ্ট নয়, কারণ recycled slot পরে অন্য state ধারণ করতে পারে।
Example classifier:
```
def classify_cache_trace(events):
    failed = any(e["event"] == "lazy_alloc_failed" for e in events)
    overwritten = any(e["event"] == "inplace_write" for e in events)
    stale_donated = any(e["event"] == "stale_donation" for e in events)
    stale_restored = any(e["event"] == "stale_restore" for e in events)
    if stale_restored and stale_donated:
        return "STALE_RESTORE_EXPOSED"
    if failed and overwritten:
        return "FALLBACK_EXPOSED_NOT_CERTIFIED"
    return "NO_CONFIRMED_STALE_RESTORE"
```
stale_donation/stale_restore তৈরি করতে আগে source-level consistency predicate implement করতে হবে; এগুলো upstream-এর প্রস্তুত event names নয়।
Internal debug assertion আছে? SGLang 0.5.21-এ mamba_last_track_seqlen metadata এবং cache donation paths আছে—আপনার source review-এ তা নিশ্চিত হয়েছে। কিন্তু সব donated Mamba state recomputation-এর সঙ্গে compare করে এমন কোনো built-in end-to-end assertion আমি নিশ্চিত করতে পারিনি — UNVERIFIED।
প্রস্তাবিত fail-closed diagnostic check:
```
# Diagnostic pseudocode — implement at the donation boundary,
# after validating actual v0.5.21 fields and ownership.
if checkpoint_was_overwritten_in_place:
    assert checkpoint_was_invalidated_before_donation
```
এই assertion permanent production patch নয়। E17 debug build-এ ব্যবহার করুন; production-এ upstream #43567-এর দুই-hunk invalidation যথেষ্ট কি না GPU test দিয়ে যাচাই করুন।

## ৪. প্রশ্ন ৪ — 248k-vocabulary logits সবচেয়ে সস্তায় কীভাবে পাবেন?
এখানে একটি সুনির্দিষ্ট উত্তর পাওয়া গেছে: upstream PR #43567 নিজের regression test-এ SGLang /generate endpoint-এর token_ids_logprob ব্যবহার করে সম্পূর্ণ vocabulary distribution নিয়েছে। সুতরাং প্রথমে custom CUDA/server hook দরকার নেই।
Recommended request:
```
import requests
import numpy as np
vocab_ids = list(range(actual_vocab_size))
payload = {
    "input_ids": fixed_prompt_token_ids,
    "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
    "return_logprob": True,
    "logprob_start_len": -1,
    "token_ids_logprob": vocab_ids}
result = requests.post("http://127.0.0.1:30000/generate", json=payload, timeout=120).json()
rows = result["meta_info"]["output_token_ids_logprobs"]
assert len(rows) == 1
assert [x[1] for x in rows[0]] == vocab_ids
lp = np.array([x[0] for x in rows[0]], dtype=np.float64)
assert np.isfinite(lp).all()
lp -= np.logaddexp.reduce(lp)
```
actual_vocab_size checkpoint-এর text config থেকে নেবেন। 248,320 token upstream model-এর vocabulary size; RedHat checkpoint-এ hardcode করার প্রয়োজন নেই।
KL:
```
def kl_div(logp, logq):
    p = np.exp(logp)
    return float(np.sum(p * (logp - logq)))
kl = kl_div(cold_logprobs, warm_logprobs)
```
API alternatives-এর খরচ ও সীমাবদ্ধতা:
| Method | Full-vocab KL পাওয়া যাবে? | আমার মূল্যায়ন |
|---|---|---|
| return_logprob=True only | না, selected token প্রধানত | কম খরচে selected-token comparison |
| top_logprobs_num=20/100 | না | Fast screening, full KL নয় |
| top_logprobs_num=vocab_size | তাত্ত্বিকভাবে সম্ভব মনে হলেও exact 0.5.21 API/limit UNVERIFIED | ব্যবহার করার দরকার নেই |
| token_ids_logprob=list(range(vocab_size)) | হ্যাঁ, upstream test-verified | প্রথম পছন্দ |
| Server-side raw-logits hook | সম্ভব, implementation-specific | API যথেষ্ট না হলে |
প্রতি scored position-এ 248,320 FP32 logprobs-এর raw size প্রায় 0.95 MiB; token IDs-সহ binary representation কয়েক MiB। কিন্তু JSON serialization, transfer ও memory allocation overhead অনেক বেশি হতে পারে। সেটির actual wall-time এই stack-এ UNVERIFIED।
তাই ১৬-stream দীর্ঘ run-এর প্রতিটি token-এ full-vocab score নেবেন না। প্রথমে ২–৪টি decisive probe-তে ব্যবহার করুন।
আর একটি সতর্কতা: এগুলো model-এর output log-probabilities। temperature=1, top_p=0.95, top_k=20 দিয়ে final sampling-এর পরে যে truncated distribution ব্যবহৃত হয়, সেটির সঙ্গে এদের সরাসরি KL এক নয়। আপনার T2 থেকে nucleus-audit বাদ দেওয়ার সিদ্ধান্ত ঠিক হয়েছে।

## ৫. প্রশ্ন ৫ — natural allocation failures কীভাবে গুনবেন?
আপনার hooks সঠিক জায়গা থেকে শুরু করেছে, কিন্তু যথেষ্ট নয়।
| Counter | অর্থ |
|---|---|
| lazy_boundary_attempts | Ordinary lazy tracking কতবার চেষ্টা হয়েছে |
| lazy_spec_attempts | Speculative lazy tracking কতবার চেষ্টা হয়েছে |
| lazy_alloc_failures | Optional slot request ব্যর্থ হয়েছে |
| fallback_inplace_writes | Failure-এর পরে keep slot overwrite path নিয়েছে |
| failed_with_valid_old_length | Failure-এর সময় পুরোনো checkpoint length valid ছিল |
| invalidated_before_write | Patched path যথাসময়ে length invalidated করেছে |
| stale_donations | Overwritten state পুরোনো length-এ publish হয়েছে |
| stale_restorations | পরবর্তী request সেই stale checkpoint restore করেছে |
| donation_assertions | Mamba allocation assert-এর কারণে scheduler বন্ধ হয়েছে |
এছাড়া available_size, evictable_size, locked_size (সংশ্লিষ্ট allocator থেকে নির্ভরযোগ্যভাবে পাওয়া গেলে), active Mamba slots ও live requests সংরক্ষণ করুন।
সবচেয়ে গুরুত্বপূর্ণ correction: mamba_usage-কে pressure indicator হিসেবে ব্যবহার করবেন না। আপনার দেওয়া source calculation অনুযায়ী evictable cached states আলাদা বাদ দেওয়া হয়। তাই available_size == 0 হতে পারে, যদিও reported mamba_usage কম দেখায়।
Instrumentation-এর robustness:
- প্রতিটি TP rank-এর log আলাদা রাখুন; একই JSON ফাইলে চার process লিখলে partial writes হতে পারে।
- প্রতি decoding step-এ JSON serialize করবেন না। CPU-side in-memory counters/ring-buffer রাখুন।
- alloc()-এর source-level actual return outcome গণনা করুন। Function call সংখ্যা থেকে ব্যর্থতা infer করবেন না।
- sitecustomize monkey patch সত্যিই worker ও scheduler-এর কাঙ্ক্ষিত method wrap করেছে কি না, startup-এ test event দিয়ে যাচাই করুন।
- Source instrumentation চালু/বন্ধ রেখে ছোট A/A comparison করুন—probe নিজেই scheduler timing পাল্টাচ্ছে কি না বোঝার জন্য।
Natural pressure test-এর decision logic:
```
if failures == 0:
    verdict = "NO_FAILURE_OBSERVED"
elif stale_restores == 0:
    verdict = "FAILURE_OBSERVED_NO_STALE_RESTORE"
elif cold_warm_divergence:
    verdict = "SILENT_CORRUPTION_CONFIRMED"
else:
    verdict = "STALE_RESTORE_EXPOSED_NO_OUTPUT_FLIP"
```
এই চারটি outcome আলাদা করতে পারলে O1/O2 সম্পর্কিত সবচেয়ে গুরুত্বপূর্ণ uncertainty দূর হবে।
আর T2-তে একবার server restart করে cache flush করা যথেষ্ট নয়। প্রত্যেক scored probe-এর আগে idle হওয়া, /flush_cache সফল হওয়া এবং expected cache-hit behaviour নিশ্চিত করতে হবে। Otherwise cold reference-ও warm থাকতে পারে।

## ৬. প্রশ্ন ৬ — দুই-hunk patch যথেষ্ট নিরাপদ? 0.5.22 পরীক্ষা করবেন?
#43567 alone: হ্যাঁ, isolated correctness test হিসেবে যুক্তিসংগত। Upstream #43567 ordinary decode regression-এ এই change একাই KL 0 করেছে। Source-level backport আপনার 0.5.21-এ compile হয়েছে এবং idempotent—এটিও ইতোমধ্যে CPU-তে যাচাই হয়েছে। তবে সম্পূর্ণ production-safety certification নয়।
#43559 না দিলে কী হবে? PR #43559 স্পষ্ট বলেছে #43567 prerequisite। এর নিজস্ব problem হলো consumed prefix lock ছাড়ার আগে donation allocation retry করলে pool exhausted হয়ে `AssertionError: Can not alloc mamba cache` হতে পারে। Merged: ১৮ অক্টোবর ২০২৬।
তাই:
- #43567 alone → stale checkpoint invalidation পরীক্ষা করার সঠিক ablation।
- #43567 + #43559 → upstream-এর সংশ্লিষ্ট allocation/crash path-ও সংশোধিত অবস্থার পরীক্ষা।
- 0.5.22 → দুটি fix এবং আরও পরিবর্তনসমেত release-level comparison।
আপনার ৬২টি real server log-এ এই assert শূন্যবার পাওয়া গেছে, তাই এটিকে বর্তমান production-এর confirmed crash বলছি না। কিন্তু E17-এর deliberate slot-pressure experiment-এ এই failure ঘটতে পারে। ঘটলে একে #43567 correctness failure হিসেবে গণনা করবেন না।
Length-0 insertion সম্পর্কে: mamba_last_track_seqlen=None হলে finished request-এর length-0 cloned state downstream কীভাবে ব্যবহৃত হয়—আপনার review-ও পুরো path trace করেনি। Exact Qwen3.8-27B / v0.5.21 result UNVERIFIED। তবে None path নতুন নয়; upstream fix-ও এটি ব্যবহার করে। শুধু এ কারণে backport-কে unsafe ঘোষণা করার evidence নেই। Negative tests করুন: unfinished request, finished request, zero cached length, EOS/truncation এবং next-turn prefix reuse।
আরও একটি আলাদা fix: PR #43387 speculative truncation-এর পরে Mamba checkpoint misalignment সংশোধন করেছে। এর confirmed reproduction no_buffer + NEXTN-এ—আপনার extra_buffer_lazy + DFlash path একইভাবে আক্রান্ত, তার প্রমাণ নয়। আর PR #43749 হলো ReplaySSM-specific unflushed checkpoint issue; আপনার বর্তমান ReplaySSM-disabled setup-এ সরাসরি relevant নয়।
Recommended version comparison:
| Arm | Purpose | Priority |
|---|---|---|
| v0.5.21 original | Bug-exposure baseline | P0 |
| v0.5.21 + #43567 | Isolate stale checkpoint fix | P0 |
| v0.5.21 + #43567 + #43559 | Crash-path follow-up, if triggered | P1 |
| v0.5.22 stock | Release-level robustness | P1, separate session if needed |
প্রথম দুই arm-এর মধ্যে অন্য কোনো source, quantization, sampling বা kernel backend পরিবর্তন করবেন না। 0.5.22 একই সঙ্গে অনেক পরিবর্তন আনে; faster/cleaner হলেও সেটি #43567-এর causal effect isolate করে না।

## ৭. প্রশ্ন ৭ — KL difference থেকে ARC solve-rate impact কীভাবে মাপবেন?
সবচেয়ে সস্তা অর্থবহ পরীক্ষা হলো paired task-level replay, পুরো leaderboard submission নয়। প্রথমে #43567 সত্যিই আপনার stack-এ active হলে তবেই এই পর্যায়ে যান। শুধু upstream bug আছে বলে ১২০টি ARC task পুনরায় চালানোর প্রয়োজন নেই।
Recommended paired experiment — দুই engine: A = SGLang 0.5.21 original; B = SGLang 0.5.21 + #43567। একই task, prompt, seeds, solver code, retry policy, token budget এবং wall-time cap ব্যবহার করুন। Primary comparison-এ production-এর temperature=1.0, top_p=0.95, top_k=20 বজায় রাখুন। Diagnostic greedy replay আলাদা হবে।
প্রথমে Duck-এর multi-turn tasks বেছে নিন, কারণ আপনার hypothesis অনুযায়ী সেখানে cache restoration বেশি। কিন্তু এটিকে পুরো ARC task distribution-এর representative sample বলা যাবে না।
| Stage | Task count | উদ্দেশ্য |
|---|---|---|
| Pilot | 12–16 | Cache exposure, wall time এবং trace instrumentation sanity |
| Screening | 32–48 | Patch-এর কারণে solve outcome বদলায় কি না |
| Confirmation | 80–120 | Aggregate paired difference, uncertainty এবং coverage effect |
| Full EVAL | প্রয়োজন হলে | Final candidate selection |
এই task counts আমার practical recommendations; statistical power নিশ্চিত করে না।
কীভাবে ফল বিশ্লেষণ করবেন? চারটি সংখ্যা রাখুন: both_solved, both_failed, patched_only_solved, unpatched_only_solved.
Paired solve-rate uplift: Δ = (N_patched-only − N_unpatched-only) / N_paired. একই tasks-এ solved/failed outcome comparison-এর জন্য McNemar exact test ব্যবহার করা যায়।
ধরুন ৩২টি task-এর মধ্যে ৮টি শুধু patched arm solve করেছে, আর কোনোটি শুধু unpatched arm solve করেনি। তাহলে paired difference ২৫ percentage points, এবং two-sided exact McNemar p=0.0078125। এটি hypothetical example, আপনার expected uplift নয়।
তবে আরেকটি পার্থক্য অত্যন্ত গুরুত্বপূর্ণ:
- Fixed token/completion budget: serving correctness ও solver quality তুলনা।
- Fixed wall-clock budget: correctness-এর পাশাপাশি speed, prefix hit ও completed-task coverage তুলনা।
Production impact বোঝার জন্য শেষ পর্যন্ত দুটোই দরকার। শুধুমাত্র fixed-time evaluation দিয়ে KL corruption-এর effect আলাদা করা যাবে না।
আপনার LB standard deviation প্রায় ১.৫ হলেও, সেখান থেকে সরাসরি paired evaluation-এর required sample size নির্ধারণ করা যায় না। Task-level discordance ও repeated seed variation মাপতে হবে।
আর একটি সংশোধন: E17 README-তে বলা হয়েছে, একই engine/flags থাকায় #43567 একা eval→LB gap তৈরি করতে পারে না। এটি অত্যন্ত শক্ত categorical claim। একই flags থাকলেও ১২০ বনাম ২৪০ tasks, concurrent scheduling, cache occupancy ও request churn আলাদা হলে conditional bug-এর trigger frequency বদলাতে পারে। এটি আপনার gap-এর প্রমাণ নয়, কিন্তু যৌক্তিকভাবে সম্ভাবনাটি বাদ দেওয়া যায় না।

## ৮. প্রশ্ন ৮ — আপনার পরিকল্পনার বাইরে আর কী পরীক্ষা প্রয়োজন?
| Priority | Mechanism | Evidence | Cheapest decisive test | Fix / interpretation |
|---|---|---|---|---|
| P0 | DFlash TARGET_VERIFY-এর numerical/state divergence, radix cache ছাড়াও | #35150, #36548 | Cache off, concurrency 1, fixed input_ids, target-only বনাম DFlash forced-reject; first high-margin divergence localize | Reproduction হলে #43567 দিয়ে solved বলা যাবে না |
| P1 | FP8 KV calibration load path | #39942, #31224 | Full-attention layers-এর actual runtime k_scale_float/v_scale_float বনাম checkpoint metadata | Warning একা যথেষ্ট নয়; scale provenance যাচাই |
| P1 | DFlash acceptance collapse ও repeated garbage output | #43514 | Acceptance length, repeated-token rate, empty outputs, first anomalous step monitor | Original trigger grammar + long prefill; আপনার /generate path-এ applicability UNVERIFIED |
| P2 | Wrong DFlash target taps | #37475 | Draft config-এর target_layer_ids verify | Missing হলে fail-closed; expected [5,19,33,47,61] for published checkpoint |
বিশেষভাবে FP8 KV scale-এর নতুন finding: SGLang-এর #31224 দেখিয়েছে, startup warning-এ Defaulting to scaling factors of 1.0 বলা হলেও checkpoint-embedded calibrated scales বাস্তবে load হতে পারে। তাই warning দেখে bug আছে বলা যাবে না। অন্যদিকে #39942-এ কিছু model family-তে calibrated scales silently drop হওয়ার রিপোর্ট আছে। আপনার Qwen3.8 INT4 checkpoint একইভাবে affected—UNVERIFIED।
এ কারণে O6-এর জন্য শুধু loaded object-এর float fields নয়, প্রতিটি full-attention layer-এর checkpoint-provided values ও runtime values-এর mapping record করুন। GDN layers-এ সাধারণ full-attention KV scale থাকার কথা ধরে নিয়ে ভুল alarm তুলবেন না।
আর fp16 বনাম fp32 SSM test-এ আগে একটি নিয়ম রাখুন: দুই configuration-এ identical-prefix target-only cold A/A numerical drift মাপুন, তারপর DFlash ও cache reuse-এর comparison। তা না হলে fp32-তে sequence বদলালেই fp16 corrupt ছিল বলা যাবে না।
T3-এর গুরুত্বপূর্ণ সংশোধন: বর্তমান T3-তে 40–60k stored continuation-এর teacher-forced NLL তুলনা করা ভালো model distribution diagnostic, কিন্তু এটি 40–60k token ধরে step-by-step DFlash verification হয়েছে—এমন প্রমাণ নয়। অতএব দুটি test আলাদা করুন:
- LONG_PREFILL_PARITY: 40k/60k fixed prefix-এর final next-token logits।
- LONG_INCREMENTAL_PARITY: একই initial prompt থেকে সত্যিকারের sequential decoding এবং speculative verify transitions, 8k-position snapshots সহ।
দ্বিতীয়টি অনেক বেশি ব্যয়বহুল, কিন্তু long-generation recurrent-state bug ধরার জন্য বেশি প্রাসঙ্গিক।

## ৯. Revised E17 execution order
আপনার historical startup time প্রায় ৫১৭–৫৫৫ seconds। তাই তিনটি patch variant × target-only/DFlash মিলিয়ে ছয়টি cold launches নিলে শুধু startup-তেই প্রায় ৫২–৫৬ মিনিট চলে যেতে পারে। ৬০–৯০ মিনিটে সব T1/T2/T3 নির্ভরযোগ্যভাবে শেষ হবে ধরে নেওয়া উচিত নয়।
| Order | Experiment | Expected evidence |
|---|---|---|
| 0 | CPU patch/provenance + no-GPU instrument dry run | Exact hashes, successful event probes |
| 1 | Target-only unpatched, small-slot upstream KL repro | Qwen3.8-এ #43567 trigger হয় কি না |
| 2 | Target-only patched, identical repro | Same-version causal comparison |
| 3 | DFlash W6 unpatched, targeted speculative repro | mamba_lazy_spec_prepare exposure |
| 4 | DFlash W6 patched, same replay | Fix-এর effectiveness |
| 5 | Production ১৬-stream natural pressure | বাস্তবে failure frequency |
| 6 | Forced-reject/cache-off control | Independent DFlash accuracy issue |
| 7 | fp32/FP8 scale/long-generation follow-ups | Only if earlier evidence warrants |
#43559 crash fix অথবা stock v0.5.22 comparison প্রয়োজন হলে সেটি দ্বিতীয় session-এ রাখুন। একই session-এর মধ্যে সব hypothesis শেষ করার চেষ্টা করলে গুরুত্বপূর্ণ negative control-এ GPU সময় কম পড়ে যেতে পারে।
Outcome decision matrix:
| Result | Interpretation | Action |
|---|---|---|
| Unpatched stale restore + divergence; patched clean | #43567 confirmed on your stack | Patch candidate GO, followed by workload validation |
| Failure occurs, no stale donation | Harmless or not fully exposed | Corruption unconfirmed; keep investigating only if production exposure high |
| Both versions diverge with cache off | Separate decode/verify issue likely | NO-GO for DFlash correctness claim |
| Patched version asserts during pressure | Allocation/crash path exposed | #43559 or 0.5.22 follow-up |
| No natural failures after bounded stress | Trigger not observed | GO to other hypotheses, not proof bug impossible |
| No trigger in small-slot repro | E17 reproduction inconclusive | Fix trigger/instrumentation before score experiment |

## চূড়ান্ত GO/NO-GO
- Research hypothesis এবং patch ablation: GO
- বর্তমান T1/T2 অপরিবর্তিত রেখে causal conclusion: NO-GO
- সংশোধিত E17 diagnostic GPU test: GO
- #43567 থেকে নিশ্চিত leaderboard uplift: UNVERIFIED
সবচেয়ে গুরুত্বপূর্ণ সিদ্ধান্ত: E17-তে প্রথমে upstream-এর actual small-slot sequence দিয়ে #43567 reproduction করুন। Forced-fail flag-কে primary evidence বানাবেন না। তারপর DFlash-specific lifecycle instrumentation এবং cache-off forced-reject control দিয়ে corruption-এর উৎস আলাদা করুন। Full ARC evaluation কেবল GPU-level difference reproducible হওয়ার পর করুন।
এই পর্যালোচনায় upstream source ও PR diff যাচাই করেছি এবং আপনার provided E17 পরিকল্পনার সঙ্গে মিলিয়েছি। নতুন GPU run বা নতুন benchmark result তৈরি করা হয়নি।
