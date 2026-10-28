# GPT audit of the E17 bug-hunt restart bundle (2026-10-27), pasted by the user (verbatim, Bengali)

Input: `Experiment Results/E17_BUGHUNT_RESTART_FOR_GPT_2026-10-27.zip`. The README the audit mentions (164 lines, 17 URLs) was not attached.

---

E17 Bug-Hunt Restart — স্বাধীন অডিট
২৭ অক্টোবর ২০২৬ | ARC-AGI-2 | SGLang 0.5.21 বনাম vLLM 0.19/0.32 | 4× NVIDIA L4

১. সবচেয়ে গুরুত্বপূর্ণ সিদ্ধান্ত
আগের E17 পরীক্ষায় GPU time খরচ হলেও #43567-এর silent-accuracy hypothesis পরীক্ষা সম্পূর্ণ হয়নি। চারটি arm-ই AssertionError: Can not alloc mamba cache-এ থেমেছে। এটি upstream #43559-সম্পর্কিত allocation/lock-pressure failure-এর সঙ্গে সামঞ্জস্যপূর্ণ।
আরও গুরুত্বপূর্ণ, নতুন README একটি বড় scope correction করেছে: আপনাদের best-performing S1-60K notebook SGLang নয়, vLLM 0.19 + FP8 + MTP3 ব্যবহার করে। তাই SGLang-এর #43567 bug দিয়ে S1-এর CV→LB gap বা সব submission-এর কম score ব্যাখ্যা করা যায় না।
সেজন্য আমার সুপারিশ হলো E17-কে দুটি পৃথক investigation হিসেবে দেখা:
- Production-critical: S1-এর vLLM serving correctness, prefix-cache behaviour, FP8 KV, task coverage ও deadline accounting।
- Engine-specific: SGLang #43567/#43559 reproduction, যখন লক্ষ্য SGLang/DFlash-কে আবার production candidate বানানো।
এতে একটি SGLang-only hypothesis-এ আরেকটি সম্পূর্ণ GPU session খরচ হওয়ার ঝুঁকি কমবে।

২. নিশ্চিত সমস্যা ও সমাধান
| Severity | সমস্যা | Evidence | Fix |
|---|---|---|---|
| BLOCKER — E17 rerun | ৪-slot setup-এ mandatory Mamba checkpoint donation-এর জন্য free slot পাওয়া যায়নি; চারটি arm crash করেছে। | GPU logs, hook counters ও SGLang 0.5.21 source | Forced-fail ছোট diagnostic A/B; natural-pressure test-এ #43567 prerequisite রেখে #43559-compatible fix |
| HIGH — Evidence loss | e17_t1() তিনটি reference score সফলভাবে নেওয়ার পর seed crash করলে সব partial A/A results final report থেকে হারিয়ে যায়। | notebook/e17_cell.py, e17_phase() | প্রতি successful reference ও cycle শেষে atomic checkpoint |
| HIGH — Wrong engine priority | S1-এর vLLM-specific correctness পরীক্ষা বাদ রেখে SGLang-only investigation চলছে। | Bundle README এবং historical scores | S1 baseline-কে primary target করে vLLM MTP/cache/KV ablations |
| HIGH — Deadline exposure | Reported final drain-এর প্রায় পুরো ৮-মিনিট margin re-prefill-এ চলে যেতে পারে। | Bundle-এর 20×50k-token workload ও ~2.5k tok/s observation | Deadline-aware admission এবং measured-cost-based draining |
| HIGH — Unresolved correctness risk | vLLM hybrid model-এ MTP + prefix caching থেকে wrong/empty output-এর upstream reports। | #53912, #47194 | Same-checkpoint 2×2 MTP/cache experiment; greedy ও task-level comparison |

Fix 1 — E17 partial evidence loss: notebook/e17_cell.py-তে e17_t1() প্রতিটি সফল operation-এর পরে JSON snapshot সংরক্ষণ করুন।
```python
def e17_save_t1_partial(prompt, stage, out):
    path = E17_OUT / "partial" / (f"{E17_ARM['name']}_{prompt}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in out.items() if not k.startswith("_")}
    safe["last_completed_stage"] = stage
    safe["complete"] = False
    write_json_atomic(path, safe)
```
e17_t1()-এর reference loop-এ: `refs.append(r["lp"]); out["ref_cached"].append(r["cached"]); out["sum_raw"] = r["lp"]["sum_raw"]; e17_save_t1_partial(pname, f"reference_{len(refs)}", out)`. প্রতিটি cycle-এর শেষে: `out["cycles"].append(cycle_record); e17_save_t1_partial(pname, f"cycle_{c}", out)`. এখানে cycle_record বলতে existing cycle dictionary বোঝাচ্ছি; একই data দ্বিতীয়বার append করা যাবে না। সব expected cycles শেষ হলে snapshot-এ complete=True লিখবেন। Exception path-এ incomplete partial evidence preserved থাকবে, কিন্তু verdict INCOMPLETE হবে।
Validation: reference ৩-এর পর একটি synthetic RemoteDisconnected inject করুন। Expected: তিনটি reference-এর cached depth ও score metadata সংরক্ষিত; missing seed/reuse KL-কে কোনোভাবেই zero বা PASS ধরা হবে না।

Fix 2 — E17SHORT-এর পরের পরীক্ষায় ভুল চারটি arm পুনরাবৃত্তি নয়। বর্তমান E17_SHORT execution path-এ L1/L5/L7/L9 চালায়; forced-fail L4-কে skip করে। অথচ RUN1-এর পর এই L4-ই সবচেয়ে সস্তা পরবর্তী diagnostic। সংশোধিত next run: A: Target-only + forced-fail + unpatched; B: Target-only + forced-fail + #43567; C: DFlash W6 + forced-fail + unpatched; D: DFlash W6 + forced-fail + #43567. প্রথম A/B-তে usable cold/warm logprobs ও complete event chain না পেলে C/D চালাবেন না। আর full-production natural exposure test-এ forced-fail result-কে actual failure frequency হিসেবে ব্যবহার করবেন না। Upstream sources: https://github.com/sgl-project/sglang/pull/43567 , https://github.com/sgl-project/sglang/pull/43559 . Status: Existing E17SHORT rerun unchanged = NO-GO। সংশোধিত forced-fail diagnostic = GO।

৩. S1-60K-এর জন্য নতুন গুরুত্বপূর্ণ findings
| Priority | Suspect | Evidence | সবচেয়ে সস্তা test | Fix / next action |
|---|---|---|---|---|
| P0 | Hybrid prefix caching + MTP3-এ wrong state | vLLM #53912-এ empty/punctuation outputs; #47194-এ tool-call ও recall regression। রিপোর্টের version আপনার v0.19 নয়। | Same FP8 checkpoint; MTP on/off; fixed cold/warm ARC prompts | MTP off-এ reproducible corruption দূর হলে cache+MTP combination isolate করুন |
| P0 | mamba_cache_mode="align"-এ prefix misses | #45238 এবং #51250-এ aligned checkpoint geometry-এর limitation। | Identical shared prefix রেখে unique suffix lengths sweep; cached tokens ও TTFT | Supported newer version-এ partial-prefix matching test; v0.19-এ unsupported flag নয় |
| P1 | FP8 KV uncalibrated | S1 smoke report-এ scale 1.0, checkpoint-এ KV scales নেই | FP8 KV বনাম BF16 KV, same tokens ও checkpoint | BF16 better হলে quality/memory tradeoff; --calculate-kv-scales অন্ধভাবে নয় |
| P1 | Cache-boundary NaN / garbage | vLLM #55766-এ specific prefill-boundary cache-hit failure। | Prefix length boundary-এর আগে/পরে sweep, repeated cold/warm requests | Trigger confirmed হলে cache disable বা validated engine fix |
| P1 | Final drain-এর সময় শেষ | ~20 requests × 50k re-prefill; ৮-মিনিট margin | CPU timeline replay + recorded prefill latency | Deadline-aware task admission ও early submission persistence |
এখানে গুরুত্বপূর্ণ পার্থক্য: FP8 KV scale 1.0 নিজে bug প্রমাণ করে না। আবার vLLM #37554-এ hybrid model-এর --calculate-kv-scales ব্যবহার করেও corrupted outputs রিপোর্ট হয়েছে। তাই BF16 reference ছাড়া calibration change গ্রহণযোগ্য নয়।

৪. সংশোধিত ৬০-মিনিট GPU investigation
CPU-তে notebook branch, deadline ও submission validation আগে শেষ করুন। এরপর GPU-তে নিচের ক্রম অনুসরণ করুন। প্রতিটি cap-এর মধ্যে server startup-ও ধরতে হবে।
| Arm | Configuration | Budget cap | প্রধান প্রশ্ন |
|---|---|---|---|
| V0 | S1 original FP8 + MTP3 + prefix on + FP8 KV | 17 min | Stable baseline? |
| V1 | একই, শুধু MTP off | 15 min | Speculation correctness সমস্যা? |
| V2 | একই, শুধু prefix caching off | 15 min | Cached-state corruption? |
| V3 | একই, শুধু KV BF16/auto | অবশিষ্ট ≤13 min | FP8 KV numerical impact? |
প্রতিটি arm-এ fixed prompts, 8k/32k/50k context, serial ও concurrent requests, cold/warm repetitions, first-token/finish-status validation, token-ID divergence, large-margin flips, prefix-hit ratio, TTFT এবং end-to-end output tokens/sec রেকর্ড করুন। একটি arm-এ স্পষ্ট corruption পাওয়া গেলে পরবর্তী configuration না চালিয়ে reproduction পুনরাবৃত্তি বেশি মূল্যবান হবে। GPU-budget cap পার হলে বাকি arm NOT_TESTED_BUDGET, কখনো PASS নয়।

৫. LB-এর মতো CV বানানোর মূল সংশোধন
আপনার CV-কে শুধু ছোট evaluation subset-এর accuracy হিসেবে নয়, পুরো Kaggle rerun-এর executable simulation হিসেবে চালাতে হবে।
| Requirement | কীভাবে যাচাই করবেন |
|---|---|
| Same runtime | 12-hour deadline, startup, GPU speed, concurrency, prefill, drain |
| Same code path | RERUN branch, model files, package versions, checkpoint hashes, submission writer |
| Same workload geometry | 240-task workload এবং 120-task pseudo-hidden subsets, bundle-এর framing অনুযায়ী |
| Same scorer | প্রতিটি test-output grid-এর দুটি attempt; exact-match scoring |
| Coverage awareness | Started/completed tasks, finalised outputs, fallbacks, remaining time |
| Variance | Multiple seeded replicates এবং deadline jitter |
| Output integrity | Every required task ID, grid count, valid JSON, two attempts |
Kaggle-এর official rules অনুযায়ী score হয় test-output level-এ, এবং দুটি attempts-এর যেকোনো একটি সঠিক হলেই credit পাওয়া যায়। ১২-ঘণ্টার execution limit ও identical reruns-এর timing variation-ও documented। এই কারণে আপনার CV report-এ দুটি score আলাদা চাই: completed tasks-এর conditional solve rate এবং সব required test outputs ধরে effective score। Coverage কমে গেলে প্রথমটি ভালো থেকেও leaderboard খারাপ হতে পারে।

৬. Final GO / NO-GO
| সিদ্ধান্ত | Verdict |
|---|---|
| আগের চার-slot E17 notebook অপরিবর্তিত আবার চালানো | NO-GO |
| #43567 forced-fail isolated diagnostic | GO |
| S1 vLLM correctness + prefix-cache investigation | GO — highest priority |
| Existing FP8 KV scale থেকে silent corruption নিশ্চিত | UNVERIFIED |
| Deadline-aware CV ও submission checks | GO — CPU first |
| mikelou1-এর নির্দিষ্ট "tiny kernel issue" শনাক্ত | NOT FOUND |

শেষ সিদ্ধান্ত: আপনার সবচেয়ে ভালো S1 notebook যেহেতু vLLM ব্যবহার করে, এখন SGLang #43567-কে primary investigation রাখা ঠিক হবে না। আগে S1-এর MTP3 × prefix caching × FP8 KV correctness, তারপর re-prefill এবং final-drain coverage পরীক্ষা করুন। এগুলোর কোনোটি leaderboard-এ কত পয়েন্ট উদ্ধার করবে, paired ARC results ছাড়া নির্ভরযোগ্যভাবে বলা যায় না।
