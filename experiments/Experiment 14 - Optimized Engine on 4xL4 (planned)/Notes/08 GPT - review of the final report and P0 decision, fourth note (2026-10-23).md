# GPT note 4 (verbatim, pasted by the user 2026-10-23): review of the 476-line report, decision on the two P0 tasks

ARC-AGI-2 Experiment 14 — রিপোর্ট পর্যালোচনা ও সিদ্ধান্ত
আমার সিদ্ধান্ত: দুটি CPU-only P0 কাজের জন্য GO। GPU run-এর অনুমতি এখন নয়।

৪৭৬ লাইনের রিপোর্টে আগের গবেষণার বেশ কয়েকটি গুরুত্বপূর্ণ সংশোধন এসেছে। বিশেষ করে agent-এর re-prefill overhead-কে engine bottleneck-এর পাশাপাশি গুরুত্ব দেওয়া, TP2-এর KV-memory সীমাবদ্ধতা এবং measured ও estimated performance আলাদা রাখা—এগুলো সঠিক গবেষণা-দিক।

তবে দুটি P0 কাজের জন্য কিছু শর্ত আছে:
- E14-0 — Load-path optimization: GO for implementation and offline validation। বর্তমান working loader অক্ষত রেখে নতুন path optional করতে হবে। /dev/shm-এ সম্পূর্ণ model ধরবে ধরে নেওয়া যাবে না।
- E14-P0 — Offline token-diff: GO for investigation। কিন্তু পুরোনো run-এ generated token IDs সংরক্ষিত না থাকলে সম্পূর্ণ exact token-diff করা সম্ভব নয়। সে ক্ষেত্রে reconstructed comparison-কে exact proof বলা যাবে না।
GPU, Kaggle push, Duck solver, reasoning prompt, scheduler অথবা production model পরিবর্তনের অনুমতি এই সিদ্ধান্তে অন্তর্ভুক্ত নয়।
১. রিপোর্টের দাবিগুলোর ওপর আমার রায়
দাবি	মূল্যায়ন
115–140 ms decode step	লগভিত্তিক ফল; গ্রহণযোগ্য
140 all-reduce/step	Local source-count হিসেবে রিপোর্ট করা; profiler confirmation বাকি
Communication 35–55 ms	সম্ভাব্য, কিন্তু এখনো অনুমান
GDN-এর বড় projection INT4	রিপোর্টের config analysis যুক্তিসঙ্গত; মূল config এখানে সংযুক্ত নেই
Agent-এ re-prefill বড় bottleneck	শক্তিশালী log-derived evidence
Re-rendering-ই re-prefill-এর কারণ	এখনো অপ্রমাণিত
2×TP2-এ প্রায় 170k KV/replica	Sizing-model estimate; বাস্তবে মাপা নয়
#44401 merged	Web-verified: 15 October 2026 

GitHub
600–800 effective tok/s	লক্ষ্য ও সম্ভাব্য scenario; demonstrated throughput নয়
G4B ERROR-এর কারণ dataset mounting	সম্ভাব্য, নিশ্চিত নয়

Google Cloud-এর documentation-এ parallel-download performance improvement-এর বাস্তব সমর্থন আছে। কিন্তু Kaggle-এর নির্দিষ্ট GCSFUSE configuration এবং storage capacity আলাদা করে যাচাই করতে হবে। 

Google Cloud Documentation
+1




https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/performance
২. E14-0 — Load path implementation-এর নির্দিষ্ট নির্দেশ
এখানে আমি GO, কিন্তু কয়েকটি guard বাধ্যতামূলক রাখব।
অংশ	কীভাবে করবেন	Validation
Initial diagnostics	প্রথম executable cell থেকেই mount, storage, memory, heartbeat	Server না উঠলেও diagnostics পাওয়া যায়
Storage selection	/dev/shm-এ যথেষ্ট capacity থাকলে তবেই; নয়তো /tmp	Low-space simulation
Parallel copy	64 MiB ranges, প্রথমে bounded 4/8/16 workers	Sequential baseline-এর সঙ্গে তুলনা
File integrity	Temporary destination, complete size ও integrity check-এর পর atomic publish	Short-read, interrupted-copy, corrupt-file tests
Fallback	Copy ব্যর্থ হলে original mount থেকে load	Failure injection-এ working loader অপরিবর্তিত
Stall detection	Phase-aware progress এবং absolute deadline দুটোই	Progressing process অকারণে বন্ধ হয় না
JIT cache	Version/fingerprint-verified, mismatch হলে cold JIT	Wrong-version cache rejection

বিশেষ সংশোধন: 16–32 workers default করবেন না। প্রথমে concurrency sweep-এ যা দ্রুত ও স্থিতিশীল পাওয়া যায়, সেটি ব্যবহার করুন। অতিরিক্ত worker দিয়ে FUSE বা host memory overload করে আবার startup failure তৈরি করা যাবে না।

আরও একটি কথা: /dev/shm-এ weight files রাখলে সেই memory model loading ও serving-এর অন্য কাজের জন্য কমে যেতে পারে। GPU memory ঠিক থাকলেও host/cgroup memory exhausted হতে পারে। Copy-পরবর্তী host-memory headroom পরীক্ষা দরকার।

JIT cache-এর জন্য শুধু nv/ বাদ দেওয়া যথেষ্ট compatibility proof নয়। SGLang, Triton, CUDA, torch, GPU architecture, driver-relevant settings এবং kernel source fingerprint মিলতে হবে। Known-answer test না পাস করলে cache ফেলে cold compilation করতে হবে।
৩. E14-P0 — Offline token-diff কীভাবে করবেন
এখানে একটি methodological সমস্যা আছে।

রিপোর্ট বলছে historical requests-এ return_input_ids=False এবং return_output_ids=False ছিল। সেক্ষেত্রে generated sequence-এর আসল token IDs হয়তো সংরক্ষিত নেই।

একই decoded text-এর একাধিক tokenization হতে পারে। তাই শুধু tokenizer দিয়ে text আবার encode করলেই আগের generated token IDs ফিরে পাওয়া নিশ্চিত নয়।

এই কাজ দুই স্তরে করুন:
Stage 1 — Historical reconstruction (CPU-only)

Existing transcripts, request JSON, reasoning text, tool-call objects এবং pinned chat template থেকে পরের request reconstruct করুন।

Detect করুন:
- |trim whitespace বদলেছে কি না;
- <think> / </think> boundary বদলেছে কি না;
- Tool-call XML পুনর্গঠনে কোনো পার্থক্য আছে কি না;
- Reasoning এবং content-এর মাঝখানে নতুন separator এসেছে কি না;
- কোন character/tokenized span-এ reconstruction প্রথম আলাদা হচ্ছে।
ফলাফলের নাম হবে reconstructed_divergence, exact generated-token mismatch নয়।

Stage 2 — Exact token-ID validation

যদি historical raw output IDs সত্যিই না থাকে, পরবর্তী অনুমোদিত GPU diagnostic-এ ছোট কয়েকটি representative requests-এর জন্য actual generated IDs এবং subsequent rendered prompt IDs record করতে হবে।

তখনই first_divergent_token_index-কে exact হিসেবে প্রকাশ করা যাবে।

একটি গুরুত্বপূর্ণ test: এমন দুটি আলাদা token sequence ব্যবহার করুন যেগুলো decode করলে একই text হয়। Re-encode করে original sequence উদ্ধার করা যাচ্ছে—এমন ভুল assumption যেন test pass না করে।

তাছাড়া plain assistant reply, long thinking, submit(), check(), malformed tool call এবং history trimming—এই cases পৃথকভাবে পরীক্ষা করা উচিত।
৪. GPU run-এর পরিকল্পনায় আরও দুটি সংশোধন
প্রথমত, E14-A-র baseline profiler-এ #44401 এবং CUDA graph পরিবর্তন একসঙ্গে ঢোকানো ঠিক হবে না। তাহলে baseline আর original W1 থাকবে না। Existing unchanged engine-এর একটি সংক্ষিপ্ত measurement আগে নিতে হবে। এরপর patched run-এর সঙ্গে পার্থক্য হিসাব করা যাবে। #44401-এর merged implementation যাচাই হয়েছে, কিন্তু আপনার pinned 0.5.21 stack-এ backport behaviour এখনো পরীক্ষিত নয়। 

GitHub




https://github.com/sgl-project/sglang/pull/44401

দ্বিতীয়ত, correctness gate-এ McNemar p ≥ 0.01-কে correctness প্রমাণ হিসেবে ধরবেন না। Non-significant p-value মানে দুটি configuration equivalent, এমন নয়। Predefined allowed error difference, confidence interval, baseline repeatability এবং failure taxonomy রাখতে হবে। একই অল্প কয়েকটি prompt 150 বার চালালে 150টি independent ARC task পরীক্ষা করার সমতুল্যও হয় না।

DFlash2-এর concurrent-state issue public tracker-এ নথিভুক্ত আছে, তাই ছোট correctness gate রাখা যথার্থ। 

GitHub




https://github.com/sgl-project/sglang/issues/36548
৫. এখন Claude-কে যে সিদ্ধান্ত দেওয়া উচিত
Decision: GO for E14-0 and E14-P0 CPU-only work. NO-GO for GPU execution until separate approval.

Implement and offline-test the load-path optimization and investigate the historical prompt re-render/token-diff problem.

Mandatory constraints:
1. Keep the existing working load path as a fallback. Use capacity-checked /dev/shm or /tmp, bounded parallelism, atomic copy completion, integrity checks, and fail-open behavior. Do not assume 16–32 workers or sufficient RAM.
2. Validate the JIT cache with environment/source fingerprints. Never use an incompatible or unverified cache.
3. For token-diff, distinguish reconstructed text/token differences from exact generated-token-ID differences. If original token IDs were not recorded, report that limitation explicitly rather than claiming exact root-cause proof.
4. Do not modify Duck solver, prompts, scheduler, model, or serving algorithms.
5. Preserve all existing notebooks and baseline hashes. Make changes in separate E14 files.
6. Include concrete failure-injection tests, regression results, and a before/after diff.
7. Build the E14-A diagnostic notebook separately, but do not launch or push it.
8. For the later GPU test, measure the unchanged engine baseline before activating #44401 or extra CUDA graph sizes. Correctness and quota gates must remain separate.
After CPU validation, provide the README, code changes, tests, and exact GPU wall-time/quota/stop-point plan for a final audit.

Do not launch GPU, push a notebook, or submit anything without explicit approval.




সারকথা: রিপোর্টের গবেষণা-দিক ভালো, কিন্তু সবচেয়ে বড় লাভ এখনো estimate। দুটি CPU-only P0 কাজ শুরু করার পক্ষে আমি। প্রথমটির লক্ষ্য startup failure কমানো, দ্বিতীয়টির লক্ষ্য re-prefill-এর কারণ শনাক্ত করা। এই দুটো যাচাইয়ের পরে GPU-তে কী মাপতে হবে এবং কোন engineering fix নেওয়া উচিত, তা অনেক বেশি নির্ভুলভাবে সিদ্ধান্ত নেওয়া যাবে।
