<!-- Pasted by the user on 2026-10-24 (GPT deep web research). Kept VERBATIM. Claude's validation: Notes/29. -->

ARC Prize 2026 — Qwen3.8-27B Serving ও Competitor Research
Research cut-off: ২৪ অক্টোবর ২০২৬। এই রিপোর্টে তোমাদের 4×L4, PCIe/PHB, SGLang 0.5.21, INT4+DFlash2 এবং Duck-agent setup-কে কেন্দ্র করে প্রকাশিত benchmarks, repository, model cards, issue reports ও competition evidence বিচার করছি।

প্রতিটি finding-এর evidence strength আলাদা:
- M — Measured: লেখকের প্রকাশিত benchmark-এ hardware, workload ও সংখ্যা আছে। Independent reproduction বোঝায় না।
- S — Source-confirmed: published code/config থেকে আচরণ যাচাইযোগ্য।
- C — Claim: লেখক বলেছেন, কিন্তু যথেষ্ট benchmark artifacts নেই।
- H — Hypothesis: evidence থেকে যৌক্তিক অনুমান; GPU validation বাকি।
1. সবচেয়ে গুরুত্বপূর্ণ প্রাথমিক আবিষ্কার
4×L4-এর জন্য publicly reproducible 600–800 effective tok/s configuration এখনো পাইনি। তবে কয়েকটি published result দেখাচ্ছে, শুধু DFlash2 width কমানো নয়—MTP, batching, prefix reuse এবং inference-engine implementation বড় performance differences তৈরি করতে পারে।

বিশেষ করে তিনটি source গুরুত্বপূর্ণ:
1. Alexander Ollman — Qwen3.8 on RTX 3090: দুই replica-তে concurrency 16-এ MTP k=4 610.3 aggregate tok/s, DFlash2 k=7 401.6 tok/s। এটি একই workload-এর author-reported comparison; 4×L4-এ সরাসরি transferable নয়। 
   
   Qwen3.8-27B on RTX 3090
2. tonyd2wild — 4×RTX 3090 TP4: AutoRound W4A16 + DFlash2-তে concurrency 8 হলেও ordinary-prose throughput 145.3 aggregate tok/s; code-like predictable output-এ 230–243 tok/s পর্যন্ত। এটি দেখায় benchmark text distribution কতটা গুরুত্বপূর্ণ। 
   
   GitHub
3. moki1 — single RTX 3090 patched vLLM: concurrency 8-এ MTP 379.9 tok/s, DFlash2 373.7 tok/s; আলাদা highly batched configuration-এ concurrency 64-এ প্রায় 1,094 tok/s দাবি করেছে। এটি 4×L4-এর result নয় এবং ছোট prompt/answer workload ব্যবহার করে। 
   
   GitHub
তোমাদের মাপের সঙ্গে বিরোধ? সরাসরি নয়। GPU topology, TP/replication, context length, sampling, quantization এবং throughput denominator ভিন্ন।
Public leaderboard: ২৩ অক্টোবরের একটি যাচাইকৃত snapshot
২৪ অক্টোবরের কাছাকাছি স্বাধীন leaderboard archive-এ ARC-AGI-2-এর এই অবস্থান পাওয়া যাচ্ছে:
Rank	Team	Public score
1	Tufa Labs	88.06
2	Yi-Chia Chen	84.72
3	nvbanana	81.39
4	rabbithole	80.83
5	Majkel1337	75.56
6	Dawid Kopiczko	75.14
7	Domdolus Tolus	72.92
8	ALbert RUehlman	72.64
9	ｶﾎｳｼｭｳｾｲ	72.50
10	_hans	71.81



arc-explainer-production.up.railway.app


এটি ২৩ অক্টোবরের snapshot, চূড়ান্ত private result নয়। Kaggle-এর search-indexed leaderboard-এর কিছু পৃষ্ঠা পুরোনো snapshot দেখাচ্ছে; তাই আলাদা publication dates না মিলিয়ে বিভিন্ন ranking একত্র করা নিরাপদ নয়। Official Kaggle-এর নিয়ম অনুযায়ী public leaderboard আনুমানিক অর্ধেক test set-এ গণনা হয়। 

Kaggle




2. Published serving benchmarks — সবচেয়ে প্রাসঙ্গিক comparison
এগুলো বিভিন্ন লেখকের independently published measurements; একক standardized benchmark নয়। C = concurrent requests; throughput aggregate, যেখানে আলাদা বলা নেই।
Hardware / author	Engine, model, speculation	C	Reported tok/s	Evidence / relevance
2×RTX 3090, Alexander Ollman, 2026	Patched vLLM, two replicas, MTP k4	8	464.9	M; short ~1k prompts
একই	একই	16	610.3	M; 512-token outputs
একই	Two DFlash2 k7 replicas	8	410.7	M; MTP +13%
একই	Two DFlash2 k7 replicas	16	401.6	M; MTP +52% relative to DFlash2
4×RTX 3090, tonyd2wild	vLLM 0.27.1, TP4, AutoRound INT4, DFlash2	8	145.3	M; ordinary prose, short generation
একই	একই, structured repetitive outputs	1	230–243	M; not C8 aggregate
1×RTX 3090, moki1	Patched vLLM, MTP	8	379.9	M; 1,024-token realistic outputs
একই	Patched vLLM, DFlash2	8	373.7	M; same corpus
একই	Specialized batch mode, no equivalent long contexts	64	~1,094	C/M mixed; 128 input, 512 output
1×RTX 3090, ashalliants	NInfer, MTP3	8	522.8	M; thinking off, greedy
একই	NInfer, DFlash2	4	406.2	M; C8 didn't fit
4×L4, তোমাদের logs	SGLang 0.5.21, TP4, W4A16, DFlash2 width8	~12	355–460 decode; ~280 effective	User-measured baseline

Sources: 

Qwen3.8-27B on RTX 3090
+4




Raw URLs:

https://alexander-ollman.github.io/qwen3.8-on-rtx3090/results.html

https://github.com/tonyd2wild/Qwen3.8-27B-DFlash2-4x3090-TP4

https://github.com/moki1/qwen38-27b-rtx3090test

https://github.com/ashalliants/ninfer-3090
এই benchmarks থেকে কী প্রতিষ্ঠিত হয়?
MTP বনাম DFlash2: কিছু workload ও high concurrency-তে MTP অনেক ভালো। কিন্তু তোমাদের 4×L4-এ MTP superiority প্রতিষ্ঠিত হয়নি। Single-stream ও structured code generation-এ DFlash2 আবার ভালো হতে পারে।

TP4 বনাম replicas: 4×3090-এর published comparison-এ TP4 বেশি long-context capacity দেয়; TP2-এ single-stream latency ভালো। কিন্তু তোমাদের 50–100k retained context এবং 12 active streams-এর memory constraint থাকায় শুধু 610 tok/s দেখে 2×TP2-এ যাওয়া ঠিক হবে না।

Benchmark distortion: 128–1,024-token completion, thinking off, অথবা predictable repeated JSON-এর TPS দিয়ে 60k-token XHIGH Duck-agent throughput predict করা যাবে না।
Long-context evidence
tonyd2wild-এর TP4 setup-এর reported characteristics:
- vLLM 0.27.1, four RTX 3090, AutoRound W4A16;
- DFlash2 with seven draft tokens, FP8 KV;
- configured context 262,144;
- logical KV pool 1,003,062 tokens;
- approximately 3.83 full-length equivalent sequences at 262k context;
- startup 197 seconds, of which ~100 seconds compile।
এটি তোমাদের 4×L4-এর 502-second cold launch এবং ~866–881k KV pool-এর বিরুদ্ধে উপকারী comparison, কিন্তু RTX 3090-এর power, bandwidth, NVLink-pair topology ও engine আলাদা। 

GitHub




3. TP4, 2×TP2, PP2/PP4 ও data parallelism
Comparative assessment
Topology	4×L4-এ প্রধান সুবিধা	প্রধান বাধা	Current verdict
TP4	বড় shared KV pool; 12 long-running contexts	~140 collectives/verify step; PHB latency	Keep baseline
2×TP2	Independent engines; fewer ranks per collective	দ্বিগুণ model/SSM state per rank; কম KV per replica	Memory preflight ছাড়া নয়
PP2/PP4	Per-layer TP collectives কমানোর সম্ভাবনা	Microbatch pipeline bubbles; hybrid-state placement; speculative support	No qualified measured winner found
DP4 / four full replicas	No TP communication	27B+drafter+long KV per L4 cannot straightforwardly fit	Not viable as-is
PP2×TP2	Hybrid communication trade-off	More complex serving and scheduling, reduced flexibility	Research arm only

নতুন source-level warning: Qwen3.5 hybrid family-এর pipeline-parallel implementation-এ load/compatibility issues রিপোর্ট হয়েছে। SGLang #19500-এ PP2 startup failure এবং #37186-এ visual encoder weights PP stage-range filtering-এর সমস্যা ছিল। এই findings current release-এ কোন patches আছে যাচাই ছাড়া PP performance testing safe নয়। 

GitHub
+1




https://github.com/sgl-project/sglang/issues/19500

https://github.com/sgl-project/sglang/issues/37186

4×L4, 30–100k context, concurrency 8–32, Qwen3.8-27B-এর exact TP4 বনাম PP4 বনাম 2×TP2 controlled public benchmark: পাওয়া যায়নি।

এই শূন্যস্থান গুরুত্বপূর্ণ। অন্য PCIe GPU-তে PP দ্রুত হয়েছে এমন সাধারণ claim দিয়ে তোমাদের engine বদলানো উচিত নয়।
4. Speculative decoding: DFlash2, built-in MTP, EAGLE এবং correctness
4.1 আমাদের stack-এর সবচেয়ে বিপজ্জনক public finding
SGLang issue #36548 — L3tum, ২৬ আগস্ট ২০২৬; Gabriel Bryk-এর detailed reproduction, ২৭ আগস্ট। Evidence: independently reported paired experiment, কিন্তু আমাদের hardware-এ reproduced নয়।

এই issue-তে Qwen3.8-27B + DFlash2-এর concurrent requests-এ ভুল output এবং recurrent GDN-state divergence পরীক্ষা করা হয়েছে।
Configuration	Wrong outputs
DFlash2 width8, serial	0/100
DFlash2 width8, concurrent 8	Expanded affected subset-এ 111/304
Target-only, concurrent 8	0/304
FP32 SSM mitigation	37/304
CUDA graph disabled	37/304
Draft width4	37/304
Dense BF16 lm_head, অন্য setup একই	1/304



GitHub


https://github.com/sgl-project/sglang/issues/36548

ব্যাখ্যা: 111/304 হলো affected subset-এর expanded reproduction। এটিকে সব Qwen3.8 requests-এর 36.5% error rate বলা ভুল হবে। আর BF16 lm_head mitigation-টি NVFP4 target-এর reproduction-এ কাজ করেছে; তোমাদের RedHat INT4-এ একই লাভ দাবি করা যাবে না।
4.2 Thinking-enabled greedy divergence
Forlayo, ৪ সেপ্টেম্বর ২০২৬, SGLang #38009: চারটি RTX 3090, TP4, DFlash2 width8, BF16 target/draft, greedy sampling। Thinking enabled থাকলে token 30-এ deterministic divergence; thinking disabled থাকলে 64-token comparison match। 

GitHub




https://github.com/sgl-project/sglang/issues/38009

এই finding বিশেষভাবে তোমাদের XHIGH setting-এর জন্য প্রাসঙ্গিক। কারণ model অনেক reasoning tokens generate করলেও wrong program তৈরি করলে serving TPS লাভের গুরুত্ব কমে যায়।

Recommendation: E14 profiler-এর পাশাপাশি ছোট target-only বনাম DFlash2 correctness gate রাখো। Concurrent 1/6/12, reasoning on/off, greedy deterministic control এবং full-request generated token IDs তুলনা করবে। পরে same benchmark-এর Python-solution correctness দেখবে। শুধু greedy mismatch দেখে production failure ঘোষণা নয়; কিন্তু বড়/পুনরাবৃত্ত discrepancy উপেক্ষাও নয়।
4.3 DFlash checkpoint-এর target layers
Harshal-96, ১ সেপ্টেম্বর ২০২৬, SGLang #37475: DFlash checkpoint থেকে target_layer_ids হারিয়ে গেলে fallback ভুল tap positions ব্যবহার করতে পারে, অথচ shape match হওয়ায় model load হয়। Reported trained IDs [5,19,33,47,61] বনাম guessed [1,16,31,46,61]। 

GitHub




https://github.com/sgl-project/sglang/issues/37475

এটি source/config preflight দিয়ে GPU ছাড়াই আটকানো সম্ভব। তোমাদের drafter-এর actual config.json থেকে layer IDs, block size, hidden layers এবং trained model compatibility assert করবে।
4.4 কোন draft algorithm সবচেয়ে ভালো?
Algorithm	Public evidence	4×L4 recommendation
DFlash2 width8	তোমাদের measured baseline; structured-code acceptance ভালো হতে পারে	Baseline retain
DFlash2 width6/4	Single-3090 llama.cpp-তে n4–5 বেশি efficient; অন্য engine ও workload	Controlled comparison only
Built-in MTP, EAGLE 3/1/4	Two-replica 3090 C16 benchmark-এ DFlash2 থেকে অনেক ভালো	সর্বোচ্চ-priority alternative
DSpark	SGLang cookbook-এ supported, separate draft model; extra memory	Lower priority
EAGLE-4	AMD MI355X-এ বড় measured gains	Interesting, কিন্তু L4 setup-এ direct proof নেই
N-gram / suffix / context lookup	Repetitive output-এ লাভ হতে পারে	ARC code-generation-specific experiment
EAGLE-3 separate drafter	Strong research, কিন্তু exact Qwen3.8/L4 pretrained compatible drafter ও benchmark পাওয়া যায়নি	Not immediate

Official SGLang recipe-তে built-in MTP:
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4


DFlash2 recipe:
--speculative-algorithm DFLASH \
--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
--speculative-num-draft-tokens 8


Source: SGLang Qwen3.8 cookbook, current published documentation। 

GitHub




https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx

EAGLE-4 — AMD, ২০ অক্টোবর ২০২৬: AMD-এর published benchmark-এ এক MI355X-এ Qwen3.8-27B BF16, C16-এ EAGLE-4 2,167 tok/s, built-in MTP 1,743 tok/s। C32-এ যথাক্রমে 2,784 ও 2,551। এটি EAGLE-4-এর high-load potential দেখায়, কিন্তু MI355X-এর GPU memory ও bandwidth সম্পূর্ণ ভিন্ন। 

amd.com




https://www.amd.com/en/developer/resources/technical-articles/2026/eagle-4-on-amd-instinct-gpus.html

DFlash paper: Jian Chen, Yesheng Liang, Zhijian Liu, ৫ ফেব্রুয়ারি ২০২৬। Block-diffusion drafting-এর theoretical ও measured speed benefits বর্ণনা করে। তবে paper-এর headline speedup তোমাদের high-batch PCIe hardware-এ expected speedup নয়। 

arxiv.org




https://arxiv.org/abs/2602.06036

5. Weight formats, GDN state এবং hybrid prefix caching
5.1 Quantization decision
Format	Public status	Ada L4-এ relevance
RedHat INT4 W4A16	তোমাদের working measured baseline	Keep
Official Qwen FP8 W8A8	Qwen published, Apache-2.0, 128-block quant	Quality/performance research arm; higher memory
vrfai FP8	Mixed FP8 language model, sensitive layers BF16; published evaluation	Accuracy comparison, not established fastest
huginnfork FP8_DYNAMIC	MLP-only quantization; attention/GDN BF16	More memory, possibly precision benefit
GPTQ/AWQ/AutoRound INT4	Multiple 3090 recipes	Kernel-specific gain possible; conversion/revalidation needed
NVFP4 W4A4	Blackwell-oriented acceleration	L4 sm_89 lacks native FP4 acceleration; not an easy upgrade

Official Qwen FP8: Qwen's model card states fine-grained FP8 block-size 128 and near-original performance, but does not establish a 4×L4 speed advantage over W4A16. 

Qwen/Qwen3.8-27B-FP8 at main




https://huggingface.co/Qwen/Qwen3.8-27B-FP8

vrfai FP8: Author reports 28.3 GB weights, 93.28% Linear weights quantized, with GDN in_proj_a/b, lm_head, vision left BF16। Reported MathVision testmini score 84.54 versus BF16 86.18। এটি model-quality trade-off, ARC score নয়। 

vrfai/Qwen3.8-27B-FP8 at main




https://huggingface.co/vrfai/Qwen3.8-27B-FP8

huginnfork FP8_DYNAMIC: MLP-only quantization, 35.8 GiB; attention ও GDN BF16। এখানে large recurrent projections BF16 রাখার সিদ্ধান্ত author-specific, তোমাদের RedHat INT4 quantization-কে ভুল প্রমাণ করে না। 

huginnfork/Qwen3.8-27B-FP8 at main




https://huggingface.co/huginnfork/Qwen3.8-27B-FP8
5.2 GDN/Mamba cache-এর তিনটি গুরুত্বপূর্ণ upstream issue
Issue / date	Reported behaviour	আমাদের সিদ্ধান্ত
#43204, Henry Tsang, ৮ অক্টোবর	সব cached Mamba states locked থাকলে allocation assertion engine kill করতে পারে	Actual scheduler path audit
#45400, jhsmith409, ২০ অক্টোবর	HiCache host-tier-এ 3 long prefixes কাজ করে, 4-এ cache hit 0	HiCache enable নয়
#39830, সেপ্টেম্বর	Hybrid model host-tier hit-এর পরে wrong output	Accuracy gate ছাড়া host cache নয়



GitHub
+2


https://github.com/sgl-project/sglang/issues/43204

https://github.com/sgl-project/sglang/issues/45400

https://github.com/sgl-project/sglang/issues/39830

#45400-র 20 অক্টোবরের reproduction: চারটি ~89k-token documents-এর host-tier cache test-এ cold re-prefill ~23 seconds, দ্বিতীয় query-তেও 0 cached tokens। এটি SGLang 0.5.22, NVFP4, EAGLE/ReplaySSM, PP2 hardware—অর্থাৎ আমাদের exact stack নয়। তবু hybrid-state cache design-এর বাস্তব সতর্কতা।

Franzen-এর public ARC-AGI-3 write-up, Daniel Franzen, ২০২৬:
Prefix metric, earlier demo run	Reported
Weighted prompt reuse	93.43%
Long-prompt cache misses	1/3,248
P90 effective prefill	10,553 tok/s
P90 aggregate decode	837 tok/s
Harness-effective generation	609 tok/s

এগুলো RTX PRO 6000-ভিত্তিক ARC-AGI-3 donor experiment-এর সংখ্যা; তোমাদের 4×L4-এ expected TPS নয়। তবে reasoning history retain করা, larger-block trim hysteresis, end-of-prefill Mamba checkpoint এবং cache-aware admission—এই mechanism-গুলো সরাসরি গুরুত্বপূর্ণ। 

GitHub




https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md

আরেকটি গুরুত্বপূর্ণ insight: Franzen write-up-এ reasoning history সরিয়ে ফেললে performance regression-এর কথা আছে। তাই cache optimization করতে গিয়ে Duck-এর accumulated thinking indiscriminately delete করা উচিত নয়।
6. PCIe/PHB communication: তোমাদের নতুন measurement কী বদলে দিয়েছে?
তোমাদের নতুন finding—NCCL P2P/IPC, SHM-এর চেয়ে 5.5× ধীর—এই গবেষণার recommendations বদলে দেয়।

অন্য PCIe system-এ NCCL_P2P_LEVEL=PHB লাভ দিয়েছে বলে তোমাদের Kaggle L4-এ সেটি force করার কোনো যুক্তি নেই।
Published comparative evidence
Eduardo Pessin, ১ অক্টোবর ২০২৬; HyperQwen #254

4×RTX 5060 Ti, PCIe Gen3 x8, PHB, patched P2P-capable driver, NCCL 2.29.7, vLLM 0.30:
Concurrency	SHM/default → P2P decode gain
C1	প্রায় +2–6%
C2	+10.1%
C4	+10–16%
C8	Not reliably established

Exact message-size benchmark-এ 81,920-byte all-reduce 64.09→55.51 µs; 1.31 MB-এ 498.09→467.28 µs। 

GitHub




https://github.com/syv-ai/HyperQwen/issues/254

তোমাদের result-এর সঙ্গে contradiction? হ্যাঁ, optimization direction-এ। তবে technical contradiction নয়: তাদের patched consumer P2P fabric/host-এ লাভ, তোমাদের Kaggle PHB VM-এ তীব্র ক্ষতি। PCIe topology label এক হলেও effective transport একই নয়।

NVIDIA NCCL documentation-ও বলে NCCL_P2P_LEVEL, protocol ও algorithm override performance কমাতে পারে। 

NCCL 2.32.3 documentation




https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
All-reduce optimization options
Option	Public evidence	4×L4 decision
NCCL SHM baseline	তোমাদের measured fastest transport	Keep
NCCL_P2P_LEVEL=PHB/SYS	অন্য hardware-এ +2–16%	তোমাদের measured 5.5× regression—reject
NCCL_SHM_USE_CUDA_MEMCPY=1	Mixed outcomes across NVIDIA hardware	Exact 983 KB A/B ছাড়া নয়
FlashInfer PCIe-IPC AR	SGLang PR #34528; newer FlashInfer module	L4 support/correctness/peer memory unverified
Custom all-reduce TP4	vLLM/SGLang 4 PCIe GPUs-এ normally disabled	Force করা high-risk
Quantized/FP8 all-reduce	Bandwidth-saving hypothesis	No validated drop-in 4×L4 solution found
Vocab-parallel logits selection	Could reduce all-gather volume	Nontrivial sampling-correctness patch

FlashInfer PCIe-IPC: AliceChenyy-এর SGLang PR #34528 এবং FlashInfer PR #4393-তে implementation পাওয়া যায়। কিন্তু প্রকাশিত benchmark মূলত newer GPUs/SM120-oriented। তোমাদের pinned FlashInfer 0.6.18-এ প্রয়োজনীয় module নেই—তাই 0.5.21-এ শুধু environment flag দিলেই এটি কাজ করবে না। 

GitHub
+1




https://github.com/sgl-project/sglang/pull/34528

https://github.com/flashinfer-ai/flashinfer/pull/4393

Custom TP4 all-reduce: club-3090-এর August 2026 report-এ চার PCIe GPU-তে custom all-reduce disabled ছিল, P2P transport available থাকা সত্ত্বেও। Forced custom path-এ correctness issues-ও রিপোর্ট হয়েছে। 

GitHub




https://github.com/noonghunna/club-3090/discussions/773

সিদ্ধান্ত: তোমাদের 56 ms/step TP4 all-reduce cost যদি profiler-verified হয়, এটিই সবচেয়ে বড় measured engine bottleneckগুলোর একটি। কিন্তু সেটি সরানোর আজ publicly verified, safe L4-specific patch পাওয়া যায়নি।

7. SGLang versions, scheduling এবং startup
#44401 — একটি গুরুত্বপূর্ণ low-risk candidate
hnyls2002, merged ১৫ অক্টোবর ২০২৬, SGLang PR #44401।

Continuing chunked-prefill request-এর held req_to_token row admission-এ double-count হওয়ার issue সংশোধন করে। 

GitHub




https://github.com/sgl-project/sglang/pull/44401

তোমাদের আগের run-এ 11 running + 1 queued অবস্থার সঙ্গে এটি মেলে। তবে বর্তমানে #44401 backport করা হয়েছে কি না এবং isolated gain কত—তা আলাদা মাপা প্রয়োজন।

Expected gain: code fix-এর নির্দিষ্ট speedup প্রকাশিত নয়। তোমাদের observed 11→12 occupancy gap অনুযায়ী whole-run +0–8% একটি planning range হতে পারে, benchmarked guarantee নয়।
Version migration
Version/path	Evidence	Recommendation
SGLang 0.5.21 pinned	তোমাদের working CUDA graph/DFlash setup	Production baseline
0.5.21 + #44401	Narrow scheduler patch	Isolated candidate
0.5.22	New serving/cache options, newer FlashInfer	Full upgrade এখন high regression cost
Nightly	More fixes, less pinned stability	GPU validation ছাড়া নয়
vLLM patched	3090 results promising	Separate engine migration, not one-line optimization

CUDA graph sizes: 1–13 explicit capture পরীক্ষা করতে পারো; 1..8,10,12-এর padding এড়ানোর সম্ভাবনা আছে। লাভ GPU benchmark ছাড়া নির্ধারণ করা যাবে না, আর boot/graph-capture time ও memory বাড়তে পারে।
Startup/JIT/public evidence
Source	Observation	Relevance
tonyd2wild, 4×3090 TP4	Startup 197s, ~100s compile	তোমাদের 502s cold launch বেশি; hardware/storage আলাদা
SGLang cookbook, DGX Spark	BF16 checkpoint load প্রায় 6.5 মিনিট; ~10 মিনিট READY allowance	Slow startup বাস্তব হতে পারে
Kaggle participant reports, ১৬–২০ অক্টোবর	Model server 20 মিনিটের বেশি startup, zero-score/early exit reports	Contest operational risk বাস্তব
Hugging Face Xet docs	Adaptive parallel chunk transfers, explicit cache controls	Hub downloads-এর জন্য; mounted Kaggle datasets-এর direct benchmark নয়



GitHub
+3


https://huggingface.co/docs/hub/xet/using-xet-storage
Startup optimization — safe order
আমি Kaggle-এর /kaggle/input অথবা নির্দিষ্ট GCSFUSE/NFS filesystem-এর কারণে তোমাদের 502s startup—এমন definitive evidence পাইনি।

তবে diagnosis করার জন্য weights-read time, safetensors mmap time, CUDA JIT compile, graph capture এবং SGLang ready-to-serve—পাঁচটি আলাদা timestamps দরকার।

Parallel staging copy-এর ক্ষেত্রে role-unique destinations, size/hash validation ও atomic rename থাকবে। JIT caches reuse করবে কেবল torch/Triton/FlashInfer/CUDA version-compatible keys দিয়ে। Cache warm হলেই output correct—এমন নিশ্চয়তা নেই।

Kaggle storage-specific faster copy recipe with reproducible 4×L4 Qwen3.8 numbers: পাওয়া যায়নি।
8. ARC-AGI-2 competitors: কার কী public method পাওয়া গেছে?
সবচেয়ে গুরুত্বপূর্ণ সতর্কতা: উঁচু leaderboard score থেকে model, agent, inference engine বা ensemble infer করা যায় না। Tufa Labs-এর ARC-AGI-3 Duck harness প্রকাশিত, কিন্তু সেটিই তাদের ARC-AGI-2-এর 88.06 solution—এমন দাবি করার প্রমাণ নেই।
Top competitors, ২৩ অক্টোবর ২০২৬
Team	Public evidence found	Method status
Tufa Labs — 88.06	ARC-AGI-3 Duck harness GitHub, July 1 blog, technical presentation; ARC2 method পরে জানাবে বলেছে	ARC2-specific method পাওয়া যায়নি
Yi-Chia Chen — 84.72	Public leaderboard score ও ARC-AGI-3 participation	ARC2 method/repo পাওয়া যায়নি
nvbanana — 81.39	Team roster-এ Jean-François Puget-এর cpmpml account থাকার evidence	ARC2 exact model/recipe পাওয়া যায়নি
rabbithole — 80.83	Leaderboard history	Public technical recipe পাওয়া যায়নি
Majkel1337 — 75.56	October leaderboard progression	Recipe পাওয়া যায়নি
ALbert RUehlman — 75.56	Leaderboard record	Recipe পাওয়া যায়নি
Dawid Kopiczko — 75.14	Leaderboard record	Recipe পাওয়া যায়নি
Domdolus Tolus — 73.47	Leaderboard record	Recipe পাওয়া যায়নি
ｶﾎｳｼｭｳｾｲ — 72.50	Leaderboard, Japanese-community results	Specific source/model পাওয়া যায়নি
_hans — 71.81	Leaderboard history	Recipe পাওয়া যায়নি
Youssef Nader	Contest ranking/progression	Exact submission method পাওয়া যায়নি
mikelou1	Kaggle first-person Qwen discussion; NVARC denial; local-run claims	Partial evidence, not full recipe

Leaderboard source (২৩ অক্টোবর ২০২৬, 1:05 PM EDT): 

arc-explainer-production.up.railway.app




https://arc-explainer-production.up.railway.app/arc-leaderboards
Tufa Labs: publicly reusable কিন্তু আলাদা competition-এর method
Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tešnar, Stefano Viel — ১ জুলাই ২০২৬।

Tufa-এর official ARC-AGI-3 Duck publication:

https://tufalabs.ai/research/duck-harness/

https://github.com/Tufalabs/duck-harness

Tool-using Python coding agent, persistent state ও executable environment-modeling। তোমাদের Duck approach এই family-র সঙ্গে মেলে। কিন্তু ARC2-তে তাদের selection, fine-tuning, runtime ও score contribution অপ্রকাশিত। 

Tufa Labs



mikelou1: সবচেয়ে মূল্যবান competitor statement
Kaggle discussion থেকে publicly archived observations:
- অক্টোবর ১৮: NVARC ensemble ব্যবহারের কথা অস্বীকার; একটি lower result-এর পেছনে kernel issue উল্লেখ।
- তখন প্রায় 66% public score ও ~60% local validation report।
- অক্টোবর ২০: প্রায় 87% local Qwen run দাবি করেছেন, 4×L4-equivalent time constraint-এ; নিজেই বলেছেন transfer নিয়ে সন্দেহ আছে এবং ~75% আশা করছেন।
- পূর্ণ model variant, engine config, task scheduler বা source প্রকাশ করেননি।
এগুলো author claims, verified 87% Kaggle score নয়। 

The ARC Daily Digest




Direct discussions:

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546

Research implication: তোমাদের মতো Qwen-based offline system দিয়েই বড় score সম্ভব—এমন একটি credible lead আছে। কিন্তু 34.58→70+ gap-এর মূল কারণ শুধু tok/s, model tuning নাকি candidate search—তা এই statement থেকে জানা যায় না।
Public Kaggle notebooks: যা বাস্তবে পাওয়া গেছে
Source	Reported result	Method significance
sorenravn/arc2-vanilla-exact	31.39 (public research repo claim)	Qwen grid baseline, low-cost
christopherdaleman/arc-2026-nvarc-trm-evidence-cost-v1	31.11	NVARC/TRM + evidence selection
johntaylorai/swarm-arc2-lb3389-baseline	29.72 notebook	Public Qwen 4B baseline
yusuketogashi/arc-baseline-rebuild	Best 31.81	4×L4, 6h23 run, Qwen grid model
Andres Zapke Qwen+LoRA starter	No verified strong score here	Training/inference starter



GitHub
+3


URLs:

https://github.com/OpenKaggle/arc-agi-2-paper-track-research

https://www.kaggle.com/code/johntaylorai/swarm-arc2-lb3389-baseline

https://www.kaggle.com/code/yusuketogashi/arc-baseline-rebuild

https://www.kaggle.com/code/andreshzapke/starter-notebook-qwen-lora-stop-loss

উল্লেখযোগ্য সীমা: এই notebooks-গুলো top 70–88% teams-এর published solutions নয়।

9. ARC agent/harness গবেষণা: model থেকে বেশি কাজ কীভাবে বের করা যায়?
9.1 Stateful Python REPL — সবচেয়ে প্রতিষ্ঠিত design choice
gutfeeling / Pivotools, ২০২৬ public repository

ARC-AGI-2-তে stateful IPython tool agent plain chain-of-thought baseline-এর তুলনায় কিছু model-এর accuracy অনেক বাড়িয়েছে; GPT-OSS 120B high-এর জন্য 4×-এর বেশি improvement report করেছে। এটি user-authored empirical comparison, Qwen3.8-specific measured result নয়। 

GitHub




https://github.com/gutfeeling/arc-agi-2-submission

তোমাদের existing Duck architecture এই evidence-এর সঙ্গে সঙ্গতিপূর্ণ। তাই Python REPL বাদ দিয়ে pure CoT-তে ফেরা উচিত নয়।
9.2 Multi-hypothesis portfolio, pass@2 এবং executable verification
User's Kaggle rules অনুসারে প্রতিটি test input-এ দুই output দেওয়ার সুযোগ আছে। তাই train-exact কিন্তু test-wrong candidate-এর পরেও materially different program রাখা মূল্যবান। 

ARC-AGI-2 Competition




তোমাদের Rev5/R6-এর host-clean verification ও distinct-output selection এই design-এর সঙ্গে যুক্তিসংগত। কিন্তু GPU-effective budget-এর মধ্যে দ্বিতীয় hypothesis তৈরি করাই bottleneck।

Recommended harness idea: প্রথম candidate exact হলে auto-stop নয়; remaining budget, structural ambiguity ও already distinct predictions দেখে selective continuation। Model-written transform-এর source fingerprint, train-pair pass count ও test-grid disagreement রাখতে হবে।
9.3 Genetic search seeded by LLM
Val Dyachenko, ২৩ সেপ্টেম্বর ২০২৬, arXiv 2609.27242।

Quantized Qwen3.5-4B দিয়ে initial DSL programs, তারপর genetic evolution। 60 public ARC2 tasks-এ initial LLM 2/60, genetic search আরও 4, মোট 6/60 (10%)। Pure genetic search without LLM seeds কোনো solve দেয়নি। 

arxiv.org




https://arxiv.org/abs/2609.27242

Relevance: Duck-এর train-exact candidate থেকে bounded rule variants তৈরি করার ধারণা গবেষণাগতভাবে সমর্থিত। কিন্তু broad genetic search এখন যোগ করলে code/risk বাড়বে এবং 10% paper result তোমাদের model-এ সরাসরি transferable নয়।
9.4 Persistent executable hypotheses
Schema — Guanning Zeng et al., ৩০ সেপ্টেম্বর ২০২৬।

Agent evolving hypotheses-কে executable programs হিসেবে ধরে, history-র বিরুদ্ধে verify করে। ARC-AGI-3-তে same base-model setup-এ 58.7→99.2 RHAE author-reported। এটি ARC2 নয়; তবু program check early design-এর গুরুত্ব দেখায়। 

arxiv.org




https://arxiv.org/abs/2609.39140
9.5 Thought/artifact analysis
Chen Wu, Josh Passenger, Yin Song, ৮ অক্টোবর ২০২৬।

ARC-AGI-3 coding-agent traces-এ 630 script references-এর মধ্যে মাত্র 33টি task boundary পার করেছে; পুরোনো script-এর 74% abandoned। Hard-coded state একটি বড় transfer error। 

arxiv.org




https://arxiv.org/abs/2610.11450

ARC2 implication: 4c3d4a41-এর fixed-coordinate overfit-এর মতো ঘটনা শনাক্ত করতে structural generalization feedback কার্যকর hypothesis। তবে known-solution rejection rule বানানো উচিত নয়।
9.6 Reasoning-efficient Qwen fine-tunes
Swift-Qwen3.8-27B — ukisai, September 2026 study

Independent Alexander Ollman test: 60 LiveCodeBench tasks-এ base 63.3%, Swift variant 71.7%; median output tokens 14,190→7,035, প্রায় 50% কম। এটি আমাদের ARC2 Duck run-এর result নয়। Swift-এর license Swift Open License 1.0, Apache-2.0 নয়। Prize eligibility যাচাই না করে ব্যবহার ঠিক হবে না। 

Qwen3.8-27B on RTX 3090
+1




https://huggingface.co/ukisai/Swift-Qwen3.8-27b

ThinkingCap-Qwen3.8-27B — BottleCap AI, September 2026

Author-published model, license PolyForm Small Business 1.0 + personal-use grant, with separate commercial licensing। ARC2 Kaggle prize eligibility source থেকে প্রতিষ্ঠিত নয়। 

bottlecapai/ThinkingCap-Qwen3.8-27B at main




https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B

NVIDIA Nemotron-SFT-ARC-AGI-v1 — March 2026

Public multi-turn ARC tool traces dataset; data card states CC BY 4.0, with other included licensing notes। Training data/research-এর জন্য প্রাসঙ্গিক; already-trained Qwen3.8 ARC champion model নয়। 

nvidia/Nemotron-SFT-ARC-AGI-v1 at main




https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1

Qwen3.8-Flash-Next licensing: Official model card-এ qwen-community-1.0 লেখা আছে, Apache-2.0 নয়। ARC-AGI-3 Kaggle discussions-এ eligibility নিয়ে প্রশ্ন রয়েছে, কিন্তু definitive organizer ruling পাইনি। 

Hugging Face
+1




https://huggingface.co/Qwen/Qwen3.8-Flash-Next

আমার সিদ্ধান্ত: Prize eligibility গুরুত্বপূর্ণ হলে এখন RedHat/Qwen base-এর established license lineage অক্ষত রাখো। New fine-tune বা Flash-Next-এর ক্ষেত্রে rules/licensing review ছাড়া final entry বদলাবে না।
10. Ranked speed changes — তোমাদের exact 4×L4 setup
এখানে expected gain-গুলো তোমাদের measured 280 effective tok/s baseline-এর বিরুদ্ধে। যেখানে exact L4 experiment নেই, সেখানে range-কে engineering hypothesis বলছি, proven gain নয়। একাধিক gain সরাসরি যোগ করা যাবে না।
Rank	Change	Potential gain	Evidence	Risk / decision
1	Prefix re-prefill ও Mamba checkpoint diagnostic/fix	+10–35% H	তোমাদের prompt-cache losses; Franzen 93.43% reuse	Medium–High; first do CPU token-diff, then cache-specific regression
2	#44401 scheduler backport, verify 12/12 occupancy	+0–8% H	Upstream PR + local 11+1 observations	Low–Medium; narrow source patch
3	Built-in MTP 3/1/4 vs DFlash2 width8	Unknown; +10–30% plausible H, regression possible	3090 high-C comparison	Medium–High; target-only parity and GDN correctness gates
4	CUDA graph decode sizes 1..13	+0–5% H	Existing graph padding overhead	Low–Medium; extra startup/memory
5	Prefill/decode scheduling, including chunk size and --prefill-decode-interval	+0–15% H, highly workload-dependent	SGLang scheduler source; public discussion	Medium; TTFT/retraction/tail latency
6	DFlash2 width 8/6/4 A/B	−10% to +12% H	DFlash research; single-GPU observations	Medium; acceptance and quality parity
7	NCCL SHM microbenchmark tuning	0–10% H	PHB benchmarks mixed; own P2P regression	Medium; retain proven SHM default
8	FP8 or alternative INT4 kernel	Unknown	Published checkpoint and Marlin/vLLM recipes	High; quantization accuracy and memory
9	Vocab-parallel logits / verification-kernel work	Unknown, possible single-digit to low-teens	Source-level overhead hypothesis	High engineering and correctness risk
10	Full engine upgrade or 2×TP2/PP conversion	Unknown, negative possible	Indirect PCIe benchmarks only	High; not now

আমার top-three decision
First: prefix/cache। কারণ decoding faster হলেও repeated prefill কার্যকর সময় খেয়ে ফেলে। বর্তমান source-level uncertainty মেটাতে generated token IDs ও subsequent re-rendered input IDs-এর comparison সবচেয়ে জরুরি। একই সঙ্গে Mamba checkpoint hit/eviction test করতে হবে। এটি CPU diagnostic ও bounded GPU correctness test—সরাসরি production cache patch নয়।

Second: #44401 + graphs। Narrow, independently testable engine changes। E14-A-তে isolated arm comparisons ও graph correctness already planned। অন্য configuration accidentally adopted হচ্ছে কি না verify করতে হবে।

Third: MTP vs DFlash2। এই গবেষণার সবচেয়ে মূল্যবান নতুন performance alternative। 2×3090 benchmark high-concurrency-তে MTP superiority দেখিয়েছে; তোমাদের 12-stream Qwen run-এর সঙ্গে directional মিল আছে। কিন্তু DFlash2→MTP বদলানোর আগে paired ARC correctness check বাধ্যতামূলক।
Throughput বাড়লে 240-task budget
শুধু planning arithmetic—11.3 ঘণ্টা সম্পূর্ণ effective generation ধরে:
Generated tokens available per task at different sustained effective rates
240 tasks, 11.3 hours; ignores changes in scheduling efficiency and model accuracy.


040,00080,000120,000160,000280 tok/s400 tok/s500 tok/s600 tok/s800 tok/s














বর্তমান ~280 effective tok/s-এ 240 task-এর average generated-token capacity প্রায় 47.5k/task। 400-এ ~67.8k, 600-এ ~101.7k।

কিন্তু দ্বিগুণ tokens মানেই দ্বিগুণ ARC score নয়। Code hypothesis quality, late-correct programs, generalization ও pass@2 diversity সমানভাবে গুরুত্বপূর্ণ।

11. Separate ranked harness ideas
Rank	Idea	Evidence	Recommendation
1	Earlier executable candidate + check()	ARC2 agentic coding, existing Rev5/R6 traces	Keep improving
2	No repeated grid reconstruction; targeted inspection, compact spatial views	Your Rev6d failure analysis	High priority
3	Progress-aware allocation with genuine first exposure	Franzen donor + your 240-task requirement	Keep, tune on GPU
4	Two distinct train-exact programs, not duplicate outputs	ARC pass@2 scoring	Keep
5	Continue after wrong first train-exact	Your archived late corrections	Mandatory
6	Preserve thinking + coherent tool/result blocks	Duck donor; cache findings	Mandatory
7	Detect hard-coded offsets, coordinates and input-size overfit	ARC papers and your failures	Short feedback, not automatic rejection
8	Task-local lightweight program variants	LLM-seeded search paper	Optional bounded experiment
9	Alternative representation for a genuinely independent branch	Facts-view failures / diversity hypothesis	Test, not blanket duplicate compute
10	ARC-tuned distillation or token-efficient Qwen	Nemotron traces / Swift results	Longer-term; licensing and training time

আমি এখন নতুন orchestration framework, judge ensemble, broad genetic algorithm অথবা new model migration যোগ করব না। Competition deadline ২ নভেম্বর ২০২৬; remaining experiments-এ reproducibility ও runtime correctness বেশি মূল্যবান।

12. কোথায় খুঁজেও actionable evidence পাওয়া যায়নি?
GitHub, Hugging Face model cards/discussions, Kaggle competition pages/notebooks, ARC Prize, arXiv, Reddit, X-এর indexed posts, YouTube, Bilibili, Qiita, Zenn ও Zhihu-তে ২০০টির বেশি targeted search query ব্যবহার করেছি; একই তথ্যের পুনরাবৃত্ত results বাদ দিয়েছি।

এই বিষয়গুলোতে প্রয়োজনীয় মানের public proof পাওয়া যায়নি:
- 4×L4, Qwen3.8-27B, 30k–100k context, concurrency 12-এ stable 600–800 agent-effective tok/s।
- তোমাদের exact RedHat INT4 + BF16 DFlash2 + SGLang 0.5.21 setup-এর আগে থেকে প্রকাশিত faster reference configuration।
- 4×L4-এ controlled TP4 বনাম PP2/PP4 বনাম 2×TP2 long-context benchmark।
- Tufa Labs, Yi-Chia Chen, rabbithole বা nvbanana-এর বর্তমান top-scoring ARC2 notebook-এর সম্পূর্ণ, reproducible method।
- Tufa-এর ARC2 score-এর জন্য Duck, distilled model, NVARC বা ensemble-এর নির্দিষ্ট contribution।
- Kaggle PHB L4-এ নিরাপদ ও faster FP8/quantized all-reduce-এর publicly tested drop-in kernel।
- Qwen3.8-27B-এর exact ARC-AGI-2 prize-eligible short-thinking fine-tune, যার 240-task hidden-like validation ও licença নিশ্চিত।
- Kaggle-mounted model files-এর জন্য universally fastest GCSFUSE/NFS staging strategy with measured 4×L4 results।
Japanese/Chinese-language search: Bilibili-তে local Qwen inference benchmarks পাওয়া গেছে; কিছু Japanese ARC contender score posts-ও পাওয়া গেছে। কিন্তু 4×L4-এ তোমাদের চেয়ে দ্রুত, exact flags-সহ reproducible Qwen3.8-serving method অথবা top competitors-এর গোপন implementation পাওয়া যায়নি।

Discord: Publicly indexed, technically verifiable transcript থেকে নতুন actionable result পাওয়া যায়নি। Private Discord conversations আমার কাছে accessible নয়; সেগুলো পড়েছি এমন দাবি করছি না।

13. Research reference index — raw URLs
নিচেরগুলো সবচেয়ে কার্যকর links, সরাসরি copy করা যাবে।

Serving performance ও model benchmarking
https://alexander-ollman.github.io/qwen3.8-on-rtx3090/results.html
https://github.com/tonyd2wild/Qwen3.8-27B-DFlash2-4x3090-TP4
https://github.com/tonyd2wild/Qwen3.8-27B-3090-Cookbook
https://github.com/moki1/qwen38-27b-rtx3090test
https://github.com/ashalliants/ninfer-3090
https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9
https://www.reddit.com/r/LocalLLaMA/comments/1vsccit/qwen3827b_on_2x_3090_vllm_dflash2_218_toks_single/
https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
https://www.amd.com/en/developer/resources/technical-articles/2026/eagle-4-on-amd-instinct-gpus.html
https://arxiv.org/abs/2602.06036


SGLang, NCCL, CUDA ও correctness
https://github.com/sgl-project/sglang/pull/44401
https://github.com/sgl-project/sglang/issues/36548
https://github.com/sgl-project/sglang/issues/38009
https://github.com/sgl-project/sglang/issues/37475
https://github.com/sgl-project/sglang/issues/28180
https://github.com/sgl-project/sglang/issues/35150
https://github.com/sgl-project/sglang/issues/43204
https://github.com/sgl-project/sglang/issues/45400
https://github.com/sgl-project/sglang/issues/39830
https://github.com/sgl-project/sglang/issues/38815
https://github.com/sgl-project/sglang/issues/19500
https://github.com/sgl-project/sglang/issues/37186
https://github.com/sgl-project/sglang/pull/34528
https://github.com/flashinfer-ai/flashinfer/pull/4393
https://github.com/syv-ai/HyperQwen/issues/254
https://github.com/noonghunna/club-3090/discussions/773
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
https://huggingface.co/docs/hub/xet/using-xet-storage


Models, fine-tunes ও datasets
https://huggingface.co/Qwen/Qwen3.8-27B
https://huggingface.co/Qwen/Qwen3.8-27B-FP8
https://huggingface.co/Qwen/Qwen3.8-Flash-Next
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
https://huggingface.co/vrfai/Qwen3.8-27B-FP8
https://huggingface.co/huginnfork/Qwen3.8-27B-FP8
https://huggingface.co/ukisai/Swift-Qwen3.8-27b
https://huggingface.co/ukisai/Swift-Qwen3.8-27b/blob/main/LICENSE
https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B
https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1


ARC Prize, competitors, agent methodology ও papers
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546
https://arc-explainer-production.up.railway.app/arc-leaderboards
https://arcprize.org/competitions/2026/arc-agi-2
https://arcprize.org/results/alibaba-qwen3-8-27b
https://tufalabs.ai/research/duck-harness/
https://github.com/Tufalabs/duck-harness
https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md
https://github.com/gutfeeling/arc-agi-2-submission
https://github.com/OpenKaggle/arc-agi-2-paper-track-research
https://arxiv.org/abs/2609.27242
https://arxiv.org/abs/2609.39140
https://arxiv.org/abs/2610.11450



14. Final recommendation
GO — Targeted engineering experiments

আমার মতে সবচেয়ে বেশি সম্ভাবনা রয়েছে:

প্রথম: Cache/re-prefill-এর exact token-ID ও checkpoint diagnosis; কারণ এ জায়গায় agent-effective throughput হারাচ্ছে।

দ্বিতীয়: Existing SGLang 0.5.21-এ verified narrow scheduler/CUDA-graph fixes।

তৃতীয়: Built-in MTP বনাম DFlash2, same 4×L4, same 12-stream ARC workload-এ measured comparison।

NOT GO — Without further evidence

Production-এ NCCL P2P force করা, 0.5.22/nightly indiscriminately upgrade করা, unsafe custom all-reduce patch, অথবা top teams-এর unpublished methods নিয়ে অনুমানভিত্তিক redesign।

তোমাদের 34.58 leaderboard থেকে উন্নতির জন্য শুধু 600 tok/s অর্জনই যথেষ্ট নয়। ১২ ঘণ্টায় প্রতিটি task-এ যথেষ্ট useful inference পাওয়া, Python দিয়ে hypothesis দ্রুত যাচাই করা, ভুল train-exact overfit থেকে বের হওয়া এবং two-attempt diversity—সবগুলো দরকার।
