<!-- Source: pasted by the user on 2026-10-23 (second GPT note on the same two suspects: TP4 all-reduce + DFlash2 width; adds 2xTP2 design, staged plan, acceptance criteria).
     Kept VERBATIM (Bengali). Claude's claim-by-claim validation goes into "02 Claude validation of GPT notes 01 and 03.md". -->

E12 Qwen3.8-27B + Duck — GPU Bottleneck Validation
1. প্রথম সিদ্ধান্ত: দুটি সন্দেহই গুরুত্ব পাওয়ার যোগ্য, কিন্তু এখনো প্রমাণিত নয়
তোমাদের TP4 communication এবং DFlash2 over-verification—দুটি hypothesis-ই পরীক্ষার উপযুক্ত। বিশেষ করে একই 4×L4-এ বেশি generated tokens/sec পাওয়া গেলে 240-task coverage সরাসরি উপকৃত হবে।

তবে বর্তমান হিসাব থেকে কয়েকটি দাবি আলাদা করতে হবে।
দাবি	যাচাইয়ের অবস্থান	মন্তব্য
INT4/DFlash2-এ concurrency বাড়লে decode step অনেক ধীর হয়	লগভিত্তিক পর্যবেক্ষণ	একই context, draft acceptance ও kernel path ধরে controlled comparison হয়নি
TP4 inter-GPU communication বড় bottleneck হতে পারে	শক্তিশালী hypothesis	NCCL topology ও ছোট-message latency এখনো মাপা হয়নি
64 MB all-reduce-এ 20.9 ms ⇒ প্রায় 3 GB/s	গাণিতিকভাবে ঠিক	এটি ওই message size-এর effective algorithm bandwidth
একই bandwidth 10 KB messages-এ প্রযোজ্য	যাচাই হয়নি	ছোট collective-এ latency ও protocol বেশি গুরুত্বপূর্ণ
DFlash2-এর 8-wide verification-এ প্রায় 46% positions কাজে লাগে না	অর্থপূর্ণ পর্যবেক্ষণ	Unaccepted positions থাকলেও verified-token work ও output-token accounting এক জিনিস নয়
Width 8→4 করলে 550–650 tok/s নিশ্চিত	Unverified prediction	Draft acceptance এবং target/drafter latency একসঙ্গে মাপতে হবে
2×TP2, TP4-এর চেয়ে ভালো হতে পারে	সম্ভাব্য	প্রতিটি TP2 replica-এর weights, drafter, KV ও Mamba state 24 GB/GPU-তে fit করতে হবে

একটি গুরুত্বপূর্ণ correction: NVIDIA L4-এর PCIe Gen4 x16-তে উদ্ধৃত 64 GB/s দ্বিমুখী aggregate; একমুখী transfer-এর theoretical ceiling প্রায় 32 GB/s। আর 121 TFLOPS হলো non-sparse FP16/BF16 Tensor Core peak—আমাদের INT4 kernel-এর sustained performance-এর সরাসরি benchmark নয়। 

NVIDIA
+1




NCCL-এর official documentation নিশ্চিত করে যে P2P unavailable হলে shared-memory/host-memory transport ব্যবহৃত হতে পারে। কিন্তু শুধু 3 GB/s effective all-reduce দেখে P2P নেই বলা যায় না। 

GitHub
+2



2. সবচেয়ে গুরুত্বপূর্ণ নতুন finding: DFlash2 width 8→4 ঝুঁকিমুক্ত flag change নয়
আমি তোমাদের ব্যবহার করা exact stack শনাক্ত করেছি পুরোনো ZIP-এর server log থেকে:
Component	Actual running configuration
Target	Qwen3.8-27B W4A16 INT4
Draft	Qwen3.8-27B-DFlash2 BF16
SGLang	v0.5.21
Tensor parallel	TP4
Speculative algorithm	DFLASH
Verification width	8
KV cache	FP8 E4M3

DFlash2-এর published checkpoint config-এ block_size=8, conv_kernel_size=2, পাঁচটি draft layers এবং target hidden-state tap indices [5,19,33,47,61] রয়েছে। 

incoai/Qwen3.8-27B-DFlash2 at main




https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json

SGLang 0.5.21-এর exact code-এ checkpoint block size ও requested width না মিললে warning দেয়, তারপর draft model-এর block size বদলায়। Grouped convolution-এর position indexing-ও নতুন width ব্যবহার করে। 

GitHub
+1




https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/dflash_worker_v2.py

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/dflash.py

তাই width 4 চালু করা সম্ভব হলেও acceptance length অপরিবর্তিত থাকবে—এমন নিশ্চয়তা নেই। এই বিশেষ Qwen3.8 checkpoint-এ width 4 ভুল output দেবে বা acceptance ভেঙে পড়বে, সেটিও প্রমাণিত নয়।

আমার সিদ্ধান্ত: 8 এবং 4 তুলনা অবশ্যই করা উচিত, কিন্তু width 6/4/3-এর প্রত্যেকটি experimental setting হিসেবে নেবে। শুধু latency নয়, target-only correctness, acceptance length ও output throughput যাচাই করবে।
550–650 tok/s কতটা বাস্তবসম্মত?
তোমাদের দেওয়া 12-request measurement ধরে বর্তমান theoretical instantaneous throughput:
\[
\frac{12\times4.3}{0.1155}\approx447\text{ tokens/s}
\]

ধরি width 4-এর complete draft+verify step সত্যিই 70 ms হবে। তখন:
Width-4 mean output tokens/request/step	Projected throughput
2.5	429 tok/s
3.0	514 tok/s
3.3	566 tok/s
3.5	600 tok/s
3.8	651 tok/s

এগুলো conditional projections, measurements নয়। 70 ms নিজেও unverified।অতএব 550–650 tok/s পেতে width 4-এ গড়ে প্রায় 3.21–3.79 tokens/request/step রাখতে হবে এবং 70 ms measured step time অর্জন করতে হবে।

এখানে আরেকটি তথ্যের ঘাটতি আছে: width 8-এর average acceptance length 4.3 জানলে width 4-এর acceptance length নির্ধারণ করা যায় না। তার জন্য per-position accepted-prefix distribution প্রয়োজন।

NVIDIA-এর 2026 speculative-decoding analysis-ও শুধু accepted tokens বাড়ানো নয়, verification time + drafting time + acceptance length একসঙ্গে optimize করতে বলেছে। 

NVIDIA Technical Blog




https://developer.nvidia.com/blog/co-designing-ai-models-using-speculative-decoding-for-faster-llm-inference
3. পুরোনো server logs থেকে স্বাধীন পুনর্গণনা
আগের E12_REV6F_IMPLEMENTATION_AUDIT ZIP-এ সংরক্ষিত E11-05 এবং E11-06-এর আসল SGLang server logs পেয়েছি। প্রায় 5,300 raw lines-এর মধ্যে decode records parse করে প্রতি step-এর সময় হিসাব করেছি:
\[
t_{\text{step}}\approx
\frac{B\times\text{accept length}}{\text{reported generation tok/s}}
\]

এটি server-reported quantities থেকে derived approximation, GPU-profiler measurement নয়। Prefill/mixed scheduler intervals ও rolling-throughput effects থাকতে পারে।
Log	Running requests	Records used	Median estimated step	Median accept length
E11-05, DFlash2	11	2,370	124.2 ms	3.82
E11-05, DFlash2	12	114	128.4 ms	3.90
E11-06, DFlash2	11	158	124.4 ms	4.47
E11-06, DFlash2	12	277	124.6 ms	4.34

এই raw-data median-গুলো তোমাদের 115.5 ms মডেলের একটু ওপরে। এর মানে 115.5 ভুল প্রমাণিত নয়—ওটি context-normalized regression estimate হতে পারে, raw median নয়। কিন্তু দুইটিকে এক ধরনের measurement বলা যাবে না।

আরও গুরুত্বপূর্ণ, E11-05-এর 12-running অবস্থায় median total context প্রায় 449k tokens; E11-06-তে প্রায় 357k। তাই model, batch size, context distribution ও throughput একসঙ্গে পরিবর্তিত হচ্ছে।

এই audit-এর একটি সংশোধিত সিদ্ধান্ত: DFlash2 width optimization-এর potential বাস্তব, কিন্তু 115→70 ms-কে এখনই expected measured speedup বলা যাবে না। Baseline-এ actual raw step time, acceptance distribution ও context একসঙ্গে সংরক্ষণ করতে হবে।
Context overhead সম্পর্কে
Qwen3.8-27B-তে 64 layers-এর মধ্যে 16টি full-attention। FP8 KV-এর theoretical total storage প্রায় 32.8 KB/token across TP ranks। তাই 12 requests × 50k context-এ চার GPU-তে ভাগ করলে প্রায় 4.9 GB KV data/GPU হয়।

300 GB/s device-memory bandwidth-এ read-only lower-bound প্রায় 16.4 ms। এটি তোমাদের +18 ms context-cost অনুমানের কাছাকাছি। সুতরাং context explanation quantitatively plausible, যদিও cache hits, actual read patterns ও GDN-state traffic মেপে নিশ্চিত করতে হবে। 

Hugging Face
+2



4. সন্দেহ ১ — TP4/NCCL communication bottleneck
4.1 128 all-reduce হিসাব: যুক্তিসংগত approximation, কিন্তু exact count নয়
Qwen3.8-27B-এর 64 decoder layers আছে। Standard tensor-parallel transformer-এ attention output ও MLP output-এর জন্য সাধারণত দুটি cross-rank reductions প্রয়োজন হয়।

তাই:
\[
64\times2=128
\]

এটি ভালো starting estimate। কিন্তু SGLang-এর Qwen3.5-family implementation LayerCommunicator ও fused/deferred reduction paths ব্যবহার করে। তাই actual CUDA trace ছাড়া প্রতি decode step-এ ঠিক 128টি independent NCCL kernel launch হয়েছে বলা যাবে না। 

GitHub




https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/qwen3_5.py
4.2 তোমাদের communication arithmetic
Hidden size = 5,120। BF16 activation = 2 bytes/element।

একটি verified token-এর full hidden activation:
\[
5120\times2=10{,}240\text{ bytes}
\]

12 requests × 8 verified positions হলে একটি hypothetical collective payload প্রায়:
\[
12\times8\times10{,}240
=983{,}040\text{ bytes}
\]

128টি collective ধরলে মোট nominal payload প্রায় 126 MB।

এটিকে সরলভাবে 3.1 GB/s দিয়ে ভাগ করলে প্রায় 40.6 ms পাওয়া যায়। তোমাদের 40–50 ms estimate-এর গাণিতিক ভিত্তি আছে।

কিন্তু এটি এখনো measured communication time নয়, কারণ:
- Collective traffic-এর bus bandwidth ও application payload bandwidth এক নয়।
- 64 MB message-এর transfer rate 1 MB collective-এ একই হবে না।
- একাধিক collective latency, synchronization ও kernel-launch overhead যোগ হবে।
- কিছু reductions অন্য kernels-এর সঙ্গে overlap বা fuse হতে পারে।
- GPU-তে actual collective payload dtype ও shape বদলাতে পারে।
NCCL documentation অনুযায়ী 4-GPU ring all-reduce-এর bus-bandwidth conversion factor \(2(4-1)/4=1.5\)। তাই algorithm bandwidth আর effective bus bandwidth এক করে তুলনা করা ঠিক হবে না। 

GitHub




https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
4.3 সবচেয়ে দ্রুত confirmation test
প্রথমে কোনো model server চালানোর আগেই:

nvidia-smi topo -m
nvidia-smi topo -p2p p

# If nccl-tests is already installed:
./all_reduce_perf \
  -b 8K -e 1M -f 2 \
  -g 4 -w 20 -n 100 -c 1



এরপর আলাদাভাবে 64 MB point পরীক্ষা করবে, যাতে পুরোনো 20.9 ms result reproduce করা যায়।

SGLang startup-এর সময়:

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM



NCCL log-এ কোন GPU pair P2P transport পেয়েছে, আর কোথায় SHM fallback হয়েছে তা পরীক্ষা করবে। nvidia-smi topo -m শুধু physical topology দেখায়; -p2p p capability এবং NCCL-selected transport মিলিয়ে দেখতে হবে। 

NCCL 2.32.3 documentation




https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html

NCCL-এর transport বদলানোর environmental flags এখনই স্থায়ীভাবে সেট করব না। বিশেষ করে NCCL_P2P_DISABLE=1 বা NCCL_SHM_DISABLE=1 দিয়ে performance optimize করার আগে baseline জানা জরুরি। অকারণে transport disable করলে উল্টো performance কমতে পারে।
4.4 Profiler-এ কোন প্রমাণ চাই?
Trace result	ব্যাখ্যা	পরবর্তী সমাধান
NCCL/all-reduce step-এর বড় অংশ দখল করে	Communication hypothesis শক্তিশালী	TP2×2 experiment, compatible all-reduce optimization
GEMM/INT4 dequant kernels সবচেয়ে বেশি সময় নেয়	Compute/kernel-limited	Quantized kernel optimization; TP split দিয়ে নিশ্চিত লাভ নয়
GDN kernels অনেক ছোট launches বা serial gaps তৈরি করে	Kernel-launch / recurrent-state bottleneck	Existing backend configuration পরীক্ষা
Drafter/selector বেশ সময় নেয়	Speculative overhead বেশি	Width/drafter comparison
Memcpy H2D/D2H ও synchronization অনেক	Host-staging বা CPU scheduling cost	NCCL topology, buffer/graph path পরীক্ষা
Long context-এ attention/KV read dominate করে	Memory bandwidth bottleneck	KV/cache strategy, context-aware scheduling

SGLang profiler-এর /start_profile support রয়েছে, তবে pinned v0.5.21 build-এ exact API/parameters আগে যাচাই করতে হবে। নতুন main-branch profiler code সরাসরি চালানোর অনুমান করব না। 

GitHub




https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx

এই profiling ছাড়া “আমাদের প্রায় অর্ধেক সময় NCCL-এ যাচ্ছে” বলা premature।

5. সন্দেহ ২ — DFlash2 verification width
এই hypothesis-এ আমি আরও একটি উপকারী research result পেয়েছি।

DFlash2-এর authors-এর August 18, 2026 প্রকাশিত article-এ Qwen3.8-27B-এর জন্য mean acceptance length প্রায় 4.80, বিভিন্ন benchmark মিলিয়ে। তোমাদের ARC/Duck traces-এ ~4.3 পাওয়া অস্বাভাবিক নয়; workload আলাদা। 

Inco AI




https://inco.ai/blog/dflash2/

কিন্তু article-এর per-position Recall@1 এবং তোমাদের আলোচনার late-position acceptance probability একই metric নয়।

Recall@1: নির্দিষ্ট draft position-এ proposed token সঠিক হওয়ার হার।

Survival probability: আগের সব positions-ও সঠিক ছিল এবং verifier ওই position পর্যন্ত পৌঁছেছে—এমন probability।

Late positions-এ cumulative survival খুব কম হলেও conditional token accuracy তত কম নাও হতে পারে। তাই “position 4–7-এ মাত্র 10% acceptance” দাবিতে ঠিক কোন denominator ব্যবহৃত হয়েছে জানা জরুরি।
Recommended width experiment
Width	Priority	কী দেখব
8	Baseline	Trained block-size path; canonical acceptance ও throughput
4	High	Width ছোট করলে latency সাশ্রয় acceptance loss-এর চেয়ে বেশি কি না
6	Medium	4 এবং 8-এর মধ্যে compromise আছে কি না
3	Low	Small-width diagnostic; সময় থাকলে

একই 12-request workload, same input lengths, same temperature/top-p/top-k এবং same concurrent output demand রাখবে।

Pass condition: শুধু faster step নয়; বেশি committed output tokens per second এবং unchanged deterministic target behaviour।

প্রয়োজনে width 8-ই জয়ী হতে পারে। Width কমানোর সঙ্গে draft convolution-এর block handling বদলানোয় এটিকে free optimization বলা যাচ্ছে না।

6. আরও বড় সম্ভাবনা — 2×TP2 বনাম TP4
এটি আমি width-optimization-এর পরের সবচেয়ে গুরুত্বপূর্ণ experiment মনে করি।

পুরোনো logs অনুযায়ী TP4-এ প্রতি GPU:
Component	Measured weight memory
INT4 target	4.62 GB
BF16 DFlash2 draft	1.49 GB
Total model weights	প্রায় 6.11 GB

TP2 করলে straightforward sharding approximation-এ model weights প্রতি GPU প্রায় 12.2 GB হতে পারে। L4-এ 24 GB VRAM থাকার কারণে প্রাথমিকভাবে feasible মনে হয়, তবে usable KV capacity, Mamba states, CUDA graphs, workspace ও prefill buffers যোগ করে startup test ছাড়া fit নিশ্চিত নয়।

একটি useful community comparison-এ 4×3090 TP4 এবং 2×3090 TP2-এর ভিন্ন strengths পাওয়া যায়: TP2 shorter contexts ও single-stream latency-তে সুবিধা দিয়েছে, TP4 long-context capacity-তে। কিন্তু ঐ hardware, engine ও topology Kaggle L4-এর মতো নয়। 

GitHub




https://github.com/tonyd2wild/Qwen3.8-27B-DFlash2-4x3090-TP4
কীভাবে fair comparison হবে?
Existing TP4
GPU 0
GPU 1
GPU 2
GPU 3

1 inference server

12 requests total


Proposed 2×TP2
GPU 0
GPU 1

Server A · 6 requestsGPU 2
GPU 3

Server B · 6 requests



দুই configuration-ই চারটি GPU ব্যবহার করবে এবং একই মোট 12 simultaneous requests serve করবে।

Decision metric: দুই TP2 server-এর combined committed output tokens/sec, যেখানে দুটির prefill, cache miss, queue delay, failed requests ও wall time অন্তর্ভুক্ত।

TP2 দ্রুত হলেও যদি usable context limit কমে যায় বা KV/Mamba state exhaustion হয়, সেটি আমাদের 128k+ context solver-এর জন্য উপযুক্ত নাও হতে পারে।
অতিরিক্ত implementation risk
দুটি replica ব্যবহার করলে একই task-এর পরবর্তী continuation ideally একই replica-তে যাবে, যাতে prefix cache reuse হয়। অন্য replica-তে গেলে server-side KV prefix পুনরায় build করতে হতে পারে।

অতএব 2×TP2-এর জন্য task-sticky routing দরকার; পুরো Duck reasoning contexts merge করতে হবে না। Existing candidate archive ও pass@2 logic একই থাকবে।

7. Minimum-GPU-cost experiment: আমার নির্বাচিত design
একটি 30–40-minute budget মাথায় রেখে experiment-কে staged করা উচিত। সব arms সমান গুরুত্ব দিয়ে একসঙ্গে চালাব না।
Stage	কাজ	Planned cap	Go/stop condition
A — topology	NCCL P2P/SHM, 8KB–1MB all-reduce sweep	3 min	Communication status জানা
B — TP4 baseline	DFlash2 width8, 6/12 requests, short/long contexts	7 min	Stable baseline TPS + accept + context
C — width4	একই controlled load; acceptance + output validation	7 min	Baseline-এর বিরুদ্ধে net gain
D — profiler	10–20 steps target/draft/NCCL timeline	5 min	প্রধান bottleneck শ্রেণিবদ্ধ
E — TP2 feasibility	দুটি TP2 replica boot, VRAM/KV/throughput	Remaining time	Boot ও context fit হলে compare

এই caps-এর মোট প্রায় 22 মিনিট plus TP2-এর জন্য অবশিষ্ট সময়। বাস্তবে model loading, CUDA graph capture ও checkpoint storage ধীর হলে 30–40 মিনিটে সব শেষ নাও হতে পারে। সে ক্ষেত্রে Stage E আলাদা postponed diagnostic, একই experiment-এ অসীম retry নয়।

Absolute stop rule: আগে থেকেই 40-minute wall deadline এবং shutdown/save margin রাখবে। কোনো benchmark চলমান থাকলেও deadline অতিক্রম করবে না। GPU quota ছোট হলে A–C-ই প্রথম run-এর minimum useful result।

Quota সম্পর্কে একটি সংশোধন: 4 GPU × 30–40 min = 2.0–2.67 physical GPU-hours। কিন্তু Kaggle account quota সরাসরি একই formula-তে charge করে কি না নির্দিষ্ট accelerator/account policy ছাড়া নিশ্চিত নয়। তাই 1–1.5 account quota-hours খরচ হবে—এখন বলা ঠিক নয়। Test-এর আগে ও পরে actual quota counter record করতে হবে।
Workload design — সবচেয়ে গুরুত্বপূর্ণ নিয়ম
শুধু random prose benchmark দিয়ে decision নেব না।

আমাদের Duck traces-এ দীর্ঘ reasoning, Python tool-call JSON এবং repeated prefix আছে। তাই 2 ধরনের workload ব্যবহার করব:
Workload	Purpose
Fixed-length synthetic	Engine capability isolation: prompt 2k/30k বা 50k; completion 512–1024; concurrency 6/12
Recorded Duck request replay	Actual workload performance: same system/tools/template, ARC-style reasoning, code generation ও prefix reuse

GPU না খরচ করে আগেই recorded request payloads এবং benchmark parser তৈরি করা সম্ভব। Random generated IDs-এর entropy বাস্তব code-generation acceptance-এর representative নাও হতে পারে।
প্রতিটি arm-এ mandatory metrics
Metric	কেন প্রয়োজন
Committed output tok/s, wall end-to-end	প্রধান optimization metric
Accepted tokens per verification step	Draft quality
Step-time median, p90	Kernel/communication cost
Target verification + draft timing	Width change আসলে কোথায় লাভ দিচ্ছে
GPU VRAM, KV/Mamba usage	2×TP2 feasibility
Cache hit, uncached prefill	Context/replay performance
NCCL collective time	Communication hypothesis
Output parity at temperature 0	Serving correctness
Timeout/OOM/retractions	Stability

Temperature 1.0-এ দুই run-এর exact generated tokens match করতেই হবে—এমন নয়, কারণ sampling stochastic। Deterministic comparison-এর জন্য temperature 0 এবং same input ব্যবহার করবে; production-style throughput-এর জন্য existing temperature 1.0 বজায় থাকবে।

8. গবেষণাপত্র থেকে আরও দুটি গুরুত্বপূর্ণ শিক্ষা
Learning to Draft (March 2026) দেখিয়েছে adaptive draft scheduling-এ শুধু acceptance length নয়, পুরো draft/verify cycle-এর time cost optimize করতে হয়। এটি আমাদের measurement objective সমর্থন করে, কিন্তু RL controller এখন port করতে বলছি না। 

arxiv.org




https://arxiv.org/abs/2603.01639

SpecKV (May 2026)-এর গবেষণায় optimal speculation length compression regime ও draft confidence-এর সঙ্গে বদলাতে পারে। অর্থাৎ width 8 সর্বত্র optimal নাও হতে পারে। কিন্তু তাদের trained adaptive controller আমাদের exact DFlash2 implementation-এ drop-in solution নয়। 

arxiv.org




https://arxiv.org/abs/2605.02888

বর্তমান project-এ প্রথমে static width 8/4 এবং TP topology মাপাই বেশি মূল্যবান। Adaptive width অথবা নতুন speculation algorithm-এর architecture changes এখন unnecessary।

9. Claude-এর জন্য implementation acceptance criteria
Finding / risk	Required fix or action	CPU/GPU validation
64MB NCCL result ছোট payload-এ extrapolate	8KB–1MB collective sweep	Per-size latency curve; P2P/SHM transport confirmed
128 all-reduces assumed	Profiler থেকে actual collective count	TP ranks-এর timelines match
Width4 drafter trained at width8	Checkpoint/config preflight + accepted-prefix statistics	No silent mismatch; target parity and output TPS
550–650 tok/s prediction	Actual end-to-end output TPS measure	Confidence interval/repeated runs, no model-based claim
2×TP2 may OOM	Both replicas boot + peak memory and 50k-context test	No OOM, enough KV/Mamba headroom
Short synthetic benchmark bias	Recorded Duck request replay	Code/tool workload included
Profiling overhead	Profiled vs nonprofiled throughput separately	Published TPS only from nonprofiled runs
Quota risk	Single absolute deadline, stage skips and output persistence	Intentional slow-start/timeout simulation

No solver modification: এই profiling experiment-এ Rev6f-এর Duck prompts, candidate selection, first-exposure scheduler, reasoning policy বা host verification বদলানো হবে না। Engine-only comparison হিসেবে রাখতে হবে।
10. Final recommendation
GO — ছোট, bounded engine benchmark তৈরি

GO: পুরোনো লগের দুটি hypothesis যথেষ্ট শক্তিশালী যে আলাদা benchmark করার মূল্য আছে।

NOT GO: width 8→4-কে এখনই production Rev6f solver-এ লাগানো, অথবা TP4 বদলে 2×TP2 করা—GPU evidence ছাড়া নয়।

আমার priority:

প্রথমে NCCL topology ও small collective latency। তারপর একই TP4 setup-এ width8 বনাম width4। Profiler-এ communication dominant হলে 2×TP2 সবচেয়ে গুরুত্বপূর্ণ experiment হবে।

Success = বেশি sustained committed tokens/sec, acceptable context capacity, correct serving behaviour এবং একই 4×L4 wall-hour-এ আরও বেশি useful Duck inference।

Research sources — সরাসরি URLs
বিষয়	Source
SGLang v0.5.21 DFlash worker	https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/dflash_worker_v2.py
SGLang DFlash grouped convolution	https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/dflash.py
DFlash2 official article	https://inco.ai/blog/dflash2/
DFlash2 checkpoint configuration	https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json
SGLang Qwen3.8 serving guide	https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
NCCL P2P troubleshooting	https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
NCCL benchmark methodology	https://github.com/NVIDIA/nccl-tests
Speculative throughput optimization	https://developer.nvidia.com/blog/co-designing-ai-models-using-speculative-decoding-for-faster-llm-inference
Adaptive draft scheduling	https://arxiv.org/abs/2603.01639
Compression-aware draft length	https://arxiv.org/abs/2605.02888
SGLang profiling	https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx
Community TP2/TP4 observations	https://github.com/tonyd2wild/Qwen3.8-27B-3090-Cookbook

সবচেয়ে গুরুত্বপূর্ণ সংশোধিত বক্তব্য: আমাদের bottleneck শুধু “GPU compute কম ব্যবহার হচ্ছে” নয়; INT4 target verification, TP4 reductions, DFlash2 drafting, GDN kernels এবং long-context KV traffic—এইগুলোর প্রকৃত wall-time ভাগ এখনো অজানা। এই ছোট benchmark-এর কাজ হবে ঠিক সেই ভাগ বের করা, নতুন solver architecture বানানো নয়।
