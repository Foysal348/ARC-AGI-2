<!-- Pasted by the user on 2026-10-24 (GPT deep web research). Kept VERBATIM. Claude's validation: Notes/29. -->

ARC Prize 2026 — Qwen3.8-27B Serving ও Competitor Deep Research
Research cutoff: 24 October 2026 · Public sources only · GPU benchmark নিজে চালানো হয়নি
Executive findings
ওয়েব অনুসন্ধানে সবচেয়ে গুরুত্বপূর্ণ বিষয় হলো, আপনার 4×L4-এর 355–460 generated tok/s-ই PCIe GPU-তে Qwen3.8-27B-এর সর্বোচ্চ গতি—এমন কোনো প্রমাণ নেই। তবে আপনার exact hardware, 12 concurrent agents, 30k–100k context, INT4+DFlash2 configuration-এ এর চেয়ে দ্রুত reproduced public benchmark এখনও পাওয়া যায়নি।

তিনটি বিশেষ গুরুত্বপূর্ণ public finding পেয়েছি:

1. 4×RTX 3090, no NVLink-এ 752 tok/s aggregate-এর প্রকাশিত benchmark আছে। কিন্তু সেটিতে P2P enabled, Merlin/vLLM-based engine, AWQ INT4 এবং 8 concurrent agents-এর ছোট 16k conversations ব্যবহার করা হয়েছে। আপনার PHB/P2P-unfriendly L4 setup-এর সঙ্গে সরাসরি তুলনা করা যায় না। 

Hugging Face




2. SGLang-এর Qwen3.8 cookbook-এ Mamba memory sizing-এর নির্দিষ্ট সূত্র এবং operational guidance আছে। Default --mamba-full-memory-ratio 0.9 concurrency কমাতে পারে। তবে published throughput measurements মূলত single-GPU newer hardware ও ছোট workload-এর। 

GitHub




3. SGLang-এর অক্টোবরের pipeline-parallel roadmap স্পষ্ট বলছে DFlash + PP decode এখনও unsupported। ফলে TP4→PP4 বা PP2×2 switch আমাদের বিদ্যমান DFlash2 solver-এর drop-in optimization নয়। 

GitHub



Research evidence কীভাবে পড়বেন
Evidence class	অর্থ
Measured — public	লেখক hardware, configuration ও numerical results প্রকাশ করেছেন। Independent reproduction না হলে সেটি author-reported measurement।
Verified — source	Official source, code, issue বা configuration থেকে behavior যাচাইযোগ্য।
Claim	Public description আছে, কিন্তু matching benchmark বা raw reproduction নেই।
Inference	আপনার configuration ও published evidence থেকে করা বিশ্লেষণ; পরিমাপ নয়।
পাওয়া যায়নি	অনুসন্ধানে নির্দিষ্ট দাবির সমর্থনে ব্যবহারযোগ্য public evidence মেলেনি।


১. সরাসরি ব্যবহারযোগ্য public serving benchmarks
1.1 Merlin: 4×RTX 3090 without NVLink
Source: halt95, model card ও linked bench card; benchmark date 20 October 2026।

https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin
4×3090, 8 agents
752
Aggregate tok/s

2×3090, 8 agents
517
Aggregate tok/s

4×3090, 131k context
235
Single-stream tok/s



Engine Merlin 3.0.0 (vLLM 0.32.0 fork), P2P enabled, no NVLink, 4-bit AWQ ~16 GiB checkpoint, FP8 KV, benchmarked 8 agents with 16k histories and 512-token follow-ups.
Merlin reported aggregate throughput by concurrency
4×3090 versus 2×3090; author-reported throughput, short-context workload, not an L4 benchmark.



4×3090

2×3090


0 tok/s200 tok/s400 tok/s600 tok/s800 tok/s2 agents4 agents8 agents














আমাদের setup-এর সঙ্গে সম্পর্ক: এটি দেখায় PCIe-only multi-GPU system-এ আরও throughput সম্ভব। কিন্তু এই benchmark-এ P2P চালু; আপনার GPU pairs PHB, যেখানে P2P/IPC NCCL SHM-এর চেয়ে ধীর। তাই Merlin-এর 752 tok/s দিয়ে আমাদের 4×L4-এ 752-এর prediction করা যাবে না।

আর model weights, quantization, runtime এবং context distribution বদলেছে। একই engine-এ 4×3090 বনাম 2×3090 comparison ছাড়া TP2×2-এর advantage প্রতিষ্ঠিত হয় না।
1.2 একই Merlin engine-এ ThinkingCap variant
Source: halt95/BottleCap-derived checkpoint, October 2026।

https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin

Reported aggregate:
Hardware	2 agents	4 agents	8 agents
2×3090	279	362	502
4×3090	365	504	733

Unit: output tok/s; author-reported measurements। 

Hugging Face




Important: এই fine-tune-এর polyform-small-business-1.0.0 license রয়েছে; মূল Merlin checkpoint-এর Apache-2.0 license-এর সঙ্গে এক নয়। Kaggle-এ ব্যবহার বা derivative distribution-এর আগে license eligibility আলাদাভাবে যাচাই করা প্রয়োজন। Faster reasoning-এর দাবি আর ARC-AGI-2 accuracy আলাদা বিষয়।
1.3 Single-GPU SGLang alternative: Escha-W2
Source: EschaLabs, আগস্ট ২০২৬ benchmarking ও subsequent user results।

https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2

Reported peak aggregate:
GPU	Max concurrent	Aggregate output
RTX 3090 24GB	16	383 tok/s
RTX 4090 24GB	16	649 tok/s
RTX 5090 32GB	16	955 tok/s

এটি aggressive 2-bit quantization ও আলাদা serving route ব্যবহার করে। 

EschaLabs/Qwen3.8-27B-Escha-W2 at main




Relevance: sm_89 Ada-এর একটি single-card alternative-এর throughput interesting, কিন্তু model quality, long-context capacity এবং ARC tasks-এ regression এখনও অজানা। এটিকে আপনার RedHat INT4-এর low-risk replacement বলব না।
1.4 AutoRound INT4 + built-in MTP
Source: Avuja, Hugging Face model card, সেপ্টেম্বর ২০২৬।

https://huggingface.co/Avuja/Qwen3.8-27B-int4-AutoRound

RTX 5090, vLLM 0.27.1, single stream-এ MTP দিয়ে 82 → 127 tok/s (1.55×) এবং reported 0.69 greedy draft acceptance। Checkpoint-এ MTP head BF16 রাখা হয়েছে। 

Hugging Face




এটি public proof যে built-in MTP কার্যকর হতে পারে। কিন্তু আমাদের high-batch DFlash2-এর চেয়ে দ্রুত—এমন প্রমাণ নয়।
২. TP4 বনাম TP2×2 বনাম pipeline parallel
এই অংশে একটি সত্যিকারের comparative benchmark পাওয়া গেছে, তবে hardware limitation গুরুত্বপূর্ণ।

Author: u/Mr_Moonsilver, Reddit r/LocalLLaMA, 17 August 2026।

https://www.reddit.com/r/LocalLLaMA/comments/1vr56r1/benchmarked_qwen3827b_on_4x_rtx_3090/

Configuration: 4×RTX3090, দুই NVLink pair, pair-দুটির মধ্যে PCIe; vLLM 0.20.1, AWQ INT4, FP8 KV, random prompts 1k input/512 output।
Concurrency	TP4, চার GPU	TP2, একটি NVLink pair
4	208.8	201.6
8	266.1	252.9
12	319.9	428.0
16	371.9	495.1
24	399.4	532.0
32	417.9	587.4

Aggregate generated tok/s; author-reported controlled benchmark। 

reddit.com


এখানে concurrency 12-তে TP2 একটি pair প্রায় 34% দ্রুত, যদিও অর্ধেক GPU ব্যবহার করছে। লেখক দুটি NVLink pair-এ দুইটি TP2 replica চালিয়ে আরও throughput পাওয়ার সম্ভাবনার কথাও বলেছেন; সেটি একই measured table-এর সরাসরি দুই-replica result নয়।

তবে এই ফল আপনার setup-এ সরাসরি প্রযোজ্য নয়: আপনার L4-গুলোর কোনো NVLink pair নেই, সব pair PHB। অর্থাৎ TP2-তেও communication-এর বড় penalty থাকতে পারে।

আর TP2-তে প্রতি GPU weights প্রায় দ্বিগুণ, usable KV pool ছোট, DFlash2 drafter-সহ 12 streams ও 30k–100k context fit করা কঠিন হতে পারে।
2.1 Parallelism options-এর বর্তমান অবস্থা
Architecture	Technical feasibility	Evidence / সিদ্ধান্ত
TP4 + DFlash2	Already works	আপনার baseline 355–460 tok/s; measured
2×TP2 + DFlash2	Unverified memory fit	Two replicas potentially reduce TP communication scope, কিন্তু L4 PHB-তে measured comparative result পাওয়া যায়নি
PP2 + TP2	DFlash2 currently unsupported in normal PP decode	SGLang October roadmap; new source/validation লাগবে
PP4 + MTP	Possible on supported newer paths	PP+EAGLE/MTP support এসেছে, কিন্তু overlap limitation ও hybrid-state correctness যাচাই দরকার
DP4 / TP1 replicas	Current RedHat INT4 + BF16 drafter unlikely to fit per 24GB GPU	Model/drafter footprint ও KV requirements বিবেচনায়; exact allocation preflight ছাড়া সিদ্ধান্ত নয়

2.2 কেন pipeline parallel-কে প্রথমে সুপারিশ করছি না
SGLang-এর 12 October 2026 roadmap:

https://github.com/sgl-project/sglang/issues/43885

এতে বলা হয়েছে EAGLE/MTP-র কিছু PP combination চালু থাকলেও DFlash, EAGLE3 under PP এখনও rejected, এবং overlap/auxiliary-state handling সম্পূর্ণ হয়নি। 

GitHub




প্রাসঙ্গিক implementation tracking:

https://github.com/sgl-project/sglang/issues/39634

Inference: TP4-এর all-reduce cost 56ms হলে PP আকর্ষণীয় হতে পারে। কিন্তু PP-এর pipeline bubbles, stage activation transfers, draft placement এবং short/nonuniform ARC requests-এর কারণে speedup নির্দিষ্টভাবে বলা যায় না।

প্রস্তাব: TP2 memory preflight এবং NCCL tuning আগে; PP rewrite deadline-এর আগে নয়।

৩. Speculative decoding: DFlash2, MTP, DSpark
3.1 DFlash2 paper/engineering team-এর actual acceptance data
Source: Inco AI, 18 August 2026।

https://inco.ai/blog/dflash2/

একই Qwen3.8-27B target-এ প্রকাশিত per-request mean acceptance length:
Dataset	Built-in MTP	DSpark	DFlash2
GSM8K	5.02	4.36	5.46
MATH-500	4.72	3.92	5.28
HumanEval	3.91	3.30	4.39
MBPP	3.99	3.51	4.79
MT-Bench	3.74	3.01	4.10
Mean	4.28	3.62	4.80

Author benchmark, default sampling, block size 8; concurrency-12/L4 throughput নয়। 

Inco AI


DFlash2-এর 2.7–3.4× autoregressive throughput claim batch size 1-এর জন্য। আপনার 12-stream performance-এ সেটি extrapolate করা যায় না।

আপনার measured accept length 3.9–5.1 এই public table-এর পরিসরের মধ্যে পড়ে। তাই acceptance quality-তে বড় অস্বাভাবিকতা প্রমাণিত হয়নি; বরং 56ms communication-এর কারণে verification expensive হওয়া বেশি গুরুত্বপূর্ণ।
3.2 PCIe consumer benchmark: 2×RTX3090 + DFlash2
Author: oceanplexian, 19 August 2026, Reddit।

https://www.reddit.com/r/LocalLLaMA/comments/1vsccit/qwen3827b_on_2x_3090_vllm_dflash2_218_toks_single/

Reported:
- vLLM 0.26.1rc1, custom patches
- 2×3090, no NVLink, P2P patched
- AutoRound INT4
- 7 draft tokens
- Acceptance length 3.35, draft acceptance 47.8%
- Peak memory 22.3GB/GPU
- Narrative 120.1 decode tok/s, code 218.3 decode tok/s
- Maximum context approximately 131k
এই post-এর সংখ্যায় workload sensitivity খুব বড়—coding ও narrative-তে output প্রায় 1.8× আলাদা। 

reddit.com




আমাদের জন্য শিক্ষা: ARC-agent-generated Python ও narrative reasoning-এ acceptance/throughput আলাদাভাবে মাপুন। শুধু GSM8K বা random text দিয়ে width winner নির্বাচন করবেন না।
3.3 High-batch speculative decoding সবসময় লাভজনক নয়
Mr_Moonsilver-এর একই 4×3090 test-এ built-in MTP (3 speculative tokens) উচ্চ concurrency-তে বেশিরভাগ ক্ষেত্রে baseline থেকে ধীর হয়েছিল:
Concurrency	No spec	MTP	Difference
8	266.1	285.9	+7%
12	319.9	288.8	−10%
16	371.9	308.3	−17%
32	417.9	348.3	−17%

Aggregate tok/s, vLLM 0.20.1, AWQ, mixed NVLink/PCIe. 

reddit.com


এটি দেখায় যে acceptance length ভালো হলেই speculative decoding throughput বাড়াবে—এমন নয়।
3.4 Draft width 8 / 6 / 4 / 3
অনুসন্ধানে Qwen3.8-27B-এর জন্য SGLang 0.5.21, 4×L4, concurrency 12, matched 30k–100k contexts-এ এই চার width-এর measured sweep পাওয়া যায়নি।

অন্য implementation-এ width4 কার্যকর হওয়ার evidence আছে। উদাহরণ: ravingamm, 25 August 2026, RTX3090 + llama.cpp-তে DFlash2 n_max=4, 160k context, reported 59.88 tok/s; no-spec baseline 33.16 tok/s। কিন্তু এটি TP4 SGLang নয়। 

huggingface.co




আমার সুপারিশ: width8/6/4 পরীক্ষার পরিকল্পনা বজায় রাখুন। Width3 কেবল দ্রুত startup সম্ভব হলে। Acceptance length-এর পাশাপাশি actual aggregate tok/s, target verification ms, draft ms ও correctness মাপুন।
3.5 একটি নতুন correctness warning: vLLM DFlash + prefix cache
Source: vLLM issue #60174, sinmkd এবং অন্যান্য contributors, October 6–22, 2026।

https://github.com/vllm-project/vllm/issues/60174

vLLM 0.30/0.31-এ Qwen3.8+DFlash2/DSpark-এর prefix-cache hit-এ repeated-token corruption পাওয়া যায়। LongBench NarrativeQA-এর একটি reproduced run-এ ২০০টির মধ্যে ৭৮টি garbled answers, আর drafter attention TRITON_ATTN করলে ০/২০০। Fix #55601 vLLM 0.32.0-এ এসেছে বলে October 22-এ reporter confirmed। 

GitHub




এটি আপনার SGLang 0.5.21-এ bug আছে—এমন প্রমাণ নয়। কিন্তু engine upgrade বা vLLM switch করলে long multi-turn agent replay অপরিহার্য।
৪. Memory, quantization ও hybrid prefix cache
4.1 SGLang-এর অক্টোবরের cookbook-এ সবচেয়ে ব্যবহারযোগ্য memory formula
Source: SGLang official Qwen3.8-27B cookbook; relevant benchmarks v0.5.19, source updated in October 2026।

https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx

Hybrid GDN-এর জন্য documented formula:
\[
R=\frac{(S+D)\times M_{\mathrm{state}}}{L\times M_{\mathrm{KV/token}}}
\]

এখানে:
Parameter	Documented value
S, extra_buffer	5 state slots/request
S, extra_buffer_lazy	4 state slots/request
S, no_buffer	3 state slots/request
S, radix cache disabled	1 slot/request
D, DFlash2	8 speculative intermediate states
FP32 GDN state	153.9 MB/slot
BF16 GDN state	78.4 MB/slot
FP8 attention KV	32.8 KB/token

বিশেষ সতর্কতা: explicit --max-mamba-cache-size সেট করলে D-কে slot-count-এর সঙ্গে আবার যোগ করা যাবে না। Verify buffer আলাদাভাবে allocated। 

GitHub




আপনার 12 concurrent agents-এর জন্য, extra_buffer_lazy ধরে প্রায় 48 state slots প্রয়োজন। extra_buffer হলে প্রায় 60। কিন্তু server already যদি 12 requests admit করতে পারে এবং KV/cache capacity যথেষ্ট থাকে, শুধু ratio বদলিয়ে কোনো throughput improvement নাও হতে পারে।
4.2 extra_buffer_lazy এবং SSM dtype
SGLang-এর cookbook-এ extra_buffer_lazy per-request slot cost 5→4 করে। BF16 state FP32-এর তুলনায় প্রায় অর্ধেক memory ব্যবহার করে।

তবে checkpoint-এর own config-এ SSM precision float32; BF16 বা FP16 ব্যবহার করলে numerical parity/accuracy validation প্রয়োজন। 

GitHub


 

Qwen/Qwen3.8-27B at main




আমাদের setup: ইতিমধ্যে FP16 recurrent state ব্যবহার হচ্ছে। তাই FP32→FP16 memory saving নতুন সম্ভাবনা নয়। Cache strategy বদলালে যতটা memory/concurrency headroom পাওয়া যায় সেটাই মাপার বিষয়।

নতুন প্রস্তাব: effective max_running_requests, state slots per request, GDN pool allocated, KV pool capacity, cache-hit tokens এবং context-trim frequency একসঙ্গে report করুন। শুধু memory fraction দেখে সিদ্ধান্ত নয়।
4.3 SGLang issue: concurrency কমে গেলে বড় throughput loss
Issue #36889, beastllama, August 28, 2026:

https://github.com/sgl-project/sglang/issues/36889

এখানে hybrid KDA model-এ Mamba-state allocation-এর কারণে advertised concurrency-এর নিচে scheduler cap হওয়ার সমস্যা এবং একটি configuration change-এ reported +75% throughput improvement উল্লেখ আছে। 

GitHub




কিন্তু এটি GLM-5.3-Flash-এর case; আপনার Qwen3.8-27B-তে +75% প্রত্যাশা করা যাবে না।
4.4 FP8 W8A8, INT4 Marlin/Machete এবং AWQ
Quantization	Public evidence	আমাদের জন্য
RedHat INT4 W4A16	আমাদের working baseline	Keep initially
AutoRound W4A16	Avuja: 19.6GB checkpoint, RTX5090 MTP 82→127 tok/s	Potential alternate, not a verified L4 speed gain
AWQ W4A16	Merlin: ~16GiB and 4×3090 measured aggregate 752	Engine-specific, quality validation required
FP8 W8A8	Huginnfork checkpoint ~35.8GiB; MLP-only dynamic FP8	Larger weight footprint; not automatically faster on L4
W2	Escha: high single-card aggregate results	Greater numerical/accuracy risk

Sources: 

Hugging Face
+3




Official model architecture has 64 layers and only 16 full-attention layers। GDN projection quantization, lm_head quantization ও verification logit accuracy—এই তিনটি ARC long reasoning-এ গুরুত্ব পায়। 

Hugging Face




Finding: INT4 → FP8 W8A8 করলে L4-এর theoretical FP8 compute সুবিধা থাকলেও larger weight footprint, GEMM shapes, communication ও memory allocation-এর কারণে বাস্তব throughput winner অনিশ্চিত।

--enable-quant-communications-কে L4 CUDA solution হিসেবে সুপারিশ করছি না: SGLang-এর documented INT8 TP communication support বর্তমানে Qwen3-এর জন্য NPU-specific। 

GitHub




৫. PHB/PCIe communication: সবচেয়ে সম্ভাবনাময় engine-level optimization
আপনার GPU profiling অনুযায়ী:
All-reduce per decode step
~56 ms
আপনার GPU measurement

Total decode step
115–133 ms
12 running requests



এখানে communication সত্যিই অত্যন্ত গুরুত্বপূর্ণ। তবে 140 × 345µs ≈ 48.3ms, 56ms নয়; profiler-এ kernel durations-এর পাশাপাশি synchronization, gaps বা অতিরিক্ত collective থাকলে পার্থক্য ব্যাখ্যা হতে পারে। তাই 56ms-কে total communication-path measurement এবং 48ms-কে 140 measured calls-এর আনুমানিক sum হিসেবে আলাদা report করা উচিত।
5.1 NCCL-এর official documentation-এর গুরুত্বপূর্ণ তথ্য
Source: NVIDIA NCCL user guide, updated 2026।

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NVIDIA বলছে NCCL_P2P_DISABLE=1 GPU direct P2P বন্ধ করে এবং NCCL_SHM_DISABLE=1 shared-memory transport বন্ধ করে। P2P না হলে SHM host RAM ব্যবহার করতে পারে। আর debugging/tuning variables production-এ স্থায়ীভাবে hard-code না করার সতর্কতাও আছে। 

NCCL 2.32.3 documentation




আপনার measurement-এ P2P/IPC SHM-এর চেয়ে 5.5× ধীর। সেক্ষেত্রে NCCL P2P off + SHM on একটি যৌক্তিক benchmark candidate।

কিন্তু NCCL ইতিমধ্যে auto-select করে SHM ব্যবহার করলে explicit NCCL_P2P_DISABLE=1-এ speedup নাও হবে।
5.2 Exact NCCL A/B candidates
# Baseline: allow NCCL to auto-select
NCCL_DEBUG=INFO

# Candidate A: prohibit P2P, retain SHM
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0

# Candidate B: use the same SHM path, test protocol separately
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
NCCL_PROTO=Simple


NCCL_PROTO=LL বা LL128, NCCL_ALGO=Ring/Tree, এবং channel counts-ও bounded experiment হিসেবে যুক্ত করা যায়—তবে available NCCL build-এর support ও logging আগে যাচাই করতে হবে।

এগুলো test candidates, recommended production flags নয়। NCCL নির্দিষ্ট protocol ও algorithm নিজেই tune করে; জোর করে বদলালে slower বা incorrect result হতে পারে।
5.3 Gain-এর mathematically defensible bound
ধরি মোট step 115ms এবং communication 56ms। একই workload-এ শুধুমাত্র communication time কমে:
Communication improvement	New estimated step	Predicted throughput multiplier
0%	115 ms	1.00×
25% কম	101 ms	1.14×
50% কম	87 ms	1.32×
75% কম	73 ms	1.58×
সম্পূর্ণ শূন্য	59 ms	1.95×

এটি Amdahl-style sensitivity calculation, নতুন benchmark নয়। বাস্তবে overlap, graph scheduling, re-batching ও kernel changes থাকলে সংখ্যাগুলো বদলাবে।

এই কারণে আপনার ক্ষেত্রে communication path optimize করা প্রথম সারির engineering priority।
5.4 CustomAllReduce-এর সীমা
SGLang এবং vLLM উভয় project-এ more-than-two PCIe-only GPUs-এর জন্য certain custom all-reduce paths disabled থাকার code evidence আছে। 

GitHub
+1




যা পাওয়া যায়নি: SGLang 0.5.21 + 4×L4 PHB-তে 140 sub-1MB all-reduce-এর জন্য production-tested, guaranteed faster quantized/FlashInfer PCIe-IPC replacement।

আপনার PCIe-IPC benchmark SHM-এর চেয়ে ধীর হওয়ায় FlashInfer PCIe-IPC blindly enable করা যুক্তিযুক্ত নয়।

৬. Startup, CUDA graphs এবং Kaggle storage
6.1 আপনার 502-second startup অস্বাভাবিকভাবে অসম্ভব নয়
SGLang-এর CUDA graph initialization কয়েক মিনিট নিতে পারে। GitHub-এ প্রায় 10-minute capture/restart overhead নিয়েও issue আছে। 

GitHub




আপনার cold 502s বনাম warm relaunch 71s-এর পার্থক্য strongly suggests JIT/graph-capture or weight-cache reuse matters। কিন্তু 431-second difference-এর সবটাই compilation—এটি profiler ছাড়া প্রমাণিত নয়।
6.2 Daniel Franzen-এর সবচেয়ে প্রাসঙ্গিক startup evidence
Author: Daniel Franzen; September 2026 ARC-AGI-3 Milestone 2 write-up।

https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md

Franzen লিখেছেন:
- Checkpoint shards loader-এর প্রয়োজনীয় order-এ prefetch;
- Multiple parallel readers;
- One shard ahead prefetch;
- External background prefetcher;
- Huge shards ছোট করে reshard, model weights unchanged;
- FlashInfer / inference runtime-এর practical startup ও prefix-cache changes। 
  
  GitHub
এই changes ARC-AGI-3-এ ব্যবহার হয়েছে, ARC-AGI-2-এ নয়। কিন্তু Kaggle startup optimization-এর জন্য এগুলো directly relevant।
6.3 Cache strategy
যেসব cache পৃথকভাবে version ও persist করা দরকার:
- TRITON_CACHE_DIR
- TORCHINDUCTOR_CACHE_DIR
- CUDA_CACHE_PATH
- FlashInfer JIT cache
- Model shard staging directory
সব cache একসঙ্গে অপরিবর্তিত রেখে reused run এবং single-cache-disabled control দিয়ে startup breakdown মাপুন। Different engine builds-এর cache reuse correctness নিশ্চিত না করে arbitrary cache copy করবেন না।

আর storage fallback নিলে unfinished background prefetcher বন্ধ করতে হবে; নইলে একই GCSFUSE/NFS source-এ concurrent load তৈরি হতে পারে।

Kaggle GCSFUSE বনাম NFS-এর জন্য Qwen3.8-27B/4×L4-এর একটি universally reproducible speedup percentage পাওয়া যায়নি।

৭. ARC-AGI-2 agent harness: প্রকাশিত code ও গবেষণা থেকে কী শেখা যায়
এই অংশে সবচেয়ে গুরুত্বপূর্ণ distinction হলো, কোনো দলের ARC-AGI-3 harness প্রকাশিত থাকলেই সেটি তাদের ARC-AGI-2 leaderboard submission-এর implementation নয়।
7.1 Tufa Labs-এর Duck harness
Authors: Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tešnar ও Stefano Viel। প্রকাশ: জুলাই ২০২৬।

https://tufalabs.ai/research/duck-harness/

https://github.com/Tufalabs/duck-harness

Public implementation ও write-up-এ Python REPL tool loop, persistent state, execution feedback, agent control এবং prompt/harness design সম্পর্কে বিস্তারিত আছে। এটি ARC-AGI-3-এর interactive-game setting-এ প্রকাশিত কাজ। 

Tufa Labs
+1




আপনার Duck adaptation-এর সঙ্গে এর সবচেয়ে প্রাসঙ্গিক বিষয় হচ্ছে:
- Reasoning-এর মধ্যে Python ব্যবহার ও execution feedback।
- State retention এবং controlled continuation।
- Independent attempts থেকে candidate সংগ্রহ।
- Agent response আর environment-observed result আলাদা রাখা।
এগুলোর অনেক কিছু আপনার harness-এ ইতিমধ্যে আছে। তাই শুধু Duck repo-এর নতুন revision অনুলিপি করলে বড় performance gain পাওয়া যাবে—এমন evidence নেই।

Tufa Labs-এর 88.06 ARC-AGI-2 submission-এর actual algorithm এখনও প্রকাশিত পাওয়া যায়নি। ১০ অক্টোবরের public community reference অনুযায়ী তারা deadline-এর পরে বিস্তারিত দেওয়ার কথা বলেছিল। 

arc-explainer-production.up.railway.app



7.2 Daniel Franzen-এর প্রকাশিত ARC-AGI-3 solution
Author: Daniel Franzen, সেপ্টেম্বর ২০২৬।

https://github.com/da-fr/arc-agi-3-solution

https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md

Public write-up-এ Qwen3.8-27B ও পরবর্তী Flash-Next experiments, দশ concurrent streams, long-context retention, prefix-cache, resource scheduling, retained Python functions এবং startup optimization বর্ণিত হয়েছে। 

GitHub




কিছু published configuration:
Parameter	Reported choice
Serving GPU	RTX PRO 6000
Concurrent streams	10
Model context	131,072
Per-request output	12,288 tokens
Context draining threshold	59,392
KV cache	FP8
Python tool	Persistent REPL
Sampling	Temperature 0.7, top-p 0.95, top-k 20

এটি আপনার E12 short-request, accumulating-thinking architecture-এর কাছাকাছি public comparison, কিন্তু full ARC-AGI-2 benchmark নয়।

গুরুত্বপূর্ণ শিক্ষা: per-request 60k reasoning বাধ্যতামূলক না করেও tool-centric agent তৈরি করা যায়। তবে এর মানে আপনার 60k total reasoning allowance কেটে ফেলতে হবে, তা নয়। একই computation দ্রুত useful Python candidate-এ রূপান্তর করাই লক্ষ্য।
7.3 NVARC এবং অন্য প্রকাশিত ARC-AGI-2 notebook
Public artifact	প্রকাশিত ফল / setup	আমাদের জন্য শিক্ষা
mikelou1 minimal augmentation specialist	L4×4, প্রায় 30m44s, public 28.89; Qwen-based 4B specialist	Cheap specialist এবং candidate diversity পরীক্ষা করা যায়; 27B leaderboard system-এর proof নয়
NVARC + TRM aggressive cost order	Public 31.11, প্রায় 26m30s L4×4	Cheap-first task ordering/solver portfolio
NVARC 2026 results	Public 30.97, প্রায় 25 min L4×4	27B-এর বাইরে complementarity খোঁজার baseline
Deterministic ARC rules	21 deterministic rules; public eval-এ coverage সীমিত	Computationally cheap train-exact filters, কিন্তু general solver replacement নয়



Kaggle
+3


আরও একটি reproducibility lesson: Beiciccc-এর public ARC-AGI-2 experiment log-এ একই public notebook variant পুনরায় চালিয়ে 25.97–32.22 score variation রিপোর্ট করা হয়েছে। এটি 2026-এ leaderboard noise ও sampling variation-এর বাস্তব উদাহরণ, যদিও আপনার specific model-এর variability নয়। 

GitHub




https://github.com/Beiciccc/arc-prize-2026-arc-agi-2-experiments
7.4 ARC agent-এর জন্য গবেষণা
ARCANA: Multi-Agent Program Synthesis for ARC, Kunbo Zhang et al., 10 July 2026।

https://arxiv.org/abs/2607.09059

এটি multi-agent program synthesis ও iterative reflection নিয়ে গবেষণা। Public abstract method সম্পর্কে ধারণা দেয়, কিন্তু আপনার 4×L4 offline Kaggle limit-এর মধ্যে reproducible 70%+ score বা 12-hour feasibility প্রতিষ্ঠা করে না। 

arXiv




অর্থাৎ বেশ কয়েকটি agent ও judge যুক্ত করা সম্ভব হলেও এখনকার সময়সীমায় তা automatically better choice নয়।
Harness ideas — গবেষণা থেকে নেওয়া, কিন্তু আপনার setup অনুযায়ী ranked
Priority	পরিবর্তন	Public evidence	আমাদের প্রত্যাশিত প্রভাব
1	Early runnable transform() + immediate check()	Duck REPL ও program-synthesis literature	No-transform branches কমানো; score gain অমাপা
2	Distinct verified predictions দিয়ে pass@2	Competition rule + public solver portfolios	Duplicate attempt কমানো; baseline already implemented
3	Progress-aware continuation	Franzen-এর scheduler/context strategy	Late productive candidate ধরে রাখা; token allocation efficiency
4	Cheap deterministic/NVARC specialist first	Kaggle public notebooks	27B-র compute বাঁচানো, যদি complementarity পাওয়া যায়
5	Preserve raw reasoning only when valuable	Franzen context retention + আপনার own drop evidence	Prefix-cache ও state-loss কমানোর সম্ভাবনা
6	Model-specific ARC fine-tune/distill	Public fine-tune model cards	Accuracy opportunity, কিন্তু unverified transfer ও licensing risk
7	Multi-agent reviewer	ARCANA ও অন্য program-synthesis research	দুই বা বেশি strong candidate থাকলে মূল্যবান; সব task-এ costly

বিশেষভাবে: আপনার বর্তমান LB 34.58 থেকে 70+ যাওয়ার জন্য শুধু 20–30% throughput বাড়ানো যথেষ্ট হবে—এমন কোনো evidence নেই। Reasoning quality, branch distribution ও candidate selection-ও বড় ভূমিকা পালন করবে।

৮. আপনার উল্লেখ করা ১২টি শীর্ষ competitor-এর public investigation
২৪ অক্টোবর ২০২৬-এর কাছাকাছি score যাচাইয়ের জন্য ব্যবহার করেছি ২৩ অক্টোবরের timestamped leaderboard snapshot এবং Kaggle-এর public competition pages। Kaggle search engine-এর cached leaderboard অনেক পুরোনো হতে পারে, তাই তা বর্তমান score হিসেবে ব্যবহার করিনি। 

ARC Explainer
+1




এখানে leaderboard presence এবং method disclosure সম্পূর্ণ আলাদা করে দেখাচ্ছি।
Team	Score, user-supplied	Public materials found by Oct 24	Exact ARC-AGI-2 method
Tufa Labs	88.06	Duck harness code, ARC-AGI-3 paper/blog, competition posts	পাওয়া যায়নি; ARC-3 code public
Yi-Chia Chen	84.72	Leaderboard/team presence, public profiles	পাওয়া যায়নি
nvbanana	81.39	Team roster ও leaderboard updates; Jean-François Puget-এর linked public profile	পাওয়া যায়নি
rabbithole	80.83	Public team listing/roster	পাওয়া যায়নি
Majkel1337	75.56	ARC-AGI-2/3 leaderboard presence	পাওয়া যায়নি
ALbert RUehlman	75.56	Leaderboard presence; ২৩ অক্টোবর snapshot-এ 72.64	পাওয়া যায়নি
Dawid Kopiczko	75.14	Leaderboard/team listing	পাওয়া যায়নি
Domdolus Tolus	73.47	Team listing; ২৩ অক্টোবর snapshot-এ 72.92	পাওয়া যায়নি
カトウシュウセイ	72.50	Leaderboard presence	পাওয়া যায়নি
_hans	71.81	ARC-related older discussion/notebook references	এই 2026 ARC-AGI-2 system-এর method পাওয়া যায়নি
Youssef Nader	70.56	Leaderboard presence	পাওয়া যায়নি
mikelou1	69.72	Published 4B specialist notebook	69.72 submission-এর method পাওয়া যায়নি

Scores in the second column are from your supplied 24 October context, not all independently verified at that same timestamp. Dated public snapshot discrepancies are explicitly shown. A public profile, notebook, or another competition's code does not establish a team's private submission architecture.কয়েকটি নির্দিষ্ট discovery:

nvbanana: ২৩ অক্টোবরের public report-এ team membership হিসেবে cpmpml (Jean-François Puget-এর Kaggle identity) ও darraghdog উল্লেখ আছে। এটি team provenance, implementation disclosure নয়। 

ARC Explainer




Tufa: ৮৮.০৬ score verified by dated standings। কিন্তু Tufa-র ARC-AGI-3 Duck harness-কে তাদের high-scoring ARC-AGI-2 solution বলে উপস্থাপন করা ভুল হবে। 

ARC Explainer
+1




mikelou1: আপনার উল্লেখ করা pure Qwen3.8-27B, no NVARC, no distillation দাবি আমি স্বাধীন public post বা matching code দিয়ে নিশ্চিত করতে পারিনি। তার পাওয়া 4B specialist notebook-ও ওই দাবির বিরোধিতা প্রমাণ করে না—একজন competitor একাধিক পৃথক experiment চালাতে পারেন।

ALbert RUehlman ও Domdolus Tolus: আপনার দেওয়া score এবং timestamped ২৩ অক্টোবর snapshot আলাদা। সময়ভেদে leaderboard result বদলাতে পারে; user-supplied newer figure-কে ভুল ধরে নিচ্ছি না। 

ARC Explainer




ফল: top competitor-দের 70–88% score-এর পেছনে model architecture, token budgets, multi-agent scheduling, distillation, NVARC integration কিংবা private ensemble নিয়ে confidence সহকারে ঘোষণা করা সম্ভব হয়নি।
৯. আমাদের 4×L4-এর জন্য ranked speed improvements
এখানে expected gain দুটি ভাবে আলাদা করেছি: measurement-supported এবং scenario-only। কোনো tested improvement-কে নিশ্চিত production speedup বলছি না।
Rank	পরিবর্তন	Expected gain	Evidence strength	Risk
1	NCCL P2P→SHM A/B, protocol ও collective benchmarking	Communication 25–50% কমলে 1.14–1.32× theoretical	আপনার ~56ms profiler + NVIDIA docs; actual gain unmeasured	Low–medium
2	DFlash2 width 8/6/4 matched sweep	পাওয়া যায়নি; gain বা regression দুটোই সম্ভব	Public batch-1 evidence; exact L4 sweep নেই	Medium
3	Hybrid prefill interleaving, --chunked-prefill-size পরীক্ষা	পাওয়া যায়নি	SGLang cookbook-এ smaller chunks-এর উপকারের report	Low–medium
4	Mamba slot/ratio/cache admission tuning	Already 12 streams ধরে রাখলে throughput gain 0-ও হতে পারে	Official formula + অন্য hybrid model-এর measured cap সমস্যা	Medium
5	Prefix reuse ও unnecessary prompt prefill কমানো	Agent-effective gain অমাপা	Your 280 vs 355–460 decode tok/s	Low–medium
6	2×TP2 replicas	পাওয়া যায়নি; memory fit প্রথম শর্ত	3090 NVLink-pair comparison; L4 PHB নয়	High
7	Startup cache ও staging optimization	Cold startup saving hardware-dependent; আপনার warm-vs-cold gap 431s	Your startup logs + Franzen practices	Low–medium
8	Quantization/engine replacement	INT4→AWQ/W2/FP8 gain পাওয়া যায়নি on L4	Different GPUs/benchmarks	High
9	PP2/PP4 implementation	Current DFlash2 route unsupported	SGLang roadmap	Very high

9.1 NCCL change কত লাভ দিতে পারে?
আপনার মাপা communication cost 56ms, total step 115–133ms। তাই প্রথমে NCCL tuning চেষ্টা সবচেয়ে যুক্তিসঙ্গত। তবে actual production selection অবশ্যই measured step-time, effective tok/s এবং parity দিয়ে হবে।

Suggested controlled configurations:
# A: current automatic selection
NCCL_DEBUG=INFO

# B: force non-P2P, allow SHM
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0

# C: test Simple protocol separately
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
NCCL_PROTO=Simple


এই পরিবেশ variables-কে Kaggle notebook-এর global permanent defaults বানাবেন না। Server initialization-এর আগে apply করে একই context/load-এ benchmark, তারপর কেবল measured winner রাখুন।

যদি 56ms-এর অর্ধেক সত্যিই বাদ দেওয়া যায়, 115ms baseline ধরে expected step ~87ms। কিন্তু communication layer কতটা overlap করছে এবং kernels বদলাবে কি না, তা না জেনে 1.32× নিশ্চিত নয়।
9.2 Chunked prefill কেন গুরুত্বপূর্ণ?
SGLang-এর Qwen3.8 hybrid cookbook-এ বড় prefill chunks-এর সময় GDN/attention contention-এর উদাহরণ আছে। সেখানে ছোট --chunked-prefill-size ব্যবহার করার পরামর্শ দেখা যায়। 

GitHub




আপনার run-এ যখন 12 long-running agent একসঙ্গে reasoning করছে, মাঝেমধ্যে নতুন request-এর long prompt prefill অন্যদের decode stall ঘটাতে পারে।

প্রথমে বর্তমান resolved value লিখে রাখুন। যদি 8192 বা তার বেশি হয়, 2048-এর সঙ্গে তুলনা করুন। যদি আগে থেকেই ছোট হয়, এই experiment-এর priority কমিয়ে দিন।
9.3 Agent-effective throughput gap
আপনার baseline:
- Server decode: 355–460 tok/s
- Agent-effective: প্রায় 280 tok/s
অর্থাৎ সময়ের একটি অংশ generation ছাড়া prefill, tool execution, Python sandbox, admission gaps, context handling ও scheduling-এ যাচ্ছে।

একটি upper-bound sanity check:
\[
\frac{355}{280}=1.27,\qquad
\frac{460}{280}=1.64
\]

এটি বলে, অন্য সব অপরিবর্তিত রেখে effective throughput server decode rate পর্যন্ত পৌঁছাতে পারলে 1.27–1.64× ratio দেখা যায়। কিন্তু বাস্তবে সব overhead মুছে ফেলা সম্ভব নয়। এটি improvement forecast নয়।

এখানে highest-value diagnostics: server_generated_tokens, cache_hit_tokens, prefill_wall, decode_wall, tool_wall, scheduler_idle_wall, launch_wall এবং verified candidates per wall-hour।

১০. Deadline-এর আগে কী করব, কী করব না
26 October 2026 — Entry deadline

Competition rules accept করা ও team merger-এর deadline।



2 November 2026 — Final submission

11:59 PM UTC, unless organizers change the deadline।



এই dates Kaggle-এর competition page-এ নিশ্চিত আছে। 

Kaggle




এখন recommended execution order:
1. Existing proven baseline freeze করুন। SHA, model artifacts, serving flags, parser, notebook entrypoint ও rollback সংরক্ষণ করুন।
2. NCCL A/B + width sweep, একই model/inputs/context 12-stream workload-এ। একবারে একটি variable।
3. Candidate correctness ও runtime দুটোই দেখুন। Faster server কিন্তু failed check() বা corrupted output গ্রহণযোগ্য নয়।
4. Only-if-needed TP2 memory preflight। TP4 বন্ধ করে full alternative benchmark তখনই, যখন TP2-এর model+drafter+cache fit করার শক্ত evidence আছে।
5. Final solver integration test। Cold launch, 12-hour deadline safety, 240-task coverage, cache state, saved candidates ও submission.json integrity।
এখন নতুন PP implementation, complete engine rewrite, aggressive W2 conversion, untested large multi-agent architecture কিংবা speculative kernel rewrite শুরু করার পক্ষে যথেষ্ট evidence নেই।

১১. যে তথ্যগুলো খুঁজেও পাওয়া যায়নি
অনুসন্ধান করেছি Hugging Face model cards/community discussions, GitHub issues/PRs, Kaggle notebooks/discussions, ARC Prize site, arXiv, Reddit, X-indexed public posts, YouTube talks এবং Chinese/Japanese public-indexed resources-এ। শতাধিক লক্ষ্যভিত্তিক search query ব্যবহার করেছি; এটি পুরো ইন্টারনেটের প্রতিটি private post বা Discord channel কভার করার দাবি নয়।
Requested evidence	Finding
Qwen3.8-27B, 4×L4, all-PHB, 12 streams, 30k–100k context, 550+ tok/s reproduced result	পাওয়া যায়নি
4×L4 TP4 vs 2×TP2, identical model/context/DFlash2, numerical comparison	পাওয়া যায়নি
SGLang 0.5.21 width8/6/4/3 controlled benchmark at batch12	পাওয়া যায়নি
PCIe-only L4, FlashInfer FP8/custom all-reduce plug-in showing verified advantage over SHM	পাওয়া যায়নি
Verified SGLang 0.5.22 upgrade giving measurable L4/Qwen3.8 throughput boost	পাওয়া যায়নি
240-task, 12-hour ARC-AGI-2 pure Qwen3.8-27B public write-up scoring 70%+	পাওয়া যায়নি
Tufa Labs' exact 88.06 ARC-AGI-2 code or token-allocation policy	পাওয়া যায়নি
Top 12 teams' matched public disclosures of models, distillation, NVARC use or orchestration	বেশিরভাগ ক্ষেত্রেই পাওয়া যায়নি
GCSFUSE→NFS startup speedup under exactly this Kaggle hardware/model	পাওয়া যায়নি


১২. Raw research URLs
নিচের URLs copyable। এগুলো findings-এর সঙ্গে উপরে citations-এও যুক্ত করা হয়েছে।

Official model, speculative decoding ও benchmarks
https://huggingface.co/Qwen/Qwen3.8-27B

https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin

https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin

https://huggingface.co/Avuja/Qwen3.8-27B-int4-AutoRound

https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2

https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9

https://inco.ai/blog/dflash2/

https://huggingface.co/TeichAI/Qwen3.8-27B-Fable-Distill

https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16

SGLang, vLLM, PCIe ও NCCL
https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx

https://github.com/sgl-project/sglang/issues/43885

https://github.com/sgl-project/sglang/issues/39634

https://github.com/sgl-project/sglang/issues/36889

https://github.com/vllm-project/vllm/issues/60174

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

https://www.reddit.com/r/LocalLLaMA/comments/1vr56r1/benchmarked_qwen3827b_on_4x_rtx_3090/

https://www.reddit.com/r/LocalLLaMA/comments/1vsccit/qwen3827b_on_2x_3090_vllm_dflash2_218_toks_single/

ARC harness, notebooks, competitors এবং papers
https://tufalabs.ai/research/duck-harness/

https://github.com/Tufalabs/duck-harness

https://github.com/da-fr/arc-agi-3-solution

https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md

https://github.com/Beiciccc/arc-prize-2026-arc-agi-2-experiments

https://www.kaggle.com/code/mikelou1/arc-agi2-minimal-augmentation-specialist

https://www.kaggle.com/code/christopherdaleman/arc-2026-nvarc-trm-aggressive-cost-order

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code

https://arcprize.org/results/alibaba-qwen3-8-27b

https://arcprize.org/competitions/2026/arc-agi-2

https://arc-explainer-production.up.railway.app/arc-leaderboards

https://arxiv.org/abs/2607.09059

Final research verdict
আপনার current bottleneck-এ সবচেয়ে শক্ত evidence হলো measured TP4 all-reduce overhead। তাই low-risk engineering priority হচ্ছে SHM/NCCL transport benchmark, followed by DFlash2 width sweep এবং hybrid prefill/cache optimization।

কিন্তু 70–88% leaderboard-এর সবচেয়ে গুরুত্বপূর্ণ অজানা অংশ serving speed নয়—সেই দলগুলোর actual reasoning, training, search এবং candidate-allocation methods। তাদের অধিকাংশ implementation এখনও public নয়। ARC-AGI-3 Duck code কিংবা unrelated Kaggle notebook দিয়ে সেই missing evidence পূরণ করা যাবে না।

একই সঙ্গে official Qwen3.8-27B xhigh public ARC-AGI-2 evaluation 42.4%, low 22.8%, medium 13.2%—এই substantial effort sensitivity-ও মনে রাখতে হবে; এটি competition submission score নয়, dedicated model evaluation। 

ARC-AGI Results




সুতরাং engine speed + productive reasoning + pass@2 diversity—এই তিনটি একসঙ্গে উন্নত করা প্রয়োজন। শুধু faster token generation-কে high leaderboard score-এর সমার্থক ধরা যাবে না।

নতুন SGLang/NCCL releases ও competitor write-ups প্রকাশিত হলে সেগুলোর পরিবর্তন নিয়মিত নজরদারিতে রাখা যেতে পারে।
