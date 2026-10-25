<!-- Source: pasted by the user on 2026-10-23 (GPT analysis of Claude's "two suspects" message: TP4 all-reduce + DFlash2 8->4).
     Kept VERBATIM (Bengali). Claude's claim-by-claim validation lives in "02 Claude validation of GPT note 01.md". -->

ARC-AGI-2 — GPU Bottleneck Research & Validation
মূল সিদ্ধান্ত: দুটি সন্দেহই পরীক্ষাযোগ্য, কিন্তু কোনোটিই এখনো প্রমাণিত প্রধান bottleneck নয়
আপনার প্রস্তাবের সবচেয়ে শক্তিশালী অংশ হলো TP4 communication overhead এবং DFlash2 verification width—দুটোকে আলাদা করে মাপার চিন্তা। তবে কয়েকটি গুরুত্বপূর্ণ গণনাগত ও architectural সংশোধন দরকার। এগুলো না করলে GPU profiling-এর ফল ভুলভাবে ব্যাখ্যা হওয়ার ঝুঁকি থাকবে।
১. পুরোনো GPU logs থেকে যা independently যাচাই করেছি
আগের আপলোড করা E12 rev 6f ZIP-এ পাওয়া গেছে:
- E11-05_EVAL30_sglang_server.log — 3,470 lines
- E11-06_P1-60K_sglang_server.log — 1,876 lines
- আরও E12 historical server logs
এই দুটো E11 log-এর server_args থেকে নিশ্চিত হয়:
Configuration	Recorded value
Tensor parallelism	TP=4
Speculative algorithm	DFLASH
Verification width setting	8
Speculation steps	1
Draft model	Qwen3.8-27B-DFlash2 BF16
KV cache	FP8 E4M3

Decode records-এ #running-req, accept len, accept rate, gen throughput (token/s) আছে। অর্থাৎ throughput এবং acceptance সম্পর্কে আপনার অনুমানকে log দিয়ে যাচাই করার সুযোগ রয়েছে।
Log-derived DFlash2 acceptance
Log	Running requests	Mean accept length
E11-06	8 (408 records)	4.46
E11-06	12 (278 records)	4.37
E11-05	12 (114 records)	3.91

এগুলো decode-log sample-এর arithmetic mean; প্রতি verification step-এর exact, time-weighted বা token-weighted aggregate নয়।
সুতরাং গড়ে প্রায় 4.3 accepted tokens—এই observation বিশ্বাসযোগ্য। কিন্তু 4.3/8 থেকে সরাসরি 46% GPU compute নষ্ট হয়েছে বলা যায় না। কেন, সেটি নিচে ব্যাখ্যা করছি।
২. বড় আবিষ্কার: TP4 communication-path সমস্যার সরাসরি প্রমাণ পাওয়া গেছে
এটি আপনার প্রথম সন্দেহকে উল্লেখযোগ্যভাবে শক্তিশালী করে।

E11-06-এর আসল server log-এ চারটি TP worker-ই জানিয়েছে:
CustomAllReduceV2 is disabled because
it's not supported on more than two PCIe-only GPUs.


পুরোনো CustomAllreduce-ও একই কারণে disabled।

এটি কোনো theoretical অনুমান নয়—আপনার নিজের recorded GPU run-এর startup message।

SGLang-এর source code-এও একই limitation আছে: চারটি PCIe-only GPU-তে NVLink ছাড়া optimized custom all-reduce path disabled হয়। তবে এর মানে NCCL all-reduce কাজ করছে না, এমন নয়। সাধারণ NCCL collective-এ fallback হতে পারে। 

GitHub
+1




এখন যা নিশ্চিত:
- SGLang আপনার চার L4-কে PCIe-only configuration হিসেবে শনাক্ত করেছে।
- TP4-তে optimized custom all-reduce kernels ব্যবহার করতে পারেনি।
- তাই TP4 বনাম TP2 communication-path পার্থক্যের একটি বাস্তব, code-supported কারণ আছে।
এখনো যা অজানা: NCCL আসলে P2P, shared-memory/host staging, নাকি অন্য transport ব্যবহার করেছে; এবং এই collective-গুলো মোট decode time-এর কত শতাংশ নিয়েছে।

NCCL documentation অনুযায়ী P2P না থাকলে shared-memory transport ব্যবহার হতে পারে। nvidia-smi topo -m এবং ছোট-message benchmark দিয়ে এই দুটি অবস্থাকে আলাদা করতে হবে। 

NCCL 2.4.6 documentation
+1




এখন পর্যন্ত গবেষণায় এটিই সবচেয়ে শক্তিশালী finding।
৩. DFlash2-এর ৮→৪ পরিবর্তনে আরেকটি বড় ঝুঁকি
Checkpoint compatibility যাচাই ছাড়া পরিবর্তনটি গ্রহণযোগ্য নয়
আপনার ব্যবহৃত drafter-এর public reference checkpoint-এ:
"dflash_config": {
  "block_size": 8,
  "conv_group_size": 16,
  "conv_kernel_size": 2,
  "target_layer_ids": [5, 19, 33, 47, 61]
}


SGLang v0.5.21-এর dflash_worker_v2.py-তে configured width এবং checkpoint block size আলাদা হলে warning দিয়ে set_block_size() কল করা হয়। অর্থাৎ command-line flag পরিবর্তন করা সম্ভব হলেও সেটি একই trained 8-position model থেকে শুধু শেষ চারটি prediction বাদ দেওয়ার সমতুল্য নয়। 

z-lab/Qwen3.8-27B-DFlash2 at main
+1




এই checkpoint মূলত 8-position verification দিয়ে প্রকাশিত এবং benchmark করা হয়েছে। 

Hugging Face
+1




এ কারণে আমার প্রস্তাবিত পরীক্ষার ক্রম:
1. 8-token checkpoint অপরিবর্তিত রেখে baseline মাপুন।
2. 6 ও 4 width-এ model load হয় কি না, warning, acceptance এবং greedy-equivalence যাচাই করুন।
3. তারপর effective output tokens/second তুলনা করুন।
4. 3-width-কে optional arm রাখুন; 8 ও 4-এর মধ্যে যথেষ্ট পার্থক্য না থাকলে আগে সেটি চালানোর দরকার নেই।
আরেকটি সতর্কতা: SGLang-এর issue #36548-এ concurrency-র সঙ্গে DFlash2 output inconsistency রিপোর্ট করা হয়েছে। Issue-টির reported environment আপনার exact pinned stack নয়, তাই নিজের run-এ bug ঘটছে ধরে নেওয়া যাবে না। কিন্তু TP4/TP2 বা draft-width comparison-এর আগে একই input-এর output correctness পরীক্ষা করার শক্ত কারণ এটি। 

GitHub




অর্থাৎ throughput বৃদ্ধি যদি output বদলে দেয়, তা ARC competition-এর জন্য গ্রহণযোগ্য optimization নয়।
৪. Latency model-এর গণনাগত অডিট
৪.১ প্রতি-request marginal cost
আপনার দেওয়া সংখ্যাগুলো থেকে:
Configuration	Reported marginal cost	আমার মূল্যায়ন
No speculation	~1.3–1.45 ms/request	Arithmetic মোটামুটি ঠিক
MTP (~4 verified tokens)	4.4 ms/request	1.10 ms/verified token
DFlash2 (8 verified tokens)	6.8–6.9 ms/request	0.85–0.86 ms/verified token

কিন্তু এই তিনটি slope থেকে প্রতি GPU মাত্র 10–16 TFLOPS ব্যবহৃত হচ্ছে—এটি প্রতিষ্ঠিত হয় না।

কারণ observed marginal time-এর মধ্যে GEMM ছাড়াও communication, kernel launches, synchronization, quantization/dequantization, GDN state computation এবং speculative verification overhead থাকতে পারে। তিনটি আলাদা serving mode-এর slope একই ধরনের কাজের মাপও নয়।

আর NVIDIA-এর official L4 specification অনুযায়ী dense FP16/BF16 Tensor Core peak 121 TFLOPS, sparsity-সহ 242 TFLOPS। 121 সংখ্যাটি ঠিক, কিন্তু এটি sustained inference throughput-এর নিশ্চয়তা নয়। 

NVIDIA L4 GPU Full Specifications




সংশ্লিষ্ট specification:

https://www.nvidia.com/en-in/data-center/l4/
৪.২ পুরোনো logs থেকে DFlash2-এর সময় পুনর্গণনা
আপনার formula ব্যবহার করে, অর্থাৎ:
\[
t_{\mathrm{inferred}} \approx
\frac{N_{\mathrm{running}}\times L_{\mathrm{accept}}}
{\mathrm{reported\ output\ tok/s}}
\]

১২টি request-এর জন্য পেয়েছি:
Log	Estimated median decode-step time
E11-06	124.6 ms
E11-05	128.4 ms

E11-06-এ central 80% আনুমানিক 116–133 ms।

অর্থাৎ আপনার 115.5 ms estimate অসম্ভব নয়, কিন্তু historical log-এর কেন্দ্রীয় মান কিছুটা বেশি।

এই পুনর্গণনাটিও approximation, কারণ SGLang-এর throughput field একটি logging interval-এর পরিসংখ্যান; প্রতিটি decode kernel-এর আলাদা CUDA-event duration নয়। তাই profiler ছাড়া এগুলোকে precise per-step latency হিসেবে ব্যবহার করা উচিত নয়।

এছাড়া ZIP-এ E11-03-এর no-speculation log এবং আপনার 64 MB all-reduce benchmark-এর raw output পাইনি। ফলে ওই সংখ্যাগুলোর arithmetic যাচাই করেছি, কিন্তু measurement reproducibility নিশ্চিত করতে পারিনি।
৪.৩ All-reduce-এর 40–50 ms hypothesis
আপনার হিসাবের ধারণাটি যুক্তিসঙ্গত:
- Hidden width 5,120
- BF16 activation = 2 bytes
- প্রতি token-এর hidden activation প্রায় 10 KiB
- 12 request × 8 positions ≈ 0.98 MB per collective
- 128 collective হলে মোট logical payload প্রায় 126 MB
3.1 GB/s effective bandwidth বসালে প্রায় 40 ms পাওয়া যায়।

কিন্তু এটি measured NCCL time নয়।

দুটি বড় অনিশ্চয়তা আছে। প্রথমত, 64টি model layer × 2 all-reduce হলো conventional tensor-parallel architecture-এর একটি হিসাব; actual Qwen3.8 hybrid model এবং SGLang fused execution-এ কতটি collective হয়েছে, trace ছাড়া নিশ্চিত নই। দ্বিতীয়ত, 64 MB message-এর effective bandwidth দিয়ে 10 KB–1 MB message-এর latency হিসাব করা যায় না। ছোট collective-এ fixed startup ও synchronization latency গুরুত্বপূর্ণ।

NVIDIA-এর nccl-tests নিজেই ছোট message-এর জন্য latency এবং বড় message-এর জন্য bandwidth আলাদা করে মূল্যায়ন করতে বলে। 

GitHub




https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

Verdict: 40–50 ms communication cost একটি গ্রহণযোগ্য hypothesis, verified measurement নয়।
৫. ৮→৪ draft width-এ 550–650 tok/s: কতটা বাস্তবসম্মত?
এখানে সবচেয়ে প্রয়োজনীয় সংশোধন।

যদি 12-request DFlash2 setup-এ 8-width-এর acceptance 4.3 এবং step time 115.5 ms ধরি:
\[
T_8 \approx \frac{12\times4.3}{0.1155}
\approx447 \text{ tokens/s}
\]

আপনার proposed 4-width step time 70 ms হলে 550–650 tok/s অর্জনের জন্য প্রয়োজন:
\[
L_4 =
\frac{T_4\times0.070}{12}
\]

অর্থাৎ 4-width-এ গড়ে 3.21–3.79 output tokens per verification।

এটি অত্যন্ত ভালো acceptance প্রয়োজন করে, কারণ 4-width-এর theoretical maximum output tokens per step হলো 4।

একটি সরল illustrative model ধরে দেখা যাক—প্রতিটি subsequent draft position-এর conditional success probability একই \(\alpha\)।
\[
L_8 = 1+\alpha+\alpha^2+\cdots+\alpha^7=4.3
\]

এখান থেকে \(\alpha\approx0.811\)। একই probability model-এ:
\[
L_4\approx3.00,\qquad L_6\approx3.78
\]

70 ms-এ 4-width-এর implied throughput প্রায় 515 tok/s, 550–650 নয়।

এটি prediction নয়; simplified model দিয়ে আপনার দাবির sensitivity যাচাই। বাস্তবে conditional acceptance positional এবং workload-dependent। NVIDIA-এর speculative decoding analysis-ও optimum নির্ধারণে acceptance length, verification cost ও draft cost—তিনটিই ব্যবহার করে। 

NVIDIA Technical Blog




https://developer.nvidia.com/blog/co-designing-ai-models-using-speculative-decoding-for-faster-llm-inference
12 requests ধরে illustrative throughput comparison

02004006008008-width baseline4-width model4-width optimistic









8-width: আপনার 115.5 ms assumption; 4-width: 70 ms assumption ও 3.00 acceptance; 6-width-এর step time অনুমান না করে দেখানো হয়নি।





এটি hypothetical arithmetic, benchmark result নয়। 650 tok/s-এর জন্য প্রয়োজনীয় 3.79 acceptance এখনো প্রমাণিত নয়।
আরও একটি গুরুত্বপূর্ণ বিষয়: DFlash2-এর developers নিজেরাই fixed-width verification-এ wasted computation শনাক্ত করে dynamic verification feature proposal করেছেন (#36127)। এটি আপনার optimization idea-কে সমর্থন করে, কিন্তু SGLang v0.5.21-এ এই feature কার্যকর আছে—এমন প্রমাণ নয়। 

GitHub




https://github.com/sgl-project/sglang/issues/36127

সুতরাং আমার সিদ্ধান্ত:
- 4-width: worthwhile experiment
- 6-width: অত্যন্ত গুরুত্বপূর্ণ intermediate experiment
- 3-width: lower priority
- 550–650 tok/s: optimistic hypothesis, expected result নয়
৬. TP4 বনাম 2×TP2 — সম্ভবত সবচেয়ে লাভজনক পরীক্ষা
আপনার E11-06 server log-এ model-load memory usage ছিল:
Component	TP4-এ প্রতি GPU
INT4 target	4.62 GB
BF16 DFlash2 drafter	1.49 GB
Combined weights	6.11 GB

সাধারণ shard-size doubling ধরে TP2-এ combined weights প্রায় 12.22 GB/GPU হতে পারে। L4-এর 24 GB-এর মধ্যে এটি সম্ভব মনে হচ্ছে।

কিন্তু বাকি জায়গায় CUDA graphs, activation workspace, KV pool, GDN recurrent state এবং request batching রাখতে হবে। তাই TP2 model fits এবং TP2 6 concurrent long-context requests reliably serves—দুটি আলাদা প্রশ্ন।

2×TP2-এর সুবিধা হতে পারে:
- প্রতি replica-তে দুই GPU-র communication।
- চারটি GPU জুড়ে all-reduce এড়ানো।
- দুই independent serving engine, প্রতিটিতে পৃথক KV/cache state।
- TP2-তে optimized custom all-reduce পুনরায় available হওয়ার সম্ভাবনা।
তবে SGLang-এর custom all-reduce TP2-তে চলবে কি না, সেটিও actual P2P capability-এর ওপর নির্ভর করে। দুটি GPU ব্যবহার করলেই optimized collective নিশ্চিত হয় না।

আরেকটি বিষয়: 2×TP2 মানে প্রতিটি replica তার নিজস্ব subset of requests serve করবে। Total throughput যোগ করতে হবে; একটি TP2-এর tok/s দ্বিগুণ ধরে নেওয়া যাবে না, যদি দুই replica-র workloads আলাদা হয়।

প্রস্তাবিত topology-aware pairing: nvidia-smi topo -m এবং nvidia-smi topo -p2p p দেখে সবচেয়ে ভালো GPU pairs বেছে নিন। শুধু 0–1 ও 2–3 pairing অন্ধভাবে নির্ধারণ করবেন না। NVIDIA GPU troubleshooting documentation-ও P2P matrix যাচাই করার নির্দেশ দেয়। 

NCCL 2.32.3 documentation




https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
৭. নতুন সম্ভাব্য correctness problem — throughput-এর চেয়েও গুরুত্বপূর্ণ
আমি আরও কয়েকটি SGLang report যাচাই করেছি:
Issue	Evidence	আপনার stack-এ গুরুত্ব
#36548	DFlash2 high concurrency-তে inconsistent outputs; report-এ target-only control ঠিক ছিল	HIGH preflight priority
#38009	Thinking enabled অবস্থায় greedy DFlash2 বনাম target-only divergence	HIGH preflight priority
#37475	Missing target_layer_ids হলে wrong trained tap layers বেছে নেওয়ার ঝুঁকি	Config validation প্রয়োজন
#39087	Quantized drafter load হলেও near-zero acceptance হতে পারে	BF16 drafter বজায় রাখা
#28180	DFlash penalty accumulation/scaling issue	Repetition penalty experiment-এর আগে যাচাই

এগুলো public issue reports, আপনার pinned Kaggle build-এর confirmed failures নয়। বিশেষ করে #36548-এ একই 8→4 ধরনের width adjustment দিয়েও underlying inconsistency পুরোপুরি যায়নি। 

GitHub
+4




Raw links:

https://github.com/sgl-project/sglang/issues/36548

https://github.com/sgl-project/sglang/issues/38009

https://github.com/sgl-project/sglang/issues/37475

https://github.com/sgl-project/sglang/issues/39087

https://github.com/sgl-project/sglang/issues/28180

Practical implication: Benchmark শুরুতে 20–50টি fixed prompt নিয়ে greedy mode-এ target-only বনাম DFlash2 token IDs তুলুন। Serial এবং concurrent দুটো mode রাখুন। Difference হলে সেটি nondeterministic scheduling, FP8 numerical behavior, অথবা speculative integration—কোনটির কারণে, তা আলাদা করে অনুসন্ধান করুন। কোনো throughput arm correctness gate fail করলে সেটিকে production candidate হিসেবে গ্রহণ করবেন না।
৮. সবচেয়ে কার্যকর GPU profiling plan
আপনার quota বাঁচানোর জন্য আমি proposed four-part experiment-কে দুটি স্তরে ভাগ করব।
Stage A — Communication এবং current TP4 bottleneck
Step	পরীক্ষা	মাপা হবে
A1	GPU topology / P2P matrix	GPU pairing ও P2P availability
A2	NCCL 8 KB–1 MB all-reduce	Latency curve, bandwidth নয় শুধু
A3	Existing TP4 DFlash8 server	Baseline throughput, acceptance, memory
A4	12-request short profiling window	NCCL vs GEMM vs GDN vs drafter
A5	Greedy serial/concurrent comparison	Correctness

যদি nccl-tests আগে থেকেই available থাকে, উদাহরণ:
nvidia-smi topo -m
nvidia-smi topo -p2p p

NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET \
./all_reduce_perf \
  -b 8K -e 1M -f 2 -g 4 -w 20 -n 100


একই sweep TP2-এর জন্যও চালাতে হবে, এবং 64 MB result পুনর্মাপতে হবে। all_reduce_perf binary না থাকলে offline PyTorch distributed benchmark ব্যবহার করা যাবে; runtime-এ নতুন package download-এর ওপর নির্ভর করা উচিত নয়।

Profiler-এ মোট সময়কে অন্তত চার ভাগে classify করুন: collective kernels, quantized GEMM, GDN/attention kernels, এবং speculative draft/accept/commit overhead। CUDA-graph replay-র ভেতরের kernels দেখতে trace capture করতে হবে; শুধু Python profiler-এর outer-function timing যথেষ্ট নয়।

SGLang-এর profiling documentation:

https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx


GitHub


Stage B — Configuration comparison
Stage A-তে engine healthy হলে:
Arm	GPU partition	Draft width	প্রধান প্রশ্ন
B0	TP4	8	Current baseline
B1	TP4	6	Middle-width improvement আছে?
B2	TP4	4	Verification কমানোর net benefit কত?
B3	2×TP2	8	Communication reduction থেকে improvement হয়?
B4	2×TP2	Best validated width	Combined best-case

প্রতিটি arm-এ একই fixed prompts, sampling settings, context-length distribution, target/draft checkpoint revisions এবং total workload ব্যবহার করুন।

বেঞ্চমার্কের scorecard:
engine_startup_sec
total_wall_sec
total_output_tokens
end_to_end_output_tok_per_sec
steady_state_output_tok_per_sec
median_decode_step_ms
accepted_tokens_per_verification
collective_gpu_critical_path_ms
peak_gpu_memory_gb
kv_and_mamba_usage
greedy_token_match_rate
request_failures


collective_gpu_critical_path_ms-কে চার GPU-র summed kernel times দিয়ে হিসাব করবেন না; overlapping execution double-count হতে পারে।
Stage C — বাস্তব ARC workload
Synthetic 12-request throughput-এর winner-কে 8–12টি আসল ARC-style prompts দিয়ে confirm করুন। Long thinking, Python tool calls, cache reuse, 10k–50k context, এবং request churn থাকতে হবে।

কারণ uniform 1,024-token benchmark-এর winner বাস্তব agentic workload-এ winner নাও হতে পারে।
৯. ৩০–৪০ মিনিটের হিসাব কি বাস্তবসম্মত?
এখানে আমি আপনার সময় ও quota estimate সংশোধন করব।

Recorded startup timeline:
Run	server_args থেকে ready
E11-05	প্রায় 7 মিনিট 39 সেকেন্ড
E11-06	প্রায় 7 মিনিট 30 সেকেন্ড
Rev 6d G4B	প্রায় 19 মিনিট 9 সেকেন্ড

Rev 6d-তে model loading/storage stall startup দীর্ঘ করেছে। E11 runs-এ CUDA-graph capture-ই বড় অংশ নিয়েছে।

অর্থাৎ চার-পাঁচটি আলাদা server configuration cold-start করলে benchmark শুরু হওয়ার আগেই উল্লেখযোগ্য সময় চলে যেতে পারে।

আমি একসঙ্গে সব arms চালিয়ে ৩০–৪০ মিনিট শেষ হওয়ার নিশ্চয়তা দেব না।

Kaggle competition-এর official rule-এ আরও গুরুত্বপূর্ণ তথ্য আছে: L4x4 notebooks পুরোনো T4x2/P100-এর তুলনায় দ্বিগুণ quota consume করে। 

Kaggle




https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators

তাই 40 মিনিট L4x4 wall time হলে reference GPU quota-তে প্রায় 80 মিনিট কাটতে পারে। আপনার ১–১.৫ ঘণ্টা quota estimate শুধু ছোট run-এর জন্য যুক্তিসঙ্গত; complete 5-arm sweep-এর জন্য নয়।

আমার প্রস্তাবিত bounded execution plan:
Phase	Planned cap
Topology + NCCL microbenchmarks	5–8 মিনিট
TP4 DFlash8 startup ও profiler	15–25 মিনিট
TP4 DFlash4/6 selected follow-up	অতিরিক্ত 10–20+ মিনিট প্রতি restart
2×TP2 startup ও comparison	আলাদা GPU budget প্রয়োজন

এগুলো operational caps, guaranteed execution times নয়। Server startup failure, quota cancellation, profiler incompatibility বা storage delay হলে fail-closed stop দরকার। GPU process বন্ধ হওয়ার আগে trace এবং JSON summaries অবশ্যই disk-এ flush করতে হবে।
Minimum viable experiment
Quota খুব সীমিত হলে প্রথম run-এ শুধু Stage A করুন। সেটি communication-কে সত্যিকার bottleneck হিসেবে প্রমাণ বা খণ্ডন করতে পারবে। তারপর result দেখে সিদ্ধান্ত নিন TP2 পরীক্ষা বেশি গুরুত্বপূর্ণ, নাকি DFlash width sweep।

এই ক্রমে একটি expensive GPU run থেকেই পরের optimization direction বেছে নেওয়া সম্ভব।

১০. সবচেয়ে মূল্যবান sources
বৃহত্তর গবেষণায় প্রায় ২০০টি targeted web query ব্যবহার করেছি—SGLang source, issues, NCCL/CUDA documentation, NVIDIA hardware specifications, DFlash2 model/config, papers ও community benchmarks। নিচেরগুলো আপনার নির্দিষ্ট hardware/software setup-এর জন্য সবচেয়ে প্রাসঙ্গিক:
Source	কেন গুরুত্বপূর্ণ
SGLang v0.5.21 custom all-reduce source	TP4 PCIe-only limitation-এর সরাসরি code evidence
SGLang v0.5.21 DFlash worker	Runtime block-size override এবং warning
NVIDIA NCCL troubleshooting	Actual GPU P2P/SHM transport শনাক্ত করার পদ্ধতি
NVIDIA nccl-tests	ছোট collective latency মাপার স্বীকৃত পদ্ধতি
Inco AI DFlash2 paper/blog	Acceptance বনাম throughput এবং checkpoint-এর মূল design
SGLang #36548, #38009	Concurrency ও greedy-equivalence correctness risks

Raw URLs:

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/dflash_worker_v2.py

https://github.com/NVIDIA/nccl-tests

https://inco.ai/blog/dflash2/

https://huggingface.co/incoai/Qwen3.8-27B-DFlash2

https://github.com/sgl-project/sglang/issues/36127
Final research verdict
Hypothesis 1: TP4 communication bottleneck
Strong evidence

চারটি PCIe-only GPU-তে optimized custom all-reduce disabled—আপনার log এবং pinned source উভয়েই নিশ্চিত। তবে 40–50 ms collective cost এখনো অনুমান।
Hypothesis 2: DFlash2 8→4 improves throughput
Unproven

Fixed-width wasted verification বাস্তব; কিন্তু runtime block-size override, acceptance কমা এবং correctness risk বিবেচনায় 550–650 tok/s এখনই justified expectation নয়।
TP4 বনাম 2×TP2
High-priority experiment

TP2 memory-fit সম্ভাবনাময় এবং communication-path উন্নত হতে পারে। Real concurrency, memory ও correctness পরীক্ষা ছাড়া 2.4× speedup ধরে নেওয়া যাবে না।
আমার চূড়ান্ত সুপারিশ: আগে GPU topology + small-message NCCL benchmark + TP4 kernel trace করুন। বিশেষ করে TP4-তে actual NCCL critical-path time যদি 30–50 ms পাওয়া যায়, তাহলে 2×TP2-কে প্রধান optimization হিসেবে অনুসরণ করার শক্ত ভিত্তি তৈরি হবে। অন্যদিকে NCCL যদি মাত্র 5–10 ms হয়, তাহলে DFlash verification/GDN/quantized GEMM profiling-এ গুরুত্ব দিতে হবে।

কোনো solver code, Duck architecture বা prompt পরিবর্তনের প্রয়োজন নেই। এটি পুরোপুরি serving-engine performance এবং correctness validation-এর কাজ।
