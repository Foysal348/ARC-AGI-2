<!-- Source: pasted by the user on 2026-10-23 (third GPT note: independent validation; adds the accept-len-includes-bonus-token correction
     from SGLang metrics_reporter.py, per-run accept means incl. E12 EVAL30, CUDA-graph memory, gated benchmark A-D, result sheet).
     Kept VERBATIM (Bengali). Claude's claim-by-claim validation goes into "02 Claude validation of GPT notes 01, 03 and 04.md". -->

ARC-AGI-2 — GPU Throughput Bottleneck: Independent Validation ও Research
বিশ্লেষণের ভিত্তি: আপনার দেওয়া হিসাব, আগের E12 Rev 6f ZIP-এ সংরক্ষিত E11-05/E11-06/E12 server logs, SGLang source code, NVIDIA hardware documentation, NCCL documentation এবং DFlash-related research।
১. প্রথম গুরুত্বপূর্ণ আবিষ্কার: TP4 communication নিয়ে আপনার সন্দেহের পক্ষে সরাসরি evidence আছে
পুরনো লগে শুধু throughput নয়, এই warning-টিও পাওয়া গেছে:
CustomAllReduceV2 is disabled because it's not supported
on more than two PCIe-only GPUs.


এটি E11-05, E11-06 এবং E12 EVAL30-এর logs-এ আছে।

SGLang-এর implementation-ও নিশ্চিত করছে যে দুইটির বেশি PCIe-only GPU হলে তার optimized custom all-reduce implementation disable হতে পারে। 

GitHub




এর অর্থ:
- আমাদের 4×L4 TP4 configuration-এ একটি গুরুত্বপূর্ণ optimized communication path ব্যবহার হচ্ছে না।
- TP2 configuration-এ ওই নিষেধাজ্ঞা প্রযোজ্য নয়; hardware ও P2P support ঠিক থাকলে optimized path পাওয়া সম্ভব।
- তবে P2P নেই বা সব all-reduce host RAM দিয়ে যাচ্ছে—এটি এখনও প্রমাণিত নয়। NCCL নিজেও PCIe P2P ব্যবহার করতে পারে। 
  
  NCCL 2.32.3 documentation
এটি আপনার TP4 বনাম TP2 hypothesis-কে আগের চেয়ে অনেক বেশি গুরুত্ব দেয়।

এই evidence নতুন GPU খরচ ছাড়াই পাওয়া গেছে। তাই expensive experiment-এর আগে NCCL transport এবং SGLang all-reduce backend আলাদাভাবে পরীক্ষা করা উচিত।
২. Draft-token ৮→৪: আকর্ষণীয় hypothesis, কিন্তু 550–650 tok/s এখনও প্রমাণিত নয়
আপনার model-এর সংখ্যাগুলো দিয়ে:
- 8-wide verification: 115 ms/step
- Mean accepted: 4.3 tokens/request
- 12 requests-এর throughput: প্রায় 449 tok/s
4-wide verification সত্যিই 70 ms/step হলে:
4-wide mean accepted	Calculated aggregate throughput	8-wide-এর তুলনায়
2.5	429 tok/s	−4%
3.0	514 tok/s	+15%
3.5	600 tok/s	+34%
4.0 (theoretical maximum)	686 tok/s	+53%

অর্থাৎ আপনার 550–650 tok/s অনুমান সম্ভব, কিন্তু সেটির জন্য 4-wide-এ প্রায় 3.21–3.79 accepted tokens এবং 70 ms step time দরকার।

সবচেয়ে গুরুত্বপূর্ণ অজানা তথ্য হলো প্রতি position-এর acceptance distribution। শুধু 8-wide mean 4.3 জানলে 4-wide mean নির্ণয় করা যায় না।

আরেকটি সংশোধন: ৮টির মধ্যে ৪.৩টি গ্রহণ করা মানে ৪৬% verification positions final output হয়নি—এটি arithmetic হিসেবে ঠিক। কিন্তু সব rejected-position computation অপচয় ছিল, এমন সিদ্ধান্ত সঠিক নয়; বেশি verification width ভবিষ্যতের sequential target passes বাঁচায়।
৩. পুরনো server logs পুনরায় গণনা করে যা পেলাম
E12 Rev 6f ZIP-এর runs/server_logs_12_streams/ থেকে decode records পুনরায় parse করেছি। নিচের values log-derived, profiler-measured kernel latency নয়।
Run	Decode records	Mean accept length	12-stream median step-time proxy
E11-05 EVAL30	2,509	3.87	128.4 ms
E11-06 P1-60K	1,299	4.58	124.6 ms
E12 EVAL30 Rev 1	1,209	5.10	137.9 ms

Step-time proxy হিসাব করেছি:
\[
T_{\mathrm{proxy}}=\frac{N_{\mathrm{running}}\times L_{\mathrm{accept}}}{\mathrm{logged\ tok/s}}
\]

এখানে zero/warm-up ধরনের অস্বাভাবিক throughput record বাদ দিয়েছি। এটি নির্ভুল per-kernel measurement নয়, কারণ log-এর throughput aggregation ও request transitions-ও প্রভাব ফেলতে পারে।

মূল পর্যবেক্ষণ: Mean acceptance 3.87 থেকে 5.10 পর্যন্ত পরিবর্তিত হয়েছে। তাই fixed 4.3 ধরে draft-width optimize করলে ভুল সিদ্ধান্ত হতে পারে। একই ARC workload, sampling settings, context distribution এবং concurrency ধরে পরীক্ষা জরুরি।
Context estimate নিয়ে যাচাই
Qwen3.8-27B-এর 64 layers-এর মধ্যে 16টি full-attention layer; KV-head configuration অনুযায়ী FP8 KV cache-এর হিসাব:
\[
12\times50{,}000\times16\times
\frac{4}{4}\times256\times2\times1
\approx4.92\ \mathrm{GB/GPU}
\]

অর্থাৎ আপনার আনুমানিক 4.8 GB সঠিক মাত্রার হিসাব। NVIDIA L4-এর 300 GB/s memory bandwidth-এ শুধু ওই data একবার পড়তে theoretical প্রায় 16.4 ms লাগবে। 

Qwen/Qwen3.5-27B at main


 

NVIDIA L4 GPU Full Specifications




তবে এই agreement দিয়ে attention implementation optimal প্রমাণিত হয় না। KV memory access efficiency, cache reuse, GDN operations এবং অন্য kernels-এর সময়ও profiler-এ দেখতে হবে।
৪. আরও দুটি গুরুত্বপূর্ণ finding
4.1 TP4 বনাম 2×TP2 শুধু communication benchmark নয়—memory feasibility-ও সমস্যা
E11-06-এর startup log-এ প্রতি GPU:
Allocation	Observed memory
INT4 target weights	4.62 GB
BF16 DFlash2 weights	1.49 GB
CUDA graph capture (target)	0.68 GB
CUDA graph capture (draft)	0.35 GB

TP4 থেকে TP2-তে গেলে model-এর অনেক sharded weights প্রতি GPU-তে প্রায় দ্বিগুণ জায়গা নিতে পারে। ফলে 2×TP2-তে KV cache capacity ও maximum context TP4-এর তুলনায় কমে যাওয়ার আশঙ্কা আছে।

পুরোনো log-এ target CUDA graph capture একবারেই 372.5 seconds নিয়েছে। অর্থাৎ model configurations বারবার cold-start করলে benchmark time অনেক ব্যয় হতে পারে।

এজন্য 2×TP2-কে প্রথমেই full benchmark করা উচিত নয়। আগে per-GPU weight memory, usable KV pool এবং target context-এর admission পরীক্ষা করা দরকার।
4.2 ৮→৪ flag সরাসরি production-এ দেওয়া নিরাপদ নয়
Public DFlash2 configuration:

https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json

এতে block_size=8। SGLang v0.5.21-এর source-এ runtime override করার ব্যবস্থা আছে, কিন্তু mismatch হলে warning দিয়ে drafter-এর block size বদলায়। 

incoai/Qwen3.8-27B-DFlash2 at main


 

GitHub




এবং grouped convolution-এর block indexing-ও সেই নতুন size অনুসরণ করে। 

GitHub




এতে ৪-token mode নিষিদ্ধ হয়ে যায় না। কিন্তু ৮-wide-এর প্রথম চার candidate একইভাবে ৪-wide-এ তৈরি হবে, এমন নিশ্চয়তা নেই। Acceptance পুনরায় মাপতে হবে।

Hugging Face-এর সরাসরি আলোচনাটি পেয়েছি:

https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9

সেখানে একজন ব্যবহারকারী শেষের speculative positions-এ প্রায় 0.1 acceptance বলেছেন, আরেকজন ৪-token setting-এ coding workload-এ 60–75% acceptance জানিয়েছেন। তবে hardware, engine, prompt ও acceptance definition আমাদের setup-এর সঙ্গে controlled comparison নয়। 

Hugging Face




এই কারণে আমার বর্তমান অগ্রাধিকার ৮ বনাম ৬ বনাম ৪ পরীক্ষা, সরাসরি ৪ নির্বাচন নয়।
৫. আরও গুরুত্বপূর্ণ সংশোধন: accept len আর accepted draft tokens এক জিনিস নয়
এটি research-এ পাওয়া সবচেয়ে গুরুত্বপূর্ণ সংশোধনগুলোর একটি।

SGLang v0.5.21-এর নিজের metrics_reporter.py যাচাই করে নিশ্চিত হয়েছি:
- accept len হলো প্রতি verification step-এ মোট output tokens, যার মধ্যে bonus token-ও থাকে।
- accept rate হিসাবের জন্য accepted draft tokens আলাদাভাবে গণনা করা হয়।
- speculative_num_draft_tokens = 8 হলে metric denominator হিসেবে সাধারণত ৭টি draft position ব্যবহৃত হয়। 
  
  GitHub
সুতরাং আগের 4.3/8 = 54% acceptance হিসাবটি যথাযথ নয়।

যদি accept len = 4.3 হয় এবং standard linear drafting accounting প্রযোজ্য হয়, তাহলে:
\[
\text{Mean correct drafts}=4.3-1=3.3
\]

\[
\text{Draft acceptance rate}\approx\frac{3.3}{7}=47.1\%
\]

অর্থাৎ draft rejection আনুমানিক 52.9% হতে পারে, 46% নয়। তবে precise figure বের করতে actual accept rate log ব্যবহার করাই সঠিক।

এটি width-4 benchmark-এর গুরুত্ব আরও বাড়ায়। আগে দেওয়া width-4 throughput table-কে তাই illustrative sensitivity calculation হিসেবে দেখতে হবে—চূড়ান্ত upper bound হিসেবে নয়। Width-4 configuration-এ bonus-token convention অনুযায়ী ৪-এর বেশিও output tokens per verification step হতে পারে।

Source:

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler_components/metrics_reporter.py

NVIDIA-এর acceptance metric ব্যাখ্যাও একই পার্থক্য নিশ্চিত করে। 

NVIDIA NeMo AutoModel



৬. আপনার মূল hypothesis-গুলোর চূড়ান্ত validation
দাবি	Verdict	কারণ
TP4-এ communication overhead বড়	Strong hypothesis	CustomAllReduceV2 disabled — directly verified
TP4-এ GPU-to-GPU P2P নেই	Not proven	NCCL fallback PCIe P2P ব্যবহার করতে পারে
১২৮ all-reduce/step	Plausible estimate	Layer structure দিয়ে যুক্তিসংগত; profiler দিয়ে actual count হয়নি
Communication 40–50 ms/step	Unverified quantitative estimate	64 MB measurement ছোট collective-এর প্রতিনিধি নয়
GPU মাত্র 10–16 TFLOPS কাজে লাগাচ্ছে	Approximate equivalent compute rate	Full GPU utilization বা bottleneck identification নয়
GPU compute bottleneck নয়	Not established	INT4 dequantization, small GEMM, GDN ও kernel overhead এখনও সম্ভব
Context cost প্রায় 16–18 ms	Plausible lower-bound agreement	Theoretical KV bandwidth-এর সঙ্গে মেলে
DFlash2 width 8 অকার্যকর	Not established	Width-4/6 acceptance ও step latency দরকার
Width 4-এ 550–650 tok/s	Possible, not verified	Acceptance এবং actual verify time নির্ণায়ক
2×TP2 faster হতে পারে	Worth testing	Communication advantage সম্ভব, কিন্তু memory risk আছে
শীর্ষ Kaggle দলগুলো একই optimization করেছে	No supporting evidence found	Public leaderboard result থেকে harness architecture জানা যায় না

Communication estimate-এর নির্দিষ্ট সীমাবদ্ধতা
ধরি TP4-এর প্রতিটি all-reduce-এ 96 verification positions, hidden size 5,120, BF16 2 bytes। তাহলে একটি collective-এর payload প্রায় 0.98 MB।
\[
128\times96\times5120\times2
\approx126\ \mathrm{MB}
\]

আপনার 64 MB collective benchmark-এর 3.06 GB/s-কে সরাসরি প্রয়োগ করলে প্রায় 41 ms পাওয়া যায়। হিসাবটি arithmetically consistent। কিন্তু physical latency measurement নয়।

NCCL নিজেই বলে small-message all-reduce latency এবং large-message bandwidth আলাদা regime। আর 4-GPU ring-এর effective busbw বনাম সাধারণ size/time এক নয়। 

GitHub




একটি স্পষ্ট সংশোধন: PCIe Gen4 x16-এর প্রায় 64 GB/s হলো দুই দিক মিলিয়ে theoretical transfer rate; একদিকে প্রায় 31.5 GB/s। সেটির সঙ্গেও 3 GB/s collective bandwidth সরাসরি তুলনা করা যাবে না, কারণ all-reduce-এ একাধিক rank ও transfers রয়েছে।
TFLOPS estimate
২৭ বিলিয়ন parameters এবং প্রতি verification step-এ ৯৬ target tokens ধরে আনুমানিক:
\[
\frac{2\times27\text{B}\times96}
{4\times0.115}\approx11.3\ \mathrm{TFLOPS/GPU}
\]

এটি dense-equivalent target-model work rate। এতে drafter, attention, data movement, kernel launch এবং dequantization-এর সব খরচ যথাযথভাবে ধরা নেই।

L4-এর 121 TFLOPS BF16 theoretical peak-এর সঙ্গে এটিকে তুলনা করে compute idle—এমন সিদ্ধান্ত নেওয়া যাবে না। NVIDIA-এর hardware specifications-এ L4-এর 24 GB memory, 300 GB/s bandwidth এবং dense BF16 121 TFLOPS নিশ্চিত। 

৭. Research literature আমাদের কী বলছে?
এখানে কয়েকটি সবচেয়ে প্রাসঙ্গিক, যাচাই করা source একত্র করলাম।
Research/source	আমাদের জন্য evidence
MLSys 2026 — Speculative Decoding: Performance or Illusion?	Production-grade evaluation-এ target verification অনেক সময় প্রধান খরচ; acceptance position, task ও batching অনুযায়ী বদলায়। 

MLSys Proceedings
DFlash2 model/config	Original block size 8; serving stack ও checkpoint configuration একসঙ্গে পরীক্ষা করতে হবে। 

incoai/Qwen3.8-27B-DFlash2 at main
Hugging Face DFlash2 discussion	Width কমানোর বাস্তব অভিজ্ঞতা আছে, কিন্তু আমাদের L4 TP4 environment-এর জন্য সরাসরি evidence নয়। 

Hugging Face
SGLang CustomAllReduceV2 code, pinned v0.5.21	২টির বেশি PCIe-only GPU-এ optimized custom path disabled; TP2 hypothesis-এর সবচেয়ে শক্তিশালী code evidence। 

GitHub
NVIDIA NCCL tests	8KB–64MB collective benchmark ও algorithm/bus bandwidth ব্যাখ্যা। 

GitHub
DFlash2 published model benchmark	H200-এ performance improvement আছে; কিন্তু batch scaling ও hardware dependency থাকায় L4 projection করা যায় না। 

z-lab/Qwen3.8-27B-DFlash2 at main
NVIDIA profiler/Nsight	GPU kernel durations, communication timeline ও overlapping work যাচাইয়ের উপযুক্ত উপায়। 

GitHub

প্রাসঙ্গিক মূল URLs:
https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html

https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py

https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9

https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json

https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf

৮. সবচেয়ে কম GPU quota খরচ করে সিদ্ধান্ত নেওয়ার benchmark design
আমার প্রস্তাব একাধিক আলাদা exploratory run নয়; একটি gated engineering benchmark, যাতে প্রতিটি ধাপের ফল পরের ধাপের প্রয়োজন নির্ধারণ করে।

Gate A — GPU connectivity এবং NCCL microbenchmark

প্রথমে topology ও P2P matrix সংগ্রহ:
nvidia-smi topo -m
nvidia-smi topo -p2p p


তারপর NCCL tests থাকলে:
./build/all_reduce_perf -b 8K -e 64M -f 2 -g 4


একই benchmark প্রতিটি candidate TP2 pair-এ চালাতে হবে। Per-size latency, algbw, busbw এবং GPU pair identity সংরক্ষণ করতে হবে।

Decision: TP2 pair P2P সমর্থন করে কি না এবং TP4-তে actual small-message cost কত, সেটা এখানে নির্ধারিত হবে।




Gate B — Existing TP4 + width8 baseline profiling

১২ concurrent requests, warm context, fixed output budget। CUDA graph enabled রেখেই representative decode interval profile করুন।

Kernel durations আলাদা করুন: target GEMM/dequant, GDN, attention, all-reduce, draft network, sampling এবং scheduler gaps।

Profiler-এর overhead থাকায় profiled throughput এবং ordinary steady-state throughput আলাদাভাবে রিপোর্ট করবেন।

Decision: NCCL সত্যিই 40–50ms খাচ্ছে, নাকি target GEMM/dequant বা drafter বড় অংশ নিচ্ছে—এখানেই জানা যাবে।




Gate C — Draft widths 8, 6, 4

Same ১২ requests, same prompt/context distributions, same serving settings। প্রতিটি width-এ server initialization ও CUDA graph capture নতুন করে করতে হবে।

Measure: mean/median accept length, draft-only accept rate, p50/p90 step time, generated tok/s, prefill+decode wall time, GPU memory এবং tool-call correctness।

Decision: ৬ বা ৪-এর measured effective throughput যদি baseline থেকে যথেষ্ট বাড়ে এবং correctness ঠিক থাকে, তখনই narrower width গ্রহণ করুন। প্রথম পরীক্ষায় width 3 অপরিহার্য নয়।




Gate D — Optional 2×TP2, only after memory preflight

দুই pair-এ target + drafter + CUDA graphs + required KV cache fit করে কি না আগে যাচাই করুন।

Fit হলে একই মোট ১২ requests দুটি replica-তে ছয়টি করে চালিয়ে TP4 baseline-এর সঙ্গে comparison করুন।

Decision: শুধু aggregate tok/s নয়, 50k context support, cache retention, startup reliability এবং OOM risk-ও PASS করতে হবে।



Benchmark result sheet
পরবর্তী ফল প্রকাশের জন্য অন্তত এই fields চাই:
Configuration	Generated tok/s	Accept len	Draft accept rate	Step p50/p90	NCCL ms/step	Target ms/step	KV headroom
TP4 × width8	Baseline	Measure	Measure	Measure	Profile	Profile	Measure
TP4 × width6	Measure	Measure	Measure	Measure	Optional	Optional	Measure
TP4 × width4	Measure	Measure	Measure	Measure	Optional	Optional	Measure
2×TP2 × best width	Optional	Measure	Measure	Measure	Profile	Profile	Measure

একই input distribution না থাকলে throughput তুলনাটি invalid হবে। বিশেষত context 2k বনাম 50k, 6 বনাম 12 concurrency এবং cached বনাম uncached prefill একসঙ্গে মিশিয়ে performance ranking করা যাবে না।
৯. GPU quota ও time-risk: একটি সংশোধন
আপনার দেওয়া ৩০–৪০ মিনিট wall-clock এবং ১–১.৫ ঘণ্টা quota-এর সম্পর্কটি এখনও নিশ্চিত নয়।

যদি accounting GPU-hours হিসেবে চারটি GPU-র সময় যোগ করে, তাহলে ৩০–৪০ মিনিট wall-clock-এর অর্থ ২.০–২.৬৭ GPU-hours হতে পারে। Kaggle-এর নির্দিষ্ট competition quota কীভাবে charge হয়, সেটি dashboard-এর before/after usage দিয়ে মাপতে হবে; অনুমান করে GPU budget বরাদ্দ করা ঠিক হবে না।

আর cold startup-এর একটি historical run-এ target CUDA graph capture-ই 372.5 seconds নিয়েছিল। ফলে width8, width6, width4 এবং TP2—চারটি cold-start configuration-কে একটি নির্দিষ্ট ৩০–৪০ মিনিটে শেষ করা যাবে, এমন প্রতিশ্রুতি সমর্থিত নয়।

এই কারণে benchmark script-এ wall-clock hard stop, GPU quota guard, completed-measurements persistence এবং optional gates skip করার ব্যবস্থা থাকা প্রয়োজন। Full solver চালানোর প্রয়োজন নেই।

১০. চূড়ান্ত সিদ্ধান্ত
TP4 communication profiling
Highest priority

Draft width 8/6/4 benchmark
Strong second priority

2×TP2 production switch
Not yet justified


বর্তমান evidence অনুযায়ী TP4 communication overhead একটি বাস্তব ও গুরুত্বপূর্ণ সন্দেহ। কিন্তু all-reduce ঠিক 40–50ms নিচ্ছে, কিংবা width4 বেছে নিলেই 600 tok/s পাওয়া যাবে—এ দুটি দাবি এখনও অনুমান।

পরবর্তী GPU run-এর আগে সবচেয়ে মূল্যবান কাজ হলো profiler-ready benchmark-এর CPU/static validation সম্পূর্ণ করা, তারপর নির্দিষ্ট quota সীমার মধ্যে একবার measured experiment চালানো। Solver architecture বদলানো, DFlash2 rewrite, penalty patch অথবা TP2 deployment—কোনোটিই এই মুহূর্তে mandatory নয়।

আর শেষ পার্থক্যটি গুরুত্বপূর্ণ: engine throughput 500–600 tok/s হলেও ARC-AGI-2 score স্বয়ংক্রিয়ভাবে বাড়বে না। অতিরিক্ত compute কত দ্রুত correct transform() ও verified candidate তৈরি করছে, সেটি Rev 6f-এর candidate-production metrics-এর সঙ্গে মিলিয়েই প্রকৃত লাভ বিচার করতে হবে।
