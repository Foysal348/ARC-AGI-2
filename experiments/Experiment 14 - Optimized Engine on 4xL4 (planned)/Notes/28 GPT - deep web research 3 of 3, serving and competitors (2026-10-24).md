<!-- Pasted by the user on 2026-10-24 (GPT deep web research). Kept VERBATIM. Claude's validation: Notes/29. -->

ARC-AGI-2 2026 — Deep Web Research
তথ্যসীমা: ২৪ অক্টোবর ২০২৬ · Public evidence only · আপনার 4×L4 / SGLang 0.5.21 / DFlash2 setup কেন্দ্র করে
প্রথম গুরুত্বপূর্ণ সিদ্ধান্ত
আপনার 4×L4 setup-এর 355–460 decode tok/s-কে অস্বাভাবিক ধীর প্রমাণ করে—এমন কোনো একই hardware, একই model, 12 concurrent request ও 30k–100k context-এর reproducible public benchmark এখনো পাওয়া যায়নি।

ইতোমধ্যে পাওয়া তুলনাযোগ্য গবেষণাগুলো থেকে তিনটি বিষয় পরিষ্কার:
1. Speculative decoding-এর সেরা পদ্ধতি hardware ও concurrency অনুযায়ী বদলায়। এক RTX 5090-এ MTP-এর 1.81× gain কিংবা এক RTX PRO 6000-এ DFlash2-এর 2.26× gain সরাসরি আপনার TP4 12-stream environment-এ প্রযোজ্য নয়।
2. Hybrid GDN + DFlash correctness এখনো একটি সক্রিয় সমস্যা। সেপ্টেম্বর–অক্টোবর ২০২৬-এর একাধিক SGLang issue-তে thinking-mode divergence, concurrency-state corruption ও zero acceptance-এর রিপোর্ট রয়েছে।
3. একটি নতুন মডেল বা kernel benchmark-এ দ্রুত হলেই ARC score বাড়বে না। আপনার 60k-token agent workload-এ prefix reuse, useful Python calls, candidate quality এবং full-task coverage আলাদাভাবে গুরুত্বপূর্ণ।
একটি প্রকাশিত benchmark-এ Qwen3.8-27B-এর built-in MTP, RTX 5090 / GGUF / llama.cpp-তে 73.6→133.6 tok/s করেছে (n=3)। অন্যদিকে DFlash2, RTX PRO 6000-তে 100টি coding problem-এ 67.97→153.91 tok/s করেছে। দুটোই measured user reports—কিন্তু কোনোটিই TP4 L4 high-batch comparison নয়। 

huggingface.co
+1



১. Serving benchmarks — বাস্তবে প্রকাশিত সংখ্যাগুলো
এখানে Measured report বলতে লেখক hardware-এ benchmark চালিয়ে সংখ্যা প্রকাশ করেছেন; আমি স্বাধীনভাবে সেই GPU benchmark পুনরায় চালাইনি। আপনার hardware থেকে ফল আলাদা হলে সেটি contradiction নয়, যতক্ষণ setup identical না হয়।
লেখক / তারিখ	Hardware, software	Aggregate speed	Evidence	আপনার setup-এ প্রযোজ্যতা
halt95, সেপ্টেম্বর–অক্টোবর ২০২৬	4×3090, vLLM 0.30, INT4 lm_head/MTP, FP8 KV, TP4	C8: 701.9, C32: 1,120.9 tok/s	Published benchmark + repo	অত্যন্ত গুরুত্বপূর্ণ architecture lead; কিন্তু P2P-enabled, short prompts
Mr_Moonsilver, 17 আগস্ট	4×3090, vLLM 0.20.1, AWQ INT4, FP8 KV	TP4 C12: 319.9; TP2 C12: 428.0	User benchmark	TP split মূল্যায়নে গুরুত্বপূর্ণ; NVLink pairs ছিল
tonyd2wild, আগস্ট–অক্টোবর	4×3090, vLLM 0.27.1 + DFlash2 n=7, TP4, FP8 KV	C2–8: 145–147 tok/s prose	Reproducible recipe	DFlash2 সবসময় দ্রুত নয়—এমন শক্ত counterexample
xirang-husir, 20 আগস্ট	4×3090, vLLM, BF16 TP4 + MTP-3	C8: 367.5 tok/s	Published benchmark data	আপনার measured range-এর কাছাকাছি
xjx546, 19 আগস্ট	2×3090, PCIe Gen4 P2P, vLLM 0.26.1rc1 + INT4/DFlash2	Code: 218.3 tok/s, single stream	User benchmark	Fast single stream; 12-stream throughput নয়
engchina, 20 আগস্ট	4×4090, WSL2, vLLM + MTP/DFlash2	Single: 29→58–59 tok/s	Japanese benchmark	Ada generation হলেও stack ও workload আলাদা
Inco AI, আগস্ট ২০২৬	1×H200, SGLang + DFlash2, C8	HumanEval: 1,291.5 tok/s, GSM8K: 1,328.7	Model-author benchmark	H200 result; L4 projection হিসেবে ব্যবহারযোগ্য নয়

Sources: 

halt95/Qwen3.8-27B-W4A16-Merlin at main
+6




03507001,0501,400Your 4×L4 C124×3090 BF16 C84×3090 TP2 C124×3090 Merlin C84×3090 Merlin C32H200 DFlash2 C8









ভিন্ন hardware ও benchmark protocol; এটি ranking of engines নয়।





সবচেয়ে গুরুত্বপূর্ণ repository: Merlin
halt95/qwen38-27b-3090s, v1.0.0, vLLM 0.30.0 + পাঁচটি Python patch।

এর recipe-তে আছে:
- lm_head এবং built-in MTP head-এ INT4 GPTQ।
- Full-attention KV cache-এ calibrated FP8 E4M3 scales।
- Token embedding table pinned host RAM-এ রেখে VRAM বাঁচানো।
- TP2 ও TP4 দুটো configuration।
- Multi-concurrency benchmark cards এবং GSM8K-1319 paired evaluation।
Published configuration-এ 4×3090 TP4-তে প্রায় 1.95 million token KV pool এবং 32 stream-এ 1,120.9 tok/s। একই লেখক সতর্ক করেছেন, non-P2P hosts ও sm_89-এ তাঁর measurements করা হয়নি। আরও গুরুত্বপূর্ণ, তাঁর prefix caching-ও আগের generated assistant reply পুরোপুরি reuse করে না—যা আপনার re-prefill observation-এর সঙ্গে মিলে যায়। 

halt95/Qwen3.8-27B-W4A16-Merlin at main




https://github.com/halt95/qwen38-27b-3090s

https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin

গবেষণার অর্থ: অন্য serving stack-এ বড় throughput সম্ভব, কিন্তু সেটি সম্ভবত একই সঙ্গে quantization, GPU communication, KV headroom, workload length ও scheduling-এর ফল। আপনার পরিবেশে vLLM 0.30-এ সরাসরি switch করার সুপারিশ করার মতো head-to-head evidence নেই।
২. TP4 বনাম TP2×2 বনাম PP2/PP4
একই hardware-এ পরীক্ষার সবচেয়ে উপযোগী public comparison এসেছে Reddit-এর ১৭ আগস্ট ২০২৬-এর 4×3090 benchmark থেকে।

লেখক: Mr_Moonsilver · vLLM 0.20.1 · AWQ INT4, FP8 KV · 1,024 input / 512 output tokens · দুটি NVLink-paired 3090

TP4 aggregateTP2 aggregate

0 tok/s200 tok/s400 tok/s600 tok/s800 tok/s124812162432














Source-এর TP2 run একটি TP2 configuration-এর reported throughput; chart-এ সেটিকে দুই replica-এর মোট throughput হিসেবে দ্বিগুণ করা হয়নি। 

reddit.com


ব্যাখ্যা: ছোট concurrency-তে TP4 এগিয়ে, কিন্তু C12-এর পরে TP2 এগিয়েছে। এটি memory trade-off বা অন্য PCIe topology-তে একই ফল হবে, এমন প্রমাণ নয়।

আপনার ক্ষেত্রে সিদ্ধান্ত আরও কঠিন:
Design	সুবিধা	আপনার 4×24GB L4-তে সমস্যা	সিদ্ধান্ত
TP4	বড় unified KV pool; 12টি দীর্ঘ context চালানোর ক্ষমতা	140টি collective/step; PHB transport	বর্তমান baseline রাখুন
2×TP2	কম TP communication; independent engines	প্রতি replica-তে weight দ্বিগুণ, KV pool ছোট; long-context residency সীমিত	শুধু fit smoke test, এখন primary নয়
PP2/PP4	প্রতিটি layer-এর all-reduce কমতে পারে	Pipeline bubbles, stage synchronization, hybrid/DFlash compatibility	পূর্ণ benchmark ছাড়া production নয়
Full DP4	TP communication নেই	27B INT4+drafter+runtime memory এক L4-তে পর্যাপ্ত নয়	বর্তমান model-এ কার্যকর নয়

Pipeline parallel: ২০২৬ সালের সেপ্টেম্বরের SGLang development issue-তে hybrid Qwen3.8, PP এবং speculation-এর compatibility নিয়ে কাজ পাওয়া গেছে। কিন্তু 4×L4, 30k–100k context, DFlash2, PP2/PP4 বনাম TP4-এর measured comparison পাওয়া যায়নি। 

GitHub




আরেকটি benchmark-এ 2×4090D 48GB-এ vLLM দিয়ে single GPU বনাম TP2 বনাম PP2 দেখা হয়েছে: concurrency 4-এ যথাক্রমে 128.1, 163.4, 119.9 tok/s। সেখানে PP2 decode throughput-এ পিছিয়েছে। ফলে “PP communication কমায়, তাই অবশ্যই দ্রুত”—দাবিটি সমর্থিত নয়। 

Houtini




আপনার 2×TP2 সম্পর্কে: প্রতি replica আনুমানিক 170k KV ধরে—এটি আপনার দেওয়া sizing analysis। Public 3090 benchmark এই L4 memory limitation খণ্ডন করে না। তাই আগের TP2 recommendation-কে low-priority exploratory test-এ নামানোই যুক্তিযুক্ত।
৩. MTP বনাম DFlash2 বনাম DSpark/EAGLE-3
নতুন বড় উৎস: SGLang-এর official Qwen3.8 serving cookbook
SGLang-এর প্রকাশিত Qwen3.8-27B cookbook-এ ২০২টি measured evaluation configuration বর্ণিত আছে, v0.5.19-এ RTX 5090, RTX PRO 6000 ও DGX Spark, বিভিন্ন speculation ও GDN dtype setting নিয়ে। GSM8K-এর 1,319টি প্রশ্ন ব্যবহার করে accuracy-ও যাচাই করা হয়েছে।

এটি শক্তিশালী reproducibility resource, কিন্তু serving sweep-এর throughput/acceptance measurements এবং GSM8K accuracy measurements একই protocol নয়। বিশেষ করে cookbook-এর অনেক performance result concurrency 1-এ। 

GitHub




https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
২০২৬-এর গুরুত্বপূর্ণ speculative-decoding evidence
উৎস ও সময়	Hardware / workload	ফলাফল	আপনার জন্য অর্থ
Inco AI, আগস্ট ২০২৬	H200, C8, SGLang, HumanEval	AR 483.4; MTP 934.2; DSpark 956.5; DFlash2 1,291.5 tok/s	DFlash2 high-batch-এ ভালো হতে পারে, কিন্তু H200
Inco AI, আগস্ট	H200, C8, GSM8K	AR 467.2; MTP 1,022.1; DSpark 1,040.8; DFlash2 1,328.7	Workload-specific
Mr_Moonsilver, আগস্ট	4×3090, C12, TP4	No-spec 319.9; MTP3 288.8 tok/s	Speculation-এ regress হতে পারে
z-lab community, আগস্ট	llama.cpp, single GPU, দীর্ঘ context	Draft width 4: 56.96; width 7: 54.94 tok/s	ছোট width কখনো ভালো, high-batch প্রমাণ নয়
Paolo Rosson, সেপ্টেম্বর	DGX Spark, SGLang, DFlash2 width 16	C16: 360; C32: 387 aggregate tok/s	Wider draft acceptance বাড়ালেও total speed সীমিত
aphex3k, সেপ্টেম্বর–অক্টোবর	DGX Spark, NVFP4 + DFlash2	Peak C16: 480.7 tok/s at width 8	Width optimum workload অনুযায়ী বদলায়



Hugging Face
+4


DFlash2 width-এর মূল trade-off হচ্ছে accepted tokens per step বনাম verification rows-এর খরচ। আপনার 8-width-এ 12×8=96 target rows চলছে; width 4 করলে 48 rows হবে, কিন্তু acceptance-ও কমবে। উপরন্তু checkpoint-এর block attention bidirectional; width 4 কেবল width 8-এর শেষ চারটি token বাদ দেওয়া নয়।

বর্তমান evidence অনুযায়ী 6 এবং 4—উভয় width-এর net outcome অনিশ্চিত। আপনার নিজস্ব 4×L4 benchmark-ই এখানে সিদ্ধান্তমূলক হবে।
তিনটি correctness warning
1. SGLang #36548: concurrent DFlash2-তে state/output inconsistency-এর public report।
2. SGLang #38009: thinking-mode-এ target-only ও speculative greedy output divergence-এর report।
3. SGLang #37475: checkpoint-এর target_layer_ids ভুল বা অনুপস্থিত হলে nonmatching target hidden layers ব্যবহার হওয়ার ঝুঁকি।
এগুলো bug reports, আপনার run-এর confirmed defects নয়। কিন্তু optimization করার আগে fixed prompt set-এ serial ও concurrent consistency test নেওয়া প্রয়োজন। 

GitHub
+2




EAGLE-3 সম্পর্কে: আপনার exact Qwen3.8-27B / 4×L4 / long-context / C12 configuration-এ EAGLE-3 বনাম DFlash2-এর public measured benchmark পাওয়া যায়নি। অন্য model-এর EAGLE-3 speedup দিয়ে এখানে সুপারিশ করা ঠিক হবে না।
৪. INT4, FP8, GDN state এবং prefix cache
Quantization: কোন পথে বাস্তব evidence আছে?
Format / implementation	Public evidence	4×L4 recommendation
RedHat W4A16 INT4 + Marlin	আপনার measured working baseline	অপরিবর্তিত রাখুন
Merlin W4A16 GPTQ + INT4 lm_head/MTP	halt95-এর 4×3090 benchmark, C32 1,120.9 tok/s	আলাদা research arm; direct replacement নয়
AWQ INT4	4×3090 benchmark-এ TP4 C12 319.9 tok/s	বর্তমান INT4-এর চেয়ে ভালো—প্রমাণ নেই
FP8 W8A8	H200-তে benchmark আছে	4×24GB memory headroom কমার ঝুঁকি; এখন নয়
FP8 E4M3 KV	বহু reproducible recipes, আপনার baseline-এও ব্যবহৃত	রাখুন
INT8/FP8 SSM state	Hybrid model-এ memory-saving experiment	Accuracy ও state-cache equivalence gate ছাড়া নয়

Merlin-এর বড় সুবিধা সম্ভবত শুধু quantization speed নয়; quantized output head, smaller weight footprint, KV capacity এবং concurrent residency-ও। ফলে ঐ benchmark থেকে “Marlin-এর পরিবর্তে Merlin দিলেই 2× throughput” সিদ্ধান্ত নেওয়া যাবে না। 

halt95/Qwen3.8-27B-W4A16-Merlin at main
+2



SGLang cookbook থেকে একটি গুরুত্বপূর্ণ memory-sizing rule
Hybrid GDN model-এ শুধু --mem-fraction-static বাড়ালে concurrency বাড়বেই না। GDN state pool এবং attention KV pool-এর ভারসাম্য প্রয়োজন।

Cookbook-এর relevant quantities:
- extra_buffer: সাধারণত প্রতি active request-এ 5টি SSM state slot।
- extra_buffer_lazy: প্রায় 4টি।
- no_buffer: প্রায় 3টি।
- BF16 state: প্রায় 78.4 MB/slot।
- FP8 attention KV: প্রায় 32.8 KB/token।
Official calculator --mamba-full-memory-ratio এবং --max-mamba-cache-size মডেল করে। কিন্তু এগুলো বর্তমান 12-running setup-এ যথাযথভাবে configured কি না, তা boot log-এর resolved pool sizes দিয়ে যাচাই করতে হবে। 

GitHub




আপনার কাছে KV pool যথেষ্ট থাকলে blind SSM-memory reduction লাভের বদলে state correctness problem তৈরি করতে পারে।
Prefix reuse: সবচেয়ে মূল্যবান agent-side opportunity
আপনার historical measurements অনুযায়ী Duck run-এ 13–20%, কিছু ক্ষেত্রে 55%, সময় re-prefill-এ গেছে। এই observation-এর সঙ্গে Daniel Franzen-এর ARC-AGI-3 write-up-এর বক্তব্য মেলে: generated assistant response পুনরায় chat template দিয়ে tokenize করলে token sequence বদলে যেতে পারে এবং cache reuse নষ্ট হতে পারে। 

GitHub




কিন্তু কারণ এখনো নির্দিষ্টভাবে প্রমাণিত নয়। অন্তত তিনটি সম্ভাবনা রয়েছে:
- Generated token IDs এবং পরবর্তী request-এর re-rendered history-এর mismatch;
- GDN/Mamba checkpoint eviction বা state tracking;
- Context trimming, different messages অথবা cache-capacity changes।
আপনার জন্য সঠিক next step হলো already planned CPU-side prompt reconstruction এবং প্রথম token divergence শনাক্ত করা। Historical raw generated token IDs না থাকলে reconstructed mismatch ও exact generated-ID mismatch আলাদা করে report করতে হবে।
HiCache কেন এখন নিম্ন অগ্রাধিকার
সেপ্টেম্বরের SGLang #39830-এ hybrid model-এর host-tier cached state থেকে ভুল output-এর report রয়েছে। ২০ অক্টোবরের #45400-এ Qwen3.8-27B hybrid workload-এ কিছু দীর্ঘ prefix-এর পরে host-tier cache reuse ভেঙে পড়ার report আছে। 

GitHub
+1




এগুলো অন্য environment-এর reports, কিন্তু আপনার GDN + DFlash2 stack-এর সঙ্গে প্রাসঙ্গিক ঝুঁকি। বর্তমানে KV residency যদি প্রকৃত bottleneck না হয়, তাহলে HiCache দিয়ে root cause সমাধান হওয়ার শক্ত কারণ নেই।

Verdict: extra_buffer_lazy ও বর্তমান FP8 KV অক্ষত রেখে প্রথমে cache-miss classification করুন; HiCache migration নয়।
৫. PCIe/PHB communication: আপনার নতুন measurement সিদ্ধান্ত বদলে দিয়েছে
আপনি এখন জানাচ্ছেন:
- সব GPU pair PHB।
- NCCL P2P/IPC path SHM-এর তুলনায় 5.5× ধীর।
- TP4 collective প্রায় 56 ms/step।
এই নতুন measurement বিশ্বাসযোগ্য benchmark protocol-এ পাওয়া হয়ে থাকলে P2P forcing-কে আর primary optimization হিসেবে রাখা যাবে না।

Public evidence-এ অনুরূপ topology-তে ভিন্ন ফল পাওয়া গেছে। উদাহরণ: HyperQwen-এর 4×RTX 5060 Ti PCIe study-তে NCCL_P2P_LEVEL=PHB forcing কিছু concurrency-তে 2–16% decode gain দিয়েছে। এটি দেখায় topology/driver-dependent outcome কতটা ভিন্ন হতে পারে। আপনার বিপরীত ফলের সঙ্গে তার কোনো logical contradiction নেই। 

GitHub




https://github.com/syv-ai/HyperQwen/issues/254
Communication techniques-এর বর্তমান রায়
Technique	Public evidence	আপনার 4×L4 সিদ্ধান্ত
NCCL default SHM	আপনার fastest measured transport	Baseline রাখুন
NCCL_P2P_LEVEL=PHB/SYS	অন্য PHB machines-এ কখনো লাভ	আপনার 5.5× slowdown থাকলে adopt নয়
NCCL_SHM_USE_CUDA_MEMCPY=1	কিছু PCIe setup-এ উন্নতি, অন্যগুলোতে regression	Optional bounded A/B
NCCL_ALGO=Tree/Ring, NCCL_PROTO	Size/topology-dependent	Exact 983,040-byte replay ছাড়া নয়
FlashInfer PCIe-IPC all-reduce	Newer implementation; hardware-dependent	P2P ইতিমধ্যে ধীর হলে upgrade-এর যুক্তি দুর্বল
FP8/quantized all-reduce	Potential lower communication volume	আপনার stack-এ verified safe implementation পাওয়া যায়নি
TP communication compression	Research possibility	Numerical quality ও implementation risk বেশি

SGLang-এর TP4 PCIe-only setup-এ optimized CustomAllReduce disabled হওয়া source-verified। তবে communication cost-এর বড় অংশ সত্যিই all-reduce—এই কথার decisive confirmation kernel trace। 

GitHub



SGLang 0.5.22: ১৯ অক্টোবরের release
এই research-এ একটি গুরুত্বপূর্ণ version update পাওয়া গেছে: SGLang v0.5.22, ১৯ অক্টোবর ২০২৬। এর release notes-এ hybrid caching ও speculative decoding সম্পর্কিত একাধিক changes আছে। 

GitHub




https://github.com/sgl-project/sglang/releases/tag/v0.5.22

কিন্তু তিনটি বিষয় আলাদা:
1. Upstream bug fix release হয়েছে।
2. আপনার exact model/runtime combination-এ fix কার্যকর।
3. সেই release আপনার W1-এর চেয়ে দ্রুত।
প্রথমটি কিছু ক্ষেত্রে verified; দ্বিতীয় ও তৃতীয়টির পরীক্ষা বাকি। আপনার pinned wheelhouse-এ FlashInfer, DFlash2 ও quantization dependencies আছে—তাই deadline-এর এক সপ্তাহ আগে blind full upgrade উচ্চ ঝুঁকির কাজ।

Recommendation: 0.5.22-কে optional parallel serving branch রাখুন; Kaggle production baseline বদলাবেন না যতক্ষণ না load, correctness, memory এবং throughput চারটি gate পাশ করছে।
৬. Startup: প্রথম launch 502 সেকেন্ড, relaunch 71 সেকেন্ড
প্রথম launch বনাম relaunch পার্থক্য:
Cold launch
502 s
আপনার measurement

Relaunch
71 s
আপনার measurement


প্রায় 431 সেকেন্ডের ব্যবধান, কিন্তু এর কতটা storage, filesystem cache, Triton JIT, FlashInfer, CUDA-graph capture বা torch initialization—এখনো আলাদা করে জানা যায়নি।
Public evidence
Google Cloud Storage FUSE: Google-এর documentation large-file parallel downloads ও local caching থেকে performance gain-এর কথা বলে। কিন্তু নির্দিষ্ট Kaggle environment-এ 16 বা 32 reader দ্রুত হবে—এমন measured evidence পাওয়া যায়নি। 

Google Cloud
+1




DGX Spark reproducibility: Paolo Rosson-এর সেপ্টেম্বর ২০২৬ repository-তে prepared FlashInfer autotune cache সহ cold boot প্রায় 210 সেকেন্ড এবং তার benchmark পুনরুৎপাদনের নির্দেশনা আছে। এটি JIT/autotune cache-এর গুরুত্ব দেখায়, যদিও GB10-এর সংখ্যা L4-তে প্রযোজ্য নয়। 

GitHub




Kaggle runtime accounting: Kaggle staff-এর অক্টোবরের public reply অনুযায়ী dataset download ও interactive quota হিসাবের কিছু সময় পরে সংশোধিত হতে পারে; live quota display চূড়ান্ত হিসাবের সমান নয়। এটি প্রকৃত Kaggle submission ও interactive session-এর policy অনুযায়ী যাচাই করতে হবে। 

The ARC Daily Digest



আপনার জন্য concrete startup order
প্রথমে load-path diagnostic-এ cold timings ভাগ করুন:

mount_read → staged_copy → tokenizer/model_load → JIT → graph_capture → first_ready_response

এরপর প্রমাণ-অনুযায়ী fix করুন। সব startup delay-কে GCSFUSE বা শুধু CUDA graph-এর সমস্যা বলবেন না।

/dev/shm ব্যবহার করলে host/cgroup free memory এবং copy integrity নিশ্চিত করা জরুরি। JIT caches-এ source hash, CUDA/Triton/torch/SGLang version ও GPU architecture যাচাই করতে হবে; mismatched cache ব্যবহার করা যাবে না।
৭. ARC agent ও harness গবেষণা
এই বিভাগে সরাসরি 4×L4 speed-এর চেয়ে competition score-এর জন্য গুরুত্বপূর্ণ public evidence আছে।
ARC Prize official Qwen result
ARC Prize-এর ১৪ আগস্ট প্রকাশিত model result এবং সেপ্টেম্বরের pricing snapshot অনুযায়ী Qwen3.8-27B-এর semi-private ARC-AGI-2 accuracy:
42.4%
XHigh

22.8%
Low

13.2%
Medium

1.5%
None


এটি dedicated model evaluation, Duck/Poetiq competition kernel নয়। তবু XHigh-কে পূর্ণমাত্রায় বাতিল করে Low-only coverage নেওয়া কেন বিপজ্জনক, তার শক্ত evidence। 

ARC-AGI Results




https://arcprize.org/results/alibaba-qwen3-8-27b
Public harness designs থেকে কার্যকর শিক্ষা
উৎস	প্রকাশিত approach	আপনার জন্য প্রযোজ্যতা	Evidence
Daniel Franzen / Duck, ARC-AGI-3 write-up	Python tool REPL, retained state, context reuse, request scheduling	আপনার donor-এর সঙ্গে সরাসরি সম্পর্কিত	Public code/write-up
Poetiq, public repository	Iterative executable hypotheses, feedback, candidate refinement	Multi-attempt logic-এর ভালো reference	Public code; 4×L4 result নয়
Dyachenko, সেপ্টেম্বর ২০২৬ ARC research	Small Qwen-ভিত্তিক program seeds + evolutionary refinement	Candidate diversity, task-specific DSL-এর ধারণা	Small-sample experiment
ARCANA, জুলাই ২০২৬	Structured program synthesis ও agent reasoning	ARC hypothesis search-এর conceptual lead	Paper
Tufa/other leaders	Competition score public	তাদের ARC2 harness পুরোপুরি প্রকাশিত নয়	Method পাওয়া যায়নি



GitHub
+3


Daniel Franzen-এর open write-up:

https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md

সতর্কতা: এই repository ARC-AGI-3 solution-এর। Tufa Labs-এর 88.06 ARC-AGI-2 entry ঠিক একই harness ব্যবহার করেছে—এমন প্রমাণ পাওয়া যায়নি।
Harness-এর জন্য আলাদা ranked ideas
1. প্রথম executable transform/check আগে আনুন। আপনার দীর্ঘ চিন্তার মধ্যে প্রথম check() দেরিতে হওয়ার observation-এর সঙ্গে code-first prompting মেলে। তবে fixed short thought cap যেন late-correct refinement বন্ধ না করে। প্রত্যাশিত score gain অজানা।
2. Progress-aware compute scheduling। প্রথমে সব 240 task-কে exposure, পরে verified-progress অনুযায়ী continuation। একেবারে অনুৎপাদনশীল branch-এ unlimited reasoning না দিয়ে bounded chunks। এর gain offline replay-তে যাচাই করতে হবে।
3. Pass@2 candidate diversity বজায় রাখা। দুইটি distinct train-exact test prediction থাকলে উভয়কে রাখুন। Latest correct candidate বাদ দিয়ে majority vote করবেন না।
4. Prefix/token history preservation। Reasoning ও tool history-র exact representation যাচাই করে পুনরায় prefill কমানো। এখানে সম্ভাব্য throughput benefit আছে, কিন্তু root cause নিশ্চিত করা বাকি।
5. Task-wise budget allocation। মাত্র runtime speed নয়, প্রতি GPU-hour-এ নতুন correct task/input মাপুন। Easy task-এ early stop এবং কঠিন task-এ continuation—এই trade-off evidence দিয়ে ঠিক করুন।
6. Poetiq ও Duck-এর complementary predictions। একই input-এ দুটো agent ব্যবহারের লাভ নির্ধারণ করুন তাদের unique solves দিয়ে; model count বা average confidence দিয়ে নয়।
7. Small-model distillation বা ARC-tuned model। Public concept আছে, কিন্তু আপনার 4×L4 offline submission-এ 70%+ প্রদর্শিত নির্ভরযোগ্য distilled Qwen3.8-27B model পাওয়া যায়নি। তাই near-deadline প্রধান পরিকল্পনা নয়।
৮. শীর্ষ ১২ প্রতিযোগী — public footprint audit
২৪ অক্টোবরের জন্য পাওয়া সবচেয়ে নির্দিষ্ট dated leaderboard snapshot হলো ২৩ অক্টোবর ২০২৬, সকাল ৯:০১ EDT। এতে প্রথম দশের score পাওয়া গেছে। User-provided তালিকার সঙ্গে কিছু historical পার্থক্য রয়েছে—বিশেষ করে ALbert RUehlman এবং Japanese team name। 

ARC Explainer




Public scoreboard ≠ public technique. কোনো team 80%+ করেছে মানেই তারা নির্দিষ্ট model, Duck, NVARC বা distillation ব্যবহার করেছে—এমন সিদ্ধান্ত নেওয়া যায় না।
প্রতিযোগী	Published score / date	Public method-related evidence
Tufa Labs	88.06 · 23 Oct	প্রতিযোগিতা শেষে পদ্ধতি জানানোর কথা বলেছেন। ARC2 solver recipe পাওয়া যায়নি।
Yi-Chia Chen	84.72 · 23 Oct	ARC2 ও ARC3-তে উচ্চ score; ARC2 model/harness implementation পাওয়া যায়নি।
nvbanana	81.39 · 23 Oct	Public team/leaderboard footprint পাওয়া গেছে; ARC2 ensemble composition পাওয়া যায়নি।
rabbithole	80.83 · 23 Oct	Team roster: cookizesong, xiezejian, yuhuake240506। Method পাওয়া যায়নি।
Majkel1337	75.56 · 23 Oct	ARC2/ARC3 competition presence; ARC2 recipe পাওয়া যায়নি।
ALbert RUehlman	72.64 · 23 Oct	আপনার 75.56 সংখ্যার সঙ্গে মেলে না। Source-specific latest dated value 72.64; method পাওয়া যায়নি।
Dawid Kopiczko	75.14 · 23 Oct	Public Kaggle identity; exact model/notebook পাওয়া যায়নি।
Domdolus Tolus	73.47 · 23 Oct	Public score; technique পাওয়া যায়নি।
ｶﾎｳｼｭｳｾｲ	72.50 · 23 Oct	Officially captured team string এটি; আপনার লেখা カトウシュウセイ-এর সমান কি না নিশ্চিত নয়। Method পাওয়া যায়নি।
_hans	71.81 · 23 Oct	Public roster/profile association deepdreamer; ARC2 solver detail পাওয়া যায়নি।
Youssef Nader	আপনার দাবি 70.56	২০ অক্টোবরের independent snapshot-এ 62.22; 70.56-এর dated primary confirmation পাওয়া যায়নি।
mikelou1	আপনার দাবি 69.72	Kaggle comments-এ Qwen3.8 family, no NVARC ensemble, ~66% score এবং kernel issue-এর কথা বলেছেন। 69.72-এর exact dated confirmation ও full recipe পাওয়া যায়নি।



ARC Explainer
+6


mikelou1 সবচেয়ে কার্যকর public lead। অক্টোবরের Kaggle discussion-এ তিনি Qwen-only approach, local validation ও kernel issue নিয়ে মন্তব্য করেছেন; NVARC combination ব্যবহার করছেন না বলেও জানিয়েছেন। কিন্তু তাঁর exact agent graph, model quantization, token budget ও scheduler প্রকাশিত হয়নি। তাই “Qwen3.8-27B একাই 69–70% করতে পারে”—এটিকে user claim/competition observation হিসেবে রাখা যায়, reproducible public method হিসেবে নয়। 

The ARC Daily Digest




Original competition discussion:

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546

আরেকটি গুরুত্বপূর্ণ inference: Tufa Labs-এর ARC2 result এবং Tufa/Duck-এর ARC3 repository একই জিনিস নয়। Public repo থেকে code শেখা বৈধ, কিন্তু private winning implementation সম্পর্কে অনুমান করা উচিত নয়।
৯. আপনার জন্য ranked concrete speed changes
নিচের gain percentages আপনার exact 4×L4 stack-এ measured নয়, যদি আলাদা করে না বলা থাকে। Estimate হলো আপনার measured bottleneck ও বাহ্যিক গবেষণা মিলিয়ে পরীক্ষার অগ্রাধিকার নির্ধারণের জন্য।
Rank	পরিবর্তন	Potential improvement	Evidence class	Risk / সিদ্ধান্ত
1	Prompt re-render / prefix reuse diagnosis এবং validated fix	Effective agent throughput-এ ~0–15%, কিছু pathological run-এ বেশি হতে পারে	আপনার log-derived model; external Duck evidence	Medium; CPU investigation আগে
2	#44401 scheduler admission fix	Overall ~0–8% সম্ভাব্য	Source fix + 11-running observation; unmeasured end-to-end gain	Low–medium; matched comparison
3	CUDA graph exact batch sizes 1..13	~0–8% hypothesis	Padding arithmetic; direct benchmark বাকি	Low–medium; memory ও capture time
4	Reduce cold startup / JIT cache / safe staging	Cold startup-এ 502→71 s gap-এর কিছু অংশ বাঁচানো সম্ভব	আপনার measured difference; public caching techniques	Medium; integrity ও portability
5	DFlash2 width 6 / 4	−10% থেকে +12% scenario	Acceptance model; single-GPU evidence	Medium; correctness gate আবশ্যক
6	Nodelay/admission occupancy tuning	Occupancy 11→12 হলে affected windows-এ প্রায় +9% output opportunity	Log arithmetic; actual overall gain অজানা	Medium; concurrency safety
7	SGLang 0.5.22 selective fixes	Unknown performance; সম্ভাব্য correctness improvement	Public release/source	High integration risk; separate branch
8	Merlin-style quantized head/MTP memory saving	Unknown; potentially more KV capacity	4×3090 short-context benchmark	High porting/quality risk
9	NCCL SHM environment micro-tuning	Unknown, দুদিকেই যেতে পারে	Mixed public hardware measurements	Low if isolated; measured winner ছাড়া adopt নয়
10	Vocab-parallel logits verification / GDN kernel tuning	Theoretical কয়েক ms/step	Static computational model, no matched benchmark	High correctness/kernel risk
11	2×TP2 বা PP2/PP4	Unknown, regress হতে পারে	Other-hardware comparisons; L4 memory problem	High; near-deadline primary নয়
12	P2P forcing / PCIe-IPC upgrade	আপনার measured setup-এ নেতিবাচক	আপনার 5.5× slower result	Deprioritize

সবচেয়ে গুরুত্বপূর্ণ বাস্তব পার্থক্য: Engine decode speed-এ +10% এলেও Duck-effective speed-এ +10% আসবে না, যদি re-prefill ও occupancy একই থাকে। আবার prompt reuse ঠিক করে engine kernel না বদলিয়েও agent-effective throughput বাড়ানো সম্ভব।
600–800 tok/s লক্ষ্য
আপনার current 355–460 decode tok/s থেকে 700–900 decode tok/s-এ যেতে হলে বড় multiple bottleneck removals দরকার। Public Merlin result প্রমাণ করে শক্তিশালী PCIe workstation-এ C8–C32-তে উচ্চ aggregate throughput সম্ভব; কিন্তু আপনার 12× long-context PHB/SHM load-এ সেই target অর্জিত—এমন প্রমাণ নেই।

600–800 agent-effective tok/s-এর verified route: পাওয়া যায়নি।
১০. Evidence gap — কী পাওয়া যায়নি
প্রায় ২০০টি targeted search-এ Kaggle, Hugging Face Discussions, GitHub, ARC Prize, Reddit, X-এর public references, YouTube, Qiita, Zhihu ও গবেষণাপত্র খুঁজেছি। নিচের প্রশ্নগুলোর শক্ত evidence পাওয়া যায়নি:
- 4×L4, sm_89, TP4, RedHat W4A16, DFlash2 width 8, FP8 KV, FP16 SSM, 12–16 stream ও 50k+ context—এই exact setup-এ independent benchmark।
- একই Kaggle 4×L4 environment-এ TP4 বনাম 2×TP2 বনাম PP4-এর controlled comparison।
- Qwen3.8-27B DFlash2 width 4/6/8-এর high-concurrency long-thinking paired measurement।
- 4×L4-তে FP8/quantized all-reduce-এর verified safe implementation ও positive speed result।
- আপনার 12টি named leaderboard team-এর সম্পূর্ণ ARC2 private solving recipe।
- Tufa 88.06-এর নির্দিষ্ট model/agent/compute allocation।
- mikelou1-এর সম্পূর্ণ Qwen-only 69%+ reproducible implementation।
- 2026-এর কোনো publicly reproduced 70%+ ARC2 fine-tuned/distilled Qwen3.8-27B pipeline যা আপনার 4×L4 offline constraints-এ fit করে।
এই gap-গুলো গুরুত্বপূর্ণ: missing evidence-কে negative result হিসেবে ধরা যাবে না; আবার অনুমান দিয়ে পূরণও করা যাবে না।
১১. মূল raw URLs — কপি করার উপযোগী
Serving ও quantization
https://github.com/halt95/qwen38-27b-3090s
https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin
https://www.reddit.com/r/LocalLLaMA/comments/1vr56r1/benchmarked_qwen3827b_on_4x_rtx_3090/
https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2
https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
https://github.com/aphex3k/Qwen3.8-27B-on-DGX-Spark
https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
https://github.com/sgl-project/sglang/issues/42024
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
https://arxiv.org/abs/2602.06036


SGLang, NCCL, correctness ও caching
https://github.com/sgl-project/sglang/releases/tag/v0.5.22
https://github.com/sgl-project/sglang/pull/44401
https://github.com/sgl-project/sglang/issues/36548
https://github.com/sgl-project/sglang/issues/38009
https://github.com/sgl-project/sglang/issues/37475
https://github.com/sgl-project/sglang/issues/43514
https://github.com/sgl-project/sglang/issues/39830
https://github.com/sgl-project/sglang/issues/45400
https://github.com/sgl-project/sglang/issues/28180
https://github.com/sgl-project/sglang/issues/36127
https://github.com/syv-ai/HyperQwen/issues/254
https://github.com/NVIDIA/nccl-tests
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
https://docs.flashinfer.ai/api/comm.html
https://github.com/flashinfer-ai/flashinfer/pull/4393
https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/performance


ARC Prize, Kaggle ও agents
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082
https://arcprize.org/results/alibaba-qwen3-8-27b
https://arc-explainer-production.up.railway.app/arc-leaderboards
https://arc-explainer-production.up.railway.app/news/community
https://github.com/da-fr/arc-agi-3-solution
https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md


চূড়ান্ত গবেষণা-ভিত্তিক সিদ্ধান্ত
আপনার setup-এ NCCL P2P-তে আর সময় খরচ করাকে আমি প্রধান অগ্রাধিকার দেব না, কারণ আপনার নিজের 5.5× regression external positive claims-এর চেয়ে অনেক বেশি প্রাসঙ্গিক।

এখন সবচেয়ে সম্ভাবনাময় ক্রম:

প্রথমে existing E14-A baseline ও correctness gate দিয়ে actual profiler measurement নিশ্চিত করা। তারপর #44401 ও graph batch coverage-এর bounded comparison। সমান্তরালে GPU ছাড়া token-history/prefix reconstruction। তারপর measured evidence দেখে width 6/4 অথবা state/cache fix নির্বাচন।

Merlin, TP2×2, PP4, PCIe-IPC, vLLM migration এবং custom logits/GDN kernels—এগুলো প্রযুক্তিগতভাবে আকর্ষণীয়, কিন্তু বর্তমানে সবচেয়ে কম ঝুঁকিতে leaderboard improvement পাওয়ার পথ নয়।

আর প্রতিযোগিতামূলক স্কোরের দিক থেকে সবচেয়ে বড় শিক্ষা হলো: সর্বোচ্চ tok/s নয়, সীমিত 12-hour budget-এ নতুন সঠিক ARC task কত পাওয়া যাচ্ছে—সেটিই প্রকৃত optimization target।
