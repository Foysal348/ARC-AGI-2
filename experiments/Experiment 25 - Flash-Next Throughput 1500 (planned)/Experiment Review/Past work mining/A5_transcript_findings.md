# A5: Throughput findings mined from the conversation transcripts

**Date:** 2026-10-31.
**Scope:** the main transcript `C:\Users\hp\.claude\projects\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f.jsonl` (625 MB, 142,2xx JSONL lines, 2026-09-15 to 2026-10-31 05:58Z) and the 297 subagent transcripts under `...\1c6db7aa-...\subagents\` (+ 1 workflow).
**Method:** a streaming Python scan pulled the text blocks of user and assistant messages (tool results skipped, keys redacted) into a 18.9 MB text file. That file was searched by keyword (tok/s, throughput, preempt, KV, concurrency, stream, MTP, acceptance, DFlash, speculative, prefix hit, cudagraph, PIECEWISE, NCCL, P2P, Flash-Next, qwen4_exp, Merlin, 1500, NVIDIA, গতি, স্পিড, কনকারেন্সি, স্ট্রিম, টার্গেট). For the subagents, the final hand-back of every agent that mentions Flash-Next or Merlin was read (159 files matched).
**Citation format:** `L<n>` is the line number in the main JSONL, followed by the UTC timestamp. Subagent citations give the agent id. "Folder status" says whether the fact was found in `experiments/**/*.md` (checked with ripgrep).
**Read-only:** no file was changed except this one. No secret was copied.

---

## 1. User directives on speed (date | quote / translation | meaning)

| Date (UTC) | Source | Quote (Bengali, abridged) / translation | Meaning for E25 |
|---|---|---|---|
| 2026-10-04 | L6530 | "কনকারেন্সি ১২ হলে ১২ টা টাস্ক দিও। জিপিউ নষ্ট করতে চ্যাঁই না" = "If concurrency is 12, give it 12 tasks. I don't want to waste GPU." | Every slot must be busy. Idle slots count as waste. |
| 2026-10-09 | L13185 | "১২ ঘন্টার ১০ মিনিট আগে থামবে এমন করে দিও" = "Make it stop 10 minutes before 12 h." | Run budget ends at 11 h 50 min (later memory says ~11.5 h usable). |
| 2026-10-12 | L21666 | Quoting Claude "normally 16 run, up to 18 if KV allows": "আমার উদ্দেশ্যই নষ্ট করে ফেলেছিস" = "You destroyed my whole purpose." | The KV controller was meant to push concurrency up dynamically, not cap it near 16. |
| 2026-10-14 | L27357 | "সহজ + কঠিন এমন পাজল ভাগ ভাগ করে একবারে পাঠালে তো আর এই সমস্যা টা থাকে না" = "Mix easy and hard puzzles and the simultaneous KV peak goes away." | Mixed admission to flatten the KV peak. Repeated 10-26 (L112583). |
| 2026-10-14 | L28775 | "Nvidia Team মাত্র ৭ ঘন্টা + পাজল সলভ করে ৭৫+ পেয়েছে, তাই অপ্টিমাইজের জায়গা আছে" = "NVIDIA team got 75+ in about 7 h, so there is room to optimize." | First mention of the NVIDIA-team benchmark. |
| 2026-10-15 | L29688 | E9 86k LB 22.92 "৮৬k চিন্তার জন্য প্রাইভেটের ২৪০ টা পাজল সলভ ই করার সময় পায় নি"; R5 "প্রায় ৯ ঘন্টা নিয়েছে প্রথম ইটারেশন শেষ করতে" = "At 86k thinking it had no time for the 240 private puzzles; R5 took ~9 h for the first iteration." | User-reported LB: coverage binds. |
| 2026-10-15 | L29997 | "কনকারেন্সি আমাদের ১৬, এটা ধরো ২৫/৩০ করতে চাই, গাণিতিক হিসাব করো" = "Our concurrency is 16; I want 25/30. Do the math." | Target 25-30 concurrent streams. |
| 2026-10-16 | L34662 | "KV ৮৫% পার হলে নতুন request নেওয়া হয় না ... এটা অত্যন্ত কড়া ... ৯৫%+ ও তো নিরাপদ" = "An 85 % KV admission gate is far too strict; 95 %+ is safe." | Admission gates must not leave KV idle. |
| 2026-10-16 | L36077 | "গতি: T 0.6-এ tok/s আর MTP acceptance ... অল্প কিছু টাস্কেও মাপতে পারি, ১২ ঘন্টার রান প্রয়োজন নেই" = "Measure tok/s and MTP acceptance at T 0.6 on a few tasks; no 12 h run is needed." | Speed is measured in short runs, not 12 h runs. |
| 2026-10-17 | L41076, L42660 | "৩২ বা এই কাছাকাছি কনকারেন্সি কি আসবে?"; "২-৩ গুণ গতি ও আর ২৫-৩২ কনকারেন্সি আমরা কোনভাবেই পাচ্ছি না?" = "Will we get ~32 concurrency? Is there really no way to 2-3× speed and 25-32 concurrency?" | Targets on 10-17: 2-3× speed, 25-32 streams. |
| 2026-10-22 | L79784 | "আমাদের ইঞ্জিনের কোন সমস্যা এর গতি ৬০০–৮০০ এ যেতে দিচ্ছে না?" = "What in our engine stops it reaching 600-800?" | Target band on 10-22: 600-800 tok/s. |
| 2026-10-23 | L88968 | "আগে ২-৩ গুন বলে এখন এত কম কেন বলছিস? এটা হলেও তো ১০০% কভারেজ পাবো না?" = "Why do you now say +10-25 % when you said 2-3× before? Even that won't give 100 % coverage." | Small gains are rejected; coverage of all 240 tasks is the bar. |
| 2026-10-23 | L89737 | "গত ৪ দিনের সকল রিসোর্স কথোপকথোন পড়ে আসো, আমরা গতি বাড়ানোর নতুন এক্সপেরিমেন্ট করবো" = "Read all resources and conversations of the last 4 days; we do a new speed experiment." | Start of E14. |
| 2026-10-26 | L112542, L112569 | "আমরা কি ৮০০-৯০০ গতি পাবো?"; "কয়টা কনকারেন্সি পাওয়ার সম্ভাবনা আছে সাথে গতি?" | Target band on 10-26: 800-900 tok/s. |
| 2026-10-26 | L116585 | "শুরুতে ১০-১৫ মিনিটের গতি মাপ যোগ করবে, এর পর অডিট করবে দুইটা এজেন্ট দিয়ে। এরপর অনুমতি নিয়ে কমিট" = "Add a 10-15 min speed measurement at the start, audit with two agents, then commit after permission." | Pattern for every FN notebook: short speed stage first, 2-agent audit, push only after go. |
| 2026-10-27 | L119274, L119302 | "speed kome gelo keno?"; "NVIDIA Team 7 hours 46 sec এ ৭৫+ পায়। আর আমরা একটা ইটারেশনই শেষ করতে পারি না কেন?" | NVIDIA team: 75+ in 7 h 46. We cannot finish one iteration. |
| 2026-10-29 | L126264 | "অনেক গুলো গতির পরিক্ষা করেছি, কেন কভারেজ এখনো দিতে পারলাম না ... এনভিডিয়া টিম ৭ ঘন্টা ৪৬ মিনিটে" | Same; coverage is the problem. |
| 2026-10-29 | L126349, L126384 | Asks for a GPT prompt to search "GitHub, Hugging Face, web, Reddit, the whole internet" for 4×L4 speed; "স্পিডের রিসার্স থামালি কেন?" = "Why did you stop the speed research?" | Exhaustive external search is expected. |
| **2026-10-29** | **L126812** | "**১৫০০+ সম্ভব এটা NVIDIA Team করেছে, এটার প্রমাণ খুজতে হবে না। তুই মেমরিতে রাখ। তাই এইটা সম্ভব আর আমাদের টার্গেট ১০০০ +**" = "1500+ is possible, NVIDIA team did it; don't look for proof; keep it in memory. So it is possible, and our target is 1000+." | Origin of the 1500 figure. **Note:** in this message the stated target is 1000+, with 1500+ as the proven-possible level. From 10-31 the user uses 1500+ as the target (L142091, L142115). Memory file `speed-target-1000-plus.md` says "OUR TARGET 1500+". |
| 2026-10-29 | L127737 | "তুমি নিজের মত এগোচ্ছো কেন? আগে আমি এই ৬ টা এক্সপেরিমেন্ট এক নোটবুকে করতে চাই। ... এক নোটবুকে ২০-২৫ টা আর্ম ... তা চ্যাঁই না" = "Why are you going your own way? I want these 6 experiments in one notebook. I do not want 20-25 arms in one notebook." | Keep speed notebooks to a few arms; follow the user's list. |
| 2026-10-30 | L134508 | "Merlin-এর default reasoning_effort medium ... xhigh কেন দিস নি?" = "Why didn't you set xhigh?" | Merlin must be forced to xhigh. |
| 2026-10-30 | L135827 | "টপরা স্বীকার করতেছে যে তারা Qwen3.8-Flash-Next alongwith the Duck harness ইউজ করতেছে" (discussion 750546) | Model switch to Flash-Next becomes the priority. |
| 2026-10-30 | L135935 | "আমাদের গতি বাড়ানোর এক্সপেরিমেন্ট গুলো কি Flash-Next এ কাজে লাগবে না?" | Reuse the 27B speed work on FN. |
| 2026-10-30 | L135975 | Approves two FN experiments first (fix the "!!!!" output; grow KV space): "এই দুইটার জন্য এক্সপেরিমেন্ট করি চলো। আর ফুল রান পরে দিবো" = "Full run later." | Correctness and KV before any full run. |
| 2026-10-30 | L136909 | "SGLang and vLLM এ flashnext সেট আপ করতে হবে।" | Two engines for FN. |
| 2026-10-31 | L139511, L139693 | "B. Merlin + Flash-Next — এটায় যাওয়ার আগে আরো ডিপলি এনালাইসিস করো। প্রম্পট দিও"; "Merlin এর ডাউনলোডের কাজ চলো করা জাঁক" | Merlin path approved after deeper analysis. |
| 2026-10-31 | L141725 | Quoting "4×L4-এ এর কোনো গতির সংখ্যা এখনো আমাদের হাতে নেই": "এটার গতি মাপার এক্সপেরিমেন্ট করা হয়েছি কি?" | User noticed: no valid FN speed number exists yet. |
| 2026-10-31 | L141844, L141858 | "পুশ পরে করিও, আমি অনুমতি দিলে"; "আগে দেখিও, ফিক্স লিস্ট বানাও" | Push only with go; show a fix list first. |
| 2026-10-31 | L142091, L142115 | "১৫০০+ এর জন্য প্রম্পট দিও ... Flash-Next এর জন্য রিসোর্স + পরিকল্পনা"; "৩ টা [GPT] পরামর্শ দিবো ... রিসার্স প্ল্যান ... আপডেট করতে থাকবে। নতুন ফোল্ডার খুলে নাও FLASH NEXT Throughput" | E25 mandate: 1500+ for Flash-Next, living plan updated after each of 3 GPT reports. |
| 2026-10-31 | L142167 | "৪ এজেন্টকে কাজ দিও ... আগে যে যে Throughput ... করেছি যা FLASH NEXT Throughput এ লাগবে ... কোন গুলোতে আমরা ভালো করে ছিলাম, আর কোনগুলোতে খারাপ ... প্রয়োজনে ১০ টা" | This mining task. |

**Related standing rules (from memory, re-stated in the transcript):** thinking floor 60k, never cut below without approval (L119319 cites "49k LB 32.08 vs 60k 34.58"); thinking up to 120k (max-model-len 131072); never import Duck's 12,288 cap; never call the target impossible; trust the user's observations of other teams (L64366, 10-20: "ওরা সবাই Qwen3.8 + Optimized Duck Use করছে").

---

## 2. Numbers found in transcripts (number | context | source)

### 2a. Flash-Next on our hardware

| Number | Context | Source | Folder status |
|---|---|---|---|
| ~6 tok/s single stream (5.95) | E3 FN-PROBE v4.2.1, Pennyroyal SGLang, eager, FA2 shim, output "!!!!", TileLang "data race" warning | L14625 (10-11T02:58) | In E3 folder |
| 0.66 GiB free per GPU | E3, after load | L14625, L57554 | In E3 |
| KV 118k tokens, server up in 862 s | E3 | L55917, L116505 | In E3 |
| "30-50 tok/s at 8 streams at best" | Claude's extrapolation from E3 | L14625 | Not measured |
| KV 94,528 tokens (mem 0.93), 161,984 (mem 0.95) | E13 SGLang 0.5.21 TP4+EP4 | L118206, L139662 | In E13 |
| 40.8 tok/s single stream (garbage output) | E13 LOADPROBE, 10-26 | L118192, L141753 | In E13 |
| weight 19.1 GB per rank; load eager 1348 s, graph 716 s | E13 SGLang | L139662 | In E13 (NANPROBE/RC1 results) |
| "1 stream at 60k fits in 94.5k KV"; 41 tok/s × 43,200 s ≈ 1.8M tokens ≈ 30 requests in 12 h vs 27B production 583 tok/s ≈ 25M (14×) | Claude's comparison on 10-26 that pushed FN back | L118206 | Partly |
| Stock vLLM 0.32 TP4 BF16 KV pool 30,208 tokens; RC1 4-stream speed test died on `-lcuda` before any tok/s | E13 RC1 (10-31) | L141753, PLAN_1500 §2 | In E13 |
| Merlin capacity on L4 (Merlin's block rule, not measured): 2.31 GiB KV per card = 150 blocks; FP8 ≈ 558k tokens = 5 streams at 70k, 3 at 131k; BF16 ≈ 292k = 3 / 2 | Merlin synthesis | L139879, L140731 | In E13 synthesis |
| Merlin projection on L4: 50-80 single, 200-300 at 8, 150-250 at 3-4 streams; 200-320 at 5-6 long streams | Claude estimates (L4/3090 bandwidth 300/936) | L139572, L139662, L139879 | Projections only |
| Flash-Next active parameters 5-6B; 2 KV heads, head dim 256; 48 layers 3 GDN : 1 QSA; 512 experts, 10 routed + 1 shared; QSA indexer budget 2048, compress 4; PLE 51B | Research agent ac5bf92583e9b8eab (10-29) | subagent | In E24 Resources research |
| "pass if output correct and aggregate ≥ 700 tok/s (≈ 2× the 27B)" | E13 stage-1 pass criterion set on 10-19 | L55917, L56439 | Probably in E13 PLAN |

### 2b. Flash-Next elsewhere (other hardware, other teams)

| Number | Context | Source | Folder status |
|---|---|---|---|
| ~240 tok/s, concurrency 28 | Our ARC-AGI-3 track: Flash-Next-NVFP4, 1× RTX PRO 6000, vLLM nightly, MTP-3 | L244 (09-15) | ARC-AGI-3 track, not in ARC-AGI-2 experiments |
| 946-1,159 tok/s | ARC-AGI-3 numbers on RTX PRO 6000 | L48609 (10-18) | Not checked |
| 590-617 tok/s; score 36.56 (v1) / 46.49 (v3); host RAM 106/177 GB; GPU 91/96 GB; weight read 321 s; server ready 562 s | Franzen ARC-AGI-3 milestone-2, 1× RTX PRO 6000, TP1, sm120 env | L56292 (10-19) | In E12/E13 notes |
| 10 streams, prefix reuse 93.43 %, P90 aggregate decode 836.6 tok/s, 609 harness-side | Franzen Duck setup | L126551; agent ac5bf92583e9b8eab | In E24 research and E12 |
| **14 streams → peak ~470 then ~280 tok/s from KV preemption; score 47.30 → 29.90** (baseline 10 streams 600-620) | Franzen ARC-AGI-3, Kaggle topic 750885 | agent af0250ade484cc769 (10-29) | **Only in E24 Resources research notes; not in E13/E25** |
| Merlin 4×3090 FN: 168-184 single, 638 at 8 (16K, thinking off), 574.5 for 6 × 126,976 sessions, KV 924,993, GSM8K 199/200 | Merlin README | L139508, L141753 | In E13 synthesis, PLAN_1500 |
| ~69 tok/s on 4×3090 (Intel W4A16 AutoRound, stock vLLM) | HF Intel discussion #2 "Works Great!" | L139439 | **Not in folders** |
| 3× RTX 4090 48 GB, FN NVFP4, vLLM 0.30: 124 / 387 / 618 / 635 tok/s at 1 / 4 / 8 / 16 streams; needs ~28 GiB per card | chenxin199305 recipe | agent ac5bf92583e9b8eab | In E24 research only |
| 56 tok/s claimed, 296-expert pruned checkpoint, SGLang fork with "Marlin repack GC fix for W4A16 load OOM" and "mixed-chunk × QSA crash fix" | MinChen05 4×3090 guide (repo now 404) | L139439 | **Not in folders** |
| Darren Amadeus Martin (topic 750546): 900 tok/s top, 980 tok/s on 1-min harness work incl. tool calls and prefill, ~1,100 pure decode at the same concurrency, 710 average over 6 h × 2 passes of public eval ("tail starved of concurrency"), ~250 under heavy prefill; definition = generated tokens ÷ wall clock; model not stated; the optimization "did not transfer" to ARC-AGI-3 (score 4) | Kaggle thread | L135910, agent a2e67755474e7b266, agent af0250ade484cc769 | 980/710 in PLAN_1500; the 1,100 pure-decode figure, the definition and the "did not transfer" note are in E24 research only |
| Anil Thomas: FN ≈ 75 % one pass / 80 % two passes; 365 tok/s for one 64k pass of 240 tasks, 440 tok/s with a 20 % second pass | topic 750546, 10-27 | L135910 | Partly (E25 prompt) |
| TheDivine Aurora: 250-400 tok/s → 19-20 %; ~40 % needs 600-800; FN solves far more than 27B on public eval; "quantization hurts quality"; needs 128k-256k context | topic 750546 | L106051, L140712 | Partly |
| Adarsh: on 4×L4, 27B and FN "almost breakeven in speed and accuracy" (Duck) | topic 750546 | L135910 | In E24 research |
| mikelou1: CV 87 % → 79 ± 5 %, one sub 66; "a tiny kernel issue dropped the score from 60 to 40" | topics 750546 / 748082 | L140712, L106051 | Partly (E17) |
| NguyenThanhNhan (65.97): SGLang 27B FP8, fp8_e4m3 KV, flashinfer, triton linear-attn, chunk 2048, max-prefill 4096, mamba extra_buffer, full-memory-ratio 0.5, ssm float32, EAGLE 3/1/4 | topic 745787 | L106051 | In E14/E24 |
| Leaderboard 10-29: Tufa Labs 88.06, Yi-Chia Chen 84.72, Domdolus Tolus 83.47, nvbanana 81.39, rabbithole 80.83; Franzen rank 40 at 57.64; us rank 123 at 34.58 | Tong Hui Kang monitor | agent af0250ade484cc769 | Not in E25 |

### 2c. Our engine numbers that bound what FN must beat (mostly 27B)

| Number | Context | Source | Folder status |
|---|---|---|---|
| vLLM 0.19 FP8 MTP3, 16 running: gen 338.8-365 tok/s, KV 65-73 %, prefix hit 1.9-2.4 %, acceptance 2.94-3.08; prompt 555.9 tok/s cuts gen to 315 | User-pasted live logs | L23265, L23303 (10-13) | In E7/E8 run logs |
| Poetiq on vLLM 0.19, 12 → 7 running: 353 → 288 (accept 2.68) → 159 gen with 256 prompt tok/s (prefix hit 59.6 %) → **52.3 gen tok/s while prompt is 1,796 tok/s** (prefix hit 76.7 %) | User-pasted live logs | L26385, L26446, L26634, L27049 (10-14) | In E10 logs |
| 27B "373 tok/s" was the steady 16-running decode; the whole-run average was 248.9 | E1/E4 | L14533 (10-10) | In E1 |
| S1 whole-run ~256 tok/s vs steady window > 400 | S1-60K | L141317 | In E18/E20 |
| SGLang INT4 sweep c=1 55.1, c=4 205.1, c=8 297.3 tok/s, accept 3.1-3.2 | E11-03 B_int4_mtp | L40899 (user-pasted) | In E11 |
| DFlash2 gain vs INT4 alone: 2.36× at 1, 1.9× at 4, 1.39× at 12; INT4 no drafter 347 at 16, 349 at 22; step ~137 ms at c=12 vs ~22 ms at c=1 | E11 | L42677 (10-17) | In E11 |
| Per-step cost from logs: INT4 no-spec 21.9 ms @1, 37.9 @12 (50 @22, ~1.3 ms per extra request); MTP 40.3 @1, 79.9 @10 (~4.4 ms each); DFlash2 39.9 @1, 115.5 @12 (~6.8 ms each); ~0.85-1.3 ms per verified token; 64 MB all-reduce 20.9 ms ≈ 3 GB/s | Log analysis | L79809 (10-22) | In E14 Notes |
| Profiler: NCCL all-reduce 50.9 ms of a 116 ms step (44 %); + all-gather 8.1 ms = 51 % communication | E14A SGLang TP4 DFlash | L112566, agent ae6e896917f686064 | In E14 |
| NCCL default SHM, 2 channels: 983,040 B all-reduce 343.9 µs in a graph; 64 MiB 4.90 GB/s busbw. Forced P2P (`NCCL_P2P_LEVEL=SYS`): 1,880 µs (5.5× slower), 0.42 GB/s. `NCCL_SHM_USE_CUDA_MEMCPY=1` slower (503 vs 382 µs eager) and timed out in graph phase. All pairs PHB, `can_device_access_peer` True, NCCL 2.29.7, driver 580.178.04, **nvidia-smi in the VM reports pcie.link.gen.max = 3** | E14A nccl_bench.json | agent ae6e896917f686064 (10-29) | bench json in E14/E24; the "PCIe gen 3" reading is not in E25 |
| Step 107-125 ms; weight read only 16-22 ms; ~80 ms overhead; 900 tok/s needs a ~52 ms step; without communication the step is ~57 ms (900-1,000 tok/s at short context) | 27B analysis | L126535, L126778 (10-29) | In research report `reports/৪xL4 এ ৯০০ টোকেন গতির উপায়.md` |
| Best measured 27B, 8k / 40k / 90k: SGLang INT4 + DFlash2 W6 20 streams 597 / 536 / 364; vLLM 0.32 INT4 + MTP3 20 streams 518 / 439 / 332; whole-run 254-380; FP8 20 → 24 streams 423 → 424 | Same report | L126778 | In reports/ and E22/E24 |
| E14A 531 (12 streams, ~6k, DFlash W6); E15B real ARC 451-469 (16 streams); E15P Poetiq 6 h whole run 389 | | L112566 | In E14/E15 |
| E12 Duck: engine decode ~420, effective ~187 tok/s because 144 requests were evicted and 8.4M tokens re-prefilled; 27 % of requests hit the 12,288 cap | E12 EVAL30 | L57155 (10-19) | In E12 |
| Duck effective 167-279 vs engine peak 450-600 → **effective ≈ 50-60 % of benchmark peak** | | L127118 (10-29) | **The ratio is not stated in E25** |
| Merlin 27B TP1 GPU0: 93.9 tok/s at 40k@2 (47/stream), step 92 ms, accept 4.3; 41.0 at 120k@1; TP1 ×4 replicas 359.5 at 40k (92.0/91.8/90.0/85.7, 2 per GPU); KV per GPU 220,897; TP4 421.5 at 40k@8, 329.0 at 120k@8 (accept 4.227, step 102.8 ms, KV 1,789,034) | E23 SMOKE (10-30) | L141285, L134505 | In E23 |
| FP8 MTP3 TP4 40k@16: PIECEWISE G0 435.7 (accept 3.569, step 131.1 ms, KV 1,159,379) vs FULL G3 + PR 60532 448.9 (+3.0 %, accept 3.584, step 127.7 ms); NVFP4 KV MTP off 120k@8 133.0 tok/s, step 60.2 ms, KV 2,356,868 (1.67× FP8), TF drift 0.379 (TP4→TP1 alone 0.077) | E23 SMOKE | L134505, L140712, L140731 | In E23 |
| jsboige fork (vLLM 0.32, sm_89) NVFP4 KV: KV 429k → 751k (1.75×) but aggregate 886 → 768 tok/s at N=48 (−13-17 %); fork rejected it | Research agent ab21c481f100b4f14 | L126804 | In E24 research only |
| club-3090 4×3090 c=16 1,811 tok/s; ÷ bandwidth ratio 3.1 ≈ 584 ≈ our 597 → "we are near bandwidth parity; more needs fewer bytes read" | | L126804 | In E24 research / reports |
| 4× RTX PRO 2000 Blackwell (~288 GB/s, closest to L4) TP4 NVFP4 MTP3: 386.9 at c8 and still rising | agent af0250ade484cc769 | subagent | In E24 research |
| E18 align mode prefix hit 7.7 %; prefix hit 55 % on vLLM 0.32 vs 69-82 % on 0.19 | | L141317, L126287 | In E18/E21 |
| Target arithmetic, 11.3 h usable: 300 tok/s → 204 × 60k attempts (36 tasks get nothing); 400 → 1 per task; 700 → 2 per task; 1000 → 2.8; 1500 → 61.1M tokens = 4.2 × 60k per task | Claude table | L127118 (10-29) | Similar table in PLAN_1500 §1 |

---

## 3. Decisions and reasons

| Date | Decision | Reason given | Source |
|---|---|---|---|
| 10-05 | Put Flash-Next behind; go 27B FP8 + MTP at xhigh, target ~350 tok/s | Discussion 745787 said 4×L4 is slow for FN; Anil's math: 350 tok/s → ~30 LB | L8514 |
| 10-07 | Run FN-PROBE v4 alongside, to see whether FN is 2-3× faster than 27B (373) | Biggest possible lever | L13083 |
| 10-11 | Stop Flash-Next ("not viable") | 6 tok/s, "!!!!" output, 0.66 GiB free | L14625, L49049 |
| 10-18 | FN kept blocked | License concern (qwen-community-1.0) plus KV | L48609, L116270 |
| 10-19 | Re-open FN as E13 on SGLang 0.5.21 (qwen4_exp, auto_round, PLE offload present); route QSA decode to SGLang's own Triton kernel instead of FA4; copy Franzen except TP4+EP4 and Triton GDN | User: "the same architecture with Flash-Next, much work was done before" (L55766); FA4 crashes on L4 | L56439, L56499 |
| 10-19 | Claude: "only 1-3 Duck streams, ≤ ~420 tok/s, NEXTN won't fit". GPT audit: all three "not proven" | Claude extrapolated from E3 (Pennyroyal, BF16 KV, no graphs, no NEXTN) | L57551, L57554 |
| 10-26 | "We never used Flash-Next in the ARC-AGI-2 Duck" (only harness/prompt from Franzen); after LOADPROBE: FN deprioritized again | KV 94.5k = 1 stream; 41 tok/s; 27B production 583 tok/s = 14× more tokens | L116270, L118206 |
| 10-29 | Final research report: "no measured path to 700-1,000 tok/s for 27B at 50-130k on 4×L4; PCIe communication holds us" | 5 research agents | L126778 |
| 10-29 | User overrides: 1500+ is possible (NVIDIA team), no proof needed; target 1000+ | User observation | L126812 |
| 10-29 | Six-lever E23 in one notebook (Merlin TP1×4 / PP4, host AR, FULL graph, SM89 FP8, NVFP4 KV, MTP depth), not 20-25 arms | User instruction | L127737 |
| 10-30 | FN becomes the priority again after topic 750546; first fix "!!!!" (NaN probe), then KV | Top teams on FN + Duck | L135910, L135946, L135975 |
| 10-30 | What carries from 27B speed work to FN: measurement harness, Duck work, KV capacity (most needed), MTP depth, INT8 all-reduce idea, FULL-graph idea (PR 60532 patch does not apply: QSA runs on TileLang), graph-proof parser. **Not carried:** Merlin 27B W4A16, TP1×4 (73.6 GB weights do not fit one L4), SM89 dense FP8 GEMM (little), PP4 (later; crashed at 8k on stock vLLM for 27B). Host-staged all-reduce dropped: slower than NCCL | Architecture of FN | L135946 |
| 10-31 | Choose Merlin + Flash-Next (path B) over SGLang (broken MoE) and stock vLLM (correct but KV 30k at TP4 BF16) | Merlin has an sm89 profile, FN correct on 3090, engine already proven on Kaggle via E23, TP2×PP2 splits the 2 KV heads and halves layers per card | L139508, L139572, L139662 |
| 10-31 | Download only 72.55 GB of the 124 GB Merlin checkpoint (5 CPU kernels, rev cec53e9f); PLE (51.2 GB) from our FP8-PLE overlay, which is byte-identical to albucino by Merlin's sha list | Saves upload; no GPU quota | L139802 |
| 10-31 | Smaller GGUF / 2-3 bit checkpoints not taken for now (llama.cpp/Strata, no continuous-batching proof, quality risk); keep vLLM-loadable small checkpoints as an open branch (Minachist INT4/INT6 mixed, klee100 3bpw-MTP, Haberstroh int2-mixed); PLE compression does not add streams | GPU memory per card is the bind, not disk | L139822 |
| 10-31 | Merlin verdict GO-WITH-GATES; three must-haves: lower `KV_BYTES` to L4 size, force xhigh, run `host_phase` | 9-agent workflow | L139879 |
| 10-31 | Pick the model for the 1500 work by **solves per GPU-hour** = (avg tok/s × 3600 ÷ tokens per task) × solve rate | Fair 27B vs FN comparison | L141343 |
| 10-31 | RC2 pushed (foysalemonshanto, 12eaa96c, 240 min budget) | User go | L142022 |
| 10-31 | FP8 KV is the production target, BF16 the fallback; gate = BF16 A/A noise vs FP8 drift | 67 % more streams | L140731 |

---

## 4. Failures and what fixed them

| Failure | Where / when | Fix or status | Source |
|---|---|---|---|
| FA4 kernel crashes on L4 (sm89) for QSA decode | E3, 10-10 | Own FA2 shim (math verified correct later). SGLang PR #36968 (fail early without FA2) still open | L56439, agent ac5bf92583e9b8eab |
| "!!!!" output (token 0), NaN | E3, E13 SMOKE1, LOADPROBE | NaN probe: first non-finite at `model.layers.2.mlp.experts` (FusedMoE, GPTQMarlinMoEMethod, SGLang 0.5.21, EP4). Fix = leave SGLang; stock vLLM 0.32 is correct (RC1), Merlin uses `MERLIN_FULL_K` (no split-K lock path) | L137272, L138583, L139879 |
| Load OOM on SGLang 0.5.21 | E13 LOADPROBE 10-26 | Delayed free after Marlin repack; `gc.collect()` hook after repack. MinChen05's fork has the same "Marlin repack GC fix" | L118192, L139439 |
| FlashInfer JIT `-lcuda` link error → EngineDeadError | Q27 v1.3.x 10-05; again in E13 RC1 4-stream test 10-31 | libcuda linker fix (`/usr/local/nvidia/lib64/libcuda.so.1`) worked on 10-05 (L10273). RC1 shows the fix must be present in every new engine path | L9725, L10273, L141753 |
| Stock vLLM 0.32 TP4 BF16 KV only 30,208 tokens | E13 RC1 | Merlin TP2×PP2: under TP4 the 2 KV heads are replicated on every card; TP2 splits them and PP2 halves the layers per card | L139439, L139662 |
| Merlin default reasoning_effort "medium" | E23 | `--default-chat-template-kwargs` with xhigh (rev 3h); same fix needed for FN | L134508, L139684 |
| Merlin needs GLIBC_2.38 | E23 | Weak symbol + `__isoc23_strto*` LD_PRELOAD shim | L139879 |
| Merlin `KV_BYTES` defaults to 4.1e9 (4090 size) | RC2 design | Lower to L4 size (2,479,838,700 B in PLAN) | L139879 |
| Stock vLLM recipe for Intel FN on 24 GB: util 0.94 OOMs; indexer default 512 MiB too big | todiadiyatmo 3090 recipe | util 0.88, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, `NCCL_CUMEM_ENABLE=0`, `--disable-custom-all-reduce`, `--max-num-seqs 3`, batched 1024, FULL_AND_PIECEWISE ("decode about 2× PIECEWISE"), BF16 PLE needs ~149 GiB host RAM | L139439 |
| GCSFUSE slowdown: SGLang ready 1,197 s vs the 30-min launch timeout (= LB 0) | 10-21+ | Weights as Kaggle Models (NFS, ~15× faster than notebook output), warm page cache, longer timeout | memory, L142112 |
| PCIe-IPC all-reduce (FlashInfer 0.7) 0.68× vLLM, 0.4× SGLang; host-staged AR 0.93×; 2×TP2 DP KV 0.57× at 40k; PP4 illegal memory access at 8k (stock vLLM, 27B) | E22/E23 | Dropped; NCCL SHM remains the best. Untested: more NCCL channels (`NCCL_MIN_CTAS`/`NCCL_MAX_CTAS` 4 or 8, `NCCL_NTHREADS=512`); INT8 AR estimate +30-35 % (accuracy risk) | L126287, L126804, agent ae6e896917f686064 |
| FULL_AND_PIECEWISE fell back to PIECEWISE on vLLM because FlashInfer supports only single-token decode graphs; symm-mem "capability 8.9 not supported"; FlashInfer all-reduce not supported at world size 4 or 2 | vLLM 27B logs | PR 60532 gave FULL (+3 % only) | agent ae6e896917f686064 |
| SGLang DFlash min-free-slots delayer kept 1 of 12 slots empty (11 real) | E11-05/E15 | Fixed (delayer off) | L88164, L47794 |
| E12 Duck KV thrash (8.4M re-prefill, effective 187 vs 420) | E12 EVAL30 | Fewer workers / admission control | L57155 |
| ARC-AGI-3 Franzen 14 streams → preemption, 280 tok/s, score −17 | topic 750885 | Keep streams within the resident fit | agent af0250ade484cc769 |
| MTP + prefix caching OFF gives wrong tokens (4-12 flips) | E19 | Never disable prefix caching with MTP | memory, L135946 |
| VMS log lines cut at 700 chars hid vLLM tok/s | E22 VMS, 10-26 | Print fix (cannot be changed mid-run) | L113383 |
| E13 rev 6 speed stage could kill the server before Duck with no liveness check; decode-summary filter `"TP0]"` misses `" TP0 EP0"` under EP | audit agent ac8ce9a5ce8561e85 (10-26) | Post-speed smoke + `"TP0" in line` | subagent |

---

## 5. Things relevant to Flash-Next throughput that are NOT in the experiment folders

Checked with ripgrep over `experiments/**/*.md` (some appear only under `research_notes/`, `reports/` or E24 `Resources/sources/research/`, which E25 does not cite).

1. **The source of the 1500 figure and the exact target.** L126812 (10-29): "1500+ is possible, NVIDIA team did it ... our target 1000+". Later messages use 1500+ as the target. PLAN_1500 does not record this, nor whether 1500 is a whole-run average or a peak.
2. **Effective ≈ 50-60 % of engine peak** (L127118). If 1500 means whole-run effective tok/s, the engine peak has to be ~2,500-3,000, or the harness losses (re-prefill, tail drain, tool time) must be cut. Darren's own ratio is 710/980 ≈ 72 %. PLAN_1500 §1 uses 1500 without this factor.
3. **Darren's definition and extra numbers:** tok/s = generated tokens ÷ wall clock (prefill not counted in the numerator); pure decode at the same concurrency ~1,100; the method did not transfer to ARC-AGI-3 (score 4). Only in E24 research notes.
4. **Franzen's ARC-AGI-3 KV-preemption collapse** (topic 750885): 10 → 14 streams took 600-620 to ~280 tok/s and the score from 47.30 to 29.90. This is direct evidence for the L7 risk (seqs above the resident fit). Not in E13/E25.
5. **The stock-vLLM Flash-Next recipe** (github.com/todiadiyatmo/rtx-3090-vllm-recipe-qwen-flash-next): the repo name, the claim "FULL graph decode ≈ 2× PIECEWISE", `NCCL_CUMEM_ENABLE=0`, `--disable-custom-all-reduce`, BF16 PLE ~149 GiB host RAM, PP>1 fixes #54709/#54793/#54795, and the ~69 tok/s 4×3090 report on the Intel HF page. Some settings (util 0.88) are in E13 RC1/RC2 files, the source is not.
6. **MinChen05's 4×3090 SGLang guide** (now 404): 56 tok/s, 296-expert checkpoint, same repack-GC OOM fix as ours, plus a "mixed-chunk × QSA crash fix". Not in folders.
7. **Small vLLM-loadable FN checkpoints** to test for more streams: `Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`, `HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang`; and the estimate that main weights of ~47 GB (vs 66.2 GiB) would give ~7 GiB KV per card, ~2-3× the KV, ~8-10 streams (L139822). Caveat: Marlin runs only 4- and 8-bit. Not in folders.
8. **Scaling curve for FN on Ada:** chenxin199305 3×4090-48GB NVFP4 124/387/618/635 at 1/4/8/16. It flattens after 8 streams, which matters for the "12 streams × 3 accepted / 24 ms" plan. Only in E24 research.
9. **Align-mode costs:** vLLM align mode with prefix caching costs 13-16 % throughput even at zero hits (#60008); align mode does not keep the Mamba state at the end of a reply, so every agent turn re-prefills the previous reply (fix PR #50551 `--enable-mamba-decode-checkpoint`, open, rejects hidden-state spec decode, so probably not with MTP). Merlin's FN profile uses `--mamba-cache-mode align`. Only in research notes (agent ab21c481f100b4f14).
10. **KV offload does not add live streams** (a decoding request needs its KV on GPU); offload on hybrids is fragile (#58653, #63565); a fork on sm_89 "wedged" at 0.3-0.9 tok/s when GPU KV hit 100 %. This contradicts the "KV-to-host offload" next step listed in L140712. Only in research notes.
11. **Speculative decoding stops paying above ~4-8 concurrent requests unless the state pool is enlarged** (community pattern, agent af0250ade484cc769). Supports L3 (MTP k vs concurrency).
12. **PCIe in the VM reports gen 3** (`pcie.link.gen.max = 3`) and NCCL runs at ~35 % of a ~120 µs host-path floor with only 2 channels; the channel knob is untested. Only in research notes / E14 json.
13. **L4 72 W power cap → clock drop at high stream counts**; proposed `nvidia-smi dmon` per GPU (L141317). In the E25 prompt but not as a telemetry item in PLAN §4.
14. **SGLang #36807**: `fast_topk<512>` for QSA gave wrong top-k on SM89 (RTX 4090) in 20/20 runs with concentrated logits; FlashInfer 0.6.17 `top_k_ragged_transform(deterministic=True)` passed 29/29. Relevant if any FN path uses that indexer top-k (L57554). In two E13 review files, not in E25.
15. **"MoE at many streams reads almost all experts"**, so FN's per-step cost may grow faster with streams than its 5-6B active size suggests (L141343). A measurement point, not in PLAN.
16. **Live log signatures** worth copying into FN telemetry: on vLLM 0.19, a prefill burst of 1,796 prompt tok/s cut generation to 52 tok/s with 7 running (L27049). Prefill isolation (chunk size, L2) is therefore a large lever.
17. **Leaderboard snapshot 10-29** (Tufa 88.06 ... Franzen 57.64 rank 40, us 34.58 rank 123); nvbanana and Tufa will publish code at the end; ARC Prize staff reportedly confirmed on Discord that Flash-Next is allowed (second-hand, topic 752709). Not in E25.

---

## 6. Open questions

1. **Metric of the target:** is 1500 the whole-run effective tok/s (Darren-style, generated ÷ wall clock) or a steady-window peak? With the measured 50-72 % effective/peak ratio this changes the engine target from ~1,500 to ~2,100-3,000.
2. **Merlin FN graph mode under MTP K3:** does FULL_AND_PIECEWISE really replay FULL for multi-token verify on sm89, or does it fall back to PIECEWISE as stock FlashInfer did for 27B? The RC2 log must show the mode.
3. **Align-mode tax:** does Merlin's `--mamba-cache-mode align` cost 13-16 % at zero prefix hits (#60008) on FN as well? Is there an FN-safe mode with MTP?
4. **QSA indexer top-k on sm89 in vLLM/Merlin:** is it free of the SGLang #36807 bug class? An RC2 long-context needle row tests it only indirectly.
5. **Resident fit vs seqs 8:** Merlin's block rule gives 5 FP8 streams at 70k. Admitting 8 risks the Franzen-style preemption collapse. What is the real KV pool on L4 after RC2 boot (`--language-model-only` delta, `KV_BYTES` +0.25/0.5 GiB)?
6. **Stream scaling of MoE decode on L4:** does aggregate tok/s keep rising from 5 to 8 to 12 streams, or flatten as in the 3×4090 NVFP4 curve (618 → 635 from 8 to 16)?
7. **Communication on TP2×PP2:** NCCL SHM with 2 channels and PCIe gen 3; does `NCCL_MIN_CTAS`/`MAX_CTAS` 4-8 help? Does P2P (can_access_peer True, but 5.5× slower) stay off?
8. **Power cap:** does the 72 W limit drop clocks at 8+ streams on L4?
9. **Smaller main weights:** can Merlin or stock vLLM load any of the INT4/INT6-mixed or 3-bit FN checkpoints on sm89, and is long-reasoning quality kept?
10. **Model choice by solves/GPU-hour:** RC2 gives FN tok/s; an FN Duck EVAL is still needed for its solve rate on our harness (Adarsh "breakeven" vs TheDivine "much better").
11. **Pending data:** E23 FULL (esfersami50, ends ~17:45 BST 10-31) has the TP1×4 level ladder for 27B; RC2 (foysalemonshanto) has the first valid FN numbers. Neither result is in the transcript yet.
12. **Darren's model:** unstated; his team is not identifiable on the leaderboard. Treat 980/710 as a 4×L4 harness benchmark, model unknown.
