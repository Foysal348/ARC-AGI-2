# PCIe যোগাযোগই আমাদের ৯০০ tok/s আটকে রেখেছে

Kaggle 4×L4-এ Qwen3.8-27B-এর জন্য 50k–130k কনটেক্সটে **700–1,000 tok/s sustained-এর কোনো মাপা পথ এখনো নেই, আমাদের কাছেও না, প্রকাশিত কোনো সূত্রেও না।** আমাদের সেরা মাপা decode SGLang 0.5.21 INT4 + DFlash2 W6, 20 স্ট্রিম: **597 / 536 / 364 tok/s** (8k / 40k / 90k)। vLLM 0.32 INT4 + MTP3-এর সেরা 518 / 439 / 332। পুরো production রানে আমরা পেয়েছি 254–380, আর Duck agent-এর ভেতরে 167–279 ([rev2g ফল](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md>))। মূল কারণ এখন মাপা। SGLang-এর 116 ms decode step-এর **51 % (59 ms) NCCL SHM all-reduce আর all-gather**। Step প্রায় **26 ms স্থির খরচ + প্রতি verify সারিতে 0.94 ms** হারে বাড়ে। তাই স্ট্রিম বাড়ালে খরচ ভাগ হয় না: 40k-এ vLLM 20 স্ট্রিমেই নিজের ছাদের 5 %-এর মধ্যে পৌঁছে যায় ([E14A R2 নোট](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/30 Claude - E14A SLIM R2 run results (2026-10-24).md>))। এই GCP VM-এ P2P কাজ করে, কিন্তু NCCL SHM-এর চেয়ে **5.5× ধীর**। তাই বাজারের প্রতিটি দ্রুত all-reduce এখানে হারে: FlashInfer PCIe-IPC মাপা হয়েছে 0.39–0.68×, আর 2×TP2 40k-এ 0.57× ([NCCL bench](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs/E14A SLIM R2 run1 (hosen42, 2026-10-23)/bundle/e14/nccl_bench.json>))। Darren Amadeus Martin-এর 900–980 tok/s peak আর 6 ঘণ্টায় 710 গড় শুধু দাবি। তাঁর model, engine বা পদ্ধতি অপ্রকাশিত ([Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546))। তবে সংখ্যাটি অসম্ভব নয়। আমাদের 116 ms step থেকে শুধু যোগাযোগের 59 ms সরালে যা থাকে, ছোট কনটেক্সটে তা প্রায় 900–1,000 tok/s দেয়। তাই সবচেয়ে বড় লিভার হলো যোগাযোগ কমানো: নিজেদের লেখা 4-rank host-staged all-reduce, INT8 payload arm সহ (+14–35 %)। এরপর আসে harness-এ tail আর occupancy ঠিক রাখা (কার্যকর হারে +10–25 %), vLLM-এ MTP-সহ FULL CUDA graph (#60532 বা TRITON_ATTN, +3–20 %), NCCL channel sweep (0–17 %), আর লম্বা কনটেক্সটে NVFP4 KV। আমার মডেলে সব একসাথে বসলে 8k-এ প্রায় 850–1,000, 40k-এ 650–780, আর 90k-এ 450–560 tok/s দাঁড়ায়। Custom code ছাড়া ছাদ প্রায় 550–650 (8k) আর 450–520 (40k)। পুরোনো কয়েকটি ধারণা এখন ভুল প্রমাণিত, যেমন P2P/PCIe-IPC-এর আশা, #44401 দিয়ে 11-of-12 সারানো, আর L4-এ FP8-এর kernel path।

---

## পড়ার নিয়ম: স্ট্যাটাস আর সূত্র

প্রতিটি সংখ্যার পাশে তার স্ট্যাটাস আছে। **[M]** মানে মাপা, আমাদের Kaggle log বা প্রকাশিত পরীক্ষা থেকে। **[D]** মানে মাপা সংখ্যা থেকে হিসাব করা। **[E]** মানে অনুমান, মডেল বা roofline গণিত। **[C]** মানে কোনো ব্যক্তি বা vendor-এর দাবি, পদ্ধতি ছাড়া। **[S]** মানে source code সরাসরি পড়ে পাওয়া। **[R]** মানে অযাচাইকৃত দাবি বা গুজব, যা আলাদা অংশে রাখা আছে। এই রিপোর্টের নিজস্ব নতুন হিসাব হলো "per-stream cost মডেল" (অংশ ২), যেখানে "এই রিপোর্টের হিসাব" লেখা আছে।

মূল নোট এই ফোল্ডারে: `D:/ARC AGI 2/research_notes/৪xL4 এ ৯০০ টোকেন গতির উপায়/` ([community_evidence](<D:/ARC AGI 2/research_notes/৪xL4 এ ৯০০ টোকেন গতির উপায়/community_evidence.md>), [engines_flashnext_duck](<D:/ARC AGI 2/research_notes/৪xL4 এ ৯০০ টোকেন গতির উপায়/engines_flashnext_duck.md>), [kv_capacity_and_harness](<D:/ARC AGI 2/research_notes/৪xL4 এ ৯০০ টোকেন গতির উপায়/kv_capacity_and_harness.md>), [pcie_parallelism_and_graphs](<D:/ARC AGI 2/research_notes/৪xL4 এ ৯০০ টোকেন গতির উপায়/pcie_parallelism_and_graphs.md>), [specdecode_and_kernels](<D:/ARC AGI 2/research_notes/৪xL4 এ ৯০০ টোকেন গতির উপায়/specdecode_and_kernels.md>))। এগুলোর সাথে আগের দুটি রিপোর্ট আর তাদের নোট পড়া হয়েছে। যেখানে নতুন নোট বা নতুন Kaggle মাপ আগের রিপোর্টকে বদলায়, সেখানে নতুনটিই নেওয়া হয়েছে। অংশ ৩-এ প্রতিটি সংশোধন আলাদা করে লেখা আছে। আমাদের তিনটি মূল local সূত্রকে ছোট নামে উল্লেখ করা হয়েছে: **[rev2g]** = engine matrix-এর GPU ফল (2026-10-26), **[E14A]** = E14A SLIM R2-এর ফল ও profiler (2026-10-23/24), **[bench]** = Kaggle 4×L4-এ NCCL bench। তিনটির link ওপরের অনুচ্ছেদে দেওয়া আছে।

---

## ৯০০-এর দাবি আর আমাদের ৩০০: মাপকাঠিও আলাদা, প্রসঙ্গও আলাদা

ARC-AGI-2-এ 4×L4-এ ≥600 tok/s-এর একমাত্র নির্দিষ্ট দাবি Darren Amadeus Martin-এর (2026-10-27/28) [C]। তাঁর কথাগুলো: "top speed of 900 tok/s", তারপর "980 tok/s on 1 minute harness work which includes tool call and prefill", একই concurrency-তে pure decode "like 1100 tok/s", আর "on 6 hours and 2 pass of the public evaluation, the average decode speed is 710 tok/s (due to the tail being starved of concurrency)"। prefill বেশি হলে তা "could drop to 250 tok/s"। তাঁর সংজ্ঞা হলো harness-এর মোট generated token ÷ wall time। যেমন 1 মিনিটে 60,000 generated আর 5,000 prefill token হলে তিনি তাকে 1,000 tok/s ধরেন ([Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546))। তিনি নিজের model, engine, quantization, speculative পদ্ধতি বা স্ট্রিম সংখ্যা কোনোটিই বলেননি। কৃতিত্ব দিয়েছেন "lots and lots of optimizations by claude code"-কে। একই পদ্ধতি ARC-AGI-3-এ মাত্র 4 স্কোর দিয়েছে। RTX PRO 6000-এ এটি "much faster than franzen's setup but it doesn't translate to the leaderboard"। তাঁর দলের CV 75 %-এর ওপরে থাকলেও "doesn't translate that well to LB" [C] (একই topic)। অন্য প্রতিযোগীদের সংখ্যা অনেক নিচে। TheDivine Aurora "250-400tps" বলেছেন, আর Adarsh 900-এর "half"-ও পাচ্ছেন না [C] (একই topic)।

তুলনাটি সমান মাপকাঠিতে করতে হবে। Darren-এর 980 হলো রানের শুরুতে এক মিনিটের peak, যখন কনটেক্সট ছোট। আমাদের সমতুল্য সংখ্যা 8k band-এর 597 (SGLang DFlash W6, 20 স্ট্রিম)। তাই peak-এ ব্যবধান প্রায় **1.6×**। তাঁর 6 ঘণ্টার গড় 710-এর সাথে তুলনীয় আমাদের পুরো রানের হার 329–380 (W1) আর 254–333 (vLLM FP8)। সেখানে ব্যবধান প্রায় **2×** [M/D] ([rev2g]; আগের রিপোর্ট "Qwen 27B দ্রুততম সার্ভিং সেটআপ")। Duck-এর ভেতরে আমাদের কার্যকর হার 167.8–279.0, কারণ এই engine-এর prefill মাত্র প্রায় 1,500 tok/s আর prefill চলার সময় সব decode থেমে থাকে [M] (আগের রিপোর্ট "৪xL4 ইঞ্জিন বটলনেক যাচাই")।

কেন 700 লক্ষ্য, সেটি coverage-এর হিসাব থেকে আসে। Anil Thomas-এর হিসাবে 64k গড়ে 240 টাস্কের এক pass-এর জন্য "~ 365 tok/s" লাগে, আর 20 % টাস্কে দ্বিতীয় pass দিলে "~ 440 tok/s" [E] ([Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546))। আমাদের 60k thinking floor-এ এক attempt-এ 240 × 60k = 14.4M token লাগে, যা 12 ঘণ্টায় প্রায় 333–350 tok/s। দুই attempt-এ 28.8M token, অর্থাৎ প্রায় **667–700 tok/s** [D]। Darren-এর 710 tok/s প্রায় 11.5 ঘণ্টা চললে প্রায় 29M token হয়, টাস্ক-প্রতি প্রায় 122k, যা তাঁর "2 pass"-এর সাথে মেলে [D] (community_evidence §1)। আমাদের 330–380 হারে 14–16M token হয়, অর্থাৎ প্রায় একটি 60k attempt। তাই **700 sustained মানে আমাদের floor-এ প্রতি টাস্কে দুটি স্বাধীন attempt**। আমাদের নিজের D66/L66 ডেটায় দুটি স্বাধীন attempt পরস্পরের পরিপূরক: 28 আর 34 থেকে union 50 (memory, d66-l66)। LB-র প্রেক্ষাপট হলো: Tufa Labs 88.06 শীর্ষে, 40তম Daniel Franzen 57.64, আর আমরা 34.58, rank 123 ([arc3.huikang.dev](https://arc3.huikang.dev/leaderboard/?comp=arc2))।

প্রসঙ্গও আলাদা। Anil-এর পর্যবেক্ষণ "many of the leading teams are using Qwen3.8-Flash-Next alongwith the Duck harness"। এটি score বাড়ার সময় দেখে করা অনুমান। কোনো শীর্ষ দল তা নিশ্চিত করেনি [R] ([Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546))। Duck-এর ARC-AGI-3 সেটআপে request-প্রতি output 12,288 token, 2,048-token yield, আর context প্রায় 57 Ki-তে ছাঁটা হয় ([Franzen WRITEUP](https://github.com/da-fr/arc-agi-3-solution))। এমন ছোট কনটেক্সটে decode অনেক দ্রুত। আমাদের ARC-AGI-2 নিয়ম হলো লম্বা, বিরতিহীন thinking (memory: arc-agi2-needs-long-thinking)। তাই Darren-এর সংখ্যা আমাদের 40k–90k band-এর সাথে সরাসরি তুলনীয় নয়। তাঁর দল কোন rank-এ, তাও যাচাই করা যায়নি: 10-29-এর leaderboard-এ 3 নম্বরে "Domdolus Tolus" (83.47), আর তাঁর সাথে কোনো যোগসূত্র নেই [R] (community_evidence §1)।

---

## ১১৬ ms-এর step-এর অর্ধেক PCIe-এ যায়, আর স্ট্রিম বাড়ালে তা ভাগ হয় না

### মাপা ভাগ: কোথায় যায় 116 ms

E14A R2-এর profiler-এর সেটআপ: SGLang 0.5.21, TP4, INT4 + DFlash2 W8, 12 স্ট্রিম × 8 verify সারি = 96 সারি, প্রায় 5.9k কনটেক্সট, 20 decode step, TP rank 0। চারটি rank-এর মধ্যে পার্থক্য ছিল 0.35 ms-এর কম। নিচের "52–65 ms লক্ষ্য" কলামটি এই রিপোর্টের হিসাব [E]। 900 tok/s পেতে 12 × 4.86 = 58.3 token/step-এ step প্রায় 65 ms হতে হবে। vLLM-এর 16 × 2.9 = 46.4 token/step-এ লাগবে প্রায় 52 ms।

| বিভাগ | মাপা ms/step [M] | ভাগ | প্রতি সারিতে | 52–65 ms লক্ষ্যে যা লাগবে [E] | কোন লিভার |
|---|---|---|---|---|---|
| NCCL all-reduce (140 call, call-প্রতি 983,040 B) | **50.9** | 44 % | 0.53 ms | ≤ 8–12 | host-staged AR, INT8 payload, NCCL CTA |
| NCCL all-gather (3 call, logits) | 8.1 | 7 % | 0.08 ms | ≤ 1–2 | sharded sampling |
| Marlin INT4 GEMM (512 call) | 25.3 | 22 % | — | প্রায় 25 (থাকবে) | INT4-ই সেরা; FP8-এ বাড়ে |
| অন্যান্য GEMM (BF16 drafter + lm_head) | 14.5 | 12.5 % | — | 12–14 | ছোট width |
| GDN (Triton) | 8.0 | 7 % | — | 6–8 | ছোট লিভার |
| elementwise / attention / sampling / norm | 2.9 / 2.7 / 2.4 / 1.1 | 8 % | — | প্রায় 8 | — |
| **মোট** | **116.1** (wall 116.7) | | | **≈ 57 + যোগাযোগ** | |

সূত্র: [E14A]। bench-এর পূর্বাভাস ছিল 345 µs × 140 + 8 = 56.3 ms, আর profiler তা নিশ্চিত করেছে [M]। টেবিলের মূল অর্থ সরল। **যোগাযোগ বাদে বাকি সব মিলিয়ে প্রায় 57 ms।** তাই 6k কনটেক্সটে W8 12 স্ট্রিমে যোগাযোগ পুরো সরিয়ে দিলে step প্রায় 57 ms হয়, যা 58.3 ÷ 0.057 ≈ **1,020 tok/s** দেয় [D]। যোগাযোগ 15 ms-এ নামালে প্রায় 810 [D]। Darren-এর 900–980 peak তাই আমাদের stack থেকে "যোগাযোগ প্রায় শূন্য" অবস্থার সমান। এটি কোনো প্রমাণ নয়, শুধু দেখায় যে সংখ্যাটি L4-এর পদার্থবিদ্যার বাইরে নয় [E]।

### প্রতি সারি আর প্রতি স্ট্রিমের খরচ (এই রিপোর্টের হিসাব) [D]

E14A একই engine-এ চারটি verify width মেপেছে, সবগুলো 12 স্ট্রিমে। Target-only (12 সারি) step 35.3 ms, W4 (48 সারি) 74.5 ms, W6 (72 সারি) 96.4 ms, W8 (96 সারি) 114.0 ms [M] ([E14A])। চারটি বিন্দুর least-squares fit: **step ≈ 26 ms + 0.94 ms × verify সারি**, আর প্রতিটি বিন্দুর residual ±3 ms-এর মধ্যে [D]। প্রতি সারির 0.94 ms-এর মধ্যে প্রায় 0.61 ms হলো যোগাযোগ (59 ms ÷ 96)। অর্থাৎ **প্রতি সারির খরচের প্রায় দুই-তৃতীয়াংশ PCIe**। All-reduce message-এর আকার সারির সংখ্যার সাথে প্রায় রৈখিকভাবে বাড়ে, আর 4×5060 Ti SHM-এও একই রৈখিকতা দেখা গেছে: 64 token 267.6 µs, 128 token 498.1 µs ([HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254))। তাই batch বড় করলে weight পড়ার খরচ ভাগ হয়, কিন্তু যোগাযোগের খরচ ভাগ হয় না।

[rev2g]-এর জোড়া মাপ দিয়ে একই কথা প্রতিটি stack-এ দেখা যায়। ধরি tok/s = a·N ÷ (c + s·N)। এখানে a হলো স্ট্রিম-প্রতি step-প্রতি accepted token, c স্থির খরচ, s স্ট্রিম-প্রতি খরচ। N অসীমে গেলে ছাদ দাঁড়ায় a ÷ s। vLLM-এর a = 2.9 ধরা হয়েছে (বাস্তব রানে MTP3-এর গড় প্রায় 1.9 accepted draft + 1), আর SGLang-এর a log থেকে নেওয়া। দুই-বিন্দুর fit-এ ±10–15 % অনিশ্চয়তা আছে, কারণ একই reference arm দুই রানে 572–583 দিয়েছে।

| Stack (band) | মাপা স্ট্রিম → tok/s [M] | a | c (ms) | s (ms/স্ট্রিম) | ছাদ a/s [D] |
|---|---|---|---|---|---|
| vLLM 0.32 FP8 MTP3 (8k) | 16→434, 20→452, 24→472 | 2.9 | ≈ 26 | ≈ 5.1 | **≈ 570** |
| vLLM 0.32 FP8 MTP3 (40k) | 16→399, 20→423, 24→424 | 2.9 | ≈ 20 | ≈ 6.0 | **≈ 485** |
| vLLM 0.32 INT4 MTP3 (8k) | 16→495, 20→518 | 2.9 | ≈ 21 | ≈ 4.6 | **≈ 635** |
| vLLM 0.32 INT4 MTP3 (40k) | 16→434, 20→439 | 2.9 | ≈ 6 | ≈ 6.3 | **≈ 460** |
| SGLang INT4 DFlash W6 (8k) | 16→583, 20→597 | 4.2 | ≈ 19 | ≈ 6.05 | **≈ 690** |
| SGLang INT4 DFlash W6 (40k) | 16→521, 20→536 | 4.2 | ≈ 3 | ≈ 7.7 | **≈ 545** |
| SGLang INT4 MTP 2/1/3 (8k) | 16→509, 20→542 | 2.48 | ≈ 22 | ≈ 3.5 | **≈ 715** |
| SGLang INT4 target-only (≈4–6k) | 12→339.6, 16→405 | 1 | ≈ 23 | ≈ 1.05 | ≈ 950 |

এই টেবিল থেকে তিনটি জিনিস দাঁড়ায়। প্রথমত, **40k-এ vLLM 20 স্ট্রিমেই নিজের ছাদের প্রায় 95 %-এ।** FP8-এ 20 থেকে 24 স্ট্রিমে লাভ মাত্র 423 → 424। তাই "concurrency বাড়ানো" এখন আর লিভার নয়। Darren-এর "only high speed when we increase the concurrency" কথাটি শুধু এমন stack-এ খাটে, যার s অনেক ছোট। দ্বিতীয়ত, 8k থেকে 40k-এ s বাড়ে প্রায় 1.4–1.7 ms। অর্থাৎ কনটেক্সটের প্রতি 1k token স্ট্রিম-প্রতি প্রায় 0.03–0.045 ms যোগ করে, যা fp8 KV পড়ার খরচ। 90k-এ এটি প্রায় 2.7–4 ms। তার ওপর KV pool-এর কারণে 90k band-এ আসল স্ট্রিম মাত্র 7–12 [M] ([rev2g])। তৃতীয়ত, 700 পেতে s কতটা নামতে হবে তা হিসাব করা যায় [D]। vLLM MTP3-এ 24 স্ট্রিম আর c ≈ 20 ধরলে 700-এর জন্য s ≤ 3.3 ms লাগে (40k-এ এখন 6.3), আর 900-এর জন্য s ≤ 2.4 (8k-এ এখন 4.6)। SGLang DFlash W6-এ 20 স্ট্রিমে 700-এর জন্য s ≤ 5.05 (40k-এ এখন 7.7), আর 900-এর জন্য s ≤ 3.7 (8k-এ এখন 6.05)। **দুই পথেই স্ট্রিম-প্রতি খরচ 35–50 % কমাতে হবে।** সেই খরচের সবচেয়ে বড় অংশ যোগাযোগ।

### vLLM-এর 107–112 ms: profile নেই, তাই হিসাব [E]

vLLM 0.32-এর step-এর কোনো profile আমাদের কাছে নেই। এটিই প্রথম মাপার বিষয় (pcie_parallelism §Q5)। মাপা step থেকে হিসাব হলো: FP8 MTP3 16 স্ট্রিমে 8k-এ প্রায় 107 ms, INT4 MTP3 20 স্ট্রিমে প্রায় 112 ms (specdecode §Q1)। একটি সম্ভাব্য ভাগ এরকম। যোগাযোগ 64–80 সারিতে প্রায় 34–45 ms। Marlin GEMM INT4-এ প্রায় 22–25 ms, FP8 W8A16-এ বেশি। BF16 lm_head আর ছোট projection প্রায় 10–15 ms। KV প্রায় 4–6 ms। বাকি প্রায় 15–30 ms যায় GDN, PIECEWISE mode-এর eager attention, sampling, MTP head আর host-এ। প্রতি সারির হিসাবে vLLM-এর s/সারি প্রায় 1.15–1.25 ms, আর SGLang-এর প্রায় 0.94–1.0 [D]। **এই প্রায় 0.2–0.3 ms/সারির পার্থক্যই সর্বোচ্চ সেই অংশ, যা FULL CUDA graph আর host overhead কমিয়ে ফেরত পাওয়া যেতে পারে।** এটি আসলে ঊর্ধ্বসীমা, কারণ পার্থক্যের সবটাই graph-এর জন্য নয়।

### Communication-only ছাদ: verify সারি-প্রতি accepted token-ই আসল মাপকাঠি [D]

যোগাযোগের খরচ সারি-প্রতি হওয়ায়, অন্য সব খরচ শূন্য ধরলেও একটি ছাদ থাকে: tok/s ≤ (accepted token ÷ সারি) ÷ (0.61 ms)। হিসাবগুলো এরকম:

| পদ্ধতি | সারি-প্রতি token | যোগাযোগ-একা ছাদ |
|---|---|---|
| DFlash W8, accept 4.86 | 0.61 | **≈ 1,000** |
| DFlash W6, accept 4.2 | 0.70 | ≈ 1,150 |
| DFlash W4, accept 3.3 | 0.82 | ≈ 1,350 |
| MTP3, accept 2.9 | 0.725 | ≈ 1,190 |
| No-spec | 1.0 | ≈ 1,640 |

এর অর্থ, চওড়া draft প্রত্যাখ্যাত সারির জন্যও PCIe খরচ দেয়। এই কারণেই [rev2g]-এ W8 40k-এ হেরেছে (477 বনাম W6-এর 521), আর W4 ও W6 সমান। যোগাযোগ কমানো গেলে এই অনুপাত কম গুরুত্বপূর্ণ হবে, আর তখন আবার বেশি accept-এর পদ্ধতি জিতবে [E]।

---

## নতুন প্রমাণে যে পুরোনো ধারণাগুলো ভাঙল

| আগের ধারণা (উৎস) | নতুন প্রমাণ | সংশোধন |
|---|---|---|
| L4-এ block-128 FP8 চলে Triton block kernel-এ (vLLM 0.19); FP8 Marlin "Ampere-এর পথ" (রিপোর্ট "Qwen 27B দ্রুততম…") | vLLM 0.32-এর L4 log: `Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod` [M] ([E21 log](<D:/ARC AGI 2/experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/Experiment Results/SMOKE_GPU_run1_2026-10-29/nb2/q27x/halfB/vllm_server_halfB_primary.log>)); PR #61224 নিজেও বলে SM89-এ block-FP8 "currently run through Marlin weight-only" ([vLLM #61224](https://github.com/vllm-project/vllm/pull/61224)) | **0.32-এ FP8 = W8A16 Marlin।** এর কোনো compute সুবিধা নেই, আর INT4-এর দ্বিগুণ byte পড়ে। মাপা INT4 20 স্ট্রিমে 518 বনাম FP8 452 (8k)। CUTLASS W8A8 SM89 এখনো খোলা PR (#61224, L4 Qwen3-8B 1,929 → 2,739 tok/s, compute-bound shape-এ) |
| fp16 SSM মাপা নিরাপদ পথ, কোনো গতির খরচ নেই | vLLM 0.32 source: fused CUDA MTP verify kernel-এর জন্য state dtype `(float32, bfloat16)` লাগে [S] ([source](<D:/ARC AGI 2/_scratch/engine_audit_pack_rev2e/source_refs/vllm_0.32.0/vllm_model_executor_layers_mamba_gdn_qwen_gdn_linear_attn.py>)); E21 log: "Falling back to the Triton GDN spec decode path" [M] | **fp16 সত্যিই fused kernel বন্ধ করে।** তবে L4-এ এর খরচ মাপা অনুযায়ী প্রায় 0। `--linear-backend triton` arm 438/398/283 বনাম ref 434/399/281, আর fp16 SSM arm বরং দ্রুত: 450/414/312, কারণ XL-এ 11টি স্ট্রিম ধরে ([rev2g])। তাই fused kernel L4-এ বড় লিভার নয়। Prefill মেশানো প্রতিটি step-ও এমনিতে Triton-এ যায় |
| CUDA graph আগে থেকেই চালু, বাকি লাভ প্রায় শূন্য | vLLM FlashInfer + spec-এ `cudagraph_mode=PIECEWISE` [M]; L4-এ 0.30-এ PIECEWISE ≈ eager, 1.01× ([vLLM #59548](https://github.com/vllm-project/vllm/issues/59548)); PR #60532 merge হয়েছে 2026-10-22-এ, যা pre-SM90-এ FlashInfer-কে `UNIFORM_BATCH` দেয়, batch 1-এ +36–47 % ([vLLM #60532](https://github.com/vllm-project/vllm/pull/60532)); TRITON_ATTN = `ALWAYS` ([vLLM CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)) | **vLLM-এ MTP-সহ FULL graph এখন সম্ভব:** nightly ≥ 10-22, বা 0.32.0-এ cherry-pick, বা TRITON_ATTN। লাভের পরিসর বিতর্কিত, +3–20 % (অংশ ৪)। SGLang পুরো step আগেই capture করে, তাই সেখানে এই লিভার খাটে না |
| All-reduce অমাপা, হিসাবে 30–45 % (35–55 ms) | Profiler: AR 50.9 + AG 8.1 = **59 ms of 116 (51 %)** [M] ([E14A]) | **নিশ্চিত, আর আগের হিসাবের চেয়েও বড়।** Step বাজেটের সবচেয়ে বড় অংশ এটাই |
| Kaggle-এ P2P অজানা; `NCCL_P2P_LEVEL=SYS/PHB` বা FlashInfer PCIe-IPC সবচেয়ে বড় communication লিভার (+20–45 %) | Topology PHB, `can_device_access_peer` = True; কিন্তু P2P/IPC 983 KB-এ 1,880.5 µs বনাম SHM 343.9 µs, 64 MiB busbw 0.42 বনাম 4.90 GB/s [M] ([bench]); PCIe-IPC: SGLang DFlash 0.39×, MTP 0.42×, vLLM 0.68× [M] ([rev2g]); `NCCL_SHM_USE_CUDA_MEMCPY=1` graph capture-এ hang করেছে [M] | **এই VM-এ P2P-ভিত্তিক সব পথ বাদ।** GCP-র virtualized P2P পথ ধীর। দ্রুত all-reduce-কে host memory-র মধ্য দিয়ে যেতে হবে, কম byte পাঠাতে হবে, অথবা layout বদলাতে হবে |
| PCIe link Gen4 x16 | VM-এর ভেতরে nvidia-smi দেখায় `pcie.link.gen.max = 3`, width 16 [M] ([bench]) | সীমা আসলে Gen3-শ্রেণির ভার্চুয়াল link, বা field-টি ভুল রিপোর্ট করা। host-link-এর হিসাবে প্রায় 12.4–13.2 GB/s ধরতে হবে |
| #44401 backport 11-of-12 latch সারাবে, +3–8 % | GPU-তে backport 508.1 বনাম 505.0 tok/s, churn test-এ 95.6 % সময় এখনো 11-এ আটকে [M] ([E14A]) | **#44401 এটি ঠিক করে না।** আসল কারণ delayer। `--min-free-slots-delay 1` দিলে 13 running-এ DFlash buffer crash করে (`size of tensor a (12) must match … (13)`) |
| 2×TP2 INT4 "সবচেয়ে বড় অপরীক্ষিত লিভার" (প্রথম রিপোর্ট), পরে "প্রায় বাতিল" | মাপা: SGLang DFlash W6 2×TP2 8k-এ 658 (1.15×) কিন্তু 40k-এ 308 (0.57×), replica-প্রতি 152k KV; vLLM 2×TP2 40k-এ 22/92 [M] ([rev2g]) | **বাতিল, এখন মাপা প্রমাণে।** |
| DFlash width 4/6: −10 % থেকে +12 % (মডেল) | W6 +6.3 %, W4 +5.3 % (6k, gate PASS) [M] ([E14A]); rev2g-এ W4 = W6 = ref, W8 40k-এ হারে [M] | ছোট width সামান্য এগিয়ে। কারণটি ওপরের সারি-খরচ মডেল |
| Flash-Next-এর "6B active", তাই L4-এ দ্রুত | প্রতি rank 19.06–19.10 GB, KV pool মাত্র **94,528–118k token** [M] (E3/E13 LOADPROBE); batch 16-এ প্রতি layer-এর প্রায় 27 % expert পড়া হয় [E]; output "!!!!" | **4×L4-এ লম্বা reasoning-এর জন্য Flash-Next বাস্তবসম্মত নয়।** 60k-এর বেশি কনটেক্সটের মাত্র 1–2টি স্ট্রিম ধরে |
| "NVFP4 L4-এ চলে না" | Patched vLLM 0.30, 3×RTX 4090 48 GB (SM89): Flash-Next NVFP4 expert MARLIN দিয়ে চলে, 16 স্ট্রিমে 635.3 tok/s, কিন্তু প্রায় 28 GiB/card লাগে, "24GB cards: OOM at load" ([chenxin199305](https://github.com/chenxin199305/qwen38-flash-next-3x4090)) | SGLang আর native FP4 GEMM-এর জন্য কথাটি সত্য, patched vLLM Marlin-এর জন্য নয়। L4-এ আসল বাধা মেমরি |
| vLLM MTP scheduler-এর 3-sequence cap (#55533) আমাদের আটকাতে পারে | আমাদের 0.32-এর রান বাস্তবে 16/20/24 running-এ পৌঁছেছে [M] ([rev2g]) | আমাদের pool-এ এটি বাঁধে না |
| High batch-এ dynamic SD বা spec বন্ধ করা লাভজনক | #49548: `num_speculative_tokens_per_batch_size` 8 concurrent-এ 232 থেকে 24–157 tok/s-এ নামিয়েছে; fix-গুলো unmerged ([vLLM #49548](https://github.com/vllm-project/vllm/issues/49548)) | **ব্যবহার করা যাবে না।** Static K রাখতে হবে |
| Darren-এর দল "3rd place" | #3 Domdolus Tolus (83.47); কোনো যোগসূত্র নেই (community_evidence §1, §5) | অযাচাইকৃত |

---

## র‍্যাঙ্ক করা লিভার: স্ট্রিম-প্রতি খরচ কমানোই একমাত্র পথ

নিচের তালিকা প্রত্যাশিত লাভের ক্রমে সাজানো, সবচেয়ে বড় প্রথমে। "লাভ" আমাদের workload-এর জন্য [E], baseline হিসেবে vLLM 0.32 INT4 MTP3 20 স্ট্রিম (518/439/332) বা SGLang DFlash W6 (597/536/364) ধরে। পরে প্রতিটি লিভারের বিস্তারিত দেওয়া আছে।

| # | লিভার | প্রত্যাশিত লাভ [E] | Accuracy ঝুঁকি | Offline sm_89 |
|---|---|---|---|---|
| 1 | নিজেদের 4-rank host-staged graph-capturable all-reduce (bf16, তারপর INT8 arm) | bf16 +14–17 %; INT8 +30–35 % (8k), 40k-এ +20–30 % | bf16: যোগের ক্রম বদলায় (bit-exact নয়); INT8: সত্যিকারের ঝুঁকি | হ্যাঁ, কিন্তু code লিখতে হবে (কয়েক দিন), nvcc লাগবে |
| 2 | Harness occupancy: tail refill + KV-ভিত্তিক admission + `--max-num-active-seqs` | কার্যকর হারে +10–25 % (decode-এ 0) | নেই (মডেল যা দেখে তা বদলায় না) | হ্যাঁ, client code |
| 3 | vLLM-এ MTP-সহ FULL CUDA graph (#60532 nightly / TRITON_ATTN) | +3–20 % (বিতর্কিত) | কম; greedy gate লাগবে (#42271 deadlock) | হ্যাঁ: nightly wheel বা Python cherry-pick |
| 4 | NCCL channel sweep (`NCCL_MIN_CTAS`/`MAX_CTAS` 4/8, `NCCL_NTHREADS=512`) | 0 থেকে +17 % | নেই (একই NCCL যোগফল) | হ্যাঁ, শুধু env |
| 5 | vLLM-এ FP8-এর বদলে INT4 W4A16 | 8k +14 %, 40k +4 %, XL +18 % [M] | INT4-এ প্রায় −1.3 pt (আগের হিসাব); vLLM-এ INT4 oracle নেই | হ্যাঁ, 0.32-এ (0.19-এ garbage) |
| 6 | NVFP4 KV (vLLM ≥ 0.32.0, #46963) | 8k −15 % থেকে +5 %; XL −15 % থেকে +25 % | মাঝারি; ARC gate লাগবে | হ্যাঁ, 0.32-এ আছে; SM89-এ অপরীক্ষিত |
| 7 | Re-prefill কমানো: token-identical history, `--max-num-batched-tokens` 8192–16384, SGLang streaming session; prefill-এর জন্য #61224 | Duck-এ কার্যকর হারে +5–15 % | কম | আংশিক: #61224 source build চায় |
| 8 | Sharded sampling: logits all-gather-এর বদলে per-shard top-k | +5–7 % | sampling-এর distribution সমান রাখতে হবে | engine patch |
| 9 | DFlash width 6/4 বা MTP 2 (static) | 0 থেকে +6 % [M] | কম | হ্যাঁ, flag |
| 10 | Two-micro-batch overlap (comm বনাম GEMM) | +10–30 % | মাঝারি | বড় engine পরিবর্তন |
| 11 | vLLM TP2×PP2, V2 runner + MTP (#46994) | 0 থেকে +20 % | GDN rollback-এ PP-র ঝুঁকি | release-এ আছে কি না অযাচাইকৃত |
| বাদ | P2P level, CE-memcpy SHM, PCIe-IPC, 2×TP2, DP, live concurrency-র জন্য HiCache/offload, TurboQuant, dynamic SD, SGLang fp4 KV, 4×L4-এ Flash-Next | মাপা ঋণাত্মক বা প্রয়োগযোগ্য নয় | — | — |

### ১. Host-staged 4-rank all-reduce: একমাত্র নকশা যা এই VM-এ খাটে

**কী:** এই VM-এ P2P ধীর, কিন্তু GPU↔host পথ দ্রুত (g2-standard-48-এ প্রায় 13.2 GB/s GPU→host আর 12.4 GB/s host→GPU, Spare Cores-এর মাপ, আগের nccl_p2p_l4 নোটে উদ্ধৃত)। Ring all-reduce n = 4-এ প্রতিটি GPU থেকে 1.5 × S বাইরে যায় আর ভেতরে আসে। S = 983,040 B হলে প্রতি দিকে 1.47 MB, যার link-floor প্রায় **115–120 µs**। NCCL SHM মাপা হয়েছে 343.9 µs-এ, অর্থাৎ floor-এর মাত্র প্রায় 35 % দক্ষতায় [D] (pcie_parallelism §Q1)। প্রস্তাবটি এরকম: pinned, mapped host memory-তে two-shot all-reduce (reduce-scatter + all-gather layout), sequence-numbered flag সহ, যা CUDA graph-এ capture করা যায়। এর ভিত্তি SGLang PR #39605-এর LL kernel (`SGLANG_LL_ALLREDUCE=1`, host-mapped 8-byte store, graph-capturable)। কিন্তু সেই PR শুধু world_size 2-এর জন্য, unmerged, আর তার bulk path (≥ 1 MiB) graph-capturable নয় ([SGLang #39605](https://github.com/sgl-project/sglang/pull/39605))। আমাদের 0.5–1 MB message দুটি পথের মাঝখানে পড়ে, তাই 4-rank version আমাদেরই লিখতে হবে। এটি বসবে vLLM-এর `cuda_communicator` dispatch-এ PyNCCL-এর আগে, বা SGLang `parallel_state`-এ যেখানে `pcie_ipc_ar` dispatch হয়। একটি পরীক্ষামূলক env নাম যেমন `ARC_HSAR=1 ARC_HSAR_DTYPE=bf16|int8` রাখা যায়। এটি আমাদের নিজের নাম, upstream flag নয়।

**প্রমাণ:** PR #39605 TP2-তে 1.31 MB 142 µs, 12.90 MB 946 µs, আর NCCL SHM 20.97 MB-এ 10,859 µs বনাম 1,494 µs মেপেছে। 2×5070 Ti-তে Qwen3.5-35B-A3B decode graph সহ 64.7 থেকে 71.0 tok/s (+10 %) [M] ([SGLang #39605](https://github.com/sgl-project/sglang/pull/39605))। INT8 payload-এর ভিত্তি Flash Communication: INT8 group-128 LLaMA-3-70B zero-shot-এ 80.00 বনাম FP16 80.00, আর INT4-এ 78.29। এটি 4×L40 PCIe-তে পরীক্ষিত, কিন্তু decode মাপা হয়নি আর P2P DMA ব্যবহার করে [M] ([arXiv 2412.04964](https://arxiv.org/html/2412.04964))। ROCm QuickReduce একই ধারণার INT8 group-32 রূপ, শুধু MI300-এ ([vLLM quick_all_reduce](https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/quick_all_reduce/))।

**আমাদের জন্য লাভ [E]:** bf16 two-shot-এ call-প্রতি প্রায় 220–250 µs (−30 %), এই শর্তে যে চারটি GPU একসাথে host DRAM-এ লিখলে মোট প্রায় 50 GB/s DRAM বাঁধা পড়ে না। step-প্রতি সাশ্রয় প্রায় 14–17 ms, অর্থাৎ **+14–17 %**। INT8-এ call-প্রতি প্রায় 130–150 µs, সাশ্রয় প্রায় 28–30 ms, অর্থাৎ **+30–35 %**। সারি-মডেলে INT8 SGLang-এর s/সারি 0.94 থেকে প্রায় 0.58-এ নামায়। তখন DFlash W6 20 স্ট্রিমে 8k-এ প্রায় 850–900 আর 40k-এ প্রায় 700। **Accuracy:** bf16 two-shot-এ যোগের ক্রম NCCL ring-এর থেকে আলাদা, তাই output bit-exact হবে না, তবে গাণিতিকভাবে সমান। INT8-এর ঝুঁকি বাস্তব, কারণ 64-layer reasoning মডেলের residual stream-এ outlier থাকে, আর 60k+ token-এ ছোট বিচ্যুতিও জমতে পারে। তাই INT8 একটি মাপা arm, default নয়। এর জন্য greedy parity, logprob drift আর EVAL subset লাগবে। **30 মিনিটের যাচাই:** কোনো server লাগবে না। বিদ্যমান `e14_nccl_bench.py` harness-এ kernel compile করে 983,040 B-এর message 140 বার graph replay করতে হবে, সাথে sum = 10.0 correctness পরীক্ষা আর 11.9 MB all-gather। লক্ষ্য: call-প্রতি ≤ 250 µs (bf16) বা ≤ 150 µs (INT8)। তারপর SGLang W6-এ 90 s-এর দুটি window (8k, 40k) আর greedy gate। **Offline sm_89:** সম্ভব। torch cpp_extension বা আগে-compile করা `.so` payload dataset-এ পাঠানো যায় (`TORCH_CUDA_ARCH_LIST=8.9`, image-এ nvcc 13.0.88 pinned আছে)। তবে এটি কয়েক দিনের kernel কাজ, আর deadline-এর আগে hang-guard (timeout সহ spin) অপরিহার্য।

### ২. Harness occupancy আর tail: Darren নিজেই 980 থেকে 710-এ নেমেছেন এখানে

**কী:** লাইভ স্ট্রিম যখন লক্ষ্যের নিচে নামে (যেমন 24-এর নিচে), তখন সাথে সাথে পরের টাস্কের attempt বা কোনো অসমাপ্ত টাস্কের দ্বিতীয় স্বাধীন attempt শুরু করতে হবে। শর্ত হলো Σ(admitted request-এর প্রত্যাশিত peak context) ≤ 0.9 × KV pool। vLLM ≥ 0.31-এ `--max-num-active-seqs <admission cap>` দিতে হবে, আর `--max-num-seqs` 8-এর গুণিতকে রাখতে হবে (#57355 cliff) (kv_capacity §4–5)। **প্রমাণ:** Darren-এর নিজের peak থেকে গড়ে পতন 980 → 710 (−28 %), "due to the tail being starved of concurrency" [C] ([Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546))। আমাদের G4B-এ 27.7 % সময় মাত্র 6 running-এ কেটেছে, আর 11 বনাম 12 running-এ প্রায় −9 % [M] (আগের রিপোর্ট)। ARC-AGI-3-এ KV-র চেয়ে বেশি স্ট্রিম দিলে throughput 600–620 থেকে প্রায় 280-এ নেমেছিল, preemption-এর কারণে [M] ([Kaggle 750885](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/750885))। **লাভ:** decode হার বাড়ে না, কিন্তু কার্যকর হার +10–25 % বাড়ে। দ্বিতীয় attempt দিয়ে tail ভরলে accuracy-ও বাড়ার কথা (D66/L66)। **ঝুঁকি:** নেই, শর্ত শুধু preemption এড়ানো। **যাচাই:** আগের রানের রেকর্ড করা metric-এর ওপর controller replay, CPU-তে। memory-র নিয়ম অনুযায়ী নিজেদের বানানো "healthy" fixture নয়। GPU smoke-এ real running == লক্ষ্য assert করতে হবে। **Offline:** সম্পূর্ণ।

### ৩. vLLM-এ MTP-সহ FULL CUDA graph

**কী, দুটি পথ।** (ক) vLLM nightly ≥ 2026-10-22, অথবা 0.32.0-এ PR #60532 (commit `71e9320`) cherry-pick। এটি শুধু Python পরিবর্তন, `flashinfer.py`-এ। এটি uniform multi-token verify query-কে FlashInfer fa2 decode পথে পাঠায় আর pre-SM90-এ `UNIFORM_BATCH` রিপোর্ট করে। reviewer-এর পর config field-এর নাম `use_fa2_native_decode`, তাই install-এর source-এ default আর নাম যাচাই করতে হবে। সাথে দিতে হবে `--attention-backend FLASHINFER --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`। (খ) stock 0.32.0-এ `--attention-backend TRITON_ATTN --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`। Triton attention `ALWAYS` declare করে, আর যেকোনো head size-এ fp8 KV নেয় ([vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/))।

**প্রমাণ:** #60532-এর লেখক SM80-এ batch 1-এ Qwen3.8-27B MTP3 TP1-এ 73.6 → 108.1, আর TP2-এ 61.0 → 83.1 tok/s মেপেছেন, output token হুবহু এক ([vLLM #60532](https://github.com/vllm-project/vllm/pull/60532))। কিন্তু fp8 KV, head_dim 256 আর batch > 1 পরীক্ষা হয়নি। বিপক্ষে প্রমাণ আছে: GB10-এ drafter-এর FULL-graph prototype conc 16-এ মাত্র প্রায় 1.3 % e2e দিয়েছে ([vLLM #45258](https://github.com/vllm-project/vllm/issues/45258))। আর PIECEWISE mode-এ all-reduce আগেই captured piece-এর ভেতরে থাকে। **লাভ [E]: +3–20 %।** নিম্নসীমা আসে #45258 থেকে। ঊর্ধ্বসীমা আসে vLLM বনাম SGLang-এর 0.2–0.3 ms/সারি পার্থক্য থেকে: 80 সারিতে 16–24 ms of 112। **ঝুঁকি:** MTP k=1 + `FULL_AND_PIECEWISE` bonus-token shape-এ deadlock করেছিল (v0.20.1), যার fallback `FULL_DECODE_ONLY` ([vLLM #42271](https://github.com/vllm-project/vllm/issues/42271))। MTP + DCP-তে নিঃশব্দ corruption-এর রিপোর্টও আছে ([vLLM #45425](https://github.com/vllm-project/vllm/issues/45425)), যদিও আমরা DCP ব্যবহার করি না। greedy gate লাগবে। **যাচাই:** একই notebook-এ দুটি launch। log-এ "setting cudagraph_mode=PIECEWISE" লাইন যেন না থাকে, আর GDN backend কোন mode নামায় তা দেখতে হবে। 8k আর 40k-এ 90 s window, 8টি prompt-এ greedy, আর `cudagraph_capture_sizes`-এ 1+K-এর প্রতিটি গুণিতক আছে কি না। **Offline:** nightly wheel wheelhouse-এ আনতে হবে, অথবা 0.32.0-এ এক ফাইলের patch। TRITON_ATTN arm আজই চালানো যায়।

### ৪. NCCL channel sweep: সবচেয়ে সস্তা পরীক্ষা

**কী:** আমাদের NCCL মাত্র 2টি collective channel ব্যবহার করে (`coll channels 2`, transport `SHM/direct/direct`) [M] ([bench])। প্রতিটি SHM channel একটি CTA, যা mapped host memory-তে PCIe load/store করে। দুটি CTA x16 link-এর দুই দিক ভরাতে পারে না, এটাই সম্ভাব্য কারণ [E]। Sweep-এর arm: `NCCL_MIN_CTAS=4 NCCL_MAX_CTAS=4`, `=8/8`, `NCCL_NTHREADS=512`, আর দ্বিতীয় স্তরে `NCCL_ALGO=Ring|Tree`, `NCCL_BUFFSIZE=1048576|8388608`, `NCCL_CUMEM_ENABLE=1`। **যা দেওয়া যাবে না:** `NCCL_P2P_LEVEL`, `NCCL_SHM_USE_CUDA_MEMCPY` আর `NCCL_SHM_DISABLE`, কারণ এগুলো মাপা ঋণাত্মক বা hang করে ([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html))। **প্রমাণ:** RTX PRO 6000 PCIe-তে `NCCL_MIN_NCHANNELS=8`-কে "significant bandwidth improvement" বলা হয়েছে, কিন্তু কোনো সংখ্যা নেই [C] ([rtx6kpro](https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/sglang.md))। 4×A10G-তে `MIN/MAX_CTAS=1` একটি regression ফিরিয়েছিল [M] ([NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298))। বেশি CTA ছোট message-এ ক্ষতিও করতে পারে [C]। **লাভ [E]:** 0 থেকে −20 ms/step, অর্থাৎ 0 থেকে +17 %। এমনকি নিখুঁত NCCL (120 µs) হলেও যোগাযোগ 59 থেকে প্রায় 20 ms-এ নামবে, তাই ঊর্ধ্বসীমা প্রায় 1.4× step। **ঝুঁকি:** নেই। **যাচাই:** `e14_nccl_bench.py`-এ 2–3 মিনিট, server ছাড়া। **Offline:** শুধু env। এই A/B লিভার ১-এর আগে চালানো উচিত, কারণ CTA দিয়ে floor-এর কাছে পৌঁছানো গেলে custom kernel লাগবেই না।

### ৫. vLLM-এ INT4: FP8 এখানে W8A16, তাই দ্বিগুণ byte বৃথা

**কী:** `vllm serve <RedHat Qwen3.8-27B INT4>` vLLM 0.32-এ (0.19-এ INT4 garbage output দিয়েছিল, E21 run 1)। **প্রমাণ:** INT4 MTP3 20 স্ট্রিমে 518/439/332 বনাম FP8 20 স্ট্রিমে 452/423/281। XL-এ INT4 12টি স্ট্রিম ধরে, FP8 ধরে 10টি [M] ([rev2g])। **লাভ:** +14 % (8k), +4 % (40k), +18 % (XL), আর KV pool 1.35M থেকে 1.71M token। **ঝুঁকি:** আগের রিপোর্টে INT4 প্রায় −1.3 pt ধরা হয়েছিল। rev2g-এ vLLM INT4-এর কোনো oracle ছিল না ("diagnostic"), তাই একটি target-only greedy gate আর EVAL subset লাগবে। আমাদের LB প্রমাণ শুধু FP8-এর (S1 34.58, আসল স্তর প্রায় 32)। **যাচাই:** একটি boot, 3টি window, INT4 target-only oracle-এর বিপক্ষে greedy, মোট প্রায় 20 মিনিট। **Offline:** 0.32 wheelhouse-এ আছে।

### ৬. NVFP4 KV: লম্বা কনটেক্সটের ধারণক্ষমতা, তবে SM89-এ দাম আছে

**কী:** vLLM ≥ 0.32.0-এ `--kv-cache-dtype nvfp4`। এটি FlashInfer FA2-এর native prefill/decode wrapper দিয়ে চলে (#46963, merge 2026-10-07) ([vLLM #46963](https://github.com/vllm-project/vllm/pull/46963))। **প্রমাণ:** একমাত্র sm_89 মাপ একটি fork-এর (vLLM 0.32): KV 429,319 থেকে 750,747 token (×1.75), কিন্তু N=48-এ aggregate 886.4 থেকে 767.5 (−13 %) আর N=1-এ 41.1 থেকে 33.9 (−17 %)। quality gate 11/11। fork নিজেই এটি বাতিল করেছে, কারণ তাদের occupancy ছিল মাত্র 2–7 % ([jsboige #123](https://github.com/jsboige/vllm/pull/123))। PR-এর কথা: "on SM12x and SM8x, NVFP4 groups capture only single-token decode graphs"। তাই MTP-এর verify step সম্ভবত graph ছাড়া চলবে, যা লিভার ৩-এর সাথে বিরোধী। Qwen3.8-27B-এ accuracy (Blackwell `trtllm_mha` kernel, FA2 নয়): GSM8K −0.31, GPQA −1.01, AIME সমান 98.33 %, SWE −1.6 pt ([LMSYS NVFP4 KV](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache/))। **লাভ [E]:** KV-র অংশ শুধু 16টি full-attention layer-এ, তাই INT4 pool 1.71M থেকে প্রায় 3.0M token হয়। 90k band-এ এখন মাত্র 7–12টি স্ট্রিম চলে। সেখানে স্ট্রিম 20-এ উঠলে আর KV-পাঠ প্রায় 0.56× হলে XL-এ 332 থেকে প্রায় 400–420 হতে পারে। কিন্তু fork-এর −13–17 % kernel দাম পড়লে লাভ প্রায় শূন্য। 8k-এ ধারণক্ষমতা দরকার নেই, তাই সেখানে প্রায় −15 % থেকে +5 %। **ঝুঁকি:** মাঝারি। FA2 NVFP4 পথের accuracy শুধু sm_80-এর ছোট মডেলে মাপা (Qwen3-4B-এ GSM8K −4 pt), তাই T=0-এ fp8 KV-র বিপক্ষে ARC gate বাধ্যতামূলক। **যাচাই:** boot করে log থেকে KV token সংখ্যা আর cudagraph mode পড়া, তারপর 90k-এ সর্বোচ্চ স্ট্রিমে window আর greedy gate, মোট প্রায় 25 মিনিট। **Offline:** 0.32.0-এ আছে, তবে SM89-এ upstream অপরীক্ষিত ("SM86, SM89, SM120 and performance are untested")।

### ৭. Re-prefill কমানো: 4×L4-এ prefix reuse-এর দাম 3–4 গুণ

**কী:** Duck turn-এ আগের reply আবার prefill হয়। আগের রিপোর্টে দেখা গেছে, আমাদের ভালো রানেও 72–91 % hit ঠিক আগের prompt-এর শেষে থেমে যায় [M]। vLLM align mode কেবল prompt-এর শেষ block boundary-তে Mamba state রাখে, reply-এর শেষে নয়। #50551 (`--enable-mamba-decode-checkpoint`) এক 5-turn পরীক্ষায় hit rate 63.6 থেকে 94.7 %-এ তুলেছে, কিন্তু PR খোলা আর "rejects hidden-state speculative decoding", তাই MTP-র সাথে চলে না ([vLLM #50551](https://github.com/vllm-project/vllm/pull/50551))। যা এখনই করা যায়: (ক) history token-identical রাখা, অর্থাৎ generated id হুবহু ফেরত দেওয়া; (খ) `--max-num-batched-tokens` 4096 থেকে 8192–16384, যাতে 90k re-prefill প্রায় 22টি mixed step-এর বদলে 6–11টিতে শেষ হয়। 25K-TPS recipe-তে 16384-এ +8 % দেখা গেছে (kv_capacity §5)। (গ) SGLang-এ `--enable-streaming-session`, যা session-এর KV আর Mamba state eviction থেকে আড়ালে রাখে। (ঘ) prefill গতির জন্য PR #61224 (CUTLASS block-FP8 W8A8 SM89, L4-এ Qwen3-8B 1.42×), যদিও এটি FP8 checkpoint-এর জন্য আর source build চায় ([vLLM #61224](https://github.com/vllm-project/vllm/pull/61224))। **সতর্কতা:** prefix caching কখনো বন্ধ করা যাবে না। MTP-সহ তা বন্ধ করলে correctness ভাঙে (E19), আর 40k-এ গতি 3 tok/s-এ নামে [M] ([rev2g])। align mode-এর 11–16 % খরচ ([vLLM #60008](https://github.com/vllm-project/vllm/issues/60008)) তাই মেনে নিতে হবে। **লাভ:** Duck-এ কার্যকর হারে +5–15 % [E]। আগের রিপোর্টের G4B scenario ছিল 279 → 325 → 357। **যাচাই:** অফলাইন token-diff (CPU), তারপর একটি ছোট Duck window। **Offline:** (ক), (খ), (গ) সম্পূর্ণ, (ঘ) দুর্বল।

### ৮–১১. ছোট বা ঝুঁকিপূর্ণ লিভার

**Sharded sampling (৮):** E14A-তে logits all-gather 8.1 ms, 3টি call-এ rank-প্রতি 11.9 MB। Greedy verify-এর জন্য শুধু shard-প্রতি (max, index) লাগে। temperature sampling-ও Gumbel-max দিয়ে shard-এ ঠিকভাবে করা যায়, তারপর সারি-প্রতি কয়েক byte all-gather করলেই হয় (pcie_parallelism §Q5)। লাভ −5–7 ms, অর্থাৎ +5–7 %। Upstream-এ এমন feature নেই, তাই engine patch আর distribution-equivalence test লাগবে। **Width/MTP shape (৯):** static K রাখতে হবে। DFlash W6 বা W4 (gate PASS), vLLM-এ MTP2 ≈ MTP3 > MTP4 (448/397/260 বনাম 434/399/281 বনাম 399/381/267) [M] ([rev2g])। `--speculative-num-draft-tokens 6` বা `{"method":"mtp","num_speculative_tokens":2}`। যোগাযোগ কমানোর পরে আবার sweep করতে হবে। **Two-micro-batch overlap (১০):** 96 সারির batch দুই ভাগ করে আলাদা stream-এ চালানো, যাতে একটির all-reduce অন্যটির GEMM-এর আড়ালে পড়ে। আদর্শ ক্ষেত্রে প্রায় 50 ms লুকানো যায়, কিন্তু Marlin দুবার weight পড়ে (+15–25 ms), তাই নিট −10–30 ms [E]। Ada PCIe-তে কোনো প্রকাশিত implementation নেই। SGLang আর vLLM দুটোই পুরো step একটি stream-এ capture করে, তাই এটি বড় কাজ। **TP2×PP2 (১১):** `--tensor-parallel-size 2 --pipeline-parallel-size 2 --disable-custom-all-reduce` আর `VLLM_USE_V2_MODEL_RUNNER=1`, MTP k=3। PR #46994 merge হয়েছে 2026-09-11, কিন্তু কোন release-এ, তা নিয়ে তথ্য পরস্পরবিরোধী ([vLLM #46994](https://github.com/vllm-project/vllm/pull/46994))। হিসাবে প্রতি GPU-র যোগাযোগ প্রায় 3.3× কমে, KV pool প্রায় একই থাকে, কিন্তু weight বাড়তি পড়তে হয়। তাই 1.1–1.2× [E]। Qwen3.8-27B-এর একমাত্র PP মাপ single-stream, সেখানে PP2 হারে: 32.9 বনাম TP2 48.9 tok/s ([Houtini](https://houtini.com/articles/the-dual-4090-96gb-vllm-benchmark-runbook/))। SGLang DFlash `pp_size == 1` ছাড়া চলে না।

### বাদ দেওয়ার কারণ

P2P level, CE-memcpy SHM আর PCIe-IPC মাপা ঋণাত্মক (অংশ ৩)। 2×TP2 আর DP replica KV-র কারণে 40k-এ 0.57×। Torch symm-mem log করে "Device capability 8.9 not supported", আর FlashInfer AR log করে "not supported for world_size=4" [M] (pcie_parallelism)। CPU KV offload live concurrency বাড়ায় না, কারণ decode চলার সময় KV GPU-তে থাকতেই হয়। Hybrid-এ offload "silently write-only" হয়ে যেতে পারে (#58653), আর একটি fork-এ tier fetch block করে decode 0.3–0.9 t/s-এ নামিয়েছিল ([jsboige #128](https://github.com/jsboige/vllm/pull/128))। TurboQuant-এর পরিচিত crash agent re-prefill-এর ঠিক পথে ([vLLM #41726](https://github.com/vllm-project/vllm/issues/41726))। SGLang `fp4_e2m1` KV B200-এ প্রায় 90 tok/s দিয়েছে, যেখানে auto-তে প্রায় 5,000 ([SGLang #17365](https://github.com/sgl-project/sglang/issues/17365))। 4×L4-এ Flash-Next বাদ, কারণগুলো অংশ ৩-এ আছে।

---

## ৭০০+-এর সবচেয়ে সম্ভাব্য তিনটি কনফিগ

তিনটি কনফিগই অমাপা, আর প্রতিটি 700-এর ওপরে যায় শুধু তখন, যখন লিভার ১ (host-staged AR) কাজ করে। নিচের সংখ্যাগুলো সারি-মডেল থেকে [E]। ধারণাগুলো: acceptance অপরিবর্তিত থাকে, INT8 AR call-প্রতি 130–150 µs দেয়, FULL graph vLLM-এর s/সারি 0.2 ms কমায়, আর কনটেক্সটের প্রতি 1k token স্ট্রিম-প্রতি 0.03 ms (fp8) বা 0.017 ms (NVFP4) যোগ করে। Custom code ছাড়া প্রতিটির ছাদও দেওয়া আছে। Memory-র নিয়ম অনুযায়ী প্রতিটি GPU রান ব্যবহারকারীর স্পষ্ট অনুমতি নিয়ে চলবে, আর প্রতিটির আগে capacity check লাগবে।

### কনফিগ A: vLLM INT4 + MTP3 + FULL graph + host-staged AR (accuracy-base-এর বংশধর)

```bash
# vLLM nightly >= 2026-10-22 (contains PR #60532, commit 71e9320), or 0.32.0 + cherry-pick #60532 (flashinfer.py only)
export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_MIN_CTAS=8 NCCL_MAX_CTAS=8 NCCL_NTHREADS=512   # keep only if it wins the bench sweep
export ARC_HSAR=1 ARC_HSAR_DTYPE=bf16                      # OUR host-staged AR patch (not upstream); int8 = measured arm
vllm serve /kaggle/input/<redhat-qwen3.8-27b-int4> --served-model-name q27 --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype fp8_e4m3 --enable-prefix-caching \
  --mamba-ssm-cache-dtype float16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --attention-backend FLASHINFER \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --max-num-seqs 24 --max-num-active-seqs 24 --max-num-batched-tokens 8192 \
  --max-model-len 139264 --gpu-memory-utilization 0.90 --seed 0 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
# Fallback arm on stock 0.32.0: --attention-backend TRITON_ATTN (same compilation-config)
# If MTP + FULL_AND_PIECEWISE deadlocks (#42271): {"cudagraph_mode":"FULL_DECODE_ONLY"}
```

**প্রত্যাশা [E]:** custom code ছাড়া (FULL graph + NCCL CTA + 24 স্ট্রিম) প্রায় 560–650 (8k), 470–520 (40k), 340–380 (XL)। bf16 HSAR সহ প্রায় 700–780 / 540–600 / 380–420। INT8 HSAR সহ প্রায় 850–900 / 600–650 / 420–450। **কেন এটি:** এটি vLLM পথ, যার S1-60K LB 34.58 আর E19 correctness PASS আছে। তবে এই তিনটি বদল আছে: INT4 (oracle লাগবে), FULL graph (greedy gate লাগবে), আর SSM fp16 (rev2g-এ gate PASS, আর L4-এ fused kernel হারানোর দাম প্রায় 0)। `--max-model-len` আর স্ট্রিম সংখ্যা ঠিক করার আগে capacity check লাগবে: 24 স্ট্রিম × প্রত্যাশিত peak context ≤ 0.9 × 1.71M। না হলে admission cap কমাতে হবে, স্ট্রিম কমানো নয়। **বাস্তবতা:** 40k-এ 700-এ পৌঁছাতে INT8 HSAR আর কনফিগ C-এর NVFP4 দুটোই লাগে।

### কনফিগ B: SGLang INT4 + DFlash2 W6 + host-staged AR (সর্বোচ্চ peak)

```bash
# SGLang 0.5.21 (our wheelhouse) or 0.5.22; clear stale CUDA compute cache first (#42917)
rm -rf "${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}/nv"
export NCCL_MIN_CTAS=8 NCCL_MAX_CTAS=8 NCCL_NTHREADS=512   # only if it wins
export ARC_HSAR=1 ARC_HSAR_DTYPE=bf16                      # OUR patch, registered before NCCL in parallel_state
python -m sglang.launch_server --model-path /kaggle/input/<redhat-qwen3.8-27b-int4> --tp-size 4 \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.83 \
  --speculative-algorithm DFLASH --speculative-draft-model-path /kaggle/input/<qwen3.8-27b-dflash2> \
  --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 6 \
  --mamba-ssm-dtype float16 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests 21 --cuda-graph-bs-decode $(seq -s ' ' 1 21) --max-mamba-cache-size 84 \
  --disable-prefill-cuda-graph --chunked-prefill-size 4096 --schedule-policy lpm \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
# Accuracy-first variant: --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**প্রত্যাশা [E]:** custom code ছাড়া প্রায় 600–640 (8k), 540–560 (40k), 360–400 (XL), কারণ মাপা ভিত্তি 597/536/364। bf16 HSAR সহ প্রায় 720–780 / 600–640 / 400–440। INT8 HSAR + sharded sampling সহ প্রায় 850–950 / 680–760 / 430–470। **কেন এটি:** এর মাপা ভিত্তি সবচেয়ে দ্রুত, আর প্রতি step-এ সারি বেশি বলে যোগাযোগ কমানোর লাভও এখানে সবচেয়ে বড়। **বাধা:** accuracy। SGLang #36548 (DFlash c8-এ greedy probe-এ 36.5 % ভুল) কোনো fix ছাড়াই "stale" হিসেবে বন্ধ হয়েছে ([SGLang #36548](https://github.com/sgl-project/sglang/issues/36548))। আমাদের rev2g greedy gate PASS করেছে, 0টি large-margin flip, কিন্তু তা 60k sampling-এর ক্ষেত্র ধরে না। আমাদের একমাত্র SGLang LB 29.17, যদিও তার সাথে prompt পরিবর্তন মেশানো। 24 স্ট্রিম DFlash W6 OOM হয়েছে [M] ([rev2g]), তাই 20-ই সীমা। `--min-free-slots-delay 1` দেওয়া যাবে না যতক্ষণ graph buffer-এর আকার max running-এর সমান না হয় (E14A crash)।

### কনফিগ C: কনফিগ A + NVFP4 KV + 32 স্ট্রিম + MTP2 (লম্বা কনটেক্সটের জন্য)

```bash
# same env as config A (NCCL CTAs, ARC_HSAR)
vllm serve /kaggle/input/<redhat-qwen3.8-27b-int4> --served-model-name q27 --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype nvfp4 --enable-prefix-caching \
  --mamba-ssm-cache-dtype float16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --attention-backend FLASHINFER \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --max-num-seqs 32 --max-num-active-seqs 32 --max-num-batched-tokens 8192 \
  --max-model-len 139264 --gpu-memory-utilization 0.90 --seed 0 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**প্রত্যাশা [E]:** custom code ছাড়া 8k-এ প্রায় 480–600, কারণ NVFP4-এর kernel দাম আছে আর SM8x-এ NVFP4 শুধু single-token graph পায়। 40k-এ প্রায় 450–540, আর XL-এ প্রায় 380–450, কারণ 20+ স্ট্রিম ধরে। INT8 HSAR সহ 40k-এ প্রায় 650–800 আর XL-এ প্রায় 500–560, তবে শুধু যদি NVFP4 FA2 kernel bandwidth-এর কাছে চলে আর verify graph হারানোর দাম ছোট হয়। **কেন এটি:** 90k band-এ এখন মাত্র 7–12 স্ট্রিম চলে, আর লম্বা thinking-এর সময়ের বড় অংশ সেখানেই কাটে। আমাদের 60k floor-এ request-এর গড় কনটেক্সট প্রায় 30–65k। তাই "sustained"-এর সবচেয়ে কাছের প্রতিনিধি 40k কলাম, আর tail-এ XL কলাম। **বাধা:** তিনটি অজানা একসাথে আছে: NVFP4 FA2-এর SM89 গতি, MTP verify-এর graph, আর head_dim 256-এ accuracy। এটি এক ধাপে production-এ যাবে না। আগে আলাদা gate পাস করতে হবে।

---

## ৩০ মিনিটের Kaggle যাচাই: কোন ক্রমে কী মাপব

মোট তিনটি ছোট smoke, প্রতিটিতে 30 মিনিটের hard stop, আর প্রতিটি phase-এর পরে append-only JSON fsync হবে। সব রান ব্যবহারকারীর অনুমতি নিয়ে, আর memory অনুযায়ী আগে Kaggle CPU kit দিয়ে যাচাই করে। **Smoke 1 (communication, server ছাড়া, প্রায় 15 মিনিট):** T+0–5-এ environment আর `nvidia-smi topo -m`। T+5–9-এ NCCL CTA sweep: 983,040 B × 140 replay একটি CUDA graph-এ, 11.9 MB all-gather, সাথে sum-check। T+9–15-এ host-staged AR-এর bf16/INT8 microbench। সিদ্ধান্তের নিয়ম: call-প্রতি ≤ 250 µs হলে লিভার ১ engine-এ যাবে; CTA-তে ≤ 180 µs হলে custom kernel পিছিয়ে যাবে। **Smoke 2 (vLLM graph + INT4, প্রায় 30 মিনিট):** weight আগে local disk বা Kaggle Models থেকে আনতে হবে। Kaggle staff Models ব্যবহারের পরামর্শ দিয়েছেন: "I strongly suggest putting Models in Kaggle Models instead of datasets" ([Kaggle 751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344))। Arm A হলো stock 0.32.0 INT4 MTP3 24 স্ট্রিম FLASHINFER PIECEWISE, সাথে vLLM torch profiler-এ 20 step। এটিই প্রথম vLLM step-ভাগ দেবে। Arm B হলো TRITON_ATTN + `FULL_AND_PIECEWISE`, বা nightly #60532। প্রতিটির 8k আর 40k window-এ 90 s, real running == 24 assert, আর 8টি prompt-এ greedy gate। **Smoke 3 (NVFP4 KV, প্রায় 25 মিনিট):** boot করে log থেকে KV token আর cudagraph mode পড়া। তারপর 90k-এ সর্বোচ্চ স্ট্রিমে window, আর fp8 KV-র বিপক্ষে greedy ও logprob drift। প্রতিটি smoke-এর আগে tokens-needed বনাম KV capacity check, আর graph-এর প্রতিটি capture size যাচাই।

---

## Ada/L4 আর তুলনীয় PCIe GPU-র প্রতিটি throughput সংখ্যা

সব সংখ্যা aggregate output decode, যদি না অন্যভাবে লেখা থাকে। স্ট্যাটাস লেখকের [M] বা [C]।

| # | সূত্র | মডেল / quant | Engine | GPU, link, layout | Concurrency, দৈর্ঘ্য | Throughput |
|---|---|---|---|---|---|---|
| 1 | আমাদের rev2g [M] | Qwen3.8-27B INT4 + DFlash2 W6 | SGLang 0.5.21 | 4×L4, TP4 | 20 / 20 / 7 স্ট্রিম; 3.9k / 40.9k / 90.6k | **597 / 536 / 364** |
| 2 | আমাদের rev2g [M] | একই, W6 16 স্ট্রিম (দুটি রান) | SGLang 0.5.21 | 4×L4 TP4 | 16 | 572–583 / 521–544 / 370–403 |
| 3 | আমাদের rev2g [M] | INT4 W4 / W8 / MTP 3/1/4 / MTP 2/1/3 (20) / target-only | SGLang 0.5.21 | 4×L4 TP4 | 16 (MTP2 20) | 592/526/363; 573/477/363; 478–492/396–402/292–296; 542/413/263; 400–405/348–354/274–275 |
| 4 | আমাদের rev2g [M] | INT4 MTP3 16 / 20 | vLLM 0.32.0 | 4×L4 TP4 | 16 / 20 | 495/434/321; **518/439/332** |
| 5 | আমাদের rev2g [M] | FP8 MTP3 16 / 20 / 24; SSM fp16; MTP2; MTP4; Triton linear | vLLM 0.32.0 | 4×L4 TP4 | 16–24 | 434/399/281; 452/423/281; 472/424/256; 450/414/312; 448/397/260; 399/381/267; 438/398/283 |
| 6 | আমাদের rev2g [M] | PCIe-IPC AR (FlashInfer 0.7) | SGLang / vLLM | 4×L4 TP4 | 16 | DFlash 230 (0.39×), MTP 248 (0.42×), vLLM 297 (0.68×) |
| 7 | আমাদের rev2g [M] | 2×TP2 DFlash W6 / vLLM FP8, INT4 | SGLang / vLLM | 4×L4, 2×TP2 | 8→3 / replica | 658 (8k) → 308 (40k); vLLM 191/483 → 22/92 |
| 8 | আমাদের E14A [M] | INT4 DFlash2 W8 / W6 / W4 / target-only | SGLang 0.5.21 | 4×L4 TP4 | 12; প্রায় 5.9k | 505.0 / 531.4 / 526.8 / 339.6 |
| 9 | আমাদের E18 smoke v1 [M] | 27B S1 | vLLM 0.32 | 4×L4 TP4 | 20 | 382 steady, 0 preempt (memory) |
| 10 | আমাদের পুরো রান [M] | W1 / vLLM 0.19 FP8 MTP3 | SGLang / vLLM | 4×L4 TP4 | 11–16 | 328.9, 379.9 / 254.3–333.3; Duck কার্যকর 167.8–279.0 |
| 11 | Darren, [Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546) [C] | অপ্রকাশিত | অপ্রকাশিত | 4×L4 | "high concurrency" | peak 900–980, pure decode প্রায় 1,100, 6 h গড় 710, prefill-এ 250 |
| 12 | TheDivine Aurora / Adarsh, একই topic [C] | 27B / FN | — | 4×L4 | — | 250–400; "half" of 900-এর কম |
| 13 | Dominic, [Kaggle 732723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723) [M] | Nex-N2-mini (35B-A3B MoE) fp8 | vLLM 0.26 | 4×L4 | — | 438 / 452 |
| 14 | [AWS G7e blog](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-on-amazon-sagemaker-ai-with-g7e-instances/) [M] | Qwen3-32B BF16 dense | vLLM | 4×L40S TP4 | C1/8/16/32; ~1k in / ~560 out | 37 / 242 / 416 / **686**; 1× RTX PRO 6000 C32 592 |
| 15 | [dgallitelli](https://dgallitelli.github.io/blog/posts/qwen36-sagemaker-benchmark/) [M] | Qwen3.6-27B FP16/FP8 + MTP | vLLM 0.19.0 | 4×L40S TP4; 8×L4 TP8 | 11 in / 50 out | 19.82 / 20.19 req/s (tok/s-এ রূপান্তরযোগ্য নয়); 8×L4 FP8 **CUDA OOM** |
| 16 | [club-3090 #1288](https://github.com/noonghunna/club-3090/discussions/1288) [M] | 27B W8A8 INT8 + DFlash2 | SGLang 0.5.19 | 4×RTX 3090, NVLink জোড়া, TP4 | c4 / c8 / c16 | 487.7 / 730.6 / **927.5** (narrative); code-এ 893.9 / 1,418.3 / 1,811.0 |
| 17 | [HyperQwen](https://github.com/syv-ai/HyperQwen) [M] | 27B int8/Marlin mix | vLLM 0.31 + patch | 1×RTX 3090, 250 W | 64; 128 in / 512 out | প্রায় 1,035 decode, 948 e2e, সব layer int8-এ 1,222; single stream 127 |
| 18 | [vLLM #55533](https://github.com/vllm-project/vllm/issues/55533) [M] | RedHat 27B INT4 GPTQ Marlin | vLLM | 1×RTX 4090 D (sm89) | bs1 / 8 / 20 | no-spec 56.4 / 367.1 / **649.0**; MTP k=1 72.0 / 197.0 / 210.9 |
| 19 | [FriendliAI sakamakismile](https://friendli.ai/models/sakamakismile/Qwen3.8-27B-MTP-NVFP4) [M] | 27B NVFP4 | vLLM 0.22.0 | 4×RTX PRO 2000 Blackwell TP4 | c1/2/4/8 | no-MTP 49.0/93.2/180.9/318.3; MTP3 72.6/110.9/232.8/**386.9** |
| 20 | [alexander-ollman](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/qwen3.6-on-rtx3090.html) [M] | 3.6-27B INT4; 35B-A3B AWQ | vLLM 0.19.2 | 2×RTX 3090 PCIe, NVLink নেই | 1024/1024 | 27B 2 replica C4 **225.4**, C8 167.0; TP2 single 55–68; 35B-A3B C4 282.7 |
| 21 | [FriendliAI avyukth](https://friendli.ai/models/avyukth/Qwen3.8-27B-AWQ-INT4) [M] | 27B AWQ INT4 | vLLM | 1×RTX 3090 | c1/4/8/16 | 41.1 / 136.4 / **221.5** / 216.9 |
| 22 | [Reddit snapshot](https://reddit.sentinel-team.org/posts/1te9o18/snapshots/2026-05-23T00%3A36%3A34.219378Z) [M] | 3.6-27B FP16 | vLLM 0.20.2 TP4 | 4×3090, PCIe Gen3 x16/x8/x8/x4 | — | 29 t/s (x4 all-reduce bottleneck) |
| 23 | [NVIDIA forum](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102) [M] | 27B | vLLM | 4×DGX Spark | C8 | TP2+DP2 36.0 / request বনাম TP4 27.0 |
| 24 | [club-3090 #1092](https://github.com/noonghunna/club-3090/discussions/1092) [M] | 27B W4A8 | vLLM 0.27/0.31 TP2 | 2×3090 | C32 | MTP4 241 বনাম off 308 |
| 25 | [tonyd2wild](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090) [M] | Flash-Next W4A16 | vLLM nightly TP4+EP, MTP3 | 4×3090 | 6 agent | 252.6 (317 sustained / 570 peak) |
| 26 | [chenxin199305](https://github.com/chenxin199305/qwen38-flash-next-3x4090) [C] | Flash-Next NVFP4 (MARLIN) + FP8 | vLLM 0.30 + 15 patch | 3×RTX 4090 48 GB TP3+EP3 | 1/4/8/16 | 124.3 / 387.4 / 617.6 / **635.3**; c16-এ K0 559, K2 644.5, K3 541.1 |
| 27 | [jsboige #123](https://github.com/jsboige/vllm/pull/123) [M] | 27B (অনুমিত) fp8 / nvfp4 KV | vLLM 0.32 fork | SM89, GPU সংখ্যা অজানা | N=1 / N=48 | 41.1 / **886.4** (fp8); 33.9 / 767.5 (nvfp4) |
| 28 | [vLLM #38230](https://github.com/vllm-project/vllm/issues/38230) [M] | Qwen3.5-4B FP8 (ছোট) | vLLM + LMCache | RTX 4080 Super | 50 | 1,061, 95 % external hit |
| 29 | [Strata #2668](https://github.com/Niko1221/Strata/issues/2668) [M] | Flash-Next UD-Q4_K_XL | llama.cpp-derived | 1×RTX 4090 48 / 24 GB | 1 slot; 1–8 client | 72.4–85.8; 24 GB-এ 64.6–71.2 |
| 30 | [vLLM #61224](https://github.com/vllm-project/vllm/pull/61224) [M] | Qwen3-8B-FP8 | vLLM + CUTLASS SM89 | 1×L4 | batch 32 | Marlin 1,929 → CUTLASS 2,739 |
| 31 | [vLLM #60532](https://github.com/vllm-project/vllm/pull/60532) [M] | 27B NVFP4 + NVFP4 KV MTP3 | vLLM | SM80, TP1 / TP2 | batch 1 | 73.6 → 108.1; 61.0 → 83.1 |
| 32 | [vLLM #59366](https://github.com/vllm-project/vllm/pull/59366) [C] | Flash-Next NVFP4 MTP3 | vLLM draft PR | 2×RTX PRO 6000 | c32 | 1,295 → 1,519 |
| 33 | [Houtini](https://houtini.com/articles/the-dual-4090-96gb-vllm-benchmark-runbook/) [M] | 27B | vLLM 0.26 | 2×RTX 4090 48 GB | single, 29k prompt | TP1 34.4, TP2 48.9, PP2 32.9 |
| 34 | [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254) [M] | AR microbench | NCCL SHM | 4×5060 Ti Gen3 x8 KVM | 64 / 128 token | 267.6 / 498.1 µs per AR; P2P_LEVEL c4-এ +13.3 % |
| 35 | Franzen M2, [ARC-AGI-3 744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792) [M] | Flash-Next W4A16 + NEXTN | Pennyroyal SGLang | 1×RTX PRO 6000 TP1 | 10 স্ট্রিম | P90 836.6, harness 609.0, peak 946; lordhansolo 1,135; sirikilohit 1,159 |
| 36 | [Kaggle 750885](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/750885) [M] | Franzen M2 | একই | RTX PRO 6000 | 10 বনাম 14 স্ট্রিম | 600–620 বনাম peak 470 → 280 |
| 37 | [SGLang FN cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next) [M] | Flash-Next | SGLang ≥ 0.5.22 | 1×RTX PRO 6000 | 16 MTP / 64 no-MTP | 613 / 861 |

টেবিল থেকে একটি নিদর্শন বারবার আসে। ≥600-এর প্রতিটি মাপা সংখ্যা এমন GPU-তে, যার ব্যান্ডউইথ L4-এর প্রায় 3 গুণ (L40S, 3090, 4090, PRO 6000), অথবা ছোট কনটেক্সটে। L4-এর ব্যান্ডউইথের কাছের সবচেয়ে ভালো analog হলো 4×RTX PRO 2000 Blackwell-এর c8-এ 386.9 [M]। আরেকটি নিদর্শন: PCIe consumer card-এ replica TP4-কে হারায় (সারি 7 বাদে, যা আমাদের KV-তে হেরেছে), আর 4–8 concurrent-এর ওপরে spec-এর লাভ কমে যায় (সারি 17, 18, 24)। AWS-এর 4×L40S-এ 686, ব্যান্ডউইথের অনুপাতে (প্রায় 2.9×) ভাগ করলে L4-এ BF16 32B-এর জন্য প্রায় 240 দাঁড়ায়। এটি এই রিপোর্টের হিসাব। L40S, L4 আর RTX PRO 2000-এর ব্যান্ডউইথ সংখ্যা NVIDIA datasheet-এর background জ্ঞান, এই কাজে fetch করা হয়নি [E]।

---

## অযাচাইকৃত দাবি আর গুজব (আলাদা রাখা)

এই অংশের কিছুই সিদ্ধান্তের ভিত্তি নয়। **Darren-এর সংখ্যা** (900, 980, প্রায় 1,100, 710, 250) [R]: model, engine, quant, স্ট্রিম বা কনটেক্সট কোনোটিই প্রকাশিত নয়, আর তাঁর দলের নাম ও rank অযাচাইকৃত ([Kaggle 750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546))। তিনি Flash-Next-এর eligibility নিয়ে প্রশ্ন তুলেছিলেন আর লিখেছিলেন "Many teams in both competitions have spent a lot of time building pipelines around this model" ([Kaggle 752709](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/752709))। এতে তাঁর মডেল Flash-Next হওয়ার ইঙ্গিত থাকে, প্রমাণ নয়। কিন্তু 4×L4-এ Flash-Next-এর মাপা KV pool (94–118k) 100k-এর বেশি reasoning-এর "high concurrency"-র সাথে মেলে না। তাই হয় তিনি 27B ব্যবহার করেন, নয়তো খুব ছোট Flash-Next footprint, যেমন sub-4-bit বা pruned expert [E]। **Anil Thomas-এর** দাবি "many of the leading teams are using Qwen3.8-Flash-Next alongwith the Duck harness" সময়ের মিল দেখে করা অনুমান। তাঁর আগের "distillation" অনুমানের জবাবে mikelou1 বলেছিলেন "I don't use distillation" [R] (একই topic)। **TheDivine Aurora:** OpenRouter-এ 16টি টাস্কে Flash-Next 27B-এর চেয়ে বেশি সমাধান করেছে, আর "quantization hurts quality"। এটি পুনরুৎপাদনযোগ্য নয় [R]। **Adarsh:** L4×4-এ 27B আর Flash-Next "almost breakeven in speed and accuracy" [R]। **mikelou1:** "around 4xL4 speed I got Qwen to 87%", CV 79 % ± 5 [R]। **xz:** CPU throttling থেকে 5–10 % কম স্কোর [R] ([Kaggle 751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344))।

ওয়েবের snippet-স্তরের দাবিগুলো, যার পুরো পাতা পড়া হয়নি: houtini-র "Block-FP8 … none for Ada (SM89) … quietly loses you around 40% of your decode" ([houtini](https://houtini.com/articles/vllm-tuning-48gb-rtx-4090/)); 4×RTX 5090-এ "Avg gen throughput: 2000 tok/s" ([CSDN](https://gitcode.csdn.net/6a0ceb7b10ee7a33f273cb87.html)); Benjamin Marie-র TurboQuant "similar accuracy - 4x smaller KV cache" ([X](https://x.com/bnjmn_marie/status/2043950978538279336)); GigaGPU-র দাবি যে PCIe-তে TP 1.5×-এ থামে আর PP 1.7–1.8× পায় ([GigaGPU](https://gigagpu.com/tensor-vs-pipeline-parallelism/)); ik_llama.cpp-এ qwen4exp MTP merge হয়েছে ([prismix](https://prismix.dev/news/6bf14fe7cba2)); RTX 5090-এ TRT-LLM-এর চেয়ে vLLM 57 % দ্রুত ([computingforgeeks](https://computingforgeeks.com/tensorrt-llm-vs-vllm/))। leaderboard monitor-এর "minutes_taken" field 720 মিনিটের বেশি দেখায় (1,074–1,525), তাই এতে queue সময়ও ধরা আছে, আর শীর্ষ দলের আসল run time জানা যায় না (community_evidence §2)। jsboige fork-এর 886.4 tok/s-এর মডেল আর GPU সংখ্যা উল্লেখ নেই, তাই সেটি 4×L4-এর সাথে তুলনীয় কি না অজানা।

---

## যা এখনো কেউ জানে না

vLLM 0.32 MTP step-এর কোনো profile নেই, তাই vLLM-এর 107–112 ms-এর ভাগ হিসাব মাত্র। NCCL CTA বাড়ালে 983 KB-এ SHM 120 µs-এর floor-এর কতটা কাছে যায়, কেউ মাপেনি। চারটি L4 একসাথে host DRAM-এ লিখলে g2 VM-এর memory bandwidth ধরে কি না, তাও মাপা হয়নি। Kaggle VM-এর "Gen3 max" সত্যি না ভুল রিপোর্ট, তা অনিশ্চিত। #60532 SM89-এ fp8 KV আর head_dim 256-এর সাথে কাজ করে কি না অপরীক্ষিত। NVFP4 KV-র sm_89 গতি MTP আর long context-এ, আর FA2 পথে ARC accuracy, অজানা। INT8 all-reduce Qwen3.x hybrid-এ 60k token-এ accuracy কতটা ধরে, তার কোনো ডেটা নেই। DFlash2-এর #36548-শ্রেণির ভুলের হার আমাদের stack-এ 60k sampling-এ মাপা হয়নি। SGLang 0.5.22 বা main-এ SM89 QSA decode-এর কোনো fix আছে কি না, আর আমাদের Flash-Next "!!!!"-এর আসল উৎস (QSA, নাকি Marlin/EP MoE), অনির্ণীত। Top-40-এর কোনো দল ARC-AGI-2-এর tok/s, মডেল বা code প্রকাশ করেনি। Tufa Labs আর nvbanana প্রতিযোগিতার শেষে প্রকাশ করবে বলেছে (community_evidence §1)।

---

## উপসংহার

নতুন মাপগুলো প্রশ্নটাই বদলে দিয়েছে। আগে প্রশ্ন ছিল কোন engine, কোন spec পদ্ধতি বা কতগুলো স্ট্রিম। এখন উত্তর একটি সংখ্যায় আসে: স্ট্রিম-প্রতি step-খরচ s। আমাদের সব stack-এ s 4.6–7.7 ms। এর প্রায় দুই-তৃতীয়াংশ এমন PCIe যোগাযোগ, যা batch বড় করলে ভাগ হয় না, আর P2P ধীর বলে বাজারের কোনো kernel এখানে খাটে না। তাই 4×L4-এ concurrency বাড়ানো বা নতুন version-এ যাওয়া এখন ছোট লিভার। একমাত্র বড় লিভার হলো host memory-র মধ্য দিয়ে আরও দক্ষ বা কম-byte all-reduce, যা আমাদের নিজেদেরই লিখতে হবে। এর সবচেয়ে সস্তা প্রথম ধাপ একটি 3 মিনিটের NCCL CTA sweep। এর আরেকটি অর্থ আছে: যোগাযোগ কমলে চওড়া draft (DFlash) আবার জিতবে। তাই width আর MTP-র সিদ্ধান্ত যোগাযোগের fix-এর পরেই আবার মাপতে হবে, আগে নয়।

দ্বিতীয় অন্তর্দৃষ্টি হলো, Darren-এর সংখ্যা আর আমাদের সংখ্যার ব্যবধান রহস্যময় নয়। আমাদের 116 ms step থেকে যোগাযোগ সরালে ঠিক তাঁর peak-এর কাছাকাছি সংখ্যা আসে। আর তাঁর নিজের 980 থেকে 710-এ পতন দেখায় যে tail-এ occupancy ধরে রাখা engine-এর মতোই গুরুত্বপূর্ণ। Deadline-এর আগে বাস্তবসম্মত লক্ষ্য দুই স্তরে ভাগ হয়। Config-স্তরের পরিবর্তন (INT4 on vLLM, FULL graph, NCCL CTA, 24 স্ট্রিম, tail refill) কয়েক দিনে বসানো যায়। এগুলো দিয়ে প্রায় 550–650 decode আর এক 60k attempt-এর পূর্ণ coverage পাওয়া সম্ভব। 700+ sustained, অর্থাৎ প্রতি টাস্কে দুটি 60k attempt, পেতে হলে host-staged all-reduce কাজ করতেই হবে, আর তা accuracy gate পার হতে হবে।
