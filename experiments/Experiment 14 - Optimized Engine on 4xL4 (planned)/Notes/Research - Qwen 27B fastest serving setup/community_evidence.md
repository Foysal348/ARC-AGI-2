# Community and competition evidence: how people serve Qwen3.8-27B and Qwen3.8-Flash-Next on Kaggle 4×L4 and similar hardware, and the throughput they report (as of 2026-10-23)

**Scope and method.**
- **Local Kaggle discussion dumps, read in full:**
  - all 35 topic files and the summary in `D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/`;
  - the 10-21 folder, whose topic files are earlier, smaller copies of the same threads;
  - the ARC-AGI-3 dumps in `D:/ARC AGI 2/_scratch/webresearch/` (`d_*.md`, `topteam_comments.md`).
- **Public Kaggle notebooks:** 14 pulled read-only to `D:/DevCache/Temp/research_kernels/`, parsed as JSON and never executed. Kaggle model file listings were fetched with the CLI.
- **Web sources:** about 100 tool calls by me plus two research subagents, covering Hugging Face model cards and their discussion tabs, GitHub issues, PRs and discussions, blogs, and arcprize.org.
  - Subagent findings came through WebFetch summaries. I re-fetched and confirmed the two most load-bearing ones: the DFlash2 card's concurrency table, and the club-3090 4×3090 SGLang post.
  - Treat all other short quotes as "as summarized".
- **Labels used throughout:**
  - **[MEASURED]** = a number the author says they measured.
  - **[ESTIMATE]** = arithmetic or a guess.
  - **[CLAIM]** = a vendor's or author's self-report with no detail.
- **Reddit:** r/LocalLLaMA is not indexed by the search tool. No Reddit thread could be read (see Gaps).

---

## 1. Kaggle 4×L4 (ARC-AGI-2): which setups competitors use, and the throughput they report

### Takeaway
- **Every public ARC-AGI-2 Qwen3.8 notebook serves the 27B with tensor parallelism across all 4 L4s (TP4)**:
  - **vLLM** (0.19 from the `arc3-vllm-h100-wheelhouse-v3` dataset) with FP8 weights, FP8 KV and MTP;
  - or **SGLang** with FP8 weights and EAGLE/MTP 3/1/4 (NguyenThanhNhan).
- **Reported aggregate throughput on 4×L4 sits at about 250–500 tok/s.** Measured values cluster at 250–430: our own vLLM FP8+MTP and SGLang INT4+DFlash2 runs, and TheDivine Aurora's "300-500 tps". Anil Thomas thinks 350 is plausible for FP8.
- **Competitors themselves say 600–800+ tok/s is needed to reach every task at xhigh effort.**
- **The top scorers (60–70 LB) do not publish their engine or tok/s.**

### Cited Findings

**Table A — who used what on Kaggle 4×L4 (ARC-AGI-2)**

| Who / notebook | Engine & weights | Spec decode / KV / key flags | Concurrency, context, token budget | Throughput reported | Score reported |
|---|---|---|---|---|---|
| **NguyenThanhNhan** (topic 745787, 10-05) | **SGLang**, Qwen3.8-27B **FP8** | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`; `--kv-cache-dtype fp8_e4m3`; `--attention-backend flashinfer`; `--linear-attn-backend triton`; `--chunked-prefill-size 2048`; `--max-prefill-tokens 4096`; `--mamba-radix-cache-strategy extra_buffer`; `--mamba-full-memory-ratio 0.5`; `--mamba-ssm-dtype float32` | xhigh, **1 pass**, 65k cap, forced commit after **49k** thinking tokens; concurrency not disclosed (asked, not answered) | **not stated**. Darren computed that ≥360 tok/s would be needed [ESTIMATE] | 2 submissions, "pure reasoning", ~23 LB [MEASURED, screenshot] |
| NguyenThanhNhan, later (topic 751344, 10-21/22) | SGLang, model not stated; reads **~180 GB** from `/kaggle/input`; "mtp draft setup" | — | ~12 h run | not stated | **52.08** (10-18) → **65.97** (10-20); two 0.00s caused by the GCSFUSE storage |
| **Adarsh** `madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9` (pulled) | **vLLM** OpenAI server. Wheels: `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` plus the `keithtyser/taaf-duck-qwen38-serving-v1` overlay. Weights: **`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`** | Startup ladder, first rung: `--kv-cache-dtype fp8 --language-model-only --speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":2}` plus `--reasoning-config` (think budget). Falls back down to plain serving. Always on: `--tensor-parallel-size 4 --enable-prefix-caching --gpu-memory-utilization 0.90` | `max_model_len` 131,072; `max_num_seqs` 16; concurrency 16; "direct" mode; 2 samples; xhigh; temp 1.0 / top_p 0.95 / top_k 20; `thinking_token_budget` **98,304**; `max_tokens` 106,496 | config default `"tok_per_s": 140` (an ad hoc planning assumption, not a measurement) | **21.53** LB; "re-ran with a small change, score dropped to 13" |
| Adarsh `madarshbb/arc-qwen-3-8-27b-agent` (pulled) | same stack | same ladder | `max_model_len` 81,920; concurrency 10; think budget 40,960; `max_tokens` 49,152 | — | "about 15" LB |
| **russcore** (Russell Kirk) `russcore/arcagi2-qwen38-freeform-direct` (pulled) | **vLLM 0.19.0** (wheelhouse v3), the FP8 repack served as `Qwen/Qwen3.8-27B-FP8` | **No MTP, no FP8 KV**; `--dtype bfloat16 --language-model-only --enable-prefix-caching`; `VLLM_USE_FLASHINFER_SAMPLER=0`; `qwen3_coder` tools | context 65,536; `max-num-seqs` 24; 2 samples; temp 0.6; xhigh; `direct_max_tokens` 57,344; `turn_max_tokens` 40,960; up to 10 tool turns | logs vLLM "Avg generation throughput", but the value is not published | about 10 LB. Anil: `turn_max_tokens` 40,960 "is too low. try setting it to 64k" |
| **khoinguyennguyen** `arc-agi-2-agentic-multi-turn-reasoning` (pulled) | vLLM (wheelhouse v3), `mikedan7/qwen3-8-27b-fp8-official` | `--kv-cache-dtype fp8`; `--attention-backend TRITON_ATTN` **plus** env `VLLM_FORCE_ATTN_BACKEND=TRITON_ATTN`; MTP 3 in the config (the text says 6); `--no-async-scheduling`; prefix caching; `preserve_thinking` | `gpu_memory_utilization` 0.96; `max_model_len` **262,144**; `max_num_seqs` 16; multi-turn python agent | no tok/s. "Startup after preload: 90–240 s on 4×L4"; 2–5 min total with page-cache preload vs 6–10 min without | none; "At 2 agents/task it takes too much time" |
| **delongyang** `arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours` (pulled) | vLLM 0.19, official FP8 (mikedan7) | fp8 KV + `TRITON_ATTN` + `VLLM_FORCE_ATTN_BACKEND`; MTP **6** tokens; `--no-async-scheduling`; default `reasoning_effort` **medium** | `max-model-len` 131,072; `max-num-seqs` 16; util 0.96 | none; "Section 2 takes 5–12 minutes to load the weights and capture CUDA graphs" | none |
| yjsh2003 `hyperqwen-v7-fast-arc-submission-20261009` (pulled) | HyperQwen runtime (syv-ai; patched vLLM) | "W4A16/MTP3, TP4/C4, FP8 KV, full 131072 context, 60K/xhigh" | concurrency 4 | none | none |
| Souvik_D_Biswas (749756) | HF transformers `from_pretrained` of the FP8 repack | — | — | — | load fails (`TypeError ... NoneType`); 4B TTT 28.06 |
| Dominic (732723); a model-class analog, not Qwen3.8 | vLLM 0.26, `Nex-N2-mini` / `Aquila-mini` (Qwen3.5-MoE, 35B total / 3B active), BF16 quantized to fp8 per-tensor at load, Triton MoE | — | 4×L4 | **438 / 452 tok/s** [MEASURED]; "W4A16 ... 20% higher throughput" | 0/64 on eval |
| **Our team (Hack2Publish)**, local reports | (a) vLLM 0.19 FP8 repack + MTP + fp8 KV; (b) SGLang 0.5.21 + RedHat INT4 + BF16 DFlash2 | (a) MTP acceptance ~0.66; (b) DFlash2 | (a) 16 running; (b) 6–12 running | (a) **249–353.5 tok/s** [MEASURED]; (b) **321–431 tok/s** [MEASURED] (368 at 6 running → 431 at 12; 371 at 12 in another run; G4B end-to-end 277) | LB 34.58 (S1-60K), 29.17 (E11-06 SGLang P1-60K), per project memory |

**Sources for Table A:**
- NguyenThanhNhan config:
  - quote: "I managed to submit twice with qwen 3.8 27b fp8 + mtp at 65k tokens and xhigh reasoning effort"; "force the model to stop reasoning and commit ... after 49k tokens"; "Those 2 submissions used pure reasoning ... you could solve many more tasks within 12 hours by implementing a simple harness";
  - [topic 745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787); local `Kaggle discussions (2026-10-22)/topic_745787.md`.
- NguyenThanhNhan 180 GB and server startup:
  - quote: "I need to read ~180 GB from /kaggle/input. At normal NFS speed of ~300MB/s, it would take around 10 minutes"; "45-minute timeout for the inference server startup. Normally, it would take at most 15 minutes"; save runs "took between 875s and 1300s to finish loading the model"; "a 6-hour run on a rented 4×L4 machine";
  - [topic 751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344); local `topic_751344.md`. The 52.08 and 65.97 scores come from the leaderboard history in local `00_SUMMARY.md` (2026-10-22).
- Adarsh:
  - [notebook](https://www.kaggle.com/code/madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9?scriptVersionId=358248812), pulled to `D:/DevCache/Temp/research_kernels/madarshbb_arc-agent-sub-q38-direct-b98k-c16-792ef9/` (cells 0, 10, 11, 12; `kernel-metadata.json`);
  - quote: "scored 21.53 ... when I re-ran with a small change, score dropped to 13" ([topic 748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082));
  - "It scored around 20-24" (topic 749756).
- russcore:
  - [notebook](https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct), pulled (cell 1, `RUNTIME_CONFIG`);
  - score "about 10" from topic 745787 (screenshot) and local `00_SUMMARY.md`; Anil's reply in 745787.
- khoinguyennguyen:
  - [notebook](https://www.kaggle.com/code/khoinguyennguyen/arc-agi-2-agentic-multi-turn-reasoning), pulled (cells 2, 3, 4);
  - "At 2 agents/task it takes too much time" (748082).
- delongyang: [notebook](https://www.kaggle.com/code/delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours), pulled (cells 0, 3, 4).
- yjsh2003: [notebook](https://www.kaggle.com/code/yjsh2003/hyperqwen-v7-fast-arc-submission-20261009), pulled (markdown cell 0).
- Souvik: [topic 749756](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/749756).
- Dominic: [topic 732723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723).
- Ours:
  - `D:/ARC AGI 2/experiments/Experiment 8 - .../E8_FINAL_REPORT.md` ("353.5 tok/s, 16 active ... MTP 0.663");
  - `Experiment 7 .../README.md` ("249–285 tok/s with 16 running; MTP acceptance 0.66");
  - `Experiment 6 .../README_DUAL_v2_GPU_RESULT_FOR_GPT.md` (288.5 tok/s);
  - `Experiment 12 - Duck REPL .../Experiment Review/E12 rev 4 EVAL6 - audit request (2026-10-20).md` ("decode at 431 tok/s"; 368 at 6 running);
  - `E12 next fix list (running, 2026-10-20).md` ("371 tok/s at 12 running (321 at 6)"; "G4B end-to-end was 277").

**Throughput statements in the ARC-AGI-2 threads:**
- **Darren Amadeus Martin (team rank ~9–12)** [CLAIM]:
  - "4 L4 seems to be very slow when running these models due to lower bandwidth and slow gpu communication. Has anybody got like an aggregate above 500 tok/s?"
  - Later he says his team "built every pipeline on top of the 4 L4 ... optimized everything so that we can get fast generation ... it fixes some issue that is present in the library code itself".
  - Sources: [745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787); `_scratch/webresearch/topteam_comments.md` lines 359-368 (the "Petition to delay..." thread, 10-11/12).
- **Anil Thomas** [ESTIMATE]:
  - "350 tokens/s on a 4x L4 seems plausible for the FP8 quantized model".
  - Back-of-envelope: `42*12*60*60/(64*1024*330/350) ~ 30` (330 queries, 64k mean generation, 42% accuracy).
  - "the generations at xhigh effort are very long (median around 64K from what I have seen)".
  - Source: [745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787).
- **TheDivine Aurora** [MEASURED for the first part, ESTIMATE for the rest]:
  - "at around 300-500 tps i had ~19-20 % score. only attempted half the dataset. so practically 40 % is possible with 600-800 tps, 1 attempt per task ... many times attempts dont end up with a answer. but when done, theyre mostly right. it just takes 128k - 256k context window".
  - Source: [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546).
- **Kris Dreemurr** [MEASURED, CV]:
  - "my qwen harness at xhigh at full policy median 70k tokens for pass@2 (so 35k per slot) gets %45 solves or about 52~ out of 120 ... The issue is I cant fit it in the box".
  - NVARC "run on all 240 tasks in about 4 hours".
  - Source: [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546).
- **Adarsh:**
  - "Qwen 3.8 27b on xhigh is just so slow that it never reaches most of the tasks, even optimizations on inference speed, task picking, etc… aren't really cutting it for me";
  - "im milking the life out of the inference speed".
  - Sources: [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546), [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082).
- **Hiểu Vy:**
  - "I think the other teams are doing something similar, but they've found ways to make their models run faster";
  - 26.81 LB with the original 27B; "80% accuracy" on the 120 validation tasks (it was asked, and not answered, whether that was within 12 h).
  - Source: [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082).
- **TheDivine Aurora** (746149): the October jump came from "an inference setup with maximum throughput to arleady extract the exsiting models ability". Source: [746149](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/746149).
- **mikelou1** (LB 66.39 on 10-17, 69.72 on 10-21) [CLAIM]:
  - "With time constraints at around 4xL4 speed I got Qwen to 87%" (CV);
  - "79% ± 5% in CV runs"; "Not a NVARC ensemble";
  - "a tiny kernel issue I found that made my program score only 40% (instead of the 60% projected by CV)".
  - He never answered whether the model is 27B or Flash-Next. Closed source.
  - Sources: [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546), [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082).
- **Darren on the ARC Prize cost:** "$0.4469/task and $2.55 per million output tokens, shouldnt it be roughly 170k tokens per task (considering 2 pass per task, maybe 85k-90k ish tokens)" [ESTIMATE]. Source: [745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787).
- **Cho Royou:** "Although the 27B model can run on four L4 GPUs, the cost is quite high. Given the inference-time limit, we probably wouldn't be able to use a sufficiently long context window anyway." Source: [742796](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742796).
- **FOYSAL** (owner of the FP8 repack):
  - "the model weights are split into four parts and loaded across the four L4 GPUs. After loading Qwen 3.8 27B, only a small amount of VRAM remains available for the KV cache";
  - "You are using my repacked Qwen3.8 27B";
  - NVARC-vs-Qwen correlation 0.43.
  - Source: [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082).

**The FP8 repack (`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`) is the de-facto shared 27B checkpoint on Kaggle:**
- **Used in ARC-AGI-2** by Adarsh (both notebooks), russcore, and Souvik.
- **Used in ARC-AGI-3** by `keithtyser/duck-qwen3-8-27b-fp8` and `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b`.
  - Source: the `model_sources` field in each pulled `kernel-metadata.json` under `D:/DevCache/Temp/research_kernels/`.
- **Release post:** announced 2026-08-16 ("You can now use Qwen 3.8 27B https://www.kaggle.com/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/"), local `_scratch/webresearch/d_735243.md`.
- **Files** [MEASURED, CLI file listing of `.../PyTorch/hf-fp8/1`, 2026-10-23]:
  - 16 × 1.524 GB `model-layers-*.safetensors`;
  - `outside.safetensors` 6.007 GB;
  - `mtp.safetensors` 0.477 GB;
  - total **30.89 GB**, uploaded 2026-08-16.
- **Second shared checkpoint:** `mikedan7/qwen3-8-27b-fp8-official`, used by khoinguyennguyen and delongyang (notebook paths).

**L4 (sm89) engine traps documented in public notebooks:**
- **delongyang:**
  - "FlashInfer's FP8 attention kernels exist only on SM90+ (Hopper and newer). The L4 is SM89", so fp8 KV crashes with "illegal memory access" unless `TRITON_ATTN` is used.
  - "The MTP drafter does not inherit `--attention-backend`" (cites vLLM PR #53450). Only `VLLM_FORCE_ATTN_BACKEND` pins it.
  - "MTP acceptance rate is 0% — a separate bug" (issue #44879).
  - "MTP ≥ 2 was the crash-prone region historically".
  - Source: delongyang notebook cell 3/4, [kaggle link](https://www.kaggle.com/code/delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours).
- **khoinguyennguyen:** "Without this variable the main model runs Triton while the drafter tries FlashInfer and dies with an illegal memory access mid-generation". Source: khoi notebook block 02.
- **Dominic:**
  - "`VLLM_USE_FLASHINFER_SAMPLER=0` is necessary — the image has no linkable `libcuda.so`";
  - "Don't install wheels under `/kaggle/working`" (7 GiB committed as output, then an ERROR at teardown).
  - Source: [732723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723).
- **Adarsh's ladder comment:** "Untested-on-L4 options (MTP, --reasoning-config) must never cost the whole run". Source: Adarsh notebook cell 11.

**Official ARC Prize numbers for context:**
- **Qwen3.8-27B (XHigh):** 42.36% on the semi-private set, $0.447/task. Sources: [Chris Deotte, topic 750577](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750577); [arcprize.org results page](https://arcprize.org/results/alibaba-qwen3-8-27b).
- **By effort level:** xhigh 42.4%, medium 13.2%, low 22.8%, none 1.5%. Provider Baseten; at $0.425 / $2.55 per 1M input / output tokens; no token counts per task are published. Source: [arcprize.org](https://arcprize.org/results/alibaba-qwen3-8-27b) (subagent fetch).

### Inferences
- **The bottleneck is bandwidth.** About 250–430 tok/s aggregate is what TP4 serving of the 27B achieves on 4×L4 today, with either vLLM FP8+MTP or SGLang INT4+DFlash2. That is roughly 30–45% of a naive bandwidth bound:
  - 4 × 300 GB/s across ~5–6 GB of weights per GPU per decode step gives about 50–60 steps/s;
  - × 16 sequences, that is about 800–950 tok/s before KV reads, all-reduce and speculative overheads.
  - The 600–800 tok/s that TheDivine Aurora says is needed is therefore plausible only with smaller weights (INT4), effective speculation at high concurrency, and low TP overhead.
- **Our measurements are consistent with the 4×3090 data.**
  - A 3090 has ~936 GB/s, about 3.1× an L4.
  - 4×3090 SGLang TP4 W8A8+DFlash2 reaches 927 tok/s on narrative text at c16.
  - Scaled by bandwidth alone, that predicts ~300 tok/s on 4×L4, in line with our measured 321–431 (Section 3). This assumes decode is bandwidth-bound and PCIe all-reduce is not the limit.
- **The public notebooks set the token budget per task but do not size it to the measured rate.** russcore's xhigh episodes are capped at 3,600 s; Adarsh's 98k thinking budget × 2 samples × 240 tasks needs far more than 12 h at ~300 tok/s. Their ~10–21 LB is consistent with the coverage problem Adarsh describes ("never reaches most of the tasks").
- **NguyenThanhNhan's ~180 GB read cannot be the 27B alone** (the FP8 repack is 30.9 GB).
  - It matches the size of a Flash-Next FP8 checkpoint: 172.78 GiB per the vLLM recipe, ~173 GB per SGLang PR #36968 (Section 3).
  - So his 65.97 run *may* be Flash-Next FP8, or a 27B plus large extra artifacts.
  - This is an inference only; he did not state the model.

### Gaps
- **No measured tok/s from any 50+ LB team.** NguyenThanhNhan, mikelou1, Darren's team, artificialagencylab, mtg, Youssef Nader and guoqin gu did not publish engine throughput or a notebook.
- **Model for the 60+ runs:** whether mikelou1 or NguyenThanhNhan (65.97) used the 27B or Flash-Next is unanswered in the threads.
- **No vLLM tok/s from the public notebooks:** their saved outputs were not included in the pull, so their logged vLLM "Avg generation throughput" values could not be read.
- **Missing pages:** the Kaggle model pages for the FP8 repacks render titles only, with no description of how the repack was made.

---

## 2. ARC-AGI-3 (single RTX PRO 6000; H100 originally planned): what the competitors use, and what transfers to L4

### Takeaway
- **All three Milestone-2 winners serve Qwen3.8-Flash-Next on one RTX PRO 6000 (TP1).**
  - Weights: 4-bit (W4A16 AutoRound or NVFP4).
  - The ~95 GB BF16 (or 47.7 GB FP8) per-layer-embedding (PLE) n-gram table is offloaded to host RAM.
  - FP8 KV; MTP/NEXTN with 3 draft tokens.
  - Engine: SGLang "Pennyroyal" (0.5.19 fork) or vLLM nightly.
- **Throughput:** peak decode 946–1,159 tok/s; Franzen's P90 aggregate decode is 837 tok/s.
- **What transfers to L4:** the harness and cache engineering (prefix-cache-aware admission, trim hysteresis, Mamba-checkpoint retention, prefetch, FP8 KV).
- **What does not transfer:** the SM120 kernels, NVFP4, FlashInfer GDN, the Pennyroyal QSA path and TP1.

### Cited Findings
- **Daniel Franzen, Milestone 2 (LB 27.89), model path:**
  - Path: Qwen3.6-27B FP8 → Qwen3.8-27B FP8 → Qwen3.8-Flash-Next; "Each switch brought a substantial improvement".
  - Quant: RadixArk NVFP4 first, then Intel W4A16 AutoRound ("similar quality and throughput, while leaving more VRAM available for the KV cache").
  - Draft: albucino INT4 MTP drafter. The BF16 PLE is kept in host RAM.
  - Engine: Pennyroyal (John Pezzulli's SGLang fork) v2.5.3 at `d00d88e`.
  - Patches: low-M BF16 GEMM, a spec-state memory budget fix, a Marlin scale-dtype fix, a prefix-cache patch, and a checkpoint-prefetch patch ("read shards in the loader's order with parallel readers and keep prefetching one shard ahead"). The model was resharded "to make loading more practical".
  - Source: [WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md); local `D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md`.
- **Franzen settings table** [MEASURED config]:
  - 10 parallel streams; harness context 131,072; max output 12,288; server context 139,264; drain 59,392.
  - FP8 E4M3 KV; Mamba cache 60 slots; mem fraction 0.96; chunked prefill 8,192; 3 speculative steps.
  - Sampling: temp 0.7 / top_p 0.95 / top_k 20.
  - Source: [WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md) §1.
- **Franzen exact launch** (pulled notebook cell 12):
  - Core: `sglang serve ... --tensor-parallel-size 1 --dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96 --context-length 139264 --page-size 64 --max-running-requests 10 --chunked-prefill-size 8192 --max-prefill-tokens 16384`.
  - CUDA graphs: `--cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10`.
  - Mamba and linear attention: `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --mamba-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer`.
  - Model and parsing: `--moe-runner-backend auto --ple-offload-embedding --reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"preserve_thinking":true}'`.
  - Scheduling and monitoring: `--schedule-policy lpm --enable-cache-report --enable-metrics`.
  - Loading: `--weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load`.
  - Speculation: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, draft quantization `compressed-tensors`, plus the FR-Spec 64k hot-token map.
  - Env: `SGLANG_MAMBA_CONV_DTYPE=bfloat16`; `SERVER_STARTUP_TIMEOUT = 12*60`.
  - Source: [notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), pulled to `D:/DevCache/Temp/research_kernels/dfranzen_arc-agi-3-milestone-2-solution/`; local cell dump `_scratch/ext/dfranzen_cells_4_12_16.txt`.
- **Franzen serving metrics** (earlier demo run) [MEASURED]:
  - prefix-cache reuse 93.43% of prompt tokens;
  - P90 effective prefill 10,552.6 tok/s;
  - **P90 aggregate decode 836.6 tok/s**; **harness-side generation 609.0 tok/s**;
  - prefill time fraction 15.25%.
  - Source: [WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md) §2.
  - Our validation of his version outputs: "ready at 562 s; 590-617 tok/s; host RAM 106/177 GB; GPU 91/96 GB"; local `D:/ARC AGI 2/experiments/Experiment 13 - .../VALIDATION.md` row 2.
- **Tong Hui Kang's comparison of the three Milestone-2 winners** [MEASURED from their notebooks]:

  | | dfranzen 27.89 | lordhansolo 23.84 | sirikilohit 22.53 |
  |---|---|---|---|
  | Engine | SGLang Pennyroyal v2.5.3 | vLLM 0.29.1rc1 nightly e975732 | SGLang Pennyroyal v2.5.0 |
  | Weights | Intel W4A16 AutoRound + BF16 PLE; albucino INT4 MTP draft | primitive-ai mixed NVFP4-FP8 + BF16 PLE (built-in MTP) | Intel W4A16 + RadixArk FP8 PLE; NVFP4 MTP draft |
  | GPU weights | 73.64 GB | 71.94 GB | 73.57 GB |
  | Offloaded to host | 95.37 GB | 96.55 GB | 95.68 GB |
  | KV dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
  | KV pool (tokens) | 1,011,264 | 1,417,100 | 1,004,288 |
  | Context (server / harness) | 139,264 / 131,072 | 147,072 / 127,488 | 69,632 / 69,632 |
  | Streams | 10 | 14 | 16 |
  | Notebook start to ready | **531 s** | **544 s** | **615 s** |
  | Peak decode | **946 tok/s** | **1,135 tok/s** | **1,159 tok/s** |
  | Temperature | 0.7 | 0.6 | 0.6 |

  Source: local `D:/ARC AGI 2/_scratch/webresearch/d_744792.md` (ARC-AGI-3 topic 744792, "Comparing the Milestone 2 winning solutions").
- **Comments on that thread:**
  - **rellik13:** "the kv pool sat somewhere between 91-97% full and requests queued for cache space, so the memory was the actual limit ... fp8 kv ... I spent the free memory on history (the trims moved from 37k->27k to 57k->45k). I guess this alone took the score from 14.5 to 22.5".
  - **xz:** "very little NLL penalty to use nvfp4 kv".
  - **Son Pham:** a REAP-pruned Qwen3.8 opened "an entire untapped area".
  - **NNbody:** "redundancy in the mamba slot allocation for Qwen 3.8 Flash Next".
  - Source: local `d_744792.md`.
- **keithtyser `duck-qwen3-8-flash-next-nvfp4-mtp`** (RTX PRO 6000, pulled):
  - "pinned `RadixArk/Qwen3.8-Flash-Next-NVFP4` ... ModelOpt NVFP4 weights, BF16 compute, async scheduling, chunked prefill, native three-token NEXTN MTP speculative decoding, 32K context, an 8K batched-token cap, eight vLLM sequences, CUDA graphs, disabled prefix caching ... Duck retains 28-way game concurrency".
  - Identical forks: `tantan0327/arc3-flashnext-asis` and `wuliao0/duck-qwen3-8-anim-base` (same md5).
  - Source: pulled notebook markdown cell 0; [kaggle](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp).
- **27B FP8 in ARC-AGI-3:**
  - `keithtyser/duck-qwen3-8-27b-fp8` and `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` both use the foysal FP8 repack on RTX PRO 6000 (`kernel-metadata.json`).
  - **Ya Xu:** Qwen3.8-27B 8-bit was "achieving a consistent 2x score on the local 25 dataset" vs Qwen3.6 27B; "'xhigh' is simply better than 'low'" (local `d_735243.md`).
- **Tufa Labs (Milestone 1):**
  - hardware: "a RTX Pro 6000 GPU with 96 GB of VRAM and 9 hours of runtime to play 110 games";
  - model: Qwen 3.6 27B FP8 on vLLM;
  - host statement: "Our plan is to implement the H100s for the ARC-AGI-3 competition".
  - Sources: local `d_717133.md`; [topic 689054](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/689054).
- **Training Flash-Next is hard** (ARC-AGI-3 topic 748593): "Its MoE structure makes training unstable"; Scott Le Grand "got 2 RHAE points ... out of 3.8-27B with SFT ... haven't been able to get anything out of flash-next". Source: local `d_748593.md`.
- **SGLang cookbook, Flash-Next on 1× RTX PRO 6000 TP1** [MEASURED, 1024 in / 256 out]:
  - 1 request: 148 tok/s with MTP vs 83 without;
  - 16 requests: 613 vs 524;
  - 64 requests (no MTP): 861 tok/s.
  - NVFP4 is Blackwell-only. Needs SGLang ≥ 0.5.22.
  - Source: [docs.sglang.io cookbook Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next) (subagent fetch).

**What transfers to 4×L4, and what does not:**
- **Does not transfer:**
  - SM120 env vars (`SGLANG_ENABLE_SM120_LOWM_BF16_GEMM`, etc.) are absent from SGLang 0.5.21.
  - FlashInfer GDN is "only for the validated SM90/SM100 domains, otherwise Triton".
  - Pennyroyal's QSA decode has no sm89-safe path; our validation verdict: "Pennyroyal dropped for L4".
  - Source: local `Experiment 13 .../VALIDATION.md` rows 4, 14, 15.
- **NVFP4:** Blackwell-only (SGLang cookbook; NVIDIA NVFP4 card targets B200/B300). Source: [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (subagent).
- **QSA sparse decode on sm89:** "QSA sparse decode on NVIDIA SM89 requires classic FlashAttention / FA2"; FA4/CuTe at head dim 256 fails. Source: [sglang PR #36968](https://github.com/sgl-project/sglang/pull/36968) (subagent).
- **Transfers:**
  - prefix-cache-aware admission ("limited admission to ten game streams, matching the server's request limit, to keep their cached prefixes resident");
  - trim hysteresis; end-of-prefill Mamba checkpoint retention;
  - `--schedule-policy lpm`, `--page-size 64`, `--mamba-track-interval 64`;
  - FP8 KV (on vLLM sm89 only with the Triton attention backend);
  - parallel prefetch / reshard.
  - Source: [WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md) §1–2. Flag existence in 0.5.21 is per our VALIDATION row 10.

### Inferences
- **Most of the ARC-AGI-3 throughput advantage is hardware.** The RTX PRO 6000 has ~1.8 TB/s on one GPU with no TP all-reduce, against 4 × 300 GB/s over PCIe. Its 946–1,159 tok/s peak decode is about 2.5–3× the best measured 27B numbers on 4×L4.
- **The ARC-AGI-3 software lessons most relevant to ARC-AGI-2 on L4 are memory and cache management, not kernels:**
  - KV-resident admission and FP8 KV (rellik13's 14.5 → 22.5 from more KV and history);
  - trimming in large blocks;
  - fast, parallel weight loading.

### Gaps
- **No ARC-AGI-3 competitor reports Qwen3.8 on H100.** The H100 was the original plan; all ARC-AGI-3 evidence found is RTX PRO 6000.
- **Final-config demo run:** Franzen did not publish one ("I reached Kaggle's compute limit before I could do a full demo run with the final submission configuration").

---

## 3. Model cards and community quants: serving recommendations and reported speeds

### Takeaway
- **What the official recipes give:** MTP (EAGLE 3/1/4 in SGLang; `mtp` with 3 tokens in vLLM), plus the DFlash2 and DSpark drafters.
- **Fastest single-stream setup across community reports:** W4A16/INT4 weights with the DFlash2 drafter.
- **The key caveat for batch serving:** speculative gains shrink, and can turn negative, at high concurrency on this hybrid GDN model.
  - DFlash2 is 3.34× at c1 but only 1.30× at c32 on an H200; MTP falls to 0.94×.
  - On sm89 with vLLM INT4, MTP caps the scheduler at about 3 sequences.
- **Hardware coverage:** no card or recipe gives an L4 or multi-L4 vLLM/SGLang number.

### Cited Findings

**Qwen/Qwen3.8-27B card:**
- **Architecture:**
  - 64 layers = "16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))";
  - Gated Attention: 24 Q / **4 KV heads**, head dim 256;
  - GDN: 48 V / 16 QK heads.
- **Context:** 262,144 native.
- **Effort:** xhigh is the default.
- **Sampling (thinking):** temp 1.0, top_p 0.95, top_k 20.
- **Recipes linked:** SGLang cookbook, vLLM recipes, TokenSpeed.
- **Speed:** no numbers.
- Source: [huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (subagent).

**SGLang cookbook (27B):**
- **Speculative options:**
  - MTP `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` ("NEXTN is an alias of EAGLE");
  - DFlash2 `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8`;
  - DSpark (`RadixArk/Qwen3.8-27B-DSpark`).
- **SSM state:** `--mamba-ssm-dtype` slot 153.9 MB at fp32 vs 78.4 MB at bf16; "Treat bfloat16 as an accuracy gate".
- **Cache:** `--mamba-radix-cache-strategy extra_buffer_lazy` cuts state slots per request from 5 to 4.
- **Coverage:** validated on v0.5.19; no L4/A100/H100 mention.
- **[MEASURED] RTX 5090:**
  - DFlash2 4.92 ms TPOT at accept length 4.29;
  - single-request EAGLE: NVFP4 152.9 / 144.5 tok/s (fp32 / bf16 state), FP8 106.3 / 116.1.
- Source: [docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) (subagent).

**vLLM recipe (27B):**
- **Versions:** vLLM ≥ 0.17.0 (≥ 0.28.0 for DFlash).
- **Commands:**
  - `vllm serve Qwen/Qwen3.8-27B-FP8 --tensor-parallel-size 4 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`;
  - DFlash: `num_speculative_tokens` 7.
- **[MEASURED] 2× RTX 5090, TP2:**
  - KV tokens: FP8 377,456; Inferact NVFP4 445,875; unsloth NVFP4 920,517;
  - MTP acceptance 0.771 / 0.897 / 0.788.
- **[MEASURED] 1× 5090:** `--language-model-only` raised KV from 91,022 to 135,926 tokens.
- Source: [recipes.vllm.ai/Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) (subagent).

**Qwen/Qwen3.8-27B-FP8:**
- **Format:** fine-grained FP8, block 128.
- **Discussion #16** [MEASURED log, RTX 4090 48 GB, vLLM v0.27.1]: "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!"
  - Workaround: tuned configs (github.com/massif-01/vllm_benchmark_block_fp8, kryoz/4090-48gb-vllm-fp8).
  - Our own E12-K1 plan found the same warning for all 5 TP4 shapes on L4 (local `D:/ARC AGI 2/experiments/Experiment 12 - FP8 kernel tuning on L4 (planned)/README.md`).
- **Discussion #13:** claims the FP8 MTP head lowers acceptance by about 3–4% per position (unconfirmed).
- **Discussion #19** [MEASURED]: 132.0 tok/s single request on a 4090 48 GB with the custom NInfer engine (DFlash2, 5 draft tokens, 62.8% acceptance).
- Sources: [discussions/16](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/16), [/13](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/13), [/19](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/19) (subagent).

**RedHatAI/Qwen3.8-27B-INT4:**
- **Format:** "AWQ smoothing and the W4A16 GPTQ scheme, exported in compressed-tensors format", FP8 KV. Kept unquantized: vision, lm_head, embed_tokens, `linear_attn.in_proj_a/b`.
- **vLLM command:** `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; "for optimal performance", DSpark with 7 tokens.
- **Accuracy recovery [CLAIM]:** GPQA-D 98.49%, AIME25 98.69%, MMLU-Pro 98.81%.
- **Speed and adoption:** no speed numbers; 341,574 downloads.
- Source: [huggingface.co/RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) (subagent).
- **Our measured size:** "INT4 19.47 GB + DFlash2 3.85 GB = 23.3 GB" (local `Kaggle discussions (2026-10-22)/00_SUMMARY.md`).

**incoai/Qwen3.8-27B-DFlash2:**
- **What it is:** "block-diffusion drafter", 2B BF16, block 8 (7 draft tokens); lossless.
- **Setup [MEASURED, re-fetched by me]:** 1× H200, SGLang, FA3, temp 1.0 / top_p 0.95 / top_k 20, xhigh, 4,096 max new tokens.
- **MATH-500 throughput (plain decoding / MTP / DFlash2):**

  | Concurrency | Plain tok/s | MTP tok/s | DFlash2 tok/s |
  |---|---|---|---|
  | 1 | 69.0 | 172.8 (2.51×) | 230.7 (3.34×) |
  | 8 | 480 | 1,023.5 (2.13×) | 1,368.3 (2.85×) |
  | 32 | 1,505.8 | 1,415.6 (**0.94×**) | 1,951.8 (**1.30×**) |

- **At c32 on MT-Bench:** MTP 0.77×, DFlash2 1.01×.
- **Acceptance length:** DFlash2 5.28 vs MTP 4.72 (MATH-500).
- **Discussions:**
  - DFlash2 "works with the FP8 quant target model ... acceptance length is very close to the BF16 target";
  - "Positions 4-7 usually get around 0.1 or less";
  - "Reducing to 3 tokens makes DFlash perform similarly to MTP".
- Sources: [card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2); [discussions/2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/2); [/9](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9).
- **Prefix-cache interaction** [MEASURED, vLLM v0.30.0, RTX PRO 6000]:
  - hybrid cache blocks are 1,584 tokens (1,616 with MTP K=4), so prompts under ~5K never hit the cache;
  - with DFlash2, "the lookup granularity becomes 6,592 tokens".
  - Source: [Qwen3.8-27B discussions/169](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/169) (subagent).

**ukisai/Swift-Qwen3.8-27b and Swift-1.5** [CLAIM]:
- **Token savings:** "58.3% fewer thinking tokens" (GPQA-D median; mean 41.0% at xhigh); "1.95× speed-up" on unspecified hardware.
- **Scores:** GPQA-D 88.38 → 88.28; AIME26 98.67 → 94.00. Swift 1.5: AIME26 98.67 → 96.00 with 22,014 → 13,203 tokens.
- **Missing:** no ARC result. A separate licence applies above US$1M revenue.
- **Quants available:** INT4, W4A16-AWQ, AutoRound, NVFP4, GGUF.
- Sources: [Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b), [Swift-1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (subagent).
- **First raised in the competition by** Darshan Makwana in [746149](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/746149).
- **ThinkingCap-Qwen3.8-27B:** "cuts reasoning tokens by 37% on average"; gated, PolyForm licence. Source: [card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) (subagent).

**Qwen/Qwen3.8-Flash-Next:**
- **Size:** "125B with 6B activated, plus 51B n-gram embedding and 4B MTP".
- **Architecture:** 48 layers = "12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))"; 512 experts, 10 routed + 1 shared, expert size 640; 2 KV heads.
- **Benchmarks vs the 27B [CLAIM]:** GPQA 91.7 vs 89.2; HLE 35.9 vs 30.8.
- **vLLM recipe:**
  - needs ≥ 0.30.0;
  - memory: FP8 172.78 GiB, BF16 335.28 GiB;
  - the n-gram table is CPU-offloaded by default ("≥51 GB of host RAM");
  - "4xH100 with plain TP4 runs out of memory without that offload".
- **LMSYS:** offloading the 51.2B n-gram embedding to host changed throughput by −0.07% (H200 TP4).
- **Intel/Qwen3.8-Flash-Next-W4A16-AutoRound:** experts only; 99.64% average recovery; no speed numbers.
- **unsloth GGUF:** UD-IQ4_XS 93.7 GB … Q8_0 192 GB.
- Sources: [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), [LMSYS blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/), [Intel AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) (subagents).

**Concurrency and speculation reports from GitHub and community benchmarks** (all [MEASURED] by their authors):
- **vLLM issue #55533** (RTX 4090 D, sm89; GPTQ INT4 Marlin; MTP k=1; also reproduced on an RTX 6000 Ada with RedHatAI/Qwen3.8-27B-INT4):
  - no-spec vs MTP: bs1 56.4 vs 72.0; bs8 367.1 vs 197.0; **bs20 649.0 vs 210.9 tok/s**;
  - the scheduler ran "exactly 3 of the 8 sequences", because MTP reserves extra GDN state blocks;
  - no fix yet.
  - Source: [vllm#55533](https://github.com/vllm-project/vllm/issues/55533).
- **club-3090 #1288** (re-fetched by me): 4× RTX 3090 at 220 W, NVLink pairs, SGLang v0.5.19, TP4, W8A8 INT8, bf16 KV, unquantized DFlash2 with 8 draft tokens, HiCache, `max_running_requests=16`, `NCCL_P2P_LEVEL=SYS`:
  - c1: narrative 146.6, code 265.5 tok/s;
  - aggregate narrative / code: c4 487.7 / 893.9, c8 730.6 / 1,418.3, **c16 927.5 / 1,811.0**;
  - prefill peak 5,058 tok/s;
  - "TTFT grows roughly linearly with context because DeltaNet/SSM state is not prefix-cacheable".
  - Source: [club-3090 discussion #1288](https://github.com/noonghunna/club-3090/discussions/1288).
- **club-3090 #1092** (2× 3090 PCIe, vLLM v0.27.1, TP2, W4A8 AutoRound):
  - MTP n=4 vs spec-off: C1 71 vs 58; C8 201 vs 211; **C32 241 vs 308**;
  - "DFlash2 can't do concurrency on 24 GB" (OOM at C8);
  - pinning `MAX_MAMBA_CACHE_SIZE=20` raised the KV pool from 451k to 548k tokens.
  - Source: [club-3090 #1092](https://github.com/noonghunna/club-3090/discussions/1092).
- **club-3090 #1746** (2× 3090 NVLink, vLLM 0.31.0, TP2, Swift 1.5):
  - single-stream DFlash2 136.5 tok/s vs MTP k=3 101.9;
  - speculation "stops paying at 4+ clients".
  - Source: [club-3090 #1746](https://github.com/noonghunna/club-3090/issues/1746).
- **SGLang #36889:** the DFLASH mamba pool silently caps concurrency (5 state slots per request); measured on GLM-5.3-Flash, c8 went from 47.3 to 80.3 tok/s when the pool was raised from 10 to 40 slots.
  - Related: club-3090 #1319 reports 33 mamba slots with DFlash2 vs 55 with MTP.
  - Sources: [sglang#36889](https://github.com/sgl-project/sglang/issues/36889), [club-3090#1319](https://github.com/noonghunna/club-3090/issues/1319).
- **RTX 4090 48 GB sm89, SGLang 0.5.16 → 0.5.19** (single card):
  - W4A16-AWQ + DFlash2 vs FP8 + MTP: code **245.6 vs 75.7** tok/s; math 167.4 vs 64.9; Chinese prose 56.3 vs 47.3;
  - 2-way concurrent aggregate 349 vs 136;
  - 0.5.19 adds a 1.69 GB `intermediate_ssm_state_cache`;
  - nvcc pinned to 13.0.88.
  - Source: [lcz.me/topic/1712](https://lcz.me/topic/1712) (subagent).
- **vLLM PR #59232:** on sm89, block-FP8 falls back to Marlin W8A16; tuning Triton block-FP8 for L20 gave ~1% in the latest revision (5–31% earlier).
  - Related: L40S FP8 Triton config tuning gave +30.5% at c=1 and +3.8% at c=128.
  - Sources: [vllm#59232](https://github.com/vllm-project/vllm/pull/59232), [L40S tuning repo](https://github.com/amanoxsolutions/vllm-qwen3-8-27b-fp8-l40s-kernel-config-tuning) (subagent).
- **Single L4, llama.cpp only:**
  - vramcalculator, UD-Q4_K_M, no MTP: 13.1 tok/s at 29K context, 6.1 at 236K; prefill 647 → 217;
  - hanxiao, UD-Q4_K_XL: 14.98 tok/s plain, 28.05 with MTP depth 2 plus a kernel patch, **37.4 with DFlash2** (7 drafts); "Effective bandwidth 252.8 of 300 GB/s".
  - Sources: [vramcalculator](https://vramcalculator.com/qwen3-8-27b-speed/), [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) (subagents).
- **HyperQwen** (RTX 3090, patched vLLM 0.31.0):
  - 127 tok/s single user, ~1,035 tok/s at 64 concurrent;
  - "Speculation wins below ~8 concurrent users, plain batching above";
  - "Multi-GPU works at TP=2 and TP=4"; "TP=3 ... invalid".
  - The Kaggle notebook yjsh2003 "HyperQwen v7" uses this stack at TP4/C4.
  - Source: [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) (subagent).
- **DGX Spark TP vs DP study:** "two TP=2 replicas are the more effective configuration"; at C8, TP2 + DP2 gives 36.0 tok/s per request vs TP4 27.0. Source: [NVIDIA forum](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102) (subagent).
- **Quality notes:**
  - FP8 KV at 256k: MRCR 0.717 vs BF16 KV 0.833 (n=15) ([discussions/182](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/182));
  - thinking loops of 20k–55k tokens fixed with `thinking_budget` ([discussions/178](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/178));
  - SGLang defaults top_k to −1 vs the card's 20 (same thread);
  - `--mamba-full-memory-ratio 2.5` "consumes 15.2 GB, starving the KV pool" ([discussions/160](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/160)).
  - All via subagent.

### Inferences
- **On 4×L4, the weights choice that the evidence supports is INT4/W4A16 + DFlash2**, provided the mamba-slot and KV budget is checked.
  - The only same-architecture (sm89) A/B shows ~3× faster decode on structured output (lcz.me 4090).
  - INT4 roughly halves the bytes read per step vs FP8.
  - On sm89, block-FP8 uses default or Marlin kernels (HF #16, vLLM #59232).
- **At ARC batch sizes (10–16 streams), DFlash2's edge over MTP should persist but shrink.** Two vLLM-on-sm89 results suggest MTP can be net-negative above about 4–8 concurrent requests: the #55533 scheduler cap and the club-3090 C32 result. This is a candidate explanation for weak vLLM+MTP@16 public notebooks, but it was not tested on L4.
- **The L4 is relatively compute-rich for its bandwidth.** Per the NVIDIA datasheet (my background figures, not fetched): ~121 dense BF16 TFLOPS against 300 GB/s, vs an H200's ~989 TFLOPS against 4.8 TB/s. So the concurrency at which speculation stops paying is likely *higher* on L4 than the H200's ~c32. This is an untested inference.
- **DP2×TP2 may beat TP4 per request** (DGX Spark study). On 4×L4, the 27B INT4 at TP2 puts ~10 GB of weights on each 24 GB card, leaving ~10 GB each for KV and mamba state. This is untested; the Spark result may not transfer to PCIe L4.

### Gaps
- **No SGLang or vLLM throughput for Qwen3.8 on L4**, single or multi-GPU, was found in any card, issue or blog.
- **No direct measurement of the `mamba_ssm_dtype` fp16 vs fp32 throughput** on Qwen3.8 was found.
- **No SGLang 0.5.20/0.5.21 regression issue** was found.

---

## 4. Is Flash-Next faster than the 27B on 4×L4, and is there evidence on its ARC accuracy?

### Takeaway
- **No one has published a working, fast Flash-Next measurement on L4.**
  - **Our team's E3 attempt:** about 6 tok/s with garbage output.
  - **Kaggle competitors:**
    - Adarsh: "hasn't worked out that well".
    - Darren: "4 L4 seems to be very slow ... due to lower bandwidth and slow gpu communication".
- **Memory constrains it.** The ~70 GB of 4-bit weights on GPU leave little KV room on 96 GB, and Flash-Next needs expert parallelism (EP) on TP4.
- **The ARC-AGI-2 accuracy evidence on Kaggle is thin and mixed:**
  - Hiểu Vy: 26.81 LB (27B) vs **8.61** (Flash);
  - amn: an agent claim without numbers;
  - mikelou1 and NguyenThanhNhan: model undisclosed.
- **Flash-Next clearly beat the 27B in ARC-AGI-3**, on a single RTX PRO 6000 (Franzen).

### Cited Findings
- **Hiểu Vy:** "I scored 26.81 with Qwen 3.8 27B and 8.61 with Qwen Flash"; also "quantizing [dots3] down to 68 GB ... still very slow—much slower than Qwen Flash". Sources: [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082), [746149](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/746149).
- **amn:** "FN with an acceleration (eg. quant?) might reach a better score according to my experiments, by letting it work as an agent". No numbers.
  - Our earlier check: amn's high LB score is "**not** shown to be Flash-Next" (local `Experiment 13 .../VALIDATION.md` row 23).
  - amn's 0.00 runs read "~110 GiB" (topic 751344).
  - Source: [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082).
- **Adarsh:** "Qwen 3.8 Flash-Next was promising initially as Franzen's milestone soln. on arc-agi-3 was based on flash next, but it hasn't worked out that well for me as L4 GPUs based on Triton architectures aren't that fast as hopper ones". Source: [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546).
- **Darren:**
  - "In arc agi 3, everybody seems to be using qwen 3.8 flash next since rtx pro 6000 has a very high bandwidth. However 4 L4 seems to be very slow when running these models";
  - "Qwen 3.8 flash next might have a better shot at 4 L4 since we can offload its embedding layer into ram".
  - Sources: [745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787), [746149](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/746149).
- **NguyenThanhNhan (10-05):** "Next stop is flash next, but the queue for 4xL4 has been insane". He did not report a Flash-Next result later. Source: [745787](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787).
- **Our E3 run** (Flash-Next on 4×L4, SGLang Pennyroyal wheel, Intel W4A16 AutoRound + FP8 PLE overlay):
  - `--weight-loader-prefetch-checkpoints` brought weight loading to 345 s;
  - server ready at 862 s with a KV pool of 118k tokens;
  - output was "garbage ("!!!!", token 0; a TileLang data-race warning in the QSA kernel)";
  - "single-stream decode about 6 tok/s";
  - "Not competitive with the 27B (about 373 tok/s at 16 running)".
  - Source: local `D:/ARC AGI 2/experiments/Experiment 3 - Flash-Next Feasibility (FN)/README.md`.
- **Our E13 validation:**
  - "EP4 needed: TP4 MoE split 640/4 = 160 is not a multiple of group 128; AutoRound MoE needs runner `auto`";
  - "W4A16 loads on 4xL4 at 19.06 GB/GPU";
  - "Host RAM on Kaggle 4xL4 is 202.6 GB";
  - sm89 QSA decode falls back to FA2, then FA4 CuTe, and our wheelhouse "has `flash_attn_4-4.0.0b35` and `tilelang-0.1.12`, and no classic FA2".
  - Source: local `Experiment 13 .../VALIDATION.md` rows 5, 6, 13, 16, 17.
- **Upstream on sm89:** "QSA sparse decode on NVIDIA SM89 requires classic FlashAttention / FA2", tested on four L20s with the FP8 checkpoint (~173 GB). No tok/s. Source: [sglang PR #36968](https://github.com/sgl-project/sglang/pull/36968) (subagent).
- **Closest multi-GPU consumer analog: 4× RTX 3090**, vLLM nightly, TP4 + EP, W4A16 AutoRound experts (Marlin), FP8 n-gram table read from NVMe, MTP 3, fp8_e5m2 KV ("FP8 e4m3 KV does not compile on 3090s"):
  - single stream: prose 109.5, code 145.8 tok/s;
  - **6 concurrent agents: 252.6 tok/s aggregate** (317 sustained / 570 peak).
  - Source: [tonyd2wild/Qwen38-Flash-Next-4x3090](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090) (subagent).
- **Same-class MoE on 4×L4:** a Qwen3.5-MoE 35B-A3B (Nex-N2-mini) on vLLM 0.26 fp8 ran at 438–452 tok/s. Source: [732723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723).
- **ARC-AGI-3 (RTX PRO 6000):** Franzen says each model switch, including 27B → Flash-Next, "brought a substantial improvement"; all three Milestone-2 winners use Flash-Next. Sources: [WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md); local `d_744792.md`.
- **No official ARC result exists for Flash-Next:** `https://arcprize.org/results/alibaba-qwen3-8-flash-next` returns 404 (subagent).
- **Model card [CLAIM]:** Flash-Next beats the 27B on GPQA (91.7 vs 89.2) and HLE (35.9 vs 30.8) ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), subagent).
- **Licence:** whether Flash-Next's Qwen Community License satisfies the OSI prize rule was asked twice and not answered by the hosts. Sources: [746307](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/746307), [750575](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750575); ARC-AGI-3 topics 745079 / 745837 (local list `_scratch/webresearch/a3topics.txt`).

### Inferences
- **Why "6B active" does not translate into 4–5× the 27B's speed under batching:**
  - With 512 experts and 10 routed per token, a decode step at batch 16 touches about 1 − (502/512)^16 ≈ 27% of each layer's experts; at batch 64, about 72%.
  - So per-step weight traffic grows quickly with batch. At ARC-style 10–16 streams, Flash-Next's per-GPU bytes per step are of the same order as a 27B INT4 under TP4.
  - Flash-Next also needs EP and the sm89 sparse-attention fallback, and has much less free VRAM for KV.
  - This is my arithmetic; nobody has measured it on L4.
- **Bandwidth-scaled estimate:** the 4×3090 Flash-Next result (252.6 tok/s at 6 agents) scaled by bandwidth (~1/3) suggests roughly 80–190 tok/s on 4×L4 at 6 streams. That is *below* the 27B's measured 321–431. This is a speculative estimate.
- **Memory:** after ~70 GB of 4-bit weights on 96 GB of VRAM, the KV/mamba pool for long ARC reasoning (60–100k tokens per stream) would be small. Contrast the 27B INT4 (19.5 GB weights), which leaves ~70 GB.
- **Net:** on current evidence, Flash-Next on 4×L4 is unproven and likely slower per useful token than the 27B. It is a research bet (E13), not a drop-in speed-up. That is consistent with Hiểu Vy's 8.61 and Adarsh's report.

### Gaps
- **No tok/s for Flash-Next on any Ada GPU** (L4, L20, L40, 4090), and no ARC-AGI-2 score for Flash-Next on 4×L4 other than Hiểu Vy's single 8.61.
- **The model behind the 60+ LB runs is unknown.** It is unknown whether mikelou1 or NguyenThanhNhan (180 GB read) used Flash-Next.

---

## 5. Does the GCSFUSE/NFS `/kaggle/input` slowdown (from about 2026-10-21) change which setup is best?

### Takeaway
- **Yes. It strongly favours small weight files and parallel, page-cache-friendly loading.**
- **Every reported 0.00 involved large reads (≥100 GB):**
  - NguyenThanhNhan ~180 GB;
  - amn ~110 GiB;
  - Jeroen 100.5 GiB.
- **The 27B INT4 + DFlash2 (23.3 GB) or the FP8 repack (30.9 GB) reads 4–8× fewer bytes than any Flash-Next setup (~120–185 GB).** The new risk becomes a fixed launch timeout, not throughput.

### Cited Findings
- **Mount change and failures:**
  - "part of Kaggle's uploaded files by category comes to NFS, and the other part is GCSFUSE reportedly much slower" (amn);
  - NguyenThanhNhan: "FUSE.GCSFUSE is much slower than NFS and reading would take hours ... the server startup never finished in time, so no inference ran and placeholder submission jsons were returned".
  - Source: [751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344); local `topic_751344.md`.
- **amn's NFS probes** [MEASURED]: 0.55 GiB/s (10-19) → 0.95 (10-20) → 0.32 → 0.040 → **0.026 GiB/s** (10-21 05:37). On 10-22: "single-thread ... ~10.5MB/s, and for 16-thread it reported 107 MB/s". Source: [751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344).
- **Jeroen Cottaar** [MEASURED, ARC-AGI-3]:
  - "a 100.5 GiB safetensors checkpoint (10 shards) ... After 7 h 22 min, only 3 of the 10 shards had loaded, at 1.8–2.8 h per shard";
  - "An earlier run ... (2026-10-17) logged ... NFS and loaded all 10 shards in 88 s";
  - "Reads through the mount don't fill the page cache".
  - Source: [751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344).
- **Reported effects:**
  - Jeroen: "now usually scoring 0.00, and when it doesn't it scores 5-10% lower";
  - Kha Vo: "previous 2 submissions were unexpectedly worse than expected by 10 points" plus a Kaggle Error;
  - CPMP: "we lost one sub with Kaggle error after 12 hours ... The day before we already had FUSE";
  - mp: "~10 points lower";
  - amn: a second 0.00, "In my intentionally limited 3 hours, the inference engine has never managed to get started, since the reading of ~110 GiB data has not completed".
  - Source: [751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344).
- **mp's mitigation (untested):** "checking whether /kaggle/input is a FUSE mount ... If it's FUSE, try copying it to local disk once at startup, instead of letting all 4 GPU processes read it from the slow GCS Fuse mount". Source: [751344](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344).
- **Earlier slow-download sign:** CPMP (10-17): "I am observing very slow datasets download in interactive mode from time to time. And I see the GPU quota decrease in the meantime". Source: local `_scratch/webresearch/topteam_comments.md` (topic "Are datasets download time include din submission sunning time?").
- **Loader techniques already used by competitors:**
  - Franzen: resharding plus "parallel readers and keep prefetching one shard ahead" plus an external background prefetcher; `--weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load`; startup timeout 12 min.
  - sirikilohit: "parallel PLE row copy on 16 threads", "prefetch once".
  - khoinguyennguyen: page-cache preload with a `ThreadPoolExecutor`, 64 MiB chunks ("2–5 minutes, versus 6–10 minutes without preload").
  - Sources: [WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md); local `d_744792.md`; khoi notebook block 02.
- **Our exposure** (local `Kaggle discussions (2026-10-22)/00_SUMMARY.md`):
  - weights read at launch: 23.3 GB;
  - SGLang ready in 507–562 s on 10-19/20, but **1,197 s** on 10-21;
  - first-launch timeout 30 min;
  - a single-threaded `cat` prefetch at ~10.5 MB/s would take ~37 min.
- **Weight sizes on Kaggle** [MEASURED where stated]:
  - FP8 repack 30.89 GB (CLI listing);
  - RedHat INT4 + DFlash2 23.3 GB (our log);
  - Franzen Flash-Next W4A16 + BF16 PLE + drafter: "his precache read 185.4 GB (model + drafter)" (local `VALIDATION.md` row 3);
  - Flash-Next FP8 172.78 GiB ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next));
  - NVFP4 checkpoint 126 GiB ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)).

### Inferences
- **Load time by checkpoint at the measured read rates** (my arithmetic; ignores the engine's own init and CUDA-graph time):

  | Checkpoint (bytes to read) | 16 threads, ~107 MB/s (10-22) | 1 thread, ~10.5 MB/s (10-22) | NFS at its best, ~0.95 GiB/s |
  |---|---|---|---|
  | 27B INT4 + DFlash2 (23.3 GB) | ≈ 3.6 min | ≈ 37 min | ≈ 23 s |
  | 27B FP8 repack (30.9 GB) | ≈ 4.8 min | ≈ 49 min | ≈ 30 s |
  | Flash-Next W4A16 + FP8 PLE + drafter (~121 GB) | ≈ 19 min | ≈ 3.2 h | ≈ 2 min |
  | Flash-Next W4A16 + BF16 PLE + drafter (185 GB) | ≈ 29 min | ≈ 4.9 h | ≈ 3 min |

  The first row's sum uses Tong Hui Kang's on-GPU sizes: 69.8 GB target + 47.7 GB FP8 PLE + 3.8 GB draft.
- **At Jeroen's GCSFUSE rate,** about 1–1.5 MB/s through the loader's own reads, even the 23.3 GB INT4 set would take hours. So parallel bulk copying to local disk or RAM before launch matters for every setup, not just large ones.
  - Kaggle 4×L4 host RAM is 202.6 GB (local `VALIDATION.md` row 13), so a 23–31 GB 27B checkpoint fits easily in `/dev/shm` or the page cache; a 185 GB Flash-Next set does not fit comfortably.
  - This is an untested inference. The local-disk size on the 4×L4 image was not found.
- **Which setup is best:** the slowdown changes the choice more than any throughput difference does. The small 27B INT4 set loses minutes even in bad conditions, while a Flash-Next set can lose hours or the whole run. A bigger weight set needs a large measured throughput gain to pay back its load-time risk, and no such gain exists on L4 (Section 4).
- **Rerun timing:** the slowdown varies with time of day (amn's probes went from 0.32 to 0.026 GiB/s within about 2 h). A run's load time is therefore not predictable from a previous save run, which argues for a launch timeout tied to the real start-up progress rather than a fixed 30 min.

### Gaps
- **No host or Kaggle statement** on the cause, the duration, or which input categories are on GCSFUSE vs NFS. The threads are user observations only.
- **No reported result** yet for mp's copy-to-local-disk mitigation.
- **Missing facts for a copy-to-local plan:** the local scratch-disk size and throughput on the 4×L4 competition image, and whether `/dev/shm` is capped.
- **Weight sizes for the 0.00 runs:** whether CPMP's and Kha Vo's failures involved large weights (their models are NVARC-based, ~8 GB for a 4B) is unknown. So size is likely not the only factor.
