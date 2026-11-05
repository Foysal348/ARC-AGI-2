# Public-best parts report: what other public sources do better than, or in addition to, Domdolus (2026-11-04)

## Scope and sources

Every line of these files was read:
- `resources/Kaggle Writeups/writeup_755564_darrenamadeusmartin_FlashNext_AllYouNeed_2026-11-04.md` (Domdolus team writeup) - **DW**
- `resources/Kaggle Writeups/writeup_755321_notnitsuj_FlashNext_1100toks_2026-11-03.md` - **NW**
- `resources/Kaggle Writeups/COMPARE_ours_vs_notnitsuj_755321_2026-11-03.md` - **CMP**
- `resources/Kaggle Writeups/NOTES_andy2709_inducer_2026-11-03.md` - **AN1**
- `resources/Kaggle Writeups/NOTES_andy2709_inducer_runtime_2026-11-04.md` - **AN2**
- `resources/Kaggle Writeups/andy2709_inducer_latest/arc-flash-next-w4a16-inducer.py` (1,471 lines) - **K**
- `resources/Kaggle Writeups/NOTES_mikelou1_arc-agi2_2026-11-04.md` - **MN**

To fill the "present in Domdolus?" column, the extracted Domdolus cells in `Build/domdolus_cells/` were also searched (grep only) for the specific items. Those cites are given as `dd:cellNN:line`.

Labels:
- **[F]** = fact, quoted or cited from the source.
- **[I]** = my inference.
- AN2 and MN are our own agents' notes on code. Their [F]/[I] labels are kept as they were.

---

## 1. Per source: facts

### 1.1 Domdolus Tolus / darrenamadeusmartin (DW): LB 83.47
- **LB [F]:** "**16** | **83.47** / 72.36 / 74.58 / 79.19 / 78.06 | + sparse CUDA graphs, kernel trims, no token numbers in the prompt". Second pick: "**13** | **82.08** / 80.00 | 2 tries, 150k tokens".
  - [I] The same notebook (#16) scored 72.36-83.47 over 5 submissions, a spread of 11 points. 83.47 is the best of 5 draws, not the mean (mean of the 5 = 77.5).
- **Model [F]:** "the Q2_0 checkpoint from ISTA-DASLab" (GSQ) for experts and the n-gram table, official BF16 for the rest. "shared experts, routers, embeddings, norms, and vision tower were kept in BF16, while the attention and DeltaNet projections, the hyper-connection mixers, the LM head ..., and the n-gram table are stored in FP8".
- **Engine and layout [F]:** patched vLLM (0.31.0 per `dd:cell06:9`); TP + EP over 4 cards; DCP for the 2 QSA KV heads.
  - DCP details: the query is sent "in the background while the indexer runs"; partial attention outputs go as "BF16" instead of FP32.
  - Sampling: "we assign each request to 1 card"; "each card send only its 64 best tokens and their ids".
- **KV [F]:** "a capacity of about 4.4 million tokens, which is roughly 63k tokens per task at 70 tasks".
  - 4-bit KV with a Hadamard rotation and a per-token scale and offset.
  - FP8 index keys; 4 QSA layers per page group instead of 12.
  - DeltaNet state "FP16 with stochastic rounding".
- **Memory trims [F]:**
  - a shared top-k buffer;
  - a rotary cache sized to the window;
  - FP8 mixer padding to 64;
  - token embeddings in host RAM;
  - util 0.945;
  - a 128 MB indexer scratch;
  - images ≤ 2 MP, video off;
  - expandable segments.
- **Concurrency [F]:** "we managed to run 70 tasks concurrently". The final cell has `CONCURRENCY = 70` (`dd:cell46:6`); the server has `MAX_SEQS = 128` and `MTP_TOKENS = 0` (`dd:cell43:9-10`).
- **Speed [F]:** "roughly 30 million tokens in the 12-hour run (estimated from our 6-hour validation)". [I] That is ≈ 694 tok/s averaged over 12 h.
- **Context [F]:** a short-term memory window: "When it reaches 64k tokens, we cut the middle of the attention cache and keep only the system prompt and the last generated tokens, 48k tokens in total, while preserving the DeltaNet state". Plus a "1M window" in #15.
- **Recompute prevention [F]:**
  - prefix hits "every 800 tokens instead of every 6,400";
  - DeltaNet states at every 800-token boundary;
  - "exact resume" (parking each call's exact final state);
  - "token-id reuse";
  - newest DeltaNet state only, kept in RAM;
  - "RAM offload" of evicted KV;
  - a "collapse fix" that "holds back some calls when memory is under pressure and releases the held calls one by one".
- **Kernels [F]:**
  - indexed MoE GEMM;
  - split-K HC GEMM;
  - redundant Hadamard removed;
  - cut positions once per step;
  - BF16 top-k;
  - butterfly Hadamard;
  - fewer GDN copies;
  - fused HC-down+SiLU, QSA output gate and selected-token shift-back;
  - "Sparse CUDA graphs" that also cover mixed prefill+decode steps;
  - "NCCL instead of vLLM's custom all-reduce".
- **Startup [F]:** "from 13 minutes ... under 4 minutes": parallel reads of the utility into RAM, background model prefetch, each card reading only its own experts, the n-gram table once into /dev/shm, prebuilt Triton/Humming/FlashInfer caches.
- **Harness [F]:**
  - built on pivotools' REPL code (DUCK and Franzen inspired);
  - "puzzles in image format", text only via the REPL;
  - helpers including "a check function";
  - "It can only give its final answer if it creates a function that passes all training examples";
  - "2 tries ... with a maximum of 200k decoded tokens each";
  - best function used on budget-out;
  - "If both tries give the same answer, we use one of the functions generated along the way that produces a different answer".
- **Sampling, from the cells (not the writeup) [F]:**
  - temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0 (`dd:cell30:1770-1775`);
  - `REASONING_EFFORT = "xhigh"` (`dd:cell46:7`);
  - no per-request `max_tokens`: "max_tokens is not sent, so a reply may use whatever is left of the window" (`dd:cell30:219`);
  - task order `random.Random(SEED=0).shuffle(ids)` (`dd:cell46:38`);
  - try 2 "plain and queued from the start" (`dd:cell46:120`).
- **Deadline [F]:** "12 hours minus 5 minutes"; "One minute before this limit, the tries are stopped"; "The submission file is also rewritten every time a try finishes".

### 1.2 notnitsuj / okiek (NW, CMP): LB 75.69
- **LB [F]:** "Public leaderboard: 75.69 with RC8, 15th of 3,626 teams".
  - Local vs LB: v3 79.6 % / 72.64, v7 84.7 % / 70.42, RC8 86.6 % / 75.69.
  - "12-hour replicas score 84–87% locally".
- **Speed [F]:** "It reaches 1,087–1,120 tokens/s on Kaggle, where stock vLLM managed 140–310." Kaggle 2-h checks: "110 to 122 to 134 of 240 as throughput rose from 1,021 to 1,087 to 1,120 tokens/s".
- **Concurrency [F]:** "two data-parallel engines of 24 sequences each, each tensor-parallel over 2 GPUs, with the experts spread across all four"; "48 sessions running at once".
- **KV [F]:** "FP8 KV and dense projections, packed recurrent state and a shared top-k buffer raised the KV cache from 879k to 974k tokens per engine".
  - Under spec decoding: "Folding accepted tokens into one BF16 checkpoint, instead of keeping a copy per draft position, allowed 49% more concurrent sequences".
- **Model [F]:**
  - "320 of 512 experts in every layer, scored by a REAP-style saliency (gate × output norm) plus tool-call routing mass";
  - "GPTQ puts the 160 most-routed experts per layer at 4 bits and the rest at 3 bits: 8.7 GB of experts per GPU";
  - cost "About 6 REPL points against FP8".
  - Format warning: "a fifth of tool calls then lost their `<parameter=code>` header, and sessions verified 0.34 of the time instead of 0.78".
  - "In direct-answer mode, compressed builds were indistinguishable from FP8. In REPL mode, the same builds lost 5–12 points."
- **RL on quant scales [F]:** "We learn a log-offset per quantization group (589.8M parameters) and the kept router rows (39.3M)"; "+7.0 pass@2 points [+3.8, +10.6] ... with 12% shorter sessions"; "ships as a 1.3 GB delta".
- **Draft [F]:** "The MTP layer drafts 3 tokens per step."
  - Fine-tuning it for expected accepted length gave "2.75 → 3.01 tokens per step on 4×L4, +6.1% tokens/s".
  - "An int8 copy limited to the 32,768 most-used tokens made sampled drafting pay off: +19% tokens/s."
- **Collectives [F]:** "a one-kernel all-reduce over a registered shared-memory buffer, and host-staged all-gathers and reduce-scatters" inside the CUDA graph: "52–271 µs, against 114–347 µs with NCCL: +8.6% tokens/s". Expert dispatch and reduce-scatters in int8: "1.9–4.0% off each step for expert traffic and 0.8–1.4% for tensor-parallel traffic".
- **Server-side sessions [F]:** "Keeping each session's cache on the server and appending only the tool output cut prompt processing from 6.1 to 0.2 tokens per generated token."
- **Harness [F]:**
  - "digit grids", an `ak` API sheet, one tool `python`, a "fresh sandboxed process" per call; a `transform` gets "a masked diff".
  - "A session ends at its first verified program, after 24 calls or after 98,304 tokens".
  - Verification: "reproduces every training output, returns valid test grids, and breaks none of 11 invariants".
  - `ak`: "+2.6 [+0.2, +5.2] at 100k tokens per input, with verified precision 0.82 against 0.72".
  - Import fix "+2.25 points"; traceback fix "+0.88 points of verification".
- **Selection [F]:**
  - "Verified programs from about 3 sessions per task vote".
  - Tie-break by symmetry and "no code that runs only on a test input": "+0.54 [+0.28, +0.84]".
  - "attempt_2 is the most common other grid any program produced".
  - "A perfect pick among the produced candidates would add only +0.8".
- **Scheduling [F]:**
  - "Every task first gets one session; more follow while budget remains, each to the task with the fewest so far."
  - A task with "at least 2 votes and a lead goes to the back of the queue: +0.8".
  - A warm start from the best unverified program gives "+0.21 right answers per session, with 17% fewer tokens".
  - Admission: "fewer than 48 are running and the projected contexts of all sessions fit".
  - "a drain lets running sessions finish and answer: +1–3% tokens".
- **Token curve [F]:** "pass@2 on the shipped model is 80.6 / 83.2 / 86.8 / 89.9% at 80 / 100 / 160 / 320k tokens per test input"; "Every extra 100 tokens/s buys each test input about 12k more tokens over a run".
- **Sampling [F]:** "the best sampling temperature was model-specific (0.7 for Dots3-Note, 1.0 for Flash-Next)".

### 1.3 andy2709 "Inducer" (K, AN1, AN2): public LB 74.31 per the AN2 title
- **LB:** 74.31 appears only in the AN2 title. It is not in the code. Local public-120: "79.55 over 4 seeds" (K:156) and "81.11" for the v31 rehearsal (K:128).
- **Engine [F]:** SGLang 0.5.20 with pinned patches; `--tensor-parallel-size 4 --expert-parallel-size 4 --decode-context-parallel-size 2` (K:1155-1159).
- **Concurrency [F]:** `MAX_RUNNING_REQUESTS = 10` (K:52). "at 12 running free device memory fell to 0.05 GiB" (K:47-48).
- **KV [F]:**
  - `L4_KV_TOKENS = 331_776` rows a rank, "663,552 tokens across each DCP pair" (K:65-71);
  - FP8 KV (K:1164-1165); mem-fraction 0.92 (K:58).
- **Speed [F]** (measured warmup report, AN2 §6): decode "283.7 tok/s mean over a 639 s window at mean 5.94 running, mean accept length 3.013"; the median at 10 running is **402.5 tok/s**.
- **Model [F]:** W4A16 AutoRound (dfranzen); "480 of 512 routed experts (mass-ranked REAP selection) in every target layer and in the MTP draft"; INT8 dense layers 4-43 "frees 0.23 GiB per L4 at the same speed" (K:152-158).
- **Spec decoding [F]:** NEXTN, 3 steps, 4 draft tokens (K:1240-1247); the draft is refit "accept length 2.807 -> 2.992 on 1,939 requests" (K:161-163); a hot-token map (27,982 ids, held-out coverage 0.999581, AN2 §3.2).
- **Kernel gains [F], from K comments:**
  - MoE down + AR fusion: "-6.4 / -4.9% at 8/10 running" (K:131-137);
  - HC mix CUDA: "-4.6 / -5.1%" (K:138-141);
  - HC local up INT8: "decode step -4/-5% at 8/10 running, prefill -18% a batch" (K:1085-1087);
  - top-32 logits: "-3.6% decode step" (K:118-121);
  - breakable prefill graph: "-6.5 to -6.8%" (K:125-127);
  - QSA indexer decode kernel: TileLang "costs 2.46 ms per decode step at 10 running" (K:143-147);
  - host all-reduce: "neutral on Kaggle's L4s" (K:108-111);
  - `NCCL_GRAPH_MIXING_SUPPORT=0`: graph mixing "only adds 17-60 us per captured collective" (K:1088-1090).
- **Expert placement [F]** (AN2 §3.2): balanced placement, "held-out max activated per GPU: identity 37.582 -> placed 35.33".
- **Harness [F]:**
  - Codex-CLI clone with `exec_command`/`write_stdin`; `solution.py` + `predictions.json`; `arc_tools.py`.
  - Sampling: temperature 0.7, top_p 0.95, top_k 20 (K:42; AN2 §1.10); `REASONING_EFFORT = "medium"` (K:43); `TOKEN_CAP = 65_536` per attempt (K:44).
  - Per-request cap 12,288: "6.9 % fewer tokens, +1.1 to +2.2 under Kaggle-sized budgets" (AN2 §1.1).
  - Logit bias -1 on "Hmm"/"Alternatively": "cut generated tokens 4.5 %".
  - Images since v28.
  - A reminder at 32,768 tokens when no solution.py exists; a finalize window with thinking off in the last 8,192-12,288 tokens.
  - Carry with a 150-word recap; "secondary unverified first".
  - Alternates: "9 of 91 alternates were right where the answer was wrong".
  - Verification: "correct 83 times in 100"; "4.6 solves per million tokens after an unverified first, against 2.6 after a verified one".
- **Admission [F]:** adaptive, grows only at KV ≤ 60 %, trims at > 80 %, cuts on any retraction (AN2 §2.2).
- **Deadline [F]:** `KERNEL_BUDGET_SECONDS = 11*3600 + 55*60` (K:33); no drain; provisional answers are published continuously (AN2 §1.6).
- **Startup [F]:** compile-cache kernel; "server ready after 1,491.6 s" in the warmup (AN2 §6); "every run's first real prefill stalls ~115 s" (K:88).

### 1.4 mikelou1 "ARC AGI2" (MN): public LB 69.72
- **Engine [F]:** "Pennyroyal SGLang (0.5.19+gd00d88efc8d6)", TP4 + EP4, no DCP; "Pair-owned QSA" (each QSA layer stored once per TP pair, owner = layer j % 2) gives "about 2x KV capacity".
- **Concurrency [F]:** `--max-running-requests 41`; `INFLIGHT = min(MAXREQ, 41, KV_TOKENS // 44000)`; `--max-total-tokens 1850000`; HiCache ratio 2 on rung 0.
- **Speed [I]** (MN's inference from their fit `step_ms = 26.97 + 1.092 n + 0.416 KV/100k`): about 500-520 tok/s at n = 41, with **no spec decoding** (all rungs `SPEC: false`).
- **Model [F]:** W4A16 AutoRound; experts pruned to 384/512 by "counts.npz decode + 1.0*counts_code.npz decode + 0.5*counts_code.npz prefill"; FP8 rowwise dense (`densefp8`, "~3.2 GB/step of bf16 dense weights per GPU"); FP8 PLE.
- **KV/memory patches [F]:**
  - embed in host memory: "~98k more pool tokens (+7%)";
  - mamba slots 3 -> 2 per request: "~100k KV tokens";
  - `mambafinal`: no per-chunk mamba checkpoints, against "re-prefill cascades" ("19 mid-session misses, 286k re-prefill tokens");
  - `INPROJ_NODUP` "frees about 10.5 MB/GDN layer/GPU".
- **Decode patches [F]:**
  - QSA split-K: "13-20 rows used 13-20 of 58 SMs";
  - MQA grid ×4 smaller: "80-95% of CTAs still launch" while empty;
  - HC mix 32 rows;
  - top-32 logits exchange: "~1-1.6 ms per step at B 13-20, plus ~0.5 ms of full-vocab sampling";
  - fused FP8 dequant: "~17% of a 1024-token prefill chunk".
- **Prefix cache [F]:** "73% hits vs 96.8% on Kaggle v86"; `--schedule-policy lpm`.
- **Harness [F]:**
  - "direct" answer mode with `propose(j, grid)`: "Nothing is rejected for hard-coding";
  - effort xhigh; temperature 0.7 (1.0 in the fresh second session); top_p 0.95, top_k 20;
  - `--turn-max-tokens 32768`, `--session-cap 100000`, `--cap-stages 150000,200000`;
  - geometry kit plus object tables in the prompt; "Two kinds of tasks" prompt; images with rulers; a proposal echo with an image;
  - continue-cut with FORCE_NOTE_HYP; salvage turns;
  - a fresh second session at T=1.0 for tasks with ≥ 80k tokens or a cap, gated on spare capacity;
  - hidden budget ("hide-budget");
  - pin context mode above 110k.
- **Ops [F]:**
  - a rung ladder; a stall watchdog (10 min); up to 4 server restarts with `--load-state`;
  - LBPROBE (the LB outcome encodes the failure class);
  - PLE copied to local /tmp with 32 readers, because the mount read "at 7-32 MB/s";
  - solve deadline at T0 + 11 h 40 min.

---

## 2. Combined technique table

Effort to add to a vLLM 0.31 stack (Domdolus base), my estimate [I]:
- **S** = config or harness-only, under 1 day.
- **M** = a vLLM patch or a new harness module, 1-3 days.
- **L** = new kernels or training, more than 3 days.

"Domdolus?" = present in the Domdolus writeup or cells (yes / no / unknown).

### 2.1 Engine and model

| # | Technique | Who | Measured effect (as stated) | Domdolus? | Effort |
|---|---|---|---|---|---|
| E1 | MTP / spec decoding at high concurrency | notnitsuj (K3, AL 3.01), andy (NEXTN 3/4, AL 3.01) | notnitsuj's engine 1,087-1,120 tok/s with MTP; andy 402.5 tok/s at 10 running | **no** (`MTP_TOKENS = 0`, dd:cell43:10; "MTP off") | M (vLLM MTP exists; DCP + 4-bit KV + GDN state checkpointing interplay is the work) |
| E2 | Draft head fine-tuned for expected accepted length | notnitsuj; andy (refit to pruned target) | +6.1 % tok/s (2.75→3.01); andy AL 2.807→2.992 | no (no MTP) | L (training) |
| E3 | Int8 draft lm_head over a hot vocab (32,768 / 27,982 ids) | notnitsuj, andy | "+19% tokens/s" (notnitsuj); coverage 0.999581 (andy) | no | M |
| E4 | One BF16 GDN checkpoint for accepted tokens under spec decoding | notnitsuj | "allowed 49% more concurrent sequences" | n/a (no spec) | M-L |
| E5 | GPU-driven host-memory collectives inside CUDA graphs (all-reduce, all-gather, reduce-scatter) | notnitsuj; andy (host AR) | notnitsuj +8.6 % tok/s; andy "neutral" at 10 running | no: "NCCL instead of vLLM's custom all-reduce" | L |
| E6 | Int8 expert dispatch / reduce-scatter with group scales | notnitsuj | -1.9-4.0 % step (expert), -0.8-1.4 % (TP) | no | L |
| E7 | DP2 × TP2 engines with experts spread over all 4 GPUs | notnitsuj | part of the 1,100 tok/s stack (no isolated number) | no (one TP4/EP4 engine) | L |
| E8 | Fused MoE down GEMM + all-reduce (overlapped) | andy | -6.4 / -4.9 % decode step at 8/10 running | no (indexed MoE GEMM, not fused with AR) | L |
| E9 | HC up projection kept locally in INT8 (drops [T, hidden] all-gather) | andy | decode -4/-5 %, prefill -18 % | unknown (Domdolus has split-K HC GEMM and FP8 mixers, not this) | M |
| E10 | CUDA HC mix kernel (1-64 tokens) | andy; mikelou (HC mix 32 rows) | -4.6 / -5.1 % (andy) | unknown (Domdolus fuses HC down+SiLU) | M |
| E11 | CUDA QSA indexer decode kernel for verify graphs | andy | saves the 2.46 ms/step TileLang cost at 10 running | n/a without MTP | M |
| E12 | QSA split-K decode + MQA indexer grid shrink | mikelou | rationale: "13-20 of 58 SMs"; "80-95% of CTAs still launch" (no end-to-end %) | unknown | M |
| E13 | Prefill CUDA graph (breakable, one bucket) | andy | -6.5 to -6.8 % prefill time; cold prompts -10 % | partial: "sparse CUDA graphs ... cover the mixed steps" | M |
| E14 | Balanced expert placement from recorded routing | andy | max activated per GPU 37.58 → 35.33 (no end-to-end %) | no | M |
| E15 | Format-aware expert pruning (REAP + tool-call routing mass) | notnitsuj (320), andy (480), mikelou (384) | pruning without format mass: verified 0.34 vs 0.78; pruning alone cost 5-6.6 pts (notnitsuj) | no (all 512 experts, Q2_0) | L |
| E16 | RL on quantization scales + router (REINFORCE LOO) | notnitsuj | "+7.0 pass@2 points [+3.8, +10.6]", 12 % shorter sessions | no | L (H200 training) |
| E17 | Server-side sessions (append tool output only) | notnitsuj | prompt tokens per generated token 6.1 → 0.2 | partial: prefix hits every 800, exact resume, token-id reuse (no stated ratio) | M |
| E18 | Pair-owned QSA KV (layer-level de-dup per TP pair) | mikelou | "about 2x KV capacity" | equivalent: DCP | - |
| E19 | Mamba slots per request 3 → 2; no per-chunk mamba checkpoints | mikelou | "~100k KV tokens"; avoids "286k re-prefill tokens" | partial: "newest DeltaNet state only", "in RAM" | S-M |
| E20 | GPTQ INT8 dense layers | andy | "frees 0.23 GiB per L4 at the same speed" | yes-ish (FP8 projections) | - |
| E21 | Host embedding table | andy, mikelou | +7 % pool tokens (mikelou) | **yes** | - |
| E22 | HiCache host KV tier (2×) | mikelou | not measured | partial: "RAM offload" | - |
| E23 | Top-k logits exchange (32) | andy, mikelou | -3.6 % step (andy) | **yes** (64 best tokens) | - |
| E24 | `NCCL_GRAPH_MIXING_SUPPORT=0` | andy | "only adds 17-60 us per captured collective" | unknown | S |
| E25 | Compile-cache kernel + SHA-pinned restore | andy, mikelou (JIT export) | 0 compiles at start; first prefill still "~115 s" | **yes** (prebuilt caches) | - |
| E26 | Copy PLE / weights off the mount with parallel readers | mikelou (32 readers), andy (16 range readers) | mount "7-32 MB/s uncached" | **yes** (/dev/shm once, parallel reads) | - |
| E27 | Int8 HC / 2:4-bit mix (4-bit hot experts, 3-bit cold) | notnitsuj | 8.7 GB experts per GPU; ~6 REPL pts vs FP8 | no (uniform Q2_0) | L |

### 2.2 Harness

| # | Technique | Who | Measured effect (as stated) | Domdolus? | Effort |
|---|---|---|---|---|---|
| H1 | Verify-then-vote over ~3 sessions per task (fewest-sessions-first) | notnitsuj | 1,553 verified sessions "right 75% of the time"; core of LB 75.69 | **no** (2 tries; no voting) | M |
| H2 | 11 content invariants in the verifier (palette subset, output ≠ input, colour monotonicity, ...) | notnitsuj | precision 0.82 vs 0.72 with `ak` (incl. invariants) | **no** (train-pass only) | M |
| H3 | Tie-break: symmetry kept on test + no test-only code | notnitsuj | +0.54 [+0.28, +0.84] | no | S-M |
| H4 | attempt_2 = most common other grid any program produced | notnitsuj | (part of selection; perfect pick only +0.8) | partial: "a function ... that produces a different answer" | S |
| H5 | Deprioritise tasks with ≥ 2 votes and a lead | notnitsuj | +0.8 | no | S |
| H6 | Warm start of a new session from the best unverified program + per-example results | notnitsuj; andy (carry + 150-word recap + diffs) | +0.21 right answers/session, -17 % tokens (notnitsuj); 4.6 vs 2.6 solves/M tokens (andy) | **no in final**: Domdolus' "informed try 2" (#9) was part of a 59.72 row and dropped (`INFORMED_TRY2 = False`, dd:cell46:120) | M |
| H7 | Unverified-first ordering of second attempts | andy | 4.6 vs 2.6 solves/M tokens | no (try 2 "queued from the start") | S |
| H8 | Per-request output cap (12,288) with a "write solution.py now" nudge | andy | "6.9 % fewer tokens, +1.1 to +2.2" | **no**: "max_tokens is not sent" (dd:cell30:219) | S |
| H9 | Logit bias -1 on "Hmm"/"Alternatively" | andy | "cut generated tokens 4.5 %" | no (sampler supports logit_bias, dd:cell22:36; no such bias set) | S |
| H10 | Reminder at 32k used tokens with no solution file; thinking-off finalize window at the end | andy | not isolated | partial (Domdolus asks for a function after no-answer ends) | S |
| H11 | Alternates inside one attempt (`predictions_alt.json`, finalize-window passing programs) | andy; mikelou (`propose_alt`) | "9 of 91 alternates were right where the answer was wrong" | partial (diversity rule on identical answers) | S |
| H12 | Helper library `ak` / `arc_tools` / geometry kit + object tables in the prompt | notnitsuj, andy, mikelou | `ak`: +2.6 [+0.2, +5.2] | partial: "bounding boxes and a check function" | M |
| H13 | Import alias fix + traceback line fix | notnitsuj | +2.25 and +0.88 points | unknown (specific to their sandbox) | S |
| H14 | Drain near the deadline | notnitsuj | "+1–3% tokens" | no ("One minute before this limit, the tries are stopped") | S |
| H15 | Admission by projected context fit, with a KV reserve | notnitsuj (48 cap, ~14 % reserve); andy (≤ 60 % KV growth) | KV-aware routing of the reserve "lost 2.2% tokens per hour" | partial ("collapse fix" holds calls) | S-M |
| H16 | Continuous provisional answers + atomic submission rewrite | andy, mikelou | robustness | **yes** ("rewritten every time a try finishes") | - |
| H17 | Fresh second session at T=1.0 only for capped/≥ 80k tasks, gated on spare capacity | mikelou | not isolated | no | S |
| H18 | Salvage turns (≤ 2 × 6,144) when a session ends without a proposal | mikelou | not isolated | partial (best function used on budget-out) | S |
| H19 | Stall watchdog + server restart + `--load-state` resume | mikelou | robustness | unknown ("MAX_REVIVALS = 3", dd:cell46:15, [I] likely similar) | M |
| H20 | LBPROBE: failure class encoded in the LB result | mikelou | diagnostic | no | S |
| H21 | Temperature 0.7 + medium effort | andy | low and xhigh "tried and retired" | no (T 1.0, xhigh) | S, **contradicted** by notnitsuj "1.0 for Flash-Next" |

---

## 3. Harness ideas ranked by evidence

Ranking: the strength of the measurement first (paired A/B with a CI > a stated delta > a stated rationale), then the size of the effect.

1. **Verify-then-vote over several sessions, with fewest-sessions-first scheduling** (notnitsuj).
   - Evidence: the 2,177-session analysis; "Verified programs ... right 75% of the time".
   - Note: "Selection is nearly saturated. A perfect pick among the produced candidates would add only +0.8, and a learned scorer added +0.74". So voting's value is mainly in generating more independent verified programs. Fancy picking adds little.
   - [I] Domdolus' 2 tries at 200k is the opposite trade (fewer, longer sessions). Notnitsuj's cap 98k plus ~3 sessions and Domdolus' 2 × 200k spend similar tokens per task. Neither team published a head-to-head.
2. **Stronger verifier: content invariants plus the `ak` helper** (notnitsuj).
   - Evidence: a pre-registered A/B, "+2.6 [+0.2, +5.2]"; precision 0.82 vs 0.72.
   - Verified-but-wrong is the largest loss bucket: "38%: only wrong programs verified".
3. **Sandbox correctness fixes** (notnitsuj): import alias "+2.25 points ... [+0.60, +3.85] over about 4,400 sessions per arm"; traceback fix +0.88. Paired A/B with a CI. [I] The lesson to transfer: audit the Domdolus REPL for the same two failure modes.
4. **Tie-break on symmetry and test-only code** (notnitsuj): "+0.54 [+0.28, +0.84]". It is small but has a CI.
5. **Per-request output cap of 12,288 with a nudge** (andy): "6.9 % fewer tokens, +1.1 to +2.2 under Kaggle-sized budgets".
   - Domdolus sends no max_tokens.
   - [I] It conflicts with our memory rule "no token cuts below 60k". It caps one reply, not the total budget. Raise it for approval before any use.
6. **Warm start / carry after an unverified session.**
   - notnitsuj: "+0.21 right answers per session, with 17% fewer tokens".
   - andy: 4.6 vs 2.6 solves per million tokens.
   - Counter-evidence: Domdolus' row 9 ("informed try 2" together with RAM offload off and token-id reuse) scored 59.72, and the final keeps try 2 plain. That row bundled 3 changes, so it is not a clean negative.
7. **Deprioritise tasks with ≥ 2 votes and a lead** (notnitsuj): +0.8.
8. **Drain near the deadline** (notnitsuj): "+1–3% tokens". Domdolus stops tries 1 minute before the limit. [I] At 70 concurrent sessions × a ~63k average context, a hard stop discards a lot of in-flight work.
9. **Anti-hedging logit bias** (andy): "cut generated tokens 4.5 %". Measured in tokens, not points.
10. **Unverified-first second attempts** (andy): solves per million tokens 4.6 vs 2.6.
11. **Alternates from one session** (andy 9/91; mikelou `propose_alt`): a diversity source for attempt_2.
12. **Geometry kit, object tables, "Two kinds of tasks" prompt, salvage, fresh second session gated on capacity** (mikelou): no isolated measurements in the code. mikelou's LB of 69.72 sits below Domdolus.
13. **Temperature / effort.** andy uses 0.7 / medium, with low and xhigh "tried and retired". notnitsuj: "1.0 for Flash-Next". Domdolus: 1.0 / xhigh. The evidence conflicts; no recommendation.

**Negatives other teams reported** (useful to avoid):
- notnitsuj: "Review turns, requests for alternative programs and generalization prompts were all negative in within-sample tests. Aborting slow sessions early cost 3–18 points."
- notnitsuj: "distillation toward the full-precision model hurt both models"; "Lower KL is not more solved tasks".
- notnitsuj: a cap raise "worth +1.9 to +3.1 on the FP8 model gave only +0.9 on the shipped one".
- notnitsuj: "Flash-Next answering directly" 45.9 vs REPL 70.9 at 100k tokens per input.
- notnitsuj: KV-aware routing using the reserve "lost 2.2% tokens per hour".
- andy: retired "filler2, phase caps, the 8k cap, the 'act' carry note, low/xhigh effort, dropped reasoning".
- andy: prefill-graph full ladder "raised teacher-forced NLL ... (+0.017 nats over 35 tasks)"; 1024 chunk rejected for memory.

---

## 4. Domdolus-team ablations and negatives (from DW)

**Submission history and what each step implies [F, row text quoted; deltas are my arithmetic]:**
- #1 → #2: "3 rounds: ... as is, then transposed, then mirrored": 51.67 → **65.14** (+13.5).
- #2 → #3: "2 tries, grids given as text": **58.75** (-6.4).
- #3 → #4: "2 tries, grids as images only": 66.81 / 65.42 (+8.1 vs #3).
  - [I] Images-only beats text by ≈ 7-8 points in this pair.
- #5: "Smaller KV pages, newest DeltaNet state only, sampling on 1 card, embeddings in RAM; 60 tasks at once": 63.33 (-3.5 vs #4).
- #6: "4-bit KV cache, DeltaNet state in RAM, FP8 index keys; 100 tasks at once": 66.25.
- #7: "+ exact resume; 100 tasks at once": 62.50 (-3.8).
- #8: "Harness fixes; 80 tasks at once": 71.53 (+9.0). The fixes: tool outputs cut at whole lines with a note, bbox background = most common colour, conversation kept after a rejected final answer.
- #9: "RAM offload off, token-id reuse, informed try 2; 80 tasks at once": **59.72** (-11.8). The negative bundle.
- #10: "3 tries, RAM offload back on": 71.67.
- #11: "+ fast server startup": 72.92 / 73.47.
- #12: "48-64k short-term memory window, prompt fixes, 3 plain tries, 100k tokens; 60 tasks at once": **78.47** (+5).
- #13: "2 tries, 150k tokens": **82.08** / 80.00.
- #14: "+ faster kernels and communication (no collapse fix)": 77.64 (-4.4).
- #15: "70 tasks at once, collapse fix, 200k tokens, 1M window": 79.44.
- #16: "+ sparse CUDA graphs, kernel trims, no token numbers in the prompt": **83.47** / 72.36 / 74.58 / 79.19 / 78.06.

**Stated negatives and dropped ideas [F]:**
- "in the final days of the competition, despite faster kernels, our score went down, which might have been caused by the server issues."
- NVARC port to vLLM: "it did not translate well to the leaderboard."
- "ARC is a vision problem" small-network pipeline: "it failed miserably ... it seemed to plateau quickly".
- DeltaNet-state test-time tuning (PICASO-style averaged states): "this did not affect the intelligence at all in our experiment."
- 4-bit model with 8-bit KV in RAM: "we did not manage to get the pipeline working in time, even though the 4-bit model seemed smarter and reached the same accuracy with fewer tokens."
- Other models: "Qwen 3.8 27B ..., Qwen 3.6 35B-A3B ..., and a Swift variant of Qwen 3.8 Flash-Next ... all of them fell short under the same settings"; DeepSeek V4.1 Flash: "couldn't get it to run at a decent speed on the L4s".
- Rounds (transposed/mirrored) were dropped for "plain tries" in #12.
- The informed try 2 is off in the final (`INFORMED_TRY2 = False`, "no more 'evolve style'", dd:cell46:120); the cells note an RTX test with "no measurable effect" for INFORMED_NEW (dd:cell46:119).
- `THIRD_TRY = False` in the final cell (dd:cell46:16).

**Stated positives without an A/B [F]:**
- 4-bit KV: "We managed to increase capacity without hurting accuracy at all."
- Short-term memory: "increasing the length of the window and how many tokens we save helps the model retain its recent reasoning."

**Noise warning [I]:**
- The same #16 notebook ranges 72.36-83.47, and #13 ranges 80.00-82.08.
- Most single-row deltas above (≤ 5 points) are within that spread, so only the large moves are reliable: images vs text, harness fixes, the #9 bundle, the window plus prompt fixes.
- DW itself: "we submitted some notebooks several times to get a luckier submission".

---

## 5. What looks addable on top of Domdolus, by evidence and size [I]

1. **MTP K3 with a fine-tuned draft and an int8 hot-vocab head** (E1-E3): notnitsuj +6.1 % and +19 % tok/s. Domdolus runs MTP off.
   - Risk: GDN state under spec decoding costs memory, and notnitsuj needed the one-BF16-checkpoint trick (+49 % sequences).
   - The Domdolus cells show an OOM fallback "retrying without MTP" (dd:cell43:269), which suggests memory was the reason MTP was off.
2. **Verifier invariants, tie-break, and voting / more sessions** (H1-H5): the best-evidenced harness gains, absent in Domdolus.
3. **Deadline drain** (H14): +1-3 % tokens, a small change.
4. **Per-request cap + nudge and anti-hedging bias** (H8, H9): token-efficiency gains measured on Flash-Next. H8 needs user approval under the 60k rule.
5. **Fused MoE-down + AR, HC local INT8 up, CUDA HC mix** (E8-E10): andy measured each at -4 to -6 % step on 4×L4, on SGLang. Porting to vLLM 0.31 is kernel work.
6. **Host collectives inside graphs** (E5): +8.6 % at notnitsuj, but neutral at andy's 10 running, and Domdolus found NCCL better than custom AR. The evidence is mixed.
7. **RL on quant scales / format-aware pruning** (E15, E16): the biggest measured model gain (+7.0 pass@2), but it is a training project. notnitsuj's model and code are "public soon"; see our memory item "notnitsuj 320-expert 2-bit plan".
