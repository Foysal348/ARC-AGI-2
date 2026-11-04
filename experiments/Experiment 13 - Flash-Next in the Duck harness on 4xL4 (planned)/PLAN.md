# >>> rev 5 (2026-10-20, user): SIMPLE PATH. Supersedes the Stage-1 probe below (FN-PROBE-2 notebook kept, not used).
# E13 = the E12 Duck SMOKE1 notebook unchanged except CELL 5 (engine) and the engine inputs/limits in CELL 1.
# Engine = Franzen launch + D1-D7 + memory items (NEXTN off, chunk 2048, mem 0.93, 4 running); FA2 for QSA decode;
# patch #3; the user-mandated top-k test (recorded only); a host-RAM guard; an '!!!!' smoke; Duck streams = min(4, KV // 60k).
# Builder: Experiment Notebooks/Build/build_e13_duck.py. Audit request: Experiment Review/E13 simple SMOKE1 - audit request (2026-10-20).md

# Experiment 13: Qwen3.8-Flash-Next in the same Duck harness on 4xL4 (plan rev 4, 2026-10-19)

## Goal and principle (user, 2026-10-19)

- Try Flash-Next once in the Duck architecture.
- **Minimal change:** take everything from Franzen's notebook (`dfranzen/arc-agi-3-milestone-2-solution`) and change only what 4xL4 makes unavoidable.
- **Reuse our proven work** instead of redoing it, so we do not loop.
- **No repeat of the E12 EVAL30 waste:** every GPU run gets a written pre-run capacity check (section 3).
- Order: E13 first, then E12 rev 4 (27B Duck fixes: per-branch cap, resident contexts ≤ KV, drain), which is not forgotten.

Constraints:
- Licence: qwen-community-1.0. Kaggle staff declined to interpret model licences (2026-10-16, topic 745079). LB use is the user's risk decision.
- Every GPU step needs the user's go. No push without approval.

Evidence:
- `VALIDATION.md` (23 claims);
- `Experiment Review/E13 rev 3 - three GPT audits validated (2026-10-19).md` (F1-F13, N1-N8).

## What changed from rev 3

1. **QSA decode = classic FA2 (F1).**
   - Upstream sm89 evidence: L20 parity 16/16 against FP32; 4090 TP4/EP4 full serving. The `fogandfrog` 2.8.3.post1 wheel for torch 2.13 / cu13 / cp312 (sha `74984f62…`).
   - Zero SGLang source change: 0.5.21's resolver picks FA2 when it is importable.
   - Our P1 (Triton packed decode, ROCm-validated only) is the fallback, decided by the micro-preflight.
2. **Patch #3 = Franzen's two substitutions with his own BEFORE/AFTER SHAs (F3).** The 0.5.21 file sha equals his BEFORE_SHA; only the Pennyroyal git-pin assert is dropped.
3. **No file-backend PLE fallback (F4).** It needs host page tables, which L4 lacks.
4. **A micro-preflight in the same GPU session, before the weights load (no extra launch):**
   - decode kernels;
   - top-k;
   - the indexer;
   - pinned PLE allocation.
5. **B2 changes exactly one variable,** chosen by the failure signature; no NEXTN re-enable inside the budget (F7).
6. **Two gates** (serving and competition); 700 tok/s is a target, not a gate (F8).
7. **Known differences recorded:** patches #2/#4 absent; D7 semantics unverified (F11).

## 1. Reuse inventory (proven work, taken as is)

Status:
- **K** = worked on Kaggle;
- **L** = local harness;
- **X** = do not reuse.

| # | Piece | Where | Status |
|---|---|---|---|
| R1 | SGLang 0.5.21 wheelhouse `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse` + `install_wheelhouse.py` + flashinfer-jit rename + libcuda shim + venv | E11 Build, E12 CELL 5 | **K** |
| R2 | E11 bring-up (diagnosed relaunch, caps from the log, fail-closed capacity gate) | E12 CELL 5 | **K** |
| R3 | Flag-existence preflight | E3 builder stage 3 | **K** |
| R4 | Pin probe | E3 stage 3_pinned_memory | **K** (1 GiB only; F5 extends it) |
| R5 | `--weight-loader-prefetch-checkpoints` (345 s / 124.8 GB on 4xL4) | E3 | **K** |
| R6 | Launcher safety (process group + killpg, port gate, /health only while the launcher lives, progress wait) | E3 v4.2.1, `fnh/fn_harness_v42.py` | **K + L** |
| R7 | Stage report JSON, input check, symlinked model dir | E3 stages 0/1 | **K** |
| R8 | Launch rules: MoE runner `auto`; EP4 (N1); no `--language-model-only`; page 64; mamba slots | E3 crashes | **K** |
| R9 | Duck package + `arc2_duck.py` + sandbox / memory self-tests | E12 CELL 3 | **K** |
| R10 | E12 solver, report, dual-mode CELL 0, request and thinking logs | E12 CELL 0/6/7 | **K** |
| R11 | Duck milestone-2 env mapped to ARC-2 | E12 settings doc | **K** |
| R12 | Fake-server notebook harness | `e12_harness/harness_e12_nb.py` | **L** |
| R13 | Push script with account/sha checks, `--hamid` | `push_e12/push_e12_duck.py` | **K** |
| R14 | Franzen's Intel W4A16 (BF16 PLE) + albucino NEXTN drafter + FR-Spec `hot_tokens_64k.pt` | Kaggle (public) | **K** (Intel attached in E3) |
| R15 | Franzen patch #3 + his SHAs | `dafr/serving/build_bundle_pennyroyal.sh:250-290` | **K** |
| R16 | **New:** FA2 wheel (fogandfrog 2.8.3.post1, sha `74984f62…`) | to download and upload as a private dataset (user approval) | upstream evidence only |
| X1-X4 | our PyTorch shim; E3 "minimal first boot"; FP8 PLE overlay (last resort only, with approval); Pennyroyal on L4 | E3 | **X** |

## 2. The complete delta from Franzen (everything else verbatim)

| # | Franzen | E13 | Why |
|---|---|---|---|
| D1 | Pennyroyal 0.5.19 | SGLang 0.5.21 (R1) | Pennyroyal has no sm89 decode path and fp8 decode scratch |
| D2 | patch #3 in his wheel | his patch #3 + his SHAs on 0.5.21 (no git pin) | bug still in 0.5.21 |
| D3 | TRT-LLM QSA decode (SM100/SM120) | **classic FA2 installed (R16); no source change.** Fallback P1, only if the micro-preflight says so. | FA4 crashes on sm89 |
| D4 | TP1 | TP4 + EP4 | 73.6 GB weights vs 24 GB per GPU; TP4 Marlin N=320 is invalid (N1) |
| D5 | FlashInfer GDN / mamba | Triton | SM90/SM100 only |
| D6 | arch 12.0 + SM120 env | 8.9; SM120 env removed | different GPU |
| D7 | `--gdn-mtp-cache-mode none` | dropped (does not exist in 0.5.21) | semantics unverified, measured |
| — | model, BF16 PLE, NEXTN 3 + FR-Spec (Launch 2 only if the memory allows, 3b), ctx 139,264, 10 running (capped by the measured mamba/KV limits), chunk 8192 (Launch 1 uses 2048, 3b), prefill 16384, **fp8 KV**, page 64, extra_buffer + track 64, mamba cache 60, CUDA graphs ≤ 10, lpm, preserve_thinking, parsers, prefetch, drop cache, mem 0.96, `fast_topk` (F6), TileLang | **verbatim** | — |

Not taken:
- patch #1 (SM120 GEMM);
- patch #2 (spec-state budget): measured;
- patch #4 (prefix retention): a later option if the cache % is low;
- patch #5 (prefetch lookahead).

## 3. Pre-run capacity check (mandatory; numbers from our own runs)

| Item | Estimate | Source | Fail rule |
|---|---|---|---|
| Quota rate | L4x4 burns about **2.0x wall** (EVAL30: 4.15 h wall → 8.22 h quota) | hosen42 quota before/after | — |
| Session wall time | install about 3 min + micro-preflight about 5 + pinned test about 2 + weight load about 8.5 (185 GB at 0.36 GB/s) + JIT/graphs about 10 + Gate C about 10 + speed about 20 → **about 60-70 min**; with B2 at most about 2 h | E11 install 181 s; E3 load 345 s/124.8 GB; E3 warm-up 7.4 min | notebook hard stop at 2.5 h |
| Quota | about 2.5-4 h of hosen42's 40.25 h | — | — |
| Host RAM | pinned BF16 PLE about 95 GB (up to about 128 GB with rounding) + 4 rank RSS + loader; of 202.6 GB | E3 stage 0; F5 | the pinned test fails, or available < need + 20 GB → stop |
| GPU memory | weights 19.06 GB/rank (E3) + NEXTN drafter + graphs; keep ≥ 1.5 GiB free | E3 | free < 1.5 GiB → report, no tuning loop |
| KV | estimate about 170-250k fp8 tokens (6 KiB/token/rank); **measured at launch** | VALIDATION #20 | recorded; it sets the Duck concurrency rule below |
| Duck concurrency (for Stage 2/3, not Stage 1) | E12 EVAL30 prompts p50 47.9k / p90 97.7k → resident contexts must be ≤ KV / about 60k; E12 thrash = 8.4 M re-prefill tokens | E12 EVAL30 analysis | Stage 2/3 notebooks must pass this check before they are proposed |

### 3b. Per-GPU memory budget (rev 4.1, 2026-10-19; GPT capacity review applied)

Labels: **[M]** = measured (source given); **[E]** = estimate / hypothesis. Units are stated.

| Item (per L4 rank) | Value | Label / source |
|---|---|---|
| Free before the weight load | 21.62 GB | [M] E3 v4.2.1 serve.log |
| Target weights (Intel W4A16, EP4) | 19.06 GB | [M] E3 |
| Free **after the weight load** | **2.55 GB** | [M] E3 (post-weight stage) |
| Free **after the memory pool + warm-up** (bf16 KV 118,080 tokens = 1.36 GB, 10 mamba slots = 0.16 GB, graphs **off**, NEXTN **off**) | 0.97 GB after the pool, **0.66 GiB** after warm-up | [M] E3 (a different stage from the line above) |
| NEXTN drafter payload | 4,139,535,872 B = 3.855 GiB total → about 0.96 GiB/rank if sharded perfectly | [M] payload (HF file sizes / albucino README); [E] per-rank VRAM (replication and buffers not known) |
| Mamba pool | about 0.016 GB per slot (E3: 10 slots = 0.16 GB). The pool is reserved by `--max-mamba-cache-size`, **not** by the number of active streams (60 slots ≈ 0.96 GB) | [M] per slot; [E] linear scaling |
| CUDA graphs (bs ≤ 10) | about 0.3-0.6 GB | [E] |
| Prefill activation peak | the cookbook measured 1.5-2.6 GB above the post-graph level at chunk 4096, and an OOM with 2.4 GB free, on an RTX PRO 6000 (TP1) | [M] on another GPU; **not an L4 minimum** |
| KV | 6,144 B/token/rank fp8 (12 layers x 1 KV head x 256 x 2): 48k tokens = 0.295 GB, 98k = 0.602 GB, 170k = 1.044 GB | [M] arithmetic; pool size [E] |
| Duck request size on ARC-2 | E12 EVAL30: n = 454 requests in 60 branch files; prompt p50 47,880, p75 74,934, p90 97,712, max 117,312 | [M] `Experiment 12/Experiment Results/Analysis scripts/eval30_request_stats.py` + `_output.json` |

What the numbers allow, as scenarios [E], not verdicts:
- **NEXTN + 10 streams + graphs + a useful KV pool:** very tight. An illustrative budget (2.55 − 0.96 − 0.80 − 0.40 − 0.10) leaves about 0.2 GB for KV + prefill.
- **Target-only (NEXTN off), small mamba pool, fp8 KV:** a KV pool of roughly 80-170k tokens. That is about 1-3 resident Duck contexts at the measured p50-p90, before any prefix sharing.
- **Throughput:** unknown. 420 tok/s (27B W1 decode at 11-12 running) is a **comparison threshold**, not a ceiling. To reach it, the per-stream rate would need to be 420 / 210 / 140 tok/s at 1 / 2 / 3 streams.
- **Chunk:** 8192 has the highest OOM risk and 2048 survived in E3. This is a measured choice, not a fixed requirement.

**Launch sequence (rev 4.1, memory-aware; still at most 2 full-model launches):**

| Step | What | Go-on condition |
|---|---|---|
| Micro-preflight (same session, no weights) | decode boundary tests incl. 129, masks/offsets, graph replay; top-k gates (below); TileLang vs torch; pinned PLE test | correctness PASS |
| **Launch 1, target-only** | NEXTN **off**; `--max-mamba-cache-size` = (planned running) x 5 slots; chunk **2048** (E3-proven); fp8 KV; graphs on; everything else Franzen-verbatim | Gate C PASS + memory gate |
| Memory gate | per-rank free VRAM after load / after pool / at the measured prefill peak; effective running (from the startup log); KV tokens; slots | quantified headroom |
| **Launch 2** (pre-declared, one of) | (a) Gate C failed → B2 diagnostic, one variable by signature (section 5); (b) Gate C passed and the measured free memory ≥ drafter + 1.5 GB → NEXTN on; (c) passed, but NEXTN does not fit → chunk 4096 if the measured peak allows, else no second launch | as section 6 |

Early stops: any Gate C failure in Launch 2 → stop. Each launch has its own hard stop (Launch 1 75 min, Launch 2 60 min). The notebook's total hard stop is 2.5 h.

This deviates from Franzen in one respect (NEXTN off in Launch 1), and only to avoid a likely-OOM launch. Launch 1 alone does not qualify NEXTN.

## 3c. Audit items re-checked on 2026-10-19 (were missing from rev 4 and are now included in Stage 1)

- **QSA decode tests** (A): lengths 1/63/64/65/129/2048; FP8 storage vs dequantized reference; eager vs graph replay; -1 masking and packed-offset asserts; diagnostic tolerance atol = rtol = 0.01, with greedy-token agreement as the stricter gate.
- **Top-k test, mandatory (user + B):**
  - bundled `fast_topk<512>` vs `torch.topk` on the same logits;
  - in-range / unique checks; selected-score multiset and cutoff check (ties allowed);
  - uniform, Gaussian, concentrated, near-tie and long-context distributions;
  - compressed lengths 2k/8k/16k/35k (= 8k-139k tokens);
  - 20 repeats for stability; graph capture + replay with changed logits;
  - failing inputs saved. A score-generation (TileLang) fix would not fix top-k: they are separate components.
  - Gate T1 (original): compressed lengths 2k / 8k / 16k / 24.5k / 30,938 (the #36807 repro length) / 34,816. Inputs: normal, concentrated, tie and near-tie. Also -1 padding and short rows (< 512). 20 repeats each.
  - Oracle: tie-aware (selected-value multiset, threshold validity, unique and in-range). Exact index equality is not required.
  - Gate T2 (replacement candidate, measured only): FlashInfer `top_k_ragged_transform(deterministic=True, tie_break=NONE, dsa_graph_safe=False)`, if present in the wheelhouse flashinfer 0.6.18. #36807 reports 29/29 on sm89 for 0.6.17, and says tie_break / dsa_graph_safe give `cudaErrorNotSupported` on sm89.
  - T2 checks: score-multiset parity, relative indices, nonzero row starts, changed-input graph replay, and its speed cost. A switch to T2 needs the user's approval.
  - Saved fixtures carry seed, dtype, shape, row starts/ends, kernel revision and graph mode. Every later change is regression-tested on them.
- **Marlin (A):**
  - print the effective expert partition shapes before launch;
  - classify the first failure as `Invalid thread config` / scale dtype / NaN;
  - require zero Marlin errors through graph capture.
- **PLE / host (A, B):**
  - log MemAvailable, the cgroup memory limit, the pinned capability, the projected PLE footprint (BF16 PLE = 107.6 GB in total, HF sizes), per-rank and peak RSS;
  - an explicit stop reason if PLE init fails;
  - RSS / pinned / GPU memory stable over 10-15 min of **Duck-like** traffic (repeated prefill + continuation), not an idle server.
  - Also log cgroup `memory.max` / `memory.current` and major page faults.
  - Host need: BF16 PLE is **107.6 GB in total** (HF sizes), not 95 GB. Total RAM is not freely allocatable RAM.
- **Speculative-state memory (B):** intermediate SSM allocated vs reserved, mamba slots, `max_total_num_tokens`, drafter KV. Read the effective running limit from the startup log, not `/get_server_info` (cookbook).
- **Static asserts before launch (A):**
  - version 0.5.21, arch 8.9, TP4 / EP4, page 64;
  - FA2 importable (`flash-attn` version, `torch.compiled_with_cxx11_abi()`), patch #3 after-SHA;
  - no `--gdn-mtp-cache-mode` / `--language-model-only`; `--moe-runner-backend auto` present; `desc_act` absent.
- **Gate C additions (A, B):**
  - `reasoning_effort` default (xhigh) on the tool-call test;
  - impossible token-id check;
  - the fallback `submission.json` stays schema-valid after a server failure.
- **B2 row (B):** an unknown cause with every micro-test passing → B2-D (TileLang off + NEXTN off) as a **diagnostic only**, never a candidate.
- **B2 row (A):** corruption with the micro-preflight showing an eager-vs-graph mismatch → `--disable-cuda-graph`. A graph-off pass is confirmed only with long-context probes (short, 32k, 64k, the feasible maximum); graph-off is not a full fix (#37052).
- **Competition-gate metrics (B, C, capacity review):** **correct test grids per GPU-hour** (primary), coverage, useful-token ratio, plus:
  - output tokens per GPU-hour, useful tool calls per hour, completed branches per hour, train-passing programs, exact hits, preemptions;
  - E12 vs E13 compared with sampling, prompt, task order and budget **frozen**.
- **Builder (A):** atomic temp-file replace for patches.
- **Failure classes (capacity review):** load/shape incompatibility, kernel compile, CUDA OOM, NaN/Inf logits, valid-but-wrong tokens, graph-only divergence. Each is recorded separately; never one "garbage" bucket.
- **Cookbook facts (C):**
  - Flash-Next always thinks; E3's "thinking off" sanity prompt was not a valid mode;
  - generation_config sampling is the default;
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 4. Stage 1 session (one notebook, one GPU session)

**1a. Install and preflight (CPU):**
- R1, plus the FA2 wheel and its import / ABI check (torch 2.13, cxx11abi, cp312).
- D2 patch with SHA asserts.
- Flag existence (R3).
- Model files and config: `desc_act` absent or false; indexer 2048/4.

**1b. Micro-preflight on the GPUs, before the weights. Small tensors, no server; record everything.**
- QSA decode: FA2 vs P1-Triton vs FP32 reference. Cases:
  - BF16 and FP8 storage;
  - 6 Q / 1 KV heads, head dim 256;
  - batch 1/4/10;
  - lengths 1/63/64/65/2048;
  - padding rows;
  - eager vs CUDA-graph replay.
- `fast_topk<512>` vs `torch.topk`: drop rate on normal, concentrated and near-tie logits, compressed lengths 8k-35k.
- TileLang MQA vs torch MQA: max error and memory.
- Pinned allocation: the real per-rank PLE size, 4 processes at once; then free.
- **Pre-declared choice:**
  - decode = FA2 if it passes; else P1 if it passes; else stop.
  - The indexer stays TileLang unless it mismatches the torch reference.
  - `fast_topk` stays unless drops are frequent on realistic logits; then report only.

**1c. Launch 1 (target-only, memory-aware; see 3b).** 0.5.21 + D1-D7 + the 1b choices. Franzen settings verbatim except the pre-declared memory items: NEXTN off; chunk 2048; `--max-mamba-cache-size` = planned running x 5. Launch 2 follows the 3b table: the B2 diagnostic (below), NEXTN on, or chunk 4096.

**Gate C (serving correctness):**
- prefill logprobs of a known sequence;
- greedy counting, repeated twice (determinism);
- a grid echo;
- exact marker retrieval at 8k / 32k / 58k context (as far as KV allows);
- a structured `python` tool call with thinking, plus a 2-turn preserve_thinking replay;
- concurrent 1/4/8/10 requests correct;
- finite logits, valid token ids, no invalid probabilities;
- NEXTN acceptance logged.

**1d. Speed stage (only after Gate C passes):**
- decode tok/s at 1/4/8/10 streams with Duck-sized contexts;
- KV tokens, real running count, queue, preemptions, cache %, free memory.

## 5. B2: one variable, by the failure signature (at most 2 launches)

| Signature in B | B2 change | Isolates |
|---|---|---|
| Divergence or invalid probabilities only with drafts; target path fine | NEXTN off | speculation |
| Indexer mismatch seen in 1b / layer evidence | TileLang off (torch indexer) | indexer |
| Decode mismatch at full-model level although 1b passed | the other decode path (FA2 ↔ P1) | QSA decode |
| Weight / Marlin / PLE load failure, or OOM | **no B2**: stop with diagnostics | — |

A second failure means stop. A B2 pass qualifies only the B2 configuration.

## 6. Gates

- **Serving gate (Stage 1):**
  - Gate C passes;
  - the measured real concurrency is stable;
  - KV and memory are recorded;
  - no silent corruption.
  - Target: at least 700 tok/s aggregate. Not a gate.
- **Competition gate (Stage 3):** the same 30 tasks / 48 inputs as E12 EVAL30 (7/48) and R5 (16/48), compared on score x coverage with the same budget rules and the Duck concurrency rule.

## 7. Stages

- **Stage 0 (offline, no GPU):**
  - download and pin the FA2 wheel;
  - build the Stage 1 notebook from R1-R8, R15, R16 + 1a-1d + Gate C;
  - local harness R6/R12 (plumbing only);
  - the capacity check of section 3 written into the run request;
  - audit zip.
- **Stage 1:** as above (after the user's go; the FA2 dataset upload also needs a go).
- **Stage 2:** Duck SMOKE1, after the Duck concurrency check.
- **Stage 3:** Duck EVAL30, after the Duck concurrency check.
- **Stage 4:** LB only after Stage 3 wins, the licence decision and the user's go.
