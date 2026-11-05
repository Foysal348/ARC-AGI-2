# Experiment 35 framework: what we take, what we add, what we measure

Built 2026-11-04 from the five reports in `Experiment Review/Agent Reports/` (Domdolus cells 00-20, cell10, 21-46, cell30 harness; other public solutions). Every item cites its report. Nothing here has run yet.

## 0. Reference facts

- Domdolus notebook #16: 5 submissions scored 72.36-83.47, mean about 77.5. A change of 5 points or less is inside that spread.
- Their run: 70 concurrent runs (server allows 128), 651 tok/s decode at 60 runs, 94 % prefix hit, 0 preemptions (cell46).
- KV: 8.81 GiB per GPU = 4,064,286 tokens (cell36). MTP off.
- notnitsuj: MTP K3, 48 sequences, 1,087-1,120 tok/s.

## 1. Base B0 = Domdolus #16 unchanged

Take all of it: TP4 + EP + DCP2 (a2a), Q2 GSQ experts (Humming, indexed MoE GEMM), FP8 lm_head / HC / dense, K4 KV + FWHT, FP8 indexer keys, FP16 stochastic-rounding DeltaNet state, pinnew hold states, SINK_JUMP 49,152 / 65,536 + trim-free, RAM offload 30 GiB (evict), exact resume, token splice, miss-deferral collapse fix, top-k exchange 64, HC-down split-K, nine decode trims, fast startup (prefetch, /dev/shm n-gram, EP load filter, AOT caches), REPL harness (cells 29-33, 46): xhigh, temperature 1.0, 200k tokens per run, try 1 + independent try 2, check()-ranked candidates.

Dependencies to copy to our accounts first: model `qwen38-flash-next-q2-gsq-vllm` version 2, the utility dataset (`site-vllm.tar`, `site-arc.tar`, caches, `UTILITY_MANIFEST.json`), optional overlays `darrenamadeusmartin/qwen38-flash-next-fp8-dense(-hc)`.

Gate: B0 on our account reproduces their speed line (about 650 tok/s at 60 runs) and an eval score inside 72-83.

## 2. Bundle H: harness (from notnitsuj / andy2709 measured A/Bs, plus our own selection work)

| item | evidence |
|---|---|
| vote across sessions on identical verified test outputs; new sessions to the task with the fewest | notnitsuj, verified sessions right 75 % |
| 11 content invariants + `ak` helper library on candidates | notnitsuj +2.6 [+0.2, +5.2] |
| tie-break: symmetry kept, no test-only code | notnitsuj +0.54 |
| tasks with 2+ votes and a lead go to the back of the queue | notnitsuj +0.8 |
| warm start from best unverified program | notnitsuj +0.21 right per session, 17 % fewer tokens |
| second attempts to unverified tasks first | andy2709 4.6 vs 2.6 solves per M tokens |
| sandbox fixes: import alias, correct traceback lines | notnitsuj +2.25, +0.88 |
| deadline drain instead of stop at -1 min | notnitsuj +1-3 % tokens |
| note to the model when a reply hits the window | cell30 observation |

## 3. Bundle S: speed and concurrency

| item | evidence |
|---|---|
| MTP K3 (needs memory fit; disables the top-k exchange path) | notnitsuj 1,100 tok/s; folding accepted tokens into one DeltaNet checkpoint gave +49 % sequences |
| int8 draft head over top 32k tokens | notnitsuj +19 % |
| concurrency 70 / 96 / 128 | cell46 vs MAX_SEQS 128 |
| split-K route only for small M (slower at M 1024) | cell23 timings |
| QAR int8 compressed all-reduce (built, off) | cell10; notnitsuj host collectives +8.6 % |
| fuse K4 query rotation; gpu util 0.945 → higher | cells 00-20 / cell10 observations |
| andy2709 kernels: fused MoE-down + all-reduce, INT8 HC up, HC mix, prefill graph | -4 to -6 % decode step each (SGLang; port is kernel work) |

## 4. Arms (measured separately, never in the base)

- Pool: KV 3-bit, sink-jump keep/limit changes, bigger RAM tier, expert pruning (must count tool-call routing: verified sessions 0.78 → 0.34 otherwise).
- Try 3 on; final review on; deadline salvage on.
- Logit bias -1 on "Hmm"/"Alternatively" (andy2709, -4.5 % tokens).
- 12,288-token per-request cap + "write solution now" nudge (andy2709 +1.1 to +2.2): conflicts with the 60k thinking floor, only with the user's approval.

## 5. Order

1. Copy dependencies (CPU, needs the user's go for uploads).
2. Live 4xL4 session: B0 boots, speed line (GPU, needs go).
3. Bundle H is CPU-testable (harness + replay); then H + S measured live; one eval run per bundle.
