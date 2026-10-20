# E13 Stage 1 (FN-PROBE-2): audit request + pre-run capacity check (2026-10-19)

## What to audit

**Notebook:** `E13-FN-PROBE2-4XL4.ipynb` (sha256 `79132ab9…`).
- Built by `Build/build_e13_probe.py` from `Build/e13_scripts/*.py`.
- It implements PLAN rev 4.1, Stage 1.
- Nothing has run on a GPU yet. Nothing has been pushed.

**Ask:** find **blockers and bugs** (file:line) that would waste the GPU run or give a wrong conclusion. Please:
- separate "certain" from "possible";
- do not re-litigate the plan decisions in PLAN rev 4.1 (minimal change, Franzen-verbatim, FA2-first, target-only Launch 1);
- check the GPU-only code paths below with special care: the local harness cannot run them.

## Cell map

| Cell | Content | Origin |
|---|---|---|
| 0 | Mode + identity fallback `submission.json` | **verbatim** E12 EVAL30 CELL 0 (ran on Kaggle) |
| 1 | Config: input discovery by glob, FA2/patch SHAs, LAUNCH1 config, hard stops (2.5 h total; L1 75 min; L2 60 min) | new |
| 2 | **Verbatim** E12 engine helpers (`E11Fail`, `_e11_post`, `_e11_gpu_used/_free`, `e11_install`, `E11Server`).<br>New: scripts unpacked, FA2 install (SHA, '+' name restore, then FA4 re-install, because the FA2 wheel carries 21 old `flash_attn/cute` files), Franzen patch #3 with his SHAs, `apply_p1` / `apply_p2` (applied only if chosen), static asserts. Franzen's draft-view helpers are appended verbatim. | E12 + Franzen cell 12 + new |
| 3 | Micro-preflight on GPU, no model: `mp_decode` (FA2 / P1 vs FP32), `mp_topk` (T1 fast_topk, T2 FlashInfer, tie-aware oracle), `mp_mqa` (TileLang vs torch indexer), `mp_pinned` (real per-rank BF16 PLE pin, sequential with a MemAvailable guard). Pre-declared decode choice. | new |
| 4 | Launch 1 (target-only) → memory gate (server log) → Gate C → speed stage → `launch2_decision` → optional Launch 2 | new |
| 5 | Report JSON + bundle zip | new |

## Pre-run capacity check (mandatory; measured numbers)

| Item | Number | Source |
|---|---|---|
| Quota rate | 2.0x wall on L4x4 | hosen42 quota: EVAL30 4.15 h wall → 8.22 h |
| hosen42 quota left | 60 − 20.07 = **39.9 h** (2026-10-19 23:20) | check-only output |
| Session wall time, worst case | 2.5 h hard stop (L1 ≤ 75 min, L2 ≤ 60 min) → **≤ 5 h quota** | notebook constants |
| Expected | install 3 + micro 10-15 + load about 9 + JIT about 10 + Gate C about 10 + speed about 10 → **about 60 min L1**, about 2 h quota | E11 install 181 s; E3 345 s / 124.8 GB; E3 warm-up 7.4 min |
| Host RAM | BF16 PLE 107.6 GB in total → 26.9 GB/rank pinned (one at a time, each needing MemAvailable ≥ 26.9 + 15 GB) of 202.6 GB | HF sizes; E3 stage 0 |
| GPU per rank | weights 19.06 GB; free after load 2.55 GB (E3). L1 mem 0.93 keeps about 1.5 GiB outside the static pools | E3 serve.log |
| KV (L1) | expected about 150-210k fp8 tokens (0.93 x 22.5 − 19.06 − 40 mamba slots x 0.016) | arithmetic; measured at launch |
| Duck relevance | E12 prompts p50 47.9k / p90 97.7k (n = 454) → L1 KV fits about 2-4 contexts | `eval30_request_stats_output.json` |

The run is worth its cost only as a measurement. It answers four things:
- numerics (FA2 / P1, top-k, indexer);
- the real KV and free memory;
- the real tok/s at Duck-like contexts;
- whether NEXTN can fit.

## Decision rules coded in the notebook (pre-declared)

- **Decode:**
  - FA2 if `mp_decode` FA2 passes (incl. graph replay) **and** the server resolves to FA2;
  - else P1 if it passes;
  - else stop.
- **Indexer:** TileLang unless `mp_mqa` fails; then P2 (torch reference) before Launch 1.
- **Top-k:** T1 kept (as Franzen); T1/T2 drop rates are reported only.
- **Launch 2 (`e13_gate.launch2_decision`):**
  - load / Marlin / OOM → stop;
  - mismatch in micro tests → B2 with one variable;
  - unknown → B2-D diagnostic;
  - pass + memory budget ≥ 2 x 48k fp8 → NEXTN on;
  - else chunk 4096 if peak headroom allows;
  - else none.

## Local evidence (CPU, no model)

`_scratch/session_1c6db7aa/e13_harness/` runs the real notebook cells against fake inputs and `fake_e13_server.py`.

| Scenario | What it checks | Result |
|---|---|---|
| `ok` | flag set (Franzen + deltas, banned flags absent), env, memory parse, Gate C, speed, decision | PASS |
| `garbage` ("!!!!", the E3 symptom) | signature `decode_wrong` → B2-D (TileLang off + graph off) → stop | PASS |
| `loadfail` (Marlin error) | stop, no B2 | PASS |
| `oom` (OOM after start) | OOM is read from the log at the end → stop | PASS |
| `parser` | `memory_gate` on the **real** E3 and E12 server logs | PASS (E3: 2.55 / 19.06 / 118,080; E12: 866,744 fp8, graphs 1.03 GB) |
| oracles | top-k: correct impl 0/49 fails, broken impl 49/49 caught. Decode: reference passes, wrong-causal impl caught (54 cases). | PASS |

**Not testable locally (please review closely):**
- the GPU branches of the four `mp_*.py` scripts (FA2 / Triton / fast_topk / FlashInfer / TileLang APIs and shapes);
- `e11_install` + FA2/FA4 pip steps on Kaggle;
- the real SGLang 0.5.21 behaviour with these flags;
- `/generate` logprob fields;
- the chat tool-call path with Flash-Next's template.

## Specific questions

1. `mp_decode.py`: is the packed layout identical to 0.5.21 `_forward_paged_attention` (scratch in q.dtype, cu_seqlens_k, topk 2051, `causal=True`, `max_seqlen_q=1`)? Is the P1 relative-index construction identical to the `is_hip()` branch?
2. `mp_topk.py`: is the oracle correct for ties and short rows? Are the T1/T2 call signatures right (`fast_topk(score, lengths, topk, row_starts)`; `top_k_ragged_transform(input, offsets, lengths, k, deterministic, tie_break, dsa_graph_safe)`)?
3. `mp_mqa.py`: are the shapes valid for `tilelang_qsa_mqa_decode` (q [B,4,128] bf16, k_cache [pages,64,1,128], page_table int32)?
4. `mp_pinned.py`: can the pinned test itself OOM-kill the kernel despite the guard? Is `torch.empty(..., pin_memory=True)` the same allocation path as SGLang's PLE table?
5. Cell 2: is reinstalling FA4 after FA2 safe, and does it really restore the `flash_attn/cute` files?
6. Cell 4 flags: any flag that 0.5.21 rejects for Qwen4Exp + auto-round + EP4, or any flag combination conflict?
7. Gate C (`e13_gate.py`): any false pass or false fail (logprob threshold −0.5 median; counting; grid echo; marker; tool call)?
8. `launch2_decision`: is the NEXTN memory budget formula reasonable, and is anything missing?
