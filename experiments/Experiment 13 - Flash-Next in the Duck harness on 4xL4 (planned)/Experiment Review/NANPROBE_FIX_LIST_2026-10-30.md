# E13 NANPROBE: fix list (2026-10-30)

Sources:
- G1, G2 = the two GPT audits;
- A = the agent runtime audit;
- B = the agent measurement audit;
- C = my own analysis.

Verdict of all five: the notebook as built (c48adfa5) must not run, but the probe design is right.

Every point below was checked:
- against the SGLang 0.5.21 source (`model_executor/runner/eager_runner.py`, `mem_cache/kv_cache_configurator.py`, `layers/logits_processor.py`);
- against the LOADPROBE log;
- with local reproductions.

| # | Sev | Finding (sources) | Evidence | Fix |
|---|---|---|---|---|
| F1 | BLOCKER | The top-level pre/post hooks never fire: SGLang calls `model_runner.model.forward(...)` directly. No `nan_forward` record would be written, the hooks would never be removed, and the slowdown would last for the whole arm (G1, A1, B-B1) | eager_runner.py:265/371; local repro: `m.forward()` gives 0 events | `functools.wraps` wrapper on the instance `forward` (the signature is kept: model_runner.py:473 reads it). Pre / post logic in the wrapper; it restores the original after the window |
| F2 | HIGH | The logits are never checked. The top output is the `LogitsProcessorOutput` dataclass and lm_head is a plain matmul. `!!!!` = token 0 = argmax of NaN **or of all-equal logits** (C: torch.argmax(all-NaN) = argmax(all-zero) = 0) (G-, A6, B-H1, C) | logits_processor.py:210; local torch check | The wrapper logs `next_token_logits` per forward: finite, min/max, argmax, top-1 minus top-2 gap, row std |
| F3 | HIGH | The `break` after the first floating tensor misses a NaN in the 2nd output (G2). The 12 slots are shared with \|x\| > 1e4, so harmless large values can push out the first NaN (A6, B-H4) | G2 repro; B toy C | Check every tensor and record its index. Keep `first_nonfinite` (always recorded) and `first_big` as separate lists |
| F4 | HIGH | Origin vs propagated: the post hook ignores the inputs, and the layers are called with keywords only, so a NaN made in plain function code (hyper-connection mix, PLE add, all-reduce, gates) is blamed on the next module (B-H2) | B toy B | Pre-hook `with_kwargs=True` records whether the inputs were bad. Origin = the first module with finite inputs and a bad output; else "between the previous module and this one" |
| F5 | HIGH | No collapse detection: finite but zero / constant states give `!!!!` without any NaN (C, B-H1) | the argmax check above | Per module and per decoder layer: min / max (aminmax) and a `constant` flag (min == max); layer stats for every checked forward, not only fw < 4 |
| F6 | HIGH | `abs().max()` + `isfinite` make full-size temporaries; on logits (chunk x 248k vocab) that is about 1 GB against about 1 GB free, so the probe itself can OOM (C). fp8 outputs raise NotImplementedError in a hook, which kills the scheduler (A3, B-M7). The 2 syncs per module are heavy (G2) | local: aminmax propagates NaN / Inf; fp8 isfinite raises | One `torch.aminmax` per tensor (no copy; one `.tolist()` sync); only fp16 / bf16 / fp32; the whole hook body in try/except, with one `hook_error` record |
| F7 | HIGH | The QSA indexer int32 outputs are skipped. The TileLang race candidate gives wrong indices, not NaN (B-H3) | qsa_indexer.py ~435-453 | For modules with `indexer` in the name: int outputs min / max and the count of -1 entries |
| F8 | HIGH | kv_lean mamba 8 caps running to 1 (extra_buffer + overlap = 5 slots per request; 8 // 5 = 1), and it changes two variables at once (G1, G2, A2, B-H5) | kv_cache_configurator.py:2232-2261, 2307-2322; LOADPROBE: 20 slots = 4 running | Keep mamba 20; change only `--mem-fraction-static 0.95`. Capacity gate: `capped_running` must stay 4, else `CAPACITY_GATE_FAILED`; also parse the "Mamba Cache is allocated" line |
| F9 | HIGH | kv_lean memory safety is not tested (smokes ≤ 122 tokens; about 0.6 GB free at 0.95) (A7, B-H6, C) | LOADPROBE: 1.05 GB free after graphs at 0.93 | After the smoke: 4 concurrent long prompts (about 20k tokens each, max_new 16, ignore_eos): OOM yes / no, server alive, `available_gpu_mem`, KV usage |
| F10 | HIGH | Smoke budget: the chat is 4096 tokens with a 1200 s timeout, not bound by the arm or HARD_STOP, and a chat timeout loses the count result (G1, G2-4, A5, B-M5) | e13_smoke | The count result is stored first. nan arm: no chat if the count failed; chat max_tokens 256. Every timeout is clamped to min(the arm deadline, HARD_STOP - 5 min) |
| F11 | MED | The vLLM check runs first: its TimeoutExpired is uncaught (both arms lost), and it costs about 4 min of arm time. It only answers "registered", not "can serve" (A4, B-M6) | CELL 5 | Run it after the arms if ≥ 6 min are left, in its own try/except; also record the implementing module and whether its source mentions ple / offload |
| F12 | MED | No warm-up / prefill / decode labels; clean forwards from fw 4 on are not logged (B-M1, A6) | - | Per checked forward: wall time, input shapes, layer stats and logits stats; smoke start / end timestamps in the arm record |
| F13 | MED | If nan_eager OOMs again, Q1 is lost (B-M4) | LOADPROBE: OOM in 2 of 3 launches | On an nan_eager OOM, retry once with `E13_GC_MIB=0` (collect after every module) |
| F14 | MED | The hooks change the timing, which can hide a race (B-M2) | - | After the hook window: run the count smoke again in the same arm and compare |
| F15 | MED | The summary keeps too little: the bad forward's layer stats are missing, the list is cut at 6, there is no per-rank table (B-M3); the mamba line is not parsed (B-L2) | - | Summary per rank: `first_nonfinite`, `first_origin`, `first_big`, the bad forward's layer / logits stats, logits stats of the first forwards. Caps: mamba slots + GB |
| F16 | MED | The harness runs canned records, not the new code, which is why F1 slipped through (B-H7) | fake server nanprobe branch | A real unit test of the decoded sitecustomize: fake loader; toy model driven through `.forward(...)` with keyword-called layers, an int32 indexer, dataclass logits, an fp8 output, a NaN made between modules, a constant collapse. The fake server prints the "capped" line when the mamba slots / 5 < running |
| F17 | LOW | A busy GPU does not stop the next arm (A8) | - | Skip the arm when `gpu_free_before.free` is False |
| F18 | LOW | Only part of q27x reaches the output zip (B-L1). The FP8 overlay is attached but unused (B-L3) | LOADPROBE zip | Download the full kernel output (unchanged notebook behaviour) and drop the overlay from the metadata |

Order of work:
1. All fixes in one pass (mp_nanprobe.py rewrite, builder CELL 1 / 5, harness, unit test).
2. Local unit test and harness.
3. Rebuild the notebook and run the sha checks; LOADPROBE 107aa182 stays untouched.
4. Hand the result to the user for the GPU go.
