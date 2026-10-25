# E13 (simple path): Duck SMOKE1 on Qwen3.8-Flash-Next, audit request (2026-10-20)

## Ask

Find **blockers and bugs** (file:line) that would waste the GPU run or give a wrong conclusion. Please:
- separate *certain* from *possible*;
- **do not** propose new tests, new stages or architecture changes. The user explicitly chose the simple path (minimal change, Franzen-verbatim, reuse E12);
- review the GPU-only code closely (the local harness cannot run it).

## What the notebook is

**Notebook:** `notebook/Q27-XHIGH-E13_DUCK_FLASHNEXT_SMOKE1.ipynb` (sha `9a8d17ca…`).
- Built by `build/build_e13_duck.py` from the E12 SGLang SMOKE1 notebook (`base/…E12…SMOKE1.ipynb`, sha `0fea4d0c`).
- That E12 notebook **ran and passed on Kaggle** on 2026-10-19: task solved, structured tool calls, cache 80 %.

| Cell | E13 vs E12 |
|---|---|
| 0 (fallback submission) | unchanged |
| 1 (config) | only: model / draft path + discovery predicate (auto-round; Franzen's drafter test), `SGL_CONTEXT_LEN` 139,264 (Franzen), `SGL_MAX_RUNNING` / `SGL_SERVER_MAX_RUNNING` 4, SMOKE1 budget 50 → 90 min, plus an appended E13 constants block |
| 2 (install + tokenizer) | unchanged. The Flash-Next `tokenizer.json` is byte-identical to our RedHat INT4 one (sha `06b95093`, also Franzen's `TOKENIZER_SHA`) and the EOS ids match, so the existing tokenizer check passes (harness: ok) |
| 3 (Duck package + adapter + sandbox) | unchanged |
| 4 (tasks) | unchanged |
| **5 (engine)** | **new, Duck-only** (structure of the E12-V vLLM cell); details below |
| 6 (Duck solver) | unchanged; uses `ENGINE_READY`, `SGL_URL`, `metrics_snapshot`, `CONCURRENCY`, `HARD_STOP`, `SGL_MODEL_PATH` |
| 7 (report) | unchanged; uses `E11_SERVER`, `ENGINE_INFO`, `engine`, `e11_decode_summary`, `e11_kill_server` |

### CELL 5 in order

1. `e11_install` (verbatim E12): the 0.5.21 wheelhouse venv.
2. FA2 wheel:
   - SHA check; Kaggle drops '+' from the file name, so the name is restored;
   - `--no-deps` install, then the wheelhouse FA4 is re-installed (the FA2 wheel ships 21 old `flash_attn/cute` files).
3. Franzen patch #3 with his BEFORE/AFTER SHAs. The 0.5.21 `gptq_moe.py` equals his BEFORE.
4. A probe asserting that SGLang's QSA decode resolves to classic FA2 (not cute).
5. The top-k test (`mp_topk.py`, SGLang's JIT `fast_topk<512>` and FlashInfer `top_k_ragged_transform` vs a tie-aware torch oracle). **Recorded only, never gating** (user-mandated).
6. Host-RAM guard: MemAvailable ≥ sum(model-ple shards) + 20 GB.
7. Franzen's launch (his cell 12 flags) with:
   - D1 0.5.21; D4 TP4 + EP4; D5 triton GDN / mamba; D6 `TORCH_CUDA_ARCH_LIST=8.9`; D7 no `--gdn-mtp-cache-mode`;
   - memory items: NEXTN off, chunk 2048, mem 0.93, 4 running, mamba 20.
8. Readiness: `/health` only while the launcher lives; 15 min of log silence = stall.
9. Caps parsed from the TP0 log.
10. Smoke:
    - raw greedy counting via `/generate`;
    - a chat "2 + 2";
    - fail on "!!!!" (the Experiment 3 symptom).
11. `CONCURRENCY = min(4, capped, KV // 60k)`, the pre-run capacity rule (E12 EVAL30 prompts p50 47.9k).

## Local evidence

`harness/` runs the real cells with `fake_e13_duck_server.py`. The server refuses to start unless Franzen's flags + deltas are present and NEXTN / `--gdn-mtp-cache-mode` / `--language-model-only` are absent.

| Scenario | Result |
|---|---|
| smoke | 24/24 checks: engine ready, streams 3, Duck solves the task, task-norm 100 %, cache 88 % |
| garbage | 7/7: engine refused, Duck not run, fallback submission kept |
| launchfail | 7/7 |

## Capacity check

**Run:**
- 1 task x 2 branches (2 conversations);
- hard stop 90 min → ≤ 3 h of quota (hosen42 has 33 h left);
- E12 solved this task in 5-6k generated tokens.

**Per GPU (measured in E3 or estimated):**
- weights 19.06 GB;
- 2.55 GB free after load;
- at mem 0.93, KV ≈ 150-210k fp8 tokens (estimate) → 2-3 streams.

## Questions

1. Are there any CELL 6/7 dependencies the new CELL 5 does not provide, or provides with a different meaning?
2. FA2 install: is the `--python` + `--find-links /tmp/e11-wheels` + FA4 force-reinstall sequence correct with `e11_install`'s venv? Is there any risk that pip resolves something else?
3. Are any flags rejected by SGLang 0.5.21 for Qwen4Exp + auto-round + EP4, or do any conflict (`--cuda-graph-bs-decode 1 2 4` with 4 running; `--warmups structured_output`; `--chat-template` from the model dir)?
4. Smoke: can the counting check or the "2 + 2" check give a false fail with Flash-Next (it always thinks; max_tokens 4096)?
5. Is there any path where a failure leaves the GPUs busy or the submission invalid?
6. Is the CELL 1 discovery predicate for the Intel model correct given its real `config.json` (`quantization_config` at the top level?).
