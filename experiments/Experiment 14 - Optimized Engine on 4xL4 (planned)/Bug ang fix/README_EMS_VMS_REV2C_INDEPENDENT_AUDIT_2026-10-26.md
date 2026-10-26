# EMS/VMS rev2c independent pre-GPU implementation audit — 2026-10-26

## Scope and evidence

Read first: `README_FOR_GPT.md`, `notes/README_PURPOSE_AND_WORK.md`, `notes/README_FINDINGS.md`, `notes/Deep_reread_rev2c_2026-10-26.md` in `ENGINE_MATRIX_EMS_VMS_REV2C_AUDIT_FOR_GPT_2026-10-26.zip`.

Actual notebook sha256 prefixes match bundle MANIFEST: EMS `c062255e`, VMS `675db353`. Kaggle CPU-kit run3 records EMS 26/26, VMS 19/19, and VMS negative ignore-list 19/19 checks. These are fake-server/CPU logic tests, NOT real GPU serving verification. No real rev2c GPU run was supplied. Older GPU results are reported in `notes/README_FINDINGS.md` and must not be treated as rev2c measurements.

**Verdict: CPU-KIT PASS; production-correctness certification NO-GO until HIGH items fixed.** The matrix can be run as a bounded diagnostic, provided no FAIL/INCOMPLETE arm is adopted and the pre-GPU CPU regression suite below passes.

## New independently identified findings beyond prior GPT/4-agent reports

### HIGH 1 — Missing target-only oracle falls back to speculative baseline (certification fail-open)
- EMS source: `builders/build_ems.py:564-610`, `_gated():503-518`. If `EMS_TGT` or probe A/A fails, `EMSR['oracle']='dflash_ref'`; `_gated` can still adopt an arm that agrees with speculative DFlash, which is exactly the path subject to upstream wrong-output reports.
- VMS source: `builders/build_vms.py:587-611`; if `fp8_target_only_noprefix` has an unusable A/A, VMS retains `ref_fp8_mtp3_s16` as the comparison reference.
- Fix: retain benchmarking but set `oracle_ready=False` and `certified=False` for every arm until a complete, non-degenerate, target-only oracle A/A is available. EMS `_gated` must return False unless `EMSR.get('oracle')=='target_only'`. VMS candidate production PASS must require `VMS.get('oracle_ready') is True`.
- Validation: corrupt the oracle A/A or probe by removing one row/margin. Candidate with perfect speed cannot become certified. Separate `diagnostic_gate` from `production_gate` in report.

### HIGH 2 — VMS incomplete SSE stream counts as valid identical greedy row
- Source: `builders/build_vms.py:223-262,355-376,407-435`. `VGen` stores streamed text/tokens/logprobs, but not per-request `finish_reason` or terminal `[DONE]`; `vms_diverge` sees two identical partial token arrays and returns errors=0/identical=1. Independently reproduced using `vms_diverge` extracted from the attached builder: `tokens=['foo']` with incomplete finish vs `tokens=['foo']` with stop -> identical 1, errors 0.
- Fix: initialize `self.done_seen=False`, `self.finish_reason=None`; update finish_reason from SSE `choices[].finish_reason`, set done_seen on `[DONE]`. A non-aborted greedy request lacking BOTH must set `self.err='incomplete SSE response'`. Include `done_seen` and `finish_reason` in `row()` and reject incomplete rows before compare.
- Validation: complete stop, complete length, EOF without DONE, truncated JSON, socket reset, and stalled timeout cases.

### HIGH 3 — Partial stream failures do not automatically invalidate throughput arms
- EMS: `harness/e14_measure.py:321-367` returns `errors` list and `real_running_share`; `builders/build_ems.py:450-518` uses singular `short_error`/`long_error` and median speed only; it does not hard-reject nonempty `window['errors']` or low `real_running_share`.
- VMS: `builders/build_vms.py:345-350,613-624` stores `stream_errors` and `degenerate` but `a['ok']` is just both `tok_s` truthy, ignoring those data.
- Fix EMS: copy `window_errors_short/long` and `running_share_short/long` from window results to record and `_gated()` reject nonempty errors and sustained share below an explicitly documented threshold (e.g. 0.8; this is a proposed quality bar, NOT an upstream invariant).
- Fix VMS: require no window error, both `stream_errors==0`, `degenerate==0`, measured level and running_mean acceptable; keep capped arms diagnostic-only.
- Validation: inject one simulated client error while token counter stays positive; FAIL production selection.

### HIGH 4 — VMS window corruption heuristic silently drops empty outputs
- Source: `builders/build_vms.py:350`: `vms_degenerate(["".join(g.text) for g in gens if g.text])`. This excludes the empty-output responses that `vms_degenerate` intends to catch.
- Fix: `texts=["".join(g.text) for g in gens]`; `out['degenerate']=vms_degenerate(texts)`; `out['empty_outputs']=sum(not x.strip() for x in texts)`; report `stream_errors` even when text empty.
- Validation: 16 streams, 1 empty, 1 repeated `!` 100 times => at least 2 degenerate and 1 empty.

### HIGH 5 — Decode-only rate is presented as if it includes 40k-prefix recompute
- Source: `builders/build_vms.py:307-345`: waits for `vllm:prompt_tokens_total` plateau, THEN starts `generation_tokens_total` delta; headline `tok_s` is decode-window rate, not end-to-end agent throughput. The report note about no-prefix recomputing the 36k prefix does NOT make decode-only tok/s include that prefill time.
- Fix: separate `steady_decode_tok_s`, `end_to_end_output_tok_s`, `prefill_settle_sec`, `prefill_settled`, `running_mean`, and `request_success`; treat a 120-s settle timeout as mixed-load / inconclusive, not successful decode-only result.
- Validation: fake server with 60s prefill and fast decode should show high decode rate but much lower end-to-end rate; both must be recorded.

### MEDIUM/HIGH 6 — PR #45151 patch needs atomic write, exact marker, and memory-delta proof
- Source: `builders/build_ems.py:282-323`, paired with `source_refs/sglang_0.5.21/srt_speculative_eagle_worker_v2.py:159-241`, `scheduler.py:1049-1075` and `notes/upstream_sglang_pr45151.diff`. Upstream PR is open and changes init ordering to share draft embedding/head before KV pool profiling. Your env-gated implementation closely follows that logic; `torch.cuda.empty_cache()` follows binding. However, `if 'EMS_PR45151' in src: applied=True` cannot distinguish a complete patch from a partial/stray marker. `Path(fp).write_text(src)` directly rewrites installed Python module non-atomically. A killed notebook during write can break ALL later arms even with the env var off.
- Fix: read original, check two unique anchors or verify BOTH complete patch markers, compile, write to an adjacent temp file, `os.replace(temp, fp)` atomically, keep original backup, and assert the installed source SHA after change; on any failure disable the patched arm. Record actual KV tokens, GPU reserved, free HBM and startup time before and after with same flags. Do NOT assume +1.19 GiB/GPU fully becomes KV capacity.
- Validation: unpatched -> valid atomic patch; patched -> idempotent; partial marker -> fail closed; wrong anchors -> fail closed; boot both arm versions; compare KV pool and greedy tokens under same cache hygiene.

### MEDIUM 7 — 40k band is not a long-context (>64k) correctness test
- EMS docs report bands ~8k, ~40k + limited generated tokens; VMS `vms_flags(max_len=106496)` versus EMS 155648. Zero rev2c high-context parity evidence. A 60-90k generated-token path can encounter state-cache bugs outside 256/1024 token gate.
- Fix: add a bounded dedicated gate of >=70k actual **total sequence position** plus one >=100k position if memory permits, with changing concurrent traffic and a same-prefix recovery request. Preflight memory before running. This is separate from the 75s throughput matrix; do not claim long-context correctness without it.

### MEDIUM 8 — Timing and winner selection
- EMS PLAN has 12 arms plus reference, target, two 2xTP2 starts. With historical 7-8 min cold starts, many arms will not fit 200min. EMS `E14_RESERVE=30min` does not guarantee BOTH TP2 arms when each cold TP2 startup can approach 30min. VMS 35min reserve likewise has two TP2 arms with 16min per-arm cap EXCLUDING startup.
- Fix: prioritise target-only oracle, reference, base MTP, PR45151 MTP A/B, 20-stream DFlash and one TP2 diagnostic; run extra NTN/FP8 variants last or on separate quota. Keep formal reserve for both TP2 only if measured startup budget makes it feasible.
- Validation: simulate 8/12/20/30-min startup durations, confirm no arm enters mandatory reserve; report skipped arms and reason.

### MEDIUM 9 — MTP and DFlash silent semantic contamination is not necessarily caught by punctuation heuristic
- Upstream SGLang #36548 reports cross-request context bleed; the resulting output can be coherent but answer the wrong task. Existing 12 greedy prompts/256 tokens and 1024 probe are not proof against rare state corruption. vLLM #53912 reports sub-1% intermittent malformed responses; 0 incidents in a very small sample is weak evidence.
- Fix: a canary sweep with request-specific random sentinel text injected into 16 concurrent prompts, distinct prefixes, warm-cache replay and churn; compare cross-request token/string membership and top-logprobs, include 0-accept + partial accept controls; use some long >8k generations. Fail on any cross-request sentinel leakage.

### HIGH 10 — Margin-based PASS is not bit-for-bit greedy identity
- Source: `harness/e14_measure.py:691-718` and `builders/build_vms.py:560-575`. A near-tie first-token divergence can pass with zero 'large_margin_flips', yet the rest of a 60k-token reasoning chain may diverge. This is a valid *diagnostic tolerance* but not proof of identical answers.
- Fix: classify `INCOMPLETE`, `FAIL` (confident), `BOUNDARY` (any divergence only near tie), `PASS_EXACT` (every expected sequence identical). Only `PASS_EXACT` may be automatically certified without separate task-quality evaluation. Document a distinct rule if the team intentionally accepts BOUNDARY.
- Test: same oracle A/A, candidate changes one argmax at logit margin 0.5. Current margin gate may PASS; corrected strict gate must return BOUNDARY and not auto-adopt.

## Upstream evidence (literal raw URLs, copy/paste to Claude)
- https://github.com/sgl-project/sglang/issues/44145
- https://github.com/sgl-project/sglang/pull/45151
- https://github.com/sgl-project/sglang/issues/36548
- https://github.com/sgl-project/sglang/issues/38009
- https://github.com/vllm-project/vllm/issues/53912
- https://github.com/vllm-project/vllm/issues/63593
- https://github.com/vllm-project/vllm/issues/51250
- https://github.com/vllm-project/vllm/issues/45238
- https://docs.vllm.ai/en/v0.31.0/design/metrics/
- https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/eagle_worker_v2.py
- https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py

## Boundaries of external claims
- #45151 was open when checked 2026-10-26; the bundler's opt-in patch must not be called a merged upstream fix.
- vLLM #63593 is a v0.32.0 high-margin forced-reject failure for **K>=5**, tested on a Blackwell RTX PRO 6000; your K=2/3/4 arms are **not directly proven affected**. It reinforces need for rejection-boundary tests; it is not evidence that your K2-4 fail.
- #53912 reports several older vLLM versions/hardware, not confirmed your exact 0.32.0 L4 stack. Intermittent risk remains a validation concern, not a finding of actual wrong outputs in the attached notebook.
- 26/26 + 19/19 CPU harness confirms orchestration/fake-server scenarios only. It does not validate Marlin/FP8/GDN kernel numerics or the speculative state commit on L4.

## Practical GO/NO-GO
- GO: read-only CPU review, static test, fake-server negative test, bounded diagnostics that can NEVER select a production arm.
- NO-GO: automatically promote fastest result until HIGH1-5 fixes and real-GPU oracle/stream/concurrency/long-context gates pass.
- Partial GO: PR45151 A/B diagnostic can be included with atomic/idempotent patch verification and separate KV gain measurement; no claim of an actual speed gain before GPU evidence.

## Mandatory acceptance checklist before choosing production winner

1. Oracle: target-only A/A complete, no fallback to DFlash/MTP reference for certification.
2. SSE: `[DONE]` and an acceptable `finish_reason` recorded for each expected completed request.
3. Window health: zero transport/generation errors, zero unexplained empties/degenerate output, declared full occupancy and measured active-share.
4. Performance: report both steady-state decode and end-to-end output tok/s including prefill, with real completion rates.
5. Correctness: label exact token equality separately from near-tie boundary differences and incomplete comparisons.
6. Resource fit: memory pool, Mamba slots, CUDA graph size, and measured peak VRAM within each 24GB L4; no silent concurrency reduction.
7. Long run: at least one >64k-position replay with churn and rejected drafts before production approval.
8. Regression: bundle CPU suites 26/26 EMS and 19/19 VMS (including negative ignore-list), plus NEW synthetic negative tests from findings above.
9. Provenance: effective engine/model revisions, flags, installed source hash, cache path, exact task prompts and output IDs saved in receipts.
10. Decision: any missing evidence is `INCOMPLETE`, never `PASS`. Diagnostic-only variants are not auto-adoptable.

**No GPU experiments were run as part of this independent audit.**
