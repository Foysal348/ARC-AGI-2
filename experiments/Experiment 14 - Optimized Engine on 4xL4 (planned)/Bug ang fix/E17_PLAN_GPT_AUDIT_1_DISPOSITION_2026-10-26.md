# Disposition of GPT audit 1 on PLAN_E17 (2026-10-26)

The audit is saved verbatim in `GPT_E17_PLAN_AUDIT_1_2026-10-26.md`. Each claim was checked against:
- the local SGLang 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src/src/sglang`);
- the upstream PR text, https://github.com/sgl-project/sglang/pull/43567;
- the upstream regression test, https://raw.githubusercontent.com/sgl-project/sglang/main/test/registered/radix_cache/test_mamba_cache_kl.py.

Verdict: **accept the "T1 as designed is not decisive" finding**. Two of GPT's six blockers are partly wrong (B2, B4). The upstream test gives a better recipe than either our plan or GPT's.

## 1. The upstream test (new facts; GPT did not have the file)

`test_mamba_cache_kl.py` has two classes.

**Class 1, `TestMambaLazyInplaceKL` (ordinary decode).**
- Environment: `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0` and `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0`. **The failure comes from natural pressure, not from the env flag.**
- Server arguments:
  - `--mamba-radix-cache-strategy extra_buffer_lazy`, `--mamba-track-interval 128`, `--max-mamba-cache-size 4`, `--max-running-requests 1`;
  - `--max-total-tokens 8192`, `--cuda-graph-max-bs-decode 1`, `--cuda-graph-backend-prefill disabled`, `--chunked-prefill-size 2048`;
  - triton backends, `--random-seed 42`, `--skip-tokenizer-init`.
- Requests: temperature 0, `ignore_eos`.
- Prompt: `[42]*128 + [43]*127` (255 tokens).
- Reference run:
  1. warm `[44]*128` (1 token);
  2. warm `[42]*128` (1 token);
  3. score the prompt for 1 token; `cached_tokens == 128`.
- Reuse run:
  1. warm both prefixes again;
  2. generate 32 tokens from the prompt (seed; `cached == 128`);
  3. score the prompt for 1 token (`cached >= 128`).
- Metric: KL = `sum(exp(ref) * (ref - reused))`, threshold `1e-4`.
  - Unpatched: KL 0.649 (a "stale 192-token hit").
  - Patched: KL 0 (a "safe 128-token hit").
- **Full-vocabulary logprobs come from `token_ids_logprob` = `range(vocab_size)`**, with `logprob_start_len -1`. The test asserts the probabilities sum to 1.

**Class 2, `TestMambaSpecTruncationKL` (speculative).**
- Arguments: `--mamba-radix-cache-strategy no_buffer`, `--page-size 1`, `--disable-overlap-schedule`, `--disable-cuda-graph`, `--max-running-requests 8`, `--max-mamba-cache-size 128`.
- Speculation: NEXTN, 3 steps, top-k 1, 4 draft tokens.
- Seed: generate 9 or 5 tokens with chat prompts. The test asserts that verify ran and that drafts were accepted.
- Probe = `prompt + output_ids + "\n"`. Score it warm, flush, score it cold. KL threshold `5e-3`.

This class is a ready template for the warm-vs-cold check of the spec path. It is not lazy and has no overlap, so it does not exercise #43567 itself.

Every flag used by the test exists in our 0.5.21:
- `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` (`environ.py:1448`);
- `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL` (`environ.py:425`);
- `mamba_track_interval` (default **256**, `arg_groups/fields/exec_.py:403`);
- `--cuda-graph-backend-prefill`.

The Qwen3.8-27B vocab size is **248,320** (our model `config.json`).

## 2. The exact mechanism (from the source; my reading, consistent with the PR text)

1. **Prefill** of a 255-token prompt: `schedule_batch.py:~3110` sets `mamba_last_track_seqlen` to the prefill checkpoint, aligned to the cache chunk grid. This is the 192 of "stale 192-token hit". The lazy keep slot holds that state.
2. The **first decode** goes 255 → 256 and crosses the track boundary of 128. `mamba_lazy_prealloc_at_boundary` (`:3464-3493`) calls `alloc(1)`, which fails: 4 slots, and alloc never evicts. Tracking then writes the 256 state **in place into the keep slot**. The length stays 192.
3. **Overlap:** the prefill result processing of step 1 runs after that decode was launched. `cache_unfinished_req` donates the keep slot under length 192, but the slot now holds the 256 state.
4. The next request with the same prefix restores the node at depth 192 with the wrong state. That is KL 0.649.

So the trigger is "the first decode/verify right after a prefill crosses a track boundary while `alloc(1)` fails". **255 = interval − 1 is the key detail.** Our T1 (2-8k prompts, max_new 1, a vague seed) would most likely have missed it.

### Production exposure (hypothesis; UNVERIFIED, this is what T2 must measure)

- **Allocation failure.** `--max-mamba-cache-size 68` with 16 running. Lazy `alloc(1)` never evicts. Once the radix cache fills with evictable checkpoints, `available` can sit at 0 most of the time. If so, r1 = optional-alloc-fail / attempts would be close to 1 in steady state.
- **Non-spec trigger.** It needs `prompt_len ≡ interval − 1` (mod the track grid), about 1/256 per request.
- **Spec trigger** (DFlash, EAGLE). `mamba_lazy_spec_in_window` (`schedule_batch.py:2187-2197`) opens a window of `2 × max_draft_tokens`. A boundary is reachable whenever it lies within that window after `kv_committed_len`, which is about 12-14 / 256 ≈ 5 % of requests. Whether a reachable crossing is actually committed depends on acceptance.
- **Poetiq and Duck are the worst case.** Every expert or iteration of a task shares the task-prompt prefix. If the first request donates a stale checkpoint at prompt depth, every later request of that task restores it.
  - This would hit all experts of the task, not one sample.
  - This is the one way #43567 could cost many tasks rather than a few tokens. It also fits "SGLang < vLLM" (P1-60K 29.17 vs S1-60K 34.58), but that comparison is confounded by the prompt.
  - It must be measured, not assumed.

## 3. GPT's six blockers

| # | GPT claim | Disposition | Evidence |
|---|---|---|---|
| B1 | T1 does not guarantee seeding, stale donation and restore | **ACCEPT** | §2: the trigger needs a boundary crossing on the first decode after prefill under overlap. Our T1 did not arrange this. |
| B2 | A global forced-fail flag may stop seeding | **PARTLY WRONG** | In 0.5.21 the flag only skips the *optional pending* slot (`:3485`, `:3521`). The prefill checkpoint (`:3110`) needs no lazy alloc. Tracking still runs in place, and the length is still set (`batch_result_processor.py:1422`, `:1487`). Seeding is not blocked. Still, **upstream uses natural 4-slot pressure (env = 0)**, so that is our primary arm. Env = 1 becomes a secondary arm. A request-scoped one-shot injector is not needed. |
| B3 | A reachable spec window is not an actual crossing | **ACCEPT** | `_mamba_lazy_spec_update` (`batch_result_processor.py:~1430-1490`) promotes or commits only when `crossed` is true. The trace must log `crossed`, `planned_pos` vs keep, and the accepted length. |
| B4 | Top-k-union KL is not full-vocab KL; full logits need a server hook (`token_ids_logprob` over all ids is "NO-GO") | **First half ACCEPT; second half WRONG** | Upstream's own test passes `token_ids_logprob = range(vocab)` at one position and checks that the sum is 1. At 248,320 ids, that is about 1.7 MB of JSON in and a few MB out, per scored position. That is fine for 1 position × a few dozen probes. No server hook is needed. |
| B5 | Teacher-forced T3 does not exercise DFlash TARGET_VERIFY | **ACCEPT** | Split into T3a (numerical drift: fp16 vs fp32 SSM, FP8 vs BF16 KV) and T3b (spec-path warm-vs-cold, as in upstream Class 2, plus DFlash vs target-only greedy). |
| B6 | Function wrappers cannot see `new_slot is None` | **ACCEPT** | The forced branch never calls `alloc`, and the functions return nothing. Use a diagnostic source patch with counter lines at the call sites, applied to the E17 venv only and never to production. |

Other points:
- Q3, the three-level evidence (trace + state + logits): **accept the trace and the logits**. Reading the state tensor needs the TP worker and adds risk. Full-vocab KL plus a lifecycle trace that links fail → in-place → donated length → restored depth for one rid is enough to prove the chain. State tensors are optional.
- Q5, the counters r1-r4 linked by rid, buffered JSONL every 10 s, no sync in the decode step: **accept**. One addition: log the **donation event at prefill-result time**, with `last_track_seqlen` and whether an in-place write already ran on that request. This is the exact stale event.
- Q6: the two hunks are GO for the isolated test, and production adoption needs more evidence: **accept**. Add slot-leak checks (allocator `available` + evictable + in-use = size at the end) and abort / EOS / `ignore_eos` cases. 0.5.22 is a later secondary arm, not part of E17.
- Q7, staged paired ARC (16-24, then 48-64, then 120), greedy for root cause, McNemar: **accept**. It is outside E17. E17's verdict is at most `ENGINE_CORRECTNESS_VALIDATED`.
- Q8, the extra suspects:
  - #36548, #43029, #25790, #31224, #36880 / #36885: **I have not verified these issue numbers or their claims** (UNVERIFIED). Each must be checked against the 0.5.21 source before an arm is built.
  - KV scales (O6) are already in Phase 0.
- The fail-closed gate `e17_adoptable` with `trigger_path_proven` = the full chain observed: **accept**.

## 4. Changes for PLAN_E17 rev 2 (to apply after the other GPT answers arrive)

1. **T1a, the upstream repro ported unchanged:** the upstream Class 1 arguments, with our model and our production numerics (FP8 KV, fp16 SSM, TP4).
   - The 128 / 255 / 32 numbers work with both track intervals: the first decode goes 255 → 256, and 256 is a boundary for interval 128 and for our default of 256.
   - **Run both:** `--mamba-track-interval 128` (exactly as upstream), and the production default of 256. Add a 511-token variant (`[42]*128 + [43]*383`) for interval 256.
   - Env = 0 (natural pressure). Unpatched vs patched. Plus a flush-between A/A control.
2. **T1b, env = 1:** a secondary arm that forces the failure with production slot counts.
3. **T1c, spec:** the same lazy and overlap setup with DFlash W6, and with EAGLE MTP if time allows.
   - Choose prompt lengths so that the next boundary lies 1..2×draft tokens after the prompt end.
   - Log `crossed` and `planned_pos`.
   - Also run upstream Class 2 (no_buffer, NEXTN) unchanged as a known-good control.
4. **KL:** full vocab through `token_ids_logprob = range(248320)` at one position. Drop the top-k-union KL.
5. **Instrumentation:** a source-level diagnostic patch.
   - Counters: eligible / attempt / fail / in-place; spec `in_window` / `has_pending` / `crossed`.
   - The prefill-result donation length vs in-place-already flag, the insert length in `cache_unfinished_req`, and the matched mamba depth on the next request.
   - Linked by rid. Buffered JSONL.
6. **T2:** production 16-stream, Poetiq-shaped, many requests sharing a task prompt.
   - Measure r1-r4. For each shared-prefix node, compare the warm probe vs a cold probe after a flush (full vocab, one position).
7. **T3a / T3b:** as in B5.
8. **The extra Q8 suspects:** add arms only after source verification.

No GPU run before the other GPT answers are merged and the user gives the go.
