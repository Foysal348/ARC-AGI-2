# EMS / VMS rev 2b: 4-agent audit (2026-10-26), disposition of every finding

## Agents and inputs

Four independent read-only agents audited the rev 2 notebooks:
1. EMS code paths.
2. EMS flags vs the SGLang 0.5.21 source.
3. VMS code paths.
4. VMS flags vs the vLLM v0.32.0 tag source.

Their full reports are in the audit pack (`agent_reports/`). I re-checked every claim used below in the code or the source before changing anything.

Builds after the fixes:
- EMS `Q27-XHIGH-EMS_ENGINE_MATRIX_SGLANG.ipynb` sha `f9201b43`: local harness 25/25.
- VMS `Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb` sha `94c3cc05`: harness 18/18 normal, plus the `VMS_BAD_INT4=1` mode.

## EMS (agents 1 + 2)

| ID | Finding | Verified | Action (rev 2b) |
|---|---|---|---|
| A1-F1 | `_gated` passes an arm when its probe has no evidence: 0 compared, errored, or a skipped reference probe | yes (`compare_passes` counts only completed rows) | `_cmp_complete()` helper; `_gated` needs a complete probe compare and a complete oracle probe A/A |
| A1-F2 | dirty-server flags from the A/A and probe passes are lost | yes (only `arm_gate` propagated `server_not_idle`) | `_aa_rows` and every probe return `server_not_idle` (so `e14_phase` relaunches clean). The target A/A is not used when dirty, and `_gate_from_rows` gives INCOMPLETE on dirty base rows |
| A1-F3 | a dead server or a failed 40k band can still be "ok" and gated | yes | `_windows` stops on a dead server and records `server_alive_after`. `capped` includes the 40k band. `_gated` needs alive, both bands, no window error |
| A1-F4 | TP2 ok, sum and gate cover replica 0 only | yes | both replicas gated in parallel; ok only when all four windows have tok/s and no error; log signs per replica |
| A1-F5 | after the oracle switch, DFlash arms cannot be told apart from "DFlash differs from plain decoding" | yes | own gate `_gate2` (same rule as `arm_gate`, identity=False), plus a second verdict vs the DFlash reference rows (`gate_vs_dflash_ref` column) |
| A1-F6 | the best-MTP pick misses ntn_int4 | yes | `family` tag (mtp / dflash / target); the picks filter on it |
| A1-F7 | the picks do not require valid windows | yes | `_gated` needs >= 8 decode lines (1 in the harness scale), not capped, both bands |
| A1-F8 / A2-F7 | fallback paths: the reference label is wrong, dflash_wN = reference, s20 / s24 doomed after streams12 | yes | `REF_NAME` from `E11_SGL["launch_flags"]`. dflash_wN is skipped when N = the reference width, and s20 / s24 under streams12. `--cuda-graph-max-bs-decode` is dropped with any explicit list |
| A1-F9 / F10 | gate and probe can use different oracles; the target A/A is accepted with missing margins | yes | one flag `EMSR["oracle"]`: target-only only when its gate A/A AND probe A/A are complete; otherwise both stay on DFlash |
| A1-F11 | a failed launch shows gate "reference" | yes | "no data" when there are no rows |
| A1-F12 | empty prompts → ZeroDivisionError per window | yes | EMS stops before the first launch with a reason |
| A1-F13 | no top-level guard; a TP2 start failure leaks a server | yes | the whole EMS body is in try / except (the report cell always runs); the TP2 start loop is guarded |
| A1-F14 | skip message says 15 min (real floor ~16.7 min) | yes | text fixed |
| A1-F15 / A2 | FlashInfer workspace shared by the TP2 replicas | **no issue** | SGLang `third_party_cache_defaults()` sets `FLASHINFER_WORKSPACE_BASE` and `TORCHINDUCTOR_CACHE_DIR` under SGLANG_CACHE_DIR with setdefault (environ.py:1963-1975), so they follow each replica's own dir |
| A1-F16 | capacity: ~7-9 of 12 PLAN launches fit in 200 min | estimate | kept; order = priority; reported to the user |
| A2-F1 | NTN arms capped at ~11 (INT4) / ~9 (FP8) by the auto mamba size (ratio 0.5, fp32, 5 slots) | yes (kv_cache_configurator.py:2292-2330, 2505-2545) | `--max-mamba-cache-size 85` (5 x 17) added; the ratio is then ignored (:2462-2484). Noted in the report |
| A2-F2 | 2 x TP2: ~115-150k KV per replica (estimate) cannot hold 8 x 50k | plausible, estimate | `kv_ok_for_production` flag per TP2 arm; the arm still measures (user-requested) |
| A2-F3 | TP2 legacy CustomAllreduce turns on at world size 2 when P2P passes; P2P measured 5.5x slower (E14A p2p_sys 1,880 us vs 344 us) | yes (custom_all_reduce_utils.py:442-512) | `--disable-custom-all-reduce` on both TP2 replicas (VMS TP2 likewise) |
| A2-F4 | ntn_fp8 reads FP8 from GCSFUSE under a 20-min limit | yes (not staged) | 30-min launch timeout for ntn_fp8; FP8 shards pre-read into the page cache in the background at EMS start |
| A2-F5 | TP2 double cold compile under a hard 20-min wall | yes | 30-min wall, never past the hard stop |
| A2-F6 | NTN attention flashinfer / linear triton / max-prefill 4096 are no-ops | yes | kept as the flag bundle; noted in the report |
| A2-F8 | mem 0.86 headroom | info | log signs record any OOM |
| A2 note | E14P 43567 prealloc patch is not applied | yes | unchanged (production does not apply it either) |

## VMS (agents 3 + 4)

| ID | Finding | Verified | Action (rev 2b) |
|---|---|---|---|
| A3-F1 | the reserve / cap apply only at launch; one arm can eat the 2 x TP2 reserve | yes | one deadline per arm, `min(ready + cap, HARD_STOP - reserve - 120)`, passed to every greedy pass and window |
| A3-F2 | `left` computed before the kill + GPU wait | yes | recomputed after; the GPU-free result is recorded |
| A3-F3 | window waits 900 s on a dead server; a KV plateau is an error | yes | dead-server stop; plateau accepted (waiting > 0, running flat 30 s) and recorded |
| A3-F4 | the window starts while the last requests are still prefilling | yes (a request counts as running once its first chunk is scheduled) | wait until nothing waits and `vllm:prompt_tokens_total` stops (<= 120 s) |
| A3-F5 / A4-F4c | DP: KV usage summed across engines; only one KV-capacity line kept | yes | per-engine max (`@max`); all KV lines kept (`kv_tokens_all`); `gpu_cache_usage_perc` dropped |
| A3-F6 | in-stream error events ignored | yes | an error event sets the row error |
| A3-F7 | `VGen.stop()` can block on the reader lock | plausible | socket shutdown before close |
| A3-F8 | greedy joins have no deadline (12 x 900 s worst case) | yes | one deadline per pass; stragglers stopped and marked |
| A3-F9 | p0 / p1 prom calls outside any try | yes | try / except / finally; streams always stopped |
| A3-F10 | whole-text equality vs a no-spec oracle may FAIL every MTP arm; rows not saved | yes (design) | token-level gate with top-1 / top-2 logprob margins (`logprobs=2`); large-margin flips may not exceed the oracle's A/A flips (the EMS rule); INCOMPLETE on missing margins; every row saved to `vms_greedy_rows.jsonl` |
| A3-F11 | reference A/A has no usability check | yes | `complete` flag; an incomplete base makes later gates INCOMPLETE |
| A3-F12 / A4-F3a,b | INT4 static check: the ok rule passes a partly packed MTP; prefix matching is not vLLM's exact rule | yes (config_utils.py:344-362) | exact name or `re.match`, fused qkv / gate_up names added, ok = every dense MTP linear exempted. RedHat passes (`re:^mtp.*`, 15 dense keys); the bad-ignore mode skips all INT4 arms |
| A3-F13 | tp2x2_int4 leaves no record on a failed static check | yes | recorded as skipped with the reason |
| A3-F14 | the report hides why an arm failed | yes | the last column shows the error / skip / window error |
| A3-F15 | SIGKILL at once | yes | SIGTERM, wait 30 s, then SIGKILL |
| A3-F16 / A4-F4b | DP=2 starts two API servers by default | yes (cli/serve.py:107-130) | `--api-server-count 1` on the TP2 arms |
| A3 "outside scope" | the install cell pre-reads INT4 + DFlash2 (SGLang default), not FP8 | yes | VMS pre-reads the FP8 checkpoint |
| A4-F1 | `fp8_mtp3_lb_marlin` = the reference: on SM 8.9 auto already picks MarlinFP8ScaledMM (W8A16); CUTLASS block FP8 needs SM 90+ | yes (kernels/linear/__init__.py:478-489, cutlass_scaled_mm_entry.cu:161-174) | arm dropped (recorded); `lb_triton` = the only real W8A8 FP8 arm |
| A4-F2 | the kernel regex misses the `--linear-backend=` fallback lines | yes | case-insensitive, plus `--linear-backend=` lines |
| A4-F3 | INT4 + MTP: draft under prefix `mtp`, `re:^mtp.*` exempts every MTP linear | yes | no change needed; #58807 not a risk for RedHat |
| A4-F4d | `--disable-custom-all-reduce` would remove a TP2 lever | considered | **kept off** in TP2: the P2P path measured 5.5x slower than NCCL on this VM (E14A); consistent with EMS |
| A4-F4e | TP2 gated against the TP4 oracle | addressed | the margin-flip gate tolerates numeric near-ties |
| A4-F5 | prefix caching switches mamba_cache_mode (align / none); Model Runner V2 is the default | info | unchanged; the 40k-band prefix recompute of the no-prefix arms is noted in the report |
| A4-F6 | the vision tower is loaded in every arm (`--language-model-only` exists) | info | unchanged: equal across arms; a possible later arm |
