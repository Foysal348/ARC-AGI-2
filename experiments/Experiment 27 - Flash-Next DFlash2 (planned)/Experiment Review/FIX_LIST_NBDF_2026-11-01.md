# E27 NBDF fix list (2026-11-01), from 3 GPT analyses + 3 GPT audits + 8 agent reports + kits v1 / v2 + upstream issues (re-verified in full)

**Notebook audited:** sha `3e2df689…`. All three audits were NO-GO, and every claim below was verified in the code.

**Owners:**
- **NBDF:** agent acfe40bd525f7510b (`build_e27_nbdf.py`, `nbdf_lib.py`, `nbdf_cell7_body.py`, kits);
- **SEAM:** agent aefbfc6004f4dc34f (`dflash_seam.py`).

| # | Source | Problem (file:line) | Fix | Owner | Status |
|---|---|---|---|---|---|
| F1 | kit v1 | R4: RecursionError for all 5 arms. The export ran the CELL 7 body twice, so `merlin_fn_server` wrapped itself. | Export = the notebook's cells; `wrap_once`; T11 tests (7). | NBDF | DONE: kit v2 host_phase / launch / argv all 5 PASS |
| F2 | kit v1 | R9 / R10 / R12: "Failed to infer device type" on the CPU box. | Copy the klee100 kit method: libcuda stub, a fake L4 platform (sm_89, 23,034 MiB), a single-process gloo group. | NBDF | DONE: kit v2 FakeL4 platform OK, argparse for 4 D arms PASS |
| F3 | audits 1-3, B1 | `KV_KIT_GATE.passed=false`, so every D arm is skipped. | Rebuild with `--kv-gate-report` once the real kit passes. Test: embedded `passed is True`, build-facts `gpu_ready` true, and D4 / D3 / D7 / D4b in ARMS. | NBDF | waits for the kit pass |
| F4 | audit 1, B2 | A low D4 AL set `stop_d` and skipped D3 / D7 (`nbdf_cell7_body.py:1111-1112`, `nbdf_lib.py:1069`, ratio 1.0). | The D4 exit is D4-only and based on throughput: if both 8k windows are below Q by more than max(A/A, 2 %), skip only D4's long windows. AL is diagnostic only. D3 always runs; D7 runs if budget remains. Fixture tests. | NBDF | sent |
| F5 | audits 1-3, B3 | The gate opens on 2 booleans (`build_e27_nbdf.py:199`). | Require all 17 real-kit checks. `drafter_loaded_weights_ok` may be `CPU_LIMIT` only when explicitly recorded, and then the GPU `draft_load_check` read-back is a hard smoke gate. Provenance binding: notebook / seam / nb4_ppack / drafter / Merlin shas, Merlin commit ad42e37, tp 2, pp 2, partition [25, 23], and a per-K KV summary for every D arm that will run. A missing field closes the gate. Negative tests. | NBDF | sent |
| F6 | audits 2-3, HIGH-3 | `memory_gate` returns None on missing data and True for 2 ranks; boot stops only on False; `arm_verdict` ignores it (`nbdf_lib.py:1116-1122`, `nbdf_cell7_body.py:670-697`). | Require 4 finite readings. A low rank gives False and steps the pin down. Missing data: retry once after 5 s, then None (no step-down). WIN / SHORT_ONLY_WIN need both memory gates True, otherwise `INCONCLUSIVE_MEMORY_GATE`. Tests on 5 inputs. | NBDF | sent |
| F7 | vLLM #63587 (audit 3) | On 0.32, DFlash acceptance is about 0 under FULL graphs: the in-graph draft metadata path (#60692) skips the kernel block-table mapping. Merlin has the same code (`dflash/speculator.py:495-507`). | Port PR #64059 (8 lines: `map_to_kernel_block_table` in the FULL branch) as a seam hunk under `DFLASH_SEAM=1`. Fallback env `DFLASH_NO_INGRAPH_META=1`; `ingraph_meta` read-back; one `INGRAPH_META_FALLBACK` relaunch if the arm hits `DFLASH_ZERO_ACCEPT`. | SEAM, then NBDF re-embeds | SEAM DONE: seam `2df1da39…`, 555/555 tests; 7th file `dflash/speculator.py` (pre `b731b09c…`, post `85f57f0e…`). NBDF re-embed sent |
| F8 | vLLM #53786 (audit 3) | The drafter's SWA group can disable fine-grained prefix hits (Merlin `kv_cache_utils.py:779-790`), and an EAGLE hit can lose a whole block. | Read back the warning line for each arm, plus the prefix-hit rate per window and per turn. `PREFIX_REUSE_DROP` flag (recorded) when D is more than 10 points below Q. | NBDF | sent |
| F9 | vLLM #58638 (audit 2) | Hybrid + DFlash can multiply the KV groups, adding per-step metadata overhead. | Kit R9 and GPU read-back: group count, layout, the drafter's groups, metadata builders. `KV_GROUP_EXPLOSION` flag (recorded). K7's summary is missing → only D7 is skipped. | NBDF | sent |
| F10 | vLLM #47930 / #58894 (audits 1, 3) | Prefix-hit acceptance collapse. | Prefix test widened to 2 prompts × 3 turns; warm-turn acceptance ≥ 0.8 × cold on each prompt; hits recorded per turn. | NBDF | sent |
| F11 | vLLM #61072 | A mismatched pairing gives silent 0 % acceptance. | Zero-accept canary (< 2 % after 200 drafts → `DFLASH_ZERO_ACCEPT`). | NBDF | DONE (in build) |
| F12 | vLLM #54690 | Draft-only FP8 KV init failure on sm_89. | D4b stays optional and runs last. | NBDF | DONE (order) |
| F13 | vLLM #54691 | Long-context draft-attention slowdown. | 90k / 116k windows are required for WIN. | NBDF | DONE (in build) |
| F14 | kit v2 | R9: `get_kv_cache_configs` raises "KV cache layout has not been resolved yet" (`kv_cache_utils.py:1909`, `config/cache.py:375`). | Call Merlin's own `resolve_kv_cache_layout` first, as the engine core does. Q must calibrate to 213-217 blocks. | NBDF | sent |
| F15 | kit v2 | R12: `assert speculative_config is not None` (`qwen3_dflash.py:966`). | Build the draft VllmConfig the way `load_dflash_model` does (target config + `SpeculativeConfig` dflash). If a CPU limit remains, record `CPU_LIMIT` (F5). | NBDF | sent |
| M1 | GPT analysis 3 (accepted, never sent) | SGLang release DFlash2 fixes (grouped conv, NaN in the selector, hybrid PP recurrent state) were not diffed against Merlin. | Item-by-item diff against Merlin's dflash2 code plus post-0.32 vLLM dflash2 PRs. Port correctness fixes that apply as seam hunks; list the rest with reasons. | SEAM | DONE: seam `7356d473…`, 591/591. Ported the grouped-conv A/B (`DFLASH_CONV_IMPL=torch`, 8th file `qwen3_dflash2.py`). The selector NaN (SGLang #41737) and the hybrid state (vLLM #55601) are already in Merlin or do not apply. Report: `E27_REPORT_seam_M1_sglang_vllm_diff_2026-11-01.md` |
| M2 | verify B §4; GPT analysis 2 Bug D (vLLM #57755) | Q-defer packing at K ≠ 3 is untested: no unit test covers `nb4_pack_views` at [n,4] / [n,7]. | Pack / unpack round-trip unit test at K 3 / 4 / 7. GPU: per-rank Q-defer fallback share for each D arm; `QDEFER_K_FALLBACK` flag. | NBDF | sent |
| M3 | vLLM #57032 / #63587 log; verify A U4 | Merlin warns "no KV cache group could be identified as the draft model's" (`kv_cache_utils.py:2122-2151`) when the drafter group is not an eagle group. | Read back the line in each D arm and record it with the F9 KV-group summary. | NBDF | sent |
| M4 | GPT analysis 1 | Draft-context correctness after preemption / recompute was not measured. | Per D window: preemptions; if above 0, acceptance before / after the first preemption plus the long-oracle token-ID check. `PREEMPT_ACCEPT_DROP` flag (< 0.8×). | NBDF | sent |
| M11 | seam open risk 2 | The aux relay under `@support_torch_compile` on PP0 is new in Merlin. | Target cudagraph mode and capture count per stage, Q against D. `PP0_GRAPH_DEGRADED` flag. | NBDF | sent |
| M12 | seam M1 (vLLM #58250) | Merlin's fused grouped conv (#55960) lowered DFlash AL 3.195 → 3.060 in the vLLM nightly. | Optional arm D4t = D4 + `DFLASH_CONV_IMPL=torch` (smoke, greedy oracle, 8k ×2), after D3, budget-gated. `CONV_IMPL_TORCH_BETTER` flag. | NBDF | sent |
| M13 | seam M1 | `DFLASH_DRAFT_SW_BLOCK` lowers `cache_config.block_size` engine-wide (`core.py:436-448`); its other readers are untested. | Unset in every GPU arm, enforced by the launch check; only the kit R9 probe uses it. | NBDF | sent |

## Not adopted as asked (with reason)
- **"KV report for K3, K4 and K7 all mandatory":** only the K values of the D arms that will run are mandatory. A missing K7 skips only D7.
- **"Missing telemetry = False":** False steps the pin down, which would waste capacity. Missing telemetry is None, which blocks WIN instead.
- **"Controlled stop when telemetry is missing":** the arm still runs (Q-defer / DFlash read-backs are still useful), but it cannot be WIN.

- **"Q-defer OFF first"** (GPT analyses 1 and 2, and the model-choice note): all arms share Q-defer, so Q and D are matched. The first failing D arm reruns once with Q-defer off (`QDEFER_INTERACTION` / `DFLASH_ITSELF`), and M2 adds the K ≠ 3 packing test. GPT analysis 3 itself recommended "same Q-defer".
- **"First captures [1,2,4], then [1,2,4,8]"** (GPT analysis 1): exact multiples of 1+K up to 8×(1+K) are used instead. Smaller lists would push most decode batches into an eager drafter (verify A B8: the drafter has no PIECEWISE).
- **"draft_tensor_parallel_size 2, draft_sample_method greedy"** (GPT analysis 2): unset draft TP already resolves to the target TP 2 (kit R10: `draft_tp_unset_is_target_tp` true). Probabilistic draft sampling is supported by DFlash2 with fp32 draft logits (`dflash2/speculator.py:200-204`, verify B). The greedy oracle covers greedy correctness.
- **"Tokenizer files / embedding sidecar folder in the drafter Model"** (GPT analyses 1 and 2): not needed. The target tokenizer is used (`speculative.py:1319-1326`), and the embedding is inside `model.safetensors` (`bf16-embed`, verified on Kaggle).
- **"Ramp 1 → 2 → 4 → 8 → 9 / 16 → 21 as separate runs"** (GPT analysis 3): this is the in-smoke ladder (probe alone, then with 2 / 4 / 7 others); W4 `max_num_seqs` is 8.
- **Bug A, vLLM #55600** (GPT analysis 2): already fixed in Merlin (`mamba_hybrid.py:159-165` uses `mamba_block_size`).

## Later, not in this notebook
- **Tap-mode A/B** (`DFLASH_TAP_MODE=pre_combine|norm_input`): only if D acceptance is clearly below MTP and not zero (verify B risk 2, seam risk 3).
- **klee100 port** (step 6): measure DFlash2 acceptance on klee100 first. Its rank-1 agreement with BF16 is 0.51-0.65 (verify B §4).
- **W4 production with 9 workers:** captures up to 9×(1+K) (verify A B8).

## Order to GPU-ready
1. SEAM: F7 DONE (seam `2df1da39…`); M1 in progress.
2. NBDF: F4-F10, F14, F15, M2-M4, M11, then re-embed the seam, units, rebuild.
3. Kit v3 on Kaggle (real + units). It must pass all 17 checks, plus the R9 Q-calibration (213-217 blocks) and the per-K KV summaries.
4. Gated rebuild (`--kv-gate-report`), F3 asserts, a refreshed audit zip.
5. GPU push: needs the user's go. Order Q → D4 → D3 (→ D7 if budget remains), 140 min.
