# E27 NBDF agent (acfe40bd525f7510b): final report, verbatim

## NBDF: built, not pushed

The notebook builds at **386,896 B** (under 400 kB). Units tests: **116/116 PASS** locally. The units kit, run once locally, prints `@@KIT units ALL OK@@`; that run included the seam agent's tests (all pass). Nothing was pushed to Kaggle and no git was run. All files are LF.

**The GPU run is not ready yet.** The build is `gpu_ready: false`: without a passing real-kit KV report, every D arm is `SKIPPED_KV_KIT_GATE` at run time and only Q runs. To make it GPU-ready:
1. Run the real kit.
2. If its `kv_cache_configs_ok` and `seam_apply_verify_revert_ok` checks pass, rebuild with `python build_e27_nbdf.py --kv-gate-report <nbdf_kit_real_report.json>`.

### Files (Build folder unless stated)
| File | Lines | sha256 |
|---|---|---|
| `build_e27_nbdf.py` | 1538 | `11bfe398dbfb094468837ece66b9b377ea4970b74821d237787cdd35ea92e62d` |
| `nbdf_lib.py` (v2, rewritten) | 1483 | `cc92dbfbef485f2325e8f7b4d0dc8922714c062bc38fc62fcf569603dcbccf29` |
| `nbdf_unit_tests.py` (new) | 416 | `588ecb656471140e0a276f24b7a96be36b8d2b03f0cdf7c41dbe392c2a72f560` |
| `build_e27_nbdf_cpu_kit.py` (new) | 536 | `1d72f3b0e67b9b9fde08ba8a8733c895c3ec65fb9ba235aedf6ae8a62ae78458` |
| `nbdf_code_export.py` (readable code) | 3416 | `8e64f992d572c3ba7d0e42760399fb1296986b6cff847aeb509935071ffa171d` |
| `nbdf_build_facts.json` | 343 | `859a114a30b279e816a559426d4b060b81dfa61675db035bdf6f461323d8b6a8` |
| `../Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` (10 cells) | — | `3e2df6891995432da9a156c91a83f063407060b94c846fcba16dc63b7b329a84` |

- **Embedded seam** (not edited): `dflash_seam.py` `c4ea1ecc287036bcadbc98cd33e40309d17f2c58b471e5c876ea7acddba51f78`.
- **Byte-identity check:** the 9 NB4T / E25 files the builder reads were unchanged during the build.
- **CELL 7 body:** sha `78104cc1…`.
- **Backups of the previous agent's files:** `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\nbdf_lib_v1_prev_agent.py` and `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\build_e27_nbdf_v1_prev_agent.py`.

### Changes from the previous agent's draft
- **Seam interface was wrong:** the draft expected `apply` / `verify` / `revert` and events (`aux_relay`) that the seam does not have. It is now wired to `apply_dflash_patch` / `verify_dflash_patch` / `revert_dflash_patch(ring=…)`, `combined_tree_state`, `server_env(mask_row_sha=…)`, `readback_label`, and the real `E27_DFLASH` event fields.
- **Order, captures and spec now match your brief and the coordinator's later messages:**
  - order is Q → D4 → D3 → D7 → D4b;
  - D-arm captures are exact multiples of 1+K;
  - the spec adds `attention_backend` FLASH_ATTN.
- **Size:** to stay under 400 kB, the embedded modules are lzma + base85, and the CELL 7 body is a compressed blob. The CELL 7 stub checks its sha and execs it into the notebook globals.

### Cells and base
- **CELL 0:** E18 R2 cell 1, verbatim.
- **CELLs 1-6:** NB4T verbatim, plus marked edits: 140-min budget, `OUT` = nbdf, `DFLASH_` env keys recorded.
- **Reused from NB4T:** the RC2 W4 boot, FP8 KV, the Q pin 3,553,580,524, `--language-model-only`, #60490 and nb4_ppack Q-defer.

### Arm flags, with the Merlin source behind them (ad42e37)
**Q (control):** Merlin's own spec, `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}` (`launch.py:441`, `registry.py:49`). Captures are Merlin's [1..32] (`launch.py:459-460`, `registry.py:83-89`). No DFLASH env.

**D4 / D3 / D7:** Q plus one change: the seam env plus a trailing `--speculative-config` and `--compilation-config`. Extra args come last, so the last occurrence wins (`launch.py:110-111`).
- Spec: `{"method":"dflash","model":<resolved dir>,"num_speculative_tokens":K,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"}`.
- Captures: D4 5..40, D3 4..32, D7 8..64.
- Env: `DFLASH_SEAM=1` and `DFLASH_MASK_ROW_SHA=8606…1a19`. D7 alone adds `DFLASH_QSA_RING_WIDEN=1`. No sidecar; host embed is off.

**D4b:** D4 with `kv_cache_dtype` fp8_e4m3 and TRITON_ATTN. Optional, runs last.

**Source references for the spec fields and engine behaviour:**
- `speculative.py`:
  - `:395` / `:398` / `:401` — K, model, method (method is explicit because `:1357-1373` detects the path);
  - `:409` — draft TP left unset, so it takes the target TP (`:1579-1583`, `:1744-1773`);
  - `:426` / `:430` — backend and KV dtype (overrides at `:380-385`);
  - `:602` — `draft_sample_method` probabilistic, which DFlash2 supports with fp32 draft logits (`speculator.py:246-258`, `dflash2/speculator.py:200-204`).
- `dflash/speculator.py:126-153` (graphs) and `:170` (progress bar).
- `cuda.py:446`: "Using X backend." is logged only for the explicitly selected (drafter) backend.
- `cudagraph_utils.py:332-357`: capture sizes are rounded up to multiples of 1+K.
- `qsa_cache.py:960-965`: QSA ring capacity is 8 at K3/K4 and 12→16 at K7 with the widening.

### What runs per arm, in order
1. **Three-state tree check** before every boot (nb4_ppack and the seam).
2. **Pins and memory gate:**
   - D arms start at Q's pin minus 1-2 rungs: 3,285,145,068 / 3,016,709,612. The plan accounts for the drafter (0.727 GiB), the embed (0.592 GiB on the GPU), fp32 `draft_logits`, a 512 MiB margin, MTP freed, and Q's PP1 free memory.
   - A launch OOM, or any rank under 512 MiB free after ready or after the first decode, steps one rung lower (up to 2 more). Later D arms start at the pin that worked.
3. **Read-backs:**
   - env, config, and Q-defer exec / fallback share per rank;
   - DFlash checks, combined with the seam's `readback_label`: DFlash2DraftModel resolved, no MTP; drafter graphs captured and not eager; "Using FLASH_ATTN" on both PP1 workers; aux taps on 4 ranks, local [6,16 | 26,36,46], `slot_base` 2, `upstream_total` 2, tap mode combine; embed own, head shared, mask-row sha matches, B5 `lm_head` released bytes > 0; `draft_load_check` × 2; the `qsa_ring` line at K7; SpecDecoding lines with K positions;
   - KV layout from `nb1_cache_info`: block sizes, ring capacity, drafter KV dtype;
   - graph counts;
   - memory per stage plus the memory log lines;
   - pool / F / max concurrency vs Q; a pool drop over 10 % is noted in the verdict.
4. **Smoke:**
   - 2 streams × 2,048 tokens;
   - zero-accept canary: under 2 % acceptance after at least 200 drafts gives `DFLASH_ZERO_ACCEPT`;
   - windowed garbage check;
   - concurrency ladder: a greedy probe alone, then alongside 2, 4 and 7 others. A mismatch is `CONCURRENCY_CONTAMINATION`, or NEAR_TIE when the margin is low.
5. **Greedy oracle vs Q by token ids,** with its acceptance recorded. At D4, if greedy AL ≤ Q's, the arm exits early and D3, D7 and D4b are skipped.
6. **Long oracle** at 90k and 116k.
7. **Prefix cold / warm two-turn test:** if warm AL < 0.8 × cold, the arm gets `PREFIX_HIT_ACCEPTANCE_COLLAPSE`.
8. **Windows:** 8k@8 × 2, then 40k / 90k / 116k at the resident level from this boot's pool and F. Q runs 90k and 116k twice for A/A noise. Each window records the prefix-hit rate and the parity of its micro-batches.
9. **Profiles** at 8k, and at 116k if time remains, with graph spans per rank to separate drafter time from target time.

### Responses, garbage gate and verdict
- **Saved responses:** every response (text and token ids) goes to `responses/responses_<launch>.jsonl.gz`, plus NB4T's `requests.jsonl`.
- **Garbage gate:** the windowed 2,000-token check gates. The stream-level LOW_VARIETY check is recorded only, and a window it flagged is restored when the windowed check is clean.
- **Q-defer isolation:** the first failing D arm reruns once with Q-defer off, labelled `QDEFER_INTERACTION` or `DFLASH_ITSELF`.
- **Verdict:**
  - WIN needs: 8k PASS on both windows, a valid 40k window, no regression at 90k or 116k beyond max(Q A/A, 2 %), and long oracle and prefix both PASS;
  - otherwise SHORT_ONLY_WIN, NO_GAIN, REGRESSION, or the named failure.
- **Budget:** at 140 min, Q (45 min) + D4 (38) + D3 (37) fit; D7 and D4b normally fall to SKIPPED_BUDGET.

### Tests and kits
- **`nbdf_unit_tests.py`, 116 checks**, including:
  - the plan and captures;
  - the launch check;
  - the pin ladder and memory gate;
  - read-back parsers on synthetic logs in the seam's real event format, plus the seam's own `readback_label` returning ENGAGED;
  - the KV layout on the real NB4T Q record: 215 blocks, F 7, ring 8, residency 6 at 90k / 5 at 116k;
  - the windowed gate on a 298k-character digit-heavy healthy ARC text: not flagged, while the stream-level check flags it;
  - the smoke / ladder / prefix / early-exit / isolation gates;
  - the drafter pins;
  - SeamSeq with the real seam and nb4_ppack on copies of the Merlin files;
  - the verdict and the budget walk.
- **Kit push folders**, under `D:/ARC AGI 2/_scratch/push_e25/`. The CPU kits are private, CPU-only, and on the pinned image.

| Folder | Size | sha256 | Attached Models |
|---|---|---|---|
| `q27-e27-nbdf-kit-units__foysalemonshanto` | 290,894 B | `8c12faf1…` | none |
| `q27-e27-nbdf-kit-real__foysalemonshanto` | 266,959 B | `3c063109…` | all 8, including the drafter |
| `q27-xhigh-e27-nbdf__foysalemonshanto` (GPU) | — | — | the 7 RC2 Models + `qwen38-flash-next-dflash2v2/transformers/bf16-embed/1` |

- **The real kit gates on:**
  - the CELL 0 identity submission, inputs, merge and sha;
  - **the drafter pins on the real mount;**
  - the Merlin install and the flag-source shas;
  - #60490;
  - seam apply / verify / revert with ring and a valid combined state;
  - host_phase, launch check, spawned-child env and tree check for all 5 arms;
  - the argv fields;
  - the spec-config preflight (field names, Literals, backend enum, draft TP default);
  - **`kv_cache_configs_ok`;**
  - **`drafter_loaded_weights_ok`.**
- **Recorded only in the real kit:** the patched-model import probe and the argparse result.
- No probe script is named after a stdlib module.

### Audit zip
`D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip`
- 679,857 B, sha `6417137e…`, 25 files plus a sha manifest.
- `00_GPT_PROMPT.md` asks for GO / NO-GO first, then blockers only.

### Open dependencies and risks
1. **The real kit's KV probe is untested.** No vllm here, so it could not run locally; a first Kaggle run will likely need a fix pass.
   - It builds per-stage specs from the NB4T GPU record at K3 state shapes and adds 5 drafter sliding-window layers on PP1 (BF16, plus a block-1024 variant).
   - It checks itself against Q's 215 blocks first.
   - K4 / K7 state growth (U2) is not modelled there; the GPU read-back measures it.
   - Until it passes, all D arms are skipped.
2. **Two real-kit probes may not run on a CPU box without `vllm._C`:** the drafter loaded-weights probe (which instantiates `DFlash2Qwen3ForCausalLM` on meta) and the patched-model import. If so, the loaded-weights gate shows NOT OK and the GPU boot becomes the proof. That result also decides whether `DFLASH_LOAD_CHECK=0` is needed.
3. **The drafter Model's pinned shas are only verifiable on Kaggle.** Inside the notebook a mismatch gives `DRAFTER_SHA_MISMATCH`, which skips the D arms only; Q and the submission path still run.
4. **Seam report vs file:** the seam report's post-sha table differs from the shas on disk for `nvidia/model.py` (`7871…` on disk) and `dflash/utils.py` (`fb76…` on disk). I used the file's constants. Your stated sha matches the seam file itself.
5. **Seam tests sha differs:** `dflash_seam_tests.py` on disk is `452bb231…`, not the `1fdde1e6…` you gave. It was changing while I worked. The units kit bundles the on-disk version, which passes.
6. **Placeholder ops you may want to change:**
   - the ladder uses 7 others (7 + probe = `max_num_seqs` 8) instead of 8;
   - the AL early exit at K=4 uses the greedy oracle acceptance;
   - the drafter host embedding is off, as you asked.

