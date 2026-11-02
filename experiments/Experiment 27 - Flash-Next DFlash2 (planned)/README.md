# Experiment 27: DFlash2 speculative decoding for Qwen3.8-Flash-Next on Merlin v3.0.0 (4×L4)

**Status:** building (2026-11-01). This README is the living plan. It is updated after every GPT analysis and every agent or GPU result. The change log is at the bottom.

## Goal
Replace MTP K3 with the DFlash2 drafter (`incoai/Qwen3.8-Flash-Next-DFlash2`) in our two Flash-Next notebooks, and measure whether it is faster at equal correctness, up to 120k context.

**The two target notebooks:**
- **W4:** NB6Q = NB6 SUBMIT + Q-defer (Exp 25).
- **klee100:** K100Q, the low-bit accuracy test (Exp 26), and the klee100 submission notebook that follows it.

**Today's baselines:**
- **W4 + Q-defer:** synthetic 8k@8 302.9 tok/s, 40k@8 287.4 (NB4T Q). Duck MTP AL 2.83, positions 0.795 / 0.599 / 0.439.
- **klee100 (MTP K3, no Q-defer):** 460 tok/s at 20k@16, 562 tok/s at 20k@24.

**Expected:** central estimates vs MTP K3 are ×1.24 (K=3), ×1.34 (K=4) and ×1.44 (K=7). All published numbers are at contexts of 4k or less, so the gain at long context is unknown.

## Ground rules (standing user rules that apply here)
- **Long context:** every speed and oracle test goes up to 120k context: 8k, 40k, 90k and about 120k. 8k / 40k alone never decide a winner.
- **Saved responses:** every GPU run saves the complete response of every request.
- **Notebook format:** every notebook carries the proven eval + private format (CELL 0 = E18 R2) and stays under 400 kB. Payloads go in sha-pinned datasets or Models.
- **Patch style:** patches are applied like the Q-defer patch: pre / post sha, atomic writes, py_compile, fail-safe restore. They never touch the files Q-defer patches (`pp_utils.py`, `model_runner.py`, `gpu_worker.py`).
- **Gates:** CPU kits on Kaggle come before any GPU run. Audit zips go in `Experiment Review/Audit Zips/`, and agent reports are saved verbatim in `Experiment Review/Agent Reports/`.
- **Pushes:** a GPU push needs the user's go.

## Step-by-step plan

### Step 0: feasibility (DONE)
- **First feasibility report:** feasible on Merlin, days not weeks. Saved as `Experiment Review/DFLASH2_FEASIBILITY_2026-11-01.md` and `Agent Reports/E27_REPORT_feasibility_first_2026-11-01.md`.
- **Verify A, engine:** GO to build, NO-GO for GPU until the CPU kit measures the drafter KV packing (U1) and the PP1 memory (B9). Blockers B1-B9, unknowns U1-U5. See `Agent Reports/E27_REPORT_verify_A_engine_2026-11-01.md`.
- **Verify B, drafter and gain:**
  - the drafter tensors match Merlin's loader exactly;
  - the embedding must be supplied;
  - `mask_token_id` 248077 is the first unused padding row;
  - MTP K3 drafting measured at 9.8-13.4 ms per micro-batch on stage 1, of which 3 lm_head reads take 7.5 ms. The DFlash2 draft is estimated at about 5.5-6.6 ms.
  
  See `Agent Reports/E27_REPORT_verify_B_drafter_gain_2026-11-01.md`.

### Step 1: drafter Kaggle Model (DONE 2026-11-01)
- **Result:** kernel `foysalemonshanto/q27-e27-dflash2-embed-repack` v364422784, REPACK OK in 95 s.
  - **Embeddings:** the W4 (`model-00015`) and klee100 (`model-00002`) embeddings are byte-identical. tensor sha256 `62288a3f…c9be3a`; row-248077 sha `860615…1a19`, norm 0.2627; zero rows 0. **One variation, `bf16-embed`, serves both notebooks.**
  - **`bf16-embed/model.safetensors`:** 2,477,538,608 B, sha256 `d2e0464d15936265bb75eda9d09149d27ac6986327ca269a0335cd3d49661d49`.
  - **`config.json`:** 1,236 B, sha256 `24095759…b1b9`.
  - **Drafter source:** HF revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, sha `d6723f2a…3186f`.
- **Model (CREATED by the user 2026-11-01):** `foysalemonshanto/qwen38-flash-next-dflash2v2`, Transformers, variation `bf16-embed`, v1, private.
  - **Files:** `bf16-embed/{config.json 1236, dflash2_manifest.json 1029, model.safetensors 2477538608}` + `repack_report.json` (sizes match the repack).
  - **Drafter dir:** `/kaggle/input/models/foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1/bf16-embed/`.
  - **The shas are verified on Kaggle, in the kit.**
- **The `bf16` variation (no embed) is not needed.** `bf16-embed` is a superset of it, so the stalled local upload of the plain variation is dropped.
- **NBDF:** the path is changed to `bf16-embed`, and the shas above are pinned in the drafter gate.

#### Original step-1 plan
- **Source:** incoai `model.safetensors`, 1,206,140,008 B: 81 BF16 tensors, about 603M params, no `embed_tokens`, no `lm_head`.
- **Repack:** append `embed_tokens.weight` [248320, 2560] BF16, byte-copied from the served target checkpoint:
  - W4: `model.language_model.embed_tokens.weight` in `model-00015`;
  - klee100: in `model-00002`.
- **How:** a Kaggle CPU kernel extracts the embedding, asserts BF16 and the shape, and records the full-tensor sha and the row-248077 sha.
- **Model:** `foysalemonshanto/qwen38-flash-next-dflash2/transformers/<variation>/1`. Variations: `bf16` (original), `bf16-embed` (one variation if W4 and klee100 embeddings are identical; otherwise `-w4` and `-k100`).
- **Layout:** `config.json` + `model.safetensors` + a manifest. No tokenizer, no zip, no `mask_embedding.pt`.

### Step 2: Merlin patch module `Build/dflash_seam.py` (IN PROGRESS)
Each hunk carries pre / post shas. Everything is gated by `DFLASH_SEAM=1` where possible; with the env off, behaviour is byte-identical.
1. **Allowlist:** add `"dflash"` at `model_executor/models/config.py:1049-1058`.
2. **Seam:**
   - `Qwen4ExpModel` gets EagleModelMixin.
   - Both `Qwen4ExpForCausalLM` and `Qwen4ExpForConditionalGeneration` get SupportsEagle3.
   - Taps are the completed layer outputs `mlp_hyper_connection.combine(...)`, post-combine and pre-norm, averaged over the 4 HC streams. Layer ids (6, 16, 26, 36, 46).
   - An optional switch `DFLASH_TAP_MODE` allows an A/B against incoai's variant.
3. **PP2 relay:**
   - set `supports_aux_hidden_states_over_pp = True`;
   - PP0 packs taps (6, 16) into `IntermediateTensors`;
   - the last rank collects remote + local, in that order, with slot base 2;
   - template: `deepseek_v4/nvidia/model.py:1287-1288, 1462-1527`.
4. **Drafter embedding:**
   - load it from the repacked drafter;
   - set `has_own_embed_tokens` on the draft class, so a missing embedding raises;
   - add a drafter-only host (UVA) option for W4 (`DFLASH_DRAFT_EMBED_HOST=1`);
   - assert row 248077 equals the target's.
5. **Release the orphaned draft lm_head** (about 636 MB per card) after sharing, as `_release_replaced_draft_head` does.
6. **Loader checks:** the drafter linears must be `UnquantizedLinearMethod`, and every drafter parameter must have received a tensor.
7. **Optional QSA ring widening** for K ≥ 5 (`qsa_cache.py:960-965`, PR #54912), env-gated.
8. **Tests:** apply / verify / revert, env-off identity, combined apply with nb4_ppack in both orders, AST checks, aux shape / order, slot base, and `get_kv_cache_configs` with the drafter group (pool / F / groups).

### Step 3: GPU A/B notebook `Build/build_e27_nbdf.py`, NBDF (IN PROGRESS)
- **Base:** NB4T (Exp 25), with the RC2 Merlin W4 boot and Q-defer.
- **Arms:** Q (MTP K3 + Q-defer) → D4 → D3 → D7 (only if the ring verifies) → D4b (optional: FP8 drafter KV + TRITON_ATTN).
- **Spec config:** `{"method":"dflash","model":<drafter>,"num_speculative_tokens":K,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"}`.
- **Captures:** multiples of K+1, up to max_num_seqs × (K+1).
- **Per arm:**
  - teacher-forced per-position acceptance first, with an early exit if AL ≤ MTP's at K=4;
  - oracle (short, and long at about 90k / 116k);
  - steady windows at 8k × 2, 40k, 90k and about 120k;
  - AL per position, pool / F, memory, prefix hits, backend and graph read-backs, Q-defer ENGAGED;
  - the complete responses saved;
  - the windowed garbage gate.
- **Verdict:** a winner needs a clean oracle, speed PASS at 8k and no regression at 40k / 90k / 120k; otherwise the arm is labelled SHORT_ONLY_WIN.
- **Added after GPT analysis 1:**
  - every D arm starts with a fail-fast smoke (2 streams, short output, small captures);
  - if a D arm fails with Q-defer on, it reruns once with Q-defer off (the isolation arm);
  - peak memory is recorded per stage (load / KV init / capture / first decode);
  - the effective KV block size and the ring capacity are read back;
  - early stop on zero acceptance or severe divergence.
- **Hard stop:** about 130-140 min.

### Step 4: CPU kits on Kaggle (NEXT)
- **units:** the seam tests, the nbdf tests, and the windowed-garbage test.
- **real:** on the real installed Merlin tree:
  - the patches apply, together with nb4_ppack;
  - the config resolves to DFlash2DraftModel, with fc 12800 and aux layers (6, 16, 26, 36, 46);
  - the reserved slot keys exist on rank 1;
  - the drafter embedding loads, and row 248077 equals the target's;
  - **`get_kv_cache_configs` pool / F / groups per K (U1, U2)**;
  - the ring capacity at K=7 is 16;
  - the argv per arm.
- **Gate:** the GPU run happens only if the kit passes and U1 is within budget.

### Step 5: GPU run of NBDF (user's go)
W4 first, because its Q baseline has been measured.

### Step 6: port the winner
- **W4:** NB6Q + DFlash2, the winning K.
- **klee100:** K100Q / the klee100 submission + DFlash2. Measure klee100 acceptance first; K=4 then K=3, no K=7 at ≥ 16 streams.
- Both go through a build, CPU kits, an audit, a commit, then a submit.

## GPT analyses (the user feeds them one by one; the plan is updated after each)
| # | Received | Verdict | Accepted changes | Rejected (why) |
|---|---|---|---|---|
| 1 | 2026-11-01 | PORT FEASIBLE, NOT YET GPU-READY. Blockers: allowlist, HC seam, PP2 relay, drafter embedding, K7 ring, memory fit. All of these are already in steps 2-4. GPT applied the roozich patch cleanly on an isolated Merlin copy (12 edits, idempotent) and confirmed that it lacks the PP2 relay and the embedding fix. | (a) **Q-defer isolation:** if a DFlash arm fails (hang / garbage / NCCL error) with Q-defer on, the same arm reruns once with Q-defer OFF, so the cause is isolated. The production comparison stays with Q-defer on. (b) **Fail-fast smoke per D arm:** 2 streams, short output, small captures, before the 8-stream windows. (c) **Peak memory recorded per stage:** model load, KV init, graph capture, first decode. Explicit KV reservation skips Merlin's memory profiling. (d) **Runtime read-back of the effective KV block size and the QSA ring capacity;** do not assume 4096. (e) **CPU preflight that Merlin's parser accepts the exact speculative config** (incl. `draft_sample_method`); drop that field if DFlash rejects it. (f) **Transactional patch install** with rollback (already the plan). (g) **Stop early on zero drafting acceptance or severe divergence.** | (1) "Test DFlash2 with Q-defer OFF first, as the baseline": not taken as the default. Verify-A found separate communicators (`pp_broadcast` vs the p2p device group) and an unchanged order; Q-defer is our measured base (+21 %). Rule (a) covers the risk. (2) "Tokenizer files in the drafter Model": not needed. Verify-B: `speculative.py:1319-1326` uses the target tokenizer when the vocab is the same. (3) "K3 first": both K3 and K4 run; K4 first (no ring patch, best central estimate), K3 as the control. |
| 2 | 2026-11-01 | CPU port research GO; direct GPU port NO-GO. GPT confirmed the same blockers (allowlist, seam, PP2 relay, PP1 embedding, K7 ring, memory) and ran the roozich patch on a copy (exit 0, compiles, no PP2 relay). It also raised upstream bug classes. | (a) **Prefix-cache cold vs warm acceptance test (HIGH, Duck is multi-turn).** Upstream #47930 / #58894 report draft acceptance collapsing to 0 after prefix-cache hits, and Merlin has no visible handling of unwritten draft-KV blocks (grep of `kv_cache_utils` / `kv_cache_manager` / `dflash/`). NBDF must run the same multi-turn prompt cold and then warm (prefix hit), compare AL per position and the outputs, and gate on it. If it collapses, the fix direction is to exclude unwritten draft blocks from reuse (PR #47926 as reference). (b) **Drafter KV BF16 in the first port** (upstream #54690: an SM89 draft-only FP8 KV crash). This was already the plan (B6). (c) **Effective drafter attention context measured at runtime.** Upstream #54691 saw long-context DFlash slowdowns on hybrid GDN; our drafter has `sliding_window` 2048, so the cost should be bounded. Verify it with the drafter tick cost at 40k / 90k / 120k. (d) **Three-state tree check before boot (ORIGINAL / FULLY_PATCHED / INVALID_MIXED)** for both patches (nb4_ppack + dflash_seam); INVALID_MIXED aborts in commit / measurement notebooks. (e) **CPU-kit checks of `draft_sample_method` (probabilistic vs greedy) and `draft_tensor_parallel_size` parsing for dflash;** use what Merlin's DFlash2 speculator supports. (f) **Final-tensor checks:** 5 taps, each [T, 2560], tap order [6, 16 \| 26, 36, 46], consistent on all 4 ranks (raise `DFLASH_AUX_COUNT_MISMATCH` / `_WIDTH_MISMATCH`). | (1) "Bug A: mamba_hybrid `block_size` vs `mamba_block_size` (#55600)": already fixed in Merlin. `v1/worker/gpu/model_states/mamba_hybrid.py:159-165` uses `cache_config.mamba_block_size`. (2) "Q-defer OFF for the first boot": same as analysis 1; covered by the isolation rerun. (3) "K3 first": K4 then K3, both run. (4) The `embedding/` subfolder layout: the Merlin loader reads `embed_tokens.weight` from `model.safetensors` (verify-B), so the repack goes into `model.safetensors`. |
| 3 | 2026-11-01 | Existing MTP K3 notebooks GO; DFlash2 port NO-GO until the blockers are fixed and GPU-tested. Same blockers as analyses 1-2 (PP2 relay, embedding ownership, memory). It also recommends keeping Q-defer ON in the A/B, which matches our plan. | (a) **Memory gate:** at least 512 MiB free on every rank after drafter load, KV init, graph capture and the first decode. If a D arm misses it, it steps down to a lower KV pin rung, recorded. On W4 the D arms start at a reduced pin, because W4 has only about 191 MiB free on PP1. (b) **Numerical tap test (CPU):** compare the seam's HC-averaged tap against a pure-torch reference of `combine(...).unflatten(...).mean(-2)` on random inputs, within tolerance, for SP off (and on, if constructible). Also check finite values and IDs. (c) **Token-ID equivalence:** the greedy oracle compares generated token IDs (not text) between Q and each D arm under the same seed / cache state. (d) **Concurrency contamination ladder** in each D arm's smoke: the same greedy prompt run alone, then among 2 / 4 / 8 concurrent requests, must give identical token IDs. Per-request ids, output hashes, drafted / accepted counts and prefix-cache state are saved. This is upstream SGLang #36548 / #43514 class evidence. (e) **The QSA ring widening ports the upstream helper from PR #54912** (with its informative error), not a hard-coded 16. (f) **Paired repeated windows;** no winner within run-to-run noise (already: A/A spread threshold). | (1) "1 worker first boot, scale 2→4→9 / 2→4→8→16→21 as separate runs": folded into the per-arm smoke and the concurrency ladder (d) inside one notebook, which costs less GPU. (2) "Tokenizer compatibility info in the Model": not needed (same vocab; the target tokenizer is used). (3) "K7 later": D7 stays optional, last in the order, and runs only if the ring helper verifies. |

## Change log
- **2026-11-01: GPU commit of `8b1760ed` (foysal, 69 min).**
  - Q was clean: 8k@8 287.5 / 305.2, 40k@8 294.6, 90k@6 236.0 / 244.8, 116k@5 205.3 / 216.3 tok/s, 0 preemptions, long oracle PASS.
  - But every D arm was SKIPPED_DEPENDENCY. The cause was a base gate on the Q-defer read-back label (the NB4S rule exec_share ≥ 0.9): flush_empty 3000, fallback_share 0.0, so no real failure. That is a measurement label that should never gate.
  - Fix: real-failure-only gates (`base_usable`, `qdefer_real_failure`); 10 label gates demoted to record-only.
  - **USER DECISION: pivot to NBDF-K100** (the klee100 target: PP1 ~719 MiB free, pool 479, 385 tok/s real Duck). Arms Q / D4s / D3s, 150 min. The W4 NBDF re-run is dropped.
- **2026-11-01: FINAL gated build after the 3 final GPT audits** (HIGH-1 F validation plus mixed-block F, the admission watchdog vLLM #55800, D4s probes at 16k / 40k). Kit v6 on `461187fa` is ALL OK (real on foysal, units on hamid). **Gated notebook `8b1760edc70478b60b341b7a127a5ae5b79672919f492b120f3896ed2b284c0b`, 396,872 B, `gpu_ready` true.** The GPU push folder has the same file. GPU needs the user's go.
- **2026-11-01: Kit v5 on `61abc8aa` is ALL OK.** R9: Q 215, K3/K4 SW-1024 205, BF16 4096 71, FP8 143. R12 PASS (67 = 67). **Gated rebuild done: `Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` sha `ffa605e8aa037fbb148692f5b4975402b3fa90c9741592240963a8189cadbd48`, 393,767 B, `gpu_ready` true**, provenance closed_rebuild. The GPU push folder `push_e25/q27-xhigh-e27-nbdf__foysalemonshanto` holds this notebook. **GPU push needs the user's go.** Arms: Q → D4s → D3s (→ D4 / D4t if time remains), 150-min hard stop, deadman at 155.
- **2026-11-01: Kit v4 on notebook `487a4dc3` is ALL OK** (18 / 18). R9 calibrated Q to 215 blocks exactly, and measured the drafter KV packing (U1):
  - BF16 drafter KV at block 4096: **71 blocks** (page 41.9 MB sets the stride), max concurrency at 131k 1.84;
  - FP8 drafter KV: 143 blocks;
  - **drafter SW block 1024: 205 blocks** (page 10.5 MB, stride unchanged), max concurrency 4.86, all at the D pin 3.02 GB.
  - **Decision:** the primary DFlash arms become D4s / D3s with `DFLASH_DRAFT_SW_BLOCK=1024`. This reverses M13 for these arms, guarded by the oracle, long oracle, ladder and prefix gates and the `cache_config.block_size` read-back. Plain D4 stays as the 71-block control (short windows only). Kit v5 is needed.
- **2026-11-01: Second independent GPT audit of NBDF: NO-GO.** It agrees on B1 / B3. New items:
  - **HIGH-3:** `memory_gate` gave None on missing telemetry and True for 2 ranks, and the verdict did not require it (`nbdf_lib.py:1116`, `nbdf_cell7_body.py:670-697`). Fix: 4 finite readings required, and WIN needs both memory gates OK, otherwise `INCONCLUSIVE_MEMORY_GATE`.
  - **vLLM #58638** (KV-group explosion with hybrid + DFlash): the kit and the GPU read-back record the KV group count and layout per K, and flag `KV_GROUP_EXPLOSION`.
- **2026-11-01: GPT audit of NBDF: NO-GO.** All three blockers were verified in the code.
  - **B1:** `KV_KIT_GATE` false makes every D arm skip. This is expected; it closes with the kit v2 pass and the gated rebuild.
  - **B2:** a low D4 AL set `stop_d` and skipped D3 / D7 (`nbdf_cell7_body.py:1111`). Fix: the D4 exit becomes D4-only and is based on 8k throughput; AL is diagnostic only.
  - **B3:** the gate opened on 2 booleans only (`build_e27_nbdf.py:199`). Fix:
    - every real-kit check is required, plus provenance binding (notebook / seam / ppack / drafter / Merlin shas, per-K KV);
    - R12 may be `CPU_LIMIT` only when it is recorded explicitly, and then the GPU `draft_load_check` read-back becomes a hard smoke gate.
  - **Also added:** the prefix test widened to 2 prompts × 3 turns (#47930 class). D4b stays optional and last (#54690).
- **2026-11-01: NBDF built.**
  - `Q27-XHIGH-E27_NBDF_DFLASH2.ipynb`: 386,896 B, sha `3e2df689…`, 10 cells.
  - Units: 116/116 PASS.
  - Arm order: Q → D4 → D3 → D7 → D4b, in a 140-min budget.
  - **`gpu_ready: false`** until the real CPU kit passes `kv_cache_configs_ok` + seam apply / verify / revert. Then rebuild with `--kv-gate-report`.
  - Kits `q27-e27-nbdf-kit-units` and `q27-e27-nbdf-kit-real` were pushed on foysal.
  - Audit zip: `E27_NBDF_audit_FLAT_2026-11-01.zip`.
  - Report: `Agent Reports/E27_REPORT_nbdf_build_2026-11-01.md`.
- **2026-11-01: Seam built.**
  - `dflash_seam.py` sha `c4ea1ecc…1f78`, 520 CPU checks ALL PASS. Report: `Agent Reports/E27_REPORT_seam_build_2026-11-01.md`.
  - It patches 6 Merlin files and keeps nb4's files untouched.
  - **Open risks for the kit:** `get_kv_cache_configs` per K and drafter SW block (U1); the aux relay under `@support_torch_compile` (PP0 capture mode read back on GPU); the loaded-weights check on the real drafter.
- **2026-11-01: The three GPT model choices** agree with our drafter: incoai Flash-Next DFlash2 BF16 (not the 27B drafter, not GGUF).
- **2026-11-01: New upstream evidence.**
  - vLLM #60765 / PR #61180 is a PP2-validated aux relay for hybrid GDN. The seam is being compared against it.
  - vLLM #61072: a mismatched DFlash2 pairing gives silent 0 % acceptance. Added an acceptance canary: `DFLASH_ZERO_ACCEPT` if under 2 % after 200 steps, and the arm fails fast.
  - **Candidate-selector workspace:** `qwen3_dflash2.py:408` builds full-vocab vocab-parallel logits, about 2 GB per rank at an 8192-token profile. Read back the profiled peak and pool against Q.
- **2026-11-01:** GPT analysis 3 integrated: the 512-MiB-free memory gate with a reduced pin on W4, the numerical tap reference test, the greedy token-ID equivalence, the concurrency contamination ladder in the smoke, the upstream ring helper, and paired windows.
- **2026-11-01:** GPT analysis 2 integrated: the prefix-cache cold / warm acceptance gate, the BF16 drafter KV confirmed, the drafter tick cost at long context, the three-state tree check, the sample-method / draft-TP parse checks, and the final aux-tensor checks. Bug A (#55600) is already fixed in Merlin (`mamba_hybrid.py:159-165`).
- **2026-11-01:** GPT analysis 1 integrated (see the table): the Q-defer isolation rerun, the per-arm smoke, per-stage peak memory, block-size / ring read-back, the spec-config parser preflight, and early stop.
- **2026-11-01:** created. Steps 0-3 started; 5 agents: 2 verifiers (done) and 3 builders (drafter, seam, NBDF).
