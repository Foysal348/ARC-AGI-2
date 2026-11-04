# DFlash2 on klee100 3bpw: expected gain, acceptance risk, the 150-min NBDF-K100 design, and the E28 adoption path

This was read-only work: no project files were edited, nothing was pushed and git was not run. My model script is `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/k100df/model.py`.

## 0. Headline
1. **Central expected gain at the production level (24 streams, 8k).**
   - Against MTP K3 with the BF16 head: D3s ×1.15, D4s ×1.19.
   - Against MTP K3 with I8 (int8 draft head): D3s ×1.09, D4s ×1.13.
   - Pessimistic: ×0.91–1.01 (a loss against I8). Optimistic: ×1.16–1.33.
   - The gain is larger at 8 streams (D3 ×1.21, D4 ×1.31 against the BF16 head) and shrinks with streams.
2. **Acceptance risk verdict.** A gradual drop is likely; a quantisation-caused structural 0 % is unlikely.
   - Everything DFlash2 reads from klee100 is structurally the same as W4: the tap path, HC weights, dtype, layer layout, embedding and lm_head.
   - The real 0 % risk is the port itself. The W4 NBDF D arms never ran, so the seam, PP relay, prefix cache, SW-1024 block and graph-metadata path are all untested on GPU.
   - If D3s shows 0 %, it cannot be blamed on klee100.
3. **Recommended K:** K3 (D3s) runs first and is the default. K4 is adopted only if the measured verify cost per token is low (see §3).
4. **I8 is a key confound.** Most of DFlash2's drafting saving is three BF16 lm_head reads (7.5 ms). I8 already halves those, so against an I8 baseline the gain rests mostly on acceptance length (AL).
   - The Q arm has to be exactly the E28 production config. Run it with I8 if E28's klee run read back `I8_ENGAGED`, otherwise with the BF16 head.

---

## 1. Acceptance risk on the low-bit target

### Upstream and our own evidence

**vLLM #61072** (https://github.com/vllm-project/vllm/issues/61072, read in full via gh):
- These are per-draft-token rates at k = 15, PP2, vLLM 0.31.
- SergiioB GPTQ-Int4 repack of the *official* 27B × z-lab 27B DFlash2: **exactly 0.0 %** (4,485 drafts, 3/3 rounds). The reporter suspects that `target_layer_ids` / the layer registry did not survive the repack, i.e. a structural fault, not a weight-perturbation effect.
- Swift-1.5 (a *retrain* of the base, W4A16 AutoRound) × the same drafter: 16–23 %.
- The paired control (4B): 25.6 %. Own MTP: 29–43 %.
- Reading: even a full retrain degrades gradually. Only a structural fault gives exactly 0.
- Follow-up PR #61084 adds a warning below 1 % after 4,096 drafts.

**Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter** (https://huggingface.co/Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter):
- **"0 of 2,681 drafts accepted"** when the drafter was cast to fp16. This is a second known structural-zero cause.
- An INT4 drafter on an INT4 target: −1.6 %.

**Our own E11-03, the closest analog** (4×L4, ARC-type workload; RedHat W4A16 27B target × z-lab BF16 DFlash2 trained on the unquantised target, SGLang, block 8):
- Source: `experiments/Experiment 11 .../Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md`.
- DFlash2 sweep AL 4.37–4.95; MTP-3 on the same target 3.09–3.17.
- ARC sanity AL 3.77–5.50 (7/8 solved); longctx 4×85k AL 4.58.
- So **a W4 quantised target lost essentially nothing** against the blog's BF16 27B block-8 AL of 4.80.
- **But DFlash2 degrades more than MTP over long generation:**
  - in the 12×24,576 long run, MTP went 3.10–3.17 → 3.02 (−3 %) and DFlash2 4.70–4.95 → 4.20 (−12 %);
  - my bucketing of `decode_rows_C_int4_dflash2_long.json` by per-request context: AL 4.76 (0–10k), 4.10 (10–20k), 3.93 (20–30k), 3.35 (30k+).
  - I use this as the long-band haircut.

**Other sources:**
- **syvai W4A16** (https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16): on a W4A16 target, acceptance equals the BF16 drafter at greedy (3.34–3.65 vs 3.54) and is about 5 % lower at default sampling. It also notes that a non-BF16 lm_head is refused for the candidate top-k.
- **Doopeworld DSpark** (https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM): FP8-trained drafter on a GPTQ-Int4 target gives 2.45–2.79 vs the published 3.39. The author says this is not like-for-like, and argues that "heavier quantisation perturbs the very hidden states it learned to read".
- **arXiv 2607.04244** (https://arxiv.org/html/2607.04244): DFlash on Qwen3.5-4B. BF16-trained drafter on the BF16 target 4.92; after stage-2 adaptation to the INT4 target 5.03; direct INT4 training 4.97. A BF16 drafter on the INT4 target unadapted is not reported. Adaptation gains only +2 %, which suggests the INT4 penalty is small.
- **No published data exists for 2–3-bit targets** with any hidden-state drafter. I searched EAGLE / DFlash / QuantSpec / ML-SpecQD (https://arxiv.org/pdf/2502.10424, https://www.alphaxiv.org/abs/2503.13565) and arXiv 2505.22179 (https://arxiv.org/abs/2505.22179, which has no acceptance numbers). klee100 is out of distribution for all published evidence.
- **incoai card** (https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2): evaluated on the official FP8 checkpoint at block 4 (K3), "limited by the current QSA implementation". It says nothing about quantised targets or long context. The training target is not stated on the card.

**klee100's own signals** (E26 K3ONLY v2):
- MTP per-position acceptance 0.864 / 0.731 / 0.607, AL 3.21. That is at or above W4: NBDF Q at 8k gave 0.853–0.864 / 0.696–0.715 / 0.557–0.584, AL 3.12–3.18.
- So a hidden-state-conditioned head trained on the full-precision model did **not** lose acceptance on klee100's perturbed hidden states. The text klee100 generates also looks more predictable: it uses ×1.44 the tokens.
- Caveat: the MTP layer was AutoRound-calibrated inside klee100's own pipeline (W4, `block_name_to_quantize` includes `mtp.layers`). DFlash2 gets no such adaptation.
- Teacher-forced klee100 vs W4 BF16:
  - mean |Δlogprob| 1.10–1.70, rank-1 agreement 0.51–0.65;
  - against the yardstick W4 FP8-KV vs BF16: 0.22–0.93 / 0.75–0.93;
  - FP8 KV cost MTP about 4 % AL (2.957 → 2.833);
  - linear extrapolation of the divergence gives a −9 to −14 % AL risk for an unadapted drafter.

### Quantified range for DFlash2 on klee100 (flat conditional c; AL_K = (1 − c^(K+1)) / (1 − c))

| Scenario | 8k–40k AL3 (c) | AL4 | vs klee100 MTP AL3 3.21 | 90k / 116k AL3 |
|---|---|---|---|---|
| Optimistic (published ×1.12 with a 4.5 % agentic haircut, as verify-B; no quantisation loss) | 3.43 (0.90) | 4.09 | ×1.07 | 3.37 / 3.32 |
| **Central** (as optimistic, −5 % quantisation) | **3.26 (0.865)** | **3.82** | **×1.02** | 3.04 / 2.99 (−5 % long haircut) |
| Pessimistic (−15 % quantisation, −10 % long) | 2.92 (0.79) | 3.32 | ×0.91 | 2.58 / 2.54 |
| Structural | ≈ 1.0 (accept ≈ 0) | — | — | — |

On real Duck, MTP AL is 2.8. Scale DFlash by the same ratios, then take about ×0.95 more for the DFlash-specific long-generation decay seen in E11-03.

### What would cause a structural 0 % on klee100, and what is klee-specific

**Checked against klee100's config** (`_scratch/e25_kits_out_klee_v2/q27-e25-klee100-cpu-kit/fixkeys_overlay/config.json`):
- dtype bfloat16;
- 48 layers, the same `layer_types` (36 GDN + 12 QSA), hc_count 4, vocab 248,320;
- `attn_/mlp_hyper_connection.block_inject_weight` are **bits 16** (unquantised), so the HC combine that produces the taps runs in BF16;
- the embedding is byte-identical to W4 (sha `62288a3f…c9be3a`, row 248077 `860615…1a19`; E27 README:41);
- lm_head is BF16 (verify-B §3).

**There is no klee-specific structural path. Every structural cause is shared with W4:**

| # | Cause | Typical signature |
|---|---|---|
| S1 | Tap ids / PP relay order (remote [6,16] before local [26,36,46], slot base 2) | ≈ 0–5 % |
| S2 | Tap semantics: pre- vs post-combine, HC-mean vs one stream (seam `DFLASH_TAP_MODE`) | a large but non-zero drop |
| S3 | Embedding / mask row 248077 wrong | ≈ 0 |
| S4 | lm_head not shared, or quantised. **I8 leaking in: `DRAFT_HEAD=int8` must not reach a dflash spec.** | refused or garbage |
| S5 | Drafter cast to fp16 (Launch80) | exactly 0; klee is bf16 ✓ |
| S6 | Drafter KV inheriting FP8 without scales (`kv_cache_dtype` None = fp8; `auto` = BF16). FA2 refuses FP8 on sm_89, which falls back to an eager drafter. | degraded / slow |
| S7 | Prefix-cache warm turns with unwritten draft KV (#47930 / #58894). **The biggest Duck risk: K100Q had an 85.4 % prefix hit.** | collapse on warm turns only |
| S8 | SW-1024 lowers the engine block to 1024 for the target too (`core.py:436-448`; kit R9 did not model it) against the Mamba page (min block 3,385, padded 21 %) and RecoverSSM | boot failure, F explosion or wrong state (the oracle catches it) |
| S9 | Stale FULL-graph metadata (#63587 / PR #64059; the notebook's INGRAPH_META_FALLBACK relaunch) | 0 % under graphs |

**klee-specific risk is gradual only:**
- The W2 g64 experts in layers 12–35 shift the residual stream at taps 16 and 26 and downstream at 36 and 46. The fc sees off-distribution features.
- The drafter's RMSNorms (`hidden_norm`, `input_layernorm`) absorb a scale drift, but not a direction drift.
- Distinguishing signatures:
  - **structural:** pos0 < 0.05;
  - **tap order / semantics:** pos0 0.05–0.5;
  - **quantisation:** pos0 0.70–0.85 with a flat-ish decay.

---

## 2. Cost model on klee100 (stage 1 binding after Q-defer; tok/s = N·AL / (2·tick))

### Inputs
- **Q tick with Q-defer:** projected as the K3ONLY v2 no-Q-defer ticks / 1.20, using the W4 Q-defer gain of +21 % / +20 %.
  - 20k@16: 55.8 → 46.5 ms; 20k@24: 68.7 → 57.2 ms.
  - 8 streams: 55.8 − 16 × 0.806 = 42.9 → 35.8 ms.
  - 90k@8: 39.5 ms; 116k@6: 36.0 ms, from W4's measured long/8k tick ratios (NBDF Q: 8k@8 41.7–43.4, 90k@6 38.0–39.5, 116k@5 36.1).
  - **These Q ticks are projections.** Klee100 with Q-defer has never been measured synthetically.
- **MTP K3 drafting per micro-batch:** 11.5 / 13.5 / 14.0 ms at 8 / 16 / 24 streams. This is verify-B's measured 9.8 ms at 1 request and 13.4 ms at 7, plus its klee100 estimate of 13–14 ms at 8–12 requests.
  - I8 saves about 3 ms. NB4T measured +3.4 % at 8k and +11.4 % at 40k; half of the 7.5 ms of three lm_head reads would be about 3.8 ms.
- **DFlash2 drafting:** 5.8 / 7.0 / 7.5 ms.
  - Verify-B: 5.5–6.6 at 7 requests, 6.5–7.5 for klee100.
  - My check of the SW attention term: 5 layers × 2 KB/token/card × 2048 tokens = 20 MB per request per card. That is 246 MB per micro-batch at 24 streams, about 1.0 ms, and is included.
  - K4 adds about 0.5 ms (a 5-position block, candidates for 4 positions; one lm_head read either way).
- **Verify cost per extra verify token per micro-batch on the binding stage (v):**
  - central 0.5 ms, optimistic 0.35, pessimistic 0.8 (the measured klee100 slope of 0.806 ms per token, which mixes per-stream costs and the no-Q-defer serialisation);
  - ×1.1 at 40k, ×1.3 at 90k, ×1.4 at 116k for QSA per-query attention (W4 per-stream tick grows +22 % from 8k to 90k);
  - bottom-up check of the MoE part: unique experts for top-10 of 512 go 313 → 355 at 48 → 60 tokens (+13 %), and 237 → 280 at 32 → 40 tokens (+18 %). At about 69 µs + 1.6–2.7 µs per expert over 23 layers that is +1.5–2.6 ms, plus lm_head / sampler / AllGather / GDN-verify at about 0.15–0.25 ms per token. The total, v ≈ 0.3–0.47, supports 0.5 as central-to-conservative.
- **D3 tick** = tick_Q − (D_mtp − D_df). **D4 tick** = D3 tick + 0.5 + (N/2)·v·ctx.
  - Caveat: if stage 0 becomes binding after the ~6 ms cut, the saving is capped. On W4, stage 0 is 21.6 ms vs stage 1 33 ms, so not binding. Read `binding_stage` back on klee100.

### Expected gain table (× vs Q, with projected tok/s in brackets)

| Band@N | Q base | Q tok/s (proj.) | D3s pess / **central** / opt | D4s pess / **central** / opt |
|---|---|---|---|---|
| 8k@8 | BF16 head | 359 | 1.05 / **1.21 (433)** / 1.29 | 1.06 / **1.31 (469)** / 1.45 |
| 8k@8 | I8 | 391 | 0.96 / **1.11** / 1.19 | 0.97 / **1.20** / 1.33 |
| 8k@16 | BF16 | 552 | 1.03 / **1.18 (652)** / 1.26 | 1.00 / **1.24 (687)** / 1.38 |
| 8k@16 | I8 | 590 | 0.97 / **1.10** / 1.18 | 0.94 / **1.16** / 1.29 |
| 8k@24 | BF16 | 673 | 1.01 / **1.15 (772)** / 1.22 | 0.96 / **1.19 (802)** / 1.33 |
| 8k@24 | I8 | 711 | 0.95 / **1.09** / 1.16 | 0.91 / **1.13** / 1.26 |
| 40k@16 | BF16 | 533 | 1.03 / **1.18 (627)** / 1.25 | 0.99 / **1.23 (655)** / 1.37 |
| 40k@16 | I8 | 569 | 0.96 / **1.10** / 1.18 | 0.93 / **1.15** / 1.28 |
| 90k@8 | BF16 | 319 | 0.93 / **1.13 (360)** / 1.27 | 0.90 / **1.19 (378)** / 1.40 |
| 90k@8 | I8 | 345 | 0.86 / **1.04** / 1.17 | 0.83 / **1.10** / 1.30 |
| 116k@6 | BF16 | 258 | 0.93 / **1.14 (293)** / 1.28 | 0.91 / **1.20 (309)** / 1.41 |
| 116k@6 | I8 | 282 | 0.86 / **1.04** / 1.17 | 0.83 / **1.10** / 1.30 |

**Real-Duck projection:** K100Q window average 385 tok/s (21 workers). Taking the central K3 ×1.09–1.15 and the ×0.95 Duck long-generation factor gives about **400–420 tok/s with D3s and 405–430 with D4s**. The pessimistic case gives 350–390.

### Break-even AL (AL_D* = AL_Q · tick_D / tick_Q)

| Band@N | K3, BF16 base (central / pess tick) | K4, BF16 base | K3, I8 base | K4, I8 base |
|---|---|---|---|---|
| 8k@8 | 2.70 / 2.79 | 2.92 / 3.12 | 2.95 / 3.04 | 3.19 / 3.41 |
| 8k@16 | 2.76 / 2.83 | 3.07 / 3.31 | 2.95 / 3.03 | 3.28 / 3.53 |
| 8k@24 | 2.85 / 2.90 | 3.21 / 3.47 | 3.00 / 3.06 | 3.39 / 3.66 |
| 40k@16 | 2.75 / 2.81 | 3.07 / 3.32 | 2.93 / 3.00 | 3.28 / 3.54 |
| 90k@8 | 2.70 / 2.78 | 2.94 / 3.15 | 2.92 / 3.00 | 3.18 / 3.41 |
| 116k@6 | 2.64 / 2.72 | 2.86 / 3.05 | 2.87 / 2.97 | 3.12 / 3.33 |

As a flat conditional c at 24 streams, break-even is about **c ≥ 0.78 against the BF16 head and c ≥ 0.82 against I8** (K4 at pessimistic v: c ≥ 0.84). Compare klee100 MTP's conditionals of 0.864 / 0.846 / 0.830.

---

## 3. Interaction with streams; recommended K

- **The gain shrinks with streams.**
  - The DFlash2 saving is roughly fixed per micro-batch (about 5.7–6.5 ms; one lm_head read instead of three).
  - The tick grows 35.8 → 46.5 → 57.2 ms from 8 to 16 to 24 streams, so the relative tick saving falls from 16 % to 14 % to 11 %.
  - K4's extra verify cost grows with streams: (N/2)·v = 2.0 / 4.0 / 6.0 ms central, 3.2 / 6.4 / 9.6 ms pessimistic.
  - MoE dominates that growth, along with sampler / AllGather tokens and QSA query reads at long context.
- **K4 over K3 needs c above a threshold that rises with streams:**
  - 8 streams: c ≥ about 0.80 (any v);
  - 16 streams: c ≥ 0.80 (v 0.5) or ≥ 0.89 (v 0.8);
  - 24 streams: c ≥ 0.80 (v 0.5) or ≥ 0.92 (v 0.8).
  - Central K4 over K3 at 24 streams is only +3.5 %, and it is negative in the pessimistic case. At 8 streams it is +8 %.
- **Recommendation: K = 3 (D3s) as the default for klee100 at 24 workers.**
  - K3 is incoai's validated block 4, adds no verify cost, keeps the captures at 4..96, and is the most robust choice against an I8 baseline.
  - Adopt K4 only if, in the same run, D4s beats D3s at **8k@24** by more than the A/A spread and does not regress at 40k / 90k / 116k.
  - The D3s vs D4s tick difference at equal N measures v directly.
- **No K7** at ≥ 16 streams: 96–192 verify tokens per micro-batch, the ring widening, and captures beyond the budget.

---

## 4. The decisive 150-min NBDF-K100 experiment

### Arms and order: Q → D3s → D4s (D3s first; this reverses the W4 plan)
**Why D3s first:**
- D3s has the same verify shape as Q, so it isolates the drafter's cost and acceptance cleanly. It is also the likely production winner at 24 streams.
- A structural fault is shared by both arms, so a D3s zero-accept saves the whole D4s slot (about 38 min).
- D3s's measured conditionals c predict D4s and allow a data gate before D4s runs.

**Arm definitions:**
- **Q:** the E28 production config exactly:
  - MTP K3 + Q-defer, plus I8 if E28's klee run read back `I8_ENGAGED`, otherwise the BF16 head (record which);
  - `--max-num-seqs` 24;
  - K100_CAPS;
  - pin 7,933,526,016.
- **D3s / D4s:**
  - the NBDF spec `{"method":"dflash","model":<bf16-embed dir>,"num_speculative_tokens":K,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"}`;
  - `DFLASH_DRAFT_SW_BLOCK=1024`, `DFLASH_SEAM=1`;
  - `DRAFT_HEAD` **unset**;
  - captures = multiples of (1 + K) up to 24 × (1 + K), i.e. K3 4..96 and K4 5..120 (the NBDF rule "largest capture = SEQS × (1+K)", with SEQS 24, because the micro-batch split is IMBALANCED, modal 7 / 1 on W4);
  - pin: start at Q's 7,933,526,016 and step down in **0.25 GiB** steps (7,665,090,560 → 7,396,655,104) on an OOM or the 512 MiB gate. Do not use K100Q's 1.07 GB rungs, which cost about 13 % of the pool each.
  - Memory estimate for PP1: 719 MiB worker-free, minus 0.03–0.12 GiB net weights, minus about 48 MB of fp32 draft logits at 24 × 4 → about 550–610 MiB. That passes the first rung marginally.

### Levels and bands
The 8k / 40k / 90k / 116k bands are all kept (user rule). There is time for 16-stream windows, and for one 24-stream window per arm.

| Window | Level | Prefill (3.3–3.5k tok/s uncached) | Why |
|---|---|---|---|
| 8k@16 × 2 | 16 | ~0.6 min | A/A spread + the K3 / K4 tick difference |
| **8k@24 × 1** | 24 | ~1 min | the **production level**; decisive for K and adoption |
| 40k@16 | 16 | ~3.1 min | the mid band at production-like concurrency |
| 90k@8 | 8 | ~3.5 min | long band; regression check (16 would cost 7 min / arm) |
| 116k@6 | 6 | ~3.4 min | long band; regression check |

- Each window is about 90 s steady, running == level, 0 preemptions.
- Long bands get one window each, so their noise threshold uses W4's measured A/A: 90k cold / warm 236.0 / 244.8 (3.7 %), 116k 205.3 / 216.3 (5.4 %). Compare warm against warm where possible.

### Time budget

| Item | Min |
|---|---|
| Setup: install, shas, inputs, page-cache drop, ppack + seam apply, 3-state tree check, drafter gate | ~10–12 |
| **Q:** boot (klee100 first boot 519–777 s) ~11 + read-backs and memory gate 1 + smoke and ladder 2 + **greedy oracle twice (Q A/A)** 3 + windows ~20.6 + long oracle 1.5 + prefix 2×3 2.5 + one 30-s profile | **~42–45** |
| **D3s:** reboot ~6 + smoke / canary 2 + oracle 2 + windows ~20.6 + long oracle 1.5 + prefix 2.5 + profile | **~36–38** (+6 per pin step) |
| **D4s:** same | **~36–38** |
| Total | **~126–133**, leaving 17–24 min for one pin step-down and the drain |

The W4 reference was setup + Q in 61.2 min. If the budget runs short, D4s drops 116k by `arm_fits` and becomes SHORT_ONLY, so it cannot be adopted; D3s is then the fallback.

### Early checks
1. **Zero-accept canary (structural), kept as built** (`nbdf_lib.py:129-130`, `:1341`). The smoke is 2 streams × 2048 tokens. After ≥ 200 drafts (a few seconds), accept < 2 % → `FAIL_SMOKE_DFLASH_ZERO_ACCEPT`.
   - **Add:** record pos0. Then:
     - pos0 < 0.05 = STRUCTURAL;
     - 0.05–0.5 = TAP_SUSPECT.
   - On D3s STRUCTURAL / TAP_SUSPECT: **skip D4s.** Spend at most 10 min on a smoke-only diagnostic: the D3s variant with the alternative `DFLASH_TAP_MODE`, plus the existing Q-defer-off isolation smoke. Then end.
2. **Add a "no-hope" cut.** After ≥ 2,000 drafted tokens, smoke AL3 < 2.40, i.e. below 0.75 × Q's smoke AL (W4 Q smoke was 3.06; the break-even AL3 is never below 2.64), gives `DFLASH_LOW_ACCEPT`. Run one 8k@16 window for the record, skip the rest of the arm, and skip D4s too if it happened on D3s.
3. **D3s → D4s data gate:** run D4s fully only if D3s's 8k@16 mean conditional c ≥ 0.80. Below that, D4s cannot beat D3s at 16–24 streams by §3.
4. **Prefix canary:** the 2 × 3 cold / warm test. The Duck rule should be **warm AL ≥ 0.9 × cold**, tighter than the built 0.8 (`PREFIX_WARM_MIN`), because Duck runs at 85 % prefix hit.
5. **Correctness gate fix for klee100 (important).**
   - klee100 is not run-to-run deterministic: fresh vs repeat rank-1 agreement 0.856 / 0.949 / 0.914, mean |Δlogprob| 0.66 at 2k (E26 K3ONLY v2), whereas W4's A/A was 0.0.
   - So the NBDF greedy token-ID EXACT / NEAR_TIE oracle and the concurrency ladder will **false-fail** D arms.
   - Fix: run Q's oracle and ladder twice (A/A). A D arm passes if its first-divergence position and high-margin flip count against Q are within Q's own A/A envelope, it has 0 windowed garbage, and its long output is coherent.

### What to read back (per arm)
- **Speculation:** SpecDecoding drafted / accepted, AL and per-position acceptance per window and per band; smoke and oracle acceptance; pos0.
- **Speed:** tick ms per window, running == level, preemptions, KV usage max, prefix hit rate.
- **Capacity:**
  - `num_gpu_blocks`;
  - **engine block** ("Setting attention block size to N tokens": 1024 expected on D);
  - max concurrency at 131,072;
  - F at the boot's own block (HIGH-1 `f_blocks`);
  - KV groups and layout (`KV_GROUP_EXPLOSION`, #58638);
  - drafter `SlidingWindowSpec` dtype BF16, ring capacity 8;
  - **resident bound at 40k / 64k / 90k / 120k against Q's 29 / 21 / 17 / 13** (the 24-worker question).
- **Memory:** free MiB per rank after load, KV init, graph capture, first decode and **after the 24-stream window**; ≥ 512 MiB required.
- **Profile** (one 30-s 8k@16 or 8k@24 trace): stage-0 vs stage-1 busy time, the MTP vs DFlash2 draft ms, lm_head ms, and the verify ms per token. This validates D_mtp, D_df and v and shows whether stage 0 binds after the cut.
- **Seam events:**
  - aux layers [6, 16 | 26, 36, 46] on 4 ranks, slot base 2, finite taps;
  - embed row sha, `has_own_embed_tokens`;
  - `draft_load_check` 67 = 67;
  - `draft_sw_block` 1024;
  - conv_impl;
  - drafter FULL_DECODE_ONLY graphs captured, with no eager decode batches.
- **Q-defer ENGAGED:** record only. Gate on real failures only (the fix after the W4 run's label-gate loss).
- **Responses:** every response saved in full (user rule).

### Decision rule: adopt DFlash2 in NB-K100 iff all of these hold
1. **Correctness:**
   - smoke PASS, 0 garbage in every window;
   - oracle and ladder within Q's own A/A envelope;
   - long oracle (90k / 116k) coherent;
   - prefix warm AL ≥ 0.9 × cold.
2. **Speed at the production level:**
   - gain ≥ max(2 × Q's 8k@16 A/A spread, 6 %) at **8k@24** and at **40k@16**;
   - measured against the production Q (with I8 if I8 engaged on klee100 in E28).
3. **No long regression:** 90k@8 and 116k@6 gain ≥ −max(Q A/A, 3 %).
4. **No long-context acceptance collapse:** D's AL at 90k and 116k ≥ 0.9 × its own 8k AL.
5. **Capacity:**
   - D's resident bound at 64k ≥ 24, or pool tokens ≥ 0.95 × Q's;
   - memory gate ≥ 512 MiB on all ranks after the 24-stream window;
   - 0 preemptions.
6. **K:** D3s unless D4s beats D3s at 8k@24 by more than the A/A spread and is not worse in any other band.

If the gain is positive only at 8 streams, or only short, the label is SHORT_ONLY and NB-K100 keeps MTP K3.

---

## 5. Adoption checklist for the E28 K100 notebook (`build_e28_k100.py`)

1. **Speculative config.**
   - Add the trailing `--speculative-config` dflash JSON with the winning K. NBDF's launch override works because the last occurrence wins (`launch.py:110-111`).
   - Drafter dir: `/kaggle/input/models/foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1/bf16-embed/`.
   - Read back "Resolved architecture: DFlash2DraftModel".
2. **I8 off.**
   - `E28_I8.want/on = False` and `DRAFT_HEAD` never set. The I8 gate (`e28_i8_gate`, builder :367-379) then returns early by design.
   - Record I8 as N/A_DFLASH.
   - A guard should refuse any `draft_lm_head_quant` in a dflash spec: the candidate selector needs the BF16 lm_head (S4).
3. **Seam env and install.**
   - `DFLASH_SEAM=1`, `DFLASH_DRAFT_SW_BLOCK=1024`, and the winning arm's tap and conv settings.
   - Install `dflash_seam.py` sha `7356d4738962…6937` with nb4_ppack rev 2 (both apply orders were kit-tested):
     - pre / post shas, py_compile, fail-safe restore;
     - the 3-state tree check, with INVALID_MIXED aborting.
4. **Drafter gate.**
   - `model.safetensors` sha `d2e0464d15936265bb75eda9d09149d27ac6986327ca269a0335cd3d49661d49` (2,477,538,608 B) and `config.json` `24095759…b1b9`;
   - row 248077 of the drafter embedding = klee100's (`860615…1a19`), embed sha `62288a3f…c9be3a`.
5. **Captures.** Multiples of (1 + K) up to 24 × (1 + K) (K3 4..96, K4 5..120). Parametrise the K3 hard-codes:
   - `PP2_MICROBATCH_TOKENS = -(-WORKERS * (1 + 3) // 2)` (:107);
   - `captures_cover_pp2_microbatch … -(-_w * 4 // 2)` (:396);
   - `K100_CAPS` (max 84) and `captures_cover_workers` (:394).
6. **Pins and capacity facts.**
   - `assert POOL1 == 479 and F1 == 6` (:164) and `capacity_note(POOL1, 4096, F1, …)` (:165) are MTP-K3 facts. Replace them with the D boot's measured pool, mc and F **at its own engine block** (1024).
   - D pin ladder in 0.25 GiB steps from 7,933,526,016.
   - The rung seq table is recomputed from D's measured capacity.
7. **Workers.** 24 workers and `--max-num-seqs` 24 only if NBDF-K100 showed the D resident capacity is at least about Q's. Otherwise use the R2 rule on D's measured pool.
8. **In-run gates.**
   - The zero-accept canary in the 6-h run: accept < 2 % after 200 drafts, then stop.
   - An hourly AL monitor: flag when AL < 0.7 × the first hour.
   - Read back the prefix hit rate, seam events, engine block, KV layout and Q-defer ENGAGED (record only).
9. **Fallback.** Any DFlash boot failure, canary hit or memory-gate miss relaunches at the same pin with the proven K100Q MTP K3 config. This is the analog of `I8_UNSUPPORTED → BF16 head`; a 6-h run cannot afford a dead engine.
10. **Size.** The E28 K100 notebook is 394,184 B (5.8 kB margin). Put the seam and its tests in the sha-pinned payload dataset, not inline, and keep the notebook under 400 kB.
11. **Proven format unchanged.** CELL 0 = E18 R2 verbatim; E15B writer; rerun = every test task. RERUN_PROD must use exactly the commit-measured DFlash config.
12. **Kits and audit.**
    - Units, plus the real kit on the klee100 overlay: seam apply / verify / revert, `get_kv_cache_configs` with the drafter group at 1024, `draft_load_check` 67 = 67, the parse preflight for the klee spec;
    - e2e commit and rerun;
    - the audit zip in `Experiment 28/Experiment Review/Audit Zips/`.
    - The GPU push needs the user's go.
13. **Timeouts.** Allow about +30–60 s of boot for the 2.48 GB drafter load. `MERLIN_SHARE_DRAFT_WORKSPACE` is inert (0 B measured); drop it or record it.
14. **Report fields.** Spec method and K; drafted / accepted totals; AL per position per hour; engine block; resident bound.

## Key file evidence
- `D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Review/Agent Reports/E27_REPORT_verify_B_drafter_gain_2026-11-01.md` §4 (cost model, ×1.24 / ×1.34; klee100 @16 / @24 projections).
- `D:/ARC AGI 2/_scratch/e27_gpu_out/q27-xhigh-e27-nbdf/nbdf/nbdf_verdict.json` (Q windows; long A/A 3.7–5.4 %; capacity `binding_stage` 1; all D arms SKIPPED_DEPENDENCY).
- `D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build/nbdf_lib.py:120-140` (canary, `PREFIX_WARM_MIN`) and `:1326-1407` (smoke / prefix gates).
- `D:/ARC AGI 2/experiments/Experiment 26 - Flash-Next klee100 3bpw low-bit (planned)/Experiment Results/E26_K3ONLY_GPU_v2_2026-11-01.md` (ticks, AL, TF divergence, self-repeat non-determinism) and `E26_K100Q_GPU_v1_2026-11-01.md` (385 tok/s, AL 2.8, prefix 85.4 %).
- `D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md` and `decode_rows_C_int4_dflash2_long.json` (the W4-target DFlash2 analog and the long-generation decay).
- `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build/build_e28_k100.py` lines 106-112, 164-165, 367-444 (workers, K3 hard-codes, I8).
- My model: `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/k100df/model.py`.

Sources: [vLLM #61072](https://github.com/vllm-project/vllm/issues/61072), [vLLM PR #61084](https://github.com/vllm-project/vllm/pull/61084), [incoai DFlash2 card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), [Launch80 INT4 drafter](https://huggingface.co/Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter), [syvai W4A16 drafter](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), [Doopeworld DSpark](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM), [arXiv 2607.04244](https://arxiv.org/html/2607.04244), [arXiv 2505.22179](https://arxiv.org/abs/2505.22179), [QuantSpec](https://arxiv.org/pdf/2502.10424), [ML-SpecQD](https://www.alphaxiv.org/abs/2503.13565), [Friendli dfischermittwald NVFP4-DFlash2](https://friendli.ai/models/dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2).