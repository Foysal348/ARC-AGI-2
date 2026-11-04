I found 124 engine configurations our team ran on Kaggle 4×L4 GPUs: 84 on the Qwen3.8-27B track and 40 on the Qwen3.8-Flash-Next track. The script check confirms 84 rows starting "| T" and 40 starting "| F".

File: `D:/ARC AGI 2/papers/5 - Kaggle 4xL4 Engine Configuration Speedup/engine_config_inventory_2026-11-04.md` (46,888 B, new folder). Nothing else in the project was changed. My scratch helper scripts are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\` (`sum_nbdf.py`, `sum_nbdf2.py`).

How I counted:
- **Crashed / OOM / garbage runs are included.** They are measured outcomes on the GPU, and each row carries a status.
- **Repeated runs of one configuration share one row.** For example, F21 holds NB4S QD, NB4T Q and NBDF-W4 Q; F31 holds NBDF-K100 Q, the E28 K100 6 h run and E31 Direct30.
- **Some numbers have no write-up and were read straight from raw JSON in `_scratch`.** These are NB-S1 GPU/SMOKE/FULL, NBDF W4, NBDF-K100, E28 K100/W4 and E31.
- **Some rows are second-hand.** Rows that cite the E25 mining reports (A1/A2/A3) or part_C (PC) use numbers from our earlier mining, which quotes the primary logs. I spot-checked them against the EMS/VMS results file.
- **Left out of the count:** kernel-only microbenchmarks (NB3, NCCL/GEMM). Also left out as off-scope: the ARC-AGI-3 runs on one RTX PRO 6000.
- **Rows with weak engine detail:** for T04 (E9/E10) and T05 (E2) the sources do not restate the engine flags. For T15 (archive) the vLLM version is not stated. All three are marked so in the file.

Key findings:
- **Best measured Flash-Next steady speed:** DFlash2 D3s with EMBED_HOST on klee100 (F36), 763.0 tok/s at 20k@24. It beats klee100 MTP K3 + Q-defer + I8 (F31) at every band: 631/610 vs 575/498 at 8k@16, 383 vs 339 at 90k@8, 324 vs 312 at 116k@6.
- **The same DFlash2 path failed several times first.** NBDF-K100 failed at launch (F32-F34), and the first NB-S1 run hit OOM without EMBED_HOST (F35, F37, F38).
- **Best real-run (whole-window) numbers:**
  - klee100 MTP + Q-defer + I8 (F31): E28 K100 EVAL120 6 h ran at 419.5 tok/s, task-norm 65.0 (107/172). E31 Direct30 ran at 407.1.
  - W4 NB6 SUBMIT (F25) is the only Flash-Next config with a recorded LB: 54+.
- **27B track:**
  - Best single-server steady speed: SGLang DFlash2 W6 at 20 streams (T62), 597 / 536 / 364 at 8k / 40k / XL.
  - Best 27B aggregate: Merlin TP1×4 replicas at 40k@16 (T44), about 504-507 tok/s (E23 FULL).
  - Best 27B LB: 34.58, from vLLM 0.19 FP8 MTP3 (T02).
- **Contradictions worth knowing before writing the paper:**
  - The same stock vLLM 0.32 FP8 MTP3 config at 40k@16 measured 399 (VMS), 435.7 (E23 SMOKE, medium prompt render) and 353-361 (E23 FULL).
  - Merlin 27B acceptance was 4.18-4.23 in SMOKE but 3.6-3.76 in FULL.
  - W4 FP8 8k@8 varies about 10 % between runs (254-280 tok/s).
  - Flash-Next FP8 KV failed the teacher-forced check (NO_GO), yet on 15 real Duck tasks FP8 scored 13/15, the same as BF16.
- **Missing context bands:**
  - None of the 27B steady sweeps has a ~120k band, except E23 G0, NVFP4 KV and Merlin TP1/TP4.
  - The W4 levers S, QP, I8, P26 and C12 were measured at 8k/40k only.
  - DFlash2 D4s and DFlash2 on W4 were never measured.
  - The E29 D3s production engine has only a 12-task smoke.

Full content of the file:

# Engine configuration inventory: Kaggle 4x NVIDIA L4, ARC-AGI-2 project (2026-11-04)

Scope: every inference-engine configuration that our team ran on a Kaggle 4x L4 GPU kernel (sm_89, 23,034 MiB per card, PCIe, no NVLink) for the ARC-AGI-2 project, with the number and the file it comes from. Configurations that were only built or planned are listed separately at the end. Read-only inventory: nothing was re-measured.

Counts: **124 measured configurations** = 84 on the Qwen3.8-27B track + 40 on the Qwen3.8-Flash-Next track. A row is one distinct engine configuration; several runs of the same configuration (different workloads or days) share one row. "Measured" includes runs that crashed, OOMed or produced garbage on the GPU, since those are measured outcomes.

## Conventions

- `B@N` = context band B (prompt tokens) at N real running streams. tok/s = aggregate generated (accepted) tokens per second in a steady window unless the cell says "window" (whole solve window), "whole" (whole run) or "p50" (median of server stats lines). "/s" after a number = per stream.
- AL = mean acceptance length (accepted tokens per verify step, including the bonus token). "acc/draft" = accepted tokens per draft step (AL - 1).
- task-norm = per-task normalised eval score (%), "q" = eval query hits. LB = Kaggle public leaderboard.
- Dates are the GPU run dates given in the source (UTC dates).
- All paths are relative to `D:/ARC AGI 2/`. `E<n>` = `experiments/Experiment <n> - .../`.

### Source keys

| Key | File |
|---|---|
| E1R | `E1/README.md` |
| PC | `E14/Notes/Speed review 2026-10-24/part_C_measured_runs_and_insights.md` (§1.13 rows 178-195; secondary quotes of older runs) |
| A1 | `E25/Experiment Review/Past work mining/A1_vllm_engine_speed.md` (mining of E14 VMS / E18 / E19 / E20 / E23; primary paths listed in its "Evidence keys") |
| A2 | `E25/Experiment Review/Past work mining/A2_sglang_speculative.md` (mining of E11 / E14 / E15 / E17; primary paths in its source index) |
| A3 | `E25/Experiment Review/Past work mining/A3_flashnext_history.md` (mining of E3 / E13; primary paths per row) |
| EMS | `E14/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md` (EMS SAFE, EMS RISKY v1, VMS) |
| E11-03 | `E11/Experiment Results/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md` + `E11/Experiment Results/E11-03 run1 analysis (2026-10-17).md` |
| E11-04 | `E11/Experiment Results/Kaggle Runs/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/report.md` + `E11/Experiment Results/E11-04 run1 analysis (2026-10-18).md` |
| E11-05 | `E11/Experiment Results/E11-05 run1 analysis (2026-10-18).md` |
| E11-06 | `E11/Experiment Results/E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md` |
| N30 | `E14/Notes/30 Claude - E14A SLIM R2 run results (2026-10-24).md` |
| CENG | `E14/Notes/EVAL120 analysis 2026-10-24/C_engine.md` |
| E15N11 / E15N10 / E15N17 / E15N20 | `E15/Notes/11 Claude - 45K30K and 40K35K GPU commit results (2026-10-25).md` / `.../10 Claude - 2X35K GPU run findings and rev 3h (2026-10-25).md` / `.../17 Claude - Duck vs Poetiq vs Direct oracle and overlap (2026-10-25).md` / `.../20 Claude - E15P rev 2 EVAL120 6-h result (2026-10-26).md` |
| E17S | `E17/Experiment Results/README_E17_SHORT_GPU_RUN1_FOR_GPT_2026-10-26.md` |
| E18S | `E18/Experiment Results/E18_SMOKE_GPU_v1_2026-10-27.md` |
| E18R2 | `E18/Experiment Results/GPU_commit_R2_2026-10-29/` (`vllm_server_s1.log`, `e18_engine_stats.json`) |
| E19 | `E19/Experiment Results/E19_GPU_run1_results_2026-10-27.md` |
| E20 | `E20/Experiment Results/E20_LB_analysis_2026-10-29.md` |
| E21S | `E21/Experiment Results/E21_SMOKE_GPU_run1_2026-10-29.md` |
| E21R | `E21/README.md` |
| E23S | `E23/Experiment Results/smoke_rev3g/SMOKE_REV3G_RESULTS_2026-10-30.md` (+ `SMOKE_A/`, `SMOKE_B/e23_results.json`) |
| E23F | `E23/Experiment Results/E23_FULL_esfersami50_v1_2026-10-31.md` |
| MEM | user auto-memory `C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/` (`MEMORY.md` index; `experiment-20-s1-drain15.md`; `experiment-21-duckb-vllm.md` final-results table; `our-best-lb-54.md`; `dflash2-over-mtp-decision.md`) |
| RC1 | `E13/Experiment Results/E13_RC1_GPU_run1_2026-10-31.md` |
| RC2 | `E13/Experiment Results/E13_RC2_GPU_v1_2026-10-31.md` |
| NB01 | `E25/Experiment Results/E25_GPU_wave1_NB0_NB1_2026-10-31.md` |
| NB1R2 | `E25/Experiment Results/E25_GPU_NB1_R2_2026-10-31.md` |
| NB23 | `E25/Experiment Results/E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md` |
| NB4S | `E25/Experiment Results/E25_NB4S_GPU_v1_2026-11-01.md` |
| NB4T | `E25/Experiment Results/E25_NB4T_GPU_v1_2026-11-01.md` |
| NB5 | `E25/Experiment Results/E25_GPU_NB5_QUALITY_2026-11-01.md` |
| NB6 | `E25/Experiment Results/E25_NB6_COMMIT_v1_2026-11-01.md` |
| SPD | `E25/Experiment Review/Agent Reports/E25_REPORT_speed_measured_data_2026-11-01.md` |
| K3v1 / K3v2 / K100Q | `E26/Experiment Results/E26_K3ONLY_GPU_v1_2026-11-01.md` / `..._K3ONLY_GPU_v2_2026-11-01.md` / `..._K100Q_GPU_v1_2026-11-01.md` |
| E27R | `E27/README.md` (change log) |
| NBDF-W4 | `_scratch/e27_gpu_out/q27-xhigh-e27-nbdf/nbdf/nbdf_results.json` |
| NBDF-K | `_scratch/e27_nbdf_k100_gpu_out/q27-xhigh-e27-nbdf-k100/nbdf/nbdf_results.json` |
| NBS1-G / NBS1-S / NBS1-F | `_scratch/e25_nbs1_gpu_out/nbdf/nbdf_results.json` (start 2026-11-01 17:04 UTC) / `_scratch/e25_nbs1_smoke_out/nbdf/nbdf_results.json` (18:09 UTC) / `_scratch/e25_nbs1_full_out/nbdf/nbdf_results.json` (18:51 UTC) |
| E28K | `_scratch/e28_k100_out/q27-xhigh-e28-k100-eval120-6h/q27x/e28_k100_report.json` |
| E28W | `_scratch/e28_w4_out/q27-xhigh-e28-w4-eval120-6h/q27x/nb5_report.json` + `.../q27-xhigh-e28-w4-eval120-6h.log` |
| E29S | `E29/Experiment Results/E29_SMOKE_hosen42_v1_2026-11-02.md` |
| E31 | `_scratch/e31_hamid_out/q27x/e31_direct_report.json` + `.../e28_k100_report.json` |
| F6 | `E25/Experiment Results/F6_breakdown_2026-11-03.md` |

Rows citing A1 / A2 / A3 / PC are secondary: those files are our own mining reports that quote the primary logs/JSONs named in them. Rows citing EMS, RC2, NBxx, E26, the `_scratch` JSONs, E21S, E23S/F were read directly for this inventory.

---

## Table 1. Qwen3.8-27B track (84 configurations)

All rows: Qwen3.8-27B (dense GDN hybrid: 48 GDN + 16 full-attention layers), Kaggle 4x L4. "FP8" = our repacked block-FP8 checkpoint `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`; "INT4" = RedHat W4A16 g128 (Marlin); "Merlin W4A16" = Merlin's 27B checkpoint.

### 1a. vLLM 0.19.0 (production era) - 15 configurations

| ID | Exp / date | Weights | Parallel | Spec | Key flags | Streams | Measured throughput (band@streams) | Accuracy / correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| T01 | E1 Q27-XHIGH v1.3.5-v1.4.4 (2026-10-06..12) | FP8 | TP4 | MTP3 (acceptance 0.661) | offline engine, fp8 KV, max_model_len 81,920, max_num_batched_tokens 4,096, libcuda shim (FlashInfer JIT) | 16 | v1.3.5 307 at 12 active / 247.8 overall; v1.4.1 285.3 (prefix hit 0.786); v1.4.4 274.6 (KV 78.7 %, 0 preemptions) | v1.4 LB 18.75; v1.4.4 EVAL12 3/22, LB 26.94; E7 ORDERING_ONLY v2 (v1.4.4 base) LB 27.92 | worked | E1R; PC row 184; MEM (`MEMORY.md`, `arc-agi2-qwen27b-project.md`) |
| T02 | "V" / S1 server (S1-60K, R5, S1COMMIT4; 2026-10-14..27) | FP8 | TP4 | MTP3 | fp8 KV, prefix caching (mamba align), FlashInfer, max_num_seqs 16 | 16 | decode by real running 1/5/10/12/16: 44/163/268/293/352 (~22/s at 16); 333.3 whole; R5 Poetiq EVAL120 280.9 whole (KV 99.5 %, 9 preemptions); E20 commit KV pool 1,353,600 | AL 3.0 (T 0.6) / 2.67 (T 1.0) / 3.45 (low effort); S1-60K LB 34.58; S1 family LB 32.08 / 31.81 / 31.94 / 31.53; R5 EVAL120 78/172 (task-norm 48.61), LB 21.39; E19 arm D (= S1) PASS 66/66 | worked | PC rows 178, 180-181; A1 §1c; E20; E19; MEM |
| T03 | L66 (2026-10-17) | FP8 | TP4 | MTP3 | V engine, max context 155,648 | 12 | 254.1 whole; KV peak 90 %, 0 preemptions (D66 at 16 / 77.8k: 255.8 whole, 66 preemptions) | D66 28 / L66 34 of 103 (MEM index) | worked | PC row 179; A1 §1c; MEM |
| T04 | E9 86k / E10 ORDER2 EVAL120 (2026-10-14..15) | FP8 | TP4 | MTP3 | Q27-XHIGH line; flags not restated in the source | 16 | E9 306.6 (KV 91.4 %); E10 ORDER2 333 | E9 LB 22.92; E10 ORDER2 EVAL120 task-norm 36.94 | worked | PC rows 188-189; MEM index |
| T05 | E2 R6 code loop | FP8 | TP4 | MTP | not restated in the source | — | 220.7 | MTP acceptance 0.678 | worked | PC row 185 |
| T06 | E19 arm E (2026-10-27) | FP8 | TP4 | MTP3 | S1 flags, async scheduling OFF | ~60 req | correctness only | PASS | worked | E19; MEM `experiment-19-vllm-correctness.md` |
| T07 | E19 arm K | FP8 | TP4 | MTP3 | S1 flags, BF16 KV | ~60 req | KV 176,800 (bf16) vs 337,600 (fp8) at E19 settings | PASS; MTP accept per position min / median 0.58 / 0.80 | worked | E19; A1 §1c |
| T08 | E19 arm C | FP8 | TP4 | none | prefix caching ON | ~60 req | correctness only | PASS | worked | E19 |
| T09 | E19 arm A (reference) | FP8 | TP4 | none | prefix caching OFF | ~60 req | correctness only | PASS (reference) | worked | E19 |
| T10 | E19 arm B | FP8 | TP4 | MTP3 | prefix caching OFF | ~60 req | correctness only | **FAIL**: 4 flips of 4-12 nat | wrong output | E19 |
| T11 | E21 smoke NB1 half A, fallback (2026-10-29) | FP8 | TP4 | MTP3 | = S1 engine (no SSM flag), Duck config B | 16 (11.68 real) | p50 301.5 / mean 269.7; KV 855 blocks x 1600; prefix hit 69.0 %; 0 preemptions | acc/draft 1.89; 0/22 (20-min window, no signal) | worked | E21S |
| T12 | E21 NB2 half A (smoke + full, 2026-10-29) | FP8 | TP4 | MTP3 | `--mamba-ssm-cache-dtype float16`, Duck B | 16 | smoke p50 303.9 / mean 291.8 (KV 1627 x 832 blocks, 11.38 real); full p50 327, prefix hit 88.6 % | full 52/172, task-norm 33.33 | worked | E21S; MEM `experiment-21-duckb-vllm.md` |
| T13 | E21 NB1 half A full (2026-10-29) | FP8 | TP4 | MTP3 | SSM fp16, Duck B | 18 | smoke-2 p50 312 (prefix 82 %); full p50 329, prefix hit 89.0 % | full 54/172, task-norm 34.31 | worked | MEM `experiment-21-duckb-vllm.md` |
| T14 | E21 smoke NB1 half A primary (2026-10-29) | RedHat INT4 | TP4 | MTP3 | Marlin W4A16 | — | ready 365 s | multilingual token soup | **garbage** | E21S |
| T15 | archive (older vLLM; version not stated) | FP8 | TP4 / TP2 | — | CUDA graphs vs eager; custom all-reduce; prefix caching at 64 seqs; TP2 | 8 / 64 | graphs 48 -> 129 tok/s at 8 seqs; custom AR at TP4 0.9995x; TP2 17k KV | prefix caching at 64 seqs killed the engine; TP2 failed | mixed | PC row 190 |

### 1b. vLLM 0.32.0 stock - 27 configurations

Bands in VMS rows: 8k = ~3.85k-token prompts, 40k = ~40.9k, XL = ~90.6k; 75-s windows, T 1.0, ignore_eos (A1 §1).

| ID | Exp / date | Weights | Parallel | Spec | Key flags | Streams | Measured throughput | Accuracy / correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| T16 | E14 VMS ref (2026-10-26) | FP8 | TP4 | MTP3 | fp8 KV, prefix (align), FlashInfer, chunk 4096, util 0.90, custom AR off; FULL_AND_PIECEWISE requested, downgraded to PIECEWISE | 16 | 8k 434 (27.1/s) / 40k 399 (24.9/s) / XL 281 @10; e2e 322 / 195; KV 1,019,504 | AL 2.96 / 3.05 / 3.08; gate PASS (BOUNDARY) | worked | EMS; A1 §1a |
| T17 | VMS | FP8 | TP4 | MTP3 | seqs 20 | 20 | 452 / 423 / 281 @10; KV 1,006,517 | PASS | worked | EMS; A1 |
| T18 | VMS | FP8 | TP4 | MTP3 | seqs 24 | 24 | 472 / 424 / 256 @9; KV 992,231 | PASS | worked | EMS; A1 |
| T19 | VMS | FP8 | TP4 | MTP3 | `--mamba-ssm-cache-dtype float16` | 16 | 450 / 414 / 312 @11; KV 1,123,048 | PASS | worked | EMS; A1 |
| T20 | VMS | FP8 | TP4 | MTP2 | — | 16 | 448 / 397 / 260; KV 1,056,871 | AL 2.47-2.54; PASS | worked | EMS; A1 |
| T21 | VMS | FP8 | TP4 | MTP4 | — | 16 | 399 / 381 / 267 @9; KV 962,267 | AL 3.38-3.53; PASS | worked | EMS; A1 |
| T22 | VMS | FP8 | TP4 | MTP3 | `--linear-backend triton` (W8A8) | 16 | 438 / 398 / 283 | PASS | worked | EMS; A1 |
| T23 | VMS | FP8 | TP4 | MTP3 | FlashInfer PCIe-IPC all-reduce | 16 | 297 / 278 / 206 (0.68x) | PASS | worked (slower) | EMS |
| T24 | VMS | FP8 | TP4 | MTP3 | prefix caching OFF | 16 / 2 | 8k 435; 40k 3 tok/s at 2 running; XL 0 | — | collapse at 40k | EMS |
| T25 | VMS | INT4 | TP4 | MTP3 | Marlin W4A16 | 16 | 495 / 434 / 321 @12; KV 1,297,432 | diagnostic only (no INT4 oracle) | worked | EMS; A1 |
| T26 | VMS | INT4 | TP4 | MTP3 | seqs 20 | 20 | 518 (25.9/s) / 439 / 332 @12; KV 1,284,445 | diagnostic | worked | EMS; A1 |
| T27 | VMS | FP8 | 2x TP2 (DP2) | MTP3 | — | 14-16 / 2 | 191 / 22; 146,756 KV per replica, 4 preemptions | — | KV-bound, unusable | EMS; A1 |
| T28 | VMS | INT4 | 2x TP2 | MTP3 | — | 16 / 2 | 483 / 92; KV 422,087 | — | KV-bound | EMS; A1 |
| T29 | E18 S1-60K on vLLM 0.32 (smoke v1 2026-10-27, R2 commit 2026-10-29, SMOKE30 v3 2026-10-30) | FP8 | TP4 | MTP3 | S1 flags + `--language-model-only`, max_len 106,496, real Poetiq T 1.0 | 20 | v1: 457-466 at ~7k -> 345-353 at ~30k; steady 382, whole 284; R2: steady mean 382.8, whole 279.2, drain phase 4.9 gen tok/s; v3: steady 410.9, whole 241.8; KV 1,110,415; prefix hit 7.0 % | AL 2.69 (per position 0.732 / 0.543 / 0.416); R2 local 7/7 (MEM); v3 never submitted | worked | E18S; E18R2; A1 §1a |
| T30 | E21 smoke NB1 half B (2026-10-29) | FP8 | TP4 | MTP3 | Duck B, s16 | 16 (11.65) | p50 305.5 / mean 259.3; KV 865 x 1600; prefix 54.2 % | 0/22 (no signal) | worked | E21S |
| T31 | E21 NB2 half B (smoke + full) | FP8 | TP4 | MTP3 | SSM fp16, Duck B | 16 | smoke p50 328.8 (KV 1663 x 832, prefix 55.1 %); full p50 278, prefix 68.3 % | full 53/172, task-norm 33.19 | worked | E21S; MEM |
| T32 | E21 NB1 half B (smoke-2 + full) | INT4 | TP4 | MTP3 | Duck B, KV 1.71M | 20 | smoke-2 p50 322 (prefix 56 %); full p50 252, prefix 61.6 % | coherent at 135k; full 43/172, task-norm 28.19 | worked | MEM |
| T33 | E23 G0 control (SMOKE_B 2026-10-30; FULL 2026-10-31) | FP8 | TP4 | MTP3 | FlashInfer, PIECEWISE | 16 / 20 / 12 / 8 | SMOKE 40k@16 435.7 (27.2/s, step 131.1 ms, KV 1,159,379); FULL 40k@16 359 / 355 / 353 / 361 (A/A), 40k@20 367-370, 90k@12 261-275, 120k@8 202-203 | SMOKE AL 3.569 (synthetic, medium render); A/A +-1 % | worked | E23S; E23F; A1 §1a |
| T34 | E23 G3 | FP8 | TP4 | MTP3 | FlashInfer + PR #60532 backport, FULL_AND_PIECEWISE (FULL in 29/29 scopes) | 16 | SMOKE 40k@16 448.9 (+3.0 %, KV 1,110,733); FULL 364.7 | AL 3.584; oracle NEAR_TIE | worked | E23S; E23F; A1 |
| T35 | E23 G1 (FULL) | FP8 | TP4 | MTP3 | FULL-graph variant G1 | 16 | FULL 40k@16 330.7 | — | worked (-7 %) | E23F |
| T36 | E23 G2 (FULL) | FP8 | TP4 | MTP3 | FULL-graph variant G2 | 16 | FULL 40k@16 334.9 | — | worked (-7 %) | E23F |
| T37 | E23 MTP0 (target only) | FP8 | TP4 | none | FULL + PIECEWISE captured | 16 / 20 / 12 | FULL 40k@16 222.5, @20 248; 90k@12 142.6; KV 1,412,664 | oracle reference | worked | E23F; A1 |
| T38 | E23 MTP1 | FP8 | TP4 | MTP1 | — | 16 / 20 | FULL 40k@16 319.2, @20 354 | — | worked | E23F |
| T39 | E23 NVFP4 KV | FP8 | TP4 | none | NVFP4 KV cache | 8 / 12 | SMOKE 120k@8 133.0 (16.6/s, step 60.2 ms); FULL 90k@12 169.8; KV 2,356,868 (1.67x) | TF 40k rank-1 0.9766, p99 2.451 (> 3x limit) -> FAIL; 90k rank-1 0.945; NVFP4 + MTP3 skipped by gate | worked, accuracy gate FAIL | E23S; E23F |
| T40 | E23 PP4 | INT4 | TP1 x PP4 | none | eager | — | KV 1,374,799; 2k EXACT | 8k: CUDA illegal memory access | **crash** | A1 §1a (E23B `e_pp4_int4_eager`) |
| T41 | E23 TF reference | FP8 | TP4 | none | BF16 KV (`auto`) | — | KV 619,202 | TF reference (FP8 KV vs BF16 KV at 40k rank-1 0.98, mean 0.06) | worked | A1; E23F |
| T42 | E23 INT4 MTP0 reference | INT4 | TP4 | none | — | — | KV 1,788,404 | CLEAN, oracle saved | worked | A1; E23S |

### 1c. Merlin v3.0.0 (vLLM 0.32.0 fork), Merlin W4A16 27B - 3 configurations

Profile flags (TP4): block 1024, util 0.96, `kv_cache_memory_bytes` 16,301,686,784, fp8_e4m3 KV, prefix caching, SSM fp16, mamba align, prefix_match_unit 16, shared-prefix checkpoint, ReplaySSM, chunk 4096, seqs 24, MTP4 probabilistic draft, FULL_AND_PIECEWISE requested (draft decode ran NONE; target downgraded to PIECEWISE with FlashInfer); server default reasoning_effort "medium" (A1 §1b).

| ID | Exp / date | Parallel | Spec | Streams | Measured throughput | Accuracy / correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|
| T43 | E23 SMOKE_A Merlin TP1 (2026-10-30) | TP1 (GPU0) | MTP4 | 2 / 1 | 40k@2 93.9 (47.0/s, step 92.4 ms); 120k@1 41.0; KV 220,897 | AL 4.338 / 3.924 | worked | A1 §1b; E23S |
| T44 | E23 Merlin TP1 x4 replicas (SMOKE 2026-10-30, FULL 2026-10-31) | 4 x TP1 | MTP4 | 8 / 16 | SMOKE 40k (2 per replica) 359.5; FULL 40k@16 503.8 / 506.8 / 506.2, 90k@8 262.4 / 264.9 / 266.7 | A/A +-0.6 % | worked | A1; E23F |
| T45 | E23 Merlin TP4 (SMOKE + FULL) | TP4 | MTP4 | 8 / 12 / 16 / 20 / 24 | SMOKE 40k@8 421.5 (52.7/s, step 79.4 ms), 120k@8 329.0; FULL 40k@16 456, @20 472, @24 503; 90k@16 364; 120k@12 324; KV 1,789,034 | SMOKE AL 4.18-4.23; FULL accept 3.6-3.76; TP4 vs TP1 greedy 90k EXACT, TF 40k mean 0.0772, rank-1 0.9883; FULL 90k@20 / 120k@16 stalled stream (1 preemption) | worked | A1 §1b; E23S; E23F |

### 1d. SGLang 0.5.21, RedHat INT4 27B, TP4 - 39 configurations

DFlash2 = `z-lab` DFlash2 drafter for the 27B (BF16, `--speculative-draft-model-quantization unquant`); "W" = verify width. Bands in EMS rows as in VMS.

| ID | Exp / date | Spec | Key flags | Streams | Measured throughput | Accuracy / correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|
| T46 | E11-03 A (2026-10-17) | none | fp32 SSM, extra_buffer, auto mem 0.62 | 1/4/8/12/16 | ~4.3k: 45.6 / 145 / 229 / 289 / 347; long 12 x 24,576 out 282.4; KV 604,274 | — | worked (24/32 requested capped at 22) | E11-03; A2 §1.1 |
| T47 | E11-03 B | NEXTN MTP 3/1/4 | fp32 SSM, auto | 1/4/8, capped 10 | 55.1 / 205 / 297; "16" = 314.5 (really 10); long 262.3; KV 474,203 | AL 3.17 / 3.10 / 3.09 / 3.11 / 3.17 | worked, silent cap 10 (mamba pool) | E11-03; A2 |
| T48 | E11-03 C | DFlash2 W8 | fp32 SSM, auto | 1/4, capped 6 | 93.7 / 250; "8-32" 268-307 (really 6); 4 x ~85.5k ctx ~238; KV 394,559 | AL 3.89-4.95; T1.0 4.34 / T0.6 4.60 | worked, silent cap 6 | E11-03; A2 |
| T49 | E11-03 D | DFlash2 W8 | fp16 SSM | 1/4/8/12/16(13) | 107.5 / 278 / 356 / 404 / 381; KV 386,563 | AL 4.31-4.85 | worked (cap 13) | E11-03; A2 |
| T50 | E11-04 W1 (2026-10-18) + production E11-05 / E11-06 | DFlash2 W8 | fp16 SSM, extra_buffer_lazy, mamba 56, no prefill graph, mem 0.83 | 1/4/8/12(11) | sweep 110.6 / 257.5 / 363.8 / 325.9; PROD 12 real 416.5 (<10k 458, 10-20k 413, 20-40k 388); E11-05 EVAL30 11 real 328.9 whole; E11-06 379.9 (87.2 % cached); E11-05 S60 6H partial 325-405; KV 881,336 | AL ~4.5 (E11-05 median 4.55); marker 8/8; E11-06 P1-60K LB 29.17 | worked | E11-04; E11-05; E11-06; A2 §1.1-1.2; MEM |
| T51 | E11-04 W2 | DFlash2 W8 | as W1 with bf16 SSM | 12 | 412.1 (466 / 402 / 375 by ctx); KV 881,336 | sanity 7/8 (W1 8/8) | worked | E11-04; A2 |
| T52 | E11-04 W3 | DFlash2 W8 | as W1 with fp32 SSM | 1/4/8 | 100 / 273.5 / 337.9; KV 598,712 | AL 4.15-4.79 | worked | E11-04; A2 |
| T53 | E14A SLIM R2 W8 (2026-10-24) | DFlash2 W8 | — | 12 | ~5.9k 505.0 (step 114.0 ms); long (~40.9k) 449.8 (123.1 ms) | AL 4.86 / 4.62 | worked | N30; A2 |
| T54 | E14A R2 W6 | DFlash2 W6 | — | 12 | ~5.9k 531.4 (96.4 ms) | AL 4.24 | worked | N30; A2 |
| T55 | E14A R2 W4 | DFlash2 W4 | — | 12 | 526.8 (74.5 ms) | AL 3.27 | worked | N30; A2 |
| T56 | E14A R2 target-only | none | — | 12 | 339.6 (35.3 ms) | — | worked | N30; A2 |
| T57 | E14A R2 #44401 | DFlash2 W8 | #44401 backport | 12 | 508.1 | AL 4.84 | worked | N30; A2 |
| T58 | E14A R2 nodelay | DFlash2 W8 | `--min-free-slots-delay 1`, graph max 12 | 12-13 | 490.3 | AL 4.63; `tensor a (12) must match b (13)` at 13 running | **crash** at 13 | N30; A2 |
| T59 | E14A R2 graphs13 | DFlash2 W8 | graphs 1..13 | 12 | 500.1 | AL 4.79 | worked | N30; A2 |
| T60 | E14A R2 int8 mamba | DFlash2 W8 | int8 Mamba checkpoint | — | — | allocate 200 MiB, free 89 MiB | **OOM** | N30; A2 |
| T61 | EMS ref (SAFE + RISKY, 2026-10-26) | DFlash2 W6 | lazy, mamba 68, mem 0.83, lpm, page 1 | 16 | SAFE 583 / 521 / 370 @8 (step 115.7 / 127.3 / 86.8 ms); RISKY 572 / 544 / 403 @8; KV 852,152 | AL 4.21 / 4.11 (SAFE), 4.14 / 4.33 (RISKY); gate PASS (BOUNDARY), XL PASS, canary 0 | worked | EMS; A2 |
| T62 | EMS | DFlash2 W6 | seqs 20, mrr 21, mamba 84 | 20 | 597 / 536 / 364 @7; KV 779,192 | AL 4.17 / 4.21; PASS | worked | EMS |
| T63 | EMS | DFlash2 W4 | — | 16 | 592 / 526 / 363 @9 (step 89.2 / 100.9 ms); KV 917,816 | AL 3.30 / 3.31; PASS | worked | EMS; A2 |
| T64 | EMS | DFlash2 W8 | — | 16 | 573 / 477 / 363 @7; KV 786,488 | AL 5.00 / 4.47; PASS | worked | EMS; A2 |
| T65 | EMS | DFlash2 W6 | seqs 24 | 24 | — (KV 706,232) | out of memory + traceback, CAPPED | **OOM** | EMS; A2 |
| T66 | EMS | DFlash2 W6 | FlashInfer 0.7 PCIe-IPC all-reduce | 16 | 230 / 202 / 233 @8 (0.39x) | PASS | worked (slower) | EMS |
| T67 | EMS SAFE / RISKY | EAGLE MTP 3/1/4 | — | 16 | 478 / 402 / 296 (SAFE), 492 / 396 / 292 (RISKY); KV 1,142,688 | AL 2.9-3.0; PASS | worked | EMS; A2 |
| T68 | EMS RISKY | MTP 3/1/4 | + #45151 (duplicate embed / lm_head freed) | 16 | 492 / 394 / 295 @12; KV 1.29M | AL 2.98 / 2.90; PASS | worked | EMS |
| T69 | EMS | MTP 3/1/4 | + PCIe-IPC | 16 | 248 / 205 / 192 (0.42x) | PASS | worked (slower) | EMS |
| T70 | EMS | MTP 2/1/3 | — | 16 | 509 / 403 / 269 (step 77.7 / 97.6 ms); KV 1,183,245 | AL 2.47 / 2.45; PASS | worked | EMS; A2 |
| T71 | EMS | MTP 2/1/3 | seqs 20 | 20 | 542 / 413 / 263; KV 1,120,156 | AL 2.48 / 2.45; PASS | worked | EMS; A2 |
| T72 | EMS | MTP 3/1/4 | mem 0.86 | 16 | 483 / 408 / 296; KV 1,223,053 | AL 2.98 / 3.00; PASS | worked | EMS; A2 |
| T73 | EMS RISKY | MTP 3/1/4 | ReplaySSM fp16 (no_buffer) | 16 | 497 / 410 / 313 | AL 2.91 / 2.90; PASS (#43387 / #43749 risk) | worked | EMS |
| T74 | EMS RISKY | MTP 3/1/4 | ReplaySSM fp32 | 16 | 501 / 415 / 288 | AL 2.91 / 2.98; PASS | worked | EMS |
| T75 | EMS RISKY | MTP 3/1/4 | ReplaySSM fp32, seqs 20 | 20 | 543 / 428 / 275 | AL 3.01 / 2.95; PASS | worked | EMS |
| T76 | EMS RISKY | MTP | NTN flags on INT4 | 16 | 444 / 363 / 230 @7 (0.78x) | AL 2.95 / 2.88; PASS | worked | EMS |
| T77 | EMS RISKY | MTP | NTN flags on FP8 weights | 16 / 4 | 400 / 343 / 151 @4 (0.70x); KV 419k | diagnostic | worked | EMS |
| T78 | EMS SAFE / RISKY | none | target-only | 16 / 15 | 405 / 354 / 275 @15 (SAFE), 400 / 348 / 274 (RISKY); step 39.5 / 45.1 / 54.6 ms; KV 1,576,170 | oracle | worked | EMS; A2 |
| T79 | EMS RISKY | DFlash2 W6 | 2 x TP2 replicas | 8 / 3 per replica | 8k 658 (sum), 40k 308 (sum); 152k KV per replica | PASS | KV-bound at 40k (0.57x) | EMS |
| T80 | EMS RISKY | MTP ReplaySSM fp32 | 2 x TP2 replicas | 8 / 4 | 511 / 279 | PASS | KV-bound | EMS |
| T81 | E12 Duck EVAL120 G4B XE16 / XNO / LOW (analysed 2026-10-24) | DFlash2 W6 | mrr 13, graphs 1..13, mamba 56, delayer default | 12 offered, 11.21 mean | whole 344.0 / 341.4 / 317.8; pure decode at 12: 434.5 / 434.9 / 445.2; KV 917,816; prefill 19-27.5 % of window | AL p50 3.9 / 3.8 / 4.0 | worked (11-of-12 delayer 73 % of time) | CENG; A2 §1.2 |
| T82 | E15B Duck config B (45K30K commit 2026-10-25; R3K1 / R3K2 EVAL120) | DFlash2 W6 | delayer off (`--min-free-slots-delay 1`), mrr 17, graphs 1..17, mamba 68 | 16 | gen p50 by running 16: 451, 15: 470, 14: 433, 12: 383, 8: 312, 2: 144, 1: 78; 2X35K partial 257-311 then 430-442; KV 852,152, cache hit 94 % | AL p50 3.8; EVAL120 task-norm 33.47 (R3K1) / 39.03 (R3K2) | worked | E15N11; E15N10; E21R; A2 |
| T83 | E15P Poetiq EVAL120 6 h (2026-10-26) | DFlash2 W6 | E15B engine, 15 streams | 15 | 389 whole over 20,127 s | AL median 3.4; EVAL120 64/172 (task-norm 39.31); LB 26.94; 0 retractions | worked | E15N20; A2; MEM |
| T84 | E17 SHORT (2026-10-26) | DFlash2 / none (small-slot repro arms L1/L7/L5/L9) | 4-slot server repro arms | — | — | all four arms `Can not alloc mamba cache` (#43559 path) before the #43567 path; verdict NOT_EXPOSED_OR_INCONCLUSIVE | **crash** | E17S; A2 §3 |

---

## Table 2. Qwen3.8-Flash-Next track (40 configurations)

Model facts (A3): 48 layers (36 GDN + 12 QSA), 512 experts top-10, PLE n-gram table (host), one MTP layer.
- **W4** = Merlin / Intel W4A16 checkpoint (+ FP8 PLE overlay).
- **klee100** = `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` (routed experts W3 g128 / W2 g64 on humming kernels, MTP W4).
- **Merlin base** (rows F11-F40 unless the row says otherwise): Merlin v3.0.0 (vLLM 0.32.0 fork), profile `fn-quad-24g-sm89`, TP2 x PP2 + EP, partition 25/23, V2 runner, async scheduling, FULL_AND_PIECEWISE target graphs (draft decode FULL_DECODE_ONLY), RecoverSSM (commit eager), prefix caching + mamba align (unit 64), block 4096, chunk 1024, P2P off, PYNCCL, MTP K3 probabilistic (BF16 draft head) unless noted (RC2 §2.3-2.10).
- **Q-defer** = our `nb4_ppack` patch with `NB4_PPDEFER=1`: deferred PP receive of the sampled tokens (NB4S).
- **I8** = `DRAFT_HEAD=int8` (`draft_lm_head_quant: int8`).
- **D3s** = DFlash2 drafter `incoai/Qwen3.8-Flash-Next-DFlash2` (repacked with embeddings, `foysalemonshanto/qwen38-flash-next-dflash2v2/bf16-embed`), K3, drafter sliding-window block 1024, BF16 drafter KV, FLASH_ATTN drafter backend, our `dflash_seam.py`.
- **EMBED_HOST** = `DFLASH_DRAFT_EMBED_HOST=1`.

### 2a. SGLang (Pennyroyal 0.5.19 / 0.5.21) and stock vLLM 0.32 - 10 configurations

| ID | Exp / date | Engine | Weights | Parallel | Key flags | Streams | Measured | Correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| F01 | E3 FN-SMOKE v3 (2026-10-05) | SGLang 0.5.19 Pennyroyal | W4A16 + FP8 PLE overlay (124.8 GB) | TP4 / EP4 | — | — | launch rc 1 after 15 s (MoE runner) | — | **launch fail** | A3 §1a (E3 `fn_smoke_report.json`) |
| F02 | E3 v3.3 (2026-10-05) | SGLang 0.5.19 | same | TP4 / EP4 | ctx 4096, running 1, chunk 1024, bf16 KV | 1 | weights 19.06 GB/rank; KV 19,968 then 15,424 tokens | — | **killed -9** at 930 s / 428 s | A3 §1a |
| F03 | E3 FN-PROBE v4 (2026-10-10) | SGLang 0.5.19 | same | TP4 / EP4 | no prefetch | — | not ready at 1,080 s (x2); hash 0.29 GB/s | — | **not ready** | A3 §1a |
| F04 | E3 v4.2.1 (2026-10-10) | SGLang 0.5.19 | same | TP4 / EP4 | eager, mem 0.96, running 8, chunk 2048, ctx 16384, bf16 KV, FA2 shim | 1 | 5.95 gen tok/s; KV 118,080; ready 862 s | output "!!!!" | **garbage** | A3 §1a |
| F05 | E13 SMOKE1 rev 6b (2026-10-26) | SGLang 0.5.21 | Franzen Intel package, BF16 PLE | TP4 + EP4 | — | — | 20.86 GiB allocated, 154.56 MiB free on TP0 | — | **OOM** (`gptq_marlin_moe_repack`) | A3 §1a; `E13/Experiment Results/E13_SMOKE1_rev6b_GPU_run1_2026-10-26.md` |
| F06 | E13 LOADPROBE bf16ple (2026-10-27) | SGLang 0.5.21 | BF16 PLE | TP4 + EP4 | fp8 KV (scale 1.0), mem 0.93, decode graphs | 1 | 40.8 tok/s; KV 94,528; ready 1,132 s | "!!!!" (token 0) | **garbage** | A3 §1a; `E13/Experiment Results/E13_LOADPROBE_GPU_run1_2026-10-27.md` |
| F07 | E13 LOADPROBE fp8ple | SGLang 0.5.21 | FP8 PLE overlay | TP4 + EP4 | same | — | +0.30 GiB per MoE layer not freed | — | **OOM** on TP1 (repack layers.8) | A3 §1a |
| F08 | E13 NANPROBE rev2 kv_lean (2026-10-30) | SGLang 0.5.21 | W4A16 | TP4 + EP4 | mem 0.95, mamba 20 slots, graphs on | 4 x 20k | KV 161,984; 4 x 20k prefill 37.6 s; max running 2 | "!!!!"; NaN from layer 2 experts (EP4 GPTQ-Marlin MoE) | **garbage** | A3 §1a; `E13/Experiment Results/E13_NANPROBE_rev2_GPU_run1_2026-10-30.md` |
| F09 | E13 NANPROBE nan_eager | SGLang 0.5.21 | W4A16 | TP4 + EP4 | eager + NaN hooks | — | first eager prefill 428 s; ~22 s per later forward (hooks) | diagnostic (NaN located) | diagnostic | A3 §1a |
| F10 | E13 RC1 vllm_fn (2026-10-31) | stock vLLM 0.32.0 | Intel W4A16 | TP4 + EP | eager, BF16 KV, util 0.93, max_model_len 8192, seqs 4, `--language-model-only`, engram cpu_offload | 1 (4 planned) | KV 30,208 tokens; ready 825 s; model 18.43 GiB/rank | first correct FN output; 4 x 256 stream test died (HTTP 500, FlashInfer sampler JIT `-lcuda`) | partial | RC1; A3 §1a |

### 2b. Merlin v3.0.0, W4 checkpoint - 17 configurations

| ID | Exp / date | KV | Pin (B) / pool | Changes vs Merlin base | Streams | Measured throughput (band@streams) | Accuracy / correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|---|
| F11 | E13 RC2 BF16 boot (2026-10-31) | BF16 | 2,464,415,744 / 78 usable blocks, 269,042 tokens | seqs 3, captures to 12 | short probes only | prefill 40k cold 4,409 tok/s; ready 411.1 s | CLEAN; MTP AL 3.057 (193 drafts) | worked | RC2 |
| F12 | E13 RC2 FP8 boot (2026-10-31) + NB2 SMOKE_CTRL (2026-10-31) | FP8 E4M3 QSA (static scales; MTP layer scale 1.0) | 2,479,838,700 / 149 blocks, 513,940 tokens | seqs 8, captures to 32 | 1-8 | 40k@1 70.0 (44.1 ms step), @2 107.4, @4 161.0, @5 190.5; 8k@8 261.8 (32.7/s, 48.2 ms); 90k@3 48.7 (inconclusive), 90k@4 159.8; 116k@3 140.5, @4 160.3; natural 116k single stream 58.8; CTRL 8k@8 259.1 | fp8_gate PASS, oracle NEAR_TIE, PASS_LONG (116,217-token natural row closed); AL 3.010 whole boot; TF 40k FP8 vs BF16 mean 1.8764, rank-1 0.3672 (UNDETERMINED in RC2) | worked | RC2; NB23 |
| F13 | E25 NB0 byte-decoder boot (2026-10-31) | FP8 via sm_86 byte decoder | — | TF-only boot | — | — | full-prompt FP8 vs BF16 mean 0.335 / 0.231 / 0.959 / 1.293 at 2k / 8k / 40k / 90k | worked (TF only) | NB01 |
| F14 | E25 NB0 MTP-off boots (BF16 KV and FP8 KV) | BF16 / FP8 | — | MTP off | — | — | AC / BD (MTP on vs off) = 0 up to 40k; 90k last256 0.61 / 0.45; FP8 verdict NO_GO_FP8_TF_DIVERGENCE (40k / 90k) | worked (TF only) | NB01 |
| F15 | E25 NB1 B1 (v1 + R2, 2026-10-31) | FP8 | 2,479,838,700 / 149 | + `--language-model-only`, #60490 backport, profiler | 3-8 | v1: 8k@3 141.1, 8k@8 280.0, 40k@5 195.3, 40k@8 248.6; R2: 8k@3 141.3, 8k@8 261.0 (48.2 ms), 40k@5 196.3, 40k@7 227.9; 90k@5 NOT_ALL_RESIDENT (1 preemption) | AL 3.04-3.18; model 16.89-16.91 GiB/card; profiler: NCCL 63.5 % of kernel time, stage-0 stall 20.96 ms/tick | worked (v1 post-processing crash, data partial) | NB01; NB1R2 |
| F16 | E25 NB1 R2 B2 | FP8 | 149 blocks | + SSM bf16, seqs 16, captures to 64, `MERLIN_SHARE_DRAFT_WORKSPACE=1`, `VLLM_RECOVERSSM_COMMIT_GRAPH=graph` | — | load tests skipped | oracle FAIL_HIGH_MARGIN (row 1 diverges at token 31, margin 0.75) | **NO_GO_ORACLE** | NB1R2 |
| F17 | E25 NB1 R2 B3 | FP8 | 3,553,580,524 / 214 blocks, 738,142 tokens | B1 + pin +1.0 GiB | 3-8 | 8k@3 141.2, 8k@8 254.6, 40k@5 195.6, 40k@8 249.5 (49.9 ms); 90k@5 ALL_RESIDENT 187.8 (0 preemptions) | AL 3.08-3.18; CLEAN | worked | NB1R2 |
| F18 | E25 NB2 SMOKE + SMOKE_SEED2 (2026-10-31) | FP8 | — | MTP K2 + INT8 draft head + seqs 16 + CAPS16 | 8 (16 admission test) | 8k@8 242.6 (42.7 ms tick) and 240.9 | AL 2.592 (per position 0.860 / 0.731); SMOKE: 1 of 8 streams looped ("row? row?") -> NO_GO_GARBAGE_UNDER_LOAD; SEED2 8/8 clean | worked; one loop not reproduced | NB23 |
| F19 | E25 NB4S arm S (2026-11-01) | FP8 | 3,553,580,524 | `CUDA_DEVICE_MAX_CONNECTIONS=32` | 8 | 8k@8 278.9 / 291.0 (mean 285.0, +11.6 %); 40k@8 281.3 (+13.4 %); stall median 10.84 ms | oracle EXACT vs B | worked | NB4S |
| F20 | E25 NB4S arm QP | FP8 | 3,553,580,524 | packed PP broadcast (`NB4_PPACK`) | 8 | 8k@8 301.9 / 301.5 (+18.2 %); 40k@8 294.3 (+18.7 %); stall 0.037 ms | EXACT | worked | NB4S |
| F21 | **W4 + Q-defer**: NB4S QD (2026-11-01), NB4T Q (2026-11-01), E27 NBDF W4 Q (2026-11-01) | FP8 | 3,553,580,524 / 215 blocks, 738,142 tokens | Q-defer (+ text-only, #60490) | 8 / 6 / 5 | NB4S QD 8k@8 305.7 / 312.5 (mean 309.1, +21.1 % vs NB4S B 255.3), 40k@8 297.1; NB4T Q 8k@8 304.6 / 301.1, 40k@8 287.4; NBDF-W4 Q 8k@8 287.5 / 305.2 (41.7 ms tick), 40k@8 294.6, 90k@6 236.0 / 244.8, 116k@5 205.3 / 216.3 | oracle EXACT (NB4S, NBDF-W4); long oracle 90k / 116k PASS; AL 2.96-3.20; 0 preemptions | worked | NB4S; NB4T; NBDF-W4; E27R |
| F22 | E25 NB4T I8 | FP8 | 3.29 GB (OOM at 3.55 GB) | Q + INT8 draft head | 8 | 8k@8 299.1 / 327.3 (mean 313.2, +3.4 %); 40k@8 320.2 (+11.4 %) | EXACT; AL 3.06-3.11 | worked | NB4T |
| F23 | E25 NB4T P26 | FP8 | 3.29 GB (OOM at 3.55 GB) | Q + `VLLM_PP_LAYER_PARTITION=26,22` | 8 | 8k@8 287.3 / 303.2 (mean 295.2, -2.5 %); 40k@8 301.5 (+4.9 %) | NEAR_TIE | worked (OOM at base pin) | NB4T |
| F24 | E25 NB4T C12 | FP8 | 3,553,580,524 | Q + 12 streams | 12 | 8k@12 386.1 / 389.6 (mean 387.9, +28 %); 40k@12 374.6 (+30 %); KV max 0.99 at 40k | EXACT; AL 3.14-3.17; 40k window flagged LOW_VARIETY on 1/12 streams (text not saved) | worked (garbage flag, likely false positive per source) | NB4T |
| F25 | E25 NB5 QUALITY FP8 (2026-11-01) + NB6 SUBMIT commit (2026-11-01) | FP8 | pin +1.0 GiB / 214 blocks, 738,142 tokens | no Q-defer, no `--language-model-only`, #60490; Duck config B | 9 workers | NB5 window 172.0, steady p50 225.0, per-request 39.5/s at 3 running -> 25.0/s at 9; NB6 window 102.5 (3 tasks, 2.12 mean running) | NB5 13/15 tasks, 18/20 q, task-norm 86.67; NB6 3/3; NB6 SUBMIT LB 54+ (our best LB); Duck AL 2.833 | worked | NB5; NB6; SPD; MEM `our-best-lb-54.md` |
| F26 | E25 NB5 QUALITY BF16 (2026-11-01) | BF16 | pool 112 | as F25 | 5 workers | window 158.0, steady p50 169.2; per-request 45.1/s at 3 running | 13/15 tasks, 18/20 q, task-norm 90.00 (McNemar p 1.0 vs FP8); Duck AL 2.957 | worked | NB5; SPD |
| F27 | E28 NB-W4 EVAL120 6 h (2026-11-01) | FP8 | 3,285,145,068 / 197 blocks (KV 679,504 logged) | Q-defer + I8 + C12, `MERLIN_SHARE_DRAFT_WORKSPACE=1`; A/B/C scheduler, T1 104k / B2 120k | 12 workers (10.39 server mean) | solver window 182.3 (21,604 s); steady-mixed p50 309.4; 2,028 preemptions; prefix hit 38.08 % | EVAL120 90/172 q (pass1 89, pass2 1), task-norm 53.75; AL 2.793; running check FAIL_RUNNING_NE_INFLIGHT | worked | E28W |

### 2c. Merlin v3.0.0, klee100 3bpw checkpoint - 13 configurations

All with FP8 E4M3 QSA KV (W4 static scales reused), `--language-model-only`, #60490, `MERLIN_SHARE_DRAFT_WORKSPACE=1` where the source lists it.

| ID | Exp / date | Pin (B) / pool | Spec | Changes | Streams | Measured throughput | Accuracy / correctness | Status | Source |
|---|---|---|---|---|---|---|---|---|---|
| F28 | E26 K3_ONLY v1 (2026-11-01) | 5,270,142,976 / 1,096,865 tokens (267.8 blocks of 4096), 8.37x at 131k | MTP K3 | — | probes | ready 468.1 s; free after ready 4,583 / 4,579 / 3,257 / 3,257 MiB | chat probes 3/3; probe AL 4.00 (9/9, tiny sample); method read-back defect -> early exit | worked (stopped by a false mismatch) | K3v1 |
| F29 | E26 K3_ONLY v2 (2026-11-01) | 7,933,526,016 / 479 blocks, 1,652,197 tokens, 12.61x | MTP K3 | no Q-defer | 16 / 24 | 20k@16 460.4 (28.8/s, 55.8 ms tick), 20k@24 562.0 (23.4/s, 68.7 ms), 40k@16 444.7 (57.2 ms), 60k@16 admission 431.1; prefill uncached 3,269-3,536 tok/s | AL 3.18-3.22; TF vs W4 BF16: rank-1 0.513 / 0.652 / 0.602 at 2k / 8k / 40k (LOWBIT_TF_DIVERGENT_SEVERE); self-repeat 40k rank-1 0.914 | worked | K3v2; SPD |
| F30 | E26 K100Q quality (2026-11-01) | 7,933,526,016 / 479 blocks | MTP K3 | Q-defer, Duck config B, 2 x 104k | 21 workers (16.47 mean) | window 385.0 (2,543,542 tokens / 6,600.4 s); median 471.6 at concurrency, 413.6 all busy; 0 preemptions; prefix hit 85.4 % | 23/30 tasks, 31/38 q, task-norm 80.0; AL 2.8; Q-defer ENGAGED (0.9976) | worked | K100Q |
| F31 | **klee100 + Q-defer + I8, 24 seqs**: E27 NBDF-K100 Q (2026-11-01), E28 NB-K100 EVAL120 6 h (2026-11-01), E31 Direct30 (2026-11-02) | 7,933,526,016 / 479 blocks, 1,652,197 tokens | MTP K3 + INT8 draft head | Q-defer; captures 4..96 | 24 | NBDF-K Q: 8k@16 574.9 / 497.8, 20k@24 658.2 (27.4/s, 58.7 ms), 40k@16 562.0, 90k@8 339.2, 116k@6 311.7; E28: solver window 419.5 (21,599.8 s, 23.41 mean running, 9,062,803 tokens), 438 preemptions, prefix hit 79.79 %; E31 Direct: 407.1 (16.1 mean running), 27 preemptions, prefix hit 3.49 % | NBDF-K: long oracle PASS, oracle INCONCLUSIVE_NO_ORACLE, 116k@6 LOW_VARIETY flag -> NO_GO_GARBAGE_UNDER_LOAD; greedy AL 2.971; E28 EVAL120 107/172 q (pass1 100, pass2 7), task-norm 65.0; E31 Direct 11 q, task-norm 25.0 | worked | NBDF-K; E28K; E31 |
| F32 | E27 NBDF-K100 D3s (2026-11-01) | 7,933,526,016 | DFlash2 K3 (old seam) | Q-defer | — | — | LAUNCH_FAILED (probes missing) | **launch fail** | NBDF-K |
| F33 | E27 NBDF-K100 D4s | 7,933,526,016 | DFlash2 K4, SW 1024 | Q-defer | — | — | LAUNCH_FAILED | **launch fail** | NBDF-K |
| F34 | E27 NBDF-K100 D3s_ISO | 7,933,526,016 | DFlash2 K3 | Q-defer OFF | — | — | LAUNCH_FAILED | **launch fail** | NBDF-K |
| F35 | E25 NB-S1 (first GPU run) arm B (2026-11-01) | 7,933,526,016 / 1,605,632 tokens, 12.25x | D3s | Q-defer, no EMBED_HOST | — | ready 732.1 s | NO_GO_SAMPLED_PATH; OOM in log | **OOM / fail** | NBS1-G; E29 README ("without EMBED_HOST the first request OOMs on PP1") |
| F36 | **D3s + EMBED_HOST**: E25 NB-S1 SMOKE arm B + NB-S1 FULL arm B (2026-11-01) | 7,933,526,016 / 1,605,632 tokens | D3s | Q-defer, EMBED_HOST (retry after first-try OOM) | 6-24 | SMOKE: 8k@24 774.6 (32.3/s, 50.5 ms), 116k@6 326.7; FULL: 8k@16 630.8 / 609.8 (42.3 ms), 20k@24 763.0 (31.8/s, 50.7 ms), 40k@16 598.7, 90k@8 383.0 (47.9/s), 116k@6 324.0 (54.0/s); 0 preemptions | AL 3.11-3.31; long oracle PASS; oracle INCONCLUSIVE_NO_ORACLE (no MTP oracle in run); greedy AL 2.788; TF A/A 2k / 8k / 40k rank-1 0.856 / 0.961 / 0.660; Q-defer ENGAGED_FALLBACK (exec share 0.89) | worked | NBS1-S; NBS1-F; MEM `dflash2-over-mtp-decision.md` |
| F37 | E25 NB-S1 (first GPU run) arm F8 | 7,933,526,016 / 1,682,090 tokens | D3s variant | drafter KV fp8_e4m3, TRITON_ATTN drafter, SW block 2048, Q-defer, no EMBED_HOST | — | ready 285.1 s | NO_GO_SAMPLED_PATH; OOM in log | **OOM / fail** | NBS1-G |
| F38 | E25 NB-S1 (first GPU run) arm F8_ISO | 7,933,526,016 / 1,682,090 tokens | as F37 | Q-defer OFF | — | ready 255.0 s | NO_GO_SHORT_PROBE; OOM in log | **fail** | NBS1-G |
| F39 | E25 NB-S1 FULL arm F8 (2026-11-01) | 7,933,526,016 / 1,682,090 tokens, 12.83x | D3s variant | drafter KV fp8_e4m3, TRITON_ATTN, SW 2048, Q-defer, EMBED_HOST | 6-24 | 8k@16 634.0 / 628.2, 20k@24 767.9 (49.8 ms), 40k@16 610.0, 90k@8 387.5, 116k@6 317.6; 0 preemptions | CLEAN, oracle NEAR_TIE, long gate PASS, quality PASS; greedy AL 2.8927 | worked | NBS1-F |
| F40 | E29 DFLASH_D3S GPU smoke (2026-11-02) | first try 7,933,526,016 / 539 blocks -> second try 7,665,090,560 / 521 blocks | D3s | EMBED_HOST from first try, seam 341d859b + GRD, QSA logits 128 / 64, Q-defer, attempts_supply + DIRECT branch, 24 seqs | 24 (7.36 mean; 12 tasks) | solve window 262.4 (312 / 238 / 188 per 15 min); 130k first-prefill probe free [999, 995, 465, 465] MiB on try 1 -> LOW_FREE step | AL 2.823 (canary 2.845 / 2.817); prefix hit 62.6 %; 0 preemptions; eval 6/15 q, 5/12 tasks (44.44, not a quality measurement) | worked | E29S |

---

## Built, not measured on GPU

| Item | What exists | Source |
|---|---|---|
| E12 FP8 kernel tuning (E12-K1, vLLM 0.19 tuned block-FP8 configs / FP8 Marlin) | notebook + builder + local dry run; `Experiment Results/` empty | `E12 - FP8 kernel tuning/README.md` |
| E22 speed lab vLLM 0.32 + SGLang | CPU flag kit only (never on GPU) | A1 evidence keys (E22K); `E22/Experiment Results/cpu_kit_v1` |
| E24 SGLang speed track | plan + base-engine copies; no notebook | MEM `experiment-24-sglang-track.md`; `E24/Experiment Notebooks/` empty |
| E23 arms not run: EAGLE-4 (P6, CPU feasibility only), compiled PP4, NVFP4 KV + MTP3 (SKIPPED_CORRECTNESS_GATE), INT4 arms in FULL (INT4 not mounted), P5 host all-reduce inside vLLM (microbenchmark only) | — | E23S line 20-21; E23F |
| E27 NBDF on W4: D4s, D3s, D4, D4t, D7, D4b | all SKIPPED_DEPENDENCY (label gate on Q-defer read-back) | NBDF-W4 summary; E27R |
| E25 NB-S1 arms G1, G2, EH, DYN | SKIPPED_SWITCH_OFF in all three NB-S1 runs | NBS1-G/S/F summary |
| E25 NB2 FULL (K2 / INT8 head sweep) | deprioritised after SMOKE | NB23 |
| E25 NB4S combinations QP + QD, S + QD | "not measured here" | NB4S |
| E25 NB4T P27 | SKIPPED_P26_OOM | NB4T |
| E25 NB4U long-band confirmation (Q, I8, C12, C10/C11, I8 + C12) | planned | NB4T "Next" |
| E25 NB-M1 (MTP levers, vocab V32/V65) / NB-M1L / NB-S2 / NBD1 | NB-M1 copy A started on GPU (Q arm INCONCLUSIVE_NO_ORACLE at 60.2 min, then V65 arm launched); no results saved locally; user dropped MTP lever work 2026-11-02; others kits only | `_scratch/live_logs/q27-xhigh-e25-nbm1-a.jsonl`; MEM `dflash2-over-mtp-decision.md` |
| E28 B2 200k / max-model-len 262,144 plan | replaced by T1 104k / B2 120k / 131,072 before the run | `E28/README.md` change log |
| E29 DEFAULT, MTP_V64, ALL_OFF notebooks; E29_DFLASH_D3S full 6 h / 12 h run | built; only the D3S smoke (F40) ran | `E29/README.md`; E29S |
| E30 capacity port + generated-tail checkpoint | design + builder | `E30/README.md` |
| E32 lever bundle (seqs 28/32, D4s / DYN spec, LP, SPC, IDX, G1, G2) | SMOKE + FULL notebooks built; CPU kits | `E32/README.md`; `_scratch/kits_e32` |
| E33 Duck + Poetiq + Direct attempt chain | notebook + CPU kits; speed only simulated | `E33/Experiment Results/E33_SPEED_COVERAGE_ESTIMATE_2026-11-03.md` |
| E34 SSM bf16 + block 2048 capacity | SMOKE + FULL notebooks; `e34_capacity_probe.json` is a CPU computation on a Merlin source copy (not a GPU measurement) | `E34/Experiment Results/e34_capacity_probe.json`; `E34/Experiment Notebooks/` |

Kernel-only GPU microbenchmarks (measured, but not engine configurations, so not counted): E25 NB3 Marlin W4 MoE per-layer timings (T 1-128: 37-2,411 us, 204-252 GB/s), humming W3/W2 vs Marlin ratios, D7 world-2 host all-reduce (1.39-1.68x for 5-123 KB), cuBLAS BF16 baselines (NB23); E14 / E23 NCCL, P2P, host-staged and INT8 all-reduce, FP8 GEMM Marlin vs CUTLASS (A1 §1d).

Excluded by scope (not 4x L4): ARC-AGI-3 P1_SPARSE Flash-Next NVFP4 runs on 1x RTX PRO 6000 (~243-248 tok/s; A3 §1d).

---

## Gaps and contradictions

1. **Same nominal stock vLLM 0.32 FP8 MTP3 config, 40k@16, three different numbers:** VMS 399 (T16), E23 SMOKE G0 435.7, E23 FULL G0 353-361 (T33). E23H fix #4 (cited in A1 §1) states the SMOKE prompts were the medium render without the xhigh system line; VMS used other prompts and 75-s windows. The FULL vs SMOKE gap is not explained in E23F.
2. **Merlin 27B TP4 (T45):** SMOKE AL 4.18-4.23 vs FULL accept 3.6-3.76 at the same profile; SMOKE 40k@8 421.5 vs FULL 40k@16 456. The SMOKE prompts were medium render (A1 §4.12).
3. **Run-to-run spread on W4 FP8 8k@8 at the same config:** RC2 261.8, NB1 v1 B1 280.0, NB1 R2 B1 261.0, NB2 CTRL 259.1, NB4S B 255.3 / 254.0 (F12, F15, F19 base). Up to ~10 % between runs.
4. **Within-run window spread:** NBDF-K100 MTP Q 8k@16 574.9 vs 497.8 (-13 %, F31); NB4T I8 8k@8 299.1 vs 327.3 (F22); NBDF-W4 Q 8k@8 287.5 vs 305.2 (F21).
5. **klee100 MTP K3 throughput differs by build:** K3_ONLY v2 (no Q-defer, no I8) 20k@24 562.0 / 40k@16 444.7 (F29) vs NBDF-K100 Q (Q-defer + I8) 20k@24 658.2 / 40k@16 562.0 (F31). Q-defer and I8 were changed together, so the split is not measured.
6. **DFlash2 vs MTP on klee100 (MEM table)** compares NB-S1 FULL arm B (F36) with NBDF-K100 Q (F31) from different kernels on the same day; JSON values match the MEM table (631 / 610 vs 575 / 498; 763 vs 658; 599 vs 562; 383 vs 339; 324 vs 312). The DFlash arms have no in-run MTP oracle (INCONCLUSIVE_NO_ORACLE).
7. **DFlash2 on GPU first failed, then worked:** NBDF-K100 D3s/D4s/D3s_ISO LAUNCH_FAILED (F32-F34); NB-S1 first run NO_GO with OOM (no EMBED_HOST, F35, F37, F38); NB-S1 SMOKE/FULL worked after an OOM retry with EMBED_HOST (F36, F39); E29 smoke needed a pin step down (F40).
8. **Pool units differ between files:** E29 smoke reports pool 539 / 521 "blocks" and E34 probe `num_blocks` 540 for the D3s layout, while NB-S1 reports 1,605,632 KV tokens (12.25x at 131k). For klee100 MTP the same pin gives 479 blocks = 1,652,197 tokens (K3v2). Blocks counted across all KV groups and token capacity are not interchangeable.
9. **Fixed blocks per request (F):** Merlin's rule and the RC2 builder use ceil(L/4096) + 7; RC2 measured F = 6 (RC2 §2.2); F6 (2026-11-03) explains F = 6 as 3 Mamba groups x 2 pages. A3 §1b still quotes +7.
10. **klee100 pool estimate vs measurement:** E26 README estimate ~463 blocks; measured 267.8 blocks at pin 5.27 GB (F28) and 479 at 7.93 GB (F29).
11. **FP8 QSA KV quality:** teacher-forced NO_GO (NB0: 40k full rank-1 0.75, 90k 0.66; F14) vs real-Duck parity (NB5 FP8 13/15 = BF16 13/15, McNemar p 1.0, underpowered; F25 / F26). On the 27B, FP8 KV vs BF16 KV TF was rank-1 0.98 (E23F).
12. **SGLang vs vLLM LB gap (27B):** SGLang INT4 + DFlash2 Poetiq LB 29.17 (T50) / 26.94 (T83) vs vLLM 0.19 FP8 S1 family 31.53-34.58 (T02); per-request eval quality looked equal (A2 §3.9). Cause never isolated (E17 crashed, T84).
13. **vLLM 0.19 vs 0.32 under Duck:** smoke said equal speed (E21S: 301.5-328.8 p50), full runs said 0.32 slower (329 / 327 vs 252 / 278 p50) with lower prefix hit (89 / 88.6 % vs 61.6 / 68.3 %) (T12, T13, T31, T32).
14. **E18 (T29) steady-window spread on one engine:** 382 (smoke v1), 382.8 (R2), 410.9 (SMOKE30 v3); whole-run 284 / 279.2 / 241.8.
15. **W4 Duck speed quoted as "225 tok/s"** in E26/E28 docs and MEM is NB5's steady p50 (F25); NB5's whole-window rate was 172.0 and NB6's commit window 102.5 (3 tasks only).
16. **E28 runs vs their smaller predecessors:** E28 K100 438 preemptions and prefix hit 79.79 % (F31) vs K100Q 0 preemptions and 85.4 % (F30; different scheduler, 24 vs 21 workers, I8). E28 W4 (F27) 2,028 preemptions, prefix hit 38.08 %, window 182.3 at 12 workers vs NB4T C12 synthetic 387.9 (F24).
17. **Missing context bands:**
    - vLLM 0.19 production rows (T01-T05) have no per-band steady windows, only whole-run and by-running rates.
    - VMS (T16-T28) and EMS (T61-T80) stop at XL ~90.6k; no ~120k band.
    - E23 FULL MTP0 / MTP1 / G1-G3 (T34-T38) are 40k only; Merlin TP1 x4 (T44) has no 120k.
    - SGLang E11 / E14A sweeps (T46-T60) are at ~4.3-5.9k (E11-03 C has one 4 x 85.5k point).
    - W4 Flash-Next levers S, QP (NB4S) and I8, P26, C12 (NB4T) are 8k / 40k only (F19, F20, F22-F24); only Q (F21) has 90k / 116k.
    - NB2 K2 + INT8 head (F18) is 8k only.
    - klee100 K3_ONLY v2 (F29) has no 90k / 120k (KV-bound skips).
    - DFlash2 D4s and DFlash2 on W4 were never measured; the E29 D3s production engine has only a 12-task smoke (F40).
18. **Missing LB numbers:** no LB is recorded in the sources for E28 K100, E28 W4, E29 or any klee100 / DFlash2 notebook; the only Flash-Next LB in the sources is NB6 SUBMIT 54+ (F25, MEM `our-best-lb-54.md`).
19. **Partial / unrecorded GPU runs:** E31 foysal run (`_scratch/e31_gpu_out`) stops at 4.8 min after Merlin launch with no throughput; the E23 FULL INT4 arms were skipped; E25 NB1 v1 lost its B1 record to a post-processing crash (only the live windows survived).
20. **Engine flags not restated:** T04 (E9 / E10) and T05 (E2 R6) are listed with throughput from PC, but the source does not restate their exact engine flags; T15 (archive) does not state the vLLM version.