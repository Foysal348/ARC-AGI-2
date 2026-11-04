## NBDF-K100: final build ready for your kit push

The notebook is rebuilt to the final design, the units pass locally and in the units kit, and both kits and the audit zip are built. Nothing was pushed and git was not run (beyond the read-only `git status` slip I reported last round). Every file is LF (CR count 0).

The real kit's new Merlin-venv probe code (the KVPROBE / WEIGHTPROBE additions behind items 1–6) only compiles locally. Merlin's vllm will not import here (`cbor2` and other packages are missing), so your Kaggle run is its first real execution.

### Notebook
- **K100 (closed build):** `Q27-XHIGH-E27_NBDF_K100.ipynb`, sha **6b83cbbb2e35a4c3b1882b06786bf2f9fad42007a03d2314619cd17c21fa3e57**, **375,450 B**, 11 cells.
  - Closed-build sha = the same 6b83cbbb.
  - It is also in the GPU push folder `push_e25/q27-xhigh-e27-nbdf-k100__foysalemonshanto` (foysal, L4, private, with the tf-ref dataset and the klee100 + drafter Models).
- **W4 notebook changed too:** it is rebuilt because `nbdf_lib` changed. New sha 2abac1af0e48798498fb7c68db7f23c525fc0160f5e9adbbad3bfbce977b68cf, 364,707 B.
  - The gated W4 rebuild had grown to about 400,050 B, over the limit.
  - I packed W4 CELL 1's 54 kB lab literal as a sha-checked blob (same lab, sha ef393fbb). The W4 gate test is 11/11 again.

### Tests

| Run | Result |
|---|---|
| `nbdf_unit_tests.py` (shared lib + W4), local | 281/281 PASS |
| `nbdf_k100_unit_tests.py`, local | 43/43 PASS |
| `nbdf_k100_nan_tests.py`, local | 173/173 PASS |
| `nbdf_build_gate_test.py` | 11/11 PASS |
| K100 units kit, run locally | `@@KIT units ALL OK@@` (k100 37/37, NaN 173/173; the 6 checks that read the notebook skip there) |
| W4 units kit, run locally | `@@KIT units ALL OK@@` (279/279, seam tests pass) |

The K100 units kit was 406,984 B at first, over the limit. Its files are now only what the K100 and NaN tests read. The shared lib's 281 tests and the seam tests run in the W4 units kit, on the same `nbdf_lib.py`.

### Kits (folders under `D:/ARC AGI 2/_scratch/push_e25/`)

| Kit | Bytes | sha256 |
|---|---|---|
| `q27-e27-nbdf-k100-kit-units__foysalemonshanto` | 174,034 | 5ac937ca7e25ff761a62740c990326c8ae291f2ecc9aceb11306eda147976f3e |
| `q27-e27-nbdf-k100-kit-real__foysalemonshanto` | 333,646 | 50dd0f5fcd53322e9aa776c775b1fc0251b2ab338354c55a6e2448c555d2b551 |
| `q27-e27-nbdf-kit-units__foysalemonshanto` (W4) | 371,357 | 6b5a17b630d06a6b25abec54768a49d80fd72b8c40fef309c429b2448a061da7 |
| `q27-e27-nbdf-kit-real__foysalemonshanto` (W4) | 295,164 | 1a98edab6124f856fd7490baf6d71e028a4684f67a9e56035c4cc64e9f4dff52 |

### Audit zips (flat, in `Experiment 27/Experiment Review/Audit Zips/`)
- `E27_NBDF_K100_audit_FLAT_2026-11-01.zip`, sha 05d19dff3c2241e00ae21e2a2a8a5ab68d362528520c71bacc6727098be8541b. Its prompt states the final design. It also holds both research reports, `e28k100_lib.py`, the NaN test and the libraries.
- W4 zip refreshed: `E27_NBDF_audit_FLAT_2026-11-01.zip`, sha 3c427edc….

### What changed in the files (all in `Experiment 27/Experiment Notebooks/Build/`)
- **`build_e27_nbdf_k100.py`** (3075b447…)
  - `K100_CFG` holds the final design.
  - The CELL 2b additions carry the E28 / NB4T `i8_readback` functions, copied verbatim by AST (sha f534166c).
  - `edit_body` holds every body edit.
- **`nbdf_lib.py`** (42314240…), all additive. The W4 defaults are unchanged and the W4 units still pass.
  - New functions: `configure`, `_wkeys`, `k100_rel`, `q_aa_clean`, `tf_gate`, `pos0_label`, `cond_c`, `prefix_ratios`, `k100_records`.
  - The `seqs=SEQS` default-binding trap is fixed in `level_for_band`, `resident_level`, `plan_level` and `draft_logits_mib`.
  - `restrict_arms` now keeps the given order.
  - Q gets a trailing argv (`Q_TAIL`), and `check_arm_plan` accepts it.
  - `launch_check` has the S4 guard and allows the Q INT8 head.
  - The config read-back expects the INT8 head on an I8 boot.
  - An EAGER drafter is now a real failure (SPEC_MISMATCH).
  - The smoke ladder is judged relative to a non-deterministic Q.
  - The verdict:
    - accepts NONDET_BASE_RECORDED;
    - fails on the TF gate (CORRECTNESS_FAIL_TF);
    - records the prefix result only;
    - uses the W4 long A/A as a threshold floor;
    - keeps the measured verdict when the server dies in the last-in-boot TF (flag TF_SERVER_DIED_AT_END);
    - records the tap diagnostic.
- **`nbdf_k100_unit_tests.py`** (b8793132…), the W4 test **`nbdf_unit_tests.py`** (f6b027a1…; the T16 EAGER case moved to the real-failure list), and the subagent's `nbdf_k100_nan_tests.py` (e8959688…).
- **Kit builders:** `build_e27_nbdf_cpu_kit.py` (53d8bb88…) and `build_e27_nbdf_k100_cpu_kit.py` (d02893af…).
- The subagent's kit items: the 9 real-kit checks are in `_k100_sub` / `k100_item`, builder lines ~786–1393.
- Generated outputs: body `nbdf_k100_cell7_body.py` 60a682c0…, export `nbdf_k100_code_export.py` 7fe0b7d8….
- Unchanged: `dflash_seam.py` (7356d473); the W4 builder `build_e27_nbdf.py` is c206151c… (from the last round).

### How the user's gate rule is applied

**Real failures, which stop only their own arm:**
- crash or boot failure;
- garbage, request errors, short output;
- zero acceptance;
- the TF envelope;
- NaN (non-finite logprobs);
- SPEC_MISMATCH: wrong method, wrong K, wrong drafter, MTP built on a D arm, DFlash2DraftModel not resolved, an EAGER drafter;
- S4 (a quantized draft head on a dflash spec);
- missing init (`nb4_ppack` NOT_INTEGRATED);
- the draft load check;
- the klee100 boot gates (method map, static scales).

**Q-specific:** a failed I8 read-back is not a stop. It triggers the E28 same-pin retry with the BF16 head.

**Global setup failures** (merge, #60490, k100_setup, sha gate) boot no arm. That is not an arm-to-arm dependency.

**No cross-arm skips:**
- If Q is not a usable base, it is recorded and D3s and D4s still run.
- A broken tree before one arm is recorded, and each arm's own tree check decides.
- The reruns run after D4s.
- The closed-build KV kit gate still skips the D arms until your gated rebuild.

### Coordinator's later additions (both record-only)
- **KV-group metadata:** `k100_records.kv_groups` holds q_groups vs d_groups (KV_GROUP_EXPLOSION).
- **Metadata-build time:** `nb4_timeline` has no attention / GDN metadata-build span. The record points to the kept 30-s traces (`profile_8k@16`) instead, so this number needs an offline read.
- **Concurrency trend:** `k100_records.conc_trend` reports D/Q separately for 8k@16 and 20k@24, plus a direction (RISES or FALLS with concurrency).

### Research A → status

| Item | Status | Reason / where |
|---|---|---|
| B1 non-determinism | done | Q oracle run twice with a cache reset, plus Q's ladder = `q_aa`. `k100_rel` turns D's FAIL_HIGH_MARGIN / CONTAMINATION into NONDET_BASE_RECORDED when Q's A/A is not clean (oracle, long oracle, smoke ladder). TF gate at 2k/8k/40k against Q's A/A, +0.05 / −0.03. |
| B2 seqs 24 | done | `N.SEQS` = 24 and ladder (2, 8, 23) via `configure`. Q captures = registry `capture_sizes_for(24,3)` 1..96 as Q's trailing `--compilation-config`; D3s 4..96, D4s 5..120. Every arm gets `--max-num-seqs 24`. Default-binding trap fixed; `expect_cfg.max_num_seqs` = 24. |
| B3 pins | done | Every arm starts at 7,933,526,016. D OOM: EMBED_HOST at the same pin, then 7,665,090,560 → 7,396,655,104 → 6,859,784,192. Q OOM: 7,665,090,560. Never the −1 GiB step first. |
| Check 1: boot gates | done | `need_mtp` only on MTP arms; scales unchanged (unit-tested on the real lib). |
| Check 2: PP1 memory | done (recorded) | The memory gate is a flag only. |
| Check 3: KV packing | done | The MIXED path uses B = 4096. Recorded: residents at 40k/64k/90k/120k next to Q's 29/21/17/13, F at the boot's own block, the pool. |
| Check 4: humming at > 64 tokens | recorded | Ticks per window; the profile. |
| Check 5: seam | unchanged | — |
| Check 6: Q-defer / extension / workspace | done | `MERLIN_SHARE_DRAFT_WORKSPACE=1` set on every arm (inert with DFlash). |
| Check 7: I8 / long budget / tokenizer | done | I8 = Q only (D refused by S4). Admit limit computed at ≤ 3,000 tok/s. Tokenizer is real-kit item 8. |
| A recommendation: windows 8k@8, 20k@16 + 20k@24, paired long levels | changed | Replaced by the coordinator's list (decision 5). |
| A recommendation: hard gates | changed | Applied as the user's real-failure list. FULL-graph count is covered by the EAGER fallback. |
| A recommendation: memory ≥ 512 MiB gate | rejected | Record-only (user rule). |
| Kit R1–R10 | done | Subagent's 9 real-kit items plus R9_IDX. Item 10 (B1 / B3 fixtures) is in the K100 units. |

### Research B → status

| Item | Status | Reason / where |
|---|---|---|
| Q = E28 production with I8 | done | Switch `q_i8` (default True). I8_ENGAGED required, else the same pin with the BF16 head. |
| K3 default; order Q → D3s → D4s | done | — |
| D4s only if c ≥ 0.80 | rejected | Recorded only (`d4s_data_gate_c_ge_0_80`). D4s always runs. |
| No-hope cut (smoke AL3 < 2.40) | rejected | Recorded only. |
| Zero-accept canary | kept | Real failure, that arm only. |
| pos0 STRUCTURAL / TAP_SUSPECT | done | Recorded. |
| On STRUCTURAL: skip D4s | rejected | D4s always runs. The tap-mode diagnostic (`DFLASH_TAP_MODE=pre_combine`, smoke + oracle + 8k) runs at the end, budget permitting. |
| Pins in 0.25 GiB steps on OOM or on the memory gate | changed | OOM only, with the EMBED_HOST step first (A's B3); the memory gate is a flag. |
| Windows 8k@16×2, 8k@24, 40k@16, 90k@8, 116k@6 | changed | 8k@24 replaced by 20k@24 (coordinator decision 5). Long threshold floor = W4's A/A, 3.7 % at 90k and 5.4 % at 116k. |
| Prefix warm ≥ 0.9 × cold | changed | Record-only, with both the 0.8 and 0.9 ratios. |
| Correctness §4.5 | done | As A's B1. |
| Read-back list | done | Existing NBDF read-backs plus engine block, residents, KV groups, pos0. The memory gate is recorded. |
| Decision rule | changed | NBDF's verdict rule is kept. B's adoption criteria are left to the reader; the conc_trend / k100_records fields hold the data. |
| S4 guard | done | — |
| S1–S9 | covered | Seam read-backs, zero-accept canary, nometa rerun, tree checks. |
| §5 E28 adoption checklist | not in scope | It applies to `build_e28_k100.py`. |

### Priority list v4, section C
- **(a) NaN draft scores:** Merlin has no cheap read-back hook (no isnan / finite counter in `dflash*/`). Recorded as `NOT_AVAILABLE`, with no engine patch.
- **(b) NaN unit test:** done, in the units kit (173/173).
  - Finding: Merlin's original torch fallback leaks a NaN into the first row of the next block (through 0 × NaN).
  - The test asserts that leak as a negative control. The seam's torch path does not leak.
  - The fused CUDA kernel the arms use is not covered by any CPU test.
- **(c) S4 guard:** done.

### Open items for you
- Run the CPU kits, then do the gated rebuild with `--kv-gate-report`. The real kit's R9 now targets Q at 480 (479 usable).
- These are first runs on real klee100 data: the item 4 per-card split rule, item 8 shard reads and every real-kit probe.
- Before the GPU commit, you might want to settle whether the end-of-boot TF at 40k is safe on klee100 at seqs 24. It is the same TF the K3_ONLY v2 run did on klee100, at this pin.
