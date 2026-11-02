## E25 NB-D1 build report: closed build done. Nothing pushed to Kaggle, no dataset uploaded, nothing run on GPU.

The notebook, its 3 copies, the units kit and the real kit are built. The units kit passes in a local Py3.12 simulation; the real kit has not run (you run it). The build is **closed** (`gpu_ready` false, every arm SKIPPED_KV_KIT_GATE at run time) until the gated rebuild with the NB-D1 real-kit report.

No file outside the new `nbd1_*` / `build_e25_nbd1*` set was written:
- The builder asserts before/after byte equality of every Exp 27 Build file, both Exp 27 notebooks, all NB-S1/NB-S2 files and notebooks, and nb4t_lib, nb4_ppack, nb4_timeline and k100q_ref27b.json.
- The real-kit builder does the same check for Exp 27 and the NB-S1/NB-S2 kit builders.
- Exp 29, E31 and NB-S1 were only read.

All my files are LF (0 CR, checked).

### 1. Files
Build folder `D:\ARC AGI 2\experiments\Experiment 25 - Flash-Next Throughput 1500 (planned)\Experiment Notebooks\Build\`:

| File | sha256 | Bytes |
|---|---|---|
| nbd1_lib.py | af0f795f685fe63f6169e5171f25f5d35f25bf66cfc6538c1272554d3190bad3 | 50,304 |
| nbd1_body_add.py | f273d197a49e61f3a74a638d7ce41e70087b87c1117297287ca1cab803d8405b | 16,236 |
| build_e25_nbd1.py | dd69bca3a566b9335bbd0d7ab5d6e45a95fa9b71f4495888459b2cc5a659fd1e | 43,874 |
| nbd1_unit_tests.py | 3c6fe4784ea794f3faab7cb5f79c7f3033d9c3cc5a5c69b533f4967c2bdd2dc9 | 54,773 |
| build_e25_nbd1_units_kit.py | d96c0fd3c172174fc044d3a3316a2dd2374be28331693c41b03bfc6266b87831 | 7,919 |
| build_e25_nbd1_cpu_kit.py | 03d085aae81c6be25226acdc54fdc26e32c32098d721c4a4f64dd3c4815fcbbe | 35,504 |
| nbd1_log_fixture.json (NB-S1 GPU server-log lines + B's env / config / dflash read-backs; no generated text) | a213eba39d775d5f898229fc2dc2366aa46ca7419f452faafe9885fe68c4228a | 12,898 |
| nbd1_cell7_body.py (output) | 725319b1ffab68bfd75b266440558713440dfa474a3711ede9c592e1e9d60eac | 124,693 |
| nbd1_code_export.py (output) | c99e18e3db6b153b948b4e25e0357f256f7b1fadd75fdb384520ee5ad43e9297 | 328,341 |
| nbd1_build_facts.json (output) | b4d6c128d2c71c755554d7722558227a817e74c3daebfd51fbf63cda72c3255b | 59,667 |

Notebooks in `...\Experiment Notebooks\`. They are 11 cells, under 400 kB, and the copies differ only in the `NBD1_ON` line (asserted):

| Notebook | Role | sha256 | Bytes |
|---|---|---|---|
| Q27-XHIGH-E25_NBD1.ipynb | every key on; the kit notebook; the closed build | abbf21dfa9cda439de7e557e5782526512f9087c7e36a0c3f70625c30d13bb6d | 347,408 |
| Q27-XHIGH-E25_NBD1_A.ipynb | copy A | b8396eb269151c001e9de866e8a00d0a84606c41a8df58cbe4a68ed7d6f2e560 | 347,415 |
| Q27-XHIGH-E25_NBD1_B.ipynb | copy B | a952595a73db66e4365e4eb67e381db268e4cf13cfee7ac3ed70c8525822f8ba | 347,416 |
| Q27-XHIGH-E25_NBD1_C.ipynb | copy C | 50a83149f187ca07f9f72b65ac82d1e15bbc623521e7f117ff7c0bed0bdd0c92 | 347,417 |

The build is deterministic: a `--test-out` rebuild gives the same abbf21df.

### 2. How it is built
NB-S2's pattern:
1. The NBDF builder runs in-process twice. The closed output must be e2198285 and the gated one c20eba7f (asserted), with seam 341d859b and nbdf_lib b0147751 (asserted).
2. Exact-once text edits are applied to CELL 1, the CELL 7 body and CELL 8.
3. The body and nbd1_lib live in the payload dataset (sha-pinned loader `_nbd1_ds`; the env fallback is `NBD1_PAYLOAD_DIR`).
4. CELL 0 is E18 R2 verbatim. A competition rerun writes the identity submission.

**Base B = E29_DFLASH:**
- NBDF-K100 D3s: K3, SW-1024, BF16 drafter KV, FLASH_ATTN, Q-defer, seqs 24, pin 7,933,526,016, captures 4..96, no draft head.
- Plus `DFLASH_DRAFT_EMBED_HOST=1` from the first try on every arm.
- Plus `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` / `VLLM_E2_QSA_LOGITS_GRANULE_MB=64` in NB1_ENV. Those are applied after Merlin's `build()`; the export check confirms `env.update(NB1_ENV)` comes after `ML.build`. Merlin's launch.py:419-420 sets 256/256.
- `check_arm_plan` enforces that B's env differs from D3s by exactly these 3 keys.

**Body edits:**
- **OOM ladder:** `oom = bool(o["oom_in_log"]) and not o["xid_in_log"]`. The D ladder drops the same-pin EMBED_HOST rung (`D1.retry_plan`), so the ladder is 7.93G → 7.67G → 7.40G → 6.86G. The retry keeps the QSA/GDN keys in `expect`.
- **4-worker env read-back:** `RPC_KEYS` now includes the QSA and GDN keys, so the RPC returns them from all 4 workers. Every arm's `expect` includes them, so `launch_check`, `env_readback` and the kit's spawned child all check them.
- **Light design:**
  - no ladder, oracle, long oracle, prefix test or TF on any arm (recorded NOT_IN_PLAN_NB_D1);
  - `gated` runs on the hard-only evidence;
  - the prefill probe runs on B, or on a lever when B's rate was never measured;
  - the smoke hard gate (FAIL_*) stops any arm, B included.
- **PROF:** two 30-s torch.profiler windows (8k@16, 20k@24) inside B's boot. The `analyze_arm` graph spans run on both, stall/nb4_timeline on the 8k trace.
- **SPC:** the sibling probe runs (recorded).
- **Step-down:** a lever's KV-bound window steps down on any band.
- **[LIVE] line:** printed after every window, skipped ones included: `[LIVE <min>] <arm> <window> tok/s x | AL y | running n | preempt p | vs B +z %`. B itself, and a lever without a B window, prints `vs B (ref)` against the NB-S1 full numbers.
- **Records:** KV ledger, K_eff, QSA over the whole log (Merlin's logged granule), `nbd1_verdict.json`. All labels are record-only.

**Real failures (that arm only, NOT_INTEGRATED):**
- the arm's own change read back with another value: Q256 QSA 256/256 in the 4 workers, the G1/G2 switch, the IDX indexer cache fp8, LP's logged (2048, 1024, True), SPC's flag, S28/S32's logged seqs, D4s/DYN's logged K and schedule, NB-S1's DYN fallback lines;
- another lever's change present on the arm.

**My call (please confirm):** B's base settings read back differently are **recorded only**, not stops. That covers QSA not 128/64 on a non-Q256 arm, and EMBED_HOST not applied. This matches E29's record-only QSA read-back. An unreadable read-back is also recorded only.

### 3. Arms per copy, times (min; est / body cap)
Time model: NB-S1 full GPU data (DFlash D3s + EMBED_HOST: first ready 714 s, warm 252-282 s, window spacing 125 / 217 / 351 / 273 / 272 s) plus NBM1L's items and caps.

**Ready classes:**
- B first boot 12 min.
- Q256, D4s, DYN, IDX, LP and SPC are counted as recompiles (7). Q256 is one because `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` is not in Merlin's compile-hash ignore list; the GDN keys are, so G1/G2 are warm (4.8).
- S28/S32 warm + new captures (5.5).

| Copy | Owner | Arms (est/cap) | Runtime | Full-cap end |
|---|---|---|---|---|
| A | foysalemonshanto | setup 6, B (BAA + PROF) 52/44, Q256 28/24, D4s 28/24, DYN 28/24 | **142** | 145.8 |
| B | hosen42 | setup 6, B light 34/25, G1 26/24, G2 26/24, IDX 28/24, LP 28/24 | **148** | 151.8 |
| C | foysalemonshanto | setup 6, B light 34/25, S28 25/23, S32 26/23, SPC 29/25 | **120** | 123.8 |

- **Hard stop:** 160 min for all copies, deadman 165. The build asserts BUDGET = ceil5(max(runtime, full-cap end) + 5), each copy ≤ 150, and every cap ≥ body + 1.
- **B windows:** in copy A, 8k@16 ×2, 20k@24, 40k@16, 90k@8, 116k@6, plus PROF. In copies B and C ("B light"): 8k@16, 20k@24, 90k@8, 116k@6. This is selected by the switch key BAA.
- **Lever windows:** 8k@16, 20k@24, 90k@8, 116k@6. S28/S32 run 20k@28/32, 90k@8 and 116k@6.
- **Switch keys:** `B BAA PROF Q256 D4s DYN G1 G2 IDX LP S28 S32 SPC`. A missing key means off.

### 4. Sources (kernel-metadata of every copy)
- **Datasets:** `foysalemonshanto/q27-e25-nb0-tf-ref` and `foysalemonshanto/q27-e25-nbd1-payload`.
- **Models:**
  - `foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1`
  - `foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1`
  - `foysalemonshanto/e23-merlin-env/Other/v1/1`
  - `foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1`
- **Competition:** arc-prize-2026-arc-agi-2. Pinned image, NvidiaL4.

**hosen42 needs access to all six private foysalemonshanto sources:** the 2 datasets (the payload one new) and the 4 Models.

### 5. Tests
**Local unit tests:** `nbd1_unit_tests.py` gives **354/354 PASS** on Py3.14 and on Py3.12.10. They cover:
- the arm plan (B = D3s + EH + QSA, each lever = B + one change, 14 mutations refused);
- `nbdf_lib.launch_check` per arm on the simulated launch env (Merlin 256 → NB1_ENV), plus negatives: Merlin 256 on B without the override, GDN on B, no EH, wrong seqs, wrong spec;
- switches and copies, walks, budget formula, B light vs full, PROF plan;
- read-backs on the real NB-S1 GPU fixture: QSA, GDN, EH, IDX, LP, SPC, S32, D4s, DYN, the DYN override line, unreadable cases, `base_mismatch` recorded;
- `retry_plan`, the [LIVE] format, `gate_from_report` (ok / stale sha / missing key / not ALL_OK / base closed);
- Merlin's own `build_dynamic_sd_schedule_lookup` == `dense_schedule`;
- the body's markers and the KV gate vs facts;
- `arm_body` with fakes: every arm of every copy runs only its light items and windows; PROF only on B in A; sibling only on SPC; prefill only on B; a harness contrast; smoke FAIL → NO_GO_SMOKE; read-back failure → NOT_INTEGRATED; `d1_readback_failure` on the fixture;
- `arm_boot` with fakes: OOM after ready → 7,665,090,560, port + 1, EH and QSA kept; Xid → no retry; ladder of 4 rungs; non-OOM → no retry; Q256 keeps 256;
- `nbd1_main` with fakes: each copy launches exactly its arms; B off → levers run; no cross-arm skip; closed gate → nothing boots; seam fail; all off;
- `run_window` with fakes: a lever steps down on any band; B does not at 8k; a LIVE line per window, skipped included;
- facts, export and the switch line read from the export (once, line 5 of CELL 1, literal = default, runs, not in body or lib); copies differ only in that line.

**Negative control:** run against NB-S2's body/facts/export it gives 196/221 (the tests tell the two builds apart).

**Units-kit layout simulation:** an empty folder with `python -I` on Py3.12.10 gave **346/346 PASS, `@@KIT units ALL OK@@`**. The only skips were the `SKIP notebook` checks, by design. Folder: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\q27-e25-nbd1-kit-units-sim2`.

**Gated path, with synthetic reports in `--test-out`:** an all-keys-True report on abbf21df gives `gpu_ready` true (3f119d85, 347,408 B). One key False keeps it closed.

**Static check:** no undefined names in the notebook cells or the real kit.

### 6. Kit folders (not pushed; `D:/ARC AGI 2/_scratch/push_e25/`)
- **`q27-e25-nbd1-kit-units__foysalemonshanto`:** ipynb sha 6c3ff340a56813d33839d77e4da2d5056a722d96951d3109b279599e0bc7445f, 383,082 B, no sources. Report: `nbd1_kit_units_report.json`.
- **`q27-e25-nbd1-kit-real__foysalemonshanto`:** ipynb sha 929af7891e41241858f24caae94d0ae11e3442ff5d1d1c54b801d530de09f6b5, 388,314 B (compact JSON). Sources: the tf-ref dataset and the 4 Models; the payload is embedded because it is not uploaded yet.
  - It is NB-S2's real kit retargeted, with every NB-S2 edit copied, plus NB-S1's spawn-keys = expect edit.
  - It runs the notebook's own pre-launch sequence for all 11 arms, the `--help` gate and the vLLM argparse one-change check per arm (env-only arms parse identical to B), plus DYN Merlin dense == lib.
  - KV probes: K3_swblock1024, K4_swblock1024 (D4s/DYN), B_idxfp8 (IDX), and max concurrency at 24/28/32 (S28/S32).
  - It checks the final launch env QSA, GDN and EH per arm, the captures per arm and the pack views up to 32. Merlin's launch.py 256 lines are recorded.
  - Gate keys `nbd1_*`: arm_plan, launch_path_all_arms, help_flag_gate_real, argparse_one_change_per_arm, kv_probes, qsa_env_final, captures_argv_packviews, switch_default_all_on.
  - Report: `nbd1_kit_real_report.json`.

Both kits are bound to the closed sha abbf21df; any builder change means rebuilding them. GPU push folders: `q27-xhigh-e25-nbd1-a__foysalemonshanto`, `q27-xhigh-e25-nbd1-b__hosen42`, `q27-xhigh-e25-nbd1-c__foysalemonshanto`.

### 7. Dataset to create
`foysalemonshanto/q27-e25-nbd1-payload`, from `D:/ARC AGI 2/_scratch/push_e25/ds_e25_nbd1_payload/`. It holds nbd1_cell7_body.py, nbd1_lib.py (af0f795f), SHA256SUMS.json and dataset-metadata.json.

**Upload only after the gated rebuild.** The closed body there now (725319b1) carries the closed KV gate; the gated rebuild rewrites the body (new sha) and this folder. Steps:
1. Run both kits.
2. `python build_e25_nbd1.py --kv-gate-report <...>/nbd1_kit_real_report.json`, then check facts `gpu_ready` is true.
3. Upload the payload folder as the new dataset and give hosen42 access.
4. GPU commits need the user's go.

### 8. Open risks
- **Timing:** copy B is at 148 of 150 min (hard stop 160). Slower boots would make LP SKIPPED_BUDGET. The time model comes from one NB-S1 run.
- **OOM any stage (your rule):** a late OOM reboots the whole arm one rung lower if at least 20 min + its cap are left. That can starve later arms.
- **20k levels:** 20k is level-capped by this boot's resident bound (NB-S2's CAPPED_BANDS). With F = 14, S32's 20k may run at about 27 (key `20k@27`). A lever's runtime KV-bound step-down can repeat, one window per step, and eat its cap.
- **QSA read-back:** the RPC read-back of the QSA keys has never run on GPU (NB-S1's RPC_KEYS lacked them). Only Q256 can stop on it; a base mismatch is recorded only.
- **Never GPU-run:** D4s/DYN with EMBED_HOST, IDX fp8 on sm_89, LP, SPC with DFlash, S28/S32 at seqs 28/32.
- **Real-kit argparse:** the model_tag fix carried over from NB-S2's real kit is still unconfirmed on Kaggle (NB-S2's wave2 kit failed only on that check, before the fix).
- **"B light" interpretation:** I read "windows only" in copies B and C as the lever window set (no 8k A/A pair, no 40k). Setting BAA True in those copies restores the full set, at about +8 min.
- **PROF:** the timeline analysis runs on the 8k trace only; the 20k trace gets graph spans only.
- **Kit notebook:** the all-on notebook does not fit 160 min; it is the kit notebook only.
- **Size margins:** the notebooks are about 52 kB under 400 kB; the kits are 383 kB and 388 kB.
- This report is not yet saved to Agent Reports (that is the coordinator's step). Scratch helpers are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\nbd1\`; the NBDF reference build is in `D:/ARC AGI 2/_scratch/nbd1_build_ref`.