## E25 NB-M1L (light) build report: closed build done. Nothing pushed, no datasets uploaded, nothing run on Kaggle.

The light build is finished and its two kits are built; you run the kits next. The closed light notebook has `gpu_ready` false, so the gated rebuild is still to do. Without `--light`, the builder still produces the current NB-M1 byte for byte. The running copies A and C were not touched (0 differences in a before/after hash check).

**The light copies are still above your 100-120 min target:** B is 142 min and C is 146 min (section 3).

### 1. Files
All files are LF. Folder: `D:\ARC AGI 2\experiments\Experiment 25 - Flash-Next Throughput 1500 (planned)\Experiment Notebooks\Build\`

**Changed:**
- **`build_e25_nbm1.py`** (91a64c08, 42,619 B), now takes `--light`:
  - Light outputs go to new names only.
  - `nbm1l_overlay.py` is embedded and sha-checked.
  - Two body edits run in light mode only (section 2).
  - Light budget assertions.
  - `FROZEN` (the hash-check list) now also covers every NB-M1 run notebook, the nbm1 build files, the `q27-xhigh-e25-nbm1-*` push folders and the `q27-e25-nbm1-kit-*` kit folders. They are checked before and after the copies.
  - New guard: a default build outside `--test-out` refuses to run while the run notebooks (gated 9e052435, A 8d183071, B 56fce34d, C 48996e78) are on disk.
- **`build_e25_nbm1_units_kit.py`** (670ad4c6) and **`build_e25_nbm1_cpu_kit.py`** (7002ee20): each takes `--light` and `--out-root DIR`.

**New:**
- **`nbm1l_overlay.py`** (2ce100d1, 5,220 B): runs inside nbm1_lib's namespace, so `nbm1_lib.py` stays unchanged (e64482c4).
- **`nbm1l_unit_tests.py`** (fa0f5fbb).

**Light outputs (Build folder):**
- `nbm1l_cell7_body.py` (47dec18f)
- `nbm1l_code_export.py` (ce6407dc)
- `nbm1l_build_facts.json` (01256279)

**Notebooks** (in `...\Experiment Notebooks\`; the copies differ only in their `NBM1_ON` line):

| Notebook | Role | sha | Size (B) |
|---|---|---|---|
| `Q27-XHIGH-E25_NBM1L.ipynb` | every arm on; the kit notebook | 0083e7045bc7e06607ed6e836f05394d6e4ab5b6c69c6e7a8ba724479f50fe1f | 396,543 |
| `Q27-XHIGH-E25_NBM1L_B.ipynb` | copy B | 92c27cfe799b802a | 396,551 |
| `Q27-XHIGH-E25_NBM1L_C.ipynb` | copy C | 1b63395da0484615 | 396,550 |

Unchanged: `nbm1_unit_tests.py` (ebe63cd5) and every NB-M1, Exp 27, NB-S1/S2, E29 and E31 file.

### 2. What was removed per arm
**Q in copies B and C:**
- **Removed:** the smoke concurrency ladder (2/8/23), Q's greedy oracle (the reference), the oracle A/A and its cache reset, the long oracle, the prefix cold/warm test, Q's own two profiles (8k@16 and 116k), and the TF reference plus TF A/A.
- **Kept:**
  - facts (chat probes and the sampled first request);
  - every read-back and gate: the I8 head, the K100 boot gates, Q-defer;
  - the 2-stream smoke;
  - the prefill probe;
  - warmup;
  - the six windows: 8k@16 ×2, 20k@24, 40k@16, 90k@8, 116k@6;
  - the end records.
- **Copy C:** PROF keeps its own two 30-s windows (8k@16, 20k@24) inside Q's boot.
- **Why the prefill probe stays (my call):** the windows' `due()` timing check and their admission both use the measured prefill rate. Without the probe the rate stays at 700 tok/s, so the 116k@6 window would need about 20 min left and would be skipped. The probe takes 0.5 min.

**Every lever (IDX, S28, S32, LP, G1, G2, K4, DYN; also V65/V32/I4 in the kit notebook):**
- **Removed:** the oracle vs Q with its quality gate and the AL-vs-Q diagnostic, and TF (IDX and LP had it).
- **Kept:**
  - facts and warmup;
  - the smoke hard gate (any FAIL_* label stops the arm);
  - the read-back of the arm's one change (a real failure means NOT_INTEGRATED for that arm only);
  - the window sets, unchanged: 90k and 116k always; a KV-bound window steps down one level and is never skipped.

**How it works in the body (light only):**
- The whole oracle block runs only when `M1.item_on(arm, "oracle")`. Otherwise it is recorded `NOT_IN_PLAN_NB_M1L`.
- At the end of `arm_boot`, an arm with no oracle is now gated on the hard-only evidence: probe request errors, the sampled first request, and hard garbage in any window. Without this change, the missing oracle would have overwritten a finished arm's DONE status with INCONCLUSIVE_NO_ORACLE.
- **Unchanged:** every real-failure rule, the OOM ladder at any stage, the `NBM1_ON` switches, and the hard-only gates.
- **Labels:** every lever's nbdf verdict becomes INCONCLUSIVE_ORACLE (record only, never a stop). The raw per-band numbers vs Q stay in `nbm1_records`.

### 3. Time estimates (minutes)
Same GPU time model as before. The body caps are new: each one is sized with the body's own `win_need`/`due()` rule at the 3,000 tok/s admission floor, plus 1 min. A unit test checks that the overlay's formula equals the body's `win_need`.

| Copy | Arms: estimate (body cap) | Total | End with every full cap |
|---|---|---|---|
| B | setup 6, Q 37 (29), IDX 24 (20), S28 25 (22), S32 26 (23), LP 24 (20) | **142** | 145.8 |
| C | setup 6, Q+PROF 48 (40), G1 22 (20), G2 22 (20), K4 24 (20), DYN 24 (20) | **146** | 149.8 |

- The current copies are 175 min each, so this saves 33 (B) and 29 (C) min.
- **Q breakdown:**
  - B: ready 11 + body 25.0 + post 0.8.
  - Body items: facts 1.5, smoke 1.3, prefill 0.5, warmup 1.0, windows 19.5, end 1.2.
  - C adds PROF 7.3, and its post is 3.8.
- **Levers:** ready (7 recompile / 4.8 warm / 5.5 warm with new captures) + body (5.0 + windows) + post 0.8.
- **Budget:** hard stop **155**, deadman **160** (CELL 1: `BUDGET` 155×60, `NBDF_DEADMAN_SEC` 9600). The build asserts:
  - runtime ≤ 150;
  - hard stop = 5-min round-up of (max of runtime and the full-cap end) + 5;
  - every cap ≥ each timed item's start + its need + 1.
- Left at the end: B 13 min, C 9 min.
- **Why it is still over 100-120:** the windows alone are 68.6 (B) and 70.8 (C) min, and the five boots about 35 min. Getting to 120 would need fewer arms per copy, which is your decision; I changed nothing beyond your list.

### 4. Tests
- **Default unchanged** (rebuilt in `--test-out`, twice including the final state):
  - closed 6505026007ed23cf…, gated 9e052435eb7862be…;
  - copies 8d183071 / 56fce34d / 48996e78;
  - body, export and metadata identical.
- **Default unit tests:** `nbm1_unit_tests.py` 142/142.
- **Kit builders unchanged by default:** I rebuilt the original builder text (1fc370ee / 79fa7b21) and ran it against the new builder on the same inputs. The kit bytes are identical: units 3b3790f1, real f8fd01eb.
- **Light unit tests:** `nbm1l_unit_tests.py` 94/94. They cover:
  - each lever = the default NB-M1 plan except items and time;
  - Q light items;
  - no TF or oracle on any lever;
  - times, caps and budget;
  - the embedded overlay = the file (sha = facts);
  - the switch line read from the export, and the facts;
  - `arm_body` with fakes: Q in B and C and every B/C lever call only their light items (removed items never called; PROF 2×30 s in C);
  - a contrast check: the default Q/IDX plans do call the oracle, TF, prefix, profiles and long oracle on the same harness;
  - a lever smoke FAIL → NO_GO_SMOKE;
  - a lever read-back real failure → NOT_INTEGRATED;
  - Q smoke FAIL stays recorded only;
  - the `arm_boot` evidence gate and the OOM/Xid ladder;
  - `nbdf_main` for B and C, Q off, the closed gate, a failed lever not skipping the next;
  - the hard-only gates without an oracle, and the verdict with no exception.
- **Units kit simulated:** empty folder, `python -I`, Py3.12.10: 95/95, `@@KIT units ALL OK@@`.
- **Notebook-level block:** the NB-M1 block of CELL 7 ran with the real libraries for all-on, B and C.
- **Gate:** a synthetic gated light build in `--test-out` gave `gpu_ready` true, 397,873 B, and `nbdf_gate_verify` ok. With one `nbm1_*` key False the gate stays closed.
- **Determinism and isolation:** the light build is deterministic (`--test-out` and the real build give the same 0083e704). The before/after hash check of every NB-M1 file, push and kit folder and the E27 build folder shows 0 differences.

### 5. Kit folders (foysalemonshanto, `D:/ARC AGI 2/_scratch/push_e25/`)
- `q27-e25-nbm1l-kit-units__foysalemonshanto`: kit fab68dc4…, 228,619 B. Report: `nbm1l_kit_units_report.json`.
- `q27-e25-nbm1l-kit-real__foysalemonshanto`: kit f425ad25…, 384,940 B. Same checks as the NB-M1 real kit (all 12 arms' launch path, help gate, argparse, KV probes, vocab tables). Sources: tf-ref, e29 payload, the 4 models.

**After the real kit passes:**
1. `python build_e25_nbm1.py --light --kv-gate-report <nbdf_kit_real_report.json>`
2. `nbdf_gate_verify.py nbm1l_build_facts.json nbm1l_cell7_body.py ..\Q27-XHIGH-E25_NBM1L.ipynb`

The kits are bound to closed sha 0083e704; any builder change means rebuilding them.

### 6. Push folders
- `q27-xhigh-e25-nbm1l-b__hosen42`: id hosen42/q27-xhigh-e25-nbm1l-b.
- `q27-xhigh-e25-nbm1l-c__foysalemonshanto`.
- Both list every source: tf-ref, e29 payload, klee100, the FP8 PLE overlay, the Merlin env, and dflash2v2.
- The light all-on base push folder was removed, as in NB-M1.

### 7. Open risks
- **Running copies A and C (default build): some windows may be skipped.** I found this while sizing the light caps. The current cap rule (body + 1) can be shorter than what the body's `due()` check needs for the last window. With the model times:
  - G1/G2/K4/DYN in C: the 116k@6 window needs 19.1 min against a 19-min cap;
  - S28 in the default B: 21.5 against 21;
  - PROF's 20k@24 window in C: about 60 against 55.

  If items run at the model times, these windows can come back SKIPPED_BOOT_BUDGET. Please check those keys when the results land. The light build sizes its caps for exactly this.
- **Light Q no longer has the oracle's hard stop.** Q's hard stops are now the probes/first request, the I8 and K100 gates, Q-defer, and windowed hard garbage (status set after the boot). Q's smoke FAIL stays recorded only (NBDF-K100's rule, unchanged).
- **Prefill rate:** if the prefill probe measures below 3,000 tok/s, the window needs grow. At 2,000 tok/s, 116k@6 needs 9.3 instead of 7.4 min, and the 1-min margin may not hold.
- **Size:** about 397.9 kB gated, so about 2.1 kB under the 400 kB rule.
- **hosen42 access:** copy B on hosen42 lists foysal's private dflash2v2 model and e29 payload. The C copy already on hosen42 had dflash2v2 removed from its metadata. I kept it, as you asked, but if hosen42 cannot see it the push may fail.
- **Labels:** every lever verdict is INCONCLUSIVE_ORACLE (record only); judge on the raw `nbm1_records` numbers.
- **Kit notebook walk:** the all-on light notebook does not fit all 12 arms in 155 min. It is the kit notebook only, same as before.

Helper and verification scripts are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\`. This report has not been saved to Agent Reports yet.