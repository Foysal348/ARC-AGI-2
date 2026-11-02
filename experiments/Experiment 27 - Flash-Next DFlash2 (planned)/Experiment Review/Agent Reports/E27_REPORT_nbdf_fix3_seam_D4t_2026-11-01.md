# E27 NBDF agent (acfe40bd525f7510b): fix 3 report (seam 7356d473, D4t, SW-block guard, est_min), verbatim

## Seam `7356d473` re-embedded, D4t added, SW-block guard and new est_min values in; notebook rebuilt with the gate closed, kits refreshed, nothing pushed

**Final kit v4 must run on this notebook:** `d8f94d7b3bffaba53ec43ccfb2906b1352e790169d38712dd0b6940dc06ba1f9` (388,115 B, about 12 kB under 400 kB). The closed-build sha is the same value (`closed_build_sha256` = notebook sha). The gate is closed, so `gpu_ready` is false.

**Tests (all run locally):**
- `nbdf_unit_tests.py`: **188/188 PASS**.
- `nbdf_build_gate_test.py` (the real builder with synthetic reports): **11/11 PASS**.
- Units kit, run once: **`@@KIT units ALL OK@@`**, with nbdf 186/186 (the 2 export-layout checks need the notebook file, which the kit does not bundle) and seam tests 591/591.

**Seam:** `dflash_seam.py` `7356d4738962bdbab4091264861c4bfbbdcd7098d6c75b9c48db69eeb73e6937` and its tests `ae856fef…f22b` are embedded and bundled. The provenance seam sha is now this embedded one.

### 1. Seam re-embed
- The builder embeds the file as found.
- The `readback_label` call passes `conv_impl=` when the seam accepts it (`"torch"` on D4t, `"fused"` elsewhere); `no_ingraph_meta=` is still passed.
- `seam_env_for` sets the torch conv only for `CONV_TORCH_ARMS`.

### 2. D4t (optional, after D3, before D7)
- **Definition** (`nbdf_lib.py:42-46`): `ORDER = Q > D4 > D3 > D4t > D7 > D4b`. `SHORT_ARMS` and `CONV_TORCH_ARMS` are both `("D4t",)`. D4t has its own port (8061).
- **Config:** exactly D4 plus `DFLASH_CONV_IMPL=torch`. Same spec, same captures 5..40, and it uses D4's K4 KV summary (`KV_RUN_OF`).
- **Body** (`nbdf_cell7_body.py:771`): the usual read-backs, smoke + concurrency ladder, and the greedy oracle (token ids vs Q), then a warm-up and the two 8k@8 windows only. No long oracle, prefix test, 40k / 90k / 116k windows or profiles.
- **Verdict** (`nbdf_lib.py:1565` `conv_compare`): compares with D4 on the 8k AL, per-position acceptance, 8k tok/s and greedy oracle AL.
  - `CONV_IMPL_TORCH_BETTER` when D4t's mean 8k AL beats D4's by more than D4's own two-window AL spread;
  - otherwise `CONV_IMPL_NO_DIFFERENCE` / `CONV_IMPL_TORCH_WORSE`, or `INCONCLUSIVE_CONV_IMPL`.
  - D4t is never throughput-eligible.
- **Read-back:** D4t must show `conv_impl` lines with impl torch on both PP1 workers. A torch line on a fused arm is a definitive contradiction.
- **Budget:** est 22 min, gated by `arm_fits`.
- **Real kit:** now launch-plans and checks all 6 arms. I kept the existing check names (e.g. `host_phase_all_5_ok`) so the gate's 17-check list does not change.
- **Tests (T1 / T13b):**
  - D4t = D4 + torch;
  - `CONV_IMPL_TORCH_BETTER` (D4 stays the best arm);
  - a difference within the spread gives `CONV_IMPL_NO_DIFFERENCE`;
  - the conv_impl read-back is required on D4t;
  - a torch line on a fused arm is caught.

### 3. `DFLASH_DRAFT_SW_BLOCK` is never set on a GPU arm
- `seam_env_for` strips it.
- `check_arm_plan` refuses it on any arm (`nbdf_lib.py` "on a GPU arm").
- `launch_check` refuses it on the final launch env (`nbdf_lib.py:631`).
- Only kit R9 uses it, as its block-1024 capacity probe.
- Tests T1: no arm sets it; the plan check and the launch check both refuse it when injected.

### 4. est_min raised
`nbdf_lib.py:418`: Q 50, D4 41, D3 40, D4t 22, D7 42, D4b 40. Body caps: Q 41, D4 33, D3 32, D4t 15, D7 34, D4b 32. The optional profiles are not counted; they run only when due.

**Consequence at the 140-min hard stop (planned walk, test T10):**
- Q and D4 are planned.
- D3 starts only if Q and D4 finish faster than their minimums. At Q 45 + D4 36 min, D3 runs (tested).
- Otherwise D3 gets SKIPPED_BUDGET, and D4t (22 min) fills the remaining time.

If you want D3 guaranteed under the new minimums, the hard stop would need to be about 150 min. I did not change it: the user's rule is "about 130-140 min". Deciding that is yours.

### Trims
- The markdown cell is cut to 4 lines.
- The notebook JSON is written compact (`separators=(",", ":")`, `build_e27_nbdf.py:1554`; plain JSON formatting, nothing else changes).
- Result: 388,115 B, versus 397,812 B before D4t.

### New shas (all LF)
| File | Lines | sha256 |
|---|---|---|
| `build_e27_nbdf.py` | 1647 | `c0c2434e7b896fb2296da56ae22ebb3860d807145e971eef6e88f0f27577eb7c` |
| `nbdf_lib.py` | 1829 | `e7e363869ef4f0edacfaadb95d7644ec20c51d1e9cc9c41561824b0db97bb5f6` |
| `nbdf_cell7_body.py` | 1247 | `75e637de5c207735b65cc232638cde9f634e9dd9b36289877651c7bae1b4a9f0` |
| `nbdf_code_export.py` | — | `693533f6dd01060c760ed61e705d0b991bc851fcc09147f054a8b83bb3dd300e` |
| `nbdf_unit_tests.py` | 640 | `b0dd0cad3bc8b2ec63fa4503b905b9ee17c7be5435bc64c5740930f37649926e` |
| `nbdf_build_gate_test.py` | 85 | `7546d27791d4134907e39d3d0cea8b03997844af27e6540184ec91e68b462f1e` |
| `build_e27_nbdf_cpu_kit.py` | 740 | `30153bc5e89acc1e984387bf1fc32f91dea5f82f2e9eb5148a9931e18d07e8d9` |

The 9 NB4T / E25 files the builder checks are unchanged.

**Kit folders** (refreshed, not pushed), under `D:/ARC AGI 2/_scratch/push_e25/`:
| Folder | Size | sha256 |
|---|---|---|
| `q27-e27-nbdf-kit-units__foysalemonshanto` | 352,834 B | `f1130cd6…` |
| `q27-e27-nbdf-kit-real__foysalemonshanto` | 274,844 B | `ff6a63dd…` |
| `q27-xhigh-e27-nbdf__foysalemonshanto` (GPU) | — | the new notebook |

The audit zip `Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip` is refreshed with 29 files, including the M1 seam report: 740,944 B, sha `e5bd7fea…`.

### Open
- **Kit v3 ran the old notebook** (`a40be02b`, seam `2df1da39`). Its report will close the gate as stale, by design, so it cannot serve as the gate report. Its R9 / R12 results are still useful diagnostics: send them to me and I will fix any probe issues on this builder before kit v4.
- **Kit v4** must run on `d8f94d7b`. Then rebuild with `python build_e27_nbdf.py --kv-gate-report <v4 nbdf_kit_real_report.json>`; the provenance will match as `closed_rebuild`.
