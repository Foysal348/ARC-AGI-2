# ENGINE MATRIX EMS / VMS REV2E–REV2F: Independent, Source-Grounded Pre-GPU Audit

**Date:** 2026-10-26  
**Verdict:** **NO-GO for automatic production adoption**; **CONDITIONAL GO for instrumented GPU diagnostic only**.  
**Scope:** User bundle `ENGINE_MATRIX_EMS_VMS_REV2E_AUDIT_FOR_GPT_2026-10-26.zip`; its `README_FOR_GPT.md`, `notes/EMS_VMS_rev2e_independent_audits_disposition_2026-10-26.md`, current generated notebooks, builders, harnesses, previous CPU reports, and pinned source snapshots. No real GPU run executed. The bundle calls itself rev2e, but the current notebooks are the **rev2f SAFE/RISKY split**.

## How to use this file with Claude

Read this README alongside the original ZIP. Keep `https://...` source URLs literal. Do **not** infer that the old rev2c CPU-kit reports certify the new rev2f hashes. Audit the findings below and add negative tests to the actual generated notebook paths. Apply fixes in the `builders/` sources, regenerate and hash each notebook, then run Kaggle CPU kit on **those exact hashes** before considering GPU. Preserve diagnostic results on incomplete/failed arms but never mark them `adoptable`.

## 1. Material directly verified

| Artifact | SHA256 prefix independently computed |
|---|---|
| `notebooks/Q27-XHIGH-EMS_SAFE_ENGINE_MATRIX_SGLANG.ipynb` | `c0ade787` |
| `notebooks/Q27-XHIGH-EMS_RISKY_ENGINE_MATRIX_SGLANG.ipynb` | `b39016d6` |
| `notebooks/Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb` | `b01992c5` |
| `kaggle_cpu_kit_runs/...run3.../cpu_harness_report.json`: EMS | `c062255e` **OLD rev2c**, 26/26 |
| Same report: VMS | `675db353` **OLD rev2c**, 19/19 standard + bad-INT4 |

**Evidence-integrity warning:** The README states the Kaggle CPU kit v4 has run on the exact new notebooks, including EMS SAFE and RISKY, but the supplied ZIP contains only run1–run3 JSON reports for prior hashes and `harness_logs` entries labeled rev2c. There is **no v4 report with matching current notebook hashes inside the supplied ZIP**. This does not prove that such a test never happened; it means this pack does not establish it. Re-run or attach v4 report + raw logs + SHA binding. Do not claim new rev2f 26/26 or 19/19 from the old artifacts.

The source-level fixes for many rev2c findings are present: integer token IDs, `[DONE]` and `finish_reason`, target-only oracle, capture of stream errors, separate `measurement_ok` / `capacity_ok` / `correctness_ok`, added XL tests, fixed `VGen._stop`, atomic PR #45151 patch, and separate SAFE/RISKY notebooks. **Presence is not GPU correctness evidence.**

## 2. Finding-by-finding audit table

| ID | Severity | Failure | Evidence: bundled file & lines | Observed vs expected | Status |
|---|---|---|---|---|---|
| **R2F-01** | **HIGH fail-open** | Missing XL parity and missing canary treated as success by `vms_classify` | `builders/build_vms.py:727–739`; DP skips XL at `771,799,812`, called at `925` | Isolated classifier accepts absent `xl_gate` and absent `canary` when other fields PASS; DP arm skips XL by design | **Source + reproduction** |
| **R2F-02** | **HIGH fail-open** | EMS accepts `xl_gate=None`; oracle canary `.errors` never checked | `builders/build_ems.py:681–711`, especially `701–704`; XL may be skipped at `556–558` | A missing XL parity is not distinguished from not-applicable; oracle error is ignored if `foreign_leaks==0` | **Source-confirmed** |
| **R2F-03** | **HIGH coverage** | A request with no first token can pass VMS window | `builders/build_vms.py:415–441` | `no_first_token=1` is collected but ignored in `vms_window_healthy` | **Source + reproduction** |
| **R2F-04** | **HIGH evidence integrity** | Claimed exact-revision CPU evidence not supplied | `README_FOR_GPT.md` vs `kaggle_cpu_kit_runs/**/cpu_harness_report.json`, `MANIFEST.txt` | Run3 certifies old notebook hashes only | **Archive inspection** |
| **R2F-05** | **HIGH run isolation** | EMS dynamically selected MTP `_s20` arm sits outside per-arm `try/except` | `builders/build_ems.py:902–922` | Failure in `_arm(_best + '_s20')` falls to broad outer handler; later planned arms can be lost | **Source-confirmed** |
| **R2F-06** | **MEDIUM–HIGH blind spot** | Any output with >=30% digits bypasses all degenerate checks | `builders/build_vms.py:475–491`; EMS equivalent `build_ems.py:713–725` | `"0" * 250` and `"0"*40 + "!"*60 + "0"*60` return `text_bad=False` | **Isolated reproduction** |
| **R2F-07** | **MEDIUM evidence quality** | Canary can have zero `own_hits` and zero `foreign_leaks`, thus no demonstrated sensitivity | `build_vms.py:712–724,781,734–737`; EMS `build_ems.py:456–468,699–704` | Noninformative leak test reported clean; incomplete/errored oracle canary not excluded | **Source-confirmed risk** |
| **R2F-08** | **MEDIUM timebox** | TP2 parallel workers use unbounded `join` | `builders/build_ems.py:1047–1060` | Hung worker stalls both-replica comparison beyond planned cap | **Source-confirmed risk** |
| **R2F-10** | **HIGH oracle-readiness** | VMS promotes target-only to oracle even when its canary reports stream errors, and without checking its XL A/A completeness | `builders/build_vms.py:662–672,773–787` | `aa["complete"]` only covers serial/concurrent, while line 781 checks foreign leaks but not canary errors / XL AA | **Source-confirmed** |
| **R2F-09** | **MEDIUM metric wording** | VMS `end_to_end_tok_s` is a submission-to-window-end counter, **not** completed-request task throughput | `builders/build_vms.py:316–434`, especially `405–433` | Streams are deliberately stopped after measurement; report already partly qualifies this | **Source-confirmed limitation** |

### Why R2F-01 specifically threatens TP2

`_vms_arm` sets `_dp=True` for flags with `--data-parallel-size`, and then does **not** run `xg` or the XL window for this arm (`if _X and not _dp`). But the `tp2x2_fp8_mtp3` row is not marked diagnostic, nor does `vms_classify` require XL evidence: `a.get("xl_gate") in (None, "PASS")`. Given otherwise healthy rows, this arm can become adoptable with no XL parity. This is not a hypothetical missing-file scenario; it follows the configured DP branch. By contrast, `tp2x2_int4_mtp3` uses `gate=False`, so it cannot accidentally pass the `gate=='PASS'` test.

### Isolated executable negative tests performed (no GPU)

Functions were extracted from the builder's actual embedded VMS cell code and executed locally with synthetic records:

```text
scenario = missing XL and canary
  gate=PASS; short/long healthy; oracle_ready=True
  vms_classify => adoptable=True (SHOULD NOT CERTIFY)
scenario = XL INCOMPLETE
  vms_classify => adoptable=False (CORRECT)
scenario = canary errors=3
  vms_classify => adoptable=False (CORRECT)
scenario = short.no_first_token=1
  vms_window_healthy => True; adoptable=True (SHOULD NOT CERTIFY)
text_bad("0" * 250) => False (degenerate bypass)
text_bad("!" * 250) => True (current heuristic works here)
```

These are **classifier-level reproductions**, not claims that real Qwen GPU execution has already returned such outputs. The missing-canary data case is not necessarily reachable on an otherwise error-free normal `_vms_arm` run, because most normal paths assign canary before windows; nevertheless the DP missing-XL path **is configured**.

## 3. Minimal code-level fixes

### Fix 1 — VMS classifier: require real XL evidence, explicit canary, and label DP diagnostic

In `builders/build_vms.py`, inside `_vms_arm`, set and persist an explicit diagnostic marker **before** `vms_classify(a)`:

```python
_dp = "--data-parallel-size" in [str(x) for x in flags]
a["diagnostic_only"] = bool(_dp)  # until this path gets the same per-rank XL parity/canary coverage
```

At `vms_classify`, replace the correctness and adoption segment, retaining existing `measurement_ok` and `capacity_ok`:

```python
xl_required = bool(globals().get("_X"))
xl_ok = (a.get("xl_gate") == "PASS") if xl_required else True
c = a.get("canary")
canary_complete = (
    isinstance(c, dict)
    and c.get("foreign_leaks") == 0
    and c.get("errors") == 0
    and isinstance(c.get("own_hits"), int)
)
a["correctness_ok"] = (a.get("gate") == "PASS" and xl_ok and canary_complete)
a["adoptable"] = bool(
    VMS.get("oracle_ready") and a["measurement_ok"] and a["capacity_ok"]
    and a["correctness_ok"] and not a.get("diagnostic_only", False)
    and a.get("server_alive_after") and not a.get("log_errors") and not a.get("error")
)
a["certified_exact"] = bool(a["adoptable"] and a.get("gate_class") == "PASS_EXACT")
```

**Policy note:** `own_hits=0` may be a compliant model paraphrase; this patch requires **presence of evidence**, not arbitrary `own_hits>=1`. Record `canary_informative = (own_hits>0)` separately; do not call zero hits proof of state isolation. A positive-control test is needed before canary absence can prove anything. If XL is intentionally omitted to save budget, keep the row `DIAGNOSTIC_NO_XL`, not `adoptable=True`.

**Test cases:** (i) missing XL→not adoptable; (ii) XL `INCOMPLETE`→not adoptable; (iii) no canary→not adoptable; (iv) canary error or leak→not adoptable; (v) complete XL `PASS`, valid canary, healthy data→adoptable; (vi) `tp2x2_fp8_mtp3` with no XL→diagnostic only, never production-certified.

### Fix 1b — VMS oracle readiness must be complete too

At `builders/build_vms.py:781`, replacing `elif aa["degenerate"] == 0 and aa["complete"] and not a["canary"]["foreign_leaks"]:` with a fail-closed predicate is necessary:

```python
_xl_aa = aa.get("xl")
_xl_complete = (not _X) or (
    isinstance(_xl_aa, dict)
    and _xl_aa.get("expected", 0) > 0
    and _xl_aa.get("compared") == _xl_aa.get("expected")
    and not _xl_aa.get("errors")
    and not _xl_aa.get("missing_margins")
)
_canary_good = (
    isinstance(a.get("canary"), dict)
    and a["canary"].get("errors") == 0
    and a["canary"].get("foreign_leaks") == 0
)
if gate == "oracle" and aa["degenerate"] == 0 and aa["complete"] and _xl_complete and _canary_good:
    ...  # retain existing oracle promotion body
else:
    ...  # retain existing fail-closed status; do not set oracle_ready
```

**Integration detail:** The existing function has `if gate=="ref" ... elif ... else ...`; introduce `_xl_complete` and `_canary_good` before that conditional and replace only its oracle predicate, not its surrounding branching. The snippet above illustrates the intended predicate, **not** a literal drop-in replacement for the `if/elif` structure.

**Test:** valid short A/A with canary `errors=1`, or incomplete XL A/A, must leave `VMS["oracle_ready"]` false; full clean A/A and canary must promote the oracle.

### Fix 2 — EMS XL / oracle canary fail-closed

In `builders/build_ems.py` at `_gated`, replace the XL/canary checks with:

```python
# XL was generated from real prompts for this benchmark; if required it must have passed.
if _X and a.get("xl_gate") != "PASS":
    return False
if not _X and a.get("xl_gate") not in (None, "PASS"):
    return False
oc = EMSR.get("canary_oracle")
if not isinstance(oc, dict) or oc.get("foreign_leaks") != 0 or oc.get("errors") != 0:
    return False
if oc.get("server_not_idle") is True:
    return False
```

Keep the candidate canary/error checks already present. The test is allowed to report `XL_NOT_TESTED_BUDGET`/`INCOMPLETE` and keep its 8k/40k speed data, but cannot certify adoption.

**Test:** inject `xl_gate=None`, `xl_gate='INCOMPLETE'`, and oracle `errors=1` separately into an otherwise valid EMS record; all must return `False`. A valid `xl_gate='PASS'` plus clean oracle canary must preserve eligibility. For the reference, `probe_complete_vs_target` also needs to remain true (rev2e-e already fixed).

### Fix 3 — VMS request coverage: enforce first-token coverage separately from occupancy

Add at the end of `vms_window_healthy`:

```python
and w.get("no_first_token", level) == 0 \
    and w.get("threads_not_joined", 1) == 0 \
    and w.get("idle_after") is True
```

The last two are already indirectly mapped to `error` by the current `finally`; making them explicit avoids regression. For a diagnostic run, retain `no_first_token` without discarding the measured tok/s.

**Test:** 16 expected streams, healthy counters, but `no_first_token=1` ⇒ `measurement_ok=False`. All 16 first tokens, clean join/idle ⇒ `measurement_ok=True`.

### Fix 4 — Ensure 2f CPU evidence is SHA-bound

After building the notebooks, produce a CPU report containing the **current** three SHA256 hashes, each run's Python version, return code, number of tests, failure list and trace; plus an INT4-negative test. Store the report inside the review archive, not just in external storage. The supplied older run3 data must not be renamed as v4 evidence.

```python
import hashlib
from pathlib import Path
for path in sorted(Path("notebooks").glob("Q27-XHIGH-*ENGINE_MATRIX*.ipynb")):
    print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
```

**Test:** manifest SHA == report SHA == on-disk notebook SHA for EMS SAFE, EMS RISKY, and VMS; deliberately mutate one byte and require the checker to fail.

### Fix 5 — EMS per-arm crash isolation, including dynamic MTP extension

In `builders/build_ems.py` around lines 916–922, protect the dynamically selected `_s20` arm:

```python
if _name == "dflash_w6_s24":
    _mt = {k: v for k, v in EMSR["arms"].items()
           if v.get("family") == "mtp" and _gated(v) and v.get("short_tok_s")}
    if _mt:
        _best = max(_mt, key=lambda k: _mt[k]["short_tok_s"])
        _dyn = _best + "_s20"
        try:
            _arm(_dyn, set_=EMSR["arms"][_best]["set"],
                 drop=tuple(EMSR["arms"][_best]["drop"]), streams=20,
                 env=EMSR["arms"][_best].get("env"))
        except Exception as e:
            EMSR["arms"][_dyn] = {"ok": False, "adoptable": False,
                                  "error": f"dynamic arm crashed: {type(e).__name__}: {e}"[:500]}
            if _dyn not in EMSR["order"]:
                EMSR["order"].append(_dyn)
            try:
                e11_kill_server(wait=True)
            except Exception:
                pass
            e14_save()
```

If the `_best` arm has already been selected, include its source arm name in the failure record. Wrap the final `ems_tp2()` invocations with analogous per-arm guards if the promise is truly *every arm* crash isolation.

**Test:** monkeypatch `_arm` to raise for dynamic suffix `_s20`; verify later `mtp_2_1_3`, `dflash_w4`, etc., still execute and a failure record is saved.

### Fix 6 — Replace overly broad numeric-grid exemption

Existing heuristic `sum(ch.isdigit() for ch in core)/len(core)>=0.30` exempts all digit-heavy output, including garbage; use **structural** grid validation, not a digit proportion:

```python
def vms_grid_like(s):
    s = (s or "").strip()
    # Pure, separated numeric grid / bracketed matrix. Not a stream of 250 unseparated zeros.
    return bool(re.fullmatch(r"[0-9,\[\]\s]+", s)) and \
        bool("[" in s or "\n" in s or re.search(r"\d\s+\d", s)) and \
        len(re.findall(r"\d+", s)) >= 2
```

Equivalent update required for EMS `_text_bad`. This is a **conservative illustrative heuristic**, not a semantic ARC-grid parser; another safer policy is to retain `suspicious_output` as a reviewable warning rather than automated FAIL when there is ambiguity. In long mixed reasoning, find actual nested grid spans rather than exempting the entire message.

**Tests:** `"0"*250` bad; `"0"*40+"!"*60+"0"*60` bad; `"[[0,0],[0,0]]"` valid; multiline `"0 0 1\n0 0 0"` valid; ordinary prose valid.

### Fix 7 — Canary coverage must be reported as INCONCLUSIVE when ineffective

Retain existing `foreign_leaks`, `errors`, `own_hits`; add:

```python
c["informative"] = c.get("own_hits", 0) > 0 and c.get("errors", 0) == 0
c["classification"] = (
    "FAIL" if c.get("foreign_leaks", 0) > 0 else
    "INCOMPLETE" if c.get("errors", 0) > 0 else
    "INCONCLUSIVE" if not c["informative"] else "PASS_NO_OBSERVED_LEAK"
)
```

This does **not** prove that even `PASS_NO_OBSERVED_LEAK` guarantees state isolation: it only means this particular detector showed no foreign sentinel. For production, design a reliable positive control; do not confuse `own_hits=0` with a test proving no leak.

### Fix 8 — Bounded TP2 worker joins

`builders/build_ems.py:1047–1060`: `_both()` starts two workers then calls `join()` without a timeout. Bound each join by the per-run deadline, classify still-alive threads as `INCOMPLETE`, and kill the corresponding server(s) to unblock sockets before proceeding; do not treat partial result as PASS. Avoid a new arbitrary 30-minute wait beyond the notebook hard stop.

```python
for t in ths:
    t.join(timeout=max(0.0, min(60.0, min(c.deadline for c in ctxs) - time.perf_counter())))
if any(t.is_alive() for t in ths):
    for srv, _ in servers:
        try:
            srv.kill()
        except Exception:
            pass
    return [{"error": "TP2_WORKER_DEADLINE"}, {"error": "TP2_WORKER_DEADLINE"}]
```

**Test:** one worker sleeps indefinitely (mock), one returns; comparison exits before deadline and `gate_verdict != PASS`. Mark real threads daemon or make termination bounded; a Python thread cannot be forcibly killed safely.

## 4. Additional interpretations / limits

- `vms_window` measures `end_to_end_tok_s` as a counter delta from submission to **window end** and aborts streams in `finally`. This is *not* equivalent to ARC task completed in wall-time or fully completed streamed responses. Label it `submission_to_window_end_tok_s` to avoid over-selling it; a task-level benchmark is still needed.
- A `BOUNDARY` parity verdict is intentionally acceptable in this harness, but it must **not** be described as bit-exact. Need ARC task-level test at `temperature=1.0, top_p=0.95, top_k=20` before production. These are competing policy choices, not proof of wrong output.
- 256-token greedy + ~85k prompt only tests long **input** state; it does not certify 60k–90k **generated** token stability or churn/reject transitions. Keep production winner pending a separate long-generation gate.
- SAFE/RISKY split avoids a crash contaminating another Kaggle notebook but repeats the reference/oracle warm start and associated GPU cost. The dynamic `_s20` crash hole is **within** SAFE and still matters.
- No claim is made that PR #45151 has an actual positive memory gain on L4; it must be evaluated with resident memory + KV tokens.

## 5. Short GO / NO-GO gate

| Stage | Decision | Why |
|---|---|---|
| Bundle structural audit, source exists, notebook SHA verified | **PASS** | File inspection and hash match |
| New rev2f CPU-kit evidence | **NOT VERIFIED** | Only old-revision CPU reports provided |
| Fast diagnostics to gather evidence | **CONDITIONAL** | Can record results but forbid adoption |
| Auto-select any `adoptable=True` arm as production | **NO-GO** | Source-confirmed fail-open paths above |
| Full 12-h ARC agent run on candidate | **NO-GO until gates repaired** | Correctness and long-run test remain open |

### Most economical order

1. Fix R2F-01, -02, -03, -05, -10 and grid exception; add negative tests.
2. Regenerate all 3 notebook files; produce SHA-linked CPU kit reports for EXACT hashes. Attach as evidence.
3. Run target-only oracle and 1 reference MTP/DFlash arm on L4 to confirm real greedy/token-ID, canary, XL, stream and cache metrics.
4. Test same-config MTP vs PR #45151, then DFlash W6 s20/24; leave the risky FP8 PCIe IPC overlay, NTN and 2×TP2 until proven worthwhile.
5. Separately qualify 60k+ generated tokens under churn and real ARC score. The matrix alone cannot certify this.

## 6. Direct literal source URLs (do NOT replace with link-only titles)

These are background reference URLs corresponding to the **bundled pinned source snapshots and explicitly referenced upstream issues**, not claims of a fresh live status check. All concrete defects above were identified from the uploaded archive itself.

- https://github.com/sgl-project/sglang/tree/v0.5.21/python/sglang/srt
- https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/eagle_worker_v2.py
- https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py
- https://github.com/sgl-project/sglang/issues/44145
- https://github.com/sgl-project/sglang/pull/45151
- https://github.com/sgl-project/sglang/issues/36548
- https://github.com/sgl-project/sglang/issues/38009
- https://github.com/vllm-project/vllm/tree/v0.32.0/vllm
- https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/speculative.py
- https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/qwen3_5_mtp.py
- https://github.com/vllm-project/vllm/issues/58807
- https://github.com/vllm-project/vllm/issues/53912
- https://github.com/vllm-project/vllm/pull/56531

**Crucial:** Every above link is a standalone raw URL, so Claude can copy/paste it into its research tools. The exact builder code under audit is supplied inside the user ZIP and is not a public GitHub file in this evidence set.
