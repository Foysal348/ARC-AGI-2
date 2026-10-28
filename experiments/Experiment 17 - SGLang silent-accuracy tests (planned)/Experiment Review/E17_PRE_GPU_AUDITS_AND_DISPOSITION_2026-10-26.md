# E17 pre-GPU audits of notebook 418023a5 (three GPT audits, 2026-10-26) and our disposition

The three audits were pasted by the user. They are Bengali originals and are kept in substance below; the key claims are quoted where it matters.
- All three: **NO-GO** for 418023a5.
- All three: **GO after the fixes**.
- None of them ran a GPU; none changed a file.

## Claims, checks and disposition

| # | Claim (audits) | Our check | Disposition / fix (notebook 9793b41e) |
|---|---|---|---|
| 1 | "`e17_long_ids` uses `e17_arc_prompts(1)`, which returns ONE task key, so the T2 bases with skip 12/24/36 are empty and T3a is short" (audits 1, 2) | **Wrong.** `e17_arc_prompts(n)` returns `keys = sorted(allt)`, ALL the evaluation ids; only the prompt ids are cut to n. The harness T2 bases had real lengths (9,297 / 5,141 tokens at a 1,500 target). | Hardened anyway: wrap-around over the task pool, a real token count, a `RuntimeError` if a prompt is shorter than its target, distinct-base assertion, base lengths and T3a cuts recorded in the report. |
| 2 | T3b verdict NO_HIGH_MARGIN_DIVERGENCE when every DFlash request failed (errors / missing margins not checked) (all three; reproduced by them) | **Confirmed bug.** `compare_passes` reports `large_margin_flips` 0 for errored rows. | New `e17_t3b_verdict`: every comparison (DFlash serial / conc vs target, DFlash conc vs serial, target A/A) must have expected > 0, compared == expected, errors 0, missing margins 0; otherwise INCOMPLETE. |
| 3 | T1 BUG_EXPOSED from evidence spread over different prompts / cycles (global counters + any deep hit + any hot KL) (all three; reproduced) | **Confirmed.** | Per cycle: seed and probe rids. The hook trace is read per rid (`e17_rid_evidence`: failed rids; confirmed stale donations with their length; spec path via `spec_resolve`). BUG_EXPOSED only when ONE valid cycle has: this seed failed, this seed's confirmed stale donation of length L, this probe's `cached_tokens == L > safe depth` (the restore link: the cache was flushed at the start of the cycle, so the only node of depth L is this seed's), and KL > the tolerance. Chain records are stored per cycle. |
| 4 | `tol = max(5e-3, 10 x A/A)` is fail-open (A/A 0.10 hides KL 0.649) (all three) | **Confirmed.** | Fixed tolerance 5e-3. The reference is valid only if every reference flush was valid, every reference hit equals the safe prefix, and the A/A max KL is <= 1e-3; otherwise INCONCLUSIVE_UNSTABLE_OR_INVALID_REFERENCE. (Audit 3 proposed 5e-4; we use 1e-3 = tol / 5 as the stability bar; the A/A is expected to be ~0 on a flushed, identical path.) |
| 5 | Revert not verified before the "unpatched" T2 (all three) | **Confirmed.** | `e17_revert_verified`: kill the server, revert, then require revert ok + state `not_applied` + the Phase-0 sha of `schedule_batch.py`, or raise PATCH_REVERT_FAILED (no unpatched arm runs). It is used on the T2 path, at the end, and in the error handler. |
| 6 | T2 probes without a cleanup gate (threads alive, `server_idle` ignored, flush unchecked, errors ignored) (all three) | **Confirmed.** | Rid-tracked streams; stop event + abort of the live rids + `join_all` + an idle check (a metrics-based `e17_idle`: missing metrics are NOT idle). Alive threads, not idle, or any stream error → INCOMPLETE. Cold probes need a valid flush and `cached == 0`; cold A/A <= 1e-3, else INCOMPLETE_COLD_CONTROL. A divergence is labelled STALE_DONATIONS_AND_WARM_DIVERGENCE: T2 has no per-request provenance, so it is never "confirmed". |
| 7 | Every T1 flush must be valid (idle + ok) (audits 2, 3) | Accepted. | `e17_flush` returns `valid = ok and idle (metrics)`. A cycle with an invalid flush is excluded. No valid cycle → INCONCLUSIVE_INVALID_COLD_CONTROL. |
| 8 | The hook self-test merges names over pids; a partial per-rank install could pass (audit 3) | Accepted. | The self-test requires 4 fully patched processes (TP4; 1 on the CPU fake). |
| 9 | The KL threshold, 248k-id JSON cost on L4, and the real overlap race on our stack: UNVERIFIED (all three) | Agreed. | Measured by the session itself (timings are in the logs). |

## Regression tests added (harness, both profiles)

- **T1:**
  - disconnected evidence (cycle A hot, cycle B deep + confirmed) is not BUG_EXPOSED;
  - the linked chain is BUG_EXPOSED;
  - a confirmed donation of another length breaks the link;
  - an unstable reference gives INCONCLUSIVE;
  - an invalid flush gives INCONCLUSIVE.
- **T3b:** complete gives a verdict; 16/16 errors, 1 missing margin, or a missing comparison give INCOMPLETE.
- **Revert:** a failed revert (mock `ok=False`, state applied) raises PATCH_REVERT_FAILED and adds 0 launches.
- **Hook:** every hooked launch passes the per-rank self-test; T1 cycles carry rids and chain records.
