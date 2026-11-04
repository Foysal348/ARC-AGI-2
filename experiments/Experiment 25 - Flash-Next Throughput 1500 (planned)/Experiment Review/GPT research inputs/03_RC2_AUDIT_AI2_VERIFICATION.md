# Verification of external audit #2 (03_RC2_AUDIT_AI2_2026-10-31.md) against the RC2 files

Checked by the lead on 2026-10-31. Only claims not already verified in `02_RC2_AUDIT_VERIFICATION.md` are listed.

| Claim | Evidence | Verdict |
|---|---|---|
| **MTP head QSA FP8 KV runs with default scales 1.0** | FP8 log 318-319: `QSA E4M3 KV active on mtp.layers.48.self_attn.attn: k_scale=1 v_scale=1 (DEFAULT 1.0 -- no static calibration loaded)` | VERIFIED. New. It affects the draft (acceptance), not the target's prompt logprobs, so it cannot by itself explain the TF divergence |
| K>1 reuses the single MTP layer ("may result in lower acceptance") | log 14 (`speculative.py:1411`) | VERIFIED (engine warning) |
| NFS auto-prefetch skipped: checkpoint 115.22 GiB > 90 % of available RAM (53.01 GiB) | log 77 (`weight_utils.py:1121`) | VERIFIED. Supports the host-memory diagnosis |
| Non-writable buffer warning; unbatched NCCL P2P send/recv creating a 2-rank communicator lazily | log 296-303 (the audit cited 340-347; the content is right, the line numbers are off) | VERIFIED (PP hand-off path; cost unknown) |
| `long_quality()` does not require finish == "stop" and think_closed for the natural row | `build_e13_rc2.py` `long_quality`: natural fails only on cut_by_cap | VERIFIED gap (RC2's row had both, so the RC2 verdict stands) |
| Forced 60k uses ignore_eos, so it is a stability test only | rc2_results.json long.forced60k label "synthetic past the first EOS (ignore_eos)" | VERIFIED |
| submission.json = 240-task dummy skeleton | RC2 writes the fallback submission | VERIFIED (by design) |
| Do not force `NCCL_PROTO=LL128` (NVIDIA: data corruption risk on unsupported platforms) | NCCL env docs caution; a numeric check cannot prove the absence of rare corruption | ACCEPTED: drop the forced LL128 arm from NB3 |
| Offline Kaggle: the #60490 diff must be staged as an artefact, not fetched at runtime | Kaggle GPU commits run offline | ACCEPTED (NB1 embeds the patch) |
| Per-position acceptance from 315 interval lines: 82.5 / 64.0 / 49.5 %, last 20 intervals 76.8 / 56.1 / 39.4 % | not re-parsed by the lead; whole-run 205,517 / 306,714 = 67.0 % matches metrics_end | PARTLY VERIFIED (the run total); the late drop coincides with the long low-concurrency rows, so uptime decay is unproven |
| Capacity ceilings with F = 7 (8k 16, 40k 8, 60k 6, 90k 5, 116k/120k 4) | arithmetic from 149 units | ARITHMETIC OK; F itself still unmeasured |
| 116k natural answer consistent with task 0934a4d8 | no label in the bundle | UNVERIFIED (the lead's RC2 analysis agent checks the local eval solutions) |
