# Claude's audit of implementation A (`Q27_XHIGH_v1_4_4_E7_CODEX_ORDERING_KV.ipynb`, sha256 0277a24e), 2026-10-12

## PASS

- **Base and scope:** v1.4.4 base; only cells 1, 3, 5 and 6 changed.
- **Tests:** its 10/10 CPU tests pass. Experiment 1 harness `mixed` / `drain` / `rerun` pass.
- **Hidden-mode equivalence:** the default `rerun` `submission.json` and all 259 records are identical to v1.4.4. With Codex ordering on, the submission is also identical, and the order equals the verbatim NVARC order (tiers 178 / 52 / 10).
- **Fixed 18:** with engine cap 20, `max_running` reached 18.
- **Metrics:** the real vLLM metric names parse correctly from the run 7 snapshot.

## Issues

- **BUG (blocks adaptive):** `_E7_PREFERRED` requires `str(attention_backend_resolved).upper() in ("FLASHINFER", "NONE")`. On a real GPU the value is `"AttentionBackendEnum.FLASHINFER"`, so adaptive always degrades to baseline.
- **Missing:** no PROXY-48 or deadline-constrained profile.
- **Minor:**
  - environment-variable-only configuration;
  - the rebuild on Windows writes CRLF line endings (cells identical);
  - `math` is imported twice.
