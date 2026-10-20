# Experiment 4 — Q27-AGENT REPL Agent and HYBRID (v1 → v1.8.3)

**Status:** closed (2026-10-12). The real GPU EVAL12 gave **agent 0/22**, against DIRECT's 3/22. The failures were in the tool interface and turn design; the redesign is Experiment 5.

## Origin

**Source:** the agent package comes from the public Kaggle notebook **`madarshbb/arc-qwen-3-8-27b-agent`** (copy in `resources/Reference Notebooks/LB_compare/`).
- It is a REPL tool agent: a `python` tool plus `submit(transform)` verified on the demos. The options are described as "duck harness takeaways".
- Its own submitted config was **`mode: "direct"`** (40,960 thinking, temperature 1.0).
- Its comments record "probe24: … never submitted" and "mini2arm: the text nudge was read and ignored".

**What we added:**
- dual-mode evaluation;
- a server ladder with tool smoke and an end-to-end probe;
- a guard (audit hook);
- a HYBRID DIRECT rescue (the LB-26.94 method);
- a recorded-DIRECT comparison;
- the DRYRUN mode;
- a metrics sampler;
- static and GPU-path gates.

## Versions (each through a GPT audit; all findings verified first)

| Version | Main changes |
|---|---|
| v1–v1.4 | First builds. Fixes: libcuda shim, readiness and tool smoke, parser, official micro metric, validator, roundtrip smoke, no `preexec_fn`. |
| v1.5 | Effort `high` → xhigh (the template only knows xhigh / medium / low). |
| v1.6 | HYBRID (DIRECT rescue), audit-hook guard, two-tier caps 24k / 36k, circuit breaker. |
| v1.7 | Deployed-policy score, fail-closed guard + canary, `train` scope check, e2e probe, stage throughput. |
| v1.8 | Rescue = the LB-26.94 method (49k per test input); recorded-DIRECT eval; DRYRUN. |
| v1.8.1 | `urllib` local-import bug fixed; real GPU-path test; pyflakes gate. |
| v1.8.2 | E2E NO-GO gate; DIRECT decoupled from the agent fallback; rescue log. |
| v1.8.3 | MTP ×3; time-aware DIRECT budget; KV / metrics sampler; e2e soft limit. |

## GPU runs

| Run | Version | Account | Outcome |
|---|---|---|---|
| 1 | v1.4 | esfersami50 | Infrastructure OK; every agent request HTTP 400 (effort `high`) |
| 2 | v1.7 | esfersami50 | Crashed after 14 min: `KeyError: 'base_url'` in the rescue probe |
| 3 | v1.8 | foysal | Server up but never detected (`import urllib.request` made `urllib` local); cancelled after about 40 min |
| **4** | **v1.8.3** | foysal | **Complete.** Server 510 s, MTP ×3 acceptance 0.736, KV peak 31%, 0 preemptions. **Agent 0/22** |

DRYRUNs on Kaggle CPU: v1.8 27/27, v1.8.2 27/27, v1.8.3 29/29.

## Why run 4 failed (from `attempts.jsonl`)

1. **The model never submitted on its own.** Every one of the 82 submits came from forced named `tool_choice` turns.
2. **Forced submits carried no program.** 59 had empty `code` (an infinite loop: empty code does not reset the force counter). 22 carried exploration code using `train`, rejected as nonportable. Only 1 program reached verification.
3. **5 attempts died on vLLM 0.19 HTTP 500**: `assert content is not None` on forced tool-choice turns.
4. **Thinking was capped at 6,144 per turn**, and the effort ladder stepped down in 22 of 24 attempts. DIRECT's correct answers needed 26–37k continuous thinking. The user had forbidden budgets below the base; this is now a rule.

## Folders

**Experiment Notebooks/**
- `Q27-AGENT-v1 … v1_8_3.ipynb`.
- `Build/Q27A_build/`: builder `make_q27_agent.py` (v1.8.3, byte-identical rebuild); `direct_ref_eval12.json`; tests (`test_v18_local`, `test_gpu_path_local`, `run_notebook_dryrun_local`, …); `check_static.py`.

**Experiment Review/**
- every audit zip sent to GPT (v1 … v1.8.3);
- the DIRECT-vs-AGENT weakness zip;
- the EVAL12 result zip.

**Experiment Results/**
- `GPT Outputs/`: GPT's audits, verbatim (deep audit v1.5, NO-GO v1.6, NO-GO v1.8.1, timing review, GO v1.8.3).
- `Kaggle Runs/`: GPU runs 1–4 and the three DRYRUNs.
- `Analysis/`: EVAL12 per-attempt table, summary, every submitted program, the HTTP 500 traceback.
