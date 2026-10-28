# Independent research audit: mikelou1 40% -> 66% and our CV-to-LB gap

**Date:** 2026-10-26. **Read first:** `README_FOR_GPT.md` from `KERNEL_ISSUE_40_TO_66_RESEARCH_PACK_FOR_GPT_2026-10-26(1).zip`.

**Verdict:** The identity of mikelou1's "tiny kernel issue" is **UNVERIFIED**. The main risk directly applicable to our SGLang 0.5.21 `extra_buffer_lazy` production is **PR #43567** (confirmed upstream silent cache-state corruption; user-local failure and LB gain unverified). There is also strong evidence that the R5 48.61 eval -> 21.39 LB gap can be explained mainly by workload coverage. DO NOT turn hypotheses into a 20+ point recovery promise.

## Evidence classes
- **UPSTREAM-VERIFIED:** reproducible result from upstream issue/PR; not necessarily reproduced on our exact model and L4.
- **BUNDLE-VERIFIED:** source/log evidence within supplied zip.
- **INDEPENDENT-CPU-TESTED:** experiments run here against supplied files.
- **INFERRED:** mechanism plausible, still needs controlled test.
- **UNVERIFIED:** cannot confirm the claim.

## 1. mikelou1 timeline, exact claims, and limitations

Original forum thread: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082
Additional conversation: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546
Submission queue context: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/747515

Sources in pack: `01_competitor/A_mikelou1_all_posts_and_interpretations.md` and `01_competitor/kaggle_api_fresh_pull_2026-10-24/topic_748082.md`.

- 2026-10-11: reported ~25% Qwen LB.
- 2026-10-13: reported 36% and a projected ~50% +/- 10% next submission.
- 2026-10-16: explicitly said a “tiny kernel issue” caused actual 40% instead of 60% projected CV.
- 2026-10-18: reported 66%, ~60% CV.
- He did **not** disclose the bug, engine, checkpoint, GPU runtime, GPU kernel, or whether he used Qwen3.8-27B versus Flash-Next. His earlier usage of "kernel" referred to a Kaggle notebook session. He said CV was “LB-ish” and at “around 4xL4 speed”; no direct evidence tells us which hardware ran his CV.
- Neither the date/order nor the word "kernel" proves the cause of the 26-point gain. No direct publication resolving the mystery was found in source pack or accessible public web.

## 2. Ranked causal hypotheses for *his* 40 -> 66 leap

| Rank | Hypothesis | Supporting evidence | Weakness | Cheapest falsification | Specific fix if confirmed | His LB impact |
|---|---|---|---|---|---|---|
| 1 | **Kaggle notebook/competition rerun logic**: wrong elapsed-time origin, different task count, incomplete submission overwrites, wrong branch/accelerator | His only other use of “kernel” is Kaggle notebook; CV vs scored rerun is the key contrast; 240 hidden tasks vs local 120; different pools reportedly used | His own rerun code is private | CPU replay with 120 vs 240 synthetic tasks, diff `KAGGLE_IS_COMPETITION_RERUN` branches and submission JSON, simulate delayed startup and verify each task has a final/fallback row | One monotonic start; compute budget from real start and actual task count; atomic submission flush and task coverage assertions | Potentially large, **unquantifiable without his logs** |
| 2 | **GPU kernel/performance only on L4/sm_89**: slow GEMM, GDN, NCCL TP4 PCIe, absent CUDA graphs, speculation disabled, JIT overhead | He has published GPU-performance patches; his CV mentions speed equivalent to L4, not necessarily actual L4; time-limited task coverage can amplify throughput loss | No evidence of his engine or kernel; no direct bug disclosure | Diff task-throughput and accepted-token/sec on *real* Kaggle 4xL4 vs scaled CV; GPU kernel trace and startup/JIT timings | Version/hardware-specific kernel choice; restore graphs and speculation after correctness gates; topology-aware TP2 A/B | Potentially large when time-limited; **unknown** |
| 3 | **GPU silent output corruption**: recurrent-state speculative verify, lazy cached checkpoint, W4A16 driver cache | Actual upstream examples of HTTP-200 wrong output without crashes | His engine/checkpoint unknown; #43567 merged Oct 11 but no proof he used it | Exact token-ID/cold-warm logprob KLD controls with cache eviction pressure, serial/concurrent and longer decode | #43567 and relevant upstream state fixes; fresh driver cache; fallback target-only/MTP if corruption | **Unknown, no defensible score figure** |
| 4 | **OS/container kernel/sandbox failures**: disallowed `unshare`, process/thread caps, seccomp, mmap/shared memory, CPU scheduling | Sandbox execution can fail on scored pool but not CV | No evidence his solver uses such syscalls; he never named an OS kernel | CPU feature audit in scored-container startup, count verify_code failures, subprocess exit codes, timeout errno | Avoid unsupported namespace/syscalls; limit workers, ensure CPU availability and timeouts | Unknown, currently LOW evidence |
| Excluded | **GCSFUSE October 21-23 storage regression as his October 16 cause** | Storage issue later documented | Chronology conflicts: 40 -> 66 before outage | Compare event dates | Use Kaggle Models NFS / stage cached weights, if issue recurs | **Cannot explain his Oct 16 score** |

**No invented probabilities** (45/40/15 are prior subjective estimates in bundled notes, not measured posterior confidence).

## 3. Our stack: ranked tests for *our* risks

### P0 — SGLang #43567: lazy Mamba stale-cache reuse (**UPSTREAM-VERIFIED, our impact UNKNOWN**)

Primary: https://github.com/sgl-project/sglang/pull/43567
Follow-on dependency: https://github.com/sgl-project/sglang/pull/43559
Release containing fix: https://github.com/sgl-project/sglang/releases/tag/v0.5.22

- Upstream: ordinary decode, 1 running / 4 Mamba slots, Qwen3.5-0.8B on GB300; warm 128-token prefix, next 255-token request, 32 generated tokens. KL on stale hit 0.6494482615 and after fix 0, HTTP success in both cases. **DFlash speculative verify path not covered by the PR's registered end-to-end test.**
- Our exact flags: SGLang **0.5.21**, `extra_buffer_lazy`, `--max-mamba-cache-size 68`, 16 active streams (4 spare slots by 4-slot capacity sizing), long repeated prefixes; upstream fix postdates 0.5.21. `04_code/e14_sglang_patches.py...py` has two #43567 hunks, but `DEFAULT_NAMES = ("44401", "41872")`; importing module alone never applies the #43567 patch. E15P code imports `E14P` but has no `E14P.apply()` call in the supplied notebook.
- Cheapest CPU check: after engine installation, `E14P.check(installed_pkg_dir, ['43567'])['patches']['43567']['state']` must be `applied`; if `not_applied`, do not claim fixed.
- GPU gate: run target-only A/A at the original 128/255/32 pattern with very tight pool, cold vs cache-seeded, logprob KL and token IDs, BEFORE and AFTER applying the exact two hunks on the same engine. Then run analogous DFlash W6 with forced rejection and churn (because original PR did not prove spec path). Require no KL outside clean A/A tolerance, no stale cache length, 0 stream errors.
- Patch: **BEFORE starting the SGLang server**, after wheel installation with installed Python package directory resolved:

```python
import json
# E14P is the bundled e14_sglang_patches module.
# pkg_dir = Path(venv)/'lib/python3.12/site-packages/sglang'
pre = E14P.check(pkg_dir, ["43567"])
assert pre["patches"]["43567"]["state"] in ("not_applied", "applied"), pre
rep = E14P.apply(pkg_dir, ["43567"])
assert rep["ok"], json.dumps(rep)
post = E14P.check(pkg_dir, ["43567"])
assert post["patches"]["43567"]["state"] == "applied", post
```

Do not modify `DEFAULT_NAMES` globally and silently run all backports; explicit patch name, source matching, backup and post-check are required. **Our LB effect: UNVERIFIED (possibly zero, possibly material).**

### P1 — DFlash GDN/verification concurrency (**UPSTREAM-VERIFIED different GPUs, our impact UNKNOWN**)

- https://github.com/sgl-project/sglang/issues/36548
- https://github.com/sgl-project/sglang/issues/38009
- https://github.com/sgl-project/sglang/issues/35150

#36548: target-only 0/304 wrong vs DFlash8 111/304 wrong at concurrent 8 in a GB10 Qwen3.8 NVFP4 case; width4, fp32 SSM, disable graph each reduced but did not eliminate (37/304). Dense BF16 LM head nearly eliminated in this *different* checkpoint, not a universal cure.

Cheapest smoke: pinned token IDs, 32+ prompts / 4,096+ generated tokens with serial, 8, 16 concurrency; target-only vs DFlash W6 vs MTP; forced rejection; full logprobs at branch position and cache transition. Include 60k+ long-run probes before claiming production-safe, but do not spend first GPU slot there. Fix if reproduced: switch to proven target-only/MTP or correct spec state implementation; limit concurrency only as a measured mitigation.

**Do not conflate** https://github.com/sgl-project/sglang/issues/43514 (DFlash accept-rate 0 and repeated `!` under **grammar-constrained requests plus concurrent multi-chunk long prefill**) with our Poetiq `/generate` (no required OpenAI tool grammar). It may be relevant to a Duck OpenAI tool/grammar path only if actual request payload activates grammar. If so, log `accept_rate`, `spec_verify`, suspicious repeated tokens and grammar requests in a single batch.

### P1 — INT4/driver cache and Marlin numerics

https://github.com/sgl-project/sglang/issues/42917

Reported `dbirks` W4A16 PPL 9.98 on SGLang 0.5.21 vs 6.05 on vLLM; later isolated to **stale CUDA driver compute cache** (`SGLANG_CACHE_DIR/nv`), not a universal W4A16 weight mapping problem. Other checkpoint and sm_120; RedHat INT4 on L4 not implicated by proof. Our E15P uses `/tmp/e11-cache/sglang` which appears run-local, reducing but not eliminating risk.

Test: actual worker effective `CUDA_CACHE_PATH`, cold vs warm 2k/4k fixed token logprobs/PPL, same and multiple boots; compare vLLM FP8 and SGLang target-only. Fix: isolate/clear **actual** SGLang driver `nv` cache before process start, then retest; clearing `jit/` alone was ineffective in source report.

### P2 — fp16 GDN recurrent state vs config fp32

Our model config requests float32 but production uses `--mamba-ssm-dtype float16`. Dataset shows no long-decode (60k-90k) parity evidence. Existing short prompt comparisons are inconclusive. Test: **fixed same target weights and KV dtype**, change only SSM float16 vs float32; score *full supplied prompt+known completion IDs* in 8k position buckets; compare near-tie flips and accuracy, then decode under real concurrency. Larger state footprint may reduce KV and throughput. LB effect unknown.

### P2 — FP8 KV scale and token selection

Our checkpoint and SGLang source suggest target calibrated k/v scale, draft 1.0; runtime layer scales not dumped. Log each attention layer's scale after loading and compare logprobs for FP8 vs BF16 KV with checkpoint unchanged. `top_logprobs_num=20` is not automatically the **post-top_k/top_p sampler distribution**: don't assert emitted-token-outside-reported-top20 is a bug unless the API reports the exact filtered verify distribution with all sampling transformations applied. Use direct token rescoring / sampler instrumentation with deterministic reference.

### P2 — 0.5.22 speculative state fixes other than #43567

https://github.com/sgl-project/sglang/pull/43029
https://github.com/sgl-project/sglang/releases/tag/v0.5.22

#43029 uses committed length at speculative Mamba tracking boundary and is called out in 0.5.22 release for structured output+hybrid speculative issues. Poetiq `/generate` does not exercise `--tool-call-parser` grammar merely by launching with the flag. Test actual payload; prefer no unrelated backport without a matched reproduction. For true grammar-constrained Duck traffic, mixed admission/churn can activate different corruption bugs.

### P2 — vLLM MTP & hybrid prefix cache effectiveness

https://github.com/vllm-project/vllm/issues/45238
https://github.com/vllm-project/vllm/issues/58303
https://github.com/vllm-project/vllm/issues/60174

vLLM's hybrid prefix caching can miss or evict under checkpoint alignment/retention; losses are **performance**, not necessarily wrong answers. #60174 is DFlash/DSpark+prefix on 0.30/0.31, not your vLLM MTP-3. Existing VMS live log showed `Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod` for 0.32.0, so do not label the measured default as Triton. Check cache hits, prefill duration, total e2e effective tok/s, not just decode medians.

### P3 — Kaggle scored-rerun notebook / startup / sandbox

- Rerun switch in both submitted notebooks: `RERUN = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1','true','yes'}`; `/generate` SGL shim launched with `--speculative-num-draft-tokens 6`, not 8, after later override; ensure actual `launch_flags` not initial `e11_flags`.
- E15 live log: installation 157.3 s; SGLang ready in 522 s; KV pool 852,152; Mamba slots 68; graph batch 1-17; actual effective 16. No evidence from that startup alone of hidden catastrophic fallback.
- GCSFUSE dataset loading slowdown started around 2026-10-21, later than mikelou1's October 16 failure and 18 fix; not a causal explanation for his 40→66. Future risk exists for wheelhouse GCSFUSE; staging into `/dev/shm` and checking available memory (23.32 GB staging in live log) remains important.
- CPU-only smoke on *actual Kaggle kernel*: print `RERUN`, `KAGGLE_KERNEL_RUN_TYPE`, python version, `os.cpu_count`, `/proc/self/status` CPU affinity, `/proc/mounts` source of inputs, disk usage, wall-time origin, loaded library paths, actual TP GPU topology and `CUDA_CACHE_PATH`; assert `len(challenges)` and outputs count (all test instances) before final save.

## 4. Independent CPU code test — five unfenced programs

Executed files in `03_our_data/C_bare_code_check` with the bundled `runner.py` on their provided task/solution fixtures:

| Program | Train | Test |
|---|---:|---:|
| `21897d95_i4.py` | 0/4 | 0/2 |
| `4c3d4a41_i2.py` | 0/2 | 0/2 |
| `898e7135_i1.py` | 0/2 | 0/1 |
| `b6f77b65_i2.py` | 1/5 | 1/3 |
| `de809cff_i2.py` | 0/2 | 0/2 |

All scripts executed; **none** solves a full held-out task according to supplied fixtures. Therefore 5/88 unfenced outputs demonstrate a parser/format mismatch but are not evidence of 5 additional task solves; lenient parser recovery should be done, but plausible LB impact for these five alone is 0 whole solved tasks.

## 5. CV-to-LB gap: conservative quantitative explanation

From `03_our_data/Qwen3.8-27B LB and run deep analysis (2026-10-16).md`:

| Run | Task-normalized eval | LB | ratio |
|---|---:|---:|---:|
| Direct 49k | ~25.4 | 26.94 | ~1.06 |
| Direct 86k | 36.94 | 22.92 | 0.62 |
| Poetiq R5 | 48.61 | 21.39 | 0.44 |
| Duck E15B | ~39 | 22.92 | ~0.59 |

Poetiq R5 used ~704 min for 120 local tasks; scored hidden set ~240 tasks in 12 h. Linear naive coverage estimate 48.61 * 120/240 = 24.31; actual 21.39 is only 2.92 leaderboard points lower than this rough expectation. Because tasks and ordering are not uniform, this is a plausibility argument **not a fit or proof**. Older direct 49k completed most tasks and LB ≈ eval, a strong comparative control. We cannot attribute the 21.39 result to a "kernel" bug without paired replay showing differences at identical per-task budgets and token IDs.

Actual noncoverage quality clues in pack: 10/44 mid-thinking EOS in SGL P1, 0/88 SGL S1 and 0/432 vLLM S1; the prompt is changed between arms so the EOS observation is strongly confounded. Unfenced outputs SGL S1 5/88 vs vLLM S1 0/432 show different behaviour but checkpoint/runtime are jointly changed. Stage-2 resume prefill accounts for 58–66% of computed prefill tokens in Duck EVAL120, causing 18–30% hourly output-rate decline later in run. This is a real throughput/coverage cause, not necessarily a kernel correctness fault.

## 6. Minimum budget decision plan (all impact estimates conditional)

| Order | Test | CPU/GPU | What settles | Go/no-go |
|---|---|---|---|---|
| 0 | Inspect pinned `E14P.check` #43567, effective `CUDA_CACHE_PATH`, exact launch flags, notebook rerun branches, response completeness | CPU/log | Patch not applied, wrong cache path, incorrect rerun branch | Fail closed if expected patch advertised yet absent |
| 1 | Target-only cold/warm 128/255/32 test at Mamba pool pressure; no patch vs #43567, logprobs/KL | Short GPU | #43567 applicability | Any significant non-A/A KL/flip = block current lazy cache path |
| 2 | DFlash W6 serial/16 concurrent with seeded cache + forced rejected tokens; compare to target-only | Short GPU | Spec-state correctness | Large-margin flips/foreign context contamination = DFlash NO-GO |
| 3 | Fixed prompt+completion trace NLL 8k bins; fp16 vs fp32 SSM, FP8 vs bf16 KV, no model change | GPU | Long-context precision | Select only measured parity-safe memory setting |
| 4 | RedHat INT4 target-only versus vLLM FP8 model at identical token IDs, plus cold/warm driver cache on same target | GPU | Weight/Marlin numerical quality | If clear PPL/KL regression, reject INT4 path or fix driver cache |
| 5 | 240-task synthetic rerun CPU manifest, deadline, output flush and 12h budget scaling | CPU | Kaggle kernel mode/coverage bug | Must preserve all keys and valid outputs after premature shutdown |
| 6 | Actual 12h scoring/paired LB test only after 0–5 | Expensive | Actual LB uplift | No numeric uplift promised in advance |

**Expected LB effects:** #43567, DFlash, INT4, fp16 SSM all have **unknown direction/magnitude for your run** until paired validation. Throughput-only adjustments translate to additional evaluated tokens, not a fixed score increment. Crash/timeout branches can cause large losses when they prevent run completion, but no such failure appears in current E15 live startup evidence. A ~26-point jump cannot honestly be projected from any one upstream bug.

## 7. Source links — direct, pasteable URLs

- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/747515
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344
- https://github.com/sgl-project/sglang/pull/43567
- https://github.com/sgl-project/sglang/pull/43559
- https://github.com/sgl-project/sglang/pull/43029
- https://github.com/sgl-project/sglang/releases/tag/v0.5.22
- https://github.com/sgl-project/sglang/issues/36548
- https://github.com/sgl-project/sglang/issues/38009
- https://github.com/sgl-project/sglang/issues/35150
- https://github.com/sgl-project/sglang/issues/43514
- https://github.com/sgl-project/sglang/issues/42917
- https://github.com/sgl-project/sglang/issues/39597
- https://github.com/sgl-project/sglang/issues/43956
- https://github.com/sgl-project/sglang/issues/36889
- https://github.com/vllm-project/vllm/issues/45238
- https://github.com/vllm-project/vllm/issues/58303
- https://github.com/vllm-project/vllm/issues/60174

**Final status: research confidence HIGH that #43567 is worth an isolated pre-GPU-production gate; confidence LOW that it was mikelou1's private bug; no proven 20+ point gain on our model. No GPU run was performed in this audit.**
