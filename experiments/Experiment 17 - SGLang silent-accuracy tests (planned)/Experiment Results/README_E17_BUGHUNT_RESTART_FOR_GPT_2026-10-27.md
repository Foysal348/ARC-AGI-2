# E17 bug hunt: the failed run, and how to restart it

**What we want from you:**
1. Review the failed bug-hunt notebook and its GPU output. Tell us what went wrong in the design, not only in the code.
2. Give us a corrected plan for finding a "tiny kernel issue" in our pipeline that makes the hidden LB score much lower than CV.
3. Mark anything you cannot verify as UNVERIFIED.

## What is in the zip

- `notebook/`
  - `Q27-XHIGH-E17_SGLANG_SILENT_ACCURACY.ipynb` (sha 9793b41e, profile E17SHORT);
  - its builder and cells (`build_e17.py`, `e17_cell.py`, `e17_report_cell.py`).
- `src/`
  - the instrumentation hook `e17_hook.py` and its tests;
  - the fake-server harness generator and the harness.
- `run/`
  - the full Kaggle output of the GPU run (`foysalemonshanto/q27-xhigh-e17-sglang-silent-accuracy` v1, 2026-10-26);
  - the server logs of the four arms, the hook traces, `e17_report.json` / `.md` and the kernel log.
- `README_E17_SHORT_GPU_RUN1_FOR_GPT_2026-10-26.md`: our diagnosis of that run, and the options we saw then.

## What happened

- E17 tested one hypothesis: the SGLang #43567 stale lazy Mamba checkpoint silently corrupts our model's outputs.
- All four arms crashed with `Can not alloc mamba cache` (the #43559 path) before the #43567 path could be reached.
- The verdict was NOT_EXPOSED_OR_INCONCLUSIVE.

## New information since then

**1. Our best notebook does not run on SGLang.** S1-60K (LB 34.58 and 31.94) runs on vLLM 0.19, with FP8 weights, fp8 KV, MTP 3 and prefix caching. Every SGLang / INT4 / DFlash submission scored lower (17.50-29.17). So an SGLang-only bug cannot explain S1's gap.

**2. All posts by mikelou1, verbatim** (Kaggle ARC-AGI-2 forum, topics 748082 and 750546):

| Date | Post |
|---|---|
| 10-11 | "OK It finally graded. Qwen scored ~25%. After some minor optimizations I hope it can get to 40%. I still have a lot of ideas and I believe it has the intelligence to get to 50%, even 85% might be possible." |
| 10-13 | "OK It got 36%. I have a submission with projected ~50% ± 10% running right now." |
| 10-13 | "No plans to opensource as I believe I have a good shot at 85%" |
| 10-16 | "Not a NVARC ensemble, as my program don't have time to do that. However there is a tiny kernel issue I found that made my program score only 40% (instead of the 60% projected by CV)." |
| 10-18 | "UPDATE: It scored 66% (~60% on CV)" |
| 10-19 | "With some tricks and stuff I currently get around 79% ± 5% in CV runs." |
| 10-19 | "The CV run used LB-ish environment (as my 66% submission got 70% on CV, I expected better on this run but it appears something regressed, likely some technical problems I'll fix later today)" |
| 10-19 | "appears to be overfit as lb is slightly easier than cv (despite it scoring a ~10% lower for me too). my cv -> lb is usually within noise and pretty consistant" |
| 10-20 | "With time constraints at around 4xL4 speed I got Qwen to 87% (and from reasoning traces it got the rule to almost every problem) so Qwen certainly has the intelligence to do so." |
| 10-20 | "The 87% will likely not transfer. I would be surprised if it scores even 80%. I do hope this run gets me to 75%-ish though as I have a few other ideas I am testing." |

- His LB went 66.39 (10-17), then 69.72 (10-21).
- He never said whether he uses the 27B or Flash-Next, which engine, or what the issue was.

**3. "Kernel" may not mean a CUDA kernel.** On Kaggle a notebook is called a "kernel". The issue could be:
- a notebook or environment bug that only shows in the hidden rerun: the rerun code path, the 12-hour clock, 240 tasks, units cut at the end, or submission writing;
- or a GPU kernel / numerics fault.

E17 assumed the second reading only.

**4. Observations from our vLLM 0.32 smoke run of S1 (2026-10-27).** These are the candidates we could see:
- **fp8 KV with scale 1.0.** The log says "Using KV cache scaling factor 1.0 for fp8_e4m3" and "Checkpoint does not provide a q scaling factor". The FP8 repack has no KV scales. We never compared fp8 KV with bf16 KV on accuracy.
- **Re-prefill with almost no prefix-cache hits.** `mamba_cache_mode` "align" plus MTP gives only 7.7 % prefix hits. Every forced-close phase 2, and the final drain, re-prefills the whole phase-1 context. In R5, 54 % of requests hit the cap. At the final drain, about 20 units × 50k tokens at about 2.5k tok/s ≈ 6.7 min, against an 8-minute margin.
- **Hybrid MTP + prefix caching.** vLLM issue #53912 reports empty or punctuation output. Our short greedy gate passed, but long ARC chains were never checked.

## Questions

1. Why did E17 fail as a design: one hypothesis, an SGLang-only stack, and an upstream repro recipe that asserts on 0.5.21? What should a bug hunt for our real stack (S1 on vLLM) look like?
2. Which candidates would you test first, and how? Consider the notebook/environment candidates, fp8 KV scale 1.0, re-prefill and drain loss, and MTP + prefix caching. Give the cheapest decisive test for each. GPU runs should be at most about 60 min.
3. How do we build an "LB-ish" CV run like mikelou1's, so that CV predicts LB? Our LB is 120 hidden tasks (240 in the full test), 12 h on 4x L4, noise sd about 1.5.
