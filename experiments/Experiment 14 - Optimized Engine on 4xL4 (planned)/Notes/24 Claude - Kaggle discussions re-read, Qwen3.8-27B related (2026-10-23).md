# 24 — Kaggle discussions re-read: everything about our model, Qwen3.8-27B / Flash-Next (2026-10-23 ~18:00 UTC)

**User request:** "আবার ডিস্কাশন গুলো পড়ে আয়" … "এই আমাদের মডেল রিলেটেড পড়" ("read the discussions again … read the ones about our model").

**Source:**
- A fresh dump of all 35 ARC-AGI-2 threads on 2026-10-23: `_scratch/qwen38_search/disc_2026-10-23/` (script `disc3.py` + `dump_threads3.py`).
- 738313 failed with HTTP 429; it has no update.

**Read in full:**
- 748082, 750546, 745787, 746149, 749756, 750575, 746307;
- 742796 (from the 10-22 copy);
- 729743, 732723;
- plus the infra threads 751344, 747515 and the agent post 750577.

## A. Teams that use Qwen3.8-27B: scores and what they say

| Who | LB / CV | What they say (verbatim where short) |
|---|---|---|
| **mikelou1** (#12, 69.72 on today's LB screenshot) | LB ~25 % → 36 % → 40 % → **66 %** (10-18) → 69.72; CV ~60 % → "79 % ± 5 %" | "**Not a NVARC ensemble**" and "I don't use distillation". "A **tiny kernel issue** I found made my program score only 40 % (instead of the 60 % projected by CV)". "With time constraints at around 4xL4 speed I got Qwen to 87 % (and from reasoning traces it got the rule to almost every problem)" |
| **NguyenThanhNhan** | Pure reasoning 23.33 (10-05); later **65.97** (57422625, uses almost all 12 h) | FP8 + MTP, 65k, xhigh, 1 pass, an intervention that forces an answer after 49k. **SGLang config:** `kv fp8_e4m3`, `--attention-backend flashinfer`, `--linear-attn-backend triton`, `--chunked-prefill-size 2048`, `--max-prefill-tokens 4096`, `--mamba-radix-cache-strategy extra_buffer`, `--mamba-full-memory-ratio 0.5`, `--mamba-ssm-dtype float32`, **EAGLE (MTP) steps 3, topk 1, draft tokens 4**. "You could solve many more tasks within 12 hours by implementing a simple harness." On 10-21/22 he got 0.00 twice: server startup (45-min timeout, ~180 GB read) never finished on slow storage |
| **shiiin9** | **65.56** (10-20) → 59.31 / 58.61 under the storage slowdown | Added a workaround: parallel reads of the inputs |
| **amn** | Pipeline **72.50**; 0.00 ×3 under the slowdown | "FN [Flash-Next] with an acceleration (eg. quant?) might reach a better score … by letting it work as an agent" |
| **Kris Dreemurr** | CV **45 % (~52/120)** | Qwen harness, xhigh, "full policy median **70k tokens for pass@2 (35k per slot)**". "The issue is I can't fit it in the box" |
| **TheDivine Aurora** | ~19-20 % at **300-500 tps**, only half the tasks attempted | "practically 40 % is possible with **600-800 tps**, 1 attempt per task … more attempts certainly helps … when done, they're mostly right … it just takes **128k-256k context** window" |
| **Adarsh** | 15 → 21.5 (public agent notebook `madarshbb/arc-agent-sub-q38-direct-b98k-c16`) | "xhigh is just so slow that it never reaches most of the tasks". "**python repl should do almost all of the thinking** for qwen and reduce the token usage" |
| **Hiểu Vy** | LB 26.81 (original 27B, 10-10); "80 % on the 120 validation tasks" (offline) | "the other teams … found ways to make their models run faster" |
| **Anil Thomas** | 35 → 45 by ensembling NVARC + Qwen | Suspects distillation into a small student that fits one L4 ("the slow link between the GPUs doesn't hurt you"). mikelou1 replied that he does not distill |
| **Darren / Bushi Liu** | CV > 75 %, does not transfer to LB | "LB seems to be a bit harder than CV, or the harness may be a little bit overfit to eval set" |
| **ARC Prize official** | Qwen3.8-27B xhigh **42.36 %** semi-private, $0.447/task | Darren: ≈ 175k tokens per task at $2.55/M (2 passes ≈ 85-90k each) |
| **Darshan** | — | Token reduction: Swift-Qwen3.8-27b / ThinkingCap (≈ −50 % thinking; licence caveat in Notes/02) |

## B. Platform facts from the same threads (updated 10-23)

- **Storage (751344, Dustin 10-23 01:35):** "instability with the dataset cache the past few days in some regions … finally in a stable state now". Reruns of affected submissions are being looked at.
- **Dustin 10-23 02:11:** "**put Models in Kaggle Models instead of datasets**" so they get the right cache (NFS). Our weights are Kaggle Models (NFS, confirmed in the diags). The **SGLang wheelhouse is a dataset (GCSFUSE)**, and its venv install took 181 s in SLIM run1.
- **Kaggle Error** still hits submissions on 10-22 and 10-23 (747515), sometimes after 4-13 h. The pending limit is 1 per team, and the deadline has not been extended.
- **Submissions** run on a larger, separate pool. Save/commit runs use a small fast pool.
- **CPU throttling (xz)** can cost 5-10 % throughput, so CPU work should overlap with the GPU.
- **Dominic (732723):** "Don't install wheels under `/kaggle/working`": a finished run was marked ERROR at teardown. Ours installs the venv in `/tmp`, so we are OK.

## C. What this means for us

1. **Pure Qwen3.8-27B + harness reaches ~66-70 on the LB** on the same 4×L4 in 12 h, with no NVARC and no distillation (mikelou1; NguyenThanhNhan 65.97; shiiin9 65.56; amn 72.50). This matches the user's own observation. Our 34.58 is a harness / throughput / engineering gap, not a model limit.
2. **Engine bugs move the score by 20 points** (mikelou1's "tiny kernel issue", 40 vs 60). Correctness of the serving stack matters as much as speed.
3. **Throughput targets:**
   - 600-800 tok/s aggregate for about 40 % with 1 attempt (TheDivine);
   - ours is W1 decode 355-460 and Duck effective 279.
4. **Speculation config used by a 65.97 team:** MTP / EAGLE with **4** draft tokens, chunked prefill 2048, mamba ssm fp32, `extra_buffer`.
   - We run DFlash2 with **8** draft tokens, chunked 4096, ssm fp16, `extra_buffer_lazy`.
   - R2's width-4 arm is the closest test. An MTP-4 arm is not in R2.
5. **Token efficiency per task:** 35k per slot × 2 (Kris, 45 % CV); 49k cut + 1 pass (NguyenThanhNhan, 65.97 later with a different notebook). "Python REPL does the thinking" (Adarsh), which is the Duck direction.
6. **Storage:**
   - our weights are already Kaggle Models (NFS);
   - move the wheelhouse to a Kaggle Model too, an easy ~minutes gain and lower risk;
   - keep the staging copy.
