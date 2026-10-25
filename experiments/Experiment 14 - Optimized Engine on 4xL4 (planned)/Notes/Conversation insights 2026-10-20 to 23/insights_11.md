# Insights from part_11 (2026-10-23 05:50 - 06:30)

Source: `part_11.txt`, 939 lines, all read. Structure of the part:
- Lines 3-482: an IDE selection the user attached at **05:50**. It is the full Bengali report `reports/৪xL4 ইঞ্জিন বটলনেক যাচাই.md` (engine bottleneck validation, 477 lines). It is attached context, not words the user typed, but it holds most of the numbers in this part.
- Line 485: the user's instruction at 05:50.
- 05:51-06:25: the assistant builds E14 (builder, measurement module, fake server, harness). One subagent hands back (the #44401/#41872 backport).
- 06:30: the user asks for this 3-day insight review. The part ends while the assistant extracts the transcripts.

Times are the transcript header times. They look like UTC, because file mtimes in the same part show 12:09 local when the header says 06:10.

---

## 1. User instructions, rules, preferences and decisions

| Time | Verbatim (Bengali) | English meaning / note |
|---|---|---|
| 2026-10-23 05:50 | "একটাই নোটবুক বানাও।আর সফল হলে তোমাকে আর একটা ফিক্স করে ১২ ঘন্টার ২ থেকে ৩ টা রান দেওয়াবো ক্যাগলে" | "Build just ONE notebook. If it succeeds, I'll have you make one more fix and then give 2-3 twelve-hour runs on Kaggle." This is a **decision**: one notebook, not separate A1/A2 notebooks. It also states the **plan after E14-A**: one more fix, then 2-3 full 12 h runs. The assistant read it as "A1 and A2 together" (05:51). |
| 2026-10-23 06:30 | "এর মধ্যে আমাদের গত ৩ দিনের সমস্ত কথোপথন পড়ে আসো,কোনো ইনসাইট পাও কি না? নোটবুকে কোনো ফিক্স বসাতে হবে কি না? আর গুরুত্বপূর্ণ সবকিছুই নোট করবে যাতে পরে দেখতে পারো।" | "Meanwhile, read ALL our conversations from the last 3 days. Do you find any insight? Does any fix need to go into the notebook? Note everything important so you can look at it later." A **standing request** to persist important findings, consistent with the "read everything fully" memory. |
| 2026-10-23 06:25 (assistant, quoting earlier user permission) | "আপনি pipeline বদলানোর অনুমতি দিয়েছেন, তাই এটাকে নোটবুকে একটা **মাপা arm ("nodelay")** হিসেবে যোগ করেছি" | The assistant says "you gave permission to change the pipeline", so it added nodelay as a measured arm. The permission itself is **not in this part**; it must come from an earlier part. Treat it as claimed, not verified here. |

Standing rules restated inside the attached report (05:50 selection, section ঘ):
- "সব GPU রানের জন্য ব্যবহারকারীর স্পষ্ট অনুমতি লাগবে … সাথে লিখিত wall time, quota আর stop point, এবং রানের আগে quota পরীক্ষা।" Every GPU run needs the user's explicit permission, plus a written wall time, quota and stop point, and a quota check before the run.
- "Rev 6g 'শেষ version' নিয়মের কারণে Solver বা Duck code বদলাতে ব্যবহারকারীর অনুমোদন লাগবে (Duck prompt re-render fix, sticky routing, branch scheduling)। Engine flag আর load cell সেই নিয়মের বাইরে, কিন্তু production notebook বদলাতেও অনুমতি লাগে।" Because of the rev 6g "last version" rule, any Solver or Duck code change needs approval. Engine flags and the load cell fall outside that rule, but changing a production notebook still needs permission.
- "60k floor-এর নিয়ম অপরিবর্তিত থাকে" (token-variant section). The 60k thinking floor stays.
- "নতুন notebook (কখনো rebuild নয়)" (E14-A row). E14-A is a new notebook and is never rebuilt.
- "প্রতিটি submission-এ একাধিক প্রমাণভিত্তিক পরিবর্তন একসাথে যাবে, একটি করে ablation নয়". Bundle changes per submission; no one-change ablations.
- Assistant 05:51: "all on CPU, with no push and no run". The no-push rule was respected throughout this part.

---

## 2. Measured facts and results

### Engine step time and throughput (report, 05:50; status [M]/[D] as the report marks them)
- W1 = SGLang 0.5.21, TP4, RedHat INT4 + BF16 DFlash2 width 8, fp8 KV, fp16 SSM. Decode step at 12 running: **about 115 ms (≤10k context) to 133-140 ms (50-80k)**. Per bin at ≤10k: **E11-03 D 115.5, E11-06 115.3, E12 EVAL30 115.4 ms**. E11-06 at 50-60k: **133.3**. E12 at 60-80k: **140.2**.
- Median step at 12 running: E11-06 **124.6 ms** (p10-p90 116.2-132.5, n=278). E11-05 EVAL30 **128.4 ms** (n=114). E12 EVAL30 135.7-139.4 ms, depending on the filter.
- Windows that contain a prefill line are much slower: E11-06 B=12 median **323 ms**.
- Identity: `accept len` includes the bonus token; accept len = 1 + 7 × accept rate. Maximum deviation over 5,017 records is **0.0057**. The report calls this "এখন তা মীমাংসিত: ধরা হয়" (now settled: the bonus is included).
- Decode-line tok/s, E11-06 at 12 running: <10k **500.1**, 10-20k 460.3, 20-40k 409.0, 40-60k 389.4.
- Whole-run tok/s: **E11-05 328.9** (4,253,591 tok / 12,932.4 s). **E11-06 379.9** (2,262,599 / 5,956.2 s).
- 11 vs 12 running: E11-05 had 2,358 decode lines at "11 running + 1 queued" and only 114 at 12. At the same context, 11 vs 12 running is about **-8-9 %** (EVAL30 422 vs 462; G4B 397 vs 436).
- Mean acceptance per run: E11-05 **3.87**, E11-06 **4.58**, E12 EVAL30 **5.10**. At 12 running in E11-06: 4.37 (8 running: 4.46). E11-05 at 12 running: 3.91. E11-05 decode-line median: 3.8.
- Collectives per step (from source): **140 all-reduces of 983,040 B each** (129 in verify + 11 in draft), plus 1 logits all-gather of **47.7 MB** ([96 × 248,320] BF16) and 2 small all-gathers (5,376 B and 10,752 B).
- Custom all-reduce is disabled on all 4 ranks in every run: "CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs". No `NCCL INFO` line appears in any log, so the transport (P2P or SHM) is unknown.
- Preflight 64 MiB fp32 eager all-reduce: **20.89 ms (E11-03)** and **20.76-20.83 ms (E11-04)**, i.e. algbw **3.21 GB/s**, busbw **4.82 GB/s**. This sits in the published SHM range.
- Estimated all-reduce cost [E]: about 340-380 µs per call, so 140 calls ≈ **48-53 ms**. Loose range 35-55 ms (30-45 % of a step). This is **unmeasured**.
- Arithmetic step budget [E]: **87-131 ms (centre about 108)**. About 7-25 ms remains unexplained.
- Marginal cost from B=1 to B=12 (E11-03 D): **39.9 ms** at 1 running, **115.5 ms** at 12. That is +75.6 ms, or about 6.87 ms per request and 0.86 ms per verified token. Target-only at 1 running: 21.9 ms.
- torch BF16 4096³ GEMM: **75.9 TFLOPS (E11-03)** and **78.8 TFLOPS (E11-04)**.
- Correction to earlier notes: the GDN large projections are **INT4, not BF16** (config.json ignore list: 304 entries, 192 of them GDN, only `linear_attn`, `norm`, `in_proj_a`, `in_proj_b`). Per-GPU target is 4.62 GB.
- CUDA graphs in E11-06: the resolved decode list was `[1..8, 10, 12]`, max_bs 12, while `max_running_requests=13`. Graph memory: target-verify 0.68 GB, draft 0.35 GB.

### Duck effective throughput (report table, [M/D])
| Run | Streams | Window s | Generated | Effective tok/s | Prefill % of window | Decode-only | Uncached prompt ÷ generated |
|---|---|---|---|---|---|---|---|
| EVAL30 | 12 | 14,046 | 2,357,584 | **167.8** | **55.1 %** | 376 | 4.07 |
| EVAL6 | 6 | 3,744 | 1,217,904 | 325.3 | 13.3 % | 377 | 0.64 |
| EVAL6R5 | 6 | 3,464 | 846,080 | 244.2 | 18.6 % | 302 | 1.15 |
| EVAL12R5_LOW | 6 | 6,570 | 1,743,889 | 265.4 | 20.4 % | 335 | 1.17 |
| G4B rev 6d | 12 | 2,050 | 572,035 | **279.0** | 19.8 % | 353 | 1.17 |
- Chunked prefill burst speed: **1,474-1,578 tok/s** across runs (EVAL30 1,474; G4B 1,578). One uncached prompt token costs about 0.27 of a generated token.
- An earlier figure of 187 tok/s for EVAL30 is the same run under a different window definition. The new window includes 679 s after the deadline.
- G4B spent **27.7 %** of its time at exactly 6 running, because the rev 6d probe stepped down 12 → 10 → 8 → 6 while KV was only 0.13-0.31. "11 running + 1 queued" covered 50 % of EVAL30 and 25 % of G4B.
- Reply re-prefill: the cache hit stops at the end of the previous prompt in **72-91 %** of requests (EVAL6 84.5 %, EVAL6R5 91.3 %, LOW 72.5 %, G4B 75.6 %, EVAL30 59.1 %). It stops on 64-multiples (G4B 141 of 150, LOW 413 of 414). Median reply-reuse fraction is -0.003 to -0.009.
- G4B uncached work was 670k tokens: about 37 % re-prefilled replies (245k), 47 % zero/partial hits (317k), 5 % first requests (34k), 11 % tool-result deltas (74k).
- At EVAL30 12 running, mamba usage was 36/56. At E11-06 11 running, 33/56. G4B KV max was 0.44.
- Hidden-scale requirement: 240 × 60k = 14.4M tokens in about 11.46 h needs about **370 tok/s effective**.

### Load / startup / platform
- `/kaggle/input` is partly on `FUSE.GCSFUSE` since 2026-10-21. amn's probe: 0.95 GiB/s (10-20) → **0.026 GiB/s** (10-21 05:37). On 10-22: 1 thread about 10.5 MB/s, 16 threads about 107 MB/s.
- Our INT4 load took 12.8-58.7 s on 10-19/20, but **669.26 s** in G4B on 10-21 (the 18.6 GB shard took 657 s, about 28 MB/s). "ready in" was **1,197 s**. The DFlash2 file loaded in 0.56 s (page cache).
- Current prefetch is a single-stream `cat`. At 10.5 MB/s it needs about 37 min, which exceeds the 30-min first-launch timeout. A timeout leads to E11Fail, then the fallback submission, then LB about 0. NguyenThanhNhan's two 0.00 scores came this way.
- Cold start, server_args to "fired up": E11-05 **8:00**, E11-06 **7:52**, E12 **8:05**, G4B **19:09**. Target-verify graph capture takes 371-385 s, almost all on the first shape: `bs=12 … 1/10 [06:02<…, 362.28s/it]`. With a warm cache in the same session it takes 11.94-13.67 s.
- **G4B rev 6g ERROR (esfersami50), 10-22, about 17:16 UTC:** it ran about 114 min, `kernel_log.txt` was exactly `[]`, and there were no outputs. Quota went from 26.87 to **30.21/30 h** (3.34 quota-h, about 100 min container time at 2×). About 3.13 h (about 94 min) remained before the run. The cause is unresolved; ranked: mount/attach stall > old image > quota.
- 4×L4 is billed at **2×** (measured burn 2.0×). The 10-21 start of 1,197 s cost about 40 quota-min before the first token.
- Quota balances at report time: **foysal 4.71 h, hosen42 8.63 h, esfersami50 exhausted (30.21/30)**. Deadline **2026-11-02**. Submissions about 1 per 2 days, 12-16 h turnaround.

### Build progress in this part (05:51-06:25)
- E12 builder `build_e12_duck.py` has 1,481 lines. The `_REBUILD` redirect for run variants (EVAL12R6G4B etc.) is confirmed at L42-47.
- Created: `src/e14_measure.py`, `Experiment Notebooks/Build/build_e14.py`, `src/harness/e14_fake_server.py`, `src/harness/harness_e14_nb.py`. `src/e14_loadpath.py` (41,095 B) already existed, mtime 09:44 local.
- Agent files at 06:10: `e14_nccl_bench.py` 54,269 B and `e14_sglang_patches.py` 20,186 B, both still in progress then.
- Builder windows (06:06): short and long steady windows of **75 s** each at `level=SGL_MAX_RUNNING`. Profile window **60 s**. Latch test `decodes=SGL_MAX_RUNNING - 1, arrivals=3, secs=60`. Phase cap "A1-2_baseline_W1" is `15 * 6…` (cut in the transcript). Time scaling: `E14_TS = float(os.environ.get("E14_TIME_SCALE", "1"))`, `_ts = lambda s: max(1.5, float(s) * E14_TS)`.
- Backport agent (06:23): 6 hunks (#41872: 2, #44401: 4). **36/36 tests passed on 6 runs in a row** (runs 7-12). Run 1 failed 1 of 30 because of a bug in the test itself.

---

## 3. Failures, mistakes and their root causes

1. **The 11-of-12 cause was misattributed (reversed 06:23-06:25).** The report (05:50) said the cause was "প্রায় মীমাংসিত" (nearly settled) as the #44401 double count, and rated E14-S (#44401 + graphs) at +3-8 %. The backport agent found that 0.5.21 has a **DFlash-only min-free-slots delayer** (`scheduler.py` L1183-1193, `min_free_slots_delayer.py` L17-22). With `--min-free-slots-delay` unset, its threshold is `min(4, max(2, (mrr+5)//6))`, which is 2 at mrr 12 and **3 at mrr 13**. When one request finishes at mrr 13, 11 are running and only 2 slots are free, under the threshold of 3, so the 12th waits for a second finish. That is exactly "11 running + 1 queued". A test that ran the real delayer code from both the original and patched file got the same result: delayed at mrr 12 and 13 with 11 running, not delayed at mrr 14 or 15. **The backport alone probably does not fix the symptom.**
2. **Builder/API mismatch (06:11).** `build_e14.py` assumed the patch report key `"patches"`, but the module returns `{"results": {name: {"status": ...}}}`. The assistant found this by reading the API. The fake-server harness would not have caught it.
3. **Agent's test bug (06:23).** A one-character mutation hit the keyword `or` and made the file invalid Python. The patcher correctly refused it; the test was wrong.
4. **Graphs arm and rollback (06:25).** "graphs arm আর তার rollback আগে গ্রহণ করা settings হারিয়ে ফেলছিল; সেটাও ঠিক করেছি". The graphs arm and its rollback were losing previously adopted settings. Fixed.
5. **Errors in earlier notes, corrected by the report (05:50):**
   - "PCIe Gen4 64 GB/s": that is bidirectional; one direction is about 31.5-32.
   - "4.3 of 8 accepted, so 46 % wasted": the bonus token is included, so the draft acceptance is about 3.3/7 ≈ 47 %.
   - "10-16 TFLOPS, so not compute-bound": not established.
   - "128 all-reduces": the source count is 140.
   - "GDN is BF16": false; it is INT4.
   - "Width 4 can give more than 4 tokens": false.
   - Note 03's "115.5 ms is a context-normalised regression": false; it is a raw median.
   - The TP2 KV figure of 0.35-0.4M per replica: wrong; it is about 170k.
   - ReplaySSM + DFlash being allowed (PSO): wrong; the configurator raises an error for non-KDA models.
   - The PARITY notebook sha was stale: the current rev 6g PARITYR6 is `82b257a5…` and was never run.
6. **G4B ERROR 10-22.** Root cause unknown. The log was `[]`, so the runner most likely never started the notebook. Our code and the 30-min timeout are ruled out.
7. **Load-path failure mode (report).** A single-stream `cat` prefetch plus a fixed 30-min timeout leads to the E11Fail fallback and LB ≈ 0 on slow GCSFUSE days.
8. **Cut tool results that matter (not guessed):**
   - L880: the assistant's own source check of the delayer is cut at `[+2214]`. The claim "১ বা তার কম দিলে delayer বন্ধ হয়ে যায়" (≤1 disables it) **cannot be confirmed from this transcript**.
   - L704: the phase cap value of "A1-2_baseline_W1" is cut.
   - L596-L631: the sglext / token-id source reads are cut.
   - L506: the builder docstring is cut.

---

## 4. Decisions made and their reasons; options rejected

**Decisions:**
- **One notebook** (user, 05:50). A1 measurement and A2 arms go in a single E14-A. The cells are:
  - CELL 1b: machine facts, phase record, heartbeat, model copy, raw reasoning on.
  - CELL 5: JIT cache dir, stall watchdog, first launch extended up to the give-up time.
  - CELL 5b: A1 measurements.
  - CELL 6: 25-min Duck run on the adopted arm.
  - CELL 6b: A2 arms.
  - CELL 7b: report, JIT cache, bundle.
- `build_e14.py` runs the E12 rev 6g builder in memory and writes nothing to E12's folder, so the base stays intact (06:11).
- **nodelay becomes a measured arm** (06:25). Arm order: **W1 → +#44401 → +nodelay → +graphs**. It is kept for the Duck run and later arms only if it passes the latch test and the speed rule.
- **Backport kept** as an all-or-nothing patch with hash guards. If the Kaggle x86_64 wheel text differs from the aarch64 reference, the patch does not apply and the run continues unpatched, with the reason logged.
- The agent applied **neither** delayer fix (`--min-free-slots-delay 1`, or mrr 14 with the client at 12), because both change server flags and need a go. The assistant then added nodelay as an arm under the claimed pipeline permission.
- The delayer finding was saved to memory (`experiment-14-optimized-engine.md`, 06:24).
- Report priorities:
  - P0: load path; offline token-diff.
  - P1: E14-A measurement; #44401 + graphs 1..13; correctness gate.
  - P2: width 6/4; NCCL env.
  - P3: 0.5.22, PCIe-IPC, logits vocab-parallel, GDN kernel sweep, Duck re-render fix, Mamba protection.
  - P4: real Duck confirmation.
  - P5: 2×TP2.

**Rejected or deprioritised, with reasons (report):**
- **2×TP2: almost dropped.** It holds only about 170k KV per replica (about 340k system-wide) against the 700-920k needed. Expected gain 0.7-1.0×.
- **HiCache: low priority, high risk.** KV max was 0.44, so it would gain about 0. Hybrid host-tier bugs are open: #39830 (0/10 vs 10/10 answers), #45400.
- **PCIe-IPC all-reduce:** dead code in our install, since FlashInfer 0.6.18 lacks `pcie_ipc_ar.py`. It needs FlashInfer ≥ 0.7 (SGLang 0.5.22).
- **ReplaySSM with DFlash:** blocked in code ("requires a KDA model").
- **Do not use:** `--enable-mixed-chunk` (#43514), `num_continuous_decode_steps` (a no-op in 0.5.21), torch.compile, forced custom AR (3090 TP4 +14.8 % but greedy divergence on 4 of 5 prompts), `NCCL_SHM_DISABLE=1`.
- **Token-efficient variants (Swift, ThinkingCap):** the licences are not OSI-approved, so prize eligibility is doubtful. The expected saving is only -10-25 % on long traces.
- **Lowering effort:** rejected. ARC Prize scores medium 13.2 % < low 22.8 % < xhigh 42.4 %, and our 49k vs 60k LB was 32.08 vs 34.58.
- **`--dtype float16`:** not worth a run, because the Marlin atomic-add path is off on sm<9.
- **Stock loader flags:** `--weight-loader-prefetch-checkpoints` splits files per rank, so rank 0 would read 18.6 GB alone. `overlap` load mode rejects spec, TP4, quantization and draft. `fastsafetensors` is not in the wheelhouse.
- **SGLang 0.5.22:** does **not** contain #44401. Only nightly ≥ 2026-10-16 has it, and nightly has a manylinux_2_39 tag that pip rejects on glibc 2.35.

---

## 5. Open items

- **Delayer fix verification on GPU.** Is real running == 12 with queue ≥ 1 after `--min-free-slots-delay 1`? Are there new scheduler errors ("alloc_req_slots runs out of memory", "Not enough space for mamba cache")? Is there any over-admission? The alternative (mrr 14, client 12) needs a memory check first, because mrr also sizes the Mamba state.
- **Backport on Kaggle.** Does the x86_64 wheel match byte for byte? Log `check` before and after, and run `apply` before the server starts. The PREFILL-mode and dLLM hunks are unexercised.
- **NCCL bench + trace summariser agent:** not finished in this part.
- **Pending at end of part:** build the notebook, run the full harness on the fake server, then give the user the written time, quota and stop rules plus the notebook sha (promised at 06:11). Not done in this part.
- **Unknowns the report lists:**
  - the true split of the step (no profiler trace yet);
  - whether P2P / CUDA IPC works on Kaggle 4×L4, and which NCCL transport is chosen;
  - µs per call at 983,040 B;
  - the logits all-gather time (3.6-15 ms);
  - L4 DRAM bandwidth;
  - acceptance at width 4/6 and per-position survival;
  - DFlash2 vs target-only error rate on our stack (#36548);
  - the cause of reply re-prefill (template trim or re-tokenisation);
  - the cause of zero hits at low KV (Mamba LRU, trim, or #22935);
  - `/dev/shm` and local disk sizes;
  - which `/kaggle/input` subtree is on GCSFUSE;
  - whether a shipped JIT cache is valid across sessions or hidden-pool machines;
  - the stage at which the 10-22 ERROR happened;
  - whether the FP8 KV scale is actually loaded (it "defaults to 1.0" in the log);
  - `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` (untested);
  - DFlash2 vs NEXTN MTP at 12 real running (never measured).
- **Unresolved conflict:** note 05 (old pinned image) vs SLE (attach/mount stall) for the G4B ERROR.
- **Licence question** to the host (topics 746307, 750575): unanswered.
- **User's future plan:** "one more fix" after E14-A succeeds, then 2-3 twelve-hour Kaggle runs. Which fix is not specified.

---

## 6. Insights a fresh reader might miss

1. **The biggest reversal: 11-of-12 is the DFlash min-free-slots delayer, not #44401.** At our mrr 13 the #44401 double count only matters if 13 requests are in the system, which 12 client streams cannot produce. So an arm of "+#44401" alone is expected to show **no** gain. A null result there is not evidence against the 11-of-12 diagnosis. The report's E14-S gain (+3-8 %) now belongs to nodelay.
2. **The delayer can also block a baseline window, not only steady state.** With 11 running and the 12th request arriving separately (for example staggered thread starts), the 12th waits. Logs still show 12 running in some lines, likely when requests were admitted together.
3. **The CPU fake-server harness cannot catch real-server API mismatches.** One already happened (the patch key `"patches"` vs `"results"`). The same risk applies to the token-id request field (see 7.4).
4. **Effective rate and decode rate multiply.** Prefill is so slow on 4×L4 (about 1,500 tok/s vs about 400 decode) that prefix reuse is worth 3-4× more than on an RTX PRO 6000. Doubling decode without fixing re-prefill leaves the effective rate capped. Fixing re-prefill without engine work caps at about 400-460.
5. **Raw reasoning may not be enough to stop reply re-prefill.** The chat template rebuilds the assistant turn as `'<|im_start|>assistant\n<think>\n' + reasoning_content|trim + '\n</think>\n\n' + content`, and rebuilds tool calls from parsed arguments into XML. Even with raw reasoning kept, `|trim` and the XML re-render can still diverge from the generated tokens. A full-reply hit is not 64-aligned, so the decode-end checkpoint works when the history matches exactly. The offline token-diff is the proof step.
6. **The 64 MiB all-reduce number is a fair proxy for our 0.98 MB calls.** At that size SHM is bandwidth-bound. Only around 10 KB is the call latency-bound. This corrects note 05 A4's own correction.
7. **Stall rule for phase 2:** the first bs=12 graph capture is about 6 min silent (362 s/it), so any no-new-log-line rule must allow more than 8 min (the report suggests 12 min). For phase 1 the report says to use `rchar`, because `read_bytes` may not count FUSE reads and tqdm is silent during mmap faults.
8. **`SGLANG_JIT_CACHE_DIR` is separate from `SGLANG_CACHE_DIR`** (default `~/.cache/sglang/jit`). The stale `nv/` driver cache caused a wrong-output bug in #42917 (PPL 9.98 vs 6.28 after deleting `nv/` only).
9. **Quota coincidence.** The G4B run that ERRORed had about 94 min of quota left and burned about 100 min. The memory note "runs started with low quota still finish" is weakened by this, although E9 is a counter-example (it ran 8,879 s from 28.44/30).
10. **Width change is not "cut the last 4 drafts".** The drafter attention is bidirectional (`is_causal: false`). Max accept len equals the width. Each new width or graph shape may cost about 6 min of recapture.
11. **NCCL P2P_LEVEL needs safety checks.** It can hang silently under IOMMU/ACS even when `topo -p2p` says OK. It needs a correctness all-reduce (sum of rank+1 = 10) and a timeout. `NCCL_SHM_USE_CUDA_MEMCPY` helped on one box and halved 1 MB speed on another, so only an A/B at exactly 983,040 B inside a CUDA graph decides it. Benchmarks must use torch's own NCCL (2.29.7).
12. **Time zones.** Transcript headers are UTC (06:10) while file mtimes are local (12:09).

---

## 7. Relevance to the current E14-A notebook (concrete checks and fixes)

1. **Quota vs the 200-min E14A budget (check, high priority).** 4×L4 bills 2×. A 200-min wall budget is about **6.7 quota-h**. Available: foysal **4.71 h**, hosen42 **8.63 h**, esfersami50 **0**. Only hosen42 can hold a full run, and the margin is about 1.9 h. The 10-22 G4B ERROR happened with remaining quota (about 94 min) just under the burn (about 100 min). The written pre-run estimate must therefore state: account = hosen42, worst case 2× wall, and the expected stop point if quota runs out. The report's own estimate (Tiers A-C, before the Duck run, replay and parity were added) was 60-70 min wall, 2.0-2.4 quota-h, worst case about 3 h. The 200-min profile is far above that, so per-phase caps matter.
2. **Interpret the arm ladder with the delayer in mind (fix the acceptance logic if needed).**
   - W1 and W1+#44401 should both **fail** the latch test (11 running, 2 free < threshold 3 at mrr 13).
   - Only +nodelay should pass.
   - The rule must not reject #44401 for "no gain" and then skip later arms, and must not treat a W1 baseline that cannot reach 12 real running as a fatal error.
   - Make sure `wait_running(level=12)` in the W1/#44401 baseline windows does not time out because a staggered 12th request is held by the delayer. Submit the window's requests together, or record the failure as an expected delayer signature and continue.
   - Also confirm that `level=SGL_MAX_RUNNING` means 12 (the client count), not the server's mrr 13.
3. **Verify the nodelay flag really disables the delayer.** The assistant's source check (L880) is cut in the transcript, so "≤1 disables it" is not confirmed here. In the nodelay arm:
   - log the resolved `min_free_slots_delay` / that the delayer is None, if the server prints it;
   - assert real running == 12 with queue ≥ 1 during chunked prefill;
   - grep the server log for "alloc_req_slots runs out of memory", "Not enough space for mamba cache" and preemption.
4. **Token-id field name for the token-level replay (check).** The assistant's summary says the replay uses `return_token_ids`. The source it read in this part shows `return_input_ids` / `return_output_ids` and `_should_return_input_ids(request)`, and token ids are **omitted after an error abort**. Confirm against the 0.5.21 source:
   - the exact request field the notebook sends;
   - the response path (`sglext` input_ids / output_ids, or `prompt_token_ids`);
   - that the replay handles `None` ids on aborted requests.
   The fake server will accept any name, so the CPU harness cannot catch a mismatch (same class of bug as the `"patches"`/`"results"` miss).
5. **Raw reasoning is necessary but may not be sufficient.** The template still applies `reasoning_content|trim` and re-renders tool-call XML from parsed arguments. The 25-min Duck run should report:
   - the `prefill_end` hit share;
   - the reply-reuse fraction;
   - the 64-aligned hit share.
   Then a remaining mismatch is visible and is not taken as proof that raw reasoning fixed it. Ideally the token-level replay records the first divergent token index.
6. **Backport guard.** Run `check` before `apply` and after `apply`, and log the results, all **before** `launch_server`. If the status is "mismatch" (aarch64 reference vs x86_64 Kaggle wheel), continue unpatched and record it. Also drop stale `.pyc` files.
7. **Stall watchdog signals.** The context says "log growth + read bytes + page faults". The report recommends `rchar` from `/proc/<pid>/io` over the whole process tree, because `read_bytes` may not count FUSE reads. If the code uses `read_bytes`, add `rchar` (page faults partly cover the mmap case). Phase 2 (load end → Uvicorn) must allow more than 8 min of silence, because the first bs=12 capture is about 362 s. Every relaunch arm (graphs 1..13, width 6/4, NCCL env) may pay this again. Keep the 75-min give-up and never kill a server that is still progressing.
8. **Model copy to /dev/shm.** The report's design:
   - df `/dev/shm` first and require ≥ 24 GB free (Docker default is 64 MB);
   - fallback chain `/dev/shm` → `/tmp` or `/kaggle/tmp` → range warmer;
   - `posix_fallocate`, 64 MiB ranges, largest file first, 16-32 workers;
   - verify size + safetensors header + last tensor offset = size − 8 − header length;
   - **never launch before the copy is verified**, or SGLang mmaps a partial file;
   - fail open to `/kaggle/input`;
   - RAM peak about 47 GB of 202.6 GB.
9. **JIT cache.**
   - Set `SGLANG_JIT_CACHE_DIR` explicitly (it is not under `SGLANG_CACHE_DIR`).
   - Exclude `nv/` from the tar (#42917).
   - Name the tar with the sglang/torch/triton/flashinfer/sm89 versions and include `pip freeze`.
   - After untar, run a known-answer greedy gate (64 tokens + first-16 logprobs); on mismatch, delete the cache and relaunch cold.
10. **First-line instrumentation** (to tell a platform stall from our code, per the 10-22 ERROR):
    - Before touching `/kaggle/input`, write and flush time, hostname, `uname -r` and image digest.
    - Record `/proc/mounts`, `df`, `free`, and a timed 64 MiB read per input.
    - Read the competition JSON in a thread with a timeout.
    - Write the heartbeat file in `/kaggle/working` early.
    Cell 0 still writes the fallback `submission.json` within seconds.
11. **Profiler.**
    - One profiler window per process (CUPTI).
    - Take published tok/s only from non-profiled windows.
    - Parse the trace in a subprocess with a timeout.
    - Take the per-collective min across ranks.
    - Log `nvidia-smi dmon` clock/power (72 W throttle risk at M ≈ 96).
12. **Greedy parity.** Use T=0, an A/A reference across a restart, and the top-1/top-2 margin at the first divergence (> 1 nat = bug). Token equality at 60k is not expected. `--enable-deterministic-inference` with DFLASH + mamba is unverified, so do not depend on it.
13. **NCCL env arm.** Set `NCCL_P2P_LEVEL=SYS/PHB` only if the P0 topology shows P2P OK, after a correctness all-reduce (sum = 10) and with a hang timeout. Use `NCCL_DEBUG=INFO` to learn the transport. Test `SHM_USE_CUDA_MEMCPY` only as an A/B at 983,040 B. Do not set `NCCL_SHM_DISABLE`.
14. **CUDA graphs 1..13.** Expect three more verify graphs of memory (today 0.68 + 0.35 GB for 10 graphs) and extra capture time. Log the resolved graph list.
15. **Width 6/4 arms.**
    - The "DFLASH block size mismatch" warning is normal.
    - The acceptance gate must use max = width: W6 ≥ 3.2, W4 ≥ 2.5 (report).
    - Compare at the same context.
    - Assert real running == 12.
16. **Window lengths.** The builder uses 75 s steady windows; the report suggested 90 s at W8k/W40k. At about 40 decode forwards per log line (about 5 s), 75 s gives about 15 lines. That is fine, but check that each window gets enough 12-running lines after the delayer filter.
17. **Do not set** `--enable-mixed-chunk`, `num_continuous_decode_steps`, torch.compile, forced custom AR, or HiCache with int8 mamba checkpoints (incompatible) in any arm.
