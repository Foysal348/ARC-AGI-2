# 23 — The last 4 days: work, mistakes, insights and the next path (Claude, 2026-10-23 ~17:40 UTC)

**User request:** "আমাদের ৩-৪ দিনের সমস্ত কাজ, ভুল, ইনসাইট আর কোন পথে নেক্সট যাবো সেটা বল" — all the work, mistakes and insights of the last 3-4 days, and which way to go next.

**Sources:**
- Notes/16 (conversation insights, 10-20 → 10-23 06:30);
- Notes/20 (GPT audits + build 2);
- Notes/21-22 (today's runs);
- the session record.

## 1. What was done (timeline)

| Date | Work | Result |
|---|---|---|
| 10-20 | E13 SMOKE1; **E12 rev 4 EVAL6**; weakness analysis; rev 5 (B1-B9, R5-1..5); EVAL6R5 + EVAL12R5_LOW | rev 4: **3/6 solved** (60k out, 100k cap, ≈10 GPU-min/task) |
| 10-21 | EVAL12R5_LOW submitted; rev 6 → 6b → 6c → 6d (regex hang, gate 4b, 12 streams, KV-only gate); **G4B rev 6d GPU run**; GPT audits | **LB 17.50** (Duck). rev 5 solved ≈56 % of the tasks it reached, but reached ≈31 % of 240. G4B rev 6d: 2/19 at ≈47.7k tokens/task, 277 tok/s |
| 10-22 | rev 6e/6f/**6g** (last version); **G4B rev 6g commit on esfersami50 → ERROR after 114 min**, empty log, quota exhausted; GCSFUSE slowdown found; E14 opened | Cause found today: notebook 444 kB, over the commit size limit |
| 10-22/23 | 2 research rounds (15 agents); step-time analysis (Notes/00); P0 token-diff; load path E14-0; **re-prefill root cause**; E14-P raw-reasoning switch; E14-A plan, "one notebook" | 2×TP2 estimated at 1.3-1.6× (the 3090's 2.4× does not transfer); found that Duck deletes blank lines from reasoning, so 72-91 % of requests lose the reply prefix |
| 10-23 morning | Build 1 (a9dad9fc); 3-day insights (Notes/16); delayer found (11 of 12); 3 GPT audits (NO-GO); validation (Notes/20); **build 2 (a5318dda)**, categories A + B | Unit 165/165, harness 31/31 (local) |
| 10-23 afternoon | E14A commits fail ×3 (INT4 mount error); 4 diag runs; T2 padded control | **Commit size limit found (407 kB ran, 444 kB+ never ran)** |
| 10-23 | **SLIM build** (224 kB, payload dataset `.bin`); SLIM run1 on foysal | Started and loaded; then A1-2 crashed (`Sampler._stop`, Python 3.12). Measurements in §3 |
| 10-23 | R2 fix; **Kaggle CPU harness kit** (user's idea); test portability fix | Kaggle CPU (Py 3.12.13): pytest 167/167, harness 32/32 ×2 |
| 10-23 17:30 | R2 on hosen42: attempt 1 mount error (no quota), **attempt 2 RUNNING since 17:33** | Ends ≈20:55 UTC, ≈6.8 h quota |

- **LB:** best S1-60K **34.58** (10-17). Last submission: 10-21 (17.50), so **3 days without a submission**.
- **Quota:** esfersami50 exhausted; foysal ≈2.25 h; hosen42 8.42 h, ≈1.6 h after R2.

## 2. My mistakes

1. **The biggest lever was dropped.** 2×TP2 (the source of "2-3×") was taken out of E14-A after the GPT audits said "TP2: not now". I followed that and went to the small flag levers (nodelay, graphs13, width, int8; together ≈10-25 %, estimate). This broke two rules: the user's goal beats audits, and the biggest lever comes first.
2. **Audit and fix loops for days** instead of a short GPU smoke. The user's new rule: a 30-min smoke, 1-2 GPT audits at most, submit daily.
3. **Platform limits never checked:**
   - the commit notebook size limit cost the 10-22 G4B run (≈3.8 h of esfersami50 quota) and 3 E14A attempts;
   - the Python version: the local harness ran on 3.14 while Kaggle runs 3.12, so `Thread._stop` crashed the run (≈0.8 h quota).
4. **The fake server proved my assumptions, not reality.** This repeated six or more times:
   - the capacity probe (12 → 6);
   - blank-line normalization;
   - the patch key;
   - `/start_profile` returning plain text;
   - fixture path names (the staging collision);
   - Python 3.14.
5. **Wrong estimates and wrong statements:**
   - the "~60 min" G4B estimate (it took 114 min);
   - "2-3×" and "10-25 %" given without saying they were different levers;
   - today:
     - "T2 is stuck in mount";
     - "444 kB ran";
     - "harness failures = CPU contention";
     - reading the user's interactive output as the commit's output.
6. **Researching instead of doing** (the cancel test). Over-long reports. Earlier omissions (the user said it 3 times). Short approvals expanded into choices. Kaggle-run notebooks overwritten twice (earlier).

## 3. Insights that stay true

**Platform (Kaggle):**
- **Notebook size:** commit runs only start below roughly 407-444 kB. Keep notebooks under 400 kB and put payloads in a private dataset, as sha-named `.bin` files so Kaggle does not unpack them. Each account needs its own copy of a private dataset.
- **Mount failures** show only on the web page. The API log is empty and no quota is used. A retry can work (hosen42 attempt 2).
- **Kaggle CPU kernel** with the same pinned image is a free and exact check: Python 3.12.13, the same libraries.
- **Billing:** 1.95× wall time, measured.
- **Model copy speed:** a single-stream read of `/kaggle/input` gives 34.6 MB/s; a parallel ranged copy gives 518 MB/s (it plateaus at 16 workers, 786 MB/s).
- **Cancel:** the cancel API exists (`cancel_kernel_session`; the session id comes from the search API). Not tested.

**Engine (TP4, INT4 + DFlash2, SGLang 0.5.21):**
- **Step time:** 115-133 ms at 12 running. **About 56 ms of it is GPU-to-GPU communication**: all-reduce 345 µs × 140 per step + all-gather 8 ms.
- **Topology** is PHB, and P2P is 5.5× slower. So PCIe-IPC / custom all-reduce is dead, and **2×TP2 is the main lever against communication**. Memory is the risk: ≈5-6 long requests per replica.
- **Delayer:** the DFlash min-free-slots delayer holds 11 of 12 (≈8-9 % lost). `--min-free-slots-delay 1` disables it.
- **First launch 502 s:** about 7 min is compile. **A relaunch takes 71 s**, so a compile cache would add ≈7 min more solving per run and reduce the timeout risk.
- **Determinism:** DFlash greedy output differs across restarts (#38009), so gates judge large-margin flips against the A/A.
  - A/A 10/12 identical (the divergences are ties);
  - batching gives 2/12 identical (normal).

**Duck / agent:**
- **Re-prefill:** Duck's normalization deletes blank lines from the reasoning, so the reply prefix is lost. Fix: raw reasoning (E14-P).
- **Prefill blocks decode:** the agent's effective rate is 167-325 tok/s against 370-500 tok/s of decode.
- **The 1× budget:** 240 tasks in 11.3 h ≈ 60k tokens/task. At 2.4× it is 144k, two full attempts; an LB near 50 if quality holds (estimate).
- **rev 6 stage 1:** 240 × 33k ≈ 7.9M tokens ≈ 5.9 h, so every task gets a first try. This is a projection; it is unproven at 240.

**Process:** biggest lever first; a short GPU smoke instead of audits; Kaggle CPU harness before the GPU; submit daily.

## 4. Next path (top-down)

1. **Now:** R2 on hosen42 (ends ≈20:55 UTC). It is the first GPU run of rev 6g Duck and measures:
   - nodelay, #44401, graphs13, int8, target-only, width 4/6;
   - the profiler step split;
   - the compile cache tar.
2. **Right after R2, the same night: a submission.** The R2 notebook in rerun mode, plus the arms that passed, as flags only. Validated on the Kaggle CPU harness, then submit. **A submission every day from then on.**
3. **Build today: the 2×TP2 measurement.**
   - Two TP2 servers, 6 + 6 streams, 8k / 40k windows, KV capacity of both, against TP4 W1.
   - ≈2-3 h build + Kaggle CPU check; GPU ≈30-40 min on foysal.
   - At ≥ 1.3×: a two-server Duck client, then the next submission.
4. **Compile cache restore** (with C8's known-answer check): ≈7 min more solving per run.
5. **Then, from the R2 + 2×TP2 numbers:** the DFlash width and target-only choice, and depth per task. Any token cut below 60k needs the user's approval.
