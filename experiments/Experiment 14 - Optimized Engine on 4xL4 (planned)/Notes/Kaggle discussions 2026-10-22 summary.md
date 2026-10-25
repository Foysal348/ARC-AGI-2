# ARC-AGI-2 discussions, re-read in full (2026-10-22): Qwen3.8-27B results, the storage slowdown, and our exposure

**How this was read:**
- All 35 ARC-AGI-2 threads that a site-wide kagglesdk search finds (36 queries) were dumped in full: the first post plus every comment and reply. The dumps are `topic_*.md`; the list is in `index.json`, `index_extra.json` and `index_new.json`.
- Two linked threads were added by id: 742796 and 704408.
- Leaderboard history: `D:/ARC AGI 2/_scratch/lbmon/new_2026-10-22/arc2_history.json`. It shows each team's best score per day, so failed submissions do not appear in it.
- The 2026-10-21 summary is in `../Kaggle discussions (2026-10-21)/00_SUMMARY.md`. This file adds what changed since then.

## 1. The storage slowdown (new; it decides whether a submission scores at all)

**What changed:**
- From about 2026-10-21, part of `/kaggle/input` is mounted as `FUSE.GCSFUSE` instead of NFS (topic 751344).
- The NFS mounts are slower too. amn's probes, same notebook and files:

  | Probe (UTC) | Read speed |
  |---|---|
  | 10-19 10:54 | 0.55 GiB/s |
  | 10-20 03:26 | 0.95 GiB/s |
  | 10-21 03:21 | 0.32 GiB/s |
  | 10-21 04:35 | 0.040 GiB/s |
  | 10-21 05:37 | 0.026 GiB/s |

  On 10-22, a single thread read about 10.5 MB/s and 16 threads about 107 MB/s.
- **Jeroen Cottaar** (an ARC-AGI-3 run): 3 of 10 shards (100.5 GiB) loaded in 7 h 22 min on GCSFUSE, against 88 s on NFS. In that session, reading through the mount did not fill the page cache.
- **Effects reported:**
  - **0.00 scores:** NguyenThanhNhan (2 submissions), amn (2), and Jeroen's team ("usually").
  - **Scores 5–10 points lower:** Kha Vo, mp, Jeroen.
  - **A "Kaggle error" after 12 h:** CPMP, Kha Vo.
- **NguyenThanhNhan's 0.00 runs:** his notebook reads about 180 GB and waits at most 45 min for the server. Normally the server starts in 15 min or less. On GCSFUSE it never started, so the placeholder JSONs were submitted.
- **mp's mitigation (still on a save run, no result yet):** if `/kaggle/input` is FUSE, copy the files to local disk once at startup instead of letting 4 GPU processes read the mount.
- **Older signs of infrastructure trouble:**
  - 738313: CPMP's 10 h notebook timed out once, then ran in 9 h 10 min on resubmission.
  - 704408: a 0.00 after 15 min; the same code scored normally on resubmission.
  - 689054: L4 queue jams; L4×4 time counts double against quota.
- **The Kaggle image changed on 10-02** (CPMP lost a submission to it). We pin the old Python 3.12 image, so this one does not affect us.

### Our exposure (verified in our own logs and notebook)

| | Value |
|---|---|
| Weights read at launch | INT4 19.47 GB + DFlash2 3.85 GB = **23.3 GB** (others read 100–180 GB) |
| SGLang "ready in", 10-19/10-20 runs | 507, 522, 532, 552, 562 s (total engine `load_sec` 660–758 s) |
| SGLang "ready in", rev 6d G4B on 10-21 | **1,197 s**. Our analysis at the time called it "a slow cold start on Kaggle"; the threads above now explain it |
| First-launch timeout | `SGL_LAUNCH_TIMEOUT_SEC = 30 min`; relaunch 20 min; no launch after `SGL_GIVE_UP_AT = min(T0 + 75 min, admit deadline − 5 min)` |
| After a timeout | `e11_next()` finds no error line in a log that is still loading, so it returns None. Then `E11Fail`, `ENGINE_READY = False`, CELL 6 does not solve, and the fallback submission stays: **about 0 on the LB** |
| Prefetch | One background `cat file1 file2 file3 > /dev/null`, single-threaded. At about 10.5 MB/s single-threaded, 23.3 GB takes about 37 min, which is longer than the 30-min timeout |
| A slow but successful load | Only costs the lost time: the solve window and the rerun's per-task window are computed from the real start time |

**Reading:**
- The danger is the hard 30-min timeout, not the slowness.
- On 10-21 we used 20 of the 30 min.
- The live check is the G4B run now on esfersami50 (pushed about 17:16 UTC 10-22): read `E11_SGL.attempts[0].sec` in its report.
- **Options (not built; the user's decision under the last-version rule):**
  1. Raise the first-launch timeout to the existing 75-min give-up bound. This is one constant.
  2. Read the weights with parallel range reads into local disk or RAM and launch from that copy. This is more code.

## 2. Qwen3.8 results within Kaggle limits (all evidence found, newest first)

| Who | Result | Method evidence (thread) |
|---|---|---|
| **mikelou1** | LB 25 → 36 → 40 → 66.39 (10-17) → **69.72 (10-21)** | 748082: the 40 was caused by a kernel issue (CV projected 60 %). "Not a NVARC ensemble." 750546: "With time constraints at around 4xL4 speed I got Qwen to 87 %" and "79 % ± 5 % in CV runs"; he expects the 87 % not to transfer. **Whether this is 27B or Flash-Next was asked twice and not answered.** Closed source |
| **NguyenThanhNhan** | ~23 (10-05, pure reasoning) → 52.08 (10-18) → **65.97** (10-20, ~12 h run) | 745787: 27B FP8 + MTP/EAGLE 3/1/4, xhigh, one pass, forced commit after 49k ("a simple harness" would solve more), SGLang flags in that thread. 751344: the 65.97 notebook reads ~180 GB and uses an MTP draft. The model of the 65.97 run is not stated |
| Daniel Franzen (Duck author) | 57.64 since 10-19 | no ARC-AGI-2 write-up |
| mtg / artificialagencylab.com / Youssef Nader / guoqin gu | 62.08 / 65.56 / 70.56 / 67.50 | methods unknown |
| **Hiểu Vy** | 26.81 with the original 27B, 8.61 with Flash; best 44.03 = an NVARC finetune | 748082: "80 % on the 120 validation tasks" with 27B; whether that was within 12 h was asked, not answered |
| **Anil Thomas** | NVARC 35 → **45** with a Qwen3.8-27B ensemble | 750546: "a harness that includes a python REPL can help Qwen3.8-27B quite a bit. but squeezing it into a 12 hour window on 4x L4 is hard". 745787: xhigh median about 64k per generation; his estimate: 42 % × the coverage fraction |
| **Kris Dreemurr** | CV 45 % (~52/120) at xhigh with a median 70k tokens per task for pass@2 (35k per slot); LB 34.31 | 750546: cannot fit it into the time box. NVARC runs all 240 tasks in about 4 h, so he is considering an ensemble |
| **TheDivine Aurora** | ~19–20 % at 300–500 tok/s, attempting half the tasks | 750546: "many times attempts dont end up with an answer. but when done, theyre mostly right"; 128k–256k context; estimates 40 % at 600–800 tok/s with 1 attempt per task |
| Darren Amadeus Martin's team | CV above 75 % (120 tasks, 6 h) that does not carry over to the LB | 750546 |
| Bushi Liu | CV does not translate to the LB either | 750546 |
| Adarsh | agent: 15 → 21.53 (a rerun gave 13), "B98k C16" notebook | 748082, 749756, 750546: "27b on xhigh is just so slow that it never reaches most of the tasks"; Flash-Next on L4 has been disappointing for him |
| Gladstone025 / juliencst / russcore | 13.75 agent / about 8 / about 10 freeform | 742796, 745787. Anil's advice to russcore: `turn_max_tokens` 40,960 is too low; use 64k and a larger context |
| amn | thinks a quantized Flash-Next run as an agent might do better | 748082 |
| Souvik_D_Biswas | 4B TTT 28.06 plus a 27B hybrid; uses our FP8 repack | 749756 |

**Other facts:**
- **Official ARC Prize semi-private (750577):** Qwen3.8-27B (XHigh) = 42.36 %, $0.447 per task. Darren works this out to about 175k output tokens per task (two attempts), about 85–90k per attempt.
- **CPMP (745787):** "We got 70+ before Qwen 3.8, and this model reshuffled everything." nvbanana (79.72) uses their own weights (729509).
- **Fine-tuning NVARC on 342k synthetic examples did not move the LB (748895):** 30.83 → 27.9–29.2. Reruns of the same code vary by ±2–4.
- **Reruns of the same NVARC notebook vary by several points (745846, 743880, 685142).** The public LB data = the ARC Prize semi-private set (CPMP).

## 3. What it means for us

- Others scoring 65–70 inside 12 h on 4×L4 confirms that the model (27B, or Flash-Next for the undisclosed runs) can get there. Every high scorer's account stresses throughput plus reaching every task. Long thinking that never commits scores 0.
- The token arithmetic in the threads (Kris 70k per task → 45 % CV; Anil 42 % × coverage) matches our coverage-binds reading. Rev 6g's 48–64k per task over 240 of 240 tasks is in the same range as Kris's 70k (pass@2). It is a budget choice; whether it pays shows in the G4B correct count.
- **The urgent item is not accuracy: it is the launch timeout under the new storage.** A run that never starts the server scores 0.
