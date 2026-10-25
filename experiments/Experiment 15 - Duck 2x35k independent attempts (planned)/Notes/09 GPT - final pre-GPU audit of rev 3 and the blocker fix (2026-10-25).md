# 09 — GPT final pre-GPU audit of rev 3 (pasted by the user, 2026-10-25), verbatim, and what was done

## What was done (Claude, 2026-10-25)

- **The BLOCKER is confirmed in the code.**
  - In `solve_all_r6`, `sess()` ran after `running += 1` and `active.add()`. A session-setup exception escaped to the worker wrapper, which leaked the slot and lost the job.
  - Reproduced: the new unit test on the OLD package (the payload `19192e8006e0`, which runs on GPU now) FAILS. It shows worker_errors 4 and the run hanging until the deadline (116 s).
- **The fix (package `bc26fa16`):**
  - The session is created first.
  - A failure records a "session setup" record, advances `stage1_left`, and queues attempt 2 for that task. The failed (task, branch) goes into a `failed` set, which `stage2_items` never requeues.
  - `setup_failed` is reported in `SCHED_LAST`.
- **The test (`test_duck_b.py` section 15)** covers GPT's mandatory cases:
  - 16 workers, 20 tasks;
  - setup fails for 3 tasks' attempt 1 and one task's attempt 2.

  It checks: no slot leak (no worker error, no straggler, stage 1 complete, heap empty); every other attempt ran (36 done) and the run ended at once; the failed attempts are recorded once and never requeued; attempt 2 still ran where attempt 1 failed; the submission is valid. **PASS.**
- **Built in the same rev (3f), at the user's request:** attempt 1 = 45k, attempt 2 = 30k (rev 3f / 3g; record in Notes/10).
- **Effect on the GPU run going now (2X35K, sha 0f0d6f67):** it carries the old code. The blocker fires only if `BranchSession()` itself raises (it creates an object and a folder), so it is unlikely there.

---

E15B Rev 3 — Final Pre-GPU Audit
NO-GO — ১টি HIGH blocker
দুটি notebook-এর SHA-256, source code, scheduler, watchdog, submission path এবং CPU validation যাচাই করেছি।

একটি reproducible scheduler bug পেয়েছি। অন্য কোনো নিশ্চিত catastrophic bug পাইনি।
BLOCKER — Worker slot স্থায়ীভাবে আটকে যেতে পারে
File: src/duck_b/arc2_duck.py
Lines: 3351–3355, 3294–3305

বর্তমান code:
state["running"] += 1active.add((tid, b))s = sess(tid, b)



সমস্যা: sess() exception দিলে বাইরের worker wrapper exception ধরে restart করে, কিন্তু running কমায় না এবং active থেকে task সরায় না।

ফলে worker slot leak হয় এবং task queue থেকে হারিয়ে যায়। এমন ব্যর্থতা একাধিকবার হলে ১৬টি worker slot-ই আটকে গিয়ে solver বাকি tasks চালাতে পারবে না।
Independent reproduction
BranchSession.__init__()-এ controlled exception inject করে solve_all_r6() সরাসরি চালিয়েছি।
Injected error: OSError("session init failure")

worker_errors: 1
stage1_left: 2
heap_left: 2
tasks_with_request_sent: 0
stragglers: 1


অর্থাৎ worker restart করার ব্যবস্থা থাকলেও scheduler bookkeeping recover করেনি।

এটি failure-injection-এ নিশ্চিত, বাস্তব Kaggle GPU-তে ঘটেছে এমন প্রমাণ নয়।
Required fix
state["running"] বাড়ানোর আগে session setup সফল করতে হবে। Setup failure হলে terminal failure record সংরক্ষণ, failed-job tracking এবং queue advancement নিশ্চিত করতে হবে।
# Inside scheduler condition locktry:    s = sess(tid, b)except Exception as exc:    failed_jobs.add((tid, b))    state["worker_errors"] += 1    records[tid].append({        "task_id": tid,        "branch": b,        "status": "done",        "accepted": False,        "submissions": [],        "checked": [],        "note": f"session setup failed: {exc}",    })    stage2_items(tid)    cond.notify_all()    continuestate["running"] += 1active.add((tid, b))



stage2_items()-এ failed job পুনরায় enqueue হওয়া আটকাতে হবে:
if (    branches > 1    and (tid, 1) not in failed_jobs    and (tid, 1) not in sessions    and (tid, 1) not in queued):    push(rank(tid, 1, sub), tid, 1, "start1")



এই patch বিদ্যমান lock, branch record ও callback contract অনুসারে integrate করতে হবে।

Mandatory test: ১৬-worker synthetic run-এ একবার এবং বারবার session-setup failure inject করুন। Verify করুন:
- running বা active slot leak নেই।
- অন্য tasks চলতে থাকে।
- কোনো failed branch অনন্তকাল requeue হয় না।
- Final submission valid থাকে।
- Scheduler deadline পর্যন্ত অকারণে আটকে থাকে না।
বাকি audit
Check	Result
SUBMIT notebook SHA 0f0d6f67…	PASS
EVAL120 notebook SHA 30653201…	PASS
Notebook code compilation	PASS
Kaggle CPU kit	77/77 recorded PASS
CPU harness modes	সব recorded PASS
2×35k / fair-share disabled	Source verified
16-stream flags / capacity configuration	Source verified
Watchdog fault-injection logs	PASS; real GPU unverified
Final submission safeguards	নতুন blocker পাইনি

চূড়ান্ত সিদ্ধান্ত: শুধু এই scheduler accounting bug সংশোধন করে negative test PASS করান। অন্য কোনো redesign প্রয়োজন নেই।

তারপর GO for GPU smoke দেওয়া যুক্তিসংগত হবে; ১৬-stream-এর বাস্তব memory ও throughput এখনও GPU-তে যাচাই করতে হবে।
