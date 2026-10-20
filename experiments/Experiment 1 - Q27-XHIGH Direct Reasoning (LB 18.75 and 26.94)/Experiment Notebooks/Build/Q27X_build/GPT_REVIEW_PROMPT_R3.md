# Review request R3: Q27-XHIGH v1.3.1 (after your R2 reviews of v1.2 and v1.2.1)

Read in this order:
1. `REVIEW_RESPONSE_R2_v1_3.md`: my verdict on each of your 8 R2 points, with the fix and the test that covers it.
2. `DISCUSSION_CHECK_v1_3_1.md`: every statement of Kaggle discussion 745787 compared with the notebook's actual values.
3. `notebooks/Q27-XHIGH-v1_3_1.ipynb` (builder: `build/make_q27x.py`).
4. `build/harness_output_v1_3_1.txt`.

Reproduce from the zip (needs `tokenizers`, `jinja2`, `numpy`):
```
python build/check_defaults.py notebooks/Q27-XHIGH-v1_3_1.ipynb
python build/harness_q27x.py notebooks/Q27-XHIGH-v1_3_1.ipynb <mixed|drain|rerun|crash|eos|vdeadline|allfail|smokefail|norelease>
```

Scope rule from the user: stay on the goal — a full-thinking Qwen3.8-27B submission following the discussion recipe.
The repair turn, NVARC fusion and the import blocklist remain deferred.

## Answer
1. Do you accept the R2 ledger? Where you disagree, give evidence (file + line).
2. Look for new bugs in the v1.3 changes:
   - the cell-1 deadline derivation and assert;
   - the cell-5 engine ladder, i.e. `_engine_smoke` (`llm.generate` before the step-API loop), engine-core shutdown,
     `_wait_gpu_release`, `ENGINE_READY=False`;
   - the cell-6 skip path and metrics snapshots;
   - the cell-7 projection validity and the MTP acceptance computed from `get_metrics()`.
3. P0 / P1 / P2 only. For each: file + cell + line; VERIFIED or INFERRED; the exact fix.
4. Verdict: GO / NO-GO for the 12-task commit run on Kaggle 4×L4.
