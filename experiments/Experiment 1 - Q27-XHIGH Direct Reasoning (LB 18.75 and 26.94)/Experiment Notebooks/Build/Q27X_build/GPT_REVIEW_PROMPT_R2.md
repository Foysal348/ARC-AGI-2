# Review request R2: Q27-XHIGH v1.2 (after your R1 review and an independent agent audit)

Read the files in this order:
1. `REVIEW_RESPONSE_v1_2.md`. It holds my verdict on each of your 12 points and the agent's 13 findings, what was
   changed, and what was **deliberately deferred**.
   - The user's rule is to stay on our goal: a full-thinking Qwen3.8-27B submission following the Kaggle discussion
     recipe.
   - So the repair turn, NVARC fusion and the import blocklist are out of scope for now.
2. `PROVENANCE.md` (updated for v1.2).
3. `notebooks/Q27-XHIGH-v1_2_1.ipynb` (builder: `build/make_q27x.py`).
4. `build/harness_output_v1_2_1.txt`. All seven scenarios pass: mixed (incl. 8 answer layouts), drain, rerun (240 tasks),
   crash mid-loop, EOS before `</think>`, verification deadline with hanging programs, and the engine fallback ladder.
   - Reproduce from the zip: `python build/harness_q27x.py notebooks/Q27-XHIGH-v1_2_1.ipynb mixed` (needs `tokenizers`,
     `jinja2`, `numpy`; the fixtures are in `fixtures/`).

Sources are unchanged: `sources/` holds the discussion, the vLLM 0.19.0 source, the model files, our earlier
notebooks and russcore's notebook.

## Answer
1. Do you accept the ledger? Where you disagree, give evidence (file + line).
2. Check the v1.2 changes for new bugs, especially:
   - the fp8 KV ladder in cell 5;
   - the deadline-bounded `collect_verifications` / `_record` in cell 6;
   - `run_solution` (process group, limits) and the fence parser in cell 4;
   - the adaptive `phase1_cap`;
   - per-query scoring in cell 7.
3. P0 / P1 / P2 findings only. For each: file + cell + line; VERIFIED or INFERRED; the exact fix.
4. Verdict: GO / NO-GO for (a) the 12-task commit run, (b) a hidden-test submission after a good commit run.
