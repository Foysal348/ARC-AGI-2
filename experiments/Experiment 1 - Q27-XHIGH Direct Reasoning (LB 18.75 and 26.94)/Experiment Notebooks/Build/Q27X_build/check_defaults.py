"""Run cell 1 (configuration) of the notebook with the PRODUCTION defaults (no Q27X_* overrides), in both modes, and
assert the values that must match Kaggle discussion 745787. The harness overrides budgets for speed, so it cannot
catch a self-check that only fails at the real defaults (this happened once in v1.2.1).
usage: python check_defaults.py <notebook>"""
import json, os, sys
from pathlib import Path

nb = json.load(open(sys.argv[1], encoding="utf-8"))
c1 = "".join([c for c in nb["cells"] if c["cell_type"] == "code"][1]["source"])
for k in list(os.environ):
    if k.startswith("Q27X_"):
        os.environ.pop(k)
for rerun in (False, True):
    G = {"os": os, "json": json, "Path": Path, "RERUN": rerun, "NOTEBOOK_T0": 0.0}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        exec(c1, G)
    expect = {"REASONING_EFFORT": "xhigh", "THINK_BUDGET": 49152, "ANSWER_MAX_TOKENS": 8192, "MAX_MODEL_LEN": 81920,
              "KV_CACHE_DTYPE": "fp8_e4m3", "SPECULATIVE_CONFIG": {"method": "mtp", "num_speculative_tokens": 3},
              "SAMPLING": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}, "MAX_NUM_SEQS": 16,
              "MAX_NUM_BATCHED_TOKENS": 4096, "GPU_MEMORY_UTILIZATION": 0.90, "ENABLE_PREFIX_CACHING": True,   # v1.4.1
              "PROFILE": "RERUN_ALL" if rerun else "EVAL12"}
    bad = {k: (G[k], v) for k, v in expect.items() if G[k] != v}
    assert not bad, bad
    assert os.environ.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "1"    # v1.3.5: CUDA graphs inside the KV budget
    assert G["CONFIG"]["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] == "1"
    os.environ.pop("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS")                  # the next mode must set it again itself
    # vLLM 0.19 config check (config/scheduler.py 273-278). MTP is serial drafting: 0 extra slots (config/speculative.py 847)
    assert G["MAX_NUM_BATCHED_TOKENS"] >= G["MAX_NUM_SEQS"]
    assert G["THINK_BUDGET"] + G["ANSWER_MAX_TOKENS"] == 57344            # v1.4: commit after 49k (Nguyen); total = russcore max_tokens
    assert G["phase1_cap"](8712) == 49152                                  # longest real prompt (evaluation) keeps the full budget
    t0 = G["NOTEBOOK_T0"]
    if rerun:   # v1.4.3: the hidden rerun stops 10 min before 12 h; hard stop leaves >= 16 min of the 12 h
        assert G["TOTAL_BUDGET_SEC"] == 12 * 3600 - 10 * 60, G["TOTAL_BUDGET_SEC"]
        assert G["HARD_STOP"] - t0 == 11 * 3600 + 44 * 60 and G["VERIFY_DEADLINE"] - t0 <= 12 * 3600 - 10 * 60
        assert G["DRAIN_DEADLINE"] - t0 == 11 * 3600 + 36 * 60 and G["ADMIT_DEADLINE"] - t0 == 11 * 3600 + 34 * 60   # v1.4.4
    else:   # EVAL12 deadlines unchanged since v1.3
        assert [round((G[k] - t0) / 60, 1) for k in ("ADMIT_DEADLINE", "DRAIN_DEADLINE", "HARD_STOP")] == [140.0, 147.0, 155.0]
    assert t0 < G["ADMIT_DEADLINE"] < G["DRAIN_DEADLINE"] < G["HARD_STOP"] < G["VERIFY_DEADLINE"] < t0 + G["TOTAL_BUDGET_SEC"]
    dl = {k: round((G[k] - t0) / 60, 1) for k in ("ADMIT_DEADLINE", "DRAIN_DEADLINE", "HARD_STOP", "VERIFY_DEADLINE")}
    print("   production deadlines (min):", dl)
    print(f"defaults OK (rerun={rerun}): profile {G['PROFILE']}, think {G['THINK_BUDGET']} + answer {G['ANSWER_MAX_TOKENS']}, "
          f"ctx {G['MAX_MODEL_LEN']}, kv {G['KV_CACHE_DTYPE']}, mtp {G['SPECULATIVE_CONFIG']['num_speculative_tokens']}, "
          f"temp {G['SAMPLING']['temperature']}, budget {G['TOTAL_BUDGET_SEC'] / 3600:.1f} h")
print("DEFAULTS CHECK PASS")
