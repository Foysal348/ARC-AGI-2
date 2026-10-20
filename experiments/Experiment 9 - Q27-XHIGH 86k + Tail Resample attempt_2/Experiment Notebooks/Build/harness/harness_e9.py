"""Local harness for Q27-XHIGH-v1: the real notebook cells run with the real Qwen tokenizer + chat template, real
competition JSON, and a fake vLLM whose engine implements add_request / step / abort_request (vLLM 0.19 V1 semantics:
DELTA token ids, external request ids, finish_reason 'stop' | 'length'). Each task gets a scripted behaviour.

usage: python harness_q27x.py <notebook> <scenario>
scenarios: mixed (all behaviours, generous time), drain (drain + hard stop), rerun (competition rerun, all test tasks)"""
import json, os, sys, time, types, tempfile, shutil, random
from pathlib import Path

NB, SCEN = sys.argv[1], sys.argv[2]
HERE = Path(__file__).resolve().parent
FAKE = Path(os.environ["Q27X_FIXTURES"]) if os.environ.get("Q27X_FIXTURES") else None
if FAKE is None or not FAKE.is_dir():   # repo layout: scratchpad/fake_input; bundle layout: fixtures/fake_input next to build/
    FAKE = next(p for p in (HERE.parent / "fake_input", HERE.parent / "fixtures" / "fake_input") if p.is_dir())
COMP = FAKE / "competitions" / "arc-prize-2026-arc-agi-2"
MODEL = next((FAKE / "models" / "foysalemonshanto" / "qwen3-8-27b-fp8-repacked-v1" / "pytorch").glob("*/*"))
WORK = Path(tempfile.mkdtemp(prefix=f"q27x_{SCEN}_"))
os.environ.update(Q27X_ARC_ROOT=str(COMP), Q27X_WORKING=str(WORK), Q27X_MODEL=str(MODEL), Q27X_SKIP_INSTALL="1",
                  Q27X_THINK_BUDGET="3000", Q27X_ANSWER_MAX="40000", Q27X_GPU_WAIT_MAX="1", PYTHONIOENCODING="utf-8")
if SCEN in ("rerun", "crash"):
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "true"; os.environ["Q27X_TIME_BUDGET"] = str(60 * 60)
elif SCEN == "drain":
    os.environ["Q27X_TIME_BUDGET"] = str(3 * 3600)        # production-ordered deadlines; the scenario tightens them before cell 6
else:
    os.environ["Q27X_TIME_BUDGET"] = str(3 * 3600)

# ---------------------------------------------------------------- transformers stub (real tokenizer + template)
import tokenizers, jinja2
class _Tok:
    def __init__(self, d):
        self.t = tokenizers.Tokenizer.from_file(str(Path(d) / "tokenizer.json"))
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, **kw)
        def _raise(msg): raise ValueError(msg)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.tpl = env.from_string((Path(d) / "chat_template.jinja").read_text(encoding="utf-8"))
    def encode(self, text, add_special_tokens=False): return self.t.encode(text, add_special_tokens=add_special_tokens).ids
    def decode(self, ids, skip_special_tokens=False): return self.t.decode(list(ids), skip_special_tokens=skip_special_tokens)
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None, **kw):
        return self.tpl.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools, **kw)
TOK = _Tok(MODEL)
_tf = types.ModuleType("transformers"); _tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda d, **k: TOK)
sys.modules["transformers"] = _tf

# ---------------------------------------------------------------- nvidia-smi stub (GPU memory after a failed engine)
import subprocess as _sp
_real_run = _sp.run
def _fake_run(cmd, *a, **k):
    if isinstance(cmd, list) and cmd and cmd[0] == "nvidia-smi":
        used = "8000" if SCEN == "norelease" or any(p.poll() is None for p in ORPHANS) else "300"
        return _sp.CompletedProcess(cmd, 0, stdout=" ".join([used] * 4), stderr="")
    return _real_run(cmd, *a, **k)
_sp.run = _fake_run
ORPHANS = []      # scenario "orphan": real child processes left behind by a failed engine start (run 1 on Kaggle)

# ---------------------------------------------------------------- vLLM stub
CLOSE = TOK.encode("</think>")[0]
STOPTXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
STOP_IDS = TOK.encode(STOPTXT)
LOG = {"added": [], "aborted": [], "max_running": 0, "phase2_prompts": []}
EVAL_SOL = json.load(open(COMP / "arc-agi_evaluation_solutions.json"))
EVAL_CH = json.load(open(COMP / "arc-agi_evaluation_challenges.json"))
TEST_CH = json.load(open(COMP / "arc-agi_test_challenges.json"))

def sol_for(uid):
    """v1.4: one unit = one test input. Hidden-test inputs have no solution: the fake model answers the input with
    each row reversed (non-identity, GPT R2 point 8)."""
    uid = uid.split("~")[0]                      # E9: resample units "tid#q~r1" answer like their base unit
    tid, qi = uid.split("#"); qi = int(qi)
    if tid in EVAL_SOL:
        return EVAL_SOL[tid][qi]
    return [list(reversed(r)) for r in TEST_CH[tid]["test"][qi]["input"]]

WRONG_GRID = [[9, 9], [9, 9]]
def answer_for(uid, kind):
    s = sol_for(uid)
    spaced = "\n".join(" ".join(map(str, r)) for r in s)
    if kind == "good":
        return "The rule is clear.\n```grid\n" + spaced + "\n```"
    if kind == "good_contig":
        return "```text\n" + "\n".join("".join(map(str, r)) for r in s) + "\n```"
    if kind == "good_json_broken":            # run 3: the outer '[' was missing in 6/12 JSON blocks
        return "```json\n{\"test_outputs\": [" + ", ".join(json.dumps(r) for r in s) + "]]}\n```"
    if kind == "good_unfenced":
        return "Final answer:\n" + spaced + "\n"
    if kind == "wrong":
        return "```grid\n" + "\n".join(" ".join(map(str, r)) for r in WRONG_GRID) + "\n```"
    if kind == "nothing":
        return "I could not find the rule."
    raise ValueError(kind)

REST = {}
BEHAVIOUR = {}     # uid -> behaviour
GOOD = ("natural_good", "forced_good", "natural_cut_good", "natural_contig", "natural_json_broken", "natural_unfenced")
def plan(uids):
    kinds = ["natural_good", "forced_good", "natural_cut_good", "natural_contig", "natural_json_broken", "natural_unfenced",
             "natural_wrong", "natural_nothing", "forced_nothing"]
    if SCEN == "eos":
        kinds = ["eos_unclosed"]
    if SCEN in ("allfail", "smokefail", "norelease", "orphan"):
        kinds = ["natural_good"]
    for i, u in enumerate(uids):
        BEHAVIOUR[u] = kinds[i % len(kinds)]

class SP:
    def __init__(self, **kw): self.__dict__.update(kw); assert kw["max_tokens"] >= 1 and kw.get("detokenize") is False
class RequestOutputKind: DELTA = "delta"; CUMULATIVE = "cum"; FINAL_ONLY = "final"
class _Comp:
    def __init__(self, ids, fr): self.token_ids = ids; self.finish_reason = fr
class _Out:
    def __init__(self, rid, ids, finished, fr): self.request_id = rid; self.outputs = [_Comp(ids, fr)]; self.finished = finished

class FakeEngine:
    def __init__(self, step_sleep=0.0, per_step=400):
        self.reqs = {}; self.step_sleep = step_sleep; self.per_step = per_step
        # audit F3: vLLM copies num_gpu_blocks to the front end; with MTP ~778 blocks x 800 tokens -> concurrency 15
        self.vllm_config = types.SimpleNamespace(cache_config=types.SimpleNamespace(num_gpu_blocks=778, block_size=800))
        self.nsteps = 0
    def _script(self, rid, prompt_ids, params):
        tid, phase = rid.rsplit("|", 1)
        beh = BEHAVIOUR.get(tid, "natural_good")
        if "~r" in tid: beh = "natural_good"         # E9: a resample of any unit closes naturally with the right grid
        think = TOK.encode(" thinking" * 50)
        if phase == "1":
            if beh.startswith("forced"):
                return think * 100                                  # never closes -> hits THINK_BUDGET
            if beh == "eos_unclosed":
                return think * 3 + [248046]                         # EOS inside thinking, no </think>
            body = think * 2 + [CLOSE] + TOK.encode("\n\n")
            kind = {"natural_good": "good", "natural_cut_good": "good", "natural_contig": "good_contig",
                    "natural_json_broken": "good_json_broken", "natural_unfenced": "good_unfenced", "natural_wrong": "wrong",
                    "natural_nothing": "nothing"}[beh]
            ans = TOK.encode(answer_for(tid, kind))
            if beh == "natural_cut_good":                            # answer starts just before the budget -> cut, then continued
                pad = params.max_tokens - len(body) - 20
                return TOK.encode(" x") * max(pad, 0) + body + ans
            return body + ans
        # phase 2: prompt ends with stop text (forced) or with the cut answer (continue)
        LOG["phase2_prompts"].append((tid, prompt_ids[-len(STOP_IDS):] == STOP_IDS))
        if prompt_ids[-len(STOP_IDS):] == STOP_IDS:
            LOG.setdefault("eos_before_stop", []).append(prompt_ids[-len(STOP_IDS) - 1] in (248046, 248044))
            kind = "good" if beh in ("forced_good", "eos_unclosed") else "nothing"
            return TOK.encode(answer_for(tid, kind))
        # continuation: emit exactly the remainder of phase 1's script
        return REST[tid]
    def add_request(self, request_id, prompt, params):
        assert isinstance(request_id, str) and "prompt_token_ids" in prompt and params.output_kind == "delta"
        assert request_id not in self.reqs
        self.reqs[request_id] = {"script": self._script(request_id, prompt["prompt_token_ids"], params), "pos": 0,
                                 "max": params.max_tokens}
        LOG["added"].append(request_id); return request_id
    def abort_request(self, ids, internal=False):
        for i in ids:
            r = self.reqs.pop(i, None); LOG["aborted"].append(i)
            if r is not None and i.endswith("|1"):
                REST[i.rsplit("|", 1)[0]] = r["script"][r["pos"]:]      # what the model would have said next
    def step(self):
        self.nsteps += 1
        if SCEN == "crash" and self.nsteps == 6:
            raise RuntimeError("EngineDeadError (simulated engine-core crash)")
        if self.step_sleep: time.sleep(self.step_sleep)
        LOG["max_running"] = max(LOG["max_running"], len(self.reqs))
        outs = []
        for rid, r in list(self.reqs.items()):
            take = r["script"][r["pos"]: r["pos"] + self.per_step]
            take = take[: max(0, r["max"] - r["pos"])]
            r["pos"] += len(take)
            done_script = r["pos"] >= len(r["script"]); done_len = r["pos"] >= r["max"]
            if done_len and not done_script:
                REST[rid.rsplit("|", 1)[0]] = r["script"][r["pos"]:]
                outs.append(_Out(rid, take, True, "length")); del self.reqs[rid]
            elif done_script: outs.append(_Out(rid, take, True, "stop")); del self.reqs[rid]
            else: outs.append(_Out(rid, take, False, None))
        return outs

ENGINE = FakeEngine(step_sleep=(0.5 if SCEN == "drain" else 0.0), per_step=(60 if SCEN == "drain" else 400))
class _Metric:
    def __init__(self, name, value): self.name = name; self.value = value; self.labels = {}
class LLM:
    def __init__(self, **kw):
        LOG["engine_kwargs"] = kw
        LOG.setdefault("ctor", []).append((kw.get("kv_cache_dtype"), bool(kw.get("speculative_config"))))
        # v1.3.5: every engine attempt gets the run-2 OOM fixes (vLLM workers inherit os.environ at spawn)
        assert kw["max_num_batched_tokens"] == 4096 and kw["gpu_memory_utilization"] == 0.90, kw   # v1.4.1
        assert os.environ.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") == "1"
        if kw.get("speculative_config") and os.environ.get("HARNESS_REFUSE_SPEC") == "1":
            raise RuntimeError("spec refused (simulated)")
        if SCEN == "allfail":
            raise RuntimeError("engine start failed (simulated)")
        if SCEN == "orphan" and kw.get("speculative_config"):
            ORPHANS.append(_sp.Popen([sys.executable, "-c", "import time; time.sleep(600)"]))
            raise RuntimeError("EngineDeadError: EngineCore encountered an issue (simulated, workers left alive)")
        self.kw = kw
        assert kw["attention_config"]["backend"] in ("FLASHINFER", "TRITON_ATTN", "FLASH_ATTN"), kw.get("attention_config")
        self.llm_engine = ENGINE
    def generate(self, prompts, params, use_tqdm=False):      # engine smoke test
        if SCEN in ("smokefail", "norelease") and self.kw.get("speculative_config"):
            raise RuntimeError("FlashInfer fp8 decode failed (simulated)")
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=list(range(16)))])]
    def get_metrics(self):
        n = ENGINE.nsteps
        return [_Metric("vllm:spec_decode_num_draft_tokens", 300.0 * n), _Metric("vllm:spec_decode_num_accepted_tokens", 210.0 * n),
                _Metric("vllm:num_preemptions", 2.0), _Metric("vllm:kv_cache_usage_perc", 0.6)]
_v = types.ModuleType("vllm"); _v.LLM = LLM; _v.SamplingParams = SP; _v.__version__ = "0.19.0"
_vsp = types.ModuleType("vllm.sampling_params"); _vsp.RequestOutputKind = RequestOutputKind
sys.modules["vllm"] = _v; sys.modules["vllm.sampling_params"] = _vsp

# ---------------------------------------------------------------- run the notebook cells
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
for i, src in enumerate(codes):
    if i == 6 and SCEN == "drain":
        # tighten the clock for the drain scenario: drain soon, hard stop shortly after
        G["ADMIT_DEADLINE"] = time.perf_counter() + 2.0
        G["DRAIN_DEADLINE"] = time.perf_counter() + 6.0
        G["HARD_STOP"] = time.perf_counter() + 14.0
    if i == 5:
        plan(G["UNITS"])
    if i == 6 and SCEN == "vdeadline":
        G["VERIFY_DEADLINE"] = time.perf_counter() + 8.0; T_SCHED = time.perf_counter()
    exec(compile(src, f"cell{i}", "exec"), G)

# ---------------------------------------------------------------- checks
rep = json.load(open(WORK / "q27x" / "run_report.json"))
ALL_RECS = [json.loads(l) for l in open(WORK / "q27x" / "records.jsonl")] if (WORK / "q27x" / "records.jsonl").exists() else []
recs = [r for r in ALL_RECS if not r.get("resample_of")]          # E9: primary units only for the v1.4.4 checks
RS = [r for r in ALL_RECS if r.get("resample_of")]
ans_files = list((WORK / "q27x" / "answers").glob("*.txt"))
assert len(ans_files) >= len(recs), (len(ans_files), len(recs))    # audit F5: answers saved
sub = json.load(open(WORK / "submission.json"))
print("RESULT", json.dumps({"scenario": SCEN, "finished": rep["queries_finished"], "planned": rep["queries_planned"],
                            "modes": rep["modes"], "score": rep.get("eval_score_sum"), "sched": {k: rep["scheduler"][k] for k in
                            ("admitted", "phase2_continue", "phase2_forced", "drained", "aborted_at_hard_stop", "not_admitted", "max_active")},
                            "first_src": rep["first_attempt_source"], "max_running": LOG["max_running"],
                            "spec": rep["engine"]["speculative"], "kv": rep["engine"]["kv_cache_dtype"],
                            "engine_attempts": [(a["kv_cache_dtype"], bool(a["speculative"]), a["ok"]) for a in rep["engine"]["attempts"]]}))
by = {r["unit"]: r for r in recs}
assert set(sub) == set(TEST_CH), "submission must cover every test task"
for t, task in TEST_CH.items():
    assert len(sub[t]) == len(task["test"])
assert G["CONCURRENCY"] == (max(4, min(16, 778 * 800 // G["EXPECTED_LIVE_TOKENS_PER_SEQ"])) if G["ENGINE_READY"] else 0), G["CONCURRENCY"]
if SCEN == "mixed":
    for t, beh in BEHAVIOUR.items():
        r = by[t]; h = r["score"]["hit"]
        if beh in GOOD:
            assert h == 1 and r["src"][0] == "model" and r["has_grid"], (t, beh, r)
        if beh == "natural_wrong":
            assert h == 0 and r["src"][0] == "model" and r["has_grid"], (t, beh, r)
        if beh in ("natural_nothing", "forced_nothing"):
            assert r["src"][0] == "test_input" and not r["has_grid"], (t, beh, r)
        if beh.startswith("forced"):
            assert r["mode"] == "forced_close" and r["think_closed"] is True, (t, r)
        if beh == "natural_cut_good":
            assert r["mode"] == "continue", (t, r)
    assert all(ok for tid, ok in LOG["phase2_prompts"] if BEHAVIOUR[tid].startswith("forced")), "forced phase-2 prompt must end with the stop text"
    assert LOG["max_running"] <= G["CONCURRENCY"]
    ek = LOG["engine_kwargs"]
    assert ek["max_model_len"] == 106496 and ek["tensor_parallel_size"] == 4
    if not os.environ.get("HARNESS_REFUSE_SPEC"):   # v1.3.4: fp8 KV; no libcuda locally -> link probe FAIL -> Triton attention
        assert ek.get("kv_cache_dtype") == "fp8_e4m3" and ek["attention_config"]["backend"] == "TRITON_ATTN", (ek.get("kv_cache_dtype"), ek.get("attention_config"))
        assert rep["engine"]["cuda_link"]["probe"] == "FAIL"
    else:
        assert ek.get("kv_cache_dtype") is None and ek["attention_config"]["backend"] == "FLASH_ATTN"
    assert ek.get("speculative_config") == ({"method": "mtp", "num_speculative_tokens": 3} if not os.environ.get("HARNESS_REFUSE_SPEC") else None)
    assert G["SAMPLING"]["temperature"] == 0.6 and all(c == 3000 for c in G["P1_CAPS"].values())
    _hits = [r["score"]["hit"] for r in recs]
    _q = sum(len(EVAL_SOL[t]) for t in G["TASK_IDS"]); assert _q == len(G["UNITS"]) == len(recs)
    # E9: the report's eval_query_hits scores the FINAL pairs (after tail resamples); the primary-only score is separate
    _gain = sum(1 for r in RS if r["score_after"]["hit"] and not r["score_before"]["hit"])
    assert rep["eval_primary_only_hits"] == sum(_hits) and rep["eval_query_total"] == _q, (rep["eval_primary_only_hits"], sum(_hits))
    assert rep["eval_query_hits"] == sum(_hits) + _gain == rep["eval_primary_only_hits"] + rep["eval_resample_gain"], (rep["eval_query_hits"], sum(_hits), _gain)
    assert abs(rep["eval_score_pct"] - round(100 * (sum(_hits) + _gain) / _q, 2)) < 1e-9 and ek["enforce_eager"] is False and ek["language_model_only"] is True
    assert ek["enable_prefix_caching"] is (not os.environ.get("HARNESS_REFUSE_SPEC")) and ek["disable_custom_all_reduce"] is True
    if not os.environ.get("HARNESS_REFUSE_SPEC"):   # v1.4.1: first attempt = prefix caching on
        assert rep["engine"]["prefix_caching"] is True and rep["engine"]["attempts"][0]["prefix_caching"] is True, rep["engine"]
        assert "prefix_cache_hit_rate" in rep, "report must carry the prefix-cache hit rate"
if SCEN == "drain":   # v1.3.2: no projection from a run with drained / hard-stopped queries
    assert rep["projection_valid"] is False and rep["projection_hidden_end_to_end_hours"] is None, rep.get("projection_reason")
    print(SCEN + ": projection correctly marked invalid:", rep.get("projection_reason", "")[:60])
if SCEN == "drain":
    assert rep["scheduler"]["drained"] + rep["scheduler"]["aborted_at_hard_stop"] >= 1, rep["scheduler"]
    assert rep["queries_finished"] + rep["scheduler"]["not_admitted"] == rep["queries_planned"] - len(rep["skipped_long_prompt"]), rep
if SCEN == "mixed" and not os.environ.get("HARNESS_REFUSE_SPEC"):
    # v1.3.4: unit test of the libcuda shim + link probe with a fake driver file and a fake compiler
    import shutil as _shu, tempfile as _tf
    _c5 = CODES5 = [c for c in nb["cells"] if c["cell_type"] == "code"][5]
    _src = "".join(_c5["source"]); _src = _src[:_src.index('os.environ.setdefault("MAX_JOBS"')]
    H = {"os": os, "json": json, "Path": Path, "time": time}
    exec(compile(_src, "cell5_linkfn", "exec"), H)
    _d = Path(_tf.mkdtemp()); _drv = _d / "libcuda.so.1"; _drv.write_bytes(b"ELF fake")
    def _fake_cc(ok):
        def run(cmd, **k):
            good = ok and "-lcuda" in cmd and str(_d / "shim") in k["env"].get("LIBRARY_PATH", "")
            return _sp.CompletedProcess(cmd, 0 if good else 1, stdout="", stderr="" if good else "ld: cannot find -lcuda")
        return run
    for ok, want in ((True, "PASS"), (False, "FAIL")):
        os.environ.pop("LIBRARY_PATH", None)
        H["subprocess"] = types.SimpleNamespace(run=_fake_cc(ok)); H["_shutil"] = types.SimpleNamespace(which=lambda n: "c++", copyfile=_shu.copyfile)
        r = H["ensure_libcuda_linkable"](candidates=[str(_drv)], workdir=str(_d / "shim"))
        assert r["probe"] == want and r["needed"] is True and (_d / "shim" / "libcuda.so").exists(), r
    r = H["ensure_libcuda_linkable"](candidates=[], workdir=str(_d / "shim2"))
    assert r["probe"] == "FAIL" and r["reason"] == "libcuda.so.1 not found", r
    os.environ.pop("LIBRARY_PATH", None)
    print("linkprobe: shim + LIBRARY_PATH -> PASS; failing link -> FAIL with stderr; no driver -> FAIL 'not found'")
if SCEN == "allfail":
    assert rep["engine"]["fatal"] == "all_engine_attempts_failed" and rep["queries_finished"] == 0, rep["engine"]
    assert all(a["attempt_1"] == q["input"] for t, task in TEST_CH.items() for a, q in zip(sub[t], task["test"]))
    assert len(rep["engine"]["attempts"]) == 4 and rep["projection_valid"] is False   # v1.4.1: fp8+Triton+prefix, fp8+Triton, bf16+FA2+MTP, bf16+FA2
    print("allfail: all 3 engine attempts failed; notebook completed; identity submission + report written")
if SCEN == "norelease":
    att = rep["engine"]["attempts"]
    assert len(att) == 1 and att[0]["gpu_release"]["released"] is False and rep["engine"]["fatal"] == "gpu_memory_not_released_after_failed_engine", att
    assert rep["queries_finished"] == 0
    print("norelease: GPU memory still held after the failed attempt -> no further attempt; notebook completed with identity")
if SCEN == "smokefail":
    att = rep["engine"]["attempts"]
    assert [a["ok"] for a in att] == [False, False, False, True] and "decode failed" in att[0]["error"] and att[-1]["speculative"] is None, att
    assert rep["queries_finished"] == rep["queries_planned"]
    print("smokefail: MTP engines (fp8+Triton, bf16+FA2) started but failed the decode smoke -> bf16 without MTP ran all tasks")
if SCEN == "orphan":
    att = rep["engine"]["attempts"]
    assert att[0]["ok"] is False and att[0]["killed_children"]["found"] >= 1 and att[0]["gpu_release"]["released"] is True, att[0]
    assert att[-1]["ok"] is True and all(p.poll() is not None for p in ORPHANS) and rep["queries_finished"] == rep["queries_planned"]
    print("orphan: failed engine left a live child holding 'GPU memory' -> killed (found", att[0]["killed_children"]["found"],
          ") -> GPU released -> fallback engine ran all", rep["queries_finished"], "queries")
if SCEN in ("mixed",) and not os.environ.get("HARNESS_REFUSE_SPEC"):
    _rs_ran = rep.get("resample", {}).get("admitted", 0) > 0                # E9 fix 6: no projection once resamples ran
    assert rep["mtp_acceptance_rate"] == 0.7 and rep["projection_valid"] is (not _rs_ran), (rep.get("mtp_acceptance_rate"), rep.get("projection_valid"))
    if _rs_ran: assert "tail resamples ran" in rep["projection_reason"]
    # E9 fix 5: task-normalised score recomputed from the final submission pairs
    _fin = json.load(open(WORK / "q27x" / "run_report.json"))
    _by = {}
    for r in recs: _by.setdefault(r["task_id"], {})[r["query"]] = r["score"]["hit"]
    for r in RS:
        if r["score_after"]["hit"]: _by[r["task_id"]][r["query"]] = 1
    _tn = round(100 * sum(sum(v.values()) / len(EVAL_SOL[t]) for t, v in _by.items()) / len(G["TASK_IDS"]), 2)
    assert abs(rep["eval_task_normalized_pct"] - _tn) < 1e-9, (rep["eval_task_normalized_pct"], _tn)
    print("E9 fix5: task-normalised final", rep["eval_task_normalized_pct"], "primary-only", rep["eval_task_normalized_primary_only_pct"])
    assert (WORK / "q27x" / "vllm_metrics.json").exists()
    print("mixed: MTP acceptance in report", rep["mtp_acceptance_rate"], "| projection_valid", rep["projection_valid"],
          "| deadlines (min)", rep["deadlines_min"])
if SCEN == "crash":
    assert rep["scheduler"]["error"] and "EngineDeadError" in rep["scheduler"]["error"], rep["scheduler"].get("error")
    print("crash: notebook finished all cells; error recorded; submission valid with", rep["queries_finished"], "finished queries")
if SCEN == "eos":
    assert LOG.get("eos_before_stop") and not any(LOG["eos_before_stop"]), LOG.get("eos_before_stop")
    assert all(r["score"]["hit"] == 1 for r in recs), [r["score"] for r in recs]
    print("eos: EOS stripped before the stop text in", len(LOG["eos_before_stop"]), "forced closes")
if SCEN == "rerun":
    _model_first = [r["unit"] for r in recs if r["src"][0] == "model" and BEHAVIOUR[r["unit"]] in GOOD]
    for u in _model_first:
        t, q = u.split("#")
        assert sub[t][int(q)]["attempt_1"] == sol_for(u), u
    _nonid_written = sum(1 for u in _model_first if sub[u.split("#")[0]][int(u.split("#")[1])]["attempt_1"]
                         != TEST_CH[u.split("#")[0]]["test"][int(u.split("#")[1])]["input"])
    assert _nonid_written > 100, _nonid_written
    print("rerun: model grids written to submission.json for", len(_model_first), "queries;", _nonid_written,
          "non-identity first attempts verified")
    nonid = sum(1 for t, task in TEST_CH.items() for a, q in zip(sub[t], task["test"]) if a["attempt_1"] != q["input"])
    print("rerun: non-identity first attempts:", nonid)
    assert rep["tasks_planned"] == len(TEST_CH) and rep["queries_planned"] == sum(len(t["test"]) for t in TEST_CH.values())
    assert rep["queries_finished"] == rep["queries_planned"] - len(rep["skipped_long_prompt"]), rep["queries_finished"]
# ---- E9 checks
if RS:
    print("E9 resample:", len(RS), "finished; changed attempt_2:", sum(r["changed_attempt2"] for r in RS),
          "| report:", rep.get("resample"), "| primary-only hits", rep.get("eval_primary_only_hits"), "gain", rep.get("eval_resample_gain"))
    for r in RS:
        assert r["resample_of"] + "~r1" == r["unit"]
    base_first = {r["unit"]: r for r in recs}
    for r in RS:     # attempt_1 of the base unit is never changed by a resample
        b = r["resample_of"]; t, q = b.split("#")
        if SCEN == "rerun":
            assert sub[t][int(q)]["attempt_1"] != None
print("SCENARIO PASS", SCEN, "| work dir", WORK)
