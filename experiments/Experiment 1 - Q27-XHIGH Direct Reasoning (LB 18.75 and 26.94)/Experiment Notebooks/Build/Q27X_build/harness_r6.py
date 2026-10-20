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
                  Q27X_THINK_BUDGET="3000", Q27X_REPAIR_THINK="3000", Q27X_GRID_THINK="3000", Q27X_ANSWER_MAX="40000", Q27X_GPU_WAIT_MAX="1", PYTHONIOENCODING="utf-8")
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

def sols_for(tid):
    if tid in EVAL_SOL:
        return EVAL_SOL[tid]
    return [[list(reversed(r)) for r in q["input"]] for q in TEST_CH[tid]["test"]]   # hidden: non-identity (GPT R2 point 8)

def cheat_code(tid):
    ch = EVAL_CH.get(tid) or TEST_CH[tid]
    table = [(ex["input"], ex["output"]) for ex in ch["train"]] + [(q["input"], s) for q, s in zip(ch["test"], sols_for(tid))]
    return ("# rule: lookup (harness)\nTABLE = " + json.dumps(table)
            + "\ndef solve(grid):\n    for i, o in TABLE:\n        if i == grid: return o\n    return grid\n")

def grids_answer(tid):
    return "\n".join("```grid\n" + "\n".join(" ".join(map(str, r)) for r in g) + "\n```" for g in sols_for(tid))

def code_answer(tid, kind):
    if kind == "good":
        return "```python\n" + cheat_code(tid) + "```"
    if kind == "wrong":
        return "```python\n# rule: identity\ndef solve(grid):\n    return grid\n```"
    if kind == "crash":
        return "```python\ndef solve(grid):\n    return grid[99]\n```"
    if kind == "nocode":
        return "I think the rule is a reflection."
    raise ValueError(kind)

PLANS = {"t1_good": ["good"], "t2_good": ["wrong", "good"], "forced_t1_good": ["good"], "nocode_then_good": ["nocode", "good"],
         "crash_then_good": ["crash", "good"], "never": ["wrong"], "t4_good": ["wrong", "wrong", "wrong", "good"]}
EXPECT_TURN = {"t1_good": 1, "t2_good": 2, "forced_t1_good": 1, "nocode_then_good": 2, "crash_then_good": 2, "never": None, "t4_good": 4}
REST = {}
BEHAVIOUR = {}     # tid -> behaviour
def plan(uids):
    kinds = list(PLANS)
    if SCEN == "eos":
        kinds = ["eos_forced"]
    if SCEN in ("allfail", "smokefail", "norelease", "orphan"):
        kinds = ["t1_good"]
    for i, u in enumerate(uids):
        BEHAVIOUR[u] = kinds[i % len(kinds)]

def kind_for(beh, turn):
    if beh == "eos_forced":
        return "good"
    p = PLANS[beh]
    return p[min(turn, len(p)) - 1]

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
        tid, tag = rid.rsplit("|", 1)
        turn, phase = int(tag[1:tag.index("p")]), tag[tag.index("p") + 1:]
        beh = BEHAVIOUR.get(tid, "t1_good")
        think = TOK.encode(" thinking" * 50)
        is_grid = "Skip the code now." in TOK.decode(prompt_ids[-400:])
        if phase == "1":
            LOG.setdefault("turn_prompts", []).append((tid, turn, TOK.decode(prompt_ids)))
            if is_grid:
                LOG.setdefault("grid_turns", []).append(tid)
                return think * 2 + [CLOSE] + TOK.encode("\n\n" + grids_answer(tid))
            if beh == "forced_t1_good" and turn == 1:
                return think * 100                                  # never closes -> hits the thinking cap
            if beh == "eos_forced":
                return think * 3 + [248046]                         # EOS inside thinking, no </think>
            return think * 2 + [CLOSE] + TOK.encode("\n\n" + code_answer(tid, kind_for(beh, turn)))
        LOG["phase2_prompts"].append((tid, prompt_ids[-len(STOP_IDS):] == STOP_IDS))
        if prompt_ids[-len(STOP_IDS):] == STOP_IDS:
            LOG.setdefault("eos_before_stop", []).append(prompt_ids[-len(STOP_IDS) - 1] in (248046, 248044))
            return TOK.encode(grids_answer(tid) if is_grid else code_answer(tid, kind_for(beh, turn)))
        return REST[rid.rsplit("p", 1)[0]]
    def add_request(self, request_id, prompt, params):
        assert isinstance(request_id, str) and "prompt_token_ids" in prompt and params.output_kind == "delta"
        assert request_id not in self.reqs
        self.reqs[request_id] = {"script": self._script(request_id, prompt["prompt_token_ids"], params), "pos": 0,
                                 "max": params.max_tokens}
        LOG["added"].append(request_id); return request_id
    def abort_request(self, ids, internal=False):
        for i in ids:
            r = self.reqs.pop(i, None); LOG["aborted"].append(i)
            if r is not None and i.endswith("p1"):
                REST[i.rsplit("p", 1)[0]] = r["script"][r["pos"]:]      # what the model would have said next
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
                REST[rid.rsplit("p", 1)[0]] = r["script"][r["pos"]:]
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
        assert kw["max_num_batched_tokens"] == 4096 and kw["gpu_memory_utilization"] == 0.90, kw
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
recs = [json.loads(l) for l in open(WORK / "q27x" / "records.jsonl")] if (WORK / "q27x" / "records.jsonl").exists() else []
ans_files = list((WORK / "q27x" / "answers").glob("*.txt"))
sub = json.load(open(WORK / "submission.json"))
print("RESULT", json.dumps({"scenario": SCEN, "finished": rep["tasks_finished"], "planned": rep["tasks_planned"],
                            "final_status": rep["final_status"], "verified_at_turn": rep["verified_at_turn"],
                            "sched": {k: rep["scheduler"].get(k) for k in ("admitted", "turns_started", "repair_turns", "phase2_forced",
                                      "drained", "aborted_at_hard_stop", "not_admitted", "max_active")},
                            "first_src": rep["first_attempt_source"], "max_running": LOG["max_running"],
                            "spec": rep["engine"]["speculative"], "kv": rep["engine"]["kv_cache_dtype"],
                            "engine_attempts": [(a["kv_cache_dtype"], bool(a["speculative"]), a["ok"]) for a in rep["engine"]["attempts"]]}))
by = {r["task_id"]: r for r in recs}
assert len(ans_files) >= len([r for r in recs if r["n_turns"] and r["status"] != "hard_stop"]), (len(ans_files), len(recs))
assert set(sub) == set(TEST_CH), "submission must cover every test task"
for t, task in TEST_CH.items():
    assert len(sub[t]) == len(task["test"])
assert G["CONCURRENCY"] == (max(4, min(16, 778 * 800 // G["EXPECTED_LIVE_TOKENS_PER_SEQ"])) if G["ENGINE_READY"] else 0), G["CONCURRENCY"]
if SCEN == "mixed":
    for t, beh in BEHAVIOUR.items():
        r = by[t]
        assert r["verified_turn"] == EXPECT_TURN[beh], (t, beh, r["verified_turn"], r["turns"])
        if EXPECT_TURN[beh]:
            assert r["status"] == "verified" and all(r["score"]["query_hits"]) and r["src"][0][0] == "verified_code", (t, beh, r)
            assert r["n_turns"] == EXPECT_TURN[beh], (t, beh, r["n_turns"])
        else:   # R6 v1.1: 4 failed code turns, then the direct-grid turn (fake answers the true grids)
            assert r["status"] == "unverified_grid" and r["n_turns"] == G["MAX_TURNS"] + 1, (t, r["status"], r["n_turns"])
            assert r["turns"][-1]["kind"] == "grid" and r["turns"][-1]["grids_parsed"] == len(EVAL_SOL[t]), r["turns"][-1]
            assert all(r["score"]["query_hits"]) and r["src"][0][0] == "direct_grid", (t, r["src"], r["score"])
        if beh == "forced_t1_good":
            assert r["turns"][0]["mode"] == "forced_close" and r["turns"][0]["code_found"], r["turns"][0]
    reps = [(t, k, txt) for t, k, txt in LOG["turn_prompts"] if k >= 2]
    assert reps, "no repair turn was started"
    assert set(LOG.get("grid_turns", [])) == {t for t, b in BEHAVIOUR.items() if b == "never"}, LOG.get("grid_turns")
    assert G["THINK_BUDGET"] == 3000 and G["GRID_FALLBACK"] is True
    for t, k, txt in reps:
        beh = BEHAVIOUR[t]
        assert ("Your solve() was run" in txt) or ("did not contain" in txt), (t, k, txt[-600:])
        assert " thinking thinking" not in txt, "earlier thinking must not be re-sent (preserve_thinking=False)"
        assert "Test INPUT 1" in txt and txt.rstrip().endswith("<think>"), txt[-200:]
        if beh == "crash_then_good" and k == 2:
            assert "ERROR" in txt and "IndexError" in txt, txt[-800:]
        if beh in ("t2_good", "never", "t4_good"):
            assert "# rule: identity" in txt and ("WRONG" in txt or "CORRECT" in txt), txt[-800:]
        if beh == "never" and k == G["MAX_TURNS"] + 1:
            assert "Skip the code now." in txt and "# rule: identity" in txt and "WRONG" in txt, txt[-900:]
            assert "Find what is wrong" not in txt.split("Skip the code now.")[0][-400:], "the repair request must be dropped"
        if beh == "nocode_then_good" and k == 2:
            assert "did not contain" in txt and "I think the rule is a reflection." in txt
    assert all(ok for tid, ok in LOG["phase2_prompts"] if BEHAVIOUR[tid] == "forced_t1_good"), "forced prompt must end with the stop text"
    assert LOG["max_running"] <= G["CONCURRENCY"]
    ek = LOG["engine_kwargs"]
    assert ek["max_model_len"] == 81920 and ek["tensor_parallel_size"] == 4
    if not os.environ.get("HARNESS_REFUSE_SPEC"):
        assert ek.get("kv_cache_dtype") == "fp8_e4m3" and ek["attention_config"]["backend"] == "TRITON_ATTN"
        assert rep["engine"]["cuda_link"]["probe"] == "FAIL"
    else:
        assert ek.get("kv_cache_dtype") is None and ek["attention_config"]["backend"] == "FLASH_ATTN"
    assert ek.get("speculative_config") == ({"method": "mtp", "num_speculative_tokens": 3} if not os.environ.get("HARNESS_REFUSE_SPEC") else None)
    assert G["SAMPLING"]["temperature"] == 0.6 and all(c == 3000 for c in G["P1_CAPS"].values())
    _hits = [h for r in recs for h in r["score"]["query_hits"]]
    _q = sum(len(EVAL_SOL[t]) for t in G["TASK_IDS"])
    assert rep["eval_query_hits"] == sum(_hits) and rep["eval_query_total"] == _q
    assert ek["enforce_eager"] is False and ek["language_model_only"] is True and ek["enable_prefix_caching"] is False
    print("mixed: verified at turns", rep["verified_at_turn"], "| repair prompts carry feedback + earlier code, no earlier thinking")
if SCEN == "drain":
    assert rep["projection_valid"] is False and rep["projection_hidden_end_to_end_hours"] is None, rep.get("projection_reason")
    assert rep["scheduler"]["drained"] + rep["scheduler"]["aborted_at_hard_stop"] >= 1, rep["scheduler"]
    assert rep["tasks_finished"] + rep["scheduler"]["not_admitted"] == rep["tasks_planned"] - len(rep["skipped_long_prompt"]), rep
    print("drain: projection invalid; every admitted task finalized:", rep["final_status"])
if SCEN == "allfail":
    assert rep["engine"]["fatal"] == "all_engine_attempts_failed" and rep["tasks_finished"] == 0, rep["engine"]
    assert all(a["attempt_1"] == q["input"] for t, task in TEST_CH.items() for a, q in zip(sub[t], task["test"]))
    assert len(rep["engine"]["attempts"]) == 3 and rep["projection_valid"] is False
    print("allfail: all 3 engine attempts failed; notebook completed; identity submission + report written")
if SCEN == "norelease":
    att = rep["engine"]["attempts"]
    assert len(att) == 1 and att[0]["gpu_release"]["released"] is False and rep["tasks_finished"] == 0, att
    print("norelease: no further attempt; notebook completed with identity")
if SCEN == "smokefail":
    att = rep["engine"]["attempts"]
    assert [a["ok"] for a in att] == [False, False, True] and rep["tasks_finished"] == rep["tasks_planned"], att
    print("smokefail: bf16 without MTP ran all tasks")
if SCEN == "orphan":
    att = rep["engine"]["attempts"]
    assert att[0]["ok"] is False and att[0]["killed_children"]["found"] >= 1 and att[0]["gpu_release"]["released"] is True, att[0]
    assert att[-1]["ok"] is True and all(p.poll() is not None for p in ORPHANS) and rep["tasks_finished"] == rep["tasks_planned"]
    print("orphan: child killed, GPU released, fallback engine ran all", rep["tasks_finished"], "tasks")
if SCEN == "mixed" and not os.environ.get("HARNESS_REFUSE_SPEC"):
    assert rep["mtp_acceptance_rate"] == 0.7 and rep["projection_valid"] is True, (rep.get("mtp_acceptance_rate"), rep.get("projection_valid"))
if SCEN == "crash":
    assert rep["scheduler"]["error"] and "EngineDeadError" in rep["scheduler"]["error"], rep["scheduler"].get("error")
    print("crash: notebook finished all cells; error recorded; submission valid with", rep["tasks_finished"], "finished tasks")
if SCEN == "eos":
    assert LOG.get("eos_before_stop") and not any(LOG["eos_before_stop"]), LOG.get("eos_before_stop")
    assert all(r["verified_turn"] == 1 and all(r["score"]["query_hits"]) for r in recs), [(r["task_id"], r["verified_turn"]) for r in recs]
    print("eos: EOS stripped before the stop text in", len(LOG["eos_before_stop"]), "forced closes; all verified at turn 1")
if SCEN == "rerun":
    _ver = [r["task_id"] for r in recs if r["verified_turn"]]
    for t in _ver:
        assert [a["attempt_1"] for a in sub[t]] == sols_for(t), t
    _nonid = sum(1 for t in _ver for a, q in zip(sub[t], TEST_CH[t]["test"]) if a["attempt_1"] != q["input"])
    assert _nonid > 100, _nonid
    assert rep["tasks_planned"] == len(TEST_CH) and rep["tasks_finished"] == len(TEST_CH) - len(rep["skipped_long_prompt"])
    print("rerun: verified program outputs written for", len(_ver), "tasks;", _nonid, "non-identity first attempts")
print("SCENARIO PASS", SCEN, "| work dir", WORK)
