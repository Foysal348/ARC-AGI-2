"""Experiment 7 v3: portable CPU test suite (Claude, 2026-10-12). Only bundled files are used.

usage: python tests/test_e7_v3.py [notebook_dir]      (default: the 5 notebooks two levels above Build/E7B_build)
Covers the two final GPT audits:
  T1 source freeze + per-variant defaults + hidden guard text      T6 attempt-wise final submission + audit in report
  T2 exact Codex donor function                                    T7 integrated ADAPTIVE scheduler: 16 -> 17 -> 18
  T3 real vLLM 0.19 metrics parse (31 snapshots, labelled)         T8 integrated adaptive + preemption freeze (max 17)
  T4 shadow: ONE decision per telemetry poll                       T9 integrated B1 shadow (cap 20): never > 16 active,
  T5 start_phase2 transactional + success path == v1.4.4              shadow counted once per poll
                                                                   T10 scheduler error -> not_admitted reported
"""
import collections
import copy
import json
import math
import pathlib
import sys
import tempfile
import time
import traceback
import types

HERE = pathlib.Path(__file__).resolve().parent
FX = HERE / "fixtures"
NB_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent.parent
VARIANTS = {"COMBINED": ("PROXY48", 24, "adaptive", "1")}
BASE = json.loads((FX / "Q27-XHIGH-v1_4_4.ipynb").read_text(encoding="utf-8"))
NBS = {v: json.loads((NB_DIR / f"Q27-XHIGH-v1_4_4_E7_{v}.ipynb").read_text(encoding="utf-8")) for v in VARIANTS}
PASSED = []


def code(nb, i):
    return "".join(nb["cells"][i]["source"])


def ok(name):
    PASSED.append(name)
    print("PASS", name, flush=True)


# ------------------------------------------------------------------ T1
for v, (prof, cap, kv, order) in VARIANTS.items():
    nb = NBS[v]
    assert len(nb["cells"]) == 9
    for i in (1, 3, 5):
        assert code(nb, i) == code(BASE, i), (v, i)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile(code(nb, i), f"{v}_cell{i}", "exec")
            assert not c.get("outputs")
    c2 = code(nb, 2)
    assert f'os.environ.get("Q27X_PROFILE", "{prof}")' in c2
    assert f'os.environ.get("Q27X_MAX_NUM_SEQS", {cap}))' in c2
    assert f'os.environ.get("Q27X_KV_MODE", "{kv}")' in c2
    assert f'os.environ.get("Q27X_ORDERING", "{order}")' in c2
    assert 'Q27X_ORDERING, Q27X_KV_MODE, MAX_NUM_SEQS = False, "off", 16' in c2 and "E7_HIDDEN_GUARD_APPLIED" in c2
    assert c2.index("E7_HIDDEN_GUARD_APPLIED = bool(") < c2.index("CONFIG = {k: globals()[k]")
    assert '"PROXY48":' in c2
    assert 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 49152))' in c2
    assert 'ANSWER_MAX_TOKENS = int(os.environ.get("Q27X_ANSWER_MAX", 8192))' in c2
manifest = json.loads((FX / "proxy48_manifest.json").read_text())
assert manifest["n_tasks"] == 48 and manifest["n_test_outputs"] == 56 and manifest["all_in_eval_file"]
assert sorted(t["id"] for t in manifest["tasks"]) == sorted(json.loads(
    code(NBS["COMBINED"], 2).split('"task_list": ')[1].split("]")[0] + "]"))
ok("T1 source freeze, COMBINED defaults (PROXY48, cap 24, adaptive, ordering on), hidden guard before CONFIG, PROXY48 manifest 48/56")

# ------------------------------------------------------------------ T2
donor = json.loads((FX / "codex_donor.ipynb").read_text(encoding="utf-8"))
s = "".join(donor["cells"][6]["source"])
d_fn = s[s.index("def get_puzzle_priority"):s.index("\ndef local_worker")]
for v in VARIANTS:
    s4 = code(NBS[v], 4)
    assert s4[s4.index("def get_puzzle_priority"):s4.index("\n# Sort on")].strip() == d_fn.strip(), v
ok("T2 exact Codex get_puzzle_priority in every variant")

# ------------------------------------------------------------------ controller extraction (from the notebook itself)
S7 = code(NBS["COMBINED"], 7)
for v in VARIANTS:
    assert code(NBS[v], 7) == S7, "cell 7 must be identical across variants"
ctl_src = S7[S7.index("E7_REQUIRED_METRICS ="):S7.index("\n_E7_PREFERRED_RECIPE =")]
CNS = {"math": math, "deque": collections.deque, "Counter": collections.Counter, "NOTEBOOK_T0": 0.0,
       "Q27X_KV_EXTRA_THRESHOLD": 0.85, "Q27X_KV_PROJECTED_CEILING": 0.90, "Q27X_KV_EMERGENCY": 0.93,
       "THINK_BUDGET": 49152, "ANSWER_MAX_TOKENS": 8192, "STOP_IDS": [7, 8, 9]}
exec(ctl_src, CNS)
REQ = CNS["E7_REQUIRED_METRICS"]


def sample(kv=0.30, run=16, waiting=0, preempt=0):
    return [{"name": n, "value": x, "labels": {"engine": "0", "model_name": "Q27"}}
            for n, x in zip(REQ.values(), (kv, run, waiting, preempt))]


# ------------------------------------------------------------------ T3
real = json.loads((FX / "real_vllm_metrics_dual_v2.json").read_text())
parsed = [CNS["e7_parse_metrics"](r["metrics"]) for r in real]
assert sum(p is not None for p in parsed) >= 30, sum(p is not None for p in parsed)
assert all(0 <= p["kv"] <= 1 for p in parsed if p)
ok(f"T3 real vLLM 0.19 labelled metrics parse ({sum(p is not None for p in parsed)}/{len(parsed)} snapshots)")

# ------------------------------------------------------------------ T4
C = CNS["E7KVAdmission"]
c = C(mode="shadow", baseline_cap=16, engine_cap=24, extra_cap=24, poll_sec=12, cooldown_sec=120, kv_tokens=1353600)
c.set_load(16, 16 * 5200, 16 * 5600, 5600)        # short units: the lifetime projection allows an extra
now = 1000.0
for _ in range(50):     # no poll yet: nothing counted
    assert c.decide(now=now, active_count=16, deadline=now + 999, next_prompt_tokens=5000) == (False, "shadow_only")
assert sum(v for k, v in c.stats.items() if k.startswith("shadow_")) == 0
c.observe(sample(), now - 36, 0.01)        # v5: the growth gate needs a >= 30 s telemetry window
c.observe(sample(), now - 24, 0.01)
c.observe(sample(), now, 0.01)
for _ in range(1000):
    assert c.decide(now=now + 1, active_count=16, deadline=now + 999, next_prompt_tokens=5000) == (False, "shadow_only")
assert c.stats["shadow_would_admit"] == 1, c.stats
c.observe(sample(), now + 12, 0.01)
for _ in range(1000):
    c.decide(now=now + 13, active_count=16, deadline=now + 999, next_prompt_tokens=5000)
assert c.stats["shadow_would_admit"] == 2, c.stats
c.observe([{"name": "bad"}], now + 24, 0.01)            # invalid snapshot -> one rejection only
for _ in range(500):
    c.decide(now=now + 25, active_count=16, deadline=now + 999, next_prompt_tokens=5000)
assert c.stats["shadow_reject_missing_telemetry"] == 1, c.stats
ok("T4 shadow counts one decision per telemetry poll (1000 calls -> 1; invalid snapshot rejected once)")

# ------------------------------------------------------------------ T5
B7 = code(BASE, 7)


def p2_fn(src):
    a = src.index("def start_phase2(")
    return src[a:src.index("\ndef answer_text(", a)]


def p2_ns(src, engine):
    ns = {"THINK_CLOSE_ID": 9, "EOS_IDS": {2}, "STOP_IDS": [7, 8, 9], "MAX_MODEL_LEN": 81920, "SEED": 20261005,
          "UNIT_INDEX": {"u#0": 3}, "SCHED": {"phase2_continue": 0, "phase2_forced": 0}, "engine": engine,
          "_req": lambda uid, p: f"{uid}|{p}", "make_params": lambda cap, seed: (cap, seed), "time": time,
          "E7_GATE": types.SimpleNamespace(transition=lambda t: None)}
    exec(p2_fn(src), ns)
    return ns


class Rec:
    def __init__(self):
        self.calls, self.aborted = [], []

    def add_request(self, rid, prompt, params):
        self.calls.append((rid, list(prompt["prompt_token_ids"]), params))

    def abort_request(self, ids):
        self.aborted += ids


class Bad(Rec):
    def add_request(self, rid, prompt, params):
        raise RuntimeError("injected add_request failure")


for p1 in ([5, 6, 2, 2], [5, 9, 4, 4]):                  # forced close with trailing EOS / closed + cut answer
    st0 = {"uid": "u#0", "phase": 1, "prompt_ids": [1, 1], "p1_ids": list(p1), "p2_ids": [], "p2_prefix": [], "mode": None}
    stb, ste = copy.deepcopy(st0), copy.deepcopy(st0)
    eb, ee = Rec(), Rec()
    nb_, ne = p2_ns(B7, eb), p2_ns(S7, ee)
    assert nb_["start_phase2"](stb, 8192) is True and ne["start_phase2"](ste, 8192) is True
    assert eb.calls == ee.calls and stb == ste and nb_["SCHED"] == ne["SCHED"], (eb.calls, ee.calls, stb, ste)
    stx = copy.deepcopy(st0)
    nx = p2_ns(S7, Bad())
    try:
        nx["start_phase2"](stx, 8192)
        raise AssertionError("failure not propagated")
    except RuntimeError as e:
        assert "injected" in str(e)
    assert stx == st0 and nx["SCHED"] == {"phase2_continue": 0, "phase2_forced": 0}, (stx, nx["SCHED"])
    assert nx["engine"].aborted == ["u#0|2"]
st0 = {"uid": "u#0", "phase": 1, "prompt_ids": [1] * 81900, "p1_ids": [5, 2], "p2_ids": [], "p2_prefix": [], "mode": None}
stb, ste = copy.deepcopy(st0), copy.deepcopy(st0)
assert p2_ns(B7, Rec())["start_phase2"](stb, 8192) is False and p2_ns(S7, Rec())["start_phase2"](ste, 8192) is False
assert stb == ste
ok("T5 start_phase2: success path identical to v1.4.4 (prefix, seed, cap, state, counters); add_request failure leaves state/counters untouched; no-room path identical")

# ------------------------------------------------------------------ T6
S8 = code(NBS["COMBINED"], 8)
assert S8.index("SUBMISSION_AUDIT = ensure_final_submission()") < S8.index('report["submission_audit"] = SUBMISSION_AUDIT') \
    < S8.index('write_json_atomic(OUT / "run_report.json", report)') < S8.index('ZIP = WORKING / "Q27X_BUNDLE.zip"')
fn_src = S8[S8.index("def ensure_final_submission():"):S8.index("\ntry:\n    SUBMISSION_AUDIT")]
cell4 = code(BASE, 5)
vg_src = cell4[cell4.index("def valid_grid("):cell4.index("\n_FENCE_RE")]
with tempfile.TemporaryDirectory() as td:
    sp = pathlib.Path(td) / "submission.json"
    ch = {"t1": {"test": [{"input": [[0]]}, {"input": [[1]]}]}, "t2": {"test": [{"input": [[2]]}]}}
    ns = {"json": json, "SUBMISSION_PATH": sp, "test_challenges": ch,
          "_load_json": lambda p: json.loads(pathlib.Path(p).read_text()),
          "write_json_atomic": lambda p, o: pathlib.Path(p).write_text(json.dumps(o))}
    exec(vg_src, ns)
    exec(fn_src, ns)
    sp.write_text(json.dumps({"t1": [{"attempt_1": [[7]], "attempt_2": [[99]]}, {"attempt_1": "x", "attempt_2": [[3]]}],
                              "zz": []}))
    audit = ns["ensure_final_submission"]()
    out = json.loads(sp.read_text())
    assert out["t1"][0] == {"attempt_1": [[7]], "attempt_2": [[0]]}, out          # valid attempt_1 kept
    assert out["t1"][1] == {"attempt_1": [[1]], "attempt_2": [[3]]}, out          # valid attempt_2 kept
    assert out["t2"] == [{"attempt_1": [[2]], "attempt_2": [[2]]}]                 # missing task -> fallback
    assert set(out) == set(ch) and audit["repaired_attempts"] == 4 and audit["unknown_keys_dropped"] == 1, audit
    assert ns["ensure_final_submission"]()["repaired_attempts"] == 0              # idempotent
    sp.write_text("not json")
    assert ns["ensure_final_submission"]()["tasks"] == 2
ok("T6 attempt-wise submission repair (valid attempt kept), runs before run_report/ZIP, idempotent, audit in report")


# ------------------------------------------------------------------ integrated scheduler harness (cell 7, real code)
class T:
    def encode(self, x, add_special_tokens=False):
        return [1]

    def decode(self, toks, skip_special_tokens=False):
        return "1"


class TimedEngine:
    """Requests think for `life` seconds (one token per step), then close naturally (token 555)."""

    def __init__(self, life, fail_after_steps=None):
        self.req, self.seen, self.life, self.steps, self.fail = {}, [], life, 0, fail_after_steps
        self.max_running = 0

    def add_request(self, rid, prompt, params):
        assert rid not in self.req, "duplicate " + rid
        self.req[rid] = time.perf_counter()
        self.seen.append(rid)
        self.max_running = max(self.max_running, len(self.req))

    def abort_request(self, ids):
        for rid in ids:
            self.req.pop(rid, None)

    def step(self):
        self.steps += 1
        if self.fail and self.steps >= self.fail:
            raise RuntimeError("injected engine failure")
        time.sleep(0.01)
        out = []
        for rid, t0 in list(self.req.items()):
            done = time.perf_counter() - t0 >= self.life
            if done:
                self.req.pop(rid)
            out.append(types.SimpleNamespace(request_id=rid, finished=done, outputs=[types.SimpleNamespace(
                token_ids=[555, 101] if done else [42], finish_reason="stop" if done else None)]))
        return out


def run_sched(mode, engine_cap, n_units, life, poll=1.0, preempt_after_extra=False, fail_after=None,
              extra_cap=24, cooldown=120.0, deadline_sec=None, preempt_each_extra=False, records_broken=False):
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "q27x"
        out.mkdir()
        if records_broken:                      # records.jsonl cannot be opened for append
            (out / "records.jsonl").mkdir()
        units = [f"t{i:02d}#0" for i in range(n_units)]
        ch = {u.split("#")[0]: {"train": [], "test": [{"input": [[0]]}]} for u in units}
        eng = TimedEngine(life, fail_after)
        state = {"extra_seen": False, "extras": 0}

        def metrics():
            pre = 1 if (preempt_after_extra and state["extra_seen"]) else 0
            if preempt_each_extra:
                pre = state["extras"]
            return sample(kv=0.30, run=len(eng.req), waiting=0, preempt=pre)
        ns = {
            "OUT": out, "time": time, "json": json, "math": math, "Counter": collections.Counter,
            "deque": collections.deque, "NOTEBOOK_T0": time.perf_counter(), "RERUN": True,
            "SUBMISSION_PATH": pathlib.Path(td) / "submission.json",
            "HARD_STOP": time.perf_counter() + (deadline_sec or 600), "DRAIN_DEADLINE": time.perf_counter() + (deadline_sec or 600) - 1,
            "ADMIT_DEADLINE": time.perf_counter() + (deadline_sec or 600) - 2, "ENGINE_READY": True,
            "ENGINE_INFO": {"actual_max_num_seqs": engine_cap, "kv_tokens": 1353600, "selected_recipe_index": 0,
                            "kv_cache_dtype": "fp8_e4m3", "speculative": {"method": "mtp", "num_speculative_tokens": 3},
                            "prefix_caching": True, "prefix_caching_resolved": True, "attention_backend": "FLASHINFER",
                            "attention_backend_resolved": "AttentionBackendEnum.FLASHINFER"},
            "MAX_NUM_SEQS": engine_cap, "CONCURRENCY": 16, "Q27X_KV_MODE": mode, "Q27X_KV_EXTRA_CAP": extra_cap,
            "Q27X_KV_POLL_SEC": poll, "Q27X_KV_COOLDOWN_SEC": cooldown, "Q27X_KV_EXTRA_THRESHOLD": 0.85,
            "Q27X_KV_PROJECTED_CEILING": 0.90, "Q27X_KV_EMERGENCY": 0.93, "THINK_BUDGET": 49152, "KV_CACHE_DTYPE": "fp8_e4m3",
            "SPECULATIVE_CONFIG": {"method": "mtp", "num_speculative_tokens": 3}, "ENABLE_PREFIX_CACHING": True,
            "GEN_TOKENS_TOTAL": 0, "UNITS": units, "PROMPT_IDS": {u: [1] * 5000 for u in units},
            "P1_CAPS": {u: 49152 for u in units}, "UNIT_TASK": {u: u.split("#")[0] for u in units},
            "UNIT_Q": {u: 0 for u in units}, "UNIT_INDEX": {u: i for i, u in enumerate(units)}, "CHALLENGES": ch,
            "TASK_IDS": list(ch), "test_challenges": ch, "SEED": 20261005, "THINK_CLOSE_ID": 555, "EOS_IDS": {2},
            "STOP_IDS": [555], "MAX_MODEL_LEN": 81920, "ANSWER_MAX_TOKENS": 8192, "DRAIN_ANSWER_MAX_TOKENS": 8192,
            "LOG_EVERY_SEC": 1e9, "tokenizer": T(), "engine": eng, "make_params": lambda cap, seed: (cap, seed),
            "parse_grid": lambda text: [[1]],
            "make_attempts": lambda uid, g: {"attempt_1": g, "attempt_2": [[0]], "src_1": "model", "src_2": "test_input"},
            "score_unit": lambda uid, att: None, "write_json_atomic": lambda p, o: pathlib.Path(p).write_text(json.dumps(o)),
            "identity_submission": lambda ch_: {t: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
                                                for t, task in ch_.items()},
            "metrics_snapshot": metrics, "ORDER_ADMIT_RANK": {u: i for i, u in enumerate(units)},
            "CODEX_PRIORITY_TIERS": {u.split("#")[0]: 1 for u in units}, "CODEX_ORDERED_TASKS": list(ch),
            "Q27X_ORDERING": False, "schedule_units": lambda: units, "traceback": traceback, "E7_HIDDEN_APPROVED": True,
        }
        # test-only instrumentation: mark when the first extra admission happened (preemption scenario)
        ns["__mark_state__"] = state
        src = S7.replace("            E7_GATE.admitted(_uid, time.perf_counter(), _reason, len(active))\n",
                         "            E7_GATE.admitted(_uid, time.perf_counter(), _reason, len(active))\n"
                         "            if _reason == 'adaptive_extra': __mark_state__['extra_seen'] = True; "
                         "__mark_state__['extras'] += 1\n")
        assert src != S7
        # time-scaled test: the 30 s growth window becomes 1 s (polls every 0.2-0.3 s)
        src = src.replace("E7_GROWTH_MIN_SPAN_SEC = 30.0\n", "E7_GROWTH_MIN_SPAN_SEC = 1.0\n")
        assert "E7_GROWTH_MIN_SPAN_SEC = 1.0\n" in src
        exec(src, ns)
        return ns, eng


# ------------------------------------------------------------------ T7a long units: projection keeps it at 16
ns, eng = run_sched("adaptive", 24, 40, life=1e6, poll=0.3, deadline_sec=6)
g = ns["E7_GATE"]
assert g.mode == "adaptive", g.mode
assert eng.max_running == 16 and ns["SCHED"]["max_active"] == 16, (eng.max_running, ns["SCHED"]["max_active"])
assert g.stats.get("block_kv_lifetime_projection", 0) > 0, dict(g.stats)
assert not [e for e in g.events if e.get("reason") == "adaptive_extra"]
_polls = g.stats.get("metrics_ok", 0) + g.stats.get("metrics_invalid", 0)
assert sum(v for k, v in g.stats.items() if k.startswith("block_")) <= _polls + 3, (dict(g.stats), _polls)   # v5 dedupe
assert not [e for e in g.events if e.get("event") == "metrics_ok"]                                           # v5 events
ok("T7a integrated adaptive, long units (no finished length known -> worst case 49,152+8,192): projection blocks extras, stays at 16")

# ------------------------------------------------------------------ T7b short units: KV decides, ramps up to the user cap 24
ns, eng = run_sched("adaptive", 24, 140, life=4.0, poll=0.2)
g = ns["E7_GATE"]
assert ns["SCHED"]["error"] is None, ns["SCHED"]["error"]
extras = [e for e in g.events if e.get("event") == "admit" and e.get("reason") == "adaptive_extra"]
assert eng.max_running == 24 and ns["SCHED"]["max_active"] == 24, (eng.max_running, ns["SCHED"]["max_active"])
assert len(extras) >= 8 and len(eng.seen) == len(set(eng.seen)) == 140 and len(ns["FINAL"]) == 140
assert g.final_lens and g.last_projection is not None and g.last_projection < 0.90
ok(f"T7b integrated adaptive, short units: learned finals -> {len(extras)} KV-authorised extras, max_active 24 (= user cap), 140/140 finished, no duplicates")

# ------------------------------------------------------------------ T8 preemption: cooldown + tighter target, then resume
ns, eng = run_sched("adaptive", 24, 140, life=4.0, poll=0.2, preempt_after_extra=True, cooldown=2.0)
g = ns["E7_GATE"]
pre_t = [e["t"] for e in g.events if e.get("event") == "preemption_or_counter_reset"]
extras = [e for e in g.events if e.get("event") == "admit" and e.get("reason") == "adaptive_extra"]
assert pre_t and abs(g.penalty - 0.04) < 1e-9 and not g.frozen_expansion, (pre_t, g.penalty)
assert [e for e in extras if e["t"] > pre_t[0] + 2.0], "expansion must resume after the cooldown"
assert len(ns["FINAL"]) == 140 and ns["SCHED"]["error"] is None
ok(f"T8 preemption -> cooldown + target tightened by 0.04 (no permanent freeze); expansion resumed ({len(extras)} extras), 140/140 finished")

# ------------------------------------------------------------------ T9 shadow mode (cap 24): never admits extras, counts once per poll
ns, eng = run_sched("shadow", 24, 40, life=4.0, poll=0.2)
g = ns["E7_GATE"]
shadow = sum(v for k, v in g.stats.items() if k.startswith("shadow_"))
polls = g.stats.get("metrics_ok", 0) + g.stats.get("metrics_invalid", 0)
loops = ns["SCHED"]["steps"]
assert eng.max_running == 16 and ns["SCHED"]["max_active"] == 16
assert 1 <= shadow <= polls < loops, (dict(g.stats), polls, loops)
assert len(ns["FINAL"]) == 40
ok(f"T9 integrated shadow (cap 24): max 16 active, shadow decisions {shadow} <= polls {polls} << loop steps {loops}")

# ------------------------------------------------------------------ T10 not_admitted on scheduler error
ns, eng = run_sched("off", 16, 30, life=50.0, fail_after=40)
assert ns["SCHED"]["error"] and "injected engine failure" in ns["SCHED"]["error"]
assert ns["SCHED"]["not_admitted"] == 30 - ns["SCHED"]["admitted"] == 14, (ns["SCHED"]["not_admitted"], ns["SCHED"]["admitted"])
ok("T10 scheduler exception -> not_admitted = planned - admitted (14)")

# ------------------------------------------------------------------ v5 (two audits of v4, 2026-10-13)
# T11 length model: natural closes only, p90 after >= 8 per tier; forced / cut = full budget
c = C(mode="adaptive", baseline_cap=16, engine_cap=24, extra_cap=24, poll_sec=12, cooldown_sec=120, kv_tokens=1353600)
WORST = 49152 + 8192 + 3
for i in range(7):
    c.note_finished(1, 5000, 5000 + 1000 + i, natural=True)
assert c.expected_final(1, 4000) == 4000 + WORST                      # 7 < 8 -> worst case
c.note_finished(1, 5000, 6100, natural=True)
assert c.expected_final(1, 4000) == 4000 + 1100, c.expected_final(1, 4000)   # 8 natural -> p90 = 8th of 8
for i in range(10):
    c.note_finished(1, 5000, 5000 + 1000, natural=True)
assert c.expected_final(1, 4000) == 4000 + 1006, c.expected_final(1, 4000)   # 18 samples: p90 = 17th smallest
for i in range(3):
    c.note_finished(1, 5000, 5100, natural=False)                     # forced / cut / drained -> full budget
assert c.expected_final(1, 4000) == 4000 + WORST, c.expected_final(1, 4000)   # 21 samples: p90 = 19th -> worst
c.note_finished(2, 5000, 5050, natural=False)
assert c.final_lens[2] == [WORST], c.final_lens[2]
assert 'natural=(st["mode"] is None and st["end_reason"] == "natural")' in code(NBS["COMBINED"], 7)
ok("T11 length model: generated length from natural closes only, p90 after >= 8 per tier, forced/cut = full budget")

# T12 fixed mode no longer bypasses the projection; T13 rapid-growth gate; window too short -> no extra
for mode in ("fixed", "adaptive"):
    c = C(mode=mode, baseline_cap=16, engine_cap=24, extra_cap=24, poll_sec=12, cooldown_sec=120, kv_tokens=1353600)
    c.set_load(16, 16 * 30000, 16 * (5000 + WORST), 5000 + WORST)     # long units -> projection > 0.90
    c.observe(sample(kv=0.60), 1000.0, 0.01)
    assert c.extra_gate(now=1001.0, active_count=16, next_prompt_tokens=5000) == (False, "growth_window_short")
    c.observe(sample(kv=0.60), 1036.0, 0.01)
    assert c.extra_gate(now=1037.0, active_count=16, next_prompt_tokens=5000) == (False, "kv_lifetime_projection"), mode
c = C(mode="adaptive", baseline_cap=16, engine_cap=24, extra_cap=24, poll_sec=12, cooldown_sec=120, kv_tokens=1353600)
c.set_load(16, 16 * 5200, 16 * 5600, 5600)                            # short units: projection would allow
c.observe(sample(kv=0.40), 1000.0, 0.01)
c.observe(sample(kv=0.50), 1060.0, 0.01)                              # 0.1 / min -> +0.5 in 5 min
assert c.extra_gate(now=1061.0, active_count=16, next_prompt_tokens=5000) == (False, "kv_rapid_growth")
c.history.clear()
c.observe(sample(kv=0.50), 1400.0, 0.01)
c.observe(sample(kv=0.51), 1440.0, 0.01)                              # 0.015 / min -> +0.075 in 5 min
assert c.extra_gate(now=1441.0, active_count=16, next_prompt_tokens=5000) == (True, "adaptive_extra")
ok("T12 fixed mode uses the lifetime projection; T13 rapid KV growth blocks extras, slow growth allows them")

# T14 circuit breaker: 2 preemptions within 30 min -> extras off for good, baseline 16 continues, run completes
ns, eng = run_sched("adaptive", 24, 140, life=4.0, poll=0.2, preempt_each_extra=True, cooldown=0.5)
g = ns["E7_GATE"]
br = [e for e in g.events if e.get("event") == "circuit_breaker_open"]
extras = [e for e in g.events if e.get("event") == "admit" and e.get("reason") == "adaptive_extra"]
assert g.frozen_expansion and br and len(extras) == 2, (g.frozen_expansion, br, len(extras))
assert not [e for e in extras if e["t"] > br[0]["t"]]
assert len(ns["FINAL"]) == 140 and ns["SCHED"]["error"] is None and ns["SCHED"]["max_active"] <= 18
ok(f"T14 circuit breaker after 2 preemptions: extras stop ({len(extras)} before), 140/140 finished, no error")

# T15 records.jsonl I/O failure does not stop the scheduler
ns, eng = run_sched("adaptive", 24, 30, life=1.0, poll=0.2, records_broken=True)
assert ns["SCHED"]["error"] is None and len(ns["FINAL"]) == 30 and ns["SCHED"]["records_io_errors"] == 30
assert len(ns["RECORDS"]) == 30
ok("T15 records.jsonl write failure: counted, scheduler continues, 30/30 finished")

# T16 validator failure rebuilds from the in-memory answers (never identity over valid answers)
fb_src = S8[S8.index("def _submission_fallback():"):S8.index("\ntry:\n    SUBMISSION_AUDIT")]
with tempfile.TemporaryDirectory() as td:
    sp = pathlib.Path(td) / "submission.json"
    good = {"t1": [{"attempt_1": [[7]], "attempt_2": [[8]]}]}

    def _broken():
        raise OSError("disk")
    ns = {"traceback": traceback, "SUBMISSION_PATH": sp, "test_challenges": {"t1": {"test": [{"input": [[0]]}]}},
          "write_submission": lambda: sp.write_text(json.dumps(good)),
          "identity_submission": lambda ch_: {"t1": [{"attempt_1": [[0]], "attempt_2": [[0]]}]},
          "write_json_atomic": lambda p, o: pathlib.Path(p).write_text(json.dumps(o))}
    exec(fb_src, ns)
    assert ns["_submission_fallback"]() == "rebuilt_from_final" and json.loads(sp.read_text()) == good
    ns["write_submission"] = _broken
    assert ns["_submission_fallback"]() == "identity_written" and json.loads(sp.read_text())["t1"][0]["attempt_1"] == [[0]]
assert 'SUBMISSION_AUDIT = {"error": "validation_failed", "fallback": _submission_fallback()}' in S8
ok("T16 validator error -> submission rebuilt from FINAL (identity only if that also fails)")

print(f"\nALL {len(PASSED)} E7 v5 TESTS PASS")
