"""Build Q27-AGENT v1: the Tufa-style REPL tool agent (python + submit, verified on the demos) from the public
madarshbb/arc-qwen-3-8-27b-agent notebook, on our Qwen3.8-27B-FP8, with our dual-mode rule:

  * commit / interactive run  -> REPL agent on the 12 EVAL12 evaluation tasks, scored, report written
  * competition rerun         -> REPL agent on every hidden test task, submission.json rewritten as attempts finish

The agent package (grids, sandbox, backend, agent, repl, client, repl_agent, run_direct, run_submit, submit_main)
is copied verbatim from the source notebook; only submit_main gets a 3-line dispatch to the new run_eval.py.

usage: python make_q27_agent.py [out.ipynb]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "LB_compare" / "madarshbb__arc-qwen-3-8-27b-agent" / "arc-qwen-3-8-27b-agent.ipynb"
if not SRC.exists():                           # audit-bundle layout: build/ next to source_notebook/
    SRC = HERE.parent / "source_notebook" / "arc-qwen-3-8-27b-agent.ipynb"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(r"D:\ARC AGI 2\Research Base Experiment\Q27-AGENT-v1.ipynb") if (HERE.parent / "LB_compare").exists()
    else HERE.parent / "notebook" / "Q27-AGENT-v1_REBUILT.ipynb")

src = json.load(open(SRC, encoding="utf-8"))
cells = ["".join(c["source"]) for c in src["cells"]]
assert len(cells) == 14, len(cells)
assert cells[8].startswith("%%writefile /kaggle/working/agent/repl_agent.py")
assert cells[11].startswith("%%writefile /kaggle/working/agent/submit_main.py")

EVAL12 = ["136b0064", "20a9e565", "21897d95", "247ef758", "2b83f449", "4c3d4a41", "4c7dc4dd", "6e4f6532",
          "88bcf3b4", "898e7135", "8b7bacbf", "b6f77b65"]   # Q27-XHIGH EVAL12 (seed 20261005): direct = 3/22 (run 7)

HEAD = r'''# Q27-AGENT v1 — Qwen3.8-27B REPL tool agent (python + submit, verified on the demos), 4×L4

**Why.** Our direct-answer pipeline (Q27-XHIGH) scored LB 18.75: the model reasons up to 49k tokens and writes the
grid. The Kaggle discussions attribute the October jump to 60+ to Qwen3.8-27B *agents*; the public
`madarshbb/arc-qwen-3-8-27b-agent` notebook contains a REPL agent modelled on the Tufa Labs duck harness (Tufa Labs is
#1 at 88). This notebook runs that agent on our model.

**How the agent works** (code copied verbatim from the source notebook, cells 2–11):
* two tools: `python(code)` explores the task (grids preloaded as numpy arrays + helpers `show`, `diff`, `objects`);
  `submit(code)` runs a `transform(grid)` program on every train input and returns which examples it reproduced plus a
  cell diff. Reproducing all of them verifies the program; its test outputs become the answer.
* many short turns instead of one 49k-token monologue; per-turn thinking cap (`thinking_token_budget`), early-submit
  nudges, forced submit after 10 python calls, reasoning-effort ladder after a cut turn, function retention, context
  trimming with hysteresis (keeps the vLLM prefix cache valid), repetition-loop guard.
* answers: verified programs vote; otherwise the best unverified programs (most train examples right).
* serving: vLLM OpenAI server from our wheelhouse + the `taaf-duck-qwen38-serving-v1` transformers overlay,
  tool parser `qwen3_coder`, reasoning parser `qwen3`, prefix caching, fp8 KV, MTP (ladder falls back to plain).

**Modes** (same notebook):
* commit / interactive run: the agent on the 12 EVAL12 evaluation tasks (the same tasks as Q27-XHIGH run 7: direct =
  3/22 queries), budget 2.5 h from the first cell; report in `/kaggle/working/agent_eval/`.
* competition rerun: every hidden test task; `submission.json` is rewritten as each attempt finishes; the agent stops
  at 11 h 35 min and the process is cut at 11 h 50 min (10 min before the 12 h limit). A valid fallback submission is
  written in the first cell, and the last cell repairs any malformed row.
'''

C1 = cells[1]                                  # fallback submission + start epoch (verbatim)

RUN_EVAL = r'''%%writefile /kaggle/working/agent/run_eval.py
"""Commit-run evaluation: the REPL agent on a fixed set of evaluation tasks, scored against the public solutions.

    python run_eval.py '<json config>'   (normally called by submit_main once the server is up)

Writes /kaggle/working/agent_eval/attempts.jsonl (one line per finished attempt, with transcript) and
eval_report.json (score, per task, tokens, turns, statuses). Never touches /kaggle/working/submission.json.
"""
import glob
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ChatClient                      # noqa: E402
from repl_agent import solve_repl, AGENT_OPTS      # noqa: E402
from agent import to_submission, score             # noqa: E402


def find(name, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def main(cfg):
    t0 = time.time()
    root = cfg.get("data_dir", "/kaggle/input")
    ch = json.load(open(find("arc-agi_evaluation_challenges.json", root)))
    sol = json.load(open(find("arc-agi_evaluation_solutions.json", root)))
    ids = [t for t in cfg["eval_task_ids"] if t in ch]
    tasks = {t: ch[t] for t in ids}
    out = Path(cfg.get("eval_out", "/kaggle/working/agent_eval"))
    out.mkdir(parents=True, exist_ok=True)
    client = ChatClient(cfg["base_url"], cfg["model"], max_tokens=cfg.get("max_tokens", 16384),
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=cfg.get("thinking", True),
                        reasoning_effort=cfg.get("reasoning_effort"),
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))
    lock = threading.Lock()
    jsonl = open(out / "attempts.jsonl", "a", encoding="utf-8")

    def on_done(rec):
        with lock:
            jsonl.write(json.dumps(rec, default=str) + "\n")
            jsonl.flush()

    print(f"eval: {len(tasks)} tasks, deadline in {(cfg['deadline_epoch'] - time.time()) / 60:.0f} min", flush=True)
    answers, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 2), concurrency=cfg.get("concurrency", 16),
        deadline=cfg["deadline_epoch"], stop_after=1, max_turns=cfg.get("max_turns", 40),
        max_gen_tokens=cfg.get("eval_cap", 60000), context_budget=cfg.get("context_budget", 28000),
        nudge_every=cfg.get("nudge_every", 0), turn_max_tokens=cfg.get("turn_max_tokens"),
        in_python=cfg.get("in_python", False), log=lambda m: print(m, flush=True), on_done=on_done,
        **{k: cfg[k] for k in AGENT_OPTS if k in cfg and k not in ("turn_max_tokens", "in_python")})
    jsonl.close()
    sub = to_submission(answers, tasks)
    total, pct, per = score(sub, {t: sol[t] for t in ids})
    q_hit = q_tot = 0
    for t in ids:
        for j, s in enumerate(sol[t]):
            q_tot += 1
            q_hit += int(s in (sub[t][j]["attempt_1"], sub[t][j]["attempt_2"]))
    atts = report["attempts"]
    gen = sum(a["gen_tokens"] for a in atts)
    wall = time.time() - t0
    summary = {
        "score_tasks": round(total, 3), "score_pct": round(pct, 2), "n_tasks": len(ids),
        "query_hits": q_hit, "queries": q_tot, "per_task": per,
        "tasks_with_verified_program": sum(1 for t in ids if any(a["status"] == "verified" and a["task_id"] == t
                                                                  for a in atts)),
        "status_counts": dict(Counter(a["status"] for a in atts)),
        "notes": dict(Counter(a["note"] for a in atts if a["note"])),
        "gen_tokens_total": gen, "gen_tokens_mean_per_attempt": round(gen / max(len(atts), 1)),
        "turns_mean": round(sum(a["turns"] for a in atts) / max(len(atts), 1), 1),
        "cuts_total": sum(a["cuts"] for a in atts), "submits_total": sum(a["submits"] for a in atts),
        "python_calls_total": sum(a["python_calls"] for a in atts),
        "wall_s": round(wall), "gen_tok_per_s": round(gen / max(wall, 1e-6), 1),
        "config": {k: v for k, v in cfg.items() if k not in ("base_url",)},
        "per_attempt": [{k: a[k] for k in ("task_id", "idx", "status", "note", "turns", "python_calls", "submits",
                                           "cuts", "forced", "gen_tokens", "prompt_tokens", "wall", "best_ok",
                                           "n_train", "effort_steps", "evictions")} for a in atts]}
    (out / "eval_report.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "eval_submission.json").write_text(json.dumps(sub))
    print(f"\nEVAL SCORE {total:.2f} / {len(ids)} tasks ({pct:.1f}%) | queries {q_hit}/{q_tot} | "
          f"verified tasks {summary['tasks_with_verified_program']} | tokens {gen} ({summary['gen_tok_per_s']} tok/s) | "
          f"wall {wall / 60:.0f} min | {summary['status_counts']}", flush=True)
    print("per task:", {t: round(v, 2) for t, v in per.items()}, flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
'''

C11 = cells[11]
_old = '''        sys.path.insert(0, HERE)
        import run_submit
        run_submit.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))'''
assert C11.count(_old) == 1
C11 = C11.replace(_old, '''        sys.path.insert(0, HERE)
        if cfg.get("eval"):                  # Q27-AGENT: commit run = evaluation subset, scored
            import run_eval
            run_eval.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))
        else:
            import run_submit
            run_submit.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))''')

SERVER = {k: v for k, v in json.loads(cells[12].split('cfg = json.loads(r"""', 1)[1].split('""")', 1)[0]).items()
          if k in ("model_glob", "overlay", "parsers", "server_ladder", "max_model_len", "max_num_seqs")}
assert SERVER["max_model_len"] == 81920 and SERVER["max_num_seqs"] == 16 and len(SERVER["server_ladder"]) == 6

AGENT_CFG = {
    "mode": "repl", "attempts": 2, "concurrency": 16,
    "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    "template_kwargs": {"reasoning_effort": "high"},
    "thinking_token_budget": 8192,      # vLLM forces </think> after this many reasoning tokens in a turn
    "max_tokens": 12288, "turn_max_tokens": 12288,
    "max_turns": 40, "context_budget": 28000, "keep_recent": 6, "trim_drain": 6000,
    "nudge_every": 6, "in_python": True, "force_submit_after": 10,
    "effort_ladder": ["medium", "low"], "retain_functions": True,
    "degenerate_guard": True, "timeout_as_cut": True, "snippet_timeout": 30.0,
    "tok_per_s": 300, "min_cap": 15000, "max_cap": 60000, "eval_cap": 60000,
}

C12 = r'''# ==================================================================== launch: evaluation (commit run) or hidden test (rerun)
import json, os, subprocess, sys, time
from pathlib import Path
START = float(Path("/kaggle/working/start_epoch.txt").read_text())
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
EVAL12 = ''' + json.dumps(EVAL12) + r'''
SERVER = json.loads(r"""''' + json.dumps(SERVER) + r'''""")
AGENT = json.loads(r"""''' + json.dumps(AGENT_CFG) + r'''""")
cfg = dict(SERVER, **AGENT)
if RERUN:
    # hidden test: the agent stops at 11 h 35 min; the process is cut at 11 h 50 min (10 min before the 12 h limit)
    cfg["deadline_epoch"] = START + 12 * 3600 - 25 * 60
else:
    EVAL_HOURS = float(os.environ.get("Q27A_EVAL_HOURS", 2.5))
    _ids = os.environ.get("Q27A_EVAL_TASKS", "eval12").strip()
    cfg.update(eval=True, deadline_epoch=START + EVAL_HOURS * 3600,
               eval_task_ids=EVAL12 if _ids == "eval12" else [t.strip() for t in _ids.split(",") if t.strip()])
print("mode:", "COMPETITION RERUN (hidden test)" if RERUN else f"EVALUATION ({len(cfg['eval_task_ids'])} tasks)",
      "| agent deadline in", round((cfg["deadline_epoch"] - time.time()) / 60), "min", flush=True)
print(json.dumps({k: v for k, v in cfg.items() if k not in ("server_ladder",)}, indent=1), flush=True)
try:
    rc = subprocess.run([sys.executable, "/kaggle/working/agent/submit_main.py", json.dumps(cfg)],
                        timeout=max(60, cfg["deadline_epoch"] + 15 * 60 - time.time())).returncode
    print("submit_main exit code", rc)
except Exception as e:
    print("submit_main did not finish cleanly:", repr(e))
if not RERUN:
    try:
        rep = json.loads(Path("/kaggle/working/agent_eval/eval_report.json").read_text())
        print({k: rep[k] for k in ("score_tasks", "score_pct", "query_hits", "queries", "tasks_with_verified_program",
                                   "status_counts", "gen_tokens_total", "gen_tok_per_s", "wall_s")})
    except Exception as e:
        print("no eval report:", repr(e))
'''

C13 = cells[13]                                # last-line-of-defence validation (verbatim)

out_cells = [("markdown", HEAD), ("code", C1)] + [("code", c) for c in cells[2:11]] + \
            [("code", C11), ("code", RUN_EVAL), ("code", C12), ("code", C13)]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in out_cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
text = json.dumps(nb, indent=1, ensure_ascii=False)
DST.write_text(text, encoding="utf-8")
print("wrote", DST, hashlib.sha256(text.encode()).hexdigest())
