"""Deterministic, source-controlled builder. Base notebook is provided reference.
Run: python build_notebook.py
"""
from pathlib import Path
import json
import hashlib
import nbformat

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'reference'/'AGENT_Q27-AGENT-v1_8_3_as_run.ipynb'
# v1.9.1 (Claude, after validating GPT's v1.9.0): written next to the other Experiment 5 notebooks
OUT = ROOT.parent.parent/'Q27_AGENT_v1_9_1_SYNTH_EVAL12.ipynb'
base = nbformat.read(BASE,as_version=4)

base.cells[0].source = '''# Q27-AGENT v1.9.1 — Long-Think → Program → Verify (Phase 1+2 only)

**Purpose:** real-GPU microprobe of Qwen3.8-27B-FP8 synthesizing standalone ARC `transform(grid)` programs.

- Inherits v1.8.3 Kaggle offline wheelhouse, 4× L4, MTP3, FP8 KV, prefix-cache, startup ladder and full-schema fallback submission.
- Phase 1: one no-tools chat completion / task, **xhigh / thinking_token_budget = 49,152 / max_tokens = 57,344**. No effort ladder, no forced named tool.
- Phase 2: standalone Python AST check + sandbox verify on every train/test input. **0 repair turns.**
- EVAL uses **recorded DIRECT v1.4.4 Run 7** (LB 26.94), not live DIRECT. Only EVAL challenge files are inputs to the model; solutions are read for reporting after inference.
- Competition rerun remains **DIRECT-only** until results demonstrate the new AGENT has incremental value. Notebook writes a valid fallback immediately.
- **EVAL12 by default** (`Q27A_EVAL_TASKS=eval12`, eval deadline 2 h); `Q27A_EVAL_TASKS=136b0064,247ef758,2b83f449` for a 3-task probe.
- v1.9.1 fixes over GPT's v1.9.0: official score = Kaggle micro metric (outputs), timeout never retried / retry bounded by the deadline, thinking tokens measured per request (`think_hit_budget`), hidden-path DIRECT can never get < 49,152 thinking (`rescue_min_think` = 49,152 + `check_config` gate), a DRYCHECK cell (`dryrun_report.json`), launcher clean-up, pyflakes clean.
- `Q27A_DRYRUN=1` activates fake-server CPU plumbing checks; fake-server verification DOES NOT imply Qwen accuracy.
- Never lower the synthesis thinking budget silently. If 49,152+8,192 won't fit context, skip with an explicit `context_too_large` status.

**Safety and validity:** executing model-generated Python involves security risks. The inherited Python audit hook is not a full OS security boundary; keep solution labels isolated from inference where possible. No guarantee of bug-free GPU behavior until the Kaggle real-server probe is run.

**Expected outputs:** `/kaggle/working/agent_eval/eval_report.json`, `attempts.jsonl`, `progress.json`, `eval_submission.json`, `server_metrics.json` and `/kaggle/working/agent_server_report.json`.

**Research reference:** [gutfeeling/arc-agi-2-submission](https://github.com/gutfeeling/arc-agi-2-submission) (architecture inspiration, different model/backend). vLLM documentation: [reasoning budget](https://docs.vllm.ai/en/v0.19.1/features/reasoning_outputs/).
'''

# Replace the inherited fake-server emulator and telemetry parser with their
# v1.9 fixes; otherwise the whole-notebook CPU run would test stale files.
base.cells[14].source = '%%writefile /kaggle/working/agent/fake_vllm.py\n' + (ROOT/'modules'/'fake_vllm.py').read_text()
base.cells[12].source = '%%writefile /kaggle/working/agent/rescue.py\n' + (ROOT/'modules'/'rescue.py').read_text()
base.cells[10].source = '%%writefile /kaggle/working/agent/run_submit.py\n' + (ROOT/'modules'/'run_submit.py').read_text()
sources = ['synth.py','run_eval_v19.py','check_config.py','submit_main_v19.py']
extra=[]
for name in sources:
    contents=(ROOT/'modules'/name).read_text()
    cell=nbformat.v4.new_code_cell(f'%%writefile /kaggle/working/agent/{name}\n'+contents)
    extra.append(cell)

launch = base.cells[16].source
start = launch.index('AGENT = json.loads(r"""')
end = launch.index('\ncfg = dict(SERVER, **AGENT)', start)
conf = {
 'agent_mode':'synth', 'concurrency':16, 'temperature':0.6, 'top_p':0.95,
 'top_k':20, 'template_kwargs':{'reasoning_effort':'xhigh'},'thinking':True,
 'synth_think':49152, 'synth_answer':8192,'repair_turns':0,'synth_commit':True,
 'max_model_len':81920, 'max_num_batched_tokens':4096,
 'request_margin_s':10,'commit_min_remaining_s':120,
 'sandbox_timeout_s':10,'metrics_every_s':30,
 'eval_direct':'ref','eval_direct_run':'run7','rescue':True,
 'rescue_all':True,'rescue_effort':'xhigh','rescue_think':49152,
 'rescue_answer':8192,'rescue_commit_tokens':8192,'rescue_temperature':0.6,
 'rescue_min_start_s':300,'rescue_concurrency':16,'rescue_stream_tok_s':15.0,
 'rescue_answer_reserve_tokens':4096,'rescue_overhead_s':120,
 'rescue_min_think':49152,'rescue_kv_throttle':0.95,
 'rescue_finish_extra_s':480,
}
launch = launch[:start] + 'AGENT = json.loads(r"""'+json.dumps(conf)+'""")' + launch[end:]
# v1.9.1: EVAL12 stays the default (user decision: EVAL12 directly, not the 3-task probe)
# Replace old agent-specific smoke/timeout defaults with v1.9 budget-safe gates.
launch = launch.replace('cfg["e2e_max_s"] = 300', 'cfg["request_timeout_s"] = 7200')
launch = launch.replace('cfg["e2e_extend_s"] = 600', 'cfg["min_agent_s"] = 1200')
launch = launch.replace('cfg.update(eval=True, deadline_epoch=START + EVAL_HOURS * 3600, min_agent_s=1800,',
                        'cfg.update(eval=True, deadline_epoch=START + EVAL_HOURS * 3600, min_agent_s=600,')
launch = launch.replace('cfg.update(dryrun=True, deadline_epoch=time.time() + 20 * 60, min_agent_s=60, e2e_reserve_s=60, e2e_max_s=120,\n               e2e_min_s=10, eval_direct="live", rescue_all=True, rescue_min_start_s=5, install_timeout_s=60)',
                        'cfg.update(dryrun=True, deadline_epoch=time.time() + 12 * 60, min_agent_s=30,\n               eval_direct="ref", rescue_all=True, install_timeout_s=60)')
launch = launch.replace('"/kaggle/working/agent/submit_main.py"','"/kaggle/working/agent/submit_main_v19.py"')
# Remove the second CPU dryrun which would spend another 15 mins loading a server.
pos=launch.find('if DRYRUN:                                   # v1.8: the hidden-test path too')
end2=launch.find('for _p in ("/kaggle/working/agent_server_report.json"',pos)
if pos!=-1 and end2!=-1:
    launch=launch[:pos]+'# v1.9: no second dryrun; DIRECT-only hidden path tested separately.\n'+launch[end2:]
# replace old report keys with new concise ones
launch=launch.replace('"official_score_pct", "query_hits", "queries", "task_macro_score_pct",',
                      '"official_score_pct", "query_hits", "test_outputs", "task_macro_score_pct",')
# ---------------- v1.9.1 launcher clean-up (each replacement must hit exactly once) ----------------
def _once(text, a, b):
    assert text.count(a) == 1, (a, text.count(a))
    return text.replace(a, b)
launch = _once(launch, """    cfg["min_agent_s"] = 3600            # the server ladder never starts a rung with < 1 h left for the agent
    cfg["request_timeout_s"] = 7200               # v1.8.3: 5 min soft limit for the e2e probe in the hidden run (was 10) ...
    cfg["min_agent_s"] = 1200            # ... extended up to 10 min only when its python call already worked
""", """    cfg["min_agent_s"] = 1200            # v1.9.1: the server ladder never starts a rung with < 20 min left
    cfg["request_timeout_s"] = 7200      # hidden run = DIRECT only (v1.8.3 rescue path, 49,152 thinking or not started)
""")
launch = _once(launch, 'EVAL_HOURS = float(os.environ.get("Q27A_EVAL_HOURS", 1.5))   # v1.8: agent only (DIRECT from the run records)',
               'EVAL_HOURS = float(os.environ.get("Q27A_EVAL_HOURS", 2.0))   # v1.9.1: synthesis only (DIRECT from the run-7 records)')
launch = _once(launch, """    # v1.6: the commit run rescues EVERY task with a direct answer, so the report shows agent-only / direct-only /
    # hybrid scores and their overlap; the agent gets 60 % of the eval time
    cfg.update(rescue_all=True, agent_frac=float(os.environ.get("Q27A_AGENT_FRAC", 0.6)))
""", "")
launch = _once(launch, """# v1.8 DRYRUN: a commit run WITHOUT a GPU (or Q27A_DRYRUN=1) drives the real path against fake_vllm.py on CPU: the
# exact cfg, smoke, e2e probe, agent loop, live DIRECT rescue, eval report, then the hidden-test path on 20 tasks into
# /kaggle/working/dryrun/. Never in a competition rerun.""",
"""# DRYRUN: a commit run WITHOUT a GPU (or Q27A_DRYRUN=1) drives the real v1.9.1 path against fake_vllm.py on CPU:
# the exact cfg, budget-contract smoke, synthesis, verification, recorded-DIRECT merge and eval report; the DRYCHECK
# cell then writes dryrun_report.json. Never in a competition rerun.""")
launch = _once(launch, '"official_score_pct", "query_hits", "test_outputs", "task_macro_score_pct",',
               '"official_score_pct", "query_hits", "test_outputs", "task_macro_score_pct", "agent_task_macro_pct", "think",')
launch = _once(launch, '"agent_gen_tok_per_s", "rescue_gen_tok_per_s", "gen_tokens_total", "wall_s")}',
               '"gen_tokens_total", "wall_s")}')
launch = _once(launch, '"hybrid_all_rescued_pct", "agent_only_official_pct",', '"agent_only_official_pct",')
launch = _once(launch, '"tasks_with_verified_program", "status_counts", "circuit_breaker",\n                                         "isolation_denied_tasks", "isolation_guard_failed_tasks",\n',
               '"tasks_with_verified_program", "status_counts",\n')
base.cells[16].source = launch
drycheck=nbformat.v4.new_code_cell((ROOT/'modules'/'drycheck_cell.py').read_text())
base.cells = base.cells[:16] + extra + [base.cells[16], drycheck, base.cells[17]]
# Avoid inheriting the old DRYRUN post-flight assertions tied to forced submit.
for cell in base.cells:
    if cell.cell_type=='code':
        cell.outputs=[]
        cell.execution_count=None
for i,cell in enumerate(base.cells):
    cell.id = f'q27-v19-{i:02d}'
nbformat.validate(base)
nbformat.write(base,OUT)
print('FILE',OUT,'BYTES',OUT.stat().st_size,'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
