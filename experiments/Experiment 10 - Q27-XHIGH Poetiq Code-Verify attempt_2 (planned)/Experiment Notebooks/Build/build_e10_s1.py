"""E10 S1 = Poetiq R5 + correctness fixes + ORDER2 queue + KV admission gate + weight prefetch (Claude, 2026-10-16,
user go "বানাও"). Deterministic builder; source = the exact R5 EVAL120 notebook (LB 21.39 version).

S1 thinking = 60,000 + 8,192 (user 2026-10-16: LB test of the 60k threshold; R1 keeps 77,824).
Unchanged from R5: MAX_MODEL_LEN 106,496, 1 expert, <= 10 iterations, prompts, voting,
sandbox, engine (vLLM 0.19, FP8 KV, MTP-3, 16 slots), full thinking saved to q27x/poetiq_thinking/, hidden clock.

Changes (each one small and independently checkable):
  Q  queue: iteration 1 in ORDER2 order ([>=3 train] + [1 test], ties keep the sorted-id order); failed tasks still go
     to the back, so every task gets iteration 1 before any iteration 2 (R5 data: iteration-1 score does not predict
     which iteration 2 succeeds; 4 of 12 iteration-2 solves had iteration-1 score 0)
  F1 poetiq_parse_code: the LAST ```python block that parses and defines transform (R5 took the first block)
  F3 a train-passing program that fails on one test input is no longer dropped: it can be "best", and every executed
     candidate stays in the ranking; attempts are picked per test input from valid grids (latent: R5 had 0 such programs)
  A2 attempt_2: the second distinct valid grid; if none exists, the test input (or its mirror) instead of a copy of
     attempt_1 (R5: attempt_2 == attempt_1 in 134/172 inputs)
  F2 verifications pending at the hard stop are harvested until 2 min before the budget end (R5: min(HARD_STOP,...));
     F2b then cancels still-queued verifications so leaving the thread pool waits only for the running ones
  C1 one bad engine output no longer ends the solver: per-output try/except, unknown request ids are skipped
  KV admission gate: no new request while vLLM KV usage > E10_KV_ADMIT_MAX (0.95, user 2026-10-16) unless fewer than
     E10_KV_GATE_MIN_ACTIVE (12) are running (R5: KV 99.5 %, 9 preemptions, ~38 min of full-context recompute)
  P  weight prefetch: the model files are read into the page cache while pip runs (read-only, best effort)
Outputs:
  S1  Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb   60,000 thinking, commit = S1COMMIT4, T 1.0 -> submit
  R1  Q27-XHIGH-v1_4_4_E10_S1_T06_EVAL120.ipynb             77,824 thinking, commit = EVAL120, temperature 0.6
  A60 Q27-XHIGH-v1_4_4_E10_S1_60K_EVAL120.ipynb             = S1-60K with commit = EVAL120 (Run A, 12 h)
usage: python build_e10_s1.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R5_EVAL120.ipynb"
SRC_SHA = None   # filled below from the file; the code cells are checked against the Kaggle-submitted R5 by the caller
# D66: the 66 eval tasks R5 did not fully solve (R5 poetiq_final_generated.json vs the eval solutions; 57 zero + 9 partial)
HARD66 = ["135a2760", "13e47133", "16b78196", "195c6913", "20270e3b", "20a9e565", "21897d95", "221dfab4", "247ef758",
          "269e22fb", "271d71e2", "2b83f449", "2d0172a1", "31f7f899", "35ab12c3", "3a25b0d8", "3dc255db", "446ef5d2",
          "4c416de3", "4c7dc4dd", "4e34c42c", "5545f144", "581f7754", "5dbc8537", "64efde09", "67e490f4", "6e453dd6",
          "6e4f6532", "6ffbe589", "71e489b6", "78332cb0", "7b80bb43", "800d221b", "8698868d", "88bcf3b4", "88e364bc",
          "89565ca0", "8b7bacbf", "8e5c0c38", "9385bd28", "97d7923e", "9bbf930d", "a25697e4", "a32d8b75", "a47bf94d",
          "abc82100", "b0039139", "b10624e5", "b6f77b65", "b9e38dc0", "cb2d8a2c", "cbebaa4b", "d35bdbdc", "d59b0160",
          "da515329", "dbff022c", "de809cff", "dfadab01", "e12f9a14", "e3721c99", "e376de54", "e87109e9", "edb79dae",
          "eee78d87", "f560132c", "faa9f03d"]
assert len(HARD66) == len(set(HARD66)) == 66


def once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f"replacement not unique ({n}) in {where}: {old[:90]!r}")
    return text.replace(old, new)


def build(variant):
    nb = json.loads(SRC.read_bytes())
    c = nb["cells"]
    src = lambda i: "".join(c[i]["source"])

    # ------------------------------------------------------------------ cell 1: 12 h clock from kernel start (CLOCK)
    t = src(1)
    t = once(t, "NOTEBOOK_T0 = time.perf_counter()\n",
             "NOTEBOOK_T0 = time.perf_counter()\n"
             "# S1 (CLOCK): Kaggle's 12 h counts from kernel start; R5's cell 0 began 7 s later. Anchor the clock at the kernel\n"
             "# process start (capped at 10 min in case the value is odd).\n"
             "try:\n"
             "    import psutil as _ps\n"
             "    _since_kernel_start = max(0.0, min(600.0, time.time() - _ps.Process(os.getpid()).create_time()))\n"
             "except Exception:\n"
             "    _since_kernel_start = 0.0\n"
             "NOTEBOOK_T0 -= _since_kernel_start\n", "clock anchor")
    t = once(t, '      "| fallback submission.json:", len(test_challenges), "tasks | RUN_ID", RUN_ID)\n',
             '      "| fallback submission.json:", len(test_challenges), "tasks | RUN_ID", RUN_ID,\n'
             '      "| clock anchored", round(_since_kernel_start, 1), "s before cell 0")\n', "clock print")
    c[1]["source"] = t.splitlines(keepends=True)

    # ------------------------------------------------------------------ cell 2: profiles, temperature, KV gate config
    t = src(2)
    t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
             '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
             '    # S1 quick commit: 4 eval tasks that closed fastest in R5 iteration 1 -> a valid version in about 1 h, then submit\n'
             '    "S1COMMIT4": {"source": "eval", "task_list": ["1818057f", "2ba387bc", "bf45cf4b", "58490d8a"],\n'
             '                  "time_budget_sec": 90 * 60, "admit_frac": 0.80, "drain_frac": 0.90},\n', "profile row")
    default_profile = {"S1": "S1COMMIT4", "R1": "EVAL120", "A60": "EVAL120", "D66": "HARD66", "L66": "HARD66",
                       "W3H": "LOW3H", "S49": "S1COMMIT4"}[variant]
    t = once(t, 'os.environ.get("Q27X_PROFILE", "EVAL120")', f'os.environ.get("Q27X_PROFILE", "{default_profile}")', "default profile")
    if variant == "R1":
        t = once(t, "os.environ.get('E10_POETIQ_TEMPERATURE','1.0')", "os.environ.get('E10_POETIQ_TEMPERATURE','0.6')", "temperature")
    if variant in ("D66", "L66"):
        # Deep run (user go 2026-10-16): only the 66 eval tasks R5 did not fully solve, fresh seeds, R5 settings otherwise
        t = once(t, '    # R5 EVAL120: full evaluation run',
                 '    # D66: the 66 eval tasks R5 did not fully solve (57 zero + 9 partial), on the hidden-run clock\n'
                 f'    "HARD66":    {{"source": "eval", "task_list": {json.dumps(HARD66)},\n'
                 '                  "time_budget_sec": 12.0 * 3600 - 3 * 60, "admit_frac": 0.90, "drain_frac": 0.94,\n'
                 '                  "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},\n'
                 '    # R5 EVAL120: full evaluation run', "HARD66 profile")
    if variant == "D66":
        t = once(t, "\nSEED = 20261005\n", "\nSEED = 20261016   # D66: fresh sampling seeds (R5: 20261005)\n", "D66 seed")
        t = once(t, "os.environ.get('E10_POETIQ_SEED','0')", "os.environ.get('E10_POETIQ_SEED','1')", "D66 example-order seed")
    if variant == "L66":
        # L66 (user go 2026-10-16): D66 with 122,880 thinking on the same 66 tasks; D66 (77,824) is the paired control.
        # MAX_MODEL_LEN must grow too (else phase1_cap clips to ~93k); 12 slots (16 x ~125k would overflow the 1.35M KV pool)
        t = once(t, "\nSEED = 20261005\n", "\nSEED = 20261017   # L66: fresh sampling seeds (R5: 20261005, D66: 20261016)\n", "L66 seed")
        t = once(t, "os.environ.get('E10_POETIQ_SEED','0')", "os.environ.get('E10_POETIQ_SEED','2')", "L66 example-order seed")
        t = once(t, 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 77824))   # E9/E10: 86k total (77,824 + 8,192); was 49,152',
                 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 122880))   # L66: long-thinking measurement (D66/R5: 77,824)',
                 "L66 think budget")
        t = once(t, 'os.environ.get("Q27X_MAX_MODEL_LEN", 106496))',
                 'os.environ.get("Q27X_MAX_MODEL_LEN", 155648))   # L66: prompt (<= ~20k) + 122,880 + 8,192 fits;', "L66 max model len")
        t = once(t, 'MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 16))',
                 'MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 12))   # L66: 12 x ~125k fits the KV pool better than 16;',
                 "L66 slots")
    if variant == "W3H":
        # W3H (user go 2026-10-17): the submitted S1-60K with reasoning_effort "low" (template adds "keep your thinking brief";
        # medium was dropped: ARC Prize v2 semi-private medium 13.2 % < low 22.8 % < xhigh 42.4 %); 3 h on all 120 eval
        # tasks to measure tokens per request, tasks reached and iteration-1 accuracy vs R5 (xhigh 77.8k)
        t = once(t, '    # R5 EVAL120: full evaluation run',
                 '    # W3H: all 120 eval tasks, 3 h measurement clock (user 2026-10-17) (same margins as the hidden run)\n'
                 '    "LOW3H":     {"source": "eval", "n_tasks": 120, "time_budget_sec": 3.0 * 3600,\n'
                 '                  "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,\n'
                 '                  "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},\n'
                 '    # R5 EVAL120: full evaluation run', "LOW3H profile")
        t = once(t, 'REASONING_EFFORT = "xhigh"', 'REASONING_EFFORT = os.environ.get("Q27X_REASONING_EFFORT", "low")   # W3H (xhigh in S1/R5)',
                 "W3H reasoning effort")
    if variant in ("S1", "A60", "W3H"):
        # 60K (user, 2026-10-16): LB test of the 60,000 thinking threshold against R5's 77,824 (still >= the 49,152 floor)
        t = once(t, 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 77824))   # E9/E10: 86k total (77,824 + 8,192); was 49,152',
                 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 60000))   # S1-60K: LB test of 60,000 (R5/E9: 77,824; floor 49,152)',
                 "60k think budget")
    if variant == "S49":
        # S49 (user 2026-10-18, "প্রার্থী ২"): the submitted S1-60K with thinking 49,152 (= the user's floor). ONE change.
        t = once(t, 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 77824))   # E9/E10: 86k total (77,824 + 8,192); was 49,152',
                 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 49152))   # S49: LB test of 49,152 (S1: 60,000; R5/E9: 77,824; floor 49,152)',
                 "49k think budget")
    t = once(t, "POETIQ_SANDBOX_WORKERS = int(os.environ.get('E10_POETIQ_SANDBOX_WORKERS','4'))\n",
             "POETIQ_SANDBOX_WORKERS = int(os.environ.get('E10_POETIQ_SANDBOX_WORKERS','4'))\n"
             "# S1: KV admission gate (no new request above this vLLM KV usage unless fewer than MIN_ACTIVE are running)\n"
             "E10_KV_ADMIT_MAX = float(os.environ.get('E10_KV_ADMIT_MAX','0.95'))\n"
             "E10_KV_GATE_MIN_ACTIVE = int(os.environ.get('E10_KV_GATE_MIN_ACTIVE','12'))\n", "kv gate config")
    if variant == "L66":   # with 12 slots the gate must be able to act below 12 running
        t = once(t, "E10_KV_GATE_MIN_ACTIVE = int(os.environ.get('E10_KV_GATE_MIN_ACTIVE','12'))\n",
                 "E10_KV_GATE_MIN_ACTIVE = int(os.environ.get('E10_KV_GATE_MIN_ACTIVE','8'))   # L66: 12 slots, gate from 8 running\n",
                 "L66 kv gate floor")
    # CLOCK: R5 finished its report + notebook save 16 s after the solver, at 11:43:40, leaving 16 min of the 12 h unused.
    # New hidden clock (EVAL120 uses the same): budget 12 h - 3 min, hard stop 11:55, verify harvest until 11:56,
    # drain 11:47 (8 min kept for the forced answers), last admission 11:45. About 3.5 min stay as safety margin.
    if t.count('"time_budget_sec": 12.0 * 3600 - 10 * 60') != 2 or t.count('"hard_margin_sec": 6 * 60, "verify_margin_sec": 3 * 60') != 2:
        sys.exit("clock rows not found exactly twice (EVAL120 + RERUN_ALL)")
    t = t.replace('"time_budget_sec": 12.0 * 3600 - 10 * 60', '"time_budget_sec": 12.0 * 3600 - 3 * 60')
    t = t.replace('"hard_margin_sec": 6 * 60, "verify_margin_sec": 3 * 60', '"hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60')
    c[2]["source"] = t.splitlines(keepends=True)

    # ------------------------------------------------------------------ cell 3: weight prefetch while pip runs
    t = src(3)
    t = once(t, 'if os.environ.get("Q27X_SKIP_INSTALL") != "1":     # Q27X_SKIP_INSTALL is for the local harness only\n',
             '# S1: read the model files into the page cache in the background while pip runs (read-only, best effort)\n'
             'if os.environ.get("Q27X_SKIP_INSTALL") != "1":\n'
             '    try:\n'
             '        import shlex as _shlex\n'
             '        _wfiles = sorted(str(p) for p in MODEL_PATH.glob("*.safetensors"))\n'
             '        if _wfiles:\n'
             '            subprocess.Popen(["bash", "-c", "cat " + " ".join(_shlex.quote(p) for p in _wfiles) + " > /dev/null"],\n'
             '                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
             '            print("weight prefetch started:", len(_wfiles), "files")\n'
             '    except Exception as _e:\n'
             '        print("weight prefetch not started:", repr(_e)[:200])\n'
             '\n'
             'if os.environ.get("Q27X_SKIP_INSTALL") != "1":     # Q27X_SKIP_INSTALL is for the local harness only\n', "prefetch")
    c[3]["source"] = t.splitlines(keepends=True)

    # ------------------------------------------------------------------ cell 5: F1 parser, A2 attempts
    t = src(5)
    t = once(t, "def poetiq_parse_code(response):\n"
                "    match=re.search(r'```python\\s*(.*?)```',response,re.DOTALL|re.IGNORECASE)\n"
                "    return match.group(1).strip() if match else None\n",
             "def poetiq_parse_code(response):\n"
             "    # S1 (F1): the LAST ```python block that parses and defines transform(); R5 took the first block, so a short\n"
             "    # illustrative snippet written before the real solution was the one executed. Falls back to the R5 choice.\n"
             "    import ast as _ast\n"
             "    blocks=re.findall(r'```python\\s*(.*?)```',response or '',re.DOTALL|re.IGNORECASE)\n"
             "    for b in reversed(blocks):\n"
             "        b=b.strip()\n"
             "        try:\n"
             "            tree=_ast.parse(b)\n"
             "        except (SyntaxError,ValueError):\n"
             "            continue\n"
             "        if any(isinstance(n,_ast.FunctionDef) and n.name=='transform' for n in tree.body):\n"
             "            return b\n"
             "    return blocks[0].strip() if blocks else None\n", "F1 parser")
    t = once(t, "        if grids:\n"
                "            a1=grids[0];a2=grids[1] if len(grids)>1 else grids[0]\n"
                "        else:\n"
                "            a1=a2=test['input'] # only if no valid program output; required schema\n",
             "        # S1 (A2): never repeat attempt_1 (R5 did in 134/172 inputs); the next distinct grid, else the test input,\n"
             "        # else its mirror\n"
             "        mirror=[list(reversed(row)) for row in test['input']]\n"
             "        if grids:\n"
             "            a1=grids[0]\n"
             "            a2=next((g for g in grids[1:]+[test['input'],mirror] if g!=a1),grids[0])\n"
             "        else:\n"
             "            a1=test['input'] # only if no valid program output; required schema\n"
             "            a2=mirror if mirror!=a1 else test['input']\n", "A2 attempts")
    if variant in ("D66", "L66", "W3H"):   # independent review F4: RecursionError / MemoryError from ast.parse must not close the unit
        t = once(t, "        except (SyntaxError,ValueError):\n", "        except Exception:   # D66: any parse failure -> next block\n",
                 "D66 parser except")
    c[5]["source"] = t.splitlines(keepends=True)

    # ------------------------------------------------------------------ cell 7: queue, F3, KV gate, C1, F2
    t = src(7)
    t = once(t, "for _task_idx, _tid in enumerate(TASK_IDS):\n"
                "    for _expert in range(POETIQ_NUM_EXPERTS):\n",
             "# S1 (Q): iteration 1 in ORDER2 order ([>=3 train] + [1 test], ties keep the sorted-id order). Seeds still use the\n"
             "# sorted-id index. Failed tasks go to the back (unchanged), so every task gets iteration 1 before any iteration 2.\n"
             "def e10_order2_score(tid):\n"
             "    _t=CHALLENGES[tid]\n"
             "    return int(len(_t['train'])>=3)+int(len(_t['test'])==1)\n"
             "_E10_INDEX={t:i for i,t in enumerate(TASK_IDS)}\n"
             "E10_ORDER=sorted(TASK_IDS,key=lambda t:(-e10_order2_score(t),_E10_INDEX[t]))\n"
             "for _tid in E10_ORDER:\n"
             "    _task_idx=_E10_INDEX[_tid]\n"
             "    for _expert in range(POETIQ_NUM_EXPERTS):\n", "Q queue")
    if variant == "S49":
        # S49 (user 2026-10-18): ORDER2 tie-breaker "smaller total grid cells first" (lever analysis 2026-10-17:
        # +0.7 / +2.4 at a 70 % budget on ORDER2-DIRECT / R5 data, ARC Prize xhigh AUC 0.603 -> 0.627)
        t = once(t, "E10_ORDER=sorted(TASK_IDS,key=lambda t:(-e10_order2_score(t),_E10_INDEX[t]))\n",
                 "def e10_total_cells(tid):   # S49: every train input + output and every test input\n"
                 "    _t=CHALLENGES[tid]\n"
                 "    _c=lambda g:len(g)*len(g[0]) if g and g[0] else 0\n"
                 "    return sum(_c(p['input'])+_c(p['output']) for p in _t['train'])+sum(_c(p['input']) for p in _t['test'])\n"
                 "E10_ORDER=sorted(TASK_IDS,key=lambda t:(-e10_order2_score(t),e10_total_cells(t),_E10_INDEX[t]))   # S49: + cells tie-break\n",
                 "S49 cells tie-break")
    t = once(t, "             'sandbox_verifications':0,'voting_updates':0,'feedback_trims':0,'sandbox_startup':E10_SANDBOX_SELFTEST}\n",
             "             'sandbox_verifications':0,'voting_updates':0,'feedback_trims':0,'sandbox_startup':E10_SANDBOX_SELFTEST,\n"
             "             'kv_gate_blocks':0,'kv_gate_last':None,'unit_errors':[]}\n", "sched stats")
    t = once(t, "        usable=bool(result.get('test_usable',False))\n"
                "        if usable and score >= st['best_score']:\n",
             "        usable=bool(result.get('test_usable',False))\n"
             "        # S1 (F3): a train-passing program stays eligible as best even if one test input fails; attempts are then\n"
             "        # picked per test input from valid grids (R5 dropped it and lost every query of the task).\n"
             "        if (usable or result['passed']) and score >= st['best_score']:\n", "F3 best")
    t = once(t, "             for r in s['candidate_history']\n"
                "             if id(r) not in already and r.get('test_usable')]\n",
             "             for r in s['candidate_history']\n"
             "             if id(r) not in already]   # S1 (F3): keep partly usable candidates; grids are checked per test input\n",
             "F3 history")
    t = once(t, "E10_METRICS=[]\nE10_SCHED_ERROR=None\n",
             "E10_METRICS=[]\nE10_SCHED_ERROR=None\n"
             "_E10_KV={'t':-1e9,'v':0.0}\n"
             "def e10_kv_usage():\n"
             "    # S1: vLLM KV usage (0..1), read at most every 5 s; 0.0 if the metric is unavailable (gate then stays open)\n"
             "    now=time.perf_counter()\n"
             "    if now-_E10_KV['t']>=5.0:\n"
             "        _E10_KV['t']=now\n"
             "        try:\n"
             "            vals=[float(getattr(m,'value',0) or 0) for m in llm.get_metrics() if 'kv_cache_usage' in getattr(m,'name','')]\n"
             "            _E10_KV['v']=max(vals) if vals else 0.0\n"
             "        except Exception:\n"
             "            _E10_KV['v']=0.0\n"
             "    return _E10_KV['v']\n"
             "\n"
             "def e10_kv_gate_open():\n"
             "    if len(E10_ACTIVE)<E10_KV_GATE_MIN_ACTIVE:return True\n"
             "    kv=e10_kv_usage()\n"
             "    if kv<=E10_KV_ADMIT_MAX:return True\n"
             "    E10_SCHED['kv_gate_blocks']+=1;E10_SCHED['kv_gate_last']=round(kv,3)\n"
             "    return False\n", "kv helpers")
    t = once(t, "            while E10_QUEUE and len(E10_ACTIVE)<CONCURRENCY and now<ADMIT_DEADLINE and not drained:\n",
             "            while E10_QUEUE and len(E10_ACTIVE)<CONCURRENCY and now<ADMIT_DEADLINE and not drained and e10_kv_gate_open():\n",
             "kv gate")
    old_loop = (
        "            for o in outs:\n"
        "                match=re.fullmatch(r'(.+)~e(\\d+)~i(\\d+)\\|([12])',o.request_id)\n"
        "                if not match:raise RuntimeError(f'Unexpected request ID: {o.request_id}')\n"
        "                tid,ex,iteration,phase=match.group(1),int(match.group(2)),int(match.group(3)),int(match.group(4))\n"
        "                key=(tid,ex)\n"
        "                req=E10_ACTIVE.get(key)\n"
        "                if req is None or req['phase']!=phase:continue\n"
        "                if E10_STATES[key]['iteration']!=iteration:continue\n"
        "                comp=o.outputs[0] if o.outputs else None\n"
        "                if comp is not None:\n"
        "                    token_ids=list(comp.token_ids);E10_TOTAL_GEN+=len(token_ids)\n"
        "                    (req['p1_ids'] if phase==1 else req['p2_ids']).extend(token_ids)\n"
        "                if not o.finished:continue\n"
        "                reason=comp.finish_reason if comp is not None else None\n"
        "                if phase==1:\n"
        "                    req['p1_finish']=reason\n"
        "                    if THINK_CLOSE_ID in req['p1_ids'] and reason!='length':\n"
        "                        e10_after_decode(req,'natural',pool)\n"
        "                    elif not e10_start_phase2(req):\n"
        "                        e10_after_decode(req,'no_room_for_code',pool)\n"
        "                else:\n"
        "                    req['p2_finish']=reason\n"
        "                    e10_after_decode(req,req['mode'],pool)\n")
    new_loop = (
        "            for o in outs:\n"
        "              # S1 (C1): one bad output must not end the whole solver; the unit is closed and the run continues\n"
        "              try:\n"
        "                match=re.fullmatch(r'(.+)~e(\\d+)~i(\\d+)\\|([12])',o.request_id)\n"
        "                if not match:\n"
        "                    E10_SCHED['unit_errors'].append(f'unexpected request id {o.request_id!r}'[:300]);continue\n"
        "                tid,ex,iteration,phase=match.group(1),int(match.group(2)),int(match.group(3)),int(match.group(4))\n"
        "                key=(tid,ex)\n"
        "                req=E10_ACTIVE.get(key)\n"
        "                if req is None or req['phase']!=phase:continue\n"
        "                if E10_STATES[key]['iteration']!=iteration:continue\n"
        "                comp=o.outputs[0] if o.outputs else None\n"
        "                if comp is not None:\n"
        "                    token_ids=list(comp.token_ids);E10_TOTAL_GEN+=len(token_ids)\n"
        "                    (req['p1_ids'] if phase==1 else req['p2_ids']).extend(token_ids)\n"
        "                if not o.finished:continue\n"
        "                reason=comp.finish_reason if comp is not None else None\n"
        "                if phase==1:\n"
        "                    req['p1_finish']=reason\n"
        "                    if THINK_CLOSE_ID in req['p1_ids'] and reason!='length':\n"
        "                        e10_after_decode(req,'natural',pool)\n"
        "                    elif not e10_start_phase2(req):\n"
        "                        e10_after_decode(req,'no_room_for_code',pool)\n"
        "                else:\n"
        "                    req['p2_finish']=reason\n"
        "                    e10_after_decode(req,req['mode'],pool)\n"
        "              except Exception as _uerr:\n"
        "                E10_SCHED['unit_errors'].append(f'{getattr(o,\"request_id\",\"?\")}: {type(_uerr).__name__}: {_uerr}'[:300])\n"
        "                print('[E10] unit error (isolated):',E10_SCHED['unit_errors'][-1],flush=True)\n"
        "                try:engine.abort_request([o.request_id])\n"
        "                except Exception:pass\n"
        "                _m=re.fullmatch(r'(.+)~e(\\d+)~i(\\d+)\\|([12])',str(getattr(o,'request_id','')))\n"
        "                if _m:\n"
        "                    _k=(_m.group(1),int(_m.group(2)))\n"
        "                    E10_ACTIVE.pop(_k,None)\n"
        "                    if _k in E10_STATES:\n"
        "                        E10_STATES[_k]['finished']=True;E10_STATES[_k]['status']='unit_error'\n")
    t = once(t, old_loop, new_loop, "C1 loop")
    t = once(t, "        endwait=min(HARD_STOP,time.perf_counter()+POETIQ_SANDBOX_TIMEOUT*2)\n",
             "        # S1 (F2): HARD_STOP is already past at the hard stop, so R5 never harvested verifications started there\n"
             "        endwait=min(NOTEBOOK_T0+TOTAL_BUDGET_SEC-60,time.perf_counter()+POETIQ_SANDBOX_TIMEOUT*2)\n", "F2 harvest")
    t = once(t, "        E10_SCHED['remaining_verifications']=len(E10_VERIFY)\n",
             "        E10_SCHED['remaining_verifications']=len(E10_VERIFY)\n"
             "        # S1 (F2b): leaving the with-block would wait for every queued verification; cancel the queued ones so only\n"
             "        # the <= POETIQ_SANDBOX_WORKERS running jobs are awaited (GPT audit 2026-10-16)\n"
             "        pool.shutdown(wait=False,cancel_futures=True)\n", "F2b cancel queued")
    if variant in ("D66", "L66", "W3H"):
        # D66 (GPT audit F1 + user, 2026-10-16): per-iteration evidence, logging only. The solver never sees test truth;
        # test_correct is computed from SOLUTIONS (eval only, None in a rerun) after selection is already done.
        t = once(t, "    base_rec['train_passed']=bool(result and result['passed'])\n    E10_RECORDS.append(base_rec)\n",
                 "    base_rec['train_passed']=bool(result and result['passed'])\n"
                 "    # D66: per-iteration evidence for the deep-run analysis (logging only)\n"
                 "    try:\n"
                 "        base_rec['test_usable']=bool(result and result.get('test_usable'))\n"
                 "        base_rec['test_grids']=[r.get('grid') for r in result['test_results']] if result else None\n"
                 "        base_rec['test_errors']=[(str(r.get('error'))[-300:] if r.get('error') else None) for r in result['test_results']] if result else None\n"
                 "        base_rec['train_soft_scores']=[r.get('soft_score') for r in result['train_results']] if result else None\n"
                 "        _truth=SOLUTIONS.get(st['task_id']) if (result and SOLUTIONS) else None\n"
                 "        base_rec['test_correct']=[(g is not None and q<len(_truth) and g==_truth[q]) for q,g in enumerate(base_rec['test_grids'])] if _truth else None\n"
                 "        base_rec['t_done_sec']=round(time.perf_counter()-NOTEBOOK_T0,1)\n"
                 "    except Exception as _lerr:\n"
                 "        base_rec['log_error']=repr(_lerr)[:300]\n"
                 "    E10_RECORDS.append(base_rec)\n", "D66 per-iteration evidence")
        # D66 (independent review 2026-10-16): admit time + phase-1 finish (drained vs cap), unharvested list, and the
        # deployable two-attempt score after every commit (eval only); all logging only
        t = once(t, "              'sec':round(time.perf_counter()-req['t_admit'],2),'has_python':bool(code)}\n",
                 "              'sec':round(time.perf_counter()-req['t_admit'],2),'has_python':bool(code),\n"
                 "              't_admit':round(req['t_admit']-NOTEBOOK_T0,1),'p1_finish':req.get('p1_finish')}   # D66\n",
                 "D66 admit time + p1_finish")
        t = once(t, "    e10_update_task_submission(st['task_id'])\n",
                 "    e10_update_task_submission(st['task_id'])\n"
                 "    try:   # D66: deployable two-attempt score after every commit (eval only; logging only)\n"
                 "        if SOLUTIONS is not None:\n"
                 "            _g=SOLUTIONS[st['task_id']];_fa=E10_FINAL.get(st['task_id'],[])\n"
                 "            with open(OUT/'poetiq_attempt_curve.jsonl','a',encoding='utf-8') as _fh:\n"
                 "                _fh.write(json.dumps({'task_id':st['task_id'],'iterations_done':st['iteration'],\n"
                 "                    't':round(time.perf_counter()-NOTEBOOK_T0,1),\n"
                 "                    'pass1':[a['attempt_1']==_g[i] for i,a in enumerate(_fa) if i<len(_g)],\n"
                 "                    'hit':[(a['attempt_1']==_g[i] or a['attempt_2']==_g[i]) for i,a in enumerate(_fa) if i<len(_g)]})+'\\n')\n"
                 "    except Exception:pass\n", "D66 attempt curve")
        t = once(t, "        E10_SCHED['remaining_verifications']=len(E10_VERIFY)\n        # S1 (F2b)",
                 "        E10_SCHED['remaining_verifications']=len(E10_VERIFY)\n"
                 "        E10_SCHED['unharvested']=[f\"{_r['task_id']}~i{_r['iteration']}\" for _k,_c,_r in E10_VERIFY.values()]   # D66\n"
                 "        # S1 (F2b)", "D66 unharvested list")
    c[7]["source"] = t.splitlines(keepends=True)

    # ------------------------------------------------------------------ cell 0: header
    note = {"S1": "**S1-60K (2026-10-16), submission build and the Experiment 11 base:** R5 + ORDER2 queue + F1/F3/A2/F2/C1 "
                  "fixes + KV admission gate (0.95, >= 12 running) + weight prefetch + 12 h clock from kernel start (last "
                  "admission 11:45, drain 11:47, hard stop 11:55; R5 left 16 min unused). **Thinking 60,000 + 8,192** (user: LB "
                  "test of the 60k threshold against R5's 77,824; `Q27X_THINK_BUDGET` overrides). Temperature 1.0 unchanged. "
                  "Commit run = 4 quick eval tasks (`S1COMMIT4`, about 1 h) so the version can be submitted; the hidden rerun "
                  "solves every task. `Q27X_PROFILE=EVAL120` gives the full evaluation run.\n",
            "R1": "**S1-T06 (2026-10-16), Run 1:** identical to S1 except Poetiq temperature 0.6 (`E10_POETIQ_TEMPERATURE`) "
                  "and the commit run = EVAL120 (120 tasks / 172 inputs on the hidden-run clock). Purpose: measure speed "
                  "(MTP acceptance, tok/s) and quality at T 0.6 against R5 (T 1.0).\n",
            "A60": "**S1-60K EVAL120 (2026-10-16), Run A:** identical to the submitted S1-60K except the commit run = "
                   "EVAL120 (120 tasks / 172 inputs on the hidden-run clock). Purpose: the iteration-1 score at 60,000 "
                   "thinking when every task is reached (R5 at 77,824: about 39.4 %), forced-close quality at 60k, and "
                   "tokens/time per task; read together with the S1-60K LB.\n",
            "D66": "**D66 deep run (2026-10-16):** the 66 eval tasks R5 did not fully solve (57 zero + 9 partial), with the full "
                   "12 h hidden-run clock, fresh seeds (`SEED` 20261016, `E10_POETIQ_SEED` 1) and R5's settings otherwise "
                   "(77,824 + 8,192 thinking, T 1.0, up to 10 iterations, S1 fixes). Purpose: new information that R5 cannot "
                   "give: the value of a fresh attempt on failed tasks, of iterations 3+, and of re-attempts on train-passing "
                   "but test-wrong tasks; with R5 this gives the compute -> score curve. Not a submission.\n",
            "L66": "**L66 long-thinking run (2026-10-16):** D66 with **122,880 + 8,192** thinking on the same 66 eval tasks R5 "
                   "did not fully solve (in R5, 74 % of their iteration-1 requests were cut at the 77,824 cap, against 33 % for "
                   "the solved tasks). `MAX_MODEL_LEN` 155,648, 12 slots, KV gate from 8 running, fresh seeds (`SEED` 20261017, "
                   "`E10_POETIQ_SEED` 2); everything else as D66 (T 1.0, up to 10 iterations, hidden-run clock, full logging). "
                   "Purpose: paired with D66 (77,824) on the same tasks: more attempts vs longer thinking per attempt. "
                   "Not a submission.\n",
            "W3H": "**W3H low-effort probe (2026-10-17):** the submitted S1-60K (LB 34.58) with **`reasoning_effort` "
                   "\"low\"** instead of xhigh (`Q27X_REASONING_EFFORT` overrides). Commit run = all 120 eval tasks on a "
                   "**3 h** clock (`LOW3H`), with per-iteration test logging and the attempt curve. Purpose: tokens per "
                   "request, tasks reached in 3 h and iteration-1 accuracy, compared task by task with R5 (xhigh). "
                   "Not a submission.\n",
            "S49": "**S49 (2026-10-18), submission build:** identical to the submitted S1-60K (LB 34.58) except **thinking "
                   "49,152 + 8,192** (the user's floor; S1: 60,000; R5: 77,824) **and the ORDER2 tie-breaker \"smaller total "
                   "grid cells first\"** (user 2026-10-18: both changes together). LB test of the next cap step (77.8k -> 60k "
                   "gave 21.39 -> 34.58 together with the S1 fixes). Commit run = `S1COMMIT4` (4 quick eval tasks) so the "
                   "version can be submitted; the hidden rerun solves every task.\n"}[variant]
    c[0]["source"] = (src(0) + "\n\n" + note).splitlines(keepends=True)

    for cell in c:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    name = {"S1": "Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb", "R1": "Q27-XHIGH-v1_4_4_E10_S1_T06_EVAL120.ipynb",
            "A60": "Q27-XHIGH-v1_4_4_E10_S1_60K_EVAL120.ipynb", "D66": "Q27-XHIGH-v1_4_4_E10_D66_HARD_DEEP.ipynb",
            "L66": "Q27-XHIGH-v1_4_4_E10_L66_HARD_120K.ipynb", "W3H": "Q27-XHIGH-v1_4_4_E10_W3H_LOW_3H.ipynb",
            "S49": "Q27-XHIGH-v1_4_4_E10_S1_49K_POETIQ_FIX_ORDER2.ipynb"}[variant]
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    (HERE.parent / name).write_bytes(data)
    print("BUILT", name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())


print("source", SRC.name, "sha256", hashlib.sha256(SRC.read_bytes()).hexdigest()[:16])
build("S1")
build("R1")
build("A60")
build("D66")
build("L66")
build("W3H")
build("S49")
