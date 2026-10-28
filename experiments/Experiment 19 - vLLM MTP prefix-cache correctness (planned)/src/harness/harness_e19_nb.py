"""CPU harness for E19 (vLLM 0.19 MTP / prefix-cache / async-scheduling / KV-dtype correctness). Runs the REAL notebook
cells in order (CELL 0, CELL 1, the task order, the Poetiq core, E19-J, E19-1, E19-2) against fake_vllm019_server.py.
CELL 2 (pip + tokenizer) is replaced by the real Qwen3.8 tokenizer fixture.
usage: python harness_e19_nb.py <notebook> <scenario>
scenarios (all but split / rerun / budget run profile E19ALL = the six arms A B C D E K on one fake server each):
  normal     every arm PASS; flags = S1's ENGINE_CONFIG + the arm's own switch; caches exercised; churn really churns;
             low-margin divergences (MTP vs not, concurrency, cache) tolerated
  launchfail the B arm's server exits at start -> B LAUNCH_FAILED, GPU wait, the other arms still run and PASS (the
             churn calibration falls back to A alone)
  death      the D arm's server dies mid-stream (14th request) -> D INCOMPLETE, its finished rows kept, later arms run
  corrupt    the D arm emits '!' x N with a 9-nat margin on every cache hit -> D FAIL, the others PASS
  race       every MTP arm with async scheduling emits confidently wrong tokens when the batch changes -> D, B, K FAIL
             (B is disqualified as a calibrator), E (--no-async-scheduling), A, C PASS; serial rows untouched
  collapse   the D arm reports 0 accepted MTP tokens (outputs fine) -> D FAIL on the acceptance windows alone
  kvretry    the bf16-KV arm K fails the KV check at 106,496 -> relaunched at 65,536 -> PASS
  budget     a tiny deadline: D starts and is cut -> NOT_TESTED_BUDGET (never PASS); C not launched -> NOT_TESTED_BUDGET
  rerun      KAGGLE_IS_COMPETITION_RERUN -> no server, identity submission only
  split      the three account notebooks (P1, P2, P3) with their DEFAULT profiles, then src/e19_compare.py over the
             three outputs (two folders + one bundle zip) -> all six arms, churn calibrated by A and B, all PASS"""
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import zipfile
from pathlib import Path

NB, SCEN = Path(sys.argv[1]), sys.argv[2]
HERE = Path(__file__).resolve().parent
SRC = HERE.parent
DATA = Path(os.environ.get("E19_DATA", r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 "
                                     r"(planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data"))
TOKDIR = Path(os.environ.get("E19_TOKDIR", r"D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok"))
sys.stdout.reconfigure(encoding="utf-8")
if os.name == "nt":
    import signal as _sig
    if not hasattr(_sig, "SIGKILL"):
        _sig.SIGKILL = 9
DEFAULTS = {"Q27-XHIGH-E19_MTPPC_P1_D_C.ipynb": ("E19P1", [["D", 0], ["C", 22]]),
            "Q27-XHIGH-E19_MTPPC_P2_E_A.ipynb": ("E19P2", [["E", 0], ["A", 22]]),
            "Q27-XHIGH-E19_MTPPC_P3_B_K.ipynb": ("E19P3", [["B", 0], ["K", 18]])}
ALL = ["A", "B", "C", "D", "E", "K"]
fails = []


def check(cond, msg, info=""):
    print(("PASS " if cond else "FAIL ") + msg + ("" if cond else f"  -> {str(info)[:900]}"), flush=True)
    if not cond:
        fails.append(msg)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup(tag):
    work = Path(tempfile.mkdtemp(prefix=f"e19_{tag}_"))
    arc = work / "arc"
    arc.mkdir()
    for f in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
        shutil.copy(DATA / f, arc / f)
    ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json", encoding="utf-8"))
    pick = sorted(ch)[:4]                        # the "hidden test" file of this local run (identity fallback only)
    json.dump({t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in pick},
              open(arc / "arc-agi_test_challenges.json", "w"))
    return work, arc


def cells_of(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 8 and "CELL E19-J" in code[5] and "CELL E19-1" in code[6] and "CELL E19-2" in code[7], len(code)
    return code


def profile_check(nb_path):
    """CELL 0 + CELL 1 in a fresh namespace with no profile env: the notebook's own commit default."""
    code = cells_of(nb_path)
    work, arc = setup("profile")
    os.environ.update(Q27X_ARC_ROOT=str(arc), Q27X_WORKING=str(work), Q27X_MODEL=str(TOKDIR))
    saved = {k: os.environ.pop(k, None) for k in ("Q27X_PROFILE", "KAGGLE_IS_COMPETITION_RERUN")}
    G = {"__name__": "__main__"}
    exec(compile(code[0], "<cell0>", "exec"), G)
    exec(compile(code[1], "<cell1>", "exec"), G)
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
    want = DEFAULTS[nb_path.name]
    return (G["PROFILE"] == want[0] and G["P"]["e19_arms"] == want[1] and G["TOTAL_BUDGET_SEC"] == 3600
            and round(G["HARD_STOP"] - G["NOTEBOOK_T0"]) == 3420 and G["E19_ON"] and len(G["P"]["task_list"]) == 12), \
        (G["PROFILE"], G["P"].get("e19_arms"), G["TOTAL_BUDGET_SEC"], round(G["HARD_STOP"] - G["NOTEBOOK_T0"]))


def run_nb(nb_path, scen, profile, tag, pre_run=None):
    work, arc = setup(tag)
    flog = work / "fake_log.jsonl"
    env = dict(Q27X_ARC_ROOT=str(arc), Q27X_WORKING=str(work), Q27X_MODEL=str(TOKDIR), Q27X_SKIP_INSTALL="1",
               PYTHONIOENCODING="utf-8", E19_PORT=str(free_port()), E19_FAKE_SERVER=str(HERE / "fake_vllm019_server.py"),
               E19_FAKE_SCEN=scen, E19_FAKE_LOG=str(flog), Q27X_GPU_WAIT_MAX="1", E19_SAMPLE_SEC="0.3",
               E19_CHURN_STAGGER_SEC="0.02")
    for k in ("Q27X_PROFILE", "KAGGLE_IS_COMPETITION_RERUN", "E19_LAUNCH_MIN_SEC", "E19_MIN_SERIAL_SEC", "E19_MIN_CHURN_SEC",
              "E19_MIN_LONG_SEC", "E19_REPORT_RESERVE_SEC"):
        os.environ.pop(k, None)
    os.environ.update(env)
    if profile:
        os.environ["Q27X_PROFILE"] = profile
    if scen == "rerun":
        os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "true"
    if tag == "budget":
        os.environ.update(E19_LAUNCH_MIN_SEC="8", E19_MIN_SERIAL_SEC="1", E19_MIN_CHURN_SEC="1", E19_MIN_LONG_SEC="1",
                          E19_REPORT_RESERVE_SEC="0.5")
    code = cells_of(nb_path)
    G = {"__name__": "__main__"}
    exec(compile(code[0], "<cell0>", "exec"), G)
    exec(compile(code[1], "<cell1>", "exec"), G)
    from transformers import AutoTokenizer
    G["tokenizer"] = AutoTokenizer.from_pretrained(str(TOKDIR))      # CELL 2 replacement
    exec(compile(code[3], "<cell3 task order>", "exec"), G)
    exec(compile(code[4], "<cell4 poetiq>", "exec"), G)
    exec(compile(code[5], "<cell5 judge>", "exec"), G)
    exec(compile(code[6], "<cell6 core>", "exec"), G)
    if pre_run:
        pre_run(G)
    t0 = time.time()
    exec(compile(code[7], "<cell7 run>", "exec"), G)
    G["_WALL"] = time.time() - t0
    G["_WORK"], G["_FLOG"] = work, flog
    print(f"[{tag}] run cell wall {G['_WALL']:.0f} s", flush=True)
    return G


def fake_reqs(G):
    p = G["_FLOG"]
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def flag_val(f, k):
    return f[f.index(k) + 1] if k in f else None


def check_flags(G, name):
    a = G["E19_STATE"]["arms"][name]
    f, cfg = a["flags"], G["E19_ARMS"][name]
    s1 = (flag_val(f, "--tensor-parallel-size") == "4" and flag_val(f, "--distributed-executor-backend") == "mp"
          and flag_val(f, "--dtype") == "auto" and flag_val(f, "--load-format") == "safetensors" and "--language-model-only" in f
          and flag_val(f, "--max-num-seqs") == "16" and flag_val(f, "--max-num-batched-tokens") == "4096"
          and flag_val(f, "--gpu-memory-utilization") == "0.9" and "--disable-custom-all-reduce" in f
          and flag_val(f, "--seed") == "20261005" and flag_val(f, "--model") == str(G["MODEL_PATH"])
          and "--trust-remote-code" not in f and "--enforce-eager" not in f and "--enable-prompt-tokens-details" in f
          and flag_val(f, "--attention-backend") in ("FLASHINFER", "TRITON_ATTN"))
    arm = (flag_val(f, "--kv-cache-dtype") == cfg["kv"]
           and (("--enable-prefix-caching" in f) == cfg["prefix"]) and (("--no-enable-prefix-caching" in f) == (not cfg["prefix"]))
           and ((json.loads(flag_val(f, "--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 3})
                if cfg["spec"] else "--speculative-config" not in f)
           and (("--no-async-scheduling" in f) == (cfg.get("async") is False)) and "--async-scheduling" not in f)
    return s1 and arm, f


def common_checks(G, arms):
    V, ST = G["E19_VERDICTS"], G["E19_STATE"]
    rows = G["e19_load_rows"](G["E19_ROWS_PATH"])
    check(G["E19_ROWS_PATH"].exists() and (G["E19_DIR"] / "e19_state.json").exists() and (G["E19_DIR"] / "e19_report.md").exists()
          and (G["_WORK"] / "Q27X_E19_BUNDLE.zip").exists(), "rows / state / report / bundle written")
    check(ST["order"] == arms, f"arms in order {arms}", ST["order"])
    sub = json.load(open(G["_WORK"] / "submission.json", encoding="utf-8"))
    tc = json.load(open(Path(os.environ["Q27X_ARC_ROOT"]) / "arc-agi_test_challenges.json", encoding="utf-8"))
    check(set(sub) == set(tc) and all(a["attempt_1"] == q["input"] for t in tc for a, q in zip(sub[t], tc[t]["test"])),
          "submission.json = CELL 0's identity fallback (commit mode writes nothing else)")
    with zipfile.ZipFile(G["_WORK"] / "Q27X_E19_BUNDLE.zip") as z:
        check("e19/e19_rows.jsonl" in z.namelist() and "e19/e19_state.json" in z.namelist(), "bundle holds rows + state")
    reqs = [r for r in fake_reqs(G) if "len" in r]
    check(reqs and all(r["temperature"] == 0.0 and r["logprobs"] == 2 and r["return_token_ids"] is True
                       and r["return_tokens_as_token_ids"] is True and r["stream"] is True
                       and (r["stream_options"] or {}).get("include_usage") is True and r["model"] == "q27" for r in reqs),
          f"every request ({len(reqs)}): greedy, logprobs 2 by token id, return_token_ids, streaming, include_usage")
    okrows = [r for r in rows if r.get("status") == "ok"]
    check(okrows and all(isinstance(r.get("prompt_tokens"), int) and r.get("cached_tokens") is not None and r.get("ttft") is not None
                         and r.get("n_tokens") == len(r["tokens"]) and r.get("finish_reason") for r in okrows),
          "every ok row records prompt_tokens, cached tokens, TTFT, generated tokens, finish_reason")
    return V, ST, rows


def scen_arms(G, arms, expect=None):
    V, ST, rows = common_checks(G, arms)
    expect = expect or {a: "PASS" for a in arms}
    for name in arms:
        if "flags" in ST["arms"][name]:
            ok, f = check_flags(G, name)
            check(ok, f"arm {name}: flags = S1's ENGINE_CONFIG + its own switches ({G['E19_ARMS'][name]['what']})", f)
    d = ST["arms"].get("D")
    if d and "flags" in d:
        f = d["flags"]
        check(flag_val(f, "--kv-cache-dtype") == G["KV_CACHE_DTYPE"] and "--enable-prefix-caching" in f
              and json.loads(flag_val(f, "--speculative-config")) == G["SPECULATIVE_CONFIG"]
              and flag_val(f, "--max-model-len") == str(G["MAX_MODEL_LEN"]) and "--no-async-scheduling" not in f,
              "arm D = S1 exactly (CELL 1's KV_CACHE_DTYPE, SPECULATIVE_CONFIG, prefix caching, MAX_MODEL_LEN, async default)", f)
    for name, want in expect.items():
        v = V[name]
        check(v["final"] == want, f"arm {name}: final verdict {want}",
              (v["final"], v["in_arm"].get("reasons"), v["churn"].get("reasons")))
    by = {}
    for r in rows:
        if r.get("status") == "ok" and r["pass"] != "cont":
            by.setdefault((r["stage"], r["pass"], r["idx"]), set()).add(r["prompt_sha"])
    check(by and all(len(v) == 1 for v in by.values()), "identical prompt token ids across arms for every probe key",
          [k for k, v in by.items() if len(v) > 1][:5])
    return V, ST, rows


def main():
    if SCEN == "split":
        outs = []
        for nb_name in DEFAULTS:
            p = NB.parent / nb_name
            ok, info = profile_check(p)
            check(ok, f"{nb_name}: commit default {DEFAULTS[nb_name][0]} {DEFAULTS[nb_name][1]}, 60 min (hard stop 57)", info)
            G = run_nb(p, "normal", None, "split_" + DEFAULTS[nb_name][0])
            arms = [a for a, _ in DEFAULTS[nb_name][1]]
            check(G["E19_STATE"]["order"] == arms and all(G["E19_VERDICTS"][a]["final"] == "PASS" for a in arms),
                  f"{nb_name}: arms {arms} ran and PASS", {a: G["E19_VERDICTS"][a]["final"] for a in G["E19_VERDICTS"]})
            outs.append(G["_WORK"])
        sys.path.insert(0, str(SRC))
        import e19_compare
        cmp_dir = outs[0].parent / (outs[0].name + "_compare")
        res = e19_compare.main([str(o) for o in outs[:2]] + [str(outs[2] / "Q27X_E19_BUNDLE.zip"), "--out", str(cmp_dir)])
        check(sorted(res) == ALL, "compare: all six arms from three accounts (two folders + one bundle zip)", sorted(res))
        check(all(res[a]["final"] == "PASS" for a in res), "compare: every arm PASS", {a: res[a]["final"] for a in res})
        check(all(res[a]["churn"].get("mode") == "calibrated" for a in res) and res["D"]["churn"]["calibrators"] == ["A", "B"],
              "compare: churn calibrated for every arm (D by A and B; A by B, B by A)", {a: res[a]["churn"] for a in res})
        cx = res["D"]["vs_A_info"]["comparisons"]
        check(any(c["differ"] for c in cx.values()), "compare: D vs A token comparison present (informational, differs at near ties)", cx)
        check((cmp_dir / "e19_compare_report.md").exists(), "compare report written")
    elif SCEN == "rerun":
        G = run_nb(NB, "rerun", None, "rerun")
        check(not G["E19_ON"] and G["PROFILE"] == "RERUN_ALL", "rerun: E19 off, profile RERUN_ALL")
        check(not fake_reqs(G) and G["E19_SERVER"] is None, "rerun: no server launched, no request")
        sub = json.load(open(G["_WORK"] / "submission.json", encoding="utf-8"))
        tc = json.load(open(Path(os.environ["Q27X_ARC_ROOT"]) / "arc-agi_test_challenges.json", encoding="utf-8"))
        check(all(a["attempt_1"] == q["input"] for t in tc for a, q in zip(sub[t], tc[t]["test"])), "rerun: identity submission")
    elif SCEN == "budget":
        def pre(G):
            G["P"] = dict(G["P"], e19_arms=[["D", 0], ["C", 0.1]])
            G["HARD_STOP"] = time.perf_counter() + 30.0
        G = run_nb(NB.parent / "Q27-XHIGH-E19_MTPPC_P1_D_C.ipynb", "slow", None, "budget", pre)
        V, ST, rows = common_checks(G, ["D", "C"])
        rd = [r for r in rows if r["arm"] == "D"]
        nok, nb_ = sum(r["status"] == "ok" for r in rd), sum(r["status"] == "not_tested_budget" for r in rd)
        check(nok > 0 and nb_ > 0, f"D started and was cut: {nok} rows ok, {nb_} not_tested_budget (partials kept)")
        check(V["D"]["final"] == "NOT_TESTED_BUDGET", "D: NOT_TESTED_BUDGET (never PASS)", (V["D"]["final"], V["D"]["in_arm"]["reasons"]))
        check(ST["arms"]["C"].get("not_launched", "").startswith("NOT_TESTED_BUDGET") and V["C"]["final"] == "NOT_TESTED_BUDGET",
              "C: not launched, NOT_TESTED_BUDGET", (ST["arms"]["C"].get("not_launched"), V["C"]["final"]))
        check(all(v["final"] != "PASS" for v in V.values()), "no arm PASS under the exhausted budget")
    else:
        G = run_nb(NB, SCEN, "E19ALL", SCEN)
        if SCEN == "normal":
            V, ST, rows = scen_arms(G, ALL)
            for a in ALL:
                j = V[a]["in_arm"]
                check(j["row_status"].get("ok") == j["planned_rows"], f"arm {a}: all {j['planned_rows']} planned rows ok", j["row_status"])
            for a in ("C", "D", "E", "K"):
                c = V[a]["in_arm"]["cache"]
                check(c["warm_hits"] == c["warm_rows"] > 0 and c["cold_hits"] == 0 and c["churn_hits"] > 0,
                      f"arm {a} (prefix ON): every warm / continuation row hit the cache, no serial cold row did, churn hits", c)
            for a in ("A", "B"):
                c = V[a]["in_arm"]["cache"]
                check(c["warm_hits"] + c["cold_hits"] + c["churn_hits"] == 0, f"arm {a} (prefix OFF): no cached tokens", c)
            gc = V["C"]["in_arm"]["cache"]["cont_generated_in_hit"]
            check(gc and all(x > 0 for x in gc), "arm C (MTP off): continuation hits include server-generated tokens", gc)
            sr = [r for r in rows if r["arm"] == "D" and r["stage"] == "serial"]
            check(all(r["reset"] == 200 for r in sr if r["pass"] in ("cold", "cold2")) and all(r["reset"] is None for r in sr if r["pass"] == "warm"),
                  "serial: /reset_prefix_cache before every cold row, never before a warm row")
            check(all(r["metric_delta"]["pc_hits"] == r["cached_tokens"] for r in sr),
                  "serial: /metrics prefix-cache hit deltas = the usage chunk's cached tokens")
            ch = [r for r in rows if r["arm"] == "D" and r["stage"] == "churn"]
            fk = [r for r in fake_reqs(G) if "len" in r]
            check(len(ch) == 24 and len({r["max_tokens"] for r in ch}) == 8 and min(r["max_tokens"] for r in ch) == 256
                  and max(r["max_tokens"] for r in ch) == 2048 and all(r["conc"] == 14 for r in ch),
                  "churn: 24 requests, 14 at once, max_tokens staggered over 8 lengths 256..2,048",
                  sorted((r["max_tokens"], r["conc"]) for r in ch))
            starts = sorted(r["t_start"] for r in ch)
            check(starts[-1] > starts[13], "churn: arrivals refilled after the first 14 (the batch kept changing)", starts)
            jd = V["D"]["in_arm"]["comparisons"]["churn"]
            check(jd["compared"] == 24 and jd["large_margin_flips"] == 0, "churn: all 24 rows compared with the same prompt alone, 0 large flips", jd)
            check(any(V[a]["in_arm"]["comparisons"]["churn"]["differ"] for a in ALL),
                  "churn: benign near-tie divergences exist and are tolerated", {a: V[a]["in_arm"]["comparisons"]["churn"]["differ"] for a in ALL})
            check(all(V[a]["churn"]["mode"] == "calibrated" for a in ALL) and V["D"]["churn"]["calibrators"] == ["A", "B"],
                  "churn calibrated by A and B in the six-arm run", {a: V[a]["churn"] for a in ALL})
            for a in ("B", "D", "E", "K"):
                acc = V[a]["in_arm"]["accept"]
                check(acc["windows"] > 0 and acc["collapsed"] == 0 and 0.4 < (acc["median"] or 0) < 0.7,
                      f"arm {a}: MTP acceptance windows recorded (median {acc['median']}), no collapse", acc)
            ea = ST["arms"]["E"]["log"]["async_lines"]
            check(any("disabled" in x for x in ea) and any("enabled" in x for x in ST["arms"]["D"]["log"]["async_lines"]),
                  "arm E's server logged async scheduling disabled, D's enabled", (ea, ST["arms"]["D"]["log"]["async_lines"]))
            lg = [r for r in rows if r["arm"] == "A" and r["stage"] == "long30k" and r["pass"] == "nat"]
            check(len(lg) == 6 and all((r["pf_len"] + 256) % 1600 == 0 and r["pf_len"] - G["E19_PROMPTS"][r["idx"]]["len"] >= 1024
                                       and r["prompt_len"] >= 30720 for r in lg),
                  "long30k: 6 prompts, prompt + filler >= 30k tokens, 256 tokens before a 1,600 block boundary")
            check(all(G["E19_STATE"]["arms"][a]["stop"]["gpu"]["released"] for a in ALL) and G["E19_SERVER"] is None,
                  "every server stopped between arms (GPU wait done)")
        elif SCEN == "launchfail":
            V, ST, rows = scen_arms(G, ALL, {"A": "PASS", "B": "LAUNCH_FAILED", "C": "PASS", "D": "PASS", "E": "PASS", "K": "PASS"})
            check("gpu_after_failed_launch" in ST["arms"]["B"] and "launch_tail" in ST["arms"]["B"],
                  "B: failed launch killed, GPU wait recorded, log tail kept")
            check(V["D"]["churn"]["calibrators"] == ["A"], "churn calibration falls back to A alone", V["D"]["churn"])
        elif SCEN == "death":
            V, ST, rows = scen_arms(G, ALL, {"A": "PASS", "B": "PASS", "C": "PASS", "D": "INCOMPLETE", "E": "PASS", "K": "PASS"})
            rd = [r for r in rows if r["arm"] == "D"]
            check(ST["arms"]["D"].get("server_died") and sum(r["status"] == "ok" for r in rd) >= 10
                  and sum(r["status"] == "server_dead" for r in rd) > 0,
                  f"D: server death recorded; finished rows kept, the rest server_dead",
                  {s: sum(r['status'] == s for r in rd) for s in ('ok', 'server_dead', 'error')})
            check(any("server died" in x for x in V["D"]["in_arm"]["reasons"]), "D: verdict reason names the server death")
        elif SCEN == "corrupt":
            V, ST, rows = scen_arms(G, ALL, {"A": "PASS", "B": "PASS", "C": "PASS", "D": "FAIL", "E": "PASS", "K": "PASS"})
            j = V["D"]["in_arm"]
            check(j["verdict"] == "FAIL" and j["cache_hit_corrupt"] and j["comparisons"]["cache_serial"]["large_margin_flips"] > 0,
                  "D in-arm FAIL: corrupt cache-hit rows + large-margin flips cold vs warm", j["reasons"])
        elif SCEN == "race":
            V, ST, rows = scen_arms(G, ALL, {"A": "PASS", "B": "FAIL", "C": "PASS", "D": "FAIL", "E": "PASS", "K": "FAIL"})
            check(V["D"]["churn"]["verdict"] == "FAIL" and V["D"]["churn"]["calibrators"] == ["A"],
                  "race: D's churn FAIL against A's noise (B disqualified as a calibrator)", V["D"]["churn"])
            check(all(V[a]["in_arm"]["comparisons"]["aa_serial"]["large_margin_flips"] == 0 for a in ALL),
                  "race: serial rows untouched (no batch changes)")
        elif SCEN == "collapse":
            V, ST, rows = scen_arms(G, ALL, {"A": "PASS", "B": "PASS", "C": "PASS", "D": "FAIL", "E": "PASS", "K": "PASS"})
            check(V["D"]["in_arm"]["accept"]["collapsed"] > 0 and not V["D"]["in_arm"]["corrupt_rows"]
                  and any("acceptance collapsed" in x for x in V["D"]["in_arm"]["reasons"]),
                  "collapse: D FAIL on the acceptance windows alone", V["D"]["in_arm"]["accept"])
        elif SCEN == "kvretry":
            V, ST, rows = scen_arms(G, ALL)
            check(flag_val(ST["arms"]["K"]["flags"], "--max-model-len") == "65536" and ST["arms"]["K"]["max_len"] == 65536,
                  "K: relaunched with --max-model-len 65,536 after the KV check failed", ST["arms"]["K"].get("flags"))
    print(f"\n[{SCEN}] {'ALL CHECKS PASS' if not fails else str(len(fails)) + ' failed: ' + '; '.join(fails)}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
