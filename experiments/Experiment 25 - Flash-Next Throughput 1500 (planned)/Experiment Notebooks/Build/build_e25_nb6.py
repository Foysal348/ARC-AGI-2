"""E25 NB6 SUBMIT: the production submission notebook (lead 2026-11-01). A thin wrapper over build_e25_nb5.py: it builds
the NB5 cells of the Kaggle-run NB5 QUALITY notebook (sha 9c71ba89; this script first re-derives that sha from the NB5
builder, so the verbatim NB5 code is exactly what ran on GPU) and applies only the marked NB6 edits below:

  E1 (CELL 1) the hidden rerun profile RERUN_ALL -> RERUN_PROD (PROFILE = "RERUN_PROD" if RERUN else ...)
  E2 (CELL 1) NB6_LEVERS / NB6_COMMIT_TASKS defined before PROFILE_TABLE; RERUN_PROD + NB6COMMIT appended to the table
  E3 (CELL 1) default commit profile = NB6COMMIT (the __PROFILE__ token of the NB5 builder)
  E4 (CELL 6/7) E12_B_B1_ACC_MODE 0 -> b1_accepted_mode "" (= solve_all_r6's default, as E15B SUBMIT calls it)
  E5 (markdown) the NB6 header
  E7 (engine, nb5_rc2_inputs) the RC2 probe-prompt file through ARC_ROOT; rerun without the evaluation file -> the test file
  P1 (engine, MServer) E18 R2's wait_ready (health + own KV line + 600-s log-silence stall) and kill(fast)
  P2 (engine, with_servers) rerun after the solve: E18 R2's fast kill, no release wait
  P3 (engine, nb5_boot) E18 R2's engine ladder: any failure before solving -> next step, <= 3 attempts, 30 / 15 / 15 min
  P5 (engine, e21_relaunch) E18 R2's 20-min relaunch wait
  P6 (CELL 8) no sha gate in the rerun (E18 R2 / E15B SUBMIT have none); commit keeps NB5's
Proven-only rule (user order 2026-11-01): rerun rules come from E18 R2 (LB 31.53) and E15B SUBMIT (LB 22.92); only the
Merlin boot (NB5 QUALITY's GPU-run engine config) and the Duck solver are NB5's.
Nothing else changes (CELL 0 dual-mode + fallback submission, the engine, the runner, the report: NB5 verbatim).

usage: python build_e25_nb6.py   ->  ../Q27-XHIGH-E25_NB6_SUBMIT.ipynb + nb6_code_export.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e25_nb5 as B   # noqa: E402  (module import only: build_e25_nb5's build() is never called here)

NB5_QUALITY_SHA = "9c71ba89e4db75b5f9a63a4be652e860b629120685e8f534bed3bb8d06769049"
OUT = B.NBDIR / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb"
EXPORT = HERE / "nb6_code_export.py"
DEFAULT_PROFILE = "NB6COMMIT"
PUSHED: set = set()   # sha256 of every NB6 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)


def notebook_bytes(cl, indent=0):   # NB6: indent=0 (same JSON; NB6 is ~403 kB at indent=1)
    nb_cells = []
    for kind, text in cl:
        text = B.norm(text)
        if kind == "code":
            compile(text, "<nb6>", "exec")
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return (json.dumps(nb, indent=indent, ensure_ascii=False) + "\n").encode("utf-8"), len(nb_cells)


# ---- 0. the NB5 builder still produces the Kaggle-run NB5 QUALITY notebook byte for byte
_nb5_bytes, _ = notebook_bytes(B.cells("NB5QUALITY_R2"), indent=1)
assert hashlib.sha256(_nb5_bytes).hexdigest() == NB5_QUALITY_SHA, "build_e25_nb5.py no longer reproduces NB5 QUALITY 9c71ba89"
NB5_CELL1, NB5_CELL67, NB5_MD0 = B.CELL1, B.CELL67, B.MD0

# ---- E1
_E1_OLD = 'PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", NB5_DEFAULT_PROFILE).strip().upper()'
_E1_NEW = ('PROFILE = "RERUN_PROD" if RERUN else os.environ.get("Q27X_PROFILE", NB5_DEFAULT_PROFILE).strip().upper()'
           '   # NB6 E1: the hidden rerun = RERUN_PROD')
# ---- E2
NB6_DEFS = r'''# ---- NB6 E2 (production SUBMISSION, lead 2026-11-01): the levers of NB5QUALITY_R2 (Kaggle-run 2026-11-01, sha 9c71ba89,
# FP8 phase 13 / 15, pool 214, 9 workers): FP8 KV, MTP K3, commit variant A, KV pin ladder (RC2 pin + 1.0 GiB per card =
# 3,553,580,524 B first, then 0.75 / 0.5 / 0), #60490, measured F, slots admission, launch seqs table, CAPS16 captures,
# MERLIN_SHARE_DRAFT_WORKSPACE=1, gate 1.0, 104k per attempt. Attempt policy = E15B SUBMIT (LB 22.92): its
# solve_all_r6 call passes no skip / extra / b1-mode argument -> skip_b1_if_accepted False, extra_attempts False,
# b1_accepted_mode "" (two attempts per task, plain priority). No fixed LOCAL_ANALYZER_SEED (E15B SUBMIT has none).
# Engine attempts = E18 R2's ladder (e18_ladder: <= 3, any failure before solving -> the next pin step, 30 / 15 / 15 min).
NB6_COMMIT_TASKS = ["1818057f", "2ba387bc", "45a5af55"]      # NB5 QUALITY's solid group, solved in both phases
NB6_LEVERS = dict(nb5_gate=1.0, nb5_attempt={"b0": 104000, "b1": 104000},
                  nb5_env={"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"},
                  admit_mode="slots", f_rule="measured", pin_ladder=True, patch_60490=True, seqs_rule="table", caps16=True,
                  lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, no_pure_decode=True, continue_after_failed_boot=True,
                  e18_ladder=True,          # NB6 P3: E18 R2 cell 6 L468-507
                  nb5_phases=[["EVAL", "all", None, "on", "A", "fp8"]])
'''
_E2_TABLE_OLD = "PROFILE_TABLE = {\n"
_E2_END_OLD = ('                          nb5_phases=[["Q_FP8_K3", "all", None, "on", "A", "fp8"], ["Q_BF16_K3", "all", None, "on", "A", "bf16"]]),\n'
               '}\n')
_E2_END_NEW = ('                          nb5_phases=[["Q_FP8_K3", "all", None, "on", "A", "fp8"], ["Q_BF16_K3", "all", None, "on", "A", "bf16"]]),\n'
               '    # NB6 E2: the hidden rerun (every hidden test task, ORDER2, one EVAL phase); E18 R2 RERUN_ALL budget 12 h - 3 min\n'
               '    "RERUN_PROD": dict(_NB5_T, time_budget_sec=12.0 * 3600 - 3 * 60, source="test", n_tasks=None, **NB6_LEVERS),\n'
               '    # NB6 E2: the commit check before Submit: 3 fast eval tasks, the same levers, scored; 50-min budget (hard stop\n'
               '    # 48 min, solve end 45 min = ~26 min of solving after a ~17-min FP8 boot as in NB5 QUALITY; F8: a boot ready by\n'
               '    # ~33 min still passes the phase budget check)\n'
               '    "NB6COMMIT": dict(_NB5_T, time_budget_sec=50 * 60, source="eval", task_list=NB6_COMMIT_TASKS, **NB6_LEVERS),\n'
               '}\n')
# ---- E4
_E4_OLD = 'b1_accepted_mode={1: "after3", 2: "before3"}[int(E15B["E12_B_B1_ACC_MODE"])],'
_E4_NEW = 'b1_accepted_mode={0: "", 1: "after3", 2: "before3"}[int(E15B["E12_B_B1_ACC_MODE"])],   # NB6 E4: 0 -> "" (E15B SUBMIT)'
# ---- E5
MD0_NB6 = """# E25 NB6 SUBMIT · FN-Duck production submission · __PROFILE__

- **Rerun rules from the proven notebooks only** (user order 2026-11-01): server readiness / ladder / final kill / clock /
  CELL 0 / budget = E18 R2 (LB 31.53); Duck submission writer, task set, order, sandbox fail-closed, attempt policy =
  E15B SUBMIT (LB 22.92). Merlin boot = NB5 QUALITY's GPU-run engine config (sha 9c71ba89); Duck solver = NB5's.
- **Hidden rerun = RERUN_PROD:** every hidden test task, ORDER2, FP8 KV, MTP K3, variant A, pin ladder from RC2 + 1.0 GiB,
  #60490, measured F, slots, seqs table, CAPS16, MERLIN_SHARE_DRAFT_WORKSPACE=1, 104k per attempt, 2 attempts per task,
  717-min budget. **Commit = NB6COMMIT:** 3 fast eval tasks, the same settings, scored, 50 min.
"""


# ---- E7 (engine, nb5_rc2_inputs) F1: RC2's probe prompts read the evaluation file; the rerun never needs it
_E7_OLD = '    ARC = COMP / "arc-agi_evaluation_challenges.json"\n    if not ARC.is_file():\n'
_E7_NEW = ('    ARC = COMP / "arc-agi_evaluation_challenges.json"\n'
           '    for _nb6_arc in [ARC_ROOT / "arc-agi_evaluation_challenges.json"] + ([ARC_ROOT / "arc-agi_test_challenges.json"] if RERUN else []):\n'
           '        if not ARC.is_file() and _nb6_arc.is_file():   # NB6 E7: CELL 0\'s ARC_ROOT; rerun: the test file (train pairs) for the probe prompts\n'
           '            ARC = _nb6_arc\n'
           '    if not ARC.is_file():\n')
# ---- P1 (engine, MServer): E18 R2's E18Server.wait_ready (cell 6 L143-162: /health + this server's own "GPU KV cache
# size" line, a 600-s log-silence stall rule) and kill(fast) (L164-185), on the Merlin server class
_P1_OLD = ('class MServer(L.Server):\n'
           '    """Merlin passes --shutdown-timeout 60 and the PLE transport unlinks its /dev/shm storages only on a graceful\n'
           '    teardown: SIGTERM, wait 75 s, then E23\'s kill (SIGTERM + 40 s + SIGKILL of the whole session)."""\n'
           '    def kill(self):\n'
           '        if self.proc is not None and self.proc.poll() is None:\n')
_P1_NEW = ('E18_SILENCE_SEC = float(os.environ.get("E18_SILENCE_SEC", 600))   # NB6 P1: E18 R2 cell 6 (B-M3)\n\n\n'
           'class MServer(L.Server):\n'
           '    """Merlin passes --shutdown-timeout 60 and the PLE transport unlinks its /dev/shm storages only on a graceful\n'
           '    teardown: SIGTERM, wait 75 s, then E23\'s kill (SIGTERM + 40 s + SIGKILL of the whole session)."""\n'
           '    def wait_ready(self, timeout, silence=E18_SILENCE_SEC):\n'
           '        """NB6 P1 = E18 R2 E18Server.wait_ready: ready = /health answers AND this server\'s own "GPU KV cache size"\n'
           '        line is in its log (B-L8); a launch whose log stops growing for `silence` seconds is a stall (B-M3)."""\n'
           '        last, last_t = -1, time.perf_counter()\n'
           '        while time.perf_counter() - self.t0 < timeout:\n'
           '            if not self.alive():\n'
           '                return False, f"exited rc={self.proc.returncode}"\n'
           '            try:\n'
           '                _urlreq.urlopen(self.url + "/health", timeout=10).read()\n'
           '                if "GPU KV cache size" in self.log():\n'
           '                    self.ready_sec = round(time.perf_counter() - self.t0, 1)\n'
           '                    return True, f"ready in {time.perf_counter() - self.t0:.0f}s"\n'
           '            except Exception:\n'
           '                pass\n'
           '            sz = self.log_path.stat().st_size if self.log_path.exists() else 0\n'
           '            if sz != last:\n'
           '                last, last_t = sz, time.perf_counter()\n'
           '            elif time.perf_counter() - last_t > silence:\n'
           '                return False, f"stall ({silence:.0f} s of log silence)"\n'
           '            time.sleep(1 if FAKE else 5)\n'
           '        return False, "timeout"\n\n'
           '    def kill(self, fast=False):\n'
           '        if fast and self.proc is not None:     # NB6 P1 = E18 R2 E18Server.kill(fast=True): at once after the rerun\'s solve\n'
           '            try:\n'
           '                os.killpg(self.proc.pid, 15) if hasattr(os, "killpg") else self.proc.terminate()\n'
           '            except Exception:\n'
           '                pass\n'
           '            try:\n'
           '                self.proc.wait(timeout=0.1)\n'
           '            except Exception:\n'
           '                pass\n'
           '            try:\n'
           '                os.killpg(self.proc.pid, 9) if hasattr(os, "killpg") else self.proc.kill()\n'
           '            except Exception:\n'
           '                pass\n'
           '            try:\n'
           '                self.proc.wait(timeout=30)\n'
           '            except Exception:\n'
           '                pass\n'
           '            return\n'
           '        if self.proc is not None and self.proc.poll() is None:\n')
# ---- P2 (engine, with_servers): E18 R2's final behaviour in the rerun (cell 8b L173: kill(fast=RERUN), no release wait)
_P2_OLD = ('    finally:\n        for s in servers:\n            s.kill()\n'
           '        gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))\n'
           '        rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)\n')
_P2_NEW = ('    finally:\n'
           '        _nb6_final = RERUN and bool((globals().get("NB5_REPORT") or {}).get("phases"))   # NB6 P2: the rerun\'s solve ran\n'
           '        for s in servers:\n            s.kill(fast=True) if _nb6_final else s.kill()       # NB6 P2: E18 R2 kill(fast=RERUN)\n'
           '        gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))\n'
           '        rel = True if _nb6_final else L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)   # NB6 P2: E18 has no wait after it\n')
# ---- P3 (engine, nb5_boot): E18 R2's engine ladder (cell 6 L479-507): any failure before a phase -> the next step, at most
# 3 attempts, 30 min for the first and 15 min for each later one; a GPU-release failure stops (E18 'fatal')
_P3_OLD = ('        nb5_boot_try(trec, name, tr["kv_bytes"], phases, mtp_on)\n'
           '        fail = NB5L.boot_failure(trec.get("status"), trec.pop("log_text_for_ladder", ""), levers, trec.get("phases_started", False))\n')
_P3_NEW = ('        globals()["NB6_READY_SEC"] = ((30 if len(brec["tries"]) == 1 else 15) * 60 if P.get("e18_ladder")\n'
           '                                      else NB5_READY_TIMEOUT)          # NB6 P3: E18 R2 cell 6 L485\n'
           '        nb5_boot_try(trec, name, tr["kv_bytes"], phases, mtp_on)\n'
           '        fail = NB5L.boot_failure(trec.get("status"), trec.pop("log_text_for_ladder", ""), levers, trec.get("phases_started", False))\n'
           '        if P.get("e18_ladder"):   # NB6 P3: E18 R2 retries on ANY failed attempt (not ready, stall, smoke) before solving\n'
           '            fail["failed"] = trec.get("status") != "DONE" and not trec.get("phases_started", False)\n')
_P3B_OLD = '        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why"):\n            break\n'
_P3B_NEW = ('        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and len(brec["tries"]) >= 3):\n'
            '            break                                  # NB6 P3: E18 R2 ladder = at most 3 engine attempts\n')
_P4_OLD = '    o = with_servers(NB5_SERVERS, NB5_READY_TIMEOUT, body)\n'
_P4_NEW = '    o = with_servers(NB5_SERVERS, globals().get("NB6_READY_SEC", NB5_READY_TIMEOUT), body)   # NB6 P3\n'
# ---- P5 (engine, e21_relaunch): E18 R2's watchdog relaunch waits 20 min (cell 6 L266)
_P5_OLD = '        ok, why = srv.wait_ready(max(60.0, min(NB5_READY_TIMEOUT, t)))\n'
_P5_NEW = '        ok, why = srv.wait_ready(max(60.0, min(20 * 60, t)))   # NB6 P5: E18 R2 relaunch wait_ready(20 * 60)\n'
# ---- P6 (CELL 8): E18 R2 / E15B SUBMIT have no sha gate: none in the rerun (commit keeps NB5's)
_P6_OLD = '    _ok, _why = sha_gate(min(1500, left() - 900))\n'
_P6_NEW = ('    _ok, _why = (sha_gate(min(1500, left() - 900)) if not RERUN\n'
           '                 else (True, "rerun: no sha gate (E18 R2 / E15B SUBMIT have none)"))   # NB6 P6\n')


def post_edits(cl):
    """E7 + P1-P6 on the built cells (positions = NB5's layout: 5 engine, 7 runner)."""
    cl = list(cl)
    k, t = cl[5]
    for o, n, lab in ((_E7_OLD, _E7_NEW, "E7"), (_P1_OLD, _P1_NEW, "P1"), (_P2_OLD, _P2_NEW, "P2"), (_P3_OLD, _P3_NEW, "P3"),
                      (_P3B_OLD, _P3B_NEW, "P3b"), (_P4_OLD, _P4_NEW, "P4"), (_P5_OLD, _P5_NEW, "P5")):
        t = B.once(t, o, n, "NB6 " + lab)
    cl[5] = (k, t)
    k, t = cl[7]
    cl[7] = (k, B.once(t, _P6_OLD, _P6_NEW, "NB6 P6"))
    return cl


def nb6_parts():
    c1 = B.once(NB5_CELL1, _E1_OLD, _E1_NEW, "NB6 E1 rerun profile")
    c1 = B.once(c1, _E2_TABLE_OLD, NB6_DEFS + _E2_TABLE_OLD, "NB6 E2 levers")
    c1 = B.once(c1, _E2_END_OLD, _E2_END_NEW, "NB6 E2 profiles")
    c67 = B.once(NB5_CELL67, _E4_OLD, _E4_NEW, "NB6 E4 skip mode")
    return c1, c67


def cells(profile=DEFAULT_PROFILE):
    c1, c67 = nb6_parts()
    B.CELL1, B.CELL67, B.MD0 = c1, c67, MD0_NB6
    try:
        return post_edits(B.cells(profile))   # NB5's cell list (profile != NB5QUALITY_R2: no R2 markdown addendum)
    finally:
        B.CELL1, B.CELL67, B.MD0 = NB5_CELL1, NB5_CELL67, NB5_MD0


def build():
    cl = cells(DEFAULT_PROFILE)
    B.undefined_names_check(cl)
    data, n = notebook_bytes(cl)
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    out = OUT
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not (out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() in PUSHED), out
    out.write_bytes(data)
    EXPORT.write_text("\n\n".join(B.norm(t) for k, t in cl if k == "code"), encoding="utf-8", newline="\n")
    print("wrote", out.name, "sha256", hashlib.sha256(data).hexdigest(), "|", len(data), "bytes |", n, "cells")
    return out


if __name__ == "__main__":
    build()
