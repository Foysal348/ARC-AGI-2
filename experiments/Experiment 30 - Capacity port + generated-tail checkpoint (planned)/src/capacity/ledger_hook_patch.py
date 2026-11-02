"""Capacity-ledger hook seam: MERLIN_CAP_LEDGER_EVERY=<N> logs one "CAPACITY-LEDGER {json}" line every N scheduler
steps (capacity_ledger.live_ledger). Unset = one os.environ lookup per step, nothing else (no import, no log).

  vllm/v1/core/sched/scheduler.py  Scheduler.schedule (pristine :705-706): 3 lines after `self.current_step += 1`
  vllm/v1/core/capacity_ledger.py  NEW file = capacity_ledger.py from this folder (revert deletes it)

CLI:  python ledger_hook_patch.py apply|verify|revert <dir that contains the vllm package>
"""
import hashlib, sys, py_compile, shutil
from pathlib import Path

MARK = "CAPACITY-PORT LEDGER"
REL = "vllm/v1/core/sched/scheduler.py"
NEW_FILE = "vllm/v1/core/capacity_ledger.py"
PRE = "ea56781115c350cf0c053912bc855f86d6bb47a19cd914e1d2e045b99b2a4a7f"
OLD = '''    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
'''
NEW = '''    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        if __import__("os").environ.get("MERLIN_CAP_LEDGER_EVERY"):  # CAPACITY-PORT LEDGER
            from vllm.v1.core.capacity_ledger import ledger_tick  # CAPACITY-PORT LEDGER
            ledger_tick(self)  # CAPACITY-PORT LEDGER
'''
HERE = Path(__file__).resolve().parent


def sha(b):
    return hashlib.sha256(b).hexdigest()


def state(tree):
    p = Path(tree) / REL
    b = p.read_bytes()
    nf = Path(tree) / NEW_FILE
    return {REL: {"sha": sha(b), "state": "PRISTINE" if sha(b) == PRE else
                  ("PATCHED_MARKED" if MARK.encode() in b else "FOREIGN")},
            NEW_FILE: {"present": nf.exists(), "sha": sha(nf.read_bytes()) if nf.exists() else None,
                       "matches_source": nf.exists() and nf.read_bytes() == (HERE / "capacity_ledger.py").read_bytes()}}


def apply(tree):
    p = Path(tree) / REL
    t = p.read_text(encoding="utf-8")
    if MARK not in t:
        if t.count(OLD) != 1:
            raise SystemExit("scheduler anchor not unique, refusing")
        p.write_bytes(t.replace(OLD, NEW).encode("utf-8"))
        py_compile.compile(str(p), doraise=True)
        for pyc in p.parent.glob(f"__pycache__/{p.stem}.*.pyc"):
            pyc.unlink()
    shutil.copyfile(HERE / "capacity_ledger.py", Path(tree) / NEW_FILE)
    py_compile.compile(str(Path(tree) / NEW_FILE), doraise=True)
    return {REL: sha(p.read_bytes())}


def revert(tree):
    p = Path(tree) / REL
    t = p.read_text(encoding="utf-8")
    if MARK in t:
        if t.count(NEW) != 1:
            raise SystemExit("patched block not intact, refusing")
        p.write_bytes(t.replace(NEW, OLD).encode("utf-8"))
        for pyc in p.parent.glob(f"__pycache__/{p.stem}.*.pyc"):
            pyc.unlink()
    nf = Path(tree) / NEW_FILE
    if nf.exists():
        nf.unlink()
    return {REL: sha(p.read_bytes())}


if __name__ == "__main__":
    cmd, tree = sys.argv[1], sys.argv[2]
    if cmd == "apply":
        print(apply(tree))
    elif cmd == "revert":
        print(revert(tree))
    print(state(tree))
