"""E29 rev 2b / 2c: the print-only LIVE progress lines (the first part of the payload module e29_nbx.py; the notebook
executes e29_nbx.py into its namespace). Stdlib only."""
import time


class LiveLog:
    """rev 2b (user 2026-11-02): the print-only LIVE progress lines (never touch a record / the submission). on_rec(rec) =
    one line per finished attempt: minutes, task, phase, train pass, exact vs gold (only with solutions), per-phase totals,
    queries correct + task-norm over the finished tasks from the CURRENT attempt_1 / attempt_2 selection (select = the
    notebook's build_submission rule); DIRECT: mode, tokens, grids found, attempt_2 hits. summary() = the 15-min line."""

    ORDER = ("A", "B1", "B2", "C", "C-DIRECT", "supply")

    def __init__(self, tasks, solutions=None, select=None, cand=None, t0=0.0, clock=None):
        import time as _t
        self.tasks, self.sol, self.select, self.cand, self.t0 = tasks or {}, solutions, select, cand, float(t0)
        self.clock, self.recs, self.ph, self.q = clock or _t.perf_counter, {}, {}, {}
        self.seen = set()                     # rev 2f: (task, branch, phase) already counted
        self.nq = sum(len(t.get("test") or []) for t in self.tasks.values())

    def _m(self):
        return (self.clock() - self.t0) / 60.0

    @staticmethod
    def phase(rec):
        p = str((rec.get("e28") or {}).get("phase") or "")
        if rec.get("e29_direct"):
            return "C-DIRECT"
        return "supply" if p[:1] == "S" else (p or {0: "A", 1: "C", 2: "B1"}.get(int(rec.get("branch") or 0), "X"))

    def _totals(self):
        out = []
        for p in self.ORDER:
            if p in self.ph or p == "A":
                d, a, e = self.ph.get(p, [0, 0, 0])
                out.append(f"{p} {d}{'/' + str(len(self.tasks)) if p == 'A' else ''} done, accepted {a}" + (f", exact {e}" if self.sol else ""))
        s = " | ".join(out)
        if self.sol and self.q:
            c = sum(v[0] for v in self.q.values())
            tn = 100.0 * sum(v[0] / v[1] for v in self.q.values() if v[1]) / len(self.q)
            s += f" | queries {c}/{self.nq} correct, task-norm {tn:.1f} % ({len(self.q)} tasks)"
        return s

    def on_rec(self, rec):
        tid = rec.get("task_id")
        task = self.tasks.get(tid) or {"test": []}
        n = len(task["test"])
        self.recs.setdefault(tid, {})[rec.get("branch")] = rec
        ph = self.phase(rec)
        gold = (self.sol or {}).get(tid) if self.sol else None
        subs = [s for s in rec.get("submissions") or [] if isinstance(s, dict)]
        if ph == "C-DIRECT":
            grids = list((subs[0].get("test_outputs") if subs else None) or [None] * n)[:n]
        else:
            grids = [((self.cand(rec, qi) if self.cand else None) or {}).get("grid") for qi in range(n)]
        bt = (rec.get("e28") or {}).get("best_train") or {}
        te = next((s.get("train_exact") for s in subs if s.get("accepted")), None) or (subs[-1].get("train_exact") if subs else []) or []
        k, kn = (bt.get("k"), bt.get("n")) if bt.get("n") else (sum(bool(x) for x in te), len(te))
        ex = sum(1 for qi in range(n) if gold and qi < len(gold) and grids[qi] == gold[qi]) if gold else None
        key = (tid, rec.get("branch"), ph)
        again = key in self.seen                  # rev 2f: a re-published record updates the selection, not the counts
        self.seen.add(key)
        st = self.ph.setdefault(ph, [0, 0, 0])
        if not again:
            st[0] += 1
            st[1] += int(bool(rec.get("accepted")))
            st[2] += int(bool(gold) and ex == n and n > 0)
        if gold and self.select:
            att = self.select(tid, task, list(self.recs[tid].values()))
            c = sum(1 for qi, a in enumerate(att) if qi < len(gold) and gold[qi] in (a["attempt_1"], a["attempt_2"]))
            h = sum(1 for qi, a in enumerate(att) if qi < len(gold) and a["attempt_1"] != gold[qi] and a["attempt_2"] == gold[qi])
            self.q[tid] = (c, n, h)
        if ph == "C-DIRECT":
            x = rec.get("e31") or {}
            head = (f"{x.get('mode') or x.get('end_reason')} {int(rec.get('generated_tokens') or 0):,} tok  grid "
                    f"{sum(g is not None for g in grids)}/{n}")
        else:
            head = f"{'accepted' if rec.get('accepted') else 'not accepted'}(train {k}/{kn})"
        line = (f"[LIVE {self._m():.1f} min] {tid} {ph}{' (again)' if again else ''}  {head}" + (f" exact {ex}/{n}" if gold else "")
                + " | " + self._totals())
        if ph == "C-DIRECT" and gold:
            line += f" | attempt_2 hits {sum(v[2] for v in self.q.values())}"
        return line

    def summary(self, tok_s=None, running=None, tail=""):
        return (f"[LIVE 15-min {self._m():.1f} min] tok/s {('%.1f' % tok_s) if tok_s is not None else '?'} | running "
                f"{running if running is not None else '?'} | " + self._totals() + (tail or ""))
