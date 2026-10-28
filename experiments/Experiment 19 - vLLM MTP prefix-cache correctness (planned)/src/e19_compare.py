"""E19 offline verdicts across the three accounts' outputs. The arms run on different Kaggle accounts, so the churn
calibration by arms A / B (and the informational comparison against A) happens here. Uses the SAME rules as the
notebook (e19_judge.py is the notebook's E19-J cell).

usage: python e19_compare.py <input> [<input> ...] [--out DIR]
  <input> = a Kaggle output folder (contains q27x/e19/), an e19 folder (contains e19_rows.jsonl + e19_state.json), or a
            Q27X_E19_BUNDLE.zip
Writes e19_compare_report.md and e19_compare.json to --out (default: the current folder)."""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e19_judge import e19_report_lines, e19_verdicts  # noqa: E402


def _rows_from_text(txt):
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass          # a torn last line after a crash
    return rows


def load_input(p):
    p = Path(p)
    if p.is_file() and p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            rn = next((n for n in names if n.endswith("e19_rows.jsonl")), None)
            sn = next(n for n in names if n.endswith("e19_state.json"))
            return (_rows_from_text(z.read(rn).decode("utf-8")) if rn else []), json.loads(z.read(sn).decode("utf-8"))
    for d in (p, p / "e19", p / "q27x" / "e19"):
        if (d / "e19_state.json").exists():
            rp = d / "e19_rows.jsonl"
            return (_rows_from_text(rp.read_text(encoding="utf-8")) if rp.exists() else []), \
                json.loads((d / "e19_state.json").read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no e19_state.json under {p}")


def main(argv):
    out_dir = Path(".")
    if "--out" in argv:
        k = argv.index("--out")
        out_dir = Path(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    metas, rows, prompts, src, cfgs, sarms = {}, [], None, {}, {}, {}
    for inp in argv:
        r_, state = load_input(inp)
        if prompts is None:
            prompts = state.get("prompts")
        elif [x["sha"] for x in prompts] != [x["sha"] for x in state.get("prompts") or []]:
            raise SystemExit(f"{inp}: its prompt token ids differ from the first input's: not comparable")
        for name in state.get("order") or []:
            a = state["arms"][name]
            if name in metas:
                print(f"warning: arm {name} appears in {src[name]} and {inp}; keeping the first")
                continue
            if not a.get("meta"):
                print(f"warning: arm {name} in {inp} has no meta (the run ended before the arm closed); skipped")
                continue
            metas[name], src[name], cfgs[name], sarms[name] = a["meta"], str(inp), a.get("cfg") or {}, a
            rows += [x for x in r_ if x.get("arm") == name]
    V = e19_verdicts(rows, dict(sorted(metas.items())))
    for n in V:
        V[n]["source"] = src[n]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e19_compare.json").write_text(json.dumps(V, indent=1, default=str), encoding="utf-8")
    L = e19_report_lines(V, cfgs, sarms, "# E19 cross-account verdicts")
    L += ["", "Inputs: " + "; ".join(f"{n} <- {src[n]}" for n in sorted(src))]
    (out_dir / "e19_compare_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return V


if __name__ == "__main__":
    main(sys.argv[1:])
