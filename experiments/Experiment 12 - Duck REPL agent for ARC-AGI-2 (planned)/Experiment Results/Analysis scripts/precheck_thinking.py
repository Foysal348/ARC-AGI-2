"""What the model does before its first check(): features of all THINKING text up to (and including) the response whose
tool call first contains check(. Per branch, then fast (first check < 35k tokens) vs slow (> 55k).
Usage: python -I precheck_thinking.py <q27x dir> [...]"""
import glob, json, os, re, sys, statistics as st
sys.stdout.reconfigure(encoding="utf-8")
HDR = re.compile(r"^\[([A-Z][A-Za-z _:]*)\]\s*$")
GRIDROW = re.compile(r"^\s*(?:[-*]\s*)?(?:(?:row|r)\s*\d+\s*[:=]\s*)?(?:\d\s*[, ]\s*){5,}\d\s*$", re.I)
COORD = re.compile(r"\(\s*\d{1,2}\s*,\s*\d{1,2}\s*\)")
SWITCH = re.compile(r"\b(wait|hmm|actually|let me reconsider|alternatively|no,|that doesn't|doesn't match|not quite|let me re-?check|re-?examine)\b", re.I)
TEST = re.compile(r"\btest (input|grid)\b", re.I)
CODEINTHINK = re.compile(r"^\s*(def |for |if |return |[a-z_]+\s*=\s*\[)", re.M)
VERIFY = re.compile(r"(✓|matches|correct!|consistent|that works|confirmed)", re.I)
rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "duck", "*", "transcript.txt"))):
        tx = open(f, encoding="utf-8", errors="replace").read()
        br = json.load(open(os.path.join(os.path.dirname(f), "branch_result.json"), encoding="utf-8"))
        comp = [u["completion"] for u in br.get("usage") or []]
        blocks = []; cur = None; buf = []
        for line in tx.splitlines() + ["[END]"]:
            m = HDR.match(line)
            if m:
                if cur: blocks.append((cur, buf))
                cur, buf = m.group(1), []
            else:
                buf.append(line)
        think, n_resp, tok, reached = [], 0, 0, False
        tools_before = 0
        for name, b in blocks:
            if name == "MODEL RESPONSE META":
                if n_resp < len(comp): tok += comp[n_resp]
                n_resp += 1
                if "check(" in "\n".join(b): reached = True
            elif name == "THINKING":
                think.extend(b)
            elif name.startswith("TOOL CALL"):
                tools_before += 1
            if reached and name.startswith("TOOL CALL"):
                break
        text = "\n".join(think); ch = max(1, len(text))
        grid_chars = sum(len(l) for l in think if GRIDROW.match(l))
        rows.append(dict(run=os.path.basename(os.path.dirname(d))[:22], b=os.path.basename(os.path.dirname(f)), reached=reached, tok=tok,
                         resp=n_resp, tools=tools_before, chars=len(text),
                         grid=grid_chars / ch, coord=len(COORD.findall(text)) * 1000 / ch, switch=len(SWITCH.findall(text)) * 1000 / ch,
                         test=len(TEST.findall(text)) * 1000 / ch, code=len(CODEINTHINK.findall(text)) * 1000 / ch,
                         verify=len(VERIFY.findall(text)) * 1000 / ch))
def show(label, S):
    if not S: return
    f = lambda k: st.median(r[k] for r in S)
    print(f"{label:28} n={len(S):2} tokens {f('tok'):7.0f} resp {f('resp'):4.1f} tool calls {f('tools'):4.1f} | grid-row text {100*f('grid'):4.1f}% "
          f"| (r,c) coords/1k {f('coord'):5.2f} | wait/hmm/actually/1k {f('switch'):4.2f} | 'test input'/1k {f('test'):4.2f} | code lines in thinking/1k {f('code'):4.2f} | verify words/1k {f('verify'):4.2f}")
show("all branches", rows)
show("reached check() fast (<35k)", [r for r in rows if r["reached"] and r["tok"] < 35000])
show("reached check() 35-55k", [r for r in rows if r["reached"] and 35000 <= r["tok"] <= 55000])
show("reached check() slow (>55k)", [r for r in rows if r["reached"] and r["tok"] > 55000])
show("never reached check()", [r for r in rows if not r["reached"]])
print()
for r in sorted(rows, key=lambda r: r["tok"]):
    print(f"  {r['b']:12} {'chk' if r['reached'] else '---'} tok {r['tok']:6} resp {r['resp']:2} tools {r['tools']:2} grid {100*r['grid']:4.1f}% coord {r['coord']:5.2f} switch {r['switch']:4.2f} test {r['test']:4.2f} code {r['code']:4.2f}")
