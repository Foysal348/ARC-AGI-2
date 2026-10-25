"""Cross-run validation of the two reader reports (2026-10-21).
Per response (MODEL RESPONSE META block): reasoning chars, tool call count, finish reason,
code-intent phrases, self-detected misread phrases, whether the call defines transform / calls check,
whether the call compares against p['output'] itself. Counts only responses BEFORE the first check() call."""
import re, json, os, sys, glob, statistics as st
RUNS = {
 'xhigh': r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Results/Kaggle Runs/E12 SGLang EVAL6R5 run1 (hosen42, 2026-10-20)/q27x/duck",
 'low':   r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Results/Kaggle Runs/E12 SGLang EVAL12R5_LOW run1 (foysal, 2026-10-20)/q27x/duck",
}
INTENT = re.compile(r"(in code|with code|use python|using python|let me (just )?(write|code|compute|run|verify)[^.\n]{0,40}(code|python|program)|code this up|stop guessing|stop hand|compute (this|it) (with|in) (code|python)|let me (just )?code)", re.I)
MISREAD = re.compile(r"(miscount|misread|my (manual|hand)[- ]?count|i made an error|i made a mistake|counted wrong|wrong count|recount|let me re-?read|completely different from my|i misremembered)", re.I)
MARK = re.compile(r"^\[([A-Z _:a-z0-9-]+)\]\s*$", re.M)

def blocks(text):
    ms = list(MARK.finditer(text))
    for i, m in enumerate(ms):
        end = ms[i+1].start() if i+1 < len(ms) else len(text)
        yield m.group(1), text[m.end():end]

def responses(text):
    """Group: META then THINKING/ASSISTANT/TOOL CALL until next META or SYSTEM/USER/TOOL RESULT."""
    cur = None
    for tag, body in blocks(text):
        if tag == 'MODEL RESPONSE META':
            if cur: yield cur
            meta = dict(re.findall(r"^(\w+): (.*)$", body, re.M))
            cur = {'meta': meta, 'think': '', 'calls': []}
        elif cur is None:
            continue
        elif tag == 'THINKING':
            cur['think'] += body
        elif tag.startswith('TOOL CALL'):
            cur['calls'].append(body)
        elif tag.startswith('TOOL RESULT'):
            cur.setdefault('results', []).append(body)
    if cur: yield cur

out = {}
for run, root in RUNS.items():
    rows = []
    for bdir in sorted(glob.glob(root + '/*_b*')):
        name = os.path.basename(bdir)
        tp = os.path.join(bdir, 'transcript.txt')
        br = json.load(open(os.path.join(bdir, 'branch_result.json')))
        usage = br.get('usage') or []
        text = open(tp, encoding='utf-8', errors='replace').read()
        resp = list(responses(text))
        cum = 0; first_check = None
        pre = dict(n=0, intent=0, intent_nocall=0, misread=0, nocall=0, cap=0, owncmp=0, think_tokens=0,
                   intent_resp=0, think_after_intent=0)
        for i, r in enumerate(resp):
            comp = usage[i]['completion'] if i < len(usage) else 0
            code = '\n'.join(r['calls'])
            is_check = bool(re.search(r"\bcheck\s*\(", code))
            if first_check is None:
                pre['n'] += 1
                pre['think_tokens'] += comp
                ni = len(INTENT.findall(r['think'])); pre['intent'] += ni
                pre['misread'] += len(MISREAD.findall(r['think']))
                if not r['calls']: pre['nocall'] += 1; pre['intent_nocall'] += ni
                if r['meta'].get('finish_reason') == 'length': pre['cap'] += 1
                if re.search(r"\[['\"]output['\"]\]\s*(==|!=)|==\s*\w+\[['\"]output['\"]\]|np\.array_equal\([^)]*output", code): pre['owncmp'] += 1
                m = INTENT.search(r['think'])
                if m:
                    pre['intent_resp'] += 1
                    frac = 1 - m.start() / max(1, len(r['think']))
                    pre['think_after_intent'] += int(comp * frac)
            cum += comp
            if is_check and first_check is None:
                first_check = cum
        rows.append(dict(branch=name, accepted=bool(br.get('accepted')), total=br.get('generated_tokens'),
                         first_check=first_check, **pre))
    out[run] = rows
    print(f"\n=== {run}: {len(rows)} branches ===")
    print(f"{'branch':16} acc  total   1st_chk  resp nocall cap intent(nocall) misread owncmp tok_after_1st_intent")
    for r in rows:
        print(f"{r['branch']:16} {'Y' if r['accepted'] else '-'}  {r['total']:>7} {str(r['first_check']):>8}  {r['n']:>4} {r['nocall']:>6} {r['cap']:>3} {r['intent']:>6}({r['intent_nocall']:>3}) {r['misread']:>7} {r['owncmp']:>6} {r['think_after_intent']:>8}")
    fc = [r['first_check'] for r in rows if r['first_check']]
    tot = sum(r['think_tokens'] for r in rows)
    print(f"first check: median {st.median(fc):.0f}, never {sum(1 for r in rows if not r['first_check'])}/{len(rows)}")
    print(f"pre-check tokens {tot}; tokens after first code-intent in the same response: {sum(r['think_after_intent'] for r in rows)} "
          f"({100*sum(r['think_after_intent'] for r in rows)/max(1,tot):.0f} %)")
    print(f"branches with >=1 self-detected misread before check: {sum(1 for r in rows if r['misread'])}/{len(rows)}; "
          f"with own output comparison: {sum(1 for r in rows if r['owncmp'])}/{len(rows)}; with a cap-hit before check: {sum(1 for r in rows if r['cap'])}/{len(rows)}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'precheck_behaviours.json'), 'w'), indent=1)
