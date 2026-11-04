"""Step 1: tokenize the E26 K100Q multi-turn workload and label every generated token with a segment type.
Prints only aggregate numbers. Output: /_scratch/prep_suffix/data.pkl
Segments: 0 reasoning prose, 1 code inside reasoning/content fences, 2 numbers/grid lines, 3 tool-call code (parameter values),
4 tool/JSON protocol (tool-call wrapper tokens), 5 content / final answer prose."""
import json, glob, os, re, pickle, sys
import numpy as np
from tokenizers import Tokenizer
ROOT = "D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3/duck/"
TOK = "D:/ARC AGI 2/_scratch/e13_rc2_local/kit_tmp/tmp_main/fn_tok/tokenizer.json"
OUT = "D:/ARC AGI 2/_scratch/prep_suffix/data.pkl"
tok = Tokenizer.from_file(TOK)
MARK = re.compile(r'^\[([A-Z_ ]+)\]\s*$')
GRIDCH = set("0123456789 ,.[](){}-|:rR\t")
CODE_RE = re.compile(r'^\s*(def |for |if |elif |else:|while |import |from |return |class |try:|except|with |print\(|#|assert |lambda )|^\s{4,}\S|^\s*[\w\.\[\]]+\s*=\s*\S|^\s*[\w\.]+\(.*\)\s*$')

def line_labels(text, in_content):
    """char-level label array for text of reasoning (0) or content (5)."""
    lab = np.zeros(len(text), np.uint8)
    pos = 0; fence = False
    for line in text.split('\n'):
        L = len(line)
        s = line.strip()
        if s.startswith('```'):
            fence = not fence; l = 4 if False else (1)
        elif fence:
            l = 1
        else:
            digits = sum(c.isdigit() for c in s)
            if s and digits >= 6 and sum(c in GRIDCH for c in s) >= 0.8 * len(s):
                l = 2
            elif CODE_RE.search(line):
                l = 1
            else:
                l = 5 if in_content else 0
        lab[pos:pos+L+1] = l
        pos += L + 1
    return lab[:len(text)]

def render_gen(d):
    """returns (text, charlabels)"""
    parts = []; labs = []
    r = d.get('reasoning') or ''
    if r:
        parts.append(r); labs.append(line_labels(r, False))
    parts.append("\n</think>\n\n"); labs.append(np.full(len("\n</think>\n\n"), 4, np.uint8))
    c = d.get('content') or ''
    if c:
        parts.append(c); labs.append(line_labels(c, True))
    for tc in (d.get('tool_calls') or []):
        name = tc.get('name', ''); args = tc.get('arguments', '{}')
        try: a = json.loads(args) if isinstance(args, str) else args
        except Exception: a = {'_raw': args}
        if not isinstance(a, dict): a = {'_raw': a}
        w = "\n<tool_call>\n<function=%s>\n" % name
        parts.append(w); labs.append(np.full(len(w), 4, np.uint8))
        for k, v in a.items():
            h = "<parameter=%s>\n" % k
            parts.append(h); labs.append(np.full(len(h), 4, np.uint8))
            vs = v if isinstance(v, str) else json.dumps(v)
            parts.append(vs); labs.append(np.full(len(vs), 3, np.uint8))
            t = "\n</parameter>\n"
            parts.append(t); labs.append(np.full(len(t), 4, np.uint8))
        e = "</function>\n</tool_call>"
        parts.append(e); labs.append(np.full(len(e), 4, np.uint8))
    return ''.join(parts), np.concatenate(labs)

def enc(text, charlab=None):
    e = tok.encode(text, add_special_tokens=False)
    ids = np.array(e.ids, np.int32)
    if charlab is None: return ids, None
    off = e.offsets
    tl = np.zeros(len(ids), np.uint8)
    n = len(charlab)
    for i, (a, b) in enumerate(off):
        tl[i] = charlab[a] if a < n else 4
    return ids, tl

def parse_prompt_segments(path):
    """returns list of (n_assistant_before, kind, text) for SYSTEM/USER blocks"""
    segs = []; cur = None; buf = []; nass = 0
    def flush():
        if cur in ('SYSTEM', 'USER', 'SYSTEM PROMPT', 'USER PROMPT'):
            segs.append((nass, cur, '\n'.join(buf)))
    for line in open(path, encoding='utf8').read().split('\n'):
        m = MARK.match(line.strip())
        if m:
            flush(); cur = m.group(1); buf = []
            if cur == 'ASSISTANT': nass += 1
        else: buf.append(line)
    flush()
    return segs

branches = []
tot = {'req': 0, 'gen': 0, 'completion': 0}
for d in sorted(glob.glob(ROOT + '*_b*')):
    bn = os.path.basename(d); rf = d + '/nb5_requests.jsonl'; pf = d + '/prompts/prompt.log'
    if not os.path.exists(rf): continue
    reqs = [json.loads(l) for l in open(rf, encoding='utf8')]
    segs = parse_prompt_segments(pf) if os.path.exists(pf) else []
    segtok = []
    for na, kind, text in segs:
        ids, _ = enc(text)
        segtok.append((na, ids))
    R = []
    for i, r in enumerate(reqs):
        if r.get('completion') is None: continue
        text, cl = render_gen(r)
        ids, tl = enc(text, cl)
        R.append(dict(idx=i, kind=r['kind'], t0=r['t0'], t1=r['t1'], completion=r['completion'], ids=ids, seg=tl,
                      prompt=r.get('prompt'), finish=r.get('finish')))
        tot['req'] += 1; tot['gen'] += len(ids); tot['completion'] += r['completion']
    branches.append(dict(name=bn, segtok=segtok, reqs=R))
pickle.dump(branches, open(OUT, 'wb'))
print('branches', len(branches), tot, 'rendered/reported token ratio %.4f' % (tot['gen'] / tot['completion']))
allseg = np.concatenate([r['seg'] for b in branches for r in b['reqs']])
print('segment token shares', {k: round(float((allseg == k).mean()), 4) for k in range(6)})
print('prompt-seg tokens total', sum(len(i) for b in branches for _, i in b['segtok']))
