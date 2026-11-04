"""Step 3: aggregate p1 (suffix only), p2 (hybrid) json files into tables (stdout + results.json)."""
import json, sys, numpy as np
D0 = "D:/ARC AGI 2/_scratch/prep_suffix/"
OUT = "D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/suffix_replay/results_suffix_replay.json"
SEGN = ['reasoning prose', 'code (fenced / code-like lines)', 'numbers / grid lines', 'tool-call code (param values)', 'tool/JSON protocol', 'content / final answer']
TBN = ['turn 0', 'turn 1', 'turn 2-3', 'turn 4-7', 'turn 8+']
p1 = json.load(open(D0 + 'p1_full.json')); p1n = json.load(open(D0 + 'p1_noreason.json'))
p2 = None
try:
    for D in (3, 8, 24):
        x = json.load(open(D0 + 'p2_full_D%d.json' % D))
        if p2 is None: p2 = x
        else:
            p2['ST'].update(x['ST']); p2['SRC'].update(x['SRC'])
except Exception as e:
    print('no p2', e)
MTP_E = p1['MTP_E']; MTP_TOK = 1 + MTP_E


def agg(res, var, seg=None, tb=None):
    t = np.zeros(8)
    for k, v in res['ST'].items():
        a, s, b = k.split('|')
        if a != var: continue
        if seg is not None and int(s) != seg: continue
        if tb is not None and int(b) != tb: continue
        t += np.array(v)
    return t


def row(t):
    steps, toks, npro, slen, sacc, nsuf, ver, _ = t
    if steps == 0: return None
    return dict(steps=int(steps), tokens=int(toks), tokens_per_step=toks / steps, verify_per_step=ver / steps,
                prop_share_steps=npro / steps, mean_draftlen_when_prop=slen / max(npro, 1),
                mean_accepted_when_suffix=sacc / max(nsuf, 1), suffix_share=nsuf / steps,
                tokens_per_verify_token=toks / ver)


R = dict(MTP_expected_accepted_drafts=MTP_E, MTP_tokens_per_step=MTP_TOK, MTP_verify_tokens=4)
out = {}
for res, tag in ((p1, 'full'), (p1n, 'noreason')):
    for D in (3, 8, 24):
        v = 'S%d' % D
        out['%s_%s' % (v, tag)] = dict(total=row(agg(res, v)),
                                       by_seg={SEGN[s]: row(agg(res, v, seg=s)) for s in range(6)},
                                       by_turn={TBN[b]: row(agg(res, v, tb=b)) for b in range(5)})
# position probe
pr = {}
for D in (3, 8, 24):
    tot = np.zeros(7); segs = {}
    for k, v in p1['PROBE'].items():
        d, s = map(int, k.split('|'))
        if d != D: continue
        tot += np.array(v); segs[s] = np.array(v)
    def prow(a):
        n, npro, sl, sa, ss, hr, ho = a
        return dict(n_positions=int(n), proposal_share=npro / n, mean_draftlen=sl / max(npro, 1), mean_accepted_when_proposing=sa / max(npro, 1),
                    mean_accepted_all_positions=sa / n, mean_score=ss / max(npro, 1), hybrid_raw_exp_accepted=hr / n, hybrid_oracle_exp_accepted=ho / n,
                    mtp_exp_accepted=MTP_E)
    pr['D%d' % D] = dict(total=prow(tot), by_seg={SEGN[s]: prow(segs[s]) for s in sorted(segs)})
out['probe'] = pr
if p2:
    for D in (3, 8, 24):
        for rule in ('raw', 'cal', 'orc'):
            v = 'H%d_%s' % (D, rule)
            out[v] = dict(total=row(agg(p2, v)),
                          by_seg={SEGN[s]: row(agg(p2, v, seg=s)) for s in range(6)},
                          by_turn={TBN[b]: row(agg(p2, v, tb=b)) for b in range(5)})
            srcs = {}
            for k, c in p2['SRC'].items():
                a, s = k.split('|')
                if a == v: srcs[s] = c
            tot = sum(srcs.values()); out[v]['step_share_by_method'] = {s: c / tot for s, c in srcs.items()}
out['calibration'] = p1['cal']
out['SRC_suffix_only'] = {k: v for k, v in p1['SRC'].items()}
R['results'] = out
json.dump(R, open(OUT, 'w'), indent=1)

def f(x): return '-' if x is None else '%.3f' % x
print('MTP baseline: %.3f tokens/step, verify 4' % MTP_TOK)
for tag in ('full', 'noreason'):
    for D in (3, 8, 24):
        r = out['S%d_%s' % (D, tag)]['total']
        print('S%d %s: tok/step %.3f verify/step %.3f prop-share %.3f draftlen|prop %.2f acc|suffix %.3f' % (D, tag, r['tokens_per_step'], r['verify_per_step'], r['prop_share_steps'], r['mean_draftlen_when_prop'], r['mean_accepted_when_suffix']))
for D in (3, 8, 24):
    print('S%d full per-segment (steps, tok/step, prop share, acc|prop):' % D)
    for s in SEGN:
        r = out['S%d_full' % D]['by_seg'][s]
        if r: print('   %-34s %7d %.3f %.3f %.3f' % (s, r['steps'], r['tokens_per_step'], r['prop_share_steps'], r['mean_accepted_when_suffix']))
    print('S%d full per-turn:' % D)
    for s in TBN:
        r = out['S%d_full' % D]['by_turn'][s]
        if r: print('   %-10s %7d %.3f %.3f %.3f' % (s, r['steps'], r['tokens_per_step'], r['prop_share_steps'], r['mean_accepted_when_suffix']))
for D in (3, 8, 24):
    t = pr['D%d' % D]['total']
    print('PROBE D%d: prop share %.3f acc|prop %.3f draftlen %.2f hybrid_raw_exp %.3f oracle %.3f vs MTP %.3f' % (D, t['proposal_share'], t['mean_accepted_when_proposing'], t['mean_draftlen'], t['hybrid_raw_exp_accepted'], t['hybrid_oracle_exp_accepted'], MTP_E))
if p2:
    for D in (3, 8, 24):
        for rule in ('raw', 'cal', 'orc'):
            v = 'H%d_%s' % (D, rule); r = out[v]['total']
            print('%s: tok/step %.3f (MTP %.3f, gain %+.1f%%) verify/step %.3f tokens/verify-token %.3f (MTP %.3f) suffix share %.3f shares %s' % (
                v, r['tokens_per_step'], MTP_TOK, 100 * (r['tokens_per_step'] / MTP_TOK - 1), r['verify_per_step'], r['tokens_per_verify_token'], MTP_TOK / 4, r['suffix_share'], {k: round(x, 3) for k, x in out[v]['step_share_by_method'].items()}))
        print('H%d_raw per segment (tok/step, suffix share):' % D)
        for s in SEGN:
            r = out['H%d_raw' % D]['by_seg'][s]
            if r: print('   %-34s %7d %.3f %.3f' % (s, r['steps'], r['tokens_per_step'], r['suffix_share']))
