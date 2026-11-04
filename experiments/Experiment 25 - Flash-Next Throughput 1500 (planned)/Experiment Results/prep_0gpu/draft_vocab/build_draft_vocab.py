"""Exp 25 item 1-1 (0 GPU): ARC-derived reduced draft vocabularies for Merlin's Qwen4Exp MTP drafter.

Re-runnable, CPU only:
    python -I build_draft_vocab.py            # full run (tokenize -> 2-fold coverage -> tables -> loader check)

What it does
  1. Rebuilds the generated token stream of every saved Duck request (reasoning, content, tool calls) with the
     chat template's assistant layout, tokenizes it with the klee100 tokenizer.json (sha 06b95093), and labels
     every token with a segment type from token-level markers (thinking / code / numbers_grids / protocol /
     final_answer). Saved token ids (speed-bench traces) are used directly where they exist.
  2. Compares the two tokenizer.json files (06b95093 klee100, 0997f410 earlier Flash-Next) and the HF
     AutoTokenizer on the same strings.
  3. 2-fold task split on the 30 E26 K100Q tasks: build frequencies on one half, measure coverage on the other,
     swap, report the mean. Also held-out coverage on other runs and the klee100 speed-bench token ids.
  4. Final tables (all 30 tasks) for N = 16k / 24k / 32k / 48k / 64k: always-in set (33 added tokens, 256 byte
     tokens, all numeric tokens) + most frequent generated tokens (+ fill tiers), sorted unique int64, saved with
     torch.save, sha256 recorded.
  5. Loads every file with Merlin's own load_draft_vocab_ids(path, 248320, tp_size=2) (and tp 1 / 4, and with
     VLLM_DRAFT_VOCAB_SHA256 set).

Safety rule: the script never prints response text; only counts, ids and hashes go to stdout / JSON.
"""
from __future__ import annotations

import collections
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = "D:/ARC AGI 2"
SCR = ROOT + "/_scratch"
OUT = ROOT + "/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/draft_vocab"
WORK = SCR + "/prep_vocab"
TOK06 = SCR + "/e13_rev6/harness/fixtures/tok_int4/tokenizer.json"   # klee100 tokenizer.json (06b95093...)
TOK09 = SCR + "/e13_rev6/harness/fixtures/tok_fp8/tokenizer.json"    # earlier Flash-Next tokenizer.json (0997f410...)
TOK06_DIR = os.path.dirname(TOK06)
DRAFT_HEAD_PY = SCR + "/e23/p2/merlin-src/vllm/models/qwen4_exp/nvidia/draft_head.py"
E26_LOG = SCR + "/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/engine/logs/server_merlin_fp8_mtp_pin508.log"

V = 248320                 # model vocab (lm_head rows)
HIDDEN = 2560
NS = [16384, 24576, 32768, 49152, 65536]
SEGS = ["thinking", "code", "numbers_grids", "protocol", "final_answer"]
S_THINK, S_CODE, S_NUM, S_PROTO, S_ANS = range(5)
SPLIT_SEED = "e25-1-1-draft-vocab"

DUCK = {
    # name: (glob, model, primary?)
    "e26_k100q": (SCR + "/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3/duck/*/nb5_requests.jsonl",
                  "klee100 3bpw-MTP, FP8 KV, MTP K3 (E26 K100Q)"),
    "e25_nb5_bf16": (SCR + "/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_BF16_K3/duck/*/nb5_requests.jsonl",
                     "Flash-Next W4A16 Merlin p1-3, BF16 KV, MTP K3 (E25 NB5)"),
    "e25_nb5_fp8": (SCR + "/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3/duck/*/nb5_requests.jsonl",
                    "Flash-Next W4A16 Merlin p1-3, FP8 KV, MTP K3 (E25 NB5)"),
    "e25_nb6_eval": (SCR + "/e25_gpu_out/q27-xhigh-e25-nb6-submit/q27x/phase_EVAL/duck/*/nb5_requests.jsonl",
                     "Flash-Next W4A16 Merlin (E25 NB6 submit smoke)"),
}
BENCH = {
    # saved token ids returned by the server (klee100 runs only)
    "e27_nbdf_q": (SCR + "/e27_gpu_out/q27-xhigh-e27-nbdf/nbdf/responses/responses_nbdf_q.jsonl.gz",
                   "klee100 NBDF arm Q speed bench (E27), ignore_eos"),
    "e25_k3v2": (SCR + "/e25_gpu_out_k3_v2/q27-xhigh-e25-klee100-load-k3only/k100/traces/requests.jsonl",
                 "klee100 K3 load + sweeps (E25 k3_v2), ignore_eos"),
}
BASELINES = {
    # unconditional per-position acceptance (vLLM 'Per-position acceptance rate'), AL = 1 + sum
    "e26_k100_duck_measured": None,      # filled from the E26 server log
    "nb5_fp8_duck": (0.795, 0.599, 0.439),
    "klee100_synthetic_20k": (0.864, 0.731, 0.607),
}


def log(*a):
    print(*a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- reconstruction
def reconstruct(d):
    """Generated text of one request in the chat template's assistant layout, with typed char spans.

    Layout (template): [reasoning] '\\n</think>\\n\\n' [content] ['\\n\\n'] '<tool_call>\\n<function=NAME>\\n'
    '<parameter=K>\\n' VALUE '\\n</parameter>\\n' ... '</function>\\n</tool_call>' (next call prefixed '\\n') '<|im_end|>'.
    """
    parts = []

    def add(s, t):
        if s:
            parts.append((s, t))

    reasoning = d.get("reasoning")
    content = d.get("content") or ""
    calls = d.get("tool_calls") or []
    finish = d.get("finish")
    has_think = isinstance(reasoning, str) and reasoning != ""
    if has_think:
        add(reasoning, S_THINK)
        if content or calls or finish in ("stop", "tool_calls"):
            add("\n</think>\n\n", S_PROTO)
    if content:
        add(content, S_ANS)
    for i, tc in enumerate(calls):
        name = tc.get("name") or ""
        vt = S_ANS if name == "submit" else S_CODE
        pre = ("\n\n" if (i == 0 and content.strip()) else ("\n" if i > 0 else ""))
        add(pre + "<tool_call>\n<function=" + name + ">\n", S_PROTO)
        raw = tc.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            args = None
        if isinstance(args, dict):
            for k, v in args.items():
                add("<parameter=" + str(k) + ">\n", S_PROTO)
                add(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), vt)
                add("\n</parameter>\n", S_PROTO)
        elif isinstance(raw, str) and raw:
            add("<parameter=code>\n", S_PROTO)
            add(raw, vt)
            add("\n</parameter>\n", S_PROTO)
        add("</function>\n</tool_call>", S_PROTO)
    if finish in ("stop", "tool_calls"):
        add("<|im_end|>", S_PROTO)
    text = "".join(s for s, _ in parts)
    spans, pos = [], 0
    for s, t in parts:
        spans.append((pos, pos + len(s), t))
        pos += len(s)
    return text, spans


# ----------------------------------------------------------------------------- tokenizer helpers
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def vocab_tables(tok):
    """Per-id decoded strings (byte-level decoded, utf-8 'replace'), byte-token ids, added ids."""
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    tj = json.load(open(TOK06, encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    added = {a["id"]: a["content"] for a in tj["added_tokens"]}
    dec = [""] * V
    for s, i in vocab.items():
        try:
            dec[i] = bytes(u2b[c] for c in s).decode("utf-8", errors="replace")
        except KeyError:
            dec[i] = s
    for i, s in added.items():
        dec[i] = s
    byte_ids = sorted(vocab[b2u[b]] for b in range(256))
    return dec, byte_ids, sorted(added), vocab


def numeric_ids(dec, vocab_max):
    out = []
    for i in range(vocab_max + 1):
        s = dec[i]
        st = s.strip(" ")
        if not st or "\ufffd" in st:
            continue
        if all("0" <= c <= "9" for c in st) or (len(st) == 1 and st.isnumeric()):
            out.append(i)
    return out


# ----------------------------------------------------------------------------- data loading
def load_duck(tok06, tok09, hf, dec_is_digit, dec_is_gridp):
    recs = {}
    for name, (pat, _model) in DUCK.items():
        rows = []
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                d = json.loads(line)
                if d.get("kind") == "no_request" or "completion" not in d:
                    continue
                text, spans = reconstruct(d)
                rows.append(dict(task=d.get("task"), branch=d.get("branch"), kind=d.get("kind"), finish=d.get("finish"),
                                 completion=d.get("completion") or 0, text=text, spans=spans))
        texts = [r["text"] for r in rows]
        e06 = tok06.encode_batch(texts, add_special_tokens=False)
        e09 = tok09.encode_batch(texts, add_special_tokens=False)
        cmp = dict(requests=len(rows), req_equal_06_09=0, tok_06=0, tok_09=0, req_equal_06_hf=0, tok_hf=0)
        for r, a, b in zip(rows, e06, e09):
            ids = np.asarray(a.ids, dtype=np.int64)
            cmp["tok_06"] += len(ids)
            cmp["tok_09"] += len(b.ids)
            cmp["req_equal_06_09"] += int(list(a.ids) == list(b.ids))
            if hf is not None:
                h = hf(r["text"], add_special_tokens=False)["input_ids"]
                cmp["tok_hf"] += len(h)
                cmp["req_equal_06_hf"] += int(list(a.ids) == list(h))
            # segment labels from the char spans (token start offset)
            starts = np.asarray([o[0] for o in a.offsets], dtype=np.int64)
            seg = np.full(len(ids), S_PROTO, dtype=np.int8)
            for s0, s1, t in r["spans"]:
                m = (starts >= s0) & (starts < s1)
                seg[m] = t
            # numbers / grids override inside thinking and code
            dig = dec_is_digit[ids]
            gp = dec_is_gridp[ids]
            adj = np.zeros(len(ids), dtype=bool)
            if len(ids) > 1:
                adj[1:] |= dig[:-1]
                adj[:-1] |= dig[1:]
            num = dig | (gp & adj)
            seg[num & ((seg == S_THINK) | (seg == S_CODE))] = S_NUM
            r["ids"] = ids
            r["seg"] = seg
            r["n_mismatch_completion"] = len(ids) - r["completion"]
            del r["text"], r["spans"]
        recs[name] = (rows, cmp)
        log(f"[duck] {name}: requests {len(rows)} tokens {cmp['tok_06']} completion_sum {sum(r['completion'] for r in rows)} "
            f"06==09 reqs {cmp['req_equal_06_09']}/{len(rows)} tok06 {cmp['tok_06']} tok09 {cmp['tok_09']} "
            f"06==HF reqs {cmp['req_equal_06_hf']}/{len(rows)} tokHF {cmp['tok_hf']}")
    return recs


def load_bench(tok06, tok09, hf):
    out = {}
    seen = set()
    for name, (p, _m) in BENCH.items():
        op = gzip.open if p.endswith(".gz") else open
        rows = []
        st = collections.Counter()
        for line in op(p, "rt", encoding="utf-8"):
            d = json.loads(line)
            toks = d.get("tokens") or []
            if len(toks) < 2:
                continue
            key = hashlib.sha256(np.asarray(toks, dtype=np.int64).tobytes()).hexdigest()
            if key in seen:
                st["dup"] += 1
                continue
            seen.add(key)
            ids = np.asarray(toks, dtype=np.int64)
            # retokenization agreement (text field = server-decoded text)
            txt = d.get("text") or ""
            a = tok06.encode(txt, add_special_tokens=False).ids
            b = tok09.encode(txt, add_special_tokens=False).ids
            st["req"] += 1
            st["tok"] += len(ids)
            st["retok06_equal"] += int(list(a) == toks)
            st["retok09_equal"] += int(list(b) == toks)
            st["retok06_vs_09_equal"] += int(a == b)
            st["retok06_len"] += len(a)
            if hf is not None:
                h = hf(txt, add_special_tokens=False)["input_ids"]
                st["retokhf_equal"] += int(list(h) == toks)
            st["ids_ge_248077"] += int((ids >= 248077).sum())
            eos = np.nonzero((ids == 248046) | (ids == 248044))[0]
            cut = int(eos[0]) if len(eos) else len(ids)
            st["tok_before_first_eos"] += cut
            rows.append(dict(ids=ids, cut=cut, retok=np.asarray(a, dtype=np.int64)))
        out[name] = (rows, dict(st))
        log(f"[bench] {name}: {dict(st)}")
    return out


# ----------------------------------------------------------------------------- tables + metrics
def fold_split(tasks):
    keyed = sorted(tasks, key=lambda t: hashlib.sha256((SPLIT_SEED + t).encode()).hexdigest())
    h = len(keyed) // 2
    return sorted(keyed[:h]), sorted(keyed[h:])


def counts(rows, filt=None):
    c = np.zeros(V, dtype=np.int64)
    for r in rows:
        if filt is None or filt(r):
            c += np.bincount(r["ids"], minlength=V)[:V]
    return c


def make_table(n, always, gen, ctx=None):
    """always-in ids first, then generated frequency desc, then ctx frequency desc, then ascending id (< 248077)."""
    always = np.asarray(sorted(set(always)), dtype=np.int64)
    cand = np.arange(248077, dtype=np.int64)
    mask = np.ones(248077, dtype=bool)
    mask[always] = False
    cand = cand[mask]
    g = gen[cand]
    c = ctx[cand] if ctx is not None else np.zeros(len(cand), dtype=np.int64)
    order = np.lexsort((cand, -c, -g))     # last key primary: -gen, then -ctx, then id
    need = n - len(always)
    pick = cand[order[:need]]
    ids = np.sort(np.concatenate([always, pick]))
    assert len(ids) == n and len(np.unique(ids)) == n
    return ids


def coverage(rows, inset, cut=False):
    tot = np.zeros(5, dtype=np.int64)
    hit = np.zeros(5, dtype=np.int64)
    win = np.zeros(4, dtype=np.int64)     # windows of length k all in table (k=1..3)
    wtot = np.zeros(4, dtype=np.int64)
    for r in rows:
        ids = r["ids"][: r["cut"]] if cut else r["ids"]
        if len(ids) == 0:
            continue
        m = inset[ids]
        seg = r["seg"][: len(ids)] if "seg" in r else np.zeros(len(ids), dtype=np.int8)
        tot += np.bincount(seg, minlength=5)
        hit += np.bincount(seg[m], minlength=5)
        run = m.copy()
        for k in range(1, 4):
            if k > 1:
                run = run[:-1] & m[k - 1:]
            win[k] += int(run.sum())
            wtot[k] += len(run)
    W = [win[k] / wtot[k] if wtot[k] else float("nan") for k in range(1, 4)]
    return dict(overall=float(hit.sum() / max(1, tot.sum())), tokens=int(tot.sum()),
                per_seg={SEGS[i]: (float(hit[i] / tot[i]) if tot[i] else None) for i in range(5)},
                seg_tokens={SEGS[i]: int(tot[i]) for i in range(5)},
                miss_tokens=int(tot.sum() - hit.sum()), W=W)


def al_est(pos, W, c):
    base = 1 + sum(pos)
    joint = 1 + sum(p * w for p, w in zip(pos, W))
    indep = 1 + sum(p * c ** (k + 1) for k, p in enumerate(pos))
    return dict(base=base, joint=joint, indep=indep, joint_ratio=joint / base, indep_ratio=indep / base)


def e26_measured_acceptance():
    acc = 0
    nd = 0.0
    pos = [0.0, 0.0, 0.0]
    rx = re.compile(r"Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d.]+), ([\d.]+), ([\d.]+)")
    for ln in open(E26_LOG, encoding="utf-8", errors="replace"):
        m = rx.search(ln)
        if not m:
            continue
        a, d = int(m.group(1)), int(m.group(2))
        k = d / 3
        acc += a
        nd += k
        for i in range(3):
            pos[i] += float(m.group(3 + i)) * k
    p = tuple(x / nd for x in pos)
    return p, 1 + acc / nd, int(round(nd))


def ctx_counts(tok06, tasks_dirs):
    """Token counts of the non-generated prompt sections (system, tools, user / tool responses) per task."""
    keep = {"AVAILABLE TOOLS", "SYSTEM", "USER", "USER PROMPT", "ANALYZER STATUS"}
    per_task = {}
    for f in tasks_dirs:
        task = os.path.basename(os.path.dirname(os.path.dirname(f))).split("_b")[0]
        cur = None
        buf = []
        for ln in open(f, encoding="utf-8").read().split("\n"):
            m = re.fullmatch(r"\[([A-Z][A-Z _]{1,30})\]\s*", ln)
            if m:
                cur = m.group(1)
                continue
            if cur in keep:
                buf.append(ln)
        ids = tok06.encode("\n".join(buf), add_special_tokens=False).ids
        c = per_task.setdefault(task, np.zeros(V, dtype=np.int64))
        c += np.bincount(np.asarray(ids, dtype=np.int64), minlength=V)[:V]
    return per_task


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    from tokenizers import Tokenizer
    tok06 = Tokenizer.from_file(TOK06)
    tok09 = Tokenizer.from_file(TOK09)
    try:
        from transformers import AutoTokenizer
        _hf = AutoTokenizer.from_pretrained(TOK06_DIR)
        hf = _hf
        hf_name = f"transformers {__import__('transformers').__version__} AutoTokenizer({os.path.basename(TOK06_DIR)})"
    except Exception as e:  # noqa: BLE001
        hf = None
        hf_name = f"unavailable ({type(e).__name__})"
    tok_info = {"tok06": {"path": TOK06, "sha256": sha256_file(TOK06)}, "tok09": {"path": TOK09, "sha256": sha256_file(TOK09)},
                "hf": hf_name}
    log("[tok]", json.dumps(tok_info))

    dec, byte_ids, added_ids, vocab = vocab_tables(tok06)
    vmax = max(vocab.values())
    num_ids = numeric_ids(dec, vmax)
    dec_is_digit = np.zeros(V, dtype=bool)
    dec_is_gridp = np.zeros(V, dtype=bool)
    for i in range(vmax + 1):
        s = dec[i]
        st = s.strip()
        if st and all("0" <= ch <= "9" for ch in st):
            dec_is_digit[i] = True
        if s and all(ch in " ,[]()\n\t" for ch in s):
            dec_is_gridp[i] = True
    always = sorted(set(added_ids) | set(byte_ids) | set(num_ids))
    log(f"[always] added {len(added_ids)} ({added_ids[0]}..{added_ids[-1]}), byte {len(byte_ids)}, numeric {len(num_ids)}, "
        f"union {len(always)}; base vocab {vmax + 1}, rows >= {vmax + 1 + len(added_ids)} are unused padding")

    duck = load_duck(tok06, tok09, hf, dec_is_digit, dec_is_gridp)
    bench = load_bench(tok06, tok09, hf)

    main_rows = duck["e26_k100q"][0]
    tasks = sorted({r["task"] for r in main_rows})
    fa, fb = fold_split(tasks)
    log(f"[split] {len(tasks)} tasks -> fold A {len(fa)}, fold B {len(fb)}")
    ctx_files = sorted(glob.glob(os.path.dirname(DUCK["e26_k100q"][0]) + "/prompts/prompt.log"))
    ctx_task = ctx_counts(tok06, ctx_files)

    def ctx_for(ts):
        c = np.zeros(V, dtype=np.int64)
        for t in ts:
            if t in ctx_task:
                c += ctx_task[t]
        return c

    sec = [r for name in ("e25_nb5_bf16", "e25_nb5_fp8", "e25_nb6_eval") for r in duck[name][0]]
    p_e26, al_e26, nd_e26 = e26_measured_acceptance()
    BASELINES["e26_k100_duck_measured"] = p_e26
    log(f"[accept] e26 measured pos {[round(x, 4) for x in p_e26]} AL {al_e26:.4f} drafts {nd_e26}")

    # ---------------- 2-fold held-out
    variants = {
        "gen_only": lambda build_t: (counts(main_rows, lambda r: r["task"] in build_t), None),
        "gen_ctx": lambda build_t: (counts(main_rows, lambda r: r["task"] in build_t), ctx_for(build_t)),
        "gen_plus_e25": lambda build_t: (counts(main_rows, lambda r: r["task"] in build_t)
                                         + counts(sec, lambda r: r["task"] in build_t), ctx_for(build_t)),
        # reference only (never chosen): always-in set + lowest ids, no frequency data
        "id_only_reference": lambda build_t: (np.zeros(V, dtype=np.int64), None),
    }
    folds = [(fa, fb), (fb, fa)]
    hold = {}
    dist = []
    for vname, fn in variants.items():
        res = {n: [] for n in NS}
        for build_t, test_t in folds:
            gen, ctx = fn(set(build_t))
            test_rows = [r for r in main_rows if r["task"] in set(test_t)]
            sec_rows = [r for r in sec if r["task"] not in set(build_t)]
            if vname == "gen_only":
                g_test = counts(test_rows)
                dist.append(dict(build_distinct=int((gen > 0).sum()), test_distinct=int((g_test > 0).sum()),
                                 test_tokens=int(g_test.sum()),
                                 test_tokens_unseen_in_build=int(g_test[gen == 0].sum()),
                                 test_distinct_unseen_in_build=int(((g_test > 0) & (gen == 0)).sum())))
            for n in NS:
                ids = make_table(n, always, gen, ctx)
                inset = np.zeros(V, dtype=bool)
                inset[ids] = True
                cv = coverage(test_rows, inset)
                cs = coverage(sec_rows, inset)
                cb = {b: coverage(bench[b][0], inset) for b in bench}
                cbc = {b: coverage(bench[b][0], inset, cut=True) for b in bench}
                res[n].append(dict(main=cv, sec=cs, bench=cb, bench_cut=cbc, build_tasks=len(build_t), test_tasks=len(test_t),
                                   test_tokens=cv["tokens"], gen_filled=int((gen[ids] > 0).sum())))
        hold[vname] = res
        log(f"[2fold:{vname}] " + " ".join(f"N{n}={np.mean([x['main']['overall'] for x in res[n]]):.5f}" for n in NS))

    # ---------------- final tables (all 30 tasks, chosen variant)
    choice = max([v for v in variants if v != "id_only_reference"], key=lambda v: np.mean([np.mean([x["main"]["overall"] for x in hold[v][n]]) for n in NS]))
    gen_all, ctx_all = variants[choice](set(tasks))
    tables = {}
    for n in NS:
        ids = make_table(n, always, gen_all, ctx_all)
        t = torch.from_numpy(ids.astype(np.int64)).contiguous()
        p = f"{OUT}/draft_vocab_k100_arc_n{n}.pt"
        torch.save(t, p)
        inset = np.zeros(V, dtype=bool)
        inset[ids] = True
        tables[n] = dict(path=p, sha256=sha256_file(p), n=n, gen_filled=int((gen_all[ids] > 0).sum()),
                         ctx_only=int(((gen_all[ids] == 0) & (ctx_all[ids] > 0)).sum()) if ctx_all is not None else 0,
                         id_fill=int(((gen_all[ids] == 0) & ((ctx_all[ids] == 0) if ctx_all is not None else True)).sum()),
                         always_in=len(always), min_id=int(ids.min()), max_id=int(ids.max()),
                         in_sample_coverage_e26=coverage(main_rows, inset)["overall"],
                         bench_coverage_full={b: coverage(bench[b][0], inset)["overall"] for b in bench},
                         head_mib_per_pp1_rank_tp2=n * HIDDEN * 2 / 2 / 2 ** 20)

    # ---------------- loader check with Merlin's own code
    spec = importlib.util.spec_from_file_location("merlin_draft_head", DRAFT_HEAD_PY)
    dh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dh)
    for n, info in tables.items():
        chk = {}
        os.environ.pop(dh.ENV_DRAFT_VOCAB_SHA256, None)
        for tp in (1, 2, 4):
            ids = dh.load_draft_vocab_ids(info["path"], V, tp_size=tp)
            chk[f"tp{tp}"] = dict(ok=True, n=int(ids.numel()), dtype=str(ids.dtype))
        os.environ[dh.ENV_DRAFT_VOCAB_SHA256] = info["sha256"]
        ids = dh.load_draft_vocab_ids(info["path"], V, tp_size=2)
        chk["tp2_with_sha_env"] = dict(ok=True, n=int(ids.numel()))
        os.environ[dh.ENV_DRAFT_VOCAB_SHA256] = "0" * 64
        try:
            dh.load_draft_vocab_ids(info["path"], V, tp_size=2)
            chk["wrong_sha_refused"] = False
        except ValueError:
            chk["wrong_sha_refused"] = True
        os.environ.pop(dh.ENV_DRAFT_VOCAB_SHA256, None)
        chk["fingerprint"] = dh.draft_vocab_fingerprint(ids)
        chk["table_factor"] = dh.draft_vocab_table_factor(info["path"])
        d2t = dh.d2t_offsets(ids)
        chk["d2t_roundtrip"] = bool(torch.equal(torch.arange(ids.numel()) + d2t, ids))
        w = torch.randn(V, 8)
        rows = dh.select_head_rows(w, ids, V, 8)
        lg = torch.randn(3, ids.numel())
        full = dh.scatter_to_full(lg, ids, V)
        chk["select_scatter_ok"] = bool(torch.equal(rows, w[ids]) and torch.equal(full[:, ids], lg)
                                        and bool(torch.isinf(full).sum() == 3 * (V - ids.numel())))
        info["loader_check"] = chk
        log(f"[loader] N{n} sha {info['sha256'][:16]} {json.dumps(chk)}")

    # ---------------- acceptance model on mean held-out numbers
    summary = {}
    for vname, res in hold.items():
        summary[vname] = {}
        for n in NS:
            r = res[n]
            m_over = float(np.mean([x["main"]["overall"] for x in r]))
            m_seg = {s: float(np.mean([x["main"]["per_seg"][s] for x in r if x["main"]["per_seg"][s] is not None])) for s in SEGS}
            m_W = [float(np.mean([x["main"]["W"][k] for x in r])) for k in range(3)]
            m_sec = float(np.mean([x["sec"]["overall"] for x in r]))
            m_secW = [float(np.mean([x["sec"]["W"][k] for x in r])) for k in range(3)]
            m_b = {b: float(np.mean([x["bench"][b]["overall"] for x in r])) for b in bench}
            m_bW = {b: [float(np.mean([x["bench"][b]["W"][k] for x in r])) for k in range(3)] for b in bench}
            m_bc = {b: float(np.mean([x["bench_cut"][b]["overall"] for x in r])) for b in bench}
            al = {bn: al_est(pos, m_W, m_over) for bn, pos in BASELINES.items()}
            al_bench = {b: al_est(BASELINES["klee100_synthetic_20k"], m_bW[b], m_b[b]) for b in bench}
            summary[vname][n] = dict(overall=m_over, per_seg=m_seg, W=m_W, folds=[x["main"]["overall"] for x in r],
                                     miss_per_1k=1000 * (1 - m_over), sec_overall=m_sec, sec_W=m_secW,
                                     bench=m_b, bench_W=m_bW, bench_cut=m_bc, al=al, al_bench=al_bench,
                                     seg_tokens_fold=[x["main"]["seg_tokens"] for x in r])
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), script=os.path.abspath(__file__), vocab_size=V, hidden=HIDDEN,
        tokenizer=tok_info, always_in=dict(added=added_ids, n_byte=len(byte_ids), n_numeric=len(num_ids), n_union=len(always)),
        split=dict(seed=SPLIT_SEED, fold_A=fa, fold_B=fb), fill_variant=choice,
        fill_rule="always-in set; then generated-token frequency over all 30 E26 tasks (desc); then prompt-context frequency "
                  "(desc, if the variant uses it); then ascending token id among ids < 248077",
        sources={k: dict(glob=v[0], model=v[1], requests=len(duck[k][0]), tokens=duck[k][1]["tok_06"],
                         completion_sum=int(sum(r["completion"] for r in duck[k][0])), tokenizer_compare=duck[k][1],
                         tasks=sorted({r["task"] for r in duck[k][0]})) for k, v in DUCK.items()},
        bench={k: dict(path=v[0], model=v[1], **bench[k][1]) for k, v in BENCH.items()},
        e26_measured_acceptance=dict(pos=p_e26, AL=al_e26, drafts=nd_e26),
        baselines=BASELINES, fold_distinct=dist, tables={str(n): v for n, v in tables.items()},
        heldout=summary, runtime_sec=time.time() - t0)
    # reconstruction check vs server completion counts
    mm = np.asarray([r["n_mismatch_completion"] for r in main_rows])
    fin = collections.Counter(r["finish"] for r in main_rows)
    manifest["reconstruction_check_e26"] = dict(requests=len(main_rows), tok_sum=int(sum(len(r["ids"]) for r in main_rows)),
                                                completion_sum=int(sum(r["completion"] for r in main_rows)),
                                                diff_abs_median=float(np.median(np.abs(mm))), diff_abs_p90=float(np.quantile(np.abs(mm), .9)),
                                                diff_abs_max=int(np.abs(mm).max()), diff_mean=float(mm.mean()), finish=dict(fin))
    seg_tot = np.zeros(5, dtype=np.int64)
    for r in main_rows:
        seg_tot += np.bincount(r["seg"], minlength=5)
    manifest["segment_tokens_e26_all"] = {SEGS[i]: int(seg_tot[i]) for i in range(5)}
    # retokenization bias on bench: coverage with saved ids vs retokenized text, full tables
    rb = {}
    for n in NS:
        inset = np.zeros(V, dtype=bool)
        inset[np.asarray(torch.load(tables[n]["path"]))] = True
        rb[n] = {b: dict(saved_ids=coverage(bench[b][0], inset)["overall"],
                         retokenized=coverage([dict(ids=x["retok"], cut=len(x["retok"])) for x in bench[b][0]], inset)["overall"])
                 for b in bench}
    manifest["bench_saved_vs_retokenized_full_tables"] = {str(k): v for k, v in rb.items()}
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, default=float)
    log(f"[done] {time.time() - t0:.0f}s manifest {OUT}/manifest.json")


if __name__ == "__main__":
    main()
