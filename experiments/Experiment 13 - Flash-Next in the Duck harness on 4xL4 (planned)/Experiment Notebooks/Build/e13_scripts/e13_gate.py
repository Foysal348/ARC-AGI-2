"""E13 Stage 1, server side (stdlib only): Gate C correctness, the memory gate from the server log, the speed stage, and
the pre-declared Launch-2 decision. Imported by the probe notebook; also used by the local fake-server harness.

All requests go to SGLang's own HTTP API: /generate (raw prompts, no chat template, so no thinking; logprobs) and
/v1/chat/completions (the Duck path: tools + reasoning). Every response is logged to OUT/gate_requests.jsonl.
"""
import json
import random
import re
import threading
import time
import urllib.request

VOCAB = 248320
GRID_RNG_SEED = 20261019
TOOL = {"type": "function", "function": {"name": "python", "description": "Run Python code and return its stdout.",
                                         "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                        "required": ["code"]}}}


class Gate:
    def __init__(self, url, served, out_dir, log=print):
        self.url, self.served, self.out, self.log = url.rstrip("/"), served, out_dir, log
        self.req_log = open(f"{out_dir}/gate_requests.jsonl", "a", encoding="utf-8")
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ http
    def _post(self, path, payload, timeout):
        t = time.perf_counter()
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        err, body = None, None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:400]
        sec = time.perf_counter() - t
        with self.lock:
            self.req_log.write(json.dumps({"path": path, "sec": round(sec, 2), "error": err,
                                           "payload": {k: v for k, v in payload.items() if k not in ("text", "messages")},
                                           "prompt_chars": len(payload.get("text", "") or json.dumps(payload.get("messages", ""))),
                                           "response": body}, default=str)[:200000] + "\n")
            self.req_log.flush()
        return body, err, sec

    def generate(self, text, max_new, logprob=False, timeout=900, temperature=0.0, ignore_eos=False):
        p = {"text": text, "sampling_params": {"max_new_tokens": int(max_new), "temperature": temperature,
                                               "ignore_eos": ignore_eos}}
        if logprob:
            p.update(return_logprob=True, logprob_start_len=0, top_logprobs_num=0)
        return self._post("/generate", p, timeout)

    @staticmethod
    def out_ids(body):
        meta = (body or {}).get("meta_info") or {}
        if body and isinstance(body.get("output_ids"), list):
            return body["output_ids"]
        return [x[1] for x in (meta.get("output_token_logprobs") or []) if isinstance(x, (list, tuple)) and len(x) > 1]

    @staticmethod
    def garbage(text):
        t = (text or "").strip()
        if not t:
            return True
        top = max(t.count(c) for c in set(t))
        return top / len(t) > 0.5 and len(t) > 8                       # e.g. "!!!!!!" (E3) or one char repeated

    # ------------------------------------------------------------------ Gate C
    def c1_prefill_logprobs(self):
        text = ", ".join(str(i) for i in range(1, 61))
        b, err, sec = self.generate(text, 1, logprob=True)
        lps = [x[0] for x in ((b or {}).get("meta_info") or {}).get("input_token_logprobs") or []
               if isinstance(x, (list, tuple)) and x and x[0] is not None]
        tail = sorted(lps[20:])
        med = tail[len(tail) // 2] if tail else None
        finite = all(isinstance(v, (int, float)) and v == v and abs(v) != float("inf") for v in lps)
        ok = err is None and med is not None and finite and med > -0.5
        return {"test": "C1_prefill_logprob", "ok": ok, "median_tail_logprob": med, "n": len(lps), "finite": finite,
                "error": err, "sec": round(sec, 1)}

    def c2_counting(self, start=1, tag="C2_counting"):
        text = ", ".join(str(i) for i in range(start, start + 8)) + ","
        want = " " + ", ".join(str(i) for i in range(start + 8, start + 14))
        runs = []
        for _ in range(2):
            b, err, sec = self.generate(text, 48, logprob=True)
            runs.append((b, err))
        texts = [((b or {}).get("text") or "") for b, _ in runs]
        ids = [self.out_ids(b) for b, _ in runs]
        ok_text = all(t.startswith(want) for t in texts)
        same = ids[0] == ids[1] and texts[0] == texts[1]
        ids_ok = all(0 <= i < VOCAB for x in ids for i in x)
        ok = all(e is None for _, e in runs) and ok_text and same and ids_ok and not any(self.garbage(t) for t in texts)
        return {"test": tag, "ok": ok, "text": texts[0][:120], "want_prefix": want, "deterministic": same,
                "ids_in_vocab": ids_ok, "errors": [e for _, e in runs if e]}

    def c3_grid_echo(self):
        rng = random.Random(GRID_RNG_SEED)
        g = [[rng.randint(0, 9) for _ in range(6)] for _ in range(5)]
        rows = "\n".join(" ".join(map(str, r)) for r in g)
        text = ("Copy the grid exactly.\nInput:\n0 1 2\n3 4 5\nOutput:\n0 1 2\n3 4 5\n\n"
                "Input:\n7 7 0 1\n2 8 8 3\nOutput:\n7 7 0 1\n2 8 8 3\n\nInput:\n" + rows + "\nOutput:\n")
        b, err, sec = self.generate(text, 64)
        out = (b or {}).get("text") or ""
        ok = err is None and out.strip().startswith(rows)
        return {"test": "C3_grid_echo", "ok": ok, "got": out[:200], "error": err}

    def c4_marker(self, target_tokens, tokens_per_sentence=13.0):
        rng = random.Random(target_tokens)
        words = ["river", "stone", "lamp", "orbit", "maple", "violet", "copper", "signal", "harbor", "pixel"]
        code = f"ZK-{rng.randint(1000, 9999)}"
        sent, parts, approx = 0, [], 0
        while approx < target_tokens:                                     # calibrated from the first (8k) probe
            parts.append(f"Note {sent}: the {rng.choice(words)} near the {rng.choice(words)} is {rng.randint(0, 999)}.")
            approx += tokens_per_sentence; sent += 1
        parts.insert(max(1, len(parts) * 15 // 100), f"Remember this: the pass code is {code}.")
        text = " ".join(parts) + "\nQuestion: What is the pass code?\nAnswer: The pass code is"
        b, err, sec = self.generate(text, 12, timeout=1800)
        out = (b or {}).get("text") or ""
        pt = ((b or {}).get("meta_info") or {}).get("prompt_tokens")
        return {"test": f"C4_marker_{target_tokens // 1000}k", "ok": err is None and code in out, "prompt_tokens": pt,
                "got": out[:60], "want": code, "error": err, "sec": round(sec, 1)}

    def c5_tool_call(self, max_tokens=8192):
        msgs = [{"role": "user", "content": "Use the python tool to compute 123456789 * 987654321 and print it. "
                                            "Call the tool; do not compute it by hand."}]
        p = {"model": self.served, "messages": msgs, "tools": [TOOL], "tool_choice": "auto", "max_tokens": max_tokens}
        b, err, sec = self._post("/v1/chat/completions", p, 1800)
        ch = ((b or {}).get("choices") or [{}])[0]; m = ch.get("message") or {}
        calls = m.get("tool_calls") or []
        ok, args = False, None
        if calls:
            try:
                f = calls[0]["function"]; args = json.loads(f["arguments"]); ok = f["name"] == "python" and isinstance(args.get("code"), str)
            except Exception:
                ok = False
        verdict = "pass" if ok else ("inconclusive_length" if ch.get("finish_reason") == "length" else "fail")
        return {"test": "C5_tool_call", "ok": ok, "verdict": verdict, "finish": ch.get("finish_reason"),
                "reasoning_chars": len(m.get("reasoning_content") or ""), "code": (args or {}).get("code", "")[:200],
                "usage": (b or {}).get("usage"), "error": err, "sec": round(sec, 1), "_message": m, "_msgs": msgs}

    def c6_two_turn(self, c5, max_tokens=4096):
        m = c5.get("_message") or {}
        if not c5.get("ok"):
            return {"test": "C6_two_turn_preserve_thinking", "ok": False, "verdict": "skipped (C5 not ok)"}
        call = m["tool_calls"][0]
        msgs = c5["_msgs"] + [
            {"role": "assistant", "content": m.get("content"), "reasoning_content": m.get("reasoning_content"), "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call.get("id"), "content": "121932631112635269\n"}]
        p = {"model": self.served, "messages": msgs, "tools": [TOOL], "max_tokens": max_tokens}
        b, err, sec = self._post("/v1/chat/completions", p, 1800)
        ch = ((b or {}).get("choices") or [{}])[0]; mm = ch.get("message") or {}
        txt = (mm.get("content") or "") + (mm.get("reasoning_content") or "")
        ok = err is None and "121932631112635269" in txt.replace(",", "")
        verdict = "pass" if ok else ("inconclusive_length" if ch.get("finish_reason") == "length" else "fail")
        return {"test": "C6_two_turn_preserve_thinking", "ok": ok, "verdict": verdict, "finish": ch.get("finish_reason"),
                "prompt_tokens": ((b or {}).get("usage") or {}).get("prompt_tokens"),
                "cached_tokens": (((b or {}).get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                "error": err, "sec": round(sec, 1)}

    def c7_concurrent(self, levels):
        res = []
        for n in levels:
            outs = [None] * n

            def one(i):
                outs[i] = self.c2_counting(start=10 * (i + 1), tag=f"C7_{n}x_{i}")
            th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [t.start() for t in th]; [t.join() for t in th]
            res.append({"concurrency": n, "ok": all(o and o["ok"] for o in outs),
                        "fails": [o["text"] for o in outs if o and not o["ok"]][:3]})
        return {"test": "C7_concurrent", "ok": all(r["ok"] for r in res), "levels": res}

    def run_gate_c(self, kv_tokens, max_running, marker_sizes=(8000, 32000, 58000)):
        r = [self.c1_prefill_logprobs(), self.c2_counting(), self.c3_grid_echo()]
        tps = 13.0                                                         # tokens per filler sentence (estimate)
        for n in marker_sizes:
            if kv_tokens and n + 2000 > kv_tokens:
                r.append({"test": f"C4_marker_{n // 1000}k", "ok": None, "verdict": f"skipped: KV {kv_tokens} < {n}+2000"})
                continue
            res = self.c4_marker(n, tps)
            r.append(res)
            if res.get("prompt_tokens") and n == marker_sizes[0]:          # calibrate the next sizes on the real tokenizer
                tps = tps * res["prompt_tokens"] / n
                res["calibrated_tokens_per_sentence"] = round(tps, 2)
        c5 = self.c5_tool_call(); r.append({k: v for k, v in c5.items() if not k.startswith("_")})
        r.append(self.c6_two_turn(c5))
        r.append(self.c7_concurrent([x for x in (1, 4, 8) if x <= max(1, max_running or 1)]))
        hard = [x for x in r if x.get("ok") is False and x.get("verdict", "fail") == "fail"]
        hard_names = [x["test"] for x in hard]
        core_ok = all(any(x["test"].startswith(p) and x.get("ok") for x in r) for p in ("C1", "C2", "C3"))
        return {"pass": core_ok and not hard, "hard_failures": hard_names, "results": r,
                "signature": self.signature(r)}

    @staticmethod
    def signature(results):
        """Failure class for the pre-declared B2 choice (PLAN section 5)."""
        txt = " ".join(json.dumps(x, default=str) for x in results).lower()
        if "out of memory" in txt or "cuda error" in txt:
            return "oom_or_cuda_error"
        if any(x["test"].startswith(("C1", "C2", "C3")) and x.get("ok") is False for x in results):
            c1 = next((x for x in results if x["test"].startswith("C1")), {})
            if c1.get("ok") is False:
                return "prefill_wrong"                              # prefill logits already wrong: not a decode-only bug
            return "decode_wrong"
        if any(x["test"].startswith("C4") and x.get("ok") is False for x in results):
            return "long_context_wrong"
        if any(x["test"].startswith("C7") and x.get("ok") is False for x in results):
            return "concurrency_wrong"
        return "none"

    # ------------------------------------------------------------------ speed stage
    def speed(self, streams_list, prompt_tokens, max_new=512, words_per_token=0.4):
        """Aggregate decode tok/s with `n` concurrent long requests. Throughput = completion tokens / wall time of the
        decode window (from when the last prompt finished prefill, approximated by min end-to-end minus its own prefill)."""
        out = []
        for n in streams_list:
            res = [None] * n

            def one(i):
                text = f"Stream {i}. " + " ".join(f"word{(i * 7919 + j) % 997}" for j in range(int(prompt_tokens * words_per_token)))
                b, err, sec = self.generate(text, max_new, timeout=3600, temperature=0.6, ignore_eos=True)
                meta = (b or {}).get("meta_info") or {}
                res[i] = {"sec": sec, "err": err, "completion": meta.get("completion_tokens"), "prompt": meta.get("prompt_tokens"),
                          "garbage": self.garbage((b or {}).get("text"))}
            t0 = time.perf_counter()
            th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [t.start() for t in th]; [t.join() for t in th]
            wall = time.perf_counter() - t0
            comp = sum((x["completion"] or 0) for x in res)
            out.append({"streams": n, "prompt_tokens": prompt_tokens, "wall_sec": round(wall, 1), "completion_tokens": comp,
                        "e2e_tok_s": round(comp / wall, 1) if wall else None, "errors": [x["err"] for x in res if x["err"]],
                        "garbage": sum(x["garbage"] for x in res)})
            self.log(f"[speed] {out[-1]}")
        return out


# ---------------------------------------------------------------------- memory gate (server log)
PAT = {
    "load_end": r"Load weight end.*?avail mem=([\d.]+) GB, mem usage=([\d.]+) GB",
    "kv": r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB",
    "mamba": r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+)",
    "capped": r"max_running_requests is capped to (\d+)",
    "graph_end": r"Capture [\w ]*?(?:cuda|CUDA) graph end\..*?mem usage=([\d.]+) GB, avail mem=([\d.]+) GB",   # 0.5.21 format
    "avail_any": r"avail mem=([\d.]+) GB",
    "tp0_attn": r"(?i)(QSA|sparse full-attention|flash_attn|FA2|FA4|cute|triton).{0,80}(decode|backend)",
}


def memory_gate(log_text):
    tp0 = "\n".join(l for l in log_text.splitlines() if "TP0" in l) or log_text
    g = {}
    m = re.findall(PAT["load_end"], tp0); g["post_weight_avail_gb"] = float(m[0][0]) if m else None
    g["weight_gb"] = float(m[0][1]) if m else None
    m = re.findall(PAT["kv"], tp0); g["kv_dtype"] = m[0][0] if m else None; g["kv_tokens"] = int(m[0][1]) if m else None
    g["kv_gb"] = round(float(m[0][2]) + float(m[0][3]), 3) if m else None
    m = re.findall(PAT["mamba"], tp0); g["mamba_cache_size"] = int(m[0]) if m else None
    m = re.findall(PAT["capped"], log_text); g["capped_running"] = min(int(x) for x in m) if m else None
    m = re.findall(PAT["graph_end"], tp0)
    g["post_graph_avail_gb"] = float(m[-1][1]) if m else None
    g["graph_mem_gb"] = round(sum(float(x[0]) for x in m), 3) if m else None
    m = re.findall(PAT["avail_any"], tp0); g["last_avail_gb"] = float(m[-1]) if m else None
    g["attn_lines"] = [l[-200:] for l in log_text.splitlines()
                       if re.search(PAT["tp0_attn"], l) and "server_args" not in l][:8]
    g["marlin_errors"] = [l[-200:] for l in log_text.splitlines() if "Invalid thread config" in l or "marlin" in l.lower() and "error" in l.lower()][:5]
    g["oom"] = bool(re.search(r"(?i)out of memory", log_text))
    return g


def launch2_decision(gate_c, mem, peak_free_gb, micro, drafter_gb_per_rank=0.96, headroom_gb=1.5, kv_per_token=6144):
    """PLAN rev 4.1 section 3b. Returns (kind, detail). Pre-declared; the notebook only executes it."""
    if not gate_c.get("pass"):
        sig = gate_c.get("signature")
        if sig in ("oom_or_cuda_error", "load_failed", "exception") or mem.get("oom") or mem.get("marlin_errors"):
            return "stop", f"{sig}: load / Marlin / PLE / OOM class -> no blind B2 (root cause logged)"
        if micro.get("mqa_pass") is False:
            return "b2_tilelang_off", f"{sig}; micro-preflight indexer mismatch"
        if micro.get("decode_graph_pass") is False:
            return "b2_no_cuda_graph", f"{sig}; micro-preflight decode graph-replay mismatch"
        if sig == "decode_wrong" and micro.get("alt_decode_pass"):
            return "b2_other_decode", f"{sig}; switch FA2 <-> P1"
        return "b2_diagnostic", f"{sig}; unknown cause: TileLang off + CUDA graph off (diagnostic only, never a candidate)"
    post = mem.get("post_weight_avail_gb")
    budget = None
    if post is not None:
        mamba_gb = 0.016 * (mem.get("mamba_cache_size") or 0) * 9 / 5    # NEXTN adds 4 draft states per request (cookbook)
        graph_gb = mem.get("graph_mem_gb") or 0.5                         # measured in Launch 1; 0.5 GB if absent
        budget = post - drafter_gb_per_rank - mamba_gb - graph_gb * 1.5 - headroom_gb   # x1.5: verify graphs with NEXTN
    need = 2 * 48000 * kv_per_token / 1e9                                # >= 2 resident contexts at the measured p50
    if budget is not None and budget >= need:
        return "nextn_on", f"budget {budget:.2f} GB >= {need:.2f} GB for 2 x 48k fp8 contexts"
    if peak_free_gb is not None and peak_free_gb >= headroom_gb + 0.5:
        return "chunk_4096", f"NEXTN does not fit (budget {budget}); prefill-peak free {peak_free_gb} GB allows chunk 4096"
    return "none", f"NEXTN does not fit (budget {budget}); prefill-peak free {peak_free_gb} GB: no second launch"
