"""E13 RC1b: apply the three RC1 audits' fixes to build_e13_duck.py (exact one-time replacements)."""
from pathlib import Path

p = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/Experiment Notebooks/Build/build_e13_duck.py")
s = p.read_text(encoding="utf-8")
R = [
    # one token-identical prompt for every PLE mode (the cache flush before each mode makes it cold)
    ('''_E13_PLE_PROMPTS = {"orig": ("1, 2, 3, 4, 5, 6, 7, 8,", "9, 10, 11"), "sync": ("11, 12, 13, 14, 15, 16, 17, 18,", "19, 20, 21"),
                    "skip": ("21, 22, 23, 24, 25, 26, 27, 28,", "29, 30, 31")}''',
     '''_E13_PLE_PROMPT = ("1, 2, 3, 4, 5, 6, 7, 8,", "9, 10, 11")   # RC1 audit: the same token ids in every PLE mode'''),
    ('''        prompt, want = _E13_PLE_PROMPTS[mode]''', '''        prompt, want = _E13_PLE_PROMPT'''),
    ('''    """RC1 (GPT audits: PLE A/B in one server): per mode, flush the radix cache (no reuse of NaN-poisoned mamba states),
    set the flag file, then a count prompt of its own (no shared prefix).''',
     '''    """RC1 (GPT audits: PLE A/B in one server): per mode, flush the radix cache (fail closed: no prompt without a
    successful flush, so the same prompt runs cold and no NaN-poisoned mamba state is reused), set the flag file, then
    the same count prompt in every mode (token-identical input, so routing differs only by the PLE mode).'''),
    # vLLM smoke gates: finite logprobs, the exact 2+2 answer without thinking, all 4 streams at 256 tokens
    ('''        out.update(count_text=text[:80], count_ok=text.strip().startswith("9, 10, 11"), garbage=_e13_garbage(text),
                   count_logprobs=[None if v is None else round(float(v), 4) for v in lp[:8]],
                   count_sec=round(time.perf_counter() - t, 1))
        t = time.perf_counter()
        m = _e13_vpost("/v1/chat/completions", {"model": "flashnext", "max_tokens": 1024, "temperature": 0, "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}, deadline, 900)
        msg = (m.get("choices") or [{}])[0].get("message") or {}
        txt = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or msg.get("reasoning") or "")
        out.update(chat_head=txt[:160], chat_tail=txt[-160:], chat_has_4="4" in txt, chat_usage=m.get("usage"),
                   chat_finish=(m.get("choices") or [{}])[0].get("finish_reason"), chat_sec=round(time.perf_counter() - t, 1))''',
     '''        lpv = [float(v) for v in lp if v is not None]
        out.update(count_text=text[:80], count_ok=text.strip().startswith("9, 10, 11"), garbage=_e13_garbage(text),
                   count_logprobs=[None if v is None else round(float(v), 4) for v in lp[:8]],
                   logprobs_ok=bool(lpv) and all(math.isfinite(v) for v in lpv), count_sec=round(time.perf_counter() - t, 1))
        t = time.perf_counter()
        m = _e13_vpost("/v1/chat/completions", {"model": "flashnext", "max_tokens": 64, "temperature": 0,
                                                "chat_template_kwargs": {"enable_thinking": False}, "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}, deadline, 600)
        msg = (m.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        answer = content.split("</think>")[-1].strip()           # the reply after any thinking, nothing from the reasoning
        out.update(chat_content=content[:160], chat_answer=answer[:40], chat_ok=re.fullmatch(r"\\**4\\**\\s*\\.?", answer) is not None,
                   chat_usage=m.get("usage"), chat_finish=(m.get("choices") or [{}])[0].get("finish_reason"),
                   chat_sec=round(time.perf_counter() - t, 1))'''),
    ('''        out.update(streams4=res, streams4_sec=round(sec, 1), streams4_tok_s=round(toks / max(sec, 1e-6), 1))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    out["ok"] = bool(out.get("count_ok")) and not out.get("garbage") and bool(out.get("chat_has_4")) and not out.get("error")
    return out''',
     '''        out.update(streams4=res, streams4_sec=round(sec, 1), streams4_tok_s=round(toks / max(sec, 1e-6), 1),
                   streams4_ok=len(res) == 4 and all(type(n) is int and n == 256 for n in res))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    # RC1 audits: correctness = count + finite logprobs + the exact answer; throughput only counts with all 4 streams
    out["smoke_correct"] = (bool(out.get("count_ok")) and not out.get("garbage") and bool(out.get("logprobs_ok"))
                            and bool(out.get("chat_ok")) and not out.get("error"))
    out["throughput_valid"] = out["smoke_correct"] and bool(out.get("streams4_ok"))
    out["ok"] = out["throughput_valid"]
    return out'''),
    # vLLM arm: fail-closed gates, install clamped to the arm deadline
    ('''    rec["host_ram"] = e13_host_ram_guard()
    rec["gpu_free_before"] = _e11_gpu_free(180)
    if rec["gpu_free_before"].get("free") is False:
        rec["outcome"] = "skipped: GPUs not free"; return rec
    t_deadline = min(time.perf_counter() + budget, HARD_STOP - 5 * 60)
    try:
        inst = e13_vllm_install()
    except Exception as e:
        rec["outcome"] = f"install error: {type(e).__name__}: {e}"[:1500]; return rec''',
     '''    rec["host_ram"] = e13_host_ram_guard()
    if not rec["host_ram"].get("ok", False):                           # RC1 audits: fail closed like the SGLang arm
        rec["outcome"] = "skipped: host RAM guard"; return rec
    rec["gpu_free_before"] = _e11_gpu_free(180)
    if rec["gpu_free_before"].get("free") is not True:
        rec["outcome"] = "skipped: GPU free status not verified"; return rec
    t_deadline = min(time.perf_counter() + budget, HARD_STOP - 5 * 60)
    try:
        inst = e13_vllm_install(t_deadline)
    except Exception as e:
        rec["outcome"] = f"install error: {type(e).__name__}: {e}"[:1500]; return rec
    if time.perf_counter() > t_deadline - 12 * 60:                     # no launch that cannot reach ready + smoke
        rec["outcome"] = f"skipped: {round(t_deadline - time.perf_counter())} s left after the install"; return rec'''),
    ('''def e13_vllm_install():
    global _E13_VLLM_INST''', '''def e13_vllm_install(deadline=None):
    global _E13_VLLM_INST'''),
    ('''                       timeout=max(60, min(2400, HARD_STOP - time.perf_counter() - 150)))
    if r.returncode:
        raise E11Fail("vLLM 0.32 install failed: "''',
     '''                       timeout=max(60, min(2400, HARD_STOP - time.perf_counter() - 150,
                                           (deadline or HARD_STOP) - time.perf_counter() - 60)))
    if r.returncode:
        raise E11Fail("vLLM 0.32 install failed: "'''),
    # SGLang arm: GPU-free fail closed too
    ('''    if rec["gpu_free_before"].get("free") is False:                  # fix list F17: never launch on busy GPUs
        rec["outcome"] = "skipped: GPUs not free"; return rec''',
     '''    if rec["gpu_free_before"].get("free") is not True:               # fix list F17 + RC1 audit: fail closed
        rec["outcome"] = "skipped: GPU free status not verified"; return rec'''),
    # the CPU reference leaves the GPU arm: queued, run after all arms with its own capped deadline
    ('''    if (cfg.get("env") or {}).get("E13_RC1_FIXTURE") == "1":            # RC1 test B: CPU reference vs Marlin
        try:
            rec["rc1_reference"] = e13_rc1_reference(tdir, model_dir)
        except Exception as e:
            rec["rc1_reference"] = {"error": f"{type(e).__name__}: {e}"[:400]}
        print(f"E13 RC1 {tag} reference:", json.dumps(rec["rc1_reference"], default=str)[:3000], flush=True)
''',
     '''    if (cfg.get("env") or {}).get("E13_RC1_FIXTURE") == "1":            # RC1 test B: CPU reference, after ALL GPU arms
        E13_RC1_PENDING.append((tag, tdir, model_dir))
        rec["rc1_reference"] = {"pending": "runs after all GPU arms (RC1 audit H1)"}
'''),
    ('''_E13_ENV = None
_t_engine = time.perf_counter()
E13_PROBE = {"arms": [], "budget_hard_stop_in_sec": round(HARD_STOP - time.perf_counter())}''',
     '''_E13_ENV = None
_t_engine = time.perf_counter()
E13_RC1_PENDING = []          # RC1: (arm tag, trace dir, model dir) whose CPU reference runs after all GPU arms
E13_PROBE = {"arms": [], "budget_hard_stop_in_sec": round(HARD_STOP - time.perf_counter())}'''),
    ('''        # rev 2 (F11): CPU-only, after the arms, its own guard (a timeout here never costs an arm)''',
     '''        # RC1 audits H1: the CPU reference after every GPU arm, with its own capped deadline; the GPU evidence is on disk
        # already (e13_loadprobe.json is written after each arm), so a truncated reference loses nothing
        for _tag, _tdir, _md in E13_RC1_PENDING:
            _arm = next((a for a in E13_PROBE["arms"] if a.get("tag") == _tag), {})
            _left = HARD_STOP - time.perf_counter() - 4 * 60
            _info["stage"] = f"rc1 reference {_tag}"
            if _left < 90:
                _arm["rc1_reference"] = {"skipped": f"{round(_left)} s left before the hard stop"}
            else:
                try:
                    _arm["rc1_reference"] = e13_rc1_reference(_tdir, _md, time.perf_counter() + min(15 * 60, _left))
                except Exception as _re:
                    _arm["rc1_reference"] = {"error": f"{type(_re).__name__}: {_re}"[:400]}
            print(f"E13 RC1 {_tag} reference:", json.dumps(_arm.get("rc1_reference"), default=str)[:3000], flush=True)
            (OUT / "e13_loadprobe.json").write_text(json.dumps(E13_PROBE, indent=1, default=str), encoding="utf-8")
        # rev 2 (F11): CPU-only, after the arms, its own guard (a timeout here never costs an arm)'''),
    # bounded expert cache (RC1 audit: ~18.75 MiB per dequantized expert)
    ('''    cache = {}

    class _Late(Exception):''',
     '''    cache = __import__("collections").OrderedDict()                  # bounded: 24 experts x ~18.75 MiB fp32

    def _put(key, val):
        cache[key] = val
        while len(cache) > 24:
            cache.popitem(last=False)

    class _Late(Exception):'''),
    ('''                W[p], W[p + "_zero_unique"], W[p + "_zero0"] = _e13_gptq_dequant(qw, qz, sc)
            cache[(L, e)] = W
        return cache[(L, e)]''',
     '''                W[p], W[p + "_zero_unique"], W[p + "_zero0"] = _e13_gptq_dequant(qw, qz, sc)
            _put((L, e), W)
            zero_seen.update({k: v for k, v in W.items() if "zero" in k})
        cache.move_to_end((L, e))
        return cache[(L, e)]'''),
    ('''                W[p] = w.float().t()
            cache[(L, "shared")] = W
        return cache[(L, "shared")]''',
     '''                W[p] = w.float().t()
            _put((L, "shared"), W)
        cache.move_to_end((L, "shared"))
        return cache[(L, "shared")]'''),
    ('''    rel = lambda a, b: float((a - b).norm() / max(float(b.norm()), 1e-9))
    for fp in sorted(Path(tdir).glob("fixture_L*_fw*_pid*.pt")):''',
     '''    rel = lambda a, b: float((a - b).norm() / max(float(b.norm()), 1e-9))
    zero_seen = {}
    for fp in sorted(Path(tdir).glob("fixture_L*_fw*_pid*.pt")):'''),
    ('''                     zero_convention={k: v for k, v in (next((w for kk, w in cache.items() if kk[1] != "shared"), {})).items() if "zero" in k})''',
     '''                     zero_convention=dict(zero_seen))'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:100]
    s = s.replace(a, b)
if "\nimport math\n" not in s.split("PROBE_SRC = r'''")[1][:3000]:
    s = s.replace("PROBE_SRC = r'''\n", "PROBE_SRC = r'''\nimport math\n", 1)
p.write_text(s, encoding="utf-8", newline="\n")
print("patched", len(R))
