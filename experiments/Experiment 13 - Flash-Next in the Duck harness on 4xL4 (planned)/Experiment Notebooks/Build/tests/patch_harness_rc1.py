"""E13 RC1 harness: scenario `rc1`. Harness: a tiny GPTQ checkpoint (layers 1-2, 16 experts, hidden 256, inter 128) in the
fake Flash-Next dir + its float truth; the fake vLLM server; the flag-file path. Fake SGLang server: RC1 env / flag checks,
SIMULATED rc1_scan / ple / fixture records (L1 Marlin output = the true MoE, L2 = NaN), /flush_cache, the count reply per
PLE mode (skip mode correct). Checks: arms, PLE runs, summary, CPU reference (L1 rel err ~0, L2 NaN vs finite ref), vLLM arm."""
from pathlib import Path

H = Path(r"D:/ARC AGI 2/_scratch/e13_rev6/harness")
f = H / "fake_e13_duck_server.py"
s = f.read_text(encoding="utf-8")
assert "SCEN == \"rc1\"" not in s
A = '''_RUN = int(argv[argv.index("--max-running-requests") + 1]) if "--max-running-requests" in argv else 4'''
B = '''if SCEN == "rc1":   # E13 RC1: eager, mem 0.93, mamba 20, the RC1 env; SIMULATED rc1 records + fixtures in the hook's format
    _td = os.environ.get("E13_MEMTRACE_DIR")
    _need = {"E13_NANPROBE": "1", "E13_RC1_SCAN": "1", "E13_RC1_FIXTURE": "1", "E13_NAN_ONLY": r"(^|\\.)layers\\.[0-3](\\.|$)"}
    _bad = [k for k, v in _need.items() if os.environ.get(k) != v]
    if _bad or "--disable-cuda-graph" not in argv or not _td or not os.environ.get("E13_PLE_FLAG"):
        print(f"ValueError: E13 harness: RC1 env/flags wrong {_bad}", flush=True); sys.exit(1)
    _L2NM = os.environ.get("E13_L2_NO_MARLIN") == "1"
    os.makedirs(_td, exist_ok=True)
    _TR = os.path.join(_td, "trace_3000.jsonl")
    _sc = lambda bad: {str(L): {"module": f"model.layers.{L}.mlp.experts", "type": "FusedMoE",
                                "quant_method": "MoeWNA16Method" if (L == 2 and _L2NM) else "GPTQMarlinMoEMethod", "scheme": "NoneType",
                                "num_experts": 16, "num_local_experts": 4, "moe_ep_rank": 0, "moe_ep_size": 4,
                                **({"src": {"quant_method": {"file": "gptq_moe.py", "sha256": "ab" * 32, "params_dtype_marker": 2,
                                                             "half_marker": 0}}} if L == 0 else {}),
                                "tensors": {"w13_scales": {"dtype": "bfloat16", "shape": [4, 2, 256], "nonfinite": 0, "min": 1e-3, "max": 0.05},
                                            "w13_qweight": {"dtype": "int32", "shape": [4, 16, 512], "sum": 12345 + L, "wsum": 999 + L + bad,
                                                            "min": -5, "max": 7}}} for L in range(4)}
    with open(_TR, "w", encoding="utf-8") as _f:
        for _rec in ([{"ev": "patched"}, {"ev": "load_weights_only_end", "dev": 0, "alloc": 1}, {"ev": "postprocess_end", "alloc": 1},
                      {"ev": "gc_summary", "collections": 1, "freed_bytes": 0}, {"ev": "rc1_ple_patched", "module": "sglang.srt.models.qwen4_exp"}]
                     + ([{"ev": "rc1_ar_patched", "module": "sglang.srt.layers.quantization.auto_round"},
                         {"ev": "rc1_l2_no_marlin", "prefix": "model.layers.2.mlp.experts", "qm": "MoeWNA16Method"}] if _L2NM else [])
                     + [{"ev": "rc1_scan", "when": "after_postprocess", "layers": _sc(0)},
                        {"ev": "nan_hooks_installed", "modules": 60, "forwards": 240, "absmax_limit": 1e4}]):
            _f.write(json.dumps(_rec) + "\\n")
    _GEN = [0]

    def _rc1_generate(prompt):
        mode = "orig"
        try:
            mode = open(os.environ["E13_PLE_FLAG"], encoding="utf-8").read().strip() or "orig"
        except OSError:
            pass
        ok = mode == "skip" or _L2NM
        recs = [{"ev": "ple_mode", "mode": mode},
                {"ev": "nan_forward", "fw": _GEN[0], "forward_mode": "EXTEND", "batch_size": 1, "ple_mode": mode,
                 "first_nonfinite": [] if ok else [{"module": "model.layers.2.mlp.experts", "type": "FusedMoE", "tensor_idx": 0,
                                                    "inputs_bad": False, "prev_clean_module": "model.layers.2.mlp.topk"}],
                 "first_big": [], "const": [], "indexer": [], "layers": {"model.layers.1": [-1.2, 1.5]},
                 "moe_in": {"model.layers.2.mlp.experts": [{"dtype": "bfloat16", "shape": [6, 256], "stride": [256, 1], "contig": True,
                                                            "min": -1.0, "max": 1.0, "nonfinite": False},
                                                           {"dtype": "int32", "shape": [6, 4], "stride": [4, 1], "contig": True,
                                                            "min": 0, "max": 15, "neg1": 0}]},
                 "logits": {"present": True, "nonfinite": not ok}}]
        if _GEN[0] == 0:
            import torch
            truth = torch.load(os.environ["FAKE_RC1_TRUTH"], weights_only=False)
            g = torch.Generator().manual_seed(7)
            for L in (1, 2):
                x = torch.randn(6, 256, generator=g).to(torch.bfloat16)
                ids = torch.stack([torch.randperm(16, generator=g)[:4] for _ in range(6)]).to(torch.int32)
                w = torch.softmax(torch.randn(6, 4, generator=g), -1)
                y = torch.zeros(6, 256)
                for t in range(6):
                    for k in range(4):
                        Wg, Wu, Wd = truth[L][int(ids[t, k])]
                        xt = x[t:t + 1].float()
                        y[t] += ((torch.nn.functional.silu(xt @ Wg) * (xt @ Wu)) @ Wd)[0] * w[t, k]
                if L == 2 and not _L2NM:
                    y[:] = float("nan")
                fix = {"name": f"model.layers.{L}.mlp.experts", "fw": 0, "rank": 0, "hidden": x, "ints": [ids], "int_shapes": [[6, 4]],
                       "floats": [x, w], "float_shapes": [[6, 256], [6, 4]], "out": y.to(torch.bfloat16),
                       "meta": {"num_experts": 16, "num_local_experts": 4, "moe_ep_rank": 0, "moe_ep_size": 4, "top_k": 4}}
                torch.save(fix, os.path.join(_td, f"fixture_L{L}_fw0_pid3000.pt"))
                recs.append({"ev": "rc1_fixture", "file": f"fixture_L{L}_fw0_pid3000.pt", "layer": L, "fw": 0, "rank": 0})
            recs.append({"ev": "rc1_scan", "when": "after_forward_0", "layers": _sc(1)})
        _GEN[0] += 1
        with open(_TR, "a", encoding="utf-8") as _f:
            for _rec in recs:
                _f.write(json.dumps(_rec) + "\\n")
        nxt = {"1,": " 9, 10, 11, 12", "11": " 19, 20, 21, 22", "21": " 29, 30, 31, 32"}[prompt.split()[0][:2] if not prompt.startswith("1,") else "1,"]
        return nxt if ok else "!" * 32
''' + A
assert s.count(A) == 1
s = s.replace(A, B)
A2 = '''        if self.path != "/generate":
            return self._send("{}", 404)
        if "text" in req:                              # E13 engine smoke: raw greedy counting, non-stream'''
B2 = '''        if self.path == "/flush_cache":
            return self._send("Cache flushed.", 200)
        if self.path != "/generate":
            return self._send("{}", 404)
        if "text" in req and SCEN == "rc1":
            return self._send(json.dumps({"text": _rc1_generate(req["text"]), "output_ids": [0] * 16,
                                          "meta_info": {"prompt_tokens": 12, "completion_tokens": 32}}))
        if "text" in req:                              # E13 engine smoke: raw greedy counting, non-stream'''
assert s.count(A2) == 1
s = s.replace(A2, B2)
f.write_text(s, encoding="utf-8", newline="\n")

h = H / "harness_e13_duck_nb.py"
t = h.read_text(encoding="utf-8")
A3 = '''if SCEN in ("loadprobe", "nanprobe"):
    os.environ.update(E13_OVERLAY=str(OV), E13_FP8PLE_DST=str(WORK / "fp8ple_model"), E13_HOOK_DIR=str(WORK / "e13_hook"))'''
B3 = A3.replace('("loadprobe", "nanprobe")', '("loadprobe", "nanprobe", "rc1")') + '''
if SCEN == "rc1":   # RC1: a tiny GPTQ checkpoint (auto_gptq v1 packing) + float truth; the fake vLLM server; the PLE flag file
    import numpy as np
    import torch
    from safetensors.torch import save_file
    gen = torch.Generator().manual_seed(3)

    def _pack(qw, qz, sc):
        K, N = qw.shape
        a = np.zeros((K // 8, N), dtype=np.uint64)
        for i in range(8):
            a |= qw[i::8, :].numpy().astype(np.uint64) << np.uint64(4 * i)
        z = np.zeros((qz.shape[0], N // 8), dtype=np.uint64)
        for i in range(8):
            z |= (qz[:, i::8].numpy().astype(np.uint64)) << np.uint64(4 * i)
        return (torch.from_numpy((a & 0xFFFFFFFF).astype(np.uint32).view(np.int32)),
                torch.from_numpy((z & 0xFFFFFFFF).astype(np.uint32).view(np.int32)), sc)
    tens, truth = {}, {1: {}, 2: {}}
    for L in (1, 2):
        for e in range(16):
            Ws = []
            for p, (K, N) in (("gate_proj", (256, 128)), ("up_proj", (256, 128)), ("down_proj", (128, 256))):
                q = torch.randint(0, 16, (K, N), generator=gen)
                sc = (torch.rand(K // 128, N, generator=gen) * 0.02 + 0.001).to(torch.float16)
                stored = torch.full((K // 128, N), 7)                     # v1: zero 8 stored as 7
                Ws.append((q.float() - 8.0) * sc.float().repeat_interleave(128, 0))
                qw, qz, s_ = _pack(q, stored, sc)
                b = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}."
                tens[b + "qweight"], tens[b + "qzeros"], tens[b + "scales"] = qw, qz, s_
            truth[L][e] = tuple(Ws)
    save_file(tens, str(FN / "model-rc1-00001.safetensors"))
    json.dump({"weight_map": {k: "model-rc1-00001.safetensors" for k in tens}}, open(FN / "model.safetensors.index.json", "w"))
    torch.save(truth, WORK / "rc1_truth.pt")
    with socket.socket() as _s2:
        _s2.bind(("127.0.0.1", 0)); VPORT = _s2.getsockname()[1]
    os.environ.update(FAKE_RC1_TRUTH=str(WORK / "rc1_truth.pt"), E13_PLE_FLAG=str(WORK / "e13_ple_mode"),
                      E13_VLLM_FAKE=str(HERE / "fake_e13_vllm_server.py"), E13_VLLM_PORT=str(VPORT))'''
assert t.count(A3) == 1
t = t.replace(A3, B3)
A4 = 'elif SCEN in ("launchfail", "garbage"):'
B4 = '''elif SCEN == "rc1":
    pr = G["ENGINE_INFO"]["e13"].get("probe") or {}
    arms = {a.get("tag"): a for a in pr.get("arms", [])}
    a1, a2, a3 = arms.get("rc1") or {}, arms.get("vllm_fn") or {}, arms.get("l2_wna16") or {}
    check("RC1: arms in order rc1, vllm_fn, l2_wna16; no Duck; fallback submission",
          [a.get("tag") for a in pr.get("arms", [])] == ["rc1", "vllm_fn", "l2_wna16"] and not G["ENGINE_READY"]
          and all(a["attempt_1"] == q["input"] for t in test for a, q in zip(sub[t], test[t]["test"])), [a.get("tag") for a in pr.get("arms", [])])
    pr1 = a1.get("ple_runs") or []
    check("RC1 arm 1: three PLE modes, cache flushed each time, skip mode counts correctly, orig/sync garbage",
          [r.get("mode") for r in pr1] == ["orig", "sync", "skip"] and all(r.get("flush") == "ok" for r in pr1)
          and pr1[2].get("ok") and not pr1[0].get("ok") and not pr1[1].get("ok"), pr1)
    check("RC1: the PLE flag file is removed after the runs", not (WORK / "e13_ple_mode").exists(), "")
    m1 = a1.get("memtrace") or {}
    s1 = (next(iter(m1.values()), {}) or {}).get("rc1") or {}
    bm = s1.get("by_ple_mode") or {}
    check("RC1 summary: scans after the repack and after forward 0, the changed checksum found, NaN per PLE mode, MoE inputs, src sha",
          set(s1.get("scans") or {}) == {"after_postprocess", "after_forward_0"}
          and any(c.get("field") == "wsum" for c in s1.get("changed_after_forward0") or [])
          and bm.get("orig", {}).get("first_module") == "model.layers.2.mlp.experts" and bm.get("skip", {}).get("with_nonfinite") == 0
          and "model.layers.2.mlp.experts" in (s1.get("first_moe_in") or {})
          and ((s1.get("runtime_src") or {}).get("quant_method") or {}).get("params_dtype_marker") == 2, s1)
    ref = a1.get("rc1_reference") or {}
    fx = {r.get("layer"): r for r in ref.get("fixtures") or []}
    check("RC1 reference: checkpoint scales finite; L1 Marlin == CPU GPTQ reference (rel err < 1e-2); L2 Marlin NaN, reference finite",
          (ref.get("ckpt_scales") or {}).get("1", {}).get("nonfinite_n") == 0 and (ref.get("ckpt_scales") or {}).get("1", {}).get("tensors") == 48
          and (fx.get(1) or {}).get("rel_err_full", 9) < 1e-2 and (fx.get(1) or {}).get("marlin_finite") is True
          and (fx.get(2) or {}).get("marlin_finite") is False and (fx.get(2) or {}).get("ref_finite") is True
          and (fx.get(2) or {}).get("marlin_nonfinite_tokens") == 6, ref)
    sm = a2.get("smoke") or {}
    check("RC1 arm 2 vLLM: ready, count ok, 2+2 ok, 4 streams tok/s, KV + PLE + INC lines parsed, the Flash-Next flags",
          a2.get("outcome") == "ready" and sm.get("ok") and sm.get("streams4_tok_s", 0) > 0 and a2.get("kv_tokens") == [61440]
          and any("pinned=True" in l for l in a2.get("log_lines") or []) and any("INC" in l for l in a2.get("log_lines") or [])
          and "--engram-config" in (a2.get("flags") or []) and "--enable-expert-parallel" in (a2.get("flags") or []), (a2.get("outcome"), sm))
    s3 = (next(iter((a3.get("memtrace") or {}).values()), {}) or {}).get("rc1") or {}
    check("RC1 arm 3: layer 2 on MoeWNA16 (patch event), counts ok under orig PLE",
          a3.get("outcome") == "ready" and any(e.get("ev") == "rc1_l2_no_marlin" for e in s3.get("events") or [])
          and (a3.get("ple_runs") or [{}])[0].get("ok"), (a3.get("outcome"), s3.get("events"), a3.get("ple_runs")))
    check("RC1: e13_loadprobe.json + both server logs saved",
          (WORK / "q27x" / "e13_loadprobe.json").is_file() and (WORK / "q27x" / "sglang_server_e13_rc1.log").is_file()
          and (WORK / "q27x" / "vllm_server_e13_vllm_fn.log").is_file(), "")
''' + A4
assert t.count(A4) == 1
t = t.replace(A4, B4)
h.write_text(t, encoding="utf-8", newline="\n")
print("harness rc1 patched")
