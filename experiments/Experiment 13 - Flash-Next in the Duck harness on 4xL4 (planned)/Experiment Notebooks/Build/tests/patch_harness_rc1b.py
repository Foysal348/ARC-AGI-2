"""RC1b harness: the fake SGLang fixtures carry the dispatcher's EP map and a rank-local Marlin output (SGLang: global
ids in, reduce_results False); new checks for the RC1 audit fixes: deadline-bounded reference after all arms, map-based
ownership (global ids on rank 2), missing map, vLLM smoke gates (NaN logprob / wrong answer / a failed or short stream),
PLE cache flush fail-closed, vLLM arm host-RAM guard, one PLE prompt for every mode."""
from pathlib import Path

H = Path(r"D:/ARC AGI 2/_scratch/e13_rev6/harness")
f = H / "fake_e13_duck_server.py"
s = f.read_text(encoding="utf-8")
R = [
    ('''                y = torch.zeros(6, 256)
                for t in range(6):
                    for k in range(4):
                        Wg, Wu, Wd = truth[L][int(ids[t, k])]''',
     '''                y = torch.zeros(6, 256)
                for t in range(6):
                    for k in range(4):
                        if int(ids[t, k]) >= 4:              # SGLang EP: rank 0 owns experts 0..3, output = local sum
                            continue
                        Wg, Wu, Wd = truth[L][int(ids[t, k])]'''),
    ('''                       "meta": {"num_experts": 16, "num_local_experts": 4, "moe_ep_rank": 0, "moe_ep_size": 4, "top_k": 4}}''',
     '''                       "meta": {"num_experts": 16, "num_local_experts": 4, "moe_ep_rank": 0, "moe_ep_size": 4, "top_k": 4,
                                "reduce_results": False},
                       "expert_map": torch.tensor([0, 1, 2, 3] + [-1] * 12, dtype=torch.int32), "hidden_from": "pre_hook"}'''),
    ('''        nxt = {"1,": " 9, 10, 11, 12", "11": " 19, 20, 21, 22", "21": " 29, 30, 31, 32"}[prompt.split()[0][:2] if not prompt.startswith("1,") else "1,"]
        return nxt if ok else "!" * 32''',
     '''        if prompt != "1, 2, 3, 4, 5, 6, 7, 8,":
            return "WRONG PROMPT"
        return " 9, 10, 11, 12" if ok else "!" * 32'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
f.write_text(s, encoding="utf-8", newline="\n")

h = H / "harness_e13_duck_nb.py"
t = h.read_text(encoding="utf-8")
A = '''    check("RC1 reference: checkpoint scales finite; L1 Marlin == CPU GPTQ reference (rel err < 1e-2); L2 Marlin NaN, reference finite",
          (ref.get("ckpt_scales") or {}).get("1", {}).get("nonfinite_n") == 0 and (ref.get("ckpt_scales") or {}).get("1", {}).get("tensors") == 48
          and (fx.get(1) or {}).get("rel_err_full", 9) < 1e-2 and (fx.get(1) or {}).get("marlin_finite") is True
          and (fx.get(2) or {}).get("marlin_finite") is False and (fx.get(2) or {}).get("ref_finite") is True
          and (fx.get(2) or {}).get("marlin_nonfinite_tokens") == 6, ref)'''
B = '''    check("RC1 reference (after all arms): checkpoint scales finite; L1 Marlin == rank-local CPU reference (map-based, rel err < 1e-2); "
          "L2 Marlin NaN vs finite reference -> verdict MARLIN PATH FAULT",
          (ref.get("ckpt_scales") or {}).get("1", {}).get("nonfinite_n") == 0 and (ref.get("ckpt_scales") or {}).get("1", {}).get("tensors") == 48
          and (fx.get(1) or {}).get("rel_err", 9) < 1e-2 and (fx.get(1) or {}).get("compare_to", "").startswith("rank-local")
          and (fx.get(1) or {}).get("mapping", {}).get("contiguous_in_order") is True
          and (fx.get(2) or {}).get("marlin_finite") is False and (fx.get(2) or {}).get("ref_finite") is True
          and ref.get("reference_validated_on_layer1") is True and str(ref.get("verdict", "")).startswith("MARLIN PATH FAULT"), ref)
    import torch as _torch
    _tdir = WORK / "q27x" / "e13_memtrace" / "rc1"
    _t0 = time.perf_counter()
    _late = G["e13_rc1_reference"](_tdir, FN, time.perf_counter() + 1)
    check("RC1 reference respects its deadline (1 s left -> truncated, returns at once)",
          _late.get("truncated") and time.perf_counter() - _t0 < 10 and "INCONCLUSIVE" in str(_late.get("verdict")), _late)
    _truth = _torch.load(WORK / "rc1_truth.pt", weights_only=False)
    _sd = WORK / "rc1_map_test"; _sd.mkdir(exist_ok=True)
    _x = _torch.randn(1, 256).to(_torch.bfloat16); _ids = _torch.tensor([[3, 9, 10, 14]], dtype=_torch.int32)
    _w = _torch.tensor([[0.4, 0.3, 0.2, 0.1]]); _y = _torch.zeros(1, 256)
    for _k, _g in enumerate([3, 9, 10, 14]):
        if 8 <= _g < 12:                                  # rank 2 owns global experts 8..11
            _Wg, _Wu, _Wd = _truth[1][_g]
            _y[0] += ((_torch.nn.functional.silu(_x.float() @ _Wg) * (_x.float() @ _Wu)) @ _Wd)[0] * _w[0, _k]
    _em = _torch.tensor([-1] * 8 + [0, 1, 2, 3] + [-1] * 4, dtype=_torch.int32)
    _meta = {"num_experts": 16, "num_local_experts": 4, "moe_ep_rank": 2, "moe_ep_size": 4, "top_k": 4, "reduce_results": False}
    _torch.save({"name": "model.layers.1.mlp.experts", "fw": 0, "rank": 2, "hidden": _x, "ints": [_ids], "int_shapes": [[1, 4]],
                 "floats": [_x, _w], "float_shapes": [[1, 256], [1, 4]], "out": _y.to(_torch.bfloat16), "meta": _meta,
                 "expert_map": _em, "hidden_from": "pre_hook"}, _sd / "fixture_L1_fw0_pid7.pt")
    _torch.save({"name": "model.layers.2.mlp.experts", "fw": 0, "rank": 2, "hidden": _x, "ints": [_ids], "int_shapes": [[1, 4]],
                 "floats": [_x, _w], "float_shapes": [[1, 256], [1, 4]], "out": _y.to(_torch.bfloat16), "meta": _meta},
                _sd / "fixture_L2_fw0_pid7.pt")
    _mr = G["e13_rc1_reference"](_sd, FN, time.perf_counter() + 300)
    _f1 = next((r for r in _mr.get("fixtures") or [] if r.get("layer") == 1), {})
    _f2 = next((r for r in _mr.get("fixtures") or [] if r.get("layer") == 2), {})
    check("RC1 reference ownership from the captured map: global ids [3, 9, 10, 14] on EP rank 2 -> experts 9 and 10 local, rel err < 1e-2",
          _f1.get("rel_err", 9) < 1e-2 and _f1.get("mapping", {}).get("selected_owned") == 2
          and _f1.get("mapping", {}).get("owned_first") == [8] and _f1.get("mapping", {}).get("contiguous_in_order") is True, _f1)
    check("RC1 reference: a fixture without an expert map under EP is not compared", "missing expert_map" in str(_f2.get("error")), _f2)
    _real_vpost = G["_e13_vpost"]

    def _fake_vpost_factory(lp=(-0.01,) * 8, answer="4", stream=lambda i: 256):
        def _vp(path, body, deadline, cap):
            if path == "/v1/chat/completions":
                assert body.get("chat_template_kwargs") == {"enable_thinking": False}
                return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}], "usage": {}}
            if str(body.get("prompt", "")).startswith("1, 2, 3"):
                return {"choices": [{"text": " 9, 10, 11, 12", "logprobs": {"token_logprobs": list(lp)}}]}
            i = int(body["prompt"].split("puzzle ")[1].rstrip("."))
            n = stream(i)
            if isinstance(n, Exception):
                raise n
            return {"usage": {"completion_tokens": n}}
        return _vp
    _cases = {"good": {}, "nan_logprob": {"lp": (-0.1, float("nan"), float("inf"))}, "answer_14": {"answer": "14"},
              "answer_in_think_only": {"answer": "<think>2+2=4</think>\\n\\n5"},
              "stream_timeout": {"stream": lambda i: TimeoutError("t") if i == 2 else 256},
              "stream_255": {"stream": lambda i: 255 if i == 1 else 256}}
    _sm = {}
    for _n, _kw in _cases.items():
        G["_e13_vpost"] = _fake_vpost_factory(**_kw)
        _sm[_n] = G["e13_vllm_smoke"](time.perf_counter() + 120)
    G["_e13_vpost"] = _real_vpost
    check("vLLM smoke gates: good -> ok; NaN/Inf logprob, answer 14, 4 only inside thinking, a timed-out stream, a 255-token stream -> not ok",
          _sm["good"]["ok"] is True and all(_sm[k]["ok"] is False for k in _cases if k != "good")
          and _sm["nan_logprob"]["logprobs_ok"] is False and _sm["answer_14"]["chat_ok"] is False
          and _sm["answer_in_think_only"]["chat_ok"] is False
          and _sm["stream_timeout"]["smoke_correct"] is True and _sm["stream_timeout"]["throughput_valid"] is False
          and _sm["stream_255"]["streams4_ok"] is False, {k: {kk: v.get(kk) for kk in ("ok", "smoke_correct", "throughput_valid",
                                                                                     "logprobs_ok", "chat_ok", "streams4_ok")} for k, v in _sm.items()})
    _pr = G["e13_ple_mode_runs"](time.perf_counter() + 60, ["sync"])     # the server is gone: the flush must fail closed
    check("PLE runs fail closed when /flush_cache fails (no prompt sent, valid_comparison False, flag file reset)",
          len(_pr) == 1 and "cache isolation failed" in str(_pr[0].get("error")) and _pr[0].get("valid_comparison") is False
          and "count_text" not in _pr[0] and not (WORK / "e13_ple_mode").exists(), _pr)
    _real_guard = G["e13_host_ram_guard"]
    G["e13_host_ram_guard"] = lambda: {"ok": False, "why": "harness"}
    _va = G["e13_vllm_arm"]("vllm_guard_test", {}, 600)
    G["e13_host_ram_guard"] = _real_guard
    check("vLLM arm fails closed on the host-RAM guard (no install, no launch)",
          _va.get("outcome") == "skipped: host RAM guard" and "flags" not in _va, _va)
    check("PLE: the same prompt in every mode", len({r.get("prompt") for r in pr1}) == 1, [r.get("prompt") for r in pr1])'''
assert t.count(A) == 1
t = t.replace(A, B)
h.write_text(t, encoding="utf-8", newline="\n")
print("harness rc1b patched")
