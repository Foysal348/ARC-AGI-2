"""E22: the JSON / choice values that vLLM's argparse does not validate, checked against the installed vLLM 0.32 classes
(Kaggle CPU kit, the vLLM 0.32 venv; no GPU). Prints one FIELDCHECK {...} line.
- every key of every --speculative-config JSON is a SpeculativeConfig field (use_local_argmax_reduction);
- every key of every --compilation-config JSON is a CompilationConfig field, and cudagraph_mode is a CUDAGraphMode name;
- --mamba-ssm-cache-dtype values are allowed by CacheConfig (bfloat16 / float16);
- --prefix-match-unit / --enable-mamba-shared-prefix-checkpoint exist as engine-arg fields;
- every key of the --profiler-config JSON is a ProfilerConfig field and "profiler" is an allowed value (rev 2b).
usage: <vllm 0.32 python> e22_fieldcheck.py <e22_flags.json>"""
import dataclasses
import json
import sys
import typing

flags_all = json.load(open(sys.argv[1], encoding="utf-8"))
import vllm  # noqa: E402
from vllm.config import CacheConfig, CompilationConfig, SpeculativeConfig  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402

out = {"vllm": vllm.__version__, "checks": [], "errors": []}


def fields(cls):
    try:
        return {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        return set(getattr(cls, "model_fields", {}) or getattr(cls, "__annotations__", {}))


SPEC_F, COMP_F, CACHE_F, EA_F = fields(SpeculativeConfig), fields(CompilationConfig), fields(CacheConfig), fields(EngineArgs)
try:
    from vllm.config.compilation import CUDAGraphMode  # noqa: E402
    MODES = {m.name for m in CUDAGraphMode}
except Exception as e:
    MODES, out["cudagraph_mode_import"] = set(), repr(e)[:200]
try:
    ssm_t = typing.get_type_hints(CacheConfig).get("mamba_ssm_cache_dtype")
    SSM_ALLOWED = set(typing.get_args(ssm_t)) or {str(ssm_t)}
except Exception as e:
    SSM_ALLOWED, out["ssm_type_error"] = set(), repr(e)[:200]
try:
    from vllm.config import ProfilerConfig  # noqa: E402
    PROF_F = fields(ProfilerConfig)
    try:
        _pt = typing.get_type_hints(ProfilerConfig).get("profiler")
        PROF_ALLOWED = {str(x) for x in typing.get_args(_pt) if x is not type(None)}
        PROF_ALLOWED = {a for x in typing.get_args(_pt) for a in (typing.get_args(x) or [x]) if isinstance(a, str)} or PROF_ALLOWED
    except Exception as e:
        PROF_ALLOWED, out["profiler_type_error"] = set(), repr(e)[:200]
except Exception as e:
    PROF_F, PROF_ALLOWED, out["profiler_config_import"] = set(), set(), repr(e)[:200]
out["profiler_fields"] = sorted(PROF_F)
out["profiler_allowed"] = sorted(str(x) for x in PROF_ALLOWED)
out["engine_arg_profiler_config"] = "profiler_config" in EA_F
out["ssm_allowed"] = sorted(str(x) for x in SSM_ALLOWED)
out["engine_arg_fields_present"] = {k: (k in EA_F) for k in ("prefix_match_unit", "enable_mamba_shared_prefix_checkpoint",
                                                              "mamba_ssm_cache_dtype", "max_num_batched_tokens")}

for key, row in sorted(flags_all.items()):
    f = row["flags"]
    rec = {"key": key, "ok": True, "notes": []}
    if "--speculative-config" in f:
        d = json.loads(f[f.index("--speculative-config") + 1])
        bad = [k for k in d if k not in SPEC_F]
        if bad:
            rec["ok"] = False
            rec["notes"].append(f"speculative-config keys not in SpeculativeConfig: {bad}")
    if "--compilation-config" in f:
        d = json.loads(f[f.index("--compilation-config") + 1])
        bad = [k for k in d if k not in COMP_F]
        if bad:
            rec["ok"] = False
            rec["notes"].append(f"compilation-config keys not in CompilationConfig: {bad}")
        if MODES and d.get("cudagraph_mode") not in MODES:
            rec["ok"] = False
            rec["notes"].append(f"cudagraph_mode {d.get('cudagraph_mode')} not in {sorted(MODES)}")
    if "--mamba-ssm-cache-dtype" in f:
        v = f[f.index("--mamba-ssm-cache-dtype") + 1]
        if SSM_ALLOWED and v not in {str(x) for x in SSM_ALLOWED}:
            rec["ok"] = False
            rec["notes"].append(f"mamba-ssm-cache-dtype {v} not in {sorted(str(x) for x in SSM_ALLOWED)}")
    for flag, field in (("--prefix-match-unit", "prefix_match_unit"),
                        ("--enable-mamba-shared-prefix-checkpoint", "enable_mamba_shared_prefix_checkpoint")):
        if flag in f and field not in EA_F:
            rec["ok"] = False
            rec["notes"].append(f"{flag}: EngineArgs has no field {field}")
    if "--profiler-config" in f:
        d = json.loads(f[f.index("--profiler-config") + 1])
        bad = [k for k in d if k not in PROF_F]
        if not PROF_F or bad:
            rec["ok"] = False
            rec["notes"].append(f"profiler-config keys not in ProfilerConfig {sorted(PROF_F)}: {bad}")
        if PROF_ALLOWED and d.get("profiler") not in PROF_ALLOWED:
            rec["ok"] = False
            rec["notes"].append(f"profiler {d.get('profiler')} not in {sorted(PROF_ALLOWED)}")
        if "profiler_config" not in EA_F:
            rec["ok"] = False
            rec["notes"].append("EngineArgs has no field profiler_config")
    out["checks"].append(rec)
    if not rec["ok"]:
        out["errors"].append(rec)
out["checked"] = len(out["checks"])
out["all_ok"] = not out["errors"] and bool(SPEC_F) and bool(COMP_F)
out["has_use_local_argmax_reduction"] = "use_local_argmax_reduction" in SPEC_F
print("FIELDCHECK " + json.dumps(out))
