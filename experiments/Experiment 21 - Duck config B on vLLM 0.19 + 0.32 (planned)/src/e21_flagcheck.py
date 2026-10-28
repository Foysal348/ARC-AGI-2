"""Parse E21's launch commands with the REAL vLLM argument parser of the installed version (Kaggle CPU kit; no GPU needed).
usage: <python with vllm X> e21_flagcheck.py <flags.json> <version>   -> prints one JSON line per command, then a summary.
flags.json = CONFIG["e21"]["flags"] of a notebook run ({"A_primary": {"vllm": "0.19.0", "flags": [...]}, ...}); only the
commands of <version> are checked. vLLM 0.19 commands use the api_server form (--model <path>); 0.32 commands use the
`vllm serve <path>` form, parsed here as --model <path> with the same make_arg_parser (the serve subcommand adds only
the positional model tag). Each parsed namespace is also turned into AsyncEngineArgs (types, choices, JSON values)."""
import json
import sys
import traceback

flags_all = json.load(open(sys.argv[1], encoding="utf-8"))
version = sys.argv[2]
import vllm  # noqa: E402

# rev 2 (Kaggle CPU kit run 1): on this CPU-only box vLLM finds no platform, and building the argument parser already
# fails (EngineArgs.add_cli_args -> DeviceConfig default -> "Failed to infer device type"). The parser needs only a
# device type, so install a stub platform that says "cuda" before the parser is built (nothing here touches a GPU).
import vllm.platforms as _vp  # noqa: E402
try:
    _vp.current_platform.device_type
    _need_stub = not _vp.current_platform.device_type
except Exception:
    _need_stub = True
if _need_stub:
    from vllm.platforms.interface import UnspecifiedPlatform  # noqa: E402

    class _E21StubPlatform(UnspecifiedPlatform):
        device_name = "cuda"
        device_type = "cuda"
        dispatch_key = "CUDA"
    _vp._current_platform = _E21StubPlatform()
    out_stub = True
else:
    out_stub = False

out = {"vllm": vllm.__version__, "want": version, "checked": 0, "ok": 0, "errors": {}, "stub_platform": out_stub}
assert vllm.__version__.startswith(version), (vllm.__version__, version)
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except Exception:
    from vllm.utils import FlexibleArgumentParser
try:      # 0.19: entrypoints/openai/cli_args.py; 0.32 moved it to entrypoints/launchers/cli_args.py (VAL_api_D1 F7)
    from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
except ImportError:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402

for key, row in sorted(flags_all.items()):
    if row["vllm"] != version:
        continue
    argv = list(row["flags"])
    if not argv[0].startswith("--"):
        argv = ["--model", argv[0]] + argv[1:]
    out["checked"] += 1
    rec = {"key": key}
    try:
        p = make_arg_parser(FlexibleArgumentParser())
        ns = p.parse_args(argv)
        # step 2 (rev 2, Kaggle CPU kit run 1): AsyncEngineArgs may need a device on this CPU-only box ("Failed to infer
        # device type"); that one error is recorded, not a flag failure. Any other error is.
        try:
            ea = AsyncEngineArgs.from_cli_args(ns)
            rec["engine_args"] = "ok"
        except Exception as e2:
            if "infer device type" not in str(e2):
                raise
            ea, rec["engine_args"] = None, "skipped on CPU: " + str(e2)[:120]
        rec.update(ok=True, parsed={k: str(getattr(ns, k, None)) for k in (
            "model", "served_model_name", "port", "max_num_seqs", "max_model_len", "kv_cache_dtype", "speculative_config",
            "attention_backend", "mamba_ssm_cache_dtype", "enable_prefix_caching", "tool_call_parser", "reasoning_parser",
            "enable_auto_tool_choice", "default_chat_template_kwargs", "language_model_only", "enable_prompt_tokens_details",
            "generation_config", "trust_remote_code", "disable_custom_all_reduce", "max_num_batched_tokens", "disable_uvicorn_access_log",
            "gpu_memory_utilization", "tensor_parallel_size")},
            engine_args_values=None if ea is None else {k: str(getattr(ea, k, None)) for k in (
                "mamba_ssm_cache_dtype", "kv_cache_dtype", "max_num_seqs", "enable_prefix_caching", "speculative_config")})
        out["ok"] += 1
    except SystemExit as e:
        rec.update(ok=False, error=f"SystemExit {e.code} (argparse rejected the command)")
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
    if not rec["ok"]:
        out["errors"][key] = rec.get("error")
    print(json.dumps(rec), flush=True)
out["all_ok"] = out["checked"] > 0 and out["ok"] == out["checked"]
print("FLAGCHECK", json.dumps(out), flush=True)
