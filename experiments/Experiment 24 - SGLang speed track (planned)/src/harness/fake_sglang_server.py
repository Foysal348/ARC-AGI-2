"""E24 fake SGLang 0.5.21 server (WP2, rev 2, 2026-10-30; stdlib only, no model, no GPU).

Run exactly like the real launcher:  [python, fake_sglang_server.py] + <sglang.launch_server argv>
  --help                        argparse help of ALL 534 option strings of 0.5.21 (the table below was generated from the
                                wheel's arg_groups/fields/*.py + ServerArgs.add_cli_args with the same rules as
                                arg_utils.add_cli_args_from_dataclass); `--help=all` fails with exit 2 like the real one.
  --e24-flag-check '<json argv>' [--resolve]
                                prints one JSON line {"ran","ok","unknown","ambiguous","parse_error","method"} (+
                                "resolve_error" with --resolve): every --flag must be an EXACT option string (argparse
                                would accept unique prefixes; --cuda-graph-bs is ambiguous), then the argv must parse.

Faithful to 0.5.21 (sources: SCR/understand/sglang_source_0521.md, the unpacked wheel, the 7 EMS server logs, the E11-04
/metrics + /server_info dumps and the E14A profiler traces):
- startup: UserWarning, `server_args={...}` (repr of the resolved dict: the real 519-key EMS W6 S20 dict as template,
  overridden by this argv), 4 TP scheduler processes + 1 detokenizer started with multiprocessing SPAWN (a sitecustomize on
  PYTHONPATH runs in each), per-rank TP-prefixed init lines, NCCL INFO lines (NCCL_DEBUG=INFO; "set by environment",
  "Channel 00/0N", subsystem filter), graph capture lines, the capacity line, uvicorn lines, the warmup chat request, "The
  server is fired up and ready to roll!"; /ready and /health answer 503 until then (the port is closed before uvicorn).
- scheduler: max-running (capped by max_mamba_cache_size // ratio), KV pool from mem-fraction / --max-total-tokens,
  new-token-ratio reservations, FCFS/LPM admission, chunked prefill that pre-empts decode (chunks back to back,
  enable_mixed_chunk False; the chunked request is neither running nor queued), --prefill-decode-interval N, the DFlash
  min-free-slots delayer, 64-token radix blocks with Mamba checkpoints (chunk ends, prompt end, sequence end, branch
  points), retraction when the KV pool is full, DFlash accept length by width, decode steps every step_ms.
- logs: "Prefill batch" / "Decode batch" (every --decode-log-interval steps) in the exact 0.5.21 formats, retraction
  warning, flush lines, access lines.
- /metrics: the real families, label sets, per-rank series and ordering of the E11-04 dump; finish-only tokenizer
  counters, realtime counters every forward on TP0, cuda_graph_passes on every rank, gauges refreshed only at prefill
  batches / every decode-log-interval / idle, num_retracted_requests_total absent until the first retraction.
- /generate (stream and not; cumulative unless --incremental-streaming-output), logprobs, input logprobs with the real
  layout ([None] + rows[:-1], entry j = lp(ids[s+j] | ids[:s+j])), token_ids_logprob (full vocab), meta_info, rid;
  /v1/completions SSE (return_token_ids, usage only with stream_options.include_usage, cached tokens only with
  --enable-cache-report and > 0), /v1/chat/completions (content, reasoning_content, tool_calls), /v1/models,
  /server_info, /model_info, /flush_cache (400 while busy), /abort_request, /health (1 s + a 1-token request),
  /start_profile + /stop_profile (per-rank <id>-TP-<r>.trace.json.gz with step[VERIFY/DRAFT/DECODE/EXTEND spans, graph
  launches and NCCL kernels whose grid is the channel count and whose name carries the ALGO and, as NCCL >= 2.19 names
  its specialised kernels, always _LL (the real E14A trace: ncclDevKernel_AllGather_RING_LL for an 11.9 MB all-gather);
  FAKE_SG_NCCL_PROTO_IN_NAME=1 puts the effective PROTO in the name instead).

Knobs (env): FAKE_TIME_SCALE (or E24_TIME_SCALE), FAKE_SG_STARTUP_SEC, FAKE_SG_SLOW_START_SEC, FAKE_SG_STEP_MS,
FAKE_SG_MIN_STEP_MS, FAKE_SG_PREFILL_TPS, FAKE_SG_NOISE, FAKE_SG_NOISE_KEY, FAKE_SG_PATH_NOISE, FAKE_SG_DIVERGE_AT,
FAKE_SG_DFLASH_BUG (+ _AT), FAKE_SG_LEAK_DEFAULT_CACHE, FAKE_SG_FAULT=crash_start|slow_start|oom_log|mamba_assert|
hang_decode|nccl_env_ignored|logprob_oom (+ FAKE_SG_FAULT_AFTER; oom_log: the allocator's recovered-OOM warning at the
first prefill after start and after every flush), FAKE_SG_RESTORE_DIVERGE_AT (a high-margin divergence only in requests
that resumed from a cached prefix), FAKE_SG_TEXT_STYLE=grids (output text = space-separated digit rows),
FAKE_SG_NCCL_PROTO_IN_NAME, FAKE_SG_HEALTH_SLEEP, FAKE_SG_STUB_MODULES,
FAKE_SG_ORIG_ARGV=0 (keep sys.orig_argv; default: the launcher reports `python -m sglang.launch_server <flags>`).
Real SGLang env honoured: SGLANG_LOGPROB_CHUNK_SIZE (logprob_oom), SGLANG_SIMULATE_ACC_LEN / _TOKEN_MODE,
SGLANG_TORCH_PROFILER_DIR, SGLANG_PROFILE_WITH_STACK / _RECORD_SHAPES, SGLANG_LOG_MS, SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION,
SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK (Mamba slot ratio), SGLANG_CACHE_DIR & co (cache redirection), NCCL_*.
"""
import argparse
import ast
import json
import math
import multiprocessing
import os
import queue
import random
import re
import signal
import socket
import sys
import threading
import time
import traceback
from http import HTTPStatus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_common as C  # noqa: E402

VERSION = "0.5.21"
NCCL_VERSION = "2.29.7"
GIB = 1024 ** 3
TOTAL_GIB = 22.034            # L4 usable (23,659,151,360 B)
MODEL_CTX = 262144            # Qwen3.8-27B max_position_embeddings
BLOCK = 64                    # radix/Mamba checkpoint alignment (FLA chunk)
CLIP_NEW = 4096               # SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION
INIT_RATIO, MIN_RATIO_FACTOR, RATIO_DECAY_STEPS, RETRACT_DECODE_STEPS = 0.7, 0.14, 600, 20
DEFAULT_PREFILL_TPS = 1650.0  # tok/s at empty context (EMS logs: 1615 -> 1280 at 86k)
# fields the EMS W6 S20 argv set explicitly (the template dict holds their EMS values; unset here -> declared default)
EMS_SET = ("model_path tp_size kv_cache_dtype context_length chunked_prefill_size mamba_ssm_dtype enable_metrics "
           "disable_flashinfer_autotune random_seed host port log_level watchdog_timeout decode_log_interval "
           "speculative_algorithm speculative_draft_model_path speculative_draft_model_quantization "
           "disable_prefill_cuda_graph mamba_radix_cache_strategy mem_fraction_static incremental_streaming_output "
           "stream_interval reasoning_parser tool_call_parser default_chat_template_kwargs schedule_policy "
           "enable_cache_report speculative_num_draft_tokens min_free_slots_delay max_running_requests "
           "cuda_graph_bs_decode max_mamba_cache_size tokenizer_path served_model_name").split()
SPEC_FIELDS = ("speculative_draft_model_revision", "speculative_num_steps", "speculative_eagle_topk",
               "speculative_draft_attention_backend")


# ================================================================================================ CLI (0.5.21 parser)
def _flag_rows():
    rows = [json.loads(l) for l in FLAGS_TABLE.strip().splitlines()]
    named = dict(json.loads(l) for l in CHOICE_LISTS.strip().splitlines())
    for r in rows:
        if isinstance(r[4], str) and r[4].startswith("@"):
            r[4] = named[r[4]]
    return rows


def _hri(value):
    """srt/utils/common.py human_readable_int: SI (k, M, G, T; decimals ok) and IEC (Ki, Mi, Gi, Ti) suffixes."""
    v = value.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(Ki|Mi|Gi|Ti|k|M|G|T)", v)
    if m:
        num, suf = m.groups()
        iec = {"Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40}
        if suf in iec:
            if "." in num:
                raise argparse.ArgumentTypeError("Decimals are not allowed with IEC suffixes like '%s'." % suf)
            return int(num) * iec[suf]
        return int(float(num) * {"k": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12}[suf])
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid human-readable integer: %r" % value)


def _json(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(str(e))


def _nullable_str(v):
    return None if not v or v == "None" else v


TYPES = {"int": int, "float": float, "str": str, "": str, "human_readable_int": _hri, "json.loads": _json,
         "json_list_type": _json, "nullable_str": _nullable_str, "parse_cuda_graph_config_arg": str}


class _DeprecatedStoreTrue(argparse.Action):
    def __init__(self, option_strings, dest, **kw):
        kw.pop("nargs", None)
        super().__init__(option_strings, dest, nargs=0, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def build_parser():
    """argparse.ArgumentParser(prog="sglang serve") + every 0.5.21 option (allow_abbrev stays True, as the real one)."""
    p = argparse.ArgumentParser(prog="sglang serve")
    for names, kind, typ, nargs, choices, default, dest, req, help_ in _flag_rows():
        kw = {"dest": dest, "help": help_ or None}
        if default != "@MISSING":
            kw["default"] = default
        if kind == "store_true":
            if dest == "disable_cuda_graph":
                p.add_argument(*names, action=_DeprecatedStoreTrue, **kw)
            else:
                p.add_argument(*names, action="store_true", **kw)
        elif kind == "bool_opt":
            p.add_argument(*names, action=argparse.BooleanOptionalAction, **kw)
        elif kind == "lora":
            p.add_argument(*names, nargs="*", type=str, **kw)
        elif kind == "list":
            p.add_argument(*names, nargs=nargs or "+", type=TYPES.get(typ, str), choices=choices, **kw)
        else:
            if nargs == "?":
                kw.update(nargs="?", const="")
            if req:
                kw["required"] = True
            p.add_argument(*names, type=TYPES.get(typ, str), choices=choices, **kw)
    return p


def option_strings(parser=None):
    p = parser or build_parser()
    return set(p._option_string_actions)


def flag_check(argv, resolve=False):
    """The lab's flag_check result for a fake run: exact option strings, then argparse parse (+ fake resolution)."""
    res = {"ran": True, "ok": False, "unknown": [], "ambiguous": [], "parse_error": None,
           "method": "fake_sglang_0.5.21_option_strings+argparse"}
    p = build_parser()
    opts = set(p._option_string_actions)
    for tok in argv:
        tok = str(tok)
        if not tok.startswith("--"):
            continue
        name = tok.split("=", 1)[0]
        if name in opts:
            continue
        pref = sorted(o for o in opts if o.startswith(name) and o.startswith("--"))
        (res["ambiguous"] if len(pref) > 1 else res["unknown"]).append(name)
    try:
        import io
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            ns = p.parse_args([str(a) for a in argv])
        finally:
            sys.stderr = old
    except SystemExit:
        res["parse_error"] = (err.getvalue().strip().splitlines() or ["argparse exit 2"])[-1][:400]
        ns = None
    if resolve and ns is not None:
        res["method"] += "+fake_resolve"
        res["resolve_error"] = None
        try:
            resolve_args(ns, [str(a) for a in argv], quiet=True)
        except Exception as e:
            res["resolve_error"] = "%s: %s" % (type(e).__name__, e)
    res["ok"] = (not res["unknown"] and not res["ambiguous"] and res["parse_error"] is None
                 and not res.get("resolve_error"))
    return res


def given_dests(parser, argv):
    """dest of every option the argv actually used (exact or unique prefix, as argparse resolves it)."""
    out = set()
    opts = parser._option_string_actions
    for tok in argv:
        if not str(tok).startswith("--"):
            continue
        name = str(tok).split("=", 1)[0]
        act = opts.get(name)
        if act is None:
            pref = [o for o in opts if o.startswith(name)]
            act = opts.get(pref[0]) if len(pref) == 1 else None
        if act is not None:
            out.add(act.dest)
    return out


# ================================================================================================ resolution
class Cfg:
    """Resolved values the simulation uses (the logged dict is separate)."""


def _declared_defaults():
    return {r[6]: (None if r[5] == "@MISSING" else r[5]) for r in _flag_rows()}


def _draft_block_size(path):
    for name in ("config.json", "dflash_config.json"):
        try:
            cfg = json.load(open(os.path.join(path, name)))
        except Exception:
            continue
        for d in (cfg, cfg.get("dflash_config") or {}, cfg.get("speculative_config") or {}):
            if isinstance(d, dict) and isinstance(d.get("block_size"), int):
                return d["block_size"]
    return None


def _model_cfg(path):
    try:
        cfg = json.load(open(os.path.join(path, "config.json")))
    except Exception:
        return {}
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    return {**cfg, **tc}


def default_graph_bs(max_bs):
    bs = [1, 2, 3, 4, 5, 6, 7, 8] + list(range(10, 33, 2)) + list(range(40, 161, 8)) + list(range(176, 513, 16))
    out = [b for b in bs if b <= max_bs]
    return out or [1]


def resolve_args(ns, argv, quiet=False):
    """-> (server_args dict exactly as 0.5.21 would log it, Cfg, [warning lines]). Raises like resolve_once()."""
    p = build_parser()
    given = given_dests(p, argv)
    dflt = _declared_defaults()
    d = ast.literal_eval(SERVER_ARGS_TEMPLATE)
    warns = []
    for k, v in vars(ns).items():
        if k in given:
            d[k] = v
        elif k in EMS_SET and k in d:
            d[k] = dflt.get(k)
    mc = _model_cfg(d["model_path"])
    if "tokenizer_path" not in given or d.get("tokenizer_path") is None:
        d["tokenizer_path"] = d["model_path"]
    if d.get("served_model_name") is None:
        d["served_model_name"] = d["model_path"]
    if d.get("random_seed") is None:
        d["random_seed"] = random.SystemRandom().randint(0, 1 << 30)
    if d.get("context_length") is None:
        d["context_length"] = int(mc.get("max_position_embeddings") or MODEL_CTX)
    if d.get("page_size") is None:
        d["page_size"] = 1
    if d.get("prefill_decode_interval") is None:
        d["prefill_decode_interval"] = 0
    if d.get("mem_fraction_static") is None:
        d["mem_fraction_static"] = 0.85
    if d.get("chunked_prefill_size") is None:
        d["chunked_prefill_size"] = 2048 if TOTAL_GIB < 25 else 8192
    if d.get("mamba_ssm_dtype") is None:
        d["mamba_ssm_dtype"] = mc.get("mamba_ssm_dtype") or "float32"
    if d.get("attention_backend") is None:
        d["attention_backend"] = "flashinfer"
    algo = d.get("speculative_algorithm")
    algo = algo.upper() if isinstance(algo, str) else None
    d["speculative_algorithm"] = algo
    q = d.get("speculative_draft_model_quantization")
    if algo is None:
        for k in SPEC_FIELDS:
            d[k] = None
        if "speculative_num_draft_tokens" not in given:
            d["speculative_num_draft_tokens"] = None
        d["_speculative_draft_quantization_explicitly_set"] = False
        d["speculative_boundary_reduction"] = "rs+rsv"
    else:
        if algo == "DFLASH":
            if d.get("speculative_draft_model_path") is None:
                raise ValueError("DFLASH speculative decoding requires setting --speculative-draft-model-path.")
            if d.get("speculative_num_draft_tokens") is None:
                bsz = _draft_block_size(d["speculative_draft_model_path"])
                if bsz is None:
                    bsz = 16
                    warns.append("Failed to infer DFLASH block_size from draft model config; defaulting "
                                 "speculative_num_draft_tokens to 16.")
                d["speculative_num_draft_tokens"] = bsz
            d["speculative_num_steps"] = 1
            d["speculative_eagle_topk"] = 1
            d["speculative_draft_attention_backend"] = d["attention_backend"]
            d["speculative_boundary_reduction"] = "ar"
            if d.get("speculative_draft_model_revision") is None:
                d["speculative_draft_model_revision"] = "main"
            if d.get("max_running_requests") is None:
                d["max_running_requests"] = 48
                warns.append("Max running requests is reset to 48 for speculative decoding. You can override this "
                             "by explicitly setting --max-running-requests.")
        elif algo == "NGRAM":
            if d.get("speculative_num_draft_tokens") is None:
                d["speculative_num_draft_tokens"] = 12
            topk = d.get("speculative_ngram_max_bfs_breadth") or 1
            d["speculative_eagle_topk"] = topk
            d["speculative_num_steps"] = max(1, d["speculative_num_draft_tokens"] // topk)
            d["enable_mixed_chunk"] = False
        else:
            if d.get("speculative_num_draft_tokens") is None:
                d["speculative_num_draft_tokens"] = (d.get("speculative_num_steps") or 3) + 1
            if d.get("speculative_draft_model_revision") is None and d.get("speculative_draft_model_path"):
                d["speculative_draft_model_revision"] = "main"
        d["_speculative_draft_quantization_explicitly_set"] = "speculative_draft_model_quantization" in given
        if q == "unquant":
            d["speculative_draft_model_quantization"] = None
    # cuda graph config (decode bs list, backends)
    dec_disabled = bool(d.get("disable_decode_cuda_graph") or d.get("disable_cuda_graph")
                        or d.get("cuda_graph_backend_decode") == "disabled")
    pre_disabled = bool(d.get("disable_prefill_cuda_graph") or d.get("disable_cuda_graph")
                        or d.get("cuda_graph_backend_prefill") == "disabled" or "disable_prefill_cuda_graph" not in given)
    mrr_req = d.get("max_running_requests")
    if d.get("cuda_graph_bs_decode"):
        bs = sorted(set(int(b) for b in d["cuda_graph_bs_decode"]))
    else:
        mx = d.get("cuda_graph_max_bs_decode") or min(mrr_req or 48, 160)
        bs = default_graph_bs(mx)
    cgc = ast.literal_eval(repr(d["cuda_graph_config"]))
    cgc["decode"].update(backend="disabled" if dec_disabled else "full", max_bs=max(bs), bs=bs)
    cgc["prefill"]["backend"] = "disabled" if pre_disabled else "full"
    d["cuda_graph_config"] = cgc
    # validations that resolve_once() runs (mamba_hook / speculative_hook / kv_cache_configurator)
    strat = d.get("mamba_radix_cache_strategy") or "auto"
    if strat == "auto":
        strat_eff = "extra_buffer_lazy" if not d.get("disable_overlap_schedule") else "extra_buffer"
    else:
        strat_eff = strat
    if strat_eff == "no_buffer" and not d.get("disable_overlap_schedule"):
        raise AssertionError("no_buffer requires --disable-overlap-schedule.")
    if strat_eff == "extra_buffer_lazy" and d.get("disable_overlap_schedule"):
        raise AssertionError("Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)")
    W = int(d.get("speculative_num_draft_tokens") or 1) if algo else 1
    if algo and strat_eff in ("extra_buffer", "extra_buffer_lazy") and (d.get("mamba_track_interval") or 256) < W:
        raise AssertionError("mamba_track_interval (%s) must be >= speculative_num_draft_tokens (%s)"
                             % (d.get("mamba_track_interval"), W))
    if d.get("mem_fraction_static") is not None and not (0 < float(d["mem_fraction_static"]) < 1):
        raise ValueError("mem_fraction_static must be in (0, 1), got %s" % d["mem_fraction_static"])
    c = Cfg()
    c.d = d
    c.argv = list(argv)
    c.given = given
    c.algo = algo
    c.spec = algo is not None
    c.dflash = algo == "DFLASH"
    c.W = W if algo else 1
    c.tp = int(d.get("tp_size") or 1)
    c.port = int(d.get("port") or 30000)
    c.host = d.get("host") or "127.0.0.1"
    c.model = d["model_path"]
    c.served = d["served_model_name"]
    c.ctx_len = int(d["context_length"])
    c.chunk = int(d["chunked_prefill_size"])
    c.max_prefill = int(d.get("max_prefill_tokens") or 16384)
    c.policy = d.get("schedule_policy") or "fcfs"
    c.pdi = int(d.get("prefill_decode_interval") or 0)
    c.log_interval = max(1, int(d.get("decode_log_interval") or 40))
    c.metrics = bool(d.get("enable_metrics"))
    c.incremental = bool(d.get("incremental_streaming_output"))
    c.stream_interval = max(1, int(d.get("stream_interval") or 1))
    c.cache_report = bool(d.get("enable_cache_report"))
    c.radix = not d.get("disable_radix_cache")
    c.strategy = strat_eff
    c.ssm = d.get("mamba_ssm_dtype") or "float32"
    c.kv_dtype = d.get("kv_cache_dtype") or "auto"
    c.graph_bs = bs
    c.dec_graph = not dec_disabled
    c.max_graph_bs = max(bs)
    c.mem = float(d["mem_fraction_static"])
    c.watchdog = float(d.get("watchdog_timeout") or 300)
    c.log_level = (d.get("log_level") or "info").lower()
    c.reasoning_parser = d.get("reasoning_parser")
    c.tool_parser = d.get("tool_call_parser")
    c.skip_warmup = bool(d.get("skip_server_warmup"))
    skip_lock = C.env_bool("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK")
    if not c.radix:
        ratio = 1
    else:
        ratio = 3 - (1 if skip_lock else 0)
        if strat_eff == "extra_buffer_lazy":
            ratio += 1
        elif strat_eff == "extra_buffer":
            ratio += 1 if d.get("disable_overlap_schedule") else 2
        elif skip_lock:
            ratio += 1
    c.ratio = ratio
    mrr = mrr_req if mrr_req is not None else 2048
    K = d.get("max_mamba_cache_size")
    if K is None:
        K = ratio * min(mrr, 256)
    c.mamba_size = int(K)
    cap = c.mamba_size // ratio
    c.capped_line = None
    if cap < mrr:
        c.capped_line = ("max_running_requests is capped to %d by the mamba state cache (max_mamba_cache_size=%d, %d "
                         "state slots per request). To raise it: increase --mamba-full-memory-ratio or "
                         "--max-mamba-cache-size, or halve the state size with --mamba-ssm-dtype bfloat16."
                         % (cap, c.mamba_size, ratio))
    if cap <= 0:
        raise RuntimeError("Hybrid (mamba/linear-attention) state cache is too small to serve any requests. "
                           "max_mamba_cache_size=%d, mamba_ratio=%d, resulting max_num_reqs=%d. Try: (1) reduce "
                           "--max-running-requests, (2) increase --mem-fraction-static, or (3) use GPUs with more memory."
                           % (c.mamba_size, ratio, cap))
    c.mrr = min(mrr, cap)
    # memory model (calibrated on the EMS logs: W6 S20 0.83 -> 779,192 tokens; target-only S16 -> 1,576,170)
    fp32 = 2.0 if c.ssm == "float32" else 1.0
    c.ssm_gb = c.mamba_size * (1.49 / 84) * fp32
    c.conv_gb = c.mamba_size * (0.06 / 84)
    c.inter_gb = ((0.01758 * fp32) * (c.mrr + 1) * c.W) if c.spec else 0.0
    c.inter_conv_gb = (0.0003 * (c.mrr + 1) * c.W) if c.spec else 0.0
    kv_bytes_tok = 8192 * (2 if c.kv_dtype in ("auto", "bf16", "bfloat16") else 1)
    c.draft_bytes_tok = (2560 * (2 if c.kv_dtype in ("auto", "bf16", "bfloat16") else 1)) if c.dflash else 0
    c.target_bytes_tok = kv_bytes_tok
    budget = (c.mem * TOTAL_GIB - 4.62 - (1.49 if c.dflash else 0.0) - c.ssm_gb - c.conv_gb - c.inter_gb
              - c.inter_conv_gb - 0.10 - (0.37 if c.spec else 0.30))
    if budget <= 0.05:
        raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static. "
                           "Current value: mem_fraction_static=%s" % c.mem)
    pool = int(budget * GIB / (kv_bytes_tok + c.draft_bytes_tok))
    c.pool_profiled = pool
    if d.get("max_total_tokens"):
        if d["max_total_tokens"] > pool:
            warns.append("max_total_tokens=%d is larger than the profiled value %d. Use the profiled value instead."
                         % (d["max_total_tokens"], pool))
        pool = min(pool, int(d["max_total_tokens"]))
    c.pool = pool
    if c.spec:
        g_tv = 1.21 * c.max_graph_bs / 21 * c.W / 6 if c.dec_graph else 0.0
        g_dr = 0.63 * c.max_graph_bs / 21 if (c.dec_graph and c.dflash) else 0.0
        c.graph_gb = {"prefill": 0.0, "decode": 0.0, "target_verify": round(g_tv, 3), "draft_prefill": 0.0,
                      "draft_decode": round(g_dr, 3), "draft_extend": 0.0}
    else:
        c.graph_gb = {"prefill": 0.0, "decode": round(0.66 * c.max_graph_bs / 17, 3) if c.dec_graph else 0.0,
                      "target_verify": 0.0, "draft_prefill": 0.0, "draft_decode": 0.0, "draft_extend": 0.0}
    c.avail_after_pool = TOTAL_GIB * (1 - c.mem)
    c.avail_final = max(0.05, c.avail_after_pool - sum(c.graph_gb.values()) - (1.0 if c.spec else 0.5))
    c.min_free_slots = None
    mfs = d.get("min_free_slots_delay")
    if mfs is not None:
        t = min(int(mfs), c.mrr)
        c.min_free_slots = t if t > 1 else None
    elif c.dflash and c.mrr >= 8:
        c.min_free_slots = min(4, max(2, (c.mrr + 5) // 6))
    return d, c, warns


# ================================================================================================ logging
class Log:
    LOCK = threading.Lock()

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.ms = C.env_bool("SGLANG_LOG_MS")

    def stamp(self):
        t = time.time()
        s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
        if self.ms:
            s += ".%03d" % int((t % 1) * 1000)
        return s

    def __call__(self, msg, prefix=None):
        p = self.prefix if prefix is None else prefix
        line = "[%s%s] %s" % (self.stamp(), p, msg)
        with Log.LOCK:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except Exception:
                pass

    @staticmethod
    def raw(text):
        with Log.LOCK:
            try:
                sys.stdout.write(text if text.endswith("\n") else text + "\n")
                sys.stdout.flush()
            except Exception:
                pass


def tp_prefix(cfg_tp, rank):
    return " TP%d" % rank if cfg_tp > 1 else ""


# ================================================================================================ NCCL INFO lines
NCCL_ENV_READ_SKIP = ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE")
NCCL_TUNING_VARS = ("NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_ALGO",
                    "NCCL_PROTO", "NCCL_NTHREADS")


def nccl_effective(env, ignored=False):
    """(channels, algo, proto, nthreads_block) NCCL would use for the 983 KB TP all-reduce on 4x L4 PCIe."""
    ch, algo, proto, block = 2, "RING", "LL", 288
    if ignored:
        return ch, algo, proto, block
    mn = env.get("NCCL_MIN_CTAS") or env.get("NCCL_MIN_NCHANNELS")
    mx = env.get("NCCL_MAX_CTAS") or env.get("NCCL_MAX_NCHANNELS")
    try:
        if mn:
            ch = max(ch, int(mn))
        if mx:
            ch = min(ch, int(mx)) if not mn else max(int(mn), min(ch, int(mx)))
    except ValueError:
        pass
    a = (env.get("NCCL_ALGO") or "").lower()
    for part in a.replace(";", ",").split(","):
        part = part.strip()
        if ":" in part:
            coll, val = part.split(":", 1)
            if coll.strip() not in ("allreduce", "all"):
                continue
            part = val
        if part in ("ring", "tree"):
            algo = part.upper()
    pr = (env.get("NCCL_PROTO") or "").lower()
    if pr in ("simple", "ll", "ll128"):
        proto = {"simple": "SIMPLE", "ll": "LL", "ll128": "LL128"}[pr]
    nt = env.get("NCCL_NTHREADS")
    if proto == "SIMPLE":
        block = (int(nt) if nt and nt.isdigit() else 512) + 32
    elif proto == "LL128":
        block = 640
    elif nt and nt.isdigit():
        block = min(int(nt), 512) + 32
    return ch, algo, proto, block


def nccl_lines(rank, tp, env, pid, ignored=False):
    """NCCL 2.29.7 INFO lines of one rank's communicator init (format of the E14A bench stdout)."""
    if (env.get("NCCL_DEBUG") or "").upper() not in ("INFO", "TRACE"):
        return []
    subs = set(x.strip().upper() for x in (env.get("NCCL_DEBUG_SUBSYS") or "INIT,BOOTSTRAP,ENV").split(","))
    on = lambda s: "ALL" in subs or s in subs
    host = socket.gethostname().split(".")[0][:12] or "localhost"
    pre = "%s:%d:%d [%d] NCCL INFO " % (host, pid, pid, rank)
    ch, algo, proto, _b = nccl_effective(env, ignored)
    comm = "0x%x" % (0x475c0040 + rank * 0x1000)
    cid = "0x%016x" % (C.mix64(pid) & ((1 << 64) - 1))
    out = []
    if on("INIT"):
        out += ["ENV/Plugin: Could not find: libnccl-env.so", "cudaDriverVersion 13000",
                "NCCL version %s+cuda13.2" % NCCL_VERSION, "Comm config Blocking set to 1"]
    if on("BOOTSTRAP"):
        out += ["Bootstrap: Using eth0:172.19.2.2<0>"]
    if on("INIT"):
        out += ["[Rank %d] ncclCommInitRankConfig comm %s rank %d nranks %d cudaDev %d nvmlDev %d busId %d0 commId %s - "
                "Init START" % (rank, comm, rank, tp, rank, rank, 3 + rank, cid)]
    if on("ENV"):
        for k in sorted(env):
            if not k.startswith("NCCL_") or k in NCCL_ENV_READ_SKIP:
                continue
            if ignored and k in NCCL_TUNING_VARS:
                continue
            out.append("%s set by environment to %s." % (k, env[k]))
    if on("GRAPH"):
        out += ["Pattern 4, crossNic 0, nChannels %d, bw 10.000000/10.000000, type PHB/PIX, sameChannels 1" % max(1, ch // 2),
                " 0 : " + " ".join("GPU/0-%d0" % (3 + r) for r in range(tp))]
    if on("INIT"):
        out.append("comm %s rank %d nRanks %d nNodes 1 localRanks %d localRank %d MNNVL 0" % (comm, rank, tp, tp, rank))
        for c_ in range(ch):
            out.append("Channel %02d/%02d : %s" % (c_, ch, " ".join(str(r) for r in range(tp))))
        nxt = (rank + 1) % tp
        out.append("Trees " + " ".join("[%d] %d/-1/-1->%d->%d" % (c_, nxt, rank, (rank - 1) if rank else -1)
                                       for c_ in range(ch)))
        out.append("P2P Chunksize set to 131072")
        for c_ in range(ch):
            out.append("Channel %02d : %d[%d] -> %d[%d] via SHM/direct/direct" % (c_, rank, rank, nxt, nxt))
        out += ["Connected all rings, use ring PXN 0 GDR 1", "Connected all trees",
                "threadThresholds 8/8/64 | %d/8/64 | 512 | 512" % (tp * 8),
                "%d coll channels, %d collnet channels, 0 nvls channels, %d p2p channels, %d p2p channels per peer"
                % (ch, ch, ch, ch), "CC Off, workFifoBytes 1048576",
                "ncclCommInitRankConfig comm %s rank %d nranks %d cudaDev %d nvmlDev %d busId %d0 commId %s - Init COMPLETE"
                % (comm, rank, tp, rank, rank, 3 + rank, cid),
                "Init timings - ncclCommInitRankConfig: rank %d nranks %d total 0.38 (kernels 0.24, alloc 0.04, "
                "bootstrap 0.02, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.06, rest 0.00)" % (rank, tp)]
    # NCCL_ALGO / NCCL_PROTO show up only as "set by environment" lines and in the trace kernel names (no invented
    # tuner line: the E14A logs have none)
    return [pre + m for m in out]


# ================================================================================================ child processes
def _install_stub_modules():
    """Stubs of the hook's SGLang targets in sys.modules (the real ones live in the scheduler process), so a hook that
    wraps them finds every class / method and its self-test can pass; the fake calls Scheduler(...) and then
    run_event_loop() in the main thread like the real one. The target table comes from the loaded hook (sitecustomize
    TARGETS / TARGET_METHODS) when there is one, else Scheduler only. Stub methods never run: hook counters stay 0."""
    import types
    sc = sys.modules.get("sitecustomize")
    methods = dict(getattr(sc, "TARGET_METHODS", None) or {})
    targets = {k: v for k, v in dict(getattr(sc, "TARGETS", None) or {}).items() if str(v[0]).startswith("sglang.")}
    targets.setdefault("Scheduler", ("sglang.srt.managers.scheduler", "Scheduler"))
    mods = {}

    def module(name):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            n = ".".join(parts[:i])
            if n not in mods:
                m = types.ModuleType(n)
                m.__file__ = "<fake-sglang:%s>" % n
                m.__fake__ = True
                mods[n] = m
                if i > 1:
                    setattr(mods[".".join(parts[:i - 1])], parts[i - 1], m)
        return mods[name]

    module("sglang").__version__ = VERSION

    class Scheduler:
        def __init__(self, server_args=None, port_args=None, init=None):
            if init is not None:
                init()

        def run_event_loop(self, loop=None):
            if loop is not None:
                loop()

        def on_idle(self):
            return None

    def _noop(self, *a, **k):
        return None

    def _noop_fn(*a, **k):
        return None

    for name, (mod_name, attr) in targets.items():
        if name == "Scheduler":
            obj = Scheduler
        elif attr[:1].isupper():
            obj = type(attr, (), {meth: _noop for meth in methods.get(name, ())})
        else:                                               # a module-level function (_grouped_conv)
            obj = types.FunctionType(_noop_fn.__code__, {}, attr)
        obj.__module__ = mod_name
        setattr(module(mod_name), attr, obj)
    if "sglang.srt.managers.schedule_batch" in mods:        # read by the hook's ScheduleBatch installer
        mods["sglang.srt.managers.schedule_batch"].mamba_lazy_spec_in_window = lambda *a, **k: False
    for name, m in mods.items():
        sys.modules.setdefault(name, m)
    return sys.modules["sglang.srt.managers.scheduler"].Scheduler


def _cache_targets(env, rank, dflash, leak):
    """(root, relative files) the rank's JIT/compile caches would write on a cold start."""
    home = os.path.expanduser("~")
    sg = env.get("SGLANG_CACHE_DIR") or os.path.join(home, ".cache", "sglang")
    tri = env.get("TRITON_CACHE_DIR") or os.path.join(sg, "triton")
    ind = env.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(sg, "inductor")
    cuda = env.get("CUDA_CACHE_PATH") or os.path.join(sg, "nv")
    jit = env.get("SGLANG_JIT_CACHE_DIR") or os.path.join(home, ".cache", "sglang", "jit")
    fi = os.path.join(env.get("FLASHINFER_WORKSPACE_BASE") or sg, ".cache", "flashinfer")
    kern = ["_fused_norm_rope_kernel_stacked", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "_causal_conv1d_fwd_kernel",
            "fused_sigmoid_gating_delta_rule_update_kernel", "_layer_norm_fwd_1pass_kernel", "assign_req_to_token_pool"]
    t = [(tri, ["%s/%s.cubin" % (C.str_key(k + str(rank)) % 10 ** 12, k) for k in kern]
          + ["%s/%s.json" % (C.str_key(k + str(rank)) % 10 ** 12, k) for k in kern])]
    if dflash:
        t.append((ind, ["fxgraph/%02x/f%012x" % (rank, C.str_key("grouped_conv%d" % rank) % 16 ** 12),
                        "triton/%d/c%012x.py" % (rank, C.str_key("score_edges%d" % rank) % 16 ** 12)]))
    t.append((cuda, ["ComputeCache/%d/%012x" % (rank, C.str_key("ptx%d" % rank) % 16 ** 12), "ComputeCache/index"]))
    t.append((jit, ["tvm_ffi/sgl_kernel_jit_%d.so" % rank]))
    t.append((fi, ["89/cached_ops/norm_rank%d/norm.so" % rank]))
    if leak:
        t.append((os.path.join(home, ".cache", "sglang", "triton"), ["leak%012x/%s.cubin" % (C.str_key("leak%d" % rank) % 16 ** 12, kern[0])]))
    return t


def scheduler_child(rank, cfg, conn):
    """One TP rank (spawned): init lines with the TP prefix, NCCL lines, cache writes, then idle until told or orphaned."""
    tp = cfg["tp"]
    title = "sglang::scheduler" + ("_TP%d" % rank if tp > 1 else "")
    C.set_proctitle(title)                                  # scheduler.py:6014
    C.watch_parent(lambda: None)
    log = Log(tp_prefix(tp, rank))
    t0 = time.time()
    T = cfg["startup_sec"]
    at = lambda frac: max(0.0, cfg["t_launch"] + frac * T - time.time())

    def init():
        log("Init parallel begin.")
        time.sleep(at(0.05))
        for line in nccl_lines(rank, tp, dict(os.environ), os.getpid(), cfg["nccl_ignored"]):
            Log.raw(line)
        if rank == 0:
            log("sglang is using nccl==%s" % NCCL_VERSION)
        if tp > 2:
            log("CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs. To silence "
                "this warning, specify disable_custom_all_reduce=True explicitly.")
            log("CustomAllreduce is disabled because it's not supported on more than two PCIe-only GPUs. To silence "
                "this warning, specify disable_custom_all_reduce=True explicitly.")
        log("Init parallel ends. elapsed=%.2f s" % (time.time() - t0))
        log("Memory baseline taken. mem usage=0.00 GB")
        log("Load weight begin. avail mem=21.72 GB")
        if cfg["fault"] == "crash_start" and rank == 0:
            time.sleep(at(0.12))
            raise RuntimeError("CUDA error: an illegal memory access was encountered (fake FAKE_SG_FAULT=crash_start)")
        time.sleep(at(0.15))
        if cfg["kv_fp8"]:
            log("Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.")
        log("Load weight end. elapsed=%.2f s, type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors, avail "
            "mem=17.10 GB, mem usage=4.62 GB." % (time.time() - t0))
        if cfg["dflash"]:
            log("Load weight end. elapsed=0.64 s, type=DFlash2DraftModel, avail mem=15.58 GB, mem usage=1.49 GB.")
            if cfg["block_mismatch"]:
                log(cfg["block_mismatch"])
            if rank == 0:
                log("Initialized DFLASH draft runner. attention_backend=flashinfer, model=DFlash2DraftModel, "
                    "block_size=%d, draft_window_size=None, compact_cache=False" % cfg["W"])
                log("DFLASH draft runner ready. mask_token=<|MASK|>, mask_token_id=248070, mask_token_id_override=248070, "
                    "noise_embed_scale=1.0")
        log("Reserving 0.10 GB of the KV budget for post-sizing multimodal allocations (feature-transport pools + "
            "embedding cache).")
        inter = (" intermediate_ssm_state_cache size: %.2fGB intermediate_conv_window_cache size: %.2fGB "
                 % (cfg["inter_gb"], cfg["inter_conv_gb"])) if cfg["spec"] else " "
        log("Mamba Cache is allocated. max_mamba_cache_size: %d, conv_state size: %.2fGB, ssm_state size: %.2fGB%s"
            % (cfg["mamba_size"], cfg["conv_gb"], cfg["ssm_gb"], inter.rstrip(" ") + " "))
        kvd = "torch.float8_e4m3fn" if cfg["kv_fp8"] else "torch.bfloat16"
        log("KV Cache is allocated. dtype: %s, #tokens: %d, K size: %.2f GB, V size: %.2f GB"
            % (kvd, cfg["pool"], cfg["pool"] * cfg["target_bytes_tok"] / 2 / GIB, cfg["pool"] * cfg["target_bytes_tok"] / 2 / GIB))
        log("Memory pool end. avail mem=%.2f GB" % cfg["avail_after_pool"])
        if cfg["dflash"]:
            log("KV Cache is allocated. dtype: %s, #tokens: %d, K size: %.2f GB, V size: %.2f GB"
                % (kvd, cfg["pool"], cfg["pool"] * cfg["draft_bytes_tok"] / 2 / GIB, cfg["pool"] * cfg["draft_bytes_tok"] / 2 / GIB))
        if rank == 0:
            log("Linear attention kernel backend: decode=triton, prefill=triton, verify=triton")
        log("Using hybrid linear attention backend for hybrid GDN models.")
        if rank == 0:
            log("GDN kernel dispatcher: decode=TritonGDNKernel, extend=TritonGDNKernel, verify=TritonGDNKernel "
                "packed_decode=True")
        log("Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled' (e.g. via "
            "--cuda-graph-backend-prefill=disabled or auto-disable rules).")
        bs = cfg["graph_bs"]
        tc = time.time()
        if cfg["dec_graph"]:
            what = "target verify" if cfg["spec"] else "target decode"
            log("Capture %s CUDA graph begin. backend=full, num_tokens_per_req=%d, bs=%s, avail mem=%.2f GB"
                % (what, cfg["W"], bs, cfg["avail_after_pool"] - 0.5))
            time.sleep(at(0.80 if cfg["spec"] else 0.86))
            files = 0
            for root, rels in cfg["cache_targets"]:
                files += C.write_cache_files(root, rels, rank=rank) if cfg["cold"] or cfg["leak"] else 0
            g = cfg["graph_gb"]["target_verify" if cfg["spec"] else "decode"]
            log("Capture %s CUDA graph end. elapsed=%.2f s, mem usage=%.2f GB, avail mem=%.2f GB."
                % (what, time.time() - tc, g, cfg["avail_after_pool"] - 0.5 - g))
            if cfg["dflash"]:
                if rank == 0:
                    log("DFLASH selector decode folded into the draft cuda graph (sampling_enabled=True).")
                log("Capture draft verify CUDA graph begin. backend=full, num_tokens_per_req=%d, bs=%s, avail mem=%.2f GB"
                    % (cfg["W"], bs, cfg["avail_after_pool"] - 0.5 - g))
                time.sleep(at(0.87))
                log("Capture draft verify CUDA graph end. elapsed=%.2f s, mem usage=%.2f GB, avail mem=%.2f GB."
                    % (time.time() - tc, cfg["graph_gb"]["draft_decode"], cfg["avail_final"]))
        else:
            time.sleep(at(0.30))
            for root, rels in cfg["cache_targets"]:
                C.write_cache_files(root, rels, rank=rank) if cfg["cold"] or cfg["leak"] else 0
        if rank == 0:
            if cfg["capped_line"]:
                log(cfg["capped_line"])
            log("max_total_num_tokens=%d, chunked_prefill_size=%d, max_prefill_tokens=%d, max_running_requests=%d, "
                "context_len=%d, available_gpu_mem=%.2f GB" % (cfg["pool"], cfg["chunk"], cfg["max_prefill"], cfg["mrr"],
                                                               cfg["ctx_len"], cfg["avail_final"]))
            if cfg["radix"]:
                log("Init Unified Radix Cache. Components: (<ComponentType.FULL: 0>, <ComponentType.MAMBA: 2>). Tree "
                    "Core: RustUnifiedTreeCore")
                log("Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=True "
                    "hicache_attached=False streaming_wrapped=False")
        time.sleep(at(0.90))

    sched = None
    try:
        if cfg["stub_modules"]:
            Sched = _install_stub_modules()
            time.sleep(min(0.3, max(0.05, 0.02 * T)))        # let a hook thread see the module before __init__ runs
            sched = Sched(None, None, init=init)
        else:
            init()
        conn.send({"ok": True, "pid": os.getpid(), "rank": rank, "t": time.time() - t0})
    except Exception:
        tb = traceback.format_exc()
        log("Scheduler hit an exception: %s" % tb.rstrip())
        try:
            conn.send({"ok": False, "pid": os.getpid(), "rank": rank, "error": tb[-400:]})
        except Exception:
            pass
        try:
            os.kill(os.getppid(), signal.SIGQUIT)
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(1)
    if sched is not None:
        sched.run_event_loop(lambda: _child_idle(conn))     # scheduler.py:6082 (the hook's late point)
    else:
        _child_idle(conn)


def detokenizer_child(cfg, conn):
    C.set_proctitle("sglang::detokenizer")                  # detokenizer_manager.py:544
    C.watch_parent(lambda: None)
    try:
        conn.send({"ok": True, "pid": os.getpid(), "rank": -1})
    except Exception:
        pass
    _child_idle(conn)


def _child_idle(conn):
    signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
    while True:
        try:
            if conn.poll(0.5):
                msg = conn.recv()
                if msg == "quit":
                    os._exit(0)
                if isinstance(msg, dict) and msg.get("log"):
                    Log(msg.get("prefix", ""))(msg["log"])
        except (EOFError, OSError):
            os._exit(0)


# ================================================================================================ radix + KV + Mamba
class Node:
    __slots__ = ("key", "parent", "depth", "children", "lock", "last", "state")

    def __init__(self, key, parent, depth):
        self.key, self.parent, self.depth = key, parent, depth
        self.children, self.lock, self.last, self.state = 0, 0, time.monotonic(), False


class Tree:
    """64-token block radix tree with Mamba checkpoints (has a state = a match can stop there)."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.nodes = {}
        self.locked = 0              # tokens in nodes with lock > 0
        self.states = 0              # nodes holding a Mamba checkpoint
        self.locked_states = 0
        self.evicted_total = 0

    @staticmethod
    def block_keys(ids, upto=None):
        n = (len(ids) if upto is None else upto) // BLOCK
        keys, prev = [], 0
        for i in range(n):
            prev = hash((prev, tuple(ids[i * BLOCK:(i + 1) * BLOCK])))
            keys.append(prev)
        return keys

    @property
    def tokens(self):
        return BLOCK * len(self.nodes)

    @property
    def evictable(self):
        return self.tokens - self.locked

    def match(self, keys, cap):
        """Deepest checkpoint (state) node on the common block path, at depth <= cap -> (tokens, [keys])."""
        if not self.enabled:
            return 0, []
        best, path = 0, []
        for i, k in enumerate(keys):
            n = self.nodes.get(k)
            if n is None or n.depth > cap:
                break
            path.append(k)
            if n.state:
                best = i + 1
        return best * BLOCK, path[:best]

    def common(self, keys):
        n = 0
        for k in keys:
            if k not in self.nodes:
                break
            n += 1
        return n * BLOCK

    def lock_keys(self, keys):
        for k in keys:
            n = self.nodes.get(k)
            if n is None:
                continue
            if n.lock == 0:
                self.locked += BLOCK
                if n.state:
                    self.locked_states += 1
            n.lock += 1
            n.last = time.monotonic()

    def unlock_keys(self, keys):
        now = time.monotonic()
        for k in keys:
            n = self.nodes.get(k)
            if n is None or n.lock == 0:
                continue
            n.lock -= 1
            n.last = now
            if n.lock == 0:
                self.locked -= BLOCK
                if n.state:
                    self.locked_states -= 1

    def insert(self, keys, locked_by_caller=False):
        """Add missing blocks along keys; returns the keys that were newly created (caller decides locking)."""
        if not self.enabled:
            return []
        new, parent, now = [], None, time.monotonic()
        for i, k in enumerate(keys):
            n = self.nodes.get(k)
            if n is None:
                n = Node(k, parent, (i + 1) * BLOCK)
                self.nodes[k] = n
                if parent is not None and parent in self.nodes:
                    self.nodes[parent].children += 1
                new.append(k)
            n.last = now
            parent = k
        return new

    def set_state(self, key):
        n = self.nodes.get(key)
        if n is None or n.state:
            return False
        n.state = True
        self.states += 1
        if n.lock:
            self.locked_states += 1
        return True

    def drop_state_lru(self):
        cands = [n for n in self.nodes.values() if n.state and n.lock == 0]
        if not cands:
            return False
        n = min(cands, key=lambda x: x.last)
        n.state = False
        self.states -= 1
        return True

    def evict(self, need):
        """Free >= need tokens from unlocked leaves (LRU); returns tokens freed."""
        freed = 0
        while freed < need:
            leaves = [n for n in self.nodes.values() if n.lock == 0 and n.children == 0]
            if not leaves:
                break
            leaves.sort(key=lambda x: x.last)
            for n in leaves:
                if freed >= need:
                    break
                del self.nodes[n.key]
                if n.state:
                    self.states -= 1
                if n.parent is not None and n.parent in self.nodes:
                    self.nodes[n.parent].children -= 1
                freed += BLOCK
        self.evicted_total += freed
        return freed

    def reset(self):
        self.nodes.clear()
        self.locked = self.states = self.locked_states = 0


# ================================================================================================ requests
class Req:
    __slots__ = ("rid", "ids", "prompt_len", "max_new", "temp", "top_p", "top_k", "ignore_eos", "stop_ids", "seed",
                 "rng", "ret_lp", "top_n", "lp_start", "tid_lp", "text_in_lp", "stream", "kind", "out", "out_lp",
                 "out_top", "out_dist", "cached", "prefilled", "state", "t_arrive", "t_first", "t_finish", "t_prefill_end",
                 "t_forward_entry", "finish", "n_retract", "verify_ct", "correct_drafts", "events", "keys", "locked",
                 "own_blocks", "abort_msg", "log_metrics", "forced", "is_streaming_metric", "mamba_held", "branch_ckpt",
                 "done_evt", "skip_special", "admitted", "last_chunk_t", "chunk_times")

    def __init__(self, rid, ids, sp, *, ret_lp=False, top_n=0, lp_start=-1, tid_lp=None, text_in_lp=False,
                 stream=False, kind="generate", log_metrics=True, forced=None):
        self.rid = rid
        self.ids = list(ids)
        self.prompt_len = len(self.ids)
        self.max_new = int(sp.get("max_new_tokens", 128) if sp.get("max_new_tokens") is not None else 128)
        t = sp.get("temperature", 1.0)
        t = 1.0 if t is None else float(t)
        self.top_k = sp.get("top_k", 1 << 30)
        self.top_k = (1 << 30) if self.top_k in (None, -1) else int(self.top_k)
        if 0 <= t < 1e-6:                     # sampling_params.py L217-221: greedy -> temperature 1.0, top_k 1
            t, self.top_k = 1.0, 1
        self.temp = t
        self.top_p = float(sp.get("top_p", 1.0) if sp.get("top_p") is not None else 1.0)
        self.ignore_eos = bool(sp.get("ignore_eos", False))
        self.stop_ids = set(int(x) for x in (sp.get("stop_token_ids") or []))
        self.seed = sp.get("sampling_seed")
        self.rng = random.Random(self.seed if self.seed is not None else C.str_key(rid) ^ int(time.time() * 1e6))
        self.skip_special = sp.get("skip_special_tokens", True) is not False
        self.ret_lp, self.top_n, self.lp_start = bool(ret_lp), int(top_n or 0), lp_start
        self.tid_lp = list(tid_lp) if tid_lp is not None else None
        self.text_in_lp, self.stream, self.kind = bool(text_in_lp), bool(stream), kind
        self.out, self.out_lp, self.out_top, self.out_dist = [], [], [], []
        self.cached, self.prefilled, self.state = 0, 0, "waiting"
        self.t_arrive, self.t_first, self.t_finish, self.t_prefill_end, self.t_forward_entry = time.time(), None, None, None, None
        self.finish, self.n_retract, self.verify_ct, self.correct_drafts = None, 0, 0, 0
        self.events = queue.Queue()
        self.keys, self.locked, self.own_blocks = [], [], 0
        self.abort_msg, self.log_metrics, self.forced = None, log_metrics, forced
        self.is_streaming_metric = "true" if stream else "false"
        self.mamba_held, self.branch_ckpt, self.admitted = 0, 0, False
        self.done_evt = threading.Event()
        self.last_chunk_t, self.chunk_times = None, []

    @property
    def fill(self):
        return self.ids + self.out

    def seq_len(self):
        return self.prompt_len + len(self.out)


# ================================================================================================ the engine
class Engine:
    def __init__(self, cfg, d, log):
        self.c, self.d, self.log = cfg, d, log
        self.tp_log = Log(tp_prefix(cfg.tp, 0))
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.waiting, self.running, self.chunked = [], [], None
        self.reqs = {}
        self.tree = Tree(cfg.radix)
        self.status = "starting"
        self.dead = False
        self.t_start = time.time()
        self.lm = C.ToyLM(cfg.weights, engine_key=cfg.engine_key, noise=cfg.noise, path_noise=cfg.path_noise,
                          diverge_at=cfg.diverge_at)
        self.lm_prefill = self.lm
        self.lm_bug = C.ToyLM(cfg.weights, engine_key=cfg.engine_key, noise=cfg.noise, path_noise=cfg.path_noise,
                              diverge_at=cfg.dflash_bug_at) if cfg.dflash_bug else None
        rd = C.env_int("FAKE_SG_RESTORE_DIVERGE_AT", -1)      # a stale restored state: only requests with a cached prefix
        self.lm_restore = C.ToyLM(cfg.weights, engine_key=cfg.engine_key, noise=cfg.noise, path_noise=cfg.path_noise,
                                  diverge_at=rd) if rd >= 0 else None
        self.accept_prob = C.accept_p(cfg.W) if cfg.spec else 0.0
        if cfg.algo == "NGRAM":
            self.accept_prob = C.accept_p(cfg.W, 1.6)
        self.sim_len = C.env_float("SGLANG_SIMULATE_ACC_LEN", -1.0)
        self.sim_mode = C.env_str("SGLANG_SIMULATE_ACC_TOKEN_MODE", "fixed")
        self.new_token_ratio = INIT_RATIO
        self.min_ratio = INIT_RATIO * MIN_RATIO_FACTOR
        self.defer_left = 0
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.step_no = 0
        # scheduler-side counters (TP0 unless noted) and window accumulators
        self.rt = {"prefill_compute": 0, "prefill_cache": 0, "decode": 0}
        self.rt_seen = set()
        self.passes = {"decode_cuda_graph": 0, "decode_none": 0, "prefill_none": 0}
        self.eff = {"input": 0, "device_hit": 0, "host_hit": 0, "storage_hit": 0}
        self.retract = None            # [reqs, in_tokens, out_tokens] after the first retraction
        self.retract_window = 0
        self.win = {"gen": 0, "acc_tokens": 0, "fwd_ct": 0, "correct": 0}
        self.last_decode_log = time.perf_counter()
        self.last_prefill_log = time.perf_counter()
        self.last_gen_tp = 0.0
        self.gauges = None
        self.gauge_t = 0.0
        self.idle_sec = [0.0] * cfg.tp
        self.busy_sec = 0.0
        self.stage = {k: 0.0 for k in ("other", "recv_requests", "process_input_requests", "get_next_batch_to_run",
                                        "process_queue", "run_batch", "process_batch_result", "sanity_check_cache", "idle")}
        # tokenizer-side metrics (finish-only)
        self.tok = {}                  # is_streaming -> dict(prompt, gen, n)
        self.cached_total = 0
        self.spec_verify_total = 0
        self.aborted_total = 0
        H = C.Hist
        self.h = {}                    # is_streaming -> dict of hists
        self.h_common = {"prompt": H(PROMPT_BUCKETS), "uncached": H(PROMPT_BUCKETS), "gen": H(PROMPT_BUCKETS),
                         "itl": H(ITL_BUCKETS)}
        self.h_rank = {"queue": H(QUEUE_BUCKETS), "prefill_forward": H(STAGE_BUCKETS), "request_process": H(STAGE_BUCKETS),
                       "chunked_prefill": H(STAGE_BUCKETS), "evict": H(EVICT_BUCKETS)}
        self.stage_seen = set()
        self.http = {}                 # (endpoint, method) -> [requests, active]
        self.http_resp = {}            # (endpoint, method, code) -> n
        self.prof = None
        self.prof_steps = []
        self.hang = False
        self.progress_t = time.time()
        self.admitted_count = 0
        self.oom_logged = False
        self.startup_time = {}
        self.child_pids = []
        self.tp0_pid = os.getpid()

    # ----------------------------------------------------------------- KV / Mamba accounting
    def own_tokens(self):
        n = 0
        for r in self.running:
            n += r.seq_len() - BLOCK * len(r.locked)
        if self.chunked is not None:
            n += self.chunked.prefilled - BLOCK * len(self.chunked.locked)
        return max(0, n)

    def kv(self):
        used = self.tree.locked + self.own_tokens()
        ev = self.tree.evictable
        avail = self.c.pool - used - ev
        return used, ev, avail

    def mamba(self):
        """(used, evictable, available) Mamba SLOTS: ~3 per running request (EMS logs: 20 running <-> mamba num 55-57,
        1 <-> 3), 1 for a request in chunked prefill; unlocked radix checkpoints are evictable."""
        req_slots = 3 * len(self.running) + (1 if self.chunked is not None else 0)
        used = min(self.c.mamba_size, req_slots)
        ev = min(self.tree.states - self.tree.locked_states, self.c.mamba_size - used)
        avail = max(0, self.c.mamba_size - used - ev)
        return used, ev, avail

    def pool_stats(self):
        used, ev, avail = self.kv()
        mu, mev, mav = self.mamba()
        return {"full_num_used": used, "full_token_usage": used / self.c.pool, "full_available": avail,
                "full_evictable": ev, "mamba_used": mu, "mamba_usage": mu / self.c.mamba_size, "mamba_available": mav,
                "mamba_evictable": mev}

    def make_room(self, need):
        used, ev, avail = self.kv()
        if avail < need and ev > 0:
            t0 = time.perf_counter()
            self.tree.evict(need - avail)
            self.h_rank["evict"].observe(time.perf_counter() - t0)

    def make_mamba_room(self, need):
        while self.mamba()[2] < need and self.tree.drop_state_lru():
            pass
        return self.mamba()[2] >= need

    def add_ckpt(self, key):
        if key is None or not self.c.radix:
            return
        n = self.tree.nodes.get(key)
        if n is None or n.state:
            return
        if self.mamba()[2] < 1 and not self.tree.drop_state_lru():
            return
        self.tree.set_state(key)

    # ----------------------------------------------------------------- timing model
    def step_seconds(self, bs, ctx_tokens):
        fixed = C.env_float("FAKE_SG_STEP_MS", 0.0)
        if fixed > 0:
            return fixed / 1000.0
        if self.c.spec:
            ms = 26.0 + 0.94 * bs * self.c.W + 2.6e-5 * ctx_tokens
        else:
            ms = 25.0 + 0.9 * bs + 1.2e-5 * ctx_tokens
        ms *= self.c.nccl_speed
        scale = max(C.time_scale(), C.env_float("FAKE_SG_MIN_STEP_MS", 3.0) / 26.0)
        return ms * scale / 1000.0

    def prefill_tps(self, ctx):
        base = C.env_float("FAKE_SG_PREFILL_TPS", 0.0)
        if base <= 0:
            base = DEFAULT_PREFILL_TPS / C.time_scale()
        return base / (1.0 + ctx / 250000.0)

    # ----------------------------------------------------------------- request intake
    def submit(self, req):
        with self.lock:
            if self.dead:
                raise RuntimeError("server is dead")
            if req.rid in self.reqs:
                raise ValueError("Duplicate request ID detected: %s" % req.rid)
            self.reqs[req.rid] = req
            req.keys = Tree.block_keys(req.ids) if self.c.radix else []
            self.waiting.append(req)
            self.cv.notify_all()

    def abort(self, rid=None, abort_all=False, msg="Abort in waiting queue", quiet=False):
        with self.lock:
            targets = list(self.reqs.values()) if abort_all else ([self.reqs[rid]] if rid in self.reqs else [])
            for r in targets:
                if r.state == "finished":
                    continue
                if r.state == "waiting":
                    self.waiting = [x for x in self.waiting if x is not r]
                    self._finish(r, {"type": "abort", "message": "Abort in waiting queue"}, release=False)
                else:
                    if r is self.chunked:
                        self.chunked = None
                    self.running = [x for x in self.running if x is not r]
                    self._finish(r, {"type": "abort", "message": msg})
                self.aborted_total += 1
            self.cv.notify_all()

    # ----------------------------------------------------------------- finishing
    def _release(self, r):
        """cache_finished_req: insert prompt+output (minus the last token) blocks, a checkpoint at the end, unlock."""
        if self.c.radix:
            n = (r.seq_len() - 1) if r.out else r.prefilled
            keys = r.keys[: n // BLOCK]
            if len(keys) < n // BLOCK:                    # decode tokens beyond the prompt blocks
                keys = Tree.block_keys(r.ids + r.out, n)
            self.tree.insert(keys)
            if keys:
                self.add_ckpt(keys[-1])
        self.tree.unlock_keys(r.locked)
        r.locked = []

    def _finish(self, r, reason, release=True):
        if r.state == "finished":
            return
        if release and r.admitted:
            self._release(r)
        r.state = "finished"
        r.finish = reason
        r.t_finish = time.time()
        self.reqs.pop(r.rid, None)
        if r.log_metrics:
            s = r.is_streaming_metric
            t = self.tok.setdefault(s, {"prompt": 0, "gen": 0, "n": 0})
            t["prompt"] += r.prompt_len
            t["gen"] += len(r.out)
            t["n"] += 1
            self.cached_total += r.cached
            self.spec_verify_total += r.verify_ct
            h = self.h.setdefault(s, {"ttft": C.Hist(TTFT_BUCKETS), "e2e": C.Hist(E2E_BUCKETS)})
            if r.t_first:
                h["ttft"].observe(r.t_first - r.t_arrive)
            h["e2e"].observe(r.t_finish - r.t_arrive)
            self.h_common["prompt"].observe(r.prompt_len)
            self.h_common["uncached"].observe(max(0, r.prompt_len - r.cached))
            self.h_common["gen"].observe(len(r.out))
        r.events.put(("fin", None))
        r.done_evt.set()

    # ----------------------------------------------------------------- the scheduler loop
    def run(self):
        with self.lock:
            self._publish_gauges()          # _maybe_log_idle_metrics publishes at the first idle tick (last_log_time 0)
        while True:
            with self.lock:
                if self.dead:
                    return
                if self.hang:
                    self.cv.wait(timeout=0.5)
                    if (self.waiting or self.running) and time.time() - self.progress_t > self.c.watchdog * C.time_scale():
                        self._crash("Scheduler watchdog timeout (self.watchdog_timeout=%s, self.soft=False)"
                                    % self.c.watchdog, "watchdog")
                    continue
                plan = self._next_batch()
                if plan is None:
                    self._idle_publish()
                    t0 = time.perf_counter()
                    self.cv.wait(timeout=0.05)
                    dt = time.perf_counter() - t0
                    self.idle_sec = [x + dt for x in self.idle_sec]
                    self.stage["idle"] += dt
                    continue
                kind, items, dur = plan
            t_begin = time.perf_counter()
            wall0 = time.time()
            deadline = t_begin + dur
            while True:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                time.sleep(min(left, 0.05))
            with self.lock:
                if self.dead:
                    return
                self.busy_sec += dur
                self.stage["run_batch"] += dur
                if kind == "prefill":
                    self._finish_prefill(items)
                else:
                    self._finish_decode(items)
                self.forward_ct += 1
                self.progress_t = time.time()
                self._profile_tick(kind, items, wall0, time.time())
                self.cv.notify_all()

    def _next_batch(self):
        c = self.c
        if self.chunked is not None or self.waiting:
            if self.defer_left > 0 and self.running:
                self.defer_left -= 1
                return self._plan_decode()
            self.defer_left = 0
            delay = (c.min_free_slots is not None and self.chunked is None and self.running
                     and (c.mrr - len(self.running)) < c.min_free_slots)
            if not delay:
                items = self._form_prefill()
                if items:
                    return items
        if self.running:
            return self._plan_decode()
        return None

    # ----------------------------------------------------------------- prefill
    def _form_prefill(self):
        c = self.c
        budget = c.chunk if c.chunk > 0 else 1 << 62
        budget = min(budget, c.max_prefill) if c.chunk <= 0 else budget
        items, new_seqs, cached_sum = [], 0, 0
        if self.chunked is not None:
            r = self.chunked
            take = min(budget, r.seq_len() - r.prefilled)
            items.append((r, r.prefilled, r.prefilled + take, False))
            budget -= take
        slots = c.mrr - len(self.running) - (1 if self.chunked is not None else 0)
        ratio = self.new_token_ratio
        reserve = sum(min(max(0, r.max_new - len(r.out)), CLIP_NEW) * ratio for r in self.running)
        used, ev, avail = self.kv()
        rem = avail + ev - reserve
        order = list(self.waiting)
        if c.policy == "lpm" and c.radix:
            order.sort(key=lambda r: -self.tree.common(r.keys))
        new_chunked = None
        for r in order:
            if budget <= 0 or slots <= 0 or new_chunked is not None:
                break
            L = r.seq_len()
            cap = L - 1
            if r.ret_lp and r.lp_start >= 0:
                cap = min(cap, r.lp_start)
            keys = r.keys
            m, path = self.tree.match(keys, cap) if c.radix else (0, [])
            new_tok = L - m
            need = new_tok + min(max(0, r.max_new - len(r.out)), CLIP_NEW) * ratio
            if need > rem and (self.running or self.chunked is not None or items or new_tok > c.pool):
                break                   # FCFS head-of-line: KV-bound (an idle server always takes the head request)
            if c.radix and not self.make_mamba_room(3):
                break
            if self._fault_mamba_assert():
                return None
            self.make_room(min(new_tok, budget))
            self.waiting.remove(r)
            r.state = "prefill"
            r.admitted = True
            r.t_forward_entry = r.t_forward_entry or time.time()
            self.tree.lock_keys(path)
            r.locked = list(path)
            r.cached = m
            r.prefilled = m
            r.keys = keys
            r.branch_ckpt = (self.tree.common(keys) // BLOCK) * BLOCK if c.radix else 0
            take = min(budget, new_tok)
            items.append((r, m, m + take, True))
            budget -= take
            slots -= 1
            rem -= need
            new_seqs += 1
            cached_sum += m
            self.admitted_count += (1 if (r.log_metrics and r.kind != "health" and self.status == "up") else 0)
            if take < new_tok:
                new_chunked = r
        if not items:
            return None
        # this is when 0.5.21 logs the prefill stats and refreshes the gauges (log_prefill_stats at scheduling time)
        new_tokens = sum(e - s for _r, s, e, _n in items)
        new_seqs = len(items)
        chunked_now = new_chunked or (self.chunked if (self.chunked is not None and
                                                       items[0][2] < self.chunked.seq_len()) else None)
        self.chunked = chunked_now
        if self.chunked is not None and self.chunked.state != "chunked":
            self.chunked.state = "chunked"
        rows = 0
        for r, s, e, _n in items:
            if r.ret_lp and r.lp_start >= 0:
                rows += max(0, e - max(s, r.lp_start))
        if self._fault_logprob_oom(rows):
            return None
        now = time.perf_counter()
        gap = max(1e-6, now - self.last_prefill_log)
        self.last_prefill_log = now
        pst = self.pool_stats()
        pending = sum(r.seq_len() for r in self.waiting)
        for r, _s, e_, _n in items:
            pending += r.seq_len() - e_
        self._log_info("Prefill batch, #new-seq: %d, #new-token: %d, #cached-token: %d, full token usage: %.2f, mamba "
                       "usage: %.2f, #running-req: %d, #queue-req: %d, #pending-token: %d, cuda graph: False, input "
                       "throughput (token/s): %.2f" % (new_seqs, new_tokens, cached_sum, pst["full_token_usage"],
                                                        pst["mamba_usage"], len(self.running), len(self.waiting),
                                                        pending, new_tokens / gap))
        self.rt["prefill_compute"] += new_tokens
        self.rt["prefill_cache"] += cached_sum
        self.rt_seen.update(("prefill_compute", "prefill_cache"))
        self.eff["input"] += new_tokens
        self.eff["device_hit"] += cached_sum
        self.passes["prefill_none"] += 1
        self._publish_gauges(prefill=True, cache_hit=(cached_sum / (cached_sum + new_tokens)) if new_tokens else 0.0)
        ctx = max(s for _r, s, _e, _n in items)
        dur = new_tokens / self.prefill_tps(ctx)
        self._oom_log_once()
        return ("prefill", items, dur)

    def _finish_prefill(self, items):
        for r, s, e, is_new in items:
            if r.state == "finished":
                continue
            if self.c.radix:
                keys = r.keys[: e // BLOCK]
                self.tree.insert(keys)
                lk = keys[len(r.locked):]
                self.tree.lock_keys(lk)
                r.locked = r.locked + lk
                if e < r.seq_len() and keys:                       # chunk-end checkpoint (cache_unfinished_req)
                    self.add_ckpt(keys[-1])
                if r.branch_ckpt and r.branch_ckpt <= e and r.branch_ckpt // BLOCK - 1 < len(keys):
                    self.add_ckpt(keys[r.branch_ckpt // BLOCK - 1])
                    r.branch_ckpt = 0
            r.prefilled = e
            if e < r.seq_len():
                continue
            # prefill complete: the extend samples the first output token (prefill logits)
            if self.chunked is r:
                self.chunked = None
            if self.c.radix and r.keys:
                pk = r.keys[: (r.seq_len() - 1) // BLOCK]
                if pk:
                    self.add_ckpt(pk[-1])
            r.t_prefill_end = time.time()
            self.h_rank["prefill_forward"].observe(r.t_prefill_end - (r.t_forward_entry or r.t_arrive))
            self.h_rank["queue"].observe((r.t_forward_entry or r.t_arrive) - r.t_arrive)
            self.h_rank["request_process"].observe(0.00008)
            if r.max_new <= 0:
                self._finish(r, {"type": "length", "length": r.max_new})
                continue
            self._emit_token(r, path="prefill", gen_ok=True)
            if r.state != "finished":
                if r.t_first is None:
                    r.t_first = time.time()
                r.state = "running"
                self.running.append(r)
                r.events.put(("tok", None))
        # scheduler._arm_prefill_decode_interval: after ANY extend batch the next N iterations skip prefill
        self.defer_left = self.c.pdi if (self.c.pdi and self.running) else 0

    # ----------------------------------------------------------------- decode
    def _plan_decode(self):
        c = self.c
        bs = len(self.running)
        if c.dflash and c.dec_graph and bs > c.max_graph_bs:
            self._crash("RuntimeError: The size of tensor a (%d) must match the size of tensor b (%d) at non-singleton "
                        "dimension 0" % (c.max_graph_bs, bs), "graph", where="dflash_worker_v2.py\", line 273")
            return None
        need = bs * max(c.W, 1)
        used, ev, avail = self.kv()
        if avail + ev < need:
            self._retract(need)
            bs = len(self.running)
            if bs == 0:
                return None
        self.make_room(need)
        ctx = sum(r.seq_len() for r in self.running)
        return ("decode", list(self.running), self.step_seconds(bs, ctx))

    def _retract(self, need):
        old_ratio = self.new_token_ratio
        kv_avail0 = self.kv()[2]
        mam0 = self.mamba()[2]
        victims = []
        order = sorted(self.running, key=lambda r: (len(r.out), -r.prompt_len), reverse=True)
        while order:
            used, ev, avail = self.kv()
            if avail + ev >= need + RETRACT_DECODE_STEPS * max(1, len(self.running) - 1) or len(self.running) <= 1:
                break
            r = order.pop()
            self.running.remove(r)
            self._release(r)
            r.state, r.admitted, r.prefilled, r.cached = "waiting", False, 0, 0
            r.n_retract += 1
            r.keys = Tree.block_keys(r.ids + r.out) if self.c.radix else []
            victims.append(r)
        if not victims:
            return
        tot_dec = sum(len(r.out) for r in self.running)
        tot_max = sum(r.max_new for r in self.running) or 1
        self.new_token_ratio = min(1.0, (tot_dec + RETRACT_DECODE_STEPS * len(self.running)) / tot_max)
        if self.retract is None:
            self.retract = [0, 0, 0]
        self.retract[0] += len(victims)
        self.retract[1] += sum(r.prompt_len for r in victims)
        self.retract[2] += sum(len(r.out) for r in victims)
        self.retract_window += len(victims)
        self.waiting.extend(victims)
        self._log_warn("KV cache pool is full. Retract requests. #retracted_reqs: %d, #new_tokens_gained: %d, "
                       "#mamba_num_gained: %d, #new_token_ratio: %.4f -> %.4f"
                       % (len(victims), self.kv()[2] - kv_avail0, self.mamba()[2] - mam0, old_ratio, self.new_token_ratio))

    def _emit_token(self, r, path="decode", gen_ok=True, bug=False):
        """Append one output token to r (forced / greedy / sampled), with logprob bookkeeping; returns True if r finished."""
        pos = len(r.ids) + len(r.out)
        gi = len(r.out)
        tail = (r.ids[-3:] + r.out[-3:])[-3:] if len(r.out) < 3 else r.out[-3:]
        tail = [0] * (3 - len(tail)) + tail
        h = C.ToyLM.ctx_hash(tail[2], tail[1], tail[0], pos)
        if r.forced is not None:
            tok = r.forced[gi] if gi < len(r.forced) else C.EOS_IDS[0]
            dist = self.lm.dist_h(h, path, None)
        else:
            lm = self.lm_bug if (bug and self.lm_bug is not None) else \
                self.lm_restore if (self.lm_restore is not None and r.cached > 0) else self.lm
            dist = lm.dist_h(h, path, gi if path == "decode" else None)
            if self.c.spec and self.sim_len >= 0 and self.sim_mode == "fixed" and path == "decode":
                tok = 100
            elif self.c.spec and self.sim_len >= 0 and self.sim_mode == "real-draft-token" and path == "decode":
                tok = dist.argmax()
            else:
                tok = dist.sample(r.rng, r.temp, r.top_p, r.top_k)
        r.out.append(tok)
        if r.ret_lp:
            r.out_lp.append(dist.lp(tok))
            if r.top_n > 0:
                r.out_top.append(dist.top(r.top_n))
            if r.tid_lp is not None:
                r.out_dist.append(dist)
        if (not r.ignore_eos and tok in C.EOS_IDS) or tok in r.stop_ids:
            self._finish(r, {"type": "stop", "matched": tok})
            return True
        if len(r.out) >= r.max_new:
            self._finish(r, {"type": "length", "length": r.max_new})
            return True
        return False

    def _finish_decode(self, plan):
        c = self.c
        bs = len(plan)
        gen = acc_tok = correct = 0
        hang_after = C.env_int("FAKE_SG_FAULT_AFTER", 20) if c.fault == "hang_decode" else None
        for r in plan:
            if r.state != "running":
                continue
            if c.spec:
                if self.sim_len >= 0:
                    a = max(1, min(c.W, int(round(self.sim_len))))
                else:
                    a = C.draw_accept(c.W, self.accept_prob, C.mix64(C.str_key(r.rid) ^ (self.step_no * 0x9E3779B1)))
                r.verify_ct += 1
            else:
                a = 1
            produced = 0
            bug = c.dflash_bug and a > 1
            for _k in range(a):
                produced += 1
                if self._emit_token(r, "decode", bug=bug):
                    break
            gen += produced
            acc_tok += produced
            correct += produced - 1 if c.spec else 0
            r.correct_drafts += (produced - 1) if c.spec else 0
            now = time.time()
            if r.last_chunk_t is not None and produced:
                self.h_common["itl"].observe((now - r.last_chunk_t) / produced, produced)
            r.last_chunk_t = now
            if r.state != "finished":
                r.events.put(("tok", None))
        self.running = [r for r in self.running if r.state == "running"]
        self.step_no += 1
        self.rt["decode"] += gen
        self.rt_seen.add("decode")
        graph = c.dec_graph and bs <= c.max_graph_bs
        self.passes["decode_cuda_graph" if graph else "decode_none"] += 1
        self.forward_ct_decode += 1
        self.win["gen"] += gen
        self.win["acc_tokens"] += acc_tok
        self.win["fwd_ct"] += bs
        self.win["correct"] += correct
        self.new_token_ratio = max(self.min_ratio, self.new_token_ratio - (INIT_RATIO - self.min_ratio) / RATIO_DECAY_STEPS)
        if self.forward_ct_decode % c.log_interval == 0:
            self._decode_log(bs, graph)
        if hang_after is not None and self.forward_ct_decode >= hang_after and self.status == "up":
            self.hang = True

    def _decode_log(self, bs, graph):
        c = self.c
        now = time.perf_counter()
        gap = max(1e-6, now - self.last_decode_log)
        self.last_decode_log = now
        self.last_gen_tp = self.win["gen"] / gap
        pst = self.pool_stats()
        msg = ("Decode batch, #running-req: %d, #full token: %d, full token usage: %.2f, mamba num: %d, mamba usage: %.2f, "
               % (bs, pst["full_num_used"], pst["full_token_usage"], pst["mamba_used"], pst["mamba_usage"]))
        acc_len = acc_rate = 0.0
        if c.spec:
            acc_len = self.win["acc_tokens"] / max(1, self.win["fwd_ct"])
            acc_rate = self.win["correct"] / max(1, self.win["fwd_ct"] * max(1, c.W - 1))
            msg += "accept len: %.2f, accept rate: %.2f, " % (acc_len, acc_rate)
        msg += "cuda graph: %s, gen throughput (token/s): %.2f, #queue-req: %d" % (graph, self.last_gen_tp, len(self.waiting))
        self._log_info(msg)
        self._publish_gauges(decode_bs=bs, acc=(acc_len, acc_rate), graph=graph)
        self.win = {"gen": 0, "acc_tokens": 0, "fwd_ct": 0, "correct": 0}

    # ----------------------------------------------------------------- gauges
    def _publish_gauges(self, prefill=False, cache_hit=None, decode_bs=None, acc=None, graph=None):
        pst = self.pool_stats()
        g = dict(self.gauges or {})
        g.update(num_running_reqs=len(self.running), num_queue_reqs=len(self.waiting),
                 num_used_tokens=pst["full_num_used"], token_usage=round(max(pst["full_token_usage"], pst["mamba_usage"]), 2),
                 full_token_usage=pst["full_token_usage"], mamba_usage=pst["mamba_usage"],
                 kv_available_tokens=pst["full_available"], kv_evictable_tokens=pst["full_evictable"],
                 kv_used_tokens=pst["full_num_used"], mamba_available_tokens=pst["mamba_available"],
                 mamba_evictable_tokens=pst["mamba_evictable"], mamba_used_tokens=pst["mamba_used"],
                 num_retracted_reqs=self.retract_window, new_token_ratio=self.new_token_ratio)
        self.retract_window = 0
        if prefill:
            g["cache_hit_rate"] = cache_hit or 0.0
            g["gen_throughput"] = self.last_gen_tp if time.perf_counter() - self.last_decode_log < 30 else 0.0
        if decode_bs is not None:
            g["gen_throughput"] = self.last_gen_tp
            g["decode_sum_seq_lens"] = sum(r.seq_len() for r in self.running)
            g["is_cuda_graph"] = 1.0 if graph else 0.0
            if self.c.spec:
                g["spec_accept_length"], g["spec_accept_rate"] = acc
                g["spec_num_steps"] = float(self.d.get("speculative_num_steps") or 1)
                g["spec_num_draft_tokens"] = float(self.c.W)
        self.gauges = g
        self.gauge_t = time.time()

    def _idle_publish(self):
        if self.gauges is None:
            return
        stale = self.gauges.get("num_running_reqs") != len(self.running)
        if stale or time.time() - self.gauge_t > 30:
            pst = self.pool_stats()
            self.gauges.update(num_running_reqs=len(self.running), num_queue_reqs=len(self.waiting), gen_throughput=0.0,
                               num_used_tokens=pst["full_num_used"],
                               token_usage=round(max(pst["full_token_usage"], pst["mamba_usage"]), 2),
                               full_token_usage=pst["full_token_usage"], mamba_usage=pst["mamba_usage"],
                               kv_available_tokens=pst["full_available"], kv_evictable_tokens=pst["full_evictable"],
                               kv_used_tokens=pst["full_num_used"], mamba_available_tokens=pst["mamba_available"],
                               mamba_evictable_tokens=pst["mamba_evictable"], mamba_used_tokens=pst["mamba_used"])
            self.gauge_t = time.time()

    # ----------------------------------------------------------------- flush
    def flush(self):
        with self.lock:
            if self.running or self.waiting or self.chunked is not None:
                self._log_warn("Cache not flushed because there are pending requests. #queue-req: %d, #running-req: %d"
                               % (len(self.waiting), len(self.running)))
                return False
            self.tree.reset()
            if self.c.fault == "oom_log":                  # EMS: the warning came 3 s after a flush, under the next load
                self.oom_logged = False
            self.forward_ct_decode = 0
            self.win = {"gen": 0, "acc_tokens": 0, "fwd_ct": 0, "correct": 0}
            for r in range(self.c.tp):
                self.log("Reset HybridReqToTokenPool", tp_prefix(self.c.tp, r))
            self._log_info("Cache flushed successfully!")
            return True

    # ----------------------------------------------------------------- logging helpers
    def _log_info(self, msg):
        if self.c.log_level in ("debug", "info"):
            self.tp_log(msg)

    def _log_warn(self, msg):
        if self.c.log_level in ("debug", "info", "warning"):
            self.tp_log(msg)

    # ----------------------------------------------------------------- faults
    def _fault_mamba_assert(self):
        if self.c.fault != "mamba_assert" or self.status != "up":
            return False
        if self.admitted_count + 1 < C.env_int("FAKE_SG_FAULT_AFTER", 2):
            return False
        self._crash("AssertionError: Can not alloc mamba cache", "mamba",
                    where="sglang/srt/mem_cache/unified_cache/components/mamba.py\", line 509, in _alloc_mamba_slot")
        return True

    def _fault_logprob_oom(self, rows):
        if self.c.fault != "logprob_oom" or rows <= 256:
            return False
        chunk = os.environ.get("SGLANG_LOGPROB_CHUNK_SIZE")
        try:
            ok = chunk is not None and int(chunk) <= 256
        except ValueError:
            ok = False
        if ok:
            return False
        sz = min(rows, int(chunk) if (chunk and chunk.isdigit()) else 2048) * C.LM_VOCAB * 4 / GIB
        self._crash("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate %.2f GiB. GPU 0 has a total capacity "
                    "of 22.03 GiB of which %.2f GiB is free. Including non-PyTorch memory, this process has 21.01 GiB "
                    "memory in use. Of the allocated memory 18.83 GiB is allocated by PyTorch, and 1.21 GiB is reserved "
                    "by PyTorch but unallocated." % (sz, self.c.avail_final), "oom",
                    where="sglang/srt/layers/logprob_processor.py\", line 698, in _forward_by_chunk")
        return True

    def _oom_log_once(self):
        if self.oom_logged or self.status != "up":
            return
        if self.c.fault == "oom_log" or self.c.avail_final < 0.5:
            self.oom_logged = True
            Log.raw("[rank0]:[W%s.%06d CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while "
                    "trying to allocate 209715200 bytes (free: 154533888, total: 23659151360)."
                    % (time.strftime("%m%d %H:%M:%S"), int((time.time() % 1) * 1e6)))

    def _crash(self, exc_line, kind, where="sglang/srt/managers/scheduler.py\", line 4363, in run_batch"):
        """A scheduler exception on every rank -> SIGQUIT to the launcher -> the whole server dies (scheduler.py:6082)."""
        if self.dead:
            return
        self.dead = True
        tb = ("Traceback (most recent call last):\n  File \"/tmp/sglang-venv/lib/python3.12/site-packages/sglang/srt/"
              "managers/scheduler.py\", line 6082, in run_scheduler_process\n    scheduler.run_event_loop()\n  File "
              "\"/tmp/sglang-venv/lib/python3.12/site-packages/%s\n    raise\n%s" % (where, exc_line))
        for rank in range(self.c.tp):
            self.log("Scheduler hit an exception: %s" % tb, tp_prefix(self.c.tp, rank))
        for r in list(self.reqs.values()):
            r.events.put(("dead", None))
            r.done_evt.set()
        threading.Thread(target=MAIN_STATE["on_sigquit"], daemon=True).start()

    # ----------------------------------------------------------------- profiler bookkeeping
    def _profile_tick(self, kind, items, w0, w1):
        p = self.prof
        if p is None:
            return
        if self.forward_ct < p["start"]:
            return
        bs = len(items)
        toks = sum(e - s for (_r, s, e, _n) in items) if kind == "prefill" else bs
        p["steps"].append((kind, bs, toks, w0, w1))
        if p["num_steps"] is not None and self.forward_ct >= p["start"] + p["num_steps"] + 1:
            self.finish_profile()

    def finish_profile(self):
        p, self.prof = self.prof, None
        if p is None:
            return
        try:
            write_traces(self, p)
        except Exception as e:
            self.log("Failed to export profile: %r" % e)


# ================================================================================================ /metrics (0.5.21 dump order)
TTFT_BUCKETS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0]
E2E_BUCKETS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0, 600.0,
               1200.0, 1800.0, 2400.0]
PROMPT_BUCKETS = [100.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
                  9000.0, 10000.0, 12500.0, 15000.0, 17500.0, 20000.0, 22500.0, 25000.0, 27500.0, 30000.0, 35000.0,
                  40000.0, 60000.0, 80000.0, 100000.0, 200000.0, 300000.0, 400000.0, 600000.0, 800000.0, 1e6, 1.1e6]
ITL_BUCKETS = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6,
               0.8, 1.0, 2.0, 4.0, 6.0, 8.0]
STAGE_BUCKETS = [0.001 * 1.62 ** i for i in range(30)]
QUEUE_BUCKETS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0,
                 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0,
                 1400.0, 1600.0, 1800.0, 2000.0, 2500.0, 3000.0]
EVICT_BUCKETS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2,
                 0.5, 1.0]
HELP = {
    "routing_keys_active": ("gauge", "Number of unique routing keys with active requests"),
    "http_requests_active": ("gauge", "Number of currently active HTTP requests"),
    "prefill_effective_tokens_total": ("counter", "Effective prefill tokens with retracted-request re-counts excluded, updated on each log interval. mode: device_hit, host_hit, storage_hit, input. Windowed prefix cache hit rate = rate(sum of *_hit) / rate(sum of all modes); per-tier rate uses a single *_hit mode in the numerator."),
    "scheduler_idle_seconds_total": ("counter", "Total wall time while the scheduler has no runnable or queued work."),
    "scheduler_process_cpu_seconds_total": ("counter", "Total CPU time consumed by the scheduler process across all threads."),
    "scheduler_stage_seconds_total": ("counter", "Total scheduler-loop wall time exclusively attributed to each stage."),
    "cuda_graph_passes_total": ("counter", "Total number of forward passes categorized by CUDA graph."),
    "evicted_tokens_total": ("counter", "The number of device KV token slots freed by eviction, regardless of whether the data was backed up to host (see sglang:hicache_backup_tokens_total) or destroyed (see sglang:hicache_dropped_tokens_total)."),
    "max_total_num_tokens": ("gauge", "Maximum total number of tokens in the KV cache pool."),
    "weight_memory_usage_gb": ("gauge", "Memory used by model weights in GB."),
    "kv_cache_memory_usage_gb": ("gauge", "Memory used by the KV cache pools in GB."),
    "graph_memory_usage_gb": ("gauge", "Memory used by captured device graphs in GB."),
    "page_size": ("gauge", "KV cache page size in tokens."),
    "num_pages": ("gauge", "Number of KV cache pages."),
    "context_len": ("gauge", "Maximum context length."),
    "startup_available_gpu_memory_gb": ("gauge", "Available GPU memory in GB at startup."),
    "num_running_reqs": ("gauge", "The number of running requests."),
    "num_queue_reqs": ("gauge", "The number of requests in the waiting queue."),
    "num_grammar_queue_reqs": ("gauge", "The number of requests in the grammar waiting queue."),
    "gen_throughput": ("gauge", "The generation throughput (token/s)."),
    "cache_hit_rate": ("gauge", "The prefix cache hit rate."),
    "decode_sum_seq_lens": ("gauge", "The sum of all sequence lengths in decode."),
    "token_usage": ("gauge", "The token usage."),
    "full_token_usage": ("gauge", "The token usage for full attention layers."),
    "swa_token_usage": ("gauge", "The token usage for SWA layers."),
    "mamba_usage": ("gauge", "The token usage for Mamba layers."),
    "num_used_tokens": ("gauge", "The number of used tokens."),
    "kv_available_tokens": ("gauge", "Number of free token slots in the KV cache pool."),
    "kv_evictable_tokens": ("gauge", "Number of evictable (radix-cached) token slots in the KV cache pool."),
    "kv_used_tokens": ("gauge", "Number of actively used token slots in the KV cache pool."),
    "swa_available_tokens": ("gauge", "Number of free token slots in the SWA pool (hybrid-SWA only)."),
    "swa_evictable_tokens": ("gauge", "Number of evictable (radix-cached) token slots in the SWA pool."),
    "swa_used_tokens": ("gauge", "Number of actively used token slots in the SWA pool."),
    "mamba_available_tokens": ("gauge", "Number of free state slots in the mamba SSM pool (hybrid-SSM only)."),
    "mamba_evictable_tokens": ("gauge", "Number of evictable (radix-cached) state slots in the mamba SSM pool."),
    "mamba_used_tokens": ("gauge", "Number of actively used state slots in the mamba SSM pool."),
    "spec_accept_length": ("gauge", "Mean acceptance length of speculative decoding (accepted drafts + bonus token per forward)."),
    "spec_accept_rate": ("gauge", "Speculative acceptance rate (`accepted drafts / proposed drafts` in batch)."),
    "spec_cap_length": ("gauge", "Mean DSpark confidence-scheduled verify window per verify step, incl the bonus slot (0 when no cap is scheduled)."),
    "spec_block_accept_length": ("gauge", "Mean uncapped full-block accept length per verify step (accept + cap-trimmed drafts; exact only in DSpark cap-accept mode)."),
    "spec_num_steps": ("gauge", "Currently active speculative_num_steps."),
    "spec_num_draft_tokens": ("gauge", "Currently active speculative_num_draft_tokens (decouples from steps under topk>1)."),
    "num_prefill_bootstrap_queue_reqs": ("gauge", "The number of requests in the prefill bootstrap queue."),
    "num_prefill_inflight_queue_reqs": ("gauge", "The number of requests in the prefill inflight queue."),
    "num_decode_prealloc_queue_reqs": ("gauge", "The number of requests in the decode prealloc queue."),
    "num_decode_transfer_queue_reqs": ("gauge", "The number of requests in the decode transfer queue."),
    "num_decode_host_receive_queue_reqs": ("gauge", "Requests in the decode transfer queue receiving into host memory or waiting for device admission."),
    "pending_prealloc_token_usage": ("gauge", "The token usage for pending preallocated tokens (not preallocated yet)."),
    "utilization": ("gauge", "The utilization."),
    "fwd_occupancy": ("gauge", "Forward pass GPU occupancy percentage."),
    "new_token_ratio": ("gauge", "The new token ratio."),
    "is_cuda_graph": ("gauge", "Whether the batch is using CUDA graph."),
    "num_unique_running_routing_keys": ("gauge", "Number of unique routing keys in running batch."),
    "routing_key_running_req_count": ("gauge", "Distribution of routing keys by running request count (gt < count <= le)."),
    "routing_key_all_req_count": ("gauge", "Distribution of routing keys by running+waiting request count (gt < count <= le)."),
    "process_cpu_seconds_total": ("counter", "Total CPU time consumed by this process (user + system)"),
    "startup_time_seconds": ("gauge", "Engine startup duration by phase in seconds."),
    "startup_cuda_graph_time_seconds": ("gauge", "CUDA graph capture duration by phase in seconds."),
    "time_to_first_token_seconds": ("histogram", "Histogram of time to first token in seconds."),
    "e2e_request_latency_seconds": ("histogram", "Histogram of End-to-end request latency in seconds"),
    "prompt_tokens_histogram": ("histogram", "Histogram of prompt token length."),
    "uncached_prompt_tokens_histogram": ("histogram", "Histogram of uncached (compute) prompt token length."),
    "generation_tokens_histogram": ("histogram", "Histogram of generation token length."),
    "inter_token_latency_seconds": ("histogram", "Histogram of inter-token latency in seconds."),
    "per_stage_req_latency_seconds": ("histogram", "The latency of each stage of requests."),
    "queue_time_seconds": ("histogram", "Histogram of queueing time in seconds."),
    "eviction_duration_seconds": ("histogram", "End-to-end time of a device eviction pass in seconds; under --hicache-write-policy write_back this includes the blocking D->H backup (see sglang:hicache_backup_duration_seconds for the copy alone)."),
    "http_requests_total": ("counter", "Total number of HTTP requests by endpoint and method"),
    "http_responses_total": ("counter", "Total number of HTTP responses by endpoint and status code"),
    "prompt_tokens_total": ("counter", "Number of prefill tokens processed."),
    "generation_tokens_total": ("counter", "Number of generation tokens processed."),
    "spec_verify_calls_total": ("counter", "Number of speculative decoding verification calls."),
    "num_requests_total": ("counter", "Number of requests processed."),
    "cached_tokens_total": ("counter", "Number of cached prompt tokens by source (device/host/storage)."),
    "num_aborted_requests_total": ("counter", "Number of requests aborted."),
    "realtime_tokens_total": ("counter", "Total number of tokens processed (updated on each log interval). mode: prefill_compute, prefill_cache, decode."),
    "num_retracted_requests_total": ("counter", "Total number of retracted requests."),
    "num_retracted_input_tokens_total": ("counter", "Total number of retracted input tokens."),
    "num_retracted_output_tokens_total": ("counter", "Total number of retracted output tokens."),
    "num_retracted_reqs": ("gauge", "The number of retracted requests."),
    "num_paused_reqs": ("gauge", "The number of paused requests by async weight sync."),
}
TP0_GAUGES = ["num_running_reqs", "num_queue_reqs", "num_grammar_queue_reqs", "gen_throughput", "cache_hit_rate",
              "decode_sum_seq_lens", "token_usage", "full_token_usage", "swa_token_usage", "mamba_usage", "num_used_tokens",
              "kv_available_tokens", "kv_evictable_tokens", "kv_used_tokens", "swa_available_tokens", "swa_evictable_tokens",
              "swa_used_tokens", "mamba_available_tokens", "mamba_evictable_tokens", "mamba_used_tokens", "spec_accept_length",
              "spec_accept_rate", "spec_cap_length", "spec_block_accept_length", "spec_num_steps", "spec_num_draft_tokens",
              "num_prefill_bootstrap_queue_reqs", "num_prefill_inflight_queue_reqs", "num_decode_prealloc_queue_reqs",
              "num_decode_transfer_queue_reqs", "num_decode_host_receive_queue_reqs", "pending_prealloc_token_usage",
              "utilization", "fwd_occupancy", "new_token_ratio", "is_cuda_graph", "num_unique_running_routing_keys"]
RK_EDGES = [("0", "1"), ("1", "2"), ("2", "3"), ("3", "5"), ("5", "7"), ("7", "10"), ("10", "20"), ("20", "50"),
            ("50", "100"), ("100", "200"), ("200", "+Inf")]


def render_metrics(e):
    c, P = e.c, "sglang:"
    out = []

    def fam(name, samples):
        typ, hlp = HELP[name]
        out.append("# HELP %s%s %s" % (P, name, hlp))
        out.append("# TYPE %s%s %s" % (P, name, typ))
        out.extend(samples)

    def s(name, labels, v):
        return "%s%s%s %s" % (P, name, C.plabels(labels), C.pnum(v))

    ranks = list(range(c.tp))
    order = (ranks[1:3] + [0] + ranks[3:]) if c.tp >= 3 else ([1, 0] if c.tp == 2 else [0])
    base = {"engine_type": "unified", "model_name": c.served, "moe_ep_rank": "0", "pp_rank": "0"}
    R = lambda r, **kw: dict(base, tp_rank=str(r), **kw)
    tokl = {"engine_type": "unified", "model_name": c.served}
    with e.lock:
        g = dict(e.gauges or {})
        fam("routing_keys_active", [s("routing_keys_active", {}, 0)])
        fam("http_requests_active", [s("http_requests_active", {"endpoint": k[0], "method": k[1]}, v[1])
                                     for k, v in e.http.items()])
        fam("prefill_effective_tokens_total", [s("prefill_effective_tokens_total", R(r, mode=m), e.eff[m] if r == 0 else 0)
                                               for r in order for m in ("input", "device_hit", "host_hit", "storage_hit")])
        up = time.time() - e.t_start
        fam("scheduler_idle_seconds_total", [s("scheduler_idle_seconds_total", R(r), e.idle_sec[r]) for r in order])
        fam("scheduler_process_cpu_seconds_total", [s("scheduler_process_cpu_seconds_total", R(r), 0.9 * up + e.busy_sec)
                                                    for r in order])
        fam("scheduler_stage_seconds_total", [s("scheduler_stage_seconds_total", R(r, category=k), v)
                                              for r in order for k, v in e.stage.items()])
        cg = []
        for r in order:
            if r == 0 and e.passes["prefill_none"]:
                cg.append(s("cuda_graph_passes_total", R(r, mode="prefill_none"), e.passes["prefill_none"]))
            for m in ("decode_cuda_graph", "decode_none"):
                if e.passes[m]:
                    cg.append(s("cuda_graph_passes_total", R(r, mode=m), e.passes[m]))
        fam("cuda_graph_passes_total", cg)
        ev = {"cache_type": "UnifiedRadixCache", "dp_rank": "0", "pp_rank": "0"}
        fam("evicted_tokens_total", [s("evicted_tokens_total", dict(ev, tp_rank=str(r)), e.tree.evicted_total) for r in order])
        const_order = [0, 2, 1, 3][:c.tp] if c.tp == 4 else ranks
        kv_gb = c.pool * (c.target_bytes_tok + c.draft_bytes_tok) / GIB
        for name, v in (("max_total_num_tokens", c.pool), ("weight_memory_usage_gb", 4.619140625 + (1.49 if c.dflash else 0)),
                        ("kv_cache_memory_usage_gb", kv_gb)):
            fam(name, [s(name, R(r), v) for r in const_order])
        fam("graph_memory_usage_gb", [s("graph_memory_usage_gb", R(r, phase=ph), v) for r in const_order
                                      for ph, v in c.graph_gb.items()])
        for name, v in (("page_size", 1), ("num_pages", c.pool), ("context_len", c.ctx_len),
                        ("startup_available_gpu_memory_gb", c.avail_final)):
            fam(name, [s(name, R(r), v) for r in const_order])
        if g:
            for name in TP0_GAUGES:
                v = g.get(name, 0.0)
                if name == "fwd_occupancy":
                    v = float("nan")
                fam(name, [s(name, R(0), v)])
            for name in ("routing_key_running_req_count", "routing_key_all_req_count"):
                fam(name, [s(name, R(0, gt=a, le=b), 0) for a, b in RK_EDGES])
        fam("process_cpu_seconds_total", [s("process_cpu_seconds_total", {"component": "detokenizer"}, 0.02 * up),
                                          s("process_cpu_seconds_total", {"component": "tokenizer"}, 0.05 * up)])
        st = e.startup_time
        if st:
            fam("startup_time_seconds", [s("startup_time_seconds", dict(tokl, phase=k), st[k])
                                         for k in ("load_weight", "kv_cache_allocation", "scheduler_e2e", "tokenizer_e2e")])
            fam("startup_cuda_graph_time_seconds", [s("startup_cuda_graph_time_seconds", dict(tokl, phase=k), v)
                                                    for k, v in st["cuda_graph"].items()])
        for key, name in (("ttft", "time_to_first_token_seconds"), ("e2e", "e2e_request_latency_seconds")):
            lines = []
            for strm, hs in sorted(e.h.items()):
                lines += hs[key].lines(P + name, dict(tokl, is_streaming=strm))
            if lines:
                fam(name, lines)
        if e.tok:
            for key, name in (("prompt", "prompt_tokens_histogram"), ("uncached", "uncached_prompt_tokens_histogram"),
                              ("gen", "generation_tokens_histogram"), ("itl", "inter_token_latency_seconds")):
                fam(name, e.h_common[key].lines(P + name, tokl))
            ps = []
            for r in [3, 2, 1, 0][-c.tp:] if c.tp <= 4 else ranks:
                for st_ in ("request_process", "prefill_forward", "chunked_prefill"):
                    ps += e.h_rank[st_].lines(P + "per_stage_req_latency_seconds", R(r, stage=st_))
            fam("per_stage_req_latency_seconds", ps)
            fam("queue_time_seconds", [x for r in ([3, 2, 1, 0][-c.tp:] if c.tp <= 4 else ranks)
                                       for x in e.h_rank["queue"].lines(P + "queue_time_seconds", R(r))])
            if e.h_rank["evict"].n:
                fam("eviction_duration_seconds", [x for r in ([3, 2, 1, 0][-c.tp:] if c.tp <= 4 else ranks)
                                                  for x in e.h_rank["evict"].lines(P + "eviction_duration_seconds",
                                                                                   dict(ev, tp_rank=str(r)))])
        fam("http_requests_total", [s("http_requests_total", {"endpoint": k[0], "method": k[1]}, v[0])
                                    for k, v in e.http.items()])
        fam("http_responses_total", [s("http_responses_total", {"endpoint": k[0], "method": k[1], "status_code": str(k[2])}, v)
                                     for k, v in e.http_resp.items()])
        if e.tok:
            fam("prompt_tokens_total", [s("prompt_tokens_total", dict(tokl, is_streaming=k), v["prompt"])
                                        for k, v in sorted(e.tok.items())])
            fam("generation_tokens_total", [s("generation_tokens_total", dict(tokl, is_streaming=k), v["gen"])
                                            for k, v in sorted(e.tok.items())])
            if c.spec:
                fam("spec_verify_calls_total", [s("spec_verify_calls_total", tokl, e.spec_verify_total)])
            fam("num_requests_total", [s("num_requests_total", dict(tokl, is_streaming=k), v["n"])
                                       for k, v in sorted(e.tok.items())])
            fam("cached_tokens_total", [s("cached_tokens_total", dict(tokl, cache_source="device"), e.cached_total)])
            fam("num_aborted_requests_total", [s("num_aborted_requests_total", tokl, e.aborted_total)])
        if e.rt_seen:
            fam("realtime_tokens_total", [s("realtime_tokens_total", R(0, mode=m), e.rt[m])
                                          for m in ("prefill_compute", "decode", "prefill_cache") if m in e.rt_seen])
        if e.retract is not None:
            for i, name in enumerate(("num_retracted_requests_total", "num_retracted_input_tokens_total",
                                      "num_retracted_output_tokens_total")):
                fam(name, [s(name, R(r), e.retract[i]) for r in order])
        if g:
            fam("num_retracted_reqs", [s("num_retracted_reqs", R(0, pid=str(e.tp0_pid)), g.get("num_retracted_reqs", 0))])
            fam("num_paused_reqs", [s("num_paused_reqs", R(0, pid=str(e.tp0_pid)), 0)])
    return "\n".join(out) + "\n"


# ================================================================================================ profiler traces
def write_traces(e, p):
    """One kineto-like <id>-TP-<r>.trace.json.gz per rank: CPU step spans, graph launches, GPU kernels incl. NCCL."""
    c = e.c
    out_dir = p["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ch, algo, proto, block = nccl_effective(dict(os.environ), c.fault == "nccl_env_ignored")
    ar_name = "ncclDevKernel_AllReduce_Sum_bf16_%s_%s(ncclDevKernelArgsStorage<4096ul>)" % (
        algo, proto if C.env_bool("FAKE_SG_NCCL_PROTO_IN_NAME") else "LL")
    ag_name = "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
    marlin = ["void sglang::device::marlin::Marlin<__nv_bfloat16, 1125899907892224l, 256, 4, 16, 4, false, 4, 8, false>"
              "(int4 const*, int4 const*, int4*, int4*, int4 const*, unsigned short const*, int4 const*, int const*)",
              "void sglang::device::marlin::Marlin<__nv_bfloat16, 1125899907892224l, 256, 2, 16, 4, false, 4, 8, false>"
              "(int4 const*, int4 const*, int4*, int4*, int4 const*, unsigned short const*, int4 const*, int const*)"]
    gdn = ["fused_sigmoid_gating_delta_rule_update_kernel", "_causal_conv1d_update_kernel", "fused_qkvzba_split_reshape_cat_contiguous_kernel"]
    attn = "void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, 64u, 1u, 4u, 16u, 16u, 4u, 1u, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16, float, int, flashinfer::DefaultAttention<false, false, false, false> > >"
    gemm = "ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn"
    norm = "kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64512051201_0"
    samp = "void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int)"
    # per-step GPU time shares (E14A R2: NCCL AR 50.9, AG 8.1, Marlin 25.3, bf16 GEMM 14.5, GDN 8.0, rest 9.1 of ~116 ms)
    plan = [("ar", 140, 50.9), ("ag", 3, 8.1), ("marlin", 64, 25.3), ("gemm", 8, 14.5), ("gdn", 48, 8.0),
            ("attn", 16, 3.0), ("norm", 32, 1.5), ("samp", 4, 2.4)]
    tot = sum(x[2] for x in plan)
    hostname = socket.gethostname().split(".")[0]
    base_ns = int(p["t0"] * 1e9) // 1_000_000_000 * 1_000_000_000
    paths = []
    for rank in range(c.tp):
        pid = e.child_pids[rank] if rank < len(e.child_pids) else os.getpid()
        title = "sglang::scheduler" + ("_TP%d" % rank if c.tp > 1 else "")
        ev = [{"ph": "M", "name": "process_name", "ts": 0, "pid": pid, "tid": 0, "args": {"name": title}},
              {"ph": "M", "name": "process_labels", "ts": 0, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
              {"ph": "M", "name": "process_name", "ts": 0, "pid": rank, "tid": 0, "args": {"name": title}},
              {"ph": "M", "name": "process_labels", "ts": 0, "pid": rank, "tid": 0, "args": {"labels": "GPU %d" % rank}}]
        corr = 1000
        skew = rank * 3.0
        for kind, bs, toks, w0, w1 in p["steps"]:
            ts0 = (w0 - base_ns / 1e9) * 1e6 + skew
            span = max(50.0, (w1 - w0) * 1e6 * 0.97)
            ev.append({"ph": "X", "cat": "user_annotation", "name": "scheduler.run_batch", "pid": pid, "tid": pid,
                       "ts": round(ts0, 3), "dur": round(span, 3), "args": {"External id": corr}})
            if kind == "prefill":
                names = [("step[EXTEND bs=%d toks=%d]" % (bs, toks), span)]
            elif c.spec:
                names = [("step[DRAFT bs=%d]" % bs, span * 0.25), ("step[VERIFY bs=%d]" % bs, span * 0.70)]
            else:
                names = [("step[DECODE bs=%d]" % bs, span * 0.95)]
            t = ts0
            for nm, du in names:
                ev.append({"ph": "X", "cat": "user_annotation", "name": nm, "pid": pid, "tid": pid, "ts": round(t, 3),
                           "dur": round(du, 3), "args": {"External id": corr, "Record function id": 0}})
                corr += 1
                launch = "cudaGraphLaunch" if (kind != "prefill" and c.dec_graph and bs <= c.max_graph_bs) else "cudaLaunchKernel"
                ev.append({"ph": "X", "cat": "cuda_runtime", "name": launch, "pid": pid, "tid": pid, "ts": round(t + 5, 3),
                           "dur": 30.0, "args": {"cbid": 311 if launch == "cudaGraphLaunch" else 211, "correlation": corr}})
                if nm.startswith(("step[VERIFY", "step[DECODE", "step[EXTEND")):
                    ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": nm, "pid": rank, "tid": 24,
                               "ts": round(t + 40, 3), "dur": round(du * 0.95, 3), "args": {"External id": corr}})
                    gt = t + 40
                    gpu_span = du * 0.92
                    for cat, n, ms in plan:
                        if kind == "prefill" and cat in ("ag", "samp"):
                            continue
                        each = gpu_span * (ms / tot) / n
                        for i in range(n):
                            if cat == "ar":
                                name, grid, blk = ar_name, [ch, 1, 1], [block, 1, 1]
                            elif cat == "ag":
                                name, grid, blk = ag_name, [ch, 1, 1], [288, 1, 1]
                            elif cat == "marlin":
                                name, grid, blk = marlin[i % 2], [58, 1, 1], [256, 1, 1]
                            elif cat == "gdn":
                                name, grid, blk = gdn[i % 3], [32, 1, 1], [128, 1, 1]
                            elif cat == "attn":
                                name, grid, blk = attn, [8, 4, 1], [32, 4, 1]
                            elif cat == "gemm":
                                name, grid, blk = gemm, [64, 1, 1], [128, 1, 1]
                            elif cat == "norm":
                                name, grid, blk = norm, [bs, 1, 1], [512, 1, 1]
                            else:
                                name, grid, blk = samp, [bs, 1, 1], [256, 1, 1]
                            k = {"ph": "X", "cat": "kernel", "name": name, "pid": rank, "tid": 24, "ts": round(gt, 3),
                                 "dur": round(each * 0.98, 3),
                                 "args": {"queued": 0, "device": rank, "context": 1, "stream": 24, "correlation": corr,
                                          "registers per thread": 96, "shared memory": 96384 if cat in ("ar", "ag") else 0,
                                          "grid": grid, "block": blk, "est. achieved occupancy %": 0,
                                          "graph id": 32 if launch == "cudaGraphLaunch" else 0}}
                            if cat == "ar":
                                k["args"].update({"Collective name": "allreduce", "In msg nelems": 491520,
                                                  "Out msg nelems": 491520, "Group size": c.tp, "dtype": "BFloat16"})
                            ev.append(k)
                            gt += each
                t += du
            corr += 1
        tr = {"schemaVersion": 1,
              "deviceProperties": [{"id": i, "name": "NVIDIA L4", "totalGlobalMem": 23659151360, "computeMajor": 8,
                                    "computeMinor": 9, "maxThreadsPerBlock": 1024, "maxThreadsPerMultiprocessor": 1536,
                                    "regsPerBlock": 65536, "warpSize": 32, "sharedMemPerBlock": 49152, "numSms": 58,
                                    "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 101376,
                                    "sharedMemPerMultiprocessor": 102400} for i in range(c.tp)],
              "trace_id": "%032X" % (C.mix64(rank + int(p["t0"])) << 64 | C.mix64(rank)),
              "distributedInfo": {"backend": "nccl", "rank": rank, "world_size": c.tp, "pg_count": 45},
              "cuda_driver_version": 13000, "cuda_runtime_version": 13000, "cupti_version": 130001,
              "host_name": hostname, "displayTimeUnit": "ms", "baseTimeNanoseconds": base_ns, "traceEvents": ev,
              "traceName": "/tmp/tmp%08x.json" % (C.mix64(rank) & 0xFFFFFFFF)}
        stem = "%s%s-TP-%d" % ((p["prefix"] + "-") if p["prefix"] else "", p["profile_id"], rank)
        path = os.path.join(out_dir, stem + ".trace.json.gz")
        C.gz_json(path, tr)
        paths.append(path)
    time.sleep(min(2.0, 0.002 * sum(len(x) for x in [p["steps"]]) + 0.05))      # export stalls serving (synchronous)
    return paths


# ================================================================================================ HTTP layer
MAIN_STATE = {}


def _resp_meta(e, r, final, sent_from=0, incremental=False):
    """meta_info like tokenizer_manager (key order: id, finish_reason, prompt_tokens, weight_version, num_retractions,
    [time stats], [logprobs], reasoning_tokens, completion_tokens, cached_tokens, [e2e + spec + finish stats])."""
    mi = {"id": r.rid, "finish_reason": r.finish if final else None, "prompt_tokens": r.prompt_len,
          "weight_version": "default", "num_retractions": r.n_retract}
    if e.c.metrics:
        if r.t_forward_entry:
            mi["forward_entry_time"] = r.t_forward_entry
        if r.t_prefill_end:
            mi["prefill_finished_time"] = r.t_prefill_end
        mi["queue_time"] = max(0.0, (r.t_forward_entry or time.time()) - r.t_arrive)
    if r.ret_lp:
        n_out = len(r.out_lp)
        lo = sent_from if incremental else 0
        txt = r.text_in_lp
        mi["input_token_logprobs"] = input_logprobs(e, r, txt) if (final or sent_from == 0) else []
        mi["output_token_logprobs"] = [[lp, t, C.tok_text(t) if txt else None] for lp, t in
                                       zip(r.out_lp[lo:n_out], r.out[lo:n_out])]
        mi["output_token_logprobs_length"] = n_out
        if r.top_n > 0:
            mi["input_top_logprobs"] = input_top(e, r, txt) if (final or sent_from == 0) else []
            mi["output_top_logprobs"] = [[[lp, t, C.tok_text(t) if txt else None] for lp, t in row]
                                         for row in r.out_top[lo:n_out]]
        if r.tid_lp is not None:
            mi["input_token_ids_logprobs"] = input_tid(e, r, txt) if (final or sent_from == 0) else []
            mi["output_token_ids_logprobs"] = [_tid_row(d, r.tid_lp, txt) for d in r.out_dist[lo:n_out]]
    mi["reasoning_tokens"] = 0
    mi["completion_tokens"] = len(r.out)
    mi["cached_tokens"] = r.cached
    if final:
        mi["e2e_latency"] = (r.t_finish or time.time()) - r.t_arrive
        if e.c.spec and r.verify_ct > 0:
            prop = r.verify_ct * (e.c.W - 1)
            if prop > 0:
                mi["spec_accept_rate"] = r.correct_drafts / prop
                mi["spec_accept_length"] = len(r.out) / r.verify_ct
                mi["spec_num_correct_drafts"] = r.correct_drafts
                mi["spec_num_proposed_drafts"] = prop
                mi["spec_verify_ct"] = r.verify_ct
        if e.c.metrics:
            mi["request_received_ts"] = r.t_arrive
            if r.t_first:
                mi["first_token_latency"] = r.t_first - r.t_arrive
            mi["response_sent_to_client_ts"] = time.time()
            mi["request_finished_ts"] = r.t_finish or time.time()
            if r.t_first and len(r.out) > 1 and (r.t_finish or 0) > r.t_first:
                mi["decode_throughput"] = (len(r.out) - 1) / (r.t_finish - r.t_first)
    return mi


def _tid_row(dist, ids, txt):
    tab = dist.lp_table()
    bg = dist.bg_lp
    if txt:
        return [[tab.get(t, bg), t, C.tok_text(t)] for t in ids]
    return [[tab.get(t, bg), t, None] for t in ids]


def _lp_start(r):
    """Absolute start of the input-logprob arrays: len(origin_input_ids[logprob_start_len:]) entries (logprob_result_
    processor._calculate_relevant_tokens_len); -1 survives only with token_ids_logprob and slices the last input id."""
    if r.lp_start is None:
        return None
    if r.lp_start < 0:
        return len(r.ids) - 1 if r.tid_lp is not None else None
    return r.lp_start


def input_logprobs(e, r, txt=False):
    """[None-entry] + rows[:-1]: entry 0 = (None, ids[s]), entry j = (lp(ids[s+j] | ids[:s+j]), ids[s+j])."""
    s = _lp_start(r)
    if s is None:
        return []
    ids = r.ids
    out = [[None, ids[s], C.tok_text(ids[s]) if txt else None]] if s < len(ids) else []
    for pos in range(s + 1, len(ids)):
        d = e.lm_prefill.dist(ids, pos, "prefill")
        out.append([d.lp(ids[pos]), ids[pos], C.tok_text(ids[pos]) if txt else None])
    return out


def input_top(e, r, txt=False):
    s = _lp_start(r)
    if s is None:
        return []
    ids = r.ids
    out = [None] if s < len(ids) else []
    for pos in range(s + 1, len(ids)):
        d = e.lm_prefill.dist(ids, pos, "prefill")
        out.append([[lp, t, C.tok_text(t) if txt else None] for lp, t in d.top(r.top_n)])
    return out


def input_tid(e, r, txt=False):
    s = _lp_start(r)
    if s is None:
        return []
    ids = r.ids
    out = [None] if s < len(ids) else []
    for pos in range(s + 1, len(ids)):
        out.append(_tid_row(e.lm_prefill.dist(ids, pos, "prefill"), r.tid_lp, txt))
    return out


def toy_encode(text):
    """Deterministic word-level ids for text prompts (the lab always sends ids; this keeps text requests working)."""
    ids = []
    for w in re.findall(r"\S+|\n", text or ""):
        ids.append(1000 + C.str_key(w) % 246000)
    return ids or [0]


class Handler(C.BaseHandler):
    def _engine(self):
        return MAIN_STATE["engine"]

    def _count(self, method, path, delta_active):
        e = self._engine()
        k = (path, method)
        with e.lock:
            v = e.http.setdefault(k, [0, 0])
            if delta_active > 0:
                v[0] += 1
            v[1] = max(0, v[1] + delta_active)

    def _access(self, code):
        super()._access(code)
        e = MAIN_STATE.get("engine")
        if e is not None:
            p = self.path.split("?", 1)[0]
            with e.lock:
                k = (p, self.command, code)
                e.http_resp[k] = e.http_resp.get(k, 0) + 1

    def _route(self, method):
        path = self.path.split("?", 1)[0]
        self._count(method, path, +1)
        try:
            return self._dispatch(method, path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self._count(method, path, -1)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def _json_body(self):
        raw = self.read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as ex:
            self.send_json({"detail": [{"type": "json_invalid", "loc": ["body", ex.pos], "msg": "JSON decode error",
                                        "input": {}, "ctx": {"error": ex.msg}}]}, 422)
            return None

    def _dispatch(self, method, path):
        e = self._engine()
        if path == "/ready" and method == "GET":
            return self.send_bytes(b"", 200 if e.status == "up" else 503, "text/plain")
        if path in ("/health", "/health_generate") and method == "GET":
            return self._health()
        if path == "/metrics" and method == "GET":
            if not e.c.metrics:
                return self.send_json({"detail": "Not Found"}, 404)
            return self.send_bytes(render_metrics(e).encode(), 200, "text/plain; version=0.0.4; charset=utf-8")
        if path in ("/model_info", "/get_model_info") and method == "GET":
            if path == "/get_model_info":
                e.log("Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use "
                      "'/model_info' instead.")
            return self.send_json(model_info(e))
        if path in ("/server_info", "/get_server_info") and method == "GET":
            if path == "/get_server_info":
                e.log("Endpoint '/get_server_info' is deprecated and will be removed in a future version. Please use "
                      "'/server_info' instead.")
            return self.send_json(server_info(e))
        if path == "/v1/models" and method == "GET":
            return self.send_json({"object": "list", "data": [{"id": e.c.served, "object": "model",
                                                               "created": int(time.time()), "owned_by": "sglang",
                                                               "root": e.c.served, "parent": None,
                                                               "max_model_len": e.c.ctx_len}]})
        if path == "/flush_cache" and method in ("GET", "POST"):
            self.read_body() if method == "POST" else None
            ok = e.flush()
            if ok:
                return self.send_bytes(b"Cache flushed.\nPlease check backend logs for more details. (When there are "
                                       b"running or waiting requests, the operation will not be performed.)\n", 200,
                                       "text/plain; charset=utf-8")
            return self.send_bytes(b"Flush cache failed.\n", 400, "text/plain; charset=utf-8")
        if path == "/freeze_gc" and method == "POST":
            self.read_body()
            return self.send_bytes(b"", 200, "text/plain")
        if path == "/abort_request" and method == "POST":
            body = self._json_body()
            if body is None:
                return
            e.abort(rid=body.get("rid"), abort_all=bool(body.get("abort_all")))
            return self.send_bytes(b"", 200, "text/plain")
        if path == "/start_profile" and method in ("GET", "POST"):
            body = self._json_body() if method == "POST" else {}
            if body is None:
                return
            return self._start_profile(body or {})
        if path == "/stop_profile" and method in ("GET", "POST"):
            self.read_body() if method == "POST" else None
            with e.lock:
                if e.prof is None:
                    return self.send_bytes(b"Internal Server Error", 500, "text/plain")
                e.finish_profile()
            return self.send_bytes(b"Stop profiling. This will take some time.\n", 200, "text/plain; charset=utf-8")
        if path == "/generate" and method in ("POST", "PUT"):
            body = self._json_body()
            if body is None:
                return
            return self._generate(body)
        if path == "/v1/completions" and method == "POST":
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self.read_body()
                return self.send_json({"detail": "Unsupported Media Type: Only 'application/json' is allowed"}, 415)
            body = self._json_body()
            if body is None:
                return
            return self._completions(body)
        if path == "/v1/chat/completions" and method == "POST":
            body = self._json_body()
            if body is None:
                return
            return self._chat(body)
        self.read_body() if method != "GET" else None
        return self.send_json({"detail": "Not Found"}, 404)

    # ------------------------------------------------------------- endpoints
    def _health(self):
        e = self._engine()
        if e.status != "up":
            return self.send_bytes(b"", 503, "text/plain")
        if not C.env_bool("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", True):
            return self.send_bytes(b"", 200, "text/plain")
        tic = time.time()
        r = Req("HEALTH_CHECK_%s" % C.new_rid(), [0], {"max_new_tokens": 1, "temperature": 0.0}, kind="health",
                log_metrics=False)
        try:
            e.submit(r)
        except Exception:
            return self.send_bytes(b"", 503, "text/plain")
        time.sleep(C.env_float("FAKE_SG_HEALTH_SLEEP", 1.0))
        ok = r.done_evt.wait(timeout=20) or e.progress_t > tic
        return self.send_bytes(b"", 200 if ok and not e.dead else 503, "text/plain")

    def _start_profile(self, body):
        e = self._engine()
        with e.lock:
            if e.prof is not None:
                return self.send_bytes(b"Internal Server Error", 500, "text/plain")
            ns = body.get("num_steps")
            st = body.get("start_step")
            start = e.forward_ct + 1
            if st is not None:
                start = max(int(st), e.forward_ct + 1)
            e.prof = {"output_dir": body.get("output_dir") or os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp"),
                      "num_steps": int(ns) if ns is not None else None, "start": start,
                      "profile_id": str(body.get("profile_id") or time.time()), "prefix": body.get("profile_prefix"),
                      "activities": body.get("activities") or ["CPU", "GPU"], "t0": time.time(), "steps": []}
        return self.send_bytes(b"Start profiling.\n", 200, "text/plain; charset=utf-8")

    def _validate(self, e, ids, sp):
        if ids is None or len(ids) == 0:
            raise ValueError("Either text, input_ids or input_embeds should be provided.")
        n = len(ids)
        if n >= e.c.ctx_len:
            raise ValueError("The input (%d tokens) is longer than the model's context length (%d tokens)." % (n, e.c.ctx_len))
        mx = sp.get("max_new_tokens")
        if mx is not None and n + int(mx) > e.c.ctx_len:
            raise ValueError("Requested token count exceeds the model's maximum context length of %d tokens. You "
                             "requested a total of %d tokens: %d tokens from the input messages and %d tokens for the "
                             "completion. Please reduce the number of tokens in the input messages or the completion to "
                             "fit within the limit." % (e.c.ctx_len, n + int(mx), n, int(mx)))
        bad = set(sp) - SAMPLING_KEYS
        if bad:
            raise ValueError("SamplingParams.__init__() got an unexpected keyword argument '%s'" % sorted(bad)[0])

    def _generate(self, body):
        e = self._engine()
        sp = dict(body.get("sampling_params") or {})
        ids = body.get("input_ids")
        if ids is None and body.get("text") is not None:
            ids = toy_encode(body["text"])
        rid = body.get("rid") or C.new_rid()
        try:
            self._validate(e, ids, sp)
            lps = body.get("logprob_start_len", -1)
            lps = -1 if lps is None else int(lps)
            ret_lp = bool(body.get("return_logprob"))
            tid = body.get("token_ids_logprob")
            if ret_lp and lps == -1 and tid is None:
                lps = len(ids)                      # scheduler.py L3047-3051: output logprobs only
            if lps > len(ids):
                raise ValueError("req.logprob_start_len=%d is higher than the number of input tokens "
                                 "len(req.origin_input_ids)=%d. Please use a smaller logprob_start_len." % (lps, len(ids)))
            r = Req(rid, ids, sp, ret_lp=ret_lp, top_n=body.get("top_logprobs_num") or 0,
                    lp_start=lps if ret_lp else -1, tid_lp=tid if ret_lp else None,
                    text_in_lp=bool(body.get("return_text_in_logprobs")), stream=bool(body.get("stream")),
                    log_metrics=body.get("log_metrics", True) is not False)
            e.submit(r)
        except ValueError as ex:
            e.log("[http_server] Error: %s" % ex)
            return self.send_json({"error": {"message": str(ex)}}, 400)
        except RuntimeError:
            return self.send_json({"error": {"message": "server is dead"}}, 503)
        want_prompt_ids = bool(body.get("return_prompt_token_ids"))
        if not r.stream:
            r.done_evt.wait()
            if r.finish is None:
                return self._dead_close()
            out = {"text": detok(r.out, r.skip_special), "output_ids": list(r.out), "meta_info": _resp_meta(e, r, True)}
            if want_prompt_ids:
                out["prompt_token_ids"] = list(r.ids)
            return self.send_bytes(C.dumps(out).encode(), 200, "application/json")
        self.start_sse()
        sent = 0
        incr = e.c.incremental
        try:
            while True:
                kind, _x = r.events.get()
                while not r.events.empty():
                    k2, _ = r.events.get_nowait()
                    kind = k2 if k2 != "tok" else kind
                if kind == "dead":
                    return self._dead_close(stream=True)
                n = len(r.out)
                final = r.state == "finished"
                if not final and n - sent < e.c.stream_interval and n % e.c.stream_interval:
                    continue
                if incr:
                    out = {"text": detok(r.out[sent:n], r.skip_special), "output_ids": r.out[sent:n],
                           "meta_info": _resp_meta(e, r, final, sent, True)}
                else:
                    out = {"text": detok(r.out[:n], r.skip_special), "output_ids": r.out[:n],
                           "meta_info": _resp_meta(e, r, final, 0, False)}
                if want_prompt_ids:
                    out["prompt_token_ids"] = list(r.ids)
                self.sse(C.dumps(out))
                sent = n
                if final:
                    break
            self.sse("[DONE]")
            self.end_chunked()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            e.abort(rid=r.rid, msg="client disconnected")

    def _dead_close(self, stream=False):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _completions(self, body):
        e = self._engine()
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        ids = prompt if isinstance(prompt, list) else toy_encode(prompt if isinstance(prompt, str) else "")
        mt = body.get("max_tokens", 16)
        if mt is not None and int(mt) <= 0:
            return self.send_json({"detail": [{"type": "value_error", "loc": ["body", "max_tokens"],
                                               "msg": "Value error, max_tokens must be positive", "input": mt}]}, 422)
        sp = {"temperature": body.get("temperature", 1.0), "max_new_tokens": int(mt) if mt is not None else 16,
              "top_p": body.get("top_p", 1.0), "top_k": body.get("top_k", -1), "ignore_eos": bool(body.get("ignore_eos")),
              "stop_token_ids": body.get("stop_token_ids"), "sampling_seed": body.get("seed"),
              "skip_special_tokens": body.get("skip_special_tokens", True)}
        lp_n = body.get("logprobs")
        rid = body.get("rid") or C.new_rid()
        stream = bool(body.get("stream"))
        try:
            self._validate(e, ids, {k: v for k, v in sp.items() if k in SAMPLING_KEYS})
            r = Req(rid, ids, sp, ret_lp=lp_n is not None, top_n=lp_n or 0, lp_start=len(ids), text_in_lp=True,
                    stream=stream, kind="completions")
            e.submit(r)
        except ValueError as ex:
            return self.send_json({"object": "error", "message": str(ex), "type": "BadRequestError", "param": None,
                                   "code": 400}, 400)
        model = body.get("model") or "default"
        created = int(time.time())
        rti = bool(body.get("return_token_ids"))
        so = body.get("stream_options") or {}
        include_usage = bool(so.get("include_usage")) or bool(e.d.get("stream_response_default_include_usage"))
        if not stream:
            r.done_evt.wait()
            if r.finish is None:
                return self._dead_close()
            ch = {"index": 0, "text": detok(r.out, r.skip_special),
                  "logprobs": _oai_logprobs(r, 0, len(r.out)) if lp_n is not None else None,
                  "finish_reason": _oai_finish(r.finish), "matched_stop": r.finish.get("matched")}
            if rti:
                ch["token_ids"] = list(r.out)
                ch["prompt_token_ids"] = list(r.ids)
            res = {"id": r.rid, "object": "text_completion", "created": created, "model": model, "choices": [ch],
                   "usage": _usage(e, r), "metadata": None}
            return self.send_bytes(C.dumps(res).encode(), 200, "application/json")
        self.start_sse()
        sent = 0
        first = True
        try:
            while True:
                kind, _x = r.events.get()
                while not r.events.empty():
                    k2, _ = r.events.get_nowait()
                    kind = k2 if k2 != "tok" else kind
                if kind == "dead":
                    return self._dead_close(stream=True)
                n = len(r.out)
                final = r.state == "finished"
                if n == sent and not final:
                    continue
                ch = {"index": 0, "text": detok(r.out[sent:n], r.skip_special),
                      "logprobs": _oai_logprobs(r, sent, n) if lp_n is not None else None,
                      "finish_reason": _oai_finish(r.finish) if final else None,
                      "matched_stop": r.finish.get("matched") if (final and r.finish) else None}
                if rti:
                    ch["token_ids"] = r.out[sent:n]
                    if first:
                        ch["prompt_token_ids"] = list(r.ids)
                self.sse(C.dumps({"id": r.rid, "object": "text_completion", "created": created, "model": model,
                                  "choices": [ch], "usage": None}))
                first = False
                sent = n
                if final:
                    break
            if include_usage:
                self.sse(C.dumps({"id": r.rid, "object": "text_completion", "created": created, "model": model,
                                  "choices": [], "usage": _usage(e, r, exclude_none=True)}))
            self.sse("[DONE]")
            self.end_chunked()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            e.abort(rid=r.rid, msg="client disconnected")

    def _chat(self, body):
        e = self._engine()
        msgs = body.get("messages") or []
        last = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "") or ""
        if isinstance(last, list):
            last = " ".join(x.get("text", "") for x in last if isinstance(x, dict))
        kw = body.get("chat_template_kwargs") or {}
        thinking = kw.get("enable_thinking", True) is not False and e.c.reasoning_parser is not None
        tools = body.get("tools") or []
        m = re.search(r"(-?\d+)\s*([-+*/x])\s*(-?\d+)\s*=", last)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            val = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a // b if b else 0}[op]
            answer = str(val)
        else:
            answer = "The grid transformation is a reflection."
        reasoning = ("The user asks: %s I compute it step by step and answer." % last.strip()[:120]) if thinking else None
        tool_calls = None
        if tools and body.get("tool_choice") != "none" and e.c.tool_parser:
            fn = (tools[0].get("function") or {})
            props = ((fn.get("parameters") or {}).get("properties") or {})
            args = {}
            for k, sch in props.items():
                t = (sch or {}).get("type")
                args[k] = 1 if t in ("integer", "number") else (True if t == "boolean" else ([] if t == "array" else "x"))
            tool_calls = [{"id": "call_%s" % C.new_rid()[:24], "index": 0, "type": "function",
                           "function": {"name": fn.get("name", "tool"), "arguments": json.dumps(args)}}]
        text = ((("<think>\n" + reasoning + "\n</think>\n\n") if reasoning else "") + (answer if not tool_calls else ""))
        forced = toy_encode(text) + [C.EOS_IDS[0]]
        prompt = [248045] + toy_encode(json.dumps(msgs))[:4000] + [248046, 248045]
        mt = body.get("max_completion_tokens") or body.get("max_tokens")
        sp = {"max_new_tokens": int(mt) if mt else len(forced), "temperature": body.get("temperature", 1.0)}
        r = Req(C.new_rid(), prompt, sp, kind="chat", forced=forced)
        try:
            e.submit(r)
        except Exception as ex:
            return self.send_json({"object": "error", "message": str(ex), "type": "BadRequestError", "param": None,
                                   "code": 400}, 400)
        r.done_evt.wait()
        if r.finish is None:
            return self._dead_close()
        trunc = r.finish.get("type") == "length"
        msg = {"role": "assistant", "content": None if tool_calls else (answer if not trunc else answer[:1]),
               "reasoning_content": reasoning, "tool_calls": tool_calls}
        fr = "tool_calls" if tool_calls else _oai_finish(r.finish)
        res = {"id": r.rid, "object": "chat.completion", "created": int(time.time()), "model": body.get("model") or "default",
               "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": fr,
                            "matched_stop": None if tool_calls else r.finish.get("matched")}],
               "usage": _usage(e, r), "metadata": None}
        return self.send_bytes(C.dumps(res).encode(), 200, "application/json")


SAMPLING_KEYS = {"max_new_tokens", "stop", "stop_token_ids", "stop_regex", "temperature", "top_p", "top_k", "min_p",
                 "frequency_penalty", "presence_penalty", "repetition_penalty", "min_new_tokens", "n", "json_schema",
                 "regex", "ebnf", "structural_tag", "ignore_eos", "skip_special_tokens", "spaces_between_special_tokens",
                 "no_stop_trim", "stream_interval", "logit_bias", "sampling_seed", "custom_params"}


GRID_TEXT = C.env_str("FAKE_SG_TEXT_STYLE", "") == "grids"


def detok(ids, skip_special=True):
    """FAKE_SG_TEXT_STYLE=grids: ARC answer grids, one digit per token, a row break every 15th token id (chunk-safe)."""
    if GRID_TEXT:
        return "".join(str(t % 10) + ("\n" if t % 15 == 0 else " ") for t in ids)
    return C.detok(ids, skip_special)


def _oai_finish(fr):
    if not fr:
        return None
    return fr.get("type")


def _oai_logprobs(r, a, b):
    """OpenAI completion logprobs (SGLang: return_text_in_logprobs=True, top_logprobs keyed by token TEXT)."""
    toks, lps, tops, offs = [], [], [], []
    off = len(detok(r.out[:a], r.skip_special))
    for i in range(a, min(b, len(r.out_lp))):
        t = r.out[i]
        txt = C.tok_text(t)
        toks.append(txt)
        lps.append(r.out_lp[i])
        offs.append(off)
        off += len(txt)
        if r.top_n > 0 and i < len(r.out_top):
            d = {}
            for lp, tid in r.out_top[i]:
                d.setdefault(C.tok_text(tid), lp)
            tops.append(d)
        else:
            tops.append(None)
    return {"text_offset": offs, "token_logprobs": lps, "tokens": toks, "top_logprobs": tops}


def _usage(e, r, exclude_none=False):
    u = {"prompt_tokens": r.prompt_len, "total_tokens": r.prompt_len + len(r.out), "completion_tokens": len(r.out),
         "prompt_tokens_details": ({"cached_tokens": r.cached} if (e.c.cache_report and r.cached > 0) else None),
         "reasoning_tokens": 0}
    if exclude_none and u["prompt_tokens_details"] is None:
        del u["prompt_tokens_details"]
    return u


def model_info(e):
    return {"model_path": e.c.model, "served_model_name": e.c.served, "tokenizer_path": e.d.get("tokenizer_path"),
            "is_generation": True, "preferred_sampling_params": e.d.get("preferred_sampling_params"),
            "weight_version": "default", "load_format": e.d.get("load_format", "auto"),
            "reasoning_parser": e.d.get("reasoning_parser"), "tool_call_parser": e.d.get("tool_call_parser"),
            "disaggregation_mode": "null", "has_image_understanding": True, "has_audio_understanding": False,
            "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}


def server_info(e):
    c = e.c
    d = dict(e.d)
    internal = dict(d)
    internal.update(disable_chunked_prefix_cache=True, pp_max_micro_batch_size=c.mrr,
                    disaggregation_decode_retraction_backup="cpu_tensor")
    internal.update({"world_size": c.tp, "last_gen_throughput": e.last_gen_tp,
                     "memory_usage": {"weight": round(4.619 + (1.49 if c.dflash else 0), 3),
                                      "kvcache": round(c.pool * (c.target_bytes_tok + c.draft_bytes_tok) / GIB, 3),
                                      "startup_available": round(c.avail_final, 3), "token_capacity": c.pool,
                                      "token_capacity_swa": None, "graph": c.graph_gb},
                     "effective_max_running_requests_per_dp": c.mrr, "decode_cuda_graph_bs": [],
                     "decode_cuda_graph_memory_gb": round(sum(c.graph_gb.values()), 3)})
    out = dict(d)
    out.update({"launch_command": " ".join(c.argv), "status": "ready" if e.status == "up" else "starting",
                "max_total_num_tokens": c.pool, "max_req_input_len": c.ctx_len - 6, "startup_time": e.startup_time or None,
                "internal_states": [internal], "version": VERSION, "frontend": "python", "kv_events": None})
    return out


# ================================================================================================ main
def _warn_launch_server():
    sys.stderr.write("%s:71: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the "
                     "recommended entrypoint.\n  Example: sglang serve --model-path <model> [options]\n  warnings.warn(\n"
                     % os.path.abspath(__file__))
    sys.stderr.flush()


def startup_seconds(cold, fault):
    if fault == "slow_start":
        v = C.env_float("FAKE_SG_SLOW_START_SEC", 0.0)
        if v > 0:
            return v
        return 700.0 * C.time_scale() if C.time_scaled_env() else 30.0
    v = C.env_float("FAKE_SG_STARTUP_SEC", -1.0)
    if v >= 0:
        return v
    if C.time_scaled_env():
        return (490.0 if cold else 62.0) * C.time_scale()
    return 2.0


def main(argv):
    if argv and argv[0] == "--e24-flag-check":
        try:
            av = json.loads(argv[1]) if len(argv) > 1 else []
        except Exception as ex:
            print(json.dumps({"ran": False, "ok": False, "unknown": [], "ambiguous": [], "parse_error": "bad json: %r" % ex,
                              "method": "fake_sglang"}))
            return 0
        if av and str(av[0]).endswith(("python", "python3", "python3.12")):
            av = av[1:]
        if len(av) >= 2 and av[0] == "-m":
            av = av[2:]
        print(json.dumps(flag_check(av, resolve="--resolve" in argv[2:])))
        return 0
    if os.environ.get("FAKE_SG_ORIG_ARGV", "1") != "0":
        # the real launcher runs as `python -m sglang.launch_server <flags>`: code that classifies the process by
        # sys.orig_argv (the E24 hook's role kind) sees the same; the OS cmdline (pgrep -f) keeps fake_sglang_server.py
        sys.orig_argv = [sys.executable, "-m", "sglang.launch_server"] + list(argv)
    _warn_launch_server()
    parser = build_parser()
    ns = parser.parse_args(argv)          # exits 2 on unknown / ambiguous / bad values, prints help on --help
    log = Log("")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")          # engine.py _set_envs_and_config (before the spawn)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    sgc = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
    for k, sub in (("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "nv"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                   ("TILELANG_CACHE_DIR", "tilelang"), ("FLASHINFER_WORKSPACE_BASE", "")):
        os.environ.setdefault(k, os.path.join(sgc, sub) if sub else sgc)       # import sglang: redirect_third_party_caches
    log("Registered linear attn model: config=GigaChat35Config, backend=GDNAttnBackend, archs=['GigaChat35ForCausalLM', "
        "'GigaChat35ForCausalLMNextN']")
    try:
        d, cfg, warns = resolve_args(ns, argv)
    except Exception:
        Log.raw(traceback.format_exc().rstrip())
        return 1
    if "attention_backend" not in cfg.given:
        log("Attention backend not specified. Use flashinfer backend by default.")
    for w in warns:
        log(w)
    if d.get("disable_cuda_graph"):
        log("--disable-cuda-graph is deprecated. Please use --cuda-graph-backend-{decode,prefill}=disabled instead.")
    fault = (C.env_str("FAKE_SG_FAULT", "") or "").strip()
    cfg.fault = fault
    cfg.weights = C.weights_family(cfg.model)
    cfg.noise = C.env_float("FAKE_SG_NOISE", 0.004)
    cfg.path_noise = C.env_float("FAKE_SG_PATH_NOISE", 0.004)
    cfg.engine_key = "sglang|%s|%s|%s%s" % (cfg.ssm, cfg.kv_dtype, "verify" if cfg.spec else "decode",
                                           ("|" + os.environ["FAKE_SG_NOISE_KEY"]) if os.environ.get("FAKE_SG_NOISE_KEY") else "")
    da = C.env_int("FAKE_SG_DIVERGE_AT", -1)
    cfg.diverge_at = da if da >= 0 else None
    cfg.dflash_bug = cfg.dflash and C.env_bool("FAKE_SG_DFLASH_BUG")
    cfg.dflash_bug_at = C.env_int("FAKE_SG_DFLASH_BUG_AT", 64)
    nignored = fault == "nccl_env_ignored"
    ch, algo, proto, _b = nccl_effective(dict(os.environ), nignored)
    speed = 1.0
    if not nignored:
        speed *= {2: 1.0, 4: 0.97, 8: 0.98}.get(ch, 1.0)
        speed *= {"RING": 1.0, "TREE": 1.05}[algo] * {"LL": 1.0, "SIMPLE": 1.04, "LL128": 1.02}[proto]
    cfg.nccl_speed = speed
    log("server_args=%r" % (d,))
    leak = C.env_bool("FAKE_SG_LEAK_DEFAULT_CACHE")
    tri = os.environ["TRITON_CACHE_DIR"]
    cold = not C.dir_has_files(tri)
    T = startup_seconds(cold, fault)
    t_launch = time.time()
    block_mm = None
    if cfg.dflash:
        bsz = _draft_block_size(d.get("speculative_draft_model_path") or "") or 8
        if bsz != cfg.W:
            block_mm = ("DFLASH block size mismatch: using speculative_num_draft_tokens=%d but draft config block_size=%d."
                        % (cfg.W, bsz))
    child_cfg = {"tp": cfg.tp, "startup_sec": T, "t_launch": t_launch, "fault": fault, "nccl_ignored": nignored,
                 "kv_fp8": cfg.kv_dtype.startswith("fp8"), "dflash": cfg.dflash, "spec": cfg.spec, "W": cfg.W,
                 "block_mismatch": block_mm, "inter_gb": cfg.inter_gb, "inter_conv_gb": cfg.inter_conv_gb,
                 "mamba_size": cfg.mamba_size, "conv_gb": cfg.conv_gb, "ssm_gb": cfg.ssm_gb, "pool": cfg.pool,
                 "target_bytes_tok": cfg.target_bytes_tok, "draft_bytes_tok": cfg.draft_bytes_tok,
                 "avail_after_pool": cfg.avail_after_pool, "avail_final": cfg.avail_final, "graph_bs": cfg.graph_bs,
                 "dec_graph": cfg.dec_graph, "graph_gb": cfg.graph_gb, "cold": cold, "leak": leak,
                 "capped_line": cfg.capped_line, "chunk": cfg.chunk, "max_prefill": cfg.max_prefill, "mrr": cfg.mrr,
                 "ctx_len": cfg.ctx_len, "radix": cfg.radix, "stub_modules": C.env_bool("FAKE_SG_STUB_MODULES", True)}
    ctx = multiprocessing.get_context("spawn")
    conns, procs = [], []
    engine = Engine(cfg, d, log)
    MAIN_STATE.update(engine=engine, procs=procs, log=log)
    _install_signals(engine, procs, log)
    time.sleep(max(0.0, t_launch + 0.03 * T - time.time()))
    for rank in range(cfg.tp):
        rd, wr = ctx.Pipe(duplex=True)
        cc = dict(child_cfg, cache_targets=_cache_targets(dict(os.environ), rank, cfg.dflash, leak))
        p = ctx.Process(target=scheduler_child, args=(rank, cc, wr))
        p.start()
        procs.append(p)
        conns.append(rd)
    drd, dwr = ctx.Pipe(duplex=True)
    dp = ctx.Process(target=detokenizer_child, args=(child_cfg, dwr))
    dp.start()
    procs.append(dp)
    conns.append(drd)
    MAIN_STATE["conns"] = conns
    infos = []
    for rank, cn in enumerate(conns):
        try:
            if not cn.poll(T * 3 + 120):
                raise EOFError("timeout")
            info = cn.recv()
        except (EOFError, OSError):
            info = {"ok": False, "error": "child died"}
        if not info.get("ok"):
            time.sleep(1.0)        # the SIGQUIT handler normally takes over; if not, fail like _wait_for_scheduler_ready
            log("Rank %d scheduler is dead. Please check if there are relevant logs." % rank)
            _kill_children(procs)
            return 1
        infos.append(info)
    engine.child_pids = [i["pid"] for i in infos[:cfg.tp]]
    engine.tp0_pid = engine.child_pids[0] if engine.child_pids else os.getpid()
    el = time.time() - t_launch
    engine.startup_time = {"load_weight": round(0.08 * el, 6), "kv_cache_allocation": round(0.02 * el, 6),
                           "scheduler_e2e": round(0.86 * el, 6),
                           "cuda_graph": {"prefill": 0.0, "decode": 0.0 if cfg.spec else round(0.6 * el, 6),
                                          "target_verify": round(0.6 * el, 6) if cfg.spec else 0.0, "draft_prefill": 0.0,
                                          "draft_decode": round(0.05 * el, 6) if cfg.dflash else 0.0,
                                          "draft_extend": 0.0},
                           "tokenizer_e2e": round(el, 6)}
    st = engine.startup_time
    log("Engine startup timings (s): load_weight=%.2f, kv_cache_allocation=%.2f, scheduler_e2e=%.2f, cuda_graph={prefill="
        "%.2f, decode=%.2f, target_verify=%.2f, draft_prefill=%.2f, draft_decode=%.2f, draft_extend=%.2f}, tokenizer_e2e=%.2f"
        % (st["load_weight"], st["kv_cache_allocation"], st["scheduler_e2e"], st["cuda_graph"]["prefill"],
           st["cuda_graph"]["decode"], st["cuda_graph"]["target_verify"], st["cuda_graph"]["draft_prefill"],
           st["cuda_graph"]["draft_decode"], st["cuda_graph"]["draft_extend"], st["tokenizer_e2e"]))
    threading.Thread(target=engine.run, daemon=True, name="fake-scheduler").start()
    Handler.owner = AccessLog(log)
    try:
        httpd = C.Server((cfg.host, cfg.port), Handler)
    except OSError as ex:
        log("ERROR: [Errno %s] error while attempting to bind on address ('%s', %d): %s"
            % (ex.errno, cfg.host, cfg.port, ex.strerror))
        _kill_children(procs)
        return 1
    MAIN_STATE["httpd"] = httpd
    log("INFO:     Started server process [%d]" % os.getpid())
    log("INFO:     Waiting for application startup.")
    if d.get("sampling_defaults", "model") == "model":
        log("Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}")
    log("INFO:     Application startup complete.")
    log("INFO:     Uvicorn running on http://%s:%d (Press CTRL+C to quit)" % (cfg.host, cfg.port))
    threading.Thread(target=_warmup, args=(engine, cfg, t_launch, T, log), daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    return 0


class AccessLog:
    def __init__(self, log):
        self.log = log

    def access_log(self, client, method, path, code):
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = ""
        self.log('INFO:     %s:%d - "%s %s HTTP/1.1" %d %s' % (client[0], client[1], method, path, code, phrase))


def _warmup(engine, cfg, t_launch, T, log):
    """_execute_server_warmup: GET /model_info, one chat request (8 tokens), POST /freeze_gc, then status Up."""
    time.sleep(max(0.0, t_launch + 0.92 * T - time.time()))
    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "") else cfg.host
    import urllib.request as U
    try:
        U.urlopen("http://%s:%d/model_info" % (host, cfg.port), timeout=30).read()
        if not cfg.skip_warmup:
            body = json.dumps({"model": cfg.served, "messages": [{"role": "user", "content": "The capital city of France is"}],
                               "max_tokens": 8, "temperature": 0.0}).encode()
            U.urlopen(U.Request("http://%s:%d/v1/chat/completions" % (host, cfg.port), data=body,
                                headers={"Content-Type": "application/json"}), timeout=120).read()
        time.sleep(max(0.0, t_launch + T - time.time()))
        U.urlopen(U.Request("http://%s:%d/freeze_gc" % (host, cfg.port), data=b"", method="POST"), timeout=30).read()
    except Exception as ex:
        log("Initialization failed. warmup error: %r" % ex)
        return
    with engine.lock:
        engine.status = "up"
        engine.progress_t = time.time()
    log("The server is fired up and ready to roll!")


def _kill_children(procs):
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(1.0)
            if p.is_alive():
                p.kill()
        except Exception:
            pass


def _install_signals(engine, procs, log):
    def die(how="sigquit"):
        try:
            with engine.lock:
                engine.dead = True
                for r in list(engine.reqs.values()):
                    r.events.put(("dead", None))
                    r.done_evt.set()
        except Exception:
            pass
        _kill_children(procs)

    def on_sigquit(*_a):
        log("SIGQUIT received. signum=None, frame=None. It usually means one child failed.")
        time.sleep(min(5.0, 5.0 * C.time_scale()))
        die()
        log("No live scheduler processes found; skipping py-spy and CUDA coredump.")
        log("kill_process_tree called: parent_pid=%d, include_parent=True, pid=%d" % (os.getpid(), os.getpid()))
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    def on_sigterm(*_a):
        log("SIGTERM received. signum=None frame=None. Draining requests and shutting down...")
        die()
        log("kill_process_tree called: parent_pid=%d, include_parent=True, pid=%d" % (os.getpid(), os.getpid()))
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    MAIN_STATE["on_sigquit"] = on_sigquit
    signal.signal(signal.SIGQUIT, lambda *a: threading.Thread(target=on_sigquit, daemon=True).start())
    signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=on_sigterm, daemon=True).start())


# ================================================================================================ data (generated)
# CLI option table: SCR/wp2/extract_sglang_flags.py over the sglang-0.5.21 wheel (srt/arg_groups/fields/*.py + manual
# add_cli_args). Row = [option strings, kind, type, nargs, choices, default, dest, required, help (truncated)].
FLAGS_TABLE = r"""
[["--device"],"scalar","str","",null,null,"device",0,""]
[["--base-gpu-id"],"scalar","int","",null,0,"base_gpu_id",0,""]
[["--gpu-id-step"],"scalar","int","",null,1,"gpu_id_step",0,""]
[["--random-seed"],"scalar","int","",null,null,"random_seed",0,""]
[["--mlx-enable-sampling"],"store_true","","",null,false,"mlx_enable_sampling",0,""]
[["--watchdog-timeout"],"scalar","float","",null,300,"watchdog_timeout",0,""]
[["--soft-watchdog-timeout"],"scalar","float","",null,null,"soft_watchdog_timeout",0,""]
[["--sleep-on-idle"],"store_true","","",null,false,"sleep_on_idle",0,""]
[["--use-ray"],"store_true","","",null,false,"use_ray",0,""]
[["--numa-node"],"list","int","+",null,null,"numa_node",0,""]
[["--gc-threshold"],"list","int","+",null,null,"gc_threshold",0,""]
[["--decoupled-spec-bind-endpoint"],"scalar","str","",null,null,"decoupled_spec_bind_endpoint",0,""]
[["--decoupled-spec-connect-endpoints"],"scalar","json_list_type","",null,null,"decoupled_spec_connect_endpoints",0,""]
[["--decoupled-spec-rank"],"scalar","int","",null,null,"decoupled_spec_rank",0,""]
[["--decoupled-spec-role"],"scalar","str","",["null","verifier","drafter"],"null","decoupled_spec_role",0,""]
[["--disaggregation-mode"],"scalar","str","",["null","prefill","decode"],"null","disaggregation_mode",0,""]
[["--disaggregation-transfer-backend"],"scalar","str","",["mooncake","nixl","ascend","fake","mori","mooncake_tcp"],"mooncake","disaggregation_transfer_backend",0,""]
[["--disaggregation-enable-kv-checksum"],"store_true","","",null,false,"disaggregation_enable_kv_checksum",0,""]
[["--disaggregation-bootstrap-port"],"scalar","int","",null,8998,"disaggregation_bootstrap_port",0,""]
[["--disaggregation-ib-device"],"scalar","str","",null,null,"disaggregation_ib_device",0,""]
[["--disaggregation-decode-enable-radix-cache"],"store_true","","",null,false,"disaggregation_decode_enable_radix_cache",0,""]
[["--disaggregation-decode-enable-offload-kvcache"],"store_true","","",null,false,"disaggregation_decode_enable_offload_kvcache",0,""]
[["--disaggregation-decode-retraction-backup"],"scalar","str","",["cpu_tensor","host_pool","none"],null,"disaggregation_decode_retraction_backup",0,""]
[["--num-reserved-decode-tokens"],"scalar","int","",null,512,"num_reserved_decode_tokens",0,""]
[["--disaggregation-decode-extra-slots"],"scalar","int","",null,null,"disaggregation_decode_extra_slots",0,""]
[["--disaggregation-decode-polling-interval"],"scalar","int","",null,1,"disaggregation_decode_polling_interval",0,""]
[["--enable-pd-role-switch"],"store_true","","",null,false,"enable_pd_role_switch",0,""]
[["--optimistic-prefill-attempts"],"scalar","int","",null,0,"optimistic_prefill_attempts",0,""]
[["--encoder-only"],"store_true","","",null,false,"encoder_only",0,""]
[["--language-only"],"store_true","","",null,false,"language_only",0,""]
[["--language-model-only"],"store_true","","",null,false,"language_model_only",0,""]
[["--encoder-transfer-backend"],"scalar","str","",["auto","zmq_to_scheduler","zmq_to_tokenizer","mooncake"],"auto","encoder_transfer_backend",0,""]
[["--encoder-urls"],"list","str","+",null,[],"encoder_urls",0,""]
[["--encoder-bootstrap-port"],"scalar","int","",null,8997,"encoder_bootstrap_port",0,""]
[["--encoder-register-urls"],"list","str","+",null,[],"encoder_register_urls",0,""]
[["--enable-adaptive-dispatch-to-encoder"],"store_true","","",null,false,"enable_adaptive_dispatch_to_encoder",0,""]
[["--enable-pdmux"],"store_true","","",null,false,"enable_pdmux",0,""]
[["--pdmux-config-path"],"scalar","str","",null,null,"pdmux_config_path",0,""]
[["--sm-group-num"],"scalar","int","",null,8,"sm_group_num",0,""]
[["--disaggregation-decode-host-receive-threshold"],"scalar","float","",null,0.0,"disaggregation_decode_host_receive_threshold",0,""]
[["--enable-fp32-lm-head"],"store_true","","",null,false,"enable_fp32_lm_head",0,""]
[["--enable-tf32-matmul"],"store_true","","",null,false,"enable_tf32_matmul",0,""]
[["--enable-memory-saver"],"store_true","","",null,false,"enable_memory_saver",0,""]
[["--enable-weights-cpu-backup"],"store_true","","",null,false,"enable_weights_cpu_backup",0,""]
[["--enable-draft-weights-cpu-backup"],"store_true","","",null,false,"enable_draft_weights_cpu_backup",0,""]
[["--enable-custom-logit-processor"],"store_true","","",null,false,"enable_custom_logit_processor",0,""]
[["--enable-return-hidden-states"],"store_true","","",null,false,"enable_return_hidden_states",0,""]
[["--return-hidden-states-mode"],"scalar","str","",["last","full"],null,"return_hidden_states_mode",0,""]
[["--enable-return-routed-experts"],"store_true","","",null,false,"enable_return_routed_experts",0,""]
[["--enable-return-indexer-topk"],"store_true","","",null,false,"enable_return_indexer_topk",0,""]
[["--enable-encoder-swa-bounded-replay"],"store_true","","",null,false,"enable_encoder_swa_bounded_replay",0,""]
[["--enable-decoder-swa-bounded-replay"],"store_true","","",null,false,"enable_decoder_swa_bounded_replay",0,""]
[["--sampling-mask-max-tokens"],"scalar","int","",null,4096,"sampling_mask_max_tokens",0,""]
[["--disable-outlines-disk-cache"],"store_true","","",null,false,"disable_outlines_disk_cache",0,""]
[["--enable-mis"],"store_true","","",null,false,"enable_mis",0,""]
[["--attention-backend"],"scalar","str","","@attention_backend",null,"attention_backend",0,""]
[["--decode-attention-backend"],"scalar","str","","@attention_backend",null,"decode_attention_backend",0,""]
[["--enable-lean-attention"],"store_true","","",null,null,"enable_lean_attention",0,""]
[["--prefill-attention-backend"],"scalar","str","","@attention_backend",null,"prefill_attention_backend",0,""]
[["--prefill-kv-cache-dequant-dtype"],"scalar","str","",["auto","nvfp4","fp8_e4m3"],"auto","prefill_kv_cache_dequant_dtype",0,""]
[["--grammar-backend"],"scalar","str","",["xgrammar","outlines","llguidance","none"],null,"grammar_backend",0,""]
[["--fp8-gemm-backend"],"scalar","str","","@fp8_gemm_runner_backend","auto","fp8_gemm_runner_backend",0,""]
[["--fp4-gemm-backend"],"scalar","str","","@fp4_gemm_runner_backend","auto","fp4_gemm_runner_backend",0,""]
[["--bf16-gemm-backend"],"scalar","str","",["auto","cutedsl","gemv","torch"],"auto","bf16_gemm_backend",0,""]
[["--dsa-prefill-backend"],"scalar","str","","@dsa_prefill_backend",null,"dsa_prefill_backend",0,""]
[["--dsv4-prefill-backend"],"scalar","str","",["auto","flashmla_sparse","flashmla_sparse_q8"],"auto","dsv4_prefill_backend",0,""]
[["--dsa-decode-backend"],"scalar","str","","@dsa_prefill_backend",null,"dsa_decode_backend",0,""]
[["--dsv4-attn-backend"],"scalar","str","",["auto","flashmla","trtllm"],"auto","dsv4_attn_backend",0,""]
[["--dsa-paged-mqa-logits-backend"],"scalar","str","",["auto","deepgemm","cutedsl","aiter"],"auto","dsa_paged_mqa_logits_backend",0,""]
[["--dsa-topk-backend"],"scalar","str","",["sgl-kernel","torch","flashinfer"],"sgl-kernel","dsa_topk_backend",0,""]
[["--disable-flashinfer-autotune"],"store_true","","",null,false,"disable_flashinfer_autotune",0,""]
[["--flashinfer-autotune-skip-ops"],"list","str","+",null,null,"flashinfer_autotune_skip_ops",0,""]
[["--triton-attention-reduce-in-fp32"],"store_true","","",null,false,"triton_attention_reduce_in_fp32",0,""]
[["--triton-attention-num-kv-splits"],"scalar","int","",null,8,"triton_attention_num_kv_splits",0,""]
[["--triton-attention-split-tile-size"],"scalar","int","",null,null,"triton_attention_split_tile_size",0,""]
[["--flashinfer-mla-disable-ragged"],"store_true","","",null,false,"flashinfer_mla_disable_ragged",0,""]
[["--enable-fused-qk-norm-rope"],"store_true","","",null,false,"enable_fused_qk_norm_rope",0,""]
[["--enable-precise-embedding-interpolation"],"store_true","","",null,false,"enable_precise_embedding_interpolation",0,""]
[["--enable-deepseek-v4-fp4-indexer"],"store_true","","",null,false,"enable_deepseek_v4_fp4_indexer",0,""]
[["--mamba-backend"],"scalar","str","",["triton","flashinfer"],"triton","mamba_backend",0,""]
[["--mamba-ssm-dtype"],"scalar","str","",["float32","bfloat16","float16"],null,"mamba_ssm_dtype",0,""]
[["--mamba-max-states-per-path"],"scalar","int","",null,-1,"mamba_max_states_per_path",0,""]
[["--enable-mamba-cache-stochastic-rounding"],"store_true","","",null,false,"enable_mamba_cache_stochastic_rounding",0,""]
[["--mamba-cache-philox-rounds"],"scalar","int","",null,0,"mamba_cache_philox_rounds",0,""]
[["--mamba-radix-cache-strategy"],"scalar","str","",["auto","no_buffer","extra_buffer","extra_buffer_lazy"],"auto","mamba_radix_cache_strategy",0,""]
[["--mamba-track-interval"],"scalar","int","",null,256,"mamba_track_interval",0,""]
[["--enable-int8-mamba-checkpoint"],"store_true","","",null,false,"enable_int8_mamba_checkpoint",0,""]
[["--int8-mamba-ckpt-size"],"scalar","int","",null,null,"int8_mamba_ckpt_size",0,""]
[["--linear-attn-backend"],"scalar","str","","@linear_attn_backend","triton","linear_attn_backend",0,""]
[["--linear-attn-decode-backend"],"scalar","str","","@linear_attn_backend",null,"linear_attn_decode_backend",0,""]
[["--linear-attn-prefill-backend"],"scalar","str","","@linear_attn_backend",null,"linear_attn_prefill_backend",0,""]
[["--linear-attn-verify-backend"],"scalar","str","","@linear_attn_verify_backend",null,"linear_attn_verify_backend",0,""]
[["--enable-linear-replayssm"],"store_true","","",null,false,"enable_linear_replayssm",0,""]
[["--linear-replayssm-cache-len"],"scalar","int","",null,16,"linear_replayssm_cache_len",0,""]
[["--enable-linear-replayssm-spec"],"store_true","","",null,false,"enable_linear_replayssm_spec",0,""]
[["--cuda-graph-config"],"scalar","parse_cuda_graph_config_arg","",null,null,"cuda_graph_config",0,""]
[["--cuda-graph-backend-decode"],"scalar","str","",["full","breakable","tc_piecewise","disabled"],null,"cuda_graph_backend_decode",0,""]
[["--cuda-graph-backend-prefill"],"scalar","str","",["full","breakable","tc_piecewise","disabled"],null,"cuda_graph_backend_prefill",0,""]
[["--cuda-graph-max-bs-decode"],"scalar","int","",null,null,"cuda_graph_max_bs_decode",0,""]
[["--cuda-graph-max-bs-prefill"],"scalar","int","",null,null,"cuda_graph_max_bs_prefill",0,""]
[["--cuda-graph-max-seq-len-prefill"],"scalar","int","",null,null,"cuda_graph_max_seq_len_prefill",0,""]
[["--cuda-graph-bs-decode"],"list","int","+",null,null,"cuda_graph_bs_decode",0,""]
[["--cuda-graph-bs-prefill"],"list","int","+",null,null,"cuda_graph_bs_prefill",0,""]
[["--cuda-graph-prefill-max-context","--context-bucket"],"scalar","human_readable_int","",null,null,"cuda_graph_prefill_max_context",0,""]
[["--cuda-graph-tc-compiler"],"scalar","str","",["eager","inductor"],null,"cuda_graph_tc_compiler",0,""]
[["--disable-prefill-cuda-graph"],"store_true","","",null,false,"disable_prefill_cuda_graph",0,""]
[["--disable-decode-cuda-graph"],"store_true","","",null,false,"disable_decode_cuda_graph",0,""]
[["--disable-cuda-graph-padding"],"store_true","","",null,false,"disable_cuda_graph_padding",0,""]
[["--enable-profile-cuda-graph"],"store_true","","",null,false,"enable_profile_cuda_graph",0,""]
[["--enable-cudagraph-gc"],"store_true","","",null,false,"enable_cudagraph_gc",0,""]
[["--debug-cuda-graph"],"store_true","","",null,false,"debug_cuda_graph",0,""]
[["--enable-torch-compile"],"store_true","","",null,false,"enable_torch_compile",0,""]
[["--enable-torch-compile-debug-mode"],"store_true","","",null,false,"enable_torch_compile_debug_mode",0,""]
[["--torch-compile-max-bs"],"scalar","int","",null,32,"torch_compile_max_bs",0,""]
[["--enable-layerwise-nvtx-marker"],"store_true","","",null,false,"enable_layerwise_nvtx_marker",0,""]
[["--enable-nccl-nvls"],"store_true","","",null,false,"enable_nccl_nvls",0,""]
[["--enable-symm-mem"],"store_true","","",null,false,"enable_symm_mem",0,""]
[["--disable-custom-all-reduce"],"store_true","","",null,false,"disable_custom_all_reduce",0,""]
[["--enable-mscclpp"],"store_true","","",null,false,"enable_mscclpp",0,""]
[["--enable-torch-symm-mem"],"store_true","","",null,false,"enable_torch_symm_mem",0,""]
[["--enable-scattered-sconv"],"store_true","","",null,false,"enable_scattered_sconv",0,""]
[["--pre-warm-nccl"],"store_true","","",null,false,"pre_warm_nccl",0,""]
[["--boundary-reduction"],"scalar","str","",["auto","ar","rs","rsv","rs+rsv"],"auto","boundary_reduction",0,""]
[["--enable-quant-communications"],"store_true","","",null,false,"enable_quant_communications",0,""]
[["--enforce-disable-flashinfer-allreduce-fusion"],"store_true","","",null,false,"enforce_disable_flashinfer_allreduce_fusion",0,""]
[["--flashinfer-allreduce-fusion-backend"],"scalar","str","",["auto","trtllm","mnnvl","cutedsl"],null,"flashinfer_allreduce_fusion_backend",0,""]
[["--enable-aiter-allreduce-fusion"],"store_true","","",null,false,"enable_aiter_allreduce_fusion",0,""]
[["--disable-aiter-allreduce-fusion-in-prefill"],"store_true","","",null,false,"disable_aiter_allreduce_fusion_in_prefill",0,""]
[["--disable-aiter-allreduce-fusion-in-decode"],"store_true","","",null,false,"disable_aiter_allreduce_fusion_in_decode",0,""]
[["--enable-fused-moe-sum-all-reduce"],"store_true","","",null,false,"enable_fused_moe_sum_all_reduce",0,""]
[["--moe-a2a-backend"],"scalar","str","","@moe_a2a_backend","none","moe_a2a_backend",0,""]
[["--enable-w4a4-mxfp4-megamoe"],"store_true","","",null,false,"enable_w4a4_mxfp4_megamoe",0,""]
[["--deepep-v2-mode"],"scalar","str","",["direct","hybrid"],"direct","deepep_v2_mode",0,""]
[["--moe-runner-backend"],"scalar","str","","@moe_runner_backend","auto","moe_runner_backend",0,""]
[["--flashinfer-mxfp4-moe-precision"],"scalar","str","",["default","bf16","fp8"],"default","flashinfer_mxfp4_moe_precision",0,""]
[["--deepep-mode"],"scalar","str","",["auto","normal","low_latency"],"auto","deepep_mode",0,""]
[["--fuseep-mode"],"scalar","int","",[1,2],2,"fuseep_mode",0,""]
[["--deepep-dispatcher-output-dtype"],"scalar","str","",["auto","bf16","fp8","int8","nvfp4"],"auto","deepep_dispatcher_output_dtype",0,""]
[["--flashinfer-a2a-dispatch-type"],"scalar","str","",["auto","bf16","nvfp4","mxfp8"],null,"flashinfer_a2a_dispatch_type",0,""]
[["--ep-num-redundant-experts"],"scalar","int","",null,0,"ep_num_redundant_experts",0,""]
[["--ep-dispatch-algorithm"],"scalar","str","",["static","dynamic","fake","lp"],null,"ep_dispatch_algorithm",0,""]
[["--init-expert-location"],"scalar","str","",null,"trivial","init_expert_location",0,""]
[["--enable-eplb"],"store_true","","",null,false,"enable_eplb",0,""]
[["--eplb-algorithm"],"scalar","str","",null,"auto","eplb_algorithm",0,""]
[["--eplb-rebalance-num-iterations"],"scalar","int","",null,1000,"eplb_rebalance_num_iterations",0,""]
[["--eplb-rebalance-layers-per-chunk"],"scalar","int","",null,null,"eplb_rebalance_layers_per_chunk",0,""]
[["--eplb-min-rebalancing-utilization-threshold"],"scalar","float","",null,1.0,"eplb_min_rebalancing_utilization_threshold",0,""]
[["--expert-distribution-recorder-mode"],"scalar","str","",["stat","stat_approx","per_pass","per_token"],null,"expert_distribution_recorder_mode",0,""]
[["--expert-distribution-recorder-buffer-size"],"scalar","int","",null,null,"expert_distribution_recorder_buffer_size",0,""]
[["--expert-balancedness-report-mode"],"scalar","str","",["off","server_log","prometheus","both"],"off","expert_balancedness_report_mode",0,""]
[["--deepep-config"],"scalar","str","",null,null,"deepep_config",0,""]
[["--elastic-ep-backend"],"scalar","str","",["none","mooncake","nixl"],null,"elastic_ep_backend",0,""]
[["--enable-elastic-expert-backup"],"store_true","","",null,false,"enable_elastic_expert_backup",0,""]
[["--mooncake-ib-device"],"scalar","str","",null,null,"mooncake_ib_device",0,""]
[["--enable-waterfill"],"store_true","","",null,false,"enable_waterfill",0,""]
[["--elastic-ep-join-mode"],"scalar","str","",["scale","recover"],null,"ep_join_mode",0,""]
[["--elastic-ep-scale-timeout"],"scalar","float","",null,600,"elastic_ep_scale_timeout",0,""]
[["--disable-flashinfer-cutlass-moe-fp4-allgather"],"store_true","","",null,false,"disable_flashinfer_cutlass_moe_fp4_allgather",0,""]
[["--disable-shared-experts-fusion"],"store_true","","",null,false,"disable_shared_experts_fusion",0,""]
[["--enforce-shared-experts-fusion"],"store_true","","",null,false,"enforce_shared_experts_fusion",0,""]
[["--kt-weight-path"],"scalar","str","",null,null,"kt_weight_path",0,""]
[["--kt-method"],"scalar","str","",null,"AMXINT4","kt_method",0,""]
[["--kt-cpuinfer"],"scalar","int","",null,null,"kt_cpuinfer",0,""]
[["--kt-threadpool-count"],"scalar","int","",null,2,"kt_threadpool_count",0,""]
[["--kt-num-gpu-experts"],"scalar","int","",null,null,"kt_num_gpu_experts",0,""]
[["--kt-max-deferred-experts-per-token"],"scalar","int","",null,null,"kt_max_deferred_experts_per_token",0,""]
[["--enable-two-batch-overlap"],"store_true","","",null,false,"enable_two_batch_overlap",0,""]
[["--enable-single-batch-overlap"],"store_true","","",null,false,"enable_single_batch_overlap",0,""]
[["--tbo-token-distribution-threshold"],"scalar","float","",null,0.48,"tbo_token_distribution_threshold",0,""]
[["--cpu-offload-gb"],"scalar","int","",null,0,"cpu_offload_gb",0,""]
[["--offload-group-size"],"scalar","int","",null,-1,"offload_group_size",0,""]
[["--offload-num-in-group"],"scalar","int","",null,1,"offload_num_in_group",0,""]
[["--offload-prefetch-step"],"scalar","int","",null,1,"offload_prefetch_step",0,""]
[["--offload-mode"],"scalar","str","",null,"cpu","offload_mode",0,""]
[["--ple-offload-embedding"],"bool_opt","","",null,null,"ple_offload_embedding",0,""]
[["--ple-offload-backend"],"scalar","str","",["pinned","file"],"pinned","ple_offload_backend",0,""]
[["--ple-offload-dir"],"scalar","str","",null,null,"ple_offload_dir",0,""]
[["--dllm-algorithm"],"scalar","str","",null,null,"dllm_algorithm",0,""]
[["--dllm-algorithm-config"],"scalar","str","",null,null,"dllm_algorithm_config",0,""]
[["--dllm-fdfo"],"bool_opt","","",null,true,"dllm_fdfo",0,""]
[["--enable-deterministic-inference"],"store_true","","",null,false,"enable_deterministic_inference",0,""]
[["--rl-on-policy-target"],"scalar","str","",["fsdp"],null,"rl_on_policy_target",0,""]
[["--enable-lora"],"store_true","","",null,null,"enable_lora",0,""]
[["--enable-lora-overlap-loading"],"store_true","","",null,null,"enable_lora_overlap_loading",0,""]
[["--max-lora-rank"],"scalar","int","",null,null,"max_lora_rank",0,""]
[["--lora-target-modules"],"scalar","str","*",null,null,"lora_target_modules",0,""]
[["--lora-paths"],"lora","","",null,null,"lora_paths",0,""]
[["--max-loaded-loras"],"scalar","int","",null,null,"max_loaded_loras",0,""]
[["--max-loras-per-batch"],"scalar","int","",null,8,"max_loras_per_batch",0,""]
[["--lora-eviction-policy"],"scalar","str","",["lru","fifo"],"lru","lora_eviction_policy",0,""]
[["--lora-backend"],"scalar","str","",["triton","csgmv","ascend","torch_native"],"csgmv","lora_backend",0,""]
[["--max-lora-chunk-size"],"scalar","int","",[16,32,64,128],16,"max_lora_chunk_size",0,""]
[["--experts-shared-outer-loras"],"bool_opt","","",null,null,"experts_shared_outer_loras",0,""]
[["--lora-use-virtual-experts"],"store_true","","",null,false,"lora_use_virtual_experts",0,""]
[["--lora-strict-loading"],"bool_opt","","",null,false,"lora_strict_loading",0,""]
[["--lora-drain-wait-threshold"],"scalar","float","",null,0.0,"lora_drain_wait_threshold",0,""]
[["--radix-eviction-policy"],"scalar","str","",["lru","lfu","slru","priority","tlru"],"lru","radix_eviction_policy",0,""]
[["--radix-eviction-policy-config"],"scalar","json.loads","",null,null,"radix_eviction_policy_config",0,""]
[["--disable-radix-cache"],"store_true","","",null,false,"disable_radix_cache",0,""]
[["--enable-page-major-kv-layout"],"store_true","","",null,false,"enable_page_major_kv_layout",0,""]
[["--enable-unified-memory"],"store_true","","",null,false,"enable_unified_memory",0,""]
[["--enable-session-radix-cache"],"store_true","","",null,false,"enable_session_radix_cache",0,""]
[["--radix-cache-backend"],"scalar","str","",null,null,"radix_cache_backend",0,""]
[["--enable-hierarchical-cache"],"store_true","","",null,false,"enable_hierarchical_cache",0,""]
[["--hicache-host-memory-mode"],"scalar","str","",["cache","buffer_only"],"cache","hicache_host_memory_mode",0,""]
[["--hicache-ratio"],"scalar","float","",null,null,"hicache_ratio",0,""]
[["--hicache-size"],"scalar","int","",null,0,"hicache_size",0,""]
[["--hicache-host-memory-fraction"],"scalar","float","",null,0.8,"hicache_host_memory_fraction",0,""]
[["--hicache-write-policy"],"scalar","str","",["write_back","write_through","write_through_selective"],"write_through","hicache_write_policy",0,""]
[["--hicache-io-backend"],"scalar","str","",["direct","kernel","kernel_ascend"],"kernel","hicache_io_backend",0,""]
[["--hicache-mem-layout"],"scalar","str","","@hicache_mem_layout","page_first","hicache_mem_layout",0,""]
[["--hicache-storage-backend"],"scalar","str","","@hicache_storage_backend",null,"hicache_storage_backend",0,""]
[["--hicache-storage-prefetch-policy"],"scalar","str","",["best_effort","wait_complete","timeout"],"timeout","hicache_storage_prefetch_policy",0,""]
[["--hicache-storage-backend-extra-config"],"scalar","str","",null,null,"hicache_storage_backend_extra_config",0,""]
[["--hicache-storage-prefetch-retry-poll-interval"],"scalar","int","",null,8,"hicache_storage_prefetch_retry_poll_interval",0,""]
[["--hicache-storage-prefetch-retry-max-attempts"],"scalar","int","",null,8,"hicache_storage_prefetch_retry_max_attempts",0,""]
[["--enable-unified-cache-external-linker"],"store_true","","",null,false,"enable_unified_cache_external_linker",0,""]
[["--unified-cache-external-linker-backend"],"scalar","str","",["mooncake","mori"],"mooncake","unified_cache_external_linker_backend",0,""]
[["--enable-linker-mla-dedup"],"store_true","","",null,false,"enable_linker_mla_dedup",0,""]
[["--enable-hisparse"],"store_true","","",null,false,"enable_hisparse",0,""]
[["--hisparse-config","--hierarchical-sparse-attention-extra-config"],"scalar","str","",null,null,"hisparse_config",0,""]
[["--enable-lmcache"],"store_true","","",null,false,"enable_lmcache",0,""]
[["--lmcache-config-file"],"scalar","str","",null,null,"lmcache_config_file",0,""]
[["--enable-flexkv"],"store_true","","",null,false,"enable_flexkv",0,""]
[["--flexkv-config-file"],"scalar","str","",null,null,"flexkv_config_file",0,""]
[["--enable-multimodal"],"store_true","","",null,null,"enable_multimodal",0,""]
[["--mm-attention-backend"],"scalar","str","","@mm_attention_backend",null,"mm_attention_backend",0,""]
[["--enable-broadcast-mm-inputs-process"],"store_true","","",null,false,"enable_broadcast_mm_inputs_process",0,""]
[["--enable-prefix-mm-cache"],"store_true","","",null,false,"enable_prefix_mm_cache",0,""]
[["--mm-enable-dp-encoder"],"store_true","","",null,false,"mm_enable_dp_encoder",0,""]
[["--mm-process-config"],"scalar","json.loads","",null,null,"mm_process_config",0,""]
[["--mm-processor-worker-num"],"scalar","int","",null,0,"mm_processor_worker_num",0,""]
[["--mm-io-worker-num"],"scalar","int","",null,0,"mm_io_worker_num",0,""]
[["--allowed-media-domains"],"list","str","+",null,[],"allowed_media_domains",0,""]
[["--media-url-max-file-size-mb"],"scalar","int","",null,64,"media_url_max_file_size_mb",0,""]
[["--mm-preprocess-cache-size-mb"],"scalar","int","",null,null,"mm_preprocess_cache_size_mb",0,""]
[["--trust-mm-content-hashes"],"store_true","","",null,false,"trust_mm_content_hashes",0,""]
[["--limit-mm-data-per-request"],"scalar","json.loads","",null,null,"limit_mm_data_per_request",0,""]
[["--enable-mm-global-cache"],"store_true","","",null,false,"enable_mm_global_cache",0,""]
[["--image-processor-backend"],"scalar","str","",["auto","torchvision","pil"],"auto","image_processor_backend",0,""]
[["--mm-global-cache-backend"],"scalar","str","",["mooncake"],"mooncake","mm_global_cache_backend",0,""]
[["--disable-fast-image-processor"],"store_true","","",null,false,"disable_fast_image_processor",0,""]
[["--mm-feature-transport"],"scalar","str","",["cpu","cuda_ipc","cuda_vmm"],null,"mm_feature_transport",0,""]
[["--model-path","--model"],"scalar","str","",null,"@MISSING","model_path",1,""]
[["--load-format"],"scalar","str","","@load_format","auto","load_format",0,""]
[["--model-loader-extra-config"],"scalar","str","",null,"{}","model_loader_extra_config",0,""]
[["--trust-remote-code"],"store_true","","",null,false,"trust_remote_code",0,""]
[["--context-length"],"scalar","human_readable_int","",null,null,"context_length",0,""]
[["--is-embedding"],"store_true","","",null,false,"is_embedding",0,""]
[["--revision"],"scalar","str","",null,null,"revision",0,""]
[["--model-impl"],"scalar","str","",null,"auto","model_impl",0,""]
[["--model-config-parser"],"scalar","str","",null,"auto","model_config_parser",0,""]
[["--json-model-override-args"],"scalar","str","",null,"{}","json_model_override_args",0,""]
[["--dtype"],"scalar","str","",["auto","half","float16","bfloat16","float","float32"],"auto","dtype",0,""]
[["--quantization"],"scalar","str","","@quantization",null,"quantization",0,""]
[["--quantization-param-path"],"scalar","nullable_str","",null,null,"quantization_param_path",0,""]
[["--kv-cache-dtype"],"scalar","str","","@kv_cache_dtype","auto","kv_cache_dtype",0,""]
[["--modelopt-quant"],"scalar","str","",null,null,"modelopt_quant",0,""]
[["--modelopt-checkpoint-restore-path"],"scalar","str","",null,null,"modelopt_checkpoint_restore_path",0,""]
[["--modelopt-checkpoint-save-path"],"scalar","str","",null,null,"modelopt_checkpoint_save_path",0,""]
[["--modelopt-export-path"],"scalar","str","",null,null,"modelopt_export_path",0,""]
[["--quantize-and-serve"],"store_true","","",null,false,"quantize_and_serve",0,""]
[["--rl-quant-profile"],"scalar","str","",null,null,"rl_quant_profile",0,""]
[["--startup-weight-load-mode"],"scalar","str","",["serial","overlap"],"serial","startup_weight_load_mode",0,""]
[["--custom-weight-loader"],"list","str","*",null,null,"custom_weight_loader",0,""]
[["--weight-loader-disable-mmap"],"store_true","","",null,false,"weight_loader_disable_mmap",0,""]
[["--weight-loader-prefetch-checkpoints"],"store_true","","",null,false,"weight_loader_prefetch_checkpoints",0,""]
[["--weight-loader-prefetch-num-threads"],"scalar","int","",null,4,"weight_loader_prefetch_num_threads",0,""]
[["--weight-loader-drop-cache-after-load"],"store_true","","",null,false,"weight_loader_drop_cache_after_load",0,""]
[["--remote-instance-weight-loader-seed-instance-ip"],"scalar","str","",null,null,"remote_instance_weight_loader_seed_instance_ip",0,""]
[["--remote-instance-weight-loader-seed-instance-service-port"],"scalar","int","",null,null,"remote_instance_weight_loader_seed_instance_service_port",0,""]
[["--remote-instance-weight-loader-send-weights-group-ports"],"scalar","json_list_type","",null,null,"remote_instance_weight_loader_send_weights_group_ports",0,""]
[["--remote-instance-weight-loader-backend"],"scalar","str","",["transfer_engine","nccl","modelexpress"],"nccl","remote_instance_weight_loader_backend",0,""]
[["--remote-instance-weight-loader-start-seed-via-transfer-engine"],"store_true","","",null,false,"remote_instance_weight_loader_start_seed_via_transfer_engine",0,""]
[["--engine-info-bootstrap-port"],"scalar","int","",null,6789,"engine_info_bootstrap_port",0,""]
[["--modelexpress-config"],"scalar","str","",null,null,"modelexpress_config",0,""]
[["--download-dir"],"scalar","str","",null,null,"download_dir",0,""]
[["--model-checksum"],"scalar","str","?",null,null,"model_checksum",0,""]
[["--delete-ckpt-after-loading"],"store_true","","",null,false,"delete_ckpt_after_loading",0,""]
[["--decrypted-config-file"],"scalar","str","",null,null,"decrypted_config_file",0,""]
[["--decrypted-draft-config-file"],"scalar","str","",null,null,"decrypted_draft_config_file",0,""]
[["--checkpoint-engine-wait-weights-before-ready"],"store_true","","",null,false,"checkpoint_engine_wait_weights_before_ready",0,""]
[["--weight-cache-mode"],"scalar","str","",["off","daemon","client"],"off","weight_cache_mode",0,""]
[["--weight-cache-socket"],"scalar","str","",null,null,"weight_cache_socket",0,""]
[["--weight-cache-timeout"],"scalar","int","",null,1800,"weight_cache_timeout",0,""]
[["--log-level"],"scalar","str","",null,"info","log_level",0,""]
[["--log-level-http"],"scalar","str","",null,null,"log_level_http",0,""]
[["--log-requests"],"store_true","","",null,false,"log_requests",0,""]
[["--log-requests-level"],"scalar","int","",[0,1,2,3],2,"log_requests_level",0,""]
[["--log-requests-format"],"scalar","str","",["text","json"],"text","log_requests_format",0,""]
[["--log-requests-target"],"list","str","+",null,null,"log_requests_target",0,""]
[["--uvicorn-access-log-exclude-prefixes"],"list","str","*",null,[],"uvicorn_access_log_exclude_prefixes",0,""]
[["--crash-dump-folder"],"scalar","str","",null,null,"crash_dump_folder",0,""]
[["--show-time-cost"],"store_true","","",null,false,"show_time_cost",0,""]
[["--enable-metrics"],"store_true","","",null,false,"enable_metrics",0,""]
[["--smg-http-sidecar-port","--grpc-http-sidecar-port"],"scalar","int","",null,null,"smg_http_sidecar_port",0,""]
[["--enable-mfu-metrics"],"store_true","","",null,false,"enable_mfu_metrics",0,""]
[["--enable-metrics-for-all-schedulers"],"store_true","","",null,false,"enable_metrics_for_all_schedulers",0,""]
[["--load-snapshot-publish-interval"],"scalar","int","",null,15,"load_snapshot_publish_interval",0,""]
[["--tokenizer-metrics-custom-labels-header"],"scalar","str","",null,"x-custom-labels","tokenizer_metrics_custom_labels_header",0,""]
[["--tokenizer-metrics-allowed-custom-labels"],"list","str","+",null,null,"tokenizer_metrics_allowed_custom_labels",0,""]
[["--extra-metric-labels"],"scalar","json.loads","",null,null,"extra_metric_labels",0,""]
[["--bucket-time-to-first-token"],"list","float","+",null,null,"bucket_time_to_first_token",0,""]
[["--bucket-inter-token-latency"],"list","float","+",null,null,"bucket_inter_token_latency",0,""]
[["--bucket-e2e-request-latency"],"list","float","+",null,null,"bucket_e2e_request_latency",0,""]
[["--prompt-tokens-buckets"],"list","str","+",null,null,"prompt_tokens_buckets",0,""]
[["--generation-tokens-buckets"],"list","str","+",null,null,"generation_tokens_buckets",0,""]
[["--gc-warning-threshold-secs"],"scalar","float","",null,0.0,"gc_warning_threshold_secs",0,""]
[["--decode-log-interval"],"scalar","int","",null,40,"decode_log_interval",0,""]
[["--enable-request-time-stats-logging"],"store_true","","",null,false,"enable_request_time_stats_logging",0,""]
[["--kv-events-config"],"scalar","str","",null,null,"kv_events_config",0,""]
[["--load-publish-endpoint"],"scalar","str","",null,null,"load_publish_endpoint",0,""]
[["--enable-forward-pass-metrics"],"store_true","","",null,false,"enable_forward_pass_metrics",0,""]
[["--forward-pass-metrics-worker-id"],"scalar","str","",null,"","forward_pass_metrics_worker_id",0,""]
[["--forward-pass-metrics-ipc-name"],"scalar","str","",null,null,"forward_pass_metrics_ipc_name",0,""]
[["--enable-trace"],"store_true","","",null,false,"enable_trace",0,""]
[["--trace-modules"],"scalar","str","",null,"request","trace_modules",0,""]
[["--otlp-traces-endpoint"],"scalar","str","",null,"localhost:4317","otlp_traces_endpoint",0,""]
[["--otlp-service-name"],"scalar","str","",null,null,"otlp_service_name",0,""]
[["--export-metrics-to-file"],"store_true","","",null,false,"export_metrics_to_file",0,""]
[["--export-metrics-to-file-dir"],"scalar","str","",null,null,"export_metrics_to_file_dir",0,""]
[["--kv-canary"],"scalar","str","",["none","log","raise"],"none","kv_canary",0,""]
[["--kv-canary-sweep-interval"],"scalar","int","",null,0,"kv_canary_sweep_interval",0,""]
[["--debug-tensor-dump-output-folder"],"scalar","str","",null,null,"debug_tensor_dump_output_folder",0,""]
[["--debug-tensor-dump-layers"],"list","int","+",null,null,"debug_tensor_dump_layers",0,""]
[["--debug-tensor-dump-input-file"],"scalar","str","",null,null,"debug_tensor_dump_input_file",0,""]
[["--forward-hooks"],"scalar","json_list_type","",null,null,"forward_hooks",0,""]
[["--msprobe-dump-config"],"scalar","str","",null,null,"msprobe_dump_config",0,""]
[["--nccl-port"],"scalar","int","",null,null,"nccl_port",0,""]
[["--dist-timeout"],"scalar","int","",null,null,"dist_timeout",0,""]
[["--dist-init-addr","--nccl-init-addr"],"scalar","str","",null,null,"dist_init_addr",0,""]
[["--gated-launch-port"],"scalar","int","",null,null,"gated_launch_port",0,""]
[["--nnodes"],"scalar","int","",null,1,"nnodes",0,""]
[["--node-rank"],"scalar","int","",null,0,"node_rank",0,""]
[["--tp-size","--tensor-parallel-size"],"scalar","int","",null,1,"tp_size",0,""]
[["--dcp-size","--decode-context-parallel-size"],"scalar","int","",null,1,"dcp_size",0,""]
[["--pp-size","--pipeline-parallel-size"],"scalar","int","",null,1,"pp_size",0,""]
[["--pp-max-micro-batch-size"],"scalar","int","",null,null,"pp_max_micro_batch_size",0,""]
[["--pp-async-batch-depth"],"scalar","int","",null,0,"pp_async_batch_depth",0,""]
[["--dp-size","--data-parallel-size"],"scalar","int","",null,1,"dp_size",0,""]
[["--load-balance-method"],"scalar","str","","@load_balance_method","auto","load_balance_method",0,""]
[["--attn-cp-size","--attention-context-parallel-size"],"scalar","int","",null,1,"attn_cp_size",0,""]
[["--moe-dp-size","--moe-data-parallel-size"],"scalar","int","",null,1,"moe_dp_size",0,""]
[["--dwdp-size"],"scalar","int","",null,1,"dwdp_size",0,""]
[["--dcp-comm-backend"],"scalar","str","",["ag_rs","a2a","fi_a2a"],null,"dcp_comm_backend",0,""]
[["--dcp-replicate-q-proj"],"bool_opt","","",null,null,"dcp_replicate_q_proj",0,""]
[["--enable-prefill-cp"],"store_true","","",null,false,"enable_prefill_cp",0,""]
[["--cp-strategy"],"scalar","str","",["zigzag","interleave"],null,"cp_strategy",0,""]
[["--enable-dsa-cache-layer-split"],"store_true","","",null,false,"enable_dsa_cache_layer_split",0,""]
[["--enable-cp-decode-attn-tp"],"store_true","","",null,false,"enable_cp_decode_attn_tp",0,""]
[["--enable-dp-attention"],"store_true","","",null,false,"enable_dp_attention",0,""]
[["--enable-dp-attention-local-control-broadcast"],"store_true","","",null,false,"enable_dp_attention_local_control_broadcast",0,""]
[["--enable-dp-lm-head"],"store_true","","",null,false,"enable_dp_lm_head",0,""]
[["--enable-tp-lm-head-all-to-all"],"bool_opt","","",null,null,"enable_tp_lm_head_all_to_all",0,""]
[["--enable-attn-tp-input-scattered"],"store_true","","",null,false,"enable_attn_tp_input_scattered",0,""]
[["--enable-shared-experts-attn-tp"],"store_true","","",null,false,"enable_shared_experts_attn_tp",0,""]
[["--shared-experts-tp-size"],"scalar","int","",null,null,"shared_experts_tp_size",0,""]
[["--enable-dense-mlp-attn-tp"],"store_true","","",null,false,"enable_dense_mlp_attn_tp",0,""]
[["--enable-layernorm-sp"],"store_true","","",null,false,"enable_layernorm_sp",0,""]
[["--disable-attn-tp-gather"],"store_true","","",null,false,"disable_attn_tp_gather",0,""]
[["--enable-p2p-check"],"store_true","","",null,false,"enable_p2p_check",0,""]
[["--ep-size","--expert-parallel-size","--ep"],"scalar","int","",null,1,"ep_size",0,""]
[["--moe-dense-tp-size"],"scalar","int","",null,null,"moe_dense_tp_size",0,""]
[["--elastic-ep-join-rank-offset"],"scalar","int","",null,0,"ep_join_rank_offset",0,""]
[["--elastic-ep-initial-size"],"scalar","int","",null,null,"elastic_ep_initial_size",0,""]
[["--max-ep-size"],"scalar","int","",null,null,"max_ep_size",0,""]
[["--mem-fraction-static"],"scalar","float","",null,null,"mem_fraction_static",0,""]
[["--max-running-requests"],"scalar","int","",null,null,"max_running_requests",0,""]
[["--max-queued-requests"],"scalar","int","",null,null,"max_queued_requests",0,""]
[["--max-total-tokens"],"scalar","human_readable_int","",null,null,"max_total_tokens",0,""]
[["--chunked-prefill-size"],"scalar","int","",null,null,"chunked_prefill_size",0,""]
[["--prefill-decode-interval"],"scalar","int","",null,null,"prefill_decode_interval",0,""]
[["--enable-dynamic-chunking"],"store_true","","",null,false,"enable_dynamic_chunking",0,""]
[["--max-prefill-tokens"],"scalar","human_readable_int","",null,16384,"max_prefill_tokens",0,""]
[["--prefill-max-requests"],"scalar","int","",null,null,"prefill_max_requests",0,""]
[["--schedule-policy"],"scalar","str","","@schedule_policy","fcfs","schedule_policy",0,""]
[["--enable-priority-scheduling"],"store_true","","",null,false,"enable_priority_scheduling",0,""]
[["--disable-priority-preemption"],"store_true","","",null,false,"disable_priority_preemption",0,""]
[["--default-priority-value"],"scalar","int","",null,null,"default_priority_value",0,""]
[["--abort-on-priority-when-disabled"],"store_true","","",null,false,"abort_on_priority_when_disabled",0,""]
[["--schedule-low-priority-values-first"],"store_true","","",null,false,"schedule_low_priority_values_first",0,""]
[["--priority-scheduling-preemption-threshold"],"scalar","int","",null,10,"priority_scheduling_preemption_threshold",0,""]
[["--retraction-policy"],"scalar","str","",["length","priority"],"length","retraction_policy",0,""]
[["--schedule-conservativeness"],"scalar","float","",null,1.0,"schedule_conservativeness",0,""]
[["--page-size"],"scalar","int","",null,null,"page_size",0,""]
[["--c128-page-size"],"scalar","int","",null,16,"c128_page_size",0,""]
[["--swa-full-tokens-ratio"],"scalar","float","",null,null,"swa_full_tokens_ratio",0,""]
[["--swa-prefix-tails"],"scalar","int","",null,null,"swa_prefix_tails",0,""]
[["--disable-hybrid-swa-memory"],"store_true","","",null,false,"disable_hybrid_swa_memory",0,""]
[["--prefill-only-disable-kv-cache"],"store_true","","",null,false,"prefill_only_disable_kv_cache",0,""]
[["--disable-chunked-prefix-cache"],"store_true","","",null,false,"disable_chunked_prefix_cache",0,""]
[["--disable-overlap-schedule"],"store_true","","",null,false,"disable_overlap_schedule",0,""]
[["--num-continuous-decode-steps"],"scalar","int","",null,1,"num_continuous_decode_steps",0,""]
[["--scheduler-recv-interval"],"scalar","int","",null,1,"scheduler_recv_interval",0,""]
[["--enable-mixed-chunk"],"store_true","","",null,false,"enable_mixed_chunk",0,""]
[["--max-mamba-cache-size"],"scalar","int","",null,null,"max_mamba_cache_size",0,""]
[["--mamba-full-memory-ratio"],"scalar","float","",null,null,"mamba_full_memory_ratio",0,""]
[["--enable-prefill-delayer"],"store_true","","",null,false,"enable_prefill_delayer",0,""]
[["--prefill-delayer-max-delay-passes"],"scalar","int","",null,30,"prefill_delayer_max_delay_passes",0,""]
[["--prefill-delayer-token-usage-low-watermark"],"scalar","float","",null,null,"prefill_delayer_token_usage_low_watermark",0,""]
[["--prefill-delayer-forward-passes-buckets"],"list","float","+",null,null,"prefill_delayer_forward_passes_buckets",0,""]
[["--prefill-delayer-wait-seconds-buckets"],"list","float","+",null,null,"prefill_delayer_wait_seconds_buckets",0,""]
[["--prefill-delayer-queue-min-ratio"],"scalar","float","",null,null,"prefill_delayer_queue_min_ratio",0,""]
[["--prefill-delayer-max-delay-ms"],"scalar","float","",null,null,"prefill_delayer_max_delay_ms",0,""]
[["--min-free-slots-delay"],"scalar","int","",null,null,"min_free_slots_delay",0,""]
[["--enable-response-store"],"store_true","","",null,false,"enable_response_store",0,""]
[["--tokenizer-path"],"scalar","str","",null,null,"tokenizer_path",0,""]
[["--tokenizer-mode"],"scalar","str","",["auto","slow"],"auto","tokenizer_mode",0,""]
[["--tokenizer-backend"],"scalar","str","",["huggingface","fastokens"],"huggingface","tokenizer_backend",0,""]
[["--tokenizer-worker-num"],"scalar","int","",null,1,"tokenizer_worker_num",0,""]
[["--detokenizer-worker-num"],"scalar","int","",null,1,"detokenizer_worker_num",0,""]
[["--skip-tokenizer-init"],"store_true","","",null,false,"skip_tokenizer_init",0,""]
[["--host"],"scalar","str","",null,"127.0.0.1","host",0,""]
[["--port"],"scalar","int","",null,30000,"port",0,""]
[["--fastapi-root-path"],"scalar","str","",null,"","fastapi_root_path",0,""]
[["--smg-grpc-mode"],"store_true","","",null,false,"smg_grpc_mode",0,""]
[["--grpc-mode"],"store_true","","",null,false,"grpc_mode",0,""]
[["--grpc-port"],"scalar","int","",null,null,"grpc_port",0,""]
[["--grpc-response-timeout-secs"],"scalar","int","",null,300,"grpc_response_timeout_secs",0,""]
[["--sidecar"],"scalar","str","",null,null,"sidecar",0,""]
[["--sidecar-args"],"scalar","json_list_type","",null,null,"sidecar_args",0,""]
[["--skip-server-warmup"],"store_true","","",null,false,"skip_server_warmup",0,""]
[["--warmups"],"scalar","str","",null,null,"warmups",0,""]
[["--enable-http2"],"store_true","","",null,false,"enable_http2",0,""]
[["--http2-max-concurrent-streams"],"scalar","int","",null,200,"http2_max_concurrent_streams",0,""]
[["--http2-initial-connection-window-size"],"scalar","int","",null,null,"http2_initial_connection_window_size",0,""]
[["--ssl-keyfile"],"scalar","str","",null,null,"ssl_keyfile",0,""]
[["--ssl-certfile"],"scalar","str","",null,null,"ssl_certfile",0,""]
[["--ssl-ca-certs"],"scalar","str","",null,null,"ssl_ca_certs",0,""]
[["--ssl-keyfile-password"],"scalar","str","",null,null,"ssl_keyfile_password",0,""]
[["--enable-ssl-refresh"],"store_true","","",null,false,"enable_ssl_refresh",0,""]
[["--api-key"],"scalar","str","",null,null,"api_key",0,""]
[["--admin-api-key"],"scalar","str","",null,null,"admin_api_key",0,""]
[["--served-model-name"],"scalar","str","",null,null,"served_model_name",0,""]
[["--weight-version"],"scalar","str","",null,"default","weight_version",0,""]
[["--chat-template"],"scalar","str","",null,null,"chat_template",0,""]
[["--trust-request-chat-template"],"store_true","","",null,false,"trust_request_chat_template",0,""]
[["--hf-chat-template-name"],"scalar","str","",null,null,"hf_chat_template_name",0,""]
[["--completion-template"],"scalar","str","",null,null,"completion_template",0,""]
[["--file-storage-path"],"scalar","str","",null,"sglang_storage","file_storage_path",0,""]
[["--enable-cache-report"],"store_true","","",null,false,"enable_cache_report",0,""]
[["--return-input-ids"],"store_true","","",null,false,"return_input_ids",0,""]
[["--return-output-ids"],"store_true","","",null,false,"return_output_ids",0,""]
[["--default-chat-template-kwargs"],"scalar","json.loads","",null,null,"default_chat_template_kwargs",0,""]
[["--strip-thinking-cache"],"store_true","","",null,false,"strip_thinking_cache",0,""]
[["--enable-strict-thinking"],"store_true","","",null,false,"enable_strict_thinking",0,""]
[["--tool-server"],"scalar","str","",null,null,"tool_server",0,""]
[["--sampling-defaults"],"scalar","str","",["openai","model"],"model","sampling_defaults",0,""]
[["--asr-max-buffer-seconds"],"scalar","int","",null,60,"asr_max_buffer_seconds",0,""]
[["--asr-max-concurrent-sessions"],"scalar","int","",null,32,"asr_max_concurrent_sessions",0,""]
[["--preferred-sampling-params"],"scalar","json.loads","",null,null,"preferred_sampling_params",0,""]
[["--allow-auto-truncate"],"store_true","","",null,false,"allow_auto_truncate",0,""]
[["--stream-interval"],"scalar","int","",null,1,"stream_interval",0,""]
[["--batch-notify-size"],"scalar","int","",null,16,"batch_notify_size",0,""]
[["--stream-response-default-include-usage"],"store_true","","",null,false,"stream_response_default_include_usage",0,""]
[["--incremental-streaming-output"],"store_true","","",null,false,"incremental_streaming_output",0,""]
[["--enable-streaming-session"],"store_true","","",null,false,"enable_streaming_session",0,""]
[["--constrained-json-whitespace-pattern"],"scalar","str","",null,null,"constrained_json_whitespace_pattern",0,""]
[["--constrained-json-disable-any-whitespace"],"store_true","","",null,false,"constrained_json_disable_any_whitespace",0,""]
[["--constrained-json-max-whitespace-cnt"],"scalar","int","",null,null,"constrained_json_max_whitespace_cnt",0,""]
[["--enable-dynamic-batch-tokenizer"],"store_true","","",null,false,"enable_dynamic_batch_tokenizer",0,""]
[["--dynamic-batch-tokenizer-batch-size"],"scalar","int","",null,32,"dynamic_batch_tokenizer_batch_size",0,""]
[["--dynamic-batch-tokenizer-batch-timeout"],"scalar","float","",null,0.002,"dynamic_batch_tokenizer_batch_timeout",0,""]
[["--enable-tokenizer-batch-encode"],"store_true","","",null,false,"enable_tokenizer_batch_encode",0,""]
[["--disable-tokenizer-batch-decode"],"store_true","","",null,false,"disable_tokenizer_batch_decode",0,""]
[["--speculative-algorithm"],"scalar","str","",null,null,"speculative_algorithm",0,""]
[["--uno-lora-path"],"scalar","str","",null,null,"uno_lora_path",0,""]
[["--speculative-draft-model-path","--speculative-draft-model"],"scalar","str","",null,null,"speculative_draft_model_path",0,""]
[["--speculative-draft-model-revision"],"scalar","str","",null,null,"speculative_draft_model_revision",0,""]
[["--speculative-draft-load-format"],"scalar","str","","@load_format",null,"speculative_draft_load_format",0,""]
[["--speculative-num-steps"],"scalar","int","",null,null,"speculative_num_steps",0,""]
[["--speculative-eagle-topk"],"scalar","int","",null,null,"speculative_eagle_topk",0,""]
[["--speculative-num-draft-tokens"],"scalar","int","",null,null,"speculative_num_draft_tokens",0,""]
[["--speculative-dflash-block-size"],"scalar","int","",null,null,"speculative_dflash_block_size",0,""]
[["--speculative-domino-candidate-pool-size"],"scalar","int","",null,2048,"speculative_domino_candidate_pool_size",0,""]
[["--speculative-dspark-block-size"],"scalar","int","",null,null,"speculative_dspark_block_size",0,""]
[["--speculative-dspark-sps-table-path"],"scalar","str","",null,null,"speculative_dspark_sps_table_path",0,""]
[["--speculative-dspark-confidence-sts-path"],"scalar","str","",null,null,"speculative_dspark_confidence_sts_path",0,""]
[["--speculative-dspark-align-verify-tokens-to-graph-tier"],"store_true","","",null,false,"speculative_dspark_align_verify_tokens_to_graph_tier",0,""]
[["--speculative-accept-threshold-single"],"scalar","float","",null,1.0,"speculative_accept_threshold_single",0,""]
[["--speculative-accept-threshold-acc"],"scalar","float","",null,1.0,"speculative_accept_threshold_acc",0,""]
[["--speculative-use-rejection-sampling"],"store_true","","",null,false,"speculative_use_rejection_sampling",0,""]
[["--speculative-token-map"],"scalar","str","",null,null,"speculative_token_map",0,""]
[["--speculative-attention-mode"],"scalar","str","",["prefill","decode"],"prefill","speculative_attention_mode",0,""]
[["--speculative-draft-attention-backend"],"scalar","str","",null,null,"speculative_draft_attention_backend",0,""]
[["--speculative-dsa-topk-backend"],"scalar","str","",["sgl-kernel","torch","flashinfer"],"sgl-kernel","speculative_dsa_topk_backend",0,""]
[["--speculative-draft-kv-cache-dtype"],"scalar","str","",["auto","fp8_e5m2","fp8_e4m3","bf16","bfloat16"],null,"speculative_draft_kv_cache_dtype",0,""]
[["--speculative-draft-window-size"],"scalar","int","",null,null,"speculative_draft_window_size",0,""]
[["--speculative-draft-sink-size"],"scalar","int","",null,null,"speculative_draft_sink_size",0,""]
[["--speculative-moe-runner-backend"],"scalar","str","","@moe_runner_backend",null,"speculative_moe_runner_backend",0,""]
[["--speculative-moe-a2a-backend"],"scalar","str","","@speculative_moe_a2a_backend",null,"speculative_moe_a2a_backend",0,""]
[["--speculative-draft-model-quantization"],"scalar","str","","@quantization",null,"speculative_draft_model_quantization",0,""]
[["--speculative-skip-dp-mlp-sync"],"store_true","","",null,false,"speculative_skip_dp_mlp_sync",0,""]
[["--enable-multi-layer-eagle"],"store_true","","",null,false,"enable_multi_layer_eagle",0,""]
[["--speculative-adaptive"],"store_true","","",null,false,"speculative_adaptive",0,""]
[["--speculative-adaptive-config"],"scalar","str","",null,null,"speculative_adaptive_config",0,""]
[["--spec-trace-dir"],"scalar","str","",null,null,"spec_trace_dir",0,""]
[["--speculative-ngram-min-bfs-breadth"],"scalar","int","",null,1,"speculative_ngram_min_bfs_breadth",0,""]
[["--speculative-ngram-max-bfs-breadth"],"scalar","int","",null,10,"speculative_ngram_max_bfs_breadth",0,""]
[["--speculative-ngram-match-type"],"scalar","str","",["BFS","PROB"],"BFS","speculative_ngram_match_type",0,""]
[["--speculative-ngram-max-trie-depth"],"scalar","int","",null,18,"speculative_ngram_max_trie_depth",0,""]
[["--speculative-ngram-capacity"],"scalar","int","",null,null,"speculative_ngram_capacity",0,""]
[["--speculative-ngram-external-corpus-path"],"scalar","str","",null,null,"speculative_ngram_external_corpus_path",0,""]
[["--speculative-ngram-external-sam-budget"],"scalar","int","",null,0,"speculative_ngram_external_sam_budget",0,""]
[["--speculative-ngram-external-corpus-max-tokens"],"scalar","int","",null,10000000,"speculative_ngram_external_corpus_max_tokens",0,""]
[["--sampling-backend"],"scalar","str","",["ascend","flashinfer","intel_xpu","pytorch"],null,"sampling_backend",0,""]
[["--reasoning-parser"],"scalar","str","","@reasoning_parser",null,"reasoning_parser",0,""]
[["--tool-call-parser"],"scalar","str","","@tool_call_parser",null,"tool_call_parser",0,""]
[["--kv-canary-real-data"],"scalar","str","",["none","partial","all"],"none","kv_canary_real_data",0,""]
[["--config"],"scalar","str","",null,null,"config",0,""]
[["--disable-cuda-graph"],"store_true","","",null,false,"disable_cuda_graph",0,""]
[["--enable-flashinfer-allreduce-fusion"],"store_true","","",null,false,"enable_flashinfer_allreduce_fusion",0,""]
"""
CHOICE_LISTS = r"""
["@attention_backend",["triton","torch_native","flex_attention","dsa","nsa","qsa","dsv4","compressed","fa3","fa4","flashinfer","flashmla","trtllm_mla","cutedsl_mla","tokenspeed_mla","trtllm_mha","hpc_ops","minicpm_flashattn","minicpm_flashinfer","aiter","wave","intel_amx","ascend","intel_xpu"]]
["@fp8_gemm_runner_backend",["auto","deep_gemm","flashinfer_trtllm","flashinfer_cutlass","flashinfer_deepgemm","flashinfer_cutedsl","cutlass","triton","aiter"]]
["@fp4_gemm_runner_backend",["auto","flashinfer_cudnn","flashinfer_cutedsl","flashinfer_cutlass","flashinfer_trtllm","marlin"]]
["@dsa_prefill_backend",["flashmla_sparse","flashmla_sparse_q8","flashmla_kv","flashmla_auto","flashinfer_sparse_mla","fa3","tilelang","triton","aiter","trtllm"]]
["@linear_attn_backend",["triton","cutedsl","flashinfer","flashkda","nvidia_kda","ptx_kda","helion","intel_xpu"]]
["@linear_attn_verify_backend",["triton","cutedsl","flashinfer","flashkda","nvidia_kda","ptx_kda","helion","intel_xpu","nv_cutedsl"]]
["@moe_a2a_backend",["none","deepep","mooncake","nixl","mori","ascend_fuseep","flashinfer","megamoe","deepep_v2","pplx","ascend_tp","flashinfer_megamoe"]]
["@moe_runner_backend",["auto","deep_gemm","triton","triton_kernel","flashinfer_trtllm","experimental_sgl_trtllm","flashinfer_trtllm_routed","flashinfer_cutlass","flashinfer_mxfp4","flashinfer_cutedsl","flashinfer_megamoe","cutlass","aiter","marlin","humming","experimental_sgl_marlin","hpc_ops","megamoe","intel_xpu"]]
["@hicache_mem_layout",["layer_first","page_first","page_first_direct","page_first_kv_split","page_head"]]
["@hicache_storage_backend",["file","sim","mooncake","npu_memcache","hf3fs","nixl","aibrix","dynamic","eic","simm","mori","shm","tensorcast"]]
["@mm_attention_backend",["sdpa","fa3","fa4","triton_attn","ascend_attn","aiter_attn","flashinfer_cudnn","amx_attn","xpu_attn"]]
["@load_format",["auto","pt","safetensors","npcache","dummy","sharded_state","presharded","gguf","expert_pack","bitsandbytes","mistral","layered","flash_rl","remote","remote_instance","fastsafetensors","private","runai_streamer"]]
["@quantization",["awq","fp8","mxfp8","gptq","gptq_marlin","awq_marlin","bitsandbytes","gguf","modelopt","modelopt_fp8","modelopt_fp4","nvfp4_online","modelopt_mixed","petit_nvfp4","w8a8_int8","w8a8_fp8","moe_wna16","w4afp8","mxfp4","auto-round","auto-round-int8","compressed-tensors","modelslim","mxfp_w4a8","quark","quark_int4fp8_moe","quark_mxfp4","mlx_q4","mlx_q8","unquant","humming"]]
["@kv_cache_dtype",["auto","fp8_e5m2","fp8_e4m3","mxfp8","bf16","bfloat16","nvfp4","fp4_mx_block16"]]
["@load_balance_method",["auto","round_robin","follow_bootstrap_room","total_requests","total_tokens"]]
["@schedule_policy",["lpm","random","fcfs","dfs-weight","lof","priority","routing-key","hrrn","shortest-prefill-first"]]
["@speculative_moe_a2a_backend",["none","deepep","mooncake","nixl","mori","ascend_fuseep","flashinfer","megamoe","deepep_v2","pplx","ascend_tp"]]
["@reasoning_parser",["auto","apertus2509","deepseek-r1","deepseek-v3","deepseek-v4","deepseek-v41","dots","glm45","ling3","hunyuan","gpt-oss","k2_horizon","kimi","kimi_k2","kimi_k3","mimo","muse","poolside_v1","qwen3","qwen3-thinking","minimax","minimax-append-think","minimax-m3","nanbeige","step3","step3p5","mistral","nemotron_3","granite_thinking_parser","interns1","gemma4","inkling","iquest_q1","cohere_command4","gigachat35"]]
["@tool_call_parser",["auto","apertus2509","cohere_command4","deepseekv3","deepseekv31","deepseekv32","deepseekv4","deepseekv41","dots","glm","glm45","glm47","gpt-oss","k2_horizon","kimi_k2","kimi_k3","lfm2","ling3","llama3","mimo","minicpm5","mistral","muse","poolside_v1","pythonic","qwen","qwen25","qwen3_coder","spark25","step3","step3p5","minimax-m2","minimax-m3","nanbeige","trinity","interns1","hermes","hunyuan","gigachat3","gigachat35","gemma4","inkling","iquest_q1"]]
"""
# resolved server_args of the real EMS dflash_w6_s20 launch (Base Engine/server_logs/sglang_server_ems_dflash_w6_s20.log
# line 8, `server_args={...}` = repr(ServerArgs.resolved_dict()), 519 keys); per-launch values are overridden above.
SERVER_ARGS_TEMPLATE = r"""{'model_path': '/dev/shm/e14-weights/m0-w4a16-g128-1-fe221be94c', 'tokenizer_path': '/dev/shm/e14-weights/m0-w4a16-g128-1-fe221be94c', 'tokenizer_mode': 'auto', 'tokenizer_backend': 'huggingface', 'tokenizer_worker_num': 1, 'detokenizer_worker_num': 1, 'skip_tokenizer_init': False, 'load_format': 'auto', 'model_loader_extra_config': '{}', 'trust_remote_code': False, 'context_length': 155648, 'is_embedding': False, 'enable_multimodal': None, 'revision': None, 'model_impl': 'auto', 'model_config_parser': 'auto', 'json_model_override_args': '{}', 'dtype': 'auto', 'quantization': None, 'quantization_param_path': None, 'kv_cache_dtype': 'fp8_e4m3', 'enable_fp32_lm_head': False, 'modelopt_quant': None, 'modelopt_checkpoint_restore_path': None, 'modelopt_checkpoint_save_path': None, 'modelopt_export_path': None, 'quantize_and_serve': False, 'rl_quant_profile': None, 'enable_tf32_matmul': False, 'mem_fraction_static': 0.83, 'max_running_requests': 21, 'max_queued_requests': None, 'max_total_tokens': None, 'chunked_prefill_size': 4096, 'prefill_decode_interval': 0, 'enable_dynamic_chunking': False, 'max_prefill_tokens': 16384, 'prefill_max_requests': None, 'schedule_policy': 'lpm', 'enable_priority_scheduling': False, 'disable_priority_preemption': False, 'default_priority_value': None, 'abort_on_priority_when_disabled': False, 'schedule_low_priority_values_first': False, 'priority_scheduling_preemption_threshold': 10, 'retraction_policy': 'length', 'schedule_conservativeness': 1.0, 'page_size': 1, 'c128_page_size': 16, 'swa_full_tokens_ratio': 0.8, 'disable_hybrid_swa_memory': False, 'radix_eviction_policy': 'lru', 'radix_eviction_policy_config': None, 'prefill_only_disable_kv_cache': False, 'disable_radix_cache': False, 'enable_page_major_kv_layout': False, 'enable_unified_memory': False, 'disable_chunked_prefix_cache': False, 'disable_overlap_schedule': False, 'num_continuous_decode_steps': 1, 'scheduler_recv_interval': 1, 'enable_mixed_chunk': False, 'nccl_port': None, 'dist_timeout': None, 'dist_init_addr': None, 'gated_launch_port': None, 'nnodes': 1, 'node_rank': 0, 'tp_size': 4, 'dcp_size': 1, 'pp_size': 1, 'pp_max_micro_batch_size': None, 'pp_async_batch_depth': 0, 'dp_size': 1, 'load_balance_method': 'round_robin', 'attn_cp_size': 1, 'moe_dp_size': 1, 'dwdp_size': 1, 'dcp_comm_backend': 'ag_rs', 'dcp_replicate_q_proj': None, 'enable_prefill_cp': False, 'cp_strategy': None, 'enable_dsa_cache_layer_split': False, 'enable_cp_decode_attn_tp': False, 'enable_dp_attention': False, 'enable_dp_attention_local_control_broadcast': False, 'enable_dp_lm_head': False, 'enable_tp_lm_head_all_to_all': False, 'enable_attn_tp_input_scattered': False, 'enable_shared_experts_attn_tp': False, 'shared_experts_tp_size': None, 'enable_dense_mlp_attn_tp': False, 'enable_layernorm_sp': False, 'disable_attn_tp_gather': False, 'enable_p2p_check': False, 'device': 'cuda', 'base_gpu_id': 0, 'gpu_id_step': 1, 'random_seed': 20261005, 'mlx_enable_sampling': False, 'watchdog_timeout': 1800.0, 'soft_watchdog_timeout': None, 'sleep_on_idle': False, 'use_ray': False, 'custom_sigquit_handler': None, 'numa_node': None, 'gc_threshold': None, 'host': '127.0.0.1', 'port': 30000, 'fastapi_root_path': '', 'smg_grpc_mode': False, 'grpc_mode': False, 'grpc_port': None, 'grpc_worker_threads': 4, 'sidecar': None, 'sidecar_args': None, 'skip_server_warmup': False, 'warmups': None, 'enable_http2': False, 'http2_max_concurrent_streams': 200, 'http2_initial_connection_window_size': 1048576, 'ssl_keyfile': None, 'ssl_certfile': None, 'ssl_ca_certs': None, 'ssl_keyfile_password': None, 'enable_ssl_refresh': False, 'api_key': None, 'admin_api_key': None, 'served_model_name': '/dev/shm/e14-weights/m0-w4a16-g128-1-fe221be94c', 'weight_version': 'default', 'chat_template': None, 'trust_request_chat_template': False, 'hf_chat_template_name': None, 'completion_template': None, 'file_storage_path': 'sglang_storage', 'enable_cache_report': True, 'return_input_ids': False, 'return_output_ids': False, 'reasoning_parser': 'qwen3', 'default_chat_template_kwargs': {'preserve_thinking': True}, 'strip_thinking_cache': False, 'enable_strict_thinking': False, 'tool_call_parser': 'qwen3_coder', 'tool_server': None, 'sampling_defaults': 'model', 'asr_max_buffer_seconds': 60, 'asr_max_concurrent_sessions': 32, 'preferred_sampling_params': None, 'allow_auto_truncate': False, 'stream_interval': 1, 'batch_notify_size': 16, 'stream_response_default_include_usage': False, 'incremental_streaming_output': True, 'enable_streaming_session': False, 'enable_session_radix_cache': False, 'log_level': 'info', 'log_level_http': None, 'log_requests': False, 'log_requests_level': 2, 'log_requests_format': 'text', 'log_requests_target': None, 'uvicorn_access_log_exclude_prefixes': [], 'crash_dump_folder': None, 'show_time_cost': False, 'enable_metrics': True, 'smg_http_sidecar_port': None, 'enable_mfu_metrics': False, 'enable_metrics_for_all_schedulers': False, 'load_snapshot_publish_interval': 15, 'tokenizer_metrics_custom_labels_header': 'x-custom-labels', 'tokenizer_metrics_allowed_custom_labels': None, 'extra_metric_labels': None, 'bucket_time_to_first_token': None, 'bucket_inter_token_latency': None, 'bucket_e2e_request_latency': None, 'prompt_tokens_buckets': None, 'generation_tokens_buckets': None, 'gc_warning_threshold_secs': 0.0, 'decode_log_interval': 40, 'enable_request_time_stats_logging': False, 'kv_events_config': None, 'load_publish_endpoint': None, 'enable_forward_pass_metrics': False, 'forward_pass_metrics_worker_id': '', 'forward_pass_metrics_ipc_name': None, 'enable_trace': False, 'trace_modules': 'request', 'otlp_traces_endpoint': 'localhost:4317', 'export_metrics_to_file': False, 'export_metrics_to_file_dir': None, 'stat_loggers': None, 'constrained_json_whitespace_pattern': None, 'constrained_json_disable_any_whitespace': False, 'constrained_json_max_whitespace_cnt': None, 'attention_backend': 'flashinfer', 'decode_attention_backend': None, 'enable_lean_attention': None, 'prefill_attention_backend': None, 'sampling_backend': 'flashinfer', 'grammar_backend': 'xgrammar', 'radix_cache_backend': None, 'mm_attention_backend': None, 'fp8_gemm_runner_backend': 'auto', 'fp4_gemm_runner_backend': 'auto', 'bf16_gemm_backend': 'auto', 'dsa_prefill_backend': None, 'dsv4_prefill_backend': 'auto', 'dsa_decode_backend': None, 'dsa_paged_mqa_logits_backend': 'auto', 'dsa_topk_backend': 'sgl-kernel', 'disable_flashinfer_autotune': True, 'flashinfer_autotune_skip_ops': None, 'mamba_backend': 'triton', 'cuda_graph_config': {'decode': {'backend': 'full', 'max_bs': 21, 'bs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 'tc_compiler': 'eager', 'max_context_size': None, 'full_prefill_max_req': None, 'full_prefill_prefix_chunk_tokens': None, 'max_seq_len': None}, 'prefill': {'backend': 'disabled', 'max_bs': 4096, 'bs': [4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096], 'tc_compiler': 'eager', 'max_context_size': None, 'full_prefill_max_req': None, 'full_prefill_prefix_chunk_tokens': None, 'max_seq_len': None}}, 'cuda_graph_backend_decode': None, 'cuda_graph_backend_prefill': None, 'cuda_graph_max_bs_decode': None, 'cuda_graph_max_bs_prefill': None, 'cuda_graph_bs_decode': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 'cuda_graph_bs_prefill': None, 'cuda_graph_prefill_max_context': None, 'cuda_graph_tc_compiler': None, 'disable_prefill_cuda_graph': True, 'disable_decode_cuda_graph': False, 'disable_cuda_graph': False, 'disable_cuda_graph_padding': False, 'enable_profile_cuda_graph': False, 'enable_cudagraph_gc': False, 'debug_cuda_graph': False, 'enable_layerwise_nvtx_marker': False, 'enable_nccl_nvls': False, 'enable_symm_mem': False, 'triton_attention_reduce_in_fp32': False, 'triton_attention_num_kv_splits': 8, 'triton_attention_split_tile_size': None, 'flashinfer_mla_disable_ragged': False, 'enable_fused_qk_norm_rope': False, 'enable_precise_embedding_interpolation': False, 'enable_fused_moe_sum_all_reduce': False, 'enable_deepseek_v4_fp4_indexer': False, 'disable_custom_all_reduce': False, 'enable_mscclpp': False, 'enable_torch_symm_mem': False, 'enable_scattered_sconv': False, 'pre_warm_nccl': False, 'boundary_reduction': 'rs+rsv', 'enable_quant_communications': False, 'enable_flashinfer_allreduce_fusion': False, 'enforce_disable_flashinfer_allreduce_fusion': False, 'flashinfer_allreduce_fusion_backend': None, 'enable_aiter_allreduce_fusion': False, 'enable_torch_compile': False, 'enable_torch_compile_debug_mode': False, 'torch_compile_max_bs': 32, 'speculative_algorithm': 'DFLASH', 'uno_lora_path': None, 'speculative_draft_model_path': '/dev/shm/e14-weights/m1-bf16-1-33ded95318', 'speculative_draft_model_revision': 'main', 'speculative_draft_load_format': None, 'speculative_num_steps': 1, 'speculative_eagle_topk': 1, 'speculative_num_draft_tokens': 6, 'speculative_dflash_block_size': None, 'speculative_dspark_block_size': None, 'speculative_dspark_sps_table_path': None, 'speculative_dspark_confidence_sts_path': None, 'speculative_dspark_align_verify_tokens_to_graph_tier': False, 'speculative_accept_threshold_single': 1.0, 'speculative_accept_threshold_acc': 1.0, 'speculative_use_rejection_sampling': False, 'speculative_token_map': None, 'speculative_attention_mode': 'prefill', 'speculative_draft_attention_backend': 'flashinfer', 'speculative_dsa_topk_backend': 'sgl-kernel', 'speculative_draft_kv_cache_dtype': None, 'speculative_draft_window_size': None, 'speculative_moe_runner_backend': 'auto', 'speculative_moe_a2a_backend': None, 'speculative_draft_model_quantization': None, '_speculative_draft_quantization_explicitly_set': True, 'speculative_skip_dp_mlp_sync': False, 'enable_multi_layer_eagle': False, 'speculative_adaptive': False, 'speculative_adaptive_config': None, 'decoupled_spec_bind_endpoint': None, 'decoupled_spec_connect_endpoints': None, 'decoupled_spec_rank': None, 'decoupled_spec_role': 'null', 'spec_trace_dir': None, 'speculative_ngram_min_bfs_breadth': 1, 'speculative_ngram_max_bfs_breadth': 10, 'speculative_ngram_match_type': 'BFS', 'speculative_ngram_max_trie_depth': 18, 'speculative_ngram_capacity': 10000000, 'speculative_ngram_external_corpus_path': None, 'speculative_ngram_external_sam_budget': 0, 'speculative_ngram_external_corpus_max_tokens': 10000000, 'ep_size': 1, 'moe_a2a_backend': 'none', 'enable_w4a4_mxfp4_megamoe': False, 'deepep_v2_mode': 'direct', 'moe_runner_backend': 'auto', 'flashinfer_mxfp4_moe_precision': 'default', 'deepep_mode': 'auto', 'fuseep_mode': 2, 'deepep_dispatcher_output_dtype': 'auto', 'ep_num_redundant_experts': 0, 'ep_dispatch_algorithm': None, 'init_expert_location': 'trivial', 'enable_eplb': False, 'eplb_algorithm': 'auto', 'eplb_rebalance_num_iterations': 1000, 'eplb_rebalance_layers_per_chunk': None, 'eplb_min_rebalancing_utilization_threshold': 1.0, 'expert_distribution_recorder_mode': None, 'expert_distribution_recorder_buffer_size': 1000, 'expert_balancedness_report_mode': 'off', 'deepep_config': None, 'moe_dense_tp_size': None, 'elastic_ep_backend': None, 'enable_elastic_expert_backup': False, 'mooncake_ib_device': None, 'enable_waterfill': False, 'ep_join_mode': None, 'ep_join_rank_offset': 0, 'elastic_ep_initial_size': None, 'max_ep_size': None, 'elastic_ep_scale_timeout': 600, 'disable_flashinfer_cutlass_moe_fp4_allgather': False, 'disable_shared_experts_fusion': False, 'enforce_shared_experts_fusion': False, 'max_mamba_cache_size': 84, 'mamba_ssm_dtype': 'float16', 'mamba_max_states_per_path': -1, 'enable_mamba_cache_stochastic_rounding': False, 'mamba_cache_philox_rounds': 0, 'mamba_full_memory_ratio': 0.9, 'mamba_radix_cache_strategy': 'extra_buffer_lazy', 'uses_mamba_radix_cache': True, 'mamba_track_interval': 256, 'enable_int8_mamba_checkpoint': False, 'int8_mamba_ckpt_size': None, 'linear_attn_backend': 'triton', 'linear_attn_decode_backend': None, 'linear_attn_prefill_backend': None, 'linear_attn_verify_backend': None, 'enable_linear_replayssm': False, 'linear_replayssm_cache_len': 16, 'enable_linear_replayssm_spec': False, 'enable_hierarchical_cache': False, 'hicache_host_memory_mode': 'cache', 'hicache_ratio': 2.0, 'hicache_size': 0, 'hicache_write_policy': 'write_through', 'hicache_io_backend': 'kernel', 'hicache_mem_layout': 'page_first', 'hicache_storage_backend': None, 'hicache_storage_prefetch_policy': 'timeout', 'hicache_storage_backend_extra_config': None, 'hicache_storage_prefetch_retry_poll_interval': 8, 'hicache_storage_prefetch_retry_max_attempts': 8, 'enable_unified_cache_external_linker': False, 'unified_cache_external_linker_backend': 'mooncake', 'enable_hisparse': False, 'hisparse_config': None, 'enable_broadcast_mm_inputs_process': False, 'enable_prefix_mm_cache': False, 'mm_enable_dp_encoder': False, 'mm_process_config': {}, 'mm_processor_worker_num': 0, 'mm_io_worker_num': 0, 'allowed_media_domains': [], 'media_url_max_file_size_mb': 64, 'mm_preprocess_cache_size_mb': None, 'trust_mm_content_hashes': False, 'limit_mm_data_per_request': None, 'enable_mm_global_cache': False, 'image_processor_backend': 'auto', 'mm_global_cache_backend': 'mooncake', 'disable_fast_image_processor': False, 'mm_feature_transport': 'cpu', 'enable_lora': None, 'enable_lora_overlap_loading': None, 'max_lora_rank': None, 'lora_target_modules': None, 'lora_paths': None, 'max_loaded_loras': None, 'max_loras_per_batch': 8, 'lora_eviction_policy': 'lru', 'lora_backend': 'csgmv', 'max_lora_chunk_size': 16, 'experts_shared_outer_loras': None, 'lora_use_virtual_experts': False, 'lora_strict_loading': False, 'lora_drain_wait_threshold': 0.0, 'enable_two_batch_overlap': False, 'enable_single_batch_overlap': False, 'tbo_token_distribution_threshold': 0.48, 'cpu_offload_gb': 0, 'offload_group_size': -1, 'offload_num_in_group': 1, 'offload_prefetch_step': 1, 'offload_mode': 'cpu', 'enable_lmcache': False, 'lmcache_config_file': None, 'enable_flexkv': False, 'flexkv_config_file': None, 'kt_weight_path': None, 'kt_method': 'AMXINT4', 'kt_cpuinfer': None, 'kt_threadpool_count': 2, 'kt_num_gpu_experts': None, 'kt_max_deferred_experts_per_token': None, 'dllm_algorithm': None, 'dllm_algorithm_config': None, 'dllm_fdfo': True, 'disaggregation_mode': 'null', 'disaggregation_transfer_backend': 'mooncake', 'disaggregation_bootstrap_port': 8998, 'disaggregation_ib_device': None, 'disaggregation_decode_enable_radix_cache': False, 'disaggregation_decode_enable_offload_kvcache': False, 'disaggregation_decode_retraction_backup': None, 'num_reserved_decode_tokens': 512, 'disaggregation_decode_extra_slots': None, 'disaggregation_decode_polling_interval': 1, 'optimistic_prefill_attempts': 0, 'encoder_only': False, 'language_only': False, 'language_model_only': False, 'encoder_transfer_backend': 'zmq_to_scheduler', 'encoder_urls': [], 'encoder_bootstrap_port': 8997, 'encoder_register_urls': [], 'enable_adaptive_dispatch_to_encoder': False, 'enable_pdmux': False, 'pdmux_config_path': None, 'sm_group_num': 8, 'startup_weight_load_mode': 'serial', 'custom_weight_loader': [], 'weight_loader_disable_mmap': False, 'weight_loader_prefetch_checkpoints': False, 'weight_loader_prefetch_num_threads': 4, 'weight_loader_drop_cache_after_load': False, 'remote_instance_weight_loader_seed_instance_ip': None, 'remote_instance_weight_loader_seed_instance_service_port': None, 'remote_instance_weight_loader_send_weights_group_ports': None, 'remote_instance_weight_loader_backend': 'nccl', 'remote_instance_weight_loader_start_seed_via_transfer_engine': False, 'engine_info_bootstrap_port': 6789, 'modelexpress_config': None, 'download_dir': None, 'model_checksum': None, 'delete_ckpt_after_loading': False, 'decrypted_config_file': None, 'decrypted_draft_config_file': None, 'checkpoint_engine_wait_weights_before_ready': False, 'enable_prefill_delayer': False, 'prefill_delayer_max_delay_passes': 30, 'prefill_delayer_token_usage_low_watermark': None, 'prefill_delayer_forward_passes_buckets': None, 'prefill_delayer_wait_seconds_buckets': None, 'prefill_delayer_queue_min_ratio': None, 'prefill_delayer_max_delay_ms': None, 'min_free_slots_delay': 1, 'enable_deterministic_inference': False, 'rl_on_policy_target': None, 'kv_canary': 'none', 'kv_canary_real_data': 'none', 'kv_canary_sweep_interval': 0, 'enable_dynamic_batch_tokenizer': False, 'dynamic_batch_tokenizer_batch_size': 32, 'dynamic_batch_tokenizer_batch_timeout': 0.002, 'enable_tokenizer_batch_encode': False, 'disable_tokenizer_batch_decode': False, 'debug_tensor_dump_output_folder': None, 'debug_tensor_dump_layers': None, 'debug_tensor_dump_input_file': None, 'enable_memory_saver': False, 'enable_weights_cpu_backup': False, 'enable_draft_weights_cpu_backup': False, 'enable_custom_logit_processor': False, 'enable_return_hidden_states': False, 'return_hidden_states_mode': None, 'enable_return_routed_experts': False, 'enable_return_indexer_topk': False, 'sampling_mask_max_tokens': 4096, 'disable_outlines_disk_cache': False, 'enable_mis': False, 'weight_cache_mode': 'off', 'weight_cache_socket': None, 'weight_cache_timeout': 1800, 'forward_hooks': None, 'msprobe_dump_config': None, 'ple_offload_embedding': None, 'ple_offload_backend': 'pinned', 'ple_offload_dir': None, 'disaggregation_enable_kv_checksum': False, 'enable_pd_role_switch': False, 'speculative_domino_candidate_pool_size': 2048, 'flashinfer_a2a_dispatch_type': None, 'cuda_graph_max_seq_len_prefill': None, 'prefill_kv_cache_dequant_dtype': 'auto', 'dsv4_attn_backend': 'auto', 'otlp_service_name': None, '_radix_eviction_policy_explicitly_set': False, 'hicache_host_memory_fraction': 0.8, 'enable_linker_mla_dedup': False, '_swa_full_tokens_ratio_explicitly_set': False, 'swa_prefix_tails': None, 'enable_encoder_swa_bounded_replay': False, 'enable_decoder_swa_bounded_replay': False, 'enable_response_store': False, 'disaggregation_decode_host_receive_threshold': 0.0, 'speculative_boundary_reduction': 'ar', 'speculative_draft_sink_size': None, 'disable_aiter_allreduce_fusion_in_prefill': False, 'disable_aiter_allreduce_fusion_in_decode': False, 'grpc_response_timeout_secs': 300}"""

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
