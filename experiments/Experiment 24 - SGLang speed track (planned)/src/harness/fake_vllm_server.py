"""E24 fake vLLM 0.32.0 OpenAI server (WP2, rev 2, 2026-10-30; stdlib only, no model, no GPU) for the section-1 oracle
arms s1_a (FP8) and s1_b (INT4).

Run like the real CLI:   [python, fake_vllm_server.py, "serve", <model>, <vllm flags>]
  --help                 short group summary (as 0.32: flags are listed only with --help=all)
  --help=all             every option string of `vllm serve` 0.32.0 (table generated from engine/arg_utils.py,
                         entrypoints/launchers/cli_args.py and the config dataclasses; bool fields get --no-<x>)
  --e24-flag-check '<json argv>'   one JSON line {"ran","ok","unknown","ambiguous","parse_error","method"}

Faithful to vLLM 0.32 (sources: the v0.32.0 tree, E21/VMS 0.32 server logs, e23_lab.py):
- processes: the API server FORKS "EngineCore" and EngineCore forks 4 workers "VllmWorker-<r>" (multiprocessing fork
  context: os.fork runs os.register_at_fork handlers; VLLM_WORKER_MULTIPROC_METHOD=spawn switches to spawn), each logging
  with its own "(EngineCore pid=N)" / "(Worker_TP<r> pid=N)" prefix; workers write VLLM_CACHE_ROOT/torch_compile_cache and
  Triton files (TORCHINDUCTOR_CACHE_DIR forced under VLLM_CACHE_ROOT as vLLM does);
- startup lines incl. "non-default args", "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: Xx",
  kernel selection, "Using KV cache scaling factor 1.0 for fp8_e4m3" (FP8), CUDA graph capture; the port is closed until
  the engine is ready, then /health = 200;
- /metrics: vllm:* families with {engine,model_name} labels, counters with _total and _created (single-process
  registry), per-iteration prompt/generation counters, iteration_tokens_total histogram, preemptions;
- /v1/completions stream + non-stream: token_ids, logprobs keyed "token_id:<id>" with return_tokens_as_token_ids,
  prompt_logprobs (entry 0 null, entry p = lp(ids[p] | ids[:p]) keyed by str(id), rejected with stream=true), seed,
  generation_config defaults (T 1.0, top_k 20, top_p 0.95) when a field is omitted; /v1/chat/completions (content +
  reasoning); /v1/models; /start_profile + /stop_profile when a profiler dir is configured (*.pt.trace.json.gz).
Knobs: FAKE_TIME_SCALE / E24_TIME_SCALE, FAKE_VLLM_STARTUP_SEC (else FAKE_SG_STARTUP_SEC), FAKE_VLLM_STEP_MS,
FAKE_VLLM_PREFILL_TPS, FAKE_VLLM_NOISE, FAKE_SG_PATH_NOISE, FAKE_SG_LEAK_DEFAULT_CACHE, FAKE_VLLM_FAULT=crash_start.
"""
import argparse
import json
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

VERSION = "0.32.0"
GIB = 1024 ** 3
MULTI = {"served_model_name", "allowed_origins", "allowed_methods", "allowed_headers", "api_key", "middleware"}


# ================================================================================================ CLI
def flag_rows():
    return [json.loads(l) for l in VLLM_FLAGS.strip().splitlines()]


def build_parser():
    p = argparse.ArgumentParser(prog="vllm serve", add_help=False, conflict_handler="resolve")
    p.add_argument("model_tag", nargs="?", default=None)
    p.add_argument("--help", "-h", nargs="?", const="__short__", default=None)
    for names, kind, field in flag_rows():
        if names[0] in ("--help",):
            continue
        field = names[0].lstrip("-").replace("-", "_")          # argparse's own dest (e.g. --kv-cache-dtype)
        if kind == "b":
            p.add_argument(*names, dest=field, action=argparse.BooleanOptionalAction, default=None)
        elif kind == "t":
            p.add_argument(*names, dest=field, action="store_true", default=None)
        elif field in MULTI:
            p.add_argument(*names, dest=field, nargs="+", default=None)
        else:
            p.add_argument(*names, dest=field, default=None)
    return p


def option_strings():
    return set(build_parser()._option_string_actions)


def flag_check(argv):
    res = {"ran": True, "ok": False, "unknown": [], "ambiguous": [], "parse_error": None,
           "method": "fake_vllm_0.32.0_option_strings+argparse"}
    av = [str(a) for a in argv]
    if av and av[0] == "serve":
        av = av[1:]
    p = build_parser()
    opts = set(p._option_string_actions)
    for tok in av:
        if not tok.startswith("--"):
            continue
        name = tok.split("=", 1)[0]
        if name in opts:
            continue
        pref = [o for o in opts if o.startswith(name)]
        (res["ambiguous"] if len(pref) > 1 else res["unknown"]).append(name)
    import io
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        p.parse_args(av)
    except SystemExit:
        res["parse_error"] = (err.getvalue().strip().splitlines() or ["argparse exit 2"])[-1][:400]
    finally:
        sys.stderr = old
    res["ok"] = not res["unknown"] and not res["ambiguous"] and res["parse_error"] is None
    return res


def print_help(full):
    p = build_parser()
    print("usage: vllm serve [model_tag] [options]\n\nLaunch a local OpenAI-compatible API server to serve LLM\n"
          "completions via HTTP. Defaults to Qwen/Qwen3-0.6B if no model is specified.\n")
    if not full:
        print("Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,\n--help=ModelConfig, "
              "--help=Frontend)\n  Use `--help=all` to show all available flags at once\n\nConfig Groups:\n  Frontend\n"
              "  ModelConfig\n  LoadConfig\n  AttentionConfig\n  ParallelConfig\n  CacheConfig\n  SchedulerConfig\n"
              "  CompilationConfig\n  SpeculativeConfig\n  ObservabilityConfig\n  VllmConfig")
        return
    print("positional arguments:\n  model_tag             The model tag to serve (optional if specified in config) "
          "(default: None)\n\noptions:")
    for names, kind, field in flag_rows():
        shown = list(names)
        if kind == "b":
            shown += ["--no-" + n[2:] for n in names if n.startswith("--")]
        print("  %s%s" % (", ".join(shown), "" if kind in ("b", "t") else " %s" % field.upper()))
        print("                        (default: None)")


# ================================================================================================ logging
class VLog:
    LOCK = threading.Lock()

    def __init__(self, who):
        self.who = who

    def __call__(self, msg, level="INFO", src="api_server.py:0"):
        t = time.time()
        line = "(%s pid=%d) %s %s [%s] %s" % (self.who, os.getpid(), level, time.strftime("%m-%d %H:%M:%S",
                                                                                          time.localtime(t)), src, msg)
        with VLog.LOCK:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except Exception:
                pass

    @staticmethod
    def raw(text):
        with VLog.LOCK:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()


# ================================================================================================ child processes
def worker_main(rank, cfg):
    C.set_proctitle("VLLM::Worker_TP%d" % rank)
    C.watch_parent(lambda: None)
    log = VLog("Worker")
    log("world_size=%d rank=%d local_rank=%d distributed_init_method=file:///tmp/vllm_dist_%s backend=nccl"
        % (cfg["tp"], rank, rank, cfg["dist_id"]), src="parallel_state.py:1906")
    wl = VLog("Worker_TP%d" % rank)
    if rank == 0:
        log("vLLM is using nccl==2.29.7", src="pynccl.py:148")
    if cfg["fault"] == "crash_start" and rank == 0:
        time.sleep(max(0.0, cfg["t0"] + 0.2 * cfg["T"] - time.time()))
        VLog.raw("(Worker_TP0 pid=%d) ERROR %s [multiproc_executor.py:771] WorkerProc failed to start.\nTraceback (most "
                 "recent call last):\nRuntimeError: CUDA error: out of memory (fake FAKE_VLLM_FAULT=crash_start)"
                 % (os.getpid(), time.strftime("%m-%d %H:%M:%S")))
        os._exit(1)
    if cfg["fp8"]:
        wl("Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod", src="__init__.py:758")
    else:
        wl("Using MarlinLinearKernel for CompressedTensorsWNA16", src="compressed_tensors_wNa16.py:138")
        if rank == 0:
            wl("Using MarlinLinearKernel for mixed-precision linear", src="__init__.py:923")
    wl("Using AttentionBackendEnum.FLASHINFER backend.", src="cuda.py:446")
    time.sleep(max(0.0, cfg["t0"] + 0.25 * cfg["T"] - time.time()))
    if cfg["fp8"] and rank == 0:
        wl("Using KV cache scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that k/v_scale scaling factors "
           "are properly set in the checkpoint.", level="WARNING", src="kv_cache.py:163")
    wl("Model loading took %.2f GiB memory and %.6f seconds" % (cfg["weights_gib"], time.time() - cfg["t0"]),
       src="model_runner.py:424")
    wl("Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size."
       % cfg["block"], src="interface.py:1096")
    root = cfg["cc_root"]
    wl("[backbone] Using cache directory: %s/rank_%d_0/backbone for vLLM's torch.compile" % (root, rank), src="backends.py:1104")
    time.sleep(max(0.0, cfg["t0"] + 0.55 * cfg["T"] - time.time()))
    if cfg["cold"] or cfg["leak"]:
        C.write_cache_files(root, ["rank_%d_0/backbone/computation_graph.py" % rank,
                                   "rank_%d_0/backbone/cache_key_factors.json" % rank], rank=rank)
        C.write_cache_files(cfg["inductor"] % rank, ["fxgraph/%02x/f%012x" % (rank, C.str_key("ind%d" % rank) % 16 ** 12)],
                            rank=rank)
        C.write_cache_files(cfg["triton"], ["%012x/triton_poi_fused_%d.cubin" % (C.str_key("tri%d" % rank) % 16 ** 12, rank)],
                            rank=rank)
        if cfg["leak"]:
            C.write_cache_files(os.path.join(os.path.expanduser("~"), ".cache", "vllm", "torch_compile_cache"),
                                ["leak/rank_%d" % rank], rank=rank)
    wl("[backbone] torch.compile took %.2f s in total" % (time.time() - cfg["t0"]), src="monitor.py:62")
    time.sleep(max(0.0, cfg["t0"] + 0.85 * cfg["T"] - time.time()))
    wl("Graph capturing finished in %d secs, took 0.73 GiB" % max(1, int(time.time() - cfg["t0"])), src="model_runner.py:1127")
    while True:
        time.sleep(0.5)


def engine_core_main(cfg, conn):
    C.set_proctitle("VLLM::EngineCore")
    log = VLog("EngineCore")
    log("Initializing a V1 LLM engine (v%s) with config: model='%s', speculative_config=None, tokenizer='%s', "
        "skip_tokenizer_init=False, tokenizer_mode=auto, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=%d, "
        "tensor_parallel_size=%d, kv_cache_dtype=%s, seed=%s, served_model_name=%s, enable_prefix_caching=%s, "
        "enable_chunked_prefill=True" % (VERSION, cfg["model"], cfg["model"], cfg["max_len"], cfg["tp"], cfg["kv_dtype"],
                                        cfg["seed"], cfg["served"], cfg["prefix"]), src="core.py:133")
    log("DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=172.19.2.2 (local), "
        "world_size=%d, local_world_size=%d" % (cfg["tp"], cfg["tp"]), src="multiproc_executor.py:153")
    ctx = multiprocessing.get_context(cfg["mp_method"])
    procs = []
    for r in range(cfg["tp"]):
        p = ctx.Process(target=worker_main, args=(r, cfg), name="VllmWorker-%d" % r, daemon=False)
        p.start()
        procs.append(p)
    pids = [p.pid for p in procs]
    C.watch_parent(lambda: [p.kill() for p in procs])        # after the forks: no thread exists at fork time
    while time.time() < cfg["t0"] + 0.92 * cfg["T"]:
        if any(p.exitcode not in (None, 0) for p in procs):
            log("Worker proc VllmWorker-0 died unexpectedly, shutting down executor.", level="ERROR",
                src="multiproc_executor.py:296")
            for p in procs:
                try:
                    p.kill()
                except Exception:
                    pass
            conn.send({"ok": False})
            os._exit(1)
        time.sleep(0.05)
    log("GPU KV cache size: {:,} tokens, Maximum concurrency for {:,} tokens per request: {:.2f}x".format(
        cfg["kv_tokens"], cfg["max_len"], cfg["kv_tokens"] / cfg["max_len"]), src="kv_cache_utils.py:2327")
    log("init engine (profile, create kv cache, warmup model) took %.2f s (compilation: %.2f s)"
        % (time.time() - cfg["t0"], 0.15 * (time.time() - cfg["t0"])), src="core.py:395")
    conn.send({"ok": True, "pid": os.getpid(), "workers": pids})

    def term(*_a):
        log("[shutdown] EngineCore: trigger received signal=SIGTERM", src="core.py:1454")
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        os._exit(0)
    signal.signal(signal.SIGTERM, term)
    while True:
        try:
            if conn.poll(0.5) and conn.recv() == "quit":
                term()
        except (EOFError, OSError):
            term()


# ================================================================================================ engine
class Req:
    def __init__(self, rid, ids, temp, top_p, top_k, max_new, ignore_eos, seed, logprobs, prompt_lp, stop_ids,
                 stream, forced=None):
        self.rid, self.ids, self.prompt_len = rid, list(ids), len(ids)
        self.temp, self.top_p, self.top_k = temp, top_p, top_k
        self.max_new, self.ignore_eos, self.stop_ids = max_new, ignore_eos, set(stop_ids or [])
        self.rng = random.Random(seed if seed is not None else C.str_key(rid) ^ int(time.time() * 1e6))
        self.logprobs, self.prompt_lp, self.stream, self.forced = logprobs, prompt_lp, stream, forced
        self.out, self.out_lp = [], []
        self.computed, self.cached, self.state = 0, 0, "waiting"
        self.finish, self.stop_reason = None, None
        self.events, self.done = queue.Queue(), threading.Event()
        self.t_arrive, self.t_first, self.t_sched, self.t_finish = time.time(), None, None, None
        self.n_preempt = 0


class Engine:
    def __init__(self, cfg):
        self.c = cfg
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.waiting, self.running = [], []
        self.status = "starting"
        self.dead = False
        self.lm = C.ToyLM(cfg.weights, engine_key=cfg.engine_key, noise=cfg.noise, path_noise=cfg.path_noise)
        self.cnt = {"prompt": 0, "gen": 0, "preempt": 0, "pq": 0, "ph": 0, "cached": 0}
        self.success = {}
        self.created = time.time()
        self.h_iter = C.Hist([1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
        self.h_ttft = C.Hist([0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
                              20.0, 40.0, 80.0, 160.0, 640.0, 2560.0])
        self.h_itl = C.Hist([0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
                             40.0, 80.0])
        self.h_e2e = C.Hist([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
                             480.0, 960.0, 1920.0, 7680.0])
        self.win = {"prompt": 0, "gen": 0, "t": time.time()}
        self.prof = None

    def kv_used(self):
        return sum(r.computed + len(r.out) for r in self.running)

    def step_seconds(self, n_dec, n_pre, ctx):
        fixed = C.env_float("FAKE_VLLM_STEP_MS", 0.0)
        scale = max(C.time_scale(), C.env_float("FAKE_SG_MIN_STEP_MS", 3.0) / 26.0)
        if fixed > 0:
            dec = fixed / 1000.0
        else:
            dec = (25.0 + 0.9 * n_dec + 1.2e-5 * ctx) * scale / 1000.0 if n_dec else 0.0
        tps = C.env_float("FAKE_VLLM_PREFILL_TPS", 0.0) or 1400.0 / C.time_scale()
        return dec + n_pre / tps

    def submit(self, r):
        with self.lock:
            if self.dead:
                raise RuntimeError("EngineDeadError")
            self.waiting.append(r)
            self.cv.notify_all()

    def abort(self, r):
        with self.lock:
            if r in self.waiting:
                self.waiting.remove(r)
            if r in self.running:
                self.running.remove(r)
            if r.state != "finished":
                self._finish(r, "abort", None)

    def _finish(self, r, reason, stop_reason):
        r.state, r.finish, r.stop_reason, r.t_finish = "finished", reason, stop_reason, time.time()
        if reason != "abort":
            self.success[reason] = self.success.get(reason, 0) + 1
        self.h_e2e.observe(r.t_finish - r.t_arrive)
        r.events.put("fin")
        r.done.set()

    def run(self):
        last_log = time.time()
        while True:
            with self.lock:
                if self.dead:
                    return
                if time.time() - last_log >= max(1.0, 10.0 * C.time_scale()):
                    self._stats_log(time.time() - last_log)
                    last_log = time.time()
                budget = self.c.max_batched
                dec = [r for r in self.running if r.computed >= r.prompt_len]
                budget -= len(dec)
                pre = []
                for r in self.running:
                    if r.computed < r.prompt_len and budget > 0:
                        take = min(budget, r.prompt_len - r.computed)
                        pre.append((r, take))
                        budget -= take
                while self.waiting and budget > 0 and len(self.running) < self.c.max_seqs:
                    r = self.waiting[0]
                    need = r.prompt_len + 1
                    if self.kv_used() + need > self.c.kv_tokens and self.running:
                        break
                    self.waiting.pop(0)
                    r.state, r.t_sched = "running", time.time()
                    if self.c.prefix:
                        hit = (max(0, r.prompt_len - 1) // self.c.block) * self.c.block if r.n_preempt else 0
                        r.cached = r.computed = hit
                    self.running.append(r)
                    take = min(budget, r.prompt_len - r.computed)
                    pre.append((r, take))
                    budget -= take
                if not dec and not pre:
                    self.cv.wait(timeout=0.05)
                    continue
                ctx = sum(r.prompt_len + len(r.out) for r in dec)
                dur = self.step_seconds(len(dec), sum(t for _r, t in pre), ctx)
            time.sleep(dur)
            with self.lock:
                if self.dead:
                    return
                n_prompt = n_gen = 0
                for r, take in pre:
                    if r.state != "running":
                        continue
                    r.computed += take
                    n_prompt += take
                    if r.computed >= r.prompt_len:
                        self._token(r, "prefill")
                        n_gen += 1
                for r in dec:
                    if r.state != "running":
                        continue
                    self._token(r, "decode")
                    n_gen += 1
                while self.kv_used() > self.c.kv_tokens and len(self.running) > 1:
                    v = self.running.pop()
                    v.state, v.computed, v.n_preempt = "waiting", 0, v.n_preempt + 1
                    v.prompt_len = len(v.ids) + len(v.out)
                    self.waiting.insert(0, v)
                    self.cnt["preempt"] += 1
                self.running = [r for r in self.running if r.state == "running"]
                self.cnt["prompt"] += n_prompt
                self.cnt["gen"] += n_gen
                self.win["prompt"] += n_prompt
                self.win["gen"] += n_gen
                self.h_iter.observe(n_prompt + n_gen)
                if self.prof is not None:
                    self.prof["steps"].append((len(dec), n_prompt, time.time() - dur, time.time()))

    def _token(self, r, path):
        seq = r.ids + r.out
        pos = len(seq)
        d = self.lm.dist(seq, pos, path, None)
        gi = len(r.out)
        if r.forced is not None:
            tok = r.forced[gi] if gi < len(r.forced) else C.EOS_IDS[0]
        else:
            tok = d.sample(r.rng, r.temp, r.top_p, r.top_k)
        r.out.append(tok)
        if r.logprobs is not None:
            r.out_lp.append((d, tok))
        now = time.time()
        if r.t_first is None:
            r.t_first = now
            self.h_ttft.observe(now - r.t_arrive)
        else:
            self.h_itl.observe(now - getattr(r, "t_last", now))
        r.t_last = now
        if (not r.ignore_eos and tok in C.EOS_IDS) or tok in r.stop_ids:
            self._finish(r, "stop", tok if tok in r.stop_ids else None)
        elif len(r.out) >= r.max_new:
            self._finish(r, "length", None)
        else:
            r.events.put("tok")

    def _stats_log(self, dt):
        used = self.kv_used()
        VLog("APIServer")("Engine 000: Avg prompt throughput: %.1f tokens/s, Avg generation throughput: %.1f tokens/s, "
                          "Running: %d reqs, Waiting: %d reqs, GPU KV cache usage: %.1f%%, Prefix cache hit rate: %.1f%%"
                          % (self.win["prompt"] / dt, self.win["gen"] / dt, len(self.running), len(self.waiting),
                             100.0 * used / self.c.kv_tokens, 0.0), src="loggers.py:335")
        self.win = {"prompt": 0, "gen": 0, "t": time.time()}


# ================================================================================================ metrics
def render_metrics(e):
    c = e.c
    lab = {"engine": "0", "model_name": c.served}
    L = C.plabels
    out = []

    def gauge(name, help_, v, extra=None):
        out.extend(["# HELP %s %s" % (name, help_), "# TYPE %s gauge" % name, "%s%s %s" % (name, L(dict(lab, **(extra or {}))), C.pnum(v))])

    def counter(name, help_, v, extra=None):
        out.extend(["# HELP %s_total %s" % (name, help_), "# TYPE %s_total counter" % name,
                    "%s_total%s %s" % (name, L(dict(lab, **(extra or {}))), C.pnum(v)),
                    "# HELP %s_created %s" % (name, help_), "# TYPE %s_created gauge" % name,
                    "%s_created%s %s" % (name, L(dict(lab, **(extra or {}))), C.pnum(e.created))])

    def hist(name, help_, h):
        out.extend(["# HELP %s %s" % (name, help_), "# TYPE %s histogram" % name])
        for line in h.lines(name, lab):
            if "_sum" in line.split("{")[0]:
                continue
            out.append(line)
        out.append("%s_sum%s %s" % (name, L(lab), C.pnum(h.s)))
        out.extend(["# HELP %s_created %s" % (name, help_), "# TYPE %s_created gauge" % name,
                    "%s_created%s %s" % (name, L(lab), C.pnum(e.created))])

    with e.lock:
        out += ["# HELP python_gc_objects_collected_total Objects collected during gc",
                "# TYPE python_gc_objects_collected_total counter",
                'python_gc_objects_collected_total{generation="0"} 12345.0',
                "# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.",
                "# TYPE process_start_time_seconds gauge", "process_start_time_seconds %s" % C.pnum(e.created)]
        gauge("vllm:num_requests_running", "Number of requests in model execution batches.", len(e.running))
        gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", len(e.waiting))
        out += ["# HELP vllm:num_requests_waiting_by_reason Number of waiting requests by reason.",
                "# TYPE vllm:num_requests_waiting_by_reason gauge",
                "vllm:num_requests_waiting_by_reason%s %s" % (L(dict(lab, reason="capacity")), C.pnum(len(e.waiting))),
                "vllm:num_requests_waiting_by_reason%s 0.0" % L(dict(lab, reason="deferred"))]
        gauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", e.kv_used() / c.kv_tokens)
        counter("vllm:prefix_cache_queries", "Prefix cache queries, in terms of number of queried tokens.", e.cnt["pq"])
        counter("vllm:prefix_cache_hits", "Prefix cache hits, in terms of number of cached tokens.", e.cnt["ph"])
        counter("vllm:num_preemptions", "Cumulative number of preemption from the engine.", e.cnt["preempt"])
        counter("vllm:prompt_tokens", "Number of prefill tokens processed.", e.cnt["prompt"])
        counter("vllm:prompt_tokens_cached", "Number of cached prompt tokens (local + external).", e.cnt["cached"])
        counter("vllm:generation_tokens", "Number of generation tokens processed.", e.cnt["gen"])
        out += ["# HELP vllm:request_success_total Count of successfully processed requests.",
                "# TYPE vllm:request_success_total counter"]
        for reason in ("stop", "length", "abort", "error", "repetition"):
            out.append("vllm:request_success_total%s %s" % (L(dict(lab, finished_reason=reason)),
                                                             C.pnum(e.success.get(reason, 0))))
        hist("vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.", e.h_iter)
        hist("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", e.h_ttft)
        hist("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", e.h_itl)
        hist("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.", e.h_e2e)
        out += ["# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig", "# TYPE vllm:cache_config_info gauge",
                'vllm:cache_config_info{block_size="%d",cache_dtype="%s",enable_prefix_caching="%s",engine="0",'
                'gpu_memory_utilization="%s",num_gpu_blocks="%d"} 1.0' % (c.block, c.kv_dtype, c.prefix, c.util,
                                                                         c.kv_tokens // c.block)]
    return "\n".join(out) + "\n"


# ================================================================================================ HTTP
STATE = {}


def _lp_token(tid, as_id):
    return "token_id:%d" % tid if as_id else C.tok_text(tid)


def _completion_logprobs(r, a, b, k, as_id):
    toks, lps, tops, offs = [], [], [], []
    off = 0
    for i in range(a, b):
        d, tok = r.out_lp[i]
        lp = max(d.lp(tok), -9999.0)
        name = _lp_token(tok, as_id)
        toks.append(name)
        lps.append(lp)
        offs.append(off)
        off += len(name)
        top = {name: lp}
        for lpt, tid in d.top(max(0, k)):
            top.setdefault(_lp_token(tid, as_id), max(lpt, -9999.0))
        tops.append(top)
    return {"text_offset": offs, "token_logprobs": lps, "tokens": toks, "top_logprobs": tops}


def _prompt_logprobs(e, ids, k):
    out = [None]
    for p in range(1, len(ids)):
        d = e.lm.dist(ids, p, "prefill")
        ent = {str(ids[p]): {"logprob": d.lp(ids[p]), "rank": d.rank(ids[p]), "decoded_token": C.tok_text(ids[p])}}
        for i, (lp, tid) in enumerate(d.top(k)):
            ent.setdefault(str(tid), {"logprob": lp, "rank": i + 1, "decoded_token": C.tok_text(tid)})
        out.append(ent)
    return out


def _err(handler, msg, param=None, code=400):
    handler.send_json({"error": {"message": msg, "type": "BadRequestError", "param": param, "code": code}}, code)


class Handler(C.BaseHandler):
    def do_GET(self):
        e = STATE["engine"]
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self.send_bytes(b"", 503 if e.dead else 200, "application/json")
        if path == "/metrics":
            return self.send_bytes(render_metrics(e).encode(), 200, "text/plain; version=0.0.4; charset=utf-8")
        if path == "/v1/models":
            return self.send_json({"object": "list", "data": [{
                "id": e.c.served, "object": "model", "created": int(time.time()), "owned_by": "vllm", "root": e.c.model,
                "parent": None, "max_model_len": e.c.max_len,
                "permission": [{"id": "modelperm-%s" % C.new_rid()[:24], "object": "model_permission",
                                "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True,
                                "allow_logprobs": True, "allow_search_indices": False, "allow_view": True,
                                "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}]}]})
        if path == "/version":
            return self.send_json({"version": VERSION})
        return self.send_json({"detail": "Not Found"}, 404)

    def do_POST(self):
        e = STATE["engine"]
        path = self.path.split("?", 1)[0]
        raw = self.read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return _err(self, "JSON decode error")
        if path == "/v1/completions":
            return self._completions(e, body)
        if path == "/v1/chat/completions":
            return self._chat(e, body)
        if path in ("/start_profile", "/stop_profile") and e.c.profile_dir:
            return self._profile(e, path)
        return self.send_json({"detail": "Not Found"}, 404)

    def _profile(self, e, path):
        with e.lock:
            if path == "/start_profile":
                e.prof = {"t0": time.time(), "steps": []}
            elif e.prof is not None:
                p, e.prof = e.prof, None
                os.makedirs(e.c.profile_dir, exist_ok=True)
                for rank in range(e.c.tp):
                    ev = []
                    for n_dec, n_pre, w0, w1 in p["steps"]:
                        ts = w0 * 1e6
                        ev.append({"ph": "X", "cat": "user_annotation", "name": "execute_model", "pid": rank, "tid": 1,
                                   "ts": ts, "dur": (w1 - w0) * 1e6})
                        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": rank, "tid": 1,
                                   "ts": ts + 5, "dur": 20, "args": {"correlation": int(ts) % 10 ** 9}})
                        ev.append({"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
                                   "(ncclDevKernelArgsStorage<4096ul>)", "pid": rank, "tid": 7, "ts": ts + 30,
                                   "dur": (w1 - w0) * 1e6 * 0.4, "args": {"correlation": int(ts) % 10 ** 9,
                                                                           "grid": [2, 1, 1], "block": [288, 1, 1]}})
                    C.gz_json(os.path.join(e.c.profile_dir, "%s_%d.%d.pt.trace.json.gz"
                                           % (socket.gethostname().split(".")[0], os.getpid() + rank + 1,
                                              int(time.time() * 1000))), {"schemaVersion": 1, "traceEvents": ev})
        return self.send_bytes(b"", 200, "application/json")

    def _sampling(self, e, body):
        gc = e.c.gen_cfg
        t = body.get("temperature")
        t = gc["temperature"] if t is None else float(t)
        top_p = body.get("top_p")
        top_k = body.get("top_k")
        if t < 1e-5:
            return 0.0, 1.0, 1
        return t, float(gc["top_p"] if top_p is None else top_p), int(gc["top_k"] if top_k is None else top_k)

    def _completions(self, e, body):
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        ids = prompt if isinstance(prompt, list) else [1000 + C.str_key(w) % 246000 for w in str(prompt or "").split()] or [0]
        pl = body.get("prompt_logprobs")
        stream = bool(body.get("stream"))
        if pl is not None and stream and (pl > 0 or pl == -1):
            return _err(self, "`prompt_logprobs` are not available when `stream=True`.", "prompt_logprobs")
        mt = body.get("max_tokens")
        mt = 16 if mt is None else int(mt)
        if len(ids) + mt > e.c.max_len:
            return _err(self, "This model's maximum context length is %d tokens. However, you requested %d tokens (%d in "
                        "the messages, %d in the completion). Please reduce the length of the messages or completion."
                        % (e.c.max_len, len(ids) + mt, len(ids), mt), "max_tokens")
        lp = body.get("logprobs")
        if lp is not None and lp > e.c.max_logprobs:
            return _err(self, "Requested sample logprobs of %d, which is greater than max allowed: %d"
                        % (lp, e.c.max_logprobs), "logprobs")
        t, tp, tk = self._sampling(e, body)
        rid = "cmpl-%s" % C.new_rid()
        r = Req(rid, ids, t, tp, tk, mt, bool(body.get("ignore_eos")), body.get("seed"), lp, pl,
                body.get("stop_token_ids"), stream)
        try:
            e.submit(r)
        except RuntimeError:
            return self.send_json({"error": {"message": "EngineCore encountered an issue.", "type": "InternalServerError",
                                             "param": None, "code": 500}}, 500)
        as_id = bool(body.get("return_tokens_as_token_ids"))
        rti = bool(body.get("return_token_ids"))
        created = int(time.time())
        model = e.c.served
        if not stream:
            r.done.wait()
            if e.dead:
                return self.send_json({"error": {"message": "EngineCore encountered an issue.", "type": "InternalServerError",
                                                 "param": None, "code": 500}}, 500)
            ch = {"index": 0, "text": C.detok(r.out), "logprobs": _completion_logprobs(r, 0, len(r.out), lp, as_id)
                  if lp is not None else None, "finish_reason": r.finish, "stop_reason": r.stop_reason,
                  "token_ids": list(r.out) if rti else None,
                  "prompt_logprobs": _prompt_logprobs(e, ids, pl if pl > 0 else 5) if pl is not None else None,
                  "prompt_token_ids": list(ids) if rti else None, "routed_experts": None}
            res = {"id": rid, "object": "text_completion", "created": created, "model": model, "choices": [ch],
                   "service_tier": None, "system_fingerprint": None,
                   "usage": {"prompt_tokens": len(ids), "total_tokens": len(ids) + len(r.out), "completion_tokens": len(r.out),
                             "prompt_tokens_details": None},
                   "kv_transfer_params": None, "ec_transfer_params": None, "metrics": None}
            return self.send_bytes(C.dumps(res).encode(), 200, "application/json")
        self.start_sse()
        sent, first = 0, True
        try:
            while True:
                ev = r.events.get()
                while not r.events.empty():
                    ev2 = r.events.get_nowait()
                    ev = ev2 if ev2 != "tok" else ev
                if ev == "dead":
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                n = len(r.out)
                fin = r.state == "finished"
                if n == sent and not fin:
                    continue
                ch = {"index": 0, "text": C.detok(r.out[sent:n]),
                      "logprobs": _completion_logprobs(r, sent, n, lp, as_id) if lp is not None else None,
                      "finish_reason": r.finish if fin else None, "stop_reason": r.stop_reason if fin else None,
                      "prompt_token_ids": list(ids) if (rti and first) else None,
                      "token_ids": r.out[sent:n] if rti else None}
                self.sse(C.dumps({"id": rid, "object": "text_completion", "created": created, "model": model,
                                  "choices": [ch]}))
                first, sent = False, n
                if fin:
                    break
            if (body.get("stream_options") or {}).get("include_usage"):
                self.sse(C.dumps({"id": rid, "object": "text_completion", "created": created, "model": model, "choices": [],
                                  "usage": {"prompt_tokens": len(ids), "total_tokens": len(ids) + len(r.out),
                                            "completion_tokens": len(r.out)}}))
            self.sse("[DONE]")
            self.end_chunked()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            e.abort(r)

    def _chat(self, e, body):
        msgs = body.get("messages") or []
        last = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "") or ""
        kw = body.get("chat_template_kwargs") or {}
        thinking = kw.get("enable_thinking", True) is not False
        m = re.search(r"(-?\d+)\s*([-+*/x])\s*(-?\d+)\s*=", str(last))
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            answer = str({"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a // b if b else 0}[op])
        else:
            answer = "The grid transformation is a reflection."
        reasoning = ("The user asks a question; I answer briefly.") if (thinking and e.c.reasoning_parser) else None
        forced = [1000 + C.str_key(w) % 246000 for w in answer.split()] + [C.EOS_IDS[0]]
        prompt = [248045] + [1000 + C.str_key(w) % 246000 for w in json.dumps(msgs).split()][:4000] + [248046, 248045]
        mt = body.get("max_completion_tokens") or body.get("max_tokens") or len(forced)
        r = Req("chatcmpl-%s" % C.new_rid(), prompt, 0.0, 1.0, 1, int(mt), False, None, None, None, None, False,
                forced=forced)
        e.submit(r)
        r.done.wait()
        msg = {"role": "assistant", "content": answer, "refusal": None, "annotations": None, "audio": None,
               "function_call": None, "tool_calls": [], "reasoning": reasoning}
        res = {"id": r.rid, "object": "chat.completion", "created": int(time.time()), "model": e.c.served,
               "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": r.finish,
                            "stop_reason": None, "token_ids": None}],
               "service_tier": None, "system_fingerprint": None,
               "usage": {"prompt_tokens": len(prompt), "total_tokens": len(prompt) + len(r.out),
                         "completion_tokens": len(r.out), "prompt_tokens_details": None},
               "prompt_logprobs": None, "prompt_token_ids": None, "kv_transfer_params": None}
        return self.send_bytes(C.dumps(res).encode(), 200, "application/json")


class AccessLog:
    def access_log(self, client, method, path, code):
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = ""
        VLog.raw('INFO:     %s:%d - "%s %s HTTP/1.1" %d %s' % (client[0], client[1], method, path, code, phrase))


# ================================================================================================ main
class Cfg:
    pass


def main(argv):
    if argv and argv[0] == "--e24-flag-check":
        try:
            av = json.loads(argv[1]) if len(argv) > 1 else []
        except Exception as ex:
            print(json.dumps({"ran": False, "ok": False, "unknown": [], "ambiguous": [], "parse_error": "bad json: %r" % ex,
                              "method": "fake_vllm"}))
            return 0
        print(json.dumps(flag_check(av)))
        return 0
    av = list(argv)
    if av and av[0] == "serve":
        av = av[1:]
    if "--help=all" in av:
        print_help(True)
        return 0
    if "--help" in av or "-h" in av:
        print_help(False)
        return 0
    p = build_parser()
    ns = p.parse_args(av)
    log = VLog("APIServer")
    c = Cfg()
    c.model = ns.model_tag or ns.model or "Qwen/Qwen3-0.6B"
    c.served = (ns.served_model_name or [c.model])[0]
    c.tp = int(ns.tensor_parallel_size or 1)
    c.port = int(ns.port or 8000)
    c.host = ns.host or "0.0.0.0"
    c.max_len = int(ns.max_model_len or 262144)
    c.max_seqs = int(ns.max_num_seqs or 1024)
    c.max_batched = int(ns.max_num_batched_tokens or 8192)
    c.util = float(ns.gpu_memory_utilization or 0.9)
    c.kv_dtype = ns.kv_cache_dtype or "auto"
    c.prefix = ns.enable_prefix_caching is not False
    c.seed = ns.seed if ns.seed is not None else 0
    c.max_logprobs = int(ns.max_logprobs or 20)
    c.reasoning_parser = getattr(ns, "reasoning_parser", None)
    c.weights = C.weights_family(c.model)
    fp8 = c.weights == "fp8"
    ssm = getattr(ns, "mamba_ssm_cache_dtype", None) or "float32"
    c.block = 832 if ssm == "float16" else 1600
    weights = 7.26 if fp8 else 4.43
    c.kv_tokens = max(10000, int((c.util * 22.034 - weights - 2.3) * GIB / (8192 if c.kv_dtype.startswith("fp8") else 16384)))
    c.noise = C.env_float("FAKE_VLLM_NOISE", C.env_float("FAKE_SG_NOISE", 0.004))
    c.path_noise = C.env_float("FAKE_SG_PATH_NOISE", 0.004)
    c.engine_key = "vllm|%s|%s" % (ssm, c.kv_dtype)
    c.gen_cfg = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    try:
        gc = json.load(open(os.path.join(c.model, "generation_config.json")))
        c.gen_cfg.update({k: gc[k] for k in ("temperature", "top_k", "top_p") if k in gc})
    except Exception:
        pass
    c.profile_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if ns.profiler_config:
        try:
            c.profile_dir = json.loads(ns.profiler_config).get("torch_profiler_dir") or c.profile_dir
        except Exception:
            pass
    nondefault = {k: v for k, v in vars(ns).items() if v is not None and k not in ("help",)}
    nondefault["model"] = c.model
    for banner in ("", "       █     █     █▄   ▄█",
                   " ▄▄ ▄█ █     █     █ ▀▄▀ █  version %s" % VERSION,
                   "  █▄█▀ █     █     █     █  model   %s" % c.model,
                   "   ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀", ""):
        log(banner, src="api_utils.py:347")
    log("non-default args: %r" % (nondefault,), src="api_utils.py:286")
    log("Resolved architecture: Qwen3_5ForConditionalGeneration", src="model.py:733")
    log("Using max model len %d" % c.max_len, src="model.py:2152")
    if c.kv_dtype.startswith("fp8"):
        log("Using fp8_e4m3 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. "
            "Meanwhile, it may cause accuracy drop without a proper scaling factor", src="cache.py:361")
    log("Chunked prefill is enabled with max_num_batched_tokens=%d." % c.max_batched, src="scheduler.py:307")
    if not c.prefix:
        pass
    else:
        log("Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is "
            "enabled", src="config.py:688")
    vroot = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(os.environ.get("XDG_CACHE_HOME") or
                                                              os.path.join(os.path.expanduser("~"), ".cache"), "vllm")
    h = "%010x" % (C.str_key("%s|%s|%s" % (c.model, c.tp, c.kv_dtype)) % 16 ** 10)
    cc_root = os.path.join(vroot, "torch_compile_cache", h)
    triton = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    cold = not C.dir_has_files(cc_root)
    v = C.env_float("FAKE_VLLM_STARTUP_SEC", C.env_float("FAKE_SG_STARTUP_SEC", -1.0))
    T = v if v >= 0 else ((500.0 if cold else 300.0) * C.time_scale() if C.time_scaled_env() else 2.0)
    cfg = {"tp": c.tp, "t0": time.time(), "T": T, "model": c.model, "served": c.served, "max_len": c.max_len,
           "kv_dtype": c.kv_dtype, "seed": c.seed, "prefix": c.prefix, "fp8": fp8, "weights_gib": weights,
           "block": c.block, "cc_root": cc_root, "inductor": os.path.join(cc_root, "rank_%d_0", "inductor_cache"),
           "triton": triton, "cold": cold, "leak": C.env_bool("FAKE_SG_LEAK_DEFAULT_CACHE"),
           "fault": C.env_str("FAKE_VLLM_FAULT", ""), "kv_tokens": c.kv_tokens,
           "dist_id": C.new_rid(), "mp_method": "spawn" if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn" else "fork"}
    engine = Engine(c)
    STATE["engine"] = engine
    # bind early (setup_server), listen only when the engine is ready: connections are refused until then
    httpd = C.Server((c.host, c.port), Handler, bind_and_activate=False)
    httpd.server_bind()
    ctx = multiprocessing.get_context(cfg["mp_method"])
    rd, wr = ctx.Pipe(duplex=True)
    core = ctx.Process(target=engine_core_main, args=(cfg, wr), name="EngineCore", daemon=False)
    core.start()

    def shutdown(*_a):
        log("[shutdown] API server: shutdown triggered", src="launcher.py:254")
        try:
            rd.send("quit")
        except Exception:
            pass
        try:
            core.terminate()
            core.join(2)
            if core.is_alive():
                core.kill()
        except Exception:
            pass
        VLog.raw("INFO:     Shutting down")
        sys.stdout.flush()
        os._exit(0)
    signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=shutdown, daemon=True).start())
    try:
        if not rd.poll(T * 3 + 120):
            raise EOFError
        info = rd.recv()
    except (EOFError, OSError):
        info = {"ok": False}
    if not info.get("ok"):
        log("Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}",
            level="ERROR", src="core_client.py:1162")
        try:
            core.kill()
        except Exception:
            pass
        return 1
    threading.Thread(target=engine.run, daemon=True, name="fake-vllm-engine").start()

    def watch_core():
        while core.is_alive():
            time.sleep(0.5)
        with engine.lock:
            engine.dead = True
            for r in engine.waiting + engine.running:
                r.events.put("dead")
                r.done.set()
        log("Engine core proc EngineCore died unexpectedly, shutting down client.", level="ERROR", src="core_client.py:594")
    threading.Thread(target=watch_core, daemon=True).start()
    log("Starting vLLM API server 0 on http://%s:%d" % (c.host, c.port), src="launcher.py:111")
    for route, meth in (("/health", "GET"), ("/metrics", "GET"), ("/v1/models", "GET"), ("/v1/chat/completions", "POST"),
                        ("/v1/completions", "POST")):
        log("Route: %s, Methods: %s" % (route, meth), src="launcher.py:195")
    Handler.owner = AccessLog()
    VLog.raw("INFO:     Started server process [%d]" % os.getpid())
    VLog.raw("INFO:     Waiting for application startup.")
    VLog.raw("INFO:     Application startup complete.")
    httpd.server_activate()
    engine.status = "up"
    httpd.serve_forever(poll_interval=0.2)
    return 0


# `vllm serve` 0.32.0 options: SCR/wp2/extract_vllm_flags.py over the v0.32.0 tree. Row = [option strings, kind
# (b = BooleanOptionalAction, also --no-<x>; t = store_true; v = takes a value), dest].
VLLM_FLAGS = r"""
[["--additional-config"],"v","additional_config"]
[["--aggregate-engine-logging"],"t","aggregate_engine_logging"]
[["--all2all-backend"],"v","all2all_backend"]
[["--allow-credentials"],"b","allow_credentials"]
[["--allow-deprecated-quantization"],"b","allow_deprecated_quantization"]
[["--allowed-headers"],"v","allowed_headers"]
[["--allowed-local-media-path"],"v","allowed_local_media_path"]
[["--allowed-media-domains"],"v","allowed_media_domains"]
[["--allowed-methods"],"v","allowed_methods"]
[["--allowed-origins"],"v","allowed_origins"]
[["--anthropic-disabled-thinking-effort"],"v","anthropic_disabled_thinking_effort"]
[["--api-key"],"v","api_key"]
[["--api-server-count","-asc"],"v","api_server_count"]
[["--async-scheduling"],"b","async_scheduling"]
[["--attention-backend"],"v","backend"]
[["--attention-config","-ac"],"v","attention_config"]
[["--aux-output-config"],"v","aux_output_config"]
[["--block-size"],"v","block_size"]
[["--chat-template"],"v","chat_template"]
[["--chat-template-content-format"],"v","chat_template_content_format"]
[["--code-revision"],"v","code_revision"]
[["--cohere-format"],"v","cohere_format"]
[["--cohere-is-reasoning-model"],"b","cohere_is_reasoning_model"]
[["--collect-detailed-traces"],"v","collect_detailed_traces"]
[["--compilation-config","-cc"],"v","compilation_config"]
[["--config"],"v","config"]
[["--config-format"],"v","config_format"]
[["--convert"],"v","convert"]
[["--cp-kv-cache-interleave-size"],"v","cp_kv_cache_interleave_size"]
[["--cpu-distributed-timeout-seconds"],"v","cpu_distributed_timeout_seconds"]
[["--cpu-offload-gb"],"v","cpu_offload_gb"]
[["--cpu-offload-params"],"v","cpu_offload_params"]
[["--cudagraph-capture-sizes"],"v","cudagraph_capture_sizes"]
[["--cudagraph-metrics"],"b","cudagraph_metrics"]
[["--custom-histogram-buckets"],"v","custom_histogram_buckets"]
[["--data-parallel-address","-dpa"],"v","data_parallel_address"]
[["--data-parallel-backend","-dpb"],"v","data_parallel_backend"]
[["--data-parallel-external-lb","-dpe"],"b","data_parallel_external_lb"]
[["--data-parallel-hybrid-lb","-dph"],"b","data_parallel_hybrid_lb"]
[["--data-parallel-multi-port-external-lb","-dpm"],"t","data_parallel_multi_port_external_lb"]
[["--data-parallel-rank","-dpn"],"v","data_parallel_rank"]
[["--data-parallel-rpc-port","-dpp"],"v","data_parallel_rpc_port"]
[["--data-parallel-size","-dp"],"v","data_parallel_size"]
[["--data-parallel-size-local","-dpl"],"v","data_parallel_size_local"]
[["--data-parallel-start-rank","-dpr"],"v","data_parallel_start_rank"]
[["--data-parallel-supervisor-port"],"v","data_parallel_supervisor_port"]
[["--dbo-decode-token-threshold"],"v","dbo_decode_token_threshold"]
[["--dbo-prefill-token-threshold"],"v","dbo_prefill_token_threshold"]
[["--dcp-comm-backend"],"v","dcp_comm_backend"]
[["--dcp-kv-cache-interleave-size"],"v","dcp_kv_cache_interleave_size"]
[["--dcp-q-replicate"],"b","dcp_q_replicate"]
[["--decode-context-parallel-size","-dcp"],"v","decode_context_parallel_size"]
[["--default-chat-template-kwargs"],"v","default_chat_template_kwargs"]
[["--default-mm-loras"],"v","default_mm_loras"]
[["--device-ids"],"v","device_ids"]
[["--diffusion-config","-dc"],"v","diffusion_config"]
[["--disable-access-log-for-endpoints"],"v","disable_access_log_for_endpoints"]
[["--disable-cascade-attn"],"b","disable_cascade_attn"]
[["--disable-chunked-mm-input"],"b","disable_chunked_mm_input"]
[["--disable-custom-all-reduce"],"b","disable_custom_all_reduce"]
[["--disable-fastapi-docs"],"b","disable_fastapi_docs"]
[["--disable-hybrid-kv-cache-manager"],"b","disable_hybrid_kv_cache_manager"]
[["--disable-log-stats"],"t","disable_log_stats"]
[["--disable-nccl-for-dp-synchronization"],"b","disable_nccl_for_dp_synchronization"]
[["--disable-sliding-window"],"b","disable_sliding_window"]
[["--disable-uvicorn-access-log"],"b","disable_uvicorn_access_log"]
[["--distributed-executor-backend"],"v","distributed_executor_backend"]
[["--distributed-timeout-seconds"],"v","distributed_timeout_seconds"]
[["--download-dir"],"v","download_dir"]
[["--dp-supervisor-probe-failure-threshold"],"v","dp_supervisor_probe_failure_threshold"]
[["--dp-supervisor-probe-interval-s"],"v","dp_supervisor_probe_interval_s"]
[["--dp-supervisor-probe-timeout-s"],"v","dp_supervisor_probe_timeout_s"]
[["--dp-sync-interval"],"v","dp_sync_interval"]
[["--dtype"],"v","dtype"]
[["--ec-manager-config"],"v","ec_manager_config"]
[["--ec-transfer-config"],"v","ec_transfer_config"]
[["--elastic-ep-max-dp-size"],"v","elastic_ep_max_dp_size"]
[["--enable-auto-tool-choice"],"b","enable_auto_tool_choice"]
[["--enable-batch-sharded-sampling"],"b","enable_batch_sharded_sampling"]
[["--enable-chunked-prefill"],"b","enable_chunked_prefill"]
[["--enable-cumem-allocator"],"b","enable_cumem_allocator"]
[["--enable-dbo"],"b","enable_dbo"]
[["--enable-elastic-ep"],"b","enable_elastic_ep"]
[["--enable-ep-weight-filter"],"b","enable_ep_weight_filter"]
[["--enable-eplb"],"b","enable_eplb"]
[["--enable-expert-parallel","-ep"],"b","enable_expert_parallel"]
[["--enable-fault-tolerance"],"b","enable_fault_tolerance"]
[["--enable-flash-late-interaction"],"b","enable_flash_late_interaction"]
[["--enable-flashinfer-autotune"],"b","enable_flashinfer_autotune"]
[["--enable-force-include-usage"],"b","enable_force_include_usage"]
[["--enable-hc-sp"],"b","enable_hc_sp"]
[["--enable-layerwise-nvtx-tracing"],"b","enable_layerwise_nvtx_tracing"]
[["--enable-log-deltas"],"b","enable_log_deltas"]
[["--enable-log-outputs"],"b","enable_log_outputs"]
[["--enable-log-requests"],"b","enable_log_requests"]
[["--enable-logging-iteration-details"],"b","enable_logging_iteration_details"]
[["--enable-lora"],"b","enable_lora"]
[["--enable-mamba-cache-stochastic-rounding"],"b","enable_stochastic_rounding"]
[["--enable-mamba-shared-prefix-checkpoint"],"b","enable_mamba_shared_prefix_checkpoint"]
[["--enable-mfu-metrics"],"b","enable_mfu_metrics"]
[["--enable-mixed-moe-lora-format"],"b","enable_mixed_moe_lora_format"]
[["--enable-mm-embeds"],"b","enable_mm_embeds"]
[["--enable-moe-shared-loras"],"b","enable_moe_shared_loras"]
[["--enable-nccl-comm-suspend"],"b","enable_nccl_comm_suspend"]
[["--enable-offline-docs"],"b","enable_offline_docs"]
[["--enable-per-request-metrics"],"b","enable_per_request_metrics"]
[["--enable-prefix-caching"],"b","enable_prefix_caching"]
[["--enable-prompt-embeds"],"b","enable_prompt_embeds"]
[["--enable-prompt-tokens-details"],"b","enable_prompt_tokens_details"]
[["--enable-request-id-headers"],"b","enable_request_id_headers"]
[["--enable-return-routed-experts"],"b","enable_return_routed_experts"]
[["--enable-scale-out"],"b","enable_scale_out"]
[["--enable-server-load-tracking"],"b","enable_server_load_tracking"]
[["--enable-shm-tensor-arena"],"b","enable_shm_tensor_arena"]
[["--enable-sleep-mode"],"b","enable_sleep_mode"]
[["--enable-ssl-refresh"],"b","enable_ssl_refresh"]
[["--enable-tokenizer-info-endpoint"],"b","enable_tokenizer_info_endpoint"]
[["--enable-tower-connector-lora"],"b","enable_tower_connector_lora"]
[["--enable-trace-replay"],"b","enable_trace_replay"]
[["--enforce-eager"],"b","enforce_eager"]
[["--engram-config"],"v","engram_config"]
[["--eplb-config"],"v","eplb_config"]
[["--exclude-tools-when-tool-choice-none"],"b","exclude_tools_when_tool_choice_none"]
[["--expert-placement-strategy"],"v","expert_placement_strategy"]
[["--fail-on-environ-validation"],"b","fail_on_environ_validation"]
[["--fault-tolerance-config"],"v","fault_tolerance_config"]
[["--fingerprint-mode"],"v","fingerprint_mode"]
[["--fingerprint-value"],"v","fingerprint_value"]
[["--fully-sharded-loras"],"b","fully_sharded_loras"]
[["--gdn-prefill-backend"],"v","gdn_prefill_backend"]
[["--generation-config"],"v","generation_config"]
[["--gpu-memory-utilization","--device-memory-utilization"],"v","gpu_memory_utilization"]
[["--grpc"],"t","grpc"]
[["--grpc-port"],"v","grpc_port"]
[["--h11-max-header-count"],"v","h11_max_header_count"]
[["--h11-max-incomplete-event-size"],"v","h11_max_incomplete_event_size"]
[["--headless"],"t","headless"]
[["--hf-config-path"],"v","hf_config_path"]
[["--hf-overrides"],"v","hf_overrides"]
[["--hf-token"],"b","hf_token"]
[["--host"],"v","host"]
[["--ignore-patterns"],"v","ignore_patterns"]
[["--interleave-mm-strings"],"b","interleave_mm_strings"]
[["--io-processor-plugin"],"v","io_processor_plugin"]
[["--ir-op-priority"],"v","ir_op_priority"]
[["--jit-monitor-mode"],"v","jit_monitor_mode"]
[["--jit-monitor-verbose"],"b","jit_monitor_verbose"]
[["--kda-decode-backend"],"v","kda_decode_backend"]
[["--kda-prefill-backend"],"v","kda_prefill_backend"]
[["--kernel-config"],"v","kernel_config"]
[["--kv-cache-dtype"],"v","cache_dtype"]
[["--kv-cache-dtype-skip-layers"],"v","kv_cache_dtype_skip_layers"]
[["--kv-cache-memory-bytes"],"v","kv_cache_memory_bytes"]
[["--kv-cache-metrics"],"b","kv_cache_metrics"]
[["--kv-cache-metrics-sample"],"v","kv_cache_metrics_sample"]
[["--kv-events-config"],"v","kv_events_config"]
[["--kv-offloading-backend"],"v","kv_offloading_backend"]
[["--kv-offloading-size"],"v","kv_offloading_size"]
[["--kv-sharing-fast-prefill"],"b","kv_sharing_fast_prefill"]
[["--kv-transfer-config"],"v","kv_transfer_config"]
[["--language-model-only"],"b","language_model_only"]
[["--limit-mm-per-prompt"],"v","limit_per_prompt"]
[["--linear-backend"],"v","linear_backend"]
[["--load-format"],"v","load_format"]
[["--log-config-file"],"v","log_config_file"]
[["--log-error-stack"],"b","log_error_stack"]
[["--log-level"],"v","log_level"]
[["--logging-config"],"v","logging_config"]
[["--logits-processors"],"v","logits_processors"]
[["--logprobs-mode"],"v","logprobs_mode"]
[["--long-prefill-token-threshold"],"v","long_prefill_token_threshold"]
[["--long-prefill-token-threshold-adaptive"],"b","long_prefill_token_threshold_adaptive"]
[["--lora-dtype"],"v","lora_dtype"]
[["--lora-modules"],"v","lora_modules"]
[["--lora-target-modules"],"v","target_modules"]
[["--mamba-backend"],"v","backend"]
[["--mamba-block-size"],"v","mamba_block_size"]
[["--mamba-cache-dtype"],"v","mamba_cache_dtype"]
[["--mamba-cache-mode"],"v","mamba_cache_mode"]
[["--mamba-cache-philox-rounds"],"v","stochastic_rounding_philox_rounds"]
[["--mamba-ssm-cache-dtype"],"v","mamba_ssm_cache_dtype"]
[["--mamba-ssu-algorithm"],"v","ssu_algorithm"]
[["--master-addr"],"v","master_addr"]
[["--master-port"],"v","master_port"]
[["--max-cpu-loras"],"v","max_cpu_loras"]
[["--max-cudagraph-capture-size"],"v","max_cudagraph_capture_size"]
[["--max-log-len"],"v","max_log_len"]
[["--max-logprobs"],"v","max_logprobs"]
[["--max-lora-cls-labels"],"v","max_lora_cls_labels"]
[["--max-lora-rank"],"v","max_lora_rank"]
[["--max-loras"],"v","max_loras"]
[["--max-model-len"],"v","max_model_len"]
[["--max-num-active-seqs"],"v","max_num_active_seqs"]
[["--max-num-batched-tokens"],"v","max_num_batched_tokens"]
[["--max-num-queued-reqs"],"v","max_num_queued_reqs"]
[["--max-num-queued-tokens"],"v","max_num_queued_tokens"]
[["--max-num-scheduled-tokens"],"v","max_num_scheduled_tokens"]
[["--max-num-seqs"],"v","max_num_seqs"]
[["--max-parallel-loading-workers"],"v","max_parallel_loading_workers"]
[["--media-io-kwargs"],"v","media_io_kwargs"]
[["--middleware"],"v","middleware"]
[["--mm-device-do-normalize"],"b","mm_device_do_normalize"]
[["--mm-encoder-attn-backend"],"v","mm_encoder_attn_backend"]
[["--mm-encoder-attn-dtype"],"v","mm_encoder_attn_dtype"]
[["--mm-encoder-fp8-scale-path"],"v","mm_encoder_fp8_scale_path"]
[["--mm-encoder-fp8-scale-save-margin"],"v","mm_encoder_fp8_scale_save_margin"]
[["--mm-encoder-fp8-scale-save-path"],"v","mm_encoder_fp8_scale_save_path"]
[["--mm-encoder-only"],"b","mm_encoder_only"]
[["--mm-encoder-tp-mode"],"v","mm_encoder_tp_mode"]
[["--mm-hasher-algorithm"],"v","mm_hasher_algorithm"]
[["--mm-ipc-gpu-memory-gb"],"v","mm_ipc_gpu_memory_gb"]
[["--mm-processor-cache-gb"],"v","mm_processor_cache_gb"]
[["--mm-processor-cache-type"],"v","mm_processor_cache_type"]
[["--mm-processor-device"],"v","mm_processor_device"]
[["--mm-processor-kwargs"],"v","mm_processor_kwargs"]
[["--mm-shm-cache-max-object-size-mb"],"v","mm_shm_cache_max_object_size_mb"]
[["--mm-tensor-ipc"],"v","mm_tensor_ipc"]
[["--model"],"v","model"]
[["--model-class-overrides"],"v","model_class_overrides"]
[["--model-impl"],"v","model_impl"]
[["--model-loader-extra-config"],"v","model_loader_extra_config"]
[["--moe-backend"],"v","moe_backend"]
[["--nnodes","-n"],"v","nnodes"]
[["--node-rank","-r"],"v","node_rank"]
[["--num-gpu-blocks-override"],"v","num_gpu_blocks_override"]
[["--numa-bind"],"b","numa_bind"]
[["--numa-bind-cpus"],"v","numa_bind_cpus"]
[["--numa-bind-nodes"],"v","numa_bind_nodes"]
[["--offload-backend"],"v","offload_backend"]
[["--offload-group-size"],"v","offload_group_size"]
[["--offload-num-in-group"],"v","offload_num_in_group"]
[["--offload-params"],"v","offload_params"]
[["--offload-prefetch-step"],"v","offload_prefetch_step"]
[["--optimization-level"],"v","optimization_level"]
[["--otlp-traces-endpoint"],"v","otlp_traces_endpoint"]
[["--override-generation-config"],"v","override_generation_config"]
[["--per-request-spec-decode-metrics"],"v","per_request_spec_decode_metrics"]
[["--performance-mode"],"v","performance_mode"]
[["--pipeline-parallel-size","-pp"],"v","pipeline_parallel_size"]
[["--pooler-config"],"v","pooler_config"]
[["--port"],"v","port"]
[["--prefill-context-parallel-size","-pcp"],"v","prefill_context_parallel_size"]
[["--prefill-schedule-interval"],"v","prefill_schedule_interval"]
[["--prefix-cache-retention-interval"],"v","prefix_cache_retention_interval"]
[["--prefix-caching-hash-algo"],"v","prefix_caching_hash_algo"]
[["--prefix-match-unit"],"v","prefix_match_unit"]
[["--profiler-config"],"v","profiler_config"]
[["--pt-load-map-location"],"v","pt_load_map_location"]
[["--quantization","-q"],"v","quantization"]
[["--quantization-config"],"v","quantization_config"]
[["--ray-workers-use-nsight"],"b","ray_workers_use_nsight"]
[["--reasoning-config"],"v","reasoning_config"]
[["--reasoning-parser"],"v","reasoning_parser"]
[["--reasoning-parser-plugin"],"v","reasoning_parser_plugin"]
[["--renderer-num-workers"],"v","renderer_num_workers"]
[["--replayssm-buffer-len"],"v","replayssm_buffer_len"]
[["--response-role"],"v","response_role"]
[["--return-sampling-mask"],"b","return_sampling_mask"]
[["--return-tokens-as-token-ids"],"b","return_tokens_as_token_ids"]
[["--revision"],"v","revision"]
[["--root-path"],"v","root_path"]
[["--runner"],"v","runner"]
[["--safetensors-load-strategy"],"v","safetensors_load_strategy"]
[["--safetensors-prefetch-block-size"],"v","safetensors_prefetch_block_size"]
[["--safetensors-prefetch-num-threads"],"v","safetensors_prefetch_num_threads"]
[["--scheduler-cls"],"v","scheduler_cls"]
[["--scheduler-reserve-full-isl"],"b","scheduler_reserve_full_isl"]
[["--scheduling-policy"],"v","policy"]
[["--seed"],"v","seed"]
[["--served-model-name"],"v","served_model_name"]
[["--show-hidden-metrics-for-version"],"v","show_hidden_metrics_for_version"]
[["--shutdown-timeout"],"v","shutdown_timeout"]
[["--skip-mm-profiling"],"b","skip_mm_profiling"]
[["--skip-tokenizer-init"],"b","skip_tokenizer_init"]
[["--sleep-mode-offload-cudagraph"],"b","sleep_mode_offload_cudagraph"]
[["--sleep-preserve-parameter-names"],"v","sleep_preserve_parameter_names"]
[["--sparse-indexer-topk-backend"],"v","sparse_indexer_topk_backend"]
[["--spec-method"],"v","method"]
[["--spec-model"],"v","model"]
[["--spec-tokens"],"v","num_speculative_tokens"]
[["--specialize-active-lora"],"b","specialize_active_lora"]
[["--speculative-config","-sc"],"v","speculative_config"]
[["--sse-keep-alive-interval"],"v","sse_keep_alive_interval"]
[["--ssl-ca-certs"],"v","ssl_ca_certs"]
[["--ssl-cert-reqs"],"v","ssl_cert_reqs"]
[["--ssl-certfile"],"v","ssl_certfile"]
[["--ssl-ciphers"],"v","ssl_ciphers"]
[["--ssl-keyfile"],"v","ssl_keyfile"]
[["--stream-interval"],"v","stream_interval"]
[["--structured-outputs-config"],"v","structured_outputs_config"]
[["--swa-bounded-replay"],"b","swa_bounded_replay"]
[["--tensor-parallel-size","-tp"],"v","tensor_parallel_size"]
[["--tokenizer"],"v","tokenizer"]
[["--tokenizer-mode"],"v","tokenizer_mode"]
[["--tokenizer-revision"],"v","tokenizer_revision"]
[["--tokens-only"],"b","tokens_only"]
[["--tool-call-parser"],"v","tool_call_parser"]
[["--tool-parser-plugin"],"v","tool_parser_plugin"]
[["--tool-server"],"v","tool_server"]
[["--tool-strict-level"],"v","tool_strict_level"]
[["--trust-remote-code"],"b","trust_remote_code"]
[["--trust-request-chat-template"],"b","trust_request_chat_template"]
[["--trust-request-mm-kwargs"],"b","trust_request_mm_kwargs"]
[["--ubatch-size"],"v","ubatch_size"]
[["--uds"],"v","uds"]
[["--use-fp64-gumbel"],"b","use_fp64_gumbel"]
[["--use-replayssm"],"b","use_replayssm"]
[["--use-tqdm-on-load"],"b","use_tqdm_on_load"]
[["--uvicorn-log-level"],"v","uvicorn_log_level"]
[["--video-pruning-method"],"v","video_pruning_method"]
[["--video-pruning-rate"],"v","video_pruning_rate"]
[["--watermark"],"v","watermark"]
[["--watermark-config"],"v","watermark_config"]
[["--weight-transfer-config"],"v","weight_transfer_config"]
[["--worker-cls"],"v","worker_cls"]
[["--worker-extension-cls"],"v","worker_extension_cls"]
"""

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
