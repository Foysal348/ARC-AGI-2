"""E12-K1 engine benchmark: one vLLM 0.19 engine (R5's exact settings) in this process; decode speed with paired workloads.

The variant is chosen by the parent before this process starts:
- default: no L4 config files, Triton block FP8 with the default config;
- tuned: L4 config files copied into vLLM's configs directory;
- marlin: VLLM_TEST_FORCE_FP8_MARLIN=1.

Workloads (prefix caching ON, so each *_prefill run warms the cache and the next run measures mostly decode):
- short_prefill / short: 16 ARC prompts (about 2-10k tokens); 3,072 new tokens each; T 1.0, top_p 0.95, top_k 20; ignore_eos.
- long_prefill / long: 16 prompts of about 36k tokens; 1,024 new tokens each.
- greedy: 4 short prompts, 192 greedy tokens, compared against the default variant by the parent.

usage: python k1_engine_bench.py --variant default --out result.json
K1_DRYRUN=1 writes fake results (local orchestration test only).
"""
import argparse, json, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
MODEL = os.environ.get("Q27X_MODEL", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1")
ARC_ROOT = os.environ.get("Q27X_ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
RESULT = {"variant": args.variant, "errors": [], "workloads": {}, "env_force_fp8_marlin": os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN")}


def save():
    with open(args.out, "w") as f:
        json.dump(RESULT, f, indent=1)


if os.environ.get("K1_DRYRUN") == "1":
    f = {"default": 1.0, "tuned": 1.15, "marlin": 1.25}.get(args.variant, 1.0)
    RESULT.update(load_sec=1.0, kv_tokens=1353600)
    for w, base in (("short", 300.0), ("long", 240.0)):
        RESULT["workloads"][w] = {"wall_sec": 10.0, "gen_tokens": 1000, "tok_s": base * f, "accept_rate": 0.55, "preemptions": 0}
        RESULT["workloads"][w + "_prefill"] = {"wall_sec": 5.0, "prompt_tokens": 50000, "prefill_tok_s": 4000.0 * f}
    RESULT["greedy_ids"] = [[1, 2, 3, 4, 5]] * 4 if args.variant != "marlin" else [[1, 2, 3, 9, 9]] * 4
    print("DRYRUN", args.variant); save(); sys.exit(0)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained(MODEL)
ch = json.load(open(os.path.join(ARC_ROOT, "arc-agi_evaluation_challenges.json")))
TIDS = sorted(ch)
INSTR = ("You are solving an ARC-AGI task. Study the training pairs, find the transformation rule, then write a Python "
         "function transform(grid) that maps each input grid to its output grid.\n\n")


def render(task_texts):
    msg = INSTR + "\n\n".join(task_texts)
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True, reasoning_effort="xhigh")


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


short = [ids_of(render([json.dumps(ch[t])])) for t in TIDS[:16]]
long_ = []
for i in range(16):        # whole tasks until the next one would pass 36k tokens, so the chat template ends correctly
    parts, j, ids = [], i * 7, None
    while True:
        cand = ids_of(render(parts + [json.dumps(ch[TIDS[j % len(TIDS)]])])); j += 1
        if len(cand) > 36000 and ids is not None:
            break
        parts.append(json.dumps(ch[TIDS[(j - 1) % len(TIDS)]])); ids = cand
        if len(ids) > 36000 or len(parts) > 60:
            break
    long_.append(ids)
RESULT["prompt_tokens"] = {"short_mean": sum(map(len, short)) / 16, "long_mean": sum(map(len, long_)) / 16}

t0 = time.time()
llm = LLM(model=MODEL, tensor_parallel_size=4, distributed_executor_backend="mp", dtype="auto", load_format="safetensors",
          trust_remote_code=False, language_model_only=True, max_model_len=106496, max_num_seqs=16,
          max_num_batched_tokens=4096, gpu_memory_utilization=0.90, enforce_eager=False, disable_custom_all_reduce=True,
          enable_prefix_caching=True, seed=20261005, disable_log_stats=False, kv_cache_dtype="fp8_e4m3",
          speculative_config={"method": "mtp", "num_speculative_tokens": 3}, attention_config={"backend": "FLASHINFER"})
RESULT["load_sec"] = round(time.time() - t0, 1)
print("ENGINE READY", args.variant, RESULT["load_sec"], "s", flush=True)
save()


def counters():
    out = {}
    try:
        for m in llm.get_metrics():
            v = getattr(m, "value", None)
            if isinstance(v, (int, float)):
                out[m.name] = out.get(m.name, 0) + v
    except Exception as e:
        RESULT["errors"].append(f"metrics: {e!r}"[:300])
    return out


def run(name, prompts, max_tokens, temperature=1.0, ignore_eos=True):
    kw = {"top_p": 0.95, "top_k": 20} if temperature > 0 else {}      # greedy: library defaults
    sps = [SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos, seed=1000 + i, **kw)
           for i in range(len(prompts))]
    c0 = counters(); t = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False)
    wall = time.time() - t; c1 = counters()
    d = lambda k: c1.get(k, 0) - c0.get(k, 0)
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    draft, acc = d("vllm:spec_decode_num_draft_tokens"), d("vllm:spec_decode_num_accepted_tokens")
    rec = {"wall_sec": round(wall, 2), "gen_tokens": gen, "tok_s": round(gen / wall, 1),
           "prompt_tokens": int(d("vllm:prompt_tokens")), "prompt_cached": int(d("vllm:prompt_tokens_cached")),
           "prefill_tok_s": round(d("vllm:prompt_tokens") / wall, 1) if max_tokens == 1 else None,
           "accept_rate": round(acc / draft, 4) if draft else None,
           "tokens_per_step": round(1 + acc / d("vllm:spec_decode_num_drafts"), 3) if d("vllm:spec_decode_num_drafts") else None,
           "preemptions": int(d("vllm:num_preemptions"))}
    RESULT["workloads"][name] = rec
    print("WORKLOAD", args.variant, name, json.dumps(rec), flush=True)
    save()
    return outs


for name, fn in (("short_prefill", lambda: run("short_prefill", short, 1)),
                 ("short", lambda: run("short", short, 3072)),
                 ("long_prefill", lambda: run("long_prefill", long_, 1)),
                 ("long", lambda: run("long", long_, 1024))):
    try:
        fn()
    except Exception:
        RESULT["errors"].append(f"{name}: " + traceback.format_exc()[-1500:]); save()
try:
    g = run("greedy", short[:4], 192, temperature=0.0)
    RESULT["greedy_ids"] = [list(o.outputs[0].token_ids) for o in g]
    RESULT["greedy_text_head"] = [tok.decode(o.outputs[0].token_ids[:60]) for o in g]
except Exception:
    RESULT["errors"].append("greedy: " + traceback.format_exc()[-1500:])
save()
print("BENCH DONE", args.variant, flush=True)
