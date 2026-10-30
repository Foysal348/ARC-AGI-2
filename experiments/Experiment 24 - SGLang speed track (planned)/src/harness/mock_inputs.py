"""E24 mock /kaggle/input tree for the local runner and the harness tests (WP5, rev 2, 2026-10-30; stdlib only).

The layout is the real Kaggle mount layout (DESIGN §2; kaggle_inputs_kits.md §2-§3; E23 CPU kit v2 `inputs` log):
  datasets/foysalemonshanto/sglang-0-5-21-cu13-wheelhouse/      BUILD_INFO.json (sglang 0.5.21: the EMS `_e11_discover`
        predicate), requirements.lock (the real 220-pin E11-02 lock when the repo has it), SHA256SUMS, dummy wheels (one
        stored '+'-stripped, as Kaggle stores flashinfer_jit_cache-0.6.18+cu130), install_wheelhouse.py stub, README.md
  datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse/        the same shape; install_wheelhouse.py stub
  models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1/   config (compressed-tensors w4a16 g128, EMS
        predicate), tokenizer files, tiny model.safetensors + model_mtp.safetensors + index json
  models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1/              config (architectures ["DFlash2DraftModel"])
  models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1/    config (qwen3_5 fp8), tokenizer files, tiny shards
        + the real index json when the repo has it
  competitions/arc-prize-2026-arc-agi-2/                                   the real ARC files (E1 fixtures, git-tracked),
        else 120 synthesized valid tasks
Both install_wheelhouse.py stubs print VENV_PYTHON=<the python that runs them>, CUDA_HOME=/tmp and PATH= (the real
installer's interface, E23 `install_stock` parses these lines); nothing is installed.

Tokenizer: a SMALL byte-level BPE tokenizer.json (256 byte tokens, no merges) with the real Qwen3.8 normalizer,
pre-tokenizer, decoder and the 33 real added tokens at their real ids (248044 <|endoftext|> ... 248069 </think> ...
248076). HF `tokenizers` loads it (one id per byte, special tokens single ids) and the notebook's fake-tokenizer path can
read the special ids from `added_tokens`. INT4 and FP8 get byte-identical tokenizer files, as on Kaggle (R3K2 tokenizer
check: same vocab, merges, added tokens, eos); `tokenizer_mismatch=True` makes INT4 differ (negative test).
Contents are deterministic, so identity hashes are stable across builds.

usage: python mock_inputs.py <root> [--n-tasks N] [--tokenizer-mismatch]   -> prints the paths as JSON
       python mock_inputs.py --selftest                                       -> builds into a temp dir and checks it
"""
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = HERE.parents[2]          # <repo>/experiments
OWNER = "foysalemonshanto"
SGL_WH = f"datasets/{OWNER}/sglang-0-5-21-cu13-wheelhouse"
VLLM_WH = f"datasets/{OWNER}/vllm-0-32-0-cu13-wheelhouse"
INT4 = f"models/{OWNER}/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1"
DFLASH = f"models/{OWNER}/qwen3-8-27b-dflash2/pytorch/bf16/1"
FP8 = f"models/{OWNER}/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
ARC = "competitions/arc-prize-2026-arc-agi-2"
ARC_FILES = ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json", "arc-agi_test_challenges.json",
             "arc-agi_training_challenges.json", "arc-agi_training_solutions.json", "sample_submission.json")
# the real Qwen3.8 added tokens (FP8 tokenizer.json, E1 fixture): id, content, special
ADDED = [(248044, "<|endoftext|>", True), (248045, "<|im_start|>", True), (248046, "<|im_end|>", True),
         (248047, "<|object_ref_start|>", True), (248048, "<|object_ref_end|>", True), (248049, "<|box_start|>", True),
         (248050, "<|box_end|>", True), (248051, "<|quad_start|>", True), (248052, "<|quad_end|>", True),
         (248053, "<|vision_start|>", True), (248054, "<|vision_end|>", True), (248055, "<|vision_pad|>", True),
         (248056, "<|image_pad|>", True), (248057, "<|video_pad|>", True), (248058, "<tool_call>", False),
         (248059, "</tool_call>", False), (248060, "<|fim_prefix|>", False), (248061, "<|fim_middle|>", False),
         (248062, "<|fim_suffix|>", False), (248063, "<|fim_pad|>", False), (248064, "<|repo_name|>", False),
         (248065, "<|file_sep|>", False), (248066, "<tool_response>", False), (248067, "</tool_response>", False),
         (248068, "<think>", False), (248069, "</think>", False), (248070, "<|audio_start|>", True),
         (248071, "<|audio_end|>", True), (248072, "<tts_pad>", True), (248073, "<tts_text_bos>", True),
         (248074, "<tts_text_eod>", True), (248075, "<tts_text_bos_single>", True), (248076, "<|audio_pad|>", True)]
SPLIT_RE = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
SGL_PINS = ("sglang==0.5.21\nsglang-kernel==0.4.7\ntorch==2.13.0\nflashinfer-python==0.6.18\nflashinfer-jit-cache==0.6.18+cu130\n"
            "transformers==5.12.1\ntriton==3.7.1\nxgrammar==0.2.7\nnvidia-nccl-cu13==2.29.7\nprometheus-client==0.26.0\n")
VLLM_PINS = ("vllm==0.32.0\ntorch==2.13.0\nflashinfer-python==0.7.0.post1\nflashinfer-jit-cache==0.7.0.post1+cu130\n"
             "transformers==5.19.0\ncompressed-tensors==0.17.0\n")
STUB_INSTALLER = '''# install_wheelhouse.py (E24 MOCK) - the real installer's interface, nothing is installed:
#     python install_wheelhouse.py [VENV_DIR]   -> prints VENV_PYTHON=, CUDA_HOME=, PATH= (parsed by the notebooks)
import sys
from pathlib import Path
venv = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock-venv")
print("E24 mock wheelhouse installer: nothing installed, venv", venv)
print("restored wheel names: none")
print("VENV_PYTHON=" + sys.executable)
print("CUDA_HOME=/tmp")
print("PATH=" + str(Path(sys.executable).parent) + ":/tmp/bin:$PATH")
'''


def _wj(p, obj, indent=1):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent), encoding="utf-8", newline="\n")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _bytes_to_unicode():
    """GPT-2 byte-level alphabet (the ByteLevel pre-tokenizer maps every byte to one of these 256 chars)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def tokenizer_json(variant=""):
    """Small byte-level BPE tokenizer.json with the real Qwen3.8 added tokens (see the module docstring)."""
    b2u = _bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    if variant:                              # a deterministic difference in the vocab (negative test)
        vocab = {k: (v + 1) % 256 for k, v in vocab.items()}
    bl = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    return {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                              "normalized": False, "special": s} for i, c, s in ADDED],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": SPLIT_RE}, "behavior": "Isolated", "invert": False}, bl]},
            "post_processor": bl, "decoder": bl,
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                      "vocab": vocab, "merges": []},
            "e24_mock": {"note": "E24 mock tokenizer: 256 byte tokens + the real added tokens", "variant": variant}}


def tokenizer_config():
    return {"add_prefix_space": False, "add_bos_token": False, "bos_token": None, "clean_up_tokenization_spaces": False,
            "eos_token": "<|im_end|>", "errors": "replace", "model_max_length": 262144, "pad_token": "<|endoftext|>",
            "split_special_tokens": False, "tokenizer_class": "Qwen2Tokenizer", "unk_token": None,
            "added_tokens_decoder": {str(i): {"content": c, "lstrip": False, "normalized": False, "rstrip": False,
                                              "single_word": False, "special": s} for i, c, s in ADDED},
            "chat_template": ("{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + "
                              "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n"
                              "<think>\\n' }}{%- endif %}")}


GEN_CONFIG = {"bos_token_id": 248044, "do_sample": True, "eos_token_id": [248046, 248044], "pad_token_id": 248044,
              "temperature": 1.0, "top_k": 20, "top_p": 0.95}


def _text_config():
    lt = (["linear_attention"] * 3 + ["full_attention"]) * 16
    return {"attention_bias": False, "attn_output_gate": True, "bos_token_id": 248044, "dtype": "bfloat16",
            "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu", "hidden_size": 5120,
            "intermediate_size": 17408, "layer_types": lt, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1, "num_attention_heads": 24, "num_hidden_layers": 64, "num_key_value_heads": 4,
            "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06, "tie_word_embeddings": False, "vocab_size": 248320,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                                "rope_theta": 10000000, "rope_type": "default"}}


def _e1_sources():
    for d in EXPS.glob("Experiment 1 - */Experiment Notebooks/Build/Q27X_build"):
        if (d / "sources" / "model_qwen3-8-27b-fp8-repacked-v1" / "config.json").is_file():
            return d
    return None


def fp8_config():
    e1 = _e1_sources()
    if e1 is not None:            # the real repack config (git-tracked in E1)
        return json.loads((e1 / "sources/model_qwen3-8-27b-fp8-repacked-v1/config.json").read_text(encoding="utf-8"))
    return {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "language_model_only": False,
            "tie_word_embeddings": False, "text_config": _text_config(),
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                    "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]}}


def int4_config():
    c = json.loads(json.dumps(fp8_config()))
    c["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
        "version": "0.18.0", "global_compression_ratio": None,
        "config_groups": {"group_0": {"targets": ["Linear"], "input_activations": None, "output_activations": None,
                                      "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                                                  "group_size": 128, "dynamic": False, "actorder": None,
                                                  "observer": "minmax"}}},
        "kv_cache_scheme": {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False, "symmetric": True},
        "ignore": ["lm_head", "re:.*mlp.gate$", "re:.*visual.*", "re:.*mtp.*"]}
    c["e24_mock"] = "RedHat INT4 W4A16 g128 (mock)"
    return c


DFLASH_CONFIG = {"architectures": ["DFlash2DraftModel"], "model_type": "dflash2", "torch_dtype": "bfloat16",
                 "num_target_layers": 64, "vocab_size": 248320, "hidden_size": 5120,
                 "dflash_config": {"block_size": 8, "target_layer_ids": [5, 19, 33, 47, 61]}}


def safetensors_blob(seed, n=16):
    """A tiny valid safetensors file (8-byte LE header length + JSON header + data), content keyed by `seed`."""
    rnd = random.Random(seed)
    data = bytes(rnd.randrange(256) for _ in range(2 * n))
    hdr = json.dumps({"__metadata__": {"format": "pt", "e24_mock": str(seed)},
                      "w": {"dtype": "BF16", "shape": [n], "data_offsets": [0, 2 * n]}}, separators=(",", ":")).encode()
    hdr += b" " * (-len(hdr) % 8)
    return struct.pack("<Q", len(hdr)) + hdr + data


def _wheel(name):
    """A small valid-looking wheel (zip with a dist-info) - only discovery and sha checks ever read it."""
    dist, ver = name.split("-")[0], name.split("-")[1]
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(f"{dist}-{ver}.dist-info/METADATA", date_time=(2026, 10, 30, 0, 0, 0))
        z.writestr(info, f"Metadata-Version: 2.1\nName: {dist}\nVersion: {ver}\nSummary: E24 mock wheel\n")
    return buf.getvalue()


def _real_sglang_lock():
    """The real 220-pin lock (E11-02 build notebook, cell 1 LOCK), or None."""
    for nb in EXPS.glob("Experiment 11 - */Experiment Notebooks/E11_02_sglang_0_5_21_wheelhouse_build_on_KAGGLE.ipynb"):
        try:
            for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
                s = "".join(c.get("source") or [])
                if 'LOCK = """' in s:
                    lock = s.split('LOCK = """', 1)[1].split('"""', 1)[0]
                    if "sglang==0.5.21" in lock:
                        return lock
        except Exception:
            pass
    return None


def _wheelhouse(d, lock, wheels, build_info):
    d.mkdir(parents=True, exist_ok=True)
    (d / "requirements.lock").write_text(lock, encoding="utf-8", newline="\n")
    sums = []
    for name in wheels:                     # SHA256SUMS keeps the original names; Kaggle stores them '+'-stripped
        b = _wheel(name)
        (d / name.replace("+", "")).write_bytes(b)
        sums.append(f"{_sha(b)}  {name}\n")
    (d / "SHA256SUMS").write_text("".join(sums), encoding="utf-8", newline="\n")
    (d / "install_wheelhouse.py").write_text(STUB_INSTALLER, encoding="utf-8", newline="\n")
    (d / "README.md").write_text("E24 mock wheelhouse (local runner). Real dataset: see DESIGN §2.\n", encoding="utf-8")
    if build_info:
        _wj(d / "BUILD_INFO.json", build_info)


def _model(d, config, shards, tok=None, index=None, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    _wj(d / "config.json", config)
    for i, name in enumerate(shards):
        (d / name).write_bytes(safetensors_blob(f"{d.parts[-4]}/{name}"))
    if tok is not None:
        _wj(d / "tokenizer.json", tok, indent=None)
        _wj(d / "tokenizer_config.json", tokenizer_config())
        _wj(d / "generation_config.json", GEN_CONFIG)
        (d / "chat_template.jinja").write_text(tokenizer_config()["chat_template"], encoding="utf-8")
    if index is not None:
        _wj(d / "model.safetensors.index.json", index)
    for k, v in (extra or {}).items():
        (d / k).write_text(v, encoding="utf-8")


def _grid(rnd, h, w, colors):
    return [[rnd.choice(colors) for _ in range(w)] for _ in range(h)]


def synth_arc(n=120, seed=20261030):
    """Valid ARC-format tasks (train pairs + test inputs, grids 0-9) with a simple per-task rule (flip / transpose / recolor)."""
    rnd = random.Random(seed)
    ch, sol = {}, {}
    rules = ("flip_h", "flip_v", "transpose", "recolor")
    for t in range(n):
        tid = f"{rnd.getrandbits(32):08x}"
        rule, cmap = rules[t % 4], list(range(10))
        rnd.shuffle(cmap)

        def f(g):
            if rule == "flip_h":
                return [r[::-1] for r in g]
            if rule == "flip_v":
                return g[::-1]
            if rule == "transpose":
                return [list(r) for r in zip(*g)]
            return [[cmap[v] for v in r] for r in g]
        pairs = []
        for _ in range(3 + t % 3):
            h, w = rnd.randint(3, 18), rnd.randint(3, 18)
            g = _grid(rnd, h, w, list(range(10)))
            pairs.append({"input": g, "output": f(g)})
        tests = []
        for _ in range(1 + (t % 5 == 0)):
            h, w = rnd.randint(3, 22), rnd.randint(3, 22)
            tests.append(_grid(rnd, h, w, list(range(10))))
        ch[tid] = {"train": pairs, "test": [{"input": g} for g in tests]}
        sol[tid] = [f(g) for g in tests]
    return ch, sol


def _arc_source():
    env = os.environ.get("E24_MOCK_ARC_DIR")
    cands = [Path(env)] if env else []
    cands += sorted(EXPS.glob("Experiment 1 - */Experiment Notebooks/Build/Q27X_build/fixtures/fake_input/competitions/"
                              "arc-prize-2026-arc-agi-2"))
    for d in cands:
        if (d / "arc-agi_evaluation_challenges.json").is_file() and (d / "arc-agi_test_challenges.json").is_file():
            return d
    return None


def build(root, n_tasks=None, tokenizer_mismatch=False, full_arc=True, arc_source="auto"):
    """Write the mock tree under `root` (created; must be empty or absent). Returns a dict of paths and facts.
    n_tasks: keep only the first n evaluation/test tasks (sorted ids); None keeps all. arc_source: "auto" (repo copy,
    else synthesized), "synth", or a directory."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"mock input root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    lock = _real_sglang_lock() or SGL_PINS
    _wheelhouse(root / SGL_WH, lock, ["sglang-0.5.21-py3-none-any.whl", "sglang_kernel-0.4.7-cp310-abi3-manylinux2014_x86_64.whl",
                                      "flashinfer_jit_cache-0.6.18+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
                                      "torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"],
                {"sglang": "0.5.21", "python": "3.12.13", "pins": len([l for l in lock.splitlines() if "==" in l]),
                 "stamp": "sglang==0.5.21 torch==2.13.0 flashinfer==0.6.18 jit-cache==0.6.18+cu130 transformers==5.12.1",
                 "e24_mock": True})
    _wheelhouse(root / VLLM_WH, VLLM_PINS, ["vllm-0.32.0-cp38-abi3-manylinux_2_31_x86_64.whl",
                                            "flashinfer_jit_cache-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"],
                {"vllm": "0.32.0", "python": "3.12.13", "e24_mock": True})
    tok = tokenizer_json()
    fp8c = fp8_config()
    e1 = _e1_sources()
    if e1 is not None:
        fidx = json.loads((e1 / "sources/model_qwen3-8-27b-fp8-repacked-v1/model.safetensors.index.json").read_text(encoding="utf-8"))
    else:
        fidx = {"metadata": {}, "weight_map": {"model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                                              "lm_head.weight": "model-00002-of-00002.safetensors"}}
    _model(root / FP8, fp8c, sorted(set(fidx["weight_map"].values())), tok=tok, index=fidx)
    _model(root / INT4, int4_config(), ["model.safetensors", "model_mtp.safetensors"],
           tok=tokenizer_json("int4_differs") if tokenizer_mismatch else tok,
           index={"metadata": {"e24_mock": True}, "weight_map": {"model.language_model.embed_tokens.weight": "model.safetensors",
                                                               "lm_head.weight": "model.safetensors",
                                                               "mtp.fc.weight": "model_mtp.safetensors"}},
           extra={"recipe.yaml": "# E24 mock: W4A16 g128 (compressed-tensors)\n"})
    _model(root / DFLASH, DFLASH_CONFIG, ["model.safetensors"], extra={"README.md": "E24 mock DFlash2 drafter\n"})
    arc = root / ARC
    arc.mkdir(parents=True, exist_ok=True)
    src = None if arc_source == "synth" else (_arc_source() if arc_source == "auto" else Path(arc_source))
    if src is not None:
        for f in ARC_FILES:
            if (src / f).is_file() and (full_arc or "training" not in f):
                shutil.copyfile(src / f, arc / f)
    else:
        ch, sol = synth_arc()
        _wj(arc / "arc-agi_evaluation_challenges.json", ch, indent=None)
        _wj(arc / "arc-agi_evaluation_solutions.json", sol, indent=None)
        _wj(arc / "arc-agi_test_challenges.json", ch, indent=None)
        _wj(arc / "sample_submission.json", {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]]
                                             for k, v in ch.items()}, indent=None)
    if n_tasks:
        for f in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json", "arc-agi_test_challenges.json"):
            p = arc / f
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                _wj(p, {k: d[k] for k in sorted(d)[:n_tasks]}, indent=None)
    return {"root": str(root), "SGL_WHEELHOUSE": str(root / SGL_WH), "VLLM_WHEELHOUSE": str(root / VLLM_WH),
            "INT4": str(root / INT4), "DFLASH": str(root / DFLASH), "FP8": str(root / FP8), "ARC": str(arc),
            "arc_source": str(src) if src else "synthesized", "sglang_lock": "real E11-02" if lock is not SGL_PINS else "short",
            "tokenizer_mismatch": bool(tokenizer_mismatch)}


# ---- self-test (stdlib unittest-free; `--selftest` exits 1 on any failed check) ----
def _discover(root, marker, pred):
    """EMS `_e11_discover` semantics: every `marker` file whose JSON satisfies `pred` (the fallback discovery)."""
    out = []
    for p in sorted(Path(root).rglob(marker)):
        try:
            if pred(json.loads(p.read_text(encoding="utf-8"))):
                out.append(p.parent)
        except Exception:
            pass
    return out


def selftest():
    fails, n = [], 0

    def ok(label, cond):
        nonlocal n
        n += 1
        if not cond:
            fails.append(label)
        print(("PASS " if cond else "FAIL ") + label)
    with tempfile.TemporaryDirectory(prefix="e24_mock_") as td:
        info = build(Path(td) / "input")
        r = Path(info["root"])
        ok("all DESIGN §2 mount dirs exist", all((r / p).is_dir() for p in (SGL_WH, VLLM_WH, INT4, DFLASH, FP8, ARC)))
        ok("E23 find(): exactly one config.json under each model slug", all(
            len(list((r / "models" / OWNER / s).rglob("config.json"))) == 1
            for s in ("qwen3-8-27b-int4-redhat", "qwen3-8-27b-dflash2", "qwen3-8-27b-fp8-repacked-v1")))
        ok("EMS discovery: INT4 by compressed-tensors", _discover(r, "config.json", lambda c: (
            c.get("quantization_config") or {}).get("quant_method") == "compressed-tensors") == [r / INT4])
        ok("EMS discovery: DFlash2 by architectures", _discover(r, "config.json", lambda c: c.get("architectures") ==
                                                                  ["DFlash2DraftModel"]) == [r / DFLASH])
        ok("EMS discovery: SGLang wheelhouse by BUILD_INFO sglang 0.5.21 + lock", [
            p for p in _discover(r, "BUILD_INFO.json", lambda c: c.get("sglang") == "0.5.21") if (p / "requirements.lock").is_file()]
            == [r / SGL_WH])
        lock = (r / SGL_WH / "requirements.lock").read_text()
        ok("SGLang lock pins sglang==0.5.21", "sglang==0.5.21" in lock.splitlines())
        names = {l.split("  ", 1)[1].strip() for l in (r / SGL_WH / "SHA256SUMS").read_text().splitlines() if l.strip()}
        ok("SHA256SUMS keeps the '+' name; the file is stored '+'-stripped (Kaggle)",
           any("+" in x for x in names) and all((r / SGL_WH / x.replace("+", "")).is_file() for x in names))
        ok("every wheel sha256 matches SHA256SUMS", all(
            _sha((r / SGL_WH / l.split("  ", 1)[1].strip().replace("+", "")).read_bytes()) == l.split("  ", 1)[0]
            for l in (r / SGL_WH / "SHA256SUMS").read_text().splitlines() if l.strip()))
        import subprocess
        for wh in (SGL_WH, VLLM_WH):
            out = subprocess.run([sys.executable, str(r / wh / "install_wheelhouse.py"), str(Path(td) / "venv")],
                                 capture_output=True, text=True, timeout=60)
            kv = dict(l.split("=", 1) for l in out.stdout.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
            ok(f"{wh.split('/')[-1]} installer stub prints VENV_PYTHON / CUDA_HOME",
               out.returncode == 0 and kv.get("VENV_PYTHON") == sys.executable and kv.get("CUDA_HOME") == "/tmp")
        for m in (INT4, DFLASH, FP8):
            good = True
            for p in (r / m).glob("*.safetensors"):
                b = p.read_bytes()
                hl = struct.unpack("<Q", b[:8])[0]
                h = json.loads(b[8:8 + hl])
                good &= h["w"]["data_offsets"][1] == len(b) - 8 - hl and hl % 8 == 0
            ok(f"safetensors placeholders parse ({m.split('/')[2]})", good and any((r / m).glob("*.safetensors")))
            idx = r / m / "model.safetensors.index.json"
            if idx.exists():
                wm = json.loads(idx.read_text())["weight_map"]
                ok(f"index weight_map shards exist ({m.split('/')[2]})", all((r / m / s).is_file() for s in set(wm.values())))
        ft, it = (r / FP8 / "tokenizer.json").read_bytes(), (r / INT4 / "tokenizer.json").read_bytes()
        ok("INT4 and FP8 tokenizer.json byte-identical (R3K2 tokenizer check)", ft == it)
        ok("tokenizer.json is small (< 64 kB)", len(ft) < 64 * 1024)
        t = json.loads(ft)
        sp = {a["content"]: a["id"] for a in t["added_tokens"]}
        ok("special ids are the real ones", sp.get("<|im_start|>") == 248045 and sp.get("<|im_end|>") == 248046 and
           sp.get("<think>") == 248068 and sp.get("</think>") == 248069 and sp.get("<|endoftext|>") == 248044)
        ok("byte vocab covers all 256 bytes with distinct ids", sorted(t["model"]["vocab"].values()) == list(range(256)))
        try:
            import tokenizers  # noqa: F401  (absent locally; the Kaggle image has it)
            tk = tokenizers.Tokenizer.from_file(str(r / FP8 / "tokenizer.json"))
            ids = tk.encode("<|im_start|>user\nab 12<|im_end|>", add_special_tokens=False).ids
            ok("HF tokenizers loads it; specials are single ids", ids[0] == 248045 and ids[-1] == 248046)
        except ImportError:
            print("SKIP HF tokenizers load (not installed)")
        c4 = json.loads((r / INT4 / "config.json").read_text())
        ok("INT4 config is W4A16 g128", c4["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] == 128)
        ch = json.loads((r / ARC / "arc-agi_evaluation_challenges.json").read_text())
        te = json.loads((r / ARC / "arc-agi_test_challenges.json").read_text())

        def gridok(g):
            return isinstance(g, list) and g and all(isinstance(row, list) and row and all(
                isinstance(v, int) and 0 <= v <= 9 for v in row) for row in g)
        ok("ARC evaluation tasks valid (train pairs + test inputs, grids 0-9)", len(ch) >= 20 and all(
            v["train"] and v["test"] and all(gridok(p["input"]) and gridok(p["output"]) for p in v["train"]) and
            all(gridok(q["input"]) for q in v["test"]) for v in ch.values()))
        ok("ARC test challenges present", len(te) >= 20)
        info2 = build(Path(td) / "input2")
        ok("deterministic: identical trees give identical config/tokenizer bytes", all(
            (Path(info2[k]) / f).read_bytes() == (Path(info[k]) / f).read_bytes()
            for k in ("INT4", "FP8") for f in ("config.json", "tokenizer.json", "model.safetensors.index.json")))
        info3 = build(Path(td) / "input3", n_tasks=8, tokenizer_mismatch=True, arc_source="synth")
        ok("tokenizer_mismatch makes INT4 differ", (Path(info3["INT4"]) / "tokenizer.json").read_bytes() !=
           (Path(info3["FP8"]) / "tokenizer.json").read_bytes())
        sch = json.loads((Path(info3["ARC"]) / "arc-agi_evaluation_challenges.json").read_text())
        ok("synthesized ARC + n_tasks", len(sch) == 8 and all(gridok(q["input"]) for v in sch.values() for q in v["test"]))
        try:
            build(Path(td) / "input")
            ok("refuses a non-empty root", False)
        except FileExistsError:
            ok("refuses a non-empty root", True)
        print(json.dumps({k: info[k] for k in ("arc_source", "sglang_lock")}))
    print(f"mock_inputs selftest: {n - len(fails)}/{n} passed" + (f"; FAILED: {fails}" if fails else ""))
    return not fails


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    nt = int(sys.argv[sys.argv.index("--n-tasks") + 1]) if "--n-tasks" in sys.argv else None
    print(json.dumps(build(sys.argv[1], n_tasks=nt, tokenizer_mismatch="--tokenizer-mismatch" in sys.argv), indent=1))
