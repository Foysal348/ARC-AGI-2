"""E27 NBDF library (2026-11-01, v2): DFlash2 vs MTP K3 for Qwen3.8-Flash-Next W4 on Merlin v3.0.0 (4 x L4, TP2 x PP2 +
EP, FP8 target KV), on the Q-defer base of NB4T (nb4t_lib, embedded verbatim and imported here as T).

Arms (this order = priority; every arm runs on the nb4_ppack + dflash_seam patched tree, Q-defer on):
  Q    NB4T's Q: Merlin's own --speculative-config {"method":"mtp","num_speculative_tokens":3,"draft_sample_method":
       "probabilistic"} (launch.py:441, registry.py:49), Merlin's captures [1,2,3,4,6,8,12,16,24,32] (launch.py:459-460),
       no DFLASH_* env. The control.
  D4   Q + the DFlash change at K = 4: the seam env (DFLASH_SEAM=1, + the drafter host-embedding env when the seam has
       one) + a trailing --speculative-config {"method":"dflash","model":<drafter>,"num_speculative_tokens":4,
       "draft_sample_method":"probabilistic","kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"} + a trailing
       --compilation-config whose captures are the multiples of 1 + K up to 8 x (1 + K) (launch.py:110-111: extra args
       come last, the last occurrence wins). BF16 drafter KV on FA2 (verify A B6: FP8 drafter KV -> FlashInfer -> eager).
  D3   the same at K = 3 (captures 4 .. 32).
  D7   the same at K = 7 (captures 8 .. 64, block 8 as trained) + DFLASH_QSA_RING_WIDEN=1; only when the seam's ring
       widening applied AND verified (qsa_cache.py:960-965: 4096 % 12 != 0), else SKIPPED_NO_RING. Dropped first.
  D4b  optional, last: D4 with drafter KV fp8_e4m3 + TRITON_ATTN (verify A B6; upstream #54690 SM89 FP8 draft crash).
  <K>_ISO  one rerun of the FIRST failing D arm with NB4_PPDEFER unset (same patched tree; the env-off path = original):
       QDEFER_INTERACTION when it is clean, DFLASH_ITSELF when it fails too. The comparison stays with Q-defer on.

Per arm (the same items in the same order; every item gated by the boot deadline):
  read-backs (env, config, Q-defer exec / fallback share, DFlash: arch, speculator graphs, draft backend, aux taps
  [6, 16 | 26, 36, 46] on all 4 ranks, embedding, draft KV dtype, captures, block size, QSA ring capacity, memory per
  stage, pool / F vs Q) -> fail-fast smoke (2 streams x 2,048 tokens: acceptance > 0, no garbage) -> short oracle vs Q
  (greedy; its spec counters = the teacher-forced acceptance on Q's own tokens) -> AL early exit (D4: greedy AL <= Q's ->
  the remaining D windows / arms are skipped, recorded) -> long oracle (90k + 116k vs Q) -> prefix cold / warm 2-turn
  acceptance (warm AL < 0.8 x cold AL -> PREFIX_HIT_ACCEPTANCE_COLLAPSE) -> 8k@8 x 2 -> 40k -> 90k -> 116k at the arm's
  resident level (Q: 90k / 116k twice = A/A) -> profiled 8k (+ 116k when time is left).
  Every response (text + token ids) of every request is saved, gzipped jsonl per arm. Garbage under load is gated by
  the WINDOWED check (2,000-token windows, the klee100 lab); the stream-level LOW_VARIETY check is recorded only.

Pure logic, stdlib only (+ nb4t_lib). Unit-tested locally and on the Py3.12 CPU kit.
"""
import inspect
import json
import math
import re
import statistics

import nb4t_lib as T

LIB_VERSION = "nbdf_lib 2.1 (2026-11-01, arms Q > D4s > D3s > D4 > D4t > D7 > D4b + one Q-defer isolation rerun; DFlash2 vs MTP K3; gates on real failures only)"
ORDER = ("Q", "D4s", "D3s", "D4", "D4t", "D7", "D4b")
D_ARMS = ("D4s", "D3s", "D4", "D4t", "D7", "D4b")
OPTIONAL = ("D4", "D4t", "D4b")
SW_BLOCK_ARMS = ("D4s", "D3s", "D4t")                      # DFLASH_DRAFT_SW_BLOCK=1024 (kit v4 R9: 205 blocks vs 71 at 4096)
SW_BLOCK = 1024
SHORT_ARMS = ("D4", "D4t")                                  # smoke + greedy oracle + the two 8k@8 windows only
CONV_TORCH_ARMS = ("D4t",)                                  # D4s + DFLASH_CONV_IMPL=torch (vLLM #55960 / #58250; SGLang #41736)
CONV_BASE = "D4s"                                           # D4t is compared with this arm
K_OF = {"Q": 3, "D4s": 4, "D3s": 3, "D4": 4, "D4t": 4, "D7": 7, "D4b": 4}
ENV_SEAM = "DFLASH_SEAM"
ENV_RING = "DFLASH_QSA_RING_WIDEN"
SEAM_ENV_KEYS = ("DFLASH_SEAM", "DFLASH_TAP_MODE", "DFLASH_EMBED_SIDECAR", "DFLASH_EMBED_KEY", "DFLASH_MASK_ROW_SHA",
                 "DFLASH_DRAFT_EMBED_HOST", "DFLASH_LOAD_CHECK", "DFLASH_QSA_RING_WIDEN", "DFLASH_DRAFT_SW_BLOCK",
                 "DFLASH_NO_INGRAPH_META", "DFLASH_CONV_IMPL")   # = dflash_seam.ALL_ENVS
WATCHED = tuple(T.WATCHED) + SEAM_ENV_KEYS
WATCHED_DEFAULT = dict(T.WATCHED_DEFAULT)
SEQS = 8
PP = 2
TP = 2
NUM_LAYERS = 48
PARTITION = (25, 23)                                        # flash-next profile layout (launch.py:373-374)
Q_LABEL = T.Q_LABEL                                         # nb4_ppack readback arm name ("Q-defer")
Q_SPEC = {"method": "mtp", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"}   # launch.py:441
Q_CAPTURES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]            # registry.capture_sizes_for(8, 3)
DRAFT_SAMPLE = "probabilistic"                              # speculative.py:602 + speculator.py:246-258 (DFlash2: fp32 logits)
D_KV = {"D4s": "auto", "D3s": "auto", "D4": "auto", "D4t": "auto", "D7": "auto", "D4b": "fp8_e4m3"}
D_BACKEND = {"D4s": "FLASH_ATTN", "D3s": "FLASH_ATTN", "D4": "FLASH_ATTN", "D4t": "FLASH_ATTN", "D7": "FLASH_ATTN", "D4b": "TRITON_ATTN"}
ENV_SW_BLOCK = "DFLASH_DRAFT_SW_BLOCK"                      # lowers cache_config.block_size engine-wide (core.py:436-448): SW arms only
ENV_CONV = "DFLASH_CONV_IMPL"
# ONE constant: the drafter Kaggle Model (repack kernel foysalemonshanto/q27-e27-dflash2-embed-repack v364422784, REPACK OK;
# variation bf16-embed = incoai a22a1744 + the target embed_tokens, byte-identical for W4 and klee100)
DRAFTER_MODEL = "foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1"
DRAFTER_BASE = "qwen38-flash-next-dflash2v2"                           # under /kaggle/input/models/foysalemonshanto
DRAFTER_REL = "qwen38-flash-next-dflash2v2/transformers/bf16-embed/1/bf16-embed"   # the files sit in a bf16-embed/ subfolder
DRAFTER_VARIATION = "bf16-embed"                                       # resolve: exactly one dir with config.json + model.safetensors
DRAFTER_PINS = {
    "model.safetensors": {"size": 2477538608, "sha256": "d2e0464d15936265bb75eda9d09149d27ac6986327ca269a0335cd3d49661d49"},
    "config.json": {"size": 1236, "sha256": "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"},
    "embed_tokens": {"shape": [248320, 2560], "dtype": "BF16", "nbytes": 1271398400,
                     "sha256": "62288a3fca91d3575b68924a54ab73629b86b609284d3fcd059a309146c9be3a"},
    "mask_row": {"row": 248077, "norm": 0.26273465, "sha256": "860615491bdf6786178b8c14e96397f51e3895c556d0336e2b5ad8cc8e061a19"},
    "tensors": 82, "zero_rows": 0,
    "origin": {"repo": "incoai/Qwen3.8-Flash-Next-DFlash2", "revision": "a22a1744e85c17364dcc54a6a420af7135d4c7f6",
               "sha256_prefix": "d6723f2a", "sha256_suffix": "3186f", "tensors": 81}}
TARGET_LAYER_IDS = (5, 15, 25, 35, 45)                      # incoai dflash_config.target_layer_ids
AUX_LAYERS = tuple(i + 1 for i in TARGET_LAYER_IDS)         # eagle3_utils.py:108-111 (+1) -> (6, 16, 26, 36, 46)
PP0_TAPS = tuple(x for x in AUX_LAYERS if x <= PARTITION[0])        # (6, 16): captured on PP0, relayed
PP1_TAPS = tuple(x for x in AUX_LAYERS if x > PARTITION[0])         # (26, 36, 46)
MASK_TOKEN_ID = 248077
DRAFT_HIDDEN = 2560
DRAFT_LAYERS = 5
DRAFT_BLOCK = 8
DRAFT_KV_HEADS = 8
DRAFT_HEAD_DIM = 128
DRAFT_WINDOW = 2048
VOCAB = 248320
DRAFT_ARCH = "DFlash2DraftModel"
QSA_COMPRESS = 4
BLOCK_TOKENS = 4096
MAX_NUM_BATCHED = 1024                                      # profile max_num_batched_tokens (captures above it are dropped)
GIB = 1 << 30
MIB = 1 << 20
PIN_STEP = 268435456                                        # Merlin: lower KV_BYTES in 0.25 GiB steps
D_OOM_RETRIES = 2                                           # a D launch OOM: up to 2 more rungs (0.25 GiB each)
# PP1 memory plan per card (verify A B9 / verify B section 3, TP2 BF16; measured on GPU by "Model loading took")
DRAFTER_GPU_MIB = 745                                       # 0.727 GiB drafter weights without the embedding
EMBED_GPU_MIB = 606                                         # 0.592 GiB embed_tokens shard, unless on the host
MTP_FREED_MIB = 1024                                        # the MTP module not built on a D arm (verify B: 1.2-1.3 GiB)
PIN_MARGIN_MIB = 512                                        # = the memory gate (GPT analysis 3)
PIN_LADDER = [3553580524 - i * 268435456 for i in range(0, 6)]     # 3,553,580,524 / 3,285,145,068 / 3,016,709,612 / ...
FIRST_RUNG_MAX = 2                                          # the D arms start at rung 1 or 2; the memory gate steps further
MEM_GATE_MIB = 512                                          # every rank >= 512 MiB free after load / KV / graphs / first decode
CONC_LADDER = (2, 4, 7)                                     # others next to the probe (7 + the probe = 8 = max_num_seqs)
CONC_TOKENS = 256
SPEED_MIN = T.SPEED_MIN
CLEAN = T.CLEAN
START_RESERVE_SEC = T.START_RESERVE_SEC
BODY_RESERVE_SEC = T.BODY_RESERVE_SEC
HARD_STOP_MIN = 150                                          # user 2026-11-01: Q + D4 + D3 guaranteed
DEADMAN_MIN = 155                                            # absolute process kill (the K100Q pattern), 5 min after
LONG_BANDS = ("90k", "116k")
BAND_TOKENS = {"8k": 8192, "40k": 40000, "90k": 90000, "116k": 116000}
LONG_NTOK = 256                                              # greedy tokens per long-oracle row
LONG_VARIANT = 7                                             # long-oracle prompt variant (windows use 0 .. level-1 <= 6)
GEN_RATE_PER_STREAM = 40.0                                   # tok/s per stream used only to size the resident level
RESIDENT_MARGIN = 1.0
SMOKE_STREAMS, SMOKE_TOKENS = 2, 2048
SMOKE_MIN_ACCEPT = 0.02                                      # accepted / drafted < 2 % after >= 200 drafts = DFLASH_ZERO_ACCEPT
SMOKE_MIN_DRAFTS = 200                                       # (upstream #61072 / #60765: a pairing mismatch = exactly 0 %)
POOL_DROP_NOTE = 0.10                                        # D pool > 10 % below Q: noted in the verdict (selector workspace)
EARLY_EXIT_RATIO = 1.0                                       # greedy AL vs Q: a DIAGNOSTIC only (GPT audit B2: never skips arms)
SHORT_EXIT_ARMS = ("D4s",)                                   # 8k throughput rule: this arm's own 40k / 90k / 116k windows only
SW_START_RUNGS = 2                                           # SW arms start at the D pin: Q's 3,553,580,524 - 2 rungs = 3,016,709,612
SW_OOM_RETRIES = 1                                           # (kit v4 R9: 205 blocks there); one step-down to 2,748,274,156
PREFIX_WARM_MIN = 0.8                                        # warm AL < 0.8 x cold AL -> PREFIX_HIT_ACCEPTANCE_COLLAPSE
PREFIX_VARIANTS = (40, 41)                                   # 2 different prompts x 3 turns (upstream #47930 / #47926 class)
PREFIX_P_TOKENS = 6144                                       # + turn 1 (2048) fills block 2 with speculative-step KV
PREFIX_T1, PREFIX_T2, PREFIX_T3 = 2048, 512, 256             # turn 2 completes block 3 (the filler sizes it); turn 3 reuses it
WIN_TOK, WIN_MIN = 2000, 200                                 # windowed garbage check (klee100 lab window_garbage)
ISO_MAX = 1
EVENT_RX = re.compile(r"(?:E27_DFLASH|DFLASH_SEAM)\s+(\{.*\})\s*$")
_WTAG = re.compile(r"\(Worker_PP(\d+)_TP(\d+)")


# ------------------------------------------------------------------ idempotent wrappers (the kit 2026-11-01 RecursionError)
def wrap_once(holder, name, factory):
    """Replace holder.name (an object attribute, or a key when holder is a dict, e.g. globals()) by factory(original), ONCE:
    a second call (the same code executed twice) returns the existing wrapper instead of wrapping the wrapper (which made
    merlin_fn_server call itself: RecursionError in the real kit R4)."""
    is_d = isinstance(holder, dict)
    cur = holder[name] if is_d else getattr(holder, name)
    if getattr(cur, "_nbdf_wrapped", False):
        return cur
    new = factory(cur)
    new._nbdf_wrapped, new._nbdf_orig = True, cur
    if is_d:
        holder[name] = new
    else:
        setattr(holder, name, new)
    return new


# ------------------------------------------------------------------ captures (registry.py:83-89 + cudagraph_utils.py:332-357)
def capture_sizes(seqs, k):
    """Every multiple of 1 + K up to seqs x (1 + K) (user rule + verify A B8): every decode micro-batch b = 1 .. seqs runs
    in an exact FULL graph of the target AND of the drafter (FULL_DECODE_ONLY, no PIECEWISE fallback)."""
    return [b * (int(k) + 1) for b in range(1, int(seqs) + 1) if b * (int(k) + 1) <= MAX_NUM_BATCHED]


def v2_decode_graphs(captures, seqs, k):
    """Request counts with a FULL uniform-decode graph in the V2 runner: each capture size is rounded UP to a multiple of
    q = 1 + K, kept when <= seqs x q and <= the largest capture (cudagraph_utils.py:332-347)."""
    q = int(k) + 1
    caps = sorted(set(int(c) for c in captures))
    if not caps:
        return []
    out = set()
    for c in caps:
        rt = -(-c // q) * q
        if rt <= seqs * q and rt <= caps[-1] and rt // q <= seqs:
            out.add(rt // q)
    return sorted(out)


def coverage(captures, seqs, k):
    g = v2_decode_graphs(captures, seqs, k)
    rows = []
    for b in range(1, int(seqs) + 1):
        r = next((x for x in g if x >= b), None)
        rows.append({"b": b, "tokens": b * (k + 1), "graph_reqs": r, "exact": r == b})
    return {"k": k, "captures": sorted(captures), "graph_reqs": g, "covers_all": all(r["graph_reqs"] for r in rows),
            "padded_b": [r["b"] for r in rows if r["graph_reqs"] and not r["exact"]],
            "uncovered_b": [r["b"] for r in rows if not r["graph_reqs"]]}


def qsa_ring(k, block=BLOCK_TOKENS, compress=QSA_COMPRESS, widen=False):
    """qsa_cache.py:960-965 (+ the seam's widening, PR #54912 semantics): (ok, capacity)."""
    cap = compress * math.ceil((compress + int(k)) / compress)
    if block % cap == 0:
        return True, cap
    if widen:
        c = cap + compress
        while c <= min(block, 2 * cap):
            if block % c == 0:
                return True, c
            c += compress
    return False, cap


# ------------------------------------------------------------------ spec / compilation strings
def dflash_spec(model_path, k, kv_dtype="auto", backend="FLASH_ATTN"):
    return {"method": "dflash", "model": str(model_path), "num_speculative_tokens": int(k), "draft_sample_method": DRAFT_SAMPLE,
            "kv_cache_dtype": kv_dtype, "attention_backend": backend}


def spec_json(d):
    return json.dumps(d, separators=(",", ":"))


def compilation_json(captures):
    """The exact shape Merlin passes (launch.py:459-460)."""
    return '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[' + ",".join(str(int(c)) for c in captures) + "]}"


def captures_from_argv(argv):
    v = T.last_value(argv, "--compilation-config")
    if v is None:
        return None
    try:
        return list(json.loads(v).get("cudagraph_capture_sizes") or [])
    except Exception:
        return None


# ------------------------------------------------------------------ the seam module (dflash_seam.py, parallel agent)
_NAMES = {"apply": ("apply_dflash_patch", "apply"), "verify": ("verify_dflash_patch", "verify"),
          "revert": ("revert_dflash_patch", "revert")}


def _has_kw(fn, *names):
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    return next((n for n in names if n in ps), None)


def seam_api(mod):
    """The interface NBDF codes against (nb4_ppack style): apply(tree, py, ring) -> record (raises on a miss),
    verify(tree, ring) -> {"state": "applied" | "not_applied" | "foreign" | "mixed", ...}, revert(tree, py, ring); plus
    server_env(**kw) and readback_label when present. Accepts apply_dflash_patch / verify_dflash_patch /
    revert_dflash_patch (the module as built) or apply / verify / revert."""
    fns, missing = {}, []
    for role, names in _NAMES.items():
        f = next((getattr(mod, n) for n in names if callable(getattr(mod, n, None))), None)
        if f is None:
            missing.append(role)
        fns[role] = f
    se = getattr(mod, "server_env", None) if mod is not None else None
    se = se if callable(se) else None
    caps = {"ring_kw": bool(fns["apply"] and _has_kw(fns["apply"], "ring")),
            "server_env": bool(se),
            "sidecar_kw": _has_kw(se, "embed_sidecar", "sidecar") if se else None,
            "host_kw": _has_kw(se, "embed_host", "host_embed", "draft_embed_host", "host") if se else None,
            "mask_kw": _has_kw(se, "mask_row_sha") if se else None,
            "conv_kw": _has_kw(se, "conv_impl") if se else None,
            "sw_kw": _has_kw(se, "draft_sw_block") if se else None,
            "draft_block_kw": bool(fns["apply"] and _has_kw(fns["apply"], "draft_block")),
            "tree_state": callable(getattr(mod, "tree_state", None)) if mod is not None else False,
            "ring_env_kw": _has_kw(se, "ring") if se else None,
            "readback_label": callable(getattr(mod, "readback_label", None)) if mod is not None else False,
            "parse_log": callable(getattr(mod, "parse_log", None)) if mod is not None else False,
            "env_names": {n: getattr(mod, n) for n in ("ENV_SEAM", "ENV_EMBED", "ENV_EMBED_KEY", "ENV_RING", "ENV_EMBED_HOST",
                                                       "ENV_HOST", "ENV_TAP") if isinstance(getattr(mod, n, None), str)} if mod is not None else {}}
    return {"ok": not missing, "missing": missing, "fns": fns, "server_env_fn": se, "caps": caps}


def seam_env_for(api, k, embed_in_ckpt, sidecar_path=None, host_embed=False, mask_row_sha=None, conv_impl=None, draft_sw_block=None):
    """The seam env of one D arm: DFLASH_SEAM=1; DFLASH_MASK_ROW_SHA (row 248077 of the served embedding, asserted at
    load); the sidecar only when the drafter checkpoint ships no embed_tokens (the seam refuses both); the ring env at
    K >= 5; the drafter host embedding only when asked (off: its UVA path is GPU-only and untested). Returns (env, notes)."""
    notes = []
    se = (api or {}).get("server_env_fn")
    caps = (api or {}).get("caps") or {}
    kw = {}
    ring = not qsa_ring(k)[0]
    if se is None:
        env = {ENV_SEAM: "1"}
        if ring:
            env[ENV_RING] = "1"
        notes.append("seam has no server_env(): DFLASH_SEAM=1 (+ ring) only")
        return env, notes
    if caps.get("ring_env_kw"):
        kw[caps["ring_env_kw"]] = ring
    if not embed_in_ckpt:
        if caps.get("sidecar_kw") and sidecar_path:
            kw[caps["sidecar_kw"]] = str(sidecar_path)
        else:
            notes.append("drafter has no embed_tokens and no sidecar option / path: the boot will raise (loud)")
    if draft_sw_block and caps.get("sw_kw"):
        kw[caps["sw_kw"]] = int(draft_sw_block)
    if conv_impl and caps.get("conv_kw"):
        kw[caps["conv_kw"]] = conv_impl
    elif conv_impl:
        notes.append("seam server_env has no conv_impl kw")
    if mask_row_sha and caps.get("mask_kw"):
        kw[caps["mask_kw"]] = mask_row_sha
    elif mask_row_sha:
        notes.append("seam server_env has no mask_row_sha kw: the mask row is not asserted at load")
    if host_embed and caps.get("host_kw"):
        kw[caps["host_kw"]] = True
    elif host_embed:
        notes.append("seam offers no drafter host-embedding option: the embedding stays on the GPU (pin plan accounts it)")
    env = {str(a): str(b) for a, b in dict(se(**kw)).items()}
    if ring and env.get(ENV_RING) != "1":
        env[ENV_RING] = "1"
        notes.append("ring env added by NBDF (server_env has no ring kw)")
    env.setdefault(ENV_SEAM, "1")
    if conv_impl and env.get(ENV_CONV) != conv_impl:
        env[ENV_CONV] = conv_impl
    if draft_sw_block:
        env[ENV_SW_BLOCK] = str(int(draft_sw_block))  # SW arms only (engine-wide block size: read back + flagged)
    else:
        env.pop(ENV_SW_BLOCK, None)
    notes.append(f"server_env kwargs {sorted(kw)}")
    return env, notes


def host_embed_on(env):
    return any(k in env and str(env[k]).strip() == "1" for k in ("DFLASH_DRAFT_EMBED_HOST",))


def tree_state(verify_state):
    """Three-state tree check (GPT analysis 2 item 4): ORIGINAL / FULLY_PATCHED / INVALID_MIXED."""
    s = str((verify_state or {}).get("state") if isinstance(verify_state, dict) else verify_state)
    return {"applied": "FULLY_PATCHED", "not_applied": "ORIGINAL"}.get(s, "INVALID_MIXED")


class SeamSeq:
    """dflash_seam on the installed tree: applied once after nb4_ppack (ring=True first; a failure is reverted and the core
    retried with ring=False -> D7 SKIPPED_NO_RING), verified before every boot, reverted first in the finally block
    (before nb4_ppack). A failed apply -> every D arm NOT_INTEGRATED; Q runs."""

    def __init__(self, apply_fn, verify_fn, revert_fn):
        self.apply_fn, self.verify_fn, self.revert_fn = apply_fn, verify_fn, revert_fn
        self.state, self.log, self.error, self.ring = "pristine", [], None, None
        self.apply_rec = self.verify_rec = self.revert_rec = None

    def apply(self):
        if self.state == "applied":
            return True, "already applied"
        if self.state != "pristine":
            return False, f"NOT_INTEGRATED: seam {self.state}: {self.error}"
        errs = []
        for ring in (True, False):
            try:
                self.apply_rec = self.apply_fn(ring)
                self.verify_rec = self.verify_fn(ring)
                if tree_state(self.verify_rec) != "FULLY_PATCHED":
                    raise RuntimeError("SEAM_VERIFY_AFTER_APPLY: " + json.dumps(self.verify_rec, default=str)[:400])
                self.state, self.ring = "applied", ring
                self.log.append(("apply", "ring" if ring else "core"))
                return True, "applied" + (" (ring)" if ring else " (core only: no ring widening)")
            except Exception as e:
                errs.append(f"ring={ring}: {type(e).__name__}: {e}"[:400])
                self.log.append(("apply_failed", "ring" if ring else "core"))
                try:
                    self.revert_rec = self.revert_fn(ring)
                    self.log.append(("revert_after_failed_apply", ring))
                except Exception as e2:
                    self.log.append(f"revert_after_failed_apply_failed: {e2!r}"[:200])
                    self.state, self.error = "revert_failed", "; ".join(errs) + f"; revert: {e2!r}"[:300]
                    return False, "NOT_INTEGRATED: " + self.error
        self.state, self.error = "failed", "; ".join(errs)[:800]
        return False, "NOT_INTEGRATED: " + self.error

    def check(self, tag):
        """Before a boot: the seam files are FULLY_PATCHED (three-state check)."""
        if self.state != "applied":
            return False, f"NOT_INTEGRATED: seam {self.state}: {self.error}"
        try:
            v = self.verify_fn(self.ring)
        except Exception as e:
            return False, f"NOT_INTEGRATED: seam verify raised {type(e).__name__}: {e}"[:300]
        st = tree_state(v)
        self.log.append(("verify", tag, st))
        if st != "FULLY_PATCHED":
            return False, f"SEAM_TREE_{st}: " + json.dumps(v, default=str)[:300]
        return True, "FULLY_PATCHED"

    def finish(self):
        if self.state == "applied":
            try:
                self.revert_rec = self.revert_fn(self.ring)
                self.state = "reverted"
                self.log.append("revert")
            except Exception as e:
                self.state, self.error = "revert_failed", f"{type(e).__name__}: {e}"[:500]
                self.log.append("revert_failed")
        return {"state": self.state, "log": self.log, "error": self.error}


def ring_status(seq):
    """(available, verified, why) from the seam sequence: the ring patch applied with ring=True AND verified."""
    if seq is None or seq.state != "applied":
        return False, False, f"seam not applied ({getattr(seq, 'state', None)})"
    if not seq.ring:
        return False, False, "seam applied without the ring widening (ring=True apply failed)"
    files = (seq.verify_rec or {}).get("files") or {}
    ring_files = [f for f in files if f.endswith("qsa_cache.py")]
    if ring_files and all((files[f] or {}).get("state") == "applied" for f in ring_files):
        return True, True, "ring widening applied and verified (qsa_cache.py post-sha + marker)"
    return True, False, "ring=True apply, but verify shows no applied qsa_cache.py"


# ------------------------------------------------------------------ arm plan
def arm_plan(base_extra, base_env, kv_bytes, drafter_path, q_env=None, seam_envs=None, ports=(8011, 8091, 8081, 8021, 8061, 8041, 8051)):
    """Q = NB4T's Q. D4 / D3 / D7 / D4b = Q + the ONE DFlash change: the seam env of that arm (seam_envs[key], default
    DFLASH_SEAM=1 (+ ring at K >= 5)) + the trailing [--speculative-config dflash, --compilation-config multiples of
    1 + K]. kv_bytes = the planned pin (D arms: replaced at run time by pin_plan after Q)."""
    q_env = dict(q_env or {T.ENV_DEFER: "1"})
    qenv = dict(base_env, **q_env)
    title = {"Q": "NB4T Q: FP8 KV, pin 3,553,580,524 B, text-only, #60490, MTP K3, seqs 8, profiler + Q-defer (nb4_ppack rev 2)",
             "D4s": "Q + DFlash2 K = 4 (BF16 drafter KV, FLASH_ATTN) + DFLASH_DRAFT_SW_BLOCK=1024, captures 5 .. 40 (primary)",
             "D3s": "Q + DFlash2 K = 3 (BF16 drafter KV, FLASH_ATTN) + DFLASH_DRAFT_SW_BLOCK=1024, captures 4 .. 32 (primary)",
             "D4": "Q + DFlash2 K = 4 (BF16 drafter KV at block 4096: the 71-block control; smoke + oracle + 8k x 2 only)",
             "D3": "Q + DFlash2 K = 3 (BF16 drafter KV, FLASH_ATTN), captures 4 .. 32",
             "D4t": "D4s + DFLASH_CONV_IMPL=torch (the pre-#55960 grouped conv; smoke + oracle + 8k x 2 only; optional)",
             "D7": "Q + DFlash2 K = 7 (block 8 as trained; QSA ring widening), captures 8 .. 64",
             "D4b": "D4 with FP8 drafter KV + TRITON_ATTN (optional, last)"}
    est = {"Q": 50, "D4s": 41, "D3s": 40, "D4": 22, "D4t": 22, "D7": 42, "D4b": 40}   # full minimum (min): launch + gates + the
    cap = {"Q": 41, "D4s": 33, "D3s": 32, "D4": 15, "D4t": 15, "D7": 34, "D4b": 32}   # 2 x 3 prefix test + windows + teardown
    arms = {}
    for i, key in enumerate(ORDER):
        k = K_OF[key]
        env, extra = dict(qenv), list(base_extra)
        if key == "Q":
            spec, ch, caps = dict(Q_SPEC), None, list(Q_CAPTURES)
            extra += list(Q_TAIL)
        else:
            caps = capture_sizes(SEQS, k)
            spec = dflash_spec(drafter_path, k, D_KV[key], D_BACKEND[key])
            senv = dict((seam_envs or {}).get(key) or ({ENV_SEAM: "1", ENV_RING: "1"} if not qsa_ring(k)[0] else {ENV_SEAM: "1"}))
            if key in CONV_TORCH_ARMS:
                senv[ENV_CONV] = "torch"
            if key in SW_BLOCK_ARMS:
                senv[ENV_SW_BLOCK] = str(SW_BLOCK)
            else:
                senv.pop(ENV_SW_BLOCK, None)
            env.update(senv)
            add = ["--speculative-config", spec_json(spec), "--compilation-config", compilation_json(caps)]
            extra += add
            ch = {"kind": "dflash", "env": senv, "argv": add, "k": k}
        expect = {w: env.get(w, WATCHED_DEFAULT.get(w)) for w in WATCHED}
        ok_r, ring_cap = qsa_ring(k, widen=key == "D7")
        arms[key] = {"key": key, "name": f"nbdf_{key.lower()}", "port": ports[i], "extra": extra, "env": env, "build_env": {},
                     "change": ch, "expect": expect, "k": k, "method": spec["method"], "spec": spec,
                     "captures": caps, "coverage": coverage(caps, SEQS, k), "ring_needed": not qsa_ring(k)[0],
                     "ring_capacity_expected": ring_cap,
                     "expect_cfg": {"max_num_seqs": SEQS, "partition": expect[T.ENV_PART], "language_model_only": True,
                                    "num_speculative_tokens": k, "method": spec["method"], "model": spec.get("model"),
                                    "kv_cache_dtype": spec.get("kv_cache_dtype"), "attention_backend": spec.get("attention_backend"),
                                    "captures": caps, "draft_head": None},
                     "streams": SEQS, "windows": _wkeys(), "long_bands": LONG_BANDS, "long_aa": key == "Q",
                     "kv_bytes": int(kv_bytes), "needs_patch": True, "needs_seam": key != "Q", "q_label": Q_LABEL,
                     "est_min": est[key], "cap_min": cap[key], "title": title[key], "lever": key != "Q",
                     "optional": key in OPTIONAL, "short_only": key in SHORT_ARMS, "sw_block": SW_BLOCK if key in SW_BLOCK_ARMS else None,
                     "iso": False}
    return arms


def variant_arm(arm, key_suffix, name_suffix, port_offset, env_add=None, env_del=(), title_add="", **kw):
    """One single-retry variant of a D arm (same patched tree, same spec / captures / pin rules): env keys added / removed."""
    a = json.loads(json.dumps(arm))
    for k_ in env_del:
        a["env"].pop(k_, None)
    a["env"].update(env_add or {})
    a["expect"] = {w: a["env"].get(w, WATCHED_DEFAULT.get(w)) for w in WATCHED}
    a.update(key=arm["key"] + key_suffix, name=arm["name"] + name_suffix, port=arm["port"] + port_offset, iso=True, parent=arm["key"],
             long_aa=False, title=arm["title"] + " | " + title_add)
    a.update(kw)
    return a


def iso_arm(arm, port_offset=5):
    """The Q-defer isolation rerun of a failed D arm: the same arm with NB4_PPDEFER removed (same patched tree)."""
    return variant_arm(arm, "_ISO", "_noqd", port_offset, env_del=(T.ENV_DEFER,), title_add="Q-defer OFF (isolation rerun)", q_label=None)


def nometa_arm(arm, port_offset=7):
    """vLLM #63587 (DFlash acceptance ~0 under FULL graphs, fixed by PR #64059): the one relaunch with DFLASH_NO_INGRAPH_META=1."""
    return variant_arm(arm, "_NOMETA", "_nometa", port_offset, env_add={"DFLASH_NO_INGRAPH_META": "1"},
                       title_add="DFLASH_NO_INGRAPH_META=1 (INGRAPH_META_FALLBACK)")


def ingraph_meta(log_text, tp=TP):
    """The seam's 'E27_DFLASH {"event": "ingraph_meta", ...}' lines (PR #64059 port) on the PP1 workers."""
    ev = [e for e in seam_events(log_text) if e.get("event") == "ingraph_meta"]
    pp1 = [e for e in ev if str(e.get("worker") or f"PP{PP - 1}").startswith(f"PP{PP - 1}")]
    return {"lines": len(ev), "pp1_lines": len(pp1), "all_pp1": len({e.get("pid") for e in pp1}) >= tp,
            "kernel_map_refresh": any(e.get("kernel_map_refresh") is True for e in ev), "events": ev[:4]}


def needs_nometa(rec):
    """One INGRAPH_META_FALLBACK relaunch: the arm hit DFLASH_ZERO_ACCEPT while kernel_map_refresh was true."""
    rec = rec or {}
    return "ZERO_ACCEPT" in str(rec.get("smoke_gate") or "") and (rec.get("ingraph_meta") or {}).get("kernel_map_refresh") is True


FINE_GRAINED_RX = "Disabling fine-grained prefix-cache hits"
DRAFT_GROUP_WARN_RX = "no KV cache group could be identified as the draft model"   # kv_cache_utils.py:2122-2151 (vLLM #57032)
_STAGE_GRAPHS = re.compile(r"\(Worker_PP(\d+)_TP\d+[^)]*\).*?Capturing (?:(\w+) )?CUDA graphs \((\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)")


def stage_graphs(log_text):
    """Target CUDA graphs per PP stage and mode ('Capturing CUDA graphs (MODE): 100%| n/N' on a Worker_PPx line; the drafter's
    'dflash2' bars are kept apart) + 'Graph capturing finished' lines per stage (seam open risk 2: the compiled aux relay)."""
    out = {}
    for pp, name, mode, n, tot in _STAGE_GRAPHS.findall(log_text or ""):
        key = f"PP{pp}" + ("" if not name else f"_{name.lower()}")
        out.setdefault(key, {})[mode] = max(int(tot), out.get(key, {}).get(mode, 0))
    fin = re.findall(r"\(Worker_PP(\d+)_TP\d+[^)]*\).*?Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", log_text or "")
    out["finished"] = sorted({(f"PP{a}", int(b), float(c)) for a, b, c in fin})[:8]
    return out


def f_blocks(pool, mc, block, max_len=131072):
    """F (fixed blocks per request) from the max-concurrency line at THIS boot's block size (nb4t_lib assumes 4096).
    None unless pool > 0, mc > 0 and finite, block a positive int and 0 <= F <= pool (GPT final audit HIGH-1: the kit's
    4096-block pool / mc with block 1024 gave F = -85)."""
    ok = (isinstance(pool, (int, float)) and not isinstance(pool, bool) and pool > 0 and isinstance(mc, (int, float))
          and not isinstance(mc, bool) and math.isfinite(mc) and mc > 0 and isinstance(block, int) and not isinstance(block, bool) and block > 0)
    if not ok:
        return None
    f = math.ceil(round(pool / mc - max_len / block, 3))
    return f if 0 <= f <= pool else None


CONSERVATIVE_LEVEL = {"40k": 4, "90k": 2, "116k": 2}       # CAPACITY_UNVERIFIED: no optimistic 8 / 6 / 5 fallback


def f_from_layout(layout, block, rank="2"):
    """A conservative F from THIS boot's KV layout read-back (the drafter rank): 2 blocks per request for every state group
    (Mamba / CircularBuffer: the current + the align checkpoint) + ceil(window / block) + 1 for a sliding-window group."""
    g = ((layout or {}).get("groups") or {}).get(str(rank))
    if not isinstance(g, list) or not g or not isinstance(block, int) or block <= 0:
        return None
    f = 0
    for spec in g:
        name = str(spec[0] if isinstance(spec, (list, tuple)) else spec)
        if "Sliding" in name:
            f += math.ceil(DRAFT_WINDOW / block) + 1
        elif "Full" not in name and "MLA" not in name:
            f += 2
    return f or None


def level_for_band(band, band_tokens, pool, mc, block, layout=None, prefill_tok_s=700.0, win_sec=90.0, seqs=None):
    """(level, info) for one long band, from THIS boot only. F = the boot's max-concurrency line + its num_gpu_blocks at its
    block size (validated); else the layout estimate; else the conservative level, labelled CAPACITY_UNVERIFIED (windows
    then count only when running == level without preemption). A valid F with no room -> level 0: SKIPPED_CAPACITY."""
    seqs = SEQS if seqs is None else seqs   # read at call time (E27 K100: SEQS 24)
    info = {"band": band, "block": block, "pool": pool, "mc": mc}
    fa = (((layout or {}).get("block_sizes") or {}).get("FullAttentionSpec") or [None])[0]
    if isinstance(fa, int) and fa > 0 and isinstance(block, int) and fa != block:
        # third final audit: the engine block (e.g. 1024 on SW arms) differs from the target FullAttention / MLA block (4096):
        # the long-context KV lives in the target groups, so F and the per-request blocks use the target block first
        info.update(mixed="MIXED_KV_BLOCK_SIZES", engine_block=block, target_block=fa)
        f = f_blocks(pool, mc, fa)
        src = "boot max-concurrency line + num_gpu_blocks at the TARGET FullAttention block (MIXED_KV_BLOCK_SIZES)"
        if f is not None:
            lv = plan_level(pool, f, band_tokens, prefill_tok_s, win_sec, seqs, block=fa)
            if not lv:
                return 0, dict(info, source=src, F=f, level=0, block=fa, skip="SKIPPED_CAPACITY")
            return lv, dict(info, source=src, F=f, level=lv, block=fa)
        f = f_from_layout(layout, fa)
        if f is not None:
            src = "kv layout read-back estimate at the target block (MIXED_KV_BLOCK_SIZES, conservative)"
            lv = plan_level(pool, f, band_tokens, prefill_tok_s, win_sec, seqs, block=fa) if isinstance(pool, (int, float)) and pool > 0 else None
            if lv:
                return lv, dict(info, source=src, F=f, level=lv, block=fa)
            if lv == 0:
                return 0, dict(info, source=src, F=f, level=0, block=fa, skip="SKIPPED_CAPACITY")
        lv = CONSERVATIVE_LEVEL.get(band, 1)
        return lv, dict(info, source="CAPACITY_UNVERIFIED", F=None, level=lv)
    f = f_blocks(pool, mc, block)
    src = "boot max-concurrency line + num_gpu_blocks (its block size)"
    if f is None:
        f = f_from_layout(layout, block)
        src = "kv layout read-back estimate (conservative)"
    if f is None or not (isinstance(pool, (int, float)) and pool > 0):
        lv = CONSERVATIVE_LEVEL.get(band, 1)
        return lv, dict(info, source="CAPACITY_UNVERIFIED", F=None, level=lv)
    lv = plan_level(pool, f, band_tokens, prefill_tok_s, win_sec, seqs, block=block)
    if not lv:
        return 0, dict(info, source=src, F=f, level=0, skip="SKIPPED_CAPACITY")
    return lv, dict(info, source=src, F=f, level=lv)


ADMISSION_WEDGE_SEC = 120.0                                  # vLLM #55800: running 0 with waiting > 0 for > 120 s
ADMISSION_PROBE_ARMS = ("D4s",)
ADMISSION_PROBE_TOKENS = (16384, 40000)


class AdmissionWatch:
    """Fires ONCE when the server's metrics show running == 0 with waiting > 0 continuously for more than `limit` s
    (a SWA drafter's full-sequence KV requirement wedging admission, upstream vLLM #55800). feed(t, running, waiting, kv)
    returns True when it fired; on_fire(info) is called once (the notebook stops the item's live streams)."""

    def __init__(self, limit=ADMISSION_WEDGE_SEC, on_fire=None, label=""):
        self.limit, self.on_fire, self.label = float(limit), on_fire, label
        self.since, self.fired, self.info, self.samples = None, False, None, 0

    def feed(self, t, running, waiting, kv=None):
        self.samples += 1
        if self.fired:
            return True
        wedged = (running or 0) == 0 and (waiting or 0) > 0
        if not wedged:
            self.since = None
            return False
        self.since = t if self.since is None else self.since
        if t - self.since > self.limit:
            self.fired = True
            self.info = {"status": "ADMISSION_DEADLOCK", "item": self.label, "wedged_sec": round(t - self.since, 1), "running": running,
                         "waiting": waiting, "kv_usage": kv}
            if self.on_fire:
                self.on_fire(self.info)
            return True
        return False


def capacity_window_status(w, unverified):
    """A window run at a CAPACITY_UNVERIFIED level counts only when valid (STEADY, running == level, 0 preemptions);
    otherwise INCONCLUSIVE_CAPACITY (never a speed number for the verdict)."""
    if not unverified or not isinstance(w, dict) or str(w.get("status", "")).startswith("SKIPPED"):
        return w
    w["capacity_unverified"] = True
    if not T.window_ok(w)[0]:
        w["status_before_capacity_rule"] = w.get("status")
        w["status"] = "INCONCLUSIVE_CAPACITY"
    return w


def engine_block(log_text, layout=None):
    """The effective attention block size: 'Setting attention block size to N tokens' (last), else the FullAttentionSpec block."""
    m = re.findall(r"Setting attention block size to (\d+) tokens", log_text or "")
    if m:
        return int(m[-1])
    b = ((layout or {}).get("block_sizes") or {}).get("FullAttentionSpec") or []
    return b[0] if b else None


def pp0_graph_degraded(q_sg, x_sg):
    """PP0_GRAPH_DEGRADED (recorded): the D arm's PP0 target FULL graphs fewer than Q's, or none (eager)."""
    qf, xf = ((q_sg or {}).get("PP0") or {}).get("FULL"), ((x_sg or {}).get("PP0") or {}).get("FULL")
    if xf is None and qf is None:
        return {"flag": None, "why": "no per-stage graph lines"}
    flag = "PP0_GRAPH_DEGRADED" if (not xf) or (isinstance(qf, int) and xf < qf) else None
    return {"flag": flag, "q_pp0_full": qf, "x_pp0_full": xf}


def qdefer_k_fallback(q_rec, x_rec):
    """QDEFER_K_FALLBACK (recorded; verify B section 4 / vLLM #57755): the D arm's max Q-defer fallback share above Q's."""
    def mx(r):
        sh = ((T.latest_q(r or {}) or {}).get("shares") or {}) if isinstance(T.latest_q(r or {}), dict) else {}
        v = [x.get("fallback_share") for x in sh.values() if isinstance(x, dict) and isinstance(x.get("fallback_share"), (int, float))]
        return max(v) if v else None
    q, x = mx(q_rec), mx(x_rec)
    return {"flag": "QDEFER_K_FALLBACK" if (q is not None and x is not None and x > q) else None, "q_max_fallback": q, "x_max_fallback": x}


def preempt_accept(w, k, spec_lines):
    """M4 (draft context after preemption / recompute): a window with preemptions > 0 records its SpecDecoding acceptance
    series; Merlin logs no preemption time, so 'before / after' = the window's first vs last SpecDecoding interval
    (APPROXIMATION, labelled). PREEMPT_ACCEPT_DROP when after < 0.8 x before (recorded)."""
    if not isinstance(w, dict) or not (w.get("preemptions") or 0) > 0:
        return None
    t0, t1 = w.get("wall_t0"), w.get("wall_t1")
    sel = [x for x in spec_lines or [] if (t0 is None or (x["ts"] or "") >= t0) and (t1 is None or (x["ts"] or "") <= t1)]
    rates = [round(x["accepted"] / x["drafted"], 4) for x in sel if x["drafted"]]
    out = {"preemptions": w.get("preemptions"), "accept_rate_series": rates[:40],
           "basis": "first vs last SpecDecoding interval of the window (no preemption timestamp in Merlin's log)"}
    if len(rates) >= 2:
        out.update(before=rates[0], after=rates[-1], flag="PREEMPT_ACCEPT_DROP" if rates[-1] < 0.8 * rates[0] else None)
    return out


def prefix_reuse(rec, log_text=None):
    """vLLM #53786: the SlidingWindowManager warning (kv_cache_utils.py:779-790), the mean prefix-hit rate of the steady
    windows and of every prefix-test turn."""
    rec = rec or {}
    rates = [w.get("prefix_hit_rate") for w in T._win(rec).values() if isinstance(w, dict) and isinstance(w.get("prefix_hit_rate"), (int, float))]
    turns = {}
    for pr in ((rec.get("prefix_cache") or {}).get("prompts") or []):
        for t, tt in (pr.get("turns") or {}).items():
            for side in ("cold", "warm"):
                turns[f"{pr.get('variant')}/{t}/{side}"] = ((tt.get(side) or {}).get("prefix") or {}).get("rate")
    return {"fine_grained_disabled": (FINE_GRAINED_RX in log_text) if log_text is not None else rec.get("fine_grained_disabled"),
            "window_rate_mean": round(statistics.mean(rates), 4) if rates else None, "windows": len(rates), "turn_rates": turns}


def prefix_reuse_drop(Q, X, pts=0.10):
    q, x = prefix_reuse(Q).get("window_rate_mean"), prefix_reuse(X).get("window_rate_mean")
    if q is None or x is None:
        return {"flag": None, "q": q, "x": x}
    return {"flag": "PREFIX_REUSE_DROP" if q - x > pts else None, "q": q, "x": x, "drop_points": round(q - x, 4)}


def check_arm_plan(arms):
    """Fail closed (ValueError): order; never-do; Q = base env + NB4_PPDEFER=1 (no NB4_PPACK / CDMC / partition / DFLASH_*),
    Merlin's own spec + captures; each D arm = Q + exactly the DFlash change (DFLASH_* env only, DFLASH_SEAM=1, the ring env
    only at K >= 5; the trailing spec + compilation pair, nothing else); spec fields; exact FULL graphs for every decode
    batch; the largest capture = 8 x (1 + K); pins (D <= Q); ports distinct."""
    errs = []
    q = arms.get("Q") or {}
    if tuple(arms) != ORDER:
        errs.append(f"order {tuple(arms)} != {ORDER}")
    qe = q.get("env") or {}
    if qe.get(T.ENV_DEFER) != "1" or any(w in qe for w in (T.ENV_PACK, T.FLAG_CDMC, T.ENV_PART) + SEAM_ENV_KEYS):
        errs.append("Q env must carry NB4_PPDEFER=1 and none of NB4_PPACK / CUDA_DEVICE_MAX_CONNECTIONS / partition / DFLASH_*")
    if "--language-model-only" not in (q.get("extra") or []):
        errs.append("Q lacks --language-model-only")
    qx = list(q.get("extra") or [])
    qbase = qx[:len(qx) - len(Q_TAIL)] if Q_TAIL else qx
    if Q_TAIL and qx[len(qx) - len(Q_TAIL):] != list(Q_TAIL):
        errs.append("Q's trailing argv != Q_TAIL")
    if "--speculative-config" in qx or (not Q_TAIL and "--compilation-config" in qx):
        errs.append("Q must keep Merlin's own spec and captures (no trailing override; K100: only Q_TAIL's captures)")
    for key, a in arms.items():
        v = T.never_do_violations(a["env"], a["extra"])
        if v:
            errs.append(f"{key}: never-do {v}")
        if a["kv_bytes"] > q.get("kv_bytes", 0):
            errs.append(f"{key}: pin above Q's")
        if not a.get("needs_patch") or a.get("q_label") != Q_LABEL:
            errs.append(f"{key}: every arm runs Q-defer on the patched tree")
        if a["captures"][-1] != SEQS * (a["k"] + 1):
            errs.append(f"{key}: largest capture {a['captures'][-1]} != {SEQS} x {a['k'] + 1}")
        if not a["coverage"]["covers_all"]:
            errs.append(f"{key}: decode batches {a['coverage']['uncovered_b']} have no FULL graph")
        if a["expect"].get(T.ENV_PART) != "25,23":
            errs.append(f"{key}: partition {a['expect'].get(T.ENV_PART)}")
        if key == "Q":
            continue
        if a["coverage"]["padded_b"]:
            errs.append(f"{key}: padded decode batches {a['coverage']['padded_b']} (captures must be exact multiples)")
        ch = a.get("change") or {}
        ed = {x for x in set(a["env"]) | set(qe) if a["env"].get(x) != qe.get(x)}
        ok = (ch.get("kind") == "dflash" and ed and ed <= set(SEAM_ENV_KEYS) and a["env"].get(ENV_SEAM) == "1"
              and a["extra"][:len(qbase)] == qbase and a["extra"][len(qbase):] == ch.get("argv") and not a["build_env"])
        if not ok:
            errs.append(f"{key}: differs from Q by env {sorted(ed)} / flags beyond the DFlash pair")
        if (a["env"].get(ENV_SW_BLOCK) == str(SW_BLOCK)) != (key in SW_BLOCK_ARMS) or (ENV_SW_BLOCK in a["env"] and key not in SW_BLOCK_ARMS):
            errs.append(f"{key}: {ENV_SW_BLOCK} {a['env'].get(ENV_SW_BLOCK)} ({SW_BLOCK} only on {SW_BLOCK_ARMS}: engine-wide block size)")
        if (a["env"].get(ENV_CONV) == "torch") != (key in CONV_TORCH_ARMS):
            errs.append(f"{key}: {ENV_CONV} {a['env'].get(ENV_CONV)} (torch only on {CONV_TORCH_ARMS})")
        if (a["env"].get(ENV_RING) == "1") != a["ring_needed"]:
            errs.append(f"{key}: ring env {a['env'].get(ENV_RING)} but ring_needed {a['ring_needed']}")
        sp = T.spec_from_argv(a["extra"]) or {}
        want = {"method": "dflash", "num_speculative_tokens": K_OF[key], "draft_sample_method": DRAFT_SAMPLE,
                "kv_cache_dtype": D_KV[key], "attention_backend": D_BACKEND[key]}
        if any(sp.get(x) != y for x, y in want.items()) or not sp.get("model") or set(sp) != set(want) | {"model"}:
            errs.append(f"{key}: spec {sp}")
        if captures_from_argv(a["extra"]) != a["captures"]:
            errs.append(f"{key}: trailing captures {captures_from_argv(a['extra'])} != plan {a['captures']}")
        if K_OF[key] + 1 > DRAFT_BLOCK:
            errs.append(f"{key}: K {K_OF[key]} needs block_size > {DRAFT_BLOCK}")
    if len({a["port"] for a in arms.values()}) != len(arms):
        errs.append("ports not distinct")
    if errs:
        raise ValueError("ARM_PLAN_INVALID: " + "; ".join(errs))
    return True


def launch_check(arm, env, argv, kv_bytes=None):
    """On the FULL final env / argv of a launch (fail closed): never-do; every watched env key; the effective (last)
    --speculative-config == the arm's spec; the effective captures; the last --max-num-seqs; --language-model-only; the pin."""
    bad = T.never_do_violations(env, argv)
    if ENV_SW_BLOCK in (env or {}) and (arm.get("parent") or arm["key"]) not in SW_BLOCK_ARMS:
        bad.append(f"{arm['key']} {ENV_SW_BLOCK} set on a non-SW launch (engine-wide block size)")
    for w, want in arm["expect"].items():
        have = (env or {}).get(w)
        if have != want:
            bad.append(f"{arm['key']} launch env {w}={have!r}, expected {want!r}")
    sp = T.spec_from_argv(argv)
    want_sp = dict(arm["spec"], draft_lm_head_quant=arm["draft_head"]) if arm.get("draft_head") and arm.get("method") != "dflash" else dict(arm["spec"])
    if arm.get("method") == "dflash" and ((env or {}).get("DRAFT_HEAD") or (isinstance(sp, dict) and sp.get("draft_lm_head_quant"))):
        bad.append(f"{arm['key']} S4: a quantized draft head (DRAFT_HEAD={(env or {}).get('DRAFT_HEAD')!r}, spec "
                   f"draft_lm_head_quant={(sp or {}).get('draft_lm_head_quant') if isinstance(sp, dict) else None!r}) on a dflash spec")
    if not isinstance(sp, dict) or sp.get("error"):
        bad.append(f"{arm['key']} --speculative-config missing / unparsable: {sp}")
    elif sp != want_sp:
        bad.append(f"{arm['key']} spec {sp} != {want_sp}")
    caps = captures_from_argv(argv)
    if caps != arm["captures"]:
        bad.append(f"{arm['key']} captures {caps} != {arm['captures']}")
    if T.last_value(argv, "--max-num-seqs") != str(arm["expect_cfg"]["max_num_seqs"]):
        bad.append(f"{arm['key']} --max-num-seqs {T.last_value(argv, '--max-num-seqs')}")
    if "--language-model-only" not in [str(x) for x in (argv or ())]:
        bad.append(f"{arm['key']} --language-model-only missing")
    if kv_bytes is not None and T.last_value(argv, "--kv-cache-memory-bytes") != str(int(kv_bytes)):
        bad.append(f"{arm['key']} --kv-cache-memory-bytes {T.last_value(argv, '--kv-cache-memory-bytes')}, expected {kv_bytes}")
    return bad


# ------------------------------------------------------------------ the D pin (PP1 memory plan)
def draft_logits_mib(k, seqs=None):
    """fp32 draft_logits [max_num_reqs, K, vocab] (speculator.py:246-258, DFlash2 fp32: dflash2/speculator.py:200-204)."""
    seqs = SEQS if seqs is None else seqs   # read at call time (E27 K100: SEQS 24)
    return seqs * int(k) * VOCAB * 4 / MIB


def pin_plan(q_pin, q_free_pp1_mib, k, host_embed, drafter_gpu_mib=DRAFTER_GPU_MIB, embed_gpu_mib=EMBED_GPU_MIB,
             mtp_freed_mib=MTP_FREED_MIB, margin_mib=PIN_MARGIN_MIB):
    """The D arm's first pin: Q's final pin lowered by whole 0.25 GiB rungs so that the PP1 cards keep the margin:
    need = drafter + (embed unless host) + draft_logits + margin - MTP freed - Q's free MiB on PP1 after ready. At least
    one rung (verify B: '+0.03 to +0.12 GiB'), at most FIRST_RUNG_MAX (the ladder 3,285,145,068 / 3,016,709,612 from Q's
    3,553,580,524); a launch OOM or the 512 MiB memory gate steps further down (recorded)."""
    free = q_free_pp1_mib if isinstance(q_free_pp1_mib, (int, float)) else 0.0
    need = drafter_gpu_mib + (0 if host_embed else embed_gpu_mib) + draft_logits_mib(k) + margin_mib - mtp_freed_mib - free
    want = max(1, math.ceil(max(0.0, need) * MIB / PIN_STEP))
    rungs = min(FIRST_RUNG_MAX, want)
    return {"pin": int(q_pin) - rungs * PIN_STEP, "rungs": rungs, "rungs_formula": want, "need_mib": round(need, 1),
            "q_free_pp1_mib": q_free_pp1_mib, "ladder": [int(q_pin) - i * PIN_STEP for i in range(0, FIRST_RUNG_MAX + D_OOM_RETRIES + 1)],
            "host_embed": bool(host_embed), "draft_logits_mib": round(draft_logits_mib(k), 1),
            "terms_mib": {"drafter": drafter_gpu_mib, "embed": 0 if host_embed else embed_gpu_mib, "margin": margin_mib,
                          "mtp_freed": -mtp_freed_mib}}


# ------------------------------------------------------------------ server-log read-backs
def log_facts(log_text):
    """nb4t_lib.log_config + the effective captures of the non-default args line."""
    f = T.log_config(log_text)
    na = next((l for l in (log_text or "").splitlines() if "non-default args:" in l), "")
    m = re.search(r"'cudagraph_capture_sizes': \[([\d, ]*)\]", na)
    f["capture_sizes"] = [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
    return f


def config_readback(arm, log_text, kv_bytes=None):
    """The server log must show the arm's config (fail closed for a positive verdict); definitive = a value contradicts
    the plan (the arm stops: nothing to measure); a missing line is not definitive."""
    f = log_facts(log_text)
    ec = arm["expect_cfg"]
    checks, contra = {}, []

    def chk(name, have, want):
        if have is None:
            checks[name] = False
            return
        checks[name] = have == want
        if not checks[name]:
            contra.append(f"{name} {have!r} != {want!r}")
    chk("max_num_seqs", f["max_num_seqs"], ec["max_num_seqs"])
    chk("language_model_only", f["language_model_only"], True)
    if kv_bytes is not None:
        chk("kv_cache_memory_bytes", f["kv_cache_memory_bytes"], int(kv_bytes))
    sp = f["speculative_config"]
    if isinstance(sp, dict):
        chk("spec_method", sp.get("method"), ec["method"])
        chk("spec_k", sp.get("num_speculative_tokens"), ec["num_speculative_tokens"])
        chk("spec_draft_sample", sp.get("draft_sample_method"), DRAFT_SAMPLE)
        if ec["method"] == "dflash":
            chk("spec_model", sp.get("model"), ec["model"])
            chk("spec_draft_kv", sp.get("kv_cache_dtype"), ec["kv_cache_dtype"])
            chk("spec_attention_backend", sp.get("attention_backend"), ec["attention_backend"])
    else:
        checks["spec_method"] = checks["spec_k"] = False
    chk("captures", f["capture_sizes"], ec["captures"])
    want_head = arm.get("draft_head") if arm.get("method") != "dflash" else None
    checks["no_draft_head_quant"] = bool(f["draft_head_any"]) if want_head else not f["draft_head_any"]
    if f["draft_head_any"] and not want_head:
        contra.append("a quantized draft head was built")
    want = {"0": [(0, PARTITION[0])], "1": [(PARTITION[0], NUM_LAYERS)]}
    sl = f["stage_layers"]
    if sl:
        checks["partition"] = {k: [tuple(x) for x in v] for k, v in sl.items()} == want
        if not checks["partition"]:
            contra.append(f"stage layers {sl} != {want}")
    else:
        checks["partition"] = False
    ok = all(checks.values())
    return {"ok": ok, "definitive": bool(contra), "checks": checks, "contradictions": contra,
            "why": "ok" if ok else ("; ".join(contra) if contra else "missing: " + ", ".join(k for k, v in checks.items() if not v)),
            "log": {k: f[k] for k in ("max_num_seqs", "language_model_only", "kv_cache_memory_bytes", "speculative_config",
                                      "capture_sizes", "stage_layers", "block_bytes_per_stage")}}


_TS = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)")
_SPEC_LINE = re.compile(r"SpecDecoding metrics: Mean acceptance length: ([\d.]+), Accepted throughput: ([\d.]+) tokens/s, "
                        r"Drafted throughput: ([\d.]+) tokens/s, Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
                        r"Per-position acceptance rate: ([\d., ]+), Avg Draft acceptance rate: ([\d.]+|nan)%")


def spec_decoding_lines(log_text):
    """Every 'SpecDecoding metrics' line (v1/spec_decode/metrics.py:121-138)."""
    out = []
    for l in (log_text or "").splitlines():
        m = _SPEC_LINE.search(l)
        if not m:
            continue
        ts = _TS.search(l)
        out.append({"ts": ts.group(1) if ts else None, "al": float(m.group(1)), "accepted": int(m.group(4)),
                    "drafted": int(m.group(5)), "per_pos": [float(x) for x in m.group(6).split(",") if x.strip()],
                    "rate_pct": None if m.group(7) == "nan" else float(m.group(7))})
    return out


def spec_summary(lines, k, t0=None, t1=None, min_drafted=1):
    """Draft-weighted aggregate of SpecDecoding lines (optionally t0 <= ts <= t1, 'MM-DD HH:MM:SS')."""
    sel = [x for x in lines if x["drafted"] >= min_drafted and (t0 is None or (x["ts"] or "") >= t0) and (t1 is None or (x["ts"] or "") <= t1)]
    if not sel:
        return {"lines": 0}
    bad_pos = [x["per_pos"] for x in sel if len(x["per_pos"]) != k]
    drafts = [x["drafted"] / k for x in sel]
    nd = sum(drafts)
    acc = sum(x["accepted"] for x in sel)
    per = [round(sum(x["per_pos"][i] * d for x, d in zip(sel, drafts) if len(x["per_pos"]) == k) / nd, 4) for i in range(k)] if nd else None
    return {"lines": len(sel), "drafted": sum(x["drafted"] for x in sel), "accepted": acc, "al": round(1 + acc / nd, 3) if nd else None,
            "per_pos": per, "positions_ok": not bad_pos, "bad_positions": bad_pos[:3], "t0": t0, "t1": t1}


def acceptance_delta(a0, a1, k):
    """Acceptance between two accept_pos() snapshots ({"pos": {p: accepted}, "drafts": n}): AL, per position, counts."""
    if not a0 or not a1:
        return {"status": "NO_METRICS"}
    d = (a1.get("drafts") or 0.0) - (a0.get("drafts") or 0.0)
    per = {int(p): (a1["pos"].get(p, 0.0) - (a0.get("pos") or {}).get(p, 0.0)) for p in (a1.get("pos") or {})}
    acc = sum(per.values())
    if d <= 0:
        return {"status": "NO_DRAFTS", "drafts": d, "accepted": acc}
    return {"status": "OK", "drafts": round(d, 1), "accepted": round(acc, 1), "drafted_tokens": round(d * k, 1),
            "al": round(1 + acc / d, 4), "accept_rate": round(acc / (d * k), 4),
            "per_pos": [round(per.get(i, 0.0) / d, 4) for i in range(k)]}


def seam_events(log_text):
    """'E27_DFLASH {json}' lines (dflash_seam.py, every worker; vLLM prefixes worker stdout with '(Worker_PPx_TPy ...)')."""
    out = []
    for l in (log_text or "").splitlines():
        m = EVENT_RX.search(l)
        if not m:
            continue
        try:
            e = json.loads(m.group(1))
        except ValueError:
            out.append({"bad": l[-200:]})
            continue
        w = _WTAG.search(l)
        if isinstance(e, dict):
            if w:
                e.setdefault("worker", f"PP{w.group(1)}_TP{w.group(2)}")
            out.append(e)
    return out


_BACKEND_SEL = re.compile(r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+) backend\.")
_BACKEND_AUTO = re.compile(r"Using ([A-Z0-9_]+) attention backend out of potential backends")
_GRAPHS = re.compile(r"Capturing (?:(\w+) )?CUDA graphs \((\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)")


def backend_lines(log_text):
    """Explicitly selected backends ('Using X backend.', cuda.py:446: only a selected backend, i.e. the drafter's
    attention_backend) and auto selections (cuda.py:507), with the worker tag."""
    sel, auto = [], []
    for l in (log_text or "").splitlines():
        w = _WTAG.search(l)
        wk = f"PP{w.group(1)}_TP{w.group(2)}" if w else None
        m = _BACKEND_SEL.search(l)
        if m:
            sel.append({"backend": m.group(1), "worker": wk})
        m = _BACKEND_AUTO.search(l)
        if m:
            auto.append({"backend": m.group(1), "worker": wk})
    return {"selected": sel, "auto": auto}


def graph_counts(log_text):
    """'Capturing [<name> ]CUDA graphs (<MODE>): 100%| n/N' (cudagraph_utils.py:471; the drafter's bar is 'Capturing
    dflash2 CUDA graphs', dflash/speculator.py:170): {target|<name>: {MODE: [N, ...]}}."""
    out = {}
    for name, mode, n, tot in _GRAPHS.findall(log_text or ""):
        out.setdefault((name or "target").lower(), {}).setdefault(mode, []).append(int(tot))
    return out


def dflash_readback(arm, log_text, plan_env=None, tp=TP):
    """Read-backs of a D arm from the server log (fail closed for a positive verdict; definitive = a contradiction):
      arch         'Resolved architecture: DFlash2DraftModel' (config/model.py:735), no MTP head built
      graphs       'Capturing model for DFlash2 speculator' (dflash/speculator.py:153) and no eager warning (:133-138);
                   the drafter's FULL_DECODE_ONLY graph count recorded ('Capturing dflash2 CUDA graphs (FULL)')
      backend      'Using <attention_backend> backend.' on the tp PP1 workers (the drafter's explicit selection)
      aux          4 'aux_layers' events (distinct pids) with ids (6, 16, 26, 36, 46); PP0 local (6, 16) slot base 0, PP1
                   local (26, 36, 46) slot base 2, upstream 2 (order [6, 16 | 26, 36, 46]); tap shapes when logged
      embed        tp 'embed' events (distinct pids, PP1): embed loaded own / shared, head shared_target, abs mean > 0
      sidecar      'embed_sidecar' events only when the plan set DFLASH_EMBED_SIDECAR
      ring         'qsa_ring' event with this K when the arm needs the widening
      spec lines   every SpecDecoding line has K positions"""
    t = log_text or ""
    env = plan_env if plan_env is not None else arm.get("env") or {}
    checks, contra, rec = {}, [], {}
    archs = re.findall(r"Resolved architecture: (\w+)", t)
    rec["archs"] = archs
    checks["arch_dflash2"] = DRAFT_ARCH in archs
    if any("MTP" in a for a in archs):
        contra.append(f"an MTP draft model was resolved on a DFlash arm: {archs}")
    caps = t.count("Capturing model for DFlash2 speculator")
    eager = len(re.findall(r"draft attention \(.*\) does not support full CUDA graphs", t))
    rec["speculator_capture_lines"], rec["speculator_eager_lines"] = caps, eager
    checks["drafter_full_graphs"] = caps >= 1 and eager == 0
    if eager:
        contra.append("the drafter runs EAGER (its attention backend has no full CUDA graphs)")
    rec["graph_counts"] = graph_counts(t)
    bl = backend_lines(t)
    rec["backend_selected"], rec["backend_auto"] = bl["selected"][:8], sorted({x["backend"] for x in bl["auto"]})
    pp1_sel = [x for x in bl["selected"] if str(x.get("worker") or "").startswith(f"PP{PP - 1}")]
    want_b = arm["expect_cfg"].get("attention_backend")
    checks["draft_backend"] = len({x["worker"] for x in pp1_sel if x["backend"] == want_b}) >= tp
    if pp1_sel and any(x["backend"] != want_b for x in pp1_sel):
        contra.append(f"draft backend {sorted({x['backend'] for x in pp1_sel})} != {want_b}")
    ev = seam_events(t)
    rec["seam_events"] = len(ev)
    rec["bad_events"] = [e for e in ev if "bad" in e][:3]
    aux = [e for e in ev if e.get("event") == "aux_layers"]
    rec["aux"] = [{k: e.get(k) for k in ("worker", "pid", "ids", "local", "start_layer", "end_layer", "slot_base", "upstream_total")}
                  for e in aux][:8]
    pids = {e.get("pid") for e in aux}
    checks["aux_all_ranks"] = len(aux) == PP * tp and len(pids) == PP * tp
    checks["aux_ids"] = bool(aux) and all(tuple(e.get("ids") or ()) == AUX_LAYERS for e in aux)
    if aux and not checks["aux_ids"]:
        contra.append(f"aux ids {sorted({tuple(e.get('ids') or ()) for e in aux})} != {AUX_LAYERS}")
    st0 = [e for e in aux if e.get("start_layer") == 0]
    st1 = [e for e in aux if e.get("start_layer") not in (0, None)]
    checks["aux_order_pp0_6_16"] = len(st0) == tp and all(tuple(e.get("local") or ()) == PP0_TAPS and e.get("slot_base") in (0, None)
                                                          and e.get("upstream_total") in (0, None) for e in st0)
    checks["aux_order_pp1_26_36_46"] = (len(st1) == tp and all(tuple(e.get("local") or ()) == PP1_TAPS for e in st1)
                                        and all(e.get("slot_base") == len(PP0_TAPS) and e.get("upstream_total") == len(PP0_TAPS) for e in st1))
    checks["tap_mode_combine"] = bool(aux) and all(e.get("tap_mode") in ("combine", None) for e in aux)
    if (st0 and not checks["aux_order_pp0_6_16"] and any(tuple(e.get("local") or ()) != PP0_TAPS for e in st0)) or \
       (st1 and any(tuple(e.get("local") or ()) != PP1_TAPS for e in st1)):
        contra.append(f"tap split {[(e.get('start_layer'), e.get('local')) for e in aux]} != [6, 16 | 26, 36, 46]")
    shp = [e for e in ev if e.get("event") in ("aux_check", "aux_shapes", "aux_shape")]
    rec["aux_shape_events"] = shp[:4]
    if shp:
        good = all(all(list(s)[-1:] == [DRAFT_HIDDEN] for s in (e.get("shapes") or [])) and len(e.get("shapes") or []) in (len(PP0_TAPS), len(PP1_TAPS), len(AUX_LAYERS))
                   for e in shp)
        checks["aux_shapes_2560"] = good
        if not good:
            contra.append("aux tap shapes are not [T, 2560]")
    else:
        rec["aux_shapes_note"] = "the seam logs no tap shapes; its _e27_check_aux raises on a count mismatch and fc 12800 on width"
    emb = [e for e in ev if e.get("event") == "embed"]
    rec["embed"] = emb[:4]
    checks["embed_last_rank"] = (len(emb) == tp and len({e.get("pid") for e in emb}) == tp
                                 and all(str(e.get("worker") or f"PP{PP - 1}").startswith(f"PP{PP - 1}") for e in emb))
    checks["embed_nonzero"] = bool(emb) and all(isinstance(e.get("sample_abs_mean"), (int, float)) and e["sample_abs_mean"] > 0 for e in emb)
    checks["head_shared"] = bool(emb) and all(e.get("head_source") == "shared_target" for e in emb)
    checks["embed_own_loaded"] = bool(emb) and all(e.get("embed_source") == "own" for e in emb)
    holders = [e for e in emb if isinstance(e.get("mask_row"), dict) and e["mask_row"].get("on_this_rank")]
    rec["mask_row"] = [e["mask_row"] for e in holders]
    want_mask = env.get("DFLASH_MASK_ROW_SHA") or DRAFTER_PINS["mask_row"]["sha256"]
    checks["mask_row_sha"] = len(holders) == 1 and holders[0]["mask_row"].get("sha256") == want_mask
    if holders and holders[0]["mask_row"].get("sha256") not in (None, want_mask):
        contra.append(f"mask row {holders[0]['mask_row'].get('sha256')} != {want_mask}")
    rel = [((e.get("released_bytes") or {}).get("lm_head") or 0) for e in emb]
    rec["released_mib"] = [e.get("released_mib") for e in emb]
    checks["b5_lm_head_released"] = bool(emb) and all(x > 0 for x in rel)
    checks["embed_host_as_planned"] = bool(emb) and all(bool(e.get("host")) == (env.get("DFLASH_DRAFT_EMBED_HOST") == "1") for e in emb)
    lc = [e for e in ev if e.get("event") == "draft_load_check"]
    rec["draft_load_check"] = lc[:4]
    lc_pids = {e.get("pid") for e in lc if e.get("pid") is not None}
    lc_counts = all(not isinstance(e.get("expected"), int) or e.get("loaded") == e.get("expected") for e in lc)
    checks["draft_load_check"] = (bool(lc) and len(lc_pids) >= tp and all(not e.get("skipped") for e in lc) and lc_counts
                                  and all(str(e.get("worker") or f"PP{PP - 1}").startswith(f"PP{PP - 1}") for e in lc))
    rec["linears_unquantized"] = "asserted in-engine before the embed line (UnquantizedLinearMethod or RuntimeError)"
    if emb and any(isinstance(e.get("sample_abs_mean"), (int, float)) and e["sample_abs_mean"] <= 0 for e in emb):
        contra.append("the drafter embedding is ZERO")
    side = [e for e in ev if e.get("event") == "embed_sidecar"]
    want_side = bool(env.get("DFLASH_EMBED_SIDECAR"))
    checks["sidecar_as_planned"] = (len(side) == tp) if want_side else not side
    rec["sidecar"] = len(side)
    if arm.get("ring_needed"):
        ring = [e for e in ev if e.get("event") == "qsa_ring" and e.get("num_speculative_tokens") == arm["k"]]
        rec["ring_events"] = ring[:2]
        checks["ring_widened"] = bool(ring) and all(e.get("capacity") == arm.get("ring_capacity_expected") for e in ring)
    swb = [e for e in ev if e.get("event") == "draft_sw_block"]
    rec["draft_sw_block"] = [(e.get("layer"), e.get("block_size"), e.get("draft_block")) for e in swb][:6]
    if env.get(ENV_SW_BLOCK):
        checks["draft_sw_block_applied"] = bool(swb) and all(e.get("draft_block") == int(env[ENV_SW_BLOCK]) for e in swb)
    elif swb:
        contra.append(f"draft_sw_block lines on an arm without {ENV_SW_BLOCK}")
    ci = [e for e in ev if e.get("event") == "conv_impl"]
    rec["conv_impl"] = [e.get("impl") for e in ci][:4]
    if env.get(ENV_CONV) == "torch":
        checks["conv_impl_torch"] = len(ci) >= tp and all(e.get("impl") == "torch" for e in ci)
    elif ci and any(e.get("impl") not in (None, "fused") for e in ci):
        contra.append(f"conv_impl {rec['conv_impl']} on a fused arm")
    sl = spec_decoding_lines(t)
    rec["spec_lines"] = len(sl)
    checks["spec_positions_k"] = bool(sl) and all(len(x["per_pos"]) == arm["k"] for x in sl)
    if sl and not checks["spec_positions_k"]:
        contra.append(f"SpecDecoding lines with {sorted({len(x['per_pos']) for x in sl})} positions, K {arm['k']}")
    ok = all(checks.values()) and not contra
    return {"ok": ok, "definitive": bool(contra), "checks": checks, "contradictions": contra,
            "why": "ok" if ok else ("; ".join(contra) if contra else "missing: " + ", ".join(k for k, v in checks.items() if not v)),
            "rec": rec}


def merge_seam_label(rb, label):
    """dflash_seam.readback_label (label, detail) joins the DFlash read-back as one more check (ENGAGED required)."""
    rb = dict(rb or {})
    ch = dict(rb.get("checks") or {})
    lab = (list(label)[0] if isinstance(label, (list, tuple)) and label else None)
    ch["seam_readback_label"] = lab == "ENGAGED"
    rb["checks"], rb["seam_label"] = ch, list(label) if isinstance(label, (list, tuple)) else label
    rb["ok"] = all(ch.values())
    if not rb["ok"] and rb.get("why") == "ok":
        rb["why"] = "missing: seam_readback_label"
    elif not ch["seam_readback_label"] and "seam_readback_label" not in str(rb.get("why")):
        rb["why"] = str(rb.get("why")) + "; seam readback_label " + str(lab)
    return rb


def q_seam_inert(log_text):
    """Q on the seam-patched tree WITHOUT DFLASH_SEAM: no DFlash2 model, no seam events (env-off = original code)."""
    t = log_text or ""
    ev = seam_events(t)
    archs = re.findall(r"Resolved architecture: (\w+)", t)
    ok = DRAFT_ARCH not in archs and not ev
    return {"ok": ok, "archs": archs, "events": ev[:4], "why": "ok" if ok else "DFlash parts active on Q"}


def kv_layout_readback(workers, arm=None):
    """From the NB1 worker extension (nb1_cache_info: kv_cache_specs + kv_cache_config per rank): the effective KV block
    sizes per spec type, the QSA ring capacity (CircularBufferSpec block_size), the drafter's KV spec (SlidingWindowSpec
    or a non-language_model layer: dtype, block, heads), the group layout and pool per rank. Do not assume 4096."""
    ranks = (workers or {}).get("ranks") or []
    out = {"block_sizes": {}, "ring_capacity": [], "drafter_specs": [], "groups": {}, "num_blocks": {}}
    for w in ranks:
        if not isinstance(w, dict):
            continue
        r = w.get("rank")
        for s in w.get("kv_cache_specs") or []:
            if not isinstance(s, dict):
                continue
            ty = str(s.get("type"))
            out["block_sizes"].setdefault(ty, set()).add(s.get("block_size"))
            if ty == "CircularBufferSpec":
                out["ring_capacity"].append(s.get("block_size"))
            ex = str(s.get("example") or "")
            if "Sliding" in ty or not (ex.startswith("language_model") or ex.startswith("mtp") or not ex):
                out["drafter_specs"].append({"rank": r, "type": ty, "dtype": s.get("dtype"), "block_size": s.get("block_size"),
                                             "num_kv_heads": s.get("num_kv_heads"), "head_size": s.get("head_size"),
                                             "layers": s.get("layers"), "example": ex[:80], "page_size_bytes": s.get("page_size_bytes")})
        kc = w.get("kv_cache_config") or {}
        out["groups"][str(r)] = [(g.get("spec"), g.get("layers"), g.get("block_size"), g.get("page_size_bytes")) for g in kc.get("groups") or []]
        out["num_blocks"][str(r)] = kc.get("num_blocks", w.get("num_gpu_blocks"))
    out["block_sizes"] = {k: sorted(x for x in v if x is not None) for k, v in out["block_sizes"].items()}
    out["ring_capacity"] = sorted({x for x in out["ring_capacity"] if x is not None})
    out["drafter_kv_dtype"] = sorted({str(d["dtype"]) for d in out["drafter_specs"] if d.get("dtype")})
    if arm is not None:
        out["ring_expected"] = arm.get("ring_capacity_expected")
        out["ring_ok"] = out["ring_capacity"] == [arm.get("ring_capacity_expected")] if out["ring_capacity"] else None
        if arm.get("method") == "dflash":
            want = "bfloat16" if arm["expect_cfg"].get("kv_cache_dtype") == "auto" else "uint8"
            out["drafter_kv_ok"] = (all(want in x for x in out["drafter_kv_dtype"]) if out["drafter_kv_dtype"] else None)
    return out


def memory_stages(rec):
    """Memory per stage and rank (GPT analysis 1 item 3; explicit kv_cache_memory_bytes skips Merlin's profiling):
    initial free / reserved, 'Model loading took' GiB + sec, graph capture GiB, free MiB after ready, after the smoke
    (first decode) and at the end (min of nvidia-smi and the worker)."""
    rec = rec or {}
    p = rec.get("pool") or {}
    return {"initial_free_reserved_gib": p.get("initial_free_reserved_gib"), "model_load_gib": p.get("rank_load_gib"),
            "model_load_sec": p.get("rank_load_sec"), "graph_capture_gib": p.get("rank_graph_gib"),
            "free_after_ready_mib": (rec.get("free_after_ready_min") or {}).get("min"),
            "free_after_smoke_mib": (rec.get("free_after_smoke_min") or {}).get("min"),
            "free_end_mib": (rec.get("free_end_min") or {}).get("min"),
            "kv_lines": [l for l in ((rec.get("census") or {}).get("startup_lines") or []) if "KV cache" in l][:6],
            "memory_log_lines": rec.get("memory_log_lines")}


def capacity_compare(q_rec, d_rec):
    """Pool / F / max concurrency / block bytes / load GiB / free of D vs Q (read before any speed comparison: K = 7 grows
    the GDN / RecoverSSM / PLE state, the drafter KV joins the pool)."""
    q, d = T.capacity_facts(q_rec), T.capacity_facts(d_rec)
    keys = ("num_gpu_blocks", "kv_tokens", "max_concurrency_131072", "F_blocks", "block_bytes_per_stage", "pin", "rank_load_gib",
            "free_after_ready_min_mib")
    out = {"q": {k: q.get(k) for k in keys}, "d": {k: d.get(k) for k in keys}}
    for k in ("kv_tokens", "max_concurrency_131072", "F_blocks", "pin"):
        if isinstance(q.get(k), (int, float)) and isinstance(d.get(k), (int, float)):
            out[f"delta_{k}"] = round(d[k] - q[k], 3)
    qb, db = q.get("num_gpu_blocks") or [], d.get("num_gpu_blocks") or []
    if len(qb) == 1 and len(db) == 1:
        out["delta_blocks"] = db[0] - qb[0]
        out["pool_drop"] = round(1 - db[0] / qb[0], 4) if qb[0] else None
    qs, ds = q.get("block_bytes_per_stage"), d.get("block_bytes_per_stage")
    if isinstance(qs, list) and isinstance(ds, list) and len(qs) == len(ds):
        out["delta_block_bytes_per_stage"] = [b - a for a, b in zip(qs, ds)]
    ql, dl = q.get("rank_load_gib") or {}, d.get("rank_load_gib") or {}
    if ql and dl:
        out["delta_load_gib_per_rank"] = {r: round(dl[r] - ql[r], 3) for r in ql if r in dl}
    return out


# ------------------------------------------------------------------ drafter checkpoint facts (stdlib)
def safetensors_keys(path):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        if n <= 0 or n > 200 * 2 ** 20:
            raise ValueError(f"bad safetensors header length {n}")
        h = json.loads(fh.read(n).decode("utf-8"))
    return {k: {"shape": v.get("shape"), "dtype": v.get("dtype"), "offsets": v.get("data_offsets")} for k, v in h.items() if k != "__metadata__"}


def drafter_check(cfg, files, keys, shas=None, manifest=None):
    """cfg = config.json; files = {name: size}; keys = {file: {tensor: {shape, dtype}}}; shas = {name: sha256};
    manifest = {name: sha | [size, sha] | {"size", "sha256"}}. ok = the incoai DFlash2 drafter (architecture, taps, mask
    id, hidden, layers, heads, window, block, vocab, selector_top_k) with weights; embed_in_checkpoint = a BF16
    [248320, 2560] embed_tokens tensor (the bf16-embed variation)."""
    cfg = cfg or {}
    dc = cfg.get("dflash_config") or {}
    why = []
    archs = cfg.get("architectures") or []
    if DRAFT_ARCH not in archs:
        why.append(f"architectures {archs}")
    tl = tuple(dc.get("target_layer_ids") or cfg.get("target_layer_ids") or ())
    if tl != TARGET_LAYER_IDS:
        why.append(f"target_layer_ids {tl}")
    if dc.get("mask_token_id", cfg.get("mask_token_id")) != MASK_TOKEN_ID:
        why.append(f"mask_token_id {dc.get('mask_token_id', cfg.get('mask_token_id'))}")
    for k, want in (("hidden_size", DRAFT_HIDDEN), ("num_hidden_layers", DRAFT_LAYERS), ("vocab_size", VOCAB),
                    ("num_key_value_heads", DRAFT_KV_HEADS), ("head_dim", DRAFT_HEAD_DIM)):
        if cfg.get(k) != want:
            why.append(f"{k} {cfg.get(k)}")
    bs = dc.get("block_size", cfg.get("block_size"))
    if not isinstance(bs, int) or bs < DRAFT_BLOCK:
        why.append(f"block_size {bs}")
    if dc.get("selector_top_k") is None:
        why.append("dflash_config.selector_top_k missing (dflash2/speculator.py:195 reads it)")
    st = [f for f in files if f.endswith(".safetensors")]
    if not st:
        why.append("no .safetensors file")
    allk = {t: v for f in keys.values() for t, v in (f or {}).items()}
    if any(t == "__error__" for t in allk):
        why.append(f"safetensors header unreadable {allk.get('__error__')}")
    emb = [t for t in allk if t.endswith("embed_tokens.weight")]
    emb_ok = bool(emb) and all((allk[t].get("shape") or [0, 0]) == [VOCAB, DRAFT_HIDDEN] and allk[t].get("dtype") == "BF16" for t in emb)
    if emb and not emb_ok:
        why.append(f"embed_tokens {[(allk[t].get('shape'), allk[t].get('dtype')) for t in emb]} != [248320, 2560] BF16")
    man = None
    if manifest:
        man = {"files": len(manifest), "mismatch": []}
        for name, want in manifest.items():
            size, wsha = (want[0], want[1]) if isinstance(want, (list, tuple)) else (
                (want.get("size"), want.get("sha256")) if isinstance(want, dict) else (None, want))
            if name not in files or (size is not None and files[name] != size) or (wsha and shas and shas.get(name) != wsha):
                man["mismatch"].append(name)
        if man["mismatch"]:
            why.append(f"manifest mismatch {man['mismatch'][:5]}")
    return {"ok": not why, "why": "; ".join(why) or "ok", "embed_in_checkpoint": emb_ok, "embed_tensors": emb,
            "tensors": len(allk), "files": dict(files), "selector_top_k": dc.get("selector_top_k"), "block_size": bs,
            "sliding_window": cfg.get("sliding_window"), "manifest": man, "shas": shas}


# ------------------------------------------------------------------ windowed garbage (klee100 lab window_garbage) + gate
def degenerate(text):
    """= e23_lab.degenerate (verbatim logic; the notebook passes L.degenerate itself)."""
    t = text or ""
    if not t.strip():
        return "EMPTY"
    if re.search(r"(\S)\1{200,}", t, re.S):
        return "REPEATED_CHAR"
    if t.count("!") > 0.5 * max(1, len(t)):
        return "BANG_GARBAGE"
    words = re.findall(r"[A-Za-z]{3,}", t)
    if len(words) > 60 and len(set(words)) < 0.08 * len(words):
        return "LOW_VARIETY"
    return None


HARD_GARBAGE = ("EMPTY", "REPEATED_CHAR", "BANG_GARBAGE")    # B2 (user rule 2026-11-01): only these stop an arm / fail a window
HARD_CORRECTNESS = ("NO_GO_ORACLE_REQUEST_ERROR", "NO_GO_ORACLE_GARBAGE", "NO_GO_GARBAGE_UNDER_LOAD", "NO_GO_GARBAGE",
                    "NO_GO_SAMPLED_PATH", "NO_GO_SHORT_PROBE")   # hard-only after the B2 rebinding of the cells' gates


def hard_only(check):
    """The degenerate check reduced to hard garbage (EMPTY / REPEATED_CHAR / BANG_GARBAGE): LOW_VARIETY (ARC digit grids
    read as low variety) and any other heuristic label -> None (recorded by the callers that keep the full check)."""
    def hard(text):
        v = check(text)
        return v if v in HARD_GARBAGE else None
    hard.e27_full = check
    return hard


def windowed_garbage(ids, decode, check=degenerate, win=WIN_TOK, min_len=WIN_MIN):
    """The first garbage 2,000-token window of a row (None when every window is clean): the klee100 lab's window_garbage.
    A whole-stream check reads long healthy ARC reasoning as LOW_VARIETY (unique words < 8 %); windows are checked one
    by one (a tail shorter than min_len is not judged)."""
    ids = list(ids or [])
    for i in range(0, len(ids), win):
        seg = ids[i:i + win]
        if len(seg) < min_len:
            break
        v = check(decode(seg))
        if v:
            return {"start_token": i, "verdict": v}
    return None


def garbage_review(rows, decode, check=degenerate, tail_chars=4000):
    """Both checks on the complete rows of a window: stream-level (e23_lab steady: the last 4,000 chars of each stream;
    RECORDED only) and windowed. B2 (user rule 2026-11-01): only HARD garbage gates (EMPTY / REPEATED_CHAR / BANG_GARBAGE);
    every other label (LOW_VARIETY: ARC digit grids) is recorded in `labels` and the window keeps its status. A complete
    row with no tokens, no text and no error is EMPTY; a row with output and no tokens is judged on its text. `check` may
    be the full or the hard-only check (hard_only keeps the full one as .e27_full)."""
    full = getattr(check, "e27_full", check)
    stream, windowed = [], []
    for r in rows or []:
        txt = r.get("text") or ""
        toks = r.get("tokens") or []
        stream.append(full(txt[-tail_chars:]) if (toks or txt) else None)
        if toks:
            windowed.append(windowed_garbage(toks, decode, full))
        elif txt:
            windowed.append({"start_token": 0, "verdict": full(txt)} if full(txt) else None)
        elif not r.get("error"):
            windowed.append({"start_token": 0, "verdict": "EMPTY"})
        else:
            windowed.append(None)
    wflag = [w for w in windowed if w]
    hard = [w for w in wflag if w["verdict"] in HARD_GARBAGE]
    labels = {w["verdict"] for w in wflag if w["verdict"] not in HARD_GARBAGE} | {x for x in stream if x and x not in HARD_GARBAGE}
    return {"rows": len(rows or []), "stream_level": [x for x in stream if x], "windowed": hard, "labels": sorted(labels),
            "gate": "GARBAGE" if hard else "CLEAN",
            "rule": f"windowed {WIN_TOK}-token check: hard garbage {HARD_GARBAGE} gates; other labels and the stream-level check are recorded"}


def apply_windowed_gate(w, review, n):
    """A window record after e23_lab.steady + NB1 window(): FAIL_GARBAGE_UNDER_LOAD there came from the stream-level check.
    Restore the pre-garbage status when the windowed check is clean (STEADY iff running_min == running_max == level, else
    CAPACITY_LIMITED; then NB1's running == level / prefill-done rule again); set FAIL_GARBAGE_UNDER_LOAD when the windowed
    check flags a STEADY / CAPACITY_LIMITED window."""
    w["garbage_review"] = review
    if review.get("labels"):
        w["labels"] = list(review["labels"])            # B2: recorded labels; the window keeps its status (its tok/s counts)
    st = w.get("status")
    if st == "FAIL_GARBAGE_UNDER_LOAD" and review.get("gate") == "CLEAN" and review.get("rows"):
        lv = w.get("level") or n
        pre = "STEADY" if w.get("running_min") == w.get("running_max") == lv else "CAPACITY_LIMITED"
        if pre == "STEADY" and not (w.get("running_eq_level") and w.get("prefill_done_before_window")):
            pre = "INCONCLUSIVE_RUNNING_NE_LEVEL" if not w.get("running_eq_level") else "INCONCLUSIVE_PREFILL_NOT_DONE"
        w["status"] = pre
        w["stream_level_flag_overridden"] = "the stream-level flag is recorded; the windowed check has no hard garbage"
    elif st in ("STEADY", "CAPACITY_LIMITED") and review.get("gate") == "GARBAGE":
        w["status"] = "FAIL_GARBAGE_UNDER_LOAD"
        w["windowed_garbage_flag"] = review["windowed"][:3]
    return w


# ------------------------------------------------------------------ smoke, early exit, prefix cold / warm, isolation
def smoke_gate(sm, is_q=False, q_nondet=False):
    """Fail-fast smoke after a boot (2 streams x 2,048 tokens): errors, windowed garbage, drafting acceptance > ~0.
    E27 K100: q_nondet (Q's own A/A not clean) -> a ladder FAIL_HIGH_MARGIN is recorded (NONDET base), never a failure."""
    if not isinstance(sm, dict):
        return "INCONCLUSIVE_SMOKE_MISSING"
    if sm.get("errors"):
        return "FAIL_SMOKE_REQUEST_ERROR"
    if (sm.get("garbage") or {}).get("gate") == "GARBAGE":
        return "FAIL_SMOKE_GARBAGE"                        # hard garbage only (garbage_review, B2)
    if not sm.get("tokens") or min(sm["tokens"]) <= 0:
        return "FAIL_SMOKE_EMPTY"
    short = min(sm["tokens"]) < SMOKE_TOKENS               # B2: recorded (INCONCLUSIVE_SMOKE_SHORT), never a stop
    acc = sm.get("acceptance") or {}
    if acc.get("status") == "NO_METRICS" or not acc:
        return "INCONCLUSIVE_SMOKE_NO_METRICS"             # /metrics not read: a read-back gap, recorded (USER RULE 2026-11-01)
    if acc.get("status") != "OK":
        return "FAIL_SMOKE_NO_DRAFTS"
    if (acc.get("drafts") or 0) >= SMOKE_MIN_DRAFTS and acc.get("accept_rate", 0.0) < SMOKE_MIN_ACCEPT:
        return "FAIL_SMOKE_DFLASH_ZERO_ACCEPT"
    if (acc.get("drafts") or 0) < SMOKE_MIN_DRAFTS:
        return "INCONCLUSIVE_SMOKE_FEW_DRAFTS"
    cg = contamination_gate(sm.get("ladder"))
    if cg == "CONCURRENCY_CONTAMINATION" and q_nondet and not is_q:
        return "INCONCLUSIVE_SMOKE_NONDET_BASE_LADDER"
    if cg == "CONCURRENCY_CONTAMINATION":                  # B2: klee100 is non-deterministic: recorded, never a stop
        return "INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION"
    if cg.startswith("INCONCLUSIVE"):
        return "INCONCLUSIVE_SMOKE_" + cg
    return "INCONCLUSIVE_SMOKE_SHORT" if short else "PASS"


def _req_err(probes):
    return any((c or {}).get("error") for c in ((probes or {}).get("cases") or []))


def sampled_hard_fail(sampled):
    """The sampled request after ready (RC1's FlashInfer JIT check): a request error, a dead server, no token or hard
    garbage stop the arm; a short / non-hard-garbage request is recorded (B2)."""
    s = sampled or {}
    return bool(s.get("error")) or s.get("alive_after") is False or not s.get("tokens") or (s.get("garbage") in HARD_GARBAGE)


def short_gate_hard(probes, sampled, fake=False):
    """B2 short gate: stop only on a probe REQUEST error (a wrong arithmetic answer is recorded) or a hard sampled failure."""
    return (not fake and _req_err(probes)) or sampled_hard_fail(sampled)


def quality_gate_hard(o, ref, fake=False, clean=CLEAN):
    """B2 quality gate: FAIL_* only for hard breakage (sampled hard failure, a probe request error, oracle request errors /
    hard garbage); FAIL_HIGH_MARGIN, wrong probe answers and a short sampled request are INCONCLUSIVE_* (recorded)."""
    orc = o.get("oracle") or {}
    vs = (orc.get("vs_" + ref) or {}).get("status")
    if sampled_hard_fail(o.get("sampled")):
        return "FAIL_SAMPLED"
    if not fake and _req_err(o.get("probes")):
        return "FAIL_PROBES"
    if any(orc.get("errors") or []) or any(g in HARD_GARBAGE for g in (orc.get("garbage") or [])):
        return "FAIL"
    if vs == "FAIL_HIGH_MARGIN":
        return "INCONCLUSIVE_HIGH_MARGIN"
    if not fake and (o.get("probes") or {}).get("ok") is not True:
        return "INCONCLUSIVE_PROBES_WRONG"
    if (o.get("sampled") or {}).get("ok") is not True:
        return "INCONCLUSIVE_SAMPLED_SHORT"
    if orc.get("complete") is True and vs in clean:
        return "PASS"
    return "INCONCLUSIVE"


def oracle_gate_hard(o, need=(), extra=(), windows=(), fake=False, clean=CLEAN):
    """B2 correctness (the cells' oracle_gate): NO_GO only for oracle request errors, hard oracle garbage, a window with
    hard garbage (FAIL_GARBAGE_UNDER_LOAD is hard-only now) or a probe request error; FAIL_HIGH_MARGIN -> INCONCLUSIVE_ORACLE."""
    orc = o.get("oracle") or {}
    if any(orc.get("errors") or []):
        return "NO_GO_ORACLE_REQUEST_ERROR"
    if any(g in HARD_GARBAGE for g in (orc.get("garbage") or [])):
        return "NO_GO_ORACLE_GARBAGE"
    if any((w or {}).get("status") == "FAIL_GARBAGE_UNDER_LOAD" for w in windows or ()):
        return "NO_GO_GARBAGE_UNDER_LOAD"
    if not fake and _req_err(o.get("probes")):
        return "NO_GO_SHORT_PROBE"
    sts = [(orc.get("vs_" + k) or {}).get("status") for k in need] + [(x or {}).get("status") for x in extra]
    if any(s_ not in clean for s_ in sts):
        return "INCONCLUSIVE_NO_ORACLE" if any(s_ in (None, "INCONCLUSIVE_NO_ORACLE") for s_ in sts) else "INCONCLUSIVE_ORACLE"
    return None


def al_vs_q(d_acc, q_acc, ratio=EARLY_EXIT_RATIO):
    """DIAGNOSTIC only (GPT audit B2): the arm's greedy oracle AL vs Q's (the same greedy tokens when the oracle is EXACT).
    DFlash drafts in one block-parallel pass (MTP: 3 sequential passes + 3 lm_head reads), so AL <= Q can still win on
    tok/s: this never skips an arm or a window."""
    da, qa = (d_acc or {}).get("al"), (q_acc or {}).get("al")
    if not isinstance(da, (int, float)) or not isinstance(qa, (int, float)):
        return {"below_q": None, "why": f"no AL to compare (D {da}, Q {qa})"}
    return {"below_q": da <= ratio * qa, "d_al": da, "q_al": qa, "why": f"greedy AL {da} vs Q {qa} (diagnostic only)"}


def short_exit(q_win, x_win, keys=None):
    """(exit, why, detail): the throughput rule after an arm's two 8k@8 windows. Exit (skip only THIS arm's 40k / 90k /
    116k windows) when BOTH 8k windows are valid and each is below Q's 8k mean by more than max(Q's A/A spread, 2 %).
    Missing / invalid windows never exit (the long windows then still run)."""
    keys = keys or _wkeys()[:2]
    qv = [q_win.get(k) for k in keys if T.window_ok(q_win.get(k))[0]]
    xv = [x_win.get(k) for k in keys if T.window_ok(x_win.get(k))[0]]
    if len(qv) != 2 or len(xv) != 2:
        return False, f"8k windows valid Q {len(qv)} / X {len(xv)}: no exit", {}
    qt = [w["tok_s"] for w in qv]
    qm = statistics.mean(qt)
    thr = max(abs(qt[0] - qt[1]) / qm, SPEED_MIN)
    xt = [w["tok_s"] for w in xv]
    det = {"q_tok_s": qt, "x_tok_s": xt, "q_mean": round(qm, 1), "threshold": round(thr, 4), "limit": round(qm * (1 - thr), 1)}
    if all(x < qm * (1 - thr) for x in xt):
        return True, f"both 8k windows {xt} below Q mean {qm:.1f} x (1 - {thr:.3f})", det
    return False, f"8k {xt} not both below {qm * (1 - thr):.1f}", det


def prefix_gate(pc):
    """Cold vs warm acceptance, 2 prompts x 3 turns (GPT analyses 2 / 3; upstream #47930 / #47926 / #58894): pc =
    {"prompts": [{"variant", "turns": {"2": {"cold", "warm"}, "3": {...}}}]}, each side {"acceptance", "prefix_hits"}.
    COLLAPSE when any warm turn's AL < 0.8 x its cold AL; PASS needs every (prompt, turn) measured with a warm prefix hit."""
    if not isinstance(pc, dict) or pc.get("error"):
        return "INCONCLUSIVE_PREFIX_TEST"
    ps = pc.get("prompts") or []
    if len(ps) != len(PREFIX_VARIANTS):
        return "INCONCLUSIVE_PREFIX_PROMPTS"
    worst = "PASS"
    for pr in ps:
        if pr.get("error"):
            return "INCONCLUSIVE_PREFIX_TEST"
        for t in ("2", "3"):
            tt = (pr.get("turns") or {}).get(t) or {}
            c, w = (tt.get("cold") or {}).get("acceptance") or {}, (tt.get("warm") or {}).get("acceptance") or {}
            if c.get("status") != "OK" or w.get("status") != "OK":
                worst = "INCONCLUSIVE_PREFIX_NO_DRAFTS" if worst == "PASS" else worst
                continue
            if not (tt.get("warm") or {}).get("prefix_hits"):
                worst = "INCONCLUSIVE_PREFIX_NO_HIT" if worst == "PASS" else worst
                continue
            if w["al"] < PREFIX_WARM_MIN * c["al"]:
                return "PREFIX_HIT_ACCEPTANCE_COLLAPSE"
    return worst


ISO_TRIGGERS = ("SERVER_DIED", "BODY_OOM", "DEVICE_FAULT", "BOOT_CAP_KILL", "BODY_ERROR", "ERROR", "TIMEOUT", "LAUNCH_FAILED",
                "NO_GO", "FAIL_SMOKE", "FAIL_GARBAGE", "QDEFER_FAILED")   # a real Q-defer failure on a D arm -> the Q-defer-off rerun


def needs_isolation(rec):
    """A D arm that FAILED with Q-defer on (hang / NCCL / garbage / RuntimeError) -> one rerun with Q-defer off. A launch
    OOM, a missing integration, a budget skip or a plain no-gain result never triggers it."""
    if not isinstance(rec, dict):
        return False
    st = str(rec.get("status") or "")
    if st.startswith("SKIPPED") or rec.get("not_integrated") or st == "NOT_INTEGRATED" or T.launch_oom(rec):
        return False
    return st.startswith(ISO_TRIGGERS) or str(rec.get("correctness", "")).startswith("NO_GO") or \
        str(rec.get("smoke_gate", "")).startswith("FAIL")


def isolation_label(main, iso):
    if not isinstance(iso, dict) or str(iso.get("status", "")).startswith("SKIPPED") or iso.get("status") is None:
        return "ISOLATION_NOT_RUN"
    clean = (iso.get("status") == "DONE" and iso.get("correctness") == "CLEAN" and str(iso.get("smoke_gate", "PASS")) == "PASS")
    return "QDEFER_INTERACTION" if clean else "DFLASH_ITSELF"


def memory_gate(free_mib, need=MEM_GATE_MIB, ranks=PP * TP):
    """(ok, why), FAIL-CLOSED (GPT audit HIGH-3): exactly 4 finite numeric readings (no bool / NaN / inf), else (None,
    INCOMPLETE_FOUR_RANK_MEMORY); any rank below 512 MiB -> (False, LOW_MEMORY_RANKS). None never steps the pin down
    (recorded) and never counts as a pass in the verdict."""
    good = (isinstance(free_mib, (list, tuple)) and len(free_mib) == ranks
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in free_mib))
    if not good:
        return None, f"INCOMPLETE_FOUR_RANK_MEMORY: {free_mib}"
    low = [i for i, x in enumerate(free_mib) if x < need]
    return (not low), (f"LOW_MEMORY_RANKS:{low} below {need} MiB: {list(free_mib)}" if low else f"all {ranks} ranks >= {need} MiB: {list(free_mib)}")


def memory_gates_ok(rec):
    """Both gates (after ready, after the first decode) recorded True."""
    rec = rec or {}
    return ((rec.get("memory_gate_ready") or {}).get("ok") is True and (rec.get("memory_gate_first_decode") or {}).get("ok") is True)


def kv_group_count(layout, rank="2"):
    """KV cache groups of the last-PP-stage rank (where the drafter lives) from kv_layout_readback()."""
    g = ((layout or {}).get("groups") or {}).get(str(rank))
    return len(g) if isinstance(g, list) else None


def kv_group_explosion(q_layout, d_layout, drafter_groups=None):
    """vLLM #58638 class (hybrid + DFlash KV-group explosion): D's group count on the drafter rank > Q's + the drafter's own
    groups (the kit's per-K count, else 1). Recorded only: the tok/s windows decide."""
    q, d = kv_group_count(q_layout), kv_group_count(d_layout)
    allow = int(drafter_groups) if isinstance(drafter_groups, int) and drafter_groups > 0 else 1
    if q is None or d is None:
        return {"flag": None, "q_groups": q, "d_groups": d, "why": "no group read-back"}
    return {"flag": "KV_GROUP_EXPLOSION" if d > q + allow else None, "q_groups": q, "d_groups": d, "drafter_groups_allowed": allow}


def contamination_gate(ladder):
    """The concurrency ladder of the smoke: the greedy probe alone vs with 2 / 4 / 8 other requests. Token ids must be
    identical; a difference is NEAR_TIE (recorded, allowed: the alone row's own top-2 margin is small at the first
    difference) or a failure (CONCURRENCY_CONTAMINATION)."""
    if not isinstance(ladder, dict) or not ladder.get("alone"):
        return "INCONCLUSIVE_LADDER_MISSING"
    if (ladder["alone"] or {}).get("error"):
        return "INCONCLUSIVE_LADDER_ALONE_ERROR"
    worst = "EXACT"
    for c in CONC_LADDER:
        st = ((ladder.get(str(c)) or {}).get("vs_alone") or {}).get("status")
        if st == "FAIL_HIGH_MARGIN":
            return "CONCURRENCY_CONTAMINATION"
        if st is None or str(st).startswith("INCONCLUSIVE"):
            worst = "INCONCLUSIVE_LADDER"
        elif st == "NEAR_TIE" and worst == "EXACT":
            worst = "NEAR_TIE"
    return worst


def bf16_row_norm(raw):
    """L2 norm of a little-endian BF16 row (stdlib: each value = the top 16 bits of a float32)."""
    import struct
    n = len(raw) // 2
    vals = struct.unpack("<%df" % n, b"".join(b"\x00\x00" + raw[2 * i:2 * i + 2] for i in range(n)))
    return math.sqrt(sum(v * v for v in vals))


def drafter_pins_check(facts, pins=DRAFTER_PINS):
    """facts = {"files": {name: [size, sha256]}, "tensors": n, "embed": {"shape", "dtype", "nbytes", "sha256"},
    "mask_row": {"sha256", "norm"}, "zero_rows": n}, measured on the mount (header + tensor bytes, no model load).
    -> (ok, mismatches). A mismatch = DRAFTER_SHA_MISMATCH: the D arms are skipped; Q and the submission path run."""
    bad = []
    f = (facts or {}).get("files") or {}
    for name in ("model.safetensors", "config.json"):
        have = f.get(name)
        if not have or list(have)[:2] != [pins[name]["size"], pins[name]["sha256"]]:
            bad.append(f"{name} {have} != [{pins[name]['size']}, {pins[name]['sha256'][:12]}..]")
    e = (facts or {}).get("embed") or {}
    for k in ("shape", "dtype", "nbytes", "sha256"):
        if e.get(k) != pins["embed_tokens"][k]:
            bad.append(f"embed_tokens {k} {str(e.get(k))[:70]} != {str(pins['embed_tokens'][k])[:70]}")
    m = (facts or {}).get("mask_row") or {}
    if m.get("sha256") != pins["mask_row"]["sha256"]:
        bad.append(f"mask row sha {str(m.get('sha256'))[:16]}")
    if not isinstance(m.get("norm"), (int, float)) or abs(m["norm"] - pins["mask_row"]["norm"]) > 1e-5:
        bad.append(f"mask row norm {m.get('norm')} != {pins['mask_row']['norm']}")
    if (facts or {}).get("tensors") != pins["tensors"]:
        bad.append(f"tensors {(facts or {}).get('tensors')} != {pins['tensors']}")
    if (facts or {}).get("zero_rows") != pins["zero_rows"]:
        bad.append(f"zero rows {(facts or {}).get('zero_rows')}")
    return not bad, bad


REAL_KIT_CHECKS = ("cell0_identity_submission", "inputs_found", "merge_ok", "sha256_all_ok", "drafter_pins_ok", "merlin_install_ok",
                   "merlin_flag_sources_equal", "patch_60490_ok", "seam_apply_verify_revert_ok", "host_phase_all_5_ok",
                   "launch_check_all_5_ok", "env_in_spawned_child_all_5", "tree_check_all_5", "argv_fields_ok",
                   "spec_config_preflight_ok", "kv_cache_configs_ok", "drafter_loaded_weights_ok",
                   "e2_guard_dflash_ok")    # E27 B1: the real e2-guard admits the DFlash2 drafter on the patched tree
PPACK_SHA_PIN = "8dadf0763d43d22966603745198ad327ed6ed36001cc60a79925ea18818f2dea"
SEAM_SHA_PIN = "c4ea1ecc287036bcadbc98cd33e40309d17f2c58b471e5c876ea7acddba51f78"
KV_RUN_OF = {"D4s": "K4_swblock1024", "D3s": "K3_swblock1024", "D4": "K4", "D4t": "K4_swblock1024", "D7": "K7", "D4b": "K4_fp8"}
CPU_LIMIT_HINTS = ("cuda", "device", "nvml", "libcuda", "vllm._c", "platform", "gpu", "cudart", "nccl", "triton")


def _cpu_limit(r12):
    """R12 could not build the drafter on a CPU box (a device / CUDA limit), and it is NOT a 'got no tensor' false raise."""
    j = (r12 or {}).get("json") or {}
    if j.get("ok") is not None or j.get("false_raise_suspect"):
        return False
    err = " ".join(str(j.get(k) or "") for k in ("error", "instantiate_error", "load_weights_error")).lower()
    return bool(err) and "got no tensor" not in err and any(h in err for h in CPU_LIMIT_HINTS)


def kv_gate_from_report(rep, expect):
    """The notebook's KV_KIT_GATE from the real kit's report (GPT audit B3: every real-kit check + provenance). expect =
    {"notebook_shas": {label: sha} (the closed-gate rebuild / the pre-rebuild file the kit ran), "seam_sha", "ppack_sha",
    "drafter_pins", "merlin_src_shas" {rel: sha}}. Returns {"passed", "provenance", "per_k" {D arm: bool}, "why" [...], ...};
    any mismatch keeps the gate closed. drafter_loaded_weights_ok may be accepted as CPU_LIMIT (recorded): the GPU
    draft_load_check read-back is then a hard pass condition of every D arm (it always is, at the smoke)."""
    why, out = [], {"passed": False, "checks": {}, "per_k": {}, "kv_summary": {}}
    chk = ((rep or {}).get("verdict") or {}).get("checks") or {}
    C = (rep or {}).get("checks") or {}
    for c in REAL_KIT_CHECKS:
        out["checks"][c] = chk.get(c)
    out["drafter_loaded_weights"] = True if chk.get("drafter_loaded_weights_ok") is True else (
        "CPU_LIMIT" if _cpu_limit(C.get("R12")) else False)
    out["require_gpu_load_check"] = True
    bad = [c for c in REAL_KIT_CHECKS if chk.get(c) is not True and not (c == "drafter_loaded_weights_ok" and out["drafter_loaded_weights"] == "CPU_LIMIT")]
    if bad:
        why.append(f"real-kit checks not True: {bad}")
    nbs = (rep or {}).get("notebook_sha256")
    prov = [lab for lab, v in (expect.get("notebook_shas") or {}).items() if v and v == nbs]
    out["provenance"] = {"kit_notebook_sha256": nbs, "matches": prov}
    if not prov:
        why.append(f"STALE_NOTEBOOK_SHA: the kit ran {nbs}, expected one of {expect.get('notebook_shas')}")
    seam = (((C.get("R1") or {}).get("embed")) or {}).get("sha256")
    if not seam or seam != expect.get("seam_sha"):
        why.append(f"seam sha in the report {seam} != the embedded {expect.get('seam_sha')}")
    plan_q = (((C.get("R4") or {}).get("plans") or {}).get("Q") or {})
    av = [str(x) for x in (plan_q.get("argv") or [])]
    lv = lambda f: av[av.index(f) + 1] if f in av and av.index(f) + 1 < len(av) else None
    ident = {"tp": lv("--tensor-parallel-size"), "pp": lv("--pipeline-parallel-size"),
             "partition": (plan_q.get("env") or {}).get(T.ENV_PART), "merlin_commit": (C.get("R2") or {}).get("merlin_commit")}
    out["identity"] = ident
    want_id = {"tp": "2", "pp": "2", "partition": "25,23", "merlin_commit": expect.get("merlin_commit")}
    miss = {k: (ident.get(k), v) for k, v in want_id.items() if not ident.get(k) or ident.get(k) != v}
    if miss:
        why.append(f"identity fields missing / different in the report (never inferred): {miss}")
    if expect.get("ppack_sha") != PPACK_SHA_PIN:
        why.append(f"nb4_ppack sha {expect.get('ppack_sha')} != {PPACK_SHA_PIN}")
    dr = (C.get("R1") or {}).get("drafter") or {}
    pins = expect.get("drafter_pins") or DRAFTER_PINS
    fs = ((dr.get("facts") or {}).get("files") or {})
    emb = ((dr.get("facts") or {}).get("embed") or {}).get("sha256")
    msk = ((dr.get("facts") or {}).get("mask_row") or {}).get("sha256")
    if not (dr.get("pins_ok") is True and list((fs.get("model.safetensors") or [None, None])[:2]) == [pins["model.safetensors"]["size"], pins["model.safetensors"]["sha256"]]
            and list((fs.get("config.json") or [None, None])[:2]) == [pins["config.json"]["size"], pins["config.json"]["sha256"]]
            and emb == pins["embed_tokens"]["sha256"] and msk == pins["mask_row"]["sha256"]):
        why.append("drafter file / tensor shas not the pinned ones in the report")
    src = (C.get("R2") or {}).get("src_shas") or {}
    ms = expect.get("merlin_src_shas") or {}
    if not src or any(src.get(k) != v for k, v in ms.items()) or set(src) - set(ms):
        why.append(f"Merlin source shas differ from the audited tree: {sorted(k for k in set(src) | set(ms) if src.get(k) != ms.get(k))[:6]}")
    runs = (((C.get("R9") or {}).get("json")) or {}).get("runs") or {}
    for arm, run in KV_RUN_OF.items():
        r = runs.get(run) or {}
        okk = bool(r.get("num_blocks")) and "error" not in r and isinstance(r.get("kv_group_count"), int)
        out["per_k"][arm] = okk
        out["kv_summary"][arm] = {k: r.get(k) for k in ("num_blocks", "kv_group_count", "drafter_group_count", "groups",
                                                         "max_concurrency_131072", "wasted_bytes_per_block", "attn_metadata_builders")}
    for arm in [a for a in ("D4s", "D3s") if a in KV_RUN_OF]:   # the primaries (restrict_arms may drop one); the others without a summary -> SKIPPED_NO_KV_* at run time
        if not out["per_k"][arm]:
            why.append(f"no per-K KV summary for {arm} ({KV_RUN_OF[arm]})")
    out["passed"] = not why
    out["why"] = why or ["every real-kit check True (drafter loaded weights: " + str(out["drafter_loaded_weights"]) + "), provenance bound"]
    return out


# ------------------------------------------------------------------ residency (long bands)
def resident_level(pool_blocks, f_blocks, prompt_tokens, gen_tokens, seqs=None, margin=RESIDENT_MARGIN, block=BLOCK_TOKENS):
    seqs = SEQS if seqs is None else seqs   # read at call time (E27 K100: SEQS 24)
    if not pool_blocks or f_blocks is None:
        return None
    per = math.ceil((prompt_tokens + gen_tokens) / block) + int(f_blocks)
    return max(0, min(int(seqs), int(margin * pool_blocks // per)))


def gen_estimate(n, prompt_tokens, prefill_tok_s, win_sec, rate=GEN_RATE_PER_STREAM):
    adm = n * prompt_tokens / max(500.0, float(prefill_tok_s or 700.0))
    return int(rate * (adm + 60.0 + float(win_sec)))


def plan_level(pool_blocks, f_blocks, band_tokens, prefill_tok_s, win_sec, seqs=None, block=BLOCK_TOKENS):
    """Resident level for a band: start at seqs, step down until the level holds its own growth estimate."""
    seqs = SEQS if seqs is None else seqs   # read at call time (E27 K100: SEQS 24)
    for n in range(int(seqs), 0, -1):
        lv = resident_level(pool_blocks, f_blocks, band_tokens, gen_estimate(n, band_tokens, prefill_tok_s, win_sec), seqs, block=block)
        if lv is None:
            return None
        if lv >= n:
            return n
    return 0


def kv_bound(w):
    if not isinstance(w, dict):
        return False
    rm = w.get("running_max")
    return (w.get("status") == "CAPACITY_LIMITED" or (w.get("preemptions") or 0) > 0
            or (isinstance(rm, int) and 0 < rm < (w.get("level") or 0)))


# ------------------------------------------------------------------ long oracle
def long_oracle_gate(rec, is_q):
    lo = (rec or {}).get("long_oracle")
    if not isinstance(lo, dict):
        return "INCONCLUSIVE_LONG_ORACLE_MISSING"
    if any(lo.get("errors") or []):
        return "NO_GO_LONG_ORACLE_ERROR"
    if any(lo.get("garbage") or []):
        return "NO_GO_LONG_ORACLE_GARBAGE"
    if lo.get("complete") is not True:
        return "INCONCLUSIVE_LONG_ORACLE_INCOMPLETE"
    if is_q:
        return "PASS"
    vs = (lo.get("vs_q_long") or {}).get("status")
    if vs == "FAIL_HIGH_MARGIN":                           # B2: recorded (klee100 is non-deterministic), never a stop
        return "INCONCLUSIVE_LONG_ORACLE_HIGH_MARGIN"
    if vs in CLEAN or vs in NONDET_OK:
        return "PASS"
    return f"INCONCLUSIVE_LONG_ORACLE_{vs}"


# ------------------------------------------------------------------ graph spans per rank (drafter cost by context)
def graph_spans(rank_graphs_by_rank, top=4):
    """{rank: [graph dicts with n / start / end]} (nb4t_lib.rank_graphs) -> per rank the node-count modes with their count
    and median span (ms): on PP1 the target's verify graph and the drafter's FULL_DECODE_ONLY graph are separate replays,
    so the draft time per tick is the span of the smaller mode (recorded; the analysis names which is which)."""
    out = {}
    for r, gs in (rank_graphs_by_rank or {}).items():
        by = {}
        for g in gs or []:
            by.setdefault(int(g["n"]), []).append((float(g["end"]) - float(g["start"])) / 1000.0)
        modes = sorted(by.items(), key=lambda kv: -len(kv[1]))[:top]
        out[str(r)] = [{"nodes": n, "count": len(v), "span_ms_median": round(statistics.median(v), 3)} for n, v in modes]
    return out


# ------------------------------------------------------------------ speed
def band_windows(rec, band):
    w = T._win(rec)
    return {k: v for k, v in sorted(w.items()) if isinstance(v, dict) and k.split("@")[0] == band}


def best_valid(ws):
    return [(k, v) for k, v in ws.items() if T.window_ok(v)[0]]


def long_compare(Q, X, band):
    """One band at each arm's own resident level: gain = mean(X) / mean(Q) - 1; threshold = max(Q's A/A spread, 2 %)."""
    qv, xv = best_valid(band_windows(Q, band)), best_valid(band_windows(X, band))
    out = {"band": band, "q_windows": [k for k, _ in qv], "x_windows": [k for k, _ in xv],
           "why_invalid": {k: T.window_ok(v)[1] for k, v in list(band_windows(Q, band).items()) + list(band_windows(X, band).items())
                           if not T.window_ok(v)[0]}}
    if not qv:
        out.update(status="INCONCLUSIVE_Q_LONG", why=f"Q has no valid {band} window")
        return out
    if not xv:
        out.update(status="INCONCLUSIVE_X_LONG", why=f"no valid {band} window")
        return out
    qt, xt = [v["tok_s"] for _, v in qv], [v["tok_s"] for _, v in xv]
    qm, xm = statistics.mean(qt), statistics.mean(xt)
    aa = abs(qt[0] - qt[1]) / qm if len(qt) >= 2 else None
    thr = max(aa if aa is not None else 0.0, SPEED_MIN, float(LONG_THR_FLOOR.get(band) or 0.0))
    gain = xm / qm - 1.0
    ql, xl = sorted({v.get("level") for _, v in qv}), sorted({v.get("level") for _, v in xv})
    out.update(q_tok_s=[round(x, 1) for x in qt], x_tok_s=[round(x, 1) for x in xt], q_mean=round(qm, 1), x_mean=round(xm, 1),
               aa_spread=None if aa is None else round(aa, 4), threshold=round(thr, 4), gain=round(gain, 4), q_level=ql, x_level=xl,
               levels_differ=ql != xl, q_tick_ms=[v.get("tick_ms") for _, v in qv], x_tick_ms=[v.get("tick_ms") for _, v in xv],
               q_al=[v.get("mean_acceptance_length") for _, v in qv], x_al=[v.get("mean_acceptance_length") for _, v in xv],
               status="PASS" if gain > thr else ("REGRESSION" if gain < -thr else "NO_REGRESSION"))
    return out


def long_pass(lc):
    bad = {b: c.get("status") for b, c in lc.items() if c.get("status") not in ("PASS", "NO_REGRESSION")}
    return not bad, bad


# ------------------------------------------------------------------ verdicts
def base_check(Q):
    """NB4T's strict base check (Q-defer label ENGAGED, env / config read-backs ok): RECORDED ONLY since 2026-11-01."""
    return T.base_check(Q)


SPEC_CONTRA_CONFIG = ("spec_method ", "spec_k ", "spec_model ")
SPEC_CONTRA_DFLASH = ("an MTP draft model was resolved", "SpecDecoding lines with", "the drafter runs EAGER")


def spec_mismatch(is_d, crb, drb):
    """(mismatch, why): the REAL-failure part of the read-backs (coordinator sign-off 2026-11-01): the arm measures the
    wrong thing when the logged speculative method, K or drafter model differ from the plan (config read-back), or, on a
    D arm, an MTP drafter is resolved, DFlash2DraftModel is not resolved (the log has 'Resolved architecture' lines but
    not this one) or the SpecDecoding lines carry another K. -> NOT_INTEGRATED, only this arm. Drafter EAGER, backend
    mismatch, captures, max_num_seqs, KV bytes, partition and missing lines stay flags (READBACK_CONTRADICTION /
    READBACK_INCOMPLETE)."""
    why = [str(c) for c in ((crb or {}).get("contradictions") or []) if str(c).startswith(SPEC_CONTRA_CONFIG)]
    if is_d:
        drb = drb or {}
        why += [str(c) for c in (drb.get("contradictions") or []) if str(c).startswith(SPEC_CONTRA_DFLASH)]
        archs = (drb.get("rec") or {}).get("archs")
        if archs and DRAFT_ARCH not in archs:
            why.append(f"{DRAFT_ARCH} not resolved (resolved: {archs})")
    return bool(why), "; ".join(why)


QDEFER_FALLBACK_MAX = 0.05            # flush_recv + flush_consume share of defer_park that counts as a real Q-defer failure


def qdefer_real_failure(qr):
    """(failed, why). USER RULE 2026-11-01: a measurement label / read-back never skips the experiment; only a REAL
    Q-defer failure gates: Q-defer not applied (nb4_ppack label NOT_INTEGRATED, fewer init lines than expected), any
    non-last rank with fallback_share > 0, or flush_recv + flush_consume > 5 % of defer_park. The labels ENGAGED /
    ENGAGED_FALLBACK / NOT_ENGAGED and exec_share are recorded only (the GPU commit 8b1760ed: Q ENGAGED_FALLBACK from
    exec_share 0.86 < 0.9 with 0 fallbacks skipped every D arm)."""
    if not isinstance(qr, dict):
        return False, "no Q-defer read-back (recorded)"
    lab = qr.get("label")
    if lab == "NOT_APPLICABLE":
        return False, "Q-defer off (isolation rerun)"
    det = qr.get("detail") if isinstance(qr.get("detail"), dict) else {}
    inits, expect = det.get("inits"), det.get("expect")
    if lab == "NOT_INTEGRATED":
        return True, f"Q-defer not applied: nb4_ppack label NOT_INTEGRATED (init lines {inits} / {expect})"
    if isinstance(inits, int) and isinstance(expect, int) and inits < expect:
        return True, f"Q-defer init lines {inits} < {expect}"
    per = det.get("defer_per_rank") if isinstance(det.get("defer_per_rank"), dict) else {}
    if not per and isinstance(qr.get("shares"), dict):
        per = qr["shares"]
    for pid, c in sorted(per.items()):
        c = c if isinstance(c, dict) else {}
        fs = c.get("fallback_share")
        if isinstance(fs, (int, float)) and not isinstance(fs, bool) and fs > 0:
            return True, f"rank {pid} fallback_share {fs} > 0"
        park = c.get("defer_park") if isinstance(c.get("defer_park"), (int, float)) else 0
        fb = sum(c.get(k) for k in ("flush_recv", "flush_consume") if isinstance(c.get(k), (int, float)))
        if fb > QDEFER_FALLBACK_MAX * max(park, 0) and fb > 0:
            return True, f"rank {pid} flush_recv + flush_consume {fb} > {QDEFER_FALLBACK_MAX:.0%} of defer_park {park}"
    shares = {str(p): (c or {}).get("exec_share") for p, c in per.items() if isinstance(c, dict)}
    return False, f"label {lab} (recorded only), exec_share {shares}, no fallback"


def base_usable(Q):
    """(ok, label, why): Q is the base for the D arms when it ran to the end (status DONE, or a post-processing error after
    a complete run), its correctness is CLEAN (windowed garbage gate), its greedy oracle / long oracle did not FAIL, the
    server stayed healthy and Q-defer has no REAL failure (qdefer_real_failure). A Q-defer label, an exec_share, a parity
    label, a memory-gate reading, an INCONCLUSIVE oracle reference or an incomplete env / config read-back are recorded
    (base_check_strict) and never skip the D arms (USER RULE 2026-11-01)."""
    Q = Q if isinstance(Q, dict) else {}
    st = str(Q.get("status") or "")
    if not Q or not st or st.startswith("SKIPPED") or st == "NOT_RUN":
        return False, "INCONCLUSIVE_NOT_RUN", f"Q {st or None}"
    if Q.get("not_integrated") or st == "NOT_INTEGRATED":
        return False, "INCONCLUSIVE_BASE_NOT_INTEGRATED", f"Q {st}: {Q.get('why') or Q.get('early_stop')}"
    done = st == "DONE" or (st == "POSTPROCESS_ERROR" and Q.get("status_before_postprocess") in (None, "NOT_RUN", "DONE"))
    if not done:
        return False, "INCONCLUSIVE_BASE_FAILED", f"Q status {st} (died in {Q.get('died_in')}): {str(Q.get('early_stop') or Q.get('error') or '')[:200]}"
    if Q.get("correctness") != "CLEAN":
        return False, "INCONCLUSIVE_BASE_CORRECTNESS", f"Q correctness {Q.get('correctness')}"
    qg = str(Q.get("quality_gate") or "")
    if qg.startswith("FAIL"):
        return False, "INCONCLUSIVE_BASE_GATE", f"Q greedy oracle gate {qg}"
    lg = str(Q.get("long_gate") or "")
    if lg.startswith(("NO_GO", "FAIL")):
        return False, "INCONCLUSIVE_BASE_LONG_ORACLE", f"Q long oracle {lg}"
    fail, why = qdefer_real_failure(T.latest_q(Q))
    if fail:
        return False, "INCONCLUSIVE_BASE_QDEFER_FAILED", why
    strict = T.base_check(Q)
    return True, "OK", (f"Q usable: status {st}, correctness CLEAN, oracle gate {qg or None}, long oracle {lg or None}, Q-defer {why}; "
                        f"strict NB4T base_check {strict[1]} (recorded only)")


def conv_compare(D4, X):
    """D4t (DFLASH_CONV_IMPL=torch) vs D4s (fused, Merlin's vLLM #55960 kernel; D is the base arm here): the 8k@8 windows' AL and per-position
    acceptance, the 8k tok/s, the greedy oracle acceptance. CONV_IMPL_TORCH_BETTER when D4t's mean 8k AL is higher than
    D4's by more than D4's own A/A AL spread (its two 8k windows); recorded, never a throughput winner."""
    keys = _wkeys()[:2]
    dw, xw = T._win(D4 or {}), T._win(X or {})
    dv = [dw.get(k) for k in keys if T.window_ok(dw.get(k))[0]]
    xv = [xw.get(k) for k in keys if T.window_ok(xw.get(k))[0]]
    out = {"d4_8k": [{"tok_s": w.get("tok_s"), "al": w.get("mean_acceptance_length"), "per_pos": w.get("accept_rate_per_position_incl_admission")}
                     for w in dv], "d4t_8k": [{"tok_s": w.get("tok_s"), "al": w.get("mean_acceptance_length"),
                                               "per_pos": w.get("accept_rate_per_position_incl_admission")} for w in xv],
           "oracle_al": [((D4 or {}).get("oracle_acceptance") or {}).get("al"), ((X or {}).get("oracle_acceptance") or {}).get("al")]}
    dal = [w.get("mean_acceptance_length") for w in dv if isinstance(w.get("mean_acceptance_length"), (int, float))]
    xal = [w.get("mean_acceptance_length") for w in xv if isinstance(w.get("mean_acceptance_length"), (int, float))]
    if len(dal) != 2 or not xal:
        out.update(label="INCONCLUSIVE_CONV_IMPL", why=f"valid 8k windows D4 {len(dal)} / D4t {len(xal)}")
        return out
    spread = abs(dal[0] - dal[1])
    gain = statistics.mean(xal) - statistics.mean(dal)
    out.update(al_gain=round(gain, 4), aa_al_spread=round(spread, 4),
               tok_s_gain=round(statistics.mean(w["tok_s"] for w in xv) / statistics.mean(w["tok_s"] for w in dv) - 1, 4))
    out["label"] = "CONV_IMPL_TORCH_BETTER" if gain > spread else ("CONV_IMPL_TORCH_WORSE" if gain < -spread else "CONV_IMPL_NO_DIFFERENCE")
    out["why"] = f"8k AL D4t {statistics.mean(xal):.3f} vs D4 {statistics.mean(dal):.3f} (A/A spread {spread:.3f}); tok/s {out['tok_s_gain']:+.1%}"
    return out


arms_recs_ref = {}


def arm_verdict(Q, X, key, base=None, arms=None):
    """One D arm vs Q. Order: not run / skipped -> launch OOM -> not integrated -> smoke -> correctness (short + long oracle)
    -> launch / arm failure -> Q-defer engaged -> env / config / DFlash read-back -> DONE -> early exit -> prefix cold / warm
    -> oracle -> base -> short speed (8k pair + 40k) -> long bands (90k / 116k) -> WIN / SHORT_ONLY_WIN / NO_GAIN / ..."""
    base = base or base_usable(Q)
    out = {"arm": key, "k": K_OF.get(key)}
    st = str((X or {}).get("status") or "")
    if X and X.get("died_in") == "tf_k100" and X.get("measured_before_tf") and st.startswith(T.FAILED_PREFIXES):
        out["tf_server_died_at_end"] = st          # E27 K100: the TF ran last; the measured windows keep their verdict (flag)
        st = "DONE"
    if not X or not st or st.startswith("SKIPPED"):
        v = ("SKIPPED_NO_RING" if "NO_RING" in st else st if "NO_KV_" in st else
             "SKIPPED_KV_KIT_GATE" if "KV_KIT" in st else "SKIPPED_BUDGET" if ("BUDGET" in st or not X) else
             "NOT_INTEGRATED" if (X or {}).get("not_integrated") else "SKIPPED_DEPENDENCY")
        out.update(verdict=v, why=(X or {}).get("why") or "not run")
        return out
    if st == "LAUNCH_FAILED" and T.launch_oom(X):
        out.update(verdict="LAUNCH_OOM", why=f"every launch try OOM: {X.get('tries')}", capacity=T.capacity_facts(X))
        return out
    if X.get("not_integrated") or st == "NOT_INTEGRATED":
        out.update(verdict="NOT_INTEGRATED", why=X.get("why") or X.get("early_stop"))
        return out
    sg = str(X.get("smoke_gate") or "")
    if "CONCURRENCY_CONTAMINATION" in sg and sg.startswith("FAIL"):
        out.update(verdict="CONCURRENCY_CONTAMINATION", why=json.dumps((X.get("smoke") or {}).get("ladder_brief"), default=str)[:400])
        return out
    if "ZERO_ACCEPT" in sg:
        out.update(verdict="DFLASH_ZERO_ACCEPT", why=f"smoke {sg}: {json.dumps((X.get('smoke') or {}).get('acceptance'), default=str)[:300]}")
        return out
    if sg.startswith("FAIL"):
        out.update(verdict="SMOKE_FAIL", why=f"smoke {sg}: {json.dumps((X.get('smoke') or {}).get('acceptance'), default=str)[:200]}")
        return out
    rec_flags = ["SMOKE_INCONCLUSIVE"] if sg.startswith("INCONCLUSIVE") else []      # recorded (USER RULE 2026-11-01)
    if "CONCURRENCY_CONTAMINATION" in sg:
        rec_flags.append("CONCURRENCY_CONTAMINATION_RECORDED")
    out["smoke_gate"] = sg or None
    so = X.get("oracle_vs")
    corr = str(X.get("correctness", ""))
    if corr in HARD_CORRECTNESS or str(X.get("quality_gate", "")).startswith("FAIL"):   # B2: hard failures only
        out.update(verdict="CORRECTNESS_FAIL", why=f"oracle vs Q {so}, correctness {X.get('correctness')}, gate {X.get('quality_gate')}")
        return out
    if so == "FAIL_HIGH_MARGIN" or corr.startswith("NO_GO"):  # B2: recorded (klee100 non-deterministic); TF is the verdict gate
        rec_flags.append("ORACLE_RECORDED_" + (so if so == "FAIL_HIGH_MARGIN" else corr))
    lg = X.get("long_gate") or long_oracle_gate(X, False)
    if lg in ("NO_GO_LONG_ORACLE_ERROR", "NO_GO_LONG_ORACLE_GARBAGE"):
        out.update(verdict="CORRECTNESS_FAIL_LONG", why=f"long oracle {lg}")
        return out
    if lg.startswith(("NO_GO", "FAIL")) or "HIGH_MARGIN" in lg:
        rec_flags.append("LONG_ORACLE_RECORDED_" + lg)
    if st in ("LAUNCH_FAILED", "GPU_NOT_EMPTY"):
        out.update(verdict=st, why=f"tries {X.get('tries')}", capacity=T.capacity_facts(X))
        return out
    if st.startswith(T.FAILED_PREFIXES) or st.startswith("TREE_"):
        out.update(verdict="INCONCLUSIVE_ARM_FAILED", why=f"arm status {st} (died in {X.get('died_in')})", capacity=T.capacity_facts(X))
        return out
    qr = T.latest_q(X)
    qf = qdefer_real_failure(qr)
    out["qdefer"] = {"label": (qr or {}).get("label"), "real_failure": qf[0], "why": qf[1]}
    if qf[0]:
        out.update(verdict="NOT_INTEGRATED" if (qr or {}).get("label") == "NOT_INTEGRATED" else "QDEFER_FAILED", why=f"Q-defer: {qf[1]}")
        return out
    sm = spec_mismatch(True, X.get("config_readback_end") or X.get("config_readback"), X.get("dflash_readback"))
    if sm[0]:                                 # the arm measured the wrong spec method / K / drafter: a real failure
        out.update(verdict="NOT_INTEGRATED", why="SPEC_MISMATCH: " + sm[1])
        return out
    if (qr or {}).get("label") not in (None, "ENGAGED", "NOT_APPLICABLE"):
        rec_flags.append("QDEFER_LABEL_" + str((qr or {}).get("label")))                # recorded only
    out["readback_issues"] = {}
    for nm in ("env_readback", "config_readback", "dflash_readback"):
        r_ = X.get(nm) or {}
        if not r_.get("ok"):                  # recorded (USER RULE 2026-11-01): the measured verdict stands, flagged
            out["readback_issues"][nm] = {"definitive": bool(r_.get("definitive")), "why": str(r_.get("why"))[:300]}
            rec_flags.append("READBACK_CONTRADICTION" if r_.get("definitive") else "READBACK_INCOMPLETE")
    if st != "DONE":
        out.update(verdict="INCONCLUSIVE_ARM_FAILED", why=f"arm status {st}")
        return out
    if out.get("tf_server_died_at_end"):
        rec_flags.append("TF_SERVER_DIED_AT_END")
    out["capacity"] = capacity_compare(Q, X)
    pd = out["capacity"].get("pool_drop")
    if isinstance(pd, (int, float)) and pd > POOL_DROP_NOTE:
        out["capacity_note"] = (f"D pool {pd:.0%} below Q (drafter KV + selector workspace + lower pin): the D arm ran at the "
                                "capacity it got; long bands compare each arm at its own resident level")
    out["acceptance_greedy"] = {"q": Q.get("oracle_acceptance"), "x": X.get("oracle_acceptance")}
    out["al_vs_q"] = X.get("al_vs_q")
    out["prefix_reuse"] = prefix_reuse_drop(Q, X)
    out["ingraph_meta"] = X.get("ingraph_meta")
    out["pp0_graphs"] = pp0_graph_degraded(Q.get("stage_graphs"), X.get("stage_graphs"))
    out["qdefer_k"] = qdefer_k_fallback(Q, X)
    out["kv_draft_group_warning"] = X.get("kv_draft_group_warning")
    out["engine_block"] = X.get("engine_block")
    out["sw_block_flag"] = ("SW_BLOCK_LOWERED_ENGINE_BLOCK" if (X.get("engine_block") or 4096) < 4096 else None) if key in SW_BLOCK_ARMS else None
    out["preempt_accept"] = {k_: v for k_, v in (X.get("preempt_accept") or {}).items() if v}
    out["flags"] = sorted({f for f in (out.get("sw_block_flag"), kv_group_explosion(Q.get("kv_layout"), X.get("kv_layout"),
                                                          (X.get("kv_kit_summary") or {}).get("drafter_group_count")).get("flag"), out["prefix_reuse"].get("flag"), out["pp0_graphs"].get("flag"),
                                        out["qdefer_k"].get("flag")) if f} |
                          {v.get("flag") for v in out["preempt_accept"].values() if isinstance(v, dict) and v.get("flag")} | set(rec_flags))
    out["short_exit"] = X.get("short_exit")
    out["kv_groups"] = kv_group_explosion(Q.get("kv_layout"), X.get("kv_layout"), (X.get("kv_kit_summary") or {}).get("drafter_group_count"))
    pg = X.get("prefix_gate") or prefix_gate(X.get("prefix_cache"))
    out["prefix_gate"] = pg
    out["prefix_ratios"] = prefix_ratios(X.get("prefix_cache"))
    if pg == "PREFIX_HIT_ACCEPTANCE_COLLAPSE" and PREFIX_RECORD_ONLY:
        out["flags"] = sorted(set(out["flags"]) | {"PREFIX_HIT_ACCEPTANCE_COLLAPSE"})
    elif pg == "PREFIX_HIT_ACCEPTANCE_COLLAPSE":
        out.update(verdict=pg, why=json.dumps({k: ((X.get("prefix_cache") or {}).get(k) or {}).get("acceptance") for k in ("cold", "warm")},
                                              default=str)[:400])
        return out
    tg = X.get("tf_gate") or {}
    out["tf_gate"] = tg
    if str(tg.get("status") or "").startswith("FAIL"):       # E27 K100: teacher-forced D vs Q outside Q's own A/A envelope
        out.update(verdict="CORRECTNESS_FAIL_TF", why=str(tg.get("why"))[:400])
        return out
    if TF_REQUIRED and tg.get("status") != "PASS":            # E27 K100 audit: a missing / INCONCLUSIVE TF never yields a WIN
        out["tf_verification_pending"] = True
        out["flags"] = sorted(set(out.get("flags") or []) | {"TF_NOT_VERIFIED"})
    out["k100_records"] = k100_records(Q, X, key)
    if so not in CLEAN and so not in NONDET_OK and so != "FAIL_HIGH_MARGIN":   # B2: HIGH_MARGIN is recorded (flag above)
        out.update(verdict="INCONCLUSIVE_ORACLE", why=f"oracle vs Q {so}")
        return out
    if not base[0]:
        out.update(verdict=base[1], why=base[2])
        return out
    if key in CONV_TORCH_ARMS:
        out["conv_impl"] = conv_compare((arms_recs_ref or {}).get(CONV_BASE), X)
        out.update(verdict=out["conv_impl"]["label"], why=out["conv_impl"]["why"])
        return out
    if key in SHORT_ARMS:                     # D4: the 71-block BF16 control, 8k only (recorded, never a winner)
        sp_ = T.speed_compare(T._win(Q), T._win(X), _wkeys()[:2], _wkeys()[:2])
        out.update(speed=sp_, verdict="SHORT_CONTROL_" + str(sp_.get("status")), why=f"8k vs Q {sp_.get('status')} (gain {sp_.get('gain')})",
                   capacity=capacity_compare(Q, X))
        return out
    qk = _wkeys()
    wq, wx = T._win(Q), T._win(X)
    k40q = next((k for k in wq if k.startswith("40k@")), None)
    k40x = next((k for k in wx if k.startswith("40k@")), None)
    spd = T.speed_compare(wq, wx, qk[:2], qk[:2])
    side = T.side_window(wq, wx, k40q or qk[3], k40x or qk[3])
    out["speed"], out["side_40k"] = spd, side
    out["stall"] = T.stall_compare(Q, X)
    out["long"] = {b: long_compare(Q, X, b) for b in LONG_BANDS}
    out["long_oracle"] = {"gate": lg, "vs_q_long": ((X.get("long_oracle") or {}).get("vs_q_long") or {}).get("status")}
    s = spd.get("status")
    if s == "PASS":
        if not (side["q_ok"] and side["x_ok"]):
            short = "INCONCLUSIVE_40K"
        elif (side.get("gain") or 0.0) < -SPEED_MIN:
            short = "REGRESSION_40K"
        else:
            short = "WIN"
    elif s in ("REGRESSION", "NO_GAIN"):
        short = s
    else:
        short = s or "INCONCLUSIVE_SPEED"
    lp, lbad = long_pass(out["long"])
    out["short_verdict"], out["long_pass"], out["long_bad"] = short, lp, lbad
    if short == "WIN":
        v = "WIN" if (lp and lg == "PASS" and (pg == "PASS" or PREFIX_RECORD_ONLY)) else "SHORT_ONLY_WIN"
    else:
        v = short
    out["memory_gates"] = {"ready": X.get("memory_gate_ready"), "first_decode": X.get("memory_gate_first_decode")}
    out["admission"] = {"deadlock": X.get("admission_deadlock"), "probes": X.get("admission_probes")}
    if v in ("WIN", "SHORT_ONLY_WIN") and X.get("admission_deadlock"):
        out["ranked_verdict"] = v
        v = "ADMISSION_DEADLOCK"
    if not memory_gates_ok(X):                # recorded (USER RULE 2026-11-01): a free-memory reading never overrides the measurement
        out["flags"] = sorted(set(out.get("flags") or []) | {"MEMORY_GATE_LOW_OR_MISSING"})
    if v in ("WIN", "SHORT_ONLY_WIN") and out.get("tf_verification_pending"):   # verdict only: the speed data stays intact
        out["speed_verdict"] = v
        v = "INCONCLUSIVE_TF"
    out.update(verdict=v, why=(f"8k {s} (gain {spd.get('gain')}, thr {spd.get('threshold')}); 40k {side.get('gain')} "
                               f"(valid {side['q_ok']} / {side['x_ok']}); long {'PASS' if lp else lbad} "
                               f"(90k {out['long']['90k'].get('gain')}, 116k {out['long']['116k'].get('gain')}); long oracle {lg}; "
                               f"prefix {pg}"))
    return out


def k100_rel(d_status, q_aa_clean):
    """A/B B1 (research A, coordinator decision 6): a D token-id / margin / ladder failure counts only when Q's own A/A is
    clean; on a non-deterministic Q it is NONDET_BASE_RECORDED (the teacher-forced envelope gate decides correctness)."""
    if q_aa_clean:
        return d_status
    return "NONDET_BASE_RECORDED" if d_status in ("FAIL_HIGH_MARGIN", "CONCURRENCY_CONTAMINATION") else d_status


def q_aa_clean(q_aa):
    """Q's A/A (greedy oracle run twice with a cache reset + its own ladder): clean when both are EXACT / NEAR_TIE."""
    q_aa = q_aa or {}
    return q_aa.get("oracle") in CLEAN and q_aa.get("ladder") in CLEAN


def tf_band_stats(cmp):
    c = cmp if isinstance(cmp, dict) else {}
    m, r = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
    ok = isinstance(m, (int, float)) and isinstance(r, (int, float)) and not isinstance(m, bool) and math.isfinite(m) and math.isfinite(r)
    return (float(m), float(r)) if ok else None


def tf_gate(d_cmp, q_aa_cmp, bands=("2k", "8k", "40k"), mean_margin=None, rank1_margin=None):
    """The hard correctness gate on a noisy base: teacher-forced prompt logprobs D vs Q per band, FAIL only outside Q's own
    A/A envelope (mean|dlp| > q_aa_mean + 0.05 or rank1 < q_aa_rank1 - 0.03). A band without both numbers is INCONCLUSIVE
    (recorded); any band outside the envelope -> FAIL_TF_OUTSIDE_Q_AA."""
    mm = TF_MEAN_MARGIN if mean_margin is None else mean_margin
    rm = TF_RANK1_MARGIN if rank1_margin is None else rank1_margin
    per, bad, missing = {}, [], []
    for b in bands:
        d, q = tf_band_stats((d_cmp or {}).get(b)), tf_band_stats((q_aa_cmp or {}).get(b))
        if d is None or q is None:
            per[b] = {"status": "INCONCLUSIVE_TF_MISSING", "d": d, "q_aa": q}
            missing.append(b)
            continue
        lim_m, lim_r = q[0] + mm, q[1] - rm
        st = "PASS" if (d[0] <= lim_m and d[1] >= lim_r) else "FAIL"
        per[b] = {"status": st, "d_mean": round(d[0], 4), "d_rank1": round(d[1], 4), "q_aa_mean": round(q[0], 4),
                  "q_aa_rank1": round(q[1], 4), "limit_mean": round(lim_m, 4), "limit_rank1": round(lim_r, 4)}
        if st == "FAIL":
            bad.append(b)
    status = "FAIL_TF_OUTSIDE_Q_AA" if bad else ("INCONCLUSIVE_TF" if missing else "PASS")
    return {"status": status, "per_band": per, "why": (f"outside Q's A/A envelope at {bad}" if bad else
                                                       f"missing {missing}" if missing else "every band inside Q's A/A envelope")}


def pos0_label(acc):
    """Research B early check (recorded): smoke per-position acceptance pos0 < 0.05 STRUCTURAL, 0.05-0.5 TAP_SUSPECT."""
    pp = (acc or {}).get("per_pos") or []
    if not pp or not isinstance(pp[0], (int, float)):
        return {"label": "NO_POS0", "pos0": None}
    p0 = float(pp[0])
    return {"label": "STRUCTURAL" if p0 < 0.05 else ("TAP_SUSPECT" if p0 < 0.5 else "OK"), "pos0": round(p0, 4)}


def cond_c(acc):
    """Mean conditional acceptance c from per-position rates (pos_i / pos_{i-1}; c_0 = pos0): research B's flat c."""
    pp = [float(x) for x in ((acc or {}).get("per_pos") or []) if isinstance(x, (int, float))]
    if not pp or pp[0] <= 0:
        return None
    cs = [pp[0]] + [pp[i] / pp[i - 1] for i in range(1, len(pp)) if pp[i - 1] > 0]
    return round(statistics.mean(cs), 4)


def prefix_ratios(pc):
    """Warm / cold AL per (prompt, turn) with the 0.8 and the 0.9 rule (recorded only in K100)."""
    out = []
    for pr in ((pc or {}).get("prompts") or []) if isinstance(pc, dict) else []:
        for t in ("2", "3"):
            tt = (pr.get("turns") or {}).get(t) or {}
            c, w = (tt.get("cold") or {}).get("acceptance") or {}, (tt.get("warm") or {}).get("acceptance") or {}
            if c.get("status") == "OK" and w.get("status") == "OK" and c.get("al"):
                r = w["al"] / c["al"]
                out.append({"variant": pr.get("variant"), "turn": t, "ratio": round(r, 4), "ge_0_8": r >= 0.8, "ge_0_9": r >= 0.9})
    return {"rows": out, "all_ge_0_8": all(x["ge_0_8"] for x in out) if out else None, "all_ge_0_9": all(x["ge_0_9"] for x in out) if out else None}


def _win_tok(rec, key):
    w = T._win(rec).get(key) if isinstance(T._win(rec), dict) else None
    return w.get("tok_s") if isinstance(w, dict) and T.window_ok(w)[0] else None


def k100_records(Q, X, key):
    """Record-only facts of one D arm vs Q (coordinator 2026-11-01): the concurrency trend (D / Q at 8k@16 and 20k@24), the
    smoke pos0 label, the conditional c at 8k@16 (the rejected D4s data gate c >= 0.80 and no-hope AL3 < 2.40, recorded),
    q_groups vs d_groups, the residents next to Q's, the NaN draft-score read-back (not available)."""
    Q, X = Q or {}, X or {}
    k1, k2 = (_wkeys()[:2] if WKEYS else (None, None))
    def ratio(kq, kx=None):
        q, x = _win_tok(Q, kq), _win_tok(X, kx or kq)
        return round(x / q, 4) if q and x else None
    k8 = [k for k in (k1, k2) if k]
    q8 = [v for v in (_win_tok(Q, k) for k in k8) if v]
    x8 = [v for v in (_win_tok(X, k) for k in k8) if v]
    trend = {"8k@16": round(statistics.mean(x8) / statistics.mean(q8), 4) if q8 and x8 else None, "20k@24": ratio("20k@24")}
    if trend["8k@16"] and trend["20k@24"]:
        trend["direction"] = "RISES_WITH_CONCURRENCY" if trend["20k@24"] > trend["8k@16"] else "FALLS_WITH_CONCURRENCY"
    sacc = (X.get("smoke") or {}).get("acceptance") or {}
    w8 = T._win(X).get(k1) if k1 and isinstance(T._win(X), dict) else None
    acc8 = {"per_pos": (w8 or {}).get("per_position_acceptance") or (w8 or {}).get("per_pos")} if isinstance(w8, dict) else {}
    c8 = cond_c(acc8) or cond_c(sacc)
    al3 = sacc.get("al")
    return {"conc_trend": trend, "pos0": pos0_label(sacc), "c_8k16": c8,
            "d4s_data_gate_c_ge_0_80": (c8 >= 0.80) if isinstance(c8, (int, float)) else None,
            "no_hope_smoke_al_lt_2_40": (al3 < 2.40) if isinstance(al3, (int, float)) and K_OF.get(key) == 3 else None,
            "kv_groups": kv_group_explosion(Q.get("kv_layout"), X.get("kv_layout"), (X.get("kv_kit_summary") or {}).get("drafter_group_count")),
            "metadata_build_ms": {"q": (Q.get("stage_compute") or {}).get("metadata_build_ms"), "d": (X.get("stage_compute") or {}).get("metadata_build_ms"),
                                  "note": "nb4_timeline has no attention / GDN metadata-build span: read it from the kept 30-s traces (profile_8k@16)"},
            "residents": {"q": Q.get("k100_residents"), "d": X.get("k100_residents")},
            "nan_draft_scores": X.get("nan_draft_scores") or {"status": "NOT_AVAILABLE", "why": "no cheap read-back hook (no engine patch)"},
            "tf_died_at_end": bool(X.get("tf_server_died")), "role": "recorded only"}


def throughput_eligible(v):
    if v.get("arm") in SHORT_ARMS:
        return False, "short-only arm (D4: the 71-block control; D4t: conv impl vs D4s; recorded)"
    if v.get("verdict") != "WIN":
        return False, f"verdict {v.get('verdict')}"
    if (v.get("speed") or {}).get("status") != "PASS":
        return False, "8k speed not PASS"
    side = v.get("side_40k") or {}
    if not (side.get("q_ok") and side.get("x_ok")) or (side.get("gain") or 0.0) < -SPEED_MIN:
        return False, "40k invalid / regression"
    if not v.get("long_pass"):
        return False, f"long bands {v.get('long_bad')}"
    if v.get("prefix_gate") != "PASS":
        return False, f"prefix {v.get('prefix_gate')}"
    return True, "ok"


def _brief(w):
    w = w or {}
    return {k: w.get(k) for k in ("status", "level", "tok_s", "tick_ms", "mean_acceptance_length", "running_eq_level", "preemptions",
                                  "kv_usage_max", "prefix_hit_rate")} | {"accept_per_pos": w.get("accept_rate_per_position_incl_admission"),
                                                                         "spec_log": w.get("spec_log")}


def arm_table(recs):
    out = {}
    for k, r in recs.items():
        if not isinstance(r, dict):
            continue
        w = T._win(r)
        out[k] = {"status": r.get("status"), "windows": {wk: _brief(v) for wk, v in w.items() if isinstance(v, dict) and not wk.startswith("profile")},
                  "long_levels": r.get("long_levels"), "spec_log_all": r.get("spec_log_all"), "oracle_acceptance": r.get("oracle_acceptance"),
                  "smoke_acceptance": (r.get("smoke") or {}).get("acceptance"), "prefix_gate": r.get("prefix_gate"),
                  "capacity": T.capacity_facts(r) if r.get("pool") else None, "kv_layout": r.get("kv_layout")}
        eights = [v.get("tok_s") for wk, v in w.items() if wk.startswith("8k@") and T.window_ok(v)[0]]
        out[k]["tok_s_8k_mean"] = round(statistics.mean(eights), 1) if eights else None
        for b in ("40k",) + LONG_BANDS:
            vv = [v for wk, v in w.items() if wk.startswith(b + "@") and T.window_ok(v)[0]]
            out[k][f"tok_s_{b}"] = [v.get("tok_s") for v in vv]
            out[k][f"tick_ms_{b}"] = [v.get("tick_ms") for v in vv]
            out[k][f"al_{b}"] = [v.get("mean_acceptance_length") for v in vv]
    return out


def verdict(arms_recs, arms=None):
    """Per-arm verdicts (D4 / D3 / D7 / D4b vs Q), the isolation labels, the throughput winner (highest 8k gain among WIN =
    short WIN + long PASS + long oracle PASS + prefix PASS), the AL / tok/s / tick table of every arm."""
    Q = arms_recs.get("Q") or {}
    base = base_usable(Q)
    strict = base_check(Q) if Q else (False, "INCONCLUSIVE_NOT_RUN", "Q not run")
    qf = qdefer_real_failure(T.latest_q(Q))
    out = {"lib": LIB_VERSION, "base": {"ok": base[0], "label": base[1], "why": base[2],
                                         "strict_nb4t_base_check": {"ok": strict[0], "label": strict[1], "why": strict[2], "role": "recorded only"},
                                         "qdefer": {"label": (T.latest_q(Q) or {}).get("label"), "real_failure": qf[0], "why": qf[1]},
                                         "memory_gate": "OK" if memory_gates_ok(Q) else "INCONCLUSIVE_MEMORY_GATE (recorded; Q is the control)",
                                         "q_long_oracle": long_oracle_gate(Q, True) if Q else None,
                                         "q_seam_inert": (Q.get("seam_inert") or {}).get("ok")},
           "criteria": {"correctness": "greedy oracle vs Q EXACT / NEAR_TIE (2k/2k/8k/40k rows and 90k/116k rows), 0 garbage (windowed)",
                        "short": f"8k gain > max(Q 8k spread, {SPEED_MIN:.0%}) on both 8k windows, 40k valid, 40k gain >= -{SPEED_MIN:.0%}",
                        "long": "90k and 116k windows valid at each arm's resident level, gain >= -max(Q A/A spread, 2 %)",
                        "prefix": f"warm 2-turn AL >= {PREFIX_WARM_MIN} x cold",
                        "winner": "WIN = short WIN + long PASS + long oracle PASS + prefix PASS; highest 8k gain; else SHORT_ONLY_WIN"},
           "parity": {k: T.parity_of(r) for k, r in arms_recs.items() if isinstance(r, dict)}, "arms": {}, "isolation": {}}
    arms_recs_ref.clear()
    arms_recs_ref.update(arms_recs)
    for k in D_ARMS:
        out["arms"][k] = arm_verdict(Q, arms_recs.get(k), k, base, arms)
        nm = arms_recs.get(k + "_NOMETA")
        if nm is not None:
            out["arms"][k]["ingraph_meta_fallback"] = {"label": "INGRAPH_META_FALLBACK", "status": nm.get("status"),
                                                       "smoke_gate": nm.get("smoke_gate"), "verdict": arm_verdict(Q, nm, k, base, arms).get("verdict")}
        tap = arms_recs.get(k + "_TAP")                 # E27 K100: the tap-mode diagnostic smoke (recorded)
        if tap is not None:
            out["arms"][k]["tap_mode_diagnostic"] = {"status": tap.get("status"), "smoke_gate": tap.get("smoke_gate"),
                                                     "pos0": tap.get("smoke_pos0"), "acceptance": (tap.get("smoke") or {}).get("acceptance")}
        iso = arms_recs.get(k + "_ISO")
        if iso is not None:
            out["isolation"][k] = {"label": isolation_label(arms_recs.get(k), iso), "iso_status": iso.get("status"),
                                   "iso_correctness": iso.get("correctness"), "iso_smoke": iso.get("smoke_gate")}
            out["arms"][k]["isolation"] = out["isolation"][k]["label"]
    out["throughput_eligible"] = {k: dict(zip(("ok", "why"), throughput_eligible(v))) for k, v in out["arms"].items()}
    elig = [(-(v["speed"]["gain"]), k) for k, v in out["arms"].items() if out["throughput_eligible"][k]["ok"]]
    if elig:
        best = min(elig)[1]
        out["best_arm"], out["best_verdict"], out["best_gain_8k"] = best, "WIN", out["arms"][best]["speed"]["gain"]
    else:
        out["best_arm"] = None
        so = [k for k, v in out["arms"].items() if v.get("verdict") == "SHORT_ONLY_WIN"]
        out["best_verdict"] = ("SHORT_ONLY_WIN:" + ",".join(so)) if so else ("NO_THROUGHPUT_WINNER" if base[0] else base[1])
    out["verdict"] = f"{out['best_arm']}:WIN" if out["best_arm"] else out["best_verdict"]
    out["table"] = arm_table(arms_recs)
    out["why"] = "; ".join(f"{k} {v['verdict']}" for k, v in out["arms"].items())
    return out


Q_TAIL = []                     # extra trailing Q argv (E27 K100: Q's --compilation-config for 24 sequences)
WKEYS = None                    # (8k #1, 8k #2, profile, 40k) window keys of the speed comparison; None = T.window_keys(SEQS)
TF_REQUIRED = False             # E27 K100: a WIN needs tf_gate PASS (W4 has no teacher-forced gate: off)
PREFIX_RECORD_ONLY = False      # E27 K100: the prefix cold / warm AL never decides a verdict (recorded, 0.8 and 0.9 ratios)
LONG_THR_FLOOR = {}             # E27 K100: per-band long threshold floor (one long window per arm: W4's measured A/A)
NONDET_OK = ("NONDET_BASE_RECORDED",)     # a token-id status judged against a non-deterministic Q (k100_rel): not a failure
TF_MEAN_MARGIN, TF_RANK1_MARGIN = 0.05, 0.03


def configure(**kw):
    """E27 NBDF-K100 (user / coordinator 2026-11-01): seqs 24, ladder (2, 8, 23), Q's captures, window keys, record-only
    prefix gate, no short exit, long threshold floors. Only known module knobs; returns the applied values."""
    allowed = {"SEQS", "CONC_LADDER", "Q_CAPTURES", "Q_TAIL", "WKEYS", "PREFIX_RECORD_ONLY", "SHORT_EXIT_ARMS", "LONG_THR_FLOOR",
               "ADMISSION_PROBE_ARMS", "TF_MEAN_MARGIN", "TF_RANK1_MARGIN", "TF_REQUIRED"}
    bad = set(kw) - allowed
    if bad:
        raise ValueError(f"configure: unknown knobs {sorted(bad)}")
    globals().update(kw)
    return {k: globals()[k] for k in kw}


def _wkeys():
    return tuple(WKEYS) if WKEYS else T.window_keys(SEQS)


def restrict_arms(keys):
    """E27 NBDF-K100: run only `keys` (Q first, in ORDER's order). Rebinds the module's arm-set constants (every function
    reads them at call time); the W4 NBDF never calls it. Returns the new ORDER."""
    global ORDER, D_ARMS, OPTIONAL, SW_BLOCK_ARMS, SHORT_ARMS, CONV_TORCH_ARMS, SHORT_EXIT_ARMS, ADMISSION_PROBE_ARMS, KV_RUN_OF
    keys = tuple(k for k in keys if k in ORDER)          # the given order (E27 K100: Q > D3s > D4s)
    if not keys or keys[0] != "Q" or len(set(keys)) != len(keys):
        raise ValueError(f"restrict_arms: Q must be in the set: {keys}")
    keep = lambda t: tuple(k for k in t if k in keys)      # noqa: E731
    ORDER, D_ARMS = keys, keys[1:]
    OPTIONAL, SW_BLOCK_ARMS, SHORT_ARMS = keep(OPTIONAL), keep(SW_BLOCK_ARMS), keep(SHORT_ARMS)
    CONV_TORCH_ARMS, SHORT_EXIT_ARMS, ADMISSION_PROBE_ARMS = keep(CONV_TORCH_ARMS), keep(SHORT_EXIT_ARMS), keep(ADMISSION_PROBE_ARMS)
    KV_RUN_OF = {k: v for k, v in KV_RUN_OF.items() if k in keys}
    return ORDER


# ------------------------------------------------------------------ budget
def arm_fits(left_sec, arm):
    return T.arm_fits(left_sec, arm)


def body_cap_sec(cap_min, left_sec):
    return T.body_cap_sec(cap_min, left_sec)


def budget_walk(minutes, hard_stop_min=HARD_STOP_MIN, setup_min=8.0, arms=None):
    """The planned walk (recorded + unit-tested): which arms start with est_min + 5 min left, given per-arm run minutes."""
    left = hard_stop_min - setup_min
    est = {k: a["est_min"] for k, a in (arms or {}).items()} or {"Q": 50, "D4s": 41, "D3s": 40, "D4": 22, "D4t": 22, "D7": 42, "D4b": 40}
    walk = []
    for k in ORDER:
        fits = left * 60 >= est[k] * 60 + START_RESERVE_SEC
        walk.append({"arm": k, "left_min": round(left, 1), "est_min": est[k], "runs": fits})
        if fits:
            left -= minutes.get(k, est[k])
    return {"walk": walk, "left_end_min": round(left, 1)}
