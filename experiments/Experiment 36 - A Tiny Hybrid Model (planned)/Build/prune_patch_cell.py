# E36 load-time expert pruning (inserted before the server launch cell; runs after all Domdolus patches).
# keep file: {"num_experts": 320, "layers": {"0": [sorted original expert ids], ..., "47": [...]}}
# - checkpoint expert tensors of pruned experts are never read; kept ones are renamed to 0..K-1 per layer
#   (before vLLM's EP weight filter, so each rank loads only its own new-id experts);
# - router rows model.language_model.layers.L.mlp.gate.weight [512, H] -> [K, H] in keep order;
# - MODEL_DIR becomes /tmp/e36_model: symlinks to every file of the original model except config.json,
#   whose text_config.num_experts is set to K.
import json as _j, os as _o, py_compile as _pc, re as _r
PRUNE_KEEP = globals().get("PRUNE_KEEP") or next(
    (str(p) for p in __import__("pathlib").Path("/kaggle/input").rglob("e36_keep_*.json")), None)
assert PRUNE_KEEP and _o.path.exists(PRUNE_KEEP), "E36: no keep file (e36_keep_*.json) attached"
_keep = _j.load(open(PRUNE_KEEP))
_K = int(_keep["num_experts"])
assert len(_keep["layers"]) == 48 and all(len(v) == _K and len(set(v)) == _K for v in _keep["layers"].values()), "E36: bad keep file"
_o.environ["FLASHNEXT_E36_KEEP"] = PRUNE_KEEP

_F_WU = "/tmp/site-vllm/vllm/model_executor/model_loader/weight_utils.py"
_LIB = '''

# ---- E36 expert pruning ----
_E36 = [None]


def _e36_state():
    if _E36[0] is None:
        import json as _json
        p = os.environ.get("FLASHNEXT_E36_KEEP")
        if not p:
            _E36[0] = False
        else:
            k = _json.load(open(p))
            _E36[0] = {int(L): {int(e): i for i, e in enumerate(ids)} for L, ids in k["layers"].items()}
            _E36.append({int(L): [int(e) for e in ids] for L, ids in k["layers"].items()})
            logger.info("E36 pruning: %d layers, %d experts kept per layer", len(_E36[0]), len(next(iter(_E36[0].values()))))
    return _E36[0]


_E36_RX = __import__("re").compile(r"(\\.layers\\.(\\d+)\\.mlp\\.experts\\.)(\\d+)(\\.)")


def _e36_prune_name(name):
    st = _e36_state()
    if not st:
        return name
    m = _E36_RX.search(name)
    if m is None:
        return name
    new = st.get(int(m.group(2)), {}).get(int(m.group(3)))
    if new is None:
        return None
    return name[: m.start()] + m.group(1) + str(new) + m.group(4) + name[m.end():]


_E36_GATE = __import__("re").compile(r"\\.layers\\.(\\d+)\\.mlp\\.gate\\.weight$")


def _e36_prune_tensor(name, t):
    st = _e36_state()
    if not st:
        return t
    m = _E36_GATE.search(name)
    if m is None:
        return t
    ids = _E36[1][int(m.group(1))]
    return t[ids].contiguous()
'''
src = open(_F_WU).read()
if "E36 expert pruning" not in src:
    _pat = _r.compile(r"( +)for name in f\.keys\(\):  # noqa: SIM118\n(\1    )if should_skip_weight\(name, local_expert_ids\)([^\n]*):\n\2    continue\n\2param = f\.get_tensor\(name\)\n\2yield name, param\n")
    _hits = _pat.findall(src)
    assert len(_hits) == 1, f"E36: weight_utils default-branch anchor count {len(_hits)}"
    def _rep(m):
        a, b, rest = m.group(1), m.group(2), m.group(3)
        return (f"{a}for _e36_orig in f.keys():  # noqa: SIM118\n"
                f"{b}name = _e36_prune_name(_e36_orig)\n"
                f"{b}if name is None:\n{b}    continue\n"
                f"{b}if should_skip_weight(name, local_expert_ids){rest}:\n{b}    continue\n"
                f"{b}param = _e36_prune_tensor(name, f.get_tensor(_e36_orig))\n"
                f"{b}yield name, param\n")
    src = _pat.sub(_rep, src, count=1)
    if "\nimport os\n" not in src:
        src = "import os\n" + src
    src += _LIB
    open(_F_WU, "w").write(src)
_pc.compile(_F_WU, doraise=True)

_SRC_DIR = str(MODEL_DIR)
_DST = "/tmp/e36_model"
_o.makedirs(_DST, exist_ok=True)
for _n in _o.listdir(_SRC_DIR):
    _d = _o.path.join(_DST, _n)
    if _n == "config.json" or _o.path.lexists(_d):
        continue
    _o.symlink(_o.path.join(_SRC_DIR, _n), _d)
_cfg = _j.load(open(_o.path.join(_SRC_DIR, "config.json")))
(_cfg.get("text_config") or _cfg)["num_experts"] = _K
_j.dump(_cfg, open(_o.path.join(_DST, "config.json"), "w"), indent=2)
MODEL_DIR = type(MODEL_DIR)(_DST) if not isinstance(MODEL_DIR, str) else _DST
print(f"[E36] pruning ON: {_K} experts per layer from {PRUNE_KEEP} | MODEL_DIR -> {_DST} (config num_experts {_K})")
