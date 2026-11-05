# E36 gEAN calibration hook (inserted before the server launch cell).
# gEAN score of expert e in layer L = sum over routed tokens of (router weight) x ||expert output||
# (Yi-Chia Chen, discussion 755801). Measured in the Humming indexed MoE apply, on the per-(token, slot)
# expert outputs right before the weighted sum. Each process (TP/EP rank) adds its local experts and
# writes /kaggle/working/gean/gean_<pid>.pt every 60 s: {layer: float64 tensor [3, E+2]}
# rows = (gEAN sum, routed count, router-weight sum). Needs --enforce-eager (S["eager"] = True).
import os as _os, re as _re, py_compile as _pc
_SITE = "/tmp/site-vllm/vllm"
_F_MOE = f"{_SITE}/model_executor/layers/fused_moe/experts/fused_humming_moe.py"
_F_MODEL = f"{_SITE}/models/qwen4_exp/nvidia/model.py"
_os.makedirs("/kaggle/working/gean", exist_ok=True)

_HOOK_LIB = '''

# ---- E36 gEAN hook ----
GEAN_LAYER = [-1]
_GEAN = {}
_GEAN_T = [None]


def _gean_dump():
    import time as _t
    while True:
        _t.sleep(60)
        try:
            snap = {k: v.detach().to("cpu") for k, v in list(_GEAN.items())}
            p = f"/kaggle/working/gean/gean_{os.getpid()}.pt"
            torch.save(snap, p + ".part")
            os.replace(p + ".part", p)
        except Exception as e:
            print("[GEAN] dump failed:", e, flush=True)


def _gean_acc(down, w, ids, emap, n_experts):
    L = GEAN_LAYER[0]
    if L < 0:
        return
    if _GEAN_T[0] is None:
        import threading as _th
        _GEAN_T[0] = _th.Thread(target=_gean_dump, name="gean-dump", daemon=True)
        _GEAN_T[0].start()
        print(f"[GEAN] hook active in pid {os.getpid()}", flush=True)
    acc = _GEAN.get(L)
    if acc is None:
        acc = _GEAN[L] = torch.zeros(3, n_experts + 2, dtype=torch.float64, device=down.device)
    m = ids.shape[0]
    ids_l = ids.long()
    valid = (ids_l >= 0) & (ids_l <= n_experts)
    safe = ids_l.clamp(min=0, max=n_experts)
    if emap is not None:
        valid &= emap[safe.clamp(max=emap.numel() - 1)] >= 0
    v = valid.double()
    nrm = down[:m].float().norm(dim=-1).double()
    wf = w[:m].double()
    idx = safe.flatten()
    acc[0].index_add_(0, idx, (wf * nrm * v).flatten())
    acc[1].index_add_(0, idx, v.flatten())
    acc[2].index_add_(0, idx, (wf * v).flatten())
'''

_ANCHOR = '''        moe_fused_mul_sum(
            inputs=buffers["down_output"].view(*topk_ids.shape, -1),'''
_CALL = '''        _gean_acc(buffers["down_output"].view(*topk_ids.shape, -1), topk_weights, topk_ids, expert_map, global_num_experts)
'''

src = open(_F_MOE).read()
if "E36 gEAN hook" not in src:
    assert src.count(_ANCHOR) == 1, f"gEAN anchor count {src.count(_ANCHOR)} in {_F_MOE}"
    if "\nimport os\n" not in src:
        src = src.replace("\nimport json\n", "\nimport json\nimport os\n", 1)
    src = src.replace(_ANCHOR, _CALL + _ANCHOR, 1) + _HOOK_LIB
    open(_F_MOE, "w").write(src)
_pc.compile(_F_MOE, doraise=True)

_MODEL_ANCHOR = '''        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
'''
_MODEL_ADD = '''        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
        import re as _gre
        _gm = _gre.search(r"layers\\.(\\d+)\\.", prefix + ".")
        self._gean_layer = int(_gm.group(1)) if _gm else -1

    def forward(self, hidden_states, already_sequence_parallel=False):
        from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe as _ghm
        _ghm.GEAN_LAYER[0] = self._gean_layer
        return super().forward(hidden_states, already_sequence_parallel)
'''
msrc = open(_F_MODEL).read()
if "_gean_layer" not in msrc:
    assert msrc.count(_MODEL_ANCHOR) == 1, f"model anchor count {msrc.count(_MODEL_ANCHOR)}"
    msrc = msrc.replace(_MODEL_ANCHOR, _MODEL_ADD, 1)
    open(_F_MODEL, "w").write(msrc)
_pc.compile(_F_MODEL, doraise=True)
print("[GEAN] hook installed:", _F_MOE, "|", _F_MODEL, "| dumps -> /kaggle/working/gean/ every 60 s")
