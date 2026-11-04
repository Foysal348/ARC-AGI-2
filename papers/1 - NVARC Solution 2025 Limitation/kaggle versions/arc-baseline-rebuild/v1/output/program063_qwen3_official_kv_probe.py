import os
import sys
from pathlib import Path
source_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
overlay_root = Path(os.environ["PROGRAM063_OVERLAY_ROOT"]).resolve()
clean = []
for item in sys.path:
    if not item:
        clean.append(item)
        continue
    try:
        resolved = Path(item).resolve()
    except OSError:
        continue
    if resolved == source_root or source_root in resolved.parents or resolved == overlay_root:
        continue
    clean.append(item)
sys.path[:] = [str(overlay_root)] + clean

import json
import torch
import torch.nn.functional as F
import unsloth.models.qwen3 as qwen3

fn = getattr(qwen3, "_program063_qwen3_cached_attention", None)
if not callable(fn):
    raise RuntimeError("PROGRAM063_QWEN3_OFFICIAL_HELPER_MISSING")
backend = getattr(fn, "__program063_backend__", None)
if backend != "official_style_kv_cache":
    raise RuntimeError(f"PROGRAM063_QWEN3_BACKEND={backend}")

torch.manual_seed(47047)
device = torch.device("cuda:0")
dtype = torch.bfloat16
reports = []


def explicit_reference(q, k, v, *, mask=None, sliding_window=None, batch_one=False):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    if sliding_window is not None and k_t.shape[-2] > sliding_window:
        start = k_t.shape[-2] - sliding_window
        k_t = k_t[:, :, start:, :]
        v_t = v_t[:, :, start:, :]
        if mask is not None:
            mask = mask[..., start:]
    if mask is not None:
        if mask.dim() == 2:
            mask = mask[:, None, None, :].to(torch.bool)
        elif mask.dim() == 4 and mask.dtype != torch.bool:
            mask = mask.eq(0)
        if mask.shape[-1] > k_t.shape[-2]:
            mask = mask[..., -k_t.shape[-2]:]
    groups = q_t.shape[1] // k_t.shape[1]
    if groups != 1:
        k_t = k_t.repeat_interleave(groups, dim=1)
        v_t = v_t.repeat_interleave(groups, dim=1)
    if batch_one:
        scores = torch.matmul(q_t * (q_t.shape[-1] ** -0.5), k_t.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_t.dtype)
        out = torch.matmul(probs, v_t)
    else:
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=(mask is None and q_t.shape[-2] == k_t.shape[-2]),
        )
    return out.transpose(1, 2).contiguous()


cases = [
    {"name": "batched_gqa_decode", "batch": 4, "q_heads": 32, "kv_heads": 8, "q_len": 1, "kv_len": 17},
    {"name": "batched_equal_heads_causal", "batch": 4, "q_heads": 8, "kv_heads": 8, "q_len": 3, "kv_len": 3},
    {"name": "batch_one_explicit", "batch": 1, "q_heads": 32, "kv_heads": 8, "q_len": 1, "kv_len": 17},
    {"name": "sliding_window", "batch": 4, "q_heads": 32, "kv_heads": 8, "q_len": 1, "kv_len": 17, "sliding_window": 8},
    {"name": "batched_masked_decode", "batch": 4, "q_heads": 32, "kv_heads": 8, "q_len": 1, "kv_len": 17, "masked": True},
]
for case in cases:
    q = torch.randn(case["batch"], case["q_len"], case["q_heads"], 16, device=device, dtype=dtype)
    k = torch.randn(case["batch"], case["kv_len"], case["kv_heads"], 16, device=device, dtype=dtype)
    v = torch.randn(case["batch"], case["kv_len"], case["kv_heads"], 16, device=device, dtype=dtype)
    mask = None
    if case.get("masked"):
        mask = torch.ones(case["batch"], case["kv_len"], device=device, dtype=torch.long)
        mask[:, :2] = 0
    window = case.get("sliding_window")
    actual = fn(q, k, v, attention_mask=mask, sliding_window=window)
    expected = explicit_reference(
        q,
        k,
        v,
        mask=mask,
        sliding_window=window,
        batch_one=(case["batch"] == 1),
    )
    torch.cuda.synchronize()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise RuntimeError(
            f"PROGRAM063_QWEN3_BAD_OUTPUT case={case['name']} actual={actual.shape} expected={expected.shape}"
        )
    max_abs_error = float((actual.float() - expected.float()).abs().max().cpu())
    if max_abs_error > 0.03:
        raise RuntimeError(
            f"PROGRAM063_QWEN3_PARITY_FAILED case={case['name']} max_abs_error={max_abs_error}"
        )
    reports.append({
        "name": case["name"],
        "batch": case["batch"],
        "q_heads": case["q_heads"],
        "kv_heads": case["kv_heads"],
        "q_len": case["q_len"],
        "kv_len": case["kv_len"],
        "sliding_window": window,
        "masked": bool(case.get("masked")),
        "shape": list(actual.shape),
        "max_abs_error": max_abs_error,
    })
print("PROGRAM063_QWEN3_OFFICIAL_KV_PROBE_OK=" + json.dumps({
    "backend": backend,
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(0),
    "cases": reports,
}, sort_keys=True))
