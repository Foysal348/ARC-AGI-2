# Per-GPU (TP4) memory model for Qwen3.8-27B RedHat INT4 on SGLang 0.5.21.
# Formulas: kv_cache_configurator.py:2232-2261 (ratio), 2462-2482 (explicit K intermediate), memory_pool.py:758-860, mamba_utils.py:116-157.
MiB = 1 << 20; GiB = 1 << 30
L = 48                      # GDN layers (64 layers, full_attention_interval 4)
hv, hk, dk, dv = 48 // 4, 16 // 4, 128, 128
conv_dim = (16 * 128 * 2 + 48 * 128) // 4   # 2560 per GPU
def ssm_slot(b): return L * hv * dv * dk * b
conv_slot = L * conv_dim * 3 * 2            # K-1 = 3, bf16
def conv_int(D): return L * conv_dim * (D + 2) * 2   # dedup window (memory_pool.py:786-833)
def ring(Lr=16):                             # GDN non-fold ReplaySSM ring, bf16 activations (memory_pool.py:664-755)
    return L * (hv * Lr * dv * 2 + hk * Lr * dk * 2 + hv * Lr * 4 + hv * Lr * dv * 2 + hk * Lr * dk * 2)
TARGET_KV = 16 * 1 * 256 * 2   # 16 full-attn layers x 1 kv head/GPU x 256 x fp8 x (K+V) = 8192
DRAFT_KV = {"dflash": 5 * 1 * 256 * 2, "mtp": 1 * 1 * 256 * 2}
# Pool budget (KV + mamba) at mem 0.83, measured: W8 KV 8.68 + mamba 3.04 = 11.72 GiB (E14 R2 graphs13 log); MTP weights 1.45 vs DFlash 1.49 GiB -> +0.04
POOL = {"dflash": 11.72, "mtp": 11.76}
def mamba(run, D, ssm_b=2, replay=False):
    mrr = run + 1
    K = 4 * (mrr + 1) if mrr % 1 == 0 else None
    K = {13: 56, 15: 60, 17: 68}[mrr]
    capped = min(mrr, K // 4)
    pers = (K + 1) * (ssm_slot(ssm_b) + conv_slot)
    inter = 0 if replay else (capped + 1) * D * ssm_slot(ssm_b)
    ci = (capped + 1) * conv_int(D)
    rg = (mrr + 1) * ring() if replay else 0
    return K, capped, pers / GiB, inter / GiB, ci / GiB, rg / GiB
print(f"ssm slot fp16 {ssm_slot(2)/MiB:.2f} MiB, fp32 {ssm_slot(4)/MiB:.2f} MiB, conv slot {conv_slot/MiB:.3f} MiB, ring/req {ring()/MiB:.2f} MiB")
arms = [("DFlash W8", "dflash", 8, 2, False), ("DFlash W6", "dflash", 6, 2, False),
        ("MTP 3/1/4 fp16", "mtp", 4, 2, False), ("MTP + ReplaySSM fp16", "mtp", 4, 2, True),
        ("MTP + ReplaySSM fp32", "mtp", 4, 4, True), ("MTP 2/1/3 fp16", "mtp", 3, 2, False)]
print("| running (mrr, K) | arm | persistent | intermediate SSM | conv int | ReplaySSM ring | mamba total | KV GiB | KV tokens | per stream |")
for run in (12, 14, 16):
    for name, fam, D, b, rp in arms:
        K, cap, p, i, c, r = mamba(run, D, b, rp)
        tot = p + i + c + r
        kv = POOL[fam] - tot
        tok = kv * GiB / (TARGET_KV + DRAFT_KV[fam])
        print(f"| {run} ({run+1}, {K}) | {name} | {p:.2f} | {i:.2f} | {c:.3f} | {r:.3f} | {tot:.2f} | {kv:.2f} | {tok/1e3:,.0f}k | {tok/run/1e3:,.1f}k |")
