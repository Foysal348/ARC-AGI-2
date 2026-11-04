"""E34 (2026-11-03): Merlin's REAL KV planner on CPU for the E34 arms (no GPU, no model load).

Derived from the capacity port's measure_capacity.py (D:/ARC AGI 2/_scratch/capacity_port, sha pinned below; read only):
the same klee100 K3 GPU worker record, the same FakeL4 platform, the same spec construction and Merlin's own
get_kv_cache_configs / generate_scheduler_kv_cache_config / get_max_concurrency_for_kv_cache_config, with three E34 knobs:
  block    the attention / Mamba (align) block: the GPU record's 4096-token specs are re-made at this block (page scaled)
  ssm      the GDN SSM state dtype (state_dtypes[1] of every linear_attn MambaSpec): float32 (record) or bfloat16
  drafter  the DFlash2 SW drafter spec (block 1024 = production DFLASH_DRAFT_SW_BLOCK, or 512)
The Mamba page is padded to the attention page block x 512 B (platforms/interface.py:_align_hybrid_block_size, the
baseline path; the capacity-port P1 env stays unset). A Mamba state page above that page = the platform would RAISE the
block (interface.py 'Setting attention block size'): recorded BLOCK_RAISED, no plan.

  <capacity venv python> -B e34_capacity_probe.py [out.json]
The derived script is written to the scratch dir (never into the capacity-port folder)."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = Path(r"D:/ARC AGI 2/_scratch/capacity_port")
SRC = CAP / "measure_capacity.py"
SRC_SHA = "27d8b1c963a2ab972968a02d348dc73757a98c8fdd83aa841b8b7677877c23f1"
TREE = str(CAP / "merlin_copy")
SCR = Path(os.environ.get("E34_SCRATCH") or r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/e34_cap")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "Experiment Results" / "e34_capacity_probe.json"

DRAFTERS = {
    "bf16_1024": {"type": "SlidingWindowSpec", "block_size": 1024, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048},
    "bf16_512": {"type": "SlidingWindowSpec", "block_size": 512, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048},
}
ARMS = [
    {"name": "B: fp32 SSM, block 4096, SW1024 (production D3s)", "k": 3, "drafter": "bf16_1024", "block": 4096, "ssm": "float32"},
    {"name": "S16: bf16 SSM, block 4096, SW1024", "k": 3, "drafter": "bf16_1024", "block": 4096, "ssm": "bfloat16"},
    {"name": "S16B2 (SW1024): bf16 SSM, block 2048, SW1024", "k": 3, "drafter": "bf16_1024", "block": 2048, "ssm": "bfloat16"},
    {"name": "S16B2: bf16 SSM, block 2048, SW512", "k": 3, "drafter": "bf16_512", "block": 2048, "ssm": "bfloat16"},
    {"name": "fp32 SSM, block 2048, SW512 (control: the platform raises the block)", "k": 3, "drafter": "bf16_512", "block": 2048, "ssm": "float32"},
]
SEQS = (24, 28, 32, 36, 40)

# exact-once edits of the pinned source (each: old, new)
EDITS = [
    ('SEQS = (24, 28, 32)\n', 'SEQS = (24, 28, 32, 36, 40)\nE34_BLOCK = 4096\n'),
    ('    sys.path.insert(0, str(Path(__file__).with_name("stubs")))',
     '    sys.path.insert(0, ' + repr(str(CAP / "stubs")) + ')\n    sys.path.insert(0, ' + repr(str(CAP)) + ')'),
    ('    arm = json.loads(arm_json)\n',
     '    arm = json.loads(arm_json)\n    global E34_BLOCK\n    E34_BLOCK = int(arm.get("block", 4096))\n    E34_SSM = arm.get("ssm", "float32")\n'),
    ('        s["state_shapes"] = sh\n        return s\n',
     '        s["state_shapes"] = sh\n'
     '        if s["type"] == "MambaSpec" and "linear_attn" in ex and E34_SSM != "float32":   # E34: the GDN SSM state dtype\n'
     '            s["state_dtypes"] = [s["state_dtypes"][0], E34_SSM] + list(s["state_dtypes"][2:])\n'
     '        if s.get("block_size") == 4096 and s["type"] in ("FullAttentionSpec", "MLAAttentionSpec", "MambaSpec"):   # E34 block\n'
     '            s["block_size"] = E34_BLOCK\n'
     '            if s["type"] != "MambaSpec" and isinstance(s.get("page_size_bytes"), int):\n'
     '                s["page_size_bytes"] = s["page_size_bytes"] * E34_BLOCK // 4096\n'
     '        return s\n'),
    ('        pad = 4096 * 512\n', '        pad = E34_BLOCK * 512\n    if max_mamba > E34_BLOCK * 512:   # E34: the platform would raise the block\n'
     '        print("@JSON " + json.dumps({"arm": arm["name"], "BLOCK_RAISED": True, "max_mamba_state_page": max_mamba, "attn_page": E34_BLOCK * 512}))\n'
     '        return\n'),
    ('        cc.block_size, cc.enable_prefix_caching, cc.cache_dtype = 4096, True, "fp8_e4m3"',
     '        cc.block_size, cc.enable_prefix_caching, cc.cache_dtype = E34_BLOCK, True, "fp8_e4m3"'),
    ('            res[basis][s] = {L: min(s, (usable - shared) // (math.ceil(L / 4096) + f)) for L in LENS}',
     '            res[basis][s] = {L: min(s, (usable - shared) // (math.ceil(L / E34_BLOCK) + f)) for L in LENS}'),
    ('        ctx = -(-(P + D) // 4096)', '        ctx = -(-(P + D) // E34_BLOCK)'),
    ('    out = {"arm": arm["name"], "layout": lay.name,', '    out = {"arm": arm["name"], "block": E34_BLOCK, "ssm": E34_SSM, "layout": lay.name,'),
]


def derived_source():
    src = SRC.read_bytes()
    assert hashlib.sha256(src).hexdigest() == SRC_SHA, "measure_capacity.py changed"
    t = src.decode("utf-8").replace("\r\n", "\n")
    for old, new in EDITS:
        assert t.count(old) == 1, ("edit anchor", old[:60], t.count(old))
        t = t.replace(old, new)
    return t


def main():
    SCR.mkdir(parents=True, exist_ok=True)
    der = SCR / "e34_measure_capacity_derived.py"
    der.write_bytes(derived_source().encode("utf-8"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MERLIN_MAMBA_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = []
    for arm in ARMS:
        a = dict(arm, drafter=DRAFTERS[arm["drafter"]])
        p = subprocess.run([sys.executable, "-B", str(der), "--child", json.dumps(a), TREE], capture_output=True, text=True,
                           env=env, timeout=1200, cwd=str(SCR))
        line = next((l for l in p.stdout.splitlines() if l.startswith("@JSON ")), None)
        r = json.loads(line[6:]) if line else {"arm": arm["name"], "error": (p.stderr or p.stdout)[-2500:]}
        r["drafter_key"], r["e34_arm"] = arm["drafter"], {k: v for k, v in arm.items() if k != "drafter"}
        res.append(r)
        print(arm["name"], "->", r.get("num_blocks"), "stride", r.get("stride_per_rank"), "F", r.get("F_admission"), r.get("F_steady"),
              "mc", r.get("max_concurrency_131072"), r.get("BLOCK_RAISED") or "", (r.get("error") or "")[-400:], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps({"source": str(SRC), "source_sha256": SRC_SHA, "tree": TREE, "derived_sha256":
                                 hashlib.sha256(der.read_bytes()).hexdigest(), "results": res}, indent=1, default=str) + "\n").encode("utf-8"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
