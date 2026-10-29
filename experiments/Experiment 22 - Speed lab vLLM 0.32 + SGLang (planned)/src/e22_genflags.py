"""Every E22 vLLM Speed Lab launch command, generated from the built notebook's own vms_flags() and PLAN text
(Kaggle paths), into src/e22_flags.json in the e21_flagcheck.py format: {key: {"vllm": "0.32.0", "flags": [...], "env": {...}}}."""
import ast, json, re
from pathlib import Path
E22 = Path(__file__).resolve().parents[1]
nb = json.load(open(E22 / "Experiment Notebooks/Q27-XHIGH-E22_SPEEDLAB_VLLM032.ipynb", encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
fn = next(ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "vms_flags")
i0, i1 = src.index("    _FULL = {"), src.index('    VMS["dropped"] = {')
plan = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i0:i1].splitlines())
g = {"json": json, "SEED": 20261005, "VMS_PORT": 8011,
     "MODEL_PATH": "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
     "VMS_INT4": "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1"}
exec(fn, g); exec(plan, g)
out = {"ref_fp8_mtp3_s16": {"vllm": "0.32.0", "flags": g["vms_flags"](g["MODEL_PATH"]), "env": {}},
       "fp8_target_only_noprefix": {"vllm": "0.32.0", "flags": g["vms_flags"](g["MODEL_PATH"], spec=0, prefix=False), "env": {}},
       "int4_target_only_noprefix": {"vllm": "0.32.0", "flags": g["vms_flags"](g["VMS_INT4"], seqs=20, spec=0, prefix=False), "env": {}}}
for name, fl, kw in g["PLAN"] + g["PLAN_INT4"]:
    out[name] = {"vllm": "0.32.0", "flags": [str(x) for x in fl], "env": kw.get("env_extra") or {}}
(E22 / "src/e22_flags.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(len(out), "commands")
for k, v in out.items():
    f = v["flags"]; sc = f[f.index("--speculative-config") + 1] if "--speculative-config" in f else None
    extra = [x for x in f if x.startswith("--") and x in ("--prefix-match-unit", "--enable-mamba-shared-prefix-checkpoint", "--mamba-ssm-cache-dtype", "--compilation-config", "--attention-backend")]
    print(f"{k:30s} seqs={f[f.index('--max-num-seqs')+1]:>3} spec={sc} {extra} env={v['env']}")
