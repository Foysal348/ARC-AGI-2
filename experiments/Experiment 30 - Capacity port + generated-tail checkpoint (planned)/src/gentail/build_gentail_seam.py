"""Builds gentail_seam.py from gentail_seam_template.py + gentail_ckpt_module.py and the pristine Merlin tree's shas.
Run: python -I build_gentail_seam.py"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERLIN = Path("D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
SCH_REL = "vllm/v1/core/sched/scheduler.py"
KVM_REL = "vllm/v1/core/kv_cache_manager.py"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    tpl = (HERE / "gentail_seam_template.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    mod = (HERE / "gentail_ckpt_module.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "'''" not in mod and "@@" not in mod
    src = tpl.replace("@@MODULE@@", mod)
    dummy = 'SCH_PRE = SCH_POST = KVM_PRE = KVM_POST = NEW_SHA = ""'
    ns = {"__name__": "gentail_seam_build"}
    exec(compile(src.replace("@@SHAS@@", dummy), "gentail_seam_build", "exec"), ns)
    pre = {rel: (MERLIN / rel).read_bytes() for rel in (SCH_REL, KVM_REL)}
    post = {rel: ns["patch_text"](pre[rel].decode("utf-8"), ns["EDITS"][rel]).encode("utf-8") for rel in pre}
    new_sha = sha(ns["new_module_bytes"]())
    shas = (
        f'SCH_PRE = "{sha(pre[SCH_REL])}"\n'
        f'SCH_POST = "{sha(post[SCH_REL])}"\n'
        f'KVM_PRE = "{sha(pre[KVM_REL])}"\n'
        f'KVM_POST = "{sha(post[KVM_REL])}"\n'
        f'NEW_SHA = "{new_sha}"'
    )
    out = src.replace("@@SHAS@@", shas)
    (HERE / "gentail_seam.py").write_bytes(out.encode("utf-8"))
    print(shas)
    print("gentail_seam.py sha256", sha(out.encode("utf-8")))


if __name__ == "__main__":
    sys.exit(main())
