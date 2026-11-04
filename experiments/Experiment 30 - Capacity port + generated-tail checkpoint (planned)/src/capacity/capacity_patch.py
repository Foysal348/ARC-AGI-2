"""Capacity port patch for Merlin (vLLM 0.32 fork): env-gated, every env unset = the original statements run.

  P1  MERLIN_MAMBA_PAD_TO_STATE=1   vllm/platforms/interface.py  _align_hybrid_block_size (pristine :1106-1126)
      Pad every Mamba state page only up to the LARGEST Mamba state page (GDN 1,732,992 B at K3 on klee100 / W4), not up
      to the 4096-token attention page (2,097,152 B). All Mamba layouts (GDN, PLE short conv) keep ONE common page, so
      they still bucket / fold together; the block-outermost packer (kv_cache_utils.py:1894-2028) already packs mixed
      page sizes side by side, so the attention page no longer has to equal the Mamba page. Use ONLY with a
      block-outermost resolved layout (BLNHC / BLHNC; klee100 + W4 resolve BLNHC): a layer-outermost layout needs equal
      pages and would fail in unify_kv_cache_spec_page_size.
  P2  MERLIN_MAMBA_GROUPS=<n>       vllm/v1/core/kv_cache_utils.py  _get_packed_kv_cache_groups (pristine :2000-2004)
      Force the sharded Mamba bucket into exactly n groups (measurement knob; the recommended setting is NOT to set it).
  P4  MERLIN_MAMBA_RETAIN_CKPT=0    vllm/v1/core/single_type_kv_cache_manager.py  _RETAIN_CHECKPOINT (pristine :51-53)
      Release-on-retire control arm (Merlin's own test switch made env-settable): an align-Mamba request no longer pins
      its newest hashed state as a prefix checkpoint (-1 block per Mamba group per request, 3 on klee100) at the price
      of that checkpoint becoming evictable. A/B arm only.

Pre-patch shas (Merlin source D:/ARC AGI 2/_scratch/e23/p2/merlin-src, LF):
  platforms/interface.py      0713467e36ac637fcc2a5c9da56c449e6d000854c7e07738b91a076971b2a517
  v1/core/kv_cache_utils.py   2126a2373ee73aa8d445cdc78d4a28ff12cc3aa7612bee40c8d471b12783c6e7
  v1/core/single_type_kv_cache_manager.py  e7a2698e527dee0f8495b3c542d58ff2216b0ed70ff1863aa6ba9dbe710f1cf0
Post-patch shas are computed and printed by `apply` and checked by `verify` (POST below, filled from the local apply).

CLI:  python capacity_patch.py apply|verify|revert <dir that contains the vllm package>
State per file: PRISTINE (sha == PRE), PATCHED (sha == POST), FOREIGN (else; apply still works when every anchor
matches exactly once and no marker is present; revert works on any file carrying the markers).
"""
import hashlib, sys, py_compile
from pathlib import Path

MARK = "CAPACITY-PORT"

P1_OLD = '''        if attn_page_size == mamba_page_size:
            return

        if (
            cache_config.mamba_page_size_padded is None
            or cache_config.mamba_page_size_padded != attn_page_size
        ):'''
P1_NEW = '''        if attn_page_size == mamba_page_size:
            return

        if os.environ.get("MERLIN_MAMBA_PAD_TO_STATE") == "1":  # CAPACITY-PORT P1
            # CAPACITY-PORT P1: pad every Mamba layout only to the largest Mamba
            # state page; the block-outermost packer packs mixed pages.
            cache_config.mamba_page_size_padded = mamba_page_size  # CAPACITY-PORT P1
            logger.info(  # CAPACITY-PORT P1
                "CAPACITY-PORT P1: Mamba page padded to the largest Mamba state "  # CAPACITY-PORT P1
                "page %d B (not the attention page %d B).",  # CAPACITY-PORT P1
                mamba_page_size,  # CAPACITY-PORT P1
                attn_page_size,  # CAPACITY-PORT P1
            )  # CAPACITY-PORT P1
            return  # CAPACITY-PORT P1

        if (
            cache_config.mamba_page_size_padded is None
            or cache_config.mamba_page_size_padded != attn_page_size
        ):'''

P2_OLD = '''            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        if num_groups == 1:'''
P2_NEW = '''            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        _cp_groups = os.environ.get("MERLIN_MAMBA_GROUPS")  # CAPACITY-PORT P2
        if (  # CAPACITY-PORT P2
            _cp_groups  # CAPACITY-PORT P2
            and isinstance(spec.first_spec, MambaSpec)  # CAPACITY-PORT P2
            and not spec.first_spec.tp_replicated  # CAPACITY-PORT P2
        ):  # CAPACITY-PORT P2
            num_groups = max(1, min(int(_cp_groups), len(spec.kv_cache_specs)))  # CAPACITY-PORT P2
        if num_groups == 1:'''

P4_OLD = '''# Align-mode Mamba prefix-checkpoint retention (MambaManager). Always on;
# a module constant only so tests can run the release-on-retire control arm.
_RETAIN_CHECKPOINT = True
'''
P4_NEW = '''# Align-mode Mamba prefix-checkpoint retention (MambaManager). Always on;
# a module constant only so tests can run the release-on-retire control arm.
_RETAIN_CHECKPOINT = __import__("os").environ.get(  # CAPACITY-PORT P4
    "MERLIN_MAMBA_RETAIN_CKPT", "1"  # CAPACITY-PORT P4
) != "0"  # CAPACITY-PORT P4
'''

FILES = {
    "vllm/platforms/interface.py": {"pre": "0713467e36ac637fcc2a5c9da56c449e6d000854c7e07738b91a076971b2a517",
                                    "post": None, "edits": [(P1_OLD, P1_NEW)]},
    "vllm/v1/core/kv_cache_utils.py": {"pre": "2126a2373ee73aa8d445cdc78d4a28ff12cc3aa7612bee40c8d471b12783c6e7",
                                       "post": None, "edits": [(P2_OLD, P2_NEW)]},
    "vllm/v1/core/single_type_kv_cache_manager.py": {
        "pre": "e7a2698e527dee0f8495b3c542d58ff2216b0ed70ff1863aa6ba9dbe710f1cf0", "post": None,
        "edits": [(P4_OLD, P4_NEW)]},
}
POST = {  # filled from the local apply on the pristine tree (see REPORT.md)
    "vllm/platforms/interface.py": "cada7cfd9f5ecafebc0368db1b42b8608c806b8ed19eb129ff3686b320eb1e14",
    "vllm/v1/core/kv_cache_utils.py": "eee13367953c7930b7d08a53277a3d7b0e0b2867a57a4b2780a7accb6b1e28e4",
    "vllm/v1/core/single_type_kv_cache_manager.py": "9c9652ee49446d79277e2fea1b1cc664f8fc6aad9904253a577f7a2b5858ce43",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def state(tree):
    out = {}
    for rel, f in FILES.items():
        p = Path(tree) / rel
        h = sha(p.read_bytes())
        t = p.read_text(encoding="utf-8")
        st = "PRISTINE" if h == f["pre"] else "PATCHED" if h == POST.get(rel) else (
            "FOREIGN_MARKED" if MARK in t else "FOREIGN")
        out[rel] = {"sha": h, "state": st}
    return out


def apply(tree):
    res = {}
    for rel, f in FILES.items():
        p = Path(tree) / rel
        t = p.read_text(encoding="utf-8")
        if MARK in t:
            res[rel] = "already patched"
            continue
        for old, new in f["edits"]:
            n = t.count(old)
            if n != 1:
                raise SystemExit(f"{rel}: anchor matched {n} times, refusing")
            t = t.replace(old, new)
        assert "\r" not in t
        p.write_bytes(t.encode("utf-8"))
        py_compile.compile(str(p), doraise=True)
        for pyc in p.parent.glob(f"__pycache__/{p.stem}.*.pyc"):
            pyc.unlink()
        res[rel] = sha(p.read_bytes())
    return res


def revert(tree):
    res = {}
    for rel, f in FILES.items():
        p = Path(tree) / rel
        t = p.read_text(encoding="utf-8")
        if MARK not in t:
            res[rel] = "not patched"
            continue
        for old, new in f["edits"]:
            if t.count(new) != 1:
                raise SystemExit(f"{rel}: patched block not found intact, refusing")
            t = t.replace(new, old)
        assert MARK not in t
        p.write_bytes(t.encode("utf-8"))
        for pyc in p.parent.glob(f"__pycache__/{p.stem}.*.pyc"):
            pyc.unlink()
        res[rel] = sha(p.read_bytes())
    return res


if __name__ == "__main__":
    cmd, tree = sys.argv[1], sys.argv[2]
    if cmd == "apply":
        print(apply(tree))
    elif cmd == "revert":
        print(revert(tree))
    print(state(tree))
