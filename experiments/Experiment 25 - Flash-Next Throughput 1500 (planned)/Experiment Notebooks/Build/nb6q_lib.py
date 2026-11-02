"""E25 NB6Q: the Q-defer patch on NB6 SUBMIT (lead 2026-11-01). NOT a standalone module: the NB6Q notebook execs this text
(sha256 checked, packed with nb4_ppack.py rev 2 in one lzma + base85 literal at the end of the engine cell) into its own
global namespace, after every NB6 definition and before CELL 8 runs. It uses the notebook's os / sys / json / time /
hashlib / traceback / Path / TMP / FAKE / R / save / log, and NB6Q_SRC / NB6Q_PPACK_SHA of the loader lines.

  nb6q_module      nb4_ppack.py rev 2 from NB6Q_SRC (sha256 = NB6Q_PPACK_SHA = the module NB4S fa9f35ca ran), written to TMP
                   and imported by its path; its PATCHES table must equal NB6Q_FILES (pre / post shas of the three files)
  nb6q_apply       once, right after the first successful merlin_install(): snapshot -> apply_ppack_patch (strict pre-sha,
                   post-sha, marker, py_compile with the Merlin venv) -> verify_ppack_patch + post shas again. ANY failure:
                   the three files are restored byte for byte from the snapshot in the REVERSE dependency order
                   (gpu_worker -> model_runner -> pp_utils, stopping at the first failed write, so a partial restore leaves
                   a consistent tree: the patched set is a prefix of pp_utils -> model_runner -> gpu_worker, recorded as
                   NB6Q["consistent"]), the pyc of all three dropped, one loud log line (stdout + stderr), state FAILED_*,
                   no NB4_PPDEFER: the run goes on as NB6. Never raises.
  merlin_install   wraps NB6's: the install, then nb6q_apply (once)
  merlin_fn_server wraps NB6's: NB4_PPDEFER / NB4_PPACK removed from the launch env, NB4_PPDEFER=1 set only when the patch is
                   verified applied (the server's 4 workers inherit it); e21_relaunch (NB6, unchanged) calls merlin_fn_server
  nb6q_readback    nb4_ppack.readback_label over the NB4_PPQ lines of the launch log (streamed, only those lines kept):
                   ENGAGED / ENGAGED_FALLBACK / NOT_ENGAGED / NOT_INTEGRATED, NOT_APPLIED without the patch, READBACK_ERROR;
                   recorded in the boot try (trec["nb6q_readback"][stage]), NB6Q["readbacks"], the report and the log.
                   Record only: it never raises and never stops a run (commit or rerun).
FAKE mode only (the CPU kits' fake server has no Merlin tree): NB6Q_FAKE_TREE names a directory holding the three files;
without it the patch is SKIPPED_FAKE_NO_TREE. A real run never reads NB6Q_FAKE_TREE (FAKE is None unless E11_HARNESS=1).
"""
NB6Q_FILES = {   # nb4_ppack.PATCHES: Merlin ad42e37 pre-sha / patched post-sha per file (asserted by the builder)
    "vllm/v1/worker/gpu/pp_utils.py": {"pre": "6be931deaf8bee6dc9571c3c84d7e92c9f26c618c09dcb58a8b384bff487a662",
        "post": "8b7fa4a6e4598ff4d4a56eb6e595efb1e494276b12b187b099973c262e90c69a"},
    "vllm/v1/worker/gpu/model_runner.py": {"pre": "05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c",
        "post": "2c25fda60dbd5f2506ff989c29ccfe8fc0ff082fbad457fc3ae9c89ed9a185d6"},
    "vllm/v1/worker/gpu_worker.py": {"pre": "a97d08ef042a53fcd139bae041fff407fce75796650d936c200c699f0d63a489",
        "post": "5642b89d918b30e5b70b1067d53776e379613035b7b6e91e93cf0ee4279cfef5"},
}
NB6Q = {"what": "NB6 SUBMIT + the Q-defer patch (nb4_ppack rev 2, NB4_PPDEFER=1)", "ppack_sha256": NB6Q_PPACK_SHA,
        "attempted": False, "applied": False, "state": "NOT_ATTEMPTED", "env": {}, "launch_env": {}, "readbacks": [],
        "last_label": None}
NB6Q_PQ = None


def nb6q_module():
    """The embedded nb4_ppack.py rev 2 (sha256 checked) as a module, loaded from TMP by its path; raises on any mismatch."""
    global NB6Q_PQ
    if NB6Q_PQ is None:
        import importlib.util as _ilu
        src = NB6Q_SRC["nb4_ppack.py"]
        if hashlib.sha256(src.encode("utf-8")).hexdigest() != NB6Q_PPACK_SHA:
            raise RuntimeError("NB6Q_EMBED_SHA_MISMATCH: nb4_ppack.py")
        f = TMP / "nb4_ppack.py"
        f.write_bytes(src.encode("utf-8"))
        if hashlib.sha256(f.read_bytes()).hexdigest() != NB6Q_PPACK_SHA:
            raise RuntimeError("NB6Q_WRITTEN_SHA_MISMATCH: " + str(f))
        spec = _ilu.spec_from_file_location("nb4_ppack", str(f))
        m = _ilu.module_from_spec(spec)
        sys.modules["nb4_ppack"] = m
        spec.loader.exec_module(m)
        if {rel: {"pre": p[0], "post": p[1]} for rel, p in m.PATCHES.items()} != NB6Q_FILES:
            raise RuntimeError("NB6Q_PATCH_TABLE_MISMATCH")
        NB6Q_PQ = m
    return NB6Q_PQ


def nb6q_tree(eng):
    """(tree, python) of the installed Merlin; FAKE (CPU kits only): NB6Q_FAKE_TREE + this python, or (None, None)."""
    if FAKE:
        t = os.environ.get("NB6Q_FAKE_TREE")
        return (Path(t), sys.executable) if t else (None, None)
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


# Dependency chain of the patched files: patched model_runner.py calls pp_handler.flush_deferred_receive() unconditionally
# (the method exists only in patched pp_utils.py); patched gpu_worker.py reaches it through getattr. A tree is CONSISTENT when
# the patched files form a prefix of this chain and the rest are original (with NB4_PPDEFER unset, patched files behave as the
# originals). nb4_ppack applies in this order; the restore runs in the REVERSE order and stops at the first failed write, so
# any partial restore leaves a consistent tree.
NB6Q_CHAIN = ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py")
NB6Q_RESTORE_ORDER = tuple(reversed(NB6Q_CHAIN))


def nb6q_consistency(tree):
    """Per file 'original' / 'patched' / 'other' (sha vs NB6Q_FILES) and consistent = patched set is a prefix of NB6Q_CHAIN."""
    st = {}
    for rel in NB6Q_CHAIN:
        try:
            h = hashlib.sha256((tree / rel).read_bytes()).hexdigest()
        except Exception:
            h = None
        st[rel] = "original" if h == NB6Q_FILES[rel]["pre"] else "patched" if h == NB6Q_FILES[rel]["post"] else "other"
    seq = [st[rel] for rel in NB6Q_CHAIN]
    n = 0
    while n < len(seq) and seq[n] == "patched":
        n += 1
    return {"consistent": "other" not in seq and all(x == "original" for x in seq[n:]), "files": st}


def nb6q_clear_pyc(tree):
    """Drop the __pycache__ entries of the three files (after any restore, even a failed one). Never raises."""
    gone = []
    for rel in NB6Q_CHAIN:
        f = tree / rel
        try:
            for pyc in (f.parent / "__pycache__").glob(f.stem + ".*.pyc"):
                try:
                    pyc.unlink()
                    gone.append(pyc.name)
                except Exception:
                    pass
        except Exception:
            pass
    return gone


def nb6q_restore(tree, snap, order=NB6Q_RESTORE_ORDER):
    """The failure path: each file back to its snapshot bytes (atomic replace, sha-verified) in the reverse dependency order
    (gpu_worker -> model_runner -> pp_utils), stopping at the first failure; then the pyc of all three are dropped and the
    tree's consistency is recorded. Never raises."""
    out, ok = {}, True
    for rel in order:
        b = snap[rel]
        f = tree / rel
        try:
            if not f.is_file() or f.read_bytes() != b:
                tmp = f.with_name(f.name + ".nb6qtmp")
                tmp.write_bytes(b)
                os.replace(tmp, f)
                act = "restored"
            else:
                act = "unchanged"
            good = hashlib.sha256(f.read_bytes()).hexdigest() == hashlib.sha256(b).hexdigest()
            out[rel] = {"action": act, "sha_equal_snapshot": good}
        except Exception as e:
            good = False
            out[rel] = {"action": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        ok = ok and good
        if not good:
            break                                     # never restore a file another still-patched file depends on
    rec = {"ok": ok, "order": list(order), "files": out, "pyc_dropped": nb6q_clear_pyc(tree)}
    try:
        c = nb6q_consistency(tree)
        rec.update(consistent=c["consistent"], disk=c["files"])
    except Exception as e:
        rec.update(consistent=None, consistency_error=repr(e)[:200])
    return rec


def nb6q_apply(eng):
    """Once: apply + verify the Q-defer patch on the installed tree. Never raises (failure -> restore, no patch, no env)."""
    if NB6Q["attempted"]:
        return NB6Q
    NB6Q["attempted"] = True
    t0, snap, tree = time.time(), None, None
    try:
        PQ = nb6q_module()
        tree, py = nb6q_tree(eng)
        if tree is None:
            NB6Q.update(state="SKIPPED_FAKE_NO_TREE", applied=False, env={})
            return NB6Q
        NB6Q["tree"] = str(tree)
        snap = {rel: (tree / rel).read_bytes() for rel in PQ.PATCHES}
        NB6Q["pre"] = {rel: hashlib.sha256(b).hexdigest() for rel, b in snap.items()}
        a = PQ.apply_ppack_patch(tree, py)
        v = PQ.verify_ppack_patch(tree)
        post = {rel: v["files"][rel]["sha"] for rel in PQ.PATCHES}
        NB6Q.update(apply={k: a.get(k) for k in ("applied", "ok", "py_compile")}, verify_state=v["state"], post=post)
        if a.get("ok") is not True or v["state"] != "applied" or any(post[rel] != NB6Q_FILES[rel]["post"] for rel in post):
            raise RuntimeError("NB6Q_POST_SHA_VERIFY_FAILED: " + json.dumps(post)[:600])
        NB6Q.update(applied=True, state="APPLIED", env=PQ.server_env("Q-defer"))
        log("NB6Q Q-defer patch APPLIED and verified (pre / post shas of " + ", ".join(sorted(post)) + "): NB4_PPDEFER=1 on every launch")
    except Exception as e:
        NB6Q.update(applied=False, env={}, error=f"{type(e).__name__}: {e}"[:900], tb=traceback.format_exc()[-1500:])
        if snap is None:
            NB6Q["state"] = "FAILED_NOT_TOUCHED"
        else:
            NB6Q["revert"] = nb6q_restore(tree, snap)
            NB6Q["state"] = "FAILED_REVERTED" if NB6Q["revert"]["ok"] else "FAILED_REVERT_INCOMPLETE"
            NB6Q["consistent"] = NB6Q["revert"].get("consistent")      # recorded; the run continues with no env either way
        msg = ("!!!!!!!!!! NB6Q Q-DEFER PATCH FAILED (" + NB6Q["state"] + "): " + NB6Q["error"][:500] +
               " -> the run continues WITHOUT the patch and WITHOUT NB4_PPDEFER (NB6 behaviour) !!!!!!!!!!")
        try:
            log(msg)
            print(msg, file=sys.stderr, flush=True)
        except Exception:
            pass
    finally:
        NB6Q["sec"] = round(time.time() - t0, 1)
        try:
            R["setup"]["nb6q_ppdefer"] = NB6Q
            if isinstance(globals().get("NB5_REPORT"), dict):
                globals()["NB5_REPORT"]["nb6q_ppdefer"] = NB6Q
            save()
        except Exception:
            pass
    return NB6Q


_nb6q_merlin_install_nb6 = globals().get("_nb6q_merlin_install_nb6") or merlin_install          # NB6's (wrapped once)
_nb6q_merlin_fn_server_nb6 = globals().get("_nb6q_merlin_fn_server_nb6") or merlin_fn_server    # NB6's (wrapped once)


def merlin_install():
    """NB6Q: NB6's merlin_install(), then (once, right after the first successful install) the Q-defer patch."""
    eng = _nb6q_merlin_install_nb6()
    if not NB6Q["attempted"]:
        nb6q_apply(eng)
    return eng


def merlin_fn_server(name, kv, port, kv_bytes):
    """NB6Q: NB6's launch plan; NB4_PPDEFER=1 in the server env only when the patch is verified applied (else removed)."""
    s = _nb6q_merlin_fn_server_nb6(name, kv, port, kv_bytes)
    popped = {k: s.env.pop(k) for k in ("NB4_PPDEFER", "NB4_PPACK") if k in s.env}
    if NB6Q.get("applied") is True:
        s.env.update(NB6Q["env"])
    NB6Q["launch_env"][name] = {"NB4_PPDEFER": s.env.get("NB4_PPDEFER"), "removed_inherited": popped or None}
    return s


def nb6q_readback(trec, s, stage):
    """Record only (never raises): nb4_ppack.readback_label over the NB4_PPQ lines of this launch's log."""
    try:
        if NB6Q.get("applied") is not True:
            rb = {"label": "NOT_APPLIED", "state": NB6Q.get("state")}
        else:
            lines = []
            with open(s.log_path, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    if "NB4_PPQ " in ln:
                        lines.append(ln.rstrip("\n"))
            lab, det = nb6q_module().readback_label("\n".join(lines), "Q-defer")
            rb = {"label": lab, "detail": det, "ppq_lines": len(lines), "env": (getattr(s, "env", None) or {}).get("NB4_PPDEFER")}
    except Exception as e:
        rb = {"label": "READBACK_ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
    rb.update(stage=stage, name=getattr(s, "name", None))
    try:
        trec.setdefault("nb6q_readback", {})[stage] = rb
        NB6Q["readbacks"].append({"name": rb["name"], "stage": stage, "label": rb["label"],
                                  "defer_per_rank": (rb.get("detail") or {}).get("defer_per_rank")})
        NB6Q["last_label"] = rb["label"]
        log("NB6Q read-back", stage, rb["name"], rb["label"],
            json.dumps((rb.get("detail") or {}).get("defer_per_rank") or rb.get("error") or rb.get("state"), default=str)[:600])
        save()
    except Exception:
        pass
    return rb
