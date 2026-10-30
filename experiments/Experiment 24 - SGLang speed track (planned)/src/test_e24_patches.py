"""CPU tests of e24_sglang_patches.py against temp COPIES of the SGLang 0.5.21 package (rev 2, 2026-10-30).

Source (never imported or executed): env E24_SGLANG_SRC (a PRISTINE `sglang` package dir, e.g. an unpacked 0.5.21
wheel), else the scratch copy of the wheel (`downloads/sglang_source_0521/src/sglang`). Tests skip when neither exists.
Also runs the backports agent's 28 CPU semantic checks (ported below from scripts_sglang_backports/test_ports_cpu.py:
the REAL 0.5.21 statements cut with ast, pristine vs THIS patcher's patched copy) and, when present, the original script.
Run: python3.12 -m unittest -v test_e24_patches.py   (stdlib only)
"""
import ast
import dataclasses
import enum
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCR = Path("/private/tmp/claude-501/-Users-md-hamidhosen-Documents-Kaggle-ARC-AGI-2/e21dd5ac-4964-44ae-8fe0-b7d1156db571/"
           "scratchpad/e24")
SRC = Path(os.environ.get("E24_SGLANG_SRC") or SCR / "downloads/sglang_source_0521/src/sglang")
PORTS_SCRIPT = Path(os.environ.get("E24_PORTS_CPU_TEST") or SCR / "scripts_sglang_backports/test_ports_cpu.py")
HAVE_SRC = (SRC / "srt/managers/schedule_batch.py").is_file()

_spec = importlib.util.spec_from_file_location("e24_sglang_patches", HERE / "e24_sglang_patches.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

# read-only files the semantic checks cut from (besides the patched ones)
EXTRA = ("srt/mem_cache/base_prefix_cache.py", "srt/mem_cache/unified_cache/unified_tree_core.py",
         "srt/mem_cache/unified_radix_cache.py")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rbytes(p):
    with open(p, "rb") as f:
        return f.read()


def rtext(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def wtext(p, s, mode="w"):
    with open(p, mode, encoding="utf-8") as f:
        f.write(s)


def wbytes(p, b):
    with open(p, "wb") as f:
        f.write(b)


def tree_shas(root):
    out = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, root)] = sha(f.read())
    return out


@unittest.skipUnless(HAVE_SRC, "no SGLang 0.5.21 source (set E24_SGLANG_SRC)")
class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e24p_")
        cls.base = os.path.join(cls.tmp, "base", "sglang")
        for rel in list(P.REFERENCE_FILE_SHA256) + ["_version.py", "__init__.py"] + list(EXTRA):
            dst = os.path.join(cls.base, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(SRC / rel, dst)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fresh(self):
        d = tempfile.mkdtemp(dir=self.tmp)
        site = os.path.join(d, "sglang")
        shutil.copytree(self.base, site)
        return site

    def file_shas(self, site):
        return {rel: sha(rbytes(os.path.join(site, rel))) for rel in P.REFERENCE_FILE_SHA256}

    def backups(self, site):
        return sorted(str(p) for p in Path(site).rglob("*" + P.BACKUP_SUFFIX))

    def assert_pristine(self, site):
        self.assertEqual(self.file_shas(site), P.REFERENCE_FILE_SHA256)
        self.assertEqual(self.backups(site), [])
        self.assertEqual(sorted(str(p) for p in Path(site).rglob("*" + P.TMP_SUFFIX)), [])
        c = P.check(site)
        self.assertTrue(c["ok"], c)
        self.assertTrue(P.state_ok(c, ()))

    def assert_compiles(self, site):
        for rel in P.REFERENCE_FILE_SHA256:
            p = os.path.join(site, rel)
            compile(rtext(p), p, "exec", dont_inherit=True)

    # ----------------------------------------------------------------------------------------------- API shape
    def test_api_constants(self):
        self.assertEqual(P.DEFAULT_NAMES, ("44401", "41872"))
        self.assertEqual(P.E24_NAMES, ("43567", "43559", "43029", "41736", "41737"))
        self.assertEqual(set(P.NAMES), set(P.DEFAULT_NAMES) | set(P.E24_NAMES))
        self.assertEqual(len(P.NAMES), 7)
        self.assertEqual(P.REQUIRES, {"43559": ("43567",)})
        for n in P.NAMES:
            for h in P.PATCHES[n]:
                self.assertEqual(sha(h["old"].encode()), h["old_sha256"], n)
                self.assertNotEqual(h["old"], h["new"])
        # 43559 is the 0.5.21 port, never the upstream req.lock text
        alloc = P._43559_ALLOC_NEW
        self.assertNotIn("req.lock ", alloc)
        self.assertNotIn("lock = req.lock", alloc)
        self.assertIn("req.last_node is not None", alloc)
        self.assertIn("not req.swa_prefix_lock_released", alloc)
        # 43567 keeps a single definition (E14's): two hunks in schedule_batch.py
        self.assertEqual([h["file"] for h in P.PATCHES["43567"]], ["srt/managers/schedule_batch.py"] * 2)

    def test_provenance_diffs_present(self):
        for n in P.E24_NAMES:
            d = HERE / "patches" / ("pr%s_0521.diff" % n)
            self.assertTrue(d.is_file(), d)
            txt = d.read_text(encoding="utf-8")
            files = {h["file"] for h in P.PATCHES[n]}
            for f in files:
                self.assertIn("+++ b/" + f, txt)

    # ----------------------------------------------------------------------------------------------- check
    def test_check_pristine(self):
        site = self.fresh()
        c = P.check(site)
        self.assertEqual(c["version"], "0.5.21")
        self.assertEqual(c["names"], {n: "not_applied" for n in P.NAMES})
        self.assertEqual(c["files"], P.REFERENCE_FILE_SHA256)
        self.assertTrue(c["files_verified"] and c["ok"])
        self.assertTrue(P.state_ok(c, ()))
        self.assertFalse(P.state_ok(c, ("43567",)))
        self.assertFalse(P.state_ok({"names": {}, "ok": True, "files_verified": True}, ()))   # fail-closed
        sub = P.check(site, "41736,41737")
        self.assertEqual(sorted(sub["names"]), ["41736", "41737"])
        self.assertFalse(P.state_ok(sub, ()))            # not all 7 checked

    # ----------------------------------------------------------------------------------------------- apply / revert
    def test_apply_each_name_alone_and_revert(self):
        for n in P.NAMES:
            if n in P.REQUIRES:
                continue
            with self.subTest(name=n):
                site = self.fresh()
                r = P.apply(site, [n])
                self.assertTrue(r["ok"], r)
                self.assertEqual(r["results"][n]["status"], "applied")
                c = r["check"]
                for m in P.NAMES:
                    self.assertEqual(c["names"][m], "applied" if m == n else "not_applied", (n, m))
                self.assertTrue(c["files_verified"])
                self.assertTrue(P.state_ok(c, (n,)))
                self.assertFalse(P.state_ok(c, ()))
                self.assert_compiles(site)
                if n in P.E24_NAMES:                     # E24 files are disjoint: patched sha = the reference
                    for rel in P._FILES_OF[n]:
                        self.assertEqual(c["files"][rel], P.REFERENCE_PATCHED_SHA256[rel], rel)
                rv = P.revert(site, [n])
                self.assertTrue(rv["ok"], rv)
                self.assert_pristine(site)

    def test_43559_alone_refused_nothing_written(self):
        site = self.fresh()
        r = P.apply(site, ["43559"])
        self.assertFalse(r["ok"])
        self.assertIn("43559 requires 43567", r["error"])
        self.assertEqual(r["results"], {})
        self.assert_pristine(site)
        self.assertFalse(P.state_ok(P.check(site), ("43559",)))   # an expected set with 43559 alone is invalid

    def test_43567_43559_unit(self):
        site = self.fresh()
        r = P.apply(site, ["43559", "43567"])                     # any order in, dependency order out
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["order"], ["43567", "43559"])
        self.assertTrue(P.state_ok(r["check"], ("43567", "43559")))
        self.assert_compiles(site)
        rv = P.revert(site, ["43567"])                            # would leave 43559 alone
        self.assertFalse(rv["ok"])
        self.assertTrue(any("43559" in e for e in rv["errors"]), rv)
        self.assertTrue(P.state_ok(P.check(site), ("43567", "43559")))
        rv = P.revert(site, ["43559"])
        self.assertTrue(rv["ok"], rv)
        self.assertTrue(P.state_ok(rv["check"], ("43567",)))
        rv = P.revert(site, ["43567"])
        self.assertTrue(rv["ok"], rv)
        self.assert_pristine(site)
        # 43559 on top of an already applied 43567 is allowed
        self.assertTrue(P.apply(site, ["43567"])["ok"])
        r = P.apply(site, ["43559"])
        self.assertTrue(r["ok"], r)
        self.assertTrue(P.state_ok(r["check"], ("43567", "43559")))
        self.assertTrue(P.revert(site)["ok"])
        self.assert_pristine(site)

    def test_valid_combos(self):
        combos = [("43567", "43559"), ("41737", "41736"), ("41737",), ("43029",), ("44401", "41872"),
                  ("44401", "41872", "43567", "43559"), P.E24_NAMES, P.NAMES]
        for combo in combos:
            with self.subTest(combo=combo):
                site = self.fresh()
                r = P.apply(site, combo)
                self.assertTrue(r["ok"], r)
                self.assertTrue(P.state_ok(r["check"], combo), r["check"]["names"])
                self.assert_compiles(site)
                rv = P.revert(site)
                self.assertTrue(rv["ok"], rv)
                self.assert_pristine(site)

    def test_all_seven_together(self):
        site = self.fresh()
        r = P.apply(site, P.NAMES)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["order"], list(P.NAMES))
        self.assertEqual(self.file_shas(site), P.REFERENCE_PATCHED_SHA256)
        self.assertTrue(all(v == "applied" for v in r["check"]["names"].values()))
        self.assertTrue(P.state_ok(r["check"], P.NAMES))
        self.assertEqual(len(self.backups(site)), 7)
        self.assert_compiles(site)
        rv = P.revert(site)
        self.assertTrue(rv["ok"], rv)
        self.assertEqual(rv["order"], list(reversed(P.NAMES)))
        self.assert_pristine(site)

    def test_idempotence(self):
        site = self.fresh()
        self.assertTrue(P.apply(site, P.E24_NAMES)["ok"])
        before = self.file_shas(site)
        r = P.apply(site, P.E24_NAMES)
        self.assertTrue(r["ok"], r)
        self.assertTrue(all(x["status"] == "already applied" for x in r["results"].values()))
        self.assertEqual(self.file_shas(site), before)
        self.assertTrue(P.revert(site)["ok"])
        r = P.revert(site)
        self.assertTrue(r["ok"], r)
        self.assertTrue(all(x["status"] == "not applied" for x in r["results"].values()))
        self.assert_pristine(site)

    def test_conflict_in_hunk_region(self):
        site = self.fresh()
        p = os.path.join(site, "srt/managers/schedule_batch.py")
        txt = rtext(p)
        mutated = txt.replace("                    has_pending = True\n            # On failure",
                              "                    has_pending = 1\n            # On failure", 1)
        self.assertNotEqual(mutated, txt)
        wtext(p, mutated)
        c = P.check(site)
        self.assertEqual(c["names"]["43567"], "conflict")
        self.assertFalse(c["ok"])
        self.assertFalse(P.state_ok(c, ()))
        r = P.apply(site, ["43567"])
        self.assertFalse(r["ok"])
        self.assertEqual(rtext(p), mutated)
        self.assertEqual(self.backups(site), [])
        r = P.apply(site, ["43567"], strict=False)                  # E14 semantics: the hunk guard alone refuses too
        self.assertFalse(r["ok"])
        self.assertEqual(r["results"]["43567"]["status"], "refused: source mismatch")
        self.assertEqual(rtext(p), mutated)

    def test_all_or_nothing_across_names(self):
        site = self.fresh()
        p = os.path.join(site, "srt/models/dflash.py")
        txt = rtext(p)
        wtext(p, txt.replace("shifted * (position >= tap)", "shifted*(position >= tap)", 1))
        # strict: refused up front (the 41736 file does not verify), nothing written
        r = P.apply(site, ["43567", "41736"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["results"], {})
        # non-strict: 43567 is applied first, the 41736 conflict then rolls it back
        r = P.apply(site, ["43567", "41736"], strict=False)
        self.assertFalse(r["ok"])
        self.assertEqual(r["results"]["43567"]["status"], "applied")
        self.assertEqual(r["rolled_back"], ["srt/managers/schedule_batch.py"])
        self.assertEqual(self.file_shas(site)["srt/managers/schedule_batch.py"],
                         P.REFERENCE_FILE_SHA256["srt/managers/schedule_batch.py"])
        self.assertEqual(self.backups(site), [])

    def test_mutation_outside_hunks_is_a_provenance_failure(self):
        site = self.fresh()
        p = os.path.join(site, "srt/models/dflash.py")
        with open(p, "a", encoding="utf-8") as f:
            f.write("# local edit\n")
        c = P.check(site)
        self.assertEqual(c["names"]["41736"], "not_applied")      # the hunk itself is intact
        self.assertFalse(c["files_verified"])
        self.assertFalse(c["ok"])
        self.assertFalse(P.state_ok(c, ()))
        self.assertFalse(P.state_ok(c, (), strict=False))          # chk["ok"] is False too
        r = P.apply(site, ["41736"])                               # strict: refused, nothing written
        self.assertFalse(r["ok"])
        self.assertIn("not the pristine", r["error"])
        self.assertTrue(rtext(p).endswith("# local edit\n"))
        self.assertEqual(P.check(site)["names"]["41736"], "not_applied")
        self.assertEqual(self.backups(site), [])

    def test_unknown_name(self):
        site = self.fresh()
        self.assertFalse(P.apply(site, ["99999"])["ok"])
        self.assertEqual(P.check(site, ["99999"])["names"]["99999"], "unknown")
        self.assertFalse(P.revert(site, ["99999"])["ok"])
        self.assert_pristine(site)

    def test_pyc_dropped(self):
        site = self.fresh()
        cache = os.path.join(site, "srt/managers/__pycache__")
        os.makedirs(cache, exist_ok=True)
        pyc = os.path.join(cache, "schedule_batch.cpython-312.pyc")
        other = os.path.join(cache, "scheduler.cpython-312.pyc")
        for f in (pyc, other):
            wbytes(f, b"stale")
        r = P.apply(site, ["43567"])
        self.assertTrue(r["ok"])
        self.assertFalse(os.path.exists(pyc))
        self.assertTrue(os.path.exists(other))
        self.assertIn("schedule_batch.cpython-312.pyc", r["results"]["43567"]["files"]["srt/managers/schedule_batch.py"]
                      ["pyc_removed"])
        wbytes(pyc, b"stale")
        self.assertTrue(P.revert(site)["ok"])
        self.assertFalse(os.path.exists(pyc))

    def test_restore_backups_recovery(self):
        site = self.fresh()
        self.assertTrue(P.apply(site, ["43029", "41736"])["ok"])
        # damage a patched file so the inverse hunk no longer matches; E14-style restore still recovers it
        p = os.path.join(site, "srt/models/dflash.py")
        wtext(p, "# damage\n", "a")
        rv = P.revert(site, ["41736"])
        self.assertFalse(rv["ok"])
        rb = P.restore_backups(site)
        self.assertTrue(rb["ok"], rb)
        self.assert_pristine(site)

    def test_cli(self):
        site = self.fresh()
        py = sys.executable
        mod = str(HERE / "e24_sglang_patches.py")
        cp = subprocess.run([py, "-I", mod, "check", "--site", site], capture_output=True, text=True, timeout=120)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(json.loads(cp.stdout)["names"], {n: "not_applied" for n in P.NAMES})
        cp = subprocess.run([py, "-I", mod, "apply", "--site", site, "--names", "43559"], capture_output=True, text=True,
                            timeout=120)
        self.assertEqual(cp.returncode, 1)
        cp = subprocess.run([py, "-I", mod, "apply", "--site", site, "--names", "43567,43559"], capture_output=True,
                            text=True, timeout=120)
        self.assertEqual(cp.returncode, 0, cp.stdout[-2000:])
        cp = subprocess.run([py, "-I", mod, "check", site, "43567", "43559"], capture_output=True, text=True,
                            timeout=120)                                        # E14 positional form
        self.assertEqual(json.loads(cp.stdout)["names"], {"43567": "applied", "43559": "applied"})
        cp = subprocess.run([py, "-I", mod, "revert", "--site", site], capture_output=True, text=True, timeout=120)
        self.assertEqual(cp.returncode, 0, cp.stdout[-2000:])
        self.assert_pristine(site)

    def test_site_dir_find_spec_only(self):
        venv = os.path.join(self.tmp, "venv")
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True, timeout=300)
        vpy = os.path.join(venv, "bin", "python")
        with self.assertRaises(RuntimeError):
            P.site_dir(vpy)
        out = subprocess.run([vpy, "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True, check=True).stdout.strip()
        pkg = os.path.join(out, "sglang")
        shutil.copytree(self.base, pkg)
        marker = os.path.join(self.tmp, "imported.marker")
        with open(os.path.join(pkg, "__init__.py"), "w") as f:
            f.write("open(%r, 'w').write('x')\nraise RuntimeError('sglang must not be imported')\n" % marker)
        got = P.site_dir(vpy)
        self.assertEqual(got, os.path.realpath(pkg))
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(P.check(got)["names"], {n: "not_applied" for n in P.NAMES})

    @unittest.skipUnless(os.environ.get("E24_TEST_FULL_COPY", "1") == "1", "E24_TEST_FULL_COPY=0")
    def test_full_package_copy_all_seven_then_revert_byte_identical(self):
        site = os.path.join(self.tmp, "full", "sglang")
        shutil.copytree(SRC, site, ignore=shutil.ignore_patterns("__pycache__"))
        before = tree_shas(site)
        r = P.apply(site, P.NAMES)
        self.assertTrue(r["ok"], r)
        mid = tree_shas(site)
        changed = {k for k in before if mid.get(k) != before[k]}
        self.assertEqual(changed, set(P.REFERENCE_FILE_SHA256))
        added = set(mid) - set(before)
        self.assertEqual(added, {rel + P.BACKUP_SUFFIX for rel in P.REFERENCE_FILE_SHA256})
        self.assertTrue(P.revert(site)["ok"])
        self.assertEqual(tree_shas(site), before)

    # ----------------------------------------------------------------------------------------------- semantic ports
    def test_28_semantic_checks_on_this_patcher_output(self):
        pristine = self.fresh()
        patched = self.fresh()
        r = P.apply(patched, P.NAMES)
        self.assertTrue(r["ok"], r)
        res = port_checks(pristine, patched)
        failed = [(n, d) for n, ok, d in res if not ok]
        self.assertEqual(len(res), 28, [n for n, _, _ in res])
        self.assertEqual(failed, [])

    @unittest.skipUnless(PORTS_SCRIPT.is_file(), "original test_ports_cpu.py not present")
    def test_original_ports_script_on_this_patcher_output(self):
        backports_pristine = SCR / "downloads/sglang_backports/sglang0521/unpacked/sglang"
        pristine = str(backports_pristine) if backports_pristine.is_dir() else self.fresh()
        patched = self.fresh()
        self.assertTrue(P.apply(patched, P.NAMES)["ok"])
        cp = subprocess.run([sys.executable, "-I", str(PORTS_SCRIPT), pristine, patched], capture_output=True,
                            text=True, timeout=300)
        self.assertEqual(cp.returncode, 0, cp.stdout[-3000:])
        self.assertIn("TOTAL 28 checks, 0 failed", cp.stdout)


UNIT = HERE / "e24_unit_41736_41737.py"


@unittest.skipUnless(HAVE_SRC, "no SGLang 0.5.21 source (set E24_SGLANG_SRC)")
class UnitScriptTests(unittest.TestCase):
    """e24_unit_41736_41737.py on this Mac: compile, prepare (stdlib), a torch-less run (ran=False, exit 0), compare."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e24u_")
        cls.site = os.path.join(cls.tmp, "venv_site", "sglang")
        for rel in list(P.REFERENCE_FILE_SHA256) + ["_version.py", "__init__.py"]:
            dst = os.path.join(cls.site, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(SRC / rel, dst)
        cls.home = os.path.join(cls.tmp, "home")
        os.makedirs(cls.home)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, *args, expect=0):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": self.home, "PYTHONDONTWRITEBYTECODE": "1"}
        cp = subprocess.run([sys.executable, "-I", "-B", str(UNIT)] + list(args), capture_output=True, text=True, env=env,
                            timeout=300)
        self.assertEqual(cp.returncode, expect, cp.stdout[-2000:] + cp.stderr[-2000:])
        return cp

    def test_compiles(self):
        compile(rtext(UNIT), str(UNIT), "exec", dont_inherit=True)

    def test_prepare_never_touches_the_site_even_when_patched(self):
        before = self.file_shas_site()
        # a patched "venv": prepare must still produce a pristine overlay_none (revert on the COPY only)
        self.assertTrue(P.apply(self.site, ["41737", "43029"])["ok"])
        patched = self.file_shas_site()
        root = os.path.join(self.tmp, "e24")
        out = os.path.join(self.tmp, "prep.json")
        self.cli("prepare", "--site", self.site, "--root", root, "--patcher", str(HERE / "e24_sglang_patches.py"),
                 "--out", out)
        with open(out) as f:
            prep = json.load(f)
        self.assertTrue(prep["ok"], prep)
        self.assertEqual(self.file_shas_site(), patched)                     # the site is only read
        self.assertEqual(sorted(prep["overlays"]), ["41736", "41737", "none"])
        for st, o in prep["overlays"].items():
            c = P.check(o["package"])
            self.assertTrue(P.state_ok(c, () if st == "none" else (st,)), (st, c["names"]))
            self.assertEqual(list(Path(o["package"]).rglob("*" + P.BACKUP_SUFFIX)) if st == "none" else [], [])
        self.assertTrue(P.revert(self.site)["ok"])
        self.assertEqual(self.file_shas_site(), before)
        # a torch-less run: ran=False, a clean exit, the patch state and the expectation recorded, own cache dirs
        for st, test, expect in (("none", "41736", "chain"), ("41736", "41736", "clean"), ("41737", "41737", "clean"),
                                 ("none", "41737", "chain")):
            rj = os.path.join(self.tmp, "run_%s_%s.json" % (test, st))
            cp = self.cli("--test", test, "--overlay", prep["overlays"][st]["dir"], "--out", rj)
            self.assertIn("E24_UNIT run", cp.stdout)
            with open(rj) as f:
                r = json.load(f)
            self.assertFalse(r["ran"])
            self.assertTrue(r["reason"].startswith("torch import failed"), r["reason"])
            self.assertEqual(r["patch_state"], "not_applied" if st == "none" else "applied")
            self.assertEqual(r["expect"], expect)
            self.assertIsNone(r["chain"])
            self.assertIsNone(r["clean"])
            self.assertTrue(r["isolated"])
            for v in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "SGLANG_CACHE_DIR", "CUDA_CACHE_PATH"):
                self.assertIn(os.sep + "unit_cache" + os.sep, r["env"][v])
        self.assertFalse(os.path.exists(os.path.join(self.home, ".cache")))
        # compare of two not-run results
        c = self.cli("compare", "--unpatched", os.path.join(self.tmp, "run_41736_none.json"), "--patched",
                     os.path.join(self.tmp, "run_41736_41736.json"), "--out", os.path.join(self.tmp, "cmp.json"))
        with open(os.path.join(self.tmp, "cmp.json")) as f:
            self.assertEqual(json.load(f)["status"], "INCONCLUSIVE_NOT_RUN")

    def file_shas_site(self):
        return {rel: sha(rbytes(os.path.join(self.site, rel))) for rel in P.REFERENCE_FILE_SHA256}

    def test_orchestrate_single_call_builder_contract(self):
        """The notebook's single call per test (no --overlay): prepare none + <test>, run both as -I children,
        compare; torch-less here -> ran False, INCONCLUSIVE_NOT_RUN, a clean exit, small details."""
        root = os.path.join(self.tmp, "orch")
        out = os.path.join(self.tmp, "orch_41736.json")
        before = self.file_shas_site()
        cp = self.cli("--test", "41736", "--out", out, "--root", root, "--site", self.site, "--timeout", "60")
        self.assertIn("E24_UNIT run", cp.stdout)
        with open(out) as f:
            r = json.load(f)
        self.assertEqual(r["mode"], "orchestrated")
        self.assertFalse(r["ran"])
        self.assertEqual(r["status"], "INCONCLUSIVE_NOT_RUN")
        self.assertIn("torch import failed", r["reason"])
        self.assertIsNone(r["chain"])
        self.assertIsNone(r["clean"])
        self.assertTrue(r["details"]["prepare"]["ok"])
        self.assertEqual(sorted(r["details"]["prepare"]["overlays"]), ["41736", "none"])
        self.assertEqual(r["details"]["unpatched"]["patch_state"], "not_applied")
        self.assertEqual(r["details"]["patched"]["patch_state"], "applied")
        self.assertTrue(r["details"]["unpatched"]["isolated"])
        self.assertLess(len(json.dumps(r["details"])), 20000)            # the notebook keeps details[:20000]
        for st in ("none", "41736"):
            self.assertTrue(os.path.isfile(os.path.join(root, "unit_41736_%s.json" % st)))
            self.assertTrue(os.path.isdir(os.path.join(root, "overlay_" + st, "sglang")))
        self.assertEqual(self.file_shas_site(), before)
        # no sglang in this python and no --site: a recorded reason, still exit 0
        out2 = os.path.join(self.tmp, "orch_nosite.json")
        self.cli("--test", "41737", "--out", out2, "--root", os.path.join(self.tmp, "orch2"))
        with open(out2) as f:
            r2 = json.load(f)
        self.assertFalse(r2["ran"])
        self.assertIn("sglang not found", r2["reason"])

    def test_41736_ast_fallback_source_compiles(self):
        spec = importlib.util.spec_from_file_location("e24_unit_t2", UNIT)
        U = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(U)
        site = os.path.join(self.tmp, "fb", "sglang")
        shutil.copytree(self.site, site)
        rel = "srt/models/dflash.py"
        src0 = U._conv_source(os.path.join(site, rel))
        self.assertTrue(P.apply(site, ["41736"])["ok"])
        src1 = U._conv_source(os.path.join(site, rel))
        for src in (src0, src1):
            compile(src, "fallback", "exec")
            self.assertTrue(src.startswith("@torch.compile(dynamic=True)\ndef _grouped_conv("))
            self.assertIn("class DFlashGroupedConv(nn.Module):", src)
        self.assertIn("* shifted * (position >= tap)", src0)
        self.assertIn("torch.where((position >= tap)", src1)

    def test_torch_without_cuda_is_not_run(self):
        ov = os.path.join(self.tmp, "fake_torch_overlay")
        shutil.copytree(self.site, os.path.join(ov, "sglang"))
        os.makedirs(os.path.join(ov, "torch"))
        wtext(os.path.join(ov, "torch", "__init__.py"),
              "import types\n__version__ = '0.0-fake'\nversion = types.SimpleNamespace(cuda=None)\n"
              "cuda = types.SimpleNamespace(is_available=lambda: False)\n")
        out = os.path.join(self.tmp, "nocuda.json")
        self.cli("--test", "41736", "--overlay", ov, "--out", out)
        with open(out) as f:
            r = json.load(f)
        self.assertFalse(r["ran"])
        self.assertEqual(r["reason"], "CUDA not available")
        self.assertEqual(r["versions"]["torch"], "0.0-fake")
        self.assertEqual((r["patch_state"], r["expect"]), ("not_applied", "chain"))

    def test_usage_errors_exit_2(self):
        self.cli("--test", "99999", "--overlay", self.tmp, "--out", os.path.join(self.tmp, "x.json"), expect=2)
        self.cli("--overlay", self.tmp, "--out", os.path.join(self.tmp, "x.json"), expect=2)

    def test_missing_overlay_is_not_run(self):
        out = os.path.join(self.tmp, "missing.json")
        self.cli("--test", "41737", "--overlay", os.path.join(self.tmp, "nope"), "--out", out)
        with open(out) as f:
            r = json.load(f)
        self.assertFalse(r["ran"])
        self.assertIn("no sglang package", r["reason"])

    def test_compare_statuses_and_max_abs_delta(self):
        spec = importlib.util.spec_from_file_location("e24_unit_t", UNIT)
        U = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(U)
        import array as _array
        import base64 as _b64

        def pack(vals):
            return {"shape": [len(vals)], "dtype": "float32",
                    "b64": _b64.b64encode(_array.array("f", vals).tobytes()).decode()}
        mod, rel = U.PATCHED_FILE["41736"]
        imp_u = {mod: {"sha256": P.REFERENCE_FILE_SHA256[rel]}}
        imp_p = {mod: {"sha256": P.REFERENCE_PATCHED_SHA256[rel]}}
        u = {"test": "41736", "ran": True, "patch_state": "not_applied", "chain": True, "imported": imp_u,
             "finite": {"conv_side0": pack([1.0, 2.0, -3.0]), "tok": {"dtype": "int64", "values": [1, 2]}}}
        p = {"test": "41736", "ran": True, "patch_state": "applied", "clean": True, "imported": imp_p,
             "finite": {"conv_side0": pack([1.0, 2.5, -3.0]), "tok": {"dtype": "int64", "values": [1, 3]}}}
        c = U.compare(u, p)
        self.assertEqual(c["status"], "PASS")
        self.assertEqual(c["max_abs_delta_finite"], {"conv_side0": 0.5, "tok": 1.0})
        self.assertEqual(c["max_abs_delta_finite_all"], 1.0)
        self.assertEqual(U.compare(dict(u, chain=False), p)["status"], "INCONCLUSIVE_NOT_EXERCISED")
        self.assertEqual(U.compare(u, dict(p, clean=False))["status"], "FAIL_PATCHED_NOT_CLEAN")
        self.assertEqual(U.compare(u, dict(p, patch_state="not_applied"))["status"], "INVALID_OVERLAY_STATE")
        self.assertEqual(U.compare(u, dict(p, imported=imp_u))["status"], "INVALID_OVERLAY_STATE")   # same file
        self.assertEqual(U.compare(u, dict(p, ran=False))["status"], "INCONCLUSIVE_NOT_RUN")
        self.assertEqual(U.compare(u, dict(p, test="41737"))["status"], "INVALID_TEST_MISMATCH")
        fb = {"path": "ast_fallback"}
        self.assertEqual(U.compare(dict(u, observed=fb), p)["status"], "PASS_AST_FALLBACK")
        self.assertEqual(U.compare(u, dict(p, observed=fb))["paths"], ["ast_fallback", "module"])
        self.assertEqual(U.compare(u, dict(p, observed={"path": "module"}))["status"], "PASS")
        nan = float("nan")
        self.assertEqual(U._max_abs_delta([nan, 1.0], [nan, 1.0]), 0.0)
        self.assertEqual(U._max_abs_delta([nan], [0.0]), float("inf"))


# ======================================================================================================================
# The 28 semantic checks of scripts_sglang_backports/test_ports_cpu.py (2026-10-30), ported into a function: the REAL
# 0.5.21 statements cut with ast (Note-14 method) on the E17 4-slot / 1-request peak. Logic unchanged.
def port_checks(PRISTINE, PATCHED):
    RESULTS = []

    def check(name, cond, detail=""):
        RESULTS.append((name, bool(cond), detail))

    def src(pkg, rel):
        with open(os.path.join(pkg, rel), encoding="utf-8") as f:
            return f.read()

    def cut(pkg, rel, cls, fn):
        text = src(pkg, rel)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == fn:
                        return textwrap.dedent(ast.get_source_segment(text, item))
        raise KeyError((rel, cls, fn))

    def cut_class(pkg, rel, cls):
        text = src(pkg, rel)
        for node in ast.parse(text).body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                return ast.get_source_segment(text, node)
        raise KeyError((rel, cls))

    def define(source, ns, name):
        exec(compile("from __future__ import annotations\n" + source, "<cut:%s>" % name, "exec"), ns)
        return ns[name]

    class ComponentType(enum.IntEnum):
        FULL = 0
        SWA = 1
        MAMBA = 2

    BPC = "srt/mem_cache/base_prefix_cache.py"
    MAMBA = "srt/mem_cache/unified_cache/components/mamba.py"
    CORE = "srt/mem_cache/unified_cache/unified_tree_core.py"
    URC = "srt/mem_cache/unified_radix_cache.py"
    import typing
    NS = {"dataclasses": dataclasses, "Optional": typing.Optional, "ComponentType": ComponentType,
          "replace": dataclasses.replace, "Sequence": typing.Sequence}
    for c in ("IncLockRefResult", "DecLockRefParams", "DecLockRefResult", "EvictParams"):
        define("@dataclasses.dataclass\n" + cut_class(PRISTINE, BPC, c), NS, c)

    class Allocator:
        def __init__(self, size):
            self.free_ids = list(range(size))

        def alloc(self, n):
            if len(self.free_ids) < n:
                return None
            return [self.free_ids.pop() for _ in range(n)]

        def free(self, v):
            self.free_ids.extend(v)

        def available_size(self):
            return len(self.free_ids)

    class Recording:
        component_type = ComponentType.FULL

        def __init__(self):
            self.released, self.acquired = [], 0

        def eviction_priority(self, is_leaf):
            return 2

        def acquire_component_lock(self, node, result, lock_host=False):
            self.acquired += 1
            return result

        def release_component_lock(self, node, params, lock_host=False):
            self.released.append(params)

    def node(nid, slot):
        cd = {ComponentType.FULL: SimpleNamespace(value=[0], lock_ref=0, host_value=None, host_lock_ref=0),
              ComponentType.MAMBA: SimpleNamespace(value=slot, lock_ref=0, host_value=None, host_lock_ref=0)}
        return SimpleNamespace(id=nid, component_data=cd)

    def build(alloc_fn_src, lazy):
        ns = dict(NS)
        ns["get_exec"] = lambda: SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=lazy))
        root = node(0, None)
        alloc = Allocator(4)
        p_slot = alloc.alloc(1)
        Pn = node(1, p_slot)
        alloc.alloc(3)  # live, keep, pending
        arena = {0: root, 1: Pn}
        full = Recording()
        core = SimpleNamespace(root_node=root, component_evictable_size_={ComponentType.MAMBA: 0},
                               component_protected_size_={ComponentType.MAMBA: 0}, host_lru_lists={})
        core._update_evictable_leaf_sets = lambda n: None
        core.node_by_id = lambda i: arena[i]
        core._assert_receipt_anchor = staticmethod(define(cut(PRISTINE, CORE, "UnifiedTreeCore", "_assert_receipt_anchor")
                                                          .replace("@staticmethod\n", ""), ns,
                                                          "_assert_receipt_anchor")).__func__
        core._release_components = types.MethodType(define(cut(PRISTINE, CORE, "UnifiedTreeCore", "_release_components"),
                                                           ns, "_release_components"), core)
        core.dec_lock_ref = types.MethodType(define(cut(PRISTINE, CORE, "UnifiedTreeCore", "dec_lock_ref"), ns,
                                                    "dec_lock_ref"), core)
        core.inc_lock_ref = types.MethodType(define(cut(PRISTINE, CORE, "UnifiedTreeCore", "inc_lock_ref"), ns,
                                                    "inc_lock_ref"), core)
        cache = SimpleNamespace(disable=False, tree_components=(ComponentType.FULL, ComponentType.MAMBA), tree_core=core,
                                req_to_token_pool=SimpleNamespace(mamba_allocator=alloc), enable_mamba_extra_buffer=True,
                                session=SimpleNamespace(try_dec_lock_ref=lambda n, p=None: None,
                                                        try_inc_lock_ref=lambda n: None))
        cache.evicted = []

        def evict_for_alloc(params):
            need = params.mamba_num
            for nid, nd in arena.items():
                cd = nd.component_data[ComponentType.MAMBA]
                if need > 0 and nid != 0 and cd.lock_ref == 0 and cd.value is not None:
                    alloc.free(cd.value)
                    cd.value = None
                    cache.evicted.append(nid)
                    need -= 1
        cache.evict_for_alloc = evict_for_alloc
        cache.dec_lock_ref = types.MethodType(define(cut(PRISTINE, URC, "UnifiedRadixCache", "dec_lock_ref"), ns,
                                                     "dec_lock_ref"), cache)
        cache.inc_lock_ref = types.MethodType(define(cut(PRISTINE, URC, "UnifiedRadixCache", "inc_lock_ref"), ns,
                                                     "inc_lock_ref"), cache)
        cache._dec_req_lock = types.MethodType(define(cut(PRISTINE, URC, "UnifiedRadixCache", "_dec_req_lock"), ns,
                                                      "_dec_req_lock"), cache)
        comp = SimpleNamespace(cache=cache, tree_core=core, component_type=ComponentType.MAMBA)
        comp.acquire_component_lock = types.MethodType(define(cut(PATCHED, MAMBA, "MambaComponent",
                                                                  "acquire_component_lock"), ns,
                                                              "acquire_component_lock"), comp)
        comp.release_component_lock = types.MethodType(define(cut(PATCHED, MAMBA, "MambaComponent",
                                                                  "release_component_lock"), ns,
                                                              "release_component_lock"), comp)
        comp.eviction_priority = lambda is_leaf: 0
        comp._alloc_mamba_slot = types.MethodType(define(alloc_fn_src, ns, "_alloc_mamba_slot"), comp)
        core.components = (full, comp)
        core.components_by_type = {ComponentType.FULL: full, ComponentType.MAMBA: comp}
        return SimpleNamespace(cache=cache, core=core, comp=comp, full=full, P=Pn, alloc=alloc, ns=ns)

    def admit(w, skip=()):
        req = SimpleNamespace(last_node=1, swa_prefix_lock_released=False, rid="r")
        req.lock_receipt = w.cache.inc_lock_ref(1, skip_lock_components=skip).to_dec_params()
        return req

    ALLOC_PATCHED = cut(PATCHED, MAMBA, "MambaComponent", "_alloc_mamba_slot")
    ALLOC_PRISTINE = cut(PRISTINE, MAMBA, "MambaComponent", "_alloc_mamba_slot")
    MAMBA_T = ComponentType.MAMBA

    w = build(ALLOC_PRISTINE, lazy=True)
    admit(w)
    try:
        w.comp._alloc_mamba_slot()
        check("43559 pristine asserts on the 4-slot peak", False, "no assertion")
    except AssertionError as e:
        check("43559 pristine asserts on the 4-slot peak", "Can not alloc mamba cache" in str(e), str(e))

    w = build(ALLOC_PATCHED, lazy=True)
    req = admit(w)
    check("43559 admission locked P's Mamba (lock_ref 1)", w.P.component_data[MAMBA_T].lock_ref == 1)
    released_before = len(w.full.released)
    slot = w.comp._alloc_mamba_slot(req)
    check("43559 patched allocates on the peak", slot is not None, slot)
    check("43559 P's Mamba lock released (lock_ref 0) and evicted",
          w.P.component_data[MAMBA_T].lock_ref == 0 and w.cache.evicted == [1])
    check("43559 receipt now skips MAMBA only", tuple(req.lock_receipt.skipped_lock_components) == (MAMBA_T,))
    check("43559 FULL lock untouched by the release", len(w.full.released) == released_before)
    try:
        w.cache._dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)
        ok, err = True, ""
    except AssertionError as e:
        ok, err = False, str(e)
    check("43559 later _dec_req_lock: no double Mamba decrement", ok, err)
    check("43559 later _dec_req_lock releases FULL once", len(w.full.released) == released_before + 1)

    w = build(ALLOC_PATCHED, lazy=False)
    req = admit(w)
    try:
        w.comp._alloc_mamba_slot(req)
        check("43559 patched non-lazy unchanged (asserts)", False)
    except AssertionError:
        check("43559 patched non-lazy unchanged (asserts)", w.P.component_data[MAMBA_T].lock_ref == 1
              and tuple(req.lock_receipt.skipped_lock_components) == ())

    w = build(ALLOC_PATCHED, lazy=True)
    req = admit(w)
    req.swa_prefix_lock_released = True
    try:
        w.comp._alloc_mamba_slot(req)
        check("43559 swa-released request is not released again", False)
    except AssertionError:
        check("43559 swa-released request is not released again", w.P.component_data[MAMBA_T].lock_ref == 1)

    w = build(ALLOC_PATCHED, lazy=True)
    req = admit(w, skip=(MAMBA_T,))
    slot = w.comp._alloc_mamba_slot(req)
    check("43559 receipt that already skips MAMBA: no extra dec, evicts", slot is not None and
          tuple(req.lock_receipt.skipped_lock_components) == (MAMBA_T,) and w.cache.evicted == [1])

    w = build(ALLOC_PATCHED, lazy=True)
    req = admit(w)
    w.alloc.free([99])
    slot = w.comp._alloc_mamba_slot(req)
    check("43559 free slot: no release, receipt unchanged", slot == [99] and
          w.P.component_data[MAMBA_T].lock_ref == 1 and tuple(req.lock_receipt.skipped_lock_components) == ())

    w = build(ALLOC_PATCHED, lazy=True)
    admit(w)
    try:
        w.comp._alloc_mamba_slot()
        check("43559 req=None unchanged (asserts)", False)
    except AssertionError:
        check("43559 req=None unchanged (asserts)", True)

    UPSTREAM = ALLOC_PATCHED.replace("""                ct = self.component_type
                if (
                    req.last_node is not None
                    and not req.swa_prefix_lock_released
                    and ct not in req.lock_receipt.skipped_lock_components
                ):""", """                lock = req.lock
                ct = self.component_type
                if (
                    lock is not None
                    and not lock.swa_released
                    and ct not in lock.receipt.skipped_lock_components
                ):""")
    check("43559 upstream text differs from the port", UPSTREAM != ALLOC_PATCHED)
    w = build(UPSTREAM, lazy=True)
    req = admit(w)
    try:
        w.comp._alloc_mamba_slot(req)
        check("43559 upstream text on 0.5.21 Req raises AttributeError", False)
    except AttributeError as e:
        check("43559 upstream text on 0.5.21 Req raises AttributeError", "lock" in str(e), str(e))

    BRP = "srt/managers/scheduler_components/batch_result_processor.py"

    def boundary(pkg, interval, req_seqlen, accept_len, retained_len=None, num_non_draft=1):
        ns = {"mamba_track_grid": lambda page: interval}
        fn = define(cut(pkg, BRP, "SchedulerBatchResultProcessor", "_mamba_check_track_boundary"), ns,
                    "_mamba_check_track_boundary")
        self_ = SimpleNamespace(tree_cache=SimpleNamespace(page_size=1))
        r = SimpleNamespace(seqlen=req_seqlen)
        batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_none=lambda: False))
        result = SimpleNamespace(num_correct_drafts_per_req_cpu=[accept_len - num_non_draft],
                                 num_non_draft_tokens_per_req=num_non_draft,
                                 grammar_retained_tokens=None if retained_len is None else [[7] * retained_len])
        return fn(self_, r, batch, result, 0)

    I = 256
    for seqlen, acc in ((260, 6), (258, 3), (300, 6), (257, 1)):
        a, b = boundary(PRISTINE, I, seqlen, acc), boundary(PATCHED, I, seqlen, acc)
        check("43029 no grammar identical seqlen=%d acc=%d" % (seqlen, acc), a == b, (a, b))
    a, b = boundary(PRISTINE, I, 260, 6, retained_len=2), boundary(PATCHED, I, 260, 6, retained_len=2)
    check("43029 truncated: pristine claims a crossing that the committed run never made", a == (True, 256), a)
    check("43029 truncated: patched sees no crossing", b == (False, 0), b)
    a, b = boundary(PRISTINE, I, 258, 6, retained_len=3), boundary(PATCHED, I, 258, 6, retained_len=3)
    check("43029 truncated with a real crossing: both report it", a == b == (True, 256), (a, b))
    b = boundary(PATCHED, I, 259, 6, retained_len=3, num_non_draft=2)
    check("43029 UNO root excluded (committed = kept - 1)", b == (False, 0), b)

    nan = float("nan")
    leak_unpatched = nan * 0.0
    leak_patched = 0.0 if not (0 >= 1) else nan
    check("41736 emulation: NaN * 0 leaks (0.5.21)", leak_unpatched != leak_unpatched)
    check("41736 emulation: where-select gives 0 (patched)", leak_patched == 0.0)

    def greedy_walk_index(scores, patched):
        top_k = len(scores)
        if not patched:
            best = max(scores) if not all(s != s for s in scores) else nan
            hits = [i for i, s in enumerate(scores) if s == best]
            return min(hits) if hits else top_k
        is_nan = [s != s for s in scores]
        if any(is_nan):
            return is_nan.index(True)
        return scores.index(max(scores))

    row = [nan] * 16
    check("41737 emulation: all-NaN row walks out of range (0.5.21, index == top_k)", greedy_walk_index(row, False) == 16)
    check("41737 emulation: all-NaN row stays in range (patched, first NaN = 0)", greedy_walk_index(row, True) == 0)
    check("41737 emulation: finite row unchanged", greedy_walk_index([0.1, 0.9, 0.9, -1.0], False)
          == greedy_walk_index([0.1, 0.9, 0.9, -1.0], True) == 1)
    return RESULTS


if __name__ == "__main__":
    unittest.main()
