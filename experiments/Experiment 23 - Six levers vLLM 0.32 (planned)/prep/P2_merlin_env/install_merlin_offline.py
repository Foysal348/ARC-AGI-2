#!/usr/bin/env python3
"""Offline installer for the Merlin 3.0.0 engine (vLLM 0.32.0 fork) on the pinned Kaggle image (Ubuntu 22.04, glibc 2.35).

    python3 install_merlin_offline.py [--venv /tmp/merlin-venv] [--work /tmp/merlin] [--dry-run] [--no-verify]

Run it from the P2 payload folder (the output of foysalemonshanto/q27-e23-p2-merlin-env, attached read-only); it never
touches the network. It mirrors release/install-env.sh <venv> <engine> in install mode with MERLIN_WHEEL set, step by
step, with the same commands and Merlin's own embedded checks (the four python heredocs of install-env.sh are executed
verbatim), and these documented differences:

  1. Offline: Python 3.13 comes from the shipped python-build-standalone archive, uv from the shipped uv archive, and
     `uv pip install` gets --offline --no-index --find-links <restored wheelhouse>. Kaggle drops '+' from file names, so
     every payload file is located by its MANIFEST.json name (with or without '+') and checked by sha256 before use.
  2. The GCC/G++ >= 13 gate of install-env.sh guards the native SOURCE build; with the release wheel nothing is compiled
     at install, so an older host compiler is recorded as a warning (runtime JIT uses cc/c++ with nvcc 13.0.88).
  3. If GNU patch or flock is missing, the five FlashInfer tile files produced by release/apply-flashinfer-patch.sh on
     the build kernel are copied in instead, and Merlin's tile hash check (same expected hashes) must pass.
  4. glibc compat (only when the host glibc is older than what an engine .so needs): the release wheel is built against
     glibc 2.39. vllm/_C_stable_libtorch.abi3.so needs GLIBC_2.38 only for __isoc23_strtol / __isoc23_strtoll (the C23
     names glibc 2.38 headers emit for strtol / strtoll). The installer re-points those two undefined dynamic symbols to
     strtol / strtoll (same string table, suffix offset) as unversioned references and renames the then-unused
     GLIBC_2.38 version-need entry to the highest version the object already needs that the host has. Nothing else in the
     file changes (byte diff recorded). C23 strtol differs from C11 strtol only by also accepting a 0b/0B prefix when
     base is 0 or 2. vllm/vllm-rs (Rust frontend, GLIBC_2.39 pidfd_spawnp/pidfd_getpid) cannot be redirected and is left
     as is: it runs only with VLLM_USE_RUST_FRONTEND=1 (default 0); never set that variable with this install.
     The products list (.build-products.sha256) is recorded AFTER this step, as install-env.sh records it after a build,
     so release/install-env.sh --verify and the Merlin launcher accept the tree. Original and patched sha256 of every
     changed file are in <engine>/.merlin-offline-install.json.

At the end it prints VENV_PYTHON=..., MERLIN_TREE=..., CUDA_HOME=..., PATH=..., LIBRARY_PATH=... (when a libcuda.so link
was made) and runs `release/install-env.sh --verify <venv> <engine>` from the checkout (unless --no-verify).
Exit code 0 only when every step passed. Stdlib only; runs on the image's python3 (3.12) and on 3.13.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERLIN_TAG = "merlin-v3.0.0"
FORBIDDEN_ENV = ("VLLM_PRECOMPILED_WHEEL_LOCATION", "VLLM_USE_PRECOMPILED", "VLLM_USE_PRECOMPILED_RUST",
                 "VLLM_DOCKER_BUILD_CONTEXT", "CMAKE_ARGS", "PYTHONPATH", "PYTHONHOME")
LOG = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    LOG.append(msg)
    print("[merlin-offline] " + msg, flush=True)


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# ELF (x86_64, little endian) version-need reader / glibc compat patch. Pure python, no pyelftools.
# ----------------------------------------------------------------------------------------------------------------------
SHT_DYNSYM, SHT_GNU_VERNEED, SHT_GNU_VERSYM = 11, 0x6ffffffe, 0x6fffffff
C23_REDIRECT = {"__isoc23_strtol": "strtol", "__isoc23_strtoll": "strtoll",
                "__isoc23_strtoul": "strtoul", "__isoc23_strtoull": "strtoull",
                "__isoc23_wcstol": "wcstol", "__isoc23_wcstoll": "wcstoll",
                "__isoc23_wcstoul": "wcstoul", "__isoc23_wcstoull": "wcstoull"}


def vtuple(s):
    return tuple(int(x) for x in re.findall(r"\d+", s))


def elf_hash(name: bytes) -> int:
    h = 0
    for c in name:
        h = ((h << 4) + c) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def is_elf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def elf_parse(data: bytes):
    """Return the version-need structure of an ELF64 LE object: sections, dynsym rows, vernaux rows."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian object")
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name, typ, flags, addr, offset, size, link, info, align, entsize = struct.unpack_from("<IIQQQQIIQQ", data, off)
        secs.append({"idx": i, "name_off": name, "type": typ, "offset": offset, "size": size, "link": link,
                     "info": info, "entsize": entsize})
    dynsym = next((s for s in secs if s["type"] == SHT_DYNSYM), None)
    versym = next((s for s in secs if s["type"] == SHT_GNU_VERSYM), None)
    verneed = next((s for s in secs if s["type"] == SHT_GNU_VERNEED), None)
    out = {"syms": [], "aux": [], "dynstr_off": None}
    if dynsym is None:
        return out
    strtab = secs[dynsym["link"]]
    out["dynstr_off"], out["dynstr_size"] = strtab["offset"], strtab["size"]

    def cstr(off):
        start = strtab["offset"] + off
        end = data.index(b"\x00", start)
        return data[start:end]

    if verneed is not None:
        off = verneed["offset"]
        for _ in range(verneed["info"]):
            vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, off)
            lib = cstr(vn_file).decode()
            a = off + vn_aux
            for _ in range(vn_cnt):
                vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, a)
                out["aux"].append({"file_off": a, "lib": lib, "hash": vna_hash, "flags": vna_flags, "other": vna_other,
                                   "name_off": vna_name, "name": cstr(vna_name).decode()})
                if not vna_next:
                    break
                a += vna_next
            if not vn_next:
                break
            off += vn_next
    n = dynsym["size"] // (dynsym["entsize"] or 24)
    for i in range(n):
        so = dynsym["offset"] + i * 24
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, so)
        vs_off = versym["offset"] + 2 * i if versym is not None else None
        vs = struct.unpack_from("<H", data, vs_off)[0] if vs_off is not None else None
        out["syms"].append({"i": i, "file_off": so, "name_off": st_name, "name": cstr(st_name).decode(errors="replace"),
                            "shndx": st_shndx, "versym": vs, "versym_off": vs_off})
    return out


def elf_needs(path: Path):
    """{'GLIBC': max version string or None, 'by_symbol': [(version, symbol)] above 2.34 for GLIBC_*}"""
    data = path.read_bytes()
    info = elf_parse(data)
    by_other = {a["other"]: a for a in info["aux"]}
    gl = [a["name"] for a in info["aux"] if a["name"].startswith("GLIBC_")]
    syms = []
    for s in info["syms"]:
        if s["shndx"] == 0 and s["versym"] is not None:
            a = by_other.get(s["versym"] & 0x7FFF)
            if a and a["name"].startswith("GLIBC_"):
                syms.append((a["name"], s["name"]))
    return {"max_glibc": max(gl, key=vtuple) if gl else None, "glibc_needs": sorted(set(gl), key=vtuple),
            "symbols": sorted(set(syms), key=lambda x: (vtuple(x[0]), x[1]))}


def glibc_compat_patch(path: Path, host: tuple, dry_run=False):
    """Make `path` loadable on glibc `host` when the only newer needs are C23 strto* redirects. Returns a report."""
    data = bytearray(path.read_bytes())
    info = elf_parse(bytes(data))
    by_other = {a["other"]: a for a in info["aux"]}
    rep = {"file": str(path), "sha256_before": hashlib.sha256(data).hexdigest(), "changes": [], "unresolved": []}
    high = {a["other"] for a in info["aux"] if a["name"].startswith("GLIBC_") and vtuple(a["name"]) > host}
    if not high:
        rep["status"] = "not needed"
        return rep
    dynstr = info["dynstr_off"]
    base = {a["lib"]: a["other"] for a in info["aux"] if a["name"] == "GLIBC_2.2.5"}  # x86_64 base version of strtol
    for s in info["syms"]:
        if s["versym"] is None or (s["versym"] & 0x7FFF) not in high:
            continue
        ver = by_other[s["versym"] & 0x7FFF]["name"]
        new = C23_REDIRECT.get(s["name"])
        if s["shndx"] != 0 or new is None or not s["name"].endswith(new):
            rep["unresolved"].append(f"{s['name']}@{ver}")
            continue
        new_off = s["name_off"] + (len(s["name"]) - len(new))
        assert bytes(data[dynstr + new_off: dynstr + new_off + len(new) + 1]) == new.encode() + b"\x00"
        lib = by_other[s["versym"] & 0x7FFF]["lib"]
        struct.pack_into("<I", data, s["file_off"], new_off)          # st_name -> suffix "strtol"
        if lib in base:   # strtol@GLIBC_2.2.5, exactly what a build against an older glibc references
            struct.pack_into("<H", data, s["versym_off"], base[lib])
            rep["changes"].append(f"symbol {s['name']}@{ver} -> {new}@GLIBC_2.2.5 (version index {base[lib]})")
        else:             # VER_NDX_GLOBAL: unversioned reference, binds to the host default version
            struct.pack_into("<H", data, s["versym_off"], 1)
            rep["changes"].append(f"symbol {s['name']}@{ver} -> {new} (unversioned, binds to the host default version)")
    if rep["unresolved"]:
        rep["status"] = "unresolved"
        return rep
    info2 = elf_parse(bytes(data))
    still = {(s["versym"] & 0x7FFF) for s in info2["syms"] if s["versym"] is not None}
    for a in info2["aux"]:
        if a["other"] not in high:
            continue
        if a["other"] in still:
            raise Fail(f"{path}: version {a['name']} still referenced after redirect")
        same_lib = [b for b in info2["aux"] if b["lib"] == a["lib"] and b["name"].startswith("GLIBC_")
                    and vtuple(b["name"]) <= host and b["other"] not in high]
        if not same_lib:
            rep["unresolved"].append(f"no host-supported {a['lib']} version to re-point {a['name']} to")
            continue
        tgt = max(same_lib, key=lambda b: vtuple(b["name"]))
        struct.pack_into("<I", data, a["file_off"], elf_hash(tgt["name"].encode()))   # vna_hash
        struct.pack_into("<I", data, a["file_off"] + 8, tgt["name_off"])             # vna_name
        rep["changes"].append(f"version-need {a['lib']} {a['name']} (index {a['other']}, now unused) -> {tgt['name']}")
    if rep["unresolved"]:
        rep["status"] = "unresolved"
        return rep
    rep["sha256_after"] = hashlib.sha256(data).hexdigest()
    rep["bytes_changed"] = sum(1 for x, y in zip(data, path.read_bytes()) if x != y)
    if not dry_run:
        tmp = path.with_name(path.name + ".compat-tmp")
        tmp.write_bytes(bytes(data))
        os.chmod(tmp, path.stat().st_mode)
        os.replace(tmp, path)
        after = elf_needs(path)
        if after["max_glibc"] and vtuple(after["max_glibc"]) > host:
            raise Fail(f"{path}: still needs {after['max_glibc']} after the compat patch")
        rep["max_glibc_after"] = after["max_glibc"]
    rep["status"] = "patched" if not dry_run else "would patch"
    return rep


def host_glibc():
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        return vtuple(libc.gnu_get_libc_version().decode())
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def sha256(p: Path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, env=None, cwd=None, dry=False, check=True, timeout=None, capture=False):
    shown = " ".join(str(c) for c in cmd)
    say("$ " + (f"(cd {cwd} && " if cwd else "") + shown + (")" if cwd else ""))
    if dry:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, timeout=timeout, text=True,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    if check and r.returncode != 0:
        tail = (r.stdout or "")[-4000:] if capture else ""
        raise Fail(f"command failed (rc {r.returncode}): {shown}\n{tail}")
    return r


def heredocs(script: str):
    """The python heredocs of install-env.sh, in order: [(argv-tail-shell, body)]."""
    out = []
    for m in re.finditer(r'"\$PY" (-[PI]) - ([^\n]*?)<<\'PY\'\n(.*?)\nPY\n', script, re.S):
        out.append((m.group(1), m.group(2).strip(), m.group(3) + "\n"))
    return out


def locate(root: Path, name: str):
    """Find a payload file by its original name; Kaggle may have dropped '+' from it."""
    for cand in (name, name.replace("+", "")):
        if (root / cand).exists():
            return root / cand
    base = Path(name).name
    hits = sorted(p for p in root.rglob("*") if p.is_file() and p.name in (base, base.replace("+", "")))
    return hits[0] if hits else None


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in FORBIDDEN_ENV and not k.endswith("_SRC_DIR")
           and not k.startswith("FETCHCONTENT_")}
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "UV_OFFLINE": "1", "UV_NO_CONFIG": "1",
                "UV_PYTHON_DOWNLOADS": "never", "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1"})
    env.update(extra or {})
    return env


# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venv", default="/tmp/merlin-venv")
    ap.add_argument("--work", default="/tmp/merlin", help="python, uv, restored wheel links, engine checkout")
    ap.add_argument("--payload", default=str(HERE), help="folder with MANIFEST.json (default: this script's folder)")
    ap.add_argument("--dry-run", action="store_true", help="resolve and check the payload, print every step, change nothing")
    ap.add_argument("--no-verify", action="store_true", help="skip release/install-env.sh --verify at the end")
    ap.add_argument("--hash-all", action="store_true", help="sha256 every payload file (default: non-wheel files and "
                                                            "renamed ones; uv --require-hashes checks the lock wheels)")
    ap.add_argument("--no-glibc-compat", action="store_true", help="never patch engine .so files")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    a = ap.parse_args(argv)
    t0 = time.time()
    dry = a.dry_run
    payload = Path(a.payload).resolve()
    venv, work = Path(a.venv), Path(a.work)
    report = {"payload": str(payload), "venv": str(venv), "work": str(work), "dry_run": dry, "steps": {},
              "deviations_from_install_env_sh": []}
    try:
        man_p = locate(payload, "MANIFEST.json")
        if man_p is None:
            raise Fail(f"MANIFEST.json not found under {payload}")
        man = json.loads(man_p.read_text(encoding="utf-8"))
        payload = man_p.parent
        files = {}
        renamed = []
        for f in man["files"]:
            p = locate(payload, f["name"])
            if p is None:
                raise Fail(f"payload file missing: {f['name']}")
            if p.name != Path(f["name"]).name:
                renamed.append(f["name"])
            if a.hash_all or f["role"] != "lock-wheel" or p.name != Path(f["name"]).name:
                got = sha256(p)
                if got != f["sha256"]:
                    raise Fail(f"sha256 mismatch for {f['name']}: {got} != {f['sha256']}")
            files[f["name"]] = (p, f)
        say(f"payload {payload}: {len(files)} files, renamed by Kaggle: {len(renamed)}")
        report["steps"]["payload"] = {"files": len(files), "renamed": renamed}
        role = {}
        for n, (p, f) in files.items():
            role.setdefault(f["role"], []).append((n, p, f))
        one = lambda r: role[r][0]  # noqa: E731
        hostv = host_glibc()
        report["host_glibc"] = ".".join(map(str, hostv)) if hostv else None
        say("host glibc", report["host_glibc"], "| python", sys.version.split()[0])

        if venv.exists() or venv.is_symlink():
            raise Fail(f"{venv} already exists; choose a new venv path (install-env.sh refuses an existing venv)")
        wheels = work / "wheels"
        pyroot, uvbin, engine = work / "python", work / "bin" / "uv", work / "merlin-engine"
        if not dry:
            if engine.exists():
                raise Fail(f"{engine} already exists; use a new --work folder")
            wheels.mkdir(parents=True, exist_ok=True)
            (work / "bin").mkdir(parents=True, exist_ok=True)
        # 1. restored wheel names (symlinks under the original names)
        for n, p, f in role["lock-wheel"] + role["engine-wheel"]:
            dst = wheels / Path(n).name
            say(f"link {dst.name} -> {p}") if p.name != dst.name else None
            if not dry and not dst.exists():
                dst.symlink_to(p)
        wheel_name, wheel_src, _ = one("engine-wheel")
        WHEEL = wheels / Path(wheel_name).name
        # 2. standalone CPython 3.13, uv, the engine checkout
        n, p, f = one("python-standalone")
        say(f"extract {n} -> {pyroot}")
        if not dry:
            with tarfile.open(p) as t:
                t.extractall(work, filter="tar") if hasattr(tarfile, "data_filter") else t.extractall(work)
            if not pyroot.exists():
                raise Fail(f"{n} did not create {pyroot}")
        py313 = pyroot / "bin" / "python3.13"
        n, p, f = one("uv")
        say(f"extract {n} -> {uvbin}")
        if not dry:
            with tarfile.open(p) as t:
                m = next(x for x in t.getmembers() if x.name.endswith("/uv"))
                with t.extractfile(m) as src, open(uvbin, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            os.chmod(uvbin, 0o755)
        n, p, f = one("engine-checkout")
        say(f"extract {n} -> {engine}")
        if not dry:
            with tarfile.open(p) as t:
                t.extractall(work, filter="tar") if hasattr(tarfile, "data_filter") else t.extractall(work)
            if not (engine / "setup.py").exists() or not (engine / "vllm" / "__init__.py").exists():
                raise Fail(f"{engine} is not a Merlin-engine source checkout")
        REQ = engine / "release" / "requirements-0.32.lock"
        env = clean_env({"PATH": f"{work / 'bin'}:{pyroot / 'bin'}:" + os.environ.get("PATH", "/usr/bin:/bin"),
                         "UV_CACHE_DIR": str(work / "uv-cache"), "UV_LINK_MODE": "hardlink"})
        # 3. source identity (install-env.sh lines 59-76)
        if shutil.which("git") is None and not dry:
            raise Fail("git is required (install-env.sh)")
        if not dry:
            top = run(["git", "-C", engine, "rev-parse", "--show-toplevel"], env=env, capture=True).stdout.strip()
            if Path(top).resolve() != engine.resolve():
                raise Fail("engine-source-path must be the Git checkout root")
            commit = run(["git", "-C", engine, "rev-parse", "--verify", "HEAD"], env=env, capture=True).stdout.strip()
            tree = run(["git", "-C", engine, "rev-parse", "--verify", "HEAD^{tree}"], env=env, capture=True).stdout.strip()
            dirty = run(["git", "-C", engine, "status", "--porcelain", "--untracked-files=no"], env=env, capture=True).stdout
            if dirty.strip():
                raise Fail("selected checkout has modified tracked source:\n" + dirty[:2000])
            if commit != man["merlin"]["commit"] or tree != man["merlin"]["tree"]:
                raise Fail(f"checkout {commit}/{tree} is not {MERLIN_TAG} {man['merlin']['commit']}/{man['merlin']['tree']}")
        else:
            commit, tree = man["merlin"]["commit"], man["merlin"]["tree"]
        source_id = f"upstream-base vllm-0.32.0\ncommit {commit}\ntree {tree}"
        say(f"SOURCE OK: commit={commit} tree={tree} base=vllm-0.32.0")
        if not dry:
            script = (engine / "release" / "install-env.sh").read_text(encoding="utf-8")
        else:  # read it from the checkout archive without extracting anything
            with tarfile.open(one("engine-checkout")[1]) as t:
                script = t.extractfile("merlin-engine/release/install-env.sh").read().decode("utf-8")
        docs = heredocs(script)
        marks = ["0.32.0 DEPENDENCY PINS OK", "FLASHINFER TILE HASHES OK", "ENGINE IMPORT OK", "BUILD PRODUCTS RECORDED"]
        if len(docs) != 4 or any(mk not in d[2] for mk, d in zip(marks, docs)):
            raise Fail(f"install-env.sh heredocs changed ({len(docs)} found); this installer mirrors {MERLIN_TAG}")
        report["install_env_sh_sha256"] = hashlib.sha256(script.encode("utf-8")).hexdigest()
        if report["install_env_sh_sha256"] != man["merlin"]["install_env_sh_sha256"]:
            raise Fail("release/install-env.sh differs from the one this installer was written against")
        if not dry:
            lock_txt = REQ.read_text(encoding="utf-8")
            if re.search(r"TODO-(PIN|HASH)", lock_txt):
                raise Fail("requirements-0.32.lock contains unresolved pins or hashes")
            for x in ("apply-flashinfer-patch.sh", "check-tile-proof.sh", "wheel.py"):
                if not (engine / "release" / x).exists():
                    raise Fail(f"missing release/{x}")
        # 4. wheel identity (install-env.sh lines 77-85)
        run([py313, "-I", engine / "release" / "wheel.py", "verify", WHEEL, engine / "release" / "merlin-wheel.sha256",
             engine], env=env, dry=dry)
        wheel_sha = sha256(WHEEL) if not dry else one("engine-wheel")[2]["sha256"]
        source_id += f"\nwheel {wheel_sha}  {WHEEL.name}"
        # 5. compilers (install-env.sh lines 90-99): recorded; the wheel route compiles nothing at install
        cc = {}
        for t in ("cc", "c++"):
            w = shutil.which(t)
            v = subprocess.run([t, "-dumpversion"], capture_output=True, text=True).stdout.strip() if w else ""
            cc[t] = {"path": w, "version": v}
            major = int(v.split(".")[0]) if v[:1].isdigit() else 0
            if not w:
                say(f"WARNING: {t} not found; runtime JIT (FlashInfer, Triton) needs cc/c++")
            elif major < 13:
                say(f"note: {t} is version {v}; install-env.sh requires >= 13 for the native source build only. "
                    "The release wheel route compiles nothing at install; runtime JIT uses it with nvcc 13.0.")
                report["deviations_from_install_env_sh"].append(f"{t} {v} < 13 accepted (wheel route)")
        report["compilers"] = cc
        # 6. venv + locked dependencies (lines 100-102), offline
        run([uvbin, "venv", "--python", py313, venv], env=env, dry=dry)
        PY = venv / "bin" / "python"
        run([uvbin, "pip", "install", "--python", PY, "--require-hashes", "--only-binary", ":all:", "--offline",
             "--no-index", "--find-links", wheels, "-r", REQ], env=env, dry=dry)
        report["deviations_from_install_env_sh"].append("uv pip install --offline --no-index --find-links <wheelhouse>")

        def heredoc(i, *args):
            flag, _shell_args, body = docs[i]
            r = run([PY, flag, "-", *args], env=env, dry=dry, capture=True, check=False) if dry else \
                subprocess.run([str(PY), flag, "-", *[str(x) for x in args]], input=body, env=env, text=True,
                               capture_output=True)
            out = (r.stdout or "") + (r.stderr or "")
            say(f"heredoc {i + 1}: rc {r.returncode} | {out.strip()[-600:]}")
            if r.returncode != 0:
                raise Fail(f"install-env.sh check {i + 1} failed: {out[-3000:]}")
            return out
        # 7. pins (lines 111-146)
        heredoc(0, REQ)
        # 8. FlashInfer tile patch (line 149) + tile hashes (152-176)
        have_tools = all(shutil.which(t) for t in ("patch", "flock"))
        if have_tools or dry:
            run(["bash", engine / "release" / "apply-flashinfer-patch.sh", venv], env=env, dry=dry)
        else:
            say("GNU patch / flock missing: installing the five pre-patched FlashInfer files from the payload")
            site = Path(subprocess.run([str(PY), "-c", 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                                       capture_output=True, text=True, check=True).stdout.strip())
            for n, p, f in role["flashinfer-patched"]:
                dst = site / "flashinfer" / f["dest"]
                shutil.copyfile(p, dst)
            for pc in (site / "flashinfer").rglob("__pycache__"):
                shutil.rmtree(pc, ignore_errors=True)
            report["deviations_from_install_env_sh"].append("FlashInfer tile files copied (patch/flock missing)")
        heredoc(1)
        # 9. CUDA links (lines 178-195)
        if not dry:
            purelib = subprocess.run([str(PY), "-P", "-c", 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                                     capture_output=True, text=True, env=env, check=True).stdout.strip()
        else:
            purelib = str(venv / "lib" / "python3.13" / "site-packages")
        CU = Path(purelib) / "nvidia" / "cu13"
        if not dry:
            if not (CU / "lib").is_dir():
                raise Fail("lock must include the qualified CUDA 13.0 compiler toolkit wheels")
            if not (CU / "lib64").exists():
                (CU / "lib64").symlink_to("lib")
            for so in sorted((CU / "lib").glob("lib*.so.[0-9]*")):
                base = CU / "lib" / (so.name.split(".so.")[0] + ".so")
                if not base.exists():
                    base.symlink_to(so.name)
            if not os.access(CU / "bin" / "nvcc", os.X_OK):
                raise Fail("no nvcc under the locked CUDA toolkit")
        env["CUDA_HOME"] = str(CU)
        env["PATH"] = f"{venv / 'bin'}:{CU / 'bin'}:" + env["PATH"]
        # 10. editable install from the release wheel (lines 198-202)
        ed_env = dict(env, PYTHONPATH=str(engine), VLLM_TARGET_DEVICE="cuda", VLLM_VERSION_OVERRIDE="0.32.0",
                      VLLM_USE_PRECOMPILED="1", VLLM_PRECOMPILED_WHEEL_LOCATION=str(WHEEL))
        run([uvbin, "pip", "install", "--python", PY, "--no-deps", "--no-build-isolation", "--offline", "--no-index",
             "--editable", "."], env=ed_env, cwd=engine, dry=dry)
        # 11. glibc compat for the extracted engine products
        compat = []
        if not dry:
            for so in sorted(p for p in (engine / "vllm").rglob("*") if p.is_file() and (p.suffix == ".so" or ".so." in p.name
                                                                                        or p.name == "vllm-rs")):
                if not is_elf(so):
                    continue
                need = elf_needs(so)
                row = {"file": str(so.relative_to(engine)), "max_glibc": need["max_glibc"],
                       "above_host": [f"{v} {s}" for v, s in need["symbols"] if hostv and vtuple(v) > hostv]}
                if hostv and need["max_glibc"] and vtuple(need["max_glibc"]) > hostv:
                    if so.name == "vllm-rs":
                        row["status"] = "left as is (Rust frontend, used only with VLLM_USE_RUST_FRONTEND=1)"
                    elif a.no_glibc_compat:
                        row["status"] = "NOT PATCHED (--no-glibc-compat)"
                    else:
                        r = glibc_compat_patch(so, hostv)
                        row.update(r)
                        if r["status"] != "patched":
                            raise Fail(f"glibc compat failed for {so}: {r}")
                        say(f"glibc compat {row['file']}: {'; '.join(r['changes'])}")
                else:
                    row["status"] = "ok"
                compat.append(row)
            if any(r["status"].startswith("patched") for r in compat):
                report["deviations_from_install_env_sh"].append("glibc compat redirect of __isoc23_strtol/strtoll")
        report["steps"]["glibc_compat"] = compat
        # 12. engine import (lines 210-240) + uv pip check (241)
        heredoc(2, engine)
        run([uvbin, "pip", "check", "--python", PY], env=env, dry=dry)
        # 13. build products (lines 243-273) + .build-source (275-276)
        heredoc(3, engine, f"wheel {WHEEL.name}")
        if not dry:
            (engine / ".build-source").write_text(source_id + "\n", encoding="utf-8")
            (engine / ".merlin-offline-install.json").write_text(json.dumps({
                "glibc_compat": compat, "deviations": report["deviations_from_install_env_sh"],
                "compilers": cc, "host_glibc": report["host_glibc"]}, indent=1), encoding="utf-8")
        if not dry:
            shutil.rmtree(work / "uv-cache", ignore_errors=True)  # venv files are hardlinks (or copies) of it
        say(f"ENGINE INSTALLED: TREE={engine} VENV={venv}")
        # 14. runtime helpers: libcuda.so for -lcuda JIT links (Kaggle ships only libcuda.so.1)
        lib_path = None
        for cand in ("/usr/local/nvidia/lib64/libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                     "/usr/local/cuda/compat/libcuda.so.1"):
            if Path(cand).exists() and not dry:
                d = work / "libcuda-link"
                d.mkdir(exist_ok=True)
                if not (d / "libcuda.so").exists():
                    (d / "libcuda.so").symlink_to(cand)
                lib_path = str(d)
                break
        # 15. Merlin's own read-only verification
        if not a.no_verify:
            r = run(["bash", engine / "release" / "install-env.sh", "--verify", venv, engine], env=env, dry=dry,
                    capture=True, check=False)
            out = r.stdout or ""
            ok = r.returncode == 0 and ("ENVIRONMENT VERIFIED" in out or dry)
            report["steps"]["install_env_verify"] = {"rc": r.returncode, "ok": ok, "tail": out[-3000:]}
            say("install-env.sh --verify:", "OK" if ok else "FAILED", out.strip()[-400:])
            if not ok:
                raise Fail("release/install-env.sh --verify failed")
        print("VENV_PYTHON=" + str(PY))
        print("MERLIN_TREE=" + str(engine))
        print("CUDA_HOME=" + str(CU))
        print("PATH=" + f"{venv / 'bin'}:{CU / 'bin'}:$PATH")
        if lib_path:
            print("LIBRARY_PATH=" + lib_path)
        report["venv_python"], report["tree"], report["cuda_home"], report["library_path"] = str(PY), str(engine), str(CU), lib_path
        report["ok"] = True
    except Exception as e:  # noqa: BLE001 - every failure ends in the report
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
        say("FAILED:", report["error"][:3000])
    report["sec"] = round(time.time() - t0, 1)
    report["log"] = LOG[-400:]
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(("MERLIN OFFLINE INSTALL OK" if report["ok"] else "MERLIN OFFLINE INSTALL FAILED") + f" ({report['sec']} s)")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
