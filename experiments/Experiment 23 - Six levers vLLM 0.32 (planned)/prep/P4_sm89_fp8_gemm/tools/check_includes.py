#!/usr/bin/env python3
"""Local (no compiler) include-graph check for the P4 sources: follows EVERY #include (also those inside #if blocks, so
the closure is a superset of what nvcc reads) from the given translation units, resolving quotes against the
including file's directory first, then the -I list, then the -isystem list (nvcc/gcc order). Reports every unresolved
header with its includer, split into C/C++ standard / POSIX headers (provided by the host toolchain) and the rest.

usage: python check_includes.py --tu FILE [--tu FILE] -I DIR [-I DIR ...] [--isystem DIR ...] [--json OUT]
"""
import argparse
import json
import re
import sys
from pathlib import Path

INC = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
STD = set("""algorithm any array atomic bit bitset cassert cctype cerrno cfenv cfloat charconv chrono cinttypes climits
clocale cmath codecvt compare complex concepts condition_variable coroutine csetjmp csignal cstdarg cstddef cstdint
cstdio cstdlib cstring ctime cuchar cwchar cwctype deque exception execution expected filesystem format forward_list
fstream functional future initializer_list iomanip ios iosfwd iostream istream iterator latch limits list locale map
memory memory_resource mutex new numbers numeric optional ostream queue random ranges ratio regex scoped_allocator
semaphore set shared_mutex source_location span sstream stack stdexcept stop_token streambuf string string_view
strstream syncstream system_error thread tuple type_traits typeindex typeinfo unordered_map unordered_set utility
valarray variant vector version execinfo.h cxxabi.h""".split())
STD |= {"ciso646"}
C_POSIX = re.compile(r"^(assert|ctype|errno|fenv|float|inttypes|limits|locale|math|setjmp|signal|stdarg|stddef|stdint|"
                     r"stdio|stdlib|string|strings|time|uchar|wchar|wctype|stdbool|unistd|pthread|dlfcn|fcntl|sched|"
                     r"malloc|memory|alloca|immintrin|x86intrin|emmintrin|xmmintrin|mmintrin|cpuid|link|elf|endian|"
                     r"features)\.h$|^(sys|bits|linux|asm|arpa|netinet)/")
# Unresolved on purpose: only reachable under a preprocessor branch that is inactive for nvcc 13.0 + g++ on Linux x86_64
# (verified by reading each guard on 2026-10-29). Anything unresolved that is NOT listed here fails the check.
GUARDED = {
    r"^(CL|sycl)/sycl\.hpp$": "torch headeronly: SYCL builds only",
    r"^hip/": "torch headeronly: ROCm builds only",
    r"^(arm_neon|arm_sve|altivec|spe)\.h$": "torch headeronly cpu/vec: non-x86 hosts only",
    r"^TargetConditionals\.h$": "torch Macros.h: Apple only",
    r"^(corecrt|yvals|windows|intrin|crtdefs|winapifamily|dbghelp)\.h$|^xutility$": "MSVC host only",
    r"^__config$|^__memory/|^__utility/": "libc++ host stdlib only (we use libstdc++)",
    r"^debug/safe_iterator\.h$": "CCCL iterator_traits.h: only with _GLIBCXX_DEBUG",
    r"^nvrtc_device_runtime\.h$|^target$": "cuda_runtime.h: only under __CUDACC_RTC__ (NVRTC)",
    r"^nvtx3/": "CCCL __nvtx: behind __has_include, falls back to its bundled cuda/__nvtx/nvtx3.h",
    r"^cuda/std/__compare/|^cuda/std/compare$": "CCCL 3.4.3: _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() is defined 0",
    r"^cuda/std/__type_traits/predicate_traits\.h$": "CCCL comp.h: only with _LIBCUDACXX_HAS_STRING",
    r"^cuda/std/detail/libcxx/include/": "cute/config.hpp: __has_include fallback for old CCCL layouts",
}


def resolve(name, quote, cur_dir, idirs, sdirs):
    cands = ([cur_dir] if quote == '"' else []) + idirs + sdirs
    for d in cands:
        p = d / name
        if p.is_file():
            return p.resolve()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu", action="append", required=True)
    ap.add_argument("-I", dest="idirs", action="append", default=[])
    ap.add_argument("--isystem", action="append", default=[])
    ap.add_argument("--json", default=None)
    ap.add_argument("--root-label", action="append", default=[], help="LABEL=DIR for the per-root file counts")
    a = ap.parse_args()
    idirs = [Path(d).resolve() for d in a.idirs]
    sdirs = [Path(d).resolve() for d in a.isystem]
    seen, todo = {}, [Path(t).resolve() for t in a.tu]
    unresolved = {}
    edges = 0
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        seen[f] = []
        for q, name in INC.findall(txt):
            edges += 1
            p = resolve(name, q, f.parent, idirs, sdirs)
            if p is None:
                unresolved.setdefault(name, set()).add(str(f))
            else:
                seen[f].append(str(p))
                if p not in seen:
                    todo.append(p)
    std = {k: sorted(v) for k, v in unresolved.items() if k in STD or C_POSIX.search(k)}
    guarded = {k: next(why for pat, why in GUARDED.items() if re.search(pat, k))
               for k in unresolved if k not in std and any(re.search(p, k) for p in GUARDED)}
    other = {k: sorted(v) for k, v in unresolved.items() if k not in std and k not in guarded}
    roots = {}
    for lab in a.root_label:
        k, d = lab.split("=", 1)
        d = str(Path(d).resolve())
        roots[k] = sum(1 for f in seen if str(f).startswith(d))
    rep = {"translation_units": a.tu, "files_reached": len(seen), "include_edges": edges, "files_per_root": roots,
           "unresolved_std_or_posix": sorted(std), "unresolved_inactive_branch": guarded, "unresolved_other": other,
           "ok": not other}
    print(json.dumps({k: v for k, v in rep.items() if k != "unresolved_other"}, indent=1))
    for k, v in sorted(other.items()):
        print(f"UNRESOLVED <{k}>  from: {v[:3]}{' ...' if len(v) > 3 else ''}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
