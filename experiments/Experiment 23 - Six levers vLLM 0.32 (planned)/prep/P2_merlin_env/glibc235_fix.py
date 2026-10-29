"""E23 P2: make the Merlin release .so files load on glibc 2.35 (Kaggle image, Ubuntu 22.04).

The Merlin 3.0.0 wheel's vllm/_C_stable_libtorch.abi3.so needs libc.so.6 version GLIBC_2.38, only for
__isoc23_strtol / __isoc23_strtoll (C23 names of strtol / strtoll that GCC 13 + glibc 2.38 headers emit).
This tool, per ELF file:
  1. lists every dynamic symbol bound to a libc.so.6 version above MAX_OK (2.35);
  2. refuses (exit 2) if any such symbol is not one the shim provides (SHIM_SYMBOLS);
  3. sets VER_FLG_WEAK on those version-need entries, so ld.so does not fail on the missing version
     (glibc dl-version.c: a weak version that is not found is not fatal); the symbols then bind to the
     unversioned definitions in the LD_PRELOADed shim (dl-lookup.c accepts an unversioned definition
     from an object that has no version info).
No other byte changes. Standard library only.
usage: python glibc235_fix.py [--dry-run] <file.so> ...   -> prints one GLIBCFIX {...} JSON line"""
import json
import struct
import sys

MAX_OK = (2, 35)
SHIM_SYMBOLS = {"__isoc23_strtol", "__isoc23_strtoll", "__isoc23_strtoul", "__isoc23_strtoull"}
SHT_DYNSYM, SHT_GNU_VERNEED, SHT_GNU_VERSYM = 11, 0x6FFFFFFE, 0x6FFFFFFF
VER_FLG_WEAK = 0x2


def glibc_ver(name):
    if not name.startswith("GLIBC_"):
        return None
    try:
        return tuple(int(x) for x in name[6:].split("."))
    except ValueError:
        return None


def cstr(buf, off):
    return buf[off:buf.index(b"\0", off)].decode()


def scan(path):
    data = bytearray(open(path, "rb").read())
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 file")
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        sh = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        secs.append({"type": sh[1], "offset": sh[4], "size": sh[5], "link": sh[6], "entsize": sh[9]})
    by_type = {}
    for s in secs:
        by_type.setdefault(s["type"], s)
    vn = by_type.get(SHT_GNU_VERNEED)
    rows = {"needs": [], "high": [], "high_symbols": []}
    if vn is None:
        return data, rows
    strtab = secs[vn["link"]]
    st = bytes(data[strtab["offset"]:strtab["offset"] + strtab["size"]])
    off, high_idx = vn["offset"], {}
    while True:
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, off)
        lib = cstr(st, vn_file)
        aoff = off + vn_aux
        for _ in range(vn_cnt):
            vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, aoff)
            name = cstr(st, vna_name)
            rows["needs"].append([lib, name, vna_flags, vna_other])
            v = glibc_ver(name)
            if lib == "libc.so.6" and v and v > MAX_OK:
                rows["high"].append({"lib": lib, "version": name, "flags": vna_flags, "index": vna_other,
                                     "flags_offset": aoff + 4})
                high_idx[vna_other] = name
            if not vna_next:
                break
            aoff += vna_next
        if not vn_next:
            break
        off += vn_next
    if high_idx:
        dynsym, versym = by_type.get(SHT_DYNSYM), by_type.get(SHT_GNU_VERSYM)
        dstr = secs[dynsym["link"]]
        ds = bytes(data[dstr["offset"]:dstr["offset"] + dstr["size"]])
        n = dynsym["size"] // dynsym["entsize"]
        for i in range(n):
            st_name, = struct.unpack_from("<I", data, dynsym["offset"] + i * dynsym["entsize"])
            ndx, = struct.unpack_from("<H", data, versym["offset"] + 2 * i)
            if (ndx & 0x7FFF) in high_idx:
                rows["high_symbols"].append([cstr(ds, st_name), high_idx[ndx & 0x7FFF]])
    return data, rows


def main(argv):
    dry = "--dry-run" in argv
    files = [a for a in argv if a != "--dry-run"]
    out = {"dry_run": dry, "files": {}, "ok": True}
    for f in files:
        rec = {}
        try:
            data, rows = scan(f)
            rec.update(high=[{k: v for k, v in h.items() if k != "flags_offset"} for h in rows["high"]],
                       high_symbols=rows["high_symbols"])
            unknown = sorted({s for s, _ in rows["high_symbols"]} - SHIM_SYMBOLS)
            rec["unknown_symbols"] = unknown
            if unknown:
                rec["action"] = "REFUSED: symbols above glibc 2.35 that the shim does not provide"
                out["ok"] = False
            elif not rows["high"]:
                rec["action"] = "none needed"
            else:
                for h in rows["high"]:
                    fl, = struct.unpack_from("<H", data, h["flags_offset"])
                    struct.pack_into("<H", data, h["flags_offset"], fl | VER_FLG_WEAK)
                if not dry:
                    open(f, "wb").write(data)
                    _, again = scan(f)
                    rec["after"] = [[h["version"], h["flags"]] for h in again["high"]]
                    if not all(h["flags"] & VER_FLG_WEAK for h in again["high"]):
                        rec["action"] = "FAILED: weak flag not set after write"
                        out["ok"] = False
                        out["files"][f] = rec
                        continue
                rec["action"] = "weak flag set" + (" (dry run, not written)" if dry else "")
        except Exception as e:
            rec["action"] = f"ERROR {type(e).__name__}: {e}"
            out["ok"] = False
        out["files"][f] = rec
    print("GLIBCFIX " + json.dumps(out))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
