"""E28 W4: the review view of the notebook (used by build_e28_w4.py and e28w4_unit_tests.py; stdlib only).

unpack_cells(cells)   the E28 W4 notebook's cells with the two size-only packings undone: the W17 loader (e21_cell7) and the
                      W18 loader (the engine donors) replaced by their decoded, sha-checked text, so the cells read as plain
                      code again (raises on a sha mismatch).
normalize(text)       every packed module literal (NB5's zlib + base64 and the lzma + base85 re-encoding of W16) replaced by
                      __PACKED_<sha256[:16] of the decoded text>__, so the two encodings of the same module compare equal.
review_diff(a, b)     a unified diff (n=0) of two cell lists after normalize(): NB6Q -> E28 W4 shows only the E28 edits.
"""
import base64
import difflib
import hashlib
import lzma
import re
import zlib

_W17 = re.compile(r'# ---- E28 W4 W17 [^\n]*\n# [^\n]*\nE28_C7_SHA = "([0-9a-f]{64})"\n_e28_c7 = __import__\("lzma"\)\.decompress\('
                  r'__import__\("base64"\)\.b85decode\("([^"]+)"\)\)\.decode\("utf-8"\)\n(?:[^\n]*\n){3}')
_W18 = re.compile(r'# ---- E28 W4 W18 [^\n]*\n# [^\n]*\n# [^\n]*\nE28_ENG_SHA = "([0-9a-f]{64})"\n_e28_eng = __import__\("lzma"\)\.decompress\('
                  r'__import__\("base64"\)\.b85decode\("([^"]+)"\)\)\.decode\("utf-8"\)\n(?:[^\n]*\n){3}\n\n')
_ZB64 = re.compile(r'__import__\("zlib"\)\.decompress\(__import__\("base64"\)\.b64decode\("([A-Za-z0-9+/=]+)"\)\)')
_LB85 = re.compile(r'__import__\("lzma"\)\.decompress\(__import__\("base64"\)\.b85decode\("([^"]+)"\)\)')


def _sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _undo(text, rx, label):
    ms = list(rx.finditer(text))
    if len(ms) > 1:
        raise ValueError(f"{label}: {len(ms)} loaders")
    for m in ms:
        seg = lzma.decompress(base64.b85decode(m.group(2))).decode("utf-8")
        if _sha(seg) != m.group(1):
            raise ValueError(f"{label}: embedded sha mismatch")
        text = text[:m.start()] + seg + text[m.end():]
    return text, len(ms)


def unpack_cells(cells):
    """cells = [(kind, text)]; returns (unpacked cells, {"W17": n, "W18": n})."""
    out, n = [], {"W17": 0, "W18": 0}
    for k, t in cells:
        if k == "code":
            t, a = _undo(t, _W17, "W17")
            t, b = _undo(t, _W18, "W18")
            n["W17"] += a
            n["W18"] += b
        out.append((k, t))
    return out, n


def normalize(text):
    def z(m):
        return "__PACKED_" + _sha(zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8"))[:16] + "__"

    def l(m):
        return "__PACKED_" + _sha(lzma.decompress(base64.b85decode(m.group(1))).decode("utf-8"))[:16] + "__"
    return _LB85.sub(l, _ZB64.sub(z, text))


def review_diff(a, b, name_a="NB6Q_SUBMIT_499a4319", name_b="E28_W4"):
    ta = "".join(f"# ---- cell {i} ({k})\n" + normalize(t).rstrip("\n") + "\n" for i, (k, t) in enumerate(a))
    tb = "".join(f"# ---- cell {i} ({k})\n" + normalize(t).rstrip("\n") + "\n" for i, (k, t) in enumerate(b))
    return "".join(difflib.unified_diff(ta.splitlines(True), tb.splitlines(True), name_a + " (cells, normalized)",
                                        name_b + " (cells, unpacked + normalized)", n=0))


def hunks(diff_text):
    """[[lines of one hunk]] of a unified diff (header lines dropped)."""
    out = []
    for ln in diff_text.splitlines():
        if ln.startswith("@@"):
            out.append([])
        elif out and ln[:1] in "+-":
            out[-1].append(ln)
    return out
