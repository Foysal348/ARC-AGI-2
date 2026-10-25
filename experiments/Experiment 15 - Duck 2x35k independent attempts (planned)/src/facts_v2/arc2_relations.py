"""Facts v2 (user 2026-10-25, lever #2 "less thinking"): deterministic RELATIONS between each train input and output, kept
only when they hold on EVERY train pair (with the same parameters). The current Duck facts (arc2_duck.arc2_facts) give raw
observations (shapes, colors, changed-cell counts, components); the model then spends 20-40k thinking tokens finding how
input and output relate. These detectors state such relations directly. Pure Python, no numpy (the Duck sandbox image).

relations(task) -> list of dicts {"kind", "text", "params"}; relations_text(task) -> the prompt block.
Each detector is also applicable to (test input, test output) so its precision can be measured against the solutions."""
from collections import Counter

Grid = list


# ------------------------------------------------------------------ grid basics
def shape(g):
    return len(g), len(g[0])


def modal(g):
    return Counter(v for row in g for v in row).most_common(1)[0][0]


def colors(g):
    return sorted({v for row in g for v in row})


def rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def flip_h(g):
    return [row[::-1] for row in g]


def flip_v(g):
    return g[::-1]


def transpose(g):
    return [list(r) for r in zip(*g)]


DIHEDRAL = {
    "identity": lambda g: [row[:] for row in g],
    "rot90": rot90,
    "rot180": lambda g: rot90(rot90(g)),
    "rot270": lambda g: rot90(rot90(rot90(g))),
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "anti_transpose": lambda g: rot90(rot90(transpose(g))),
}


def sub(g, r0, c0, h, w):
    return [row[c0:c0 + w] for row in g[r0:r0 + h]]


def components(g, ignore, conn=4):
    """same-color connected components of non-`ignore` cells: list of dict(color, cells(frozenset), bbox, size)"""
    H, W = shape(g)
    seen, out = set(), []
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if conn == 8 else [])
    for r in range(H):
        for c in range(W):
            if (r, c) in seen or g[r][c] == ignore:
                continue
            col, st, cells = g[r][c], [(r, c)], []
            seen.add((r, c))
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in nb:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen and g[ny][nx] == col:
                        seen.add((ny, nx))
                        st.append((ny, nx))
            ys, xs = [p[0] for p in cells], [p[1] for p in cells]
            out.append({"color": col, "cells": frozenset(cells), "size": len(cells),
                        "bbox": (min(ys), min(xs), max(ys), max(xs))})
    out.sort(key=lambda o: (-o["size"], o["bbox"]))
    return out


def norm_shape(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    return frozenset((p[0] - r0, p[1] - c0) for p in cells)


# ------------------------------------------------------------------ per-pair detectors (return params or None)
def d_same_shape(a, b):
    return {} if shape(a) == shape(b) else None


def d_scale(a, b):
    (ha, wa), (hb, wb) = shape(a), shape(b)
    if (ha, wa) == (hb, wb):
        return None
    if hb % ha == 0 and wb % wa == 0:
        return {"dir": "up", "fr": hb // ha, "fc": wb // wa}
    if ha % hb == 0 and wa % wb == 0:
        return {"dir": "down", "fr": ha // hb, "fc": wa // wb}
    return None


def d_dihedral(a, b):
    for k, f in DIHEDRAL.items():
        if k != "identity" and f(a) == b:
            return {"op": k}
    return None


def d_upscale_blocks(a, b):
    s = d_scale(a, b)
    if not s or s["dir"] != "up":
        return None
    fr, fc = s["fr"], s["fc"]
    if all(b[r][c] == a[r // fr][c // fc] for r in range(len(b)) for c in range(len(b[0]))):
        return {"fr": fr, "fc": fc}
    return None


def d_tiling(a, b):
    """b is a grid of tiles, each tile = some dihedral transform of a (the tile pattern recorded)"""
    s = d_scale(a, b)
    if not s or s["dir"] != "up":
        return None
    ha, wa = shape(a)
    pat = []
    for i in range(s["fr"]):
        row = []
        for j in range(s["fc"]):
            t = sub(b, i * ha, j * wa, ha, wa)
            k = next((k for k, f in DIHEDRAL.items() if f(a) == t), None)
            if k is None:
                return None
            row.append(k)
        pat.append(tuple(row))
    return {"tiles": tuple(pat)}


def d_crop(a, b):
    """b equals a subgrid of a: the positions (all of them, at most 3)"""
    (ha, wa), (hb, wb) = shape(a), shape(b)
    if (hb, wb) == (ha, wa) or hb > ha or wb > wa:
        return None
    pos = [(r, c) for r in range(ha - hb + 1) for c in range(wa - wb + 1) if sub(a, r, c, hb, wb) == b]
    if not pos:
        return None
    out = {"positions": pos[:3], "n": len(pos)}
    # is the crop window the bbox of a component? which rank / color
    m = modal(a)
    for conn in (4, 8):
        comps = components(a, m, conn)
        for rank, o in enumerate(comps):
            r0, c0, r1, c1 = o["bbox"]
            if (r1 - r0 + 1, c1 - c0 + 1) == (hb, wb) and (r0, c0) in pos:
                out.update(component_rank=rank, component_color=o["color"], conn=conn)
                return out
    # a multi-color object: the bbox of all non-modal cells
    cells = [(r, c) for r in range(ha) for c in range(wa) if a[r][c] != m]
    if cells:
        r0, c0 = min(p[0] for p in cells), min(p[1] for p in cells)
        r1, c1 = max(p[0] for p in cells), max(p[1] for p in cells)
        if (r1 - r0 + 1, c1 - c0 + 1) == (hb, wb) and (r0, c0) in pos:
            out["all_nonmodal_bbox"] = True
    return out


def d_colormap(a, b):
    """same shape and b[r][c] = f(a[r][c]) for one function f (the non-identity part recorded)"""
    if shape(a) != shape(b):
        return None
    f = {}
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            if f.setdefault(x, y) != y:
                return None
    ch = {k: v for k, v in f.items() if k != v}
    return {"map": ch} if ch else None


def d_change_scope(a, b):
    """same shape: which cells may change (only cells of the input's modal color = 'only modal cells painted'; or
    only cells of some colors) and which output colors appear in changed cells"""
    if shape(a) != shape(b):
        return None
    m = modal(a)
    chg = [(a[r][c], b[r][c]) for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c]]
    if not chg:
        return None
    src = sorted({x for x, _ in chg})
    dst = sorted({y for _, y in chg})
    kept = 1 - len(chg) / (len(a) * len(a[0]))
    return {"from": src, "to": dst, "only_modal_painted": src == [m], "only_to_modal_erased": dst == [m],
            "kept_frac": round(kept, 3)}


def d_out_symmetry(a, b):
    def sym(g):
        s = []
        if g == flip_h(g):
            s.append("left-right")
        if g == flip_v(g):
            s.append("top-bottom")
        if shape(g)[0] == shape(g)[1] and g == transpose(g):
            s.append("diagonal")
        if g == DIHEDRAL["rot180"](g):
            s.append("rot180")
        return s
    so, si = sym(b), sym(a)
    new = [x for x in so if x not in si]
    return {"symmetries": new} if new else None


def d_objects(a, b):
    """same shape: object-level change of non-modal components (4-conn): kept / moved (offset) / recolored / removed /
    added counts"""
    if shape(a) != shape(b):
        return None
    m = modal(a)
    ca, cb = components(a, m), components(b, modal(b))
    used, kept, moved, recol, offs = set(), 0, 0, 0, Counter()
    for o in ca:
        j = next((j for j, q in enumerate(cb) if j not in used and q["cells"] == o["cells"] and q["color"] == o["color"]), None)
        if j is not None:
            used.add(j)
            kept += 1
            continue
        j = next((j for j, q in enumerate(cb) if j not in used and q["cells"] == o["cells"]), None)
        if j is not None:
            used.add(j)
            recol += 1
            continue
        ns = norm_shape(o["cells"])
        j = next((j for j, q in enumerate(cb) if j not in used and q["color"] == o["color"]
                  and norm_shape(q["cells"]) == ns), None)
        if j is not None:
            used.add(j)
            moved += 1
            offs[(cb[j]["bbox"][0] - o["bbox"][0], cb[j]["bbox"][1] - o["bbox"][1])] += 1
    removed = len(ca) - kept - moved - recol
    added = len(cb) - len(used)
    return {"in": len(ca), "out": len(cb), "kept": kept, "moved": moved, "recolored": recol, "removed": removed,
            "added": added, "offsets": dict(offs.most_common(3))}


def d_out_const(a, b):
    return {"shape": shape(b), "colors": colors(b)}


PAIR = {"same_shape": d_same_shape, "scale": d_scale, "dihedral": d_dihedral, "upscale_blocks": d_upscale_blocks,
        "tiling": d_tiling, "crop": d_crop, "colormap": d_colormap, "change_scope": d_change_scope,
        "out_symmetry": d_out_symmetry, "objects": d_objects, "out_const": d_out_const}


# ------------------------------------------------------------------ consistency over all train pairs -> relations
def relations(task, pairs=None):
    pairs = pairs if pairs is not None else [(p["input"], p["output"]) for p in task["train"]]
    per = {k: [f(a, b) for a, b in pairs] for k, f in PAIR.items()}
    R = []

    def all_(k):
        return all(v is not None for v in per[k])

    def same(k, keys):
        vals = [tuple(str(v.get(x)) for x in keys) for v in per[k]]
        return len(set(vals)) == 1

    if all_("dihedral") and same("dihedral", ["op"]):
        op = per["dihedral"][0]["op"]
        R.append(dict(kind="dihedral", params={"op": op}, text=f"output = {op}(input) exactly, in every train pair"))
    if all_("upscale_blocks") and same("upscale_blocks", ["fr", "fc"]):
        v = per["upscale_blocks"][0]
        R.append(dict(kind="upscale_blocks", params=v, text=f"output = input with every cell enlarged to a "
                                                            f"{v['fr']}x{v['fc']} block, in every train pair"))
    elif all_("tiling") and same("tiling", ["tiles"]):
        v = per["tiling"][0]
        R.append(dict(kind="tiling", params=v, text=f"output = a {len(v['tiles'])}x{len(v['tiles'][0])} arrangement of "
                                                    f"transformed copies of the input, tile ops by row {list(map(list, v['tiles']))}"))
    if all_("scale") and same("scale", ["dir", "fr", "fc"]) and not R:
        v = per["scale"][0]
        R.append(dict(kind="scale", params=v, text=f"output size = input size {'x' if v['dir'] == 'up' else '/'} "
                                                   f"({v['fr']} rows, {v['fc']} cols) in every train pair"))
    if all_("crop"):
        vs = per["crop"]
        t = "output is an exact subgrid (crop) of the input in every train pair"
        params = {"positions": [v["positions"] for v in vs]}
        if all("component_rank" in v for v in vs) and len({(v["component_rank"], v["conn"]) for v in vs}) == 1:
            rk, cn = vs[0]["component_rank"], vs[0]["conn"]
            cols = {v["component_color"] for v in vs}
            t += (f"; the crop window is the bounding box of the {['largest', '2nd largest', '3rd largest'][rk] if rk < 3 else f'rank-{rk}'}"
                  f" {cn}-connected non-modal component" + (f" (color {cols.pop()})" if len(cols) == 1 else " (its color varies)"))
            params.update(component_rank=rk, conn=cn)
        elif all(v.get("all_nonmodal_bbox") for v in vs):
            t += "; the crop window is the bounding box of all non-modal cells"
            params["all_nonmodal_bbox"] = True
        elif all(v["n"] == 1 for v in vs):
            t += f"; crop positions (row, col): {[v['positions'][0] for v in vs]}"
        R.append(dict(kind="crop", params=params, text=t))
    if all_("colormap"):
        merged, ok = {}, True
        for v in per["colormap"]:
            for k, x in v["map"].items():
                if merged.setdefault(k, x) != x:
                    ok = False
        if ok:
            R.append(dict(kind="colormap", params={"map": merged},
                          text=f"same shape and a pure cell-wise color substitution {dict(sorted(merged.items()))} "
                               "(every other color unchanged), consistent across all train pairs"))
    if all_("same_shape") and not any(r["kind"] == "colormap" for r in R) and all_("change_scope"):
        vs = per["change_scope"]
        if all(v["only_modal_painted"] for v in vs):
            to = sorted({c for v in vs for c in v["to"]})
            R.append(dict(kind="paint_only", params={"to": to}, text=f"same shape; only cells of the input's modal color "
                          f"change (nothing else is touched); they become colors {to}; kept cells "
                          f"{min(v['kept_frac'] for v in vs):.0%}-{max(v['kept_frac'] for v in vs):.0%}"))
        elif all(v["only_to_modal_erased"] for v in vs):
            R.append(dict(kind="erase_only", params={}, text="same shape; changed cells only become the modal color "
                                                            "(cells are erased, nothing new is drawn)"))
        else:
            src = sorted({c for v in vs for c in v["from"]})
            dst = sorted({c for v in vs for c in v["to"]})
            R.append(dict(kind="change_scope", params={"from": src, "to": dst},
                          text=f"same shape; changed cells come from input colors {src} and become colors {dst}; kept "
                               f"cells {min(v['kept_frac'] for v in vs):.0%}-{max(v['kept_frac'] for v in vs):.0%}"))
    if all_("objects") and all_("same_shape"):
        vs = per["objects"]
        keys = ["kept", "moved", "recolored", "removed", "added"]
        tot = {k: sum(v[k] for v in vs) for k in keys}
        if tot["moved"] and not tot["added"] and not tot["removed"]:
            offs = Counter()
            for v in vs:
                offs.update(v["offsets"])
            R.append(dict(kind="objects_move", params={"offsets": dict(offs)},
                          text=f"objects keep shape and color and only move (offsets dy,dx seen {dict(offs.most_common(4))})"))
        elif tot["recolored"] and not tot["moved"] and not tot["added"] and not tot["removed"]:
            R.append(dict(kind="objects_recolor", params={}, text="no object moves, appears or disappears; some objects "
                                                                 "change color (same cells): decide which and to what"))
        elif tot["removed"] and not tot["added"] and not tot["moved"] and not tot["recolored"]:
            R.append(dict(kind="objects_remove", params={}, text="objects are only removed (the rest kept exactly): "
                                                                "decide which objects are deleted"))
    if all_("out_symmetry"):
        common = set(per["out_symmetry"][0]["symmetries"])
        for v in per["out_symmetry"][1:]:
            common &= set(v["symmetries"])
        if common:
            R.append(dict(kind="out_symmetry", params={"sym": sorted(common)},
                          text=f"every output is {', '.join(sorted(common))} symmetric while its input is not "
                               "(a completion / mirroring)"))
    oc = per["out_const"]
    if len({str(v["shape"]) for v in oc}) == 1 and not all_("same_shape"):
        R.append(dict(kind="out_shape_const", params={"shape": oc[0]["shape"]},
                      text=f"every output has the same size {oc[0]['shape'][0]}x{oc[0]['shape'][1]} whatever the input size"))
    return R


def relations_text(task):
    R = relations(task)
    if not R:
        return ""
    return ("RELATIONS (computed; each holds on EVERY train pair; a strong hint, verify in Python):\n"
            + "\n".join(f"- {r['text']}" for r in R))
