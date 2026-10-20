import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the observed ARC transformation.

    Rule implemented:
      * The background is the most frequent color.
      * Choose an anchor color: a non-background color whose components form a
        rectangular "frame" with internal background holes, and which is not
        itself part of the moving multi-color pattern.
      * The moving pattern is all non-background cells not belonging to the
        anchor color.  It may be disconnected; it is treated as one colored
        rectangular pattern.
      * Try all 8 dihedral orientations of the moving pattern.  For each
        orientation, slide it over the grid and score placements by:
          - covering as many anchor-frame background holes as possible,
          - never covering anchor cells,
          - avoiding collision with other fixed non-background cells,
          - keeping the pattern close to the anchor frame,
          - preferring rotations to reflections when all else is equal.
      * If the selected orientation is a reflection, and the moving pattern
        contains exactly two colors, swap those two colors.  This handles the
        transposed/mirrored example where the two non-anchor colors exchange.
      * If no anchor frame can be found, return the input unchanged.
    """
    if grid is None or grid.size == 0:
        return np.array(grid, copy=True)

    g = np.array(grid, copy=True)
    H, W = g.shape

    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    if not np.any(g != bg):
        return g

    # Connected components of all non-background cells.
    comp_id = np.full((H, W), -1, dtype=int)
    components = []
    cid = 0
    for i in range(H):
        for j in range(W):
            if g[i, j] != bg and comp_id[i, j] == -1:
                q = deque([(i, j)])
                comp_id[i, j] = cid
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if g[nr, nc] != bg and comp_id[nr, nc] == -1:
                                comp_id[nr, nc] = cid
                                q.append((nr, nc))
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                colors = {int(g[r, c]) for r, c in cells}
                components.append({
                    "id": cid,
                    "cells": cells,
                    "cell_set": set(cells),
                    "colors": colors,
                    "r0": min(rs),
                    "r1": max(rs),
                    "c0": min(cs),
                    "c1": max(cs),
                })
                cid += 1

    if not components:
        return g

    # Group components of the same color when their bounding boxes overlap in
    # one axis and are separated by a gap, producing a rectangular frame with
    # internal holes.
    color_groups = {}
    for comp in components:
        col = next(iter(comp["colors"])) if len(comp["colors"]) == 1 else None
        if col is None:
            continue
        color_groups.setdefault(col, []).append(comp)

    def combined_bbox(comps):
        r0 = min(c["r0"] for c in comps)
        r1 = max(c["r1"] for c in comps)
        c0 = min(c["c0"] for c in comps)
        c1 = max(c["c1"] for c in comps)
        return r0, r1, c0, c1

    def occupied_in_bbox(comps, r0, r1, c0, c1):
        occ = set()
        for comp in comps:
            for r, c in comp["cell_set"]:
                if r0 <= r <= r1 and c0 <= c <= c1:
                    occ.add((r, c))
        return occ

    def other_colors_in_bbox(exclude_color, r0, r1, c0, c1):
        bad = 0
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                v = int(g[r, c])
                if v != bg and v != exclude_color:
                    bad += 1
        return bad

    def holes_for_color(color, comps):
        r0, r1, c0, c1 = combined_bbox(comps)
        occ = occupied_in_bbox(comps, r0, r1, c0, c1)
        holes = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if g[r, c] == bg and (r, c) not in occ:
                    holes.append((r, c))
        return r0, r1, c0, c1, set(holes)

    anchor_candidates = []
    for color, comps in color_groups.items():
        if len(comps) == 1:
            r0, r1, c0, c1 = comps[0]["r0"], comps[0]["r1"], comps[0]["c0"], comps[0]["c1"]
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            occ = occupied_in_bbox(comps, r0, r1, c0, c1)
            hole_set = set()
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if (r, c) not in occ:
                        hole_set.add((r, c))
            if len(hole_set) == 0:
                continue
            if other_colors_in_bbox(color, r0, r1, c0, c1) != 0:
                continue
            # Avoid choosing a tiny one-cell-hole accidental frame.
            if len(hole_set) < 2:
                continue
            anchor_candidates.append({
                "color": color,
                "comps": comps,
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "hole_set": hole_set,
                "anchor_set": occ,
            })
        else:
            # Try grouping all components of this color, and also each adjacent
            # pair that overlaps in one axis.
            candidate_groups = [comps]
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    a, b = comps[i], comps[j]
                    # Same vertical span or same horizontal span.
                    if (a["r0"] <= b["r1"] and b["r0"] <= a["r1"]) or \
                       (a["c0"] <= b["c1"] and b["c0"] <= a["c1"]):
                        candidate_groups.append([a, b])
            for grp in candidate_groups:
                r0, r1, c0, c1 = combined_bbox(grp)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if h < 3 or w < 3:
                    continue
                occ = occupied_in_bbox(grp, r0, r1, c0, c1)
                if len(occ) < 2:
                    continue
                hole_set = set()
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if g[r, c] == bg and (r, c) not in occ:
                            hole_set.add((r, c))
                if len(hole_set) == 0:
                    continue
                if other_colors_in_bbox(color, r0, r1, c0, c1) != 0:
                    continue
                if len(hole_set) < 2:
                    continue
                anchor_candidates.append({
                    "color": color,
                    "comps": grp,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "hole_set": hole_set,
                    "anchor_set": occ,
                })

    if not anchor_candidates:
        return g

    # Moving pattern = all non-background cells not in the chosen anchor.
    # For each anchor candidate, extract that pattern and attempt to fit it.
    def orientations(cells, colors, r0, c0, H0, W0):
        """Return 8 oriented lists of (r, c, color) relative to pattern bbox."""
        rel = [(r - r0, c - c0, colors[(r, c)]) for r, c in cells]
        outs = []
        # 0: identity (rotation)
        outs.append((H0, W0, [(r, c, v) for r, c, v in rel], False))
        # 1: rotate 90 CW (rotation)
        outs.append((W0, H0, [(c, H0 - 1 - r, v) for r, c, v in rel], False))
        # 2: rotate 180 (rotation)
        outs.append((H0, W0, [(H0 - 1 - r, W0 - 1 - c, v) for r, c, v in rel], False))
        # 3: rotate 90 CCW (rotation)
        outs.append((W0, H0, [(W0 - 1 - c, r, v) for r, c, v in rel], False))
        # 4: vertical flip (reflection)
        outs.append((H0, W0, [(H0 - 1 - r, c, v) for r, c, v in rel], True))
        # 5: horizontal flip (reflection)
        outs.append((H0, W0, [(r, W0 - 1 - c, v) for r, c, v in rel], True))
        # 6: transpose (reflection)
        outs.append((W0, H0, [(c, r, v) for r, c, v in rel], True))
        # 7: anti-transpose (reflection)
        outs.append((W0, H0, [(W0 - 1 - c, H0 - 1 - r, v) for r, c, v in rel], True))
        return outs

    best = None

    for anchor in anchor_candidates:
        acolor = anchor["color"]
        a_r0, a_r1, a_c0, a_c1 = anchor["r0"], anchor["r1"], anchor["c0"], anchor["c1"]
        holes = anchor["hole_set"]
        anchor_set = anchor["anchor_set"]

        # Cells in moving pattern.
        move_cells = []
        move_colors = {}
        for comp in components:
            for r, c in comp["cells"]:
                v = int(g[r, c])
                if v == bg or v == acolor:
                    continue
                move_cells.append((r, c))
                move_colors[(r, c)] = v

        if not move_cells:
            continue

        rs = [r for r, _ in move_cells]
        cs = [c for _, c in move_cells]
        m_r0, m_r1 = min(rs), max(rs)
        m_c0, m_c1 = min(cs), max(cs)
        mH = m_r1 - m_r0 + 1
        mW = m_c1 - m_c0 + 1

        # Fixed non-background cells that are not part of the moving pattern.
        # For this anchor, fixed includes anchor cells plus any other colors not
        # being moved (normally none).
        move_set = set(move_cells)
        fixed_set = set()
        for comp in components:
            for r, c in comp["cell_set"]:
                if (r, c) not in move_set:
                    fixed_set.add((r, c))

        hole_set = holes
        n_holes = len(hole_set)

        # Pattern centroid in original pattern coords.
        pr = sum(r - m_r0 for r, _ in move_cells) / len(move_cells)
        pc = sum(c - m_c0 for _, c in move_cells) / len(move_cells)

        for oH, oW, oriented, reflection in orientations(move_cells, move_colors, m_r0, m_c0, mH, mW):
            if oH > H or oW > W:
                continue

            # Limit translations to reasonable range around anchor.
            r_start = max(0, a_r0 - oH)
            r_end = min(H - oH, a_r1 + oH)
            c_start = max(0, a_c0 - oW)
            c_end = min(W - oW, a_c1 + oW)

            for tr in range(r_start, r_end + 1):
                for tc in range(c_start, c_end + 1):
                    covered_holes = 0
                    bad_anchor = 0
                    collision = 0
                    ok = True

                    placed = []
                    for rr, cc, v in oriented:
                        nr = tr + rr
                        nc = tc + cc
                        if not (0 <= nr < H and 0 <= nc < W):
                            ok = False
                            break
                        if (nr, nc) in anchor_set:
                            bad_anchor += 1
                        if (nr, nc) in hole_set:
                            covered_holes += 1
                        if (nr, nc) in fixed_set:
                            collision += 1
                        placed.append((nr, nc, v))

                    if not ok:
                        continue
                    if bad_anchor > 0:
                        continue
                    if collision > 0:
                        continue

                    # Must cover all holes of this anchor frame.
                    if covered_holes != n_holes:
                        continue

                    # Score: prefer more hole coverage (equal here), then
                    # smaller translation distance, then rotations over
                    # reflections, then deterministic tie-breaks.
                    key_pr = sum(nr - tr for nr, _, _ in placed) / len(placed)
                    key_pc = sum(nc - tc for _, nc, _ in placed) / len(placed)
                    dist_anchor = abs((a_r0 + a_r1) / 2.0 - (tr + oH / 2.0)) + \
                                  abs((a_c0 + a_c1) / 2.0 - (tc + oW / 2.0))
                    score = (
                        covered_holes,
                        -bad_anchor,
                        -collision,
                        -dist_anchor,
                        0 if not reflection else -1,
                        -abs(pr - key_pr) - abs(pc - key_pc),
                        tr,
                        tc,
                    )

                    cand = {
                        "score": score,
                        "anchor": anchor,
                        "tr": tr,
                        "tc": tc,
                        "oH": oH,
                        "oW": oW,
                        "placed": placed,
                        "reflection": reflection,
                        "move_colors": move_colors,
                    }

                    if best is None or score > best["score"]:
                        best = cand

    if best is None:
        # Fallback: leave input unchanged rather than produce nonsense.
        return g

    anchor = best["anchor"]
    tr, tc = best["tr"], best["tc"]
    reflection = best["reflection"]
    placed = best["placed"]

    out = np.array(g, copy=True)

    # Clear moving pattern cells.
    for r, c in set(move_cells):
        out[r, c] = bg

    # If the pattern was reflected and contains exactly two colors, swap them.
    if reflection:
        colors_in_move = sorted({v for _, _, v in placed})
        if len(colors_in_move) == 2:
            a, b = colors_in_move
            mp = {a: b, b: a}
            placed = [(r, c, mp[v]) for r, c, v in placed]

    # Place transformed moving pattern.
    for r, c, v in placed:
        out[r, c] = v

    return out