import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by assembling multi-color key components into single-color anchor
    components.

    The rule inferred from the examples:
      * The background is the most frequent color.
      * Connected non-background components are extracted.
      * Components with more than one color are treated as "keys".
      * Components with exactly one color are treated as potential "anchors".
      * An anchor has a bounding box. The cells in that bounding box that are
        background are considered its "hole".
      * A key is moved, using one of the 8 dihedral orientations, so that its
        cells completely cover the anchor's hole.  If possible, it should avoid
        overlapping the anchor and other components.  If some overlap with the
        anchor is necessary, it is minimized.
      * If the chosen key orientation is a reflection (not a pure rotation),
        and the key contains exactly two colors, those two colors are swapped.
        This matches the only reflected example where the two colors exchange.
    """

    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    R, C = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # If nothing to do.
    if not np.any(grid != bg):
        return grid

    # 4-connected components of non-background cells.
    comp_id = np.full((R, C), -1, dtype=int)
    components = []
    cid = 0
    for i in range(R):
        for j in range(C):
            if grid[i, j] != bg and comp_id[i, j] == -1:
                q = deque([(i, j)])
                comp_id[i, j] = cid
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            if grid[nr, nc] != bg and comp_id[nr, nc] == -1:
                                comp_id[nr, nc] = cid
                                q.append((nr, nc))
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                comps = {int(grid[r, c]) for r, c in cells}
                components.append({
                    "id": cid,
                    "cells": cells,
                    "cell_set": set(cells),
                    "colors": comps,
                    "r0": min(rs), "r1": max(rs),
                    "c0": min(cs), "c1": max(cs),
                    "single": len(comps) == 1,
                })
                cid += 1

    if not components:
        return grid

    key_components = [c for c in components if not c["single"]]
    anchor_components = [c for c in components if c["single"]]

    if not key_components or not anchor_components:
        return grid

    # All non-background cells, used for collision checks.
    all_nonbg = set()
    for c in components:
        all_nonbg.update(c["cell_set"])

    def orient_component(comp, mode):
        """Return oriented colored cells relative to a new bbox top-left."""
        r0, c0 = comp["r0"], comp["c0"]
        H = comp["r1"] - comp["r0"] + 1
        W = comp["c1"] - comp["c0"] + 1

        rel = []
        for r, c in comp["cells"]:
            rel.append((r - r0, c - c0, int(grid[r, c])))

        out = []
        if mode == 0:      # identity, rotation
            h, w = H, W
            for r, c, col in rel:
                out.append((r, c, col))
        elif mode == 1:    # rotate 90 clockwise, rotation
            h, w = W, H
            for r, c, col in rel:
                out.append((c, H - 1 - r, col))
        elif mode == 2:    # rotate 180, rotation
            h, w = H, W
            for r, c, col in rel:
                out.append((H - 1 - r, W - 1 - c, col))
        elif mode == 3:    # rotate 90 counterclockwise, rotation
            h, w = W, H
            for r, c, col in rel:
                out.append((W - 1 - c, r, col))
        elif mode == 4:    # vertical flip, reflection
            h, w = H, W
            for r, c, col in rel:
                out.append((H - 1 - r, c, col))
        elif mode == 5:    # horizontal flip, reflection
            h, w = H, W
            for r, c, col in rel:
                out.append((r, W - 1 - c, col))
        elif mode == 6:    # transpose, reflection
            h, w = W, H
            for r, c, col in rel:
                out.append((c, r, col))
        else:              # anti-transpose, reflection
            h, w = W, H
            for r, c, col in rel:
                out.append((W - 1 - c, H - 1 - r, col))

        reflection = mode in (4, 5, 6, 7)
        if reflection and len(comp["colors"]) == 2:
            a, b = sorted(comp["colors"])
            mp = {a: b, b: a}
            out = [(rr, cc, mp[col]) for rr, cc, col in out]

        return h, w, out, reflection

    candidates = []

    for key in key_components:
        key_set = key["cell_set"]
        for anchor in anchor_components:
            if anchor["id"] == key["id"]:
                continue

            # Hole cells: background cells inside anchor bbox.
            anchor_set = anchor["cell_set"]
            hole_cells = []
            for rr in range(anchor["r0"], anchor["r1"] + 1):
                for cc in range(anchor["c0"], anchor["c1"] + 1):
                    if grid[rr, cc] == bg and (rr, cc) not in anchor_set:
                        hole_cells.append((rr, cc))

            # Ignore trivial anchors with fewer than two holes.
            if len(hole_cells) < 2:
                continue

            hole_set = set(hole_cells)

            # Other occupied cells in anchor bbox are not acceptable to cover.
            other_in_anchor = set()
            for rr in range(anchor["r0"], anchor["r1"] + 1):
                for cc in range(anchor["c0"], anchor["c1"] + 1):
                    if (rr, cc) not in anchor_set and (rr, cc) in all_nonbg:
                        other_in_anchor.add((rr, cc))

            # For each orientation and translation.
            for mode in range(8):
                h, w, oriented, reflection = orient_component(key, mode)
                if h > R or w > C:
                    continue

                for tr in range(0, R - h + 1):
                    for tc in range(0, C - w + 1):
                        # Bbox must overlap anchor bbox.
                        if max(tr, anchor["r0"]) > min(tr + h - 1, anchor["r1"]):
                            continue
                        if max(tc, anchor["c0"]) > min(tc + w - 1, anchor["c1"]):
                            continue

                        good = 0
                        bad_anchor = 0
                        bad_other = 0
                        collision = 0
                        outside_row_sum = 0
                        outside_cnt = 0

                        placed = []
                        ok = True

                        for rr, cc, col in oriented:
                            nr = tr + rr
                            nc = tc + cc

                            if not (0 <= nr < R and 0 <= nc < C):
                                ok = False
                                break

                            if anchor["r0"] <= nr <= anchor["r1"] and anchor["c0"] <= nc <= anchor["c1"]:
                                if (nr, nc) in anchor_set:
                                    bad_anchor += 1
                                elif (nr, nc) in hole_set:
                                    good += 1
                                elif (nr, nc) in other_in_anchor:
                                    bad_other += 1
                            else:
                                if (nr, nc) in all_nonbg and (nr, nc) not in key_set:
                                    collision += 1
                                outside_row_sum += nr
                                outside_cnt += 1

                            placed.append((nr, nc, col))

                        if not ok:
                            continue

                        # Must cover the whole hole.
                        if good != len(hole_cells):
                            continue

                        # Do not cover unrelated occupied cells.
                        if bad_other > 0:
                            continue

                        # Prefer no collisions with stationary pieces.
                        if collision > 0:
                            continue

                        outside_centroid = outside_row_sum / max(1, outside_cnt)
                        key_r = sum(p[0] for p in placed) / len(placed)
                        key_c = sum(p[1] for p in placed) / len(placed)
                        anc_r = (anchor["r0"] + anchor["r1"]) / 2.0
                        anc_c = (anchor["c0"] + anchor["c1"]) / 2.0
                        dist = abs(key_r - anc_r) + abs(key_c - anc_c)

                        # Higher is better.
                        score = (
                            good,
                            -bad_anchor,
                            -collision,
                            -bad_other,
                            outside_centroid,
                            -dist,
                            tr,   # lower placement breaks remaining ties, matching example 1
                            -mode
                        )

                        candidates.append({
                            "score": score,
                            "key_id": key["id"],
                            "anchor_id": anchor["id"],
                            "placed": placed,
                            "reflection": reflection
                        })

    # Sort candidates by quality.
    candidates.sort(key=lambda x: x["score"], reverse=True)

    used_keys = set()
    used_anchors = set()
    final_moves = []
    final_cells = {}
    conflicts = set()

    for cand in candidates:
        k = cand["key_id"]
        a = cand["anchor_id"]
        if k in used_keys or a in used_anchors:
            continue

        # Avoid final placement overlapping another already chosen key.
        ok = True
        for rr, cc, col in cand["placed"]:
            if (rr, cc) in conflicts:
                ok = False
                break
        if not ok:
            continue

        # Avoid colliding with components that will not move, except the anchor.
        for rr, cc, col in cand["placed"]:
            cid = comp_id[rr, cc]
            if cid == -1 or cid == k or cid == a:
                continue
            comp = components[cid]
            if comp["single"]:
                ok = False
                break
        if not ok:
            continue

        used_keys.add(k)
        used_anchors.add(a)
        final_moves.append(cand)
        for rr, cc, col in cand["placed"]:
            final_cells[(rr, cc)] = col
            conflicts.add((rr, cc))

    # Build output.
    out = np.full((R, C), bg, dtype=grid.dtype)

    # Put back all components except moved keys.
    for comp in components:
        if comp["id"] in used_keys:
            continue
        for r, c in comp["cells"]:
            out[r, c] = grid[r, c]

    # Place moved keys.
    for cand in final_moves:
        for r, c, col in cand["placed"]:
            out[r, c] = col

    return out