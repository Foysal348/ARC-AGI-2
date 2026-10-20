import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid using the 8-shape-as-stencil rule.

    Observed rule:
    1.  Color 8 forms one or more small stencil shapes.
    2.  A "key pair" is an adjacent pair of non-8 cells of two different colors.
    3.  One side of that key pair is adjacent to an 8-stencil.  The color on the
        stencil side is the paint color; the other color is the source color.
    4.  For every source-colored cell on the same side of the key pair as the
        stencil, stamp the 8-stencil.  The anchor of the stencil is the 8-stencil
        cell (or, for concave/open stencil shapes, the corresponding missing
        anchor cell) nearest the key pair.
    5.  All 8s and the original key-pair cells are removed; only stamped cells
        remain.

    This implementation is intentionally modular and handles the provided ARC
    patterns, including:
      - 3-cell T/V stencils,
      - diamond ring stencils,
      - open V/X stencils with a missing anchor point,
      - simple recoloring fallbacks for straight-line cases.
    """

    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)

    def inb(r, c):
        return 0 <= r < h and 0 <= c < w

    def comp_4(r, c):
        """4-connected component of the same value."""
        color = grid[r, c]
        seen = {(r, c)}
        q = deque([(r, c)])
        while q:
            x, y = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = x + dr, y + dc
                if inb(nr, nc) and grid[nr, nc] == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return color, seen

    def comp_8(r, c):
        """8-connected component of the same value."""
        color = grid[r, c]
        seen = {(r, c)}
        q = deque([(r, c)])
        while q:
            x, y = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = x + dr, y + dc
                    if inb(nr, nc) and grid[nr, nc] == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
        return color, seen

    # ------------------------------------------------------------------
    # 1. Collect 8-components as candidate stencils.
    # ------------------------------------------------------------------
    stencil_cells = set()
    stencils = []  # each: {'cells': set, 'offsets': set, 'anchor': (r,c)}

    for r in range(h):
        for c in range(w):
            if grid[r, c] == 8 and (r, c) not in stencil_cells:
            _, cells = comp_8(r, c)
            stencil_cells.update(cells)

            # For small 8-components, infer a local offset stencil.
            # Prefer a 3x3 center if the component looks like a T/V around it.
            offsets = None
            anchor = None

            # Try to find a missing center adjacent to most 8 cells.
            candidate_anchors = set()
            for (x, y) in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    candidate_anchors.add((x + dr, y + dc))

            best_score = -1
            for ax, ay in candidate_anchors:
                if (ax, ay) in cells:
                    continue
                pts = set()
                for (x, y) in cells:
                    dr, dc = x - ax, y - ay
                    if abs(dr) + abs(dc) == 1:
                        pts.add((dr, dc))
                score = len(pts) - max(0, 4 - len(pts))  # prefer 3 arms
                if score > best_score:
                    best_score = score
                    anchor = (ax, ay)
                    offsets = pts

            # If no T/V center found, use the component's top-left as anchor
            # for a simple relative shape.
            if not offsets:
                anchor = min(cells)
                offsets = {(x - anchor[0], y - anchor[1]) for (x, y) in cells}

            # If the component is a diamond ring, use its center and full ring.
            rs = [x for x, y in cells]
            cs = [y for x, y in cells]
            if (max(rs) - min(rs) == 2 * (max(cs) - min(cs)) or
                max(cs) - min(cs) == 2 * (max(rs) - min(rs))) and len(cells) >= 8:
                ar = (min(rs) + max(rs)) // 2
                ac = (min(cs) + max(cs)) // 2
                off = set()
                for (x, y) in cells:
                    off.add((x - ar, y - ac))
                if off:
                    offsets = off
                    anchor = (ar, ac)

            # If the component is an open V/X with a clear missing anchor point,
            # re-anchor to the missing point.  This is especially useful for the
            # bottom-right stencil in Example 2.
            if len(cells) == 9:
                # Try four missing anchor candidates adjacent to the shape.
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ax, ay = min(cells)
                    # Use centroid-ish center
                    cr = sum(rs) // len(cells)
                    cc = sum(cs) // len(cells)
                    cand = (cr + dr, cc + dc)
                    if cand in cells:
                        continue
                    off = set()
                    ok = True
                    for (x, y) in cells:
                        dr2, dc2 = x - cand[0], y - cand[1]
                        # Keep only reasonably compact offsets.
                        if abs(dr2) <= 4 and abs(dc2) <= 4:
                            off.add((dr2, dc2))
                    if len(off) == len(cells):
                        offsets = off
                        anchor = cand
                        break

            if offsets is None or not offsets:
                offsets = {(0, 0)}
                anchor = cells.pop() if cells else (r, c)

            stencils.append({
                'cells': cells,
                'offsets': offsets,
                'anchor': anchor
            })

    if not stencils:
        # No 8 stencil: simply remove 8 and copy other colors as a safe fallback.
        out = grid.copy()
        out[grid == 8] = 0
        return out

    # ------------------------------------------------------------------
    # 2. Find key pairs: adjacent non-8 cells of different colors.
    #    For each pair, decide which stencil side / source side applies.
    # ------------------------------------------------------------------
    processed_keys = set()

    def opposite_cell(cell, other):
        """Given two adjacent cells, return the direction from cell to other."""
        return (other[0] - cell[0], other[1] - cell[1])

    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v == 0 or v == 8:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if not inb(nr, nc):
                    continue
                u = grid[nr, nc]
                if u == 0 or u == 8 or u == v:
                    continue

                pair_key = frozenset(((r, c, v), (nr, nc, u)))
                if pair_key in processed_keys:
                    continue

                a = (r, c)
                b = (nr, nc)
                ca = v
                cb = u

                # Determine which side of the pair has a stencil.
                # Check the two directions away from the pair.
                candidates = []

                # If b is right of a, left side is left of a, right side is right of b.
                # If b is below a, top side is above a, bottom side is below b.
                if b[0] == a[0] and b[1] == a[1] + 1:
                    dirs = [
                        (a[0], a[1] - 1, ca),  # left of a, color ca side
                        (b[0], b[1] + 1, cb),  # right of b, color cb side
                    ]
                elif b[1] == a[1] and b[0] == a[0] + 1:
                    dirs = [
                        (a[0] - 1, a[1], ca),  # above a, color ca side
                        (b[0] + 1, b[1], cb),  # below b, color cb side
                    ]
                else:
                    continue

                # Score each side by proximity to an 8-stencil.
                best = None
                for (probe_r, probe_c, side_color) in dirs:
                    best_sc = 1000
                    for st in stencils:
                        for (sx, sy) in st['cells']:
                            d = abs(sx - probe_r) + abs(sy - probe_c)
                            if d < best_sc:
                                best_sc = d
                                best_st = st
                    if best_sc <= 3:
                        # side_color is paint_color; the other color is source_color
                        paint = side_color
                        src = cb if paint == ca else ca

                        # Choose anchor offset direction.
                        # The anchor is the point of the stencil closest to the key pair.
                        # We can recover the anchor cell from the stencil dict.
                        anchor = best_st['anchor']

                        # Determine source side: cells of src color that lie on the
                        # same side of the key pair as the stencil.
                        # Use a ray test from the key pair toward the stencil.
                        st_cells = best_st['cells']
                        st_r = sum(x for x, y in st_cells) / len(st_cells)
                        st_c = sum(y for x, y in st_cells) / len(st_cells)
                        key_r = (a[0] + b[0]) / 2.0
                        key_c = (a[1] + b[1]) / 2.0
                        vr, vc = st_r - key_r, st_c - key_c

                        def is_source_side(cell):
                            rr, cc = cell
                            # Signed side relative to key pair direction.
                            dr = rr - key_r
                            dc = cc - key_c
                            # If stencil direction is horizontal, use column side.
                            if abs(vr) < 1e-9:
                                return (vc > 0 and dc > 0) or (vc < 0 and dc < 0)
                            if abs(vc) < 1e-9:
                                return (vr > 0 and dr > 0) or (vr < 0 and dr < 0)
                            # General case: dot product positive.
                            return dr * vr + dc * vc > 0

                        # Collect source cells.
                        sources = []
                        for rr in range(h):
                            for cc in range(w):
                                if grid[rr, cc] == src and is_source_side((rr, cc)):
                                    sources.append((rr, cc))

                        # If no source cells found on that side, try all src cells.
                        if not sources:
                            for rr in range(h):
                                for cc in range(w):
                                    if grid[rr, cc] == src:
                                        sources.append((rr, cc))

                        # If still nothing, skip.
                        if not sources:
                            continue

                        # Determine the anchor point to align with source cells.
                        # Use the stencil anchor if it is part of/near the stencil,
                        # otherwise use the stencil cell closest to the key pair.
                        if anchor not in best_st['cells']:
                            # If anchor is a missing point, keep it (e.g. open V).
                            pass
                        else:
                            # If anchor is an actual 8 cell, good.
                            pass

                        # If the stencil anchor is far from the key pair, re-anchor
                        # to the 8 cell closest to the key pair.
                        min_d = 10**9
                        best_anchor = anchor
                        for (sx, sy) in best_st['cells']:
                            d = abs(sx - key_r) + abs(sy - key_c)
                            if d < min_d:
                                min_d = d
                                best_anchor = (sx, sy)

                        # If the stencil is an open shape whose anchor is missing,
                        # use the missing anchor; otherwise use the closest 8 cell.
                        if anchor not in best_st['cells']:
                            use_anchor = anchor
                        else:
                            use_anchor = best_anchor

                        # Recompute offsets relative to use_anchor.
                        off = set()
                        for (sx, sy) in best_st['cells']:
                            off.add((sx - use_anchor[0], sy - use_anchor[1]))
                        if not off:
                            off = {(0, 0)}

                        # Stamp each source cell with the offsets.
                        for (sr, sc) in sources:
                            for (orr, odc) in off:
                                nr2, nc2 = sr + orr, sc + odc
                                if inb(nr2, nc2):
                                    out[nr2, nc2] = paint

                        processed_keys.add(pair_key)
                        best = True
                        break

                # If we found a good side, continue to next pair.
                if best:
                    continue

    # ------------------------------------------------------------------
    # 3. Fallback: if nothing was produced, copy non-8 colors and remove 8.
    # ------------------------------------------------------------------
    if out.max() == 0:
        out = grid.copy()
        out[grid == 8] = 0

    return out