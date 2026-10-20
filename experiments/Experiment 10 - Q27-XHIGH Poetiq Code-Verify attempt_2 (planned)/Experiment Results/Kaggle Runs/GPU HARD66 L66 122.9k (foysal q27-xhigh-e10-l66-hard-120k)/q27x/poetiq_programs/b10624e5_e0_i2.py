import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid divided by a full row/column of 1s into four quadrants.

    The quadrant with the richest set of non-background colors is treated as
    the template. The color-2 block in each quadrant is the anchor:
    - its centered square side length determines the scale,
    - its position determines placement,
    - quadrant position determines mirroring.

    Each target quadrant is cleared and redrawn from the template using the
    appropriate mirror symmetry and scaling.
    """
    if grid is None or grid.ndim != 2:
        return grid

    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    if H == 0 or W == 0 or H < 3 or W < 3:
        return grid

    # Background is the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Find full non-background rows/columns used as separators.
    full_rows = []
    for r in range(H):
        first = grid[r, 0]
        if first != bg and np.all(grid[r, :] == first):
            full_rows.append(r)

    full_cols = []
    for c in range(W):
        first = grid[0, c]
        if first != bg and np.all(grid[:, c] == first):
            full_cols.append(c)

    hr = full_rows[len(full_rows) // 2] if full_rows else H // 2
    vc = full_cols[len(full_cols) // 2] if full_cols else W // 2

    if hr <= 0 or hr >= H - 1 or vc <= 0 or vc >= W - 1:
        return grid

    quads = {
        "TL": (0, hr, 0, vc),
        "TR": (0, hr, vc + 1, W),
        "BL": (hr + 1, H, 0, vc),
        "BR": (hr + 1, H, vc + 1, W),
    }

    def round_frac(num, den):
        """Round num / den to nearest integer, halves away from zero."""
        num = int(num)
        den = int(den)
        if den == 0:
            return 0
        if den < 0:
            num = -num
            den = -den
        if num >= 0:
            return (2 * num + den) // (2 * den)
        else:
            return -((-2 * num + den) // (2 * den))

    def get_components(bounds):
        """
        4-connected components by color inside a quadrant.
        Each component is represented by its bounding box and color.
        """
        r0, r1, c0, c1 = bounds
        sub = grid[r0:r1, c0:c1]
        h, w = sub.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []

        for i in range(h):
            for j in range(w):
                if seen[i, j] or sub[i, j] == bg:
                    continue

                color = int(sub[i, j])
                stack = [(i, j)]
                seen[i, j] = True

                min_i = max_i = i
                min_j = max_j = j
                area = 0

                while stack:
                    x, y = stack.pop()
                    area += 1

                    if x < min_i:
                        min_i = x
                    if x > max_i:
                        max_i = x
                    if y < min_j:
                        min_j = y
                    if y > max_j:
                        max_j = y

                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if not seen[nx, ny] and sub[nx, ny] == color:
                                seen[nx, ny] = True
                                stack.append((nx, ny))

                comps.append({
                    "color": color,
                    "r0": r0 + min_i,
                    "c0": c0 + min_j,
                    "r1": r0 + max_i,
                    "c1": c0 + max_j,
                    "h": max_i - min_i + 1,
                    "w": max_j - min_j + 1,
                    "area": area,
                })

        return comps

    comps_by_quad = {q: get_components(quads[q]) for q in quads}

    # Choose template quadrant: most distinct colors, then most pixels.
    def quad_score(q):
        comps = comps_by_quad[q]
        colors = {c["color"] for c in comps}
        area = sum(c["area"] for c in comps)
        return (len(colors), area, len(comps))

    source = max(quads, key=quad_score)
    source_comps = comps_by_quad[source]

    if not source_comps:
        return grid

    # Determine anchor color. Usually color 2.
    presence = {}
    for q in quads:
        for color in {c["color"] for c in comps_by_quad[q]}:
            presence[color] = presence.get(color, 0) + 1

    def area_in(q, color):
        return sum(c["area"] for c in comps_by_quad[q] if c["color"] == color)

    anchor_color = None
    if 2 in presence and any(c["color"] == 2 for c in source_comps):
        anchor_color = 2
    else:
        multi = [(c, p) for c, p in presence.items() if p > 1]
        if multi:
            anchor_color = max(
                multi,
                key=lambda cp: (cp[1], area_in(source, cp[0]))
            )[0]
        elif presence:
            anchor_color = max(presence, key=lambda c: area_in(source, c))

    anchor_candidates = [c for c in source_comps if c["color"] == anchor_color]
    if anchor_candidates:
        source_anchor = max(anchor_candidates, key=lambda c: c["area"])
    else:
        source_anchor = max(source_comps, key=lambda c: c["area"])
        anchor_color = source_anchor["color"]

    source_anchor_side = max(source_anchor["h"], source_anchor["w"])
    if source_anchor_side <= 0:
        return grid

    out = grid.copy()

    def draw_box(r0, c0, h, w, color, bounds):
        """Draw a rectangle clipped to quadrant bounds."""
        R0, R1, C0, C1 = bounds
        rs = max(R0, r0)
        cs = max(C0, c0)
        re = min(R1, r0 + h)
        ce = min(C1, c0 + w)
        if rs < re and cs < ce:
            out[rs:re, cs:ce] = color

    def make_square_anchor(comp, bounds):
        """
        Use the anchor component's centered square bounding box.
        If the anchor bounding box is not square, side length is max(h, w).
        """
        R0, R1, C0, C1 = bounds
        r0, c0, r1, c1 = comp["r0"], comp["c0"], comp["r1"], comp["c1"]

        side = max(r1 - r0 + 1, c1 - c0 + 1)
        max_side = min(R1 - R0, C1 - C0)
        if side > max_side:
            side = max(1, max_side)

        center_r2 = r0 + r1
        center_c2 = c0 + c1

        sr0 = round_frac(center_r2 - (side - 1), 2)
        sc0 = round_frac(center_c2 - (side - 1), 2)

        # Clamp inside the quadrant.
        if sr0 < R0:
            sr0 = R0
        if sr0 + side > R1:
            sr0 = R1 - side
        if sc0 < C0:
            sc0 = C0
        if sc0 + side > C1:
            sc0 = C1 - side

        return sr0, sc0, sr0 + side - 1, sc0 + side - 1, side

    for q in quads:
        if q == source:
            continue

        target_comps = comps_by_quad[q]
        target_anchor_candidates = [
            c for c in target_comps if c["color"] == anchor_color
        ]

        if target_anchor_candidates:
            target_anchor = max(
                target_anchor_candidates,
                key=lambda c: c["area"]
            )
        elif target_comps:
            target_anchor = max(target_comps, key=lambda c: c["area"])
        else:
            continue

        bounds = quads[q]

        # Clear target quadrant.
        r0, r1, c0, c1 = bounds
        out[r0:r1, c0:c1] = bg

        ta_r0, ta_c0, ta_r1, ta_c1, ta_side = make_square_anchor(target_anchor, bounds)
        if ta_side <= 0:
            continue

        sa_r2 = source_anchor["r0"] + source_anchor["r1"]
        sa_c2 = source_anchor["c0"] + source_anchor["c1"]

        row_flip = (source[0] != q[0])  # T vs B
        col_flip = (source[1] != q[1])  # L vs R
        fr = -1 if row_flip else 1
        fc = -1 if col_flip else 1

        draw_items = []

        for comp in source_comps:
            if comp is source_anchor:
                continue

            h = comp["h"]
            w = comp["w"]

            ph = max(1, round_frac(h * ta_side, source_anchor_side))
            pw = max(1, round_frac(w * ta_side, source_anchor_side))

            # Twice center coordinates.
            dr2 = (comp["r0"] + comp["r1"]) - sa_r2
            dc2 = (comp["c0"] + comp["c1"]) - sa_c2

            dr2 = fr * dr2
            dc2 = fc * dc2

            scaled_dr2 = round_frac(dr2 * ta_side, source_anchor_side)
            scaled_dc2 = round_frac(dc2 * ta_side, source_anchor_side)

            target_center_r2 = (ta_r0 + ta_r1) + scaled_dr2
            target_center_c2 = (ta_c0 + ta_c1) + scaled_dc2

            rr0 = round_frac(target_center_r2 - (ph - 1), 2)
            cc0 = round_frac(target_center_c2 - (pw - 1), 2)

            draw_items.append((rr0, cc0, ph, pw, comp["color"]))

        # Draw the target anchor square.
        anchor_draw_color = anchor_color
        if target_anchor["color"] != anchor_color:
            anchor_draw_color = target_anchor["color"]

        draw_items.append((ta_r0, ta_c0, ta_side, ta_side, anchor_draw_color))

        for rr0, cc0, hh, ww, color in draw_items:
            draw_box(rr0, cc0, hh, ww, color, bounds)

    return out