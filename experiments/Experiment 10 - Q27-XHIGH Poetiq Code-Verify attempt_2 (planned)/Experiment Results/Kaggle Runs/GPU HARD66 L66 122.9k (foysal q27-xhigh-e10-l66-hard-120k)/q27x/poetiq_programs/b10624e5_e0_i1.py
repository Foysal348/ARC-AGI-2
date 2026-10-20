import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by copying the multi-color template quadrant into the
    other three quadrants.

    Rule:
      1. The full row/column of 1s splits the grid into four quadrants.
      2. The quadrant with the most distinct non-background colors is treated
         as the source/template.
      3. In each other quadrant, the largest monochrome square/rectangle of the
         anchor color (usually color 2) is used as the placement anchor and
         determines the scale.
      4. Each monochrome connected component in the source quadrant is copied:
         - transformed by the appropriate symmetry (horizontal flip, vertical
           flip, or 180-degree rotation) depending on the target quadrant,
         - scaled relative to the anchor component,
         - positioned relative to the target anchor,
         - clipped to the target quadrant.
    """
    if grid is None or grid.ndim != 2:
        return grid

    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    # Background is the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Find the full non-background row/column used as the separator.
    full_rows = []
    for r in range(H):
        row = grid[r]
        if np.all(row == row[0]) and row[0] != bg:
            full_rows.append(r)

    full_cols = []
    for c in range(W):
        col = grid[:, c]
        if np.all(col == col[0]) and col[0] != bg:
            full_cols.append(c)

    if full_rows:
        hr = full_rows[len(full_rows) // 2]
    else:
        hr = H // 2

    if full_cols:
        vc = full_cols[len(full_cols) // 2]
    else:
        vc = W // 2

    if hr <= 0 or hr >= H - 1 or vc <= 0 or vc >= W - 1:
        return grid

    out = grid.copy()

    # Quadrants: (r0, r1, c0, c1), half-open.
    quads = {
        "TL": (0, hr, 0, vc),
        "TR": (0, hr, vc + 1, W),
        "BL": (hr + 1, H, 0, vc),
        "BR": (hr + 1, H, vc + 1, W),
    }

    def quadrant_score(q):
        r0, r1, c0, c1 = quads[q]
        sub = grid[r0:r1, c0:c1]
        colors = set(np.unique(sub).tolist())
        colors.discard(bg)
        nonbg = int(np.sum(sub != bg))
        # Prefer more colors, then more non-background pixels.
        return (len(colors), nonbg)

    # Choose the source/template quadrant.
    source = None
    best_score = None
    for q in ["TL", "TR", "BL", "BR"]:
        score = quadrant_score(q)
        if best_score is None or score > best_score:
            best_score = score
            source = q

    targets = [q for q in ["TL", "TR", "BL", "BR"] if q != source]

    def round_frac(num, den):
        """Round num/den to nearest integer, halves away from zero."""
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
        Return connected components by color inside bounds.
        Each component is:
            (color, r0, c0, r1, c1, area)
        with inclusive r0/r1/c0/c1 global coordinates.
        """
        r0, r1, c0, c1 = bounds
        sub = grid[r0:r1, c0:c1]
        h, w = sub.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []

        for r in range(h):
            for c in range(w):
                val = sub[r, c]
                if val == bg or seen[r, c]:
                    continue

                stack = [(r, c)]
                seen[r, c] = True
                min_r = max_r = r
                min_c = max_c = c
                area = 0

                while stack:
                    rr, cc = stack.pop()
                    area += 1
                    if rr < min_r:
                        min_r = rr
                    if rr > max_r:
                        max_r = rr
                    if cc < min_c:
                        min_c = cc
                    if cc > max_c:
                        max_c = cc

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not seen[nr, nc] and sub[nr, nc] == val:
                                seen[nr, nc] = True
                                stack.append((nr, nc))

                comps.append((
                    int(val),
                    r0 + min_r,
                    c0 + min_c,
                    r0 + max_r,
                    c0 + max_c,
                    area
                ))

        return comps

    source_comps = get_components(quads[source])
    if not source_comps:
        return out

    target_comps = {q: get_components(quads[q]) for q in targets}
    source_colors = {c[0] for c in source_comps}

    # Choose the anchor color. Prefer color 2 when it exists in both source
    # and at least one target quadrant.
    anchor_color = None
    if 2 in source_colors and any(
        any(comp[0] == 2 for comp in target_comps[q]) for q in targets
    ):
        anchor_color = 2
    else:
        best_area = 0
        for q in targets:
            for comp in target_comps[q]:
                if comp[0] in source_colors and comp[5] > best_area:
                    best_area = comp[5]
                    anchor_color = comp[0]

    if anchor_color is None:
        anchor_color = max(source_comps, key=lambda x: x[5])[0]

    # Source anchor: largest component of anchor color in source.
    anchor_source_candidates = [c for c in source_comps if c[0] == anchor_color]
    if anchor_source_candidates:
        source_anchor = max(anchor_source_candidates, key=lambda x: x[5])
    else:
        source_anchor = max(source_comps, key=lambda x: x[5])
        anchor_color = source_anchor[0]

    def draw_box(r, c, h, w, val, bounds):
        """Draw a rectangle, clipped to the given quadrant bounds."""
        r0, r1, c0, c1 = bounds
        rr0 = max(r0, r)
        rr1 = min(r1, r + h)
        cc0 = max(c0, c)
        cc1 = min(c1, c + w)
        if rr0 < rr1 and cc0 < cc1:
            out[rr0:rr1, cc0:cc1] = val

    # Process each target quadrant.
    for q in targets:
        bounds = quads[q]
        q_comps = target_comps[q]

        anchor_candidates = [c for c in q_comps if c[0] == anchor_color]
        if not anchor_candidates:
            # No usable anchor in this quadrant; leave it unchanged.
            continue

        target_anchor = max(anchor_candidates, key=lambda x: x[5])

        # Clear the target quadrant, then redraw the transformed template.
        r0, r1, c0, c1 = bounds
        out[r0:r1, c0:c1] = bg

        sa_r0, sa_c0, sa_r1, sa_c1, _ = source_anchor
        sa_h = sa_r1 - sa_r0 + 1
        sa_w = sa_c1 - sa_c0 + 1
        if sa_h <= 0 or sa_w <= 0:
            continue

        ta_r0, ta_c0, ta_r1, ta_c1, _ = target_anchor
        ta_h = ta_r1 - ta_r0 + 1
        ta_w = ta_c1 - ta_c0 + 1

        # Scale factors: target anchor size / source anchor size.
        sy_num, sy_den = ta_h, sa_h
        sx_num, sx_den = ta_w, sa_w

        # Twice-center coordinates avoid floating point.
        sa_r2 = sa_r0 + sa_r1
        sa_c2 = sa_c0 + sa_c1
        ta_r2 = ta_r0 + ta_r1
        ta_c2 = ta_c0 + ta_c1

        # Symmetry between source and target quadrant.
        row_flip = (source[0] != q[0])   # T <-> B
        col_flip = (source[1] != q[1])   # L <-> R
        fr = -1 if row_flip else 1
        fc = -1 if col_flip else 1

        draw_items = []

        for comp in source_comps:
            val, cr0, cc0, cr1, cc1, _ = comp

            # The source anchor itself is placed directly as the target anchor.
            if (
                val == anchor_color and
                cr0 == sa_r0 and cc0 == sa_c0 and
                cr1 == sa_r1 and cc1 == sa_c1
            ):
                continue

            ch = cr1 - cr0 + 1
            cw = cc1 - cc0 + 1

            ph = max(1, round_frac(ch * sy_num, sy_den))
            pw = max(1, round_frac(cw * sx_num, sx_den))

            # Offset between component center and source anchor center.
            dr2 = (cr0 + cr1) - sa_r2
            dc2 = (cc0 + cc1) - sa_c2

            dr2 = fr * dr2
            dc2 = fc * dc2

            scaled_dr2 = round_frac(dr2 * sy_num, sy_den)
            scaled_dc2 = round_frac(dc2 * sx_num, sx_den)

            target_center_r2 = ta_r2 + scaled_dr2
            target_center_c2 = ta_c2 + scaled_dc2

            r = round_frac(target_center_r2 - ph, 2)
            c = round_frac(target_center_c2 - pw, 2)

            draw_items.append((r, c, ph, pw, val))

        # Draw the anchor last so it is not overwritten by neighboring parts.
        draw_items.append((ta_r0, ta_c0, ta_h, ta_w, anchor_color))

        for r, c, h, w, val in draw_items:
            if h <= 0 or w <= 0:
                continue
            draw_box(r, c, h, w, val, bounds)

    return out