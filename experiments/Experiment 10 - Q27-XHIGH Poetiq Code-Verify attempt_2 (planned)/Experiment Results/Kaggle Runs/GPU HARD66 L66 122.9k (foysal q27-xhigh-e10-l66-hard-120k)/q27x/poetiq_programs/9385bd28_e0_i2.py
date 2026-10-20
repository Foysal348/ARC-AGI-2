import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    - The most frequent color is the background.
    - A legend/table is the lowest, leftmost non-background connected component.
      It is interpreted as a two-column table:
          left color = source color
          right color = target/fill color
    - For each source color:
        * Use its small non-legend connected components (size 2-4, inside 2x2).
        * If there are at least two such components, create their bounding box.
            - If target != background: fill the box with target.
            - If target == background: reserve the box as an empty locked region.
        * If there is exactly one component and it is a full 2x2 block:
            - If target != background: recolor that 2x2 block.
            - If target == background: reserve that 2x2 block.
        * A single non-2x2 component only preserves its cells; it does not create a box.
    - Smaller boxes are processed before larger boxes.
      Painted and reserved cells are locked, so later larger boxes cannot overwrite them.
    - Legend cells and preserved source cells are never overwritten.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    # Background color.
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    non_bg = g != bg
    if not non_bg.any():
        return g.copy()

    def components(mask: np.ndarray):
        """Return 4-connected components as lists of (r, c)."""
        visited = np.zeros((H, W), dtype=bool)
        comps = []

        for r in range(H):
            for c in range(W):
                if mask[r, c] and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    cells = []

                    while stack:
                        rr, cc = stack.pop()
                        cells.append((rr, cc))

                        if rr > 0 and mask[rr - 1, cc] and not visited[rr - 1, cc]:
                            visited[rr - 1, cc] = True
                            stack.append((rr - 1, cc))
                        if rr + 1 < H and mask[rr + 1, cc] and not visited[rr + 1, cc]:
                            visited[rr + 1, cc] = True
                            stack.append((rr + 1, cc))
                        if cc > 0 and mask[rr, cc - 1] and not visited[rr, cc - 1]:
                            visited[rr, cc - 1] = True
                            stack.append((rr, cc - 1))
                        if cc + 1 < W and mask[rr, cc + 1] and not visited[rr, cc + 1]:
                            visited[rr, cc + 1] = True
                            stack.append((rr, cc + 1))

                    comps.append(cells)

        return comps

    def bbox(cells):
        """Bounding box: (rmin, rmax, cmin, cmax)."""
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    # ------------------------------------------------------------------
    # Locate the legend/table.
    # Choose the non-background component that reaches lowest, then leftmost.
    # ------------------------------------------------------------------
    non_bg_comps = components(non_bg)
    if not non_bg_comps:
        return g.copy()

    def legend_key(cells):
        rmin, rmax, cmin, cmax = bbox(cells)
        # Prefer lowest bottom edge, then leftmost, then higher top, then larger.
        return (-rmax, cmin, -rmin, -len(cells))

    legend_cells = min(non_bg_comps, key=legend_key)
    l_rmin, l_rmax, l_cmin, l_cmax = bbox(legend_cells)

    left_col = l_cmin
    right_col = l_cmin + 1 if l_cmin + 1 < W else l_cmin

    # Lock the whole rectangular legend strip.
    legend_mask = np.zeros((H, W), dtype=bool)
    legend_mask[l_rmin:l_rmax + 1, left_col:right_col + 1] = True

    # ------------------------------------------------------------------
    # Read source -> target mappings from the legend strip.
    # ------------------------------------------------------------------
    mappings = {}
    legend_row = {}

    if right_col > left_col:
        for r in range(l_rmin, l_rmax + 1):
            src = int(g[r, left_col])
            tgt = int(g[r, right_col])

            if src == bg:
                continue

            # A 0 target is ignored unless 0 is actually the background.
            if tgt == 0 and bg != 0:
                continue

            if src not in mappings:
                mappings[src] = tgt
                legend_row[src] = r

    if not mappings:
        return g.copy()

    def source_objects(color: int):
        """
        Small source components of this color outside the legend.
        Valid objects are 2-4 cells and fit in a 2x2 box.
        """
        mask = (g == color) & (~legend_mask)
        comps = components(mask)
        objs = []

        for cells in comps:
            if len(cells) < 2 or len(cells) > 4:
                continue

            rmin, rmax, cmin, cmax = bbox(cells)
            if (rmax - rmin + 1) > 2 or (cmax - cmin + 1) > 2:
                continue

            objs.append(cells)

        return objs

    protected_source = set()
    ops = []

    # ------------------------------------------------------------------
    # Build operations for each legend source color.
    # ------------------------------------------------------------------
    for src, tgt in mappings.items():
        objs = source_objects(src)
        if not objs:
            continue

        if len(objs) >= 2:
            # Bounding box over all small objects of this source color.
            rmin = H
            rmax = -1
            cmin = W
            cmax = -1

            for cells in objs:
                a, b, c, d = bbox(cells)
                rmin = min(rmin, a)
                rmax = max(rmax, b)
                cmin = min(cmin, c)
                cmax = max(cmax, d)

            rect = (rmin, rmax, cmin, cmax)
            area = (rmax - rmin + 1) * (cmax - cmin + 1)

            # Preserve all source cells of this color.
            for cells in objs:
                protected_source.update(cells)

            if tgt == bg:
                # Invisible box: reserve/lock it so larger boxes cannot fill it.
                ops.append({
                    "type": "reserve",
                    "rect": rect,
                    "area": area,
                    "target": None,
                    "source": src,
                    "legend_row": legend_row.get(src, 10**9)
                })
            else:
                # Visible box: fill with target color.
                ops.append({
                    "type": "fill",
                    "rect": rect,
                    "area": area,
                    "target": int(tgt),
                    "source": src,
                    "legend_row": legend_row.get(src, 10**9)
                })

        else:
            # Exactly one small component.
            cells = objs[0]
            rmin, rmax, cmin, cmax = bbox(cells)
            rect = (rmin, rmax, cmin, cmax)

            is_full_2x2 = (
                len(cells) == 4
                and (rmax - rmin + 1) == 2
                and (cmax - cmin + 1) == 2
            )

            if is_full_2x2:
                if tgt == bg:
                    # Invisible single 2x2 box.
                    protected_source.update(cells)
                    ops.append({
                        "type": "reserve",
                        "rect": rect,
                        "area": 4,
                        "target": None,
                        "source": src,
                        "legend_row": legend_row.get(src, 10**9)
                    })
                else:
                    # Recolor the single 2x2 block.
                    ops.append({
                        "type": "recolor",
                        "rect": rect,
                        "area": 4,
                        "target": int(tgt),
                        "source": src,
                        "legend_row": legend_row.get(src, 10**9)
                    })
            else:
                # Single non-2x2 object: keep it, but do not create a box.
                protected_source.update(cells)

    # ------------------------------------------------------------------
    # Locked cells:
    #   - legend
    #   - preserved source cells
    #   - cells painted/reserved by earlier smaller operations
    # ------------------------------------------------------------------
    protected_mask = np.zeros((H, W), dtype=bool)
    for r, c in protected_source:
        protected_mask[r, c] = True

    type_rank = {"recolor": 0, "fill": 1, "reserve": 2}

    # Smaller boxes first.
    ops.sort(key=lambda op: (
        op["area"],
        op["legend_row"],
        op["rect"][0],
        op["rect"][2],
        type_rank.get(op["type"], 10),
        op["source"]
    ))

    out = g.copy()
    locked = legend_mask | protected_mask

    # ------------------------------------------------------------------
    # Apply operations.
    # ------------------------------------------------------------------
    for op in ops:
        rmin, rmax, cmin, cmax = op["rect"]

        if op["type"] == "reserve":
            # Reserve the whole rectangle; do not change colors.
            locked[rmin:rmax + 1, cmin:cmax + 1] = True

        else:
            target = op["target"]
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if not locked[r, c]:
                        out[r, c] = target
                        locked[r, c] = True

    return out