import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The background color is the most frequent color.
    2. A 2-column "legend/key" is located in the lower-left area.
       It is the lowest non-background connected component, breaking ties
       by leftmost position. Each row of this key gives:
           left cell  = source color
           right cell = target/fill color
       A target of 0 is ignored unless 0 is the actual background color.
       A target equal to the background color means "protect the source
       cells, but do not paint a visible fill".

    3. For each legend source color, find small source objects outside the
       legend strip. A small object is a 4-connected component of size 2-4
       contained in a 2x2 box (L-trominoes, dominoes, or full 2x2 blocks).

    4. If a source color has at least two small objects:
         - protect all of those source cells;
         - if the target color is visible (not background), draw the
           bounding box of the objects and fill it with the target color.
       If it has exactly one full 2x2 object and the target is visible:
         - recolor that 2x2 block to the target color.
       Single non-2x2 objects are protected but not filled.

    5. Smaller visible boxes are processed before larger visible boxes.
       Once a visible fill paints a cell, that cell is "locked" so that
       larger boxes do not overwrite smaller boxes.  Legend cells and
       protected source cells are never overwritten.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    # Background color: most frequent cell value.
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    non_bg = g != bg
    if not non_bg.any():
        return g.copy()

    def components(mask):
        """Return 4-connected components as lists of (r, c)."""
        visited = np.zeros((H, W), dtype=bool)
        comps = []
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for r in range(H):
            for c in range(W):
                if mask[r, c] and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    cells = []
                    while stack:
                        r0, c0 = stack.pop()
                        cells.append((r0, c0))
                        for dr, dc in dirs:
                            nr, nc = r0 + dr, c0 + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    comps.append(cells)
        return comps

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    # ------------------------------------------------------------------
    # Locate the legend/key component.
    # Choose the non-background component that reaches lowest, then leftmost.
    # ------------------------------------------------------------------
    non_bg_comps = components(non_bg)

    def legend_priority(cells):
        rmin, rmax, cmin, cmax = bbox(cells)
        return (rmax, -cmin, -rmin)

    legend_cells = max(non_bg_comps, key=legend_priority)
    l_rmin, l_rmax, l_cmin, l_cmax = bbox(legend_cells)

    left_col = l_cmin
    if l_cmax >= l_cmin + 1:
        right_col = l_cmin + 1
    else:
        right_col = l_cmin

    # Legend strip that must remain unchanged.
    legend_strip = np.zeros((H, W), dtype=bool)
    legend_strip[l_rmin:l_rmax + 1, left_col:right_col + 1] = True

    # ------------------------------------------------------------------
    # Read source -> target pairs from the legend strip.
    # ------------------------------------------------------------------
    mappings = {}
    legend_row = {}

    if right_col > left_col:
        for r in range(l_rmin, l_rmax + 1):
            src = int(g[r, left_col])
            tgt = int(g[r, right_col])

            if src == bg:
                continue

            # A 0 target is invalid unless the real background is 0.
            if tgt == 0 and bg != 0:
                continue

            if src not in mappings:
                mappings[src] = tgt
                legend_row[src] = r

    # ------------------------------------------------------------------
    # Find small source objects for each mapping source color.
    # ------------------------------------------------------------------
    def small_components(color):
        mask = (g == color) & (~legend_strip)
        comps = components(mask)
        good = []

        for cells in comps:
            sz = len(cells)
            if sz < 2 or sz > 4:
                continue

            rmin, rmax, cmin, cmax = bbox(cells)
            if (rmax - rmin + 1) > 2 or (cmax - cmin + 1) > 2:
                continue

            good.append({
                "cells": set(cells),
                "rect": (rmin, rmax, cmin, cmax),
                "size": sz,
            })

        return good

    protected = legend_strip.copy()
    ops = []

    for src, tgt in mappings.items():
        comps = small_components(src)

        if len(comps) >= 2:
            preserve = set()
            rmin = min(comp["rect"][0] for comp in comps)
            rmax = max(comp["rect"][1] for comp in comps)
            cmin = min(comp["rect"][2] for comp in comps)
            cmax = max(comp["rect"][3] for comp in comps)

            for comp in comps:
                preserve.update(comp["cells"])

            # Protect source cells for all such mappings.
            for r, c in preserve:
                protected[r, c] = True

            # Visible fill only if target is not the background.
            if tgt != bg:
                ops.append({
                    "type": "rect",
                    "source": src,
                    "target": tgt,
                    "rect": (rmin, rmax, cmin, cmax),
                    "preserve": preserve,
                    "area": (rmax - rmin + 1) * (cmax - cmin + 1),
                    "legend_row": legend_row.get(src, 10**9),
                })

        elif len(comps) == 1:
            comp = comps[0]
            rmin, rmax, cmin, cmax = comp["rect"]
            is_full_2x2 = (
                comp["size"] == 4
                and (rmax - rmin + 1) == 2
                and (cmax - cmin + 1) == 2
            )

            if is_full_2x2 and tgt != bg:
                # A single full 2x2 source block is recolored to the target.
                ops.append({
                    "type": "recolor",
                    "source": src,
                    "target": tgt,
                    "rect": (rmin, rmax, cmin, cmax),
                    "area": 4,
                    "legend_row": legend_row.get(src, 10**9),
                })
            else:
                # No visible fill, but the object is still protected.
                for r, c in comp["cells"]:
                    protected[r, c] = True

    # Smaller visible boxes first.  For ties, rectangles before recolors,
    # then earlier legend entries first.
    ops.sort(key=lambda op: (
        op["area"],
        1 if op["type"] == "recolor" else 0,
        op["legend_row"],
        op["rect"][0],
        op["rect"][2],
        op.get("source", -1),
    ))

    # ------------------------------------------------------------------
    # Apply visible fills.
    # ------------------------------------------------------------------
    out = g.copy()
    filled = np.zeros((H, W), dtype=bool)

    for op in ops:
        rmin, rmax, cmin, cmax = op["rect"]

        if op["type"] == "rect":
            src = op["source"]
            tgt = op["target"]
            preserve = op["preserve"]

            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if legend_strip[r, c]:
                        continue

                    if protected[r, c]:
                        if (r, c) in preserve:
                            out[r, c] = src
                        continue

                    if filled[r, c]:
                        continue

                    out[r, c] = tgt
                    filled[r, c] = True

        else:  # recolor
            tgt = op["target"]
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if legend_strip[r, c] or protected[r, c] or filled[r, c]:
                        continue
                    out[r, c] = tgt
                    filled[r, c] = True

    return out