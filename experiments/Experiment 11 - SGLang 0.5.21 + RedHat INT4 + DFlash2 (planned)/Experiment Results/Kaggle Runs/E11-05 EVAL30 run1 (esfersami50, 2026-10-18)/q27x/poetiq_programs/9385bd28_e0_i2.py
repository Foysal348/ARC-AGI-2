import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the corresponding output grid.

    Inferred rule
    ------------
    The grid contains:
      * a background color (the most frequent color);
      * a small vertical 2-column "legend" block.  Each row of the legend is
        (source_color, fill_color).  A fill_color of 0 means "paint with the
        background color".
      * one or more colors that appear exactly as two L-tromino pieces
        (3 cells in a 2x2 bounding box).  For every such source color that is
        present in the legend, the rectangle spanning the two L-pieces is
        filled with its mapped fill color.

    If multiple filled rectangles overlap, the rectangle whose source color is
    smaller has higher priority.  The original cells of the higher-priority
    L-pieces are preserved.  The legend itself is preserved unchanged.
    """

    if grid is None or grid.size == 0:
        return grid

    H, W = grid.shape
    out = grid.copy()

    # ------------------------------------------------------------------
    # 1. Determine background color.
    # ------------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    if vals.size == 0:
        return out
    max_count = counts.max()
    # Tie-break by smallest color.
    bg = int(vals[(counts == max_count).nonzero()[0][0]])

    # ------------------------------------------------------------------
    # 2. Detect the legend block.
    #
    # The legend is a 2-column block, height >= 2, whose left column has
    # non-background entries.  In the provided examples the left-column
    # colors in a true legend are all distinct.  We score candidates by:
    #   - uniqueness of left-column colors
    #   - height
    #   - number of non-background right-column cells
    # ------------------------------------------------------------------
    legend_mask = np.zeros((H, W), dtype=bool)
    mapping = {}

    def score_candidate(r0, r1, c):
        """Score a candidate legend rectangle."""
        height = r1 - r0 + 1
        left_vals = [int(grid[r, c]) for r in range(r0, r1 + 1)]
        right_nonbg = sum(1 for r in range(r0, r1 + 1) if int(grid[r, c + 1]) != bg)

        # Primary: left values must be distinct for a valid mapping.
        distinct = len(set(left_vals)) == height
        # Secondary: longer, denser, more leftward/bottomward are preferred.
        score = (
            distinct,
            height,
            right_nonbg,
            -c,
            H - r1,
        )
        return score, distinct

    best = None
    for c in range(W - 1):
        r = 0
        while r < H:
            if grid[r, c] == bg:
                r += 1
                continue

            r0 = r
            while r < H and grid[r, c] != bg:
                r += 1
            r1 = r - 1
            height = r1 - r0 + 1

            if height >= 2:
                score, distinct = score_candidate(r0, r1, c)
                if distinct and (best is None or score > best[0]):
                    best = (score, r0, r1, c)

    if best is not None:
        _, lr0, lr1, lc = best
        legend_mask[lr0:lr1 + 1, lc:lc + 2] = True
        for r in range(lr0, lr1 + 1):
            src = int(grid[r, lc])
            fill = int(grid[r, lc + 1])
            # Later duplicate source entries are kept as-is; ARC legends here
            # are one-to-one, so this is only defensive.
            mapping[src] = fill

    # Fallback if no height>=2 legend was found: try height 1 only.
    if not mapping:
        best = None
        for c in range(W - 1):
            for r in range(H):
                if grid[r, c] != bg:
                    score, distinct = score_candidate(r, r, c)
                    if (best is None or score > best[0]):
                        best = (score, r, r, c)
        if best is not None:
            _, lr0, lr1, lc = best
            legend_mask[lr0:lr1 + 1, lc:lc + 2] = True
            for r in range(lr0, lr1 + 1):
                src = int(grid[r, lc])
                fill = int(grid[r, lc + 1])
                mapping[src] = fill

    # ------------------------------------------------------------------
    # 3. Find connected components of non-background, non-legend cells.
    # ------------------------------------------------------------------
    active = (grid != bg) & (~legend_mask)

    # 4-connectivity labels by color.
    unvisited = active.copy()
    components = []  # list of (color, cells_set)

    for r in range(H):
        for c in range(W):
            if not unvisited[r, c]:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            unvisited[r, c] = False
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if unvisited[nr, nc] and int(grid[nr, nc]) == color:
                            unvisited[nr, nc] = False
                            stack.append((nr, nc))

            components.append((color, cells))

    # ------------------------------------------------------------------
    # 4. Identify L-tromino components:
    #        size == 3 and bounding box == 2x2.
    # ------------------------------------------------------------------
    l_components_by_color = defaultdict(list)

    for color, cells in components:
        if len(cells) != 3:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if h == 2 and w == 2:
            l_components_by_color[color].append(set(cells))

    # ------------------------------------------------------------------
    # 5. Build regions for mapped colors that have exactly two L-pieces.
    # ------------------------------------------------------------------
    regions = []
    target_cell_mask = {}

    for color, comp_list in l_components_by_color.items():
        if len(comp_list) != 2:
            continue
        if color not in mapping:
            continue

        all_cells = set()
        for comp in comp_list:
            all_cells |= comp

        rows = [r for r, _ in all_cells]
        cols = [c for _, c in all_cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        fill_val = int(mapping[color])
        fill_color = bg if fill_val == 0 else fill_val

        mask = np.zeros((H, W), dtype=bool)
        for rr, cc in all_cells:
            mask[rr, cc] = True

        target_cell_mask[color] = mask
        regions.append({
            "color": color,
            "fill": fill_color,
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
        })

    if not regions:
        return out

    # Sort regions by priority: smaller source color has higher priority.
    regions.sort(key=lambda x: x["color"])

    # ------------------------------------------------------------------
    # 6. Paint the output by priority.
    #
    # For each cell, use the highest-priority region covering it, i.e. the
    # region with the smallest source color.  If the original cell is one of
    # that region's target L-cells, keep it.  Otherwise paint with the region
    # fill color.
    # ------------------------------------------------------------------
    for r in range(H):
        for c in range(W):
            if legend_mask[r, c]:
                out[r, c] = grid[r, c]
                continue

            chosen = None
            for reg in regions:
                if reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"]:
                    # regions are sorted ascending by source color, so first
                    # covering region is the highest-priority one.
                    chosen = reg
                    break

            if chosen is None:
                out[r, c] = grid[r, c]
            else:
                col = chosen["color"]
                if target_cell_mask[col][r, c]:
                    out[r, c] = col
                else:
                    out[r, c] = chosen["fill"]

    # Keep legend exactly as in input (also protects any legend cells that
    # might have been incidentally covered by a region).
    if mapping:
        out[legend_mask] = grid[legend_mask]

    return out