import numpy as np
from collections import defaultdict, deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid containing a 2-column legend and colored L/square
    pieces.

    Inferred rule
    ------------
    1. The background color is the most frequent color in the grid.
    2. There is a vertical legend block of width 2. Each row contains a pair
       (source_color, fill_color).  A fill value of 0 means "use background".
    3. For every source color in the legend:
         - collect its non-legend connected components.
         - For an L-tromino component (3 cells in a 2x2 box), use its vertical
           domino as the anchor cells, but keep the whole L visible when it is
           the highest-priority region covering those cells.
         - For a solid 2x2 square component, use all four cells as anchors and
           do not preserve the original square; it is recolored by its fill.
         - The fill region for that source color is the bounding box of all its
           anchor cells.
    4. If fill regions overlap, the smaller source color has higher priority.
       The chosen region fills the cell with its mapped fill color, except that
       preserved L cells of that same source color remain unchanged.
    5. The legend itself is always preserved exactly as in the input.
    """

    if grid is None or grid.size == 0:
        return grid

    H, W = grid.shape
    original = grid.copy()

    # ---------------------------------------------------------------
    # 1. Background color
    # ---------------------------------------------------------------
    vals, counts = np.unique(original, return_counts=True)
    if vals.size == 0:
        return original

    max_count = counts.max()
    # Tie-break by smallest color for stability.
    bg = int(vals[counts == max_count].min())

    # ---------------------------------------------------------------
    # 2. Detect the legend block.
    #
    # The legend is a 2-column vertical block.  In the left column the source
    # colors are all distinct.  We take maximal non-background vertical runs
    # in the left column and score candidates by:
    #   distinct left values, height, number of non-background right cells,
    #   prefer lower / more-left blocks when tied.
    # ---------------------------------------------------------------
    best_legend = None

    for c in range(W - 1):
        r = 0
        while r < H:
            if original[r, c] == bg:
                r += 1
                continue

            r0 = r
            while r < H and original[r, c] != bg:
                r += 1
            r1 = r - 1

            height = r1 - r0 + 1
            if height < 2:
                continue

            left_vals = [int(original[i, c]) for i in range(r0, r1 + 1)]
            if len(set(left_vals)) != height:
                continue

            right_nonbg = sum(
                1 for i in range(r0, r1 + 1) if int(original[i, c + 1]) != bg
            )

            score = (
                True,                  # distinct
                height,
                right_nonbg,
                -c,                    # prefer left
                H - r1,                # prefer lower
            )

            if best_legend is None or score > best_legend[0]:
                best_legend = (score, r0, r1, c)

    legend_mask = np.zeros((H, W), dtype=bool)
    mapping = {}

    if best_legend is not None:
        _, lr0, lr1, lc = best_legend
        legend_mask[lr0:lr1 + 1, lc:lc + 2] = True

        for r in range(lr0, lr1 + 1):
            src = int(original[r, lc])
            fill = int(original[r, lc + 1])
            mapping[src] = fill

    # If no proper legend was found, do nothing.
    if not mapping:
        return original

    # ---------------------------------------------------------------
    # 3. Connected components of non-background, non-legend cells.
    # ---------------------------------------------------------------
    active = (original != bg) & (~legend_mask)
    visited = np.zeros((H, W), dtype=bool)
    components_by_color = defaultdict(list)

    for r in range(H):
        for c in range(W):
            if not active[r, c] or visited[r, c]:
                continue

            color = int(original[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if active[nr, nc] and not visited[nr, nc]:
                            if int(original[nr, nc]) == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

            components_by_color[color].append(cells)

    # ---------------------------------------------------------------
    # 4. For each mapped source color, determine anchors and preserved cells.
    #
    # L-tromino:
    #   - anchors = the vertical domino inside the 2x2 L
    #   - preserve = all three L cells
    # Solid 2x2:
    #   - anchors = all four cells
    #   - preserve = none
    # ---------------------------------------------------------------
    def vertical_pair_in_l(cells):
        """Return the two vertically adjacent cells in a 2x2 L."""
        cell_set = set(cells)
        for r, c in cells:
            if (r + 1, c) in cell_set:
                return {(r, c), (r + 1, c)}
        return set(cells)

    preserve_masks = {}
    regions = []

    for src, fill_val in mapping.items():
        comps = components_by_color.get(src, [])
        anchors = set()
        preserve = set()

        for cells in comps:
            if not cells:
                continue

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            h = max(rows) - min(rows) + 1
            w = max(cols) - min(cols) + 1

            # L-tromino.
            if len(cells) == 3 and h == 2 and w == 2:
                anchors.update(vertical_pair_in_l(cells))
                preserve.update(cells)

            # Solid 2x2 square.
            elif len(cells) == 4 and h == 2 and w == 2:
                anchors.update(cells)
                # Do not preserve solid square cells; they are recolored.

            # Fallback for unusual small components:
            # use their cells as anchors, but do not preserve them.
            elif len(cells) <= 4:
                anchors.update(cells)

        if not anchors:
            continue

        rows = [r for r, _ in anchors]
        cols = [c for _, c in anchors]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        actual_fill = bg if fill_val == 0 else int(fill_val)

        regions.append({
            "src": int(src),
            "fill": actual_fill,
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
        })

        if preserve:
            mask = np.zeros((H, W), dtype=bool)
            for r, c in preserve:
                mask[r, c] = True
            preserve_masks[int(src)] = mask

    if not regions:
        return original

    # ---------------------------------------------------------------
    # 5. Paint the final grid by priority.
    #
    # Smaller source color = higher priority.
    # Preserve L cells only for the chosen (highest-priority) source.
    # ---------------------------------------------------------------
    out = original.copy()

    for r in range(H):
        for c in range(W):
            if legend_mask[r, c]:
                out[r, c] = original[r, c]
                continue

            chosen = None
            for reg in regions:
                if reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"]:
                    if chosen is None or reg["src"] < chosen["src"]:
                        chosen = reg

            if chosen is None:
                out[r, c] = original[r, c]
            else:
                src = chosen["src"]
                pm = preserve_masks.get(src)
                if pm is not None and pm[r, c] and int(original[r, c]) == src:
                    out[r, c] = original[r, c]
                else:
                    out[r, c] = chosen["fill"]

    # Ensure legend is untouched even if a region covered it.
    out[legend_mask] = original[legend_mask]

    return out