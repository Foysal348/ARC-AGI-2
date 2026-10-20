import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by completing rectangular regions defined by paired
    corner pieces.

    The rule inferred from the examples is:

    1. The most frequent color is the background.
    2. A small bottom-left legend contains horizontal pairs:
         source_color  fill_color
       For example, "2 3" means that rectangles whose corners are color 2
       are filled with color 3.
    3. A valid rectangular object is defined either by:
         - two L-shaped 3-cell corner pieces of the same source color, or
         - a single full 2x2 block of that source color (degenerate rectangle).
    4. Active rectangles are painted from largest area to smallest area.
       A painted rectangle:
         - fills its bounding box with its legend fill color,
         - then redraws its canonical corner pieces in the source color.
       Smaller rectangles therefore overwrite larger ones where they overlap.
    5. A rectangle whose legend fill is 0 is inactive: it is not painted,
       but if its whole bounding box lies inside a larger active rectangle,
       the larger rectangle leaves that inner box untouched.
    6. Legend cells are never overwritten.

    This implementation follows that behavior.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Background color: most frequent cell value.
    vals, counts = np.unique(grid, return_counts=True)
    B = int(vals[np.argmax(counts)])

    out = grid.copy()

    # -------------------------------------------------------------------------
    # Helper: canonical bracket cells for a corner orientation.
    # -------------------------------------------------------------------------
    def bracket_cells(rmin, rmax, cmin, cmax, orientation):
        if orientation == "TL":
            return [
                (rmin, cmin),
                (rmin, cmin + 1),
                (rmin + 1, cmin),
            ]
        if orientation == "TR":
            return [
                (rmin, cmax - 1),
                (rmin, cmax),
                (rmin + 1, cmax),
            ]
        if orientation == "BL":
            return [
                (rmax - 1, cmin),
                (rmax, cmin),
                (rmax, cmin + 1),
            ]
        if orientation == "BR":
            return [
                (rmax - 1, cmax - 1),
                (rmax, cmax - 1),
                (rmax, cmax),
            ]
        return []

    # -------------------------------------------------------------------------
    # Extract legend from the bottom-left 2-column strip.
    # The legend is a vertical run of rows where the left cell is non-background.
    # We prefer runs with multiple distinct source colors.
    # -------------------------------------------------------------------------
    legend_mask = np.zeros((H, W), dtype=bool)
    mapping = {}

    if H >= 2 and W >= 2:
        max_c = min(W - 2, max(0, W // 2))
        candidates = []

        for c in range(max_c + 1):
            rows = []
            for r in range(H // 2, H):
                if grid[r, c] != B:
                    rows.append(r)

            if not rows:
                continue

            start = end = rows[0]
            for r in rows[1:]:
                if r == end + 1:
                    end = r
                else:
                    length = end - start + 1
                    distinct = len({
                        int(grid[rr, c])
                        for rr in range(start, end + 1)
                        if grid[rr, c] != B
                    })
                    candidates.append((distinct, length, -c, start, c, start, end))
                    start = end = r

            length = end - start + 1
            distinct = len({
                int(grid[rr, c])
                for rr in range(start, end + 1)
                if grid[rr, c] != B
            })
            candidates.append((distinct, length, -c, start, c, start, end))

        if candidates:
            # Prefer a true multi-entry legend.
            good = [c for c in candidates if c[0] >= 2 and c[1] >= 2]
            if good:
                pool = good
            else:
                pool = [c for c in candidates if c[1] >= 2] or candidates

            best = max(pool, key=lambda x: (x[0], x[1], x[2], x[3]))
            _, _, _, _, legend_c, legend_r0, legend_r1 = best

            for r in range(legend_r0, legend_r1 + 1):
                x = int(grid[r, legend_c])
                y = int(grid[r, legend_c + 1])
                legend_mask[r, legend_c] = True
                legend_mask[r, legend_c + 1] = True
                if x != B:
                    mapping[x] = y

    if not mapping:
        return out

    # -------------------------------------------------------------------------
    # Find connected components of non-background, non-legend cells.
    # Keep L-trominoes (3 cells in a 2x2 box) and 2x2 blocks.
    # -------------------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    l_components = {}
    block_components = {}

    for r in range(H):
        for c in range(W):
            if visited[r, c] or legend_mask[r, c] or grid[r, c] == B:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H and 0 <= nc < W
                        and not visited[nr, nc]
                        and not legend_mask[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if len(cells) not in (3, 4):
                continue

            rmin = min(x[0] for x in cells)
            rmax = max(x[0] for x in cells)
            cmin = min(x[1] for x in cells)
            cmax = max(x[1] for x in cells)

            if rmax - rmin != 1 or cmax - cmin != 1:
                continue

            cell_set = set(cells)

            if len(cells) == 3:
                # Determine which cell of the 2x2 is missing.
                missing = None
                for rr in (rmin, rmax):
                    for cc in (cmin, cmax):
                        if (rr, cc) not in cell_set:
                            missing = (rr, cc)
                            break
                    if missing:
                        break

                if missing == (rmax, cmax):
                    orientation = "TL"
                elif missing == (rmax, cmin):
                    orientation = "TR"
                elif missing == (rmin, cmax):
                    orientation = "BL"
                elif missing == (rmin, cmin):
                    orientation = "BR"
                else:
                    continue

                l_components.setdefault(color, []).append({
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                    "orientation": orientation,
                })

            elif len(cells) == 4:
                block_components.setdefault(color, []).append({
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                })

    # -------------------------------------------------------------------------
    # Build rectangles from mapping entries.
    # -------------------------------------------------------------------------
    def choose_l_pair(components):
        """Choose the best pair of L components for one source color."""
        if len(components) == 2:
            return components[0], components[1]

        best_pair = None
        best_score = None

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                a = components[i]
                b = components[j]

                rmin = min(a["rmin"], b["rmin"])
                rmax = max(a["rmax"], b["rmax"])
                cmin = min(a["cmin"], b["cmin"])
                cmax = max(a["cmax"], b["cmax"])

                area = (rmax - rmin + 1) * (cmax - cmin + 1)
                opposite = (a["orientation"], b["orientation"]) in [
                    ("TL", "BR"), ("BR", "TL"),
                    ("TR", "BL"), ("BL", "TR"),
                ]

                score = (opposite, area)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (a, b)

        return best_pair

    rectangles = []

    for src, fill in mapping.items():
        src = int(src)
        fill = int(fill)
        if src == B:
            continue

        ls = l_components.get(src, [])
        blks = block_components.get(src, [])

        rect = None

        if len(ls) >= 2:
            a, b = choose_l_pair(ls)
            if a is not None and b is not None:
                rect = {
                    "src": src,
                    "fill": fill,
                    "rmin": min(a["rmin"], b["rmin"]),
                    "rmax": max(a["rmax"], b["rmax"]),
                    "cmin": min(a["cmin"], b["cmin"]),
                    "cmax": max(a["cmax"], b["cmax"]),
                    "orientations": [a["orientation"], b["orientation"]],
                    "degenerate": False,
                }

        elif len(blks) == 1:
            bl = blks[0]
            rect = {
                "src": src,
                "fill": fill,
                "rmin": bl["rmin"],
                "rmax": bl["rmax"],
                "cmin": bl["cmin"],
                "cmax": bl["cmax"],
                "orientations": [],
                "degenerate": True,
            }

        if rect is None:
            continue

        # Fill color 0 is treated as "no painting".
        rect["active"] = (fill != 0)
        rectangles.append(rect)

    if not rectangles:
        return out

    active_rects = [r for r in rectangles if r["active"]]
    inactive_rects = [r for r in rectangles if not r["active"]]

    # Paint larger rectangles first; smaller ones go on top.
    def rect_area(r):
        return (r["rmax"] - r["rmin"] + 1) * (r["cmax"] - r["cmin"] + 1)

    active_rects.sort(key=lambda r: (rect_area(r), -r["src"]), reverse=True)

    def contained(inner, outer):
        return (
            inner["rmin"] >= outer["rmin"]
            and inner["rmax"] <= outer["rmax"]
            and inner["cmin"] >= outer["cmin"]
            and inner["cmax"] <= outer["cmax"]
        )

    # -------------------------------------------------------------------------
    # Paint active rectangles.
    # -------------------------------------------------------------------------
    for rect in active_rects:
        rmin, rmax = rect["rmin"], rect["rmax"]
        cmin, cmax = rect["cmin"], rect["cmax"]

        if rmax < rmin or cmax < cmin:
            continue

        # Holes: inactive rectangles fully contained in this rectangle.
        hole = np.zeros((H, W), dtype=bool)
        for inner in inactive_rects:
            if contained(inner, rect):
                hole[inner["rmin"]:inner["rmax"] + 1,
                     inner["cmin"]:inner["cmax"] + 1] = True

        # Fill the bounding box.
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if not hole[r, c] and not legend_mask[r, c]:
                    out[r, c] = rect["fill"]

        # Redraw canonical source-color corner pieces.
        if not rect["degenerate"]:
            for orientation in rect["orientations"]:
                for rr, cc in bracket_cells(rmin, rmax, cmin, cmax, orientation):
                    if 0 <= rr < H and 0 <= cc < W and not legend_mask[rr, cc]:
                        out[rr, cc] = rect["src"]

    return out