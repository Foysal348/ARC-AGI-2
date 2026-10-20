import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. The background color is the most common color in the input grid.
    2. Extract all non-background connected components (8-connected).
    3. Each component is moved either to the top edge or to the bottom edge:
         - Preserve its horizontal position.
         - Preserve its exact shape.
       If a component is placed at the top, its topmost row becomes row 0.
       If it is placed at the bottom, its bottommost row becomes the last row.
    4. The side is chosen using this heuristic:
         - Default: odd colors go to the top, even colors go to the bottom.
         - If that default would make two same-side components overlap in the
           horizontal direction, flip the lower-color component in each
           conflicting overlapping pair.
         - This matches the provided examples while also handling the common
           ARC-style constraint that objects must not overlap after packing.
    """
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # Determine background color as the most frequent color.
    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    out = np.full_like(grid, bg)

    # visited mask for component extraction
    visited = np.zeros((h, w), dtype=bool)

    components = []

    # 8-connected components of non-background pixels
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg and not visited[r, c]:
                color = int(grid[r, c])
                q = deque([(r, c)])
                visited[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if not visited[nr, nc] and grid[nr, nc] == color:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))

                rows = [x[0] for x in cells]
                cols = [x[1] for x in cells]
                rmin, rmax = min(rows), max(rows)
                cmin, cmax = min(cols), max(cols)

                # Store the normalized pattern
                pattern = []
                for rr in range(rmin, rmax + 1):
                    row = []
                    for cc in range(cmin, cmax + 1):
                        row.append(1 if (rr, cc) in set(cells) else 0)
                    pattern.append(row)

                components.append({
                    "color": color,
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                    "pattern": pattern,
                    "height": rmax - rmin + 1,
                    "width": cmax - cmin + 1,
                })

    if not components:
        return out

    # Default side by color parity: odd -> top, even -> bottom
    # Represent side as 0 for top, 1 for bottom.
    for comp in components:
        comp["side"] = 0 if (comp["color"] % 2 == 1) else 1

    # Resolve horizontal overlaps on the same side.
    # If two components overlap horizontally and are assigned the same side,
    # flip the one with the smaller color.
    n = len(components)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                a = components[i]
                b = components[j]
                if a["side"] != b["side"]:
                    continue
                # Horizontal overlap test
                if a["cmin"] <= b["cmax"] and b["cmin"] <= a["cmax"]:
                    # Flip the component with smaller color to the other side
                    if a["color"] < b["color"]:
                        a["side"] ^= 1
                    elif b["color"] < a["color"]:
                        b["side"] ^= 1
                    else:
                        # Tie: flip the one that appears later in row-major order
                        if (a["rmin"], a["cmin"]) > (b["rmin"], b["cmin"]):
                            a["side"] ^= 1
                        else:
                            b["side"] ^= 1
                    changed = True

    # Place components on output grid
    for comp in components:
        color = comp["color"]
        cmin = comp["cmin"]
        cmax = comp["cmax"]
        height = comp["height"]
        pattern = comp["pattern"]

        if comp["side"] == 0:
            rstart = 0
        else:
            rstart = h - height

        for i, prow in enumerate(pattern):
            r = rstart + i
            if not (0 <= r < h):
                continue
            for j, val in enumerate(prow):
                c = cmin + j
                if 0 <= c < w and val:
                    out[r, c] = color

    return out