import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the 5-marker selected pattern.

    The transformation used by the examples is:
    1. Find the rectangle implied by the two color-5 corner brackets.
       Its vertical span defines the output height.  Its horizontal span is
       used as the preferred output width.
    2. Select the non-5 pattern that the 5-rectangle is pointing to.
       In the examples this is the non-5 component whose bounding box has
       the greatest vertical overlap with the 5-rectangle; ties are broken
       by horizontal closeness, then smaller color, then leftmost position.
    3. Extract the selected component row-by-row over the 5-rectangle's
       vertical span.  If the 5-rectangle is taller than the component, the
       component is tiled vertically.
    4. If the 5-rectangle is wider than the selected component, each row is
       stretched horizontally while preserving zero gaps.  If the selected
       component is wider than the 5-rectangle, the rightmost part is kept
       (the side nearest the 5-marker in the examples).
    """
    if grid is None or grid.size == 0:
        return grid

    h, w = grid.shape

    # ------------------------------------------------------------------
    # 1. Locate the color-5 marker rectangle.
    # ------------------------------------------------------------------
    y5, x5 = np.where(grid == 5)
    if len(y5) == 0:
        # Fallback: no marker; return an empty-like grid.
        return np.zeros_like(grid)

    top, bottom = int(y5.min()), int(y5.max())
    left, right = int(x5.min()), int(x5.max())
    out_h = bottom - top + 1
    out_w = right - left + 1

    # ------------------------------------------------------------------
    # 2. Connected components of non-background, non-5 pixels.
    # ------------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == 0 or grid[r, c] == 5:
                continue

            color = int(grid[r, c])
            cells = []
            q = deque([(r, c)])
            visited[r, c] = True

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            rmin = min(r0 for r0, _ in cells)
            rmax = max(r0 for r0, _ in cells)
            cmin = min(c0 for _, c0 in cells)
            cmax = max(c0 for _, c0 in cells)

            # Ignore components that are far outside the 5-rectangle.
            vert_overlap = max(0, min(rmax, bottom) - max(rmin, top) + 1)
            if vert_overlap <= 0:
                continue

            # Horizontal distance between component and 5-rectangle.
            if cmax < left:
                hdist = left - cmax
            elif cmin > right:
                hdist = cmin - right
            else:
                hdist = 0

            comp = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
            comp[comp == 5] = 0

            components.append({
                "color": color,
                "rmin": rmin,
                "rmax": rmax,
                "cmin": cmin,
                "cmax": cmax,
                "comp": comp,
                "vert_overlap": vert_overlap,
                "hdist": hdist,
            })

    if not components:
        return np.zeros((out_h, out_w), dtype=grid.dtype)

    # Prefer maximum vertical overlap, then closest horizontally.
    components.sort(key=lambda x: (-x["vert_overlap"], x["hdist"], x["color"], x["cmin"]))
    sel = components[0]

    src = sel["comp"]
    src_h, src_w = src.shape

    # ------------------------------------------------------------------
    # 3. Build output rows by sampling/tile the selected component over
    #    the 5-rectangle's vertical span.
    # ------------------------------------------------------------------
    base_row = sel["rmin"]
    rows = []
    for i in range(out_h):
        global_r = top + i
        if global_r < sel["rmin"]:
            # If the 5-rectangle starts above the component, start at the
            # component's first row (this occurs in some extrapolation cases).
            src_r = (global_r - base_row) % src_h
        elif global_r > sel["rmax"]:
            # If the 5-rectangle extends below the component, tile it.
            src_r = (global_r - base_row) % src_h
        else:
            src_r = global_r - base_row
        rows.append(src[src_r])

    # ------------------------------------------------------------------
    # 4. Horizontally resize/crop each row to the output width.
    # ------------------------------------------------------------------
    def stretch_row(row: np.ndarray, target_w: int) -> np.ndarray:
        """
        Stretch a 1-D row to target_w while preserving zero gaps as much as
        possible.  Non-zero intervals are mapped proportionally and filled
        with their original color.
        """
        row = np.asarray(row)
        src_w = len(row)
        if src_w == target_w:
            return row.copy()

        out = np.zeros(target_w, dtype=row.dtype)
        nz = row != 0

        i = 0
        while i < src_w:
            if not nz[i]:
                i += 1
                continue

            j = i
            while j + 1 < src_w and nz[j + 1]:
                j += 1

            # Source interval [i, j] -> target interval.
            if src_w == 1:
                ti = 0
                tj = target_w - 1
            else:
                ti = int(round(i * (target_w - 1) / (src_w - 1)))
                tj = int(round(j * (target_w - 1) / (src_w - 1)))
                ti = max(0, min(target_w - 1, ti))
                tj = max(0, min(target_w - 1, tj))
                if ti > tj:
                    ti, tj = tj, ti

            out[ti:tj + 1] = row[i]
            i = j + 1

        return out

    if out_w >= src_w:
        resized = [stretch_row(row, out_w) for row in rows]
    else:
        # Keep the rightmost `out_w` columns, i.e. the side nearest the 5-box.
        resized = [row[-out_w:].copy() for row in rows]

    return np.asarray(resized, dtype=grid.dtype)