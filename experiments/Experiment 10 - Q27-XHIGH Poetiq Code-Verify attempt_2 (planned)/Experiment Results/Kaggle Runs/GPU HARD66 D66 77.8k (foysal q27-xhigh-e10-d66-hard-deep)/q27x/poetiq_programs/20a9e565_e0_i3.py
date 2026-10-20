import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The two 3-cell color-5 patterns are the top-left and bottom-right
       corners of a rectangle.  Their bounding box defines the output
       rectangle.

    2. Among all non-background, non-5 connected components, choose the
       component that can best explain the output:
       - it must overlap the vertical span of the 5-rectangle;
       - components whose bounding-box width can be stretched/cropped to
         the output width are preferred;
       - for ties, prefer larger vertical overlap, then smaller width,
         then leftmost position.

    3. Crop/tile the chosen component vertically over the 5-rectangle's
       vertical span.

    4. Horizontally resize each row to the output width:
       - if the source row is narrower, stretch its non-zero runs;
       - if the source row is wider, keep the rightmost output-width
         window.
    """
    if grid is None or grid.size == 0:
        return grid

    h, w = grid.shape

    # ---------------------------------------------------------------
    # 1. Locate the 5-marker rectangle.
    # ---------------------------------------------------------------
    y5, x5 = np.where(grid == 5)
    if len(y5) == 0:
        return np.zeros_like(grid)

    top, bottom = int(y5.min()), int(y5.max())
    left, right = int(x5.min()), int(x5.max())
    out_h = bottom - top + 1
    out_w = right - left + 1

    if out_h <= 0 or out_w <= 0:
        return np.zeros_like(grid)

    # ---------------------------------------------------------------
    # 2. Connected components of non-background, non-5 pixels.
    # ---------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == 0 or grid[r, c] == 5:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc] and int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            rmin = min(rr for rr, _ in cells)
            rmax = max(rr for rr, _ in cells)
            cmin = min(cc for _, cc in cells)
            cmax = max(cc for _, cc in cells)

            src_h = rmax - rmin + 1
            src_w = cmax - cmin + 1

            # Ignore components that do not overlap the 5 rectangle vertically.
            vert_overlap = max(0, min(rmax, bottom) - max(rmin, top) + 1)
            if vert_overlap <= 0:
                continue

            # Score components.
            width_ok = (src_w == out_w) or (out_w % src_w == 0) or (src_w % out_w == 0)
            score = (
                0 if width_ok else 1,
                -vert_overlap,
                abs(src_w - out_w),
                src_w,
                cmin
            )

            comp = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
            comp[comp == 5] = 0

            components.append((score, rmin, rmax, cmin, cmax, src_h, src_w, comp))

    if not components:
        return np.zeros((out_h, out_w), dtype=grid.dtype)

    components.sort(key=lambda x: x[0])
    _, rmin, rmax, cmin, cmax, src_h, src_w, src = components[0]

    # If the component is multi-color but the selected color is isolated,
    # keep all non-zero pixels; examples with a single color remain unchanged.
    # No explicit recoloring is needed.

    # ---------------------------------------------------------------
    # 3. Vertically crop/tile the selected component over output height.
    # ---------------------------------------------------------------
    rows = []
    for i in range(out_h):
        global_r = top + i
        # Map global row into source component coordinates.
        if global_r < rmin:
            src_r = (global_r - rmin) % src_h
        elif global_r > rmax:
            src_r = (global_r - rmin) % src_h
        else:
            src_r = global_r - rmin
        rows.append(src[src_r])

    # ---------------------------------------------------------------
    # 4. Horizontally resize/crop each row to the output width.
    # ---------------------------------------------------------------
    def stretch_row(row, target_w):
        """Stretch a 1D row to target_w, preserving non-zero runs."""
        row = np.asarray(row)
        src_w = len(row)
        if src_w == target_w:
            return row.copy()

        if target_w <= 0:
            return row[:0]

        # If target is wider, stretch non-zero intervals proportionally.
        if src_w < target_w:
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

                if src_w == 1:
                    ti, tj = 0, target_w - 1
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

        # If target is narrower, keep the rightmost target_w columns.
        return row[-target_w:].copy()

    resized = [stretch_row(row, out_w) for row in rows]
    return np.asarray(resized, dtype=grid.dtype)