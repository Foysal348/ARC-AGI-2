import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by stamping left-margin patterns into the frame at positions
    indicated by matching colored markers on the frame border.

    - Find the uniform non-zero vertical separator column.
    - The frame is the rectangle from the column right of the separator to
      the right edge, with top and bottom rows.
    - Extract 8-connected colored patterns from the left margin.
    - For each pattern color, collect matching frame markers:
        * rows from the left/right frame columns
        * columns from the top/bottom frame rows
    - Place the pattern centered at every (row, column) combination.
    - Preserve the frame and unmarked left-margin patterns.
    - If stamps overlap, the stamp with smaller target row wins; ties go to
      smaller target column, then smaller color.
    """
    if grid is None:
        return grid

    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    out = grid.copy()

    if H < 3 or W < 4:
        return out

    # ------------------------------------------------------------------
    # Find the vertical wall/separator column.
    # ------------------------------------------------------------------
    wall_col = -1
    wall_color = -1

    for c in range(W - 1):
        col = grid[:, c]
        if col.size == 0:
            continue

        val = col[0]
        if val == 0:
            continue

        if not np.all(col == val):
            continue

        # The frame should start immediately to the right.
        if grid[0, c + 1] == 0 or grid[-1, c + 1] == 0:
            continue

        # Prefer a real separator with some left margin.
        if c > 0 and not np.any(grid[:, :c] == 0):
            continue

        wall_col = c
        wall_color = val
        break

    # Fallback: leftmost uniform non-zero column.
    if wall_col == -1:
        for c in range(W - 1):
            col = grid[:, c]
            if col[0] != 0 and np.all(col == col[0]):
                wall_col = c
                wall_color = col[0]
                break

    if wall_col == -1:
        return out

    # ------------------------------------------------------------------
    # Frame rectangle.
    # ------------------------------------------------------------------
    frame_left = wall_col + 1

    frame_right = W - 1
    for c in range(W - 1, frame_left, -1):
        if grid[0, c] != 0 and grid[-1, c] != 0:
            frame_right = c
            break

    if frame_right <= frame_left:
        return out

    # ------------------------------------------------------------------
    # Collect left-margin cells by color.
    # ------------------------------------------------------------------
    left_cells = {}
    for r in range(H):
        for c in range(wall_col):
            v = grid[r, c]
            if v != 0 and v != wall_color:
                left_cells.setdefault(v, set()).add((r, c))

    if not left_cells:
        return out

    # ------------------------------------------------------------------
    # Extract 8-connected components as patterns.
    # ------------------------------------------------------------------
    components = []
    visited = set()

    for color, cells in left_cells.items():
        for start in cells:
            if start in visited:
                continue

            q = deque([start])
            visited.add(start)
            comp = []

            while q:
                r, c = q.popleft()
                comp.append((r, c))

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < wall_col:
                            if (nr, nc) in cells and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                q.append((nr, nc))

            if not comp:
                continue

            min_r = min(r for r, _ in comp)
            max_r = max(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            max_c = max(c for _, c in comp)

            h = max_r - min_r + 1
            w = max_c - min_c + 1

            pattern = np.zeros((h, w), dtype=bool)
            for r, c in comp:
                pattern[r - min_r, c - min_c] = True

            # Anchor is the bounding-box center. For odd sizes this is exact.
            anchor_r = h // 2
            anchor_c = w // 2

            components.append((color, h, w, pattern, anchor_r, anchor_c))

    # ------------------------------------------------------------------
    # Collect frame markers.
    # Rows come from the vertical frame sides, columns from horizontal sides.
    # Corners are ignored.
    # ------------------------------------------------------------------
    marker_rows = {}
    marker_cols = {}

    for r in (0, H - 1):
        for c in range(frame_left, frame_right + 1):
            v = grid[r, c]
            if v == 0 or v == wall_color:
                continue

            if frame_left < c < frame_right:
                marker_cols.setdefault(v, set()).add(c)

    for r in range(1, H - 1):
        for c in (frame_left, frame_right):
            v = grid[r, c]
            if v == 0 or v == wall_color:
                continue

            marker_rows.setdefault(v, set()).add(r)

    # ------------------------------------------------------------------
    # Build stamp operations.
    # ------------------------------------------------------------------
    ops = []
    clear_colors = set()

    for color in left_cells:
        rset = marker_rows.get(color)
        cset = marker_cols.get(color)

        if not rset or not cset:
            continue

        clear_colors.add(color)

        for r in rset:
            for c in cset:
                ops.append((r, c, color))

    if not ops:
        return out

    # Higher rows are drawn first and therefore win overlaps.
    ops.sort(key=lambda x: (x[0], x[1], x[2]))

    comps_by_color = {}
    for comp in components:
        comps_by_color.setdefault(comp[0], []).append(comp)

    # Interior is strictly inside the frame.
    r_min, r_max = 1, H - 2
    c_min, c_max = frame_left + 1, frame_right - 1

    # ------------------------------------------------------------------
    # Stamp patterns.
    # ------------------------------------------------------------------
    for r0, c0, color in ops:
        for _, h, w, pattern, ar, ac in comps_by_color.get(color, []):
            pr, pc = np.nonzero(pattern)

            for dr, dc in zip(pr, pc):
                nr = r0 + dr - ar
                nc = c0 + dc - ac

                if r_min <= nr <= r_max and c_min <= nc <= c_max:
                    if out[nr, nc] == 0:
                        out[nr, nc] = color

    # ------------------------------------------------------------------
    # Erase used left-margin patterns.
    # ------------------------------------------------------------------
    for color in clear_colors:
        for r in range(H):
            for c in range(wall_col):
                if grid[r, c] == color:
                    out[r, c] = 0

    return out