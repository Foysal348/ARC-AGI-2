import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the observed rule:

    1. Red (2) cells are edge markers.
       - A marker activates the edge it lies on.
       - A top/bottom marker creates a vertical cut through its column.
       - A left/right marker creates a horizontal cut through its row.
       - If a marker is exactly on a corner, its cut is not added as a
         partition line, but its adjacent edges are still activated.

    2. The cuts divide the grid into regions associated with grid corners.
       For each orange (7) pixel not on a cut line:
       - determine its top/bottom side and left/right side from the cuts,
         or from activated edges when no cut exists on that axis;
       - keep it for a corner if that corner touches at least one activated edge.

    3. For each kept corner, collect all orange pixels mapped to that corner,
       compute their bounding box, and translate the pattern to that grid
       corner without flipping.

    4. Orange pixels that lie exactly on a cut line remain in their original
       positions.

    5. From each marker, draw a red trail along its cut line in the inward
       direction up to the farthest original orange pixel on that line.
       Original orange pixels on the trail remain orange.

    The background is the dominant input color, usually 1.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    # Background is usually 1; if 1 is absent, fall back to the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    if 1 in vals:
        bg = 1

    SHAPE = 7
    MARKER = 2

    out = np.full((H, W), bg, dtype=int)

    # ------------------------------------------------------------------
    # 1. Locate markers and determine active sides.
    # ------------------------------------------------------------------
    marker_sides = []
    active_sides = set()

    for r in range(H):
        for c in range(W):
            if grid[r, c] != MARKER:
                continue

            sides = []
            if r == 0:
                sides.append("top")
            if r == H - 1:
                sides.append("bottom")
            if c == 0:
                sides.append("left")
            if c == W - 1:
                sides.append("right")

            # Fallback for a hypothetical interior marker: choose nearest edge.
            if not sides:
                dist = {
                    "top": r,
                    "bottom": H - 1 - r,
                    "left": c,
                    "right": W - 1 - c,
                }
                tie = {"top": 0, "bottom": 1, "left": 2, "right": 3}
                side = min(dist, key=lambda k: (dist[k], tie[k]))
                sides.append(side)

            for side in sides:
                marker_sides.append((r, c, side))
                active_sides.add(side)

    if not marker_sides:
        return out

    # ------------------------------------------------------------------
    # 2. Determine cut lines.
    #    Top/bottom markers -> vertical cuts.
    #    Left/right markers -> horizontal cuts.
    #    Skip border-corner cuts because they would lie on the frame.
    # ------------------------------------------------------------------
    vline_set = set()
    hline_set = set()

    for r, c, side in marker_sides:
        if side in ("top", "bottom"):
            if 0 < c < W - 1:
                vline_set.add(c)
        else:
            if 0 < r < H - 1:
                hline_set.add(r)

    sorted_vlines = sorted(vline_set)
    sorted_hlines = sorted(hline_set)

    # ------------------------------------------------------------------
    # 3. Preserve orange pixels that lie on cut lines.
    # ------------------------------------------------------------------
    for r in range(H):
        if r in hline_set:
            for c in range(W):
                if grid[r, c] == SHAPE:
                    out[r, c] = SHAPE

    for c in range(W):
        if c in vline_set:
            for r in range(H):
                if grid[r, c] == SHAPE:
                    out[r, c] = SHAPE

    # ------------------------------------------------------------------
    # Helpers: which corner side(s) does a non-cut cell belong to?
    # ------------------------------------------------------------------
    def horizontal_sides(r):
        """Return possible top/bottom sides for row r."""
        if not sorted_hlines:
            return [s for s in ("top", "bottom") if s in active_sides]

        if r < sorted_hlines[0]:
            return ["top"]
        if r > sorted_hlines[-1]:
            return ["bottom"]

        # Between multiple horizontal cuts: no outer top/bottom side.
        return []

    def vertical_sides(c):
        """Return possible left/right sides for column c."""
        if not sorted_vlines:
            return [s for s in ("left", "right") if s in active_sides]

        if c < sorted_vlines[0]:
            return ["left"]
        if c > sorted_vlines[-1]:
            return ["right"]

        # Between multiple vertical cuts: no outer left/right side.
        return []

    # ------------------------------------------------------------------
    # 4. Assign non-cut orange pixels to corners.
    # ------------------------------------------------------------------
    corners = {
        ("top", "left"): [],
        ("top", "right"): [],
        ("bottom", "left"): [],
        ("bottom", "right"): [],
    }

    for r in range(H):
        if r in hline_set:
            continue

        for c in range(W):
            if grid[r, c] != SHAPE:
                continue
            if c in vline_set:
                continue

            for hs in horizontal_sides(r):
                for vs in vertical_sides(c):
                    # Keep a corner if at least one of its sides is activated.
                    if hs in active_sides or vs in active_sides:
                        corners[(hs, vs)].append((r, c))

    # ------------------------------------------------------------------
    # 5. Move each kept corner region to its corresponding grid corner.
    # ------------------------------------------------------------------
    for (hs, vs), pts in corners.items():
        if not pts:
            continue

        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]

        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        for r, c in pts:
            if hs == "top":
                nr = r - min_r
            else:  # bottom
                nr = H - 1 - max_r + r

            if vs == "left":
                nc = c - min_c
            else:  # right
                nc = W - 1 - max_c + c

            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = SHAPE

    # ------------------------------------------------------------------
    # 6. Draw red trails from each marker to the farthest orange pixel
    #    along its cut line in the inward direction.
    # ------------------------------------------------------------------
    def draw_vertical_line(col, start_row, side):
        if not (0 <= col < W):
            return

        if side == "top":
            step = 1
            candidates = [
                r for r in range(start_row, H)
                if grid[r, col] == SHAPE
            ]
            end = max(candidates) if candidates else start_row
        else:  # bottom
            step = -1
            candidates = [
                r for r in range(start_row, -1, -1)
                if grid[r, col] == SHAPE
            ]
            end = min(candidates) if candidates else start_row

        if step == 1:
            row_range = range(start_row, end + 1)
        else:
            row_range = range(start_row, end - 1, -1)

        for rr in row_range:
            if 0 <= rr < H:
                if out[rr, col] == SHAPE or grid[rr, col] == SHAPE:
                    out[rr, col] = SHAPE
                else:
                    out[rr, col] = MARKER

    def draw_horizontal_line(row, start_col, side):
        if not (0 <= row < H):
            return

        if side == "left":
            step = 1
            candidates = [
                c for c in range(start_col, W)
                if grid[row, c] == SHAPE
            ]
            end = max(candidates) if candidates else start_col
        else:  # right
            step = -1
            candidates = [
                c for c in range(start_col, -1, -1)
                if grid[row, c] == SHAPE
            ]
            end = min(candidates) if candidates else start_col

        if step == 1:
            col_range = range(start_col, end + 1)
        else:
            col_range = range(start_col, end - 1, -1)

        for cc in col_range:
            if 0 <= cc < W:
                if out[row, cc] == SHAPE or grid[row, cc] == SHAPE:
                    out[row, cc] = SHAPE
                else:
                    out[row, cc] = MARKER

    for r, c, side in marker_sides:
        if side in ("top", "bottom"):
            draw_vertical_line(c, r, side)
        else:
            draw_horizontal_line(r, c, side)

    return out