import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    1. The value at grid[0, 0] is a marker.
       All other cells having this color are erased.
       If the marker color appears nowhere else, the grid is unchanged.

    2. The remaining non-zero cells are interpreted as line structures:
       - vertical walls
       - horizontal top bars
       - U-shapes: two vertical walls connected by a top bar
       - L-shapes: one vertical wall connected by a top bar

    3. Structures are settled from bottom to top.
       - For a U-shape, both side walls fall until blocked.
         The top bar then falls to the lowest row that is still:
           * below the original top row,
           * not blocked by already settled cells,
           * strictly above the shorter side wall's bottom.
         Side walls are extended upward if necessary so that they touch
         the final top-bar row.

       - For an L-shape, the whole shape shifts downward rigidly until
         either its vertical wall or its top bar would collide with an
         already settled structure.

    4. Any cells not assigned to a settled structure are dropped with
       stable column gravity.

    5. The marker cell is restored.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    target = int(grid[0, 0])

    # If the target color is only the marker, nothing changes.
    if int(np.sum(grid == target)) <= 1:
        return grid

    # Work on an erased version of the grid.
    orig = grid.copy()
    orig[orig == target] = 0
    orig[0, 0] = target

    def is_bg(r: int, c: int) -> bool:
        """Background for shape detection: zeros and the fixed marker."""
        if r < 0 or c < 0 or r >= H or c >= W:
            return True
        if r == 0 and c == 0:
            return True
        return orig[r, c] == 0

    def find_support(col: int, start_row: int, arr: np.ndarray) -> int:
        """
        First occupied row strictly below start_row in this column,
        or H if none exists.
        """
        for r in range(start_row + 1, H):
            if arr[r, col] != 0:
                return r
        return H

    # ------------------------------------------------------------------
    # Detect vertical side-wall segments.
    # A segment is a monochromatic vertical run of length at least 2.
    # If a lower cell has a horizontal non-background neighbor, the run
    # is split there; this helps separate vertical walls from horizontal
    # bars of other structures.
    # ------------------------------------------------------------------
    vertical_segments = []
    vseg_at = {}

    for c in range(W):
        r = 0
        while r < H:
            if is_bg(r, c):
                r += 1
                continue

            color = int(orig[r, c])
            start = r
            r += 1

            while r < H and not is_bg(r, c) and orig[r, c] == color:
                if r > start:
                    left_neighbor = (c > 0) and (not is_bg(r, c - 1))
                    right_neighbor = (c < W - 1) and (not is_bg(r, c + 1))
                    if left_neighbor or right_neighbor:
                        break
                r += 1

            end = r - 1
            length = end - start + 1

            if length >= 2:
                seg = {
                    "col": c,
                    "top": start,
                    "bottom": end,
                    "color": color,
                }
                vertical_segments.append(seg)
                vseg_at.setdefault((start, c), seg)

    # ------------------------------------------------------------------
    # Detect U and L structures from horizontal top runs.
    # ------------------------------------------------------------------
    groups = []
    covered = np.zeros((H, W), dtype=bool)

    for T in range(H):
        c = 0
        while c < W:
            if is_bg(T, c):
                c += 1
                continue

            a = c
            while c < W and not is_bg(T, c):
                c += 1
            b = c - 1

            wall_cols = []
            for cc in range(a, b + 1):
                if (T, cc) in vseg_at:
                    wall_cols.append(cc)

            # U-shape: exactly two vertical walls whose tops lie in this run.
            if len(wall_cols) == 2:
                L, R = wall_cols[0], wall_cols[1]
                groups.append({
                    "type": "U",
                    "top": T,
                    "left": L,
                    "right": R,
                    "a": a,
                    "b": b,
                })

                # Cover original pixels belonging to this U.
                for cc in range(a, b + 1):
                    if not is_bg(T, cc):
                        covered[T, cc] = True

                for col in (L, R):
                    seg = vseg_at[(T, col)]
                    for rr in range(seg["top"], seg["bottom"] + 1):
                        if not is_bg(rr, col):
                            covered[rr, col] = True

            # L-shape: exactly one vertical wall at one endpoint of the run.
            elif len(wall_cols) == 1:
                col = wall_cols[0]
                if (col == a or col == b) and (b - a >= 1):
                    groups.append({
                        "type": "L",
                        "top": T,
                        "side_col": col,
                        "a": a,
                        "b": b,
                    })

                    for cc in range(a, b + 1):
                        if not is_bg(T, cc):
                            covered[T, cc] = True

                    seg = vseg_at[(T, col)]
                    for rr in range(seg["top"], seg["bottom"] + 1):
                        if not is_bg(rr, col):
                            covered[rr, col] = True

    # ------------------------------------------------------------------
    # Settle structures from bottom to top.
    # ------------------------------------------------------------------
    settled = np.zeros((H, W), dtype=int)

    def merge_layer(layer: np.ndarray) -> None:
        """Merge a newly settled layer into the global settled grid."""
        empty = settled == 0
        settled[empty] = np.where(empty, layer, settled)

    def place_U(g: dict, layer: np.ndarray) -> None:
        T = g["top"]
        L = g["left"]
        R = g["right"]

        sides = []
        for col in (L, R):
            seg = vseg_at[(T, col)]

            support = find_support(col, seg["bottom"], settled)
            bottom = H - 1 if support == H else support - 1
            if bottom < seg["bottom"]:
                bottom = seg["bottom"]

            length = seg["bottom"] - seg["top"] + 1
            rigid_top = bottom - length + 1
            if rigid_top < seg["top"]:
                rigid_top = seg["top"]

            sides.append({
                "col": col,
                "seg_top": seg["top"],
                "seg_bottom": seg["bottom"],
                "rigid_top": rigid_top,
                "bottom": bottom,
                "top": rigid_top,
                "color": seg["color"],
            })

        # Limit from already settled cells under the top bar.
        limit = H - 1
        for cc in range(L, R + 1):
            s = find_support(cc, T, settled)
            lim = H - 1 if s == H else s - 1
            if lim < limit:
                limit = lim
        if limit < T:
            limit = T

        min_bottom = min(s["bottom"] for s in sides)
        fr = min(limit, min_bottom - 1)
        if fr < T:
            fr = T
        if fr > H - 1:
            fr = H - 1

        # Choose final wall tops.
        for s in sides:
            if fr < s["seg_bottom"]:
                s["top"] = min(s["rigid_top"], fr)
            else:
                s["top"] = s["seg_top"]

            # Keep the wall touching the final top-bar row.
            if s["top"] > fr:
                s["top"] = fr
            if s["top"] < T:
                s["top"] = T

        # Place side walls.
        for s in sides:
            for rr in range(s["top"], s["bottom"] + 1):
                layer[rr, s["col"]] = s["color"]

        # Place top bar.
        for cc in range(L, R + 1):
            val = orig[T, cc]
            if val != 0:
                layer[fr, cc] = int(val)

    def place_L(g: dict, layer: np.ndarray) -> None:
        T = g["top"]
        side_col = g["side_col"]
        a, b = g["a"], g["b"]
        seg = vseg_at[(T, side_col)]

        # How far the side wall can fall.
        support_wall = find_support(side_col, seg["bottom"], settled)
        wall_bottom = H - 1 if support_wall == H else support_wall - 1
        d_wall = wall_bottom - seg["bottom"]

        # How far the whole top bar can fall.
        limit = H - 1
        for cc in range(a, b + 1):
            s = find_support(cc, T, settled)
            lim = H - 1 if s == H else s - 1
            if lim < limit:
                limit = lim
        if limit < T:
            limit = T
        d_bar = limit - T

        d = max(0, min(d_wall, d_bar))
        new_T = T + d

        # Place top bar.
        for cc in range(a, b + 1):
            val = orig[T, cc]
            if val != 0:
                layer[new_T, cc] = int(val)

        # Place side wall with unchanged length.
        length = seg["bottom"] - seg["top"] + 1
        for i in range(length):
            layer[new_T + i, side_col] = seg["color"]

    top_rows = sorted({g["top"] for g in groups}, reverse=True)
    for T in top_rows:
        row_groups = [g for g in groups if g["top"] == T]
        row_groups.sort(key=lambda g: g["a"])

        layer = np.zeros((H, W), dtype=int)
        for g in row_groups:
            if g["type"] == "U":
                place_U(g, layer)
            else:
                place_L(g, layer)

        merge_layer(layer)

    # ------------------------------------------------------------------
    # Drop any uncovered stray pixels with stable column gravity.
    # ------------------------------------------------------------------
    for c in range(W):
        vals = []
        for r in range(H):
            if not is_bg(r, c) and not covered[r, c]:
                vals.append(int(orig[r, c]))

        if not vals:
            continue

        pos = H - 1
        for val in reversed(vals):
            while pos >= 0 and settled[pos, c] != 0:
                pos -= 1
            if pos >= 0:
                settled[pos, c] = val
                pos -= 1

    settled[0, 0] = target
    return settled