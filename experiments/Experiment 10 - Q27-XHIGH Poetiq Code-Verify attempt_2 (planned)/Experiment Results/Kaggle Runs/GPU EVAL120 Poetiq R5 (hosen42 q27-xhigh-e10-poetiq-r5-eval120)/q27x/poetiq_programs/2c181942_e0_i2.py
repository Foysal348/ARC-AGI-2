import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by finding the central 2x2 background hole with four
    adjacent marker dominoes. For each marker color, take all other pixels of
    that color as a remote binary pattern, choose one of its 8 symmetries so
    that the side facing the marker has a 2-cell run aligned with the marker,
    and place that pattern adjacent to the marker in the outward direction.
    Everything else is replaced by the background color.
    """
    if grid.size == 0:
        return grid

    grid = np.array(grid, copy=True)
    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    color_counts = {int(v): int(np.count_nonzero(grid == v)) for v in vals}

    # -------------------------------------------------------------------------
    # Find the central 2x2 gap surrounded by four marker dominoes.
    # Gap top-left is (r, c):
    #   gap cells: (r, c), (r, c+1), (r+1, c), (r+1, c+1)
    #   top marker: row r-1, cols c,c+1
    #   bottom marker: row r+2, cols c,c+1
    #   left marker: rows r,r+1, col c-1
    #   right marker: rows r,r+1, col c+2
    # -------------------------------------------------------------------------
    candidates = []

    for r in range(1, H - 2):
        for c in range(1, W - 2):
            if r + 2 >= H or c + 2 >= W:
                continue

            # The 2x2 gap must be background.
            if not np.all(grid[r:r + 2, c:c + 2] == bg):
                continue

            # Top marker.
            t1 = int(grid[r - 1, c])
            t2 = int(grid[r - 1, c + 1])
            if t1 == bg or t1 != t2:
                continue

            # Bottom marker.
            b1 = int(grid[r + 2, c])
            b2 = int(grid[r + 2, c + 1])
            if b1 == bg or b1 != b2:
                continue

            # Left marker.
            l1 = int(grid[r, c - 1])
            l2 = int(grid[r + 1, c - 1])
            if l1 == bg or l1 != l2:
                continue

            # Right marker.
            rt1 = int(grid[r, c + 2])
            rt2 = int(grid[r + 1, c + 2])
            if rt1 == bg or rt1 != rt2:
                continue

            colors = {
                "top": t1,
                "bottom": b1,
                "left": l1,
                "right": rt1,
            }

            marker_pixels = {}

            def add_marker(col, coords):
                marker_pixels.setdefault(int(col), []).extend(coords)

            add_marker(t1, [(r - 1, c), (r - 1, c + 1)])
            add_marker(b1, [(r + 2, c), (r + 2, c + 1)])
            add_marker(l1, [(r, c - 1), (r + 1, c - 1)])
            add_marker(rt1, [(r, c + 2), (r + 1, c + 2)])

            # Stricter check: markers should be isolated from other non-bg cells.
            def is_bg(rr, cc):
                return 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == bg

            strict = True
            if not (is_bg(r - 2, c) and is_bg(r - 2, c + 1)):
                strict = False
            if not (is_bg(r + 3, c) and is_bg(r + 3, c + 1)):
                strict = False
            if not (is_bg(r, c - 2) and is_bg(r + 1, c - 2)):
                strict = False
            if not (is_bg(r, c + 3) and is_bg(r + 1, c + 3)):
                strict = False
            if not (is_bg(r - 1, c - 1) and is_bg(r - 1, c + 2)):
                strict = False
            if not (is_bg(r - 1, c - 1) and is_bg(r + 2, c - 1)):
                strict = False
            if not (is_bg(r - 1, c + 2) and is_bg(r + 2, c + 2)):
                strict = False

            distinct = len(set(colors.values())) == 4

            remote_count = 0
            for col in set(colors.values()):
                remote_count += max(
                    0,
                    color_counts.get(int(col), 0) - len(marker_pixels.get(int(col), []))
                )

            # Prefer distinct colors, more remote pixels, then strict isolation.
            candidates.append((
                1 if distinct else 0,
                remote_count,
                1 if strict else 0,
                -r,
                -c,
                r,
                c,
                colors,
                marker_pixels,
            ))

    if not candidates:
        return grid

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
    _, _, _, _, _, gr, gc, colors, marker_pixels = candidates[0]

    # Start with an empty background grid and restore the four marker dominoes.
    out = np.full((H, W), bg, dtype=grid.dtype)
    for col, coords in marker_pixels.items():
        for rr, cc in coords:
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(col)

    gap_center = (gr + 1.0, gc + 1.0)

    # -------------------------------------------------------------------------
    # Choose and place one remote shape per direction.
    # -------------------------------------------------------------------------
    def choose_orientation(mat, direction, remote_origin):
        """
        Choose the best D4 symmetry of mat for a given outward direction.

        direction:
            'top'    -> remote is above marker, facing side is bottom of mat
            'bottom' -> remote is below marker, facing side is top of mat
            'left'   -> remote is left of marker, facing side is right of mat
            'right'  -> remote is right of marker, facing side is left of mat
        """
        h0, w0 = mat.shape
        remote_center = (
            remote_origin[0] + h0 / 2.0,
            remote_origin[1] + w0 / 2.0,
        )
        to_gap = (
            gap_center[0] - remote_center[0],
            gap_center[1] - remote_center[1],
        )
        gap_norm = float(np.hypot(to_gap[0], to_gap[1]))
        if gap_norm > 0:
            gap_unit = (to_gap[0] / gap_norm, to_gap[1] / gap_norm)
        else:
            gap_unit = (0.0, 0.0)

        # The 8 D4 transformations, with cost and a mapping from new point
        # coordinates to old point coordinates.
        transforms = [
            (0, mat, lambda r, c: (r, c)),
            (1, np.rot90(mat, 1), lambda r, c: (c, w0 - 1 - r)),
            (2, np.rot90(mat, 2), lambda r, c: (h0 - 1 - r, w0 - 1 - c)),
            (1, np.rot90(mat, 3), lambda r, c: (h0 - 1 - c, r)),
            (1, mat[:, ::-1].copy(), lambda r, c: (r, w0 - 1 - c)),
            (1, mat[::-1, :].copy(), lambda r, c: (h0 - 1 - r, c)),
            (1, mat.T.copy(), lambda r, c: (c, r)),
            (1, mat[::-1, ::-1].T.copy(), lambda r, c: (h0 - 1 - c, w0 - 1 - r)),
        ]

        # Deduplicate identical transformed matrices, keeping minimal cost.
        dedup = {}
        for cost, T, point_map in transforms:
            key = tuple(T.ravel())
            if key not in dedup or cost < dedup[key][0]:
                dedup[key] = (cost, T, point_map)

        best_score = None
        best_result = None

        for key, (cost, T, point_map) in dedup.items():
            h, w = T.shape

            # Identify the side that will face the marker.
            if direction == "top":
                vec = T[-1, :]
                L = w

                def pair_center(s, h=h, w=w):
                    return (h - 1, s + 0.5)

            elif direction == "bottom":
                vec = T[0, :]
                L = w

                def pair_center(s, h=h, w=w):
                    return (0, s + 0.5)

            elif direction == "left":
                vec = T[:, -1]
                L = h

                def pair_center(s, h=h, w=w):
                    return (s + 0.5, w - 1)

            else:  # direction == "right"
                vec = T[:, 0]
                L = h

                def pair_center(s, h=h, w=w):
                    return (s + 0.5, 0)

            if L < 2:
                continue

            side_sum = int(np.sum(vec))

            # Try every possible 2-cell alignment on this side.
            for s in range(L - 1):
                cover = int(vec[s]) + int(vec[s + 1])
                if cover == 0:
                    continue

                extra = side_sum - cover

                if cover == 2 and extra == 0:
                    exact_penalty = 0
                elif cover == 2:
                    exact_penalty = 1
                else:
                    exact_penalty = 2

                # Compute placement for this alignment.
                if direction in ("top", "bottom"):
                    left = gc - s
                    if direction == "top":
                        top = (gr - 1) - h
                    else:
                        top = (gr + 2) + 1
                else:
                    top = gr - s
                    if direction == "left":
                        left = (gc - 1) - w
                    else:
                        left = (gc + 2) + 1

                r_end = int(top) + h
                c_end = int(left) + w
                out_of_bounds = (
                    max(0, -int(top)) +
                    max(0, -int(left)) +
                    max(0, r_end - H) +
                    max(0, c_end - W)
                )

                # Centeredness of the chosen 2-cell run on the facing side.
                center_dist2 = abs(2 * s - (L - 2))

                # How strongly the corresponding original side pointed
                # toward the central hole in the input.
                nr, nc = pair_center(s)
                old_r, old_c = point_map(float(nr), float(nc))
                side_local = (
                    float(old_r) - h0 / 2.0,
                    float(old_c) - w0 / 2.0,
                )
                side_norm = float(np.hypot(side_local[0], side_local[1]))
                if side_norm > 0:
                    side_unit = (side_local[0] / side_norm, side_local[1] / side_norm)
                    dot = (
                        side_unit[0] * gap_unit[0] +
                        side_unit[1] * gap_unit[1]
                    )
                else:
                    dot = 0.0

                score = (
                    exact_penalty,
                    out_of_bounds,
                    center_dist2,
                    round(-dot, 6),
                    cost,
                    key,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_result = (T, s)

        return best_result

    for direction in ("top", "bottom", "left", "right"):
        color = int(colors[direction])

        # Remote pixels are all pixels of this color except the marker pixels.
        mask = (grid == color).astype(bool)
        for rr, cc in marker_pixels.get(color, []):
            if 0 <= rr < H and 0 <= cc < W:
                mask[rr, cc] = False

        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue

        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())

        mat = (grid[r0:r1 + 1, c0:c1 + 1] == color).astype(np.uint8)
        if not mat.any():
            continue

        chosen = choose_orientation(mat, direction, (r0, c0))
        if chosen is None:
            continue

        T, p = chosen
        h, w = T.shape

        # Final placement.
        if direction == "top":
            top = (gr - 1) - h
            left = gc - p
        elif direction == "bottom":
            top = (gr + 2) + 1
            left = gc - p
        elif direction == "left":
            top = gr - p
            left = (gc - 1) - w
        else:  # direction == "right"
            top = gr - p
            left = (gc + 2) + 1

        # Safe clipping, just in case a shape does not fit.
        rr0 = max(0, int(top))
        cc0 = max(0, int(left))
        rr1 = min(H, int(top) + h)
        cc1 = min(W, int(left) + w)

        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub = T[rr0 - int(top):rr1 - int(top), cc0 - int(left):cc1 - int(left)]
        region = out[rr0:rr1, cc0:cc1]
        region = np.where(sub > 0, color, region)
        out[rr0:rr1, cc0:cc1] = region

    return out