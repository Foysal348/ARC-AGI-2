import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform 4s by moving them in color-specific directions.

    Direction rules:
    - A rectangular frame of a single color C that contains both 1s and 2s
      is a direction indicator for color C.
    - The direction is from the centroid of all interior 1s to the centroid
      of all interior 2s, rounded to one of the 8 grid directions.
    - Each 4 is moved in the direction of the nearest indicator color.
      Ties are broken by local boundary adjacency, then color value.
    - A 4 moves step-by-step in that direction while the next cell is empty
      (0). Original 4 cells are treated as empty during movement.
    """
    g = np.asarray(grid, dtype=np.int64).copy()
    if g.ndim != 2:
        return g

    H, W = g.shape
    if g.size == 0:
        return g

    fours = np.argwhere(g == 4)
    if len(fours) == 0:
        return g

    # ------------------------------------------------------------
    # Prefix sums for counting 1s and 2s in rectangles
    # ------------------------------------------------------------
    def make_prefix(mask: np.ndarray) -> np.ndarray:
        pref = np.zeros((H + 1, W + 1), dtype=np.int64)
        pref[1:, 1:] = mask.astype(np.int64)
        pref[1:, 1:] = pref[1:, 1:].cumsum(axis=0).cumsum(axis=1)
        return pref

    pref1 = make_prefix(g == 1)
    pref2 = make_prefix(g == 2)

    def rect_sum(pref: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
        """Inclusive rectangle sum from (r1,c1) to (r2,c2)."""
        if r1 > r2 or c1 > c2:
            return 0
        return int(
            pref[r2 + 1, c2 + 1]
            - pref[r1, c2 + 1]
            - pref[r2 + 1, c1]
            + pref[r1, c1]
        )

    # ------------------------------------------------------------
    # Convert a centroid difference vector to one of 8 directions
    # ------------------------------------------------------------
    def vector_to_dir(dr: float, dc: float):
        eps = 1e-7
        if abs(dr) < eps:
            dr = 0.0
        if abs(dc) < eps:
            dc = 0.0

        if dr == 0.0 and dc == 0.0:
            return None

        if dr == 0.0:
            return (0, 1 if dc > 0 else -1)
        if dc == 0.0:
            return (1 if dr > 0 else -1, 0)

        # If one component dominates strongly, treat as cardinal.
        if abs(dr) > 2.0 * abs(dc):
            return (1 if dr > 0 else -1, 0)
        if abs(dc) > 2.0 * abs(dr):
            return (0, 1 if dc > 0 else -1)

        return (1 if dr > 0 else -1, 1 if dc > 0 else -1)

    # ------------------------------------------------------------
    # For a candidate indicator rectangle, compute interior stats
    # ------------------------------------------------------------
    def interior_stats(r1: int, c1: int, r2: int, c2: int):
        s1r = s1c = s2r = s2c = 0
        n1 = n2 = 0

        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                v = g[r, c]
                if v == 1:
                    n1 += 1
                    s1r += r
                    s1c += c
                elif v == 2:
                    n2 += 1
                    s2r += r
                    s2c += c

        if n1 == 0 or n2 == 0:
            return None

        dr = (s2r / n2) - (s1r / n1)
        dc = (s2c / n2) - (s1c / n1)

        d = vector_to_dir(dr, dc)
        if d is None:
            return None

        return n1, n2, d[0], d[1]

    # ------------------------------------------------------------
    # Find all valid rectangular indicator frames
    # ------------------------------------------------------------
    candidates_by_color = {}

    # 2 is reserved as the indicator target pixel, so it is not a frame color.
    colors = [int(c) for c in np.unique(g) if int(c) not in (0, 2, 4)]

    for C in colors:
        M = (g == C)
        row_cum = M.cumsum(axis=1)
        col_cum = M.cumsum(axis=0)

        for r1 in range(max(0, H - 2)):
            for r2 in range(r1 + 2, H):
                height = r2 - r1 + 1

                for c1 in range(max(0, W - 2)):
                    if not (bool(M[r1, c1]) and bool(M[r2, c1])):
                        continue

                    for c2 in range(c1 + 2, W):
                        if not (bool(M[r1, c2]) and bool(M[r2, c2])):
                            continue

                        width = c2 - c1 + 1

                        # Top row all C
                        top = row_cum[r1, c2] - (row_cum[r1, c1 - 1] if c1 > 0 else 0)
                        if top != width:
                            continue

                        # Bottom row all C
                        bottom = row_cum[r2, c2] - (row_cum[r2, c1 - 1] if c1 > 0 else 0)
                        if bottom != width:
                            continue

                        # Left column all C
                        left = col_cum[r2, c1] - (col_cum[r1 - 1, c1] if r1 > 0 else 0)
                        if left != height:
                            continue

                        # Right column all C
                        right = col_cum[r2, c2] - (col_cum[r1 - 1, c2] if r1 > 0 else 0)
                        if right != height:
                            continue

                        # Must contain at least one 1 and one 2 inside.
                        if rect_sum(pref1, r1 + 1, c1 + 1, r2 - 1, c2 - 1) == 0:
                            continue
                        if rect_sum(pref2, r1 + 1, c1 + 1, r2 - 1, c2 - 1) == 0:
                            continue

                        stats = interior_stats(r1, c1, r2, c2)
                        if stats is None:
                            continue

                        n1, n2, dr, dc = stats
                        area = height * width

                        candidates_by_color.setdefault(C, []).append(
                            (n2, n1, area, dr, dc)
                        )

    # Choose the best indicator for each color:
    # prefer the one containing more 2s, then smaller area, then more 1s.
    directions = {}
    for C, cands in candidates_by_color.items():
        cands.sort(key=lambda x: (-x[0], x[2], -x[1]))
        directions[C] = (cands[0][3], cands[0][4])

    if not directions:
        return g

    # ------------------------------------------------------------
    # Connected empty components, used only for tie-breaking.
    # Empty means 0 or 4.
    # ------------------------------------------------------------
    empty = (g == 0) | (g == 4)
    comp_id = -1 * np.ones((H, W), dtype=np.int64)
    comp_counts = []

    for i in range(H):
        for j in range(W):
            if empty[i, j] and comp_id[i, j] == -1:
                cid = len(comp_counts)
                comp_id[i, j] = cid
                stack = [(i, j)]
                counts = {}

                while stack:
                    r, c = stack.pop()
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if 0 <= nr < H and 0 <= nc < W:
                            if empty[nr, nc]:
                                if comp_id[nr, nc] == -1:
                                    comp_id[nr, nc] = cid
                                    stack.append((nr, nc))
                            else:
                                val = int(g[nr, nc])
                                counts[val] = counts.get(val, 0) + 1

                comp_counts.append(counts)

    # ------------------------------------------------------------
    # Precompute positions for each direction color
    # ------------------------------------------------------------
    color_points = {C: np.argwhere(g == C) for C in directions}

    def min_manhattan_to_color(r: int, c: int, C: int) -> int:
        pts = color_points[C]
        if pts.shape[0] == 0:
            return 10 ** 9
        return int(np.min(np.abs(pts[:, 0] - r) + np.abs(pts[:, 1] - c)))

    def choose_direction(r: int, c: int):
        cid = int(comp_id[r, c])
        counts = comp_counts[cid] if cid >= 0 else {}

        best_color = None
        best_key = None

        for C in directions:
            dist = min_manhattan_to_color(r, c, C)
            boundary_adj = counts.get(int(C), 0)
            key = (dist, -boundary_adj, int(C))

            if best_key is None or key < best_key:
                best_key = key
                best_color = C

        if best_color is None:
            return None
        return directions[best_color]

    # ------------------------------------------------------------
    # Move each 4.  For the same direction, process front-to-back so
    # multiple 4s do not occupy the same final cell if they line up.
    # ------------------------------------------------------------
    path_grid = g.copy()
    path_grid[path_grid == 4] = 0

    starts_with_dirs = []
    for r, c in fours:
        r = int(r)
        c = int(c)
        d = choose_direction(r, c)
        starts_with_dirs.append((r, c, d))

    groups = {}
    no_direction_targets = []

    for r, c, d in starts_with_dirs:
        if d is None:
            no_direction_targets.append((r, c))
        else:
            groups.setdefault(d, []).append((r, c))

    targets = list(no_direction_targets)

    for d, starts in groups.items():
        dr, dc = d
        blocked = path_grid.copy()

        # Furthest cell in the movement direction is processed first.
        def projection(pos):
            return pos[0] * dr + pos[1] * dc

        ordered = sorted(
            starts,
            key=lambda pos: (projection(pos), pos[0], pos[1]),
            reverse=True,
        )

        for r, c in ordered:
            rr, cc = r, c

            while True:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and blocked[nr, nc] == 0:
                    rr, cc = nr, nc
                else:
                    break

            blocked[rr, cc] = 4
            targets.append((rr, cc))

    # ------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------
    out = g.copy()
    out[g == 4] = 0

    for r, c in targets:
        out[r, c] = 4

    return out