import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete a tile-grid pattern.

    The input is a grid made of 3x3 tiles separated by 1-pixel lines.
    One tile type is the default background tile. Another tile type contains a
    distinctive color; the positions of that color inside the 3x3 tile form a
    3x3 stencil.

    The input contains some copies of the special tile, which form a partial
    copy of that stencil at the tile level. This transform places the full
    3x3 stencil so that all input special tiles lie on active stencil cells,
    then stamps the special tile at those active tile positions.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    def get_tile_bounds(g):
        """Find 3x3 tile blocks separated by 1-pixel lines."""
        H, W = g.shape

        # Separator lines are full rows/columns of a single color.
        sep_rows = [r for r in range(H) if np.all(g[r] == g[r, 0])]
        sep_cols = [c for c in range(W) if np.all(g[:, c] == g[0, c])]

        def make_bounds(lines):
            if len(lines) < 2:
                return None
            bounds = []
            for a, b in zip(lines[:-1], lines[1:]):
                # Tiles in this task are 3x3.
                if b - a - 1 != 3:
                    return None
                bounds.append((a + 1, b))
            if not bounds:
                return None
            return bounds

        row_bounds = make_bounds(sep_rows)
        col_bounds = make_bounds(sep_cols)

        if row_bounds is not None and col_bounds is not None:
            return row_bounds, col_bounds

        # Fallback for the standard 4*n - 1 layout.
        if (H - 1) % 4 == 0 and (W - 1) % 4 == 0:
            n = (H - 1) // 4
            row_bounds = [(1 + 4 * i, 4 + 4 * i) for i in range(n)]
            col_bounds = [(1 + 4 * j, 4 + 4 * j) for j in range(n)]
            if row_bounds and col_bounds:
                return row_bounds, col_bounds

        return None, None

    row_bounds, col_bounds = get_tile_bounds(grid)
    if row_bounds is None or col_bounds is None:
        return grid.copy()

    nr = len(row_bounds)
    nc = len(col_bounds)
    if nr == 0 or nc == 0:
        return grid.copy()

    th = row_bounds[0][1] - row_bounds[0][0]
    tw = col_bounds[0][1] - col_bounds[0][0]
    if th != 3 or tw != 3:
        return grid.copy()

    for r0, r1 in row_bounds:
        if r1 - r0 != th:
            return grid.copy()
    for c0, c1 in col_bounds:
        if c1 - c0 != tw:
            return grid.copy()

    tiles = []
    pat_counts = {}
    pat_to_tile = {}
    pat_grid = [[None] * nc for _ in range(nr)]

    # Extract all 3x3 tiles.
    for i, (r0, r1) in enumerate(row_bounds):
        for j, (c0, c1) in enumerate(col_bounds):
            tile = grid[r0:r1, c0:c1]
            pat = tuple(tile.ravel().tolist())
            tiles.append((i, j, tile, pat))
            pat_grid[i][j] = pat
            pat_counts[pat] = pat_counts.get(pat, 0) + 1
            if pat not in pat_to_tile:
                pat_to_tile[pat] = tile.copy()

    if not pat_counts:
        return grid.copy()

    # Identify the default tile. Usually it is the most frequent tile.
    # If all four corners share a common tile, prefer that as background.
    most_common_pat = max(pat_counts.items(), key=lambda kv: kv[1])[0]
    corners = [
        pat_grid[0][0],
        pat_grid[0][-1],
        pat_grid[-1][0],
        pat_grid[-1][-1],
    ]
    corner_pat = None
    if len(set(corners)) == 1:
        corner_pat = corners[0]

    default_pat = most_common_pat
    if (
        corner_pat is not None
        and pat_counts.get(corner_pat, 0) >= max(1, pat_counts[most_common_pat] // 2)
    ):
        default_pat = corner_pat

    nondefault = [p for p in pat_counts if p != default_pat]
    if not nondefault:
        return grid.copy()

    default_tile = pat_to_tile[default_pat]
    d_unique, d_counts = np.unique(default_tile, return_counts=True)
    d_set = set(int(x) for x in d_unique)

    # Choose the special tile type. Prefer patterns that introduce a new color.
    def special_pat_score(p):
        tile = pat_to_tile[p]
        s_vals, s_counts = np.unique(tile, return_counts=True)
        absent_counts = [
            int(cnt)
            for c, cnt in zip(s_vals, s_counts)
            if int(c) not in d_set
        ]
        return (
            pat_counts[p],
            len(absent_counts),
            max(absent_counts, default=0),
        )

    special_pat = max(nondefault, key=special_pat_score)
    special_tile = pat_to_tile[special_pat]

    # Determine the marker color inside the special tile.
    # It is normally the color in the special tile that does not appear in default.
    s_unique, s_counts = np.unique(special_tile, return_counts=True)
    absent = [
        (int(c), int(cnt))
        for c, cnt in zip(s_unique, s_counts)
        if int(c) not in d_set
    ]

    if absent:
        best_marker = max(absent, key=lambda x: x[1])[0]
    else:
        diff = special_tile != default_tile
        best_marker = None
        best_key = None
        for c, cnt in zip(s_unique, s_counts):
            v = int(c)
            diff_count = int(np.sum(diff & (special_tile == v)))
            key = (diff_count, int(cnt))
            if best_key is None or key > best_key:
                best_key = key
                best_marker = v

    if best_marker is None:
        return grid.copy()

    mask = special_tile == best_marker
    if not mask.any():
        return grid.copy()

    # Coordinates of input special tiles at the tile level.
    S = [(i, j) for (i, j, _, pat) in tiles if pat == special_pat]
    if not S:
        return grid.copy()

    # Precompute mask row/column sets and centers.
    mask_cols = [set() for _ in range(th)]
    mask_rows = [set() for _ in range(tw)]
    for ir in range(th):
        for ic in range(tw):
            if mask[ir, ic]:
                mask_cols[ir].add(ic)
                mask_rows[ic].add(ir)

    mrs, mcs = np.nonzero(mask)
    m_row_center = float(mrs.mean()) if mrs.size else 0.0
    m_col_center = float(mcs.mean()) if mcs.size else 0.0

    s_row_center = float(np.mean([r for r, _ in S]))
    s_col_center = float(np.mean([c for _, c in S]))

    S_rows = {}
    S_cols = {}
    for r, c in S:
        S_rows.setdefault(r, set()).add(c)
        S_cols.setdefault(c, set()).add(r)

    def exact_score(R, C):
        """
        Score how many stencil rows/columns are exactly matched by input tiles.
        This helps anchor the stencil when multiple placements are possible.
        """
        score = 0

        for ir in range(th):
            tr = R + ir
            if tr in S_rows:
                s_cols = {c - C for c in S_rows[tr] if 0 <= c - C < tw}
            else:
                s_cols = set()

            m_cols = mask_cols[ir]
            if m_cols and s_cols == m_cols:
                score += 1

        for ic in range(tw):
            tc = C + ic
            if tc in S_cols:
                s_rows = {r - R for r in S_cols[tc] if 0 <= r - R < th}
            else:
                s_rows = set()

            m_rows = mask_rows[ic]
            if m_rows and s_rows == m_rows:
                score += 1

        return score

    def centroid_dist(R, C):
        """Distance between the centroid of input tiles and the stencil centroid."""
        return (
            (s_row_center - (R + m_row_center)) ** 2
            + (s_col_center - (C + m_col_center)) ** 2
        )

    best = None
    best_score = None
    max_R = nr - th + 1
    max_C = nc - tw + 1

    # First try placements where every input special tile lies on an active cell.
    if max_R > 0 and max_C > 0:
        for R in range(max_R):
            for C in range(max_C):
                ok = True
                for r, c in S:
                    ir = r - R
                    ic = c - C
                    if ir < 0 or ir >= th or ic < 0 or ic >= tw or not mask[ir, ic]:
                        ok = False
                        break
                if not ok:
                    continue

                score = (
                    exact_score(R, C),
                    -centroid_dist(R, C),
                    -R - C,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = (R, C)

    # Fallback: choose the placement covering the most input special tiles.
    if best is None and max_R > 0 and max_C > 0:
        for R in range(max_R):
            for C in range(max_C):
                covered = 0
                for r, c in S:
                    ir = r - R
                    ic = c - C
                    if 0 <= ir < th and 0 <= ic < tw and mask[ir, ic]:
                        covered += 1

                score = (
                    covered,
                    -centroid_dist(R, C),
                    -R - C,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = (R, C)

    # Last resort: center the stencil.
    if best is None:
        R = max(0, (nr - th) // 2)
        C = max(0, (nc - tw) // 2)
        if R + th <= nr and C + tw <= nc:
            best = (R, C)
        else:
            return grid.copy()

    R, C = best
    target = set()
    for ir in range(th):
        for ic in range(tw):
            if mask[ir, ic]:
                target.add((R + ir, C + ic))

    # Rebuild the grid: default everywhere except active stencil tiles.
    out = grid.copy()
    for i, (r0, r1) in enumerate(row_bounds):
        for j, (c0, c1) in enumerate(col_bounds):
            if (i, j) in target:
                out[r0:r1, c0:c1] = special_tile
            else:
                out[r0:r1, c0:c1] = default_tile

    return out