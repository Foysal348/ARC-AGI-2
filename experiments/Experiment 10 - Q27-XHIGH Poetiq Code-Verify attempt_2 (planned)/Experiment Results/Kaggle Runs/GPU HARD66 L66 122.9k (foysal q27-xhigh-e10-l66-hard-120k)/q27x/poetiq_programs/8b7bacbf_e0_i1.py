import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule (inferred from examples):
    1. Identify the background color as the most frequent color.
    2. Identify the fill color as a singleton color (appears exactly once),
       preferring singleton pixels on the grid border, especially bottom/right.
    3. For each other color, detect "frame" interiors:
       - Group nearby pixels of the same frame color.
       - For each row in the frame group, use the leftmost/rightmost frame-color
         pixels as horizontal boundaries (grid edge may also serve as a boundary).
       - The interior is the background-colored cells between those boundaries.
       - Keep only components that are geometrically closed vertically and are not
         orthogonally adjacent to an unrelated non-background intruder color.
    4. For each frame color, if there are multiple distinct interior sizes,
       ignore the smallest size class and fill the larger interior classes.
       If all interiors for that color have the same size, fill all of them.
    5. Fill selected background cells with the singleton fill color.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    if grid.size == 0 or grid.ndim != 2:
        return out

    H, W = grid.shape

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def dilate8(mask: np.ndarray) -> np.ndarray:
        """Dilate a boolean mask by one cell in the 8-neighborhood."""
        p = np.pad(mask, 1, mode="constant", constant_values=False)
        h, w = mask.shape
        res = np.zeros_like(mask)
        for dr in range(3):
            for dc in range(3):
                res |= p[dr:dr + h, dc:dc + w]
        return res

    def label8(mask: np.ndarray):
        """Label 8-connected components in a boolean mask."""
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=int)
        nxt = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] and labels[i, j] == 0:
                    nxt += 1
                    labels[i, j] = nxt
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and labels[nx, ny] == 0:
                                    labels[nx, ny] = nxt
                                    stack.append((nx, ny))
        return labels, nxt

    def label4(mask: np.ndarray):
        """Label 4-connected components in a boolean mask."""
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=int)
        nxt = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] and labels[i, j] == 0:
                    nxt += 1
                    labels[i, j] = nxt
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and labels[nx, ny] == 0:
                                labels[nx, ny] = nxt
                                stack.append((nx, ny))
        return labels, nxt

    def candidate_interior_for_group(group: np.ndarray) -> np.ndarray:
        """
        For one grouped set of frame-color pixels, produce candidate interior
        background cells using row-wise left/right extremes.
        """
        cand = np.zeros((H, W), dtype=bool)

        rows = np.where(group.any(axis=1))[0]
        if rows.size == 0:
            return cand

        rmin, rmax = int(rows[0]), int(rows[-1])

        cols = np.where(group.any(axis=0))[0]
        if cols.size == 0:
            return cand
        cmin, cmax = int(cols[0]), int(cols[-1])

        # Grid-edge boundary is allowed only if the frame actually has pixels on that edge.
        edge_left = (cmin == 0 and group[:, 0].sum() >= 1)
        edge_right = (cmax == W - 1 and group[:, W - 1].sum() >= 1)

        for r in range(rmin, rmax + 1):
            row_cols = np.where(group[r])[0]
            if row_cols.size > 0:
                lc = int(row_cols.min())
                rc = int(row_cols.max())

                left = lc
                right = rc

                if edge_right and rc < W - 1:
                    right = W
                if edge_left and lc > 0:
                    left = -1
            else:
                # A row with no frame pixels can only be interior if both
                # horizontal boundaries are grid edges.
                if edge_left and edge_right:
                    left, right = -1, W
                else:
                    continue

            if right <= left + 1:
                continue

            for cc in range(left + 1, right):
                if grid[r, cc] == bg:
                    cand[r, cc] = True

        return cand

    def is_valid_component(comp: np.ndarray, group: np.ndarray, c: int) -> bool:
        """
        Validate a candidate interior component:
        - the frame group must have pixels in every row of the component span;
        - top/bottom must be closed by frame pixels, except when the component
          starts/ends at a one-cell vertex or at the grid edge;
        - no candidate cell may be orthogonally adjacent to an unrelated
          non-background intruder color.
        """
        rows = np.where(comp.any(axis=1))[0]
        if rows.size == 0:
            return False
        r0, r1 = int(rows[0]), int(rows[-1])

        # Group vertical extent.
        grow = np.where(group.any(axis=1))[0]
        if grow.size == 0:
            return False
        grmin, grmax = int(grow[0]), int(grow[-1])

        # The frame color must be present in every row covered by this component.
        for r in range(r0, r1 + 1):
            if not group[r].any():
                return False

        top_cols = np.where(comp[r0])[0]
        if top_cols.size == 0:
            return False
        top_l, top_r = int(top_cols[0]), int(top_cols[-1])

        bot_cols = np.where(comp[r1])[0]
        if bot_cols.size == 0:
            return False
        bot_l, bot_r = int(bot_cols[0]), int(bot_cols[-1])

        # Top closure.
        if r0 > grmin:
            if r0 - 1 < 0:
                return False
            if not group[r0 - 1, top_l:top_r + 1].any():
                return False
        else:
            # First row of the frame group: allow a one-cell top vertex.
            if r0 > 0 and (top_r - top_l + 1) > 1:
                return False

        # Bottom closure.
        if r1 < grmax:
            if r1 + 1 >= H:
                return False
            if not group[r1 + 1, bot_l:bot_r + 1].any():
                return False
        else:
            # Last row of the frame group: allow a one-cell bottom vertex.
            if r1 < H - 1 and (bot_r - bot_l + 1) > 1:
                return False

        # Intruder check: allowed neighboring colors are background, this frame
        # color, the fill color, and any singleton marker colors.
        allowed = {int(bg), int(c), int(fill)}
        allowed.update(int(x) for x in singleton_positions)

        p = np.pad(comp, 1, mode="constant", constant_values=False)
        neigh = np.zeros((H, W), dtype=bool)
        neigh |= p[:-2, 1:-1]   # up
        neigh |= p[2:, 1:-1]    # down
        neigh |= p[1:-1, :-2]   # left
        neigh |= p[1:-1, 2:]    # right

        neigh_vals = grid[neigh]
        if neigh_vals.size and not np.all(np.isin(neigh_vals, np.array(sorted(allowed)))):
            return False

        return True

    # ------------------------------------------------------------
    # Background and fill color
    # ------------------------------------------------------------
    max_val = int(grid.max()) if grid.size else 0
    counts = np.bincount(grid.ravel(), minlength=max(10, max_val + 1))
    bg = int(np.argmax(counts))

    singleton_positions = {}
    for c in range(len(counts)):
        if c != bg and counts[c] == 1:
            idx = np.argwhere(grid == c)
            if idx.size:
                singleton_positions[c] = (int(idx[0][0]), int(idx[0][1]))

    if singleton_positions:
        border_singletons = []
        for c, (r, cc) in singleton_positions.items():
            if r == 0 or r == H - 1 or cc == 0 or cc == W - 1:
                # Prefer bottom/right border singletons.
                score = (
                    r == H - 1,
                    cc == W - 1,
                    r,
                    cc
                )
                border_singletons.append((score, c))

        if border_singletons:
            fill = max(border_singletons)[1]
        else:
            fill = max(singleton_positions.keys())
    else:
        # Fallback: least frequent non-background color.
        nonbg = [c for c in range(len(counts)) if c != bg and counts[c] > 0]
        if not nonbg:
            return out
        fill = min(nonbg, key=lambda x: (int(counts[x]), x))

    # Optional "decoy" color: the color next to the fill singleton.
    # We do not treat this color as a frame color.
    decoy = None
    if fill in singleton_positions:
        fr, fc = singleton_positions[fill]
        neigh_counts = {}
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = int(grid[nr, nc])
                if v != bg and v != fill:
                    neigh_counts[v] = neigh_counts.get(v, 0) + 1
        if neigh_counts:
            decoy = max(neigh_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # ------------------------------------------------------------
    # Find valid interiors per frame color
    # ------------------------------------------------------------
    valid_by_color = {}

    for c in range(len(counts)):
        if c == bg or c == fill or c == decoy:
            continue

        mask = (grid == c)
        if not mask.any():
            continue

        # Merge nearby frame pixels that belong to the same drawn frame.
        dil = dilate8(mask)
        labels, nlabels = label8(dil)

        for lab in range(1, nlabels + 1):
            group = mask & (labels == lab)
            if group.sum() < 4:
                continue

            cand = candidate_interior_for_group(group)
            if not cand.any():
                continue

            comp_labels, ncomps = label4(cand)
            comps = []

            for comp_lab in range(1, ncomps + 1):
                comp = (comp_labels == comp_lab)
                area = int(comp.sum())
                if area == 0:
                    continue
                if is_valid_component(comp, group, c):
                    comps.append((comp, area))

            if comps:
                valid_by_color.setdefault(c, []).extend(comps)

    # ------------------------------------------------------------
    # Select interiors to fill:
    # for each color, ignore the smallest size class if multiple sizes exist.
    # ------------------------------------------------------------
    fill_mask = np.zeros((H, W), dtype=bool)

    for _, comps in valid_by_color.items():
        areas = [a for _, a in comps]
        min_area = min(areas)

        if len(set(areas)) == 1:
            for comp, _ in comps:
                fill_mask |= comp
        else:
            for comp, area in comps:
                if area > min_area:
                    fill_mask |= comp

    if fill_mask.any():
        out[fill_mask] = fill

    return out