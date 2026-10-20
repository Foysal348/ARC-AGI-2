import numpy as np
from collections import deque, Counter


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill frame cells with the color of the nearest connected pattern component.

    Rule:
      1. Background is the most frequent color.
      2. Frame color is identified from the most central solid 3x3 block of a
         non-background color. If no solid block exists, a 3x3 perimeter block
         is used as fallback; otherwise the largest non-background structure is
         used.
      3. All remaining non-background colors are pattern colors.
      4. Pattern cells are grouped into 4-connected components. Each component
         is represented by its dominant color.
      5. Every frame cell is replaced by the representative color of the nearest
         pattern component (4-neighbor Manhattan distance).
      6. Ties are broken by global pattern-frequency, then by larger color value.
      7. The central 3x3 frame block keeps its border, and its center is filled
         with the globally most frequent pattern color.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    colors, counts = np.unique(grid, return_counts=True)
    count_dict = {int(c): int(n) for c, n in zip(colors, counts)}

    # ------------------------------------------------------------------
    # 1. Identify background color.
    # ------------------------------------------------------------------
    max_count = max(count_dict.values())
    bg_candidates = [c for c, n in count_dict.items() if n == max_count]

    if len(bg_candidates) == 1:
        background = bg_candidates[0]
    else:
        def border_count(color):
            return int(
                (grid[0, :] == color).sum()
                + (grid[-1, :] == color).sum()
                + (grid[:, 0] == color).sum()
                + (grid[:, -1] == color).sum()
            )

        background = max(bg_candidates, key=lambda c: (border_count(c), count_dict[c]))

    non_background = [c for c in count_dict if c != background]
    if not non_background:
        return grid

    def bbox_area(color):
        mask = grid == color
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return 0
        r = np.where(rows)[0]
        c = np.where(cols)[0]
        return int((r[-1] - r[0] + 1) * (c[-1] - c[0] + 1))

    area_cache = {c: bbox_area(c) for c in non_background}

    # ------------------------------------------------------------------
    # 2. Identify frame color and the special central 3x3 block.
    # ------------------------------------------------------------------
    def solid_block_info(color):
        """Return (number_of_blocks, best_block) where best_block=(dist, r, c)."""
        a = grid == color
        num = 0
        best = None

        for r in range(H - 2):
            for c in range(W - 2):
                if (
                    a[r, c] and a[r, c + 1] and a[r, c + 2]
                    and a[r + 1, c] and a[r + 1, c + 1] and a[r + 1, c + 2]
                    and a[r + 2, c] and a[r + 2, c + 1] and a[r + 2, c + 2]
                ):
                    num += 1
                    center_r = r + 1
                    center_c = c + 1
                    dist = (2 * center_r - (H - 1)) ** 2 + (2 * center_c - (W - 1)) ** 2

                    if best is None or dist < best[0] or (
                        dist == best[0] and (r, c) < (best[1], best[2])
                    ):
                        best = (dist, r, c)

        return num, best

    solid_candidates = []
    for c in non_background:
        num, best = solid_block_info(c)
        if best is not None:
            dist, r, col = best
            solid_candidates.append(
                (dist, -count_dict[c], -area_cache[c], -num, c, r, col)
            )

    frame_color = None
    special_top_left = None

    if solid_candidates:
        solid_candidates.sort()
        _, _, _, _, frame_color, sr, sc = solid_candidates[0]
        special_top_left = (sr, sc)
    else:
        # Fallback: look for a 3x3 perimeter of a non-background color.
        def perimeter_block_info(color):
            a = grid == color
            num = 0
            best = None

            for r in range(H - 2):
                for c in range(W - 2):
                    if (
                        a[r, c] and a[r, c + 1] and a[r, c + 2]
                        and a[r + 1, c] and a[r + 1, c + 2]
                        and a[r + 2, c] and a[r + 2, c + 1] and a[r + 2, c + 2]
                    ):
                        num += 1
                        center_r = r + 1
                        center_c = c + 1
                        dist = (
                            (2 * center_r - (H - 1)) ** 2
                            + (2 * center_c - (W - 1)) ** 2
                        )
                        if best is None or dist < best[0] or (
                            dist == best[0] and (r, c) < (best[1], best[2])
                        ):
                            best = (dist, r, c)

            return num, best

        perimeter_candidates = []
        for c in non_background:
            num, best = perimeter_block_info(c)
            if best is not None:
                dist, r, col = best
                perimeter_candidates.append(
                    (dist, -count_dict[c], -area_cache[c], -num, c, r, col)
                )

        if perimeter_candidates:
            perimeter_candidates.sort()
            _, _, _, _, frame_color, sr, sc = perimeter_candidates[0]
            special_top_left = (sr, sc)
        else:
            # Last fallback: use the largest non-background structure.
            frame_color = min(
                non_background,
                key=lambda c: (-area_cache[c], -count_dict[c], c),
            )

    if frame_color is None:
        return grid

    # ------------------------------------------------------------------
    # 3. Pattern colors and global pattern statistics.
    # ------------------------------------------------------------------
    pattern_colors = [
        c for c in count_dict if c != background and c != frame_color
    ]
    if not pattern_colors:
        return grid

    global_counts = Counter()
    for c in pattern_colors:
        global_counts[c] = count_dict[c]

    if not global_counts:
        return grid

    global_majority = max(
        global_counts.items(),
        key=lambda kv: (kv[1], kv[0]),
    )[0]

    pattern_mask = np.isin(grid, pattern_colors)
    if not pattern_mask.any():
        return grid

    # ------------------------------------------------------------------
    # 4. Connected components of pattern cells.
    # ------------------------------------------------------------------
    source_color = np.full((H, W), -1, dtype=int)
    visited = np.zeros((H, W), dtype=bool)

    for r in range(H):
        for c in range(W):
            if not pattern_mask[r, c] or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            comp_counts = Counter()

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                comp_counts[int(grid[rr, cc])] += 1

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if pattern_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            representative = max(
                comp_counts.items(),
                key=lambda kv: (kv[1], global_counts.get(kv[0], 0), kv[0]),
            )[0]

            for rr, cc in cells:
                source_color[rr, cc] = representative

    rep_colors = sorted(
        {int(source_color[r, c]) for r in range(H) for c in range(W) if pattern_mask[r, c]}
    )
    if not rep_colors:
        return grid

    bit_of = {c: 1 << i for i, c in enumerate(rep_colors)}
    color_of_bit = {1 << i: c for i, c in enumerate(rep_colors)}

    # ------------------------------------------------------------------
    # 5. Multi-source BFS from all pattern components.
    # ------------------------------------------------------------------
    INF = 10 ** 9
    dist = np.full((H, W), INF, dtype=int)
    best_color = np.full((H, W), -1, dtype=int)
    mask_arr = np.zeros((H, W), dtype=np.uint64)

    q = deque()
    for r in range(H):
        for c in range(W):
            if pattern_mask[r, c]:
                col = int(source_color[r, c])
                m = bit_of[col]
                if dist[r, c] == INF:
                    dist[r, c] = 0
                    best_color[r, c] = col
                    mask_arr[r, c] = m
                    q.append((r, c))
                else:
                    if col != int(best_color[r, c]):
                        mask_arr[r, c] |= m

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        col = int(best_color[r, c])
        m = int(mask_arr[r, c])

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                nd = d + 1

                if nd < int(dist[nr, nc]):
                    dist[nr, nc] = nd
                    best_color[nr, nc] = col
                    mask_arr[nr, nc] = m
                    q.append((nr, nc))
                elif nd == int(dist[nr, nc]):
                    if col != int(best_color[nr, nc]):
                        mask_arr[nr, nc] |= m

    def choose_from_mask(mask):
        """Choose a color from a bitmask of tied representative colors."""
        mask = int(mask)
        if mask == 0:
            return global_majority

        # Single tied color.
        if mask & (mask - 1) == 0:
            return color_of_bit.get(mask, global_majority)

        best_score = None
        best_col = global_majority
        mm = mask

        while mm:
            lsb = mm & -mm
            col = color_of_bit.get(lsb, -1)
            if col != -1:
                score = (global_counts.get(col, 0), col)
                if best_score is None or score > best_score:
                    best_score = score
                    best_col = col
            mm ^= lsb

        return best_col

    # ------------------------------------------------------------------
    # 6. Build output.
    # ------------------------------------------------------------------
    out = grid.copy()
    protect = np.zeros((H, W), dtype=bool)
    center = None

    if special_top_left is not None:
        r, c = special_top_left
        center = (r + 1, c + 1)

        # Preserve the border of the special 3x3 block.
        for dr in range(3):
            for dc in range(3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if not (dr == 1 and dc == 1):
                        if grid[rr, cc] == frame_color:
                            protect[rr, cc] = True

    frame_mask = (grid == frame_color) & (~protect)
    rs, cs = np.where(frame_mask)

    for r, c in zip(rs, cs):
        if int(dist[r, c]) < INF:
            out[r, c] = choose_from_mask(mask_arr[r, c])
        else:
            out[r, c] = global_majority

    # Fill the special center with the global majority pattern color.
    if center is not None:
        cr, cc = center
        if 0 <= cr < H and 0 <= cc < W:
            out[cr, cc] = global_majority

    return out