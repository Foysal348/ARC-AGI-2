import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Use a solid rectangular legend to determine color substitutions.

    The legend is a filled rectangular component of non-zero cells.
    Its orientation determines how to read color pairs:
      - if taller than wide: read horizontally (left/right),
      - if wider than tall: read vertically (top/bottom),
      - if square: use the direction away from the nearest grid edge.

    For each color in the legend, the mapped color is the color one step
    in the chosen direction. Enclosed zero regions whose border color has
    a mapping are filled with that mapped color.
    """
    if grid is None:
        return np.array([])

    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    if H == 0 or W == 0:
        return out

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # ---------------------------------------------------------------
    # 1. Find the legend: a solid rectangular non-zero component.
    # ---------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    candidates = []

    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0 and not visited[i, j]:
                q = deque([(i, j)])
                visited[i, j] = True
                cells = []
                min_r = max_r = i
                min_c = max_c = j

                while q:
                    r, c = q.popleft()
                    cells.append((r, c))

                    if r < min_r:
                        min_r = r
                    if r > max_r:
                        max_r = r
                    if c < min_c:
                        min_c = c
                    if c > max_c:
                        max_c = c

                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and grid[nr, nc] != 0
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            q.append((nr, nc))

                h = max_r - min_r + 1
                w = max_c - min_c + 1

                # A solid rectangle has no holes inside its bounding box.
                if len(cells) == h * w and len(cells) >= 2:
                    n_colors = len({grid[r, c] for r, c in cells})
                    candidates.append(
                        (n_colors, len(cells), min_r, min_c, max_r, max_c, h, w)
                    )

    if not candidates:
        return out

    # Prefer a multi-color rectangle, then more colors, then larger area.
    multi_color = [c for c in candidates if c[0] >= 2]
    pool = multi_color if multi_color else candidates

    n_colors, area, min_r, min_c, max_r, max_c, h, w = max(
        pool, key=lambda x: (x[0], x[1])
    )

    # ---------------------------------------------------------------
    # 2. Determine the legend mapping direction.
    # ---------------------------------------------------------------
    if h == 1:
        axis = "horizontal"
    elif w == 1:
        axis = "vertical"
    elif h > w:
        axis = "horizontal"
    elif w > h:
        axis = "vertical"
    else:
        # Square legend: choose the axis away from the closer edge.
        top_dist = min_r
        bottom_dist = H - 1 - max_r
        left_dist = min_c
        right_dist = W - 1 - max_c

        if min(top_dist, bottom_dist) < min(left_dist, right_dist):
            axis = "vertical"
        elif min(left_dist, right_dist) < min(top_dist, bottom_dist):
            axis = "horizontal"
        else:
            axis = "vertical"

    if axis == "horizontal":
        left_dist = min_c
        right_dist = W - 1 - max_c
        dr, dc = (0, 1) if left_dist <= right_dist else (0, -1)
    else:
        top_dist = min_r
        bottom_dist = H - 1 - max_r
        dr, dc = (1, 0) if top_dist <= bottom_dist else (-1, 0)

    # ---------------------------------------------------------------
    # 3. Build color mapping from the legend.
    # ---------------------------------------------------------------
    raw_mapping = {}

    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            src = grid[r, c]
            nr, nc = r + dr, c + dc

            if min_r <= nr <= max_r and min_c <= nc <= max_c:
                dst = grid[nr, nc]
                if src not in raw_mapping:
                    raw_mapping[src] = Counter()
                raw_mapping[src][dst] += 1

    mapping = {}
    for src, counts in raw_mapping.items():
        # If a color maps to multiple targets, use the most common one.
        dst = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        mapping[src] = dst

    if not mapping:
        return out

    # ---------------------------------------------------------------
    # 4. Find enclosed zero regions and fill them according to mapping.
    # ---------------------------------------------------------------
    visited_zero = np.zeros((H, W), dtype=bool)

    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0 or visited_zero[i, j]:
                continue

            q = deque([(i, j)])
            visited_zero[i, j] = True
            cells = []
            touches_border = False
            border_colors = Counter()

            while q:
                r, c = q.popleft()
                cells.append((r, c))

                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    touches_border = True

                for dr2, dc2 in dirs:
                    nr, nc = r + dr2, c + dc2
                    if 0 <= nr < H and 0 <= nc < W:
                        val = grid[nr, nc]
                        if val == 0:
                            if not visited_zero[nr, nc]:
                                visited_zero[nr, nc] = True
                                q.append((nr, nc))
                        else:
                            border_colors[val] += 1

            if touches_border or not border_colors:
                continue

            # Determine the hole's border color.
            hole_color = None

            if len(border_colors) == 1:
                hole_color = next(iter(border_colors.keys()))
            else:
                # Conservative fallback: only use a strict majority mapped color.
                mapped_border = [c for c in border_colors if c in mapping]
                if len(mapped_border) == 1:
                    c = mapped_border[0]
                    if border_colors[c] * 2 > sum(border_colors.values()):
                        hole_color = c

            if hole_color is None or hole_color not in mapping:
                continue

            fill_color = mapping[hole_color]
            for r, c in cells:
                out[r, c] = fill_color

    return out