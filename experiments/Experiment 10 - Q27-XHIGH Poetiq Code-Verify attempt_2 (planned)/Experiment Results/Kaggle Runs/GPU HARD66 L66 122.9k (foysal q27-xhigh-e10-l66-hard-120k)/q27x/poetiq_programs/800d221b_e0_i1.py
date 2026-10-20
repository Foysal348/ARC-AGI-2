import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct non-background colors under a frame/separator color.

    Rule implemented:
      1. The background is the most frequent color.
      2. The frame color is the color of the most central monochromatic 3x3 block
         among non-background colors. A fallback is used if no such block exists.
      3. The central solid 3x3 frame block keeps its border and gets its center
         filled with the global majority pattern color.
      4. All other frame cells are replaced by the majority color of the nearest
         connected component of pattern colors. Ties use the global majority.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    colors, counts = np.unique(grid, return_counts=True)
    max_count = counts.max()
    background = int(colors[counts == max_count][0])
    non_background = [int(c) for c in colors if int(c) != background]

    if not non_background:
        return grid

    def component_cells(mask):
        """Return a list of connected components (4-connected) as cell lists."""
        labels = np.zeros(mask.shape, dtype=int)
        components = []
        cid = 0

        for i in range(H):
            for j in range(W):
                if mask[i, j] and labels[i, j] == 0:
                    cid += 1
                    q = deque([(i, j)])
                    labels[i, j] = cid
                    cells = []

                    while q:
                        r, c = q.popleft()
                        cells.append((r, c))

                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if mask[nr, nc] and labels[nr, nc] == 0:
                                    labels[nr, nc] = cid
                                    q.append((nr, nc))

                    components.append(cells)

        return components

    def max_bbox_area(mask):
        """Largest bounding-box area among connected components of mask."""
        best = 0
        best_cells = 0
        for cells in component_cells(mask):
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if area > best or (area == best and len(cells) > best_cells):
                best = area
                best_cells = len(cells)
        return best

    def all9_mask(color):
        """Boolean mask for top-left positions of all-9-cell 3x3 blocks of color."""
        a = grid == color
        return (
            a[0:H - 2, 0:W - 2] & a[0:H - 2, 1:W - 1] & a[0:H - 2, 2:W] &
            a[1:H - 1, 0:W - 2] & a[1:H - 1, 1:W - 1] & a[1:H - 1, 2:W] &
            a[2:H, 0:W - 2] & a[2:H, 1:W - 1] & a[2:H, 2:W]
        )

    def perimeter3x3_mask(color):
        """Boolean mask for top-left positions of 3x3 perimeters of color."""
        a = grid == color
        return (
            a[0:H - 2, 0:W - 2] & a[0:H - 2, 1:W - 1] & a[0:H - 2, 2:W] &
            a[2:H, 0:W - 2] & a[2:H, 1:W - 1] & a[2:H, 2:W] &
            a[0:H - 2, 0:W - 2] & a[1:H - 1, 0:W - 2] & a[2:H, 0:W - 2] &
            a[0:H - 2, 2:W] & a[1:H - 1, 2:W] & a[2:H, 2:W]
        )

    # Identify frame color and the special central 3x3 solid block.
    frame_color = None
    special_pos = None

    if H >= 3 and W >= 3:
        center2r = H - 1
        center2c = W - 1
        best_score = None

        for c in non_background:
            m = all9_mask(c)
            if not m.any():
                continue

            ys, xs = np.where(m)
            dists = (2 * (ys + 1) - center2r) ** 2 + (2 * (xs + 1) - center2c) ** 2
            idx = int(np.argmin(dists))
            min_dist = int(dists[idx])
            area = max_bbox_area(grid == c)
            filled_count = int(m.sum())

            # Prefer closest to center, then larger structure, then more occurrences.
            score = (min_dist, -area, -filled_count)
            if best_score is None or score < best_score:
                best_score = score
                frame_color = int(c)
                special_pos = (int(ys[idx]), int(xs[idx]))

    # Fallback: if no solid 3x3 block is found, choose the largest non-background color.
    if frame_color is None:
        best = None
        best_c = None
        for c in non_background:
            area = max_bbox_area(grid == c)
            cell_count = int((grid == c).sum())
            score = (area, cell_count)
            if best is None or score > best:
                best = score
                best_c = int(c)
        frame_color = best_c
        special_pos = None

    if frame_color is None:
        return grid

    pattern_colors = [int(c) for c in colors if int(c) != background and int(c) != frame_color]
    if not pattern_colors:
        return grid

    pattern_mask = np.isin(grid, pattern_colors)
    pattern_components = component_cells(pattern_mask)

    if not pattern_components:
        return grid

    global_counts = Counter()
    component_info = []

    for cells in pattern_components:
        cnt = Counter()
        for r, c in cells:
            v = int(grid[r, c])
            cnt[v] += 1
            global_counts[v] += 1

        best_n = max(cnt.values())
        dominant = min(v for v, n in cnt.items() if n == best_n)
        component_info.append((cells, dominant))

    best_n = max(global_counts.values())
    global_majority = min(v for v, n in global_counts.items() if n == best_n)

    protect = np.zeros((H, W), dtype=bool)
    center_mask = np.zeros((H, W), dtype=bool)

    # Preserve the border of the special central 3x3 frame block.
    if special_pos is not None:
        r, c = special_pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                protect[r + dr, c + dc] = True
        center_mask[r + 1, c + 1] = True

    out = grid.copy()

    if np.any(center_mask):
        out[center_mask] = global_majority

    assign_mask = (grid == frame_color) & (~protect) & (~center_mask)

    if np.any(assign_mask) and component_info:
        INF = H * W + 100
        best_dist = np.full((H, W), INF, dtype=int)
        best_color = np.zeros((H, W), dtype=int)
        tie = np.zeros((H, W), dtype=bool)

        for cells, dom in component_info:
            dist = np.full((H, W), INF, dtype=int)
            q = deque()

            for r, c in cells:
                if dist[r, c] != 0:
                    dist[r, c] = 0
                    q.append((r, c))

            while q:
                r, c = q.popleft()
                nd = dist[r, c] + 1

                if r + 1 < H and dist[r + 1, c] > nd:
                    dist[r + 1, c] = nd
                    q.append((r + 1, c))
                if r - 1 >= 0 and dist[r - 1, c] > nd:
                    dist[r - 1, c] = nd
                    q.append((r - 1, c))
                if c + 1 < W and dist[r, c + 1] > nd:
                    dist[r, c + 1] = nd
                    q.append((r, c + 1))
                if c - 1 >= 0 and dist[r, c - 1] > nd:
                    dist[r, c - 1] = nd
                    q.append((r, c - 1))

            less = dist < best_dist
            best_dist[less] = dist[less]
            best_color[less] = dom
            tie[less] = False

            equal = (dist == best_dist) & (best_color != dom)
            tie[equal] = True

        assigned = np.where(tie, global_majority, best_color)
        assigned = np.where(best_dist >= INF, global_majority, assigned)

        out[assign_mask] = assigned[assign_mask]

    return out