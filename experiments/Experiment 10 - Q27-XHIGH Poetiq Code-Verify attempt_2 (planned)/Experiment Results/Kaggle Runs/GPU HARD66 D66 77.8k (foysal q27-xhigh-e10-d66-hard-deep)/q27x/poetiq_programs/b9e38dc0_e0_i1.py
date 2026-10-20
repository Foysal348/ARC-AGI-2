import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The most frequent color is the background.
    2. The most frequent non-background color is the "wall" color.
    3. A "fill" color is chosen from the remaining colors:
         - prefer colors whose pixels lie inside the wall's bounding box;
         - then prefer the one with the largest connected component;
         - then prefer the one closest to the wall center.
       In all provided cases this selects the small marker color that should spread.
    4. Other non-background, non-wall, non-fill colors act as opaque obstacles.
       Each obstacle casts a straight shadow in the dominant cardinal direction
       away from the fill marker.  Shadowed background cells cannot be filled.
    5. Starting from all pixels of the fill color, flood-fill the background
       color using 4-connectivity.  The fill cannot cross:
           - wall pixels,
           - obstacle pixels,
           - shadowed background cells.
    6. All reached background pixels become the fill color.  Everything else
       remains unchanged.
    """
    grid = np.array(grid, dtype=int)
    h, w = grid.shape

    # --- 1. Background color: most common color ---
    colors = Counter(grid.ravel().tolist())
    if not colors:
        return grid
    bg_color = colors.most_common(1)[0][0]

    # --- 2. Wall color: most common non-background color ---
    non_bg_counts = Counter()
    for c, n in colors.items():
        if c != bg_color:
            non_bg_counts[c] = n

    if not non_bg_counts:
        return grid

    wall_color = non_bg_counts.most_common(1)[0][0]

    # --- 3. Choose fill color ---
    candidate_colors = [c for c in non_bg_counts if c not in (bg_color, wall_color)]
    if not candidate_colors:
        # No marker: return unchanged.
        return grid

    wall_rows, wall_cols = np.where(grid == wall_color)
    wall_bbox = (wall_rows.min(), wall_rows.max(), wall_cols.min(), wall_cols.max())
    wall_cy = (wall_bbox[0] + wall_bbox[1]) / 2.0
    wall_cx = (wall_bbox[2] + wall_bbox[3]) / 2.0

    def component_size(mask):
        """Size of the largest 4-connected component in mask."""
        if not mask.any():
            return 0
        visited = np.zeros_like(mask, dtype=bool)
        best = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    q = deque([(i, j)])
                    visited[i, j] = True
                    size = 0
                    while q:
                        r, c = q.popleft()
                        size += 1
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                                if mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))
                    best = max(best, size)
        return best

    def choose_fill_color():
        scored = []
        for c in candidate_colors:
            mask = (grid == c)
            rows, cols = np.where(mask)
            size = mask.sum()
            comp = component_size(mask)
            inside = bool(
                (rows > wall_bbox[0] + 1).all()
                and (rows < wall_bbox[1] - 1).all()
                and (cols > wall_bbox[2] + 1).all()
                and (cols < wall_bbox[3] - 1).all()
            )
            cy = rows.mean() if size else 0.0
            cx = cols.mean() if size else 0.0
            dist = (cy - wall_cy) ** 2 + (cx - wall_cx) ** 2
            # Higher score is better.
            score = (
                1_000_000 * int(inside)
                + 10_000 * comp
                + 10 * size
                - 0.001 * dist
            )
            scored.append((score, size, comp, c))
        scored.sort(reverse=True)
        return scored[0][3]

    fill_color = choose_fill_color()

    # --- 4. Build blocked mask: wall + all other non-background colors ---
    blocked = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            val = grid[r, c]
            if val != bg_color and val != fill_color:
                blocked[r, c] = True

    # --- 5. Cast shadows from secondary obstacles in dominant direction away from fill ---
    fill_rows, fill_cols = np.where(grid == fill_color)
    if fill_rows.size:
        fill_cy = fill_rows.mean()
        fill_cx = fill_cols.mean()
    else:
        fill_cy = h / 2.0
        fill_cx = w / 2.0

    shadow = np.zeros((h, w), dtype=bool)

    # All colors that are not background, wall, or fill are obstacles.
    obstacle_colors = [c for c in non_bg_counts if c not in (bg_color, wall_color, fill_color)]
    for c in obstacle_colors:
        for r, cc in zip(*np.where(grid == c)):
            dy = r - fill_cy
            dx = cc - fill_cx
            if abs(dy) >= abs(dx):
                dr, dc = (1, 0) if dy >= 0 else (-1, 0)
            else:
                dr, dc = (0, 1) if dx >= 0 else (0, -1)

            nr, nc = r + dr, cc + dc
            while 0 <= nr < h and 0 <= nc < w:
                if shadow[nr, nc]:
                    break
                shadow[nr, nc] = True
                # Stop shadow if it hits another solid pixel.
                if blocked[nr, nc]:
                    break
                nr += dr
                nc += dc

    blocked |= shadow

    # --- 6. Flood fill from fill-color pixels into background cells ---
    out = grid.copy()
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    for r, c in zip(*np.where(grid == fill_color)):
        visited[r, c] = True
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if not visited[nr, nc] and not blocked[nr, nc] and out[nr, nc] == bg_color:
                    visited[nr, nc] = True
                    out[nr, nc] = fill_color
                    q.append((nr, nc))

    return out