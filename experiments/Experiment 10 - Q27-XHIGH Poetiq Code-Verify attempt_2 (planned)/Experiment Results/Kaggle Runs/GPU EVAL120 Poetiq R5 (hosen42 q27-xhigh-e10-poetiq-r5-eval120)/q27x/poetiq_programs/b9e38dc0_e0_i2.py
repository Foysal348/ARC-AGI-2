import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. Find the background color: the most frequent color in the grid.
    2. Find the "wall" color: the most frequent non-background color.
       This is the color that forms the main outline/obstacle shape.
    3. Find the "seed" color: the least frequent non-background,
       non-wall color. If there is a tie, choose the color whose
       pixels are closest to the wall.
    4. Flood-fill from all original seed pixels through background cells,
       using 4-connectivity.  The wall and all other non-background colors
       act as barriers and remain unchanged.
    5. The reached background cells are recolored to the seed color.

    This matches the examples: a sparse seed marker is used to paint the
    connected background region associated with it, while the main outline
    and other non-background markers remain unchanged.
    """
    if grid.size == 0:
        return grid

    out = grid.copy()
    h, w = grid.shape

    # ------------------------------------------------------------------
    # 1. Background = most frequent color.
    # ------------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    # If the grid has only one color, nothing changes.
    if len(colors) == 1:
        return out

    # ------------------------------------------------------------------
    # 2. Wall = most frequent non-background color.
    # ------------------------------------------------------------------
    non_bg_mask = grid != background
    if not np.any(non_bg_mask):
        return out

    non_bg_colors, non_bg_counts = np.unique(grid[non_bg_mask], return_counts=True)
    wall = int(non_bg_colors[np.argmax(non_bg_counts)])

    # ------------------------------------------------------------------
    # 3. Seed = least frequent non-background, non-wall color.
    #    Tie-break: choose the color closest to any wall pixel.
    # ------------------------------------------------------------------
    candidates = [c for c in non_bg_colors if c != wall]
    if not candidates:
        return out

    wall_coords = np.argwhere(grid == wall)

    def min_dist_to_wall(color):
        coords = np.argwhere(grid == color)
        best = float("inf")
        for r1, c1 in coords:
            for r2, c2 in wall_coords:
                d = abs(int(r1 - r2)) + abs(int(c1 - c2))
                if d < best:
                    best = d
        return best

    # Minimum count first; then minimum distance to wall.
    best_seed = None
    best_key = None
    for c in candidates:
        cnt = int(np.sum(grid == c))
        dist = min_dist_to_wall(int(c))
        key = (cnt, dist)
        if best_key is None or key < best_key:
            best_key = key
            best_seed = int(c)

    seed = best_seed

    # ------------------------------------------------------------------
    # 4. Flood-fill from all original seed pixels.
    #    Only background cells are converted; all other pixels stay fixed.
    # ------------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    seed_coords = np.argwhere(grid == seed)
    for r, c in seed_coords:
        visited[int(r), int(c)] = True
        q.append((int(r), int(c)))

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if grid[nr, nc] == background:
                    visited[nr, nc] = True
                    out[nr, nc] = seed
                    q.append((nr, nc))

    return out