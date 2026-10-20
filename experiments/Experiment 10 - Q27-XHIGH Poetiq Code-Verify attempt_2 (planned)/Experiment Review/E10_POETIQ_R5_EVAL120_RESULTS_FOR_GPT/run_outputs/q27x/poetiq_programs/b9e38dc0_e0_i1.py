import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. Identify the background color as the most common color in the input grid.
    2. Identify the "boundary" color as the most common non-background color.
       In the examples this is the color forming the large enclosing outline.
    3. Identify the "fill" color among the remaining non-background,
       non-boundary colors. The fill color is chosen as the color whose
       pixels are closest (minimum Manhattan distance) to the boundary color.
       If there is a tie, the first encountered color in row-major order is
       used.
    4. Flood-fill the 4-connected background region containing the fill color.
       All reached background cells are recolored to the fill color.
       The boundary color and all other colors remain unchanged.

    This matches the shown examples: the small marker color inside/near the
    boundary is used to paint the connected background region associated with it.
    """
    if grid.size == 0:
        return grid

    # Work on a copy so the original grid is not modified.
    out = grid.copy()

    h, w = grid.shape

    # ---------------------------------------------------------------
    # 1. Background = most frequent color.
    # ---------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    # ---------------------------------------------------------------
    # 2. Boundary = most frequent non-background color.
    #    If all pixels are background, nothing to do.
    # ---------------------------------------------------------------
    non_bg_mask = grid != background
    if not np.any(non_bg_mask):
        return out

    non_bg_colors, non_bg_counts = np.unique(grid[non_bg_mask], return_counts=True)
    boundary = int(non_bg_colors[np.argmax(non_bg_counts)])

    # ---------------------------------------------------------------
    # 3. Choose fill color.
    #    Candidates are non-background, non-boundary colors.
    #    Pick the candidate closest to any boundary pixel.
    # ---------------------------------------------------------------
    candidate_colors = [c for c in non_bg_colors if c != boundary]
    if not candidate_colors:
        return out

    boundary_coords = np.argwhere(grid == boundary)

    best_color = candidate_colors[0]
    best_dist = float("inf")

    for color in candidate_colors:
        color_coords = np.argwhere(grid == color)
        for (r1, c1) in color_coords:
            # Minimum Manhattan distance to any boundary pixel.
            # The grids are small, so direct checking is fine and robust.
            for (r2, c2) in boundary_coords:
                d = abs(int(r1 - r2)) + abs(int(c1 - c2))
                if d < best_dist:
                    best_dist = d
                    best_color = int(color)

    fill_color = best_color

    # ---------------------------------------------------------------
    # 4. Flood fill from all original fill-color pixels.
    #    We only turn background pixels into the fill color.
    #    All non-background pixels remain untouched.
    # ---------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    # Seed the queue with every original fill-color pixel.
    fill_coords = np.argwhere(grid == fill_color)
    for r, c in fill_coords:
        visited[r, c] = True
        q.append((int(r), int(c)))

    # 4-neighbor directions.
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while q:
        r, c = q.popleft()

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc

            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if grid[nr, nc] == background:
                    visited[nr, nc] = True
                    out[nr, nc] = fill_color
                    q.append((nr, nc))

    return out