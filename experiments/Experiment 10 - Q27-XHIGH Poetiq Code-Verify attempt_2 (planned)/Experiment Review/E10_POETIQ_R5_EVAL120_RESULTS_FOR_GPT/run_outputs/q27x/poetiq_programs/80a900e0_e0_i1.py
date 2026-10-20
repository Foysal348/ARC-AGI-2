import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extend free diagonal line endpoints.

    Background colors are 0 and 1. For every non-background cell that has
    exactly one diagonal neighbor of the same color, treat it as a free
    endpoint of a diagonal segment. From that endpoint, cast rays in the
    two diagonal directions perpendicular to the segment. Rays stop at the
    first non-background input pixel or the grid boundary.

    Original non-background pixels are preserved. Extensions are added only
    on background cells.
    """
    arr = np.asarray(grid, dtype=int)

    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()

    h, w = arr.shape
    out = arr.copy()

    # In these puzzles, the checkerboard background consists of 0s and 1s.
    background = {0, 1}

    # Only consider colors that are not part of the background.
    special_mask = ~np.isin(arr, [0, 1])
    if not special_mask.any():
        return out

    colors = np.unique(arr[special_mask])

    # Diagonal directions in the original grid.
    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Store additions first so that newly added pixels do not block rays.
    additions = {}

    for color in colors:
        for r in range(h):
            for c in range(w):
                if arr[r, c] != color:
                    continue

                # Find same-color diagonal neighbors.
                same_neighbors = []
                for dr, dc in diagonal_dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == color:
                        same_neighbors.append((nr, nc, dr, dc))

                # A free endpoint has exactly one same-color diagonal neighbor.
                if len(same_neighbors) != 1:
                    continue

                _, _, dr, dc = same_neighbors[0]

                # If the segment is along the main diagonal, perpendicular
                # directions are the anti-diagonals, and vice versa.
                if dr == dc:
                    perpendicular_dirs = [(-1, 1), (1, -1)]
                else:
                    perpendicular_dirs = [(-1, -1), (1, 1)]

                for pr, pc in perpendicular_dirs:
                    rr, cc = r + pr, c + pc

                    while 0 <= rr < h and 0 <= cc < w:
                        # Stop at any original non-background pixel.
                        if arr[rr, cc] not in background:
                            break

                        key = (rr, cc)
                        if key not in additions:
                            additions[key] = color

                        rr += pr
                        cc += pc

    # Apply additions without overwriting original non-background pixels.
    for (rr, cc), color in additions.items():
        if arr[rr, cc] in background:
            out[rr, cc] = color

    return out