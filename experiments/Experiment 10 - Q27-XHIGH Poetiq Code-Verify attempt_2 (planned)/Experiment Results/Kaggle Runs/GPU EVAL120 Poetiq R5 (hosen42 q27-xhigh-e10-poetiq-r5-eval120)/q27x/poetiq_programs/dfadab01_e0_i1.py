import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform single-pixel trigger colors into 4x4 tiles.

    Triggers:
        2 -> color 4, full 4x4 frame
        3 -> color 1, frame without corners
        5 -> color 6, diagonal 2x2 blocks
        8 -> color 7, frame with corners and side pairs

    A trigger is used only if its 4x4 tile fits in the grid and its
    anchored 4x4 area does not contain another trigger.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return np.zeros_like(grid)

    H, W = grid.shape
    out = np.zeros((H, W), dtype=grid.dtype)

    # Fixed 4x4 patterns, stored as 0/1 masks.
    P2 = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ], dtype=grid.dtype)

    P3 = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ], dtype=grid.dtype)

    P5 = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=grid.dtype)

    P8 = np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ], dtype=grid.dtype)

    lookup = {
        2: (4, P2),
        3: (1, P3),
        5: (6, P5),
        8: (7, P8)
    }

    # Only these colors act as single-pixel triggers.
    trigger_mask = (
        (grid == 2) |
        (grid == 3) |
        (grid == 5) |
        (grid == 8)
    )

    if not trigger_mask.any():
        return out

    active = []

    for r, c in np.argwhere(trigger_mask):
        r = int(r)
        c = int(c)
        color = int(grid[r, c])

        # Tile must fit completely inside the grid.
        if r + 3 >= H or c + 3 >= W:
            continue

        # If this 4x4 area contains another trigger, do not place this one.
        if trigger_mask[r:r + 4, c:c + 4].sum() > 1:
            continue

        active.append((r, c, color))

    # Prefer lower/right triggers if any overlap remains.
    active.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r, c, color in active:
        out_color, pattern = lookup[color]
        region = out[r:r + 4, c:c + 4]
        region[pattern != 0] = out_color

    return out