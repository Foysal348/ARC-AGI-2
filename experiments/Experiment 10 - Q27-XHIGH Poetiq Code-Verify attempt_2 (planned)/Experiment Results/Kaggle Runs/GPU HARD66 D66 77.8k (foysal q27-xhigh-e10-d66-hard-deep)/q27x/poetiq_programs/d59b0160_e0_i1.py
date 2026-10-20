import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    - Color 7 is treated as wall/background.
    - All other cells are treated as open cells.
    - Connected components of open cells are found using 4-connectivity.
    - The open component containing any color-3 cell is always preserved.
    - Every other open component is erased by turning all of its cells to 7.

    This matches the examples: the component containing the fixed 3-shape
    is left unchanged, while the other open regions are filled with 7.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.astype(int, copy=True)
    h, w = out.shape

    WALL = 7
    KEY = 3

    # Cells considered "open" for component labeling.
    open_mask = out != WALL

    # Find 4-connected components among open cells.
    comp_id = np.zeros((h, w), dtype=int)
    next_id = 0

    for r in range(h):
        for c in range(w):
            if open_mask[r, c] and comp_id[r, c] == 0:
                next_id += 1
                stack = [(r, c)]
                comp_id[r, c] = next_id
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if open_mask[nr, nc] and comp_id[nr, nc] == 0:
                                comp_id[nr, nc] = next_id
                                stack.append((nr, nc))

    # Identify components that contain the key color 3.
    preserve = [False] * (next_id + 1)
    for r in range(h):
        for c in range(w):
            if out[r, c] == KEY:
                cid = comp_id[r, c]
                if cid:
                    preserve[cid] = True

    # Erase all non-preserved open components.
    for r in range(h):
        for c in range(w):
            cid = comp_id[r, c]
            if cid and not preserve[cid]:
                out[r, c] = WALL

    return out