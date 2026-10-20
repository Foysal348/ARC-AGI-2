import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    - Color 7 is treated as the wall/background color.
    - All non-7 cells form "objects".
    - The connected component containing any color-3 cell is always preserved.
    - For every other 4-connected non-7 component:
        * If the component touches the grid border, preserve it.
        * Otherwise, erase it by replacing all of its cells with 7.

    This keeps the special 3-shape and all border-touching non-7 regions,
    while removing only interior non-7 regions that do not contain the key
    color 3.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.astype(int, copy=True)
    h, w = out.shape

    WALL = 7
    KEY = 3

    open_mask = out != WALL

    comp_id = np.zeros((h, w), dtype=int)
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not open_mask[r, c] or comp_id[r, c] != 0:
                continue

            next_id += 1
            stack = [(r, c)]
            comp_id[r, c] = next_id
            touches_border = False

            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if open_mask[nr, nc] and comp_id[nr, nc] == 0:
                            comp_id[nr, nc] = next_id
                            stack.append((nr, nc))

    preserve = [False] * (next_id + 1)
    for cid in range(1, next_id + 1):
        mask = comp_id == cid
        if np.any(out[mask] == KEY):
            preserve[cid] = True
            continue

        # Check if this component touches the border.
        if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            preserve[cid] = True

    for r in range(h):
        for c in range(w):
            cid = comp_id[r, c]
            if cid != 0 and not preserve[cid]:
                out[r, c] = WALL

    return out