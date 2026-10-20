import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. Extract the relevant object:
       - Consider all non-zero pixels.
       - Find the largest 8-connected component.
       - Crop the grid to that component's bounding box.

    2. Canonicalize each color layer independently inside the cropped box:
       - For each non-zero color present in the crop, try the four rotations:
         0°, 90° clockwise, 180°, 270° clockwise.
       - Choose the rotation whose topmost non-zero row has the lexicographically
         largest binary pattern.
         Ties are broken by:
           a) larger maximum column in that top row,
           b) lexicographically largest full rotated binary matrix.

    3. Build the output by placing every color's chosen rotated pattern into
       the common cropped bounding box. Background is 0.

    This matches all provided examples:
    - Example 1: main 10x10 object, colors 4/1/2 are canonicalized.
    - Example 2: main 10x10 object, colors 5/3/4 are canonicalized.
    - Example 3: main 13x13 object, colors 3/8/6 are canonicalized.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # ---------- 1. Find largest 8-connected non-zero component ----------
    visited = np.zeros((h, w), dtype=bool)
    best_mask = None
    best_size = -1

    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                q = deque([(i, j)])
                visited[i, j] = True
                comp = []
                while q:
                    r, c = q.popleft()
                    comp.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if not visited[nr, nc] and grid[nr, nc] != 0:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))

                if len(comp) > best_size:
                    best_size = len(comp)
                    best_mask = np.zeros((h, w), dtype=bool)
                    for r, c in comp:
                        best_mask[r, c] = True

    if best_mask is None:
        return np.zeros_like(grid)

    # Bounding box of that component
    rows = np.any(best_mask, axis=1)
    cols = np.any(best_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
    H, W = crop.shape

    # ---------- 2. Helper: rotate a binary mask by 0/90/180/270 ----------
    def rotate_mask(mask, k):
        # k = number of 90° clockwise rotations
        m = mask
        for _ in range(k % 4):
            m = np.rot90(m, k=-1)  # clockwise
        return m

    def toprow_key(mask):
        """
        Key for choosing the canonical rotation of one color layer.
        We want the topmost non-zero row to be lexicographically largest.
        """
        nz = np.any(mask, axis=1)
        if not np.any(nz):
            return ((0,), 0, mask.tobytes())

        first = np.where(nz)[0][0]
        top = tuple(int(x) for x in mask[first].tolist())

        # For tie-breaking, also use the largest column in the top row
        cols_top = np.where(mask[first])[0]
        max_col = int(cols_top[-1]) if len(cols_top) else 0

        # Full matrix as final deterministic tie-breaker
        full = mask.astype(np.int8).tobytes()
        return (top, max_col, full)

    # ---------- 3. Canonicalize each color independently ----------
    out = np.zeros_like(crop)

    colors = [c for c in np.unique(crop) if c != 0]
    for color in colors:
        mask = (crop == color)

        best_rot = None
        best_key = None
        for k in range(4):
            rot = rotate_mask(mask, k)
            key = toprow_key(rot)
            if best_key is None or key > best_key:
                best_key = key
                best_rot = rot

        out[best_rot] = color

    return out