import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
      1. Selecting the largest 8-connected non-zero component.
      2. Cropping to that component's bounding box.
      3. For every color present in the crop, choosing one of the four
         rotations (0, 90 CW, 180, 270 CW) that is lexicographically
         smallest when the rotated binary mask is read row-major with
         0 < 1.
      4. Overwriting the crop with the chosen rotated masks.

    This reproduces the provided examples:
      - the main object is kept and cropped;
      - scattered noise outside the largest component is removed;
      - each color layer is canonically oriented.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # ---------- 1. Largest 8-connected non-zero component ----------
    visited = np.zeros((h, w), dtype=bool)
    best_mask = None
    best_size = -1

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = []

                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and not visited[nr, nc]
                                and grid[nr, nc] != 0
                            ):
                                visited[nr, nc] = True
                                q.append((nr, nc))

                if len(comp) > best_size:
                    best_size = len(comp)
                    best_mask = np.zeros((h, w), dtype=bool)
                    for rr, cc in comp:
                        best_mask[rr, cc] = True

    if best_mask is None:
        return np.zeros_like(grid)

    # ---------- 2. Crop to bounding box of that component ----------
    rows = np.any(best_mask, axis=1)
    cols = np.any(best_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
    out = np.zeros_like(crop)

    # ---------- 3. Canonical rotation for each color ----------
    def rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
        """Rotate mask k times by 90 degrees clockwise."""
        m = mask
        for _ in range(k % 4):
            m = np.rot90(m, k=-1)
        return m

    def key(mask: np.ndarray):
        """
        Lexicographic key over the row-major flattened mask,
        treating 0 as smaller than 1.
        """
        return mask.astype(np.uint8).flatten()

    colors = [int(c) for c in np.unique(crop) if c != 0]

    for color in colors:
        mask = (crop == color)

        best = None
        best_key = None
        for k in range(4):
            rotated = rotate_mask(mask, k)
            cur_key = key(rotated)

            if best_key is None or cur_key < best_key:
                best_key = cur_key
                best = rotated

        out[best] = color

    return out