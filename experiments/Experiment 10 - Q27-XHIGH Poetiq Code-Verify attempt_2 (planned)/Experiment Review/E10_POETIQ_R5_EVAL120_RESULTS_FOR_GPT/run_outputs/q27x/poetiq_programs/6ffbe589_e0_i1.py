import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC input into the compact assembled object.

    The rule implemented here:
      1. Ignore small stray pieces that are far from the main object.
      2. Find the main assembly by dilating non-zero pixels slightly and
         keeping the largest connected region.
      3. Crop to the bounding box of the main assembly.
      4. Remove completely empty rows and columns.

    For this challenge, the main assembly is the central 3-ring with the
    single 4 in its center plus the 5 fragments surrounding it.  The
    small distant 3/4/5 fragments are strays and are discarded.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    H, W = grid.shape

    # --------------------------------------------------------------------
    # 1. Non-zero mask
    # --------------------------------------------------------------------
    nz = grid > 0

    # --------------------------------------------------------------------
    # 2. Dilate the non-zero mask by one pixel (8-neighbourhood)
    #    to bridge one-cell gaps between pieces of the same object.
    # --------------------------------------------------------------------
    def dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
        out = mask.copy()
        for _ in range(iters):
            out = (
                out.copy()
                | np.roll(out, 1, axis=0)
                | np.roll(out, -1, axis=0)
                | np.roll(out, 1, axis=1)
                | np.roll(out, -1, axis=1)
            )
        return out

    dilated = dilate(nz, iters=1)

    # --------------------------------------------------------------------
    # 3. Connected components on the dilated mask
    # --------------------------------------------------------------------
    seen = np.zeros_like(dilated, dtype=bool)
    components = []

    for i in range(H):
        for j in range(W):
            if not dilated[i, j] or seen[i, j]:
                continue

            stack = [(i, j)]
            seen[i, j] = True
            comp = []

            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if dilated[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))

            components.append(comp)

    if not components:
        return np.zeros_like(grid)

    # Choose the component with the largest original non-zero pixel count
    # inside its bounding box.
    best = None
    best_score = -1
    best_mask = np.zeros_like(nz, dtype=bool)

    for comp in components:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        # Count original non-zero pixels in this dilated component's bbox
        score = int(np.sum(nz[r0:r1 + 1, c0:c1 + 1]))
        if score > best_score:
            best_score = score
            best = (r0, r1, c0, c1)

    r0, r1, c0, c1 = best

    # --------------------------------------------------------------------
    # 4. Crop to the main assembly bounding box
    # --------------------------------------------------------------------
    cropped = grid[r0:r1 + 1, c0:c1 + 1].copy()

    # --------------------------------------------------------------------
    # 5. Remove fully empty rows and columns
    # --------------------------------------------------------------------
    row_nonzero = np.any(cropped > 0, axis=1)
    col_nonzero = np.any(cropped > 0, axis=0)

    cropped = cropped[row_nonzero, :][:, col_nonzero]

    return cropped