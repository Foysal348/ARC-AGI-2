import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find a monochrome mask pattern and a multi-colour tile pattern in the input.
    The mask's bounding box defines the number of rows/columns of output tiles.
    For every non-background pixel of the mask, the tile pattern is placed in
    the corresponding tile position. All other tile positions are filled with
    the background colour.
    """
    if grid is None:
        return np.zeros((0, 0), dtype=int)

    grid = np.array(grid, copy=True)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Background: prefer a uniform border, otherwise the most frequent colour.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    bvals, _ = np.unique(border, return_counts=True)
    if bvals.size == 1:
        bg = int(bvals[0])

    nonbg = grid != bg
    if not nonbg.any():
        return np.full_like(grid, bg)

    # Find 8-connected components of non-background pixels.
    labels = np.zeros((H, W), dtype=int)
    next_label = 0
    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    for i in range(H):
        for j in range(W):
            if nonbg[i, j] and labels[i, j] == 0:
                next_label += 1
                stack = [(i, j)]
                labels[i, j] = next_label

                while stack:
                    r, c = stack.pop()
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if nonbg[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = next_label
                                stack.append((nr, nc))

    comps = []
    for lab in range(1, next_label + 1):
        rs, cs = np.where(labels == lab)
        minr, maxr = int(rs.min()), int(rs.max())
        minc, maxc = int(cs.min()), int(cs.max())
        colors = set(int(v) for v in grid[rs, cs]) - {bg}
        comps.append({
            "bbox": (minr, maxr, minc, maxc),
            "colors": colors,
            "count": int(rs.size),
        })

    mask_bbox = None
    motif_bbox = None

    # Primary case: exactly two objects, one monochrome mask and one multi-colour tile.
    if len(comps) == 2:
        singles = [c for c in comps if len(c["colors"]) == 1]
        multis = [c for c in comps if len(c["colors"]) > 1]

        if len(singles) == 1 and len(multis) == 1:
            mask_bbox = singles[0]["bbox"]
            motif_bbox = multis[0]["bbox"]
        elif len(singles) == 2:
            # Fallback if both are monochrome: use the smaller one as the mask.
            a, b = comps
            key_a = (a["count"], a["bbox"][1] - a["bbox"][0] + 1,
                     a["bbox"][3] - a["bbox"][2] + 1)
            key_b = (b["count"], b["bbox"][1] - b["bbox"][0] + 1,
                     b["bbox"][3] - b["bbox"][2] + 1)
            if key_a > key_b:
                a, b = b, a
            mask_bbox = a["bbox"]
            motif_bbox = b["bbox"]

    # Fallback: identify a pure mask colour whose bounding box contains no other
    # non-background colours. Choose the one whose removal leaves the smallest
    # bounding box for the remaining pixels (the tile).
    if mask_bbox is None:
        candidates = []

        for color in vals:
            color = int(color)
            if color == bg:
                continue

            color_mask = grid == color
            if not color_mask.any():
                continue

            rs, cs = np.where(color_mask)
            minr, maxr = int(rs.min()), int(rs.max())
            minc, maxc = int(cs.min()), int(cs.max())
            sub = grid[minr:maxr + 1, minc:maxc + 1]

            other = sub[(sub != bg) & (sub != color)]
            if other.size != 0:
                continue

            other_mask = nonbg & (grid != color)
            if other_mask.any():
                ors, ocs = np.where(other_mask)
                other_area = (int(ors.max()) - int(ors.min()) + 1) * \
                             (int(ocs.max()) - int(ocs.min()) + 1)
            else:
                other_area = 0

            candidates.append((other_area, -int(color_mask.sum()), color,
                               (minr, maxr, minc, maxc)))

        if candidates:
            candidates.sort()
            _, _, mask_color, mask_bbox = candidates[0]

            motif_mask = nonbg & (grid != mask_color)
            if motif_mask.any():
                rs, cs = np.where(motif_mask)
                motif_bbox = (int(rs.min()), int(rs.max()),
                              int(cs.min()), int(cs.max()))
            else:
                motif_bbox = mask_bbox

    # Last resort: smallest component as mask, largest as tile.
    if mask_bbox is None or motif_bbox is None:
        if not comps:
            return np.full_like(grid, bg)

        sorted_comps = sorted(comps, key=lambda c: c["count"])
        mask_bbox = sorted_comps[0]["bbox"]
        motif_bbox = sorted_comps[-1]["bbox"] if len(sorted_comps) > 1 else mask_bbox

    # Extract mask and tile.
    r1, r2, c1, c2 = mask_bbox
    mask = grid[r1:r2 + 1, c1:c2 + 1]

    r1, r2, c1, c2 = motif_bbox
    motif = grid[r1:r2 + 1, c1:c2 + 1]

    mh, mw = mask.shape
    ph, pw = motif.shape

    out = np.full((mh * ph, mw * pw), bg, dtype=grid.dtype)

    for i in range(mh):
        for j in range(mw):
            if mask[i, j] != bg:
                out[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = motif

    return out