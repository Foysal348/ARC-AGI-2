import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid using the 4-colored marker bar.

    The marker bar defines a strip:
      - if the bar is horizontal, the strip is its column range across all rows
      - if the bar is vertical, the strip is its row range across all columns

    Along the perpendicular direction, use a 6-step pattern aligned to the
    marker bar:
      - offset 0 or 4: set strip cells to 4
      - offset 2: set all original object-color cells in that row/column to 3,
                  and set strip cells to 3
      - offsets 1, 3, 5: set strip cells to the background color
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2 or g.size == 0:
        return g

    H, W = g.shape

    marker_mask = (g == 4)
    if not marker_mask.any():
        return g

    mr, mc = np.where(marker_mask)
    r0, r1 = int(mr.min()), int(mr.max())
    c0, c1 = int(mc.min()), int(mc.max())

    # Identify background and object colors.
    flat = g.ravel()
    colors, counts = np.unique(flat, return_counts=True)
    count_map = {int(c): int(cnt) for c, cnt in zip(colors, counts)}

    corner = np.array([g[0, 0], g[0, W - 1], g[H - 1, 0], g[H - 1, W - 1]])
    if len(np.unique(corner)) == 1 and int(corner[0]) != 4:
        bg = int(corner[0])
    else:
        corner_counts = {int(c): 0 for c in np.unique(corner)}
        for c in corner:
            corner_counts[int(c)] += 1

        candidates = [int(c) for c in colors if int(c) != 4]
        if candidates:
            bg = max(
                candidates,
                key=lambda c: (count_map[c], corner_counts.get(c, 0))
            )
        else:
            bg = 0

    obj = None
    for c in colors:
        c = int(c)
        if c == bg or c == 4:
            continue
        if obj is None or count_map[c] > count_map[obj]:
            obj = c

    out = g.copy()

    marker_h = r1 - r0 + 1
    marker_w = c1 - c0 + 1

    # Treat the longer side of the marker as its main axis.
    horizontal = marker_w >= marker_h

    if horizontal:
        anchor = (r0 + r1) // 2
        cols = slice(c0, c1 + 1)

        for r in range(H):
            d = (r - anchor) % 6

            if d == 0 or d == 4:
                out[r, cols] = 4
            elif d == 2:
                if obj is not None:
                    out[r, g[r] == obj] = 3
                out[r, cols] = 3
            else:
                out[r, cols] = bg

    else:
        anchor = (c0 + c1) // 2
        rows = slice(r0, r1 + 1)

        for c in range(W):
            d = (c - anchor) % 6

            if d == 0 or d == 4:
                out[rows, c] = 4
            elif d == 2:
                if obj is not None:
                    out[np.where(g[:, c] == obj)[0], c] = 3
                out[rows, c] = 3
            else:
                out[rows, c] = bg

    # Preserve the original marker cells.
    out[marker_mask] = 4
    return out