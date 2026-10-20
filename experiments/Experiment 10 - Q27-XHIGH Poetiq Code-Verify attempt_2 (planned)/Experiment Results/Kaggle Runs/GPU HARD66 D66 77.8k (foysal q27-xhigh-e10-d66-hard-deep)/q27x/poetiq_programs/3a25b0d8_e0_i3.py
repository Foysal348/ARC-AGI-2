import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the target object by collecting all rows that contain the
    'frame/wall' color, cropping each to the minimal horizontal window whose
    border cells are frame-colored, and stacking them in their original
    vertical order.

    In the observed examples, the desired output rows are already present in
    the input as short horizontal segments.  The frame color is the most
    frequent non-background color.  Each valid input row is reduced to the
    shortest segment beginning and ending with the frame color.  The final
    output is the vertical stack of these cropped row segments.
    """
    if grid.size == 0:
        return grid.copy()

    g = np.asarray(grid)
    h, w = g.shape

    colors, counts = np.unique(g, return_counts=True)
    if len(colors) == 1:
        return g.copy()

    background = int(colors[np.argmax(counts)])

    non_bg = [(int(c), int(n)) for c, n in zip(colors, counts) if int(c) != background]
    if not non_bg:
        return np.zeros_like(g)

    frame = max(non_bg, key=lambda x: x[1])[0]

    rows_out = []

    for r in range(h):
        row = g[r]

        # We only keep rows that actually contain the frame color.
        if not np.any(row == frame):
            continue

        frame_cols = np.where(row == frame)[0]
        if len(frame_cols) == 0:
            continue

        c0, c1 = int(frame_cols[0]), int(frame_cols[-1])

        # A row segment is useful only if it has some internal structure:
        # at least three frame positions, or a non-background pixel between
        # the first and last frame pixels.
        if len(frame_cols) < 3:
            if c1 <= c0 + 1:
                continue
            if not np.any(row[c0 + 1:c1] != background):
                continue

        rows_out.append(row[c0:c1 + 1].copy())

    if not rows_out:
        return np.full((1, 1), background, dtype=g.dtype)

    # Pad rows on the right if one is shorter than the others, using background.
    max_w = max(r.size for r in rows_out)
    padded = []
    for r in rows_out:
        if r.size < max_w:
            pad = np.full(max_w - r.size, background, dtype=r.dtype)
            r = np.concatenate([r, pad])
        padded.append(r)

    out = np.stack(padded, axis=0)

    # Final tight crop to the bounding box of non-background pixels.
    non_bg_mask = out != background
    if not non_bg_mask.any():
        return out

    rows = np.any(non_bg_mask, axis=1)
    cols = np.any(non_bg_mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    return out[r0:r1 + 1, c0:c1 + 1].copy()