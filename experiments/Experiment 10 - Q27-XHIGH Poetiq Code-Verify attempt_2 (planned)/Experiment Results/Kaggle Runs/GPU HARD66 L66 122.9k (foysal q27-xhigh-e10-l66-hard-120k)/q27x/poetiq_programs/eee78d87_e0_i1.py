import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 6x6 input grid into a 16x16 output grid.

    Rule inferred from the examples:
    1. The input contains one small 3x3 object pattern made of non-background
       pixels, with 7 as the background color.
    2. Extract the 3x3 binary object pattern P, where 1 means "object" and
       0 means background.
    3. From P, build a 3x3 base pattern B:
         - corners of P      -> control B[1][1]
         - top/bottom middle -> control B[1][0]
         - left/right middle -> control B[0][1]
         - center of P       -> control B[0][0]
       Each of these four values is inverted, then expanded to 3x3 by
       duplicating the second row and second column.
    4. Tile B over the entire 16x16 output, using:
         - 7 where B is 1
         - 0 where B is 0
    5. Invert the central 6x6 region:
         - cells that were 0 become 9
         - cells that were 7 remain 7
    """

    # Work on a copy.
    g = np.array(grid, dtype=np.uint8, copy=True)

    # If input is invalid or too small, return a 16x16 zero grid gracefully.
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return np.zeros((16, 16), dtype=np.uint8)

    # Background is 7.
    bg = 7

    # Find the bounding box of all non-background pixels.
    fg_mask = (g != bg)
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No object: use an all-zero 3x3 pattern.
        p = np.zeros((3, 3), dtype=np.uint8)
    else:
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        # The object bounding box should be 3x3 in the intended tasks.
        # If it is not, center/resize it into a 3x3 grid as a fallback.
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        crop = g[r0:r1 + 1, c0:c1 + 1]
        crop_bin = (crop != bg).astype(np.uint8)

        # Simple fallback: nearest-neighbor resize/crop to 3x3.
        p = np.zeros((3, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                # Sample from the cropped object.
                sr = min(h - 1, max(0, int(round(i * (h - 1) / 2))))
                sc = min(w - 1, max(0, int(round(j * (w - 1) / 2))))
                p[i, j] = crop_bin[sr, sc]

    # Extract the four independent values from the 3x3 pattern.
    # Expected pattern shape:
    #   x y x
    #   z w z
    #   x y x
    x = p[0, 0]  # corners
    y = p[0, 1]  # top/bottom middle
    z = p[1, 0]  # left/right middle
    w = p[1, 1]  # center

    # The 2x2 macro bits for the base pattern are the complements of these.
    a = 1 - w  # outer-outer
    b = 1 - z  # outer-inner
    c = 1 - y  # inner-outer
    d = 1 - x  # inner-inner

    # Expand to a 3x3 base pattern by duplicating the second row/column.
    base3 = np.array([
        [a, b, b],
        [c, d, d],
        [c, d, d]
    ], dtype=np.uint8)

    # Fill the 16x16 output with the 3x3 base pattern.
    out = np.zeros((16, 16), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            if base3[i % 3, j % 3] == 1:
                out[i, j] = 7

    # Invert the central 6x6 region: 0 -> 9, 7 stays 7.
    center = out[5:11, 5:11]
    out[5:11, 5:11] = np.where(center == 0, 9, center)

    return out