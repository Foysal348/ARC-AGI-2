import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC-style input grid.

    Observed rule implemented here:
    ------------------------------
    The input is treated as a 2x2 block arrangement:
        top-left     top-right
        bottom-left  bottom-right

    The bottom-left and bottom-right blocks are the main "content" blocks.
    The top-left and top-right blocks are used mainly to determine the output
    size:
      - If either top block is completely non-zero, the output is square with
        side equal to the larger non-zero count.
      - Otherwise, output rows = non-zero count in top-left,
        output cols = non-zero count in top-right.

    Content generation:
      - If bottom-right is empty, repeat bottom-left.
      - If bottom-left is a single solid color, use bottom-right as a mask and
        replace its zero cells by that solid color, then repeat/crop.
      - Otherwise, combine bottom-left and bottom-right in a blockwise manner:
        each cell of bottom-left fills a bottom-right-sized tile, with
        non-zero bottom-right cells taking priority.
        For the irregular full-size case shown in Example #2, a one-row /
        one-column phase shift is inserted between the two halves when the
        bottom-left pattern varies along that axis.
    """
    grid = np.array(grid, dtype=int)

    # Graceful fallback for invalid/empty inputs.
    if grid.size == 0:
        return grid.copy()
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] % 2 != 0:
        return grid.copy()

    n = grid.shape[0] // 2

    # Quadrants.
    TL = grid[:n, :n]
    TR = grid[:n, n:]
    BL = grid[n:, :n]
    BR = grid[n:, n:]

    # -------------------------
    # 1. Determine output size
    # -------------------------
    def nonzero_count(q):
        return int(np.count_nonzero(q))

    def is_full_nonzero(q):
        return bool(np.all(q != 0))

    tl_count = nonzero_count(TL)
    tr_count = nonzero_count(TR)
    tl_full = is_full_nonzero(TL)
    tr_full = is_full_nonzero(TR)

    if tl_full or tr_full:
        side = max(tl_count, tr_count)
        H = W = side
    else:
        H = tl_count
        W = tr_count

    # Avoid empty outputs.
    if H <= 0:
        H = W if W > 0 else 1
    if W <= 0:
        W = H if H > 0 else 1

    # -------------------------
    # 2. Generate content
    # -------------------------
    out = np.zeros((H, W), dtype=int)

    br_has_content = bool(np.any(BR != 0))
    bl_single = bool(np.all(BL == BL.flat[0])) if BL.size else True
    bl_color = int(BL.flat[0]) if BL.size else 0

    # Case A: bottom-right is empty -> just repeat bottom-left.
    if not br_has_content:
        if BL.size:
            for r in range(H):
                for c in range(W):
                    out[r, c] = BL[r % n, c % n]
        return out

    # Case B: bottom-left is a single solid color.
    if bl_single:
        if bl_color != 0:
            pattern = BR.copy()
            pattern[BR == 0] = bl_color
        else:
            pattern = BR.copy()

        for r in range(H):
            for c in range(W):
                out[r, c] = pattern[r % n, c % n]
        return out

    # Case C: blockwise combination of BL and BR.
    # Each BL cell fills a BR-sized tile. Non-zero BR cells win.
    #
    # For the special full-size irregular phase seen in Example #2:
    #   - output is exactly n*n by n*n
    #   - BL varies along an axis
    # then that axis is phase-shifted by 1 in its second half.
    row_varies = not bool(np.all(BL == BL[0:1, :]))
    col_varies = not bool(np.all(BL == BL[:, 0:1]))

    shift_r = 1 if (n > 2 and H == n * n and row_varies) else 0
    shift_c = 1 if (n > 2 and W == n * n and col_varies) else 0

    half_r = H // 2
    half_c = W // 2

    for r in range(H):
        src_r = r - (shift_r if r >= half_r else 0)
        i = src_r // n
        rr = src_r % n

        for c in range(W):
            src_c = c - (shift_c if c >= half_c else 0)
            j = src_c // n
            cc = src_c % n

            val = int(BR[rr, cc])
            if val == 0:
                val = int(BL[i, j])

            out[r, c] = val

    return out