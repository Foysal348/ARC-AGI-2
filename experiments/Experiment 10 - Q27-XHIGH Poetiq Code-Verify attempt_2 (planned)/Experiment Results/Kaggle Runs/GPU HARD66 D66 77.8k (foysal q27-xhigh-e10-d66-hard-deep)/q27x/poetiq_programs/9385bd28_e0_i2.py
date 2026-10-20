import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by filling bounding boxes for colors listed in a
    bottom-left 2-column palette.

    Rules implemented:
    - Background is the most frequent color.
    - Palette rows map a corner color to a fill color.
    - For each mapped corner color, fill the bounding box of all non-palette
      cells of that corner color with the mapped fill color.
    - Fill color 0 means "background" (except when background is already 0).
    - Smaller corner colors overwrite larger corner colors in overlaps.
    - L-tromino marker cells (three cells of a color inside a 2x2 block)
      are preserved for their own color.
    - The palette block itself is preserved unchanged.
    """
    arr = np.array(grid, dtype=int).copy()

    if arr.ndim != 2 or arr.size == 0:
        return arr

    H, W = arr.shape

    # ------------------------------------------------------------
    # 1. Determine background color.
    #    Prefer the global mode; on ties, prefer the border mode.
    # ------------------------------------------------------------
    vals, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    tied = vals[counts == max_count]

    if tied.size == 1:
        B = int(tied[0])
    else:
        border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
        B = int(max(tied, key=lambda x: int(np.sum(border == x))))

    total_counts = {int(v): int(np.sum(arr == v)) for v in np.unique(arr)}

    # ------------------------------------------------------------
    # 2. Find the palette.
    #
    # A palette candidate is a 2-column block whose left column is a vertical
    # suffix of a non-background run and whose left-column colors are unique.
    # We score candidates by how many occurrences of those left colors exist
    # outside the candidate block.
    # ------------------------------------------------------------
    def collect_candidates(c_limit, r_limit, min_h):
        res = []
        c_limit = min(W - 1, c_limit)

        for c in range(c_limit):
            col = arr[:, c]
            r = 0

            while r < H:
                if col[r] == B:
                    r += 1
                    continue

                start = r
                while r < H and col[r] != B:
                    r += 1
                end = r - 1

                # The palette bottom is expected to be low in the grid.
                if end < r_limit:
                    continue

                # Consider suffixes of this vertical non-background segment.
                for s in range(start, end + 1):
                    h = end - s + 1
                    if h < min_h:
                        continue

                    lefts = arr[s:end + 1, c]
                    left_ints = [int(x) for x in lefts]

                    # Palette left colors are expected to be distinct.
                    if len(set(left_ints)) != h:
                        continue

                    block = arr[s:end + 1, c:c + 2]
                    mapping = {}
                    score = 0

                    for i, lc in enumerate(left_ints):
                        fc = int(block[i, 1])
                        mapping[lc] = fc

                        in_block = int(np.sum(block == lc))
                        outside = total_counts.get(lc, 0) - in_block
                        if outside < 0:
                            outside = 0
                        if outside > 0:
                            score += outside

                    if score <= 0:
                        continue

                    # Tuple fields:
                    # score, -height, bottom_row, -left_col, start_row, col, mapping
                    #
                    # Higher score wins. On ties, prefer shorter block,
                    # then lower block, then leftmost block.
                    res.append((score, -h, end, -c, s, c, mapping))

        return res

    palette = None

    # Try increasingly permissive regions:
    # pass 0: bottom-left, height >= 2
    # pass 1: wider/lower, height >= 2
    # pass 2: anywhere, height >= 1
    for pass_idx in range(3):
        if pass_idx == 0:
            c_limit = W // 2 + 1
            r_limit = H // 2
            min_h = 2
        elif pass_idx == 1:
            c_limit = (W * 3) // 4 + 1
            r_limit = H // 3
            min_h = 2
        else:
            c_limit = W - 1
            r_limit = 0
            min_h = 1

        candidates = collect_candidates(c_limit, r_limit, min_h)
        if candidates:
            palette = max(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
            break

    # If no palette is found, do nothing.
    if palette is None:
        return arr

    _, _, end, _, pr, pc, mapping = palette
    ph = end - pr + 1

    palette_mask = np.zeros((H, W), dtype=bool)
    palette_mask[pr:pr + ph, pc:pc + 2] = True

    # ------------------------------------------------------------
    # 3. Helper: find L-tromino cells of a color.
    #    A cell is marked if it belongs to some 2x2 window containing
    #    exactly three non-palette cells of that color.
    # ------------------------------------------------------------
    def l_mask(color):
        mask = np.zeros((H, W), dtype=bool)

        if H < 2 or W < 2:
            return mask

        cm = (arr == color) & (~palette_mask)

        if int(cm.sum()) < 3:
            return mask

        for i in range(H - 1):
            for j in range(W - 1):
                cnt = (
                    int(cm[i, j]) +
                    int(cm[i, j + 1]) +
                    int(cm[i + 1, j]) +
                    int(cm[i + 1, j + 1])
                )

                if cnt == 3:
                    if cm[i, j]:
                        mask[i, j] = True
                    if cm[i, j + 1]:
                        mask[i, j + 1] = True
                    if cm[i + 1, j]:
                        mask[i + 1, j] = True
                    if cm[i + 1, j + 1]:
                        mask[i + 1, j + 1] = True

        return mask

    l_masks = {s: l_mask(s) for s in mapping}

    # ------------------------------------------------------------
    # 4. Fill boxes.
    #    Process larger corner colors first, so smaller corner colors win.
    # ------------------------------------------------------------
    out = arr.copy()

    for s in sorted(mapping.keys(), reverse=True):
        m = (arr == s) & (~palette_mask)

        if not m.any():
            continue

        rs, cs = np.where(m)
        rmin, rmax = int(rs.min()), int(rs.max())
        cmin, cmax = int(cs.min()), int(cs.max())

        fill = int(mapping[s])
        if fill == 0 and B != 0:
            fill = B

        # Paint the bounding box.
        out[rmin:rmax + 1, cmin:cmax + 1] = fill

        # Restore this color's L-tromino cells inside its own box.
        lm = l_masks.get(s)
        if lm is not None and lm.any():
            region = lm[rmin:rmax + 1, cmin:cmax + 1]
            if region.any():
                sub = out[rmin:rmax + 1, cmin:cmax + 1]
                sub[region] = s

    # Finally, make sure the palette itself is unchanged.
    out[palette_mask] = arr[palette_mask]

    return out