import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by filling bounding boxes for source colors defined by
    a bottom-left two-column palette.

    For each palette row:
      - left color  = source color
      - right color = fill color
    All non-palette pixels of the source color define a bounding box, which is
    filled with the fill color.  Original source pixels that form an L-tromino
    (exactly 3 pixels of that color in a 2x2 window) are preserved after that
    color's box is filled.  Overlapping boxes are resolved by source-color
    priority: smaller source colors are painted last and therefore win.

    Fill color 0 is treated as "background", and fill colors equal to the
    detected background also remain background.
    """
    arr = np.array(grid, dtype=int).copy()

    if arr.ndim != 2 or arr.size == 0:
        return arr

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr

    # ------------------------------------------------------------------
    # 1. Detect background color.
    # ------------------------------------------------------------------
    vals, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    tied = vals[counts == max_count]

    if tied.size == 1:
        B = int(tied[0])
    else:
        border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
        B = int(max(tied, key=lambda x: int(np.sum(border == x))))

    total_counts = {int(v): int(np.sum(arr == v)) for v in np.unique(arr)}

    def is_visible_fill(color: int) -> bool:
        """A palette fill is visible unless it is 0 or the background color."""
        c = int(color)
        return not (c == 0 or c == B)

    def effective_fill(color: int) -> int:
        """Map palette fill 0 or background fill to the actual background."""
        c = int(color)
        if c == 0 or c == B:
            return B
        return c

    # ------------------------------------------------------------------
    # 2. Detect the palette block.
    # ------------------------------------------------------------------
    def find_palette(min_h: int, require_visible: bool,
                     require_edge: bool, require_support: bool):
        candidates = []

        for c in range(W - 1):
            # Bottom contiguous non-background run in the left column.
            end = H - 1
            while end >= 0 and arr[end, c] == B:
                end -= 1
            if end < 0:
                continue

            start = end
            while start > 0 and arr[start - 1, c] != B:
                start -= 1

            # Consider suffixes of this bottom run.
            for s in range(start, end + 1):
                h = end - s + 1
                if h < min_h:
                    continue

                lefts = arr[s:end + 1, c]
                left_ints = [int(x) for x in lefts]

                # Left palette colors must be non-background and unique.
                if any(lc == B for lc in left_ints):
                    continue
                if len(set(left_ints)) != h:
                    continue

                # Prefer a palette at the left edge of the bottom region.
                if require_edge and c > 0:
                    if not np.all(arr[s:end + 1, c - 1] == B):
                        continue

                block = arr[s:end + 1, c:c + 2]
                mapping = {}
                vis = 0
                support_count = 0
                support_sum = 0

                for i, lc in enumerate(left_ints):
                    fc = int(block[i, 1])
                    mapping[lc] = fc

                    if is_visible_fill(fc):
                        vis += 1

                    in_block = int(np.sum(block == lc))
                    outside = total_counts.get(lc, 0) - in_block

                    if outside > 0:
                        support_count += 1
                        support_sum += outside

                if require_visible and vis == 0:
                    continue
                if require_support and support_count == 0:
                    continue

                is_full = (s == start)

                # Candidate tuple:
                # 0: has visible fill
                # 1: -column (prefer leftmost)
                # 2: full bottom segment?
                # 3: number of supported source colors
                # 4: total supported occurrence count
                # 5: height
                # 6: start row
                # 7: start col
                # 8: mapping
                candidates.append((
                    1 if vis > 0 else 0,
                    -c,
                    is_full,
                    support_count,
                    support_sum,
                    h,
                    s,
                    c,
                    mapping
                ))

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5])
        )

        # Return start row, start col, height, mapping.
        return best[6], best[7], best[5], best[8]

    palette = None
    detection_orders = [
        # First choice: bottom-left edge palette with visible fills and support.
        (2, True, True, True),
        # Relaxed: no left-edge requirement.
        (2, True, False, True),
        # Relaxed: support not required.
        (2, True, True, False),
        (2, True, False, False),
        # Relaxed: visible fills not required.
        (2, False, True, True),
        (2, False, False, True),
        # Allow one-row palettes.
        (1, True, True, True),
        (1, True, False, True),
        (1, False, True, True),
        (1, False, False, False),
    ]

    for min_h, req_vis, req_edge, req_sup in detection_orders:
        palette = find_palette(min_h, req_vis, req_edge, req_sup)
        if palette is not None:
            break

    if palette is None:
        return arr

    pr, pc, ph, mapping = palette

    # Safety clamp in case of unexpected detection.
    if pc + 1 >= W:
        return arr
    if pr + ph > H:
        ph = max(0, H - pr)
        if ph == 0:
            return arr

    palette_mask = np.zeros((H, W), dtype=bool)
    palette_mask[pr:pr + ph, pc:pc + 2] = True

    # ------------------------------------------------------------------
    # 3. Helper: find L-tromino marker cells for a color.
    # ------------------------------------------------------------------
    def compute_l_mask(color: int) -> np.ndarray:
        """
        Return a boolean mask of cells of `color` that belong to at least one
        2x2 window containing exactly three non-palette pixels of `color`.
        """
        color = int(color)
        mask = (arr == color) & (~palette_mask)
        lm = np.zeros((H, W), dtype=bool)

        if H < 2 or W < 2 or int(mask.sum()) < 3:
            return lm

        mi = mask.astype(int)
        window_sum = (
            mi[:-1, :-1] +
            mi[:-1, 1:] +
            mi[1:, :-1] +
            mi[1:, 1:]
        )
        three = (window_sum == 3)

        lm[:-1, :-1] |= mask[:-1, :-1] & three
        lm[:-1, 1:] |= mask[:-1, 1:] & three
        lm[1:, :-1] |= mask[1:, :-1] & three
        lm[1:, 1:] |= mask[1:, 1:] & three

        return lm

    # ------------------------------------------------------------------
    # 4. Build rectangles and fill them.
    # ------------------------------------------------------------------
    rects = []

    for src_color, fill_color in mapping.items():
        src_color = int(src_color)
        fill_color = effective_fill(fill_color)

        mask = (arr == src_color) & (~palette_mask)
        if not np.any(mask):
            continue

        rs, cs = np.where(mask)
        r1, r2 = int(rs.min()), int(rs.max())
        c1, c2 = int(cs.min()), int(cs.max())

        rects.append((src_color, fill_color, r1, r2, c1, c2))

    # Smaller source colors should win overlaps, so paint larger ones first.
    rects.sort(key=lambda x: x[0], reverse=True)

    l_masks = {
        src: compute_l_mask(src)
        for src, _, _, _, _, _ in rects
    }

    out = arr.copy()

    for src_color, fill_color, r1, r2, c1, c2 in rects:
        # Paint the bounding box.
        out[r1:r2 + 1, c1:c2 + 1] = fill_color

        # Restore this color's L-tromino marker cells inside its box.
        lm = l_masks.get(src_color)
        if lm is not None and np.any(lm):
            sub = out[r1:r2 + 1, c1:c2 + 1]
            region = lm[r1:r2 + 1, c1:c2 + 1]
            if np.any(region):
                sub[region] = src_color

    # The palette itself must remain unchanged.
    out[palette_mask] = arr[palette_mask]

    return out