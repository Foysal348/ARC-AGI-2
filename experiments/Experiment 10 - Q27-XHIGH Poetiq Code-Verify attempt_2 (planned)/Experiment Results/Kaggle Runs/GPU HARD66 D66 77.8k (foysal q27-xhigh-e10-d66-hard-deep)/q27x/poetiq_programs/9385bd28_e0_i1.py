import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by completing rectangles from corner-colored pixels.

    A bottom-left two-column palette maps corner colors to fill colors.
    For each mapped corner color, the bounding box of all its non-palette
    pixels is filled with the mapped fill color.  Overlaps are resolved by
    corner-color priority: smaller corner colors overwrite larger ones.
    L-tromino marker cells of the currently processed color are preserved.
    """
    grid = np.array(grid, dtype=int)

    if grid.size == 0:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    # Background is the most common color in the input.
    values, counts = np.unique(grid.ravel(), return_counts=True)
    B = int(values[np.argmax(counts)])

    unique_colors = np.unique(grid)
    color_counts = {int(v): int(np.sum(grid == v)) for v in unique_colors}

    def detect_palette(restrict: bool):
        """Detect the 2-column palette block. Returns (r, c, h, mapping)."""
        max_c = W - 1
        min_end = 0

        if restrict:
            # The palette is normally in the bottom-left area.
            max_c = min(W - 1, W // 2 + 1)
            min_end = max(0, H // 3)

        best = None
        best_key = None

        for c in range(max_c):
            r = 0
            while r < H:
                if grid[r, c] == B:
                    r += 1
                    continue

                start = r
                while r < H and grid[r, c] != B:
                    r += 1
                end = r - 1

                if end < min_end:
                    continue

                # Try suffixes of this non-background column segment.
                for s in range(start, end):
                    h = end - s + 1
                    if h < 2:
                        continue

                    lefts = grid[s:s + h, c]

                    # Palette left-column corner colors should be unique.
                    if len(set(lefts.tolist())) != h:
                        continue

                    # Avoid starting immediately after an identical color row.
                    if s > start and lefts[0] == grid[s - 1, c]:
                        continue

                    block = grid[s:s + h, c:c + 2]
                    mapping = {}
                    score = 0
                    ok = True

                    for i, lc in enumerate(lefts.tolist()):
                        lc = int(lc)
                        if lc == B:
                            ok = False
                            break

                        fc = int(block[i, 1])
                        total = color_counts.get(lc, 0)
                        in_block = int(np.sum(block == lc))
                        outside = total - in_block

                        # Good palette rows usually correspond to visible
                        # non-palette pixels of the corner color.
                        if outside > 0:
                            score += outside

                        mapping[lc] = fc

                    if not ok or score <= 0:
                        continue

                    # Prefer higher score, then shorter block on ties,
                    # then lower block, then leftmost column.
                    key = (score, -h, end, -c)
                    if best is None or key > best_key:
                        best = (s, c, h, mapping)
                        best_key = key

        return best

    found = detect_palette(True)
    if found is None:
        found = detect_palette(False)

    if found is None:
        return grid.copy()

    pr, pc, ph, mapping = found
    palette_mask = np.zeros((H, W), dtype=bool)
    palette_mask[pr:pr + ph, pc:pc + 2] = True

    # In the examples, a palette fill color 0 means "background".
    mapping = {int(k): (B if int(v) == 0 else int(v)) for k, v in mapping.items()}

    # Find L-tromino cells: exactly three cells of a color in a 2x2 block.
    # These are the visible corner markers that are preserved when that
    # color's rectangle is filled.
    l_cells = {}
    for color in unique_colors:
        ci = int(color)
        if ci == B:
            continue

        cells = set()

        for i in range(H - 1):
            for j in range(W - 1):
                m = grid[i:i + 2, j:j + 2] == ci
                if m.sum() == 3:
                    if m[0, 0]:
                        r, c = i, j
                        if not palette_mask[r, c]:
                            cells.add((r, c))
                    if m[0, 1]:
                        r, c = i, j + 1
                        if not palette_mask[r, c]:
                            cells.add((r, c))
                    if m[1, 0]:
                        r, c = i + 1, j
                        if not palette_mask[r, c]:
                            cells.add((r, c))
                    if m[1, 1]:
                        r, c = i + 1, j + 1
                        if not palette_mask[r, c]:
                            cells.add((r, c))

        if cells:
            l_cells[ci] = cells

    # Build one rectangle for each mapped color that has enough pixels.
    rects = []
    for lc, fc in mapping.items():
        mask = (grid == lc) & (~palette_mask)

        # At least four pixels are needed to represent a solid 2x2 marker
        # or a pair of L markers.
        if int(mask.sum()) < 4:
            continue

        rs, cs = np.where(mask)
        if rs.size == 0:
            continue

        r1, r2 = int(rs.min()), int(rs.max())
        c1, c2 = int(cs.min()), int(cs.max())
        rects.append((int(lc), int(fc), r1, r2, c1, c2))

    # Larger corner color = lower priority.  Processing in descending order
    # means smaller corner colors are painted last and win overlaps.
    rects.sort(key=lambda x: x[0], reverse=True)

    out = grid.copy()

    for lc, fc, r1, r2, c1, c2 in rects:
        # Paint the rectangle.
        out[r1:r2 + 1, c1:c2 + 1] = fc

        # Preserve the L-shaped markers of this color inside its rectangle.
        cells = l_cells.get(lc)
        if cells:
            for r, c in cells:
                if r1 <= r <= r2 and c1 <= c <= c2 and not palette_mask[r, c]:
                    out[r, c] = lc

        # The palette itself must remain unchanged.
        sub = palette_mask[r1:r2 + 1, c1:c2 + 1]
        if sub.any():
            view = out[r1:r2 + 1, c1:c2 + 1]
            orig = grid[r1:r2 + 1, c1:c2 + 1]
            view[sub] = orig[sub]

    return out