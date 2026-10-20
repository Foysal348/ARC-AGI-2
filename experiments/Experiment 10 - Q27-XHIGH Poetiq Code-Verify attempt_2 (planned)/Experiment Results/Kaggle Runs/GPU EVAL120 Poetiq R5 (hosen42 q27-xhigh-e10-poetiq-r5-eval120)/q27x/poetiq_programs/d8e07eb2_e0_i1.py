import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Match the top row's 3x3 glyphs against the four middle 3x3 glyph rows.

    Rules implemented:
    - The grid is assumed to have the same broad layout as the examples:
      background 8, separator rows of 6, a top band of 3x3 glyphs,
      four middle rows of 3x3 glyphs, and a bottom band.
    - In the top row, adjacent slots with the same color are collapsed;
      only the first glyph in each same-color run is used as a query.
    - Middle glyphs exactly matching a query glyph (same color and same
      3x3 occupancy pattern) are surrounded by a 5x5 color-3 frame.
    - If the matched middle cells form a full row or full column of the
      4x4 middle glyph grid, fill the top band and bottom rows with 3.
      Otherwise, fill the bottom rows with 2.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    BG = 8
    SEP = 6
    MARK = 3
    DEFAULT_BOTTOM = 2

    def _runs(mask):
        """Return contiguous True intervals in a 1D boolean mask."""
        runs = []
        in_run = False
        start = 0
        for i, v in enumerate(mask):
            if v and not in_run:
                in_run = True
                start = i
            elif not v and in_run:
                in_run = False
                runs.append((start, i - 1))
        if in_run:
            runs.append((start, len(mask) - 1))
        return runs

    # Identify the two full separator rows of color 6.
    sep_rows = np.where(np.all(g == SEP, axis=1))[0]
    if len(sep_rows) >= 2:
        first_sep = int(sep_rows[0])
        last_sep = int(sep_rows[-1])
    else:
        # Fallback for slightly irregular inputs.
        first_sep = 5 if H > 5 else H - 1
        last_sep = 27 if H > 27 else H - 1
        if last_sep <= first_sep:
            first_sep = 0
            last_sep = H - 1

    top_end = first_sep
    mid_start = first_sep + 1
    mid_end = last_sep
    bot_start = last_sep + 1

    # Standard slot starts, used if detection is ambiguous.
    fixed_slots = [2, 7, 12, 17]
    fixed_slots = [c for c in fixed_slots if c + 3 <= W]

    def _slot_starts():
        """Detect the four 3-column glyph slots."""
        parts = []
        if top_end > 0:
            parts.append(g[:top_end])
        if mid_end > mid_start:
            parts.append(g[mid_start:mid_end])

        if parts:
            region = np.vstack(parts)
            col_mask = np.any(region != BG, axis=0)
            col_runs = _runs(col_mask)
            width3 = [s for s, e in col_runs if e - s + 1 == 3]

            if len(width3) == 4:
                return width3

            if len(width3) >= 4:
                if fixed_slots:
                    chosen = sorted(
                        width3,
                        key=lambda s: min(abs(s - f) for f in fixed_slots)
                    )[:4]
                    return sorted(chosen)
                return sorted(width3[:4])

            if fixed_slots:
                return fixed_slots

            return width3

        return fixed_slots

    slot_starts = _slot_starts()

    def _glyph(r, c):
        """
        Extract a 3x3 glyph at top-left (r, c).
        Returns (color, flat_pattern_tuple) or None if empty.
        """
        if r < 0 or c < 0 or r + 3 > H or c + 3 > W:
            return None

        sub = g[r:r + 3, c:c + 3]
        mask = sub != BG

        if not np.any(mask):
            return None

        vals = sub[mask]
        uniq, counts = np.unique(vals, return_counts=True)
        color = int(uniq[np.argmax(counts)])
        pattern = tuple(mask.astype(int).ravel())
        return color, pattern

    # Find the 3-row band containing the top glyphs.
    top_row_start = None
    if top_end >= 3:
        top_mask = np.any(g[:top_end] != BG, axis=1)
        top_runs = _runs(top_mask)

        for s, e in top_runs:
            if e - s + 1 == 3:
                top_row_start = s
                break

        if top_row_start is None and top_runs:
            s, e = top_runs[0]
            if s + 3 <= top_end:
                top_row_start = s

    # Build active top queries, collapsing adjacent same-color slots.
    active_top = []
    if top_row_start is not None:
        prev_color = None
        for c in slot_starts:
            glyph = _glyph(top_row_start, c)
            if glyph is None:
                prev_color = None
                continue

            color = glyph[0]
            if prev_color is None or color != prev_color:
                active_top.append(glyph)

            prev_color = color

    def _mid_row_starts():
        """Detect the four 3-row glyph bands in the middle area."""
        if mid_end <= mid_start:
            return []

        region = g[mid_start:mid_end]
        row_mask = np.any(region != BG, axis=1)
        row_runs = _runs(row_mask)

        starts = [mid_start + s for s, e in row_runs if e - s + 1 == 3]

        if starts:
            if len(starts) > 4:
                expected = [
                    mid_start + 2 + 5 * i
                    for i in range(4)
                    if mid_start + 2 + 5 * i + 3 <= mid_end
                ]
                if expected:
                    chosen = sorted(
                        starts,
                        key=lambda s: min(abs(s - e) for e in expected)
                    )[:4]
                    return sorted(chosen)
                return sorted(starts[:4])
            return starts

        # Fallback for the standard layout.
        fallback = []
        for i in range(4):
            rs = mid_start + 2 + 5 * i
            if rs + 3 <= mid_end:
                fallback.append(rs)
        return fallback

    mid_row_starts = _mid_row_starts()

    # Index all middle glyphs by (color, shape).
    mid_keys = {}
    for r, rs in enumerate(mid_row_starts):
        for c, cs in enumerate(slot_starts):
            glyph = _glyph(rs, cs)
            if glyph is not None:
                key = (glyph[0], glyph[1])
                mid_keys.setdefault(key, set()).add((r, c))

    # Select middle cells matching active top glyphs.
    selected = set()
    for color, pattern in active_top:
        key = (color, pattern)
        if key in mid_keys:
            selected.update(mid_keys[key])

    # Draw 5x5 color-3 frames around selected middle glyphs.
    for r, c in selected:
        rs = mid_row_starts[r]
        cs = slot_starts[c]

        r0 = max(0, rs - 1)
        r1 = min(H - 1, rs + 3)
        c0 = max(0, cs - 1)
        c1 = min(W - 1, cs + 3)

        block = out[r0:r1 + 1, c0:c1 + 1]
        block[block == BG] = MARK

    # Determine whether selected cells form a full row or full column.
    line = False
    nr = len(mid_row_starts)
    nc = len(slot_starts)

    if selected and nr > 0 and nc > 0:
        for r in range(nr):
            if all((r, c) in selected for c in range(nc)):
                line = True
                break

        if not line:
            for c in range(nc):
                if all((r, c) in selected for r in range(nr)):
                    line = True
                    break

    # Fill top/bottom bands according to whether a full line was formed.
    if line:
        if top_end > 0:
            top_block = out[:top_end]
            top_block[top_block == BG] = MARK

        if bot_start < H:
            bottom_block = out[bot_start:]
            bottom_block[bottom_block == BG] = MARK
    else:
        if bot_start < H:
            bottom_block = out[bot_start:]
            bottom_block[bottom_block == BG] = DEFAULT_BOTTOM

    return out