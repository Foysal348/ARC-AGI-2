import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Repair the dominant horizontal line and the vertical lines attached to it.

    The transformation works in the following way:

    1. The background color is the most frequent color.
    2. The foreground/line color is the most frequent non-background color.
    3. The main horizontal line is chosen as the longest horizontal run of
       foreground color.
    4. This horizontal line is extended to the right across short gaps
       (length at most 3) when the run on the left is longer than the run on
       the right.
    5. A one-cell corner extension is allowed to connect the horizontal line
       to a strong vertical line immediately to its right, if that vertical
       line exists on both sides of the horizontal row.
    6. For every column crossed by the completed horizontal line, the vertical
       arm above and below the horizontal row is repaired:
          - pixels in a narrow band around the column are treated as part of
            that vertical arm,
          - small gaps are filled,
          - offset pixels are snapped to the arm's column,
          - unimportant offset pixels are removed.
    7. Finally, stray foreground pixels near the completed horizontal line are
       removed unless they belong to a protected horizontal/vertical run.

    This matches the examples, where broken/drifted vertical strokes are
    straightened, small gaps in the dominant horizontal stroke are filled,
    and nearby stray pixels are cleaned up.
    """
    out = np.array(grid, copy=True)
    original = out.copy()

    if out.size == 0:
        return out

    h, w = out.shape

    vals, counts = np.unique(original, return_counts=True)
    if len(vals) <= 1:
        return out

    bg = int(vals[np.argmax(counts)])
    fg_candidates = [int(v) for v in vals if int(v) != bg]
    if not fg_candidates:
        return out

    # If there are multiple non-background colors, choose the most common one.
    fg = max(fg_candidates, key=lambda c: int(np.count_nonzero(original == c)))
    mask = (original == fg)

    MAX_GAP = 3
    MAX_OFFSET = 3
    MIN_RUN = 3

    # ------------------------------------------------------------------
    # Protect pixels that are already part of reasonably long straight runs.
    # These are less likely to be mere stray pixels.
    # ------------------------------------------------------------------
    h_protected = np.zeros((h, w), dtype=bool)
    v_protected = np.zeros((h, w), dtype=bool)

    for r in range(h):
        c = 0
        while c < w:
            if mask[r, c]:
                c2 = c
                while c2 < w and mask[r, c2]:
                    c2 += 1
                if c2 - c >= MIN_RUN:
                    h_protected[r, c:c2] = True
                c = c2
            else:
                c += 1

    for c in range(w):
        r = 0
        while r < h:
            if mask[r, c]:
                r2 = r
                while r2 < h and mask[r2, c]:
                    r2 += 1
                if r2 - r >= MIN_RUN:
                    v_protected[r:r2, c] = True
                r = r2
            else:
                r += 1

    protected = h_protected | v_protected

    # ------------------------------------------------------------------
    # Find the main horizontal run.
    # ------------------------------------------------------------------
    def horizontal_runs(row):
        runs = []
        c = 0
        while c < w:
            if mask[row, c]:
                c2 = c
                while c2 < w and mask[row, c2]:
                    c2 += 1
                runs.append((c, c2 - 1, c2 - c))
                c = c2
            else:
                c += 1
        return runs

    best = None
    for r in range(h):
        for s, e, ln in horizontal_runs(r):
            if ln < MIN_RUN:
                continue

            # Tie-breaker: prefer runs in rows with more foreground pixels.
            row_count = int(np.count_nonzero(mask[r]))
            key = (ln, row_count, -r, -s)
            if best is None or key > best[0]:
                best = (key, r, s, e, ln)

    if best is None:
        return out

    _, main_row, start, end, cur_len = best

    # ------------------------------------------------------------------
    # Extend the main horizontal line to the right across short gaps.
    # Only extend when the existing left part is longer than the next part.
    # ------------------------------------------------------------------
    while True:
        c = end + 1
        while c < w and not mask[main_row, c]:
            c += 1
        if c >= w:
            break

        gap = c - end - 1
        if gap > MAX_GAP:
            break

        c2 = c
        while c2 < w and mask[main_row, c2]:
            c2 += 1
        next_len = c2 - c

        if next_len < cur_len:
            if end + 1 <= c - 1:
                out[main_row, end + 1:c] = fg
            end = c2 - 1
            cur_len += gap + next_len
        else:
            break

    # ------------------------------------------------------------------
    # Optional one-cell corner completion.
    # This handles cases where the horizontal line ends one cell before a
    # strong vertical line that should intersect it.
    # ------------------------------------------------------------------
    corner_col = end + 1
    if 0 <= corner_col < w and out[main_row, corner_col] != fg:
        above = bool(np.any(mask[:main_row, corner_col]))
        below = bool(np.any(mask[main_row + 1:, corner_col]))
        close_above = main_row > 0 and mask[main_row - 1, corner_col]
        close_below = main_row + 1 < h and mask[main_row + 1, corner_col]

        if above and below and (close_above or close_below):
            col_count = int(np.count_nonzero(mask[:, corner_col]))
            if col_count >= 5:
                out[main_row, corner_col] = fg
                end = corner_col

    # Protect and finalize the main horizontal line.
    out[main_row, start:end + 1] = fg
    protected[main_row, start:end + 1] = True

    # ------------------------------------------------------------------
    # Helper: repair one vertical arm above or below the main horizontal row.
    # ------------------------------------------------------------------
    def repair_vertical_arm(col: int, row: int, direction: int) -> None:
        """
        Repair the vertical arm in column `col`, starting from `row` and
        going in `direction` (-1 upward, +1 downward).
        """
        if not (0 <= col < w):
            return

        included_rows = []
        gap = 0

        step = 1
        while True:
            r = row + direction * step
            if r < 0 or r >= h:
                break

            found = False
            lo = max(0, col - MAX_OFFSET)
            hi = min(w - 1, col + MAX_OFFSET)

            for cc in range(lo, hi + 1):
                if not mask[r, cc]:
                    continue

                # Include pixels that are part of a protected vertical run,
                # or pixels that are not protected by a horizontal run.
                if not h_protected[r, cc] or v_protected[r, cc]:
                    found = True
                    break

            if found:
                gap = 0
                included_rows.append(r)
            else:
                gap += 1
                if gap > MAX_GAP:
                    break

            step += 1

        if not included_rows:
            return

        first = min(included_rows)
        last = max(included_rows)

        # Fill/snap the whole arm to the intended column.
        for r in range(first, last + 1):
            if out[r, col] == bg:
                out[r, col] = fg
            protected[r, col] = True

        # Remove offset pixels in the band that are not protected.
        for r in range(first, last + 1):
            lo = max(0, col - MAX_OFFSET)
            hi = min(w - 1, col + MAX_OFFSET)
            for cc in range(lo, hi + 1):
                if cc == col:
                    continue
                if mask[r, cc] and out[r, cc] == fg and not protected[r, cc]:
                    out[r, cc] = bg

    # Repair vertical arms for all columns crossed by the completed horizontal.
    for c in range(start, end + 1):
        if out[main_row, c] == fg:
            repair_vertical_arm(c, main_row, -1)
            repair_vertical_arm(c, main_row, +1)

    # ------------------------------------------------------------------
    # Final clean-up around the completed horizontal line.
    # Remove stray pixels in nearby rows, but never touch protected pixels.
    # ------------------------------------------------------------------
    for c in range(start, end + 1):
        for d in range(1, MAX_OFFSET + 1):
            for r in (main_row - d, main_row + d):
                if 0 <= r < h and out[r, c] == fg and not protected[r, c]:
                    out[r, c] = bg

    return out