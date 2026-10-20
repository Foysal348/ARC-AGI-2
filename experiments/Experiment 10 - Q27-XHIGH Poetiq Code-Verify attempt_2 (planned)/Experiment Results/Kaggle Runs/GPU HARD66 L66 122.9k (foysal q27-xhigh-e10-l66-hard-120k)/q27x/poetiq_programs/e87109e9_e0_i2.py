import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC grid.

    The input contains a top palette strip and a main grid below it.
    The palette strip tells, for certain target colors, whether a beam
    should turn counter-clockwise (marker on the left) or clockwise
    (marker on the right) when it hits that color.

    The main grid contains:
      - a background color,
      - a source block of color 8,
      - rectangular obstacle/target blocks.

    From the source, thick rays of color 8 are cast in the four cardinal
    directions.  A ray stops at the first non-background, non-8 cell.
    If the stopping color has a palette entry, the ray turns according to
    that entry and continues.  The top palette strip is removed from the
    output.
    """
    if grid is None or np.size(grid) == 0:
        return np.array([], dtype=int)

    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    # ------------------------------------------------------------------
    # 1. Find and remove the top palette strip.
    # ------------------------------------------------------------------
    sep_color = int(grid[0, 0])
    if not np.all(grid[0] == sep_color):
        sep_color = 5

    bottom_start = 6 if h > 6 else h

    if h > 0 and np.all(grid[0] == sep_color):
        for i in range(1, h):
            if np.all(grid[i] == sep_color):
                bottom_start = i + 1
                break

    if bottom_start >= h:
        return grid.copy()

    bottom = grid[bottom_start:, :].copy()
    bh, bw = bottom.shape

    if bh == 0 or bw == 0:
        return bottom

    # ------------------------------------------------------------------
    # 2. Determine background color and the source 8-block.
    # ------------------------------------------------------------------
    vals, counts = np.unique(bottom, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    ys, xs = np.where(bottom == 8)
    if len(ys) == 0:
        return bottom

    sr0, sr1 = int(ys.min()), int(ys.max())
    sc0, sc1 = int(xs.min()), int(xs.max())

    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    t = min(sh, sw)

    # If the 8-region is not perfectly square, use its central square.
    sr0 += (sh - t) // 2
    sc0 += (sw - t) // 2
    sr1 = sr0 + t - 1
    sc1 = sc0 + t - 1

    # ------------------------------------------------------------------
    # 3. Parse palette:
    #    color -> 'L' (counter-clockwise) or 'R' (clockwise)
    # ------------------------------------------------------------------
    top_side = {}

    if bottom_start >= 3:
        inner_rows = grid[1:bottom_start - 1]
        if inner_rows.shape[0] > 0:
            row = grid[1]

            cells = []
            in_cell = False
            start = 0

            for c in range(w):
                if row[c] != sep_color:
                    if not in_cell:
                        start = c
                        in_cell = True
                else:
                    if in_cell:
                        cells.append((start, c - 1))
                        in_cell = False

            if in_cell:
                cells.append((start, w - 1))

            # Fallback for the common 24-wide palette layout.
            if len(cells) != 4 and w == 24:
                cells = [(1, 4), (7, 10), (13, 16), (19, 22)]

            for cs, ce in cells:
                if ce < cs or ce >= w:
                    continue

                sub = inner_rows[:, cs:ce + 1]
                mask = (sub != sep_color) & (sub != 0)

                if not np.any(mask):
                    continue

                cvals, ccounts = np.unique(sub[mask], return_counts=True)
                color = int(cvals[np.argmax(ccounts)])

                marker_cols = np.where(mask)[1]
                cell_width = ce - cs + 1

                # Use relative columns inside the cell.
                if float(np.mean(marker_cols)) < cell_width / 2.0:
                    top_side[color] = 'L'
                else:
                    top_side[color] = 'R'

    # ------------------------------------------------------------------
    # 4. Obstacles are all non-background, non-8 cells.
    # ------------------------------------------------------------------
    obstacle = (bottom != bg) & (bottom != 8)
    out = bottom.copy()

    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

    def valid_origin(r, c):
        return 0 <= r and r + t <= bh and 0 <= c and c + t <= bw

    def fill_8(r_a, r_b, c_a, c_b):
        """Fill rectangle with 8, but never overwrite obstacles."""
        r_a = max(0, int(r_a))
        r_b = min(bh - 1, int(r_b))
        c_a = max(0, int(c_a))
        c_b = min(bw - 1, int(c_b))

        if r_a > r_b or c_a > c_b:
            return

        mask = ~obstacle[r_a:r_b + 1, c_a:c_b + 1]
        if np.any(mask):
            out[r_a:r_b + 1, c_a:c_b + 1][mask] = 8

    def extension_direction(main_dir, side):
        # Left marker  -> counter-clockwise.
        # Right marker -> clockwise.
        if side == 'L':
            return (main_dir + 3) % 4
        return (main_dir + 1) % 4

    # ------------------------------------------------------------------
    # 5. Ray casting / turning process.
    # ------------------------------------------------------------------
    queue = [
        (sr0, sc0, UP),
        (sr0, sc0, DOWN),
        (sr0, sc0, LEFT),
        (sr0, sc0, RIGHT),
    ]

    seen = set()
    max_iter = 10000
    it = 0

    while queue and it < max_iter:
        it += 1
        r0, c0, d = queue.pop(0)

        key = (r0, c0, d)
        if key in seen:
            continue
        seen.add(key)

        if not valid_origin(r0, c0):
            continue

        r1 = r0 + t - 1
        c1 = c0 + t - 1

        if d == UP:
            stop_row = -1
            stop_color = None

            for c in range(c0, c1 + 1):
                rows = np.where(obstacle[:r0, c])[0]
                if rows.size > 0:
                    r = int(rows.max())
                    if r > stop_row:
                        stop_row = r
                        stop_color = int(bottom[r, c])

            start_r = stop_row + 1 if stop_row >= 0 else 0
            fill_8(start_r, r1, c0, c1)

            if stop_row >= 0 and stop_color in top_side:
                nr = stop_row + 1
                if valid_origin(nr, c0):
                    queue.append((nr, c0, extension_direction(UP, top_side[stop_color])))

        elif d == DOWN:
            stop_row = bh
            stop_color = None
            offset = r0 + t

            for c in range(c0, c1 + 1):
                rows = np.where(obstacle[offset:, c])[0]
                if rows.size > 0:
                    r = int(rows.min()) + offset
                    if r < stop_row:
                        stop_row = r
                        stop_color = int(bottom[r, c])

            end_r = stop_row - 1 if stop_row < bh else bh - 1
            fill_8(r0, end_r, c0, c1)

            if stop_row < bh and stop_color in top_side:
                nr = stop_row - t
                if valid_origin(nr, c0):
                    queue.append((nr, c0, extension_direction(DOWN, top_side[stop_color])))

        elif d == LEFT:
            stop_col = -1
            stop_color = None

            for r in range(r0, r1 + 1):
                cols = np.where(obstacle[r, :c0])[0]
                if cols.size > 0:
                    c = int(cols.max())
                    if c > stop_col:
                        stop_col = c
                        stop_color = int(bottom[r, c])

            start_c = stop_col + 1 if stop_col >= 0 else 0
            fill_8(r0, r1, start_c, c1)

            if stop_col >= 0 and stop_color in top_side:
                nc = stop_col + 1
                if valid_origin(r0, nc):
                    queue.append((r0, nc, extension_direction(LEFT, top_side[stop_color])))

        else:  # RIGHT
            stop_col = bw
            stop_color = None
            offset = c0 + t

            for r in range(r0, r1 + 1):
                cols = np.where(obstacle[r, offset:])[0]
                if cols.size > 0:
                    c = int(cols.min()) + offset
                    if c < stop_col:
                        stop_col = c
                        stop_color = int(bottom[r, c])

            end_c = stop_col - 1 if stop_col < bw else bw - 1
            fill_8(r0, r1, c0, end_c)

            if stop_col < bw and stop_color in top_side:
                nc = stop_col - t
                if valid_origin(r0, nc):
                    queue.append((r0, nc, extension_direction(RIGHT, top_side[stop_color])))

    return out