import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input ARC grid.

    The input has a top palette strip and a main grid below it.
    The palette strip tells us four target colors and, for each color,
    whether to turn counter-clockwise (marker on the left) or clockwise
    (marker on the right) when a ray reaches that target.

    From the source block of color 8, cast thick rays in the four
    cardinal directions.  When a ray hits a target color, cast another
    thick ray in the direction determined by the palette marker.
    All drawn rays are color 8.  The top palette strip is removed.
    """
    if grid is None or np.size(grid) == 0:
        return np.array([], dtype=int)

    grid = np.array(grid, dtype=int, copy=False)
    h, w = grid.shape

    if h == 0 or w == 0:
        return grid.copy()

    # ------------------------------------------------------------------
    # 1. Separate the top palette strip from the main grid.
    # ------------------------------------------------------------------
    sep_color = int(grid[0, 0])
    if not np.all(grid[0] == sep_color):
        sep_color = 5

    bottom_start = 6 if h > 6 else h
    if np.all(grid[0] == sep_color):
        found = False
        for i in range(1, h):
            if np.all(grid[i] == sep_color):
                bottom_start = i + 1
                found = True
                break
        if not found:
            bottom_start = 6 if h > 6 else h

    if bottom_start >= h:
        return grid.copy()

    bottom = grid[bottom_start:, :].copy()
    bh, bw = bottom.shape

    if bh == 0 or bw == 0:
        return bottom

    # ------------------------------------------------------------------
    # 2. Determine background color and source 8-block.
    # ------------------------------------------------------------------
    vals, counts = np.unique(bottom, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    ys, xs = np.where(bottom == 8)
    if len(ys) == 0:
        return bottom

    sr0, sr1 = int(ys.min()), int(ys.max())
    sc0, sc1 = int(xs.min()), int(xs.max())

    th = sr1 - sr0 + 1
    tw = sc1 - sc0 + 1
    t = min(th, tw)

    # Use a square starting block.  In the intended task this is already square.
    if sr0 + t > bh or sc0 + t > bw:
        return bottom

    # ------------------------------------------------------------------
    # 3. Parse the top palette strip.
    #    Build mapping: target color -> 'L' or 'R'.
    # ------------------------------------------------------------------
    top_side = {}

    if bottom_start >= 3:
        inner_rows = grid[1:bottom_start - 1]
        if inner_rows.shape[0] > 0:
            row = grid[1]

            # Detect the four palette cells as non-separator segments.
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

            # Fallback for the known 24-wide palette layout.
            if not cells and w == 24:
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
                mid = (cs + ce) / 2.0
                side = 'L' if float(np.mean(marker_cols)) < mid else 'R'

                top_side[color] = side

    # ------------------------------------------------------------------
    # 4. Obstacles are all non-background, non-8 cells.
    #    Rays stop there, but never overwrite them.
    # ------------------------------------------------------------------
    obstacle = (bottom != bg) & (bottom != 8)
    out = bottom.copy()

    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

    def fill_8(r_a, r_b, c_a, c_b):
        """Fill a rectangle with 8, but never overwrite obstacles."""
        r_a = max(0, int(r_a))
        r_b = min(bh - 1, int(r_b))
        c_a = max(0, int(c_a))
        c_b = min(bw - 1, int(c_b))

        if r_a > r_b or c_a > c_b:
            return

        mask = ~obstacle[r_a:r_b + 1, c_a:c_b + 1]
        if np.any(mask):
            out[r_a:r_b + 1, c_a:c_b + 1][mask] = 8

    def valid_origin(r, c):
        return 0 <= r and r + t <= bh and 0 <= c and c + t <= bw

    def extension_direction(main_dir, side):
        # Left marker  -> counter-clockwise.
        # Right marker -> clockwise.
        if side == 'L':
            return (main_dir + 3) % 4
        return (main_dir + 1) % 4

    # ------------------------------------------------------------------
    # 5. Ray-casting / extension process.
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