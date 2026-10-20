import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Convert the input tile/5 scene into the drawn path output.

    Rule:
      1. Find the vertical separator line of 4s.
      2. The 5 on the right side determines the starting output column.
      3. Read the 3x3 tiles on the left in column-major order:
         left tile column top-to-bottom, then right tile column top-to-bottom.
      4. Convert each tile into a path segment:
           1 -> horizontal length 3 to the right
           2 -> horizontal length 2 to the left
           3 -> horizontal length 4 to the left
           6 -> vertical length 2 downward
      5. Draw the segments starting from the 5 and return the output grid.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W < 2:
        return grid.copy()

    def clusters(mask):
        """Return consecutive True index ranges in a 1D boolean mask."""
        result = []
        start = None
        for i, value in enumerate(mask):
            if bool(value):
                if start is None:
                    start = i
            else:
                if start is not None:
                    result.append((start, i - 1))
                    start = None
        if start is not None:
            result.append((start, len(mask) - 1))
        return result

    def find_separator():
        """Find the vertical separator column."""
        # Preferred: a full column of 4s.
        for c in range(W):
            if np.all(grid[:, c] == 4):
                return c

        # Fallback: column with the most 4s.
        counts4 = (grid == 4).sum(axis=0)
        if counts4.size > 0 and counts4.max() > 0:
            return int(np.argmax(counts4))

        # Fallback: full uniform non-zero column.
        candidates = []
        for c in range(W):
            vals = np.unique(grid[:, c])
            if vals.size == 1 and vals[0] != 0:
                candidates.append(c)
        if candidates:
            return max(candidates, key=lambda c: (min(c, W - 1 - c), c))

        # Last fallback: column with largest count of any non-zero color.
        best_c = W // 2
        best_count = -1
        for c in range(W):
            vals, counts = np.unique(grid[:, c], return_counts=True)
            if vals.size == 0:
                continue
            non_zero_counts = counts[vals != 0]
            if non_zero_counts.size > 0 and int(non_zero_counts.max()) > best_count:
                best_count = int(non_zero_counts.max())
                best_c = c
        return best_c

    sep = find_separator()
    if sep <= 0 or sep >= W:
        sep = max(1, min(W // 2, W - 1))

    out_width = sep
    out = np.zeros((H, out_width), dtype=grid.dtype)

    # Starting position from the 5.
    fives = np.argwhere(grid == 5)
    if fives.size:
        r0, c0 = int(fives[0][0]), int(fives[0][1] - sep - 1)
    else:
        r0, c0 = 0, out_width // 2

    r0 = max(0, min(H - 1, r0))
    c0 = max(0, min(out_width - 1, c0))
    out[r0, c0] = 5

    # Extract tiles from the left side.
    left = grid[:, :sep]
    row_groups = clusters(np.any(left != 0, axis=1))
    col_groups = clusters(np.any(left != 0, axis=0))

    if len(col_groups) >= 2:
        tile_cols = [col_groups[0][0], col_groups[1][0]]
    elif out_width >= 7:
        # Known task layout: two 3-wide tile columns at columns 0 and 4.
        tile_cols = [0, 4]
    else:
        tile_cols = [0]

    def tile_color(block):
        """Return the dominant non-zero color in a tile block."""
        non_zero = block[block != 0]
        if non_zero.size == 0:
            return None
        vals, counts = np.unique(non_zero, return_counts=True)
        return int(vals[np.argmax(counts)])

    # Known tile shapes, independent of color, as 3x3 boolean masks.
    MASK_PARAMS = {
        # 1 1 0 / 1 0 1 / 0 1 0
        (True, True, False,
         True, False, True,
         False, True, False): ("H", 1, 3),

        # 2 0 2 / 2 0 2 / 2 2 2
        (True, False, True,
         True, False, True,
         True, True, True): ("H", -1, 2),

        # 3 3 3 / 0 3 0 / 3 0 3
        (True, True, True,
         False, True, False,
         True, False, True): ("H", -1, 4),

        # 6 0 6 / 0 6 0 / 0 6 0
        (True, False, True,
         False, True, False,
         False, True, False): ("V", 1, 2),
    }

    # Fallback mapping by color.
    COLOR_PARAMS = {
        1: ("H", 1, 3),
        2: ("H", -1, 2),
        3: ("H", -1, 4),
        6: ("V", 1, 2),
    }

    def segment_params(color, block):
        """Return (orientation, direction, length) for a tile."""
        mask = np.zeros((3, 3), dtype=bool)
        if block.size:
            h = min(block.shape[0], 3)
            w = min(block.shape[1], 3)
            mask[:h, :w] = block[:h, :w] != 0

        key = tuple(mask.ravel())
        if key in MASK_PARAMS:
            return MASK_PARAMS[key]

        if color in COLOR_PARAMS:
            return COLOR_PARAMS[color]

        # Conservative fallback for unseen tile shapes.
        if mask[1, 1] and mask[2, 1] and int(mask.sum(axis=1).max()) <= 2:
            return ("V", 1, 2)

        direction = -1 if int(mask[:, :2].sum()) > int(mask[:, 2:].sum()) else 1
        length = max(2, int(mask.sum(axis=1).max()))
        return ("H", direction, length)

    # Store tiles by column, then read left column first, then right column.
    column_items = [[] for _ in tile_cols]

    for r_start, _ in row_groups:
        for idx, c_start in enumerate(tile_cols):
            if c_start >= sep:
                continue

            r_end = min(r_start + 3, H)
            c_end = min(c_start + 3, sep)

            if r_end <= r_start or c_end <= c_start:
                continue

            block = grid[r_start:r_end, c_start:c_end]
            color = tile_color(block)

            if color is None:
                continue

            column_items[idx].append((color, block))

    sequence = []
    for items in column_items:
        sequence.extend(items)

    # Draw the path.
    r, c = r0, c0

    for color, block in sequence:
        orientation, direction, length = segment_params(color, block)

        if length <= 0:
            continue

        if orientation == "H":
            row = r + 1
            if row < 0 or row >= H:
                break

            if c < 0 or c >= out_width:
                c = max(0, min(out_width - 1, c))

            last_col = c
            for i in range(length):
                nc = c + direction * i
                if 0 <= nc < out_width:
                    out[row, nc] = color
                    last_col = nc
                else:
                    break

            c = last_col
            r = row

        else:  # vertical
            col = c
            if col < 0 or col >= out_width:
                col = max(0, min(out_width - 1, col))
                c = col

            last_row = r
            for i in range(length):
                nr = r + 1 + i
                if 0 <= nr < H:
                    out[nr, col] = color
                    last_row = nr
                else:
                    break

            r = last_row

    return out