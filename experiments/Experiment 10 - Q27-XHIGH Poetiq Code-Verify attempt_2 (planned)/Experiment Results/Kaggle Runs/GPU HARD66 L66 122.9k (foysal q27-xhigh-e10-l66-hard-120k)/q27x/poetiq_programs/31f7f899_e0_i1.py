import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height and redraw them from shortest to longest,
    left to right, centered on the horizontal connector row.

    The connector row is the row with the most non-background cells.
    The connector color is the color in that row that appears in the fewest
    rows overall, preferring colors with more separate runs/longer length.
    Vertical bars are groups of consecutive columns on the connector row that
    are not background/connector color.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.ndim != 2 or grid.size == 0:
        return grid

    H, W = grid.shape

    # Background is the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = grid != bg
    if not nonbg.any():
        return grid

    # Find the horizontal connector row.
    row_counts = nonbg.sum(axis=1)
    h = int(np.argmax(row_counts))
    if row_counts[h] == 0:
        return grid

    row = grid[h]
    candidates = [int(v) for v in np.unique(row) if int(v) != bg]
    if not candidates:
        return grid

    # Identify the connector color.
    stats = []
    for color in candidates:
        # Number of rows where this color appears.
        row_span = int(np.sum((grid == color).any(axis=1)))

        # Number of separate horizontal runs and total length in the connector row.
        runs = 0
        length = 0
        in_run = False
        for val in row:
            if int(val) == color:
                length += 1
                if not in_run:
                    runs += 1
                    in_run = True
            else:
                in_run = False

        # Prefer colors appearing in fewer rows, then more runs, then longer.
        stats.append((row_span, -runs, -length, color))

    stats.sort()
    connector_color = int(stats[0][3])

    def column_height(col: int, color: int) -> int:
        """Height of the contiguous vertical segment of this color in one column."""
        top = h
        while top > 0 and grid[top - 1, col] == color:
            top -= 1

        bottom = h
        while bottom < H - 1 and grid[bottom + 1, col] == color:
            bottom += 1

        return bottom - top + 1

    # Group consecutive vertical columns into bars.
    groups = []
    c = 0
    while c < W:
        val = int(grid[h, c])
        if val != bg and val != connector_color:
            color = val
            start = c
            height = column_height(c, color)
            c += 1

            # Merge adjacent columns of the same color into one bar.
            while c < W and int(grid[h, c]) == color:
                ch = column_height(c, color)
                if ch > height:
                    height = ch
                c += 1

            groups.append((start, c - 1, color, height))
        else:
            c += 1

    if not groups:
        return grid

    # Sort heights and assign them to bars from left to right.
    new_heights = sorted(g[3] for g in groups)

    out = grid.copy()

    # Clear old vertical bars.
    for start, end, color, height in groups:
        for col in range(start, end + 1):
            out[:, col] = bg

    # Redraw bars with sorted heights.
    for (start, end, color, old_height), new_height in zip(groups, new_heights):
        # Center on the connector row. For even heights, bias upward.
        top = h - new_height // 2
        bottom = h + (new_height - 1) // 2

        top = max(0, top)
        bottom = min(H - 1, bottom)

        if top <= bottom:
            out[top:bottom + 1, start:end + 1] = color

    return out