import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the left 5-pattern into the right panel, then rebuild each colored
    column as:
      - top unused space: 5
      - colored bar: its original color, bottom-aligned to the 5-pattern
      - bottom mask: the moved 5-pattern
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H < 2 or W < 2:
        return grid.copy()

    out = grid.copy()

    # Locate the right panel using the top border of 5s.
    top_5 = np.where(grid[0] == 5)[0]
    if top_5.size == 0:
        return out

    right_start = int(top_5[0])
    right_end = int(top_5[-1])

    if right_end <= right_start:
        return out

    # Clear the left panel and the old interior of the right panel.
    # Keep the border columns right_start and right_end, and top/bottom rows.
    out[1:H-1, :right_start] = 0
    out[1:H-1, right_start+1:right_end] = 0

    # Find the movable 5 pattern on the left side.
    if right_start <= 0 or H <= 2:
        return out

    left_mask = (grid[1:H-1, :right_start] == 5)
    ys, xs = np.where(left_mask)

    if ys.size == 0:
        return out

    # Choose the target leftmost tower column.
    # Prefer the first column containing a colored bar; otherwise use the
    # regular panel offset right_start + 2.
    digit_cols = []
    for c in range(right_start + 1, right_end):
        col = grid[1:H-1, c]
        if np.any((col != 0) & (col != 5)):
            digit_cols.append(c)

    if digit_cols:
        target_col = int(digit_cols[0])
    else:
        target_col = right_start + 2
        if target_col < right_start + 1:
            target_col = right_start + 1
        if target_col >= right_end:
            target_col = right_end - 1

    shift = target_col - int(xs.min())

    # Move the 5 pattern into the right panel without changing its rows.
    max_y = int(ys.max())
    for y, x in zip(ys, xs):
        y = int(y)
        nx = int(x) + shift
        if 1 <= y < H - 1 and right_start + 1 <= nx < right_end:
            out[y, nx] = 5

    # Active vertical range for the towers.
    active_top = 1
    active_bottom = max_y
    if active_bottom >= H - 1:
        active_bottom = H - 2
    if active_bottom < active_top:
        active_bottom = active_top

    # Rebuild each colored tower.
    for c in digit_cols:
        if c < right_start + 1 or c >= right_end:
            continue

        col = grid[active_top:active_bottom + 1, c]
        digit_mask = (col != 0) & (col != 5)

        if not np.any(digit_mask):
            continue

        # Bar color and height.
        vals = col[digit_mask]
        unique, counts = np.unique(vals, return_counts=True)
        color = int(unique[np.argmax(counts)])
        bar_height = int(np.sum(digit_mask))

        # Top of the moved 5-mask in this column.
        mask_col = out[active_top:active_bottom + 1, c] == 5
        mask_rows = np.where(mask_col)[0]

        if mask_rows.size == 0:
            mask_top = active_bottom + 1
        else:
            mask_top = active_top + int(mask_rows[0])

        available = max(0, mask_top - active_top)
        use_height = min(bar_height, available)

        if use_height > 0:
            start = mask_top - use_height
            out[start:mask_top, c] = color

            # Fill unused top space in this tower with 5.
            if start > active_top:
                out[active_top:start, c] = 5

    return out