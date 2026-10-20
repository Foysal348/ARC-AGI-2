import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
    1. Erasing the left-hand 5-colored pattern.
    2. Moving that pattern to the right panel (aligned to columns 11-17).
    3. Rebuilding the right panel:
       - The moved 5-pattern forms the lower "mask" area.
       - Each colored vertical bar in the right panel is placed directly above
         the 5-mask in its column.
       - If the bar is shorter than the available space above the mask, the
         remaining space above it is filled with 5s.
    The fixed outer frame of the right panel is preserved.
    """
    if grid.ndim != 2:
        return grid

    H, W = grid.shape

    # The right panel starts where the top border begins.
    top_5 = np.where(grid[0] == 5)[0]
    if len(top_5) == 0:
        return grid

    right_start = int(top_5[0])
    right_end = int(top_5[-1])

    if right_start >= W or right_end <= right_start:
        return grid

    # Locate the movable 5-pattern on the left side.
    left_mask = np.zeros_like(grid, dtype=bool)
    left_ys, left_xs = np.where(grid[1:H-1, :right_start] == 5)

    if len(left_ys) == 0:
        # No movable pattern; just clear the left area and return.
        out = grid.copy()
        out[1:H-1, :right_start] = 0
        return out

    # The movable pattern is shifted so its leftmost column lands at
    # right_start + 2, matching the right panel's mask columns.
    min_col = int(left_xs.min())
    target_min_col = right_start + 2
    if target_min_col >= W:
        target_min_col = right_start + 1
    shift = target_min_col - min_col
    if shift <= 0:
        shift = right_start + 1

    # Start from a copy, then clear the regions that will be rebuilt.
    out = grid.copy()

    # Remove the original left pattern.
    out[1:H-1, :right_start] = 0

    # Clear the interior of the right panel (between the two vertical borders).
    out[1:H-1, right_start + 1:right_end] = 0

    # Shift the left 5-pattern into the right panel.
    for y, x in zip(left_ys, left_xs):
        nx = x + shift
        if 0 <= nx < W:
            out[y, nx] = 5

    # Determine the active vertical span of the right panel.
    # In these grids it is rows 1 through the bottom row of the left pattern.
    active_top = 1
    active_bottom = int(left_ys.max()) + 1
    if active_bottom < active_top:
        active_bottom = active_top

    # Process every interior column that contains at least one digit.
    for c in range(right_start + 1, right_end):
        col = grid[active_top:active_bottom + 1, c]

        # Digits are any colors other than 0 and 5.
        digit_mask = (col != 0) & (col != 5)
        if not np.any(digit_mask):
            continue

        # The bar's color is the most common non-5/non-0 color in the column.
        digit_vals = col[digit_mask]
        unique, counts = np.unique(digit_vals, return_counts=True)
        color = int(unique[np.argmax(counts)])

        # Height of the colored bar in the input.
        bar_height = int(np.sum(digit_mask))

        # Find where the shifted 5-mask starts in this column.
        mask_col = out[active_top:active_bottom + 1, c] == 5
        mask_rows = np.where(mask_col)[0]

        if len(mask_rows) == 0:
            mask_top = active_bottom + 1
        else:
            mask_top = active_top + int(mask_rows[0])

        # Available space above the mask.
        available = mask_top - active_top
        use_height = min(bar_height, max(0, available))

        # The colored bar sits immediately above the mask.
        start = mask_top - use_height

        # Fill top gap with 5s.
        if start > active_top:
            out[active_top:start, c] = 5

        # Fill the colored bar.
        if use_height > 0:
            out[start:mask_top, c] = color

        # Ensure the mask portion is 5.
        if mask_top <= active_bottom:
            out[mask_top:active_bottom + 1, c] = 5

    return out