import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each non-background color, keep only the largest subset of its pixels
    that is symmetric with respect to a vertical axis.

    Details:
    - The background is taken as the most frequent color in the grid
      (ties are broken using border frequency).
    - For every other color, we consider all possible vertical symmetry axes:
      axes can pass through a column or between two adjacent columns.
    - For a given axis, a pixel is kept only if it lies on the axis or its
      mirrored pixel at the same row is also the same color.
    - The axis with the maximum number of such symmetric pixels is chosen.
      Ties are broken by choosing the axis closest to the horizontal center
      of that color's bounding box.
    - All pixels not selected become background.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Identify background color.
    values, counts = np.unique(grid, return_counts=True)
    max_count = counts.max()
    candidates = values[counts == max_count]

    if candidates.size == 1:
        bg = int(candidates[0])
    else:
        # Tie-break with border frequency.
        border = np.concatenate([
            grid[0, :],
            grid[-1, :],
            grid[:, 0],
            grid[:, -1]
        ])
        border_vals, border_counts = np.unique(border, return_counts=True)
        border_scores = np.array([np.sum(border_vals == c) for c in candidates])
        bg = int(candidates[np.argmax(border_scores)])

    out = np.full((H, W), bg, dtype=grid.dtype)

    # Process each non-background color independently.
    for color in values:
        if int(color) == bg:
            continue

        mask = (grid == color)
        ys, xs = np.nonzero(mask)

        if xs.size == 0:
            continue

        # A single pixel is trivially vertically symmetric.
        if xs.size == 1:
            out[ys[0], xs[0]] = color
            continue

        min_col = int(xs.min())
        max_col = int(xs.max())
        center_col = (min_col + max_col) / 2.0

        best_count = -1
        best_dist = float("inf")
        best_valid = None

        # A is the doubled axis position:
        # A = 2 * axis.
        # Even A means the axis passes through a column.
        # Odd A means the axis lies between two columns.
        for A in range(2 * min_col, 2 * max_col + 1):
            # Mirror column for each pixel: mirror = 2*axis - col = A - col
            mirror = A - xs

            # First, the mirror must be inside the grid.
            valid = (mirror >= 0) & (mirror < W)

            # For pixels not on the axis, require the mirror cell to be this color.
            need_mirror = valid & (mirror != xs)
            if np.any(need_mirror):
                valid[need_mirror] = mask[ys[need_mirror], mirror[need_mirror]]

            count = int(np.count_nonzero(valid))
            dist = abs(A / 2.0 - center_col)

            if count > best_count or (count == best_count and dist < best_dist - 1e-9):
                best_count = count
                best_dist = dist
                best_valid = valid

        if best_valid is not None:
            out[ys[best_valid], xs[best_valid]] = color

    return out