import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble 4x4 squares from the input into a compact 2-column output grid.

    - A filled square is a 4x4 block completely filled with one non-zero color.
    - A hollow square is a 4x4 block whose border is one non-zero color and whose 2x2 center is 0.

    Hollow squares are placed in the left column, filled squares in the right column.
    Within each column, squares are ordered by their original position: top to bottom,
    then left to right. Missing tiles are padded with zeros.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2:
        return np.zeros((0, 8), dtype=int)

    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    objects = []  # (top_row, left_col, kind, 4x4_tile)

    # Scan every possible 4x4 top-left position.
    for r in range(h - 3):
        for c in range(w - 3):
            window = grid[r:r + 4, c:c + 4]

            # If this area already belongs to an extracted object, skip it.
            if np.any(used[r:r + 4, c:c + 4]):
                continue

            nonzero = window != 0
            if not np.any(nonzero):
                continue

            colors = np.unique(window[nonzero])
            if colors.size != 1:
                continue

            color = int(colors[0])

            is_filled = np.all(window == color)

            is_hollow = (
                np.all(window[0, :] == color) and
                np.all(window[-1, :] == color) and
                np.all(window[:, 0] == color) and
                np.all(window[:, -1] == color) and
                np.all(window[1:-1, 1:-1] == 0)
            )

            if not (is_filled or is_hollow):
                continue

            kind = "filled" if is_filled else "hollow"
            objects.append((r, c, kind, window.copy()))

            # Mark the whole 4x4 area as consumed.
            used[r:r + 4, c:c + 4] = True

    hollows = [obj for obj in objects if obj[2] == "hollow"]
    filleds = [obj for obj in objects if obj[2] == "filled"]

    hollows.sort(key=lambda obj: (obj[0], obj[1]))
    filleds.sort(key=lambda obj: (obj[0], obj[1]))

    n_rows = max(len(hollows), len(filleds))
    output = np.zeros((4 * n_rows, 8), dtype=grid.dtype)

    for i in range(n_rows):
        if i < len(hollows):
            output[4 * i:4 * i + 4, 0:4] = hollows[i][3]
        if i < len(filleds):
            output[4 * i:4 * i + 4, 4:8] = filleds[i][3]

    return output