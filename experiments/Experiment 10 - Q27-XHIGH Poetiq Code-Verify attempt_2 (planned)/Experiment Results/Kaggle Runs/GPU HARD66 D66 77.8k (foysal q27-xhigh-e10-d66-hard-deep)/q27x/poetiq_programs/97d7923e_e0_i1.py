import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each endpoint color, find the vertical bar of that color that is most
    isolated horizontally from the other bars of the same color, and fill its
    interior with the endpoint color.

    A bar is a contiguous vertical non-zero run whose top and bottom cells have
    the same color and whose interior contains at least one different color.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return grid.copy()

    out = grid.copy()
    h, w = grid.shape

    bars = []

    # Find all candidate vertical bars.
    for c in range(w):
        col = grid[:, c]
        r = 0

        while r < h:
            if col[r] == 0:
                r += 1
                continue

            start = r
            while r < h and col[r] != 0:
                r += 1
            end = r  # exclusive

            # Need at least one interior cell.
            if end - start >= 3:
                color = int(col[start])
                if int(col[end - 1]) == color:
                    interior = col[start + 1:end]
                    if interior.size > 0 and np.any(interior != color):
                        bars.append((color, c, start, end))

    if not bars:
        return out

    # Group candidate bars by their endpoint color.
    groups = {}
    for bar in bars:
        groups.setdefault(bar[0], []).append(bar)

    selected = []

    # For each color, choose the bar with the largest minimum horizontal
    # distance to another bar of the same color. Break ties rightmost.
    for color, group in groups.items():
        if len(group) == 1:
            selected.append(group[0])
            continue

        best = None
        best_key = (-1, -1, -1)

        for bar in group:
            _, c, start, end = bar
            min_dist = min(abs(c - other[1]) for other in group if other is not bar)
            key = (min_dist, c, start)

            if key > best_key:
                best_key = key
                best = bar

        selected.append(best)

    # Fill interiors of selected bars.
    for color, c, start, end in selected:
        if end - start > 1:
            out[start + 1:end, c] = color

    return out