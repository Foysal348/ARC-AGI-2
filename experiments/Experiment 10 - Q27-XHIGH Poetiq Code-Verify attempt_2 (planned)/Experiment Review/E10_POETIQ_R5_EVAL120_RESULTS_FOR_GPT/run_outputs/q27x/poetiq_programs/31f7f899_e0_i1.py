import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing a horizontal line of 6s and vertical bars.

    Rule inferred from the examples:
    1. The row with the most 6s is the horizontal center line.
    2. Bars are contiguous runs on that row whose color is neither 6 nor 8.
       Each bar keeps its color and horizontal position, but its vertical
       extent is rewritten as a centered segment.
    3. Let r be a "radius" (half-height), so height = 2*r + 1.
       - The first bar's radius is max(1, number of 6 cells to its left).
       - For each following bar:
           * if the gap of 6 cells before it has length exactly 1,
             alternate radius between start and 2*start;
           * otherwise, increase the previous target radius by 1.
    4. Rewrite each bar with the computed centered vertical segment.
    """
    arr = np.asarray(grid)
    if arr.size == 0:
        return arr.copy()

    out = arr.copy()

    # Find the horizontal 6-line.
    six_counts = (arr == 6).sum(axis=1)
    if six_counts.max() == 0:
        return out

    center_row = int(np.argmax(six_counts))
    rows, cols = arr.shape

    # Extract bars from the center row:
    # contiguous same-color runs that are not 6 and not background 8.
    bars = []
    c = 0
    while c < cols:
        color = arr[center_row, c]
        if color not in (6, 8):
            c1 = c
            while c < cols and arr[center_row, c] == color:
                c += 1
            bars.append((int(color), c1, c - 1))
        else:
            c += 1

    if not bars:
        return out

    # Compute target radii.
    left_sixes = int(np.sum(arr[center_row, :bars[0][1]] == 6))
    start = max(1, left_sixes)

    low = start
    high = 2 * start
    targets = [start]

    # alt_state: 0 -> next single-6 gap goes to high, 1 -> goes to low
    alt_state = 0

    for i in range(len(bars) - 1):
        _, c1_prev, c2_prev = bars[i]
        _, c1_next, _ = bars[i + 1]

        gap_sixes = int(np.sum(arr[center_row, c2_prev + 1 : c1_next] == 6))

        if gap_sixes == 1:
            if alt_state == 0:
                targets.append(high)
                alt_state = 1
            else:
                targets.append(low)
                alt_state = 0
        else:
            targets.append(targets[-1] + 1)
            alt_state = 0

    # Keep segments inside the grid.
    max_radius = min(center_row, rows - 1 - center_row)
    targets = [min(r, max_radius) for r in targets]

    # Rewrite bars.
    for (color, c1, c2), r in zip(bars, targets):
        # Clear old cells of this bar in its columns.
        for col in range(c1, c2 + 1):
            rows_to_clear = np.where(arr[:, col] == color)[0]
            out[rows_to_clear, col] = 8

        # Write the new centered vertical segment.
        top = center_row - r
        bottom = center_row + r
        top = max(0, top)
        bottom = min(rows - 1, bottom)
        out[top:bottom + 1, c1:c2 + 1] = color

    return out