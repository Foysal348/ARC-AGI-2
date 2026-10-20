import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the two-panel grid.

    Layout assumptions for the given task:
    - Left 5-shape: rows 2..5, columns 1..7.
    - Right active columns: 11, 13, 15, 17.
    - Right movable vertical segments: rows 1..4 in the active columns.
    - Right outer frame is preserved.

    Rule:
    1. Clear the left panel.
    2. Preserve the outer frame of the right panel.
    3. For each left column (1, 3, 5, 7), compute the height of the
       contiguous 5-bar from the bottom (row 5 upward through row 2).
       This determines how many rows are available above it in the right panel.
    4. In the matching right column, take the main non-zero vertical segment
       from input rows 1..4. Move it so its bottom is just above the 5-bar.
       If it is too long, clip its top.
    5. Translate the left 5-shape into the right panel at columns +10.
    """
    grid = np.asarray(grid)

    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H < 8 or W < 20:
        return grid.copy()

    out = np.zeros((H, W), dtype=int)

    # Preserve the right panel's outer frame.
    out[0, 9:20] = grid[0, 9:20]
    out[7, 9:20] = grid[7, 9:20]
    out[0:8, 9] = grid[0:8, 9]
    out[0:8, 19] = grid[0:8, 19]

    # Process the four vertical slots on the right.
    for c_left, c_right in ((1, 11), (3, 13), (5, 15), (7, 17)):
        # Height of the contiguous 5-bar in the corresponding left column.
        h = 0
        for r in (5, 4, 3, 2):
            if grid[r, c_left] == 5:
                h += 1
            else:
                break

        # Number of rows available above that bar, within rows 1..4.
        available = 5 - h
        if available > 4:
            available = 4

        if available <= 0:
            continue

        # Extract the movable vertical segment from the input right column.
        runs = []
        current = []
        for r in (1, 2, 3, 4):
            v = grid[r, c_right]
            if v != 0:
                current.append((r, int(v)))
            else:
                if current:
                    runs.append(current)
                    current = []
        if current:
            runs.append(current)

        if not runs:
            continue

        # Use the longest run; if tied, use the lowest one.
        segment = max(runs, key=lambda run: (len(run), run[-1][0]))

        # If the segment is longer than the available space, keep its bottom part.
        keep = segment[-available:] if len(segment) > available else segment

        # Place it with its bottom at row `available` (1-based),
        # directly above the translated 5-bar.
        top = available - len(keep) + 1
        for i, (_, color) in enumerate(keep):
            out[top + i, c_right] = color

    # Translate the left 5-shape into the right panel.
    for r in range(2, 6):
        for c in range(1, 8):
            if grid[r, c] == 5:
                out[r, c + 10] = 5

    return out