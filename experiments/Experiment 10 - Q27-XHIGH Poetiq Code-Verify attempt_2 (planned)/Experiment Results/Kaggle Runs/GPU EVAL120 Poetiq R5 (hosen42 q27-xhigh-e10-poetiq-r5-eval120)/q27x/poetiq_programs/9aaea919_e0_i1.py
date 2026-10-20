import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the ARC task:

    - Detect 3x5 plus-shaped tokens.
    - Detect 1x5 marker lines in the bottom row.
    - Bottom marker color 2 marks source columns.
    - Bottom marker color 3 marks target columns.
    - Recolor all plus tokens in source columns to 5.
    - Add to each target column, directly above its existing plus tokens,
      the total number of plus tokens found in all source columns.
    - Remove bottom marker lines.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    out = grid.copy()
    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Relative coordinates of the 3x5 plus token.
    PLUS = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3)
    ]
    PLUS_SET = set(PLUS)

    # Detect all plus tokens.
    plus_shapes = []  # (top_row, center_col, color)
    for r in range(H - 2):
        for c in range(W - 4):
            color = int(grid[r, c + 2])
            if color == bg:
                continue

            ok = True

            # Expected plus cells must be this color.
            for dr, dc in PLUS:
                if grid[r + dr, c + dc] != color:
                    ok = False
                    break

            if not ok:
                continue

            # All other cells inside the 3x5 bounding box must be background.
            for dr in range(3):
                for dc in range(5):
                    if (dr, dc) not in PLUS_SET and grid[r + dr, c + dc] != bg:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                plus_shapes.append((r, c + 2, color))

    # Group plus tokens by their center column.
    col_pluses = {}
    for top, center, color in plus_shapes:
        col_pluses.setdefault(center, []).append((top, color))

    # Detect 1x5 marker lines in the bottom row.
    line_segments = []  # (start_col, center_col, color)
    r = H - 1
    c = 0
    while c < W:
        if grid[r, c] == bg:
            c += 1
            continue

        start = c
        color = int(grid[r, c])
        while c < W and grid[r, c] == color:
            c += 1

        length = c - start
        if length == 5 and color != bg:
            line_segments.append((start, start + 2, color))

    # Identify source columns (2) and target columns (3).
    source_centers = set()
    target_centers = set()
    for _, center, color in line_segments:
        if color == 2:
            source_centers.add(center)
        elif color == 3:
            target_centers.add(center)

    # Vertical spacing between plus tokens.
    spacing = 4

    # Valid top-row bound. For 30x30 grids this gives 5.
    global_max_top = H - 5 if H >= 5 else 0
    num_slots = max(1, H // 5)
    min_top = max(0, global_max_top - spacing * (num_slots - 1))

    def draw_plus(row, center, color, safe=False, min_r=0):
        """Draw a 3x5 plus token with top row `row` and center column `center`."""
        if row < min_r or row < 0 or row + 2 >= H:
            return
        if center - 2 < 0 or center + 2 >= W:
            return

        for dr, dc in PLUS:
            rr = row + dr
            cc = center + dc

            if safe:
                if out[rr, cc] == bg or out[rr, cc] == color:
                    out[rr, cc] = color
            else:
                out[rr, cc] = color

    # Number of plus tokens in all source columns.
    source_total = sum(len(col_pluses.get(c, [])) for c in source_centers)

    # Recolor source columns to 5.
    for center in source_centers:
        for top, _ in col_pluses.get(center, []):
            draw_plus(top, center, 5, safe=False, min_r=0)

    # Extend target columns upward by source_total tokens.
    for center in target_centers:
        shapes = col_pluses.get(center, [])
        if not shapes or source_total <= 0:
            continue

        # Highest existing plus token in this column.
        top_top = min(top for top, _ in shapes)

        # Use the color of that highest token as the column's token color.
        target_color = 5
        for top, color in shapes:
            if top == top_top:
                target_color = color
                break

        for i in range(1, source_total + 1):
            new_top = top_top - i * spacing
            if new_top < min_top or new_top < 0 or new_top + 2 >= H:
                break
            draw_plus(new_top, center, target_color, safe=True, min_r=min_top)

    # Remove the bottom marker lines.
    for start, _, _ in line_segments:
        out[H - 1, start:start + 5] = bg

    return out