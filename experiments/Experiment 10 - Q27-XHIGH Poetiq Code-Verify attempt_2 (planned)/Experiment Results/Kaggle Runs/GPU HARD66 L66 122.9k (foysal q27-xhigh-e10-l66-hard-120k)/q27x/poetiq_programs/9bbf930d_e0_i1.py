import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move certain 6-markers from the left border to positions determined by
    the surrounding non-background colors.

    Rule summary:
    1. A row r with grid[r, 0] == 6 is active if the first non-7 pixel
       in row r-1 and row r+1 has the same color.
    2. For each active row, remove the 6 from column 0.
    3. Scan right along that row:
       - 7 cells are empty.
       - A non-7 cell is transparent if it has the same color directly
         above and below, and is not on the right border.
       - The first non-transparent non-7 cell is a blocker.
    4. If there is no blocker, place 6 at the rightmost 7 in the row.
    5. If there is a blocker, use the last 7 before it as a vertical column.
       Move down if possible, otherwise up. If both are blocked, use the
       opening beside the blocker and skip to the next vertical run of 7s.
       If the vertical move stops because of a wall, slide right over 7s.
    """
    if grid is None:
        return grid

    grid = np.array(grid, copy=True)

    if grid.size == 0 or grid.ndim != 2:
        return grid

    h, w = grid.shape

    BG = 7
    MARKER = 6

    if h < 3 or w < 2:
        return grid

    out = grid.copy()

    def first_non_bg(row: int) -> int:
        """Return the first non-background color in a row, ignoring column 0."""
        for c in range(1, w):
            if grid[row, c] != BG:
                return int(grid[row, c])
        return -1

    for r in range(1, h - 1):
        # Only left-border 6s are released.
        if grid[r, 0] != MARKER:
            continue

        top_color = first_non_bg(r - 1)
        bottom_color = first_non_bg(r + 1)

        # Active row: neighboring rows start with the same non-background color.
        if top_color == -1 or top_color != bottom_color:
            continue

        # Release this marker.
        out[r, 0] = BG

        wall_c = None
        last_bg = None

        # Scan right to find the first blocking non-background pixel.
        for c in range(1, w):
            val = int(grid[r, c])

            if val == BG:
                last_bg = c
                continue

            # Transparent vertically supported pixel.
            if c < w - 1 and int(grid[r - 1, c]) == val and int(grid[r + 1, c]) == val:
                continue

            # Blocking pixel.
            wall_c = c
            break

        # No blocker: use the rightmost background cell in this row.
        if wall_c is None:
            if last_bg is None:
                out[r, 0] = MARKER
            else:
                out[r, last_bg] = MARKER
            continue

        # No background cell before the blocker; keep the marker in place.
        if last_bg is None:
            out[r, 0] = MARKER
            continue

        vcol = last_bg

        # Choose vertical direction.
        down_empty = (r + 1 < h) and (grid[r + 1, vcol] == BG)
        up_empty = (r - 1 >= 0) and (grid[r - 1, vcol] == BG)

        skip_first = False

        if down_empty:
            direction = 1
            row = r
        elif up_empty:
            direction = -1
            row = r
        else:
            # Fallback: if the blocker column opens below/above, use that side.
            if (r + 1 < h) and (grid[r + 1, wall_c] == BG):
                direction = 1
                row = r + 1
            elif (r - 1 >= 0) and (grid[r - 1, wall_c] == BG):
                direction = -1
                row = r - 1
            elif r + 1 < h:
                direction = 1
                row = r + 1
            else:
                out[r, 0] = MARKER
                continue

            skip_first = True

        # In fallback, skip non-background cells until a background run is found.
        if skip_first:
            if direction == 1:
                while row < h and grid[row, vcol] != BG:
                    row += 1
            else:
                while row >= 0 and grid[row, vcol] != BG:
                    row -= 1

            if row < 0 or row >= h:
                out[r, 0] = MARKER
                continue

        # Move through the vertical background run.
        if direction == 1:
            while row + 1 < h and grid[row + 1, vcol] == BG:
                row += 1
            target_row = row
            stopped_by_wall = (row + 1 < h)
        else:
            while row - 1 >= 0 and grid[row - 1, vcol] == BG:
                row -= 1
            target_row = row
            stopped_by_wall = (row - 1 >= 0)

        target_col = vcol

        # If stopped by a wall, slide right along background cells.
        if stopped_by_wall:
            while target_col + 1 < w and grid[target_row, target_col + 1] == BG:
                target_col += 1

        out[target_row, target_col] = MARKER

    return out