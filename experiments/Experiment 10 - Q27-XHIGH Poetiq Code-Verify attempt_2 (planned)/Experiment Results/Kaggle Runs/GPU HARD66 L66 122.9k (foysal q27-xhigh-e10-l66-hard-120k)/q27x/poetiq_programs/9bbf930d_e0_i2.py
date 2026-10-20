import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move certain 6 markers from the left border.

    Rule used:
    - Color 7 is empty/background.
    - Color 6 is the marker.
    - A row is "open" if its left-border cell is 6 and the first non-7 cell
      in the row above and the first non-7 cell in the row below are in the
      same column and have the same color.
    - A non-7 cell can be crossed horizontally if its same-color connected
      component does not touch the top, bottom, or right border.
      Such components are considered "floating"; anchored components block.
    - For each open row:
        1. Remove the 6 from column 0.
        2. Scan right, crossing floating components, remembering the last 7.
        3. If no solid blocker is hit, place the 6 on that last 7.
        4. If a blocker is hit, move vertically in that last-7 column to the
           nearer end of the contiguous 7-run in that column.
           Ties prefer down.
    """
    if grid is None:
        return grid

    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2 or grid.size == 0:
        return grid

    h, w = grid.shape

    BG = 7
    MARKER = 6

    if h < 3 or w < 2:
        return grid

    original = grid.copy()

    # ------------------------------------------------------------------
    # Precompute which non-background cells are passable.
    # A non-background cell is passable iff its same-color connected
    # component does not touch the top, bottom, or right border.
    # ------------------------------------------------------------------
    passable = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0]:
                continue

            val = original[r0, c0]

            # Background and markers are not passable components.
            if val == BG or val == MARKER:
                visited[r0, c0] = True
                continue

            # Flood-fill the same-color component.
            stack = [(r0, c0)]
            visited[r0, c0] = True
            cells = []
            touches_boundary = False

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                if r == 0 or r == h - 1 or c == w - 1:
                    touches_boundary = True

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and original[nr, nc] == val
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            # If floating, internal cells may be crossed horizontally.
            if not touches_boundary:
                for r, c in cells:
                    if 0 < c < w - 1:
                        passable[r, c] = True

    # First non-background cell in a row, ignoring the left border.
    def first_non_bg(row: int):
        for c in range(1, w):
            if original[row, c] != BG:
                return c, int(original[row, c])
        return None

    # ------------------------------------------------------------------
    # Identify open rows.
    # ------------------------------------------------------------------
    active_rows = []
    for r in range(1, h - 1):
        if original[r, 0] != MARKER:
            continue

        top = first_non_bg(r - 1)
        bottom = first_non_bg(r + 1)

        if top is None or bottom is None:
            continue

        top_col, top_color = top
        bottom_col, bottom_color = bottom

        if top_col == bottom_col and top_color == bottom_color:
            active_rows.append(r)

    out = original.copy()

    # ------------------------------------------------------------------
    # Move markers.
    # ------------------------------------------------------------------
    for r in active_rows:
        # Release the marker from the left border.
        out[r, 0] = BG

        last_bg = None
        hit_wall = False

        # Scan right across the active row.
        for c in range(1, w):
            val = original[r, c]

            if val == BG:
                last_bg = c
            elif passable[r, c]:
                continue
            else:
                hit_wall = True
                break

        # If no wall was encountered, use the rightmost free background cell.
        if not hit_wall:
            if last_bg is None:
                out[r, 0] = MARKER
            else:
                out[r, last_bg] = MARKER
            continue

        # If the wall is reached immediately, leave the marker in place.
        if last_bg is None:
            out[r, 0] = MARKER
            continue

        # Move vertically in the last background column.
        vcol = last_bg

        top = r
        while top > 0 and original[top - 1, vcol] == BG:
            top -= 1

        bottom = r
        while bottom + 1 < h and original[bottom + 1, vcol] == BG:
            bottom += 1

        up_dist = r - top
        down_dist = bottom - r

        if up_dist == 0 and down_dist == 0:
            target_row = r
        elif up_dist == 0:
            target_row = bottom
        elif down_dist == 0:
            target_row = top
        elif down_dist <= up_dist:
            target_row = bottom
        else:
            target_row = top

        out[target_row, vcol] = MARKER

    return out