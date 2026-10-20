import numpy as np
from typing import List, Tuple, Optional

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by "moving" each active 6 from the left border to the end
    of a rectilinear 7-corridor.

    Rule derived from the examples:
    1. An active starting 6 is one at column 0 in a row r such that:
         - grid[r][1] is 7,
         - row r itself continues with a 7 at column 2 (i.e. it is a separator row),
         - the first non-7 cell to the right of column 1 in rows r-1 and r+1
           occurs in the same column and has the same color.
       For each active 6, change it to 7.

    2. From that active row, follow a corridor of 7s:
         - Start at (r, 1), moving east to the end of the horizontal 7-run.
         - At an end of a run, if there is a perpendicular maximal 7-run that starts at this
           cell and whose other endpoint is not orthogonally adjacent to any 6,
           turn into that run and jump to its other endpoint.
         - Stop when no valid turn exists, or when reaching the grid border.
       The final 7 cell is changed to 6.

    All other cells remain unchanged.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.copy()
    H, W = out.shape

    # Directions: (dr, dc)
    EAST = (0, 1)
    WEST = (0, -1)
    NORTH = (-1, 0)
    SOUTH = (1, 0)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def first_non7_in_row(r: int) -> Tuple[Optional[int], Optional[int]]:
        """Return (column, value) of the first cell != 7 scanning from column 1."""
        for c in range(1, W):
            if out[r][c] != 7:
                return c, out[r][c]
        return None, None

    # Find active left-border 6s.
    active_starts: List[Tuple[int, int]] = []
    for r in range(1, H - 1):
        if out[r][0] != 6 or out[r][1] != 7:
            continue
        if W > 2 and out[r][2] != 7:
            continue

        c_above, v_above = first_non7_in_row(r - 1)
        c_below, v_below = first_non7_in_row(r + 1)

        if c_above is not None and c_below is not None and c_above == c_below and v_above == v_below:
            active_starts.append((r, 0))

    # Helper: get maximal horizontal run in row r containing column c.
    # Returns (left_col, right_col) if cell is 7, else None.
    def horizontal_run(r: int, c: int) -> Optional[Tuple[int, int]]:
        if out[r][c] != 7:
            return None
        l = c
        while l > 0 and out[r][l - 1] == 7:
            l -= 1
        rr = c
        while rr + 1 < W and out[r][rr + 1] == 7:
            rr += 1
        return l, rr

    # Helper: get maximal vertical run in column c containing row r.
    def vertical_run(r: int, c: int) -> Optional[Tuple[int, int]]:
        if out[r][c] != 7:
            return None
        u = r
        while u > 0 and out[u - 1][c] == 7:
            u -= 1
        d = r
        while d + 1 < H and out[d + 1][c] == 7:
            d += 1
        return u, d

    # Is a 7-cell orthogonally adjacent to a 6?
    def adjacent_to_6(r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and out[nr][nc] == 6:
                return True
        return False

    # Determine the valid perpendicular run at (r, c) if we arrived along direction dir.
    # Returns (next_endpoint, next_dir) or None.
    def valid_turn(r: int, c: int, dir_: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # If moving horizontally, look for a vertical run; if moving vertically, look horizontal.
        if dir_ in (EAST, WEST):
            run = vertical_run(r, c)
            if run is None:
                return None
            top, bottom = run

            # Current cell must be an endpoint of this vertical run.
            if r == top and bottom > top:
                other = (bottom, c)
                if not adjacent_to_6(bottom, c):
                    return other, SOUTH
            elif r == bottom and bottom > top:
                other = (top, c)
                if not adjacent_to_6(top, c):
                    return other, NORTH
            return None

        else:  # moving NORTH or SOUTH
            run = horizontal_run(r, c)
            if run is None:
                return None
            left, right = run

            if c == left and right > left:
                other = (r, right)
                if not adjacent_to_6(r, right):
                    return other, EAST
            elif c == right and right > left:
                other = (r, left)
                if not adjacent_to_6(r, left):
                    return other, WEST
            return None

    # Follow each active start and mark the final endpoint.
    for sr, sc in active_starts:
        out[sr][sc] = 7  # move the 6 out of the left border

        # Start at the first 7 cell to the right of the start.
        r, c = sr, 1
        direction = EAST

        # Move to the east end of the first horizontal run.
        run = horizontal_run(r, c)
        if run is None:
            continue
        left, right = run
        c = right
        r = sr
        direction = EAST

        # Limit iterations to avoid accidental cycles.
        for _ in range(H * W + 5):
            # Stop at grid border.
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                break

            nxt = valid_turn(r, c, direction)
            if nxt is None:
                break

            (nr, nc), direction = nxt
            r, c = nr, nc

        # The final cell should be a 7; change it to 6.
        if out[r][c] == 7:
            out[r][c] = 6

    return out