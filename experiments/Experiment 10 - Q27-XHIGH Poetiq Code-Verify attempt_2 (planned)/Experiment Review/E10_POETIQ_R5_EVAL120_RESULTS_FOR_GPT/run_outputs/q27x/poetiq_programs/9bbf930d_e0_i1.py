import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move selected 6 markers from the first column to the end of a rightward-biased
    background corridor.

    A marker in row r is moved when row r is a background row between two rows
    whose leftmost non-background cell is the same color and in the same column.
    The marker then travels through background cells:
      - go east while possible;
      - if blocked east, try north then south;
      - while going north/south, go straight while possible, then turn east if blocked;
      - stop at the border or when no allowed background cell remains.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 3:
        return g

    # Defensively detect the marker color and background color.
    # In the given tasks marker = 6 and background = 7.
    marker = int(np.bincount(g[:, 0]).argmax())

    if W > 1 and g[0, 1] != marker and np.all(g[:, 1] == g[0, 1]):
        bg = int(g[0, 1])
    else:
        vals, counts = np.unique(g.ravel(), return_counts=True)
        mask = vals != marker
        if mask.any():
            bg = int(vals[mask][np.argmax(counts[mask])])
        else:
            bg = 0

    def first_non_bg(row):
        """Return (column, color) of the first non-background cell from column 2."""
        for c in range(2, W):
            if int(row[c]) != bg:
                return c, int(row[c])
        return None, None

    def trace(r, c):
        """
        Follow the movement rules from (r, c), starting east.
        Returns the final cell.
        """
        dr, dc = 0, 1  # east
        visited = {(r, c)}

        for _ in range(H * W * 4):
            if dc == 1:  # moving east
                if c + 1 >= W:
                    break

                if (r, c + 1) not in visited and g[r, c + 1] == bg:
                    c += 1
                    visited.add((r, c))
                    continue

                if r - 1 >= 0 and (r - 1, c) not in visited and g[r - 1, c] == bg:
                    r -= 1
                    dr, dc = -1, 0
                    visited.add((r, c))
                    continue

                if r + 1 < H and (r + 1, c) not in visited and g[r + 1, c] == bg:
                    r += 1
                    dr, dc = 1, 0
                    visited.add((r, c))
                    continue

                break

            elif dr == -1:  # moving north
                if r - 1 < 0:
                    break

                if (r - 1, c) not in visited and g[r - 1, c] == bg:
                    r -= 1
                    visited.add((r, c))
                    continue

                if c + 1 < W and (r, c + 1) not in visited and g[r, c + 1] == bg:
                    c += 1
                    dr, dc = 0, 1
                    visited.add((r, c))
                    continue

                break

            elif dr == 1:  # moving south
                if r + 1 >= H:
                    break

                if (r + 1, c) not in visited and g[r + 1, c] == bg:
                    r += 1
                    visited.add((r, c))
                    continue

                if c + 1 < W and (r, c + 1) not in visited and g[r, c + 1] == bg:
                    c += 1
                    dr, dc = 0, 1
                    visited.add((r, c))
                    continue

                break

            else:
                break

        return r, c

    out = g.copy()

    for r in range(1, H - 1):
        if g[r, 0] != marker:
            continue

        c_up, v_up = first_non_bg(g[r - 1])
        c_down, v_down = first_non_bg(g[r + 1])

        if c_up is None or c_down is None:
            continue
        if c_up != c_down or v_up != v_down:
            continue

        start_col = c_up

        if g[r, start_col] != bg:
            continue

        # The start corridor must be background from column 2 through start_col.
        if not all(int(g[r, c]) == bg for c in range(2, start_col + 1)):
            continue

        fr, fc = trace(r, start_col)

        out[r, 0] = bg
        out[fr, fc] = marker

    return out