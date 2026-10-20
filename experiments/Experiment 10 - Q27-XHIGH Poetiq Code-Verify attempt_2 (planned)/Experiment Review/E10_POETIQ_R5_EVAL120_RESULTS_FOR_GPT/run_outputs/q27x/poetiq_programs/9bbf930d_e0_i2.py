import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move selected left-border markers through the background corridor.

    A marker in the first column is selected if:
      - it is on a non-border row,
      - the first non-background cell in the row above and the row below
        are in the same column and have the same color.

    The selected marker then follows background cells:
      - initially move east,
      - if blocked east by a wall, try north then south,
      - while moving north/south, continue straight if possible,
      - if stopped by a wall (not by the grid border), turn east only if
        the east cell is background and the west cell is not background.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 3 or W < 2:
        return g

    # Detect marker color from the first column.
    marker_vals, marker_counts = np.unique(g[:, 0], return_counts=True)
    marker = int(marker_vals[np.argmax(marker_counts)])

    # Detect background/corridor color. Prefer column 1 where the marker is present.
    bg = None
    if W > 1:
        candidate = g[g[:, 0] == marker, 1]
        candidate = candidate[candidate != marker]
        if candidate.size:
            vals, counts = np.unique(candidate, return_counts=True)
            bg = int(vals[np.argmax(counts)])

    if bg is None:
        if W > 1 and np.all(g[:, 1] == g[0, 1]) and int(g[0, 1]) != marker:
            bg = int(g[0, 1])
        else:
            vals, counts = np.unique(g[:, 1:], return_counts=True)
            mask = vals != marker
            if mask.any():
                bg = int(vals[mask][np.argmax(counts[mask])])
            elif vals.size:
                bg = int(vals[np.argmax(counts)])
            else:
                return g

    def first_non_bg(row: int):
        """Return (col, color) of the first non-background cell from col 1."""
        for c in range(1, W):
            if g[row, c] != bg:
                return c, int(g[row, c])
        return None, None

    def trace(start_r: int, start_c: int):
        """
        Follow the corridor rules from (start_r, start_c), initially moving east.
        Returns the final cell.
        """
        if g[start_r, start_c] != bg:
            return None

        r, c = start_r, start_c
        visited = {(r, c)}

        # Directions: 0 = east, 1 = north, 2 = south.
        direction = 0

        while True:
            if direction == 0:  # east
                if c + 1 >= W:
                    break

                nr, nc = r, c + 1
                if g[nr, nc] == bg and (nr, nc) not in visited:
                    r, c = nr, nc
                    visited.add((r, c))
                    continue

                if (nr, nc) in visited:
                    break

                # Blocked by a wall: try north, then south.
                if r - 1 >= 0 and g[r - 1, c] == bg and (r - 1, c) not in visited:
                    r -= 1
                    direction = 1
                    visited.add((r, c))
                    continue

                if r + 1 < H and g[r + 1, c] == bg and (r + 1, c) not in visited:
                    r += 1
                    direction = 2
                    visited.add((r, c))
                    continue

                break

            elif direction == 1:  # north
                if r - 1 < 0:
                    break

                nr, nc = r - 1, c
                if g[nr, nc] == bg and (nr, nc) not in visited:
                    r = nr
                    visited.add((r, c))
                    continue

                if (nr, nc) in visited:
                    break

                # Stopped by a wall, not border. Turn east only if narrow/open to east.
                if c + 1 < W and g[r, c + 1] == bg and (r, c + 1) not in visited:
                    if c - 1 < 0 or g[r, c - 1] != bg:
                        c += 1
                        direction = 0
                        visited.add((r, c))
                        continue

                break

            else:  # south
                if r + 1 >= H:
                    break

                nr, nc = r + 1, c
                if g[nr, nc] == bg and (nr, nc) not in visited:
                    r = nr
                    visited.add((r, c))
                    continue

                if (nr, nc) in visited:
                    break

                # Stopped by a wall, not border. Turn east only if narrow/open to east.
                if c + 1 < W and g[r, c + 1] == bg and (r, c + 1) not in visited:
                    if c - 1 < 0 or g[r, c - 1] != bg:
                        c += 1
                        direction = 0
                        visited.add((r, c))
                        continue

                break

        return r, c

    out = g.copy()

    for r in range(1, H - 1):
        if g[r, 0] != marker:
            continue

        c_up, v_up = first_non_bg(r - 1)
        c_down, v_down = first_non_bg(r + 1)

        if c_up is None or c_down is None:
            continue
        if c_up != c_down or v_up != v_down:
            continue

        start_c = c_up
        if start_c < 1:
            start_c = 1

        # The entry corridor up to the start column must be background.
        if not np.all(g[r, 1:start_c + 1] == bg):
            continue

        endpoint = trace(r, start_c)
        if endpoint is None:
            continue

        out[r, 0] = bg
        er, ec = endpoint
        out[er, ec] = marker

    return out