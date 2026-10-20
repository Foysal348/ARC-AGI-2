import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform instruction tiles on the left of a vertical wall into a path
    drawn on the right side of the wall.

    The right side contains a 5 start marker. The left side contains 3x3 tiles.
    Tiles are read column-major: left column top-to-bottom, then next column.
    Each tile color/pattern gives a directed segment:
        1: 3 cells right
        2: 2 cells left
        3: 4 cells left
        6: 2 cells down
    The path starts below the 5, and after each segment the next segment
    starts one row below the current endpoint.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return np.zeros_like(grid, dtype=int)

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Find the vertical separator wall.
    sep = None
    for c in range(W):
        if np.all(grid[:, c] == 4):
            sep = c
            break

    if sep is None:
        for c in range(W):
            col = grid[:, c]
            if np.all(col == col[0]) and col[0] != 0:
                sep = c
                break

    if sep is None:
        counts = np.sum(grid == 4, axis=0)
        if counts.size and counts.max() > 0:
            sep = int(np.argmax(counts))
        else:
            sep = W // 2

    if sep <= 0 or sep >= W - 1:
        sep = W // 2

    W_out = W - sep - 1
    if W_out <= 0:
        return np.zeros((H, max(1, W)), dtype=int)

    out = np.zeros((H, W_out), dtype=int)

    # Find the 5 start marker in the right side.
    right = grid[:, sep + 1:]
    marker_found = False
    ys, xs = np.where(right == 5)
    if ys.size > 0:
        order = np.lexsort((xs, ys))  # topmost, then leftmost
        mr = int(ys[order[0]])
        mc = int(xs[order[0]])
        marker_found = True
    else:
        mr = 0
        mc = W_out // 2

    # Known 3x3 tile shapes mapped to (length, dr, dc).
    shape_map = {
        frozenset({(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}): (3, 0, 1),
        frozenset({(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}): (2, 0, -1),
        frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 2)}): (4, 0, -1),
        frozenset({(0, 0), (0, 2), (1, 1), (2, 1)}): (2, 1, 0),
    }

    # Fallback mapping by color.
    color_map = {
        1: (3, 0, 1),
        2: (2, 0, -1),
        3: (4, 0, -1),
        6: (2, 1, 0),
    }

    segments = []
    Hl, Wl = H, sep

    # Extract 3x3 tiles from the left side in column-major order.
    for bc in range(0, max(0, Wl - 2), 4):
        for br in range(0, max(0, Hl - 2), 4):
            if br + 3 > H or bc + 3 > Wl:
                continue

            block = grid[br:br + 3, bc:bc + 3]
            mask = block != 0

            if not np.any(mask):
                continue

            vals, counts = np.unique(block[mask], return_counts=True)
            color = int(vals[np.argmax(counts)])
            key = frozenset(zip(*np.where(mask)))

            if key in shape_map:
                length, dr, dc = shape_map[key]
            elif color in color_map:
                length, dr, dc = color_map[color]
            else:
                continue

            segments.append((color, length, dr, dc))

    # Draw the path.
    r = mr + 1
    c = mc

    for color, length, dr, dc in segments:
        if r < 0 or r >= H or c < 0 or c >= W_out:
            break

        last_r, last_c = r, c

        for k in range(length):
            rr = r + dr * k
            cc = c + dc * k

            if 0 <= rr < H and 0 <= cc < W_out:
                out[rr, cc] = color

            last_r, last_c = rr, cc

        r = last_r + 1
        c = last_c

    # Preserve the start marker on top of any accidental overlap.
    if marker_found:
        out[mr, mc] = 5

    return out