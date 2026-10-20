import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interiors of diagonal-line regions.

    In the examples, the non-background pixels form diagonal strokes.
    If we transform coordinates to:
        u = r + c
        v = r - c
    then those diagonal strokes become vertical line segments.

    The output is obtained by, for each fixed v, filling the background
    cells between the vertical wall segments that exist at that v.
    Original wall pixels are preserved, and the fill color is 2.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.copy()

    # Identify the background color as the most frequent color.
    flat = grid.ravel()
    counts = np.bincount(flat)
    background = int(np.argmax(counts))

    # If there is no non-background content, do nothing.
    if not np.any(grid != background):
        return out

    H, W = grid.shape
    fill_color = 2

    # Process each non-background color separately.
    colors = np.unique(grid)
    wall_colors = [int(c) for c in colors if c != background]

    for color in wall_colors:
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]

        u = rows + cols
        v = rows - cols

        # Group points by their u-coordinate.  Each group is a diagonal
        # stroke in the original grid, i.e. a vertical segment in u/v space.
        by_u = {}
        for ui, vi in zip(u, v):
            by_u.setdefault(int(ui), []).append(int(vi))

        segments = []

        # For each u, split v-values into continuous intervals.
        # Consecutive pixels on the same diagonal usually differ by 2 in v.
        for ui, vs in by_u.items():
            vs = sorted(vs)
            start = vs[0]
            prev = vs[0]

            for curr in vs[1:]:
                # If there is a larger gap than a normal step, start a new segment.
                if curr - prev > 2:
                    segments.append((ui, start, prev))
                    start = curr
                prev = curr
            segments.append((ui, start, prev))

        if not segments:
            continue

        v_min = min(seg[1] for seg in segments)
        v_max = max(seg[2] for seg in segments)

        # For every v-level, find covering segments and fill between
        # consecutive segments sorted by u.
        for vi in range(int(v_min), int(v_max) + 1):
            covering = [seg for seg in segments if seg[1] <= vi <= seg[2]]
            if len(covering) < 2:
                continue

            covering.sort(key=lambda x: x[0])

            for i in range(len(covering) - 1):
                u1 = covering[i][0]
                u2 = covering[i + 1][0]

                if u2 <= u1 + 1:
                    continue

                # Fill all transformed cells strictly between the two walls.
                for ui in range(u1 + 1, u2):
                    # Original grid coordinates must be integers.
                    if (ui + vi) % 2 != 0:
                        continue

                    r = (ui + vi) // 2
                    c = (ui - vi) // 2

                    if 0 <= r < H and 0 <= c < W and out[r, c] == background:
                        out[r, c] = fill_color

    return out