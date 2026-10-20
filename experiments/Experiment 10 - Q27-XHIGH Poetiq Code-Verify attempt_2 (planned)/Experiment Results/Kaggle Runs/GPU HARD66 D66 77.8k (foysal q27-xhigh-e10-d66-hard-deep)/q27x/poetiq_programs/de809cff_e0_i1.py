import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by finding "hole-like" zero pixels.

    A zero pixel is selected if, in its 8-neighborhood:
      - all non-zero pixels have the same color C, and
      - there are at least five such pixels of color C.

    For every selected zero:
      - the zero itself becomes 8,
      - its 3x3 neighborhood (clipped to the grid) is filled with the
        "other" main color, i.e. the other non-zero color present in the grid.

    If several neighborhoods overlap, all of their fill colors are applied.
    Conflicting fill requests are resolved by choosing the globally more frequent
    fill color. Finally, all selected centers are forced to 8.
    """
    g = np.asarray(grid, dtype=int).copy()

    if g.ndim != 2:
        return g

    h, w = g.shape
    if h == 0 or w == 0:
        return g

    # Main colors are the non-zero colors present in the input.
    colors = [int(c) for c in np.unique(g) if c != 0]
    if len(colors) < 2:
        return g

    color_counts = {c: int(np.count_nonzero(g == c)) for c in colors}

    def other_color(c: int) -> int:
        """Return the other main color, using frequency as a tie-breaker."""
        candidates = [x for x in colors if x != c]
        if not candidates:
            return 0
        return max(candidates, key=lambda x: color_counts[x])

    selected = []

    # Identify selected zero centers using the original grid.
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue

            source = None
            count = 0
            ok = True

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue

                    v = g[nr, nc]
                    if v == 0:
                        continue

                    if source is None:
                        source = int(v)
                        count = 1
                    elif int(v) == source:
                        count += 1
                    else:
                        ok = False
                        break

                if not ok:
                    break

            if ok and source is not None and count >= 5:
                selected.append((r, c, source))

    out = g.copy()

    # Collect fill requests for each cell.
    fill_requests = {}
    for r, c, source in selected:
        fill = other_color(source)
        if fill == 0:
            continue

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    fill_requests.setdefault((nr, nc), set()).add(fill)

    # Apply fills.  If multiple fills conflict, use the globally more frequent one.
    for (r, c), fills in fill_requests.items():
        if len(fills) == 1:
            out[r, c] = next(iter(fills))
        else:
            out[r, c] = max(fills, key=lambda x: color_counts.get(x, 0))

    # Centers of selected patterns are always 8.
    for r, c, _ in selected:
        out[r, c] = 8

    return out