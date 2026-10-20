import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    For every cell, look at its 8 neighbors in the original grid.

    1. Zero-cell "stamps"
       If the cell is 0 and some non-zero color appears at least 5 times
       among its 8 neighbors, make this cell an 8 and fill its 3x3
       neighborhood with the other main non-zero color.

    2. Non-zero-cell "majority cleanup"
       If the cell is non-zero and some different color, including 0,
       appears at least 5 times among its 8 neighbors, recolor this cell
       to that color.

    Stamps are applied after cleanup and can override it.  Selected stamp
    centers are finally forced to 8.  If two stamps overlap with different
    fill colors, the globally more frequent fill color wins.
    """
    g = np.asarray(grid, dtype=int).copy()

    if g.ndim != 2:
        return g

    h, w = g.shape
    if h == 0 or w == 0:
        return g

    nonzero_colors = [int(c) for c in np.unique(g) if c != 0]
    color_counts = {c: int(np.count_nonzero(g == c)) for c in nonzero_colors}

    def other_main_color(source: int) -> int:
        """Return the other main non-zero color to use for a stamp."""
        if len(nonzero_colors) <= 1:
            return 0
        return max(
            (c for c in nonzero_colors if c != source),
            key=lambda c: color_counts[c]
        )

    selected_centers = []
    recolors = {}

    # Analyze every cell using the original grid.
    for r in range(h):
        for c in range(w):
            original = int(g[r, c])

            counts = {}
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = int(g[nr, nc])
                        counts[v] = counts.get(v, 0) + 1

            if not counts:
                continue

            best_color, best_count = max(counts.items(), key=lambda x: x[1])

            # A local majority must have at least five cells.
            if best_count >= 5 and best_color != original:
                if original == 0:
                    selected_centers.append((r, c, best_color))
                else:
                    recolors[(r, c)] = best_color

    out = g.copy()

    # First apply non-zero majority cleanup.
    for (r, c), new_value in recolors.items():
        out[r, c] = new_value

    # Then collect stamp fill requests from selected zero centers.
    fill_requests = {}
    for r, c, source in selected_centers:
        fill_value = other_main_color(source)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    fill_requests.setdefault((nr, nc), set()).add(fill_value)

    # Apply stamp fills, overriding earlier cleanup where necessary.
    zero_count = h * w - sum(color_counts.values())

    for (r, c), fills in fill_requests.items():
        if len(fills) == 1:
            out[r, c] = next(iter(fills))
        else:
            def fill_key(color):
                count = zero_count if color == 0 else color_counts.get(color, 0)
                # Prefer the more frequent fill color; deterministic tie-break.
                return (count, -color)

            out[r, c] = max(fills, key=fill_key)

    # Finally force all selected centers to 8.
    for r, c, _ in selected_centers:
        out[r, c] = 8

    return out