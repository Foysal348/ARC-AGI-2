import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform vertical capped segments of the form A-B-...-B-A.

    A segment is eligible only if:
      - it is a contiguous vertical non-zero run,
      - its first and last pixels have the same color A,
      - all pixels between them have the same color B,
      - B != A,
      - there is at least one B pixel.

    For the smallest non-zero color in the grid, recolor the interior
    of the longest eligible segment with that endpoint color.
    For every other endpoint color, recolor the interior of the shortest
    eligible segment with that endpoint color.
    """
    if grid is None or grid.ndim != 2:
        return np.array(grid, copy=True)

    out = grid.copy()
    h, w = out.shape
    if h == 0 or w == 0:
        return out

    nonzero = out[out != 0]
    if nonzero.size == 0:
        return out

    special_color = int(nonzero.min())

    # Collect eligible vertical segments grouped by endpoint color.
    # Each entry is:
    # (interior_length, top_row, col, bottom_row, interior_color)
    segments_by_cap = {}

    for col in range(w):
        row = 0
        while row < h:
            if out[row, col] == 0:
                row += 1
                continue

            start = row
            while row < h and out[row, col] != 0:
                row += 1
            end = row - 1

            if end - start + 1 < 3:
                continue

            cap_color = out[start, col]
            if cap_color != out[end, col]:
                continue

            interior = out[start + 1:end, col]
            if interior.size == 0:
                continue

            body_color = interior[0]
            if body_color == cap_color:
                continue

            if not np.all(interior == body_color):
                continue

            segments_by_cap.setdefault(int(cap_color), []).append(
                (len(interior), start, col, end, int(body_color))
            )

    # Choose one segment per endpoint color and recolor its interior.
    for cap_color, segments in segments_by_cap.items():
        if not segments:
            continue

        if cap_color == special_color:
            # Longest interior; ties: highest top, then leftmost column.
            best = max(segments, key=lambda seg: (seg[0], -seg[1], -seg[2]))
        else:
            # Shortest interior; ties: lowest top, then rightmost column.
            best = min(segments, key=lambda seg: (seg[0], -seg[1], -seg[2]))

        interior_len, top, col, bottom, _ = best
        if interior_len > 0:
            out[top + 1:bottom, col] = cap_color

    return out