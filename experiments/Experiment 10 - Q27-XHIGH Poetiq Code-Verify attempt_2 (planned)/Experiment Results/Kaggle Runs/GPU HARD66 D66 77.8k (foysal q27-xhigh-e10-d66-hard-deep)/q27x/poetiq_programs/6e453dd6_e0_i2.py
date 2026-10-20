import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move each 4-connected 0 component on the left of the vertical 5 separator
    right until it touches the column just left of the separator.

    For every component, mark rows where it has a pixel in its rightmost column
    but no pixel immediately to the left of that pixel. In the output, all
    cells to the right of the 5 separator in marked rows become 2.
    """
    g = np.array(grid, dtype=int, copy=True)

    # Gracefully handle empty or non-2D inputs.
    if g.size == 0 or g.ndim != 2:
        return g

    h, w = g.shape

    # Find the vertical separator made of 5s.
    five_counts = (g == 5).sum(axis=0)
    if five_counts.max() == 0:
        return g

    full_cols = np.where(five_counts == h)[0]
    if full_cols.size > 0:
        sep = int(full_cols[0])
    else:
        sep = int(np.argmax(five_counts))

    # Output starts as 6, with the separator column fixed as 5.
    out = np.full((h, w), 6, dtype=int)
    out[:, sep] = 5

    # If there is no left side, nothing else can be transformed.
    left_w = sep
    if left_w <= 0:
        return out

    boundary = left_w - 1  # column just left of the separator
    left = g[:, :left_w]
    new_left = np.full((h, left_w), 6, dtype=int)
    visited = np.zeros((h, left_w), dtype=bool)
    mark_rows = np.zeros(h, dtype=bool)

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # Process each 4-connected 0 component on the left.
    for r in range(h):
        for c in range(left_w):
            if left[r, c] != 0 or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            max_c = c

            # Collect the whole component.
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if cc > max_c:
                    max_c = cc

                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < left_w
                        and left[nr, nc] == 0
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            # Shift this component to touch the wall.
            shift = boundary - max_c
            if shift < 0:
                shift = 0

            for rr, cc in cells:
                nc = cc + shift
                if 0 <= nc < left_w:
                    new_left[rr, nc] = 0

            # Mark rows where the component has a rightmost pixel with no
            # pixel immediately to its left.
            if max_c == 0:
                for rr, cc in cells:
                    if cc == 0:
                        mark_rows[rr] = True
            else:
                for rr, cc in cells:
                    if cc == max_c and left[rr, max_c - 1] != 0:
                        mark_rows[rr] = True

    # Place transformed left side.
    out[:, :left_w] = new_left

    # Paint marked rows to the right of the separator with 2.
    if sep + 1 < w and np.any(mark_rows):
        out[mark_rows, sep + 1:] = 2

    return out