import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Slide each connected 0-object on the left of the vertical 5-line rightward
    until it touches the line.  Then paint 2 on the right side of the line in
    rows corresponding to the vertical contact runs described by the examples.
    """
    if grid is None or grid.ndim != 2:
        return grid

    h, w = grid.shape

    # Locate the vertical divider made of 5s.
    five_counts = (grid == 5).sum(axis=0)
    if five_counts.max() == 0:
        return grid.copy()
    divider = int(np.argmax(five_counts))

    # Start with the background color 6; restore the 5s at the end.
    out = np.full((h, w), 6, dtype=grid.dtype)

    # Find 4-connected components of 0s strictly left of the divider.
    left_zero_mask = (grid == 0) & (np.arange(w)[None, :] < divider)
    visited = np.zeros_like(left_zero_mask, dtype=bool)

    def get_component(r, c):
        stack = [(r, c)]
        visited[r, c] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h
                    and 0 <= nx < divider
                    and left_zero_mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        return comp

    for r in range(h):
        for c in range(divider):
            if left_zero_mask[r, c] and not visited[r, c]:
                comp = get_component(r, c)

                # Move this component right as far as possible without crossing 5.
                max_col = max(x for _, x in comp)
                shift = (divider - 1) - max_col
                if shift < 0:
                    shift = 0

                boundary_rows = set()
                for y, x in comp:
                    nx = x + shift
                    out[y, nx] = 0
                    if nx == divider - 1:
                        boundary_rows.add(y)

                # Process contiguous vertical runs of contact rows.
                if boundary_rows:
                    rows = sorted(boundary_rows)
                    runs = []
                    start = prev = rows[0]
                    for y in rows[1:]:
                        if y == prev + 1:
                            prev = y
                        else:
                            runs.append((start, prev))
                            start = prev = y
                    runs.append((start, prev))

                    right_width = w - divider - 1
                    if right_width <= 0:
                        continue

                    for top, bottom in runs:
                        length = bottom - top + 1
                        if length < 3:
                            continue

                        if length == 3:
                            red_rows = [top + 1]
                        else:
                            height = min(3, right_width)
                            red_rows = range(top + 1, top + 1 + height)

                        for y in red_rows:
                            if 0 <= y < h:
                                out[y, divider + 1 : w] = 2

    # Preserve the original 5s.
    out[grid == 5] = 5
    return out