import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 4-connected 0-region on the left of the vertical 5-line rightward
    until it touches the line.  Then mark 2s on the right side of the line in
    rows determined by vertical zero-runs on the left/right edges of the region.

    Details:
    - The separator is the column containing the vertical line of 5s.
    - Each zero component is shifted so its rightmost column becomes
      separator_col - 1.
    - For each component, scan its leftmost and rightmost bounding-box columns.
      For every contiguous vertical run of zeros of length at least 3:
        * normally, mark the rows strictly inside the run;
        * for a 3-cell run ending at the bottom grid edge, mark its top row
          (matching the provided examples).
    - All columns to the right of the separator are filled with 2 in marked rows.
    """
    if grid is None or grid.ndim != 2:
        return grid

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    # Locate the vertical 5 separator. Prefer a full column if present.
    five_counts = (grid == 5).sum(axis=0)
    full_five_cols = np.where(five_counts == h)[0]
    if full_five_cols.size > 0:
        divider = int(full_five_cols[0])
    else:
        if five_counts.max() == 0:
            return grid.copy()
        divider = int(np.argmax(five_counts))

    # Start from background 6, preserving all original 5s.
    out = np.full((h, w), 6, dtype=grid.dtype)
    out[grid == 5] = 5

    red_rows = set()

    # If there is no left side or no right side, nothing else can happen.
    if divider <= 0 or divider + 1 >= w:
        return out

    left_zero_mask = (grid == 0) & (np.arange(w)[None, :] < divider)
    visited = np.zeros((h, w), dtype=bool)

    def add_vertical_run(start: int, end: int) -> None:
        """Add rows from a vertical zero run to the red-row set."""
        length = end - start + 1
        if length < 3:
            return

        # Edge case from the examples: a 3-cell run flush with the bottom edge
        # marks its top row instead of an interior row (there is no interior).
        if length == 3 and end == h - 1 and start > 0:
            red_rows.add(start)
        else:
            red_rows.update(range(start + 1, end))

    # Find all 4-connected zero components strictly left of the separator.
    for r in range(h):
        for c in range(divider):
            if not left_zero_mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            min_r = max_r = r
            min_c = max_c = c

            while stack:
                y, x = stack.pop()
                comp.append((y, x))

                if y < min_r:
                    min_r = y
                if y > max_r:
                    max_r = y
                if x < min_c:
                    min_c = x
                if x > max_c:
                    max_c = x

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

            # Shift the whole component right until it touches the separator.
            shift = (divider - 1) - max_c
            if shift < 0:
                shift = 0

            for y, x in comp:
                out[y, x + shift] = 0

            # Determine red rows from vertical zero runs on the component's
            # leftmost and rightmost bounding-box columns.
            comp_set = set(comp)
            for col in {min_c, max_c}:
                run_start = None
                for y in range(min_r, max_r + 1):
                    if (y, col) in comp_set:
                        if run_start is None:
                            run_start = y
                    else:
                        if run_start is not None:
                            add_vertical_run(run_start, y - 1)
                            run_start = None
                if run_start is not None:
                    add_vertical_run(run_start, max_r)

    # Paint red on the right side of the separator for all marked rows.
    for r in red_rows:
        if 0 <= r < h:
            out[r, divider + 1:] = 2

    # Preserve separator and any other original 5s.
    out[grid == 5] = 5
    return out