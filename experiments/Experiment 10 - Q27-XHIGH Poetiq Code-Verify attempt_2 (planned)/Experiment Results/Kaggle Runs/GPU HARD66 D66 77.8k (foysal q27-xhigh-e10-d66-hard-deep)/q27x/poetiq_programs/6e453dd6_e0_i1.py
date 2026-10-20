import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Push each 4-connected 0-object on the left side of the vertical 5-line
    rightward until it touches the wall.  On the right side of the wall,
    fill marked rows with 2 according to the object's final right-edge profile.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    h, w = g.shape

    # Find the vertical separator made of 5s.
    five_counts = (g == 5).sum(axis=0)
    if five_counts.max() == 0:
        return g

    full_cols = np.where(five_counts == h)[0]
    if len(full_cols) > 0:
        sep = int(full_cols[0])
    else:
        sep = int(np.argmax(five_counts))

    # Output background is 6, with the separator fixed.
    out = np.full((h, w), 6, dtype=int)
    out[:, sep] = 5

    left_w = sep
    if left_w <= 0:
        return out

    left = g[:, :left_w]
    new_left = np.full((h, left_w), 6, dtype=int)
    visited = np.zeros((h, left_w), dtype=bool)
    mark_rows = set()

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(h):
        for c in range(left_w):
            if left[r, c] != 0 or visited[r, c]:
                continue

            # Extract one 4-connected component of 0s.
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            min_c = left_w
            max_c = -1
            row_max = {}

            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))

                if cc < min_c:
                    min_c = cc
                if cc > max_c:
                    max_c = cc
                if row_max.get(rr, -1) < cc:
                    row_max[rr] = cc

                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < left_w
                        and left[nr, nc] == 0
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if not comp:
                continue

            # Push this component to the wall.
            shift = left_w - 1 - max_c
            if shift < 0:
                shift = 0

            for rr, cc in comp:
                nc = cc + shift
                if 0 <= nc < left_w:
                    new_left[rr, nc] = 0

            # Rows whose component touches the wall after shifting.
            m = max_c
            wall_rows = sorted(rr for rr, rm in row_max.items() if rm == m)

            if shift == 0:
                # Already at the wall: mark vertical runs of the rightmost
                # column, excluding only the first row of each run.
                i = 0
                while i < len(wall_rows):
                    start = wall_rows[i]
                    end = start
                    while i + 1 < len(wall_rows) and wall_rows[i + 1] == end + 1:
                        i += 1
                        end = wall_rows[i]

                    for rr in range(start + 1, end + 1):
                        mark_rows.add(rr)

                    i += 1
            else:
                # Moved object: mark rows where the wall cell is 0 and the
                # cell just left of it is a gap/background.
                if m > min_c and m - 1 >= 0:
                    for rr in wall_rows:
                        if left[rr, m - 1] != 0:
                            mark_rows.add(rr)

    out[:, :left_w] = new_left

    if sep + 1 < w:
        for rr in mark_rows:
            out[rr, sep + 1:] = 2

    return out