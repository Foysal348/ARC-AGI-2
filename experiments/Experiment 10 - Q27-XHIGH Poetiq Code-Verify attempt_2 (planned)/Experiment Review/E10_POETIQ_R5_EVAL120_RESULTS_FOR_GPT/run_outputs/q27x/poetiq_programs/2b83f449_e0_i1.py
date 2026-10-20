import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC grid:
      - Replace horizontal runs of 7 by 8,6,8.
      - Propagate the center of each 7-run into adjacent wall rows as 6.
      - Add 3s on wall-row segment ends according to how the horizontal span
        of 7-centers changes between the gap row above and below.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    if W == 0:
        return grid.copy()

    out = grid.copy()

    def runs_of_7(row):
        """Return (start, end, center) for each contiguous run of 7s."""
        runs = []
        c = 0
        while c < W:
            if row[c] == 7:
                start = c
                while c < W and row[c] == 7:
                    c += 1
                end = c - 1
                center = (start + end) // 2
                runs.append((start, end, center))
            else:
                c += 1
        return runs

    # Extract 7-runs and their outer center spans for every row.
    runs_by_row = [runs_of_7(grid[r]) for r in range(H)]
    span_by_row = []
    for runs in runs_by_row:
        if runs:
            centers = [center for _, _, center in runs]
            span_by_row.append((min(centers), max(centers)))
        else:
            span_by_row.append(None)

    # Wall rows are the rows that contain the 3/8 border material.
    wall_rows = [r for r in range(H) if np.any((grid[r] == 3) | (grid[r] == 8))]

    # Process wall rows.
    for r in wall_rows:
        # Reset wall row to its base material: 0 stays 0, 3/8 become 8.
        out[r, :] = np.where(grid[r] == 0, 0, 8)

        # Add 6 markers from centers of adjacent 7-runs.
        for nr in (r - 1, r + 1):
            if 0 <= nr < H:
                for _, _, c in runs_by_row[nr]:
                    if grid[r, c] != 0:  # holes remain holes
                        out[r, c] = 6

        above = span_by_row[r - 1] if r > 0 else None
        below = span_by_row[r + 1] if r + 1 < H else None

        active_left = False
        active_right = False
        t_left = 1
        t_right = 1

        # Determine active sides and their thickness (1 or 2).
        if above is not None:
            a_min, a_max = above
            if below is not None:
                b_min, b_max = below

                # Upper span extends farther left -> left side active.
                if a_min < b_min:
                    active_left = True
                    delta = b_min - a_min
                    t_left = 2 if delta % 2 == 0 else 1

                # Upper span extends farther right -> right side active.
                if a_max > b_max:
                    active_right = True
                    delta = a_max - b_max
                    t_right = 2 if delta % 2 == 0 else 1
            else:
                # Non-empty upper span collapsing to an empty lower span.
                active_left = True
                active_right = True
                t_left = 1
                t_right = 1

        if active_left or active_right:
            left_mid = (W - 1) // 2
            right_mid = W // 2

            c = 0
            while c < W:
                if out[r, c] != 0:
                    l = c
                    while c < W and out[r, c] != 0:
                        c += 1
                    seg_end = c - 1

                    contains6 = bool(np.any(out[r, l:c] == 6))

                    # Paint from the left end of this segment if applicable.
                    if active_left:
                        if (contains6 and l <= left_mid) or (
                            l == 0 and grid[r, 0] == 3 and l <= left_mid
                        ):
                            count = t_left
                            cc = l
                            while cc <= seg_end and count > 0:
                                if out[r, cc] != 6:
                                    out[r, cc] = 3
                                    count -= 1
                                cc += 1

                    # Paint from the right end of this segment if applicable.
                    if active_right:
                        if (contains6 and seg_end >= right_mid) or (
                            seg_end == W - 1 and grid[r, W - 1] == 3 and seg_end >= right_mid
                        ):
                            count = t_right
                            cc = seg_end
                            while cc >= l and count > 0:
                                if out[r, cc] != 6:
                                    out[r, cc] = 3
                                    count -= 1
                                cc -= 1
                else:
                    c += 1

    # Replace every 7-run in gap rows by 8 ... 8 with a 6 in the center.
    for r, runs in enumerate(runs_by_row):
        for start, end, center in runs:
            out[r, start:end + 1] = 8
            out[r, center] = 6

    return out