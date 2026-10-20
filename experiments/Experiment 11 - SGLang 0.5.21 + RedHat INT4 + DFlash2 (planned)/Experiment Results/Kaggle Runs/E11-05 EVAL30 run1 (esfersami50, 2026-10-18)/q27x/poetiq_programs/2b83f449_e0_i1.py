import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid as follows:

    1. Every length-3 horizontal run of 7s becomes:
       - a vertical line of 6s through its middle cell,
         extending one row above and one row below,
       - 8s in the two side cells of the original horizontal run.

    2. The 3s are then redistributed.
       There are "left" and "right" tokens originally at the left/right ends
       of the horizontal grid lines. Each token falls downward along its side
       until it reaches the first supported row on that side.
       A side is supported if the first/last 6 in that row is the bottom end
       of a vertical 6 created from a 7-run in the row above.

       Tokens land in available edge positions, preferring the outermost cell,
       then the adjacent cell, respecting zeros and already occupied cells.
    """
    out = grid.copy()
    H, W = out.shape

    # Step 1: replace horizontal 7-runs with 8-6-8 and vertical 6s.
    # Also remember which 6s are bottom endpoints from a 7 above.
    bottom_endpoint = np.zeros((H, W), dtype=bool)

    for r in range(H):
        c = 0
        while c < W:
            if out[r, c] == 7:
                # find run length
                start = c
                while c < W and out[r, c] == 7:
                    c += 1
                length = c - start

                if length == 3:
                    mid = start + 1

                    # middle row arms and center
                    out[r, start] = 8
                    out[r, mid] = 6
                    out[r, start + 2] = 8

                    # vertical 6s above and below
                    if r - 1 >= 0:
                        out[r - 1, mid] = 6
                    if r + 1 < H:
                        out[r + 1, mid] = 6
                        # bottom endpoint of this vertical segment
                        bottom_endpoint[r + 1, mid] = True
                # else: ignore malformed runs, but continue safely
            else:
                c += 1

    # Identify horizontal grid line rows.
    # In this task they are the rows that contain many 8/3/0 cells.
    line_rows = []
    for r in range(H):
        # A line row should have at least two 8/3/0 and not be mostly empty.
        # More robustly: rows where most non-zero cells are 8/3/0.
        row_vals = out[r]
        non_zero = row_vals[row_vals != 0]
        if len(non_zero) == 0:
            continue
        line_like = np.sum(np.isin(row_vals, [0, 3, 8, 6]))
        if line_like >= max(2, W // 2):
            line_rows.append(r)

    # If detection is too weak, fall back to even rows if that fits.
    if len(line_rows) < 2:
        line_rows = list(range(0, H, 2))

    # For each side, collect token start rows.
    # Tokens are input 3s at the left/right ends of line rows.
    left_tokens = []
    right_tokens = []
    for r in line_rows:
        if r == H - 1:
            # bottom row in examples usually has no starting 3s,
            # but if present, still allow it as a token start.
            pass
        if W > 0 and out[r, 0] == 3:
            left_tokens.append(r)
        if W > 0 and out[r, W - 1] == 3:
            right_tokens.append(r)

    # Helper: is a given row supported on the side?
    def supported(row, side):
        if row < 0 or row >= H:
            return False
        # Need at least one 6 in this row.
        cols6 = np.where(out[row] == 6)[0]
        if len(cols6) == 0:
            return False

        if side == 'left':
            c = cols6[0]
        else:
            c = cols6[-1]

        # Supported if this 6 is the bottom endpoint of a vertical 6
        # created from a 7-run in the row above.
        return bool(bottom_endpoint[row, c])

    # For each token, find first supported row at or below it.
    side_counts = {'left': np.zeros(H, dtype=int), 'right': np.zeros(H, dtype=int)}
    for side, tokens in (('left', left_tokens), ('right', right_tokens)):
        for start_r in tokens:
            for r in range(start_r, H):
                if supported(r, side):
                    side_counts[side][r] += 1
                    break

    # Clear original 3s before placing new ones.
    out[out == 3] = 8

    # Place 3s for each side/row according to counts.
    for r in range(H):
        for side in ('left', 'right'):
            need = int(side_counts[side][r])
            if need <= 0:
                continue

            placed = 0

            # Candidate columns for this side, from outer edge inward.
            if side == 'left':
                candidates = [0, 1, 2, 3]
            else:
                candidates = [W - 1, W - 2, W - 3, W - 4]

            # We can place up to 2 tokens normally.
            for c in candidates:
                if need == 0:
                    break
                if c < 0 or c >= W:
                    continue

                # Do not overwrite 6s, zeros, or existing 3s.
                if out[r, c] in (6, 0, 3):
                    continue

                # Prefer cells that were originally line cells (8).
                # This prevents placing into interior background.
                if grid[r, c] not in (3, 8):
                    continue

                out[r, c] = 3
                need -= 1
                placed += 1

            # If zeros split the side, allow one additional placement adjacent
            # to a zero on that side, matching the examples.
            if need > 0:
                if side == 'left':
                    # search from left to right for a zero followed by usable cell
                    for c in range(0, min(W - 1, 10)):
                        if out[r, c] == 0 and out[r, c + 1] not in (0, 3, 6) and grid[r, c + 1] in (3, 8):
                            out[r, c + 1] = 3
                            need -= 1
                            break
                else:
                    for c in range(W - 1, max(W - 11, -1), -1):
                        if out[r, c] == 0 and c - 1 >= 0 and out[r, c - 1] not in (0, 3, 6) and grid[r, c - 1] in (3, 8):
                            out[r, c - 1] = 3
                            need -= 1
                            break

    return out