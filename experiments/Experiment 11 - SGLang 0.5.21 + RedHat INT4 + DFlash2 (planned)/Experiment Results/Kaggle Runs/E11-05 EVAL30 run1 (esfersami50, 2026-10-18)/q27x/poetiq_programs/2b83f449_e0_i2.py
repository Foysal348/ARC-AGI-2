import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the ARC rule observed in the examples.

    Steps:
    1. Convert every horizontal run of three 7s into:
       - the same row: 8 6 8
       - the row above and below the middle cell: 6
    2. The 3s are treated as tokens that fall downward on their side
       (left/right). A line row is "supported" on a side if the nearest 6
       on that side is the bottom end of a vertical 6 created from a 7-run.
    3. Tokens fall to the first supported row with available capacity.
       Each side of a row can hold at most two tokens.
    4. Placed tokens occupy priority cells in the 8-runs of that row:
       - For a side, use the support 6 (leftmost/rightmost 6 in the row).
       - Eligible runs are mainly runs on the edge side of that support,
         plus the immediate run on the other side of the support if it is
         "open" (bounded by a 0 or grid edge, not by another 6).
       - Cells are filled from the outer edge of the run toward the support.
    """

    if grid.size == 0:
        return grid.copy()

    out = grid.copy()
    H, W = out.shape

    # ------------------------------------------------------------
    # 1) Convert 7-runs to 8/6 patterns.
    # ------------------------------------------------------------
    bottom_endpoint = np.zeros((H, W), dtype=bool)

    for r in range(H):
        c = 0
        while c < W:
            if out[r, c] == 7:
                start = c
                while c < W and out[r, c] == 7:
                    c += 1
                length = c - start

                if length == 3:
                    mid = start + 1

                    # current row: 8 6 8
                    out[r, start] = 8
                    out[r, mid] = 6
                    out[r, start + 2] = 8

                    # vertical 6s above/below
                    if r - 1 >= 0:
                        out[r - 1, mid] = 6
                    if r + 1 < H:
                        out[r + 1, mid] = 6
                        bottom_endpoint[r + 1, mid] = True
            else:
                c += 1

    # ------------------------------------------------------------
    # 2) Identify line rows: rows that are mostly 8/3/6, not 7/0 data rows.
    # ------------------------------------------------------------
    line_rows = []
    for r in range(H):
        row = out[r]
        if np.count_nonzero(row == 7) > 0:
            continue
        if np.count_nonzero(np.isin(row, [3, 8, 6])) >= max(2, W // 3):
            line_rows.append(r)

    # Fallback: in the examples line rows are even rows.
    if len(line_rows) < 2:
        line_rows = list(range(0, H, 2))

    # ------------------------------------------------------------
    # 3) Determine which side of which line row is supported.
    #    A side is supported if the nearest 6 on that side is a bottom endpoint.
    # ------------------------------------------------------------
    def side_support(r: int, side: str):
        if r < 0 or r >= H:
            return None

        six_cols = np.where(out[r] == 6)[0]
        if len(six_cols) == 0:
            return None

        if side == 'left':
            col = int(six_cols[0])
        else:
            col = int(six_cols[-1])

        if bottom_endpoint[r, col]:
            return col
        return None

    # ------------------------------------------------------------
    # 4) Collect initial 3 tokens and simulate falling with capacity 2.
    # ------------------------------------------------------------
    left_tokens = []
    right_tokens = []

    for r in line_rows:
        if W > 0 and out[r, 0] == 3:
            left_tokens.append(r)
        if W > 0 and out[r, W - 1] == 3:
            right_tokens.append(r)

    # clear original 3s; they will be re-placed
    out[out == 3] = 8

    left_counts = np.zeros(H, dtype=int)
    right_counts = np.zeros(H, dtype=int)
    MAX_PER_SIDE = 2

    def land_tokens(tokens, counts, side):
        for start_r in tokens:
            placed = False
            for r in range(start_r, H):
                if side_support(r, side) is not None and counts[r] < MAX_PER_SIDE:
                    counts[r] += 1
                    placed = True
                    break
            # If no supported row is found, the token is dropped.

    land_tokens(left_tokens, left_counts, 'left')
    land_tokens(right_tokens, right_counts, 'right')

    # ------------------------------------------------------------
    # 5) Helper: split a row into 8-runs (non-zero cells that are not 6).
    # ------------------------------------------------------------
    def get_runs(r: int):
        runs = []
        c = 0
        while c < W:
            if out[r, c] == 8:
                l = c
                while c < W and out[r, c] == 8:
                    c += 1
                runs.append((l, c - 1))
            else:
                c += 1
        return runs

    def boundary(r: int, c: int, direction: int):
        # direction = -1 left, +1 right
        nc = c + direction
        if nc < 0 or nc >= W:
            return 'edge'
        v = out[r, nc]
        if v == 0:
            return 'zero'
        if v == 6:
            return 'six'
        return 'eight'

    def is_open_boundary(r: int, c: int, direction: int):
        b = boundary(r, c, direction)
        return b in ('edge', 'zero')

    def eligible_cells(r: int, side: str, n: int):
        if n <= 0:
            return []

        s = side_support(r, side)
        if s is None:
            return []

        runs = get_runs(r)
        if not runs:
            return []

        # Candidate runs with a priority key.
        candidates = []  # (priority, far_col, run_index, step)

        for idx, (l, rr) in enumerate(runs):
            # We only want runs that are adjacent to, or in the edge direction
            # from, the support 6.
            if side == 'left':
                # Left side: prefer runs to the left of support.
                if rr < s:
                    # closer to support => smaller priority number
                    priority = (0, s - rr, idx)
                    far_col = l
                    step = 1
                elif l == s + 1:
                    # immediate right of support, used only as fallback
                    left_bound = boundary(r, l, -1)
                    if left_bound in ('edge', 'zero'):
                        priority = (1, idx, 0)
                        far_col = r
                        # For a left-side fallback run to the right of support,
                        # fill from the outer right edge inward.
                        far_col = rr
                        step = -1
                    else:
                        continue
                else:
                    continue
            else:
                # Right side: prefer runs to the right of support.
                if l > s:
                    priority = (0, l - s, idx)
                    far_col = rr
                    step = -1
                elif rr == s - 1:
                    # immediate left of support, used only as fallback
                    right_bound = boundary(r, rr, 1)
                    if right_bound in ('edge', 'zero'):
                        priority = (1, idx, 0)
                        far_col = l
                        step = 1
                    else:
                        continue
                else:
                    continue

            candidates.append((priority, far_col, idx, step))

        if not candidates:
            return []

        # Sort by priority. Ties are broken by run index to make behavior stable.
        candidates.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

        chosen = []
        # Round-robin across eligible runs: one cell from each run first,
        # then return to the first runs if more tokens are needed.
        while len(chosen) < n:
            added = False
            for _, far_col, idx, step in candidates:
                if len(chosen) >= n:
                    break
                l, rr = runs[idx]

                # Build the ordered list of cells in this run from far end inward.
                if step == 1:
                    cells = list(range(l, rr + 1))
                else:
                    cells = list(range(rr, l - 1, -1))

                # If the far cell is already chosen, try the next cell in the run.
                placed = False
                for cell in cells:
                    if cell not in chosen:
                        chosen.append(cell)
                        placed = True
                        break
                added = added or placed
            if not added:
                break

        return chosen[:n]

    # ------------------------------------------------------------
    # 6) Place 3s for each side/row.
    # ------------------------------------------------------------
    for r in line_rows:
        left_cells = eligible_cells(r, 'left', int(left_counts[r]))
        right_cells = eligible_cells(r, 'right', int(right_counts[r]))

        # A cell can be claimed by only one side; resolve conflicts by
        # giving priority to the side whose count is larger, then left.
        all_cells = []
        for c in left_cells:
            all_cells.append(('left', c))
        for c in right_cells:
            all_cells.append(('right', c))

        # Remove duplicates, preferring left if tie.
        seen = set()
        final_cells = []
        for side, c in all_cells:
            if c in seen:
                continue
            seen.add(c)
            final_cells.append(c)

        # If too many duplicates caused shortage, fill remaining requested count
        # with safe edge cells that are still 8.
        needed = int(left_counts[r]) + int(right_counts[r])
        while len(final_cells) < needed:
            # prefer left edge then right edge then any 8
            placed = False
            for c in range(W):
                if c not in seen and out[r, c] == 8:
                    final_cells.append(c)
                    seen.add(c)
                    placed = True
                    break
            if not placed:
                break

        for c in final_cells:
            if 0 <= c < W and out[r, c] == 8:
                out[r, c] = 3

    return out