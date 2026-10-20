import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the observed ARC rule.

    The main operations are:
    1. Every horizontal run of 7s is replaced by a 3-row vertical marker:
       - middle column becomes 6
       - the two neighboring columns become 8
       in the row of the run and the two adjacent rows.

    2. For each horizontal "8-line" row, compare the number of 7-runs
       in the row above and below:
       - above > below : mark both ends
       - above < below : mark neither end
       - above == below: mark the side where the upper 7-runs are
         generally to the left/right of the lower 7-runs

    3. Marking ends / gaps:
       - If the row has no gaps (0s), selected existing endpoint 3s get
         one extra adjacent 3; rows without original 3s get a single
         endpoint 3.
       - If the row has gaps, selected existing endpoint 3s are kept,
         and for each gap a 3 is added next to the more relevant
         non-zero segment (prefer segments containing 6s; if both do,
         use the longer segment).
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = grid.astype(int, copy=True)

    def runs_and_centers(row):
        """Return maximal contiguous runs of 7s and their centers."""
        runs = []
        centers = []
        i = 0
        n = len(row)
        while i < n:
            if row[i] == 7:
                j = i + 1
                while j < n and row[j] == 7:
                    j += 1
                s, e = i, j - 1
                c = (s + e) // 2
                runs.append((s, e))
                centers.append(c)
                i = j
            else:
                i += 1
        return runs, centers

    # ------------------------------------------------------------------
    # Step 1: replace every 7-run with the 3x3 6/8 marker.
    # ------------------------------------------------------------------
    set8 = set()
    set6 = set()

    for r in range(H):
        runs, centers = runs_and_centers(grid[r])
        for (s, e), c in zip(runs, centers):
            # 3x3 marker around the center column
            for dr in (-1, 0, 1):
                rr = r + dr
                if 0 <= rr < H:
                    if c - 1 >= 0:
                        set8.add((rr, c - 1))
                    set6.add((rr, c))
                    if c + 1 < W:
                        set8.add((rr, c + 1))

            # Ensure the whole original run disappears in the middle row
            for cc in range(s, e + 1):
                if cc == c:
                    set6.add((r, cc))
                else:
                    set8.add((r, cc))

    for r, c in set8:
        out[r, c] = 8
    for r, c in set6:
        out[r, c] = 6

    # ------------------------------------------------------------------
    # Identify the horizontal "line" rows (rows containing 8 in input).
    # ------------------------------------------------------------------
    line_rows = [r for r in range(H) if np.any(grid[r] == 8)]
    line_row_set = set(line_rows)

    # Remove all original 3s first; selected ones will be added back.
    for r, c in np.argwhere(grid == 3):
        if out[r, c] != 6:
            out[r, c] = 8 if r in line_row_set else 0

    def set3(r, c):
        """Set a cell to 3 unless it is a 6 marker."""
        if 0 <= r < H and 0 <= c < W and out[r, c] != 6:
            out[r, c] = 3

    # ------------------------------------------------------------------
    # Step 2 and 3: place green 3s on each line row.
    # ------------------------------------------------------------------
    for r in line_rows:
        # Count 7-runs above and below this line row.
        if r > 0:
            above_runs, above_centers = runs_and_centers(grid[r - 1])
        else:
            above_runs, above_centers = [], []

        if r + 1 < H:
            below_runs, below_centers = runs_and_centers(grid[r + 1])
        else:
            below_runs, below_centers = [], []

        A = len(above_runs)
        B = len(below_runs)

        sides = []  # 0 = left, 1 = right

        if A > B:
            sides = [0, 1]
        elif A < B:
            sides = []
        else:
            if A == 0:
                sides = []
            else:
                ac = sorted(above_centers)
                bc = sorted(below_centers)

                # Prefer a clear left/right ordering if present.
                if all(a < b for a, b in zip(ac, bc)):
                    sides = [0]
                elif all(a > b for a, b in zip(ac, bc)):
                    sides = [1]
                else:
                    # Fallback: compare average/sum of centers.
                    sa, sb = sum(ac), sum(bc)
                    if sa < sb:
                        sides = [0]
                    elif sa > sb:
                        sides = [1]
                    else:
                        sides = []

        if not sides:
            continue

        row = out[r]
        zeros = np.where(row == 0)[0]
        has_zero = len(zeros) > 0

        # Original 3 positions in this row.
        orig3 = [c for c in range(W) if grid[r, c] == 3]

        # Keep selected original endpoint 3s.
        for c in orig3:
            side = 0 if c * 2 < W else 1
            if side in sides:
                set3(r, c)

        kept_left = False
        kept_right = False
        for c in orig3:
            if out[r, c] == 3:
                if c * 2 < W:
                    kept_left = True
                else:
                    kept_right = True

        # ------------------------------------------------------------
        # No gaps in this line row.
        # ------------------------------------------------------------
        if not has_zero:
            for side in sides:
                kept = kept_left if side == 0 else kept_right

                if kept:
                    # Existing endpoint becomes a two-cell endpoint block.
                    if W > 1:
                        set3(r, 1 if side == 0 else W - 2)
                else:
                    # Row has no original 3 on this side: add a single endpoint.
                    c = 0 if side == 0 else W - 1
                    if out[r, c] == 6 and W > 1:
                        set3(r, 1 if side == 0 else W - 2)
                    else:
                        set3(r, c)

        # ------------------------------------------------------------
        # There are gaps (0s) in this line row.
        # ------------------------------------------------------------
        else:
            # Add endpoint 3s only where there is no kept original 3.
            for side in sides:
                kept = kept_left if side == 0 else kept_right
                if kept:
                    continue

                if side == 0:
                    z = int(zeros[0])
                    if z == 0:
                        continue

                    left_len = z
                    if left_len <= 0:
                        continue

                    left_has6 = bool(np.any(out[r, 0:z] == 6))
                    if not left_has6:
                        continue

                    greater = zeros[zeros > z]
                    if z + 1 < W:
                        other_start = z + 1
                        other_end = int(greater[0]) if len(greater) else W
                        other_len = other_end - other_start
                        other_has6 = (
                            bool(np.any(out[r, other_start:other_end] == 6))
                            if other_len > 0
                            else False
                        )
                    else:
                        other_len = 0
                        other_has6 = False

                    if (not other_has6) or (left_len > other_len):
                        set3(r, 0)

                else:
                    z = int(zeros[-1])
                    if z == W - 1:
                        continue

                    right_len = W - 1 - z
                    if right_len <= 0:
                        continue

                    right_has6 = bool(np.any(out[r, z + 1:W] == 6))
                    if not right_has6:
                        continue

                    less = zeros[zeros < z]
                    if z - 1 >= 0:
                        other_end = z
                        other_start = int(less[-1]) + 1 if len(less) else 0
                        other_len = other_end - other_start
                        other_has6 = (
                            bool(np.any(out[r, other_start:other_end] == 6))
                            if other_len > 0
                            else False
                        )
                    else:
                        other_len = 0
                        other_has6 = False

                    if (not other_has6) or (right_len > other_len):
                        set3(r, W - 1)

            # Add one 3 next to each gap, on the more relevant side.
            for i, z in enumerate(zeros):
                z = int(z)

                left_start = int(zeros[i - 1]) + 1 if i > 0 else 0
                left_end = z
                left_len = left_end - left_start

                right_start = z + 1
                right_end = int(zeros[i + 1]) if i + 1 < len(zeros) else W
                right_len = right_end - right_start

                left_has6 = (
                    bool(np.any(out[r, left_start:left_end] == 6))
                    if left_len > 0
                    else False
                )
                right_has6 = (
                    bool(np.any(out[r, right_start:right_end] == 6))
                    if right_len > 0
                    else False
                )

                chosen = None
                if left_has6 and right_has6:
                    if left_len > right_len:
                        chosen = 0
                    elif right_len > left_len:
                        chosen = 1
                    else:
                        chosen = 0  # deterministic tie-break
                elif left_has6:
                    chosen = 0
                elif right_has6:
                    chosen = 1

                if chosen is not None and (len(sides) == 2 or chosen in sides):
                    c = z - 1 if chosen == 0 else z + 1
                    set3(r, c)

    return out