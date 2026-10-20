import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the observed ARC rule.

    The input contains horizontal "bar" rows made mostly of 8s, with 3s at the
    ends of some bars.  Other rows contain one or more horizontal runs of 7s.

    Transformation:
    1. Every length-3 horizontal run of 7s becomes:
         - 8 6 8 in the same row,
         - a 6 directly above the middle cell,
         - a 6 directly below the middle cell.
       Only the lower 6 is treated as a "supporting" 6.

    2. The original 3s are treated as tokens that fall downward on their side:
       - a 3 in column 0 is a left token,
       - a 3 in the last column is a right token.

       A side of a row is "supported" if the outermost 6 on that side is a
       supporting lower 6.  A token falls to the first supported row on its
       side at or below its starting row.  Each side of a row can hold at most
       two tokens.  If a supported side is full, the token continues falling.
       As a final fallback, the bottom row acts as a floor.

    3. All original 3s are changed to 8.  Then the fallen tokens are placed
       back as 3s.  For a supported side, eligible 8-runs are chosen in
       round-robin order, preferring runs closer to the supporting 6.  Cells
       are filled from the far edge of a run toward the support.
       An adjacent inner run may also be used if it is a bounded pocket
       (bounded by a 0 on its far side and has another 6 farther in that
       direction).
    """
    if grid.size == 0:
        return grid.copy()

    g = np.asarray(grid)
    H, W = g.shape

    # We will mutate a copy, but keep the original for token starts and row
    # validity.
    out = g.copy()

    # ------------------------------------------------------------------
    # 1) Convert 7-runs to 8/6 patterns and record supporting lower 6s.
    # ------------------------------------------------------------------
    bottom_endpoint = np.zeros((H, W), dtype=bool)

    for r in range(H):
        c = 0
        while c < W:
            if out[r, c] == 7:
                start = c
                while c < W and out[r, c] == 7:
                    c += 1
                length = c - start

                # The examples only use runs of exactly three 7s.
                if length == 3:
                    mid = start + 1

                    # Same row: 8 6 8.
                    out[r, start] = 8
                    out[r, mid] = 6
                    out[r, start + 2] = 8

                    # Vertical extension.
                    if r - 1 >= 0:
                        out[r - 1, mid] = 6
                    if r + 1 < H:
                        out[r + 1, mid] = 6
                        bottom_endpoint[r + 1, mid] = True
            else:
                c += 1

    # ------------------------------------------------------------------
    # 2) Identify rows that can act as bars / landing rows.
    #    Data rows originally contain 7s; all-zero rows have no bar material.
    # ------------------------------------------------------------------
    def is_bar_row(r: int) -> bool:
        if np.any(g[r] == 7):
            return False
        return bool(np.any(np.isin(g[r], [3, 8])))

    bar = np.array([is_bar_row(r) for r in range(H)], dtype=bool)

    # ------------------------------------------------------------------
    # 3) Support test for a side.
    # ------------------------------------------------------------------
    def support_col(r: int, side: str):
        """
        Return the supporting 6 column for this side/row, or None.

        Left side: the leftmost 6 must be a lower endpoint.
        Right side: the rightmost 6 must be a lower endpoint.
        """
        if not bar[r]:
            return None

        cols = np.where(out[r] == 6)[0]
        if len(cols) == 0:
            return None

        if side == "left":
            c = int(cols[0])
        else:
            c = int(cols[-1])

        if bottom_endpoint[r, c]:
            return c
        return None

    # ------------------------------------------------------------------
    # 4) Collect original 3 tokens and simulate falling with capacity 2.
    # ------------------------------------------------------------------
    MAX_PER_SIDE = 2

    left_tokens = []
    right_tokens = []

    for r in range(H):
        if W > 0 and g[r, 0] == 3:
            left_tokens.append(r)
        if W > 0 and g[r, W - 1] == 3:
            right_tokens.append(r)

    left_counts = np.zeros(H, dtype=int)
    right_counts = np.zeros(H, dtype=int)

    def assign(tokens, counts, side):
        for start in tokens:
            landed = False

            # First try normal supporting rows.
            for r in range(start, H):
                if support_col(r, side) is not None and counts[r] < MAX_PER_SIDE:
                    counts[r] += 1
                    landed = True
                    break

            # Floor fallback: if no supported row has room, use the bottom
            # row if it is a bar row and has room.  This keeps tokens from
            # disappearing in edge cases.
            if not landed and H > 0:
                b = H - 1
                if bar[b] and b >= start and counts[b] < MAX_PER_SIDE:
                    counts[b] += 1

    assign(left_tokens, left_counts, "left")
    assign(right_tokens, right_counts, "right")

    # Remove all original 3s; they will be re-placed according to counts.
    out[out == 3] = 8

    # ------------------------------------------------------------------
    # 5) Placement helpers.
    # ------------------------------------------------------------------
    def get_8_runs(r: int):
        """All maximal horizontal runs of 8s in row r."""
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

    def eligible_cells(r: int, side: str, n: int):
        """
        Choose up to n cells in row r on the given side to paint with 3.
        """
        if n <= 0:
            return []

        s = support_col(r, side)

        # If no 6 support but this row is the bottom floor, place from the
        # relevant outer edge.
        if s is None:
            runs = get_8_runs(r)
            if not runs:
                return []

            cells = []
            if side == "left":
                # Use the leftmost run(s), filling from the left edge inward.
                ordered = sorted(runs, key=lambda x: x[0])
                for l, rr in ordered:
                    if len(cells) >= n:
                        break
                    for c in range(l, rr + 1):
                        if len(cells) >= n:
                            break
                        cells.append(c)
            else:
                ordered = sorted(runs, key=lambda x: -x[0])
                for l, rr in ordered:
                    if len(cells) >= n:
                        break
                    for c in range(rr, l - 1, -1):
                        if len(cells) >= n:
                            break
                        cells.append(c)
            return cells[:n]

        runs = get_8_runs(r)
        if not runs:
            return []

        six_cols = np.where(out[r] == 6)[0]

        candidates = []
        # candidate = (distance, is_inner, run_index, order_function)

        def boundary_is_zero(r_idx: int, c_idx: int) -> bool:
            return 0 <= c_idx < W and out[r_idx, c_idx] == 0

        if side == "left":
            # Outer side: all runs strictly left of the supporting 6.
            for idx, (l, rr) in enumerate(runs):
                if rr < s:
                    distance = s - rr
                    # Fill from the far/left edge toward the support.
                    order = list(range(l, rr + 1))
                    candidates.append((distance, 0, idx, order))

            # Inner pocket immediately right of support.
            for idx, (l, rr) in enumerate(runs):
                if l == s + 1:
                    # Must be bounded by zero on the far side...
                    if boundary_is_zero(r, rr + 1):
                        # ...and there must be another 6 farther to the right.
                        if np.any(six_cols > rr + 1):
                            # Fill from the far/right edge toward the support.
                            order = list(range(rr, l - 1, -1))
                            distance = 1
                            candidates.append((distance, 1, idx, order))
                    break

        else:  # right side
            # Outer side: all runs strictly right of the supporting 6.
            for idx, (l, rr) in enumerate(runs):
                if l > s:
                    distance = l - s
                    # Fill from the far/right edge toward the support.
                    order = list(range(rr, l - 1, -1))
                    candidates.append((distance, 0, idx, order))

            # Inner pocket immediately left of support.
            for idx, (l, rr) in enumerate(runs):
                if rr == s - 1:
                    if boundary_is_zero(r, l - 1):
                        if np.any(six_cols < l - 1):
                            # Fill from the far/left edge toward the support.
                            order = list(range(l, rr + 1))
                            distance = 1
                            candidates.append((distance, 1, idx, order))
                    break

        if not candidates:
            return []

        # Sort by closeness to the support; if tied, prefer outer side.
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        chosen = set()
        result = []
        pointer = {idx: 0 for _, _, idx, _ in candidates}

        # Round-robin over eligible runs: one cell from the first run, then
        # one from the second, etc., until n cells are chosen.
        while len(result) < n:
            added = False
            for dist, is_inner, idx, order in candidates:
                if len(result) >= n:
                    break

                pos = pointer[idx]
                while pos < len(order) and order[pos] in chosen:
                    pos += 1

                if pos < len(order):
                    c = order[pos]
                    chosen.add(c)
                    result.append(c)
                    pointer[idx] = pos + 1
                    added = True

            if not added:
                break

        return result[:n]

    # ------------------------------------------------------------------
    # 6) Paint the fallen tokens.
    #    Resolve rare conflicts by preferring the side with more tokens,
    #    then left, then cells farther from the center.
    # ------------------------------------------------------------------
    final_cells = []

    for r in range(H):
        lc = int(left_counts[r])
        rc = int(right_counts[r])
        if lc + rc == 0:
            continue

        left_cells = eligible_cells(r, "left", lc)
        right_cells = eligible_cells(r, "right", rc)

        row_candidates = []
        for c in left_cells:
            # Priority: higher token count first, then left, then outermost.
            row_candidates.append((lc, 0, abs(c - W // 2), "left", c))
        for c in right_cells:
            row_candidates.append((rc, 1, abs(c - W // 2), "right", c))

        # Prefer more urgent side, then left, then farther from center.
        row_candidates.sort(key=lambda x: (-x[0], x[1], -x[2], x[3]))

        used = set()
        for _, _, _, _, c in row_candidates:
            if c in used:
                continue
            if 0 <= c < W and out[r, c] == 8:
                out[r, c] = 3
                used.add(c)

    return out