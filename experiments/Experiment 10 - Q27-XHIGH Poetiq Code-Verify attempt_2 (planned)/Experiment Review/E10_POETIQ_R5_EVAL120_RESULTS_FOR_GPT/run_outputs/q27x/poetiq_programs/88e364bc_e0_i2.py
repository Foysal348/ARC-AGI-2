import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. Detect small rectangular direction indicators.
       - The border color is C.
       - Interior contains marker colors:
           * normally 1 and 2
           * if C == 1, then 5 and 2
       - Direction is from centroid of the 'from' marker to centroid of the
         'to' marker, quantized to one of the 8 directions.

    2. Every 4 is assigned to the nearest indicator rectangle.

    3. Each 4 moves in the assigned direction:
       - Through 0 cells (and already-placed 4s).
       - Orthogonal movement stops before the first non-empty cell.
       - Diagonal movement normally moves while the diagonal target is empty.
       - Special diagonal corner case:
         If the diagonal target and both orthogonal side cells in the forward
         2x2 corner are walls of the indicator color, and the cell diagonally
         beyond one of those side walls is empty, the 4 enters that side wall
         cell and stops. If both side cells qualify, the row-direction side is
         preferred.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    if not np.any(grid == 4):
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    def _prefix(mask: np.ndarray) -> np.ndarray:
        p = np.zeros((H + 1, W + 1), dtype=int)
        p[1:, 1:] = mask.astype(int)
        p = p.cumsum(axis=0).cumsum(axis=1)
        return p

    def _rect_count(pref: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
        return int(
            pref[r2 + 1, c2 + 1]
            - pref[r1, c2 + 1]
            - pref[r2 + 1, c1]
            + pref[r1, c1]
        )

    signs = []
    unique_colors = set(np.unique(grid).tolist())

    # Detect indicator rectangles.
    for C in sorted(unique_colors):
        if C in (0, 2, 4):
            continue

        from_c = 5 if C == 1 else 1
        to_c = 2

        if not np.any(grid == from_c) or not np.any(grid == to_c):
            continue

        allowed = (grid == C) | (grid == from_c) | (grid == to_c)
        bad = ~allowed

        from_mask = grid == from_c
        to_mask = grid == to_c

        bad_pref = _prefix(bad)
        from_pref = _prefix(from_mask)
        to_pref = _prefix(to_mask)

        not_c = grid != C
        not_c_pref = _prefix(not_c)

        def _row_not_c(r: int, c1: int, c2: int) -> int:
            return int(
                not_c_pref[r + 1, c2 + 1]
                - not_c_pref[r, c2 + 1]
                - not_c_pref[r + 1, c1]
                + not_c_pref[r, c1]
            )

        def _col_not_c(c: int, r1: int, r2: int) -> int:
            return int(
                not_c_pref[r2 + 1, c + 1]
                - not_c_pref[r2 + 1, c]
                - not_c_pref[r1, c + 1]
                + not_c_pref[r1, c]
            )

        candidates = []

        # Enumerate candidate rectangles of height and width at least 3.
        for r1 in range(H - 2):
            for r2 in range(r1 + 2, H):
                col_ok_pref = [0] * (W + 1)

                # For each column, check whether the vertical strip is allowed.
                for c in range(W):
                    bad_count = int(
                        bad_pref[r2 + 1, c + 1]
                        - bad_pref[r1, c + 1]
                        - bad_pref[r2 + 1, c]
                        + bad_pref[r1, c]
                    )
                    col_ok_pref[c + 1] = col_ok_pref[c] + (1 if bad_count == 0 else 0)

                for c1 in range(W - 2):
                    # c1 must itself be an allowed column in this row span.
                    if col_ok_pref[c1 + 1] == col_ok_pref[c1]:
                        continue

                    for c2 in range(c1 + 2, W):
                        width = c2 - c1 + 1

                        # Every column between c1 and c2 must be allowed.
                        if col_ok_pref[c2 + 1] - col_ok_pref[c1] != width:
                            continue

                        # The full perimeter must be color C.
                        if (
                            _row_not_c(r1, c1, c2)
                            or _row_not_c(r2, c1, c2)
                            or _col_not_c(c1, r1, r2)
                            or _col_not_c(c2, r1, r2)
                        ):
                            continue

                        # Must contain both marker colors.
                        if _rect_count(from_pref, r1, c1, r2, c2) == 0:
                            continue
                        if _rect_count(to_pref, r1, c1, r2, c2) == 0:
                            continue

                        candidates.append((r1, r2, c1, c2))

        if not candidates:
            continue

        # Keep larger candidate rectangles if they contain smaller ones.
        uniq = list(set(candidates))
        uniq.sort(
            key=lambda x: (x[1] - x[0] + 1) * (x[3] - x[2] + 1),
            reverse=True,
        )

        maximal = []
        for a in uniq:
            contained = False
            for b in maximal:
                if (
                    b[0] <= a[0]
                    and b[1] >= a[1]
                    and b[2] <= a[2]
                    and b[3] >= a[3]
                ):
                    contained = True
                    break
            if not contained:
                maximal.append(a)

        # Convert each maximal candidate into a direction sign.
        for r1, r2, c1, c2 in maximal:
            sub_from = from_mask[r1:r2 + 1, c1:c2 + 1]
            sub_to = to_mask[r1:r2 + 1, c1:c2 + 1]

            frs, fcs = np.where(sub_from)
            trs, tcs = np.where(sub_to)

            if len(frs) == 0 or len(trs) == 0:
                continue

            from_r = r1 + float(frs.mean())
            from_c = c1 + float(fcs.mean())
            to_r = r1 + float(trs.mean())
            to_c = c1 + float(tcs.mean())

            eps = 1e-6

            if to_r > from_r + eps:
                dr = 1
            elif to_r < from_r - eps:
                dr = -1
            else:
                dr = 0

            if to_c > from_c + eps:
                dc = 1
            elif to_c < from_c - eps:
                dc = -1
            else:
                dc = 0

            if dr == 0 and dc == 0:
                continue

            signs.append((C, dr, dc, r1, r2, c1, c2))

    fours = sorted(zip(*(np.where(grid == 4)).tolist()))

    if not signs or not fours:
        return out

    # Clear original 4s so they do not block paths.
    for r, c in fours:
        out[r, c] = 0

    def _inb(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    # Move each 4 according to its nearest sign.
    for r0, c0 in fours:
        best_sign = None
        best_key = None

        for C, dr, dc, r1, r2, c1, c2 in signs:
            if r0 < r1:
                drr = r1 - r0
            elif r0 > r2:
                drr = r0 - r2
            else:
                drr = 0

            if c0 < c1:
                dcc = c1 - c0
            elif c0 > c2:
                dcc = c0 - c2
            else:
                dcc = 0

            dist2 = drr * drr + dcc * dcc
            key = (dist2, C, r1, c1, r2, c2)

            if best_key is None or key < best_key:
                best_key = key
                best_sign = (C, dr, dc)

        if best_sign is None:
            continue

        C, dr, dc = best_sign
        cr, cc = r0, c0

        while True:
            nr, nc = cr + dr, cc + dc

            if not _inb(nr, nc):
                break

            val = out[nr, nc]

            # Empty cells and already placed 4s are passable.
            if val == 0 or val == 4:
                cr, cc = nr, nc
                continue

            # Diagonal special corner-door case.
            if dr != 0 and dc != 0 and val == C:
                s1r, s1c = cr + dr, cc      # row-direction side
                s2r, s2c = cr, cc + dc      # column-direction side

                chosen = None

                if (
                    _inb(s1r, s1c)
                    and _inb(s2r, s2c)
                    and out[s1r, s1c] == C
                    and out[s2r, s2c] == C
                ):
                    # Prefer the row-direction side if it has an open diagonal gap.
                    gr, gc = s1r + dr, s1c + dc
                    if _inb(gr, gc) and out[gr, gc] == 0:
                        chosen = (s1r, s1c)
                    else:
                        gr, gc = s2r + dr, s2c + dc
                        if _inb(gr, gc) and out[gr, gc] == 0:
                            chosen = (s2r, s2c)

                if chosen is not None:
                    cr, cc = chosen
                    break

            # Normal stopping condition.
            break

        if out[cr, cc] != 4:
            out[cr, cc] = 4

    return out