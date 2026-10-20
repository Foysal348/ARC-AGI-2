import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input ARC grid according to the observed rule:

    - Track rows: each horizontal 7-run becomes 8-6-8.
    - Rail rows: project 7 centers from adjacent track rows as 6s.
    - Rail 3s are removed and re-placed based on whether the nearest projected
      6 near an edge or a 0-gap comes from the track row above.

    See the detailed explanation above.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2 or g.size == 0:
        return g

    H, W = g.shape
    out = np.zeros_like(g)

    def is_rail_row(row: np.ndarray) -> bool:
        """Rail rows are rows that contain 8 or 3 and do not contain 7."""
        return (np.any(row == 8) or np.any(row == 3)) and not np.any(row == 7)

    def centers_of_sevens(row: np.ndarray) -> list:
        """Return the center columns of every contiguous horizontal 7-run."""
        centers = []
        c = 0
        while c < W:
            if row[c] == 7:
                start = c
                while c < W and row[c] == 7:
                    c += 1
                end = c - 1
                length = end - start + 1
                centers.append(start + length // 2)
            else:
                c += 1
        return centers

    rail_rows = [r for r in range(H) if is_rail_row(g[r])]
    rail_set = set(rail_rows)

    # ------------------------------------------------------------------
    # Process non-rail rows: replace 7-runs by 8-6-8.
    # ------------------------------------------------------------------
    for r in range(H):
        if r in rail_set:
            continue

        out[r] = g[r].copy()
        c = 0
        while c < W:
            if g[r, c] == 7:
                start = c
                while c < W and g[r, c] == 7:
                    c += 1
                end = c - 1
                length = end - start + 1

                out[r, start:end + 1] = 8
                out[r, start + length // 2] = 6
            else:
                c += 1

    # ------------------------------------------------------------------
    # Process rail rows.
    # ------------------------------------------------------------------
    for r in rail_rows:
        row = g[r]

        above = set()
        below = set()

        if r > 0:
            above.update(centers_of_sevens(g[r - 1]))
        if r + 1 < H:
            below.update(centers_of_sevens(g[r + 1]))

        all_centers = above | below

        # Projected 6s only appear where the rail row is not a 0-gap.
        eff = {c for c in all_centers if row[c] != 0}
        eff_above = eff & above

        # Start from the input rail row, but clear original 3 terminals.
        out_row = row.copy()
        out_row[out_row == 3] = 8

        # Place projected 6s.
        for c in eff:
            out_row[c] = 6

        def set_three(c: int) -> None:
            """Set a cell to 3 if it is a valid non-zero, non-6 cell."""
            if 0 <= c < W and out_row[c] != 0 and out_row[c] != 6:
                out_row[c] = 3

        # ----------------------------
        # Edge terminals
        # ----------------------------
        if eff:
            left6 = min(eff)
            right6 = max(eff)

            # Left edge
            if left6 in eff_above:
                connected_to_left_edge = True
                for cc in range(left6 + 1):
                    if row[cc] == 0:
                        connected_to_left_edge = False
                        break

                if (row[0] == 3 or connected_to_left_edge) and row[0] != 0:
                    set_three(0)

                if connected_to_left_edge and W > 1 and left6 <= 4 and row[1] != 0:
                    set_three(1)

            # Right edge
            if right6 in eff_above:
                connected_to_right_edge = True
                for cc in range(right6, W):
                    if row[cc] == 0:
                        connected_to_right_edge = False
                        break

                if (row[W - 1] == 3 or connected_to_right_edge) and row[W - 1] != 0:
                    set_three(W - 1)

                if connected_to_right_edge and W > 1 and (W - 1 - right6) <= 4 and row[W - 2] != 0:
                    set_three(W - 2)

        # ----------------------------
        # Gap terminals
        # ----------------------------
        for c in range(W):
            if row[c] != 0:
                continue

            left_ok = False
            cc = c - 1
            while cc >= 0 and row[cc] != 0:
                if cc in eff:
                    left_ok = (cc in eff_above)
                    break
                cc -= 1

            right_ok = False
            cc = c + 1
            while cc < W and row[cc] != 0:
                if cc in eff:
                    right_ok = (cc in eff_above)
                    break
                cc += 1

            if left_ok and right_ok:
                # Tie: prefer the right side.
                set_three(c + 1)
            elif left_ok:
                set_three(c - 1)
            elif right_ok:
                set_three(c + 1)

        out[r] = out_row

    return out