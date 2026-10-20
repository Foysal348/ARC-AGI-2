import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    1. The top-left cell is a marker. Its value is the color that must be erased
       from the rest of the grid. The marker itself remains at (0, 0).

    2. The rest of the grid consists of open-bottom rectangular frames / "U" shapes.
       Each U shape has:
         - a top horizontal row,
         - a left vertical side,
         - a right vertical side.
       The two sides may have different lengths.

    3. After removing the marked color, the remaining U shapes settle downward:
       - Vertical sides fall as far as possible until they hit the bottom edge or
         an already settled lower frame.
       - Top bars are then placed according to the remaining sides:
           * If both sides remain, the top bar moves to the lower of the two side
             tops, unless blocked by a lower frame.
           * If only one side remains and the top bar has more than the single
             corner cell, it slides down that side to the side's bottom, unless
             blocked by a lower frame.
           * If only the single corner cell remains, it stays with that side.
           * A detached top bar falls to the nearest lower support.

    4. Any remaining isolated pixels not belonging to a detected U shape are dropped
       under ordinary column gravity.
    """
    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape

    if H == 0 or W == 0:
        return grid

    target = int(grid[0, 0])
    orig = grid

    def is_bg(r: int, c: int) -> bool:
        """Background for shape detection: zeros and the fixed marker."""
        if r < 0 or c < 0 or r >= H or c >= W:
            return True
        if r == 0 and c == 0:
            return True
        return orig[r, c] == 0

    def find_support(col: int, start_row: int, arr: np.ndarray) -> int:
        """
        First occupied row strictly below start_row in this column,
        or H if none.
        """
        for r in range(start_row + 1, H):
            if arr[r, col] != 0:
                return r
        return H

    # -------------------------------------------------------------------------
    # Detect vertical side segments.
    #
    # A side is a vertical run of the same color.  The top row may touch a
    # horizontal top bar, but lower rows should be pure vertical side cells.
    # If a lower cell has a horizontal nonzero neighbor, treat it as a horizontal
    # pixel of another structure and split the vertical run there.
    # -------------------------------------------------------------------------
    vertical_segments = []
    vseg_at = {}

    for c in range(W):
        r = 0
        while r < H:
            if is_bg(r, c):
                r += 1
                continue

            color = orig[r, c]
            start = r
            r += 1

            while r < H and not is_bg(r, c) and orig[r, c] == color:
                if r > start:
                    left_neighbor = (c > 0) and (not is_bg(r, c - 1))
                    right_neighbor = (c < W - 1) and (not is_bg(r, c + 1))
                    if left_neighbor or right_neighbor:
                        break
                r += 1

            end = r - 1
            length = end - start + 1

            # Single pixels are usually top-bar cells, not side walls.
            if length >= 2:
                seg = {
                    "id": len(vertical_segments),
                    "col": c,
                    "top": start,
                    "bottom": end,
                    "color": int(color)
                }
                vertical_segments.append(seg)
                # Keep one segment for this exact (top, col).
                vseg_at.setdefault((start, c), seg)

    # -------------------------------------------------------------------------
    # Detect U shapes.
    #
    # For each horizontal nonzero run, look for two vertical segments whose tops
    # lie in that run.  If there are exactly two, they define a U shape.
    # -------------------------------------------------------------------------
    us = []
    covered = np.zeros((H, W), dtype=bool)

    for T in range(H):
        c = 0
        while c < W:
            if is_bg(T, c):
                c += 1
                continue

            a = c
            while c < W and not is_bg(T, c):
                c += 1
            b = c - 1

            cols = []
            for cc in range(a, b + 1):
                seg = vseg_at.get((T, cc))
                if seg is not None:
                    cols.append(cc)

            # In the provided task style, each U top has exactly two vertical sides.
            if len(cols) == 2:
                L, R = cols[0], cols[1]
                left_seg = vseg_at[(T, L)]
                right_seg = vseg_at[(T, R)]

                us.append({
                    "top": T,
                    "L": L,
                    "R": R,
                    "left": left_seg,
                    "right": right_seg
                })

                # Mark original U pixels as covered so they are not treated as strays.
                for cc in range(L, R + 1):
                    if not is_bg(T, cc):
                        covered[T, cc] = True

                for rr in range(left_seg["top"], left_seg["bottom"] + 1):
                    if not is_bg(rr, L):
                        covered[rr, L] = True

                for rr in range(right_seg["top"], right_seg["bottom"] + 1):
                    if not is_bg(rr, R):
                        covered[rr, R] = True

    # -------------------------------------------------------------------------
    # Settle U shapes from bottom to top.
    # -------------------------------------------------------------------------
    lower = np.zeros((H, W), dtype=int)

    top_rows = sorted({u["top"] for u in us}, reverse=True)

    for T in top_rows:
        group = [u for u in us if u["top"] == T]
        group.sort(key=lambda u: u["L"])

        # First finalize all side walls in this horizontal layer.
        side_final = {}

        for u in group:
            for side in (u["left"], u["right"]):
                sid = side["id"]
                if sid in side_final:
                    continue

                # If the side's color is the erased target color, it disappears.
                if side["color"] == target:
                    side_final[sid] = None
                    continue

                support = find_support(side["col"], side["bottom"], lower)
                final_bottom = H - 1 if support == H else support - 1

                # A wall cannot move upward.
                if final_bottom < side["bottom"]:
                    final_bottom = side["bottom"]

                length = side["bottom"] - side["top"] + 1
                final_top = final_bottom - length + 1

                side_final[sid] = {
                    "col": side["col"],
                    "top": final_top,
                    "bottom": final_bottom,
                    "color": side["color"]
                }

        group_grid = np.zeros((H, W), dtype=int)

        # Place finalized side walls.
        for fin in side_final.values():
            if fin is None:
                continue
            for rr in range(fin["top"], fin["bottom"] + 1):
                group_grid[rr, fin["col"]] = fin["color"]

        # Now place remaining top-bar pieces.
        for u in group:
            Ttop = u["top"]
            L = u["L"]
            R = u["R"]

            left_fin = side_final[u["left"]["id"]]
            right_fin = side_final[u["right"]["id"]]

            c = L
            while c <= R:
                if is_bg(Ttop, c) or orig[Ttop, c] == target:
                    c += 1
                    continue

                a = c
                while c <= R and not is_bg(Ttop, c) and orig[Ttop, c] != target:
                    c += 1
                b = c - 1

                # Which remaining side walls is this top piece attached to?
                left_conn = False
                if left_fin is not None:
                    if a <= L + 1 and b >= L - 1:
                        left_conn = True

                right_conn = False
                if right_fin is not None:
                    if b >= R - 1 and a <= R + 1:
                        right_conn = True

                # Nearest lower support for any column in this horizontal piece.
                limit = H - 1
                for cc in range(a, b + 1):
                    s = find_support(cc, Ttop, lower)
                    lim = H - 1 if s == H else s - 1
                    if lim < limit:
                        limit = lim

                if limit < Ttop:
                    limit = Ttop

                if left_conn and right_conn:
                    # Both walls remain: the top bar sits at the lower of the two tops.
                    fr = max(left_fin["top"], right_fin["top"])
                    if fr > limit:
                        fr = limit
                    if fr < Ttop:
                        fr = Ttop

                elif left_conn:
                    # One wall remains on the left.
                    if a == b and a == L:
                        # Only the corner cell remains.
                        fr = left_fin["top"]
                    else:
                        # A top bar slides down the remaining wall.
                        fr = left_fin["bottom"]
                        if fr > limit:
                            fr = limit
                        if fr < Ttop:
                            fr = Ttop

                elif right_conn:
                    # One wall remains on the right.
                    if a == b and b == R:
                        # Only the corner cell remains.
                        fr = right_fin["top"]
                    else:
                        # A top bar slides down the remaining wall.
                        fr = right_fin["bottom"]
                        if fr > limit:
                            fr = limit
                        if fr < Ttop:
                            fr = Ttop

                else:
                    # Detached horizontal piece: just fall to the nearest support.
                    fr = limit
                    if fr < Ttop:
                        fr = Ttop

                if fr < 0:
                    fr = 0
                if fr > H - 1:
                    fr = H - 1

                for cc in range(a, b + 1):
                    val = orig[Ttop, cc]
                    if val == 0 or val == target or is_bg(Ttop, cc):
                        continue
                    group_grid[fr, cc] = int(val)

        # Add this layer to the settled grid used by layers above.
        mask = group_grid != 0
        lower[mask] = group_grid[mask]

    # -------------------------------------------------------------------------
    # Build final grid and drop any leftover non-U pixels under column gravity.
    # -------------------------------------------------------------------------
    final = lower.copy()
    final[0, 0] = target

    for c in range(W):
        for r in range(H):
            if is_bg(r, c):
                continue

            val = orig[r, c]
            if val == target or covered[r, c]:
                continue

            support = find_support(c, r, final)
            if support == H:
                pos = H - 1
            else:
                pos = support - 1

            if pos < r:
                pos = r

            # Safety: if the computed spot is somehow occupied, move up.
            while pos >= 0 and final[pos, c] != 0:
                pos -= 1

            if pos >= 0:
                final[pos, c] = int(val)

    return final