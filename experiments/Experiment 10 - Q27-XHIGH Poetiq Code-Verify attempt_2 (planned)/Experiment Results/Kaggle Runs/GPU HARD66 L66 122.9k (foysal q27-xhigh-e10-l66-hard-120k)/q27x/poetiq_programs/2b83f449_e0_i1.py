import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform encoded horizontal/vertical wire diagrams.

    - Each horizontal run of 7s is treated as a vertical wire of length 3
      centered at the run's middle.
    - That vertical wire is drawn with color 6, and the original run becomes 8 6 8.
    - A 6 is "active" if it is the lower cell of such a vertical wire
      (the 7-run is immediately above it).
    - Terminals 3 are added at horizontal ends fed by active 6s:
        * at borders: one 3 for long arms, two 3s for short arms (length <= 4);
        * at zero gaps: one 3 on the side whose arm to an active 6 is longer.
    - Original 3s that are not selected as terminals become 8.
    """
    orig = np.array(grid, dtype=int)
    if orig.ndim != 2:
        return orig

    H, W = orig.shape
    if H == 0 or W == 0:
        return orig.copy()

    # Start from the input, but original terminals are provisional wires.
    out = orig.copy()
    out[out == 3] = 8

    # active[r, c] is True if out[r, c] will be an active 6:
    # it is the bottom cell of a vertical line created from a 7-run at row r-1.
    active = np.zeros((H, W), dtype=bool)

    # Find horizontal runs of 7s.
    runs = []
    for r in range(H):
        c = 0
        while c < W:
            if orig[r, c] == 7:
                s = c
                while c < W and orig[r, c] == 7:
                    c += 1
                runs.append((r, s, c - 1))
            else:
                c += 1

    # Draw each encoded vertical wire.
    for r, s, e in runs:
        # The 7-run itself becomes horizontal 8s, with the center overwritten by 6.
        out[r, s:e + 1] = 8

        length = e - s + 1
        center = s + length // 2

        # Vertical 3-cell line: row-1, row, row+1.
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < H:
                out[rr, center] = 6
                # Only the bottom cell is considered active for terminal generation.
                if rr == r + 1:
                    active[rr, center] = True

    # Rows containing 7s are not horizontal terminal rows.
    has7_row = np.any(orig == 7, axis=1)

    def mark_border_terminal(r: int, side: int, distance: int) -> None:
        """
        Mark a terminal at side 0 (left) or side W-1 (right) of row r.

        Short arms get two terminal cells, longer arms get one.
        """
        term_len = 2 if distance <= 4 else 1
        step = 1 if side == 0 else -1
        c = side
        marked = 0

        while marked < term_len and 0 <= c < W:
            v = out[r, c]
            if v == 0 or v == 6:
                break
            out[r, c] = 3
            marked += 1
            c += step

    sides = [0] if W <= 1 else [0, W - 1]

    for r in range(H):
        if has7_row[r]:
            continue

        # -------------------------
        # Border terminals
        # -------------------------
        for side in sides:
            step = 1 if side == 0 else -1

            # Length of the contiguous non-zero, non-6 arm from this border.
            c = side
            arm_len = 0
            while 0 <= c < W:
                v = out[r, c]
                if v == 0 or v == 6:
                    break
                arm_len += 1
                c += step

            # If the first blocking cell is an active 6, this border is fed.
            if 0 <= c < W and out[r, c] == 6 and active[r, c]:
                mark_border_terminal(r, side, arm_len)
                continue

            # Exception: an original border terminal may still be kept if the
            # first 6 in the row, ignoring zero gaps, is active.
            if orig[r, side] == 3:
                c = side
                first6 = None
                while 0 <= c < W:
                    if out[r, c] == 6:
                        first6 = c
                        break
                    c += step

                if first6 is not None and active[r, first6]:
                    mark_border_terminal(r, side, abs(first6 - side))

        # -------------------------
        # Zero-gap terminals
        # -------------------------
        for h in range(1, W - 1):
            if out[r, h] != 0:
                continue

            # Left arm from the gap until a 0 or 6.
            left_len = 0
            c = h - 1
            while c >= 0:
                v = out[r, c]
                if v == 0 or v == 6:
                    break
                left_len += 1
                c -= 1
            left_ok = left_len > 0 and c >= 0 and out[r, c] == 6 and active[r, c]

            # Right arm from the gap until a 0 or 6.
            right_len = 0
            c = h + 1
            while c < W:
                v = out[r, c]
                if v == 0 or v == 6:
                    break
                right_len += 1
                c += 1
            right_ok = right_len > 0 and c < W and out[r, c] == 6 and active[r, c]

            if left_ok and right_ok:
                if left_len > right_len:
                    out[r, h - 1] = 3
                elif right_len > left_len:
                    out[r, h + 1] = 3
                else:
                    out[r, h - 1] = 3
                    out[r, h + 1] = 3
            elif left_ok:
                out[r, h - 1] = 3
            elif right_ok:
                out[r, h + 1] = 3

    return out