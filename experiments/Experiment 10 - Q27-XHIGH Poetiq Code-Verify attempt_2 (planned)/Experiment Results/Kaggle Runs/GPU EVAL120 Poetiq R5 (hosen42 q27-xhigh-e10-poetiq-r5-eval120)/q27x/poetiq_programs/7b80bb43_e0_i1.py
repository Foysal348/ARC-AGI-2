import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the dominant incomplete L/T-shaped stroke.

    The rule implemented here is:
      1. Treat the most frequent color as background and the other color
         as the foreground/stroke color.
      2. Find the best intersection point of a horizontal and vertical
         stroke.  The best point is chosen by scoring how far the four
         arms can be completed.
      3. An arm may bridge short gaps (at most 3 cells).  If an arm already
         starts with a long continuous run, it is not extended beyond that
         run.  If it is short, it may be extended to the next nearby run.
      4. Draw the selected horizontal and vertical arms.
      5. Remove nearby stray foreground pixels that are not part of the
         completed arms and are not part of a genuine perpendicular line
         attaching to the completed arms.
      6. In a direction where no arm is selected at all, clear the nearest
         collinear stray run.
    """

    def in_bounds(r, c, h, w):
        return 0 <= r < h and 0 <= c < w

    def run_length(r, c, dr, dc, mask):
        """Length of the contiguous run starting at (r,c) in direction (dr,dc)."""
        length = 0
        while in_bounds(r, c, mask.shape[0], mask.shape[1]) and mask[r, c]:
            length += 1
            r += dr
            c += dc
        return length

    def has_perpendicular_support(r1, c1, r2, c2, horizontal, mask):
        """
        Return True if a horizontal/vertical run has a perpendicular
        continuation of length at least 2.  Such runs are usually part of
        another stroke and should not be swallowed as a stray short piece.
        """
        if horizontal:
            r = r1
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if run_length(r, c + 1, 1, 0, mask) >= 2:
                    return True
                if run_length(r, c - 1, -1, 0, mask) >= 2:
                    return True
        else:
            c = c1
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if run_length(r, c + 1, 0, 1, mask) >= 2:
                    return True
                if run_length(r, c - 1, 0, -1, mask) >= 2:
                    return True
        return False

    def arm_span(r, c, dr, dc, mask):
        """
        Compute the final interval in one direction from (r,c).

        Returns:
            (start_r, start_c, end_r, end_c, length, had_long_start)
            or (None, None, None, None, 0, False)
        """
        h, w = mask.shape
        pos_r, pos_c = r + dr, c + dc

        span_start = None
        span_end = None
        included_runs = 0

        max_gap = 3
        stop_len = 5
        max_runs = 4

        while True:
            gap = 0
            while in_bounds(pos_r, pos_c, h, w) and not mask[pos_r, pos_c]:
                gap += 1
                pos_r += dr
                pos_c += dc

            if not in_bounds(pos_r, pos_c, h, w):
                break

            if gap > max_gap:
                break

            run_start_r, run_start_c = pos_r, pos_c
            run_len = 0
            while in_bounds(pos_r, pos_c, h, w) and mask[pos_r, pos_c]:
                run_len += 1
                pos_r += dr
                pos_c += dc

            if run_len == 0:
                break

            run_end_r, run_end_c = pos_r - dr, pos_c - dc

            horizontal = (dc != 0)
            support = has_perpendicular_support(
                run_start_r, run_start_c, run_end_r, run_end_c, horizontal, mask
            )

            include = True
            if included_runs == 0:
                # First run in this direction.  If it is reached only across
                # a gap and is a short supported run, do not connect to it.
                if gap > 0 and support and run_len < stop_len:
                    include = False
            else:
                # Subsequent run.  Do not connect through a short supported
                # run, because that usually belongs to another stroke.
                if support and run_len < stop_len:
                    include = False

            if not include:
                break

            if span_start is None:
                span_start = (run_start_r, run_start_c)
            span_end = (run_end_r, run_end_c)

            had_long_start = (included_runs == 0 and run_len >= stop_len)
            included_runs += 1

            if run_len >= stop_len or included_runs >= max_runs:
                break

        if span_start is None or span_end is None:
            return None, None, None, None, 0, False

        if dr == 1:
            length = span_end[0] - r
        elif dr == -1:
            length = r - span_end[0]
        elif dc == 1:
            length = span_end[1] - c
        else:
            length = c - span_end[1]

        # Re-evaluate had_long_start for the first included run.
        # This is simply whether the first run from the center was long.
        first_len = run_length(r + dr, c + dc, dr, dc, mask)
        had_long_start = first_len >= stop_len

        return span_start[0], span_start[1], span_end[0], span_end[1], length, had_long_start

    def set_arm(out, r, c, dr, dc, span):
        """Draw one arm into out."""
        if span is None:
            return
        sr, sc, er, ec, _, _ = span
        if dr == 0 and dc == 1:
            out[r, min(sc, ec):max(sc, ec) + 1] = fg
        elif dr == 0 and dc == -1:
            out[r, min(sc, ec):max(sc, ec) + 1] = fg
        elif dr == 1 and dc == 0:
            out[min(sr, er):max(sr, er) + 1, c] = fg
        elif dr == -1 and dc == 0:
            out[min(sr, er):max(sr, er) + 1, c] = fg

    def clear_nearest_run_in_empty_direction(out, r, c, dr, dc, mask):
        """
        If no arm was selected in this direction, remove the nearest
        collinear foreground run, if any.
        """
        h, w = mask.shape
        pos_r, pos_c = r + dr, c + dc

        # skip background
        while in_bounds(pos_r, pos_c, h, w) and not mask[pos_r, pos_c]:
            pos_r += dr
            pos_c += dc

        if not in_bounds(pos_r, pos_c, h, w):
            return

        # clear the whole contiguous run
        while in_bounds(pos_r, pos_c, h, w) and mask[pos_r, pos_c]:
            out[pos_r, pos_c] = bg
            pos_r += dr
            pos_c += dc

    def is_keepable_attached_line(r, c, mask, selected):
        """
        Keep a foreground pixel if it belongs to a genuine horizontal or
        vertical line of length at least 2 that touches or crosses a
        selected line.  This preserves unrelated real strokes that intersect
        the main L/T shape.
        """
        h, w = mask.shape

        # vertical run containing (r,c)
        top = r
        while top > 0 and mask[top - 1, c]:
            top -= 1
        bottom = r
        while bottom < h - 1 and mask[bottom + 1, c]:
            bottom += 1
        if bottom - top + 1 >= 2:
            for sel_r in range(max(0, top - 1), min(h, bottom + 2)):
                if selected[sel_r, c]:
                    return True

        # horizontal run containing (r,c)
        left = c
        while left > 0 and mask[r, left - 1]:
            left -= 1
        right = c
        while right < w - 1 and mask[r, right + 1]:
            right += 1
        if right - left + 1 >= 2:
            for sel_c in range(max(0, left - 1), min(w, right + 2)):
                if selected[r, sel_c]:
                    return True

        return False

    # If empty or single-color, return unchanged.
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    if len(values) <= 1:
        return grid.copy()

    bg = int(values[np.argmax(counts)])
    fg = int(values[np.argmin(counts)])
    mask = (grid == fg)

    if not mask.any():
        return grid.copy()

    # Find the best intersection point.
    best = None
    best_score = -1
    best_tie = (10**9, 10**9)

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue

            spans = []
            score = 0
            valid = False

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                span = arm_span(r, c, dr, dc, mask)
                spans.append(span)
                if span[4] > 0:
                    score += span[4]
                if span[5]:
                    valid = True

            if not valid:
                continue

            # Slight bonus for having multiple arms.
            score += sum(1 for s in spans if s[4] > 0)

            tie = (r, c)
            if (score, -sum(1 for s in spans if s[4] > 0), tie) > (
                best_score,
                -best.get("arms", 0) if best else 0,
                best_tie
            ):
                best_score = score
                best = {"r": r, "c": c, "spans": spans, "arms": sum(1 for s in spans if s[4] > 0)}
                best_tie = tie

    if best is None:
        return grid.copy()

    br, bc = best["r"], best["c"]
    spans = best["spans"]

    out = grid.copy()
    selected = np.zeros((h, w), dtype=bool)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Draw selected arms.
    for (dr, dc), span in zip(directions, spans):
        set_arm(out, br, bc, dr, dc, span)
        if span[4] > 0:
            sr, sc, er, ec = span[0], span[1], span[2], span[3]
            if dr == 0 and dc == 1:
                selected[br, min(sc, ec):max(sc, ec) + 1] = True
            elif dr == 0 and dc == -1:
                selected[br, min(sc, ec):max(sc, ec) + 1] = True
            elif dr == 1 and dc == 0:
                selected[min(sr, er):max(sr, er) + 1, bc] = True
            elif dr == -1 and dc == 0:
                selected[min(sr, er):max(sr, er) + 1, bc] = True

    # Center is selected.
    selected[br, bc] = True

    # Clear nearest stray collinear run in directions with no selected arm.
    for (dr, dc), span in zip(directions, spans):
        if span[4] == 0:
            clear_nearest_run_in_empty_direction(out, br, bc, dr, dc, mask)

    # Remove nearby stray pixels that are not on selected arms and are not
    # part of an attached genuine line.
    clean_radius = 3
    selected_cells = np.argwhere(selected)

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if selected[r, c]:
                continue

            # Is it close to any selected cell?
            close = False
            for sr, sc in selected_cells:
                if abs(sr - r) <= clean_radius and abs(sc - c) <= clean_radius:
                    close = True
                    break

            if not close:
                continue

            if is_keepable_attached_line(r, c, mask, selected):
                continue

            out[r, c] = bg

    return out