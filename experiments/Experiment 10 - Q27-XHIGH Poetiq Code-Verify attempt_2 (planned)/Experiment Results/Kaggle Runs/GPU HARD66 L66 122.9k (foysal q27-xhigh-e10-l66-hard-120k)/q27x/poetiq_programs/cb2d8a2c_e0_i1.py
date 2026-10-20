import numpy as np
import heapq


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the output grid.

    Observed rule
    -------------
    1. Color 1 is only an "incomplete" part of the color-2 lines, so all
       1 pixels are first converted to 2.
    2. There is one 3 pixel on a border. It is the start of a new color-3
       path that must reach the opposite border.
    3. The color-2 lines that touch the left/right border are horizontal
       barriers when routing vertically, and the color-2 lines that touch
       the top/bottom border are vertical barriers when routing horizontally.
    4. The path is a rectilinear staircase around those barriers:
       - it keeps a one-cell gap from the barriers when possible,
       - it uses the border lane (column 1 / row 1) on the side that is
         blocked by the barrier,
       - it uses a lane two cells past the free end of the barrier on the
         other side,
       - along-axis turns are placed in the middle of the gap between
         consecutive barriers, with the first turn two cells away from the
         start border when possible.

    The implementation below builds this structured path.  If the structured
    path is invalid for some edge case, a turn-preferring A* fallback is used
    to produce a valid path avoiding the same border-touching barriers.
    """

    def detect_horizontal_barriers(work: np.ndarray):
        """
        Detect horizontal 2-runs that touch the left or right border.

        Returns list of (row, type, extent):
          type 'L': touches left border, extent = right end column
          type 'R': touches right border, extent = left start column
          type 'FULL': touches both borders
        """
        H, W = work.shape
        bars = []
        for r in range(H):
            c = 0
            while c < W:
                if work[r, c] == 2:
                    st = c
                    while c < W and work[r, c] == 2:
                        c += 1
                    en = c - 1
                    if st == 0 and en == W - 1:
                        bars.append((r, "FULL", 0))
                    elif st == 0:
                        bars.append((r, "L", en))
                    elif en == W - 1:
                        bars.append((r, "R", st))
                else:
                    c += 1
        bars.sort(key=lambda x: x[0])
        return bars

    def detect_vertical_barriers(work: np.ndarray):
        """
        Detect vertical 2-runs that touch the top or bottom border.

        Returns list of (col, type, extent):
          type 'T': touches top border, extent = bottom end row
          type 'B': touches bottom border, extent = top start row
          type 'FULL': touches both borders
        """
        H, W = work.shape
        bars = []
        for c in range(W):
            r = 0
            while r < H:
                if work[r, c] == 2:
                    st = r
                    while r < H and work[r, c] == 2:
                        r += 1
                    en = r - 1
                    if st == 0 and en == H - 1:
                        bars.append((c, "FULL", 0))
                    elif st == 0:
                        bars.append((c, "T", en))
                    elif en == H - 1:
                        bars.append((c, "B", st))
                else:
                    r += 1
        bars.sort(key=lambda x: x[0])
        return bars

    def make_mask(work: np.ndarray, direction: str, barriers):
        """Mask only the border-touching barriers that the path must avoid."""
        H, W = work.shape
        mask = np.zeros((H, W), dtype=bool)

        if direction == "down":
            for r, typ, val in barriers:
                if typ == "FULL":
                    mask[r, :] = True
                elif typ == "L":
                    mask[r, 0:val + 1] = True
                elif typ == "R":
                    mask[r, val:W] = True
        else:  # right
            for c, typ, val in barriers:
                if typ == "FULL":
                    mask[:, c] = True
                elif typ == "T":
                    mask[0:val + 1, c] = True
                elif typ == "B":
                    mask[val:H, c] = True
        return mask

    def path_valid(ways, mask, start):
        """Check that a polyline does not cross masked barrier pixels."""
        if not ways:
            return False

        for (r1, c1), (r2, c2) in zip(ways[:-1], ways[1:]):
            if r1 == r2:
                r = r1
                a, b = sorted((c1, c2))
                for c in range(a, b + 1):
                    if mask[r, c] and (r, c) != start:
                        return False
            elif c1 == c2:
                c = c1
                a, b = sorted((r1, r2))
                for r in range(a, b + 1):
                    if mask[r, c] and (r, c) != start:
                        return False
            else:
                # Should not happen for orthogonal waypoints.
                return False
        return True

    def draw_path(out: np.ndarray, ways):
        """Draw color 3 along the waypoints."""
        for (r1, c1), (r2, c2) in zip(ways[:-1], ways[1:]):
            if r1 == r2:
                a, b = sorted((c1, c2))
                out[r1, a:b + 1] = 3
            elif c1 == c2:
                a, b = sorted((r1, r2))
                out[a:b + 1, c1] = 3
            else:
                # Fallback: draw a Bresenham-like straight segment.
                dr = 1 if r2 > r1 else -1
                dc = 1 if c2 > c1 else -1
                r, c = r1, c1
                while (r, c) != (r2, c2):
                    out[r, c] = 3
                    r += dr
                    c += dc
                out[r, c] = 3

    def astar_fallback(work: np.ndarray, mask: np.ndarray, start, direction):
        """
        Fallback solver: shortest path in terms of
        (number of turns, number of steps), avoiding the barrier mask.
        """
        H, W = work.shape
        sr, sc = start
        if direction == "down":
            goal_row = H - 1
            init_dir = 1  # down
        else:
            goal_row = None
            init_dir = 3  # right

        drs = (-1, 1, 0, 0)
        dcs = (0, 0, -1, 1)

        start_state = (sr, sc, init_dir)
        best = {start_state: (0, 0)}
        parent = {}
        heap = [(0, 0, 0, sr, sc, init_dir)]
        counter = 1

        while heap:
            turns, steps, _, r, c, d = heapq.heappop(heap)
            if best[(r, c, d)] != (turns, steps):
                continue

            if direction == "down" and r == goal_row:
                # Reconstruct.
                path = [(r, c)]
                state = (r, c, d)
                while state in parent:
                    state = parent[state]
                    path.append((state[0], state[1]))
                path.reverse()
                return path

            if direction == "right" and c == W - 1:
                path = [(r, c)]
                state = (r, c, d)
                while state in parent:
                    state = parent[state]
                    path.append((state[0], state[1]))
                path.reverse()
                return path

            for nd in range(4):
                nr = r + drs[nd]
                nc = c + dcs[nd]
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if mask[nr, nc] and (nr, nc) != start:
                    continue

                nt = turns + (0 if nd == d else 1)
                ns = steps + 1
                st = (nr, nc, nd)
                if st not in best or (nt, ns) < best[st]:
                    best[st] = (nt, ns)
                    parent[st] = (r, c, d)
                    heapq.heappush(heap, (nt, ns, counter, nr, nc, nd))
                    counter += 1

        return None

    def solve_down(work: np.ndarray, start):
        """Solve the case where the start 3 is on the top border."""
        H, W = work.shape
        sr, sc = start
        out = work.copy()

        if H == 1:
            out[sr, sc] = 3
            return out

        barriers = detect_horizontal_barriers(work)
        mask = make_mask(work, "down", barriers)

        ways = []

        if not any(t == "FULL" for _, t, _ in barriers):
            # Right-side lane: two cells past the right end of every
            # left-touching barrier.  Taking the maximum reproduces the
            # examples where the same lane is reused.
            right_lane = sc
            left_ends = [val for _, typ, val in barriers if typ == "L"]
            if left_ends:
                right_lane = max(e + 2 for e in left_ends)
                if W >= 3:
                    right_lane = max(1, min(W - 2, right_lane))
                else:
                    right_lane = 0

            turn_rows = []
            lanes = []
            prev_row = 0

            for i, (row, typ, val) in enumerate(barriers):
                if i == 0:
                    # First turn: two cells below the start border when
                    # possible, but always keep at least one empty row.
                    m = min(2, max(0, row - 2))
                else:
                    low = prev_row + 2
                    high = row - 2
                    if low <= high:
                        m = (low + high) // 2
                    else:
                        m = (prev_row + row) // 2
                    m = max(0, min(H - 1, m))
                    # Keep it strictly between the two barrier rows if possible.
                    if m < prev_row:
                        m = prev_row
                    if m > row - 1:
                        m = max(prev_row, row - 1)

                if typ == "L":
                    lane = right_lane
                else:
                    # Left-side lane: one cell from the left border,
                    # clamped if the barrier is very close to the border.
                    lane = max(0, min(1, val - 2))

                turn_rows.append(m)
                lanes.append(lane)
                prev_row = row

            ways = [(0, sc)]
            cur_r, cur_c = 0, sc
            for m, lane in zip(turn_rows, lanes):
                if cur_c != lane:
                    ways.append((m, cur_c))
                    ways.append((m, lane))
                else:
                    ways.append((m, cur_c))
                cur_r, cur_c = m, lane

            ways.append((H - 1, cur_c))

        if not path_valid(ways, mask, (sr, sc)):
            fallback = astar_fallback(work, mask, (sr, sc), "down")
            if fallback is not None:
                ways = fallback

        if ways:
            draw_path(out, ways)
        return out

    def solve_right(work: np.ndarray, start):
        """Solve the case where the start 3 is on the left border."""
        H, W = work.shape
        sr, sc = start
        out = work.copy()

        if W == 1:
            out[sr, sc] = 3
            return out

        barriers = detect_vertical_barriers(work)
        mask = make_mask(work, "right", barriers)

        ways = []

        if not any(t == "FULL" for _, t, _ in barriers):
            # Bottom-side lane: two rows below the bottom end of every
            # top-touching barrier.  If the start is very close to the
            # barrier bottom, add one extra row, matching the examples.
            bottom_lane = sr
            for _, typ, val in barriers:
                if typ == "T":
                    b = val
                    if sr <= b:
                        extra = max(0, 2 - (b - sr))
                    else:
                        extra = 0
                    bottom_lane = max(bottom_lane, b + 2 + extra)

            if H >= 3:
                bottom_lane = max(1, min(H - 2, bottom_lane))
            else:
                bottom_lane = 0

            turn_cols = []
            lanes = []
            prev_col = 0
            target_col = W - 1

            for i, (col, typ, val) in enumerate(barriers):
                if i == 0:
                    if typ == "T":
                        clearance = max(2, max(0, sr - 1))
                    else:
                        clearance = max(2, max(0, (H - 1 - sr) - 1))
                    x = col - clearance
                    x = max(0, min(col - 2 if col > 2 else 0, x))
                else:
                    low = prev_col + 2
                    high = col - 2
                    if low <= high:
                        # For bottom-touching barriers and for the final
                        # turn, balance the remaining horizontal run to the
                        # target border, then clamp by the next barrier.
                        if i == len(barriers) - 1 or typ == "B":
                            pref = min((prev_col + target_col + 1) // 2, high)
                        else:
                            pref = (low + high) // 2
                    else:
                        pref = (prev_col + col) // 2
                    x = max(max(0, low), min(W - 1, high, pref))
                    x = max(0, min(W - 1, x))

                if typ == "T":
                    lane = bottom_lane
                else:
                    # Use the start row when it is safely above this
                    # bottom-touching barrier; otherwise use the top-border
                    # lane with a one-cell margin.
                    if sr <= val - 2:
                        lane = sr
                    else:
                        lane = max(0, min(1, val - 2))

                turn_cols.append(x)
                lanes.append(lane)
                prev_col = col

            ways = [(sr, 0)]
            cur_r, cur_c = sr, 0
            for x, lane in zip(turn_cols, lanes):
                if cur_c != x:
                    ways.append((cur_r, x))
                if cur_r != lane:
                    ways.append((lane, x))
                cur_r, cur_c = lane, x

            ways.append((cur_r, target_col))

        if not path_valid(ways, mask, (sr, sc)):
            fallback = astar_fallback(work, mask, (sr, sc), "right")
            if fallback is not None:
                ways = fallback

        if ways:
            draw_path(out, ways)
        return out

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    # First fill all 1s to 2.
    filled = g.copy()
    filled[filled == 1] = 2

    starts = np.argwhere(g == 3)
    if len(starts) == 0:
        return filled

    H, W = g.shape
    sr, sc = starts[0]

    work = filled.copy()
    flips = []

    # Choose a canonical direction: top-to-bottom or left-to-right.
    if sr == 0:
        direction = "down"
    elif sr == H - 1:
        work = work[::-1, :].copy()
        flips.append("v")
        direction = "down"
    elif sc == 0:
        direction = "right"
    elif sc == W - 1:
        work = work[:, ::-1].copy()
        flips.append("h")
        direction = "right"
    else:
        # The task definition has the 3 on a border.  If not, just return
        # the filled grid.
        return filled

    # Re-locate the start in the possibly flipped working grid.
    work_starts = np.argwhere(work == 3)
    if len(work_starts) == 0:
        return filled
    ws_r, ws_c = work_starts[0]

    if direction == "down":
        result = solve_down(work, (ws_r, ws_c))
    else:
        result = solve_right(work, (ws_r, ws_c))

    # Flip back to original orientation.
    if "v" in flips:
        result = result[::-1, :]
    if "h" in flips:
        result = result[:, ::-1]

    return result