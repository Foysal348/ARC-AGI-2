import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
      1. Turning every 1 into 2.
      2. Drawing a 3-path from the initial border 3 to the opposite border.

    The path goes straight in the main direction (left-to-right, right-to-left,
    top-to-bottom, or bottom-to-top). Whenever a straight line of 1/2 cells would
    block the path, the path turns early:
        offset = number of 1s in that blocking line + 1
    It turns `offset` cells before the line, then moves to a clear lane
    `offset` cells away from the nearer available end of the line.
    """

    if grid is None or grid.size == 0:
        return grid

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    out = grid.copy()

    # 1 is only a marker; in the output it becomes part of the 2-line.
    out[grid == 1] = 2

    starts = np.argwhere(grid == 3)
    if len(starts) == 0:
        return out

    r0, c0 = int(starts[0][0]), int(starts[0][1])
    out[r0, c0] = 3

    # Extract straight runs of 1/2 cells.
    # Only runs of length >= 2 are considered true blocking lines.
    block_mask = (grid == 1) | (grid == 2)

    h_lines = []  # (row, x1, x2, count_1)
    v_lines = []  # (col, y1, y2, count_1)

    # Horizontal runs.
    for r in range(H):
        c = 0
        while c < W:
            if block_mask[r, c]:
                c2 = c
                while c2 + 1 < W and block_mask[r, c2 + 1]:
                    c2 += 1
                if c2 > c:
                    cnt1 = int(np.sum(grid[r, c:c2 + 1] == 1))
                    h_lines.append((r, c, c2, cnt1))
                c = c2 + 1
            else:
                c += 1

    # Vertical runs.
    for c in range(W):
        r = 0
        while r < H:
            if block_mask[r, c]:
                r2 = r
                while r2 + 1 < H and block_mask[r2 + 1, c]:
                    r2 += 1
                if r2 > r:
                    cnt1 = int(np.sum(grid[r:r2 + 1, c] == 1))
                    v_lines.append((c, r, r2, cnt1))
                r = r2 + 1
            else:
                r += 1

    # Choose the main path direction from the border where the initial 3 sits.
    if H == 1:
        direction = "right" if c0 == 0 else "left"
    elif W == 1:
        direction = "down" if r0 == 0 else "up"
    else:
        if r0 == 0:
            direction = "down"
        elif r0 == H - 1:
            direction = "up"
        elif c0 == 0:
            direction = "right"
        elif c0 == W - 1:
            direction = "left"
        else:
            # Fallback for non-border 3: move toward nearest border.
            dist = {
                "down": H - 1 - r0,
                "up": r0,
                "right": W - 1 - c0,
                "left": c0,
            }
            direction = min(dist, key=lambda d: dist[d])

    def clear_v(r1, r2, c):
        """True if vertical segment (inclusive) is free of 2s."""
        if c < 0 or c >= W:
            return False
        if r1 > r2:
            r1, r2 = r2, r1
        if r1 < 0 or r2 >= H:
            return False
        return not np.any(out[r1:r2 + 1, c] == 2)

    def clear_h(r, c1, c2):
        """True if horizontal segment (inclusive) is free of 2s."""
        if r < 0 or r >= H:
            return False
        if c1 > c2:
            c1, c2 = c2, c1
        if c1 < 0 or c2 >= W:
            return False
        return not np.any(out[r, c1:c2 + 1] == 2)

    def paint_v(r1, r2, c):
        if c < 0 or c >= W:
            return
        if r1 > r2:
            r1, r2 = r2, r1
        r1 = max(0, min(H - 1, r1))
        r2 = max(0, min(H - 1, r2))
        out[r1:r2 + 1, c] = 3

    def paint_h(r, c1, c2):
        if r < 0 or r >= H:
            return
        if c1 > c2:
            c1, c2 = c2, c1
        c1 = max(0, min(W - 1, c1))
        c2 = max(0, min(W - 1, c2))
        out[r, c1:c2 + 1] = 3

    if direction == "down":
        pos, lane = r0, c0
        while pos < H - 1:
            best = None
            for y, x1, x2, cnt1 in h_lines:
                if y > pos and x1 <= lane <= x2:
                    if best is None or y < best[0]:
                        best = (y, x1, x2, cnt1)

            if best is None:
                break

            y, x1, x2, cnt1 = best
            off = cnt1 + 1
            turn_y = y - off

            # Keep turn inside current segment and before the obstacle.
            turn_y = max(turn_y, pos)
            if turn_y >= y:
                turn_y = y - 1
            if turn_y < pos:
                break

            base_ok = clear_v(pos, turn_y, lane)
            candidates = []

            left_lane = x1 - off
            if 0 <= left_lane < W:
                ok = base_ok and clear_h(turn_y, lane, left_lane)
                candidates.append((0 if ok else 1, abs(left_lane - lane), left_lane))

            right_lane = x2 + off
            if 0 <= right_lane < W:
                ok = base_ok and clear_h(turn_y, lane, right_lane)
                candidates.append((0 if ok else 1, abs(right_lane - lane), right_lane))

            if not candidates:
                break

            candidates.sort()
            new_lane = candidates[0][2]

            paint_v(pos, turn_y, lane)
            paint_h(turn_y, lane, new_lane)

            pos, lane = turn_y, new_lane

        paint_v(pos, H - 1, lane)

    elif direction == "up":
        pos, lane = r0, c0
        while pos > 0:
            best = None
            for y, x1, x2, cnt1 in h_lines:
                if y < pos and x1 <= lane <= x2:
                    if best is None or y > best[0]:
                        best = (y, x1, x2, cnt1)

            if best is None:
                break

            y, x1, x2, cnt1 = best
            off = cnt1 + 1
            turn_y = y + off

            turn_y = min(turn_y, pos)
            if turn_y <= y:
                turn_y = y + 1
            if turn_y > pos:
                break

            base_ok = clear_v(pos, turn_y, lane)
            candidates = []

            left_lane = x1 - off
            if 0 <= left_lane < W:
                ok = base_ok and clear_h(turn_y, lane, left_lane)
                candidates.append((0 if ok else 1, abs(left_lane - lane), left_lane))

            right_lane = x2 + off
            if 0 <= right_lane < W:
                ok = base_ok and clear_h(turn_y, lane, right_lane)
                candidates.append((0 if ok else 1, abs(right_lane - lane), right_lane))

            if not candidates:
                break

            candidates.sort()
            new_lane = candidates[0][2]

            paint_v(pos, turn_y, lane)
            paint_h(turn_y, lane, new_lane)

            pos, lane = turn_y, new_lane

        paint_v(pos, 0, lane)

    elif direction == "right":
        pos, lane = c0, r0
        while pos < W - 1:
            best = None
            for x, y1, y2, cnt1 in v_lines:
                if x > pos and y1 <= lane <= y2:
                    if best is None or x < best[0]:
                        best = (x, y1, y2, cnt1)

            if best is None:
                break

            x, y1, y2, cnt1 = best
            off = cnt1 + 1
            turn_x = x - off

            turn_x = max(turn_x, pos)
            if turn_x >= x:
                turn_x = x - 1
            if turn_x < pos:
                break

            base_ok = clear_h(lane, pos, turn_x)
            candidates = []

            top_lane = y1 - off
            if 0 <= top_lane < H:
                ok = base_ok and clear_v(top_lane, lane, turn_x)
                candidates.append((0 if ok else 1, abs(top_lane - lane), top_lane))

            bottom_lane = y2 + off
            if 0 <= bottom_lane < H:
                ok = base_ok and clear_v(bottom_lane, lane, turn_x)
                candidates.append((0 if ok else 1, abs(bottom_lane - lane), bottom_lane))

            if not candidates:
                break

            candidates.sort()
            new_lane = candidates[0][2]

            paint_h(lane, pos, turn_x)
            paint_v(turn_x, lane, new_lane)

            pos, lane = turn_x, new_lane

        paint_h(lane, pos, W - 1)

    else:  # direction == "left"
        pos, lane = c0, r0
        while pos > 0:
            best = None
            for x, y1, y2, cnt1 in v_lines:
                if x < pos and y1 <= lane <= y2:
                    if best is None or x > best[0]:
                        best = (x, y1, y2, cnt1)

            if best is None:
                break

            x, y1, y2, cnt1 = best
            off = cnt1 + 1
            turn_x = x + off

            turn_x = min(turn_x, pos)
            if turn_x <= x:
                turn_x = x + 1
            if turn_x > pos:
                break

            base_ok = clear_h(lane, pos, turn_x)
            candidates = []

            top_lane = y1 - off
            if 0 <= top_lane < H:
                ok = base_ok and clear_v(top_lane, lane, turn_x)
                candidates.append((0 if ok else 1, abs(top_lane - lane), top_lane))

            bottom_lane = y2 + off
            if 0 <= bottom_lane < H:
                ok = base_ok and clear_v(bottom_lane, lane, turn_x)
                candidates.append((0 if ok else 1, abs(bottom_lane - lane), bottom_lane))

            if not candidates:
                break

            candidates.sort()
            new_lane = candidates[0][2]

            paint_h(lane, pos, turn_x)
            paint_v(turn_x, lane, new_lane)

            pos, lane = turn_x, new_lane

        paint_h(lane, 0, pos)

    return out