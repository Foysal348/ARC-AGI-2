import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by filling 1s with 2s and drawing a 3-path from the
    initial border 3 to the opposite border, weaving around 1/2 segments.

    For each straight 1/2 segment:
      offset = count of 1s in that segment + 1
    The path turns 'offset' cells before the segment and moves to a lane
    'offset' cells away from the segment endpoint on the side that is
    available inside the grid.
    """
    if grid is None or grid.size == 0:
        return grid

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    out = grid.copy()

    # Fill all 1s to 2s.
    out[grid == 1] = 2

    # Find the starting 3.
    starts = np.argwhere(grid == 3)
    if len(starts) == 0:
        return out
    r0, c0 = int(starts[0][0]), int(starts[0][1])

    # Extract straight 1/2 runs.
    mask = (grid == 1) | (grid == 2)

    h_lines = []  # (y, x1, x2, count_1)
    v_lines = []  # (x, y1, y2, count_1)

    # Horizontal runs of length >= 2
    for r in range(H):
        c = 0
        while c < W:
            if mask[r, c]:
                c2 = c
                while c2 + 1 < W and mask[r, c2 + 1]:
                    c2 += 1
                if c2 > c:
                    cnt1 = int(np.sum(grid[r, c:c2 + 1] == 1))
                    h_lines.append((r, c, c2, cnt1))
                c = c2 + 1
            else:
                c += 1

    # Vertical runs of length >= 2
    for c in range(W):
        r = 0
        while r < H:
            if mask[r, c]:
                r2 = r
                while r2 + 1 < H and mask[r2 + 1, c]:
                    r2 += 1
                if r2 > r:
                    cnt1 = int(np.sum(grid[r:r2 + 1, c] == 1))
                    v_lines.append((c, r, r2, cnt1))
                r = r2 + 1
            else:
                r += 1

    # Isolated 1/2 cells can block in either direction.
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                has_neigh = (
                    (r > 0 and mask[r - 1, c]) or
                    (r + 1 < H and mask[r + 1, c]) or
                    (c > 0 and mask[r, c - 1]) or
                    (c + 1 < W and mask[r, c + 1])
                )
                if not has_neigh:
                    cnt1 = 1 if grid[r, c] == 1 else 0
                    h_lines.append((r, c, c, cnt1))
                    v_lines.append((c, r, r, cnt1))

    def clear_v(r1, r2, c):
        """Check if vertical segment has no 2 cells."""
        if c < 0 or c >= W:
            return False
        if r1 == r2:
            return out[r1, c] != 2
        if r1 < r2:
            return not np.any(out[r1:r2 + 1, c] == 2)
        return not np.any(out[r2:r1 + 1, c] == 2)

    def clear_h(r, c1, c2):
        """Check if horizontal segment has no 2 cells."""
        if r < 0 or r >= H:
            return False
        if c1 == c2:
            return out[r, c1] != 2
        if c1 < c2:
            return not np.any(out[r, c1:c2 + 1] == 2)
        return not np.any(out[r, c2:c1 + 1] == 2)

    def paint_v(r1, r2, c):
        """Paint vertical segment with 3."""
        if c < 0 or c >= W:
            return
        if r1 < r2:
            out[r1:r2 + 1, c] = 3
        else:
            out[r2:r1 + 1, c] = 3

    def paint_h(r, c1, c2):
        """Paint horizontal segment with 3."""
        if r < 0 or r >= H:
            return
        if c1 < c2:
            out[r, c1:c2 + 1] = 3
        else:
            out[r, c2:c1 + 1] = 3

    # Choose main direction from the border where 3 starts.
    if H == 1:
        if c0 == 0:
            direction = "right"
        elif c0 == W - 1:
            direction = "left"
        else:
            direction = "right"
    elif W == 1:
        if r0 == 0:
            direction = "down"
        elif r0 == H - 1:
            direction = "up"
        else:
            direction = "down"
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
            # Fallback if 3 is not on a border: move toward nearest border.
            dist = {
                "down": H - 1 - r0,
                "up": r0,
                "right": W - 1 - c0,
                "left": c0,
            }
            direction = min(dist, key=lambda d: dist[d])

    out[r0, c0] = 3

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

            if turn_y < pos:
                turn_y = pos
            if turn_y >= y:
                turn_y = y - 1
            if turn_y < pos:
                break

            primary_ok = clear_v(pos, turn_y, lane)
            cands = []

            nx = x1 - off
            if 0 <= nx < W:
                ok = primary_ok and clear_h(turn_y, lane, nx)
                cands.append((0 if ok else 1, abs(nx - lane), nx))

            nx = x2 + off
            if 0 <= nx < W:
                ok = primary_ok and clear_h(turn_y, lane, nx)
                cands.append((0 if ok else 1, abs(nx - lane), nx))

            if not cands:
                break

            cands.sort()
            new_lane = cands[0][2]

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

            if turn_y > pos:
                turn_y = pos
            if turn_y <= y:
                turn_y = y + 1
            if turn_y > pos:
                break

            primary_ok = clear_v(pos, turn_y, lane)
            cands = []

            nx = x1 - off
            if 0 <= nx < W:
                ok = primary_ok and clear_h(turn_y, lane, nx)
                cands.append((0 if ok else 1, abs(nx - lane), nx))

            nx = x2 + off
            if 0 <= nx < W:
                ok = primary_ok and clear_h(turn_y, lane, nx)
                cands.append((0 if ok else 1, abs(nx - lane), nx))

            if not cands:
                break

            cands.sort()
            new_lane = cands[0][2]

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

            if turn_x < pos:
                turn_x = pos
            if turn_x >= x:
                turn_x = x - 1
            if turn_x < pos:
                break

            primary_ok = clear_h(lane, pos, turn_x)
            cands = []

            ny = y1 - off
            if 0 <= ny < H:
                ok = primary_ok and clear_v(ny, lane, turn_x)
                cands.append((0 if ok else 1, abs(ny - lane), ny))

            ny = y2 + off
            if 0 <= ny < H:
                ok = primary_ok and clear_v(ny, lane, turn_x)
                cands.append((0 if ok else 1, abs(ny - lane), ny))

            if not cands:
                break

            cands.sort()
            new_lane = cands[0][2]

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

            if turn_x > pos:
                turn_x = pos
            if turn_x <= x:
                turn_x = x + 1
            if turn_x > pos:
                break

            primary_ok = clear_h(lane, pos, turn_x)
            cands = []

            ny = y1 - off
            if 0 <= ny < H:
                ok = primary_ok and clear_v(ny, lane, turn_x)
                cands.append((0 if ok else 1, abs(ny - lane), ny))

            ny = y2 + off
            if 0 <= ny < H:
                ok = primary_ok and clear_v(ny, lane, turn_x)
                cands.append((0 if ok else 1, abs(ny - lane), ny))

            if not cands:
                break

            cands.sort()
            new_lane = cands[0][2]

            paint_h(lane, pos, turn_x)
            paint_v(turn_x, lane, new_lane)

            pos, lane = turn_x, new_lane

        paint_h(lane, 0, pos)

    return out