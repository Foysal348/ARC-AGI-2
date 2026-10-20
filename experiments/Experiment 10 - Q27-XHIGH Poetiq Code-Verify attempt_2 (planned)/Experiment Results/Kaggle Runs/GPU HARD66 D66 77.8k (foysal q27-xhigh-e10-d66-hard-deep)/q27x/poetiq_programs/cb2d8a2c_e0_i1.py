import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
      1. Replacing all 1s with 2s (fill the "cracks" in the walls).
      2. Drawing a 3-path from the single 3 marker to the opposite border.

    The path is a rectilinear spiral that goes around each straight wall.
    For a wall containing k pixels of color 1, let d = k + 1.
    When passing a wall:
      - turn d cells before the wall in the travel direction;
      - then move perpendicular to a line d cells beyond the wall's open end.
    The side (left/right or top/bottom) is forced by which grid border the wall touches.
    """
    arr = np.array(grid, dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        return arr

    H, W = arr.shape
    out = arr.copy()
    out[out == 1] = 2

    def clamp(v, lo, hi):
        return max(lo, min(hi, int(v)))

    def horizontal_runs():
        runs = []
        for r in range(H):
            c = 0
            while c < W:
                if arr[r, c] in (1, 2):
                    L = c
                    while c < W and arr[r, c] in (1, 2):
                        c += 1
                    R = c - 1
                    if R - L + 1 >= 2:
                        ones = int(np.count_nonzero(arr[r, L:c] == 1))
                        runs.append({'row': r, 'L': L, 'R': R, 'ones': ones})
                else:
                    c += 1
        return runs

    def vertical_runs():
        runs = []
        for c in range(W):
            r = 0
            while r < H:
                if arr[r, c] in (1, 2):
                    T = r
                    while r < H and arr[r, c] in (1, 2):
                        r += 1
                    B = r - 1
                    if B - T + 1 >= 2:
                        ones = int(np.count_nonzero(arr[T:r, c] == 1))
                        runs.append({'col': c, 'T': T, 'B': B, 'ones': ones})
                else:
                    r += 1
        return runs

    h_runs = horizontal_runs()
    v_runs = vertical_runs()

    ys, xs = np.where(arr == 3)
    if len(ys) == 0:
        return out

    sr, sc = int(ys[0]), int(xs[0])

    # Choose the travel orientation from the border where the 3 starts.
    can_vertical = (sr == 0 or sr == H - 1) and len(h_runs) > 0
    can_horizontal = (sc == 0 or sc == W - 1) and len(v_runs) > 0

    if can_vertical and can_horizontal:
        orientation = 'horizontal' if len(v_runs) > len(h_runs) else 'vertical'
    elif can_vertical:
        orientation = 'vertical'
    elif can_horizontal:
        orientation = 'horizontal'
    else:
        if sr in (0, H - 1):
            orientation = 'vertical'
        elif sc in (0, W - 1):
            orientation = 'horizontal'
        else:
            return out

    def draw_line(r1, c1, r2, c2):
        if r1 == r2:
            cmin, cmax = sorted((c1, c2))
            if 0 <= r1 < H and 0 <= cmin < W and 0 <= cmax < W:
                for c in range(cmin, cmax + 1):
                    if out[r1, c] != 2:
                        out[r1, c] = 3
        elif c1 == c2:
            rmin, rmax = sorted((r1, r2))
            if 0 <= c1 < W and 0 <= rmin < H and 0 <= rmax < H:
                for r in range(rmin, rmax + 1):
                    if out[r, c1] != 2:
                        out[r, c1] = 3

    if orientation == 'vertical':
        dirn = 1 if sr == 0 else -1
        final_r = H - 1 if dirn == 1 else 0

        if dirn == 1:
            walls = [w for w in h_runs if w['row'] > sr]
            walls.sort(key=lambda w: w['row'])
        else:
            walls = [w for w in h_runs if w['row'] < sr]
            walls.sort(key=lambda w: w['row'], reverse=True)

        def side_col(wall, current_c):
            d = wall['ones'] + 1
            L, R = wall['L'], wall['R']

            forced = []
            if L == 0:
                forced.append(R + d)
            if R == W - 1:
                forced.append(L - d)

            if forced:
                valid = [(abs(clamp(c, 0, W - 1) - current_c), clamp(c, 0, W - 1)) for c in forced]
                return min(valid)[1]

            right = R + d
            left = L - d
            valid = []
            if 0 <= right < W:
                valid.append((abs(right - current_c), right))
            if 0 <= left < W:
                valid.append((abs(left - current_c), left))
            if valid:
                return min(valid)[1]

            if current_c < L:
                val = left
            elif current_c > R:
                val = right
            else:
                val = right if R - current_c <= current_c - L else left
            return clamp(val, 0, W - 1)

        points = [(sr, sc)]
        cur_r, cur_c = sr, sc

        for wall in walls:
            d = wall['ones'] + 1
            turn_r = wall['row'] - d if dirn == 1 else wall['row'] + d
            turn_r = clamp(turn_r, 0, H - 1)
            sc_side = side_col(wall, cur_c)

            points.append((turn_r, cur_c))
            points.append((turn_r, sc_side))

            cur_r, cur_c = turn_r, sc_side

        points.append((final_r, cur_c))

    else:
        dirn = 1 if sc == 0 else -1
        final_c = W - 1 if dirn == 1 else 0

        if dirn == 1:
            walls = [w for w in v_runs if w['col'] > sc]
            walls.sort(key=lambda w: w['col'])
        else:
            walls = [w for w in v_runs if w['col'] < sc]
            walls.sort(key=lambda w: w['col'], reverse=True)

        def side_row(wall, current_r):
            d = wall['ones'] + 1
            T, B = wall['T'], wall['B']

            forced = []
            if T == 0:
                forced.append(B + d)
            if B == H - 1:
                forced.append(T - d)

            if forced:
                valid = [(abs(clamp(r, 0, H - 1) - current_r), clamp(r, 0, H - 1)) for r in forced]
                return min(valid)[1]

            below = B + d
            above = T - d
            valid = []
            if 0 <= below < H:
                valid.append((abs(below - current_r), below))
            if 0 <= above < H:
                valid.append((abs(above - current_r), above))
            if valid:
                return min(valid)[1]

            if current_r < T:
                val = above
            elif current_r > B:
                val = below
            else:
                val = below if B - current_r <= current_r - T else above
            return clamp(val, 0, H - 1)

        points = [(sr, sc)]
        cur_r, cur_c = sr, sc

        for wall in walls:
            d = wall['ones'] + 1
            turn_c = wall['col'] - d if dirn == 1 else wall['col'] + d
            turn_c = clamp(turn_c, 0, W - 1)
            sr_side = side_row(wall, cur_r)

            points.append((cur_r, turn_c))
            points.append((sr_side, turn_c))

            cur_r, cur_c = sr_side, turn_c

        points.append((cur_r, final_c))

    for i in range(len(points) - 1):
        draw_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

    return out