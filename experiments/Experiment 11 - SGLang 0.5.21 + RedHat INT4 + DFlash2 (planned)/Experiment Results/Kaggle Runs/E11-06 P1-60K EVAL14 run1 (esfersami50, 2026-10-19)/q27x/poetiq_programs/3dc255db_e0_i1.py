import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    H, W = grid.shape
    out = grid.copy()

    # ---- 8-connected components per color ----
    colors = sorted(set(int(x) for x in grid.flatten()) - {0})
    comps = []
    for c in colors:
        remaining = {(r, cc) for r in range(H) for cc in range(W) if grid[r, cc] == c}
        while remaining:
            start = next(iter(remaining))
            stack = [start]
            comp = []
            remaining.discard(start)
            while stack:
                r, cc = stack.pop()
                comp.append((r, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r + dr, cc + dc)
                        if nb in remaining:
                            remaining.discard(nb)
                            stack.append(nb)
            comps.append([c, comp])

    def bbox(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        return min(rs), max(rs), min(cs), max(cs)

    def run_count(vals):
        vals = sorted(set(vals))
        if not vals:
            return 0
        cnt = 1
        for a, b in zip(vals, vals[1:]):
            if b != a + 1:
                cnt += 1
        return cnt

    def mid(a, b):
        L = b - a + 1
        m = max(1, L - 2)
        st = a + (L - m) // 2
        return st, st + m - 1

    # ---- assign dot components to the container whose bbox contains them ----
    assigned = [False] * len(comps)
    container_dots = {}
    for i in sorted(range(len(comps)), key=lambda i: -len(comps[i][1])):
        bi = bbox(comps[i][1])
        for j in range(len(comps)):
            if j == i or assigned[j]:
                continue
            bj = bbox(comps[j][1])
            if len(comps[i][1]) > len(comps[j][1]) and \
               bi[0] <= bj[0] and bj[1] <= bi[1] and bi[2] <= bj[2] and bj[3] <= bi[3]:
                assigned[j] = True
                container_dots.setdefault(i, []).extend(comps[j][1])

    DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # top, bottom, left, right
    OPP = {0: 1, 1: 0, 2: 3, 3: 2}

    for i in sorted(container_dots):
        color, comp = comps[i]
        dots = container_dots[i]
        dot_color = int(grid[dots[0][0], dots[0][1]])

        # remove dots from interior
        for (r, c) in dots:
            out[r, c] = 0

        r0, r1, c0, c1 = bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        mc0, mc1 = mid(c0, c1)
        mr0, mr1 = mid(r0, r1)

        seg_top = run_count([c for (r, c) in comp if r == r0 and mc0 <= c <= mc1])
        seg_bot = run_count([c for (r, c) in comp if r == r1 and mc0 <= c <= mc1])
        seg_left = run_count([r for (r, c) in comp if c == c0 and mr0 <= r <= mr1])
        seg_right = run_count([r for (r, c) in comp if c == c1 and mr0 <= r <= mr1])
        segs = [seg_top, seg_bot, seg_left, seg_right]

        zeros = [k for k in range(4) if segs[k] == 0]
        nonz = [k for k in range(4) if segs[k] > 0]

        if len(zeros) == 1:
            side = OPP[zeros[0]]
        elif len(nonz) == 1 and len(zeros) == 3:
            if len(comp) / area <= 0.5:
                side = nonz[0]
            else:
                cr = sum(r for r, _ in comp) / len(comp)
                cc = sum(c for _, c in comp) / len(comp)
                dr = sum(r for r, _ in dots) / len(dots) - cr
                dc = sum(c for _, c in dots) / len(dots) - cc
                if abs(dc) >= abs(dr):
                    side = 3 if dc < 0 else 2
                else:
                    side = 1 if dr < 0 else 0
        else:
            cr = sum(r for r, _ in comp) / len(comp)
            cc = sum(c for _, c in comp) / len(comp)
            dr = sum(r for r, _ in dots) / len(dots) - cr
            dc = sum(c for _, c in dots) / len(dots) - cc
            if abs(dc) >= abs(dr):
                side = 3 if dc < 0 else 2
            else:
                side = 1 if dr < 0 else 0

        # anchor: extreme container cell on the chosen side
        if side == 0:
            cands = [p for p in comp if p[0] == r0]
        elif side == 1:
            cands = [p for p in comp if p[0] == r1]
        elif side == 2:
            cands = [p for p in comp if p[1] == c0]
        else:
            cands = [p for p in comp if p[1] == c1]
        cr = (r0 + r1) / 2.0
        cc = (c0 + c1) / 2.0
        anchor = min(cands, key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2)

        dr_, dc_ = DIR[side]
        rr, cc_ = anchor
        placed = 0
        for _ in range(len(dots)):
            rr += dr_
            cc_ += dc_
            if rr < 0 or rr >= H or cc_ < 0 or cc_ >= W:
                break
            if out[rr, cc_] != 0:
                break
            out[rr, cc_] = dot_color
            placed += 1

    return out