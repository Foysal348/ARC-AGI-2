import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()

    # background = most common color
    colors, cnt = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(cnt)])

    # seed pixels: non-bg, not touching a 4 diagonally/orthogonally
    seeds = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or grid[r, c] == 4:
                continue
            near4 = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 4:
                        near4 = True
            if not near4:
                seeds.append((r, c, int(grid[r, c])))
    seed_colors = sorted(set(s[2] for s in seeds))

    # connected components (4-connectivity) of non-bg pixels
    mask = (grid != bg)
    lab = np.zeros((h, w), dtype=int)
    cid = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and lab[r, c] == 0:
                cid += 1
                stack = [(r, c)]
                lab[r, c] = cid
                while stack:
                    a, b = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + dr, b + dc
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and lab[na, nb] == 0:
                            lab[na, nb] = cid
                            stack.append((na, nb))

    bars = []
    for i in range(1, cid + 1):
        ys, xs = np.where(lab == i)
        r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        L = max(r1 - r0 + 1, c1 - c0 + 1)
        if L < 3:
            continue
        if c1 - c0 == 1:  # vertical bar: rim col is the one with more 4s
            n4a = int(np.sum(grid[r0:r1 + 1, c0] == 4))
            n4b = int(np.sum(grid[r0:r1 + 1, c1] == 4))
            if n4a >= n4b:
                proj = 1   # away from rim c0 -> right
                proj_c = c1 + 1
            else:
                proj = -1  # left
                proj_c = c0 - 1
            bars.append(('v', int(r0), int(r1), int(c0), int(c1), L, proj, proj_c))
        elif r1 - r0 == 1:  # horizontal bar
            n4a = int(np.sum(grid[r0, c0:c1 + 1] == 4))
            n4b = int(np.sum(grid[r1, c0:c1 + 1] == 4))
            if n4a >= n4b:
                proj = 1   # down
                proj_r = r1 + 1
            else:
                proj = -1  # up
                proj_r = r0 - 1
            bars.append(('h', int(r0), int(r1), int(c0), int(c1), L, proj, proj_r))

    # place each seed's beam on every matching-orientation bar, using the
    # bar the seed is closest to (perpendicular distance) as the reference
    for bar in bars:
        orient, r0, r1, c0, c1, L, proj, _ = bar
        if not bars or not seeds:
            continue
        # reference bar of same orientation
        cands = [b for b in bars if b[0] == orient]
        if not cands:
            continue
        if orient == 'v':
            best = min(cands, key=lambda b: min(abs(b[4] - sc[0]) for sc in seeds) + min(abs(b[3] - sc[0]), abs(b[4] - sc[0])))
            best = min(cands, key=lambda b: min((min(b[1], abs(b[1] - sr)) if sr >= b[1] and sr <= b[2] else 10**9) + abs(b[3] - sc) + abs(b[4] - sc) for (sr, sc, _) in seeds))
            s0, s1 = best[1], best[2]
        else:
            best = min(cands, key=lambda b: min((min(b[3], abs(b[3] - sc)) if sc >= b[3] and sc <= b[4] else 10**9) + abs(b[1] - sr) + abs(b[2] - sr) for (sr, sc, _) in seeds))
            s0, s1 = best[3], best[4]

        used = set()
        order = []  # (position, color)
        for (sr, sc, scol) in seeds:
            x = sr if orient == 'v' else sc
            pos = x - s0
            if pos < 0 or pos >= L:
                pos = int(x) % L
            if pos == 0 or pos == L - 1:
                for d in (1, -1, 2, -2, 3, -3, 4, -4):
                    p2 = pos + d
                    if 1 <= p2 < L - 1 and p2 not in used:
                        pos = p2
                        break
            else:
                p2 = pos
                while p2 in used:
                    p2 += 1
                    if p2 >= L - 1:
                        p2 = 1
                pos = p2
            used.add(pos)
            order.append((pos, scol))

        for (pos, scol) in order:
            if orient == 'v':
                rr = r0 + pos
                if proj == 1:
                    for cc in range(c1 + 1, w):
                        out[rr, cc] = scol
                else:
                    for cc in range(0, c0):
                        out[rr, cc] = scol
            else:
                cc = c0 + pos
                if proj == 1:
                    for rr in range(r1 + 1, h):
                        out[rr, cc] = scol
                else:
                    for rr in range(0, r0):
                        out[rr, cc] = scol
    return out