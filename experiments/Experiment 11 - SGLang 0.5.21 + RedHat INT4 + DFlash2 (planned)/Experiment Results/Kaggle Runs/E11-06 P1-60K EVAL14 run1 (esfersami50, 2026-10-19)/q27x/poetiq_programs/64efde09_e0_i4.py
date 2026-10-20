import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    colors, cnt = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(cnt)])
    out = grid.copy()

    # find candidate bar cells: non-background, non-4, part of a 2-cell-wide (vertically adjacent) non-4 pair
    mask = (grid != bg) & (grid != 4)
    adj = np.zeros((h, w), dtype=bool)
    adj[:-1] |= mask[:-1] & mask[1:]
    adj[1:] |= mask[1:] & mask[:-1]
    cands = mask & adj

    lab = np.zeros((h, w), dtype=int)
    cid = 0
    for r in range(h):
        for c in range(w):
            if cands[r, c] and lab[r, c] == 0:
                cid += 1
                stack = [(r, c)]
                lab[r, c] = cid
                while stack:
                    a, b = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + dr, b + dc
                        if 0 <= na < h and 0 <= nb < w and cands[na, nb] and lab[na, nb] == 0:
                            lab[na, nb] = cid
                            stack.append((na, nb))

    bars = []
    for i in range(1, cid + 1):
        ys, xs = np.where(lab == i)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        if rh == 2 or rw == 2:
            bars.append((r0, r1, c0, c1))

    # seed pixels: non-bg, non-4, not 4-adjacent to any 4
    seeds = []
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v == bg or v == 4:
                continue
            near4 = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 4:
                        near4 = True
            if not near4:
                seeds.append((r, c, int(v)))

    seed_colors = sorted({s[2] for s in seeds})
    if not bars or not seeds:
        return out

    n = len(seed_colors)
    L = min((max(b[1] - b[0] + 1, b[3] - b[2] + 1) for b in bars))
    M = max(1, L - 1)

    # assign each seed color to a distinct offset (1..M) via its coordinates
    used = set()
    color_to_off = {}
    for colr in sorted(seed_colors, reverse=True):
        r, c = next((sr, sc) for sr, sc, sv in seeds if sv == colr)
        cands = [v for v in (r % M, c % M, (r + c) % M) if v != 0]
        picked = None
        for v in cands:
            if v not in used:
                picked = v
                break
        if picked is None:
            for v in range(1, M + 1):
                if v not in used:
                    picked = v
                    break
        if picked is None:
            for v in cands:
                picked = v
                break
        if picked is None:
            continue
        used.add(picked)
        color_to_off[colr] = picked

    # bars we have not assigned yet
    unassigned = list(bars)
    if n >= 2:
        # first pass: horizontal bars (2 rows tall)
        for b in sorted(unassigned, key=lambda x: (x[0], x[2])):
            r0, r1, c0, c1 = b
            if r1 - r0 + 1 != 2:
                continue
            wall_top = sum(1 for c in range(c0, c1 + 1) if grid[r0, c] == 4)
            wall_bot = sum(1 for c in range(c0, c1 + 1) if grid[r1, c] == 4)
            proj_row, proj_dir = (r0, 1) if wall_top < wall_bot else (r1, -1)
            offs = sorted({off for off in color_to_off.values() if 0 < off <= c1 - c0})
            for k in range(min(n, len(offs))):
                colr = [cc for cc, vv in color_to_off.items() if vv == offs[k]][0]
                c = c0 + offs[k]
                if proj_dir == 1:
                    for rr in range(r1 + 1, h):
                        out[rr, c] = colr
                else:
                    for rr in range(r0 - 1, -1, -1):
                        out[rr, c] = colr
            unassigned.remove(b)

    # second pass: vertical bars (2 cols wide)
    for b in sorted(unassigned, key=lambda x: (x[2], x[0])):
        r0, r1, c0, c1 = b
        if c1 - c0 + 1 != 2:
            continue
        wall_left = sum(1 for r in range(r0, r1 + 1) if grid[r, c0] == 4)
        wall_right = sum(1 for r in range(r0, r1 + 1) if grid[r, c1] == 4)
        proj_col, proj_dir = (c0, 1) if wall_left < wall_right else (c1, -1)
        offs = sorted({off for off in color_to_off.values() if 0 < off <= r1 - r0})
        for k in range(min(n, len(offs))):
            colr = [cc for cc, vv in color_to_off.items() if vv == offs[k]][0]
            r = r0 + offs[k]
            if proj_dir == 1:
                for cc in range(c1 + 1, w):
                    out[r, cc] = colr
            else:
                for cc in range(c0 - 1, -1, -1):
                    out[r, cc] = colr

    return out