import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    # background = most common color
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = grid.copy()

    # find all maximal 0-rectangles (rings) via connected components of 0 cells
    visited = np.zeros((h, w), dtype=bool)
    rects = []
    for r in range(h):
        for k in range(w):
            if grid[r, k] == 0 and not visited[r, k]:
                # BFS flood fill over 0-cells (4-connected)
                comp = [(r, k)]
                visited[r, k] = True
                i = 0
                while i < len(comp):
                    rr, cc = comp[i]
                    i += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            comp.append((nr, nc))
                rmin = min(p[0] for p in comp)
                rmax = max(p[0] for p in comp)
                kmin = min(p[1] for p in comp)
                kmax = max(p[1] for p in comp)
                if rmax - rmin >= 2 and kmax - kmin >= 2:
                    rects.append((rmin, rmax, kmin, kmax))

    # dots: non-background, non-zero cells
    dots = []
    for r in range(h):
        for k in range(w):
            if grid[r, k] not in (bg, 0):
                dots.append((r, k, int(grid[r, k])))

    # assign each dot to the smallest ring containing it (or nearest if none)
    def ring_corners(rect):
        rmin, rmax, kmin, kmax = rect
        return [(rmin, kmin), (rmin, kmax), (rmax, kmin), (rmax, kmax)]

    assigned = []
    for (r, k, c) in dots:
        cands = [rect for rect in rects
                 if rect[0] <= r <= rect[1] and rect[2] <= k <= rect[3]]
        if not cands:
            best = None
            bestd = None
            for rect in rects:
                d = (abs(r - (rect[0] + rect[1]) / 2.0) + abs(k - (rect[2] + rect[3]) / 2.0))
                if bestd is None or d < bestd:
                    bestd = d
                    best = rect
            cands = [best]
        cands.sort(key=lambda rect: (rect[1] - rect[0], rect[3] - rect[2]))
        assigned.append((r, k, c, cands[0]))

    # process each dot
    for (r, k, c, rect) in assigned:
        rmin, rmax, kmin, kmax = rect
        corners = ring_corners(rect)
        # choose corner: minimize (max dist, sum dist) -> diagonal distance then tie-break
        best = None
        bestd = None
        for (cr, ck) in corners:
            d = (max(abs(r - cr), abs(k - ck)), abs(r - cr) + abs(k - ck))
            if bestd is None or d < bestd:
                bestd = d
                best = (cr, ck)
        cr, ck = best

        # diagonal path from dot toward corner (excluding dot itself), then 2x2 block
        # extending outward from the corner, but cells on the ring border stay 0.
        # Outward direction is away from the ring interior.
        # path: step by (sign(cr-r), sign(ck-k)) from dot to corner inclusive
        sdr = 0 if cr == r else (1 if cr > r else -1)
        sdc = 0 if ck == k else (1 if ck > k else -1)
        # in-ring border check: cell on ring boundary of this rect
        def is_border(rr, cc):
            return (rr == rmin or rr == rmax or cc == kmin or cc == kmax) and \
                   (rmin <= rr <= rmax and kmin <= cc <= kmax)

        # 1) path cells (excluding the dot's own cell)
        rr, cc = r, k
        while (rr, cc) != (cr, ck):
            rr += sdr
            cc += sdc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = c

        # 2) 2x2 block at corner extending outward; skip border ring cells (they stay 0)
        orr = -sdr  # outward row step
        ock = -sdc  # outward col step
        for a in (0, 1):
            for b in (0, 1):
                pr = cr + orr * a
                pc = ck + ock * b
                if 0 <= pr < h and 0 <= pc < w:
                    if is_border(pr, pc):
                        continue
                    out[pr, pc] = c

    return out