import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()

    # 1) find all connected components and test if each is exactly the
    #    1-cell-thick boundary (ring) of its bounding box
    visited = np.zeros((H, W), dtype=bool)
    rings = []
    for r in range(H):
        for c in range(W):
            if visited[r, c]:
                continue
            v = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny] and int(g[nx, ny]) == v:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            if len(comp) < 8:
                continue
            r0 = min(p[0] for p in comp)
            r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp)
            c1 = max(p[1] for p in comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            ring_cells = set()
            for rr in (r0, r1):
                for cc in range(c0, c1 + 1):
                    ring_cells.add((rr, cc))
            for cc in (c0, c1):
                for rr in range(r0 + 1, r1):
                    ring_cells.add((rr, cc))
            if set(comp) == ring_cells:
                rings.append((r0, c0, r1, c1))

    # 2) a ring that contains another ring in its interior is only a
    #    container frame; only rings with clean interiors are pattern boxes
    n = len(rings)
    is_container = [False] * n
    for i in range(n):
        r0, c0, r1, c1 = rings[i]
        for j in range(n):
            if i == j:
                continue
            s0, sc0, s1, sc1 = rings[j]
            if r0 + 1 <= s0 and s1 <= r1 - 1 and c0 + 1 <= sc0 and sc1 <= c1 - 1:
                is_container[i] = True
                break

    # 3) for each pattern box, find the small repeating tile (period)
    #    with fewest mismatches and fill every cell from it
    for idx in range(n):
        if is_container[idx]:
            continue
        r0, c0, r1, c1 = rings[idx]
        sub = g[r0 + 1:r1, c0 + 1:c1].astype(int)
        h, w = sub.shape
        if h == 0 or w == 0:
            continue
        best = None
        for pr in range(1, h + 1):
            for pc in range(1, w + 1):
                classes = {}
                for i in range(h):
                    for j in range(w):
                        classes.setdefault((i % pr, j % pc), []).append(int(sub[i, j]))
                # a class with a single observation is too unreliable
                if any(len(lst) < 2 for lst in classes.values()):
                    continue
                cost = 0
                for lst in classes.values():
                    cnt = Counter(lst)
                    cost += len(lst) - max(cnt.values())
                key = (cost, pr * pc, pr, pc)
                if best is None or key < best[0]:
                    best = (key, pr, pc)
        if best is None:
            continue
        _, pr, pc = best
        classes = {}
        for i in range(h):
            for j in range(w):
                classes.setdefault((i % pr, j % pc), []).append(int(sub[i, j]))
        global_cnt = Counter(int(x) for x in sub.flatten())
        fill = {}
        for k, lst in classes.items():
            cnt = Counter(lst)
            top = max(cnt.values())
            cands = [c for c, k2 in cnt.items() if k2 == top]
            if len(cands) == 1:
                fill[k] = int(cands[0])
            else:
                cands.sort(key=lambda c: (-global_cnt[c], int(c)))
                fill[k] = int(cands[0])
        for i in range(h):
            for j in range(w):
                out[r0 + 1 + i, c0 + 1 + j] = fill[(i % pr, j % pc)]
    return out