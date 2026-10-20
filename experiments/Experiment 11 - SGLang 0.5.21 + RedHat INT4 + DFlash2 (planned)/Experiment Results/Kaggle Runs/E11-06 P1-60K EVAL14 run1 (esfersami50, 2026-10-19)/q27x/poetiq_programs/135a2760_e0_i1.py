import numpy as np
from collections import Counter, defaultdict
from itertools import product

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    out = g.copy()
    H, W = g.shape

    def is_border_color(c):
        if g[0, 0] == c or g[-1, -1] == c or g[0, -1] == c or g[-1, 0] == c:
            return False
        if c == 0:
            return False
        return True

    def ring_ok(r0, c0, r1, c1, bc):
        if r1 - r0 < 2 or c1 - c0 < 2:
            return False
        for r in (r0, r1):
            for c in range(c0, c1 + 1):
                if g[r, c] != bc:
                    return False
        for c in (c0, c1):
            for r in range(r0 + 1, r1):
                if g[r, c] != bc:
                    return False
        return True

    def find_border_colors():
        cs = set()
        seen = set()
        for r in range(H):
            for c in range(W):
                v = g[r, c]
                if v in seen or not is_border_color(v):
                    continue
                if ring_ok(r, c, r, c + 3, v) and g[r, c + 1] == v:
                    stack = [(r, c)]
                    seen2 = set()
                    while stack:
                        x, y = stack.pop()
                        if (x, y) in seen2:
                            continue
                        seen2.add((x, y))
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and g[nx, ny] == v and (nx, ny) not in seen2:
                                stack.append((nx, ny))
                    if len(seen2) >= 8:
                        cs.add(int(v))
                if len(seen2 if False else seen2) >= 20000:
                    break
            if len(cs) > 5:
                break
        return cs

    border_colors = find_border_colors()

    used = np.zeros((H, W), dtype=bool)
    rects = []
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if int(v) in border_colors and not used[r, c] and ring_ok(r, c, r, c + 3, v) and g[r, c + 1] == v:
                stack = [(r, c)]
                seen2 = set()
                cells = []
                while stack:
                    x, y = stack.pop()
                    if (x, y) in seen2:
                        continue
                    seen2.add((x, y))
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and g[nx, ny] == v and (nx, ny) not in seen2:
                            stack.append((nx, ny))
                if len(cells) < 8:
                    continue
                rs = [a for a, _ in cells]
                cs_ = [b for _, b in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs_), max(cs_)
                if not ring_ok(r0, c0, r1, c1, v):
                    continue
                if not (r1 - r0 >= 2 and c1 - c0 >= 2):
                    continue
                expected_ring = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
                if len(cells) > expected_ring + max(1, expected_ring // 10):
                    continue
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        used[rr, cc] = True
                rects.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1))

    for (r0, c0, r1, c1) in rects:
        sub = g[r0:r1 + 1, c0:c1 + 1].astype(int)
        h, w = sub.shape
        n = h * w
        T = 2 if n < 60 else 5
        best = None
        for pr in range(1, h + 1):
            for pc in range(1, w + 1):
                classes = defaultdict(list)
                for i in range(h):
                    for j in range(w):
                        classes[(i % pr, j % pc)].append(sub[i, j])
                cost = 0
                ties = 0
                for lst in classes.values():
                    cnt = Counter(lst)
                    top = cnt.most_common(2)
                    m = top[0][1]
                    cost += len(lst) - m
                    if len(top) > 1 and top[1][1] == m:
                        ties += 1
                if cost > T:
                    continue
                key = (cost, ties, pr * pc, pr, pc)
                if best is None or key < best[0]:
                    best = (key, pr, pc, classes)
        if best is not None:
            key, pr, pc, classes = best
            for i in range(h):
                for j in range(w):
                    lst = classes[(i % pr, j % pc)]
                    cnt = Counter(lst)
                    top = cnt.most_common(2)
                    m = top[0][1]
                    if len(top) > 1 and top[1][1] == m:
                        cands = [col for col, k in cnt.items() if k == m]
                        if sub[i, j] in cands:
                            continue
                        vals = [sub[i, j]]
                        vals.append(lst[0])
                        vals.append(lst[len(lst) // 2] if len(lst) > 1 else lst[0])
                        vals.append(lst[-1])
                        sel = Counter(vals).most_common(1)[0][0]
                        out[r0 + i, c0 + j] = sel
                    else:
                        out[r0 + i, c0 + j] = top[0][0]
    return out