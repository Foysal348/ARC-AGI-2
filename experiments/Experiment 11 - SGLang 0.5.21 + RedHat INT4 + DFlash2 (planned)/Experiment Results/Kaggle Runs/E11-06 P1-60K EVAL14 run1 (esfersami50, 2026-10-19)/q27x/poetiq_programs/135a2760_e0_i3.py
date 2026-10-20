import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    if H == 0 or W == 0:
        return out

    bg = int(g[0, 0])
    mask = (g != bg).astype(bool)

    visited = np.zeros((H, W), dtype=bool)
    rings = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            r0 = min(p[0] for p in comp)
            r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp)
            c1 = max(p[1] for p in comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border_colors = set()
            for rr in (r0, r1):
                for cc in range(c0, c1 + 1):
                    border_colors.add(int(g[rr, cc]))
            for cc in (c0, c1):
                for rr in range(r0 + 1, r1):
                    border_colors.add(int(g[rr, cc]))
            if len(border_colors) == 1:
                bc = border_colors.pop()
                if bc != bg:
                    rings.append((r0, c0, r1, c1))

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

    for idx in range(n):
        if is_container[idx]:
            continue
        r0, c0, r1, c1 = rings[idx]
        sub = g[r0 + 1:r1, c0 + 1:c1].astype(int)
        h, w = sub.shape
        if h == 0 or w == 0:
            continue

        max_pr = min(h, max(12, h // 2))
        max_pc = min(w, max(12, w // 2))

        best = None
        for pr in range(1, max_pr + 1):
            for pc in range(1, max_pc + 1):
                classes = {}
                for i in range(h):
                    for j in range(w):
                        key = (i % pr, j % pc)
                        if key not in classes:
                            classes[key] = []
                        classes[key].append(int(sub[i, j]))
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
                key = (i % pr, j % pc)
                if key not in classes:
                    classes[key] = []
                classes[key].append(int(sub[i, j]))

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