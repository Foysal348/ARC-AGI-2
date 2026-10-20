import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    arrow_color = 1

    comp_id = -np.ones((H, W), dtype=int)
    comp_colors = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] == arrow_color or comp_id[r, c] != -1:
                continue
            cid = len(comp_colors)
            comp_colors.append(int(grid[r, c]))
            q = deque([(r, c)])
            comp_id[r, c] = cid
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp_id[nr, nc] == -1 and grid[nr, nc] != arrow_color:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))

    edges = []

    def find_stem(arms):
        for (dr, dc) in arms:
            others = [a for a in arms if a != (dr, dc)]
            if len(others) == 2:
                (a1, a2) = others
                if a1[0] == -a2[0] and a1[1] == -a2[1]:
                    return (dr, dc)
        return None

    visited = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != arrow_color or visited[r, c]:
                continue
            q = deque([(r, c)])
            cells = []
            visited[r, c] = True
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[nr, nc] and grid[nr, nc] == arrow_color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            if len(cells) != 4:
                continue
            cell_set = set(cells)
            center = None
            for (rr, cc) in cells:
                cnt = sum(1 for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
                         if (rr+dr, cc+dc) in cell_set)
                if cnt == 3:
                    center = (rr, cc)
                    break
            if center is None:
                continue
            cr, cc = center
            arms = [(dr, dc) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
                   if (cr+dr, cc+dc) in cell_set]
            stem = find_stem(arms)
            if stem is None:
                continue
            sr, sc = stem
            src = comp_id[cr, cc]
            if src < 0:
                nr, nc = cr - sr, cc - sc
                while 0 <= nr < H and 0 <= nc < W:
                    if grid[nr, nc] != arrow_color:
                        src = comp_id[nr, nc]
                        break
                    nr -= sr
                    nc -= sc
            if src < 0:
                continue
            tgt = -1
            r2, c2 = cr, cc
            while True:
                r2 += sr
                c2 += sc
                if not (0 <= r2 < H and 0 <= c2 < W):
                    break
                if grid[r2, c2] == arrow_color:
                    continue
                t = comp_id[r2, c2]
                if t < 0:
                    continue
                if t == src:
                    continue
                tgt = t
                break
            if tgt >= 0:
                edges.append((src, tgt))

    for r in range(H):
        for c in range(W):
            if grid[r, c] == arrow_color:
                continue
            arms = [(dr, dc) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
                   if 0 <= r+dr < H and 0 <= c+dc < W and grid[r+dr, c+dc] == arrow_color]
            if len(arms) != 3:
                continue
            stem = find_stem(arms)
            if stem is None:
                continue
            sr, sc = stem
            src = comp_id[r, c]
            if src < 0:
                continue
            tgt = -1
            r2, c2 = r, c
            while True:
                r2 += sr
                c2 += sc
                if not (0 <= r2 < H and 0 <= c2 < W):
                    break
                if grid[r2, c2] == arrow_color:
                    continue
                t = comp_id[r2, c2]
                if t < 0:
                    continue
                if t == src:
                    continue
                tgt = t
                break
            if tgt >= 0:
                edges.append((src, tgt))

    final_colors = list(comp_colors)
    for src, tgt in edges:
        final_colors[tgt] = comp_colors[src]

    out = np.empty((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            cid = comp_id[r, c]
            if cid >= 0:
                out[r, c] = final_colors[cid]
            else:
                val = 0
                for dr, dc in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
                    rr, cc = r, c
                    found = False
                    while 0 <= rr < H and 0 <= cc < W:
                        if comp_id[rr, cc] >= 0:
                            val = final_colors[comp_id[rr, cc]]
                            found = True
                            break
                        rr += dr
                        cc += dc
                    if found:
                        break
                out[r, c] = val

    if H != W:
        out = out.T

    return out