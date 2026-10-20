import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    colors, cnt = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(cnt)])

    mask = (grid != bg)
    labeled = np.zeros((h, w), dtype=int)
    cid = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labeled[r, c] == 0:
                cid += 1
                stack = [(r, c)]
                labeled[r, c] = cid
                while stack:
                    a, b = stack.pop()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        na, nb = a+dr, b+dc
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and labeled[na, nb] == 0:
                            labeled[na, nb] = cid
                            stack.append((na, nb))

    u_shapes = []
    seeds = []
    for i in range(1, cid + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 1:
            seeds.append((int(ys[0]), int(xs[0]), int(grid[ys[0], xs[0]])))
        else:
            r0, r1 = int(ys.min()), int(ys.max())
            c0, c1 = int(xs.min()), int(xs.max())
            rh, rw = r1 - r0 + 1, c1 - c0 + 1
            if (rh == 2 or rw == 2) and rh >= 2 and rw >= 2:
                u_shapes.append((r0, r1, c0, c1))
            else:
                for y, x in zip(ys, xs):
                    seeds.append((int(y), int(x), int(grid[y, x])))

    seed_colors = sorted(set(s[2] for s in seeds))
    n = len(seed_colors)
    out = grid.copy()

    for (r0, r1, c0, c1) in u_shapes:
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        if rh == 2:
            row0 = grid[r0, c0:c1+1]
            row1 = grid[r1, c0:c1+1]
            if np.sum(row0 == 2) >= np.sum(row1 == 2):
                open_row, wall_row, proj_dir = r0, r1, -1
            else:
                open_row, wall_row, proj_dir = r1, r0, 1
            wall = grid[wall_row, c0:c1+1]
            four_pos = sorted([i for i in range(rw) if wall[i] == 4])
            non4_pos = sorted([i for i in range(rw) if wall[i] != 4])
            center = rw / 2.0
            four_pos.sort(key=lambda x: (abs(x - center), x))
            non4_pos.sort(key=lambda x: (abs(x - center), x))
            selected = []
            if four_pos:
                selected.append(four_pos[0])
            for i in range(n - 1):
                if i < len(non4_pos):
                    selected.append(non4_pos[i])
            selected.sort()
            for idx, pos in enumerate(selected):
                if idx >= n: break
                col_val = seed_colors[idx]
                c = c0 + pos
                if proj_dir == -1:
                    for rr in range(0, open_row):
                        out[rr, c] = col_val
                else:
                    for rr in range(open_row + 1, h):
                        out[rr, c] = col_val
        elif rw == 2:
            col0 = grid[r0:r1+1, c0]
            col1 = grid[r0:r1+1, c1]
            if np.sum(col0 == 2) >= np.sum(col1 == 2):
                open_col, wall_col, proj_dir = c0, c1, -1
            else:
                open_col, wall_col, proj_dir = c1, c0, 1
            wall = grid[r0:r1+1, wall_col]
            four_pos = sorted([i for i in range(rh) if wall[i] == 4])
            non4_pos = sorted([i for i in range(rh) if wall[i] != 4])
            center = rh / 2.0
            four_pos.sort(key=lambda x: (abs(x - center), x))
            non4_pos.sort(key=lambda x: (abs(x - center), x))
            selected = []
            if four_pos:
                selected.append(four_pos[0])
            for i in range(n - 1):
                if i < len(non4_pos):
                    selected.append(non4_pos[i])
            selected.sort()
            for idx, pos in enumerate(selected):
                if idx >= n: break
                col_val = seed_colors[idx]
                r = r0 + pos
                if proj_dir == -1:
                    for cc in range(0, open_col):
                        out[r, cc] = col_val
                else:
                    for cc in range(open_col + 1, w):
                        out[r, cc] = col_val
    return out