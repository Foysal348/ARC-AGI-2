import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape

    def get_neighbors(mask, cells):
        seen = set()
        for (r, c) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells and mask[nr, nc]:
                        seen.add((nr, nc))
        return seen

    # --- locate the two color-5 markers ---
    marker_cells = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 5]
    comp = [set()]
    mask5 = np.zeros((H, W), bool)
    for (r, c) in marker_cells:
        mask5[r, c] = True
    seen = set()
    comps = []
    for (r, c) in marker_cells:
        if (r, c) in seen:
            continue
        q = deque([(r, c)])
        seen.add((r, c))
        cur = [(r, c)]
        while q:
            a, b = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = a + dr, b + dc
                    if 0 <= nr < H and 0 <= nc < W and mask5[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
                        cur.append((nr, nc))
        comps.append(set(cur))
    if len(comps) != 2:
        comps = [set(marker_cells)]

    def shape_key(cells):
        r0 = min(r for r, c in cells)
        c0 = min(c for r, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    shapes = [shape_key(s) for s in comps]
    if len(comps) == 2 and shapes[0] == shapes[1]:
        # same orientation: pair them; else pair by matching a 90-deg rotation
        if sum(comps[0]) <= sum(comps[1]):
            A, B = set(comps[0]), set(comps[1])
        else:
            A, B = set(comps[1]), set(comps[0])
    else:
        A, B = min(marker_cells), max(marker_cells)
        A = set([A]); B = set([B])
    # choose ordering: A = top-most cluster, B = bottom-most
    if min(r for r, c in A) > min(r for r, c in B):
        A, B = B, A

    # --- build structure graph (colors != 0, != 5), 8-connected components ---
    struct_mask = (g != 0) & (g != 5)
    comp_id = -np.ones((H, W), int)
    comps_list = []
    colors = []
    idx = 0
    for r in range(H):
        for c in range(W):
            if struct_mask[r, c] and comp_id[r, c] == -1:
                q = deque([(r, c)])
                comp_id[r, c] = idx
                cells = [(r, c)]
                while q:
                    a, b = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = a + dr, b + dc
                            if 0 <= nr < H and 0 <= nc < W and comp_id[nr, nc] == -1 and struct_mask[nr, nc]:
                                comp_id[nr, nc] = idx
                                q.append((nr, nc))
                                cells.append((nr, nc))
                comps_list.append(cells)
                colors.append(g[cells[0][0], cells[0][1]])
                idx += 1
    if idx == 0:
        return np.zeros_like(g)

    # edge-adjacency graph within each component
    node_pos = []          # (r, c)
    node_comp = []
    adj = []
    for ci, cells in enumerate(comps_list):
        cellset = set(cells)
        base = len(node_pos)
        for (r, c) in cells:
            node_pos.append((r, c))
            node_comp.append(ci)
        adj.append([])
        for i, (r, c) in enumerate(cells):
            vi = base + i
            for (nr, nc) in get_neighbors(comp_id, set()) :
                pass
    # rebuild adjacency properly
    adj = [list() for _ in range(idx)]
    for ci, cells in enumerate(comps_list):
        cellset = set(cells)
        base = sum(len(cc) for cc in comps_list[:ci])
        for i, (r, c) in enumerate(cells):
            for (nr, nc) in get_neighbors(comp_id, cellset):
                if (nr, nc) in cellset:
                    j = list(cellset).index((nr, nc))
                    adj[ci].append((base + i, base + j))

    # pick the component nearest to the markers
    def dist_comp(ci):
        cells = comps_list[ci]
        d = 10 ** 9
        for (r, c) in cells:
            for (mr, mc) in list(A) + list(B):
                d = min(d, (r - mr) ** 2 + (c - mc) ** 2)
        return d
    main = min(range(idx), key=dist_comp)
    if idx > 1:
        # prefer the largest component in near-ties
        sizes = {ci: len(comps_list[ci]) for ci in range(idx)}
        dmin = min(dist_comp(ci) for ci in range(idx))
        cands = [ci for ci in range(idx) if abs(dist_comp(ci) - dmin) <= 5]
        main = max(cands, key=lambda ci: sizes[ci])

    base = sum(len(cc) for cc in comps_list[:main])
    n = len(comps_list[main])
    adj_main = [list(d) for d in adj[main]]

    # find endpoints (degree 1)
    deg = [0] * n
    for (a, b) in adj_main:
        deg[a - base] += 1
        deg[b - base] += 1
    ends = [i for i in range(n) if deg[i] <= 1]
    if len(ends) >= 2:
        s, t = ends[0], ends[1]
    elif len(ends) == 1:
        s = ends[0]
        t = max(range(n), key=lambda i: deg[i])
    else:
        s = 0
        t = max(range(n), key=lambda i: deg[i])

    # walk the path from the endpoint nearer to A
    da = sum((node_pos[base + s][0] - mr) ** 2 + (node_pos[base + s][1] - mc) ** 2 for mr, mc in A)
    db = sum((node_pos[base + s][0] - mr) ** 2 + (node_pos[base + s][1] - mc) ** 2 for mr, mc in B)
    if da > db:
        s, t = t, s

    # BFS order along path
    order = []
    prev = {s: None}
    q = deque([s])
    while q:
        v = q.popleft()
        order.append(v)
        for u in adj_main[base + v]:
            u -= base
            if u not in prev:
                prev[u] = v
                q.append(u)
    if t not in prev:
        order = list(prev.keys())
    # ensure order follows the chain from s
    path = [s]
    while path[-1] != t and len(path) < n:
        v = path[-1]
        nxt = None
        for u in adj_main[base + v]:
            u -= base
            if u not in path:
                nxt = u
                break
        if nxt is None:
            break
        path.append(nxt)

    path_cells = [node_pos[base + v] for v in path]

    # --- straighten the chain into a 2-wide strip ---
    if len(path_cells) < 2:
        return np.zeros((3, 3), int)
    r0, c0 = path_cells[0]
    pts = [(r - r0, c - c0) for (r, c) in path_cells]
    # local coordinates: x = along-path, y = perpendicular (2 wide)
    # compute cumulative direction
    import math
    xs = [0]
    dirs = []
    for i in range(1, len(pts)):
        dr = pts[i][0] - pts[i - 1][0]
        dc = pts[i][1] - pts[i - 1][1]
        d = math.hypot(dr, dc)
        dirs.append((dr / d, dc / d))
        xs.append(xs[-1] + d)
    # perpendicular direction (left normal)
    if len(dirs) > 0:
        ux, uy = dirs[0]
    else:
        ux, uy = 0.0, 1.0
    vx, vy = -uy, ux
    # assign each cell to lane 0 (right) or 1 (left) based on dot with normal
    # and to a row based on projection onto u
    rows_data = {}
    for i, ((r, c), x) in enumerate(zip(pts, xs)):
        tproj = r * ux + c * uy
        side = r * vx + c * vy
        key = round(tproj)
        if key not in rows_data:
            rows_data[key] = {}
        col = 0 if side < 0 else 1
        # map to 2 columns: use side sign with threshold 0
        rows_data[key][col] = g[r, c]
    # sort rows by tproj
    max_x = max(xs)
    out_rows = []
    prev_r = None
    for i, ((r, c), x) in enumerate(zip(pts, xs)):
        tproj = r * ux + c * uy
        side = r * vx + c * vy
        row_idx = round(x)
        col = 1 if side > 0 else 0
        if len(out_rows) <= row_idx:
            out_rows.extend([[0, 0]] * (row_idx - len(out_rows) + 1))
        if out_rows[row_idx][col] == 0:
            out_rows[row_idx][col] = int(g[r, c])
    out = np.array(out_rows if len(out_rows) else [[0, 0]], int)
    if out.shape[0] < 1:
        out = np.zeros((1, 2), int)
    return out