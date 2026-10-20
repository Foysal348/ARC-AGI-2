import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape

    # Find color-5 components (8-connected)
    mask5 = (g == 5)
    visited = np.zeros((H, W), bool)
    comps5 = []
    for r in range(H):
        for c in range(W):
            if mask5[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                while q:
                    ar, ac = q.popleft()
                    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                        nr, nc = ar + dr, ac + dc
                        if 0 <= nr < H and 0 <= nc < W and mask5[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                comps5.append(cells)

    if len(comps5) < 2:
        return np.array([[0]], dtype=int)

    all_m = comps5[0] + comps5[1]
    r_min_m = min(r for r, c in all_m)
    r_max_m = max(r for r, c in all_m)
    c_min_m = min(c for r, c in all_m)
    c_max_m = max(c for r, c in all_m)

    row_ext_m = r_max_m - r_min_m + 1
    col_ext_m = c_max_m - c_min_m + 1

    if row_ext_m >= col_ext_m:
        is_vert = True
        long_dim = row_ext_m - 1
    else:
        is_vert = False
        long_dim = col_ext_m - 1

    if long_dim < 1:
        long_dim = 1

    # Find structure components (non-zero, non-5, 8-connected)
    struct_mask = (g != 0) & (g != 5)
    visited_s = np.zeros((H, W), bool)
    struct_comps = []
    for r in range(H):
        for c in range(W):
            if struct_mask[r, c] and not visited_s[r, c]:
                q = deque([(r, c)])
                visited_s[r, c] = True
                cells = [(r, c)]
                while q:
                    ar, ac = q.popleft()
                    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                        nr, nc = ar + dr, ac + dc
                        if 0 <= nr < H and 0 <= nc < W and struct_mask[nr, nc] and not visited_s[nr, nc]:
                            visited_s[nr, nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                struct_comps.append(cells)

    if not struct_comps:
        if is_vert:
            return np.zeros((long_dim, 1), dtype=int)
        else:
            return np.zeros((1, long_dim), dtype=int)

    def comp_dist(comp):
        min_d = float('inf')
        for (r, c) in comp:
            for (mr, mc) in all_m:
                d = (r - mr) ** 2 + (c - mc) ** 2
                min_d = min(min_d, d)
        return min_d

    # Sort components by distance to markers, take the closest
    struct_comps_sorted = sorted(struct_comps, key=comp_dist)
    main_comp = struct_comps_sorted[0]

    # Extract pattern from the main component
    if is_vert:
        c_min = min(c for r, c in main_comp)
        c_max = max(c for r, c in main_comp)
        r_min = min(r for r, c in main_comp)
        r_max = max(r for r, c in main_comp)
        pattern = []
        for r in range(r_min, r_max + 1):
            row = [int(g[r, c]) for c in range(c_min, c_max + 1)]
            pattern.append(row)
    else:
        r_min = min(r for r, c in main_comp)
        r_max = max(r for r, c in main_comp)
        c_min = min(c for r, c in main_comp)
        c_max = max(c for r, c in main_comp)
        # For horizontal: long axis = cols, short axis = rows
        # Extract as columns, then transpose
        cols_data = []
        for c in range(c_min, c_max + 1):
            col = [int(g[r, c]) for r in range(r_min, r_max + 1)]
            cols_data.append(col)
        if cols_data:
            pattern = [list(row) for row in zip(*cols_data)]
        else:
            pattern = [[0]]

    if not pattern:
        return np.array([[0]], dtype=int)

    short_dim = len(pattern[0])

    # Extend or truncate pattern to long_dim
    if len(pattern) >= long_dim:
        result = [row[:] for row in pattern[:long_dim]]
    else:
        # Find the period of the pattern
        n = len(pattern)
        period = n
        for k in range(1, n + 1):
            is_period = True
            for i in range(k, n):
                if pattern[i] != pattern[i - k]:
                    is_period = False
                    break
            if is_period:
                period = k
                break

        result = [row[:] for row in pattern]
        while len(result) < long_dim:
            idx = len(result)
            result.append(pattern[idx % period][:])

    if not result:
        return np.zeros((long_dim, short_dim), dtype=int)

    return np.array(result, dtype=int)