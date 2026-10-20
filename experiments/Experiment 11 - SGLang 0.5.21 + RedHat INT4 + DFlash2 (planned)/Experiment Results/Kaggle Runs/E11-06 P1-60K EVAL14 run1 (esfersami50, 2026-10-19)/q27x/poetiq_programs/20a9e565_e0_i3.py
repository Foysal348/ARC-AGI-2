import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape

    # Step 1: Find all color-5 cells and compute marker bbox
    mask5 = (g == 5)
    five_cells = [(r, c) for r in range(H) for c in range(W) if mask5[r, c]]
    if not five_cells:
        return np.array([[0]], dtype=int)

    r_min5 = min(r for r, c in five_cells)
    r_max5 = max(r for r, c in five_cells)
    c_min5 = min(c for r, c in five_cells)
    c_max5 = max(c for r, c in five_cells)
    out_h = r_max5 - r_min5 + 1
    out_w = c_max5 - c_min5 + 1

    # Step 2: Find all non-zero, non-5 connected components (8-connected)
    struct_mask = (g != 0) & (g != 5)
    visited = np.zeros((H, W), bool)
    components = []
    for r in range(H):
        for c in range(W):
            if struct_mask[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                while q:
                    ar, ac = q.popleft()
                    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                        nr, nc = ar + dr, ac + dc
                        if 0 <= nr < H and 0 <= nc < W and struct_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                components.append(cells)

    if not components:
        return np.zeros((out_h, out_w), dtype=int)

    # Step 3: Select the component farthest from the markers
    def comp_min_dist(comp):
        min_d = float('inf')
        for (r, c) in comp:
            for (mr, mc) in five_cells:
                d = (r - mr) ** 2 + (c - mc) ** 2
                min_d = min(min_d, d)
        return min_d

    dists = [comp_min_dist(comp) for comp in components]
    max_dist = max(dists)
    # Among those with max distance, pick the largest
    best_comps = [i for i in range(len(components)) if dists[i] == max_dist]
    best_comp = max(best_comps, key=lambda i: len(components[i]))
    main_cells = components[best_comp]

    # Step 4: Extract the pattern bounding box
    p_r_min = min(r for r, c in main_cells)
    p_r_max = max(r for r, c in main_cells)
    p_c_min = min(c for r, c in main_cells)
    p_c_max = max(c for r, c in main_cells)
    p_h = p_r_max - p_r_min + 1
    p_w = p_c_max - p_c_min + 1

    # Build the pattern grid
    pattern = np.zeros((p_h, p_w), dtype=int)
    for (r, c) in main_cells:
        pattern[r - p_r_min, c - p_c_min] = g[r, c]

    # Step 5: Determine main axis (based on marker bbox aspect ratio)
    if out_h >= out_w:
        main_axis = 'vertical'  # output is tall
    else:
        main_axis = 'horizontal'  # output is wide

    # Step 6: If horizontal main axis, transpose the pattern
    if main_axis == 'horizontal':
        pattern = pattern.T  # now pattern is (p_w, p_h) -> rows=p_w, cols=p_h
        p_h_new, p_w_new = pattern.shape
    else:
        p_h_new, p_w_new = p_h, p_w

    # Step 7: Determine the period along the main axis (rows of pattern)
    def find_period(pat, axis=0):
        n = pat.shape[axis]
        for k in range(1, n + 1):
            is_period = True
            for i in range(k, n):
                if axis == 0:
                    if not np.array_equal(pat[i], pat[i - k]):
                        is_period = False
                        break
                else:
                    if not np.array_equal(pat[:, i], pat[:, i - k]):
                        is_period = False
                        break
            if is_period:
                return k
        return n

    period = find_period(pattern, axis=0)

    # Step 8: Determine starting offset
    if main_axis == 'vertical':
        offset = r_min5 - p_r_min
    else:
        offset = c_min5 - p_c_min
    start_in_period = offset % period

    # Step 9: Build the output by tiling the period
    result = np.zeros((out_h, out_w), dtype=int)
    for i in range(out_h):
        pat_row_idx = (start_in_period + i) % p_h_new
        row_data = pattern[pat_row_idx].astype(int)  # 1D array of length p_w_new

        # Stretch or compress to out_w
        if p_w_new >= out_w:
            # Take a slice or use the first out_w elements
            result[i] = row_data[:out_w]
        else:
            # Stretch: for each output column, map to input column
            for j in range(out_w):
                src_j = min(int(j * p_w_new / out_w), p_w_new - 1)
                result[i, j] = row_data[src_j]

    return result