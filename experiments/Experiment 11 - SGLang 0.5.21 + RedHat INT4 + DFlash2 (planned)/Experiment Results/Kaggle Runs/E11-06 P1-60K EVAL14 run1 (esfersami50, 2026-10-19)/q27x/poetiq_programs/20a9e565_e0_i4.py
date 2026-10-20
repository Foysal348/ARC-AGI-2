import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape

    # Find color-5 cells
    five_cells = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 5]
    if not five_cells:
        return np.array([[0]])

    r_min5 = min(r for r, c in five_cells)
    r_max5 = max(r for r, c in five_cells)
    c_min5 = min(c for c in five_cells for r in range(H) if g[r, c] == 5)
    c_max5 = max(c for c in five_cells for r in range(H) if g[r, c] == 5)
    out_h = r_max5 - r_min5 + 1
    out_w = c_max5 - c_min5 + 1

    # Find all non-5, non-0 8-connected components
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

    # Score components: prefer those with enough cells, then leftmost
    scored = []
    for i, cells in enumerate(components):
        nonzero = len(cells)
        c_min = min(c for r, c in cells)
        c_max = max(c for r, c in cells)
        r_min = min(r for r, c in cells)
        r_max = max(r for r, c in cells)
        w = c_max - c_min + 1
        h = r_max - r_min + 1
        # Prefer components whose width is close to out_w (if out is tall) or height close to out_h (if out is wide)
        if out_h >= out_w:
            width_diff = abs(w - out_w)
        else:
            width_diff = abs(h - out_h)
        scored.append((width_diff, -nonzero, c_min, i))
    scored.sort()
    best_comp = components[scored[0][3]]

    # Extract pattern bounding box
    p_r_min = min(r for r, c in best_comp)
    p_r_max = max(r for r, c in best_comp)
    p_c_min = min(c for r, c in best_comp)
    p_c_max = max(c for r, c in best_comp)
    p_h = p_r_max - p_r_min + 1
    p_w = p_c_max - p_c_min + 1

    pattern = np.zeros((p_h, p_w), dtype=int)
    for (r, c) in best_comp:
        pattern[r - p_r_min, c - p_c_min] = g[r, c]

    # If output is wider than tall, transpose the pattern
    if out_w > out_h:
        pattern = pattern.T
        p_h, p_w = pattern.shape

    # Find vertical period
    def find_period(pat):
        n = pat.shape[0]
        for k in range(1, n + 1):
            ok = True
            for i in range(k, n):
                if not np.array_equal(pat[i], pat[i - k]):
                    ok = False
                    break
            if ok:
                return k
        return n

    period = find_period(pattern)

    # Determine starting phase: align pattern row 0 with output row 0
    # The pattern starts at p_r_min; output starts at r_min5
    # Phase offset = (r_min5 - p_r_min) % period
    phase = (r_min5 - p_r_min) % period

    # Build output by tiling the period
    result = np.zeros((out_h, out_w), dtype=int)
    for i in range(out_h):
        src_row = (phase + i) % p_h
        row_data = pattern[src_row].astype(int)
        if p_w >= out_w:
            result[i] = row_data[:out_w]
        else:
            for j in range(out_w):
                src_j = j % p_w
                result[i, j] = row_data[src_j]

    return result