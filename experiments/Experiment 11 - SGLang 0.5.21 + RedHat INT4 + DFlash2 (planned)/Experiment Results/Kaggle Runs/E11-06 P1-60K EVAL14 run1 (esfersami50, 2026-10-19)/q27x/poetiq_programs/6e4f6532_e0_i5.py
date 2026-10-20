import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()
    if h < 2 or w < 2:
        return out

    # Detect 2-px uniform frame bars
    row_bars = []
    for r in range(h - 1):
        if np.all(g[r] == g[r, 0]) and np.all(g[r + 1] == g[r + 1, 0]) and g[r, 0] == g[r + 1, 0]:
            row_bars.append((r, r + 1, int(g[r, 0])))

    col_bars = []
    for c in range(w - 1):
        if np.all(g[:, c] == g[0, c]) and np.all(g[:, c + 1] == g[0, c + 1]) and g[0, c] == g[0, c + 1]:
            col_bars.append((c, c + 1, int(g[0, c])))

    frame = np.zeros((h, w), dtype=bool)
    frame_colors = set()
    for r0, r1, cv in row_bars:
        frame[r0:r1 + 1, :] = True
        frame_colors.add(cv)
    for c0, c1, cv in col_bars:
        frame[:, c0:c1 + 1] = True
        frame_colors.add(cv)

    nf = g[~frame]
    if nf.size > 0:
        vals, cnts = np.unique(nf, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
    else:
        bg = 0

    # Find connected non-frame regions
    rid = np.full((h, w), -1, dtype=int)
    regions = []
    for i in range(h):
        for j in range(w):
            if not frame[i, j] and rid[i, j] == -1:
                q = deque([(i, j)])
                rid[i, j] = len(regions)
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not frame[ny, nx] and rid[ny, nx] == -1:
                            rid[ny, nx] = len(regions)
                            q.append((ny, nx))
                regions.append(cells)

    # Find all 8-containing components
    vis = np.zeros((h, w), dtype=bool)
    e8_comps = []
    for i in range(h):
        for j in range(w):
            if not vis[i, j] and g[i, j] != bg and not frame[i, j]:
                q = deque([(i, j)])
                vis[i, j] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not vis[ny, nx] and g[ny, nx] != bg and not frame[ny, nx]:
                            vis[ny, nx] = True
                            q.append((ny, nx))
                if any(g[y, x] == 8 for y, x in cells):
                    e8_comps.append(cells)

    if not e8_comps:
        return out

    # For each region, determine its bounding box
    region_bounds = {}
    for ri, rc in enumerate(regions):
        ys = [y for y, _ in rc]
        xs = [x for _, x in rc]
        region_bounds[ri] = (min(ys), max(ys), min(xs), max(xs))

    # Clear all non-frame, non-bg cells
    for i in range(h):
        for j in range(w):
            if not frame[i, j] and g[i, j] != bg:
                out[i, j] = bg

    # For each 8-component, extract and re-center in its region
    for comp in e8_comps:
        ys = [y for y, _ in comp]
        xs = [x for _, x in comp]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        ch = y1 - y0 + 1
        cw = x1 - x0 + 1
        mat = np.full((ch, cw), bg, dtype=g.dtype)
        for y, x in comp:
            mat[y - y0, x - x0] = g[y, x]

        # Get the region this component belongs to
        ri = int(rid[y0, x0])
        ry0, ry1, rx0, rx1 = region_bounds[ri]
        rh = ry1 - ry0 + 1
        rw = rx1 - rx0 + 1

        # Center the component in the region
        ny = ry0 + (rh - ch) // 2
        nx = rx0 + (rw - cw) // 2
        ny = max(ry0, min(ny, ry1 - ch + 1))
        nx = max(rx0, min(nx, rx1 - cw + 1))

        for yy in range(ch):
            for xx in range(cw):
                v = int(mat[yy, xx])
                if v == bg:
                    continue
                ay, ax = ny + yy, nx + xx
                if 0 <= ay < h and 0 <= ax < w and not frame[ay, ax]:
                    out[ay, ax] = v

    return out