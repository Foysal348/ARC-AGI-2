import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    H, W = g.shape
    NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # ---------- 1. locate the two color-5 marker clusters ----------
    five = [(r, c) for r in range(H) for c in range(H) if g[r, c] == 5]
    if not five:
        five = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 5]
    if not five:
        return np.zeros_like(g)
    mask5 = np.zeros((H, W), bool)
    for r, c in five:
        mask5[r, c] = True
    seen = set(); clusters = []
    for r, c in five:
        if (r, c) in seen:
            continue
        q = deque([(r, c)]); seen.add((r, c)); cur = [(r, c)]
        while q:
            a, b = q.popleft()
            for dr, dc in NEIGH:
                nr, nc = a + dr, b + dc
                if 0 <= nr < H and 0 <= nc < W and mask5[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc)); q.append((nr, nc)); cur.append((nr, nc))
        clusters.append(cur)
    if len(clusters) >= 2:
        clusters = clusters[:2]
    r_min5 = min(r for r, _ in five); r_max5 = max(r for r, _ in five)
    c_min5 = min(c for _, c in five); c_max5 = max(c for _, c in five)

    # ---------- 2. pattern components (non-0, non-5) ----------
    struct = (g != 0) & (g != 5)
    cid = -np.ones((H, W), int)
    comps = []
    for r in range(H):
        for c in range(W):
            if struct[r, c] and cid[r, c] == -1:
                q = deque([(r, c)]); cid[r, c] = len(comps); cells = [(r, c)]
                while q:
                    a, b = q.popleft()
                    for dr, dc in NEIGH:
                        nr, nc = a + dr, b + dc
                        if 0 <= nr < H and 0 <= nc < W and cid[nr, nc] == -1 and struct[nr, nc]:
                            cid[nr, nc] = len(comps); q.append((nr, nc)); cells.append((nr, nc))
                comps.append(cells)
    if not comps:
        return np.zeros((r_max5 - r_min5 + 1, c_max5 - c_min5 + 1), int)

    def bbox(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    # ---------- 3. choose main color: components stacked along the long axis ----------
    out_h = r_max5 - r_min5 + 1
    out_w = c_max5 - c_min5 + 1
    vertical = out_h >= out_w

    def score_color(color):
        cells = [p for p in range(H) for cc in range(W)]  # placeholder
        cellset = [(r, c) for r in range(H) for c in range(W) if g[r, c] == color]
        if not cellset:
            return -1
        # components of this color
        m = np.zeros((H, W), bool)
        for r, c in cellset:
            m[r, c] = True
        comp_cells = []
        vis = set()
        for r, c in cellset:
            if (r, c) in vis:
                continue
            q = deque([(r, c)]); vis.add((r, c)); cur = [(r, c)]
            while q:
                a, b = q.popleft()
                for dr, dc in NEIGH:
                    nr, nc = a + dr, b + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc] and (nr, nc) not in vis:
                        vis.add((nr, nc)); q.append((nr, nc)); cur.append((nr, nc))
            comp_cells.append(cur)
        if len(comp_cells) < 2:
            return -1
        if vertical:
            key = [ (min(r for r,_ in cc), max(r for r,_ in cc)) for cc in comp_cells ]
        else:
            key = [ (min(c for _,c in cc), max(c for _,c in cc)) for cc in comp_cells ]
        # stack extent: from first start to last end
        span = max(e for _, e in key) - min(s for s, _ in key)
        # vertical separation (components not overlapping on short axis)
        if vertical:
            overlap = sum(1 for i in range(len(key)) for j in range(i+1, len(key)) if key[i][0] <= key[j][1] and key[j][0] <= key[i][1])
        else:
            overlap = sum(1 for i in range(len(key)) for j in range(i+1, len(key)) if key[i][0] <= key[j][1] and key[j][0] <= key[i][1])
        return (span, -overlap, len(cellset))

    colors = sorted(set(int(g[r, c]) for r in range(H) for c in range(W) if g[r, c] not in (0, 5)))
    main_color = None
    best = (-1, -1, -1)
    for col in colors:
        sc = score_color(col)
        if sc and sc > best:
            best = sc; main_color = col
    if main_color is None:
        # fallback: most frequent non-5 color
        from collections import Counter
        cnt = Counter(int(g[r, c]) for r in range(H) for c in range(W) if g[r, c] not in (0, 5))
        main_color = cnt.most_common(1)[0][0]

    # ---------- 4. build the 1-D row/column profile of the main color ----------
    if vertical:
        n = H
        def row_profile(idx):
            prof = []
            any_ = False
            for r in range(H):
                if g[r, idx] == main_color:
                    prof.append(idx)
            # find the contiguous block of this row index that belongs to the pattern
            return idx
        # For each row, the cross-section = the main_color cells in that row (across columns),
        # but we only keep the columns belonging to the "active" stripe for that row.
        # Simpler: gather per-row the set of columns of main_color; the stripe for a row is
        # the run of columns where the motif is drawn. We store the row's motif row (the
        # contiguous column-run of main_color) as a vector of width = run length, anchored
        # by its left column.
        rows = {}
        for r in range(H):
            cols = [c for c in range(W) if g[r, c] == main_color]
            if cols:
                # find longest contiguous run
                best_run = [cols[0]]
                for i in range(1, len(cols)):
                    if cols[i] == cols[i-1] + 1:
                        best_run.append(cols[i])
                    else:
                        best_run = [cols[i]]
                rows[r] = best_run
        if not rows:
            return np.zeros((out_h, out_w), int)
        # period along rows: find the repeating structure of (run length, pattern of 1/0 within a fixed width)
        # Determine motif width = max run length
        motif_w = max(len(v) for v in rows.values())
        # build per-row vector of width motif_w (0/1), with the run placed at its left offset
        # We'll instead treat the row content in absolute columns; the "type" of a row is the
        # tuple of (offset, pattern) but offset varies. Use just the pattern of the run plus zeros.
        # To tile vertically we need a period. Detect period p such that rows repeat their shape.
        rs_sorted = sorted(rows.keys())
        # shape = (run length, internal pattern as the run itself which is all 1s) -> all runs are
        # contiguous 1s, so shape is just run length.
        lengths = {r: len(rows[r]) for r in rs_sorted}
        # find period
        def find_period(seq_keys):
            L = len(seq_keys)
            for p in range(1, L + 1):
                ok = True
                for i in range(p, L):
                    if lengths[seq_keys[i]] != lengths[seq_keys[i - p]]:
                        ok = False; break
                if ok:
                    return p, seq_keys[:p]
            return L, seq_keys
        period, period_rows = find_period(rs_sorted)
        # cross-section width = motif_w
        out_w2 = motif_w
        out_h2 = out_h
        # phase: which period row aligns with marker top row r_min5
        # find the period row whose stripe is active at row r_min5 (or nearest)
        start_row = r_min5
        # map start_row into the pattern: find the base row (first row of pattern occurrence) 
        # Use the period_rows list; the pattern at row R is period_rows[(R - rs_sorted[0]) % period]
        offset = (start_row - rs_sorted[0]) % period
        # build output: each output row i -> pattern row (offset + i) % period
        # The pattern row's content: a horizontal run of main_color of length = that row's run length,
        # placed starting at the row's left offset relative to motif_w... we place run at left.
        res = np.zeros((out_h2, out_w2), int)
        for i in range(out_h2):
            pr = period_rows[(offset + i) % period]
            run = rows.get(pr, [])
            # place run (all main_color) starting at column 0, width = len(run)
            for j in range(len(run)):
                if j < out_w2:
                    res[i, j] = main_color
        # trim trailing all-zero columns/rows to match expected compact size
        res = _trim(res)
        return res
    else:
        # horizontal: symmetric
        cols = {}
        for c in range(W):
            rows_c = [r for r in range(H) if g[r, c] == main_color]
            if rows_c:
                best_run = [rows_c[0]]
                for i in range(1, len(rows_c)):
                    if rows_c[i] == rows_c[i-1] + 1:
                        best_run.append(rows_c[i])
                    else:
                        best_run = [rows_c[i]]
                cols[c] = best_run
        if not cols:
            return np.zeros((out_h, out_w), int)
        motif_h = max(len(v) for v in cols.values())
        cs_sorted = sorted(cols.keys())
        lengths = {c: len(cols[c]) for c in cs_sorted}
        def find_period(seq_keys):
            L = len(seq_keys)
            for p in range(1, L + 1):
                ok = True
                for i in range(p, L):
                    if lengths[seq_keys[i]] != lengths[seq_keys[i - p]]:
                        ok = False; break
                if ok:
                    return p, seq_keys[:p]
            return L, seq_keys
        period, period_cols = find_period(cs_sorted)
        out_h2 = motif_h
        out_w2 = out_w
        offset = (c_min5 - cs_sorted[0]) % period
        res = np.zeros((out_h2, out_w2), int)
        for j in range(out_w2):
            pc = period_cols[(offset + j) % period]
            run = cols.get(pc, [])
            for i in range(len(run)):
                if i < out_h2:
                    res[i, j] = main_color
        res = _trim(res)
        return res

def _trim(arr):
    if arr.size == 0:
        return arr
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return arr
    r0, r1 = nz[0].min(), nz[0].max()
    c0, c1 = nz[1].min(), nz[1].max()
    return arr[r0:r1+1, c0:c1+1]