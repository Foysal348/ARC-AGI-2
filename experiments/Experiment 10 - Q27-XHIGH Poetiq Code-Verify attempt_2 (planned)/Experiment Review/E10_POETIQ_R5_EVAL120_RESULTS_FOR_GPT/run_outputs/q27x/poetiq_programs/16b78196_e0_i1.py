import numpy as np
from collections import deque, defaultdict
from itertools import product

def _components(mask):
    """4-connected components of a binary mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                q = deque([(i, j)])
                seen[i, j] = True
                pix = []
                while q:
                    r, c = q.popleft()
                    pix.append((r, c))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(pix)
    return comps

def _bbox_and_mask(pixels, grid):
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    m = np.zeros((h, w), dtype=int)
    for r, c in pixels:
        m[r-r0, c-c0] = 1
    return (r0, c0, h, w), m

def _place(grid, mask, r, c, color, wall_mask, out, out_mult):
    h, w = mask.shape
    if r < 0 or c < 0 or r+h > grid.shape[0] or c+w > grid.shape[1]:
        return False
    sub = mask.astype(bool)
    # Cannot cover wall pixels.
    if np.any(wall_mask[r:r+h, c:c+w] & sub):
        return False
    # Count overlaps with already placed shapes.
    overlap = int(np.sum(out_mult[r:r+h, c:c+w] & sub))
    if overlap > 1:
        return False
    ys, xs = np.where(sub)
    for y, x in zip(ys, xs):
        rr, cc = r+y, c+x
        out[rr, cc] = max(out[rr, cc], color)
        out_mult[rr, cc] += 1
    return True

def _unplace(mask, r, c, out, out_mult):
    h, w = mask.shape
    sub = mask.astype(bool)
    ys, xs = np.where(sub)
    for y, x in zip(ys, xs):
        out_mult[r+y, c+x] -= 1
        if out_mult[r+y, c+x] == 0:
            out[r+y, c+x] = 0

def _pack_left(order, ref_pos, ref_mask, ref_color, grid, wall_mask, holes):
    """
    Pack shapes to the left of the reference shape.
    order is left-to-right, so all earlier shapes go left of ref.
    Returns list of (pos, mask, color) in left-to-right order, or None.
    """
    h, w = grid.shape
    placed = [(ref_pos, ref_mask, ref_color)]
    ok = _place(grid, ref_mask, ref_pos, 0, ref_color, wall_mask,
                np.zeros((h, w), dtype=int), np.zeros((h, w), dtype=int))
    out = np.zeros((h, w), dtype=int)
    out_mult = np.zeros((h, w), dtype=int)
    if ok:
        _place(grid, ref_mask, ref_pos, 0, ref_color, wall_mask, out, out_mult)
    else:
        return None

    for idx in range(len(order)-2, -1, -1):
        pos0, m0, col0 = order[idx]
        ph, pw = m0.shape
        prev_x = placed[-1][0]
        found = None
        # Search from just left of previous shape, moving left.
        for x in range(prev_x-1, -ph-1, -1):
            if _place(grid, m0, x, 0, col0, wall_mask, out, out_mult):
                found = (x, m0, col0)
                break
            if _place.__doc__ is None:  # no-op
                pass
            # Try to undo after a failed placement is unnecessary because _place does not modify on failure.
        if found is None:
            return None
        _place(grid, found[1], found[0], 0, found[2], wall_mask, out, out_mult)
        placed.append(found)

    return placed

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identify the large wall, group non-wall shapes by their dimension parallel
    to the wall, and stack them into the best matching wall slot.
    """
    grid = np.array(grid, dtype=int)
    H, W = grid.shape

    # Find the wall: largest component touching two opposite borders.
    wall_color = None
    wall_mask = np.zeros((H, W), dtype=bool)
    best_area = 0
    for color in np.unique(grid):
        if color == 0:
            continue
        mask = (grid == color)
        for comp in _components(mask):
            if len(comp) < best_area:
                continue
            rows = {p[0] for p in comp}
            cols = {p[1] for p in comp}
            touches_lr = (0 in cols and W-1 in cols)
            touches_tb = (0 in rows and H-1 in rows)
            if touches_lr or touches_tb:
                best_area = len(comp)
                wall_color = color
                wall_mask = np.zeros((H, W), dtype=bool)
                for r, c in comp:
                    wall_mask[r, c] = True

    if wall_color is None:
        return grid.copy()

    # Orientation: horizontal wall -> stack vertically; vertical wall -> stack horizontally.
    cols_wall = np.any(wall_mask, axis=0)
    rows_wall = np.any(wall_mask, axis=1)
    horizontal = np.argmax(cols_wall) < W and np.all(cols_wall[:np.argmax(cols_wall)] == False) and (W - np.argmax(cols_wall[::-1]) - 1 == W - np.argmax(cols_wall))
    # Simpler robust orientation:
    horizontal = (wall_mask[:, 0].any() and wall_mask[:, W-1].any()) or (np.sum(wall_mask) > 0 and W >= H and wall_mask[:, 0].sum() > wall_mask[0, :].sum())

    if horizontal:
        par = 1  # x parallel
        perp = 0 # y stacking
        par_len = W
        perp_len = H
        # Wall band rows
        row_counts = wall_mask.sum(axis=1)
        rows = np.where(row_counts > max(1, W*0.5))[0]
        wall_band = (rows.min(), rows.max()) if len(rows) else (0, H-1)
    else:
        par = 0  # y parallel
        perp = 1 # x stacking
        par_len = H
        perp_len = W
        col_counts = wall_mask.sum(axis=0)
        cols = np.where(col_counts > max(1, H*0.5))[0]
        wall_band = (cols.min(), cols.max()) if len(cols) else (0, W-1)

    # Hole cells: zeros inside wall bounding box / band, useful for scoring.
    holes = np.zeros((H, W), dtype=bool)
    if horizontal:
        r0, r1 = wall_band
        c0 = np.where(wall_mask)[1].min()
        c1 = np.where(wall_mask)[1].max()
        holes[r0:r1+1, c0:c1+1] = (grid[r0:r1+1, c0:c1+1] == 0)
    else:
        c0, c1 = wall_band
        r0 = np.where(wall_mask)[0].min()
        r1 = np.where(wall_mask)[0].max()
        holes[r0:r1+1, c0:c1+1] = (grid[r0:r1+1, c0:c1+1] == 0)

    # Extract non-wall objects.
    obj_mask = ~wall_mask & (grid > 0)
    objects = []
    for comp in _components(obj_mask):
        (r0, c0, h, w), m = _bbox_and_mask(comp, grid)
        color = int(grid[r0, c0])
        par_dim = w if horizontal else h
        side = 0
        if horizontal:
            side = 0 if r0 + h <= wall_band[0] else 1
        else:
            side = 0 if c0 + w <= wall_band[0] else 1
        objects.append({
            'color': color,
            'pos': (r0, c0),
            'mask': m,
            'par_dim': par_dim,
            'side': side,
            'par_start': c0 if horizontal else r0,
            'perp_start': r0 if horizontal else c0,
        })

    # Group by parallel dimension.
    groups = defaultdict(list)
    for o in objects:
        groups[o['par_dim']].append(o)

    out = grid.copy()
    # Clear non-wall pixels; wall remains.
    out[obj_mask] = 0

    for dim, objs in groups.items():
        if not objs:
            continue

        # Order: side 0 then side 1, color ascending.
        objs_sorted = sorted(objs, key=lambda x: (x['side'], x['color'], x['perp_start'], x['par_start']))

        # For this group, search over parallel start (slot position) and reference perpendicular position.
        best = None

        par_starts = range(max(0, par_len - dim - 5), min(par_len, dim + 5) if dim <= par_len else 0)
        if horizontal:
            par_starts = range(max(0, W - dim - 5), min(W, dim + 5) if dim <= W else 0)
        else:
            par_starts = range(max(0, H - dim - 5), min(H, dim + 5) if dim <= H else 0)

        for S in par_starts:
            ref = objs_sorted[-1]
            ref_mask = ref['mask']
            rh, rw = ref_mask.shape
            ref_perp_orig = ref['perp_start']

            # Candidate reference perpendicular positions: original, and wall-fit positions.
            cand_perp = {ref_perp_orig}
            for P in range(perp_len):
                # place ref alone with parallel S, perpendicular P
                if horizontal:
                    r, c = P, S
                else:
                    r, c = S, P
                hgt, wid = rh, rw
                if r < 0 or c < 0 or r+hgt > H or c+wid > W:
                    continue
                sub = ref_mask.astype(bool)
                if np.any(wall_mask[r:r+hgt, c:c+wid] & sub):
                    continue
                cov = int(np.sum(holes[r:r+hgt, c:c+wid] & sub))
                if cov > 0:
                    cand_perp.add(P)

            for P_ref in cand_perp:
                # Place all objects in this group at parallel S, packing left/up from reference.
                # Build a temporary list with only perpendicular positions.
                # We do a direct recursive/local placement.
                placed = []
                ok = True

                # Reference first.
                if horizontal:
                    r, c = P_ref, S
                else:
                    r, c = S, P_ref
                tmp_out = np.zeros((H, W), dtype=int)
                tmp_mult = np.zeros((H, W), dtype=int)
                if not _place(grid, ref_mask, r, c, ref['color'], wall_mask, tmp_out, tmp_mult):
                    continue
                placed.append((P_ref, ref_mask, ref['color']))

                for idx in range(len(objs_sorted)-2, -1, -1):
                    o = objs_sorted[idx]
                    m = o['mask']
                    ph, pw = m.shape
                    prev_perp = placed[-1][0]
                    found = None
                    # For horizontal, place above (smaller y); for vertical, place left (smaller x).
                    for P in range(prev_perp-1, -1, -1):
                        if horizontal:
                            rr, cc = P, S
                        else:
                            rr, cc = S, P
                        if _place(grid, m, rr, cc, o['color'], wall_mask, tmp_out, tmp_mult):
                            found = P
                            break
                    if found is None:
                        ok = False
                        break
                    if horizontal:
                        rr, cc = found, S
                    else:
                        rr, cc = S, found
                    _place(grid, m, rr, cc, o['color'], wall_mask, tmp_out, tmp_mult)
                    placed.append((found, m, o['color']))

                if not ok:
                    continue

                # Score: wall-hole coverage, then less movement, then smaller span.
                coverage = 0
                movement = 0
                perps = []
                for (P, m, col) in placed:
                    if horizontal:
                        r, c = P, S
                    else:
                        r, c = S, P
                    hgt, wid = m.shape
                    sub = m.astype(bool)
                    coverage += int(np.sum(holes[r:r+hgt, c:c+wid] & sub))
                    original_perp = next(o['perp_start'] for o in objs_sorted if o['mask'] is m and o['color'] == col)
                    movement += abs(P - original_perp)
                    perps.append(P)
                span = max(perps) - min(perps) if perps else 0
                score = (coverage, -movement, -span)
                if best is None or score > best[0]:
                    best = (score, S, placed)

        if best is None:
            # Fallback: keep original objects.
            for o in objs:
                out[o['pos'][0]:o['pos'][0]+o['mask'].shape[0], o['pos'][1]:o['pos'][1]+o['mask'].shape[1]] += o['mask']
            continue

        _, S, placed = best
        for P, m, col in placed:
            if horizontal:
                r, c = P, S
            else:
                r, c = S, P
            hgt, wid = m.shape
            sub = m.astype(bool)
            ys, xs = np.where(sub)
            for y, x in zip(ys, xs):
                out[r+y, c+x] = max(out[r+y, c+x], col)

    return out