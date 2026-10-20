import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Complete zero-framed corner motifs (see analysis above)."""
    if grid is None or grid.size == 0:
        return grid.copy()
    H, W = grid.shape
    if H < 2 or W < 2:
        return grid.copy()

    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    orig = grid.copy()
    out = grid.copy()
    zero = (orig == 0)
    painted = np.zeros((H, W), dtype=bool)

    def can_paint(r, c, col):
        if not (0 <= r < H and 0 <= c < W):
            return False
        v = orig[r, c]
        return v in (0, bg)

    def paint(r, c, col):
        if can_paint(r, c, col):
            out[r, c] = col
            painted[r, c] = True

    def in_block(cr, cc, odr, odc):
        nonbg = 0
        for i in (0, 1):
            for j in (0, 1):
                r, c = cr + odr * i, cc + odc * j
                if 0 <= r < H and 0 <= c < W and orig[r, c] not in (0, bg):
                    nonbg += 1
        return nonbg >= 2

    def border_ok(v):
        return v == 0 or (v != bg) or painted_flag

    # frames: border cells must be 0 or non-background (motif) color
    def is_frame(r0, c0, r1, c1):
        if r1 - r0 < 2 or c1 - c0 < 2:
            return False
        for c in range(c0, c1 + 1):
            for r in (r0, r1):
                v = orig[r, c]
                if not (v == 0 or v != bg):
                    return False
        for r in range(r0, r1 + 1):
            for c in (c0, c1):
                v = orig[r, c]
                if not (v == 0 or v != bg):
                    return False
        return True

    frames = []
    for r0 in range(H):
        for c0 in range(W):
            if orig[r0, c0] == bg:
                continue
            r1 = r0
            while r1 + 1 < H and orig[r1 + 1, c0] != bg:
                r1 += 1
            if r1 - r0 < 2:
                continue
            c1 = c0
            while c1 + 1 < W and orig[r0, c1 + 1] != bg:
                c1 += 1
            if c1 - c0 < 2:
                continue
            if is_frame(r0, c0, r1, c1):
                frames.append((r0, c0, r1, c1))

    frames.sort(key=lambda f: -((f[2] - f[0] + 1) * (f[3] - f[1] + 1)))
    unique_frames = []
    for f in frames:
        if any(g[0] <= f[0] and g[1] <= f[1] and f[2] <= g[2] and f[3] <= g[3] for g in unique_frames):
            continue
        unique_frames.append(f)

    def seed_at(r, c, dr, dc):
        rr, cc = r + dr, c + dc
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        v = orig[rr, cc]
        return int(v) if v not in (bg, 0) else None

    def corner_info(f):
        r0, c0, r1, c1 = f
        res = []
        for cr, cc, dr, dc in [(r0, c0, 1, 1), (r0, c1, 1, -1),
                               (r1, c0, -1, 1), (r1, c1, -1, -1)]:
            s1 = seed_at(cr, cc, dr, dc)
            s2 = seed_at(cr + dr, cc + dc, dr, dc)
            blk = in_block(cr, cc, -dr, -dc)
            res.append((cr, cc, dr, dc, s1, s2, blk))
        return res

    seeded = [x for f in unique_frames for x in corner_info(f) if x[4] is not None]
    use_thick = len(seeded) >= 2 and all(x[6] for x in seeded)
    diag_touch = 0
    for f in unique_frames:
        r0, c0, r1, c1 = f
        L = min(r1 - r0, c1 - c0)
        for i in range(1, L):
            for j in range(1, L):
                if orig[r0 + i, c0 + j] == 0:
                    diag_touch += 1
    use_mid2 = diag_touch > 0

    for f in unique_frames:
        for cr, cc, dr, dc, s1, s2, blk in corner_info(f):
            odr, odc = -dr, -dc
            if s1 is not None and not blk:
                if s2 is not None:
                    if use_mid2:
                        paint(cr + dr, cc + dc, s2)
                    paint(cr + 2 * dr, cc + 2 * dc, s2)
                    for i in (0, 1):
                        for j in (0, 1):
                            paint(cr + odr * i, cc + odc * j, s2)
                else:
                    if use_thick:
                        paint(cr + odr, cc + odc, s1)
                        paint(cr + odr, cc, s1)
                        paint(cr, cc + odc, s1)
                        paint(cr, cc + dc, s1)
                        paint(cr + dr, cc, s1)
                    else:
                        paint(cr, cc, s1)
                        paint(cr + odr, cc, s1)
                        paint(cr, cc + odc, s1)
            elif blk and s1 is None:
                col = None
                for i in (0, 1):
                    for j in (0, 1):
                        r, c = cr + odr * i, cc + odc * j
                        if 0 <= r < H and 0 <= c < W and orig[r, c] not in (0, bg):
                            col = int(orig[r, c])
                            break
                    if col is not None:
                        break
                if col is not None:
                    paint(cr + dr, cc + dc, col)

    return out