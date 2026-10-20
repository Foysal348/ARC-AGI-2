import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extend colored straight arms attached to 1-colored shapes to the nearest
    grid border in the direction they point.

    The transformation:
    1. Treat the dominant color as background (8 in the given examples).
    2. Find connected components of pixels that are neither background nor 1.
    3. Each such component that touches a 1-component is an arm.
    4. Determine the outward direction of the arm (up/down/left/right).
    5. If the arm does not already reach the border in that direction,
       extend it to the border by repeating its visible color sequence.
    6. Extensions preserve 1-pixels, but may overwrite background or other
       arm pixels.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    # Determine background. Usually the dominant color. If 1 is dominant
    # for some reason, fall back to border/most-common non-1 color.
    vals_all, counts_all = np.unique(grid, return_counts=True)
    bg = int(vals_all[np.argmax(counts_all)])
    if bg == 1:
        border = np.concatenate((
            grid[0, :].ravel(),
            grid[-1, :].ravel(),
            grid[:, 0].ravel(),
            grid[:, -1].ravel()
        ))
        vals_b, counts_b = np.unique(border, return_counts=True)
        bg = int(vals_b[np.argmax(counts_b)])
        if bg == 1:
            non1 = vals_all != 1
            if np.any(non1):
                bg = int(vals_all[non1][np.argmax(counts_all[non1])])

    mask1 = grid == 1
    mask_arm = (grid != bg) & (grid != 1)

    def components(mask):
        """Return component ids and [(cells, bbox), ...] for a boolean mask."""
        comp_id = np.full((H, W), -1, dtype=int)
        comps = []

        for i in range(H):
            for j in range(W):
                if not mask[i, j] or comp_id[i, j] != -1:
                    continue

                cid = len(comps)
                stack = [(i, j)]
                comp_id[i, j] = cid
                cells = []
                min_r = max_r = i
                min_c = max_c = j

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    if r < min_r:
                        min_r = r
                    if r > max_r:
                        max_r = r
                    if c < min_c:
                        min_c = c
                    if c > max_c:
                        max_c = c

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < H and 0 <= nc < W
                            and mask[nr, nc]
                            and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = cid
                            stack.append((nr, nc))

                comps.append((cells, (min_r, max_r, min_c, max_c)))

        return comp_id, comps

    id1, shape_comps = components(mask1)
    _, arm_comps = components(mask_arm)

    def minimal_period(seq):
        """Smallest period p such that seq[i] == seq[i % p] for all i."""
        L = len(seq)
        if L == 0:
            return 1

        for p in range(1, L + 1):
            ok = True
            for i in range(p, L):
                if seq[i] != seq[i % p]:
                    ok = False
                    break
            if ok:
                return p

        return L

    tasks = []

    for cells, (tr0, tr1, tc0, tc1) in arm_comps:
        # Which 1-components does this arm touch?
        touching = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    sid = id1[nr, nc]
                    if sid != -1:
                        touching.add(sid)

        if not touching:
            continue

        candidates = []

        # Prefer direction from bounding-box relationship.
        for sid in touching:
            sr0, sr1, sc0, sc1 = shape_comps[sid][1]

            col_overlap = tc0 <= sc1 and sc0 <= tc1
            row_overlap = tr0 <= sr1 and sr0 <= tr1

            d = None
            if tr1 < sr0 and col_overlap:
                d = "up"
            elif tr0 > sr1 and col_overlap:
                d = "down"
            elif tc1 < sc0 and row_overlap:
                d = "left"
            elif tc0 > sc1 and row_overlap:
                d = "right"

            if d is not None:
                candidates.append((sid, d))

        # Fallback: infer direction from adjacent 1 pixels directly.
        if not candidates:
            dir_counts = {}
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue

                    sid = id1[nr, nc]
                    if sid == -1:
                        continue

                    if nr == r + 1:
                        d = "up"
                    elif nr == r - 1:
                        d = "down"
                    elif nc == c + 1:
                        d = "left"
                    elif nc == c - 1:
                        d = "right"
                    else:
                        continue

                    key = (sid, d)
                    dir_counts[key] = dir_counts.get(key, 0) + 1

            if dir_counts:
                (sid, d) = max(dir_counts.items(), key=lambda kv: (kv[1], kv[0]))
                candidates.append((sid, d))

        if not candidates:
            continue

        # If multiple possible shapes/directions, choose the closest one.
        if len(candidates) > 1:
            def dist(item):
                sid, d = item
                sr0, sr1, sc0, sc1 = shape_comps[sid][1]
                if d == "up":
                    return sr0 - tr1
                if d == "down":
                    return tr0 - sr1
                if d == "left":
                    return sc0 - tc1
                return tc0 - sc1

            candidates.sort(key=dist)

        sid, d = candidates[0]

        # Build an extension task.
        if d == "down":
            seq = [tuple(grid[r, tc0:tc1 + 1].tolist()) for r in range(tr0, tr1 + 1)]
            if tr1 + 1 < H:
                p = minimal_period(seq)
                tasks.append((d, tc0, tc1, tr0, tr1, seq, p))

        elif d == "up":
            seq = [tuple(grid[r, tc0:tc1 + 1].tolist()) for r in range(tr1, tr0 - 1, -1)]
            if tr0 - 1 >= 0:
                p = minimal_period(seq)
                tasks.append((d, tc0, tc1, tr1, tr0, seq, p))

        elif d == "right":
            seq = [tuple(grid[tr0:tr1 + 1, c].tolist()) for c in range(tc0, tc1 + 1)]
            if tc1 + 1 < W:
                p = minimal_period(seq)
                tasks.append((d, tr0, tr1, tc0, tc1, seq, p))

        elif d == "left":
            seq = [tuple(grid[tr0:tr1 + 1, c].tolist()) for c in range(tc1, tc0 - 1, -1)]
            if tc0 - 1 >= 0:
                p = minimal_period(seq)
                tasks.append((d, tr0, tr1, tc1, tc0, seq, p))

    # Deterministic order. This rarely matters for the provided-style inputs,
    # but gives a stable result if multiple extensions overlap.
    order = {"up": 0, "down": 1, "left": 2, "right": 3}
    tasks.sort(key=lambda t: (order[t[0]], len(t[5])))

    # Apply extensions.
    for task in tasks:
        d = task[0]

        if d == "down":
            _, c0, c1, near, far, seq, p = task
            for r in range(far + 1, H):
                val = np.array(seq[(r - near) % p], dtype=int)
                seg = out[r, c0:c1 + 1].copy()
                if not np.all(seg == 1):
                    keep = seg == 1
                    seg[~keep] = val[~keep]
                    out[r, c0:c1 + 1] = seg

        elif d == "up":
            _, c0, c1, near, far, seq, p = task
            for r in range(far - 1, -1, -1):
                val = np.array(seq[(near - r) % p], dtype=int)
                seg = out[r, c0:c1 + 1].copy()
                if not np.all(seg == 1):
                    keep = seg == 1
                    seg[~keep] = val[~keep]
                    out[r, c0:c1 + 1] = seg

        elif d == "right":
            _, r0, r1, near, far, seq, p = task
            for c in range(far + 1, W):
                val = np.array(seq[(c - near) % p], dtype=int)
                seg = out[r0:r1 + 1, c].copy()
                if not np.all(seg == 1):
                    keep = seg == 1
                    seg[~keep] = val[~keep]
                    out[r0:r1 + 1, c] = seg

        elif d == "left":
            _, r0, r1, near, far, seq, p = task
            for c in range(far - 1, -1, -1):
                val = np.array(seq[(near - c) % p], dtype=int)
                seg = out[r0:r1 + 1, c].copy()
                if not np.all(seg == 1):
                    keep = seg == 1
                    seg[~keep] = val[~keep]
                    out[r0:r1 + 1, c] = seg

    return out