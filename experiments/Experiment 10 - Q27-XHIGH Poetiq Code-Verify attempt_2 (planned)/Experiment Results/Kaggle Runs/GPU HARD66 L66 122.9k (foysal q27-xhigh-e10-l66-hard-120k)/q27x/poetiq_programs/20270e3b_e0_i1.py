import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Join two 4-colored objects marked by 7 pixels.

    The 7 pixels mark the sides of two objects. The smaller marked object is
    moved so that its marked side is directly above the marked side of the
    other object. After overlaying, the grid is cropped to the 4-object
    bounding box.
    """
    def crop_to_4(g):
        """Crop grid to the bounding box of all 4 cells."""
        if not np.any(g == 4):
            return g.copy()
        rows = np.where((g == 4).any(axis=1))[0]
        cols = np.where((g == 4).any(axis=0))[0]
        return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()

    def bbox_area(b):
        r1, c1, r2, c2 = b
        return (r2 - r1 + 1) * (c2 - c1 + 1)

    a = np.array(grid, dtype=int)
    if a.size == 0:
        return a.copy()

    # If there are no markers, nothing to do.
    if not np.any(a == 7):
        return a.copy()

    # Background is the most common non-4, non-7 color.
    vals, counts = np.unique(a[a != 7], return_counts=True)
    non_obj = vals != 4
    if np.any(non_obj):
        bg = int(vals[non_obj][np.argmax(counts[non_obj])])
    else:
        bg = 1

    H, W = a.shape

    # If there are no 4 objects, just remove markers.
    if not np.any(a == 4):
        base = a.copy()
        base[base == 7] = bg
        return crop_to_4(base)

    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Find 4-connected components of 4.
    comp_id = np.full((H, W), -1, dtype=int)
    comps = []
    bboxes = []

    for r in range(H):
        for c in range(W):
            if a[r, c] == 4 and comp_id[r, c] == -1:
                cid = len(comps)
                cells = []
                stack = [(r, c)]
                comp_id[r, c] = cid

                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))

                    for dx, dy in dirs4:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < H
                            and 0 <= ny < W
                            and a[nx, ny] == 4
                            and comp_id[nx, ny] == -1
                        ):
                            comp_id[nx, ny] = cid
                            stack.append((nx, ny))

                rs = [r0 for r0, _ in cells]
                cs = [c0 for _, c0 in cells]
                comps.append(cells)
                bboxes.append((min(rs), min(cs), max(rs), max(cs)))

    # For each 4-component, collect adjacent 7 marker cells.
    marker_cells = [set() for _ in comps]

    for r in range(H):
        for c in range(W):
            if a[r, c] == 7:
                for dx, dy in dirs4:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 4:
                        cid = comp_id[nr, nc]
                        if cid >= 0:
                            marker_cells[cid].add((r, c))

    candidates = [i for i, cells in enumerate(marker_cells) if cells]

    # Base grid with markers removed.
    base = a.copy()
    base[base == 7] = bg

    # Fallback: not enough marked objects.
    if len(candidates) < 2:
        return crop_to_4(base)

    # If more than two are found, keep the two largest marked objects.
    if len(candidates) > 2:
        candidates = sorted(candidates, key=lambda i: bbox_area(bboxes[i]), reverse=True)[:2]

    infos = []
    for cid in candidates:
        cells = marker_cells[cid]
        mr1 = min(r for r, _ in cells)
        mr2 = max(r for r, _ in cells)
        mc1 = min(c for _, c in cells)
        mc2 = max(c for _, c in cells)

        infos.append({
            "id": cid,
            "bbox": bboxes[cid],
            "area": bbox_area(bboxes[cid]),
            "m_top": mr1,
            "m_bottom": mr2,
            "m_left": mc1,
            "m_right": mc2,
        })

    A, B = infos

    # Choose the smaller object as the movable one.
    # Tie-breaker: lower marked object moves; then rightmost marked object.
    if A["area"] != B["area"]:
        src, tgt = (A, B) if A["area"] < B["area"] else (B, A)
    elif A["m_bottom"] != B["m_bottom"]:
        src, tgt = (A, B) if A["m_bottom"] > B["m_bottom"] else (B, A)
    elif A["m_left"] != B["m_left"]:
        src, tgt = (A, B) if A["m_left"] > B["m_left"] else (B, A)
    else:
        src, tgt = A, B

    # Translation so source marker sits directly above target marker.
    dr = tgt["m_top"] - src["m_bottom"] - 1
    dc = tgt["m_left"] - src["m_left"]

    r1, c1, r2, c2 = src["bbox"]

    # Mask of all 4 pixels in the source object's bounding box.
    # Keep other main marked objects if their bboxes overlap.
    exclude_ids = set(candidates) - {src["id"]}
    src_mask = np.zeros((H, W), dtype=bool)

    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if base[r, c] == 4 and comp_id[r, c] not in exclude_ids:
                src_mask[r, c] = True

    # Remove source object from original position.
    base[src_mask] = bg

    # Create a canvas large enough for original grid + translated source.
    sr1 = r1 + dr
    sr2 = r2 + dr
    sc1 = c1 + dc
    sc2 = c2 + dc

    new_r_min = min(0, sr1)
    new_r_max = max(H - 1, sr2)
    new_c_min = min(0, sc1)
    new_c_max = max(W - 1, sc2)

    new_H = new_r_max - new_r_min + 1
    new_W = new_c_max - new_c_min + 1

    new_grid = np.full((new_H, new_W), bg, dtype=int)

    off_r = -new_r_min
    off_c = -new_c_min
    new_grid[off_r:off_r + H, off_c:off_c + W] = base

    # Place translated source object.
    for r, c in np.argwhere(src_mask):
        nr = r + dr - new_r_min
        nc = c + dc - new_c_min
        if 0 <= nr < new_H and 0 <= nc < new_W:
            new_grid[nr, nc] = 4

    return crop_to_4(new_grid)