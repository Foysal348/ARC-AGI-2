import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by extracting a small canonical shape from one or
    more panels and re-placing it upright at a marker/anchor position.

    The routine is designed for the observed task family:
      - optional vertical separator columns split the grid into panels
      - one panel is selected as the active output panel
      - a small symmetric shape (T / V / arrow) appears in rotated/noisy form
      - the clean upright shape is placed using a non-core extreme marker
    """
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H_in, W_in = grid.shape
    if H_in == 0 or W_in == 0:
        return grid.copy()

    def global_background(arr):
        vals, counts = np.unique(arr, return_counts=True)
        return int(vals[np.argmax(counts)])

    global_bg = global_background(grid)

    # -------------------------------------------------------------------------
    # 1) Optional panel splitting by full vertical separator columns.
    # -------------------------------------------------------------------------
    sep_cols = []
    for c in range(W_in):
        col = grid[:, c]
        if col.size == 0:
            continue
        if np.all(col == col[0]) and int(col[0]) != global_bg:
            sep_cols.append(c)

    panels = []
    if sep_cols:
        bounds = []
        prev = 0
        for c in sep_cols:
            if c > prev:
                bounds.append((prev, c))
            prev = c + 1
        if prev < W_in:
            bounds.append((prev, W_in))

        bounds = [(a, b) for a, b in bounds if b - a > 0]
        if len(bounds) >= 2:
            widths = [b - a for a, b in bounds]
            if len(set(widths)) == 1:
                for a, b in bounds:
                    panels.append(grid[:, a:b])

    if not panels:
        panels = [grid]

    # -------------------------------------------------------------------------
    # 2) Per-panel background/object masks.
    # -------------------------------------------------------------------------
    infos = []
    for p in panels:
        vals, counts = np.unique(p, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        obj = None
        best_cnt = -1
        for v, ct in zip(vals, counts):
            if int(v) != bg and int(ct) > best_cnt:
                obj = int(v)
                best_cnt = int(ct)

        if obj is None:
            mask = np.zeros(p.shape, dtype=bool)
        else:
            mask = (p == obj)

        infos.append({
            "bg": bg,
            "obj": obj if obj is not None else bg,
            "mask": mask,
        })

    if not infos:
        return np.full_like(grid, global_bg, dtype=int)

    # Choose active panel: fewest object cells, rightmost on ties.
    nonempty = [i for i, info in enumerate(infos) if int(info["mask"].sum()) > 0]
    if not nonempty:
        idx = len(infos) - 1
        return np.full(infos[idx]["mask"].shape, infos[idx]["bg"], dtype=int)

    min_objects = min(int(infos[i]["mask"].sum()) for i in nonempty)
    selected_idx = max(i for i in nonempty if int(infos[i]["mask"].sum()) == min_objects)

    H, W = infos[selected_idx]["mask"].shape

    # -------------------------------------------------------------------------
    # 3) Shape definitions.
    #    Each shape has canonical coordinates, tip direction, and tip cell.
    # -------------------------------------------------------------------------
    def normalize(coords):
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        return tuple(sorted((r - min_r, c - min_c) for r, c in coords))

    def rotate90(coords):
        return [(c, -r) for r, c in coords]

    def reflect(coords):
        return [(r, -c) for r, c in coords]

    def all_transforms(coords):
        seen = set()
        out = []
        cur = list(coords)
        for _ in range(4):
            for variant in (cur, reflect(cur)):
                n = normalize(variant)
                if n not in seen:
                    seen.add(n)
                    out.append(list(n))
            cur = rotate90(cur)
        return out

    shape_defs = [
        {
            "name": "T",
            "coords": [(0, 0), (0, 1), (0, 2), (1, 1)],
            "tip_dir": "down",
            "tip": (1, 1),
            "priority": 0,
            "transforms": all_transforms([(0, 0), (0, 1), (0, 2), (1, 1)]),
        },
        {
            "name": "V",
            "coords": [(0, 1), (1, 0), (1, 2)],
            "tip_dir": "up",
            "tip": (0, 1),
            "priority": 1,
            "transforms": all_transforms([(0, 1), (1, 0), (1, 2)]),
        },
        {
            "name": "Arrow",
            "coords": [(0, 1), (1, 1), (2, 0), (2, 2)],
            "tip_dir": "up",
            "tip": (0, 1),
            "priority": 2,
            "transforms": all_transforms([(0, 1), (1, 1), (2, 0), (2, 2)]),
        },
    ]

    def score_min_count(coords):
        n = len(coords)
        if n <= 3:
            return n
        return max(2, n - 1)

    def core_min_count(coords, shape_name):
        n = len(coords)
        if n <= 3:
            return n
        # Allow one missing cell only for the arrow, where examples are noisy.
        if shape_name == "Arrow":
            return max(2, n - 1)
        return n

    def find_occurrences(mask, coords, min_cnt):
        h_mask, w_mask = mask.shape
        hr = max(r for r, _ in coords) + 1
        hc = max(c for _, c in coords) + 1
        occs = []

        if h_mask < hr or w_mask < hc:
            return occs

        for r in range(h_mask - hr + 1):
            for c in range(w_mask - hc + 1):
                cnt = 0
                for dr, dc in coords:
                    if mask[r + dr, c + dc]:
                        cnt += 1

                if cnt >= min_cnt:
                    sub = mask[r:r + hr, c:c + hc]
                    occs.append({
                        "r": r,
                        "c": c,
                        "hr": hr,
                        "hc": hc,
                        "cnt": cnt,
                        "support": int(sub.sum()),
                    })
        return occs

    def select_shape_from_mask(mask):
        best = None

        for sdef in shape_defs:
            best_occ = None
            best_occ_key = None

            for coords in sdef["transforms"]:
                occs = find_occurrences(mask, coords, score_min_count(coords))
                for occ in occs:
                    key = (occ["support"], occ["cnt"], occ["hr"], occ["hc"])
                    if best_occ_key is None or key > best_occ_key:
                        best_occ_key = key
                        best_occ = occ

            if best_occ is not None:
                shape_key = (
                    best_occ["support"],
                    best_occ["cnt"],
                    best_occ["hr"],
                    best_occ["hc"],
                    sdef["priority"],
                )
                if best is None or shape_key > best["shape_key"]:
                    best = {
                        "shape_key": shape_key,
                        "def": sdef,
                        "best_occ": best_occ,
                    }

        return best

    best_shape_info = select_shape_from_mask(infos[selected_idx]["mask"])

    # Fallback: if the selected panel has no recognizable shape, try other panels.
    if best_shape_info is None:
        for i, info in enumerate(infos):
            if i == selected_idx:
                continue
            if info["mask"].shape != (H, W):
                continue
            cand = select_shape_from_mask(info["mask"])
            if cand is not None:
                best_shape_info = cand
                break

    # If nothing recognizable is found, return a clean background panel.
    if best_shape_info is None:
        return np.full((H, W), infos[selected_idx]["bg"], dtype=int)

    sdef = best_shape_info["def"]
    canon = sdef["coords"]
    tip_dir = sdef["tip_dir"]
    tip_r, tip_c = sdef["tip"]

    shape_h = max(r for r, _ in canon) + 1
    shape_w = max(c for _, c in canon) + 1

    if H < shape_h or W < shape_w:
        return np.full((H, W), infos[selected_idx]["bg"], dtype=int)

    # -------------------------------------------------------------------------
    # 4) Build union object mask and core mask across all compatible panels.
    # -------------------------------------------------------------------------
    object_union = np.zeros((H, W), dtype=bool)
    for info in infos:
        if info["mask"].shape == (H, W):
            object_union |= info["mask"]

    core_union = np.zeros((H, W), dtype=bool)
    for info in infos:
        if info["mask"].shape != (H, W):
            continue

        for coords in sdef["transforms"]:
            min_core = core_min_count(coords, sdef["name"])
            for occ in find_occurrences(info["mask"], coords, min_core):
                r0, c0 = occ["r"], occ["c"]
                r1, c1 = r0 + occ["hr"], c0 + occ["hc"]
                core_union[r0:r1, c0:c1] |= info["mask"][r0:r1, c0:c1]

    pure = object_union & ~core_union
    core_coords = np.argwhere(core_union)

    # -------------------------------------------------------------------------
    # 5) Choose anchor from pure extreme markers.
    # -------------------------------------------------------------------------
    candidates = []
    for r, c in np.argwhere(pure):
        top = int(r) - tip_r
        left = int(c) - tip_c

        if top < 0 or left < 0:
            continue
        if top + shape_h > H or left + shape_w > W:
            continue

        if core_coords.size == 0:
            dist = 10**9
        else:
            dr = np.abs(core_coords[:, 0] - int(r))
            dc = np.abs(core_coords[:, 1] - int(c))
            dist = int(np.maximum(dr, dc).min())

        candidates.append((int(r), int(c), dist))

    if candidates:
        if tip_dir == "up":
            candidates.sort(key=lambda x: (x[0], x[2], x[1]))
        else:
            candidates.sort(key=lambda x: (-x[0], x[2], x[1]))
        anchor_r, anchor_c, _ = candidates[0]
    else:
        # Fallback: use the best detected occurrence's column and extreme row.
        occ = best_shape_info["best_occ"]
        r0, c0 = occ["r"], occ["c"]
        hr, hc = occ["hr"], occ["hc"]

        anchor_c = c0 + hc // 2
        anchor_c = max(tip_c, min(W - 1 - tip_c, anchor_c))

        col_cells = np.where(object_union[:, anchor_c])[0]
        if tip_dir == "up":
            anchor_r = int(col_cells.min()) if col_cells.size else max(0, r0)
        else:
            anchor_r = int(col_cells.max()) if col_cells.size else min(H - 1, r0 + hr - 1)

        # Ensure vertical fit.
        if anchor_r - tip_r < 0:
            anchor_r = tip_r
        if anchor_r - tip_r + shape_h > H:
            anchor_r = H - shape_h + tip_r

    # -------------------------------------------------------------------------
    # 6) Draw the canonical shape at the anchor.
    # -------------------------------------------------------------------------
    out = np.full((H, W), infos[selected_idx]["bg"], dtype=int)
    obj_color = infos[selected_idx]["obj"]

    dr = anchor_r - tip_r
    dc = anchor_c - tip_c

    for r, c in canon:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            out[nr, nc] = obj_color

    return out