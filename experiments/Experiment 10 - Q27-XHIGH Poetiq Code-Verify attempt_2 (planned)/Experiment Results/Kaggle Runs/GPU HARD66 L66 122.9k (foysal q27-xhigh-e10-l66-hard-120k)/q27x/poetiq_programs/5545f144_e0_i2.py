import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the panel-based template extraction task.

    The routine:
      1. Splits the grid into equal-width panels using full non-background
         separator columns, if such columns exist.
      2. Chooses the active panel as the panel with the most object pixels.
      3. Detects the best matching noisy template among:
           - T, output upright with stem pointing down
           - V, output upright with tip pointing up
           - Arrow, output upright with tip pointing up
      4. Marks the best matched template occurrence as the core.
      5. Uses the outermost non-core object pixel in the active panel as the
         anchor tip: topmost for up-pointing templates, bottommost for the
         down-pointing template.
      6. Returns the active panel size filled with the background color, with
         the clean upright template drawn from the anchor tip.
    """
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H_in, W_in = grid.shape
    if H_in == 0 or W_in == 0:
        return grid.copy()

    # ------------------------------------------------------------------
    # Small geometry helpers for template transformations.
    # ------------------------------------------------------------------
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
        cur = normalize(list(coords))
        for _ in range(4):
            for variant in (cur, normalize(reflect(list(cur)))):
                if variant not in seen:
                    seen.add(variant)
                    out.append(list(variant))
            cur = normalize(rotate90(list(cur)))
        return out

    # ------------------------------------------------------------------
    # Template definitions.
    #
    # Canonical coordinates are the upright output shape.
    # tip is the cell that will be placed at the chosen anchor.
    # tip_dir controls whether the anchor is the topmost or bottommost
    # pure object pixel.
    # Priority is used only to break near-ambiguous template matches.
    # ------------------------------------------------------------------
    shapes = []

    def add_shape(name, coords, tip, tip_dir, priority):
        canon = normalize(list(coords))
        trs = []
        for tr in all_transforms(canon):
            hr = max(r for r, _ in tr) + 1
            hc = max(c for _, c in tr) + 1
            trs.append({"coords": tuple(tr), "hr": hr, "hc": hc})
        shapes.append({
            "name": name,
            "coords": canon,
            "tip": tip,
            "tip_dir": tip_dir,
            "priority": priority,
            "size": len(canon),
            "transforms": trs,
        })

    add_shape("T", [(0, 0), (0, 1), (0, 2), (1, 1)], (1, 1), "down", 1)
    add_shape("V", [(0, 1), (1, 0), (1, 2)], (0, 1), "up", 2)
    add_shape("Arrow", [(0, 1), (1, 1), (2, 0), (2, 2)], (0, 1), "up", 3)

    # ------------------------------------------------------------------
    # Global background and optional vertical panel separation.
    # ------------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    global_bg = int(vals[np.argmax(counts)])

    sep_cols = []
    for c in range(W_in):
        col = grid[:, c]
        if col.size and np.all(col == col[0]) and int(col[0]) != global_bg:
            sep_cols.append(c)

    panels = [grid]
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
            if all(w == widths[0] for w in widths):
                panels = [grid[:, a:b] for a, b in bounds]

    # ------------------------------------------------------------------
    # Per-panel background/object extraction.
    # ------------------------------------------------------------------
    def analyze_panel(panel):
        vals, counts = np.unique(panel, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        obj = None
        best = -1
        for v, ct in zip(vals, counts):
            v = int(v)
            if v != bg and ct > best:
                obj = v
                best = ct

        if obj is None:
            mask = np.zeros(panel.shape, dtype=bool)
            return bg, bg, mask, 0

        mask = (panel == obj)
        return bg, obj, mask, int(mask.sum())

    # ------------------------------------------------------------------
    # Template matching in one binary mask.
    # ------------------------------------------------------------------
    def select_shape(mask):
        h, w = mask.shape
        best = None

        for sh in shapes:
            # V must be complete. T/Arrow may miss one pixel.
            min_cnt = sh["size"] if sh["size"] <= 3 else sh["size"] - 1

            best_occ = None
            best_occ_key = None

            for tr in sh["transforms"]:
                coords = tr["coords"]
                hr, hc = tr["hr"], tr["hc"]
                if h < hr or w < hc:
                    continue

                for r in range(h - hr + 1):
                    for c in range(w - hc + 1):
                        cnt = 0
                        for dr, dc in coords:
                            if mask[r + dr, c + dc]:
                                cnt += 1

                        if cnt >= min_cnt:
                            support = int(mask[r:r + hr, c:c + hc].sum())
                            extra = max(0, support - cnt)
                            full = 1 if cnt == sh["size"] else 0

                            # Prefer more required cells, then full matches,
                            # then fewer extra cells, then coverage.
                            occ_key = (
                                cnt,
                                full,
                                -extra,
                                support,
                                -hr,
                                -hc,
                                -r,
                                -c,
                            )

                            if best_occ_key is None or occ_key > best_occ_key:
                                best_occ_key = occ_key
                                best_occ = {
                                    "r": r,
                                    "c": c,
                                    "hr": hr,
                                    "hc": hc,
                                    "cnt": cnt,
                                    "support": support,
                                    "extra": extra,
                                    "full": full,
                                    "coords": coords,
                                }

            if best_occ is not None:
                shape_key = (
                    best_occ["full"],
                    best_occ["cnt"],
                    sh["priority"],
                    -best_occ["extra"],
                    best_occ["support"],
                    -best_occ["hr"],
                    -best_occ["hc"],
                )

                if best is None or shape_key > best["shape_key"]:
                    best = {
                        "shape": sh,
                        "occ": best_occ,
                        "shape_key": shape_key,
                    }

        return best

    # ------------------------------------------------------------------
    # Build panel information and choose active panel.
    # ------------------------------------------------------------------
    infos = []
    for i, panel in enumerate(panels):
        bg, obj, mask, count = analyze_panel(panel)
        sel = select_shape(mask) if count > 0 else None
        shape_key = sel["shape_key"] if sel is not None else (-1,) * 7
        infos.append({
            "index": i,
            "bg": bg,
            "obj": obj,
            "mask": mask,
            "count": count,
            "sel": sel,
            "shape_key": shape_key,
        })

    if not infos:
        return grid.copy()

    # Primary selection: most object pixels. Template score breaks ties.
    active_idx = max(
        range(len(infos)),
        key=lambda i: (infos[i]["count"], infos[i]["shape_key"], infos[i]["index"]),
    )
    info = infos[active_idx]

    # Fallback: if the densest panel has no recognizable template, use the
    # best template-containing panel.
    if info["sel"] is None:
        shaped = [i for i in infos if infos[i]["sel"] is not None and infos[i]["count"] > 0]
        if shaped:
            active_idx = max(
                shaped,
                key=lambda i: (infos[i]["shape_key"], infos[i]["count"], infos[i]["index"]),
            )
            info = infos[active_idx]

    H, W = info["mask"].shape
    bg = info["bg"]
    obj = info["obj"]
    mask = info["mask"]

    if info["count"] == 0 or info["sel"] is None or H == 0 or W == 0:
        return np.full((H, W), bg, dtype=int)

    sh = info["sel"]["shape"]
    occ = info["sel"]["occ"]

    canon = sh["coords"]
    tip_r, tip_c = sh["tip"]
    tip_dir = sh["tip_dir"]
    shape_h = max(r for r, _ in canon) + 1
    shape_w = max(c for _, c in canon) + 1

    if H < shape_h or W < shape_w:
        return np.full((H, W), bg, dtype=int)

    # ------------------------------------------------------------------
    # Core = cells from the best template occurrence that are present.
    # Pure = object cells not in the core, used to find the anchor marker.
    # ------------------------------------------------------------------
    core = np.zeros((H, W), dtype=bool)
    r0, c0 = occ["r"], occ["c"]
    for dr, dc in occ["coords"]:
        rr, cc = r0 + dr, c0 + dc
        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
            core[rr, cc] = True

    pure = mask & ~core

    # ------------------------------------------------------------------
    # Anchor selection.
    # ------------------------------------------------------------------
    def choose_anchor(cells_mask, allow_unfit=False):
        cells = np.argwhere(cells_mask)
        if cells.size == 0:
            return None

        core_coords = np.argwhere(core) if core.any() else np.empty((0, 2), dtype=int)
        if core_coords.size:
            center_col = float(np.mean(core_coords[:, 1]))
        else:
            center_col = float(W - 1) / 2.0

        entries = []
        for r, c in cells:
            r = int(r)
            c = int(c)

            top = r - tip_r
            left = c - tip_c
            fit = (
                0 <= top and top + shape_h <= H and
                0 <= left and left + shape_w <= W
            )

            if core_coords.size:
                dist = int(np.min(
                    np.abs(core_coords[:, 0] - r) + np.abs(core_coords[:, 1] - c)
                ))
            else:
                dist = 10**9

            col_pref = int(round(abs(c - center_col)))
            primary = r if tip_dir == "up" else -r

            sort_key = (primary, dist, col_pref, c)
            entries.append((fit, sort_key, r, c))

        if allow_unfit:
            pool = entries
            pool.sort(key=lambda e: (0 if e[0] else 1, e[1]))
        else:
            pool = [e for e in entries if e[0]]
            if not pool:
                return None
            pool.sort(key=lambda e: e[1])

        _, _, r, c = pool[0]

        top = r - tip_r
        left = c - tip_c
        top = max(0, min(top, H - shape_h))
        left = max(0, min(left, W - shape_w))

        return top + tip_r, left + tip_c

    anchor = choose_anchor(pure, allow_unfit=False)
    if anchor is None:
        anchor = choose_anchor(mask, allow_unfit=False)
    if anchor is None:
        anchor = choose_anchor(pure, allow_unfit=True)
    if anchor is None:
        anchor = choose_anchor(mask, allow_unfit=True)

    if anchor is None:
        return np.full((H, W), bg, dtype=int)

    # ------------------------------------------------------------------
    # Draw the clean upright template.
    # ------------------------------------------------------------------
    out = np.full((H, W), bg, dtype=int)
    ar, ac = anchor

    for dr, dc in canon:
        rr = ar + dr - tip_r
        cc = ac + dc - tip_c
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = obj

    return out