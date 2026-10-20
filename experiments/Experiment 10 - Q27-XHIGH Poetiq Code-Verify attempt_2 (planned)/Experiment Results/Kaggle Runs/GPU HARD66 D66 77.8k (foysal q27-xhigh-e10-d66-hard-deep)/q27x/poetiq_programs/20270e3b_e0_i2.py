import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Combine two 4-colored objects indicated by 7-colored markers.

    The target object stays in place. The source object is translated so that
    the 4-cells adjacent to its 7-marker align with the target's 7-marker.
    All 7 pixels are removed, the target marker becomes 4, and the result is
    cropped to the bounding box of the final 4 pixels.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    OBJ_COLOR = 4
    MARK_COLOR = 7

    UP, DOWN, LEFT, RIGHT = (-1, 0), (1, 0), (0, -1), (0, 1)
    DIRS = (UP, DOWN, LEFT, RIGHT)

    # Background color: most frequent color that is neither object nor marker.
    non = g[(g != OBJ_COLOR) & (g != MARK_COLOR)]
    if non.size:
        vals, counts = np.unique(non, return_counts=True)
        bg = int(vals[np.argmax(counts)])
    else:
        bg = 0

    def components(color: int):
        """Return 4-connected components of a color as a list of cell sets."""
        mask = (g == color)
        seen = np.zeros((H, W), dtype=bool)
        comps = []

        for i in range(H):
            for j in range(W):
                if mask[i, j] and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    cells = set()

                    while stack:
                        r, c = stack.pop()
                        cells.add((r, c))

                        for dr, dc in DIRS:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))

                    comps.append(cells)

        return comps

    obj_comps = components(OBJ_COLOR)
    marker_comps = components(MARK_COLOR)

    def build_output(cells):
        """Crop to bounding box of given cells and paint them OBJ_COLOR."""
        if not cells:
            return np.full((H, W), bg, dtype=int)

        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)

        out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
        for r, c in cells:
            out[r - min_r, c - min_c] = OBJ_COLOR
        return out

    # Degenerate case: not enough structure. Remove markers and crop objects.
    if not obj_comps or not marker_comps:
        cells = set()
        for comp in obj_comps:
            cells.update(comp)
        for comp in marker_comps:
            cells.update(comp)
        return build_output(cells)

    # Map each 4-cell to its component id.
    obj_id = np.full((H, W), -1, dtype=int)
    for idx, comp in enumerate(obj_comps):
        for r, c in comp:
            obj_id[r, c] = idx

    def centroid(cells):
        if not cells:
            return (0.0, 0.0)
        rs = sum(r for r, _ in cells)
        cs = sum(c for _, c in cells)
        return (rs / len(cells), cs / len(cells))

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def closest_adj_cells(marker_cells, obj_cells):
        """Fallback: object cells closest to the marker component."""
        if not marker_cells or not obj_cells:
            return set(obj_cells) if obj_cells else set()

        min_d = 10**9
        for r, c in marker_cells:
            for rr, cc in obj_cells:
                d = abs(r - rr) + abs(c - cc)
                if d < min_d:
                    min_d = d

        if min_d == 10**9:
            return set()

        if min_d == 0:
            return set(obj_cells)

        result = set()
        for r, c in obj_cells:
            best = 10**9
            for rr, cc in marker_cells:
                d = abs(r - rr) + abs(c - cc)
                if d < best:
                    best = d
            if best == min_d:
                result.add((r, c))
        return result

    def make_piece(obj_cells, marker_cells, adj_cells):
        if not obj_cells:
            return None

        min_r = min(r for r, _ in obj_cells)
        max_r = max(r for r, _ in obj_cells)
        min_c = min(c for _, c in obj_cells)
        max_c = max(c for _, c in obj_cells)

        return {
            "obj": set(obj_cells),
            "marker": set(marker_cells),
            "adj": set(adj_cells),
            "size": len(obj_cells),
            "min_r": min_r,
            "max_r": max_r,
            "min_c": min_c,
            "max_c": max_c,
        }

    # Associate each marker component with one object component.
    best_for_obj = {}

    for mcomp in marker_comps:
        mset = set(mcomp)
        mcent = centroid(mset)

        counts = {}
        adj = {}

        for r, c in mset:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == OBJ_COLOR:
                    oid = int(obj_id[nr, nc])
                    if oid < 0:
                        continue
                    counts[oid] = counts.get(oid, 0) + 1
                    adj.setdefault(oid, set()).add((nr, nc))

        # If no direct adjacency, use the nearest object component.
        if not counts:
            best_oid = None
            best_dist = 10**9
            for oid, ocomp in enumerate(obj_comps):
                d = min(abs(r - rr) + abs(c - cc) for r, c in mset for rr, cc in ocomp)
                if d < best_dist:
                    best_dist = d
                    best_oid = oid

            if best_oid is not None:
                counts = {best_oid: 0}
                adj = {best_oid: set()}

        if not counts:
            continue

        # Choose the object this marker belongs to.
        oid = max(
            counts.keys(),
            key=lambda o: (
                counts[o],
                len(obj_comps[o]),
                -manhattan(mcent, centroid(obj_comps[o]))
            )
        )

        adj_cells = adj.get(oid, set())
        dist = manhattan(mcent, centroid(obj_comps[oid]))
        key = (counts[oid], len(adj_cells), -dist)

        if oid not in best_for_obj or key > best_for_obj[oid][0]:
            best_for_obj[oid] = (key, mset, adj_cells)

    pieces = []
    for oid, (key, mset, adj_cells) in best_for_obj.items():
        ocomp = obj_comps[oid]
        if not adj_cells:
            adj_cells = closest_adj_cells(mset, ocomp)

        piece = make_piece(ocomp, mset, adj_cells)
        if piece is not None:
            pieces.append(piece)

    # If we could not build two pieces, do a safe fallback.
    if len(pieces) < 2:
        cells = set()
        for comp in obj_comps:
            cells.update(comp)
        for comp in marker_comps:
            cells.update(comp)
        return build_output(cells)

    # If there are more than two marked pieces, keep the two largest.
    if len(pieces) > 2:
        pieces.sort(key=lambda p: (-p["size"], p["min_r"], p["min_c"]))
        pieces = pieces[:2]

    def dominant_direction(piece):
        """
        Determine the main direction from the marker to its object.
        Returns None if no direction is clearly dominant.
        """
        counts = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}
        obj_set = piece["obj"]

        for r, c in piece["marker"]:
            for dr, dc in DIRS:
                if (r + dr, c + dc) in obj_set:
                    counts[(dr, dc)] += 1

        total = sum(counts.values())
        if total == 0:
            return None

        maxc = max(counts.values())
        if maxc > total / 2.0:
            for d, cnt in counts.items():
                if cnt == maxc:
                    return d

        return None

    def choose_target(p1, p2):
        """
        Choose the piece that remains fixed.

        Primary semantic rule:
          - vertically opposite markers: object above its marker is target.
          - horizontally opposite markers: object left of its marker is target.
        Fallback:
          - larger object.
          - upper-left object.
        """
        d1 = dominant_direction(p1)
        d2 = dominant_direction(p2)

        if d1 == UP and d2 == DOWN:
            return p1
        if d1 == DOWN and d2 == UP:
            return p2

        if d1 == LEFT and d2 == RIGHT:
            return p1
        if d1 == RIGHT and d2 == LEFT:
            return p2

        if p1["size"] != p2["size"]:
            return p1 if p1["size"] > p2["size"] else p2

        key1 = (p1["min_r"], p1["min_c"])
        key2 = (p2["min_r"], p2["min_c"])
        if key1 != key2:
            return p1 if key1 < key2 else p2

        c1 = centroid(p1["obj"])
        c2 = centroid(p2["obj"])
        return p1 if (c1[0], c1[1]) <= (c2[0], c2[1]) else p2

    target = choose_target(pieces[0], pieces[1])
    source = pieces[1] if target is pieces[0] else pieces[0]

    def shift(cells, dy, dx):
        return {(r + dy, c + dx) for r, c in cells}

    def centroid_distance(cells, target_cells, dy, dx):
        if not cells or not target_cells:
            return 0.0
        c1 = centroid(cells)
        c2 = centroid(target_cells)
        return manhattan((c1[0] + dy, c1[1] + dx), c2)

    def eval_translation(dy, dx):
        """
        Score a translation.

        Primary alignment: source marker-adjacent cells -> target marker cells.
        Secondary alignment: source marker cells -> target marker-adjacent cells.
        """
        has_primary = bool(source["adj"] and target["marker"])
        has_secondary = bool(source["marker"] and target["adj"])

        ov_p = -1
        d_p = 0.0
        ov_s = -1
        d_s = 0.0

        if has_primary:
            ov_p = len(shift(source["adj"], dy, dx) & target["marker"])
            d_p = centroid_distance(source["adj"], target["marker"], dy, dx)

        if has_secondary:
            ov_s = len(shift(source["marker"], dy, dx) & target["adj"])
            d_s = centroid_distance(source["marker"], target["adj"], dy, dx)

        # If primary is unavailable, promote secondary to primary.
        if not has_primary and has_secondary:
            ov_p, d_p = ov_s, d_s
            ov_s = -1
            d_s = 0.0

        move = abs(dy) + abs(dx)
        return (
            ov_p,
            ov_s if ov_s != -1 else -1,
            -d_p,
            -d_s if d_s else 0.0,
            -move
        )

    # Generate candidate translations from anchor-cell correspondences.
    candidates = set()

    if source["adj"] and target["marker"]:
        for a in source["adj"]:
            for b in target["marker"]:
                candidates.add((b[0] - a[0], b[1] - a[1]))

    if source["marker"] and target["adj"]:
        for a in source["marker"]:
            for b in target["adj"]:
                candidates.add((b[0] - a[0], b[1] - a[1]))

    # Also consider centroid-based translations and their neighbors.
    for anchor_s, anchor_t in ((source["adj"], target["marker"]), (source["marker"], target["adj"])):
        if anchor_s and anchor_t:
            cs = centroid(anchor_s)
            ct = centroid(anchor_t)
            base_r = int(round(ct[0] - cs[0]))
            base_c = int(round(ct[1] - cs[1]))

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    candidates.add((base_r + dr, base_c + dc))

    if not candidates:
        candidates.add((0, 0))

    best_dy, best_dx = max(candidates, key=lambda v: eval_translation(v[0], v[1]))

    # Build final set of object pixels.
    out_cells = set(target["obj"])

    for r, c in source["obj"]:
        out_cells.add((r + best_dy, c + best_dx))

    # The target marker becomes part of the object.
    out_cells.update(target["marker"])

    return build_output(out_cells)