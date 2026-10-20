import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by moving one 4-object from one 7-marker
    to the other 7-marker, aligning the adjacent edge, and cropping
    the result to the bounding box of the resulting 4-pixels.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    marker = 7

    # Determine background and object colors.
    non_marker = g[g != marker]
    if non_marker.size == 0:
        return g.copy()

    vals, counts = np.unique(non_marker, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    non_bg = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != bg]
    if non_bg:
        obj = int(max(non_bg, key=lambda x: x[1])[0])
    elif np.any(g == 4):
        obj = 4
    else:
        obj = 0

    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    UP, DOWN, LEFT, RIGHT = (-1, 0), (1, 0), (0, -1), (0, 1)

    def flood_components(mask):
        """4-connected components in a boolean mask."""
        seen = np.zeros((H, W), dtype=bool)
        comps = []

        for i in range(H):
            for j in range(W):
                if not mask[i, j] or seen[i, j]:
                    continue

                stack = [(i, j)]
                seen[i, j] = True
                comp = []

                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))

                    for dr, dc in dirs4:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and mask[nr, nc]
                            and not seen[nr, nc]
                        ):
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                comps.append(comp)

        return comps

    def merge_closest(comps):
        """If there are more than two marker components, merge closest ones."""
        comps = [set(comp) for comp in comps]

        while len(comps) > 2:
            best_i, best_j = -1, -1
            best_d = float("inf")

            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    min_d = float("inf")
                    for r1, c1 in comps[i]:
                        for r2, c2 in comps[j]:
                            d = abs(r1 - r2) + abs(c1 - c2)
                            if d < min_d:
                                min_d = d
                    if min_d < best_d:
                        best_d = min_d
                        best_i, best_j = i, j

            if best_i == -1:
                break

            comps[best_i].update(comps[best_j])
            del comps[best_j]

        return [list(comp) for comp in comps]

    def get_object(marker_cells):
        """
        For a marker component, return the set of 4-cells connected to it
        when only this marker component is temporarily treated as 4.
        """
        if not marker_cells:
            return set()

        temp = (g == obj)
        for r, c in marker_cells:
            temp[r, c] = True

        start = marker_cells[0]
        seen = np.zeros((H, W), dtype=bool)
        stack = [start]
        seen[start[0], start[1]] = True

        obj_set = set()

        while stack:
            r, c = stack.pop()
            if g[r, c] == obj:
                obj_set.add((r, c))

            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and temp[nr, nc]
                    and not seen[nr, nc]
                ):
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        return obj_set

    def centroid(cells):
        if not cells:
            return 0.0, 0.0
        rs = sum(r for r, _ in cells)
        cs = sum(c for _, c in cells)
        n = len(cells)
        return rs / n, cs / n

    def get_direction(marker_cells, obj_set):
        """
        Determine the direction from the marker to its associated object.
        Adjacent cells are prioritized, then total object mass in that direction.
        """
        if not obj_set:
            return UP

        mr, mc = centroid(marker_cells)
        best_dir = UP
        best_score = -1

        for d in dirs4:
            dr, dc = d

            adj_count = 0
            for r, c in marker_cells:
                if (r + dr, c + dc) in obj_set:
                    adj_count += 1

            mass = 0
            for r, c in obj_set:
                if (r - mr) * dr + (c - mc) * dc > 0:
                    mass += 1

            score = adj_count * 1000 + mass
            if score > best_score:
                best_score = score
                best_dir = d

        return best_dir

    def nearest_int(x):
        return int(round(float(x)))

    # Find marker components.
    marker_mask = (g == marker)
    marker_comps = flood_components(marker_mask)

    if not marker_comps:
        return g.copy()

    if len(marker_comps) > 2:
        marker_comps = merge_closest(marker_comps)

    # Degenerate fallback: only one marker component.
    if len(marker_comps) == 1:
        out = g.copy()
        for r, c in marker_comps[0]:
            out[r, c] = obj

        cells = np.argwhere(out == obj)
        if cells.size == 0:
            return out

        min_r, min_c = cells.min(axis=0)
        max_r, max_c = cells.max(axis=0)
        return out[min_r:max_r + 1, min_c:max_c + 1].copy()

    if len(marker_comps) > 2:
        marker_comps.sort(key=len, reverse=True)
        marker_comps = marker_comps[:2]

    # Analyze the two marker/object pairs.
    infos = []
    for comp in marker_comps:
        obj_set = get_object(comp)
        direction = get_direction(comp, obj_set)
        infos.append((comp, obj_set, direction))

    comp0, obj0, dir0 = infos[0]
    comp1, obj1, dir1 = infos[1]

    # Choose target/source.
    if dir0 == UP and dir1 == DOWN:
        target, source = 0, 1
    elif dir0 == DOWN and dir1 == UP:
        target, source = 1, 0
    elif dir0 == LEFT and dir1 == RIGHT:
        target, source = 0, 1
    elif dir0 == RIGHT and dir1 == LEFT:
        target, source = 1, 0
    else:
        size0, size1 = len(obj0), len(obj1)
        if size0 != size1:
            target = 0 if size0 > size1 else 1
        else:
            c0 = centroid(comp0)
            c1 = centroid(comp1)
            target = 0 if (c0[0], c0[1]) < (c1[0], c1[1]) else 1
        source = 1 - target

    src_comp, src_obj, src_dir = infos[source]
    tgt_comp, tgt_obj, _ = infos[target]

    # Translation vector: align source marker's adjacent edge to target marker.
    src_r, src_c = centroid(src_comp)
    anchor_r, anchor_c = src_r + src_dir[0], src_c + src_dir[1]
    tgt_r, tgt_c = centroid(tgt_comp)

    dy = nearest_int(tgt_r - anchor_r)
    dx = nearest_int(tgt_c - anchor_c)

    # Build result cell set.
    all_obj_cells = set()
    for r, c in np.argwhere(g == obj):
        all_obj_cells.add((int(r), int(c)))

    # Keep everything except original source object, but preserve target object.
    base_cells = all_obj_cells - src_obj
    base_cells |= tgt_obj

    # Add translated source object.
    for r, c in src_obj:
        base_cells.add((r + dy, c + dx))

    # Fill target marker cells.
    for r, c in tgt_comp:
        base_cells.add((r, c))

    if not base_cells:
        return np.full((H, W), bg, dtype=int)

    min_r = min(r for r, _ in base_cells)
    max_r = max(r for r, _ in base_cells)
    min_c = min(c for _, c in base_cells)
    max_c = max(c for _, c in base_cells)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for r, c in base_cells:
        out[r - min_r, c - min_c] = obj

    return out