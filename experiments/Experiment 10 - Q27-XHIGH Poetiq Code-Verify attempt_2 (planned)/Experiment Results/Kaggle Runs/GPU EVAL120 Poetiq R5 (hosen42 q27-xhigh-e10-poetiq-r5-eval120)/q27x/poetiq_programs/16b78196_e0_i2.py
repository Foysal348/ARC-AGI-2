import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule (inferred from the examples):

    1. Find the large "wall".  It is the biggest connected component that
       touches two opposite borders.  It is either horizontal or vertical.
    2. All other colored components are pieces.  For a horizontal wall, pieces
       are grouped by their width; for a vertical wall, by their height.
    3. In each group the pieces keep their original relative side order
       (above/left first, then below/right), and within a side they are ordered
       by color.  The group is packed into a tight stack along the perpendicular
       axis, with a common coordinate along the parallel axis.
    4. For each group, try both sides of the wall and all parallel offsets.
       The chosen candidate maximizes:
           - number of original wall gaps/zeros directly covered by pieces,
           - then minimal total movement of the pieces.
       Different width/height groups are placed with a small beam search so
       that their final cells do not overlap.
    5. Finally, wall gaps that are 4-connected to a placed piece are filled
       with that piece's color, while the wall itself is preserved.
    """
    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def _components(mask):
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
                        for dr, dc in dirs:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    comps.append(pix)
        return comps

    def _component_mask(pixels):
        m = np.zeros((H, W), dtype=bool)
        for r, c in pixels:
            m[r, c] = True
        return m

    def _find_wall():
        best = None
        best_score = -1

        for color in np.unique(grid):
            if color == 0:
                continue
            color_mask = (grid == color)
            for comp in _components(color_mask):
                rows = [p[0] for p in comp]
                cols = [p[1] for p in comp]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                touch_lr = (c0 == 0 and c1 == W - 1)
                touch_tb = (r0 == 0 and r1 == H - 1)

                if touch_lr or touch_tb:
                    area = len(comp)
                    span = max(r1 - r0 + 1, c1 - c0 + 1)
                    score = 2 * area + span
                    if score > best_score:
                        wall_mask = _component_mask(comp)
                        best = (color, wall_mask, (r0, c0, r1, c1), touch_lr, touch_tb)
                        best_score = score

        if best is not None:
            color, wall_mask, bbox, touch_lr, touch_tb = best
            r0, c0, r1, c1 = bbox
            if touch_lr and not touch_tb:
                horizontal = True
            elif touch_tb and not touch_lr:
                horizontal = False
            else:
                horizontal = (c1 - c0 + 1) >= (r1 - r0 + 1)
            return color, wall_mask, bbox, horizontal

        # Fallback: if no component perfectly touches opposite borders,
        # use the largest non-background color whose bounding box spans one axis.
        best = None
        best_score = -1
        for color in np.unique(grid):
            if color == 0:
                continue
            mask = (grid == color)
            area = int(mask.sum())
            if area == 0:
                continue
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue
            r0 = int(np.where(rows)[0][0])
            r1 = int(np.where(rows)[0][-1])
            c0 = int(np.where(cols)[0][0])
            c1 = int(np.where(cols)[0][-1])
            hspan = r1 - r0 + 1
            wspan = c1 - c0 + 1

            if wspan >= int(0.7 * W) and hspan <= int(0.8 * H):
                horizontal = True
                score = area
            elif hspan >= int(0.7 * H) and wspan <= int(0.8 * W):
                horizontal = False
                score = area
            else:
                continue

            if score > best_score:
                best_score = score
                best = (color, mask, (r0, c0, r1, c1), horizontal)

        if best is not None:
            return best
        return None, np.zeros((H, W), dtype=bool), (0, 0, H - 1, W - 1), True

    wall_color, wall_mask, wall_bbox, horizontal = _find_wall()
    if wall_color is None:
        return grid

    r0w, c0w, r1w, c1w = wall_bbox

    if horizontal:
        par_len = W
        perp_len = H
        wall_min = r0w
        wall_max = r1w
        wall_center = (r0w + r1w) / 2.0
    else:
        par_len = H
        perp_len = W
        wall_min = c0w
        wall_max = c1w
        wall_center = (c0w + c1w) / 2.0

    wall_bbox_mask = np.zeros((H, W), dtype=bool)
    wall_bbox_mask[r0w:r1w + 1, c0w:c1w + 1] = True
    wall_holes = (grid == 0) & wall_bbox_mask & (~wall_mask)

    def _object_side(r0, r1, c0, c1):
        if horizontal:
            center = (r0 + r1) / 2.0
            return -1 if center <= wall_center else 1
        else:
            center = (c0 + c1) / 2.0
            return -1 if center <= wall_center else 1

    objects = []
    for color in np.unique(grid):
        if color == 0 or color == wall_color:
            continue
        color_mask = (grid == color) & (~wall_mask)
        for comp in _components(color_mask):
            if not comp:
                continue
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = _component_mask(comp)

            if horizontal:
                p0 = c0
                q0 = r0
                par_dim = c1 - c0 + 1
                perp_extent = r1 - r0 + 1
            else:
                p0 = r0
                q0 = c0
                par_dim = r1 - r0 + 1
                perp_extent = c1 - c0 + 1

            objects.append({
                "color": int(color),
                "mask": mask,
                "p0": p0,
                "q0": q0,
                "par_dim": par_dim,
                "perp_extent": perp_extent,
                "side": _object_side(r0, r1, c0, c1),
            })

    if not objects:
        out = grid.copy()
        out[(grid > 0) & (~wall_mask)] = 0
        return out

    groups = defaultdict(list)
    for obj in objects:
        groups[obj["par_dim"]].append(obj)

    def _can_place(mask, q, p, occ):
        h, w = mask.shape
        if horizontal:
            # q = row, p = col
            if q < 0 or p < 0 or q + h > H or p + w > W:
                return False
            if np.any(wall_mask[q:q + h, p:p + w] & mask):
                return False
            if occ is not None and np.any(occ[q:q + h, p:p + w] & mask):
                return False
        else:
            # p = row, q = col
            if p < 0 or q < 0 or p + h > H or q + w > W:
                return False
            if np.any(wall_mask[p:p + h, q:q + w] & mask):
                return False
            if occ is not None and np.any(occ[p:p + h, q:q + w] & mask):
                return False
        return True

    def _add_to_occ(occ, mask, q, p):
        h, w = mask.shape
        if horizontal:
            occ[q:q + h, p:p + w] |= mask
        else:
            occ[p:p + h, q:q + w] |= mask

    def _generate_candidates(objs_sorted, group_dim, use_side_constraint=True):
        cands = []
        n = len(objs_sorted)
        max_p = par_len - group_dim

        for side in (-1, 1):
            for p in range(0, max_p + 1):
                occ = np.zeros((H, W), dtype=bool)
                positions = [None] * n
                valid = True

                if side == -1:
                    # Pack from the wall-adjacent end backwards.
                    for i in range(n - 1, -1, -1):
                        m = objs_sorted[i]["mask"]
                        pe = objs_sorted[i]["perp_extent"]
                        max_q = perp_len - pe
                        start_q = wall_max if i == n - 1 else positions[i + 1]
                        if start_q > max_q:
                            start_q = max_q

                        found = None
                        for q in range(start_q, -1, -1):
                            if use_side_constraint:
                                center = q + (pe - 1) / 2.0
                                if center > wall_center + 1e-9:
                                    continue
                            if _can_place(m, q, p, occ):
                                found = q
                                break

                        if found is None:
                            valid = False
                            break

                        positions[i] = found
                        _add_to_occ(occ, m, found, p)

                else:
                    # Pack from the wall-adjacent end forwards.
                    for i in range(n):
                        m = objs_sorted[i]["mask"]
                        pe = objs_sorted[i]["perp_extent"]
                        max_q = perp_len - pe
                        start_q = wall_min if i == 0 else positions[i - 1]
                        if start_q < 0:
                            start_q = 0

                        found = None
                        for q in range(start_q, max_q + 1):
                            if use_side_constraint:
                                center = q + (pe - 1) / 2.0
                                if center < wall_center - 1e-9:
                                    continue
                            if _can_place(m, q, p, occ):
                                found = q
                                break

                        if found is None:
                            valid = False
                            break

                        positions[i] = found
                        _add_to_occ(occ, m, found, p)

                if not valid:
                    continue

                coverage = int(np.sum(occ & wall_holes))
                movement = 0
                placements = []
                for i, o in enumerate(objs_sorted):
                    q = positions[i]
                    movement += abs(q - o["q0"]) + abs(p - o["p0"])
                    placements.append((o["color"], o["mask"], q, p))

                centroid_p = sum(o["p0"] for o in objs_sorted) / len(objs_sorted)
                extra = -abs(p - centroid_p)
                cands.append({
                    "mask": occ.copy(),
                    "coverage": coverage,
                    "movement": movement,
                    "extra": extra,
                    "placements": placements,
                    "p": p,
                })

        return cands

    def _original_candidate(objs_sorted):
        occ = np.zeros((H, W), dtype=bool)
        placements = []
        for o in objs_sorted:
            q = o["q0"]
            p = o["p0"]
            if not _can_place(o["mask"], q, p, occ):
                return None
            _add_to_occ(occ, o["mask"], q, p)
            placements.append((o["color"], o["mask"], q, p))

        coverage = int(np.sum(occ & wall_holes))
        centroid_p = sum(o["p0"] for o in objs_sorted) / len(objs_sorted)
        return {
            "mask": occ.copy(),
            "coverage": coverage,
            "movement": 0,
            "extra": 0,
            "placements": placements,
            "p": int(round(centroid_p)),
        }

    group_data = []
    for dim, objs in groups.items():
        objs_sorted = sorted(
            objs,
            key=lambda x: (x["side"], x["color"], x["q0"], x["p0"])
        )

        cands = _generate_candidates(objs_sorted, dim, True)
        if not cands:
            cands = _generate_candidates(objs_sorted, dim, False)
        if not cands:
            oc = _original_candidate(objs_sorted)
            if oc is not None:
                cands = [oc]

        if cands:
            centroid_p = sum(o["p0"] for o in objs_sorted) / len(objs_sorted)
            max_cov = max(c["coverage"] for c in cands)
            group_data.append({
                "dim": dim,
                "objects": objs_sorted,
                "cands": cands,
                "centroid_p": centroid_p,
                "max_cov": max_cov,
            })

    def _beam_search():
        if not group_data:
            return []

        # Place higher-value / larger groups first.
        order = sorted(
            range(len(group_data)),
            key=lambda i: (
                -group_data[i]["max_cov"],
                -len(group_data[i]["objects"]),
                -group_data[i]["dim"],
                group_data[i]["centroid_p"],
            )
        )

        states = [(np.zeros((H, W), dtype=bool), 0, 0, 0, [])]
        beam_width = 100

        for gi in order:
            group = group_data[gi]
            cands_sorted = sorted(
                group["cands"],
                key=lambda c: (c["coverage"], -c["movement"], c["extra"]),
                reverse=True
            )

            new_states = []
            for occ, cov, mov, extra, placed in states:
                for cand in cands_sorted:
                    if np.any(cand["mask"] & occ):
                        continue
                    new_occ = occ | cand["mask"]
                    new_cov = cov + cand["coverage"]
                    new_mov = mov + cand["movement"]
                    new_extra = extra + cand["extra"]
                    new_placed = placed + [cand["placements"]]
                    new_states.append((new_occ, new_cov, new_mov, new_extra, new_placed))

            if not new_states:
                return None

            new_states.sort(
                key=lambda s: (s[1], -s[2], s[3]),
                reverse=True
            )
            states = new_states[:beam_width]

        states.sort(key=lambda s: (s[1], -s[2], s[3]), reverse=True)
        return states[0][4]

    def _fallback_search():
        order = sorted(
            range(len(group_data)),
            key=lambda i: group_data[i]["centroid_p"]
        )
        occ = np.zeros((H, W), dtype=bool)
        placements = []

        for gi in order:
            group = group_data[gi]
            cands_sorted = sorted(
                group["cands"],
                key=lambda c: (c["coverage"], -c["movement"], c["extra"]),
                reverse=True
            )
            chosen = None
            for cand in cands_sorted:
                if not np.any(cand["mask"] & occ):
                    chosen = cand
                    break
            if chosen is None:
                chosen = cands_sorted[0]
            occ = occ | chosen["mask"]
            placements.append(chosen["placements"])

        return placements

    best_placements = _beam_search()
    if best_placements is None:
        best_placements = _fallback_search()

    out = grid.copy()
    out[(grid > 0) & (~wall_mask)] = 0

    for placements in best_placements:
        for color, mask, q, p in placements:
            h, w = mask.shape
            if horizontal:
                r0, r1 = q, q + h
                c0, c1 = p, p + w
            else:
                r0, r1 = p, p + h
                c0, c1 = q, q + w

            if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                continue

            sub = out[r0:r1, c0:c1]
            write = mask & (~wall_mask[r0:r1, c0:c1])
            sub = np.where(write, color, sub)
            out[r0:r1, c0:c1] = sub

    # Fill connected wall gaps from placed pieces.
    source_mask = (out > 0) & (~wall_mask)
    dist = -np.ones((H, W), dtype=int)
    q = deque()

    for r, c in np.argwhere(source_mask):
        dist[r, c] = 0
        q.append((int(r), int(c), int(out[r, c])))

    while q:
        r, c, col = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < H and 0 <= nc < W
                and dist[nr, nc] == -1
                and wall_holes[nr, nc]
                and out[nr, nc] == 0
            ):
                dist[nr, nc] = dist[r, c] + 1
                out[nr, nc] = col
                q.append((nr, nc, col))

    return out