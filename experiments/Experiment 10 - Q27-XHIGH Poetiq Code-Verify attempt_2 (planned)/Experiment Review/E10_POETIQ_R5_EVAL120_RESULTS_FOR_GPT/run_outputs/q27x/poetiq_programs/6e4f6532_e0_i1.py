import numpy as np
from collections import deque
from itertools import permutations

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by moving each 8-based object to a matching isolated
    9-pattern target.

    The idea inferred from the examples:
    - The grid has walls (large straight colored bars) and a background color.
    - Small components containing color 8 are "objects".  They may also contain
      color 9 payload pixels and small marker pixels whose colors match wall
      colors.
    - Components made only of 9 pixels are targets.
    - Each object is transformed by one of the 8 dihedral transformations
      (rotations/reflections) so that its marker colors face the nearest wall
      of the same color at the target location.
    - The transformed object is translated so that its 9-pixel pattern exactly
      matches the target 9-pixel pattern.
    - Original objects are cleared, targets remain, and transformed objects are
      drawn back.
    """
    if grid.size == 0:
        return grid

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def label_components(mask):
        """4-connected component labeling. Returns list of component coord lists."""
        visited = np.zeros((h, w), dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not visited[r, c]:
                    q = deque([(r, c)])
                    visited[r, c] = True
                    comp = []
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    components.append(comp)
        return components

    # ------------------------------------------------------------------
    # Identify wall components.  Walls are large straight bars, usually
    # thickness 2, spanning most of a row/column.
    # ------------------------------------------------------------------
    wall_mask = np.zeros((h, w), dtype=bool)
    wall_components = {}  # color -> list of (rmin, rmax, cmin, cmax)

    for color in vals:
        color = int(color)
        if color == bg:
            continue

        color_mask = (grid == color)
        for comp in label_components(color_mask):
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            bh = rmax - rmin + 1
            bw = cmax - cmin + 1
            size = len(comp)

            horizontal_bar = (bw >= 0.6 * w and bh <= max(2, 0.35 * h))
            vertical_bar = (bh >= 0.6 * h and bw <= max(2, 0.35 * w))
            size_ok = size >= max(4, min(h, w) // 3)

            if size_ok and (horizontal_bar or vertical_bar):
                for r, c in comp:
                    wall_mask[r, c] = True
                wall_components.setdefault(color, []).append((rmin, rmax, cmin, cmax))

    # ------------------------------------------------------------------
    # Find non-wall, non-background components:
    #   - objects: contain color 8
    #   - targets: contain only color 9
    # ------------------------------------------------------------------
    interior_mask = (grid != bg) & (~wall_mask)
    interior_components = label_components(interior_mask)

    objects = []
    targets = []

    for comp in interior_components:
        has8 = False
        has9 = False
        for r, c in comp:
            v = int(grid[r, c])
            if v == 8:
                has8 = True
            elif v == 9:
                has9 = True

        if has8:
            objects.append(comp)
        elif has9:
            targets.append(comp)

    # If there is nothing to transform, return unchanged.
    if not objects or not targets:
        return grid.copy()

    # ------------------------------------------------------------------
    # Preprocess targets.
    # ------------------------------------------------------------------
    target_data = []
    all_target_9 = set()
    for comp in targets:
        coords = [(r, c) for r, c in comp]
        rs = [r for r, c in coords]
        cs = [c for r, c in coords]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        cent_r = sum(rs) / len(rs)
        cent_c = sum(cs) / len(cs)
        target_data.append({
            "coords": coords,
            "set": set(coords),
            "bbox": (rmin, rmax, cmin, cmax),
            "centroid": (cent_r, cent_c),
        })
        all_target_9.update(coords)

    # ------------------------------------------------------------------
    # Preprocess objects.
    # ------------------------------------------------------------------
    object_data = []
    for comp in objects:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        rmin = min(rs)
        cmin = min(cs)
        hobj = max(rs) - rmin + 1
        wobj = max(cs) - cmin + 1

        rel_pixels = []
        nine_rel = []
        markers = {}

        for r, c in comp:
            rr = r - rmin
            cc = c - cmin
            val = int(grid[r, c])
            rel_pixels.append((rr, cc, val))

            if val == 9:
                nine_rel.append((rr, cc))
            elif val != 8 and val != bg:
                markers.setdefault(val, []).append((rr, cc))

        object_data.append({
            "rel_pixels": rel_pixels,
            "nine_rel": nine_rel,
            "markers": markers,
            "h": hobj,
            "w": wobj,
            "coords": comp,
        })

    def apply_transform(coords, h, w, t):
        """Apply one of the 8 dihedral transforms to local coordinates."""
        out = []
        for r, c in coords:
            if t == 0:      # identity
                nr, nc = r, c
            elif t == 1:    # rotate 90 clockwise
                nr, nc = c, h - 1 - r
            elif t == 2:    # rotate 180
                nr, nc = h - 1 - r, w - 1 - c
            elif t == 3:    # rotate 270 clockwise / 90 counter-clockwise
                nr, nc = w - 1 - c, r
            elif t == 4:    # vertical flip (top-bottom)
                nr, nc = h - 1 - r, c
            elif t == 5:    # horizontal flip (left-right)
                nr, nc = r, w - 1 - c
            elif t == 6:    # main diagonal reflection
                nr, nc = c, r
            elif t == 7:    # anti-diagonal reflection
                nr, nc = w - 1 - c, h - 1 - r
            else:
                raise ValueError("bad transform")
            out.append((nr, nc))
        return out

    def wall_directions(tr, tc, color):
        """
        Return possible required directions (N/S/E/W) for a marker of this color,
        based on the nearest wall component of that color.
        """
        comps = wall_components.get(color, [])
        if not comps:
            return []

        best_for_dir = {}

        for (r1, r2, c1, c2) in comps:
            # Closest point on the wall bounding box.
            cr = min(max(tr, r1), r2)
            cc = min(max(tc, c1), c2)

            dr = cr - tr
            dc = cc - tc
            if dr == 0 and dc == 0:
                continue

            dist = dr * dr + dc * dc
            vertical = (r2 - r1 + 1) >= (c2 - c1 + 1)

            if vertical:
                if r1 <= tr <= r2:
                    if cc < tc:
                        direction = "W"
                    elif cc > tc:
                        direction = "E"
                    else:
                        if cr < tr:
                            direction = "N"
                        elif cr > tr:
                            direction = "S"
                        else:
                            continue
                else:
                    if abs(dr) >= abs(dc):
                        direction = "N" if dr < 0 else "S"
                    else:
                        direction = "W" if dc < 0 else "E"
            else:
                if c1 <= tc <= c2:
                    if cr < tr:
                        direction = "N"
                    elif cr > tr:
                        direction = "S"
                    else:
                        if cc < tc:
                            direction = "W"
                        elif cc > tc:
                            direction = "E"
                        else:
                            continue
                else:
                    if abs(dc) >= abs(dr):
                        direction = "W" if dc < 0 else "E"
                    else:
                        direction = "N" if dr < 0 else "S"

            if direction not in best_for_dir or dist < best_for_dir[direction]:
                best_for_dir[direction] = dist

        if not best_for_dir:
            return []

        best = min(best_for_dir.values())
        return [d for d, dist in best_for_dir.items() if dist <= best + 1e-9]

    def side_ok(pixels, direction, bbox):
        """
        Check whether marker pixels are on the requested side of the target
        9-pattern bounding box.
        """
        if not pixels:
            return True

        rmin, rmax, cmin, cmax = bbox
        rs = [r for r, c in pixels]
        cs = [c for r, c in pixels]
        avg_r = sum(rs) / len(rs)
        avg_c = sum(cs) / len(cs)

        if direction == "N":
            frac = sum(r <= rmin for r, c in pixels) / len(pixels)
            return frac >= 0.5 and avg_r <= rmin + 1e-9
        if direction == "S":
            frac = sum(r >= rmax for r, c in pixels) / len(pixels)
            return frac >= 0.5 and avg_r >= rmax - 1e-9
        if direction == "W":
            frac = sum(c <= cmin for r, c in pixels) / len(pixels)
            return frac >= 0.5 and avg_c <= cmin + 1e-9
        if direction == "E":
            frac = sum(c >= cmax for r, c in pixels) / len(pixels)
            return frac >= 0.5 and avg_c >= cmax - 1e-9

        return False

    def primary_offset(pixels, direction, bbox):
        """Small bonus: prefer markers clearly outside the target bbox."""
        if not pixels:
            return 0.0

        rmin, rmax, cmin, cmax = bbox
        avg_r = sum(r for r, c in pixels) / len(pixels)
        avg_c = sum(c for r, c in pixels) / len(pixels)

        if direction == "N":
            return max(0.0, rmin - avg_r)
        if direction == "S":
            return max(0.0, avg_r - rmax)
        if direction == "W":
            return max(0.0, cmin - avg_c)
        if direction == "E":
            return max(0.0, avg_c - cmax)
        return 0.0

    def marker_score(transformed_markers, req_dirs, bbox):
        """
        Score how well marker colors face their nearest wall colors.
        Returns (integer perfect-color count, float secondary score).
        """
        int_score = 0
        float_score = 0.0

        for color, pixels in transformed_markers.items():
            dirs = req_dirs.get(color, [])
            if not dirs:
                continue

            best_dir = None
            best_float = -1.0
            for d in dirs:
                if side_ok(pixels, d, bbox):
                    f = primary_offset(pixels, d, bbox)
                    if f > best_float:
                        best_float = f
                        best_dir = d

            if best_dir is not None:
                int_score += 1
                float_score += best_float

        return int_score, float_score

    def find_translations(nine_t, target_set, target_coords):
        """Find all integer translations aligning transformed 9s to target 9s."""
        if len(nine_t) != len(target_coords):
            return []

        translations = set()
        for a in nine_t:
            for b in target_coords:
                dr = b[0] - a[0]
                dc = b[1] - a[1]
                ok = True
                for x, y in nine_t:
                    if (x + dr, y + dc) not in target_set:
                        ok = False
                        break
                if ok:
                    translations.add((dr, dc))
        return list(translations)

    def candidate_valid(obj, t, trans, target):
        """Check bounds, wall overlap, and target overlap."""
        target_set = target["set"]

        for rr, cc, val in obj["rel_pixels"]:
            transformed = apply_transform([(rr, cc)], obj["h"], obj["w"], t)[0]
            nr, nc = transformed[0] + trans[0], transformed[1] + trans[1]

            if not (0 <= nr < h and 0 <= nc < w):
                return False

            if wall_mask[nr, nc]:
                return False

            # Do not overwrite target 9 pixels with non-9 pixels.
            if (nr, nc) in all_target_9:
                if not (val == 9 and (nr, nc) in target_set):
                    return False

        return True

    # ------------------------------------------------------------------
    # Evaluate all object -> target -> transform candidates.
    # ------------------------------------------------------------------
    n_obj = len(object_data)
    n_tgt = len(target_data)

    # best[i][j] = (combined_score, t, trans, int_score, float_score)
    best = [[None for _ in range(n_tgt)] for __ in range(n_obj)]
    # candidates_by_obj[i] = list of (combined_score, j, t, trans)
    candidates_by_obj = [[] for _ in range(n_obj)]

    for i, obj in enumerate(object_data):
        if not obj["nine_rel"]:
            continue

        for j, tgt in enumerate(target_data):
            tr, tc = tgt["centroid"]
            req_dirs = {}
            for color in obj["markers"]:
                req_dirs[color] = wall_directions(tr, tc, color)

            for t in range(8):
                nine_t = apply_transform(obj["nine_rel"], obj["h"], obj["w"], t)
                translations = find_translations(nine_t, tgt["set"], tgt["coords"])

                for trans in translations:
                    if not candidate_valid(obj, t, trans, tgt):
                        continue

                    # Transform marker pixels and add translation.
                    transformed_markers = {}
                    for color, pixels in obj["markers"].items():
                        tp = apply_transform(pixels, obj["h"], obj["w"], t)
                        transformed_markers[color] = [(r + trans[0], c + trans[1]) for r, c in tp]

                    int_score, float_score = marker_score(transformed_markers, req_dirs, tgt["bbox"])
                    dist = abs(trans[0]) + abs(trans[1])
                    combined = int_score * 100.0 + float_score - 0.001 * dist

                    cand = (combined, t, trans, int_score, float_score)
                    if best[i][j] is None or combined > best[i][j][0]:
                        best[i][j] = cand

                    candidates_by_obj[i].append((combined, j, t, trans))

    # ------------------------------------------------------------------
    # Assign objects to distinct targets.
    # ------------------------------------------------------------------
    assignment = {}  # obj_index -> (target_index, t, trans)

    # Prefer exact brute force when small.
    if n_obj <= 7 and n_tgt <= 9 and n_obj <= n_tgt:
        best_total = None
        best_perm = None

        for perm in permutations(range(n_tgt), n_obj):
            total = 0.0
            ok = True
            for i, j in enumerate(perm):
                if best[i][j] is None:
                    ok = False
                    break
                total += best[i][j][0]
            if ok and (best_total is None or total > best_total):
                best_total = total
                best_perm = perm

        if best_perm is not None:
            for i, j in enumerate(best_perm):
                _, t, trans, _, _ = best[i][j]
                assignment[i] = (j, t, trans)
        else:
            # Fallback greedy.
            used_targets = set()
            for i in range(n_obj):
                cands = sorted(candidates_by_obj[i], reverse=True)
                for combined, j, t, trans in cands:
                    if j not in used_targets:
                        assignment[i] = (j, t, trans)
                        used_targets.add(j)
                        break
    else:
        # Greedy fallback.
        used_targets = set()
        # Process objects with strongest best candidate first.
        order = sorted(range(n_obj), key=lambda i: max((c[0] for c in candidates_by_obj[i]), default=-1e18), reverse=True)
        for i in order:
            cands = sorted(candidates_by_obj[i], reverse=True)
            for combined, j, t, trans in cands:
                if j not in used_targets:
                    assignment[i] = (j, t, trans)
                    used_targets.add(j)
                    break

    # If some object got no assignment, try to keep it as close to original as possible.
    for i, obj in enumerate(object_data):
        if i not in assignment:
            # Fallback: leave original object unchanged.
            # We will not remove unassigned objects below.
            pass

    # ------------------------------------------------------------------
    # Build output.
    # ------------------------------------------------------------------
    out = grid.copy()

    # Remove assigned objects.
    for i, obj in enumerate(object_data):
        if i in assignment:
            for r, c in obj["coords"]:
                if not wall_mask[r, c]:
                    out[r, c] = bg

    # Place assigned transformed objects.
    # Place in deterministic order.
    for i in sorted(assignment.keys()):
        j, t, trans = assignment[i]
        obj = object_data[i]

        for rr, cc, val in obj["rel_pixels"]:
            nr_local, nc_local = apply_transform([(rr, cc)], obj["h"], obj["w"], t)[0]
            nr = nr_local + trans[0]
            nc = nc_local + trans[1]

            if 0 <= nr < h and 0 <= nc < w and not wall_mask[nr, nc]:
                out[nr, nc] = val

    return out