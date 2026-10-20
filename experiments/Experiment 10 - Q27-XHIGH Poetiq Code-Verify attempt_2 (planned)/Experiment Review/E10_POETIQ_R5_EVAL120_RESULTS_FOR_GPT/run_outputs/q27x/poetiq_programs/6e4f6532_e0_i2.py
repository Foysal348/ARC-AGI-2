import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move each 8-containing object to a 9-only target.

    Rule inferred from examples:
    - The most frequent color is background.
    - Long straight bars of other colors are walls.
    - Components containing color 8 are objects. They may also contain color 9
      and small marker pixels whose colors match wall colors.
    - Components containing only color 9 are targets.
    - For each object, try the 8 dihedral transformations. The object's 9-pixel
      pattern must exactly cover one target 9-pattern.
    - The object orientation is chosen so that each marker color lies on the
      side of the object's 8-body bounding box that faces the nearest wall of
      that color.
    - Original objects and target 9s are cleared, then transformed objects are
      drawn.
    """
    if grid is None:
        return grid

    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    h, w = g.shape
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # ---------------------------------------------------------------
    # Connected components, 4-neighborhood
    # ---------------------------------------------------------------
    def label_components(mask):
        visited = np.zeros((h, w), dtype=bool)
        comps = []
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
                    comps.append(comp)
        return comps

    # ---------------------------------------------------------------
    # Detect wall bars. Walls are long, thin, solid components.
    # Colors 8 and 9 are reserved for objects/targets and excluded.
    # ---------------------------------------------------------------
    wall_mask = np.zeros((h, w), dtype=bool)
    wall_by_color = {}

    for color in vals:
        ci = int(color)
        if ci == bg or ci == 8 or ci == 9:
            continue

        mask = (g == ci)
        for comp in label_components(mask):
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            bh = rmax - rmin + 1
            bw = cmax - cmin + 1
            size = len(comp)

            if size < max(3, min(h, w) // 4):
                continue

            thick_h = max(2, int(0.25 * h) + 1)
            thick_w = max(2, int(0.25 * w) + 1)

            horizontal = (
                bw >= 0.6 * w and
                bh <= thick_h and
                size >= 0.75 * bh * bw
            )
            vertical = (
                bh >= 0.6 * h and
                bw <= thick_w and
                size >= 0.75 * bh * bw
            )

            if horizontal or vertical:
                for r, c in comp:
                    wall_mask[r, c] = True
                wall_by_color.setdefault(ci, []).append({
                    "rmin": rmin, "rmax": rmax,
                    "cmin": cmin, "cmax": cmax
                })

    # ---------------------------------------------------------------
    # Find interior objects and targets
    # ---------------------------------------------------------------
    interior = (g != bg) & (~wall_mask)
    comps = label_components(interior)

    objects = []
    targets = []

    for comp in comps:
        colors = set(int(g[r, c]) for r, c in comp)
        if 8 in colors:
            objects.append(comp)
        elif 9 in colors:
            if all(int(g[r, c]) == 9 for r, c in comp):
                targets.append(comp)

    if not objects or not targets:
        return g

    # ---------------------------------------------------------------
    # Preprocess targets
    # ---------------------------------------------------------------
    target_data = []
    all_target_set = set()

    for comp in targets:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        target_data.append({
            "coords": comp,
            "set": set(comp),
            "centroid": (sum(rs) / len(rs), sum(cs) / len(cs)),
        })
        all_target_set.update(comp)

    # ---------------------------------------------------------------
    # Preprocess objects
    # ---------------------------------------------------------------
    obj_data = []

    for comp in objects:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        rmin = min(rs)
        cmin = min(cs)
        rmax = max(rs)
        cmax = max(cs)
        oh = rmax - rmin + 1
        ow = cmax - cmin + 1

        all_pixels = []
        eight = []
        nine = []
        markers = {}

        for r, c in comp:
            val = int(g[r, c])
            rr = r - rmin
            cc = c - cmin
            all_pixels.append((rr, cc, val))

            if val == 8:
                eight.append((rr, cc))
            elif val == 9:
                nine.append((rr, cc))
            elif val != bg:
                markers.setdefault(val, []).append((rr, cc))

        obj_data.append({
            "coords": comp,
            "h": oh,
            "w": ow,
            "all_pixels": all_pixels,
            "eight": eight,
            "nine": nine,
            "markers": markers,
            "centroid": (sum(rs) / len(rs), sum(cs) / len(cs)),
        })

    # ---------------------------------------------------------------
    # Dihedral transformations
    # ---------------------------------------------------------------
    def apply_one(r, c, oh, ow, t):
        if t == 0:      # identity
            return (r, c)
        if t == 1:      # 90 clockwise
            return (c, oh - 1 - r)
        if t == 2:      # 180
            return (oh - 1 - r, ow - 1 - c)
        if t == 3:      # 90 counter-clockwise
            return (ow - 1 - c, r)
        if t == 4:      # vertical flip
            return (oh - 1 - r, c)
        if t == 5:      # horizontal flip
            return (r, ow - 1 - c)
        if t == 6:      # main diagonal reflection
            return (c, r)
        if t == 7:      # anti-diagonal reflection
            return (ow - 1 - c, oh - 1 - r)
        raise ValueError("bad transform")

    def transform_coords(coords, oh, ow, t):
        return [apply_one(r, c, oh, ow, t) for r, c in coords]

    # Prefer rotations, then reflections.
    TRANSFORM_ORDER = [0, 1, 3, 2, 4, 5, 6, 7]

    # ---------------------------------------------------------------
    # Helper: translations aligning transformed 9s to a target
    # ---------------------------------------------------------------
    def find_translations(nine_t, target_set, target_coords):
        if not nine_t or len(nine_t) != len(target_coords):
            return []

        trans = set()
        for a in nine_t:
            for b in target_coords:
                dr = b[0] - a[0]
                dc = b[1] - a[1]
                ok = True
                for r, c in nine_t:
                    if (r + dr, c + dc) not in target_set:
                        ok = False
                        break
                if ok:
                    trans.add((dr, dc))
        return list(trans)

    # ---------------------------------------------------------------
    # Helper: nearest wall direction for a marker color
    # ---------------------------------------------------------------
    wall_dir_cache = {}

    def wall_dirs(r, c, color):
        key = (round(r * 4), round(c * 4), int(color))
        if key in wall_dir_cache:
            return wall_dir_cache[key]

        comps = wall_by_color.get(int(color))
        if not comps:
            wall_dir_cache[key] = None
            return None

        best = []
        best_dist = float("inf")

        for comp in comps:
            cr = min(max(r, comp["rmin"]), comp["rmax"])
            cc = min(max(c, comp["cmin"]), comp["cmax"])
            dr = cr - r
            dc = cc - c
            if dr == 0 and dc == 0:
                continue

            dist = dr * dr + dc * dc
            dirs = []

            if abs(dr) > abs(dc):
                dirs.append("N" if dr < 0 else "S")
            elif abs(dc) > abs(dr):
                dirs.append("W" if dc < 0 else "E")
            else:
                if dr < 0:
                    dirs.append("N")
                elif dr > 0:
                    dirs.append("S")
                if dc < 0:
                    dirs.append("W")
                elif dc > 0:
                    dirs.append("E")

            if dist < best_dist - 1e-9:
                best = dirs
                best_dist = dist
            elif abs(dist - best_dist) <= 1e-9:
                for d in dirs:
                    if d not in best:
                        best.append(d)

        wall_dir_cache[key] = best if best else None
        return wall_dir_cache[key]

    # ---------------------------------------------------------------
    # Helper: marker side constraints relative to 8-body bbox
    # ---------------------------------------------------------------
    def side_ok(coords, direction, bbox):
        rmin, rmax, cmin, cmax = bbox
        if direction == "N":
            return all(r < rmin for r, c in coords)
        if direction == "S":
            return all(r > rmax for r, c in coords)
        if direction == "W":
            return all(c < cmin for r, c in coords)
        if direction == "E":
            return all(c > cmax for r, c in coords)
        return False

    def side_cost(coords, direction, bbox):
        rmin, rmax, cmin, cmax = bbox
        cost = 0
        for r, c in coords:
            if direction == "N":
                cost += rmin - r
            elif direction == "S":
                cost += r - rmax
            elif direction == "W":
                cost += cmin - c
            elif direction == "E":
                cost += c - cmax
        return cost

    # ---------------------------------------------------------------
    # Generate valid candidates
    # ---------------------------------------------------------------
    candidates = []

    for i, obj in enumerate(obj_data):
        if not obj["nine"]:
            continue

        for j, tgt in enumerate(target_data):
            if len(obj["nine"]) != len(tgt["coords"]):
                continue

            for t in TRANSFORM_ORDER:
                nine_t = transform_coords(obj["nine"], obj["h"], obj["w"], t)
                translations = find_translations(nine_t, tgt["set"], tgt["coords"])

                for trans in translations:
                    placed = {}
                    eight_g = []
                    ok = True

                    for rr, cc, val in obj["all_pixels"]:
                        nr0, nc0 = apply_one(rr, cc, obj["h"], obj["w"], t)
                        nr = nr0 + trans[0]
                        nc = nc0 + trans[1]

                        if not (0 <= nr < h and 0 <= nc < w) or wall_mask[nr, nc]:
                            ok = False
                            break

                        placed.setdefault(val, []).append((nr, nc))
                        if val == 8:
                            eight_g.append((nr, nc))

                    if not ok:
                        continue

                    # 9s must exactly cover this target.
                    if set(placed.get(9, [])) != tgt["set"]:
                        continue

                    # Non-9 pixels may not land on any target 9.
                    for val, coords in placed.items():
                        if val == 9:
                            continue
                        for nr, nc in coords:
                            if (nr, nc) in all_target_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue

                    # Use bounding box of 8-body for marker side tests.
                    if eight_g:
                        rmin8 = min(r for r, c in eight_g)
                        rmax8 = max(r for r, c in eight_g)
                        cmin8 = min(c for r, c in eight_g)
                        cmax8 = max(c for r, c in eight_g)
                    else:
                        all_coords = [coord for coords in placed.values() for coord in coords]
                        if not all_coords:
                            continue
                        rmin8 = min(r for r, c in all_coords)
                        rmax8 = max(r for r, c in all_coords)
                        cmin8 = min(c for r, c in all_coords)
                        cmax8 = max(c for r, c in all_coords)

                    bbox8 = (rmin8, rmax8, cmin8, cmax8)
                    center_r = (rmin8 + rmax8) / 2.0
                    center_c = (cmin8 + cmax8) / 2.0

                    satisfied = 0
                    marker_cost = 0
                    marker_ok = True

                    for color, rel_coords in obj["markers"].items():
                        placed_coords = placed.get(color, [])
                        if not placed_coords:
                            continue

                        dirs = wall_dirs(center_r, center_c, color)
                        if dirs is None:
                            # No known wall constraint for this color.
                            satisfied += 1
                            continue

                        color_ok = False
                        best_cost = float("inf")

                        for d in dirs:
                            if side_ok(placed_coords, d, bbox8):
                                c = side_cost(placed_coords, d, bbox8)
                                if c < best_cost:
                                    best_cost = c

                        if best_cost == float("inf"):
                            marker_ok = False
                            break

                        satisfied += 1
                        marker_cost += best_cost

                    if not marker_ok:
                        continue

                    transform_priority = TRANSFORM_ORDER.index(t)
                    orig_dist = int(abs(obj["centroid"][0] - tgt["centroid"][0]) +
                                    abs(obj["centroid"][1] - tgt["centroid"][1]))

                    cost = (
                        0,                 # valid
                        -satisfied,        # more satisfied constraints is better
                        transform_priority,
                        int(marker_cost),
                        orig_dist
                    )

                    candidates.append({
                        "i": i,
                        "j": j,
                        "t": t,
                        "trans": trans,
                        "placed": placed,
                        "coords_set": set(coord for coords in placed.values() for coord in coords),
                        "cost": cost,
                    })

    if not candidates:
        return g

    # ---------------------------------------------------------------
    # Keep a small number of best candidates per object-target pair
    # ---------------------------------------------------------------
    pair_cands = {}
    for cand in candidates:
        key = (cand["i"], cand["j"])
        pair_cands.setdefault(key, []).append(cand)

    for key in pair_cands:
        pair_cands[key].sort(key=lambda x: x["cost"])
        pair_cands[key] = pair_cands[key][:5]

    n_obj = len(obj_data)
    n_tgt = len(target_data)

    # ---------------------------------------------------------------
    # Assign objects to distinct targets
    # ---------------------------------------------------------------
    def add_cost(a, b):
        return tuple(x + y for x, y in zip(a, b))

    obj_order = sorted(range(n_obj), key=lambda i: len([j for (oi, j) in pair_cands if oi == i]))

    current = {}

    def backtrack(idx, used_targets, used_coords, total_cost):
        if idx == len(obj_order):
            return total_cost, dict(current)

        i = obj_order[idx]
        best = None

        for j in range(n_tgt):
            if j in used_targets:
                continue
            cands = pair_cands.get((i, j), [])
            for cand in cands:
                if cand["coords_set"] & used_coords:
                    continue

                current[i] = cand
                new_cost = add_cost(total_cost, cand["cost"])
                res = backtrack(
                    idx + 1,
                    used_targets | {j},
                    used_coords | cand["coords_set"],
                    new_cost
                )

                if res is not None and (best is None or res[0] < best[0]):
                    best = res

                del current[i]

        return best

    init_cost = (0, 0, 0, 0, 0)
    full = backtrack(0, set(), set(), init_cost)

    if full is None:
        # Fallback greedy if exact assignment fails.
        assignment = {}
        used_targets = set()
        used_coords = set()

        for i in obj_order:
            best_cand = None
            best_j = -1
            for j in range(n_tgt):
                if j in used_targets:
                    continue
                for cand in pair_cands.get((i, j), []):
                    if cand["coords_set"] & used_coords:
                        continue
                    if best_cand is None or cand["cost"] < best_cand["cost"]:
                        best_cand = cand
                        best_j = j

            if best_cand is not None:
                assignment[i] = best_cand
                used_targets.add(best_j)
                used_coords |= best_cand["coords_set"]
    else:
        assignment = full[1]

    if not assignment:
        return g

    # ---------------------------------------------------------------
    # Build output
    # ---------------------------------------------------------------
    out = g.copy()

    used_targets = set(cand["j"] for cand in assignment.values())

    # Remove assigned objects.
    for i, cand in assignment.items():
        for r, c in obj_data[i]["coords"]:
            if not wall_mask[r, c]:
                out[r, c] = bg

    # Clear used target 9s (they are replaced by placed object 9s).
    # If this was a complete one-to-one assignment, clear all targets.
    clear_targets = target_data if len(assignment) == len(obj_data) and len(assignment) == len(target_data) \
                    else [target_data[j] for j in used_targets]

    for tgt in clear_targets:
        for r, c in tgt["coords"]:
            out[r, c] = bg

    # Place transformed objects.
    for i in sorted(assignment.keys()):
        cand = assignment[i]
        for val, coords in cand["placed"].items():
            for r, c in coords:
                if not wall_mask[r, c]:
                    out[r, c] = val

    return out