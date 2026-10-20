import numpy as np
import math
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Combine two yellow (4) objects marked by orange (7) handles.

    Rule implemented:
    1. Find the two main 4-colored object components.
    2. Find the 7-colored marker components and associate each marker with
       the object component it touches / is nearest to.
    3. Choose the fixed object:
       - if the two objects are on opposite vertical sides of their markers,
         keep the object above its marker;
       - if they are on opposite horizontal sides, keep the object left of
         its marker;
       - otherwise keep the larger object, breaking ties by upper-left.
    4. Translate the other object so that the edge of it adjacent to its own
       marker aligns with the fixed object's marker.
    5. Remove all 7 pixels, keep the fixed object and the translated object,
       and crop to the bounding box of all resulting 4 pixels.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    UP, DOWN, LEFT, RIGHT = (-1, 0), (1, 0), (0, -1), (0, 1)
    DIRS = (UP, DOWN, LEFT, RIGHT)

    # ------------------------------------------------------------------
    # Color identification.  The examples use 4 for objects and 7 for markers.
    # If those colors are absent, infer reasonable colors.
    # ------------------------------------------------------------------
    has4 = bool((g == 4).any())
    has7 = bool((g == 7).any())

    if has4 and has7:
        obj_color = 4
        marker_color = 7
    else:
        vals, counts = np.unique(g, return_counts=True)
        bg_guess = int(vals[np.argmax(counts)])
        non_bg = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != bg_guess]

        if not non_bg:
            return np.full((H, W), bg_guess, dtype=int)

        non_bg.sort(key=lambda x: x[1], reverse=True)

        if has4:
            obj_color = 4
        else:
            obj_color = int(non_bg[0][0])

        if has7:
            marker_color = 7
        else:
            marker_candidates = [int(v) for v, _ in non_bg if int(v) != obj_color]
            if not marker_candidates:
                # No clear marker: crop object pixels only.
                cells = set(map(tuple, np.argwhere(g == obj_color).tolist()))
                if not cells:
                    return np.full((H, W), bg_guess, dtype=int)
                min_r = min(r for r, _ in cells)
                max_r = max(r for r, _ in cells)
                min_c = min(c for _, c in cells)
                max_c = max(c for _, c in cells)
                out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg_guess, dtype=int)
                for r, c in cells:
                    out[r - min_r, c - min_c] = obj_color
                return out
            marker_color = int(marker_candidates[0])

    # Background is the most frequent color that is neither object nor marker.
    bg_mask = (g != obj_color) & (g != marker_color)
    if bg_mask.any():
        vals, counts = np.unique(g[bg_mask], return_counts=True)
        bg_color = int(vals[np.argmax(counts)])
    else:
        bg_color = 0 if obj_color != 0 else 1

    def crop_cells(cells):
        """Crop to the bounding box of cells and paint them as object color."""
        if not cells:
            return np.full((H, W), bg_color, dtype=int)

        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)

        out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg_color, dtype=int)
        for r, c in cells:
            out[r - min_r, c - min_c] = obj_color
        return out

    def get_components(color):
        """4-connected components of one color, returned as a list of sets."""
        mask = (g == color)
        seen = np.zeros((H, W), dtype=bool)
        comps = []

        for r in range(H):
            for c in range(W):
                if not mask[r, c] or seen[r, c]:
                    continue

                stack = [(r, c)]
                seen[r, c] = True
                comp = set()

                while stack:
                    x, y = stack.pop()
                    comp.add((x, y))

                    for dr, dc in DIRS:
                        nx, ny = x + dr, y + dc
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            stack.append((nx, ny))

                comps.append(comp)

        return comps

    def centroid(cells):
        if not cells:
            return (0.0, 0.0)
        rs = sum(r for r, _ in cells)
        cs = sum(c for _, c in cells)
        n = len(cells)
        return (rs / n, cs / n)

    def set_distance(a, b):
        if not a or not b:
            return 10**9
        best = 10**9
        for r1, c1 in a:
            for r2, c2 in b:
                d = abs(r1 - r2) + abs(c1 - c2)
                if d < best:
                    best = d
        return best

    def nearest_int(x):
        # Round half away from zero.
        if x >= 0:
            return int(math.floor(x + 0.5))
        return int(math.ceil(x - 0.5))

    def shift(cells, dy, dx):
        return {(r + dy, c + dx) for r, c in cells}

    obj_comps = get_components(obj_color)
    marker_comps = get_components(marker_color)

    if not obj_comps:
        return np.full((H, W), bg_color, dtype=int)

    if not marker_comps:
        cells = set()
        for comp in obj_comps:
            cells.update(comp)
        return crop_cells(cells)

    # Map object cells to component id.
    obj_id = np.full((H, W), -1, dtype=int)
    for idx, comp in enumerate(obj_comps):
        for r, c in comp:
            obj_id[r, c] = idx

    # ------------------------------------------------------------------
    # Build two pieces: object component + its marker(s).
    # ------------------------------------------------------------------
    if len(obj_comps) == 2:
        groups = [set(), set()]

        for m in marker_comps:
            counts = [0, 0]

            # Prefer direct adjacency to an object component.
            for r, c in m:
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == obj_color:
                        oid = int(obj_id[nr, nc])
                        if 0 <= oid < 2:
                            counts[oid] += 1

            if counts[0] or counts[1]:
                if counts[0] == counts[1]:
                    oid = 0 if set_distance(m, obj_comps[0]) <= set_distance(m, obj_comps[1]) else 1
                else:
                    oid = 0 if counts[0] > counts[1] else 1
            else:
                oid = 0 if set_distance(m, obj_comps[0]) <= set_distance(m, obj_comps[1]) else 1

            groups[oid].update(m)

        # Fallback: if a marker was missed, attach the closest marker.
        if not groups[0]:
            groups[0].update(min(marker_comps, key=lambda m: set_distance(m, obj_comps[0])))
        if not groups[1]:
            groups[1].update(min(marker_comps, key=lambda m: set_distance(m, obj_comps[1])))

        pieces = [
            {"obj": obj_comps[0], "marker": groups[0]},
            {"obj": obj_comps[1], "marker": groups[1]},
        ]

    else:
        # General fallback for more than two object components.
        groups = defaultdict(set)

        for m in marker_comps:
            counts = defaultdict(int)
            for r, c in m:
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == obj_color:
                        oid = int(obj_id[nr, nc])
                        if oid >= 0:
                            counts[oid] += 1

            if counts:
                oid = max(
                    counts.keys(),
                    key=lambda o: (counts[o], len(obj_comps[o]), -set_distance(m, obj_comps[o])),
                )
            else:
                oid = min(range(len(obj_comps)), key=lambda o: set_distance(m, obj_comps[o]))

            groups[oid].update(m)

        nonempty = [oid for oid in groups if groups[oid]]
        if len(nonempty) >= 2:
            selected = sorted(
                nonempty,
                key=lambda o: (len(obj_comps[o]), sum(len(m) for m in groups[o]), -o),
                reverse=True,
            )[:2]
        else:
            selected = nonempty.copy()
            for o in sorted(range(len(obj_comps)), key=lambda o: -len(obj_comps[o])):
                if o not in selected:
                    selected.append(o)
                if len(selected) == 2:
                    break

        pieces = []
        for o in selected:
            marker_set = set(groups.get(o, set()))
            if not marker_set:
                marker_set.update(min(marker_comps, key=lambda m: set_distance(m, obj_comps[o])))
            pieces.append({"obj": obj_comps[o], "marker": marker_set})

    if len(pieces) < 2:
        cells = set()
        for comp in obj_comps:
            cells.update(comp)
        return crop_cells(cells)

    # ------------------------------------------------------------------
    # Determine the direction from each marker to its object.
    # ------------------------------------------------------------------
    def direction_for(obj_set, marker_set):
        counts = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}

        for r, c in marker_set:
            for d in DIRS:
                if (r + d[0], c + d[1]) in obj_set:
                    counts[d] += 1

        if sum(counts.values()) > 0:
            mr, mc = centroid(marker_set)
            best_dir = UP
            best_score = -1

            for d, cnt in counts.items():
                if cnt == 0:
                    continue
                mass = sum(
                    1
                    for r, c in obj_set
                    if (r - mr) * d[0] + (c - mc) * d[1] > 0
                )
                score = cnt * 1000 + mass
                if score > best_score:
                    best_score = score
                    best_dir = d

            return best_dir

        # Fallback: direction from marker centroid to object centroid.
        mr, mc = centroid(marker_set)
        or_, oc = centroid(obj_set)
        dr = or_ - mr
        dc = oc - mc
        if abs(dr) >= abs(dc):
            return DOWN if dr > 0 else UP
        return RIGHT if dc > 0 else LEFT

    def adjacent_obj_cells(marker_set, obj_set, direction=None):
        dirs = [direction] if direction is not None else list(DIRS)
        cells = set()

        for r, c in marker_set:
            for d in dirs:
                cell = (r + d[0], c + d[1])
                if cell in obj_set:
                    cells.add(cell)

        # If direction was too restrictive, retry all directions.
        if not cells and direction is not None:
            for r, c in marker_set:
                for d in DIRS:
                    cell = (r + d[0], c + d[1])
                    if cell in obj_set:
                        cells.add(cell)

        # Last fallback: closest object cells to the marker.
        if not cells:
            best = 10**9
            for m in marker_set:
                for o in obj_set:
                    d = abs(m[0] - o[0]) + abs(m[1] - o[1])
                    if d < best:
                        best = d
            cells = {
                o
                for o in obj_set
                if any(abs(m[0] - o[0]) + abs(m[1] - o[1]) == best for m in marker_set)
            }

        return cells

    pieces[0]["direction"] = direction_for(pieces[0]["obj"], pieces[0]["marker"])
    pieces[1]["direction"] = direction_for(pieces[1]["obj"], pieces[1]["marker"])

    # ------------------------------------------------------------------
    # Choose the fixed piece.
    # ------------------------------------------------------------------
    def choose_target(p0, p1):
        d0 = p0["direction"]
        d1 = p1["direction"]

        if d0 == UP and d1 == DOWN:
            return 0
        if d0 == DOWN and d1 == UP:
            return 1
        if d0 == LEFT and d1 == RIGHT:
            return 0
        if d0 == RIGHT and d1 == LEFT:
            return 1

        if len(p0["obj"]) != len(p1["obj"]):
            return 0 if len(p0["obj"]) > len(p1["obj"]) else 1

        c0 = centroid(p0["obj"])
        c1 = centroid(p1["obj"])
        if (c0[0], c0[1]) != (c1[0], c1[1]):
            return 0 if (c0[0], c0[1]) < (c1[0], c1[1]) else 1

        return 0

    target_idx = choose_target(pieces[0], pieces[1])
    source_idx = 1 - target_idx

    target = pieces[target_idx]
    source = pieces[source_idx]

    # The moving edge is the part of the source object adjacent to its marker.
    source_front = adjacent_obj_cells(source["marker"], source["obj"], source["direction"])
    target_marker = target["marker"]
    target_adj = adjacent_obj_cells(target["marker"], target["obj"], target["direction"])

    # ------------------------------------------------------------------
    # Candidate translations.
    # ------------------------------------------------------------------
    candidates = set()
    preferred_t = (0, 0)

    # Primary: align source marker-adjacent edge to target marker.
    if source_front:
        tf = centroid(target_marker)
        sf = centroid(source_front)
        base = (nearest_int(tf[0] - sf[0]), nearest_int(tf[1] - sf[1]))
        preferred_t = base

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                candidates.add((base[0] + dr, base[1] + dc))

        # Pairwise correspondences, when small enough.
        if len(source_front) * len(target_marker) <= 10000:
            for a in source_front:
                for b in target_marker:
                    candidates.add((b[0] - a[0], b[1] - a[1]))

    # Secondary: align source marker to target marker-adjacent object cells.
    if source["marker"] and target_adj:
        ta = centroid(target_adj)
        sm = centroid(source["marker"])
        base = (nearest_int(ta[0] - sm[0]), nearest_int(ta[1] - sm[1]))
        if not source_front:
            preferred_t = base

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                candidates.add((base[0] + dr, base[1] + dc))

        if len(source["marker"]) * len(target_adj) <= 10000:
            for a in source["marker"]:
                for b in target_adj:
                    candidates.add((b[0] - a[0], b[1] - a[1]))

    # Last fallback: move whole source object centroid toward target marker.
    if not candidates:
        tm = centroid(target_marker)
        so = centroid(source["obj"])
        base = (nearest_int(tm[0] - so[0]), nearest_int(tm[1] - so[1]))
        candidates.add(base)
        preferred_t = base

    def score_translation(t):
        dy, dx = t
        sf = shift(source_front, dy, dx) if source_front else set()
        sm = shift(source["marker"], dy, dx)

        overlap_fm = len(sf & target_marker) if source_front else 0
        overlap_ms = len(sm & target_adj) if target_adj else 0

        if sf:
            c1 = centroid(sf)
            c2 = centroid(target_marker)
            d_fm = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
        else:
            d_fm = 1e9

        if sm and target_adj:
            c1 = centroid(sm)
            c2 = centroid(target_adj)
            d_ms = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
        else:
            d_ms = 1e9 if target_adj else 0.0

        t_dist = abs(t[0] - preferred_t[0]) + abs(t[1] - preferred_t[1])
        move = abs(dy) + abs(dx)

        return (
            overlap_fm,
            overlap_ms,
            -d_fm,
            -d_ms,
            -t_dist,
            -move,
        )

    best_dy, best_dx = max(candidates, key=score_translation)

    # ------------------------------------------------------------------
    # Build final set of object pixels and crop.
    # ------------------------------------------------------------------
    out_cells = set(target["obj"])
    out_cells |= shift(source["obj"], best_dy, best_dx)
    out_cells |= target_marker

    return crop_cells(out_cells)