import numpy as np
from collections import deque
import math


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by moving "creature" objects to isolated 9-seeds.

    Inferred rule:
    - The most common color is the background.
    - Connected non-background components containing both 8 and 9 are movable objects.
    - Connected components made only of 9 are target seeds.
    - All other non-background components are static walls/borders.

    For each object:
    - Choose a seed, preferring a seed with at least as many 9s as the object,
      then the closest count, then the best orientation score, then proximity.
    - Choose one of the 8 grid symmetries of the object.  The symmetry is scored
      by checking whether non-8/9 marker colors on the object's sides point toward
      the nearest static wall/border of the same color from the chosen seed.
    - Place the transformed object so that the centroid of its 9-pixels aligns
      with the centroid of the chosen seed.
    - Erase all original objects and seeds, then redraw the transformed objects.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = grid != bg

    # Connected components (4-connected) of non-background pixels.
    visited = np.zeros((H, W), dtype=bool)
    comps = []

    for r in range(H):
        for c in range(W):
            if nonbg[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                pixels = []

                while q:
                    cr, cc = q.popleft()
                    pixels.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if nonbg[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                rs = [p[0] for p in pixels]
                cs = [p[1] for p in pixels]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)

                colors = set(int(grid[p[0], p[1]]) for p in pixels)
                nine_pixels = [(r, c) for r, c in pixels if grid[r, c] == 9]

                comp = {
                    "pixels": pixels,
                    "colors": colors,
                    "size": len(pixels),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "cy": sum(rs) / len(rs),
                    "cx": sum(cs) / len(cs),
                    "n9": len(nine_pixels),
                }

                if nine_pixels:
                    comp["nine_cy"] = sum(p[0] for p in nine_pixels) / len(nine_pixels)
                    comp["nine_cx"] = sum(p[1] for p in nine_pixels) / len(nine_pixels)
                else:
                    comp["nine_cy"] = comp["cy"]
                    comp["nine_cx"] = comp["cx"]

                comps.append(comp)

    # Classify components.
    static_mask = np.zeros((H, W), dtype=bool)
    objects = []
    seeds = []

    area = H * W
    for comp in comps:
        colors = comp["colors"]
        size = comp["size"]
        h = comp["h"]
        w = comp["w"]

        # Very large / long-thin 9-only components are treated as static walls,
        # not seeds.  Normal small 9-only components are seeds.
        long_thin = (
            (h <= max(4, H // 4) and w >= max(8, W // 2)) or
            (w <= max(4, W // 4) and h >= max(8, H // 2))
        )
        very_large = size >= max(20, int(0.15 * area))
        is_static_big = long_thin or very_large

        if 8 in colors and 9 in colors:
            objects.append(comp)
        elif colors == {9} and not is_static_big:
            seeds.append(comp)
        else:
            for r, c in comp["pixels"]:
                static_mask[r, c] = True

    if not objects or not seeds:
        return grid.copy()

    def round_half_up(x: float) -> int:
        return int(math.floor(x + 0.5))

    def orientation_of_coords(coords) -> int:
        """0 = single, 1 = vertical, 2 = horizontal, 3 = diagonal/square."""
        if len(coords) <= 1:
            return 0
        rs = [p[0] for p in coords]
        cs = [p[1] for p in coords]
        hr = max(rs) - min(rs)
        wc = max(cs) - min(cs)
        if hr > wc:
            return 1
        if wc > hr:
            return 2
        return 3

    # Prepare seed data.
    seed_list = []
    for comp in seeds:
        seed_list.append({
            "pixels": comp["pixels"],
            "set": set(comp["pixels"]),
            "size": comp["size"],
            "cy": comp["cy"],
            "cx": comp["cx"],
            "r0": comp["r0"],
            "r1": comp["r1"],
            "c0": comp["c0"],
            "c1": comp["c1"],
            "orient": orientation_of_coords(comp["pixels"]),
        })

    def compute_dir_info(seed):
        """
        For each of the four sides of a seed's bounding box, find the nearest
        static wall/border color in that direction.
        Returns dict direction -> list of (color, distance) at minimal distance.
        """
        info = {"up": [], "down": [], "left": [], "right": []}

        # Up
        hits = []
        for c in range(seed["c0"], seed["c1"] + 1):
            for r in range(seed["r0"] - 1, -1, -1):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), seed["r0"] - r))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["up"] = [(c, md) for c in sorted({c for c, d in hits if d == md})]

        # Down
        hits = []
        for c in range(seed["c0"], seed["c1"] + 1):
            for r in range(seed["r1"] + 1, H):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), r - seed["r1"]))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["down"] = [(c, md) for c in sorted({c for c, d in hits if d == md})]

        # Left
        hits = []
        for r in range(seed["r0"], seed["r1"] + 1):
            for c in range(seed["c0"] - 1, -1, -1):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), seed["c0"] - c))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["left"] = [(c, md) for c in sorted({c for c, d in hits if d == md})]

        # Right
        hits = []
        for r in range(seed["r0"], seed["r1"] + 1):
            for c in range(seed["c1"] + 1, W):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), c - seed["c1"]))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["right"] = [(c, md) for c in sorted({c for c, d in hits if d == md})]

        return info

    dir_cache = [compute_dir_info(s) for s in seed_list]

    def make_pattern(comp):
        """Tight local pattern for a component; background and other components are 0."""
        pat = np.zeros((comp["h"], comp["w"]), dtype=int)
        for r, c in comp["pixels"]:
            pat[r - comp["r0"], c - comp["c0"]] = int(grid[r, c])
        return pat

    def unique_transforms(pat):
        """Return 8 dihedral symmetries, deduplicated, with rotations before reflections."""
        res = []
        seen = set()

        def add(arr):
            key = arr.tobytes()
            if key not in seen:
                seen.add(key)
                res.append(arr)

        # Rotations: identity, CW, 180, CCW.
        for k in (0, -1, 2, 1):
            add(np.rot90(pat, k=k))

        # Reflections: horizontal flip of each rotation.
        for k in (0, -1, 2, 1):
            add(np.fliplr(np.rot90(pat, k=k)))

        return res

    def color_sides(pat):
        """
        Map each non-8/9 marker color to the sides of the bounding box it touches.
        """
        h, w = pat.shape
        res = {}

        def add(v, side):
            v = int(v)
            if v in (0, 8, 9):
                return
            res.setdefault(v, set()).add(side)

        for v in pat[0]:
            add(v, "top")
        for v in pat[-1]:
            add(v, "bottom")
        for v in pat[:, 0]:
            add(v, "left")
        for v in pat[:, -1]:
            add(v, "right")

        return res

    def evaluate_transform(pat, seed, dir_info):
        """
        Evaluate one transformed object against one seed.
        Returns:
            valid, top, left, match_score, dist_sum, orient_score, overlap9
        """
        h, w = pat.shape

        ys, xs = np.where(pat == 9)
        if len(ys) > 0:
            oy = float(np.mean(ys))
            ox = float(np.mean(xs))
        else:
            oy = (h - 1) / 2.0
            ox = (w - 1) / 2.0

        top = round_half_up(float(seed["cy"]) - oy)
        left = round_half_up(float(seed["cx"]) - ox)

        valid = (0 <= top and top + h <= H and 0 <= left and left + w <= W)

        if valid:
            for i in range(h):
                row = pat[i]
                for j in np.nonzero(row)[0]:
                    if static_mask[top + i, left + j]:
                        valid = False
                        break
                if not valid:
                    break

        overlap = 0
        if valid and len(ys) > 0:
            seed_set = seed["set"]
            for y, x in zip(ys, xs):
                if (top + int(y), left + int(x)) in seed_set:
                    overlap += 1

        # Marker-color side scoring.
        c_side = color_sides(pat)
        match = 0
        dist_sum = 0
        side_to_dir = {"top": "up", "bottom": "down", "left": "left", "right": "right"}

        for col, sides in c_side.items():
            best_dist = None
            for side in sides:
                for wall_color, dist in dir_info[side_to_dir[side]]:
                    if wall_color == col:
                        if best_dist is None or dist < best_dist:
                            best_dist = dist

            if best_dist is not None:
                match += 1
                dist_sum += best_dist
            else:
                # Penalize if this marker points at a side that has a different wall color.
                has_different_wall = any(
                    dir_info[side_to_dir[side]] for side in sides
                )
                if has_different_wall:
                    match -= 1

        # Orientation of 9-cluster.
        if len(ys) > 0:
            obj_orient = orientation_of_coords(list(zip(ys.tolist(), xs.tolist())))
        else:
            obj_orient = 0

        orient_score = 1 if obj_orient == seed["orient"] else 0

        return valid, top, left, match, dist_sum, orient_score, overlap

    # Build object data with precomputed symmetries.
    obj_list = []
    for comp in objects:
        pat = make_pattern(comp)
        obj_list.append({
            "comp": comp,
            "n9": comp["n9"],
            "nine_cy": comp["nine_cy"],
            "nine_cx": comp["nine_cx"],
            "transforms": unique_transforms(pat),
        })

    def best_for_pair(obj, seed_idx):
        seed = seed_list[seed_idx]
        dir_info = dir_cache[seed_idx]

        best_valid = None
        best_any = None

        for tidx, pat in enumerate(obj["transforms"]):
            valid, top, left, match, dist, orient, overlap = evaluate_transform(
                pat, seed, dir_info
            )

            # Transform key: better match, shorter matched-wall distance,
            # better 9-orientation, more 9 overlap, earlier transform.
            tkey = (-match, dist, -orient, -overlap, tidx)

            data = {
                "valid": valid,
                "top": top,
                "left": left,
                "match": match,
                "dist": dist,
                "orient": orient,
                "overlap": overlap,
                "tpat": pat,
                "tidx": tidx,
            }

            if valid:
                if best_valid is None or tkey < best_valid[0]:
                    best_valid = (tkey, data)
            if best_any is None or tkey < best_any[0]:
                best_any = (tkey, data)

        if best_valid is not None:
            return best_valid[1]
        return best_any[1] if best_any is not None else None

    # Build candidate object-seed assignments.
    obj_candidates = []
    for oi, obj in enumerate(obj_list):
        cands = []
        for si, seed in enumerate(seed_list):
            best = best_for_pair(obj, si)
            if best is None:
                continue

            cand = dict(best)
            cand["obj_idx"] = oi
            cand["seed_idx"] = si
            cand["obj_n9"] = obj["n9"]
            cand["seed_size"] = seed["size"]
            cand["dist_orig"] = math.hypot(
                obj["nine_cy"] - seed["cy"],
                obj["nine_cx"] - seed["cx"],
            )

            valid_flag = 0 if cand["valid"] else 1
            eligible = 0 if cand["seed_size"] >= cand["obj_n9"] else 1

            if cand["seed_size"] >= cand["obj_n9"]:
                count_diff = cand["seed_size"] - cand["obj_n9"]
            else:
                count_diff = 1000 + (cand["obj_n9"] - cand["seed_size"])

            cand["key"] = (
                valid_flag,
                eligible,
                count_diff,
                -cand["match"],
                cand["dist"],
                -cand["orient"],
                -cand["overlap"],
                cand["dist_orig"],
                cand["tidx"],
            )
            cands.append(cand)

        cands.sort(key=lambda c: c["key"])
        obj_candidates.append(cands)

    # Global greedy assignment to avoid two objects using the same seed when possible.
    all_cands = []
    for oi, cands in enumerate(obj_candidates):
        for si, c in enumerate(cands):
            all_cands.append((c["key"], oi, si, c))

    all_cands.sort(key=lambda x: (x[0], x[1], x[2]))

    assignments = {}
    used_seeds = set()

    for _, oi, si, cand in all_cands:
        if oi not in assignments and si not in used_seeds:
            assignments[oi] = cand
            used_seeds.add(si)

    # Fallback for any unassigned object.
    for oi in range(len(obj_list)):
        if oi not in assignments:
            chosen = None
            for cand in obj_candidates[oi]:
                if cand["seed_idx"] not in used_seeds:
                    chosen = cand
                    break
            if chosen is None and obj_candidates[oi]:
                chosen = obj_candidates[oi][0]
            if chosen is not None:
                assignments[oi] = chosen
                used_seeds.add(chosen["seed_idx"])

    if not assignments:
        return grid.copy()

    # Clear all movable objects and seeds.
    out = grid.copy()
    movable = np.zeros((H, W), dtype=bool)

    for obj in objects:
        for r, c in obj["pixels"]:
            movable[r, c] = True
    for seed in seeds:
        for r, c in seed["pixels"]:
            movable[r, c] = True

    out[movable] = bg

    # Insert transformed objects.
    for oi in sorted(assignments):
        cand = assignments[oi]
        pat = cand["tpat"]
        h, w = pat.shape

        top = max(0, min(H - h, cand["top"]))
        left = max(0, min(W - w, cand["left"]))

        for i in range(h):
            for j in range(w):
                v = int(pat[i, j])
                if v == 0:
                    continue
                ar, ac = top + i, left + j
                if 0 <= ar < H and 0 <= ac < W and not static_mask[ar, ac]:
                    out[ar, ac] = v

    return out