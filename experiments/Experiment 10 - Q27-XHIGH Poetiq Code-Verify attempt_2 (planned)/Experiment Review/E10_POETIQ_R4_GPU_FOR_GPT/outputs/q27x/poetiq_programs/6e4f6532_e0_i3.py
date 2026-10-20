import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by relocating and reorienting the 8/9 "creatures".

    Inferred rule:
    1. The most frequent color is the background.
    2. Static walls are large / long-thin components that do not contain both
       colors 8 and 9.
    3. Seeds are small components made only of 9.
    4. Creatures are components containing both 8 and 9.
    5. Each creature is assigned to a seed with the same number of 9-pixels
       when possible, otherwise to the nearest/compatible seed.
    6. The creature is chosen among its 8 dihedral symmetries so that its
       non-8/9 marker colors point toward the nearest static walls of the same
       colors.
    7. The transformed creature is placed so that its 9-centroid aligns with
       the seed's 9-centroid, without overwriting static walls.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    # Background = most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = (grid != bg)

    # 4-connected components of non-background pixels.
    visited = np.zeros((H, W), dtype=bool)
    comps: List[Dict] = []

    for r in range(H):
        for c in range(W):
            if nonbg[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                pixels: List[Tuple[int, int]] = []

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
                h, w = r1 - r0 + 1, c1 - c0 + 1

                color_set = set(int(grid[p[0], p[1]]) for p in pixels)
                nine_pixels = [(r, c) for r, c in pixels if grid[r, c] == 9]

                comp = {
                    "pixels": pixels,
                    "color_set": color_set,
                    "size": len(pixels),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "h": h,
                    "w": w,
                    "n9": len(nine_pixels),
                    "cy": sum(rs) / len(rs),
                    "cx": sum(cs) / len(cs),
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
    creatures: List[Dict] = []
    seeds: List[Dict] = []

    area = H * W
    for comp in comps:
        colors = comp["color_set"]
        h, w = comp["h"], comp["w"]
        size = comp["size"]

        long_thin = (
            (h <= max(4, H // 4) and w >= max(8, W // 2)) or
            (w <= max(4, W // 4) and h >= max(8, H // 2))
        )
        very_large = size >= max(20, int(0.15 * area))
        is_static_big = long_thin or very_large

        if 8 in colors and 9 in colors:
            creatures.append(comp)
        elif colors == {9} and not is_static_big:
            seeds.append(comp)
        else:
            for r, c in comp["pixels"]:
                static_mask[r, c] = True

    if not creatures or not seeds:
        return grid.copy()

    # Build tight local pattern for each component.
    def make_pattern(comp: Dict) -> np.ndarray:
        pat = np.zeros((comp["h"], comp["w"]), dtype=int)
        for r, c in comp["pixels"]:
            pat[r - comp["r0"], c - comp["c0"]] = int(grid[r, c])
        return pat

    # Dihedral symmetries, deduplicated.
    def unique_transforms(pat: np.ndarray) -> List[np.ndarray]:
        res = []
        seen = set()

        def add(arr: np.ndarray) -> None:
            key = arr.tobytes()
            if key not in seen:
                seen.add(key)
                res.append(arr)

        # Rotations first, then reflections.
        for k in (0, 1, 2, 3):
            add(np.rot90(pat, k=k))
            add(np.fliplr(np.rot90(pat, k=k)))
        return res

    # For a transformed pattern, map non-8/9 marker colors to the sides they touch.
    def color_sides(pat: np.ndarray) -> Dict[int, set]:
        res: Dict[int, set] = {}
        h, w = pat.shape

        def add(val: int, side: str) -> None:
            val = int(val)
            if val in (0, 8, 9):
                return
            res.setdefault(val, set()).add(side)

        for v in pat[0]:
            add(int(v), "top")
        for v in pat[-1]:
            add(int(v), "bottom")
        for v in pat[:, 0]:
            add(int(v), "left")
        for v in pat[:, -1]:
            add(int(v), "right")

        return res

    # For each seed, precompute nearest static wall colors in the 4 directions.
    def wall_info(seed: Dict) -> Dict[str, List[Tuple[int, int]]]:
        info = {"top": [], "bottom": [], "left": [], "right": []}

        # Top.
        hits = []
        for c in range(seed["c0"], seed["c1"] + 1):
            for r in range(seed["r0"] - 1, -1, -1):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), seed["r0"] - r))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["top"] = [(col, md) for col in sorted({col for col, d in hits if d == md})]

        # Bottom.
        hits = []
        for c in range(seed["c0"], seed["c1"] + 1):
            for r in range(seed["r1"] + 1, H):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), r - seed["r1"]))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["bottom"] = [(col, md) for col in sorted({col for col, d in hits if d == md})]

        # Left.
        hits = []
        for r in range(seed["r0"], seed["r1"] + 1):
            for c in range(seed["c0"] - 1, -1, -1):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), seed["c0"] - c))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["left"] = [(row, md) for row in sorted({row for row, d in hits if d == md})]

        # Right.
        hits = []
        for r in range(seed["r0"], seed["r1"] + 1):
            for c in range(seed["c1"] + 1, W):
                if static_mask[r, c]:
                    hits.append((int(grid[r, c]), c - seed["c1"]))
                    break
        if hits:
            md = min(d for _, d in hits)
            info["right"] = [(row, md) for row in sorted({row for row, d in hits if d == md})]

        return info

    seed_wall_info = [wall_info(s) for s in seeds]

    # Evaluate one transformed creature at one seed.
    def evaluate(pat: np.ndarray, seed: Dict, wall_dir: Dict[str, List[Tuple[int, int]]]) -> Tuple:
        h, w = pat.shape

        ys, xs = np.where(pat == 9)
        if len(ys) > 0:
            oy = float(np.mean(ys))
            ox = float(np.mean(xs))
        else:
            oy = (h - 1) / 2.0
            ox = (w - 1) / 2.0

        # Align 9-centroid with seed centroid.
        top = int(round(seed["nine_cy"] - oy))
        left = int(round(seed["nine_cx"] - ox))

        valid = (0 <= top <= H - h and 0 <= left <= W - w)

        if valid:
            for i in range(h):
                row = pat[i]
                for j in np.nonzero(row)[0]:
                    if static_mask[top + i, left + j]:
                        valid = False
                        break
                if not valid:
                    break

        # Marker-to-wall scoring.
        side_to_dir = {"top": "top", "bottom": "bottom", "left": "left", "right": "right"}
        match = 0
        penalty = 0
        dist_sum = 0

        for col, sides in color_sides(pat).items():
            best = None
            for side in sides:
                for wall_color, dist in wall_dir[side_to_dir[side]]:
                    if wall_color == col:
                        if best is None or dist < best:
                            best = dist
            if best is not None:
                match += 1
                dist_sum += best
            else:
                # Penalize markers whose side faces a different-colored wall.
                for side in sides:
                    if wall_dir[side_to_dir[side]]:
                        penalty += 1
                        break

        # Prefer keeping 9-count shape if possible.
        seed_n9 = seed["n9"]
        pat_n9 = int(np.sum(pat == 9))
        n9_diff = abs(pat_n9 - seed_n9)

        return valid, top, left, match, penalty, dist_sum, n9_diff

    # Prepare candidate transformed patterns for each creature.
    creature_data: List[Dict] = []
    for comp in creatures:
        pat = make_pattern(comp)
        transforms = unique_transforms(pat)
        creature_data.append({
            "comp": comp,
            "n9": comp["n9"],
            "transforms": transforms,
        })

    # Build candidate placements: creature x seed x symmetry.
    all_cands = []
    for ci, cdata in enumerate(creature_data):
        for si, seed in enumerate(seeds):
            wall_dir = seed_wall_info[si]
            for ti, tpat in enumerate(cdata["transforms"]):
                valid, top, left, match, penalty, dist_sum, n9_diff = evaluate(tpat, seed, wall_dir)

                # Sort key:
                # - valid placement first
                # - exact 9-count match first
                # - more marker/wall matches
                # - fewer penalties
                # - shorter wall distances
                # - closer original creature to seed
                # - earlier symmetry for determinism
                orig_dist = ((cdata["comp"]["nine_cy"] - seed["nine_cy"]) ** 2 +
                             (cdata["comp"]["nine_cx"] - seed["nine_cx"]) ** 2)

                key = (
                    0 if valid else 1,
                    n9_diff,
                    -match,
                    penalty,
                    dist_sum,
                    orig_dist,
                    ti,
                    ci,
                    si,
                )

                all_cands.append((key, ci, si, {
                    "valid": valid,
                    "top": top,
                    "left": left,
                    "pat": tpat,
                }))

    all_cands.sort(key=lambda x: x[0])

    # Greedy assignment: each creature gets one seed, each seed at most one creature.
    assigned_creature: Dict[int, Dict] = {}
    used_seeds = set()

    for _, ci, si, cand in all_cands:
        if ci not in assigned_creature and si not in used_seeds:
            assigned_creature[ci] = cand
            used_seeds.add(si)

    # Fallback: if some creatures remain unassigned, give them the best available seed.
    for ci in range(len(creature_data)):
        if ci not in assigned_creature:
            for _, c2, s2, cand in all_cands:
                if c2 == ci:
                    assigned_creature[ci] = cand
                    break

    if not assigned_creature:
        return grid.copy()

    # Clear all creatures and seeds; walls remain.
    out = grid.copy()
    clear_mask = np.zeros((H, W), dtype=bool)

    for comp in creatures:
        for r, c in comp["pixels"]:
            clear_mask[r, c] = True
    for comp in seeds:
        for r, c in comp["pixels"]:
            clear_mask[r, c] = True

    out[clear_mask] = bg

    # Draw assigned transformed creatures.
    for ci in sorted(assigned_creature):
        cand = assigned_creature[ci]
        pat = cand["pat"]
        h, w = pat.shape

        top = max(0, min(H - h, cand["top"]))
        left = max(0, min(W - w, cand["left"]))

        for i in range(h):
            for j in range(w):
                v = int(pat[i, j])
                if v == 0:
                    continue
                r, c = top + i, left + j
                if 0 <= r < H and 0 <= c < W and not static_mask[r, c]:
                    out[r, c] = v

    return out