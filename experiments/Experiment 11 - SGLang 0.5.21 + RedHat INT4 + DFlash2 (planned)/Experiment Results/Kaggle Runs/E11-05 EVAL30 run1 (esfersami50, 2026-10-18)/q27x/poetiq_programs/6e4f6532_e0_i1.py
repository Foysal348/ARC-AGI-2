import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by moving "key" objects onto "lock" positions.

    Assumptions derived from the examples:
    - 8 pixels form the body/ring of a movable key.
    - 9 pixels inside a key form its center pattern.
    - Isolated 9-only connected components are locks/target positions.
    - Large non-background, non-8/9 components are frame walls.
    - Each key is paired with a lock whose 9-pattern matches the key's 9-pattern
      after one of the 8 dihedral transformations.
    - Among valid transformations, choose the one that best aligns the key's
      non-8/non-9 tab colors with the frame wall colors in the directions
      facing those walls at the lock position.
    """

    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ---- color statistics -------------------------------------------------
    flat = grid.reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    bg_color = int(unique[np.argmax(counts)])

    # ---- find frame components --------------------------------------------
    # Frame candidates are pixels that are not background and not special 8/9.
    frame_candidate = (grid != bg_color) & (grid != 8) & (grid != 9)

    def connected_components(mask):
        comps = []
        seen = np.zeros_like(mask, dtype=bool)
        for r in range(H):
            for c in range(W):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)])
                    seen[r, c] = True
                    comp = []
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc))
                        for nr, nc in ((cr + 1, cc), (cr - 1, cc),
                                       (cr, cc + 1), (cr, cc - 1)):
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    comps.append(comp)
        return comps

    frame_components = connected_components(frame_candidate)
    frame_mask = np.zeros((H, W), dtype=bool)
    frame_pixels = []

    # A component is considered frame if it is large enough / spans a large part
    # of the grid. This separates walls from small colored tabs.
    min_frame_size = max(8, int(0.015 * H * W))
    for comp in frame_components:
        if len(comp) < min_frame_size:
            continue
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if len(comp) >= min_frame_size and (h >= H // 2 or w >= W // 2 or len(comp) >= 0.05 * H * W):
            for r, c in comp:
                frame_mask[r, c] = True
            frame_pixels.extend(comp)

    # ---- find key/lock components ------------------------------------------
    # Key/lock candidates are all non-background pixels not classified as frame.
    obj_candidate = (grid != bg_color) & (~frame_mask)
    obj_components = connected_components(obj_candidate)

    keys = []
    locks = []

    for comp in obj_components:
        has_8 = any(grid[r, c] == 8 for r, c in comp)
        has_9 = any(grid[r, c] == 9 for r, c in comp)
        if has_8:
            keys.append(comp)
        elif has_9:
            locks.append(comp)

    # If no locks/keys, return background + frame.
    if not keys or not locks:
        out = grid.copy()
        for comp in obj_components:
            for r, c in comp:
                out[r, c] = bg_color
        return out

    # ---- helper: normalized pattern -----------------------------------------
    def normalize_points(points):
        if not points:
            return []
        rmin = min(r for r, c in points)
        cmin = min(c for r, c in points)
        return tuple(sorted((r - rmin, c - cmin) for r, c in points))

    def key_9_points(comp):
        return [(r, c) for r, c in comp if grid[r, c] == 9]

    def lock_points(comp):
        return [(r, c) for r, c in comp if grid[r, c] == 9]

    # ---- dihedral transformations -------------------------------------------
    # Each transform maps (r, c) relative to origin.
    def transforms():
        # Identity, rotations, and reflections.
        yield "id", lambda r, c: (r, c)
        yield "rot90", lambda r, c: (c, -r)
        yield "rot180", lambda r, c: (-r, -c)
        yield "rot270", lambda r, c: (-c, r)
        yield "hflip", lambda r, c: (r, -c)
        yield "vflip", lambda r, c: (-r, c)
        yield "main_diag", lambda r, c: (c, r)
        yield "anti_diag", lambda r, c: (-c, -r)

    def transform_points(points, f):
        if not points:
            return []
        transformed = [f(r, c) for r, c in points]
        rmin = min(r for r, c in transformed)
        cmin = min(c for r, c in transformed)
        return tuple(sorted((r - rmin, c - cmin) for r, c in transformed))

    # Precompute key 9 patterns and lock patterns.
    key_9_norm = []
    lock_norm = []
    for comp in keys:
        key_9_norm.append(normalize_points(key_9_points(comp)))
    for comp in locks:
        lock_norm.append(normalize_points(lock_points(comp)))

    # Generate all valid transformations for each key-lock shape match.
    valid_transforms = {}
    for ki, k9 in enumerate(key_9_norm):
        for lj, lp in enumerate(lock_norm):
            for name, f in transforms():
                if transform_points(k9, f) == lp:
                    valid_transforms.setdefault((ki, lj), []).append(name)

    # ---- frame wall query ---------------------------------------------------
    # For a placed key bbox, query first frame color in each cardinal direction.
    def nearest_frame_color(side, r0, r1, c0, c1):
        candidates = defaultdict(int)
        if side == "top":
            for c in range(c0, c1 + 1):
                r = r0 - 1
                while r >= 0:
                    if frame_mask[r, c]:
                        candidates[grid[r, c]] += 1
                        break
                    r -= 1
        elif side == "bottom":
            for c in range(c0, c1 + 1):
                r = r1 + 1
                while r < H:
                    if frame_mask[r, c]:
                        candidates[grid[r, c]] += 1
                        break
                    r += 1
        elif side == "left":
            for r in range(r0, r1 + 1):
                c = c0 - 1
                while c >= 0:
                    if frame_mask[r, c]:
                        candidates[grid[r, c]] += 1
                        break
                    c -= 1
        elif side == "right":
            for r in range(r0, r1 + 1):
                c = c1 + 1
                while c < W:
                    if frame_mask[r, c]:
                        candidates[grid[r, c]] += 1
                        break
                    c += 1
        if not candidates:
            return None
        # Choose the color that appears most often; ties by smaller color.
        return max(candidates.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Determine which tab colors are on each side of a transformed key bbox.
    def side_tab_colors(pixels, r0, r1, c0, c1):
        side_colors = {
            "top": set(),
            "bottom": set(),
            "left": set(),
            "right": set(),
        }
        for r, c, col in pixels:
            if col in (8, 9):
                continue
            if r == r0:
                side_colors["top"].add(col)
            if r == r1:
                side_colors["bottom"].add(col)
            if c == c0:
                side_colors["left"].add(col)
            if c == c1:
                side_colors["right"].add(col)
        return side_colors

    # Score a transformed key placed at a translation.
    def score_placement(comp, tname, translation):
        tr, tc = translation
        # Transform all key pixels.
        f = next(f for name, f in transforms() if name == tname)
        transformed = []
        for r, c in comp:
            nr, nc = f(r, c)
            transformed.append((nr + tr, nc + tc, grid[r, c]))

        rows = [r for r, c, _ in transformed]
        cols = [c for r, c, _ in transformed]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        tab_sides = side_tab_colors(transformed, r0, r1, c0, c1)
        score = 0
        for side in ("top", "bottom", "left", "right"):
            wall_col = nearest_frame_color(side, r0, r1, c0, c1)
            if wall_col is None:
                continue
            if wall_col in tab_sides[side]:
                score += 1
        return score, transformed

    # Find best (key, lock, transform, translation) assignments.
    # First compute best option for each compatible pair.
    options = []
    for (ki, lj), tnames in valid_transforms.items():
        key_comp = keys[ki]
        lock_comp = locks[lj]
        k9 = key_9_points(key_comp)
        l9 = lock_points(lock_comp)
        best = None
        for tname in tnames:
            f = next(f for name, f in transforms() if name == tname)
            # Find translation that maps key 9s to lock 9s.
            for pr, pc in k9:
                for lr, lc in l9:
                    dr, dc = lr - pr, lc - pc
                    translated_k9 = {(r + dr, c + dc) for r, c in k9}
                    if translated_k9 == set(l9):
                        score, _ = score_placement(key_comp, tname, (dr, dc))
                        opt = (score, ki, lj, tname, (dr, dc))
                        if best is None or opt[0] > best[0]:
                            best = opt
                        break
                # Avoid repeated identical translations from multiple anchor points.
                seen = set()
                # Simple dedupe not essential because shapes are small.
            # Continue checking other tnames.
        if best is not None:
            options.append(best)

    if not options:
        # Fallback: simply clear all movable objects.
        out = grid.copy()
        for comp in obj_components:
            for r, c in comp:
                out[r, c] = bg_color
        return out

    # Greedy one-to-one assignment maximizing total score.
    # Number of objects is small in these tasks.
    chosen = []
    used_keys = set()
    used_locks = set()
    # Sort by score descending; tie-break by key/lock index for stability.
    for score, ki, lj, tname, tr in sorted(options, key=lambda x: (-x[0], x[1], x[2])):
        if ki in used_keys or lj in used_locks:
            continue
        chosen.append((ki, lj, tname, tr))
        used_keys.add(ki)
        used_locks.add(lj)

    # Build output grid.
    out = grid.copy()

    # Clear original key/lock pixels (but keep frame/background).
    for comp in obj_components:
        for r, c in comp:
            if not frame_mask[r, c]:
                out[r, c] = bg_color

    # Place chosen transformed keys.
    fmap = {name: f for name, f in transforms()}
    for ki, lj, tname, (dr, dc) in chosen:
        f = fmap[tname]
        key_comp = keys[ki]
        for r, c in key_comp:
            nr, nc = f(r, c)
            nr += dr
            nc += dc
            if 0 <= nr < H and 0 <= nc < W:
                # Do not overwrite frame pixels; keys are expected not to overlap them.
                if not frame_mask[nr, nc]:
                    out[nr, nc] = grid[r, c]

    return out