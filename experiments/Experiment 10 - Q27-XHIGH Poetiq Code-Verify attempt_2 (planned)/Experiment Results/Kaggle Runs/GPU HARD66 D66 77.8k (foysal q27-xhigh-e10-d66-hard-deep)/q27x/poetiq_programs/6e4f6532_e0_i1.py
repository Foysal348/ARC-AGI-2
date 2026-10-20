import numpy as np
from collections import Counter, defaultdict
from itertools import product


def _connected_component(mask, r, c):
    """BFS connected component of mask (4-connectivity)."""
    h, w = mask.shape
    if not mask[r, c]:
        return []
    seen = np.zeros_like(mask, dtype=bool)
    stack = [(r, c)]
    seen[r, c] = True
    comp = []
    while stack:
        y, x = stack.pop()
        comp.append((y, x))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return comp


def _rotate_pixels(pixels, k, h, w):
    """Rotate pixel coordinates k times 90° clockwise in an h x w bounding box."""
    pixels = [(r, c) for r, c in pixels]
    if k == 0:
        return pixels, h, w
    for _ in range(k):
        pixels = [(c, h - 1 - r) for r, c in pixels]
        h, w = w, h
    return pixels, h, w


def _minimal_bbox(pixels):
    if not pixels:
        return 0, 0, 0, 0
    rs = [r for r, _ in pixels]
    cs = [c for _, c in pixels]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    return r0, c0, r1, c1


def _translate_to_origin(pixels):
    r0, c0, _, _ = _minimal_bbox(pixels)
    return [(r - r0, c - c0) for r, c in pixels]


def _is_wall_segment(grid, color, axis):
    """Detect full/central wall strips of a given color."""
    h, w = grid.shape
    if axis == "row":
        counts = [np.count_nonzero(grid[i] == color) for i in range(h)]
    else:
        counts = [np.count_nonzero(grid[:, j] == color) for j in range(w)]
    length = h if axis == "row" else w
    return [i for i, cnt in enumerate(counts) if cnt >= max(1, length * 0.85)]


def _detect_walls(grid):
    """
    Detect border/central wall colors.

    Returns:
        walls: dict with keys 'top','bottom','left','right','hmid','vmid'
               if present.
        wall_colors: set of wall colors
    """
    h, w = grid.shape
    bg = Counter(grid.flatten()).most_common(1)[0][0]
    walls = {}
    colors = Counter(grid.flatten()).most_common(10)
    candidates = [c for c, _ in colors if c != bg]

    def try_wall(color):
        top_rows = _is_wall_segment(grid, color, "row")
        if top_rows:
            top_i = min(top_rows)
            if top_i <= 2:
                walls.setdefault("top", color)
            elif top_i > h // 2:
                walls.setdefault("bottom", color)
            else:
                walls.setdefault("hmid", color)

        left_cols = _is_wall_segment(grid, color, "col")
        if left_cols:
            left_i = min(left_cols)
            if left_i <= 2:
                walls.setdefault("left", color)
            elif left_i > w // 2:
                walls.setdefault("right", color)
            else:
                walls.setdefault("vmid", color)

    for color in candidates:
        try_wall(color)

    # If only one full row/center row exists and there is no bottom,
    # ensure it is treated as top if it is at the top.
    return walls, set(walls.values())


def _interior_mask(grid, wall_colors, bg):
    """Mask of cells that are neither background nor wall-strip pixels."""
    h, w = grid.shape
    mask = np.ones((h, w), dtype=bool)
    mask[grid == bg] = False

    # Remove wall strips: rows/cols that are mostly a wall color.
    for color in wall_colors:
        rows = _is_wall_segment(grid, color, "row")
        for r in rows:
            mask[r, :] = False
        cols = _is_wall_segment(grid, color, "col")
        for c in cols:
            mask[:, c] = False
    return mask


def _group_objects(grid, mask):
    """
    Group object pixels.

    Objects are connected components of non-background/non-wall pixels.
    In these puzzles, wall-colored pixels inside the playfield are part
    of objects, while the actual wall strips have already been masked out.
    """
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    objects = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                comp = _connected_component(mask, i, j)
                seen[comp and np.array([c[0] for c in comp]) or (i, j)] = False  # placeholder
                # Mark properly
                for (r, c) in comp:
                    seen[r, c] = True
                pixels = comp
                colors = [grid[r, c] for r, c in pixels]
                objects.append({"pixels": pixels, "colors": colors})
    return objects


def _endpoint_colors(obj, wall_colors):
    """Wall-colored pixels that appear inside the object."""
    return {c for c in obj["colors"] if c in wall_colors}


def _primary_wall(obj, walls):
    """
    Choose a target wall based on endpoint colors present in the object.
    Preference: explicit endpoint color matching a wall.
    If multiple, use the wall with the closest average distance to the object centroid.
    """
    eps = _endpoint_colors(obj, set(walls.values()))
    if not eps:
        return None

    # Map wall side -> color
    side_by_color = {color: side for side, color in walls.items()}

    rs = [r for r, _ in obj["pixels"]]
    cs = [c for _, c in obj["pixels"]]
    cy, cx = sum(rs) / len(rs), sum(cs) / len(cs)
    h, w = 1, 1  # not used; distances are relative to object size
    obj_r0, obj_c0, obj_r1, obj_c1 = _minimal_bbox(obj["pixels"])
    oh = max(1, obj_r1 - obj_r0 + 1)
    ow = max(1, obj_c1 - obj_c0 + 1)

    scored = []
    for color in eps:
        side = side_by_color.get(color)
        if side is None:
            continue
        # heuristic score: smaller is better
        if side == "top":
            score = abs(cy - (obj_r0 + oh / 2))  # not important
        elif side == "bottom":
            score = abs(cy - (obj_r0 + oh / 2))
        elif side == "left":
            score = abs(cx - (obj_c0 + ow / 2))
        elif side == "right":
            score = abs(cx - (obj_c0 + ow / 2))
        elif side == "hmid":
            score = 0
        elif side == "vmid":
            score = 0
        else:
            score = 10
        scored.append((score + 0.001 * len(eps), side, color))
    scored.sort()
    return scored[0][1] if scored else None


def _rotation_for_side(obj, target_side, wall_colors):
    """
    Choose rotation (0,1,2,3 clockwise) that puts the object's matching
    endpoint color closest to the target side.
    """
    eps = _endpoint_colors(obj, wall_colors)
    if not eps:
        return 0

    r0, c0, r1, c1 = _minimal_bbox(obj["pixels"])
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    best_k, best_score = 0, float("inf")

    for k in range(4):
        pix, h, w = _rotate_pixels([(r - r0, c - c0) for r, c in obj["pixels"]], k, oh, ow)
        # side distances for endpoint pixels
        total = 0.0
        cnt = 0
        for (r, c), color in zip(pix, [obj["pixels"][0][0]]):  # placeholder, recompute below
            pass

        # Need colors after rotation; create mapping old pixel -> color
        color_by_old = {(r, c): None for r, c in []}
        # Simpler: rotate with color
        rotated = []
        for (r, c) in obj["pixels"]:
            rr, cc = r - r0, c - c0
            for _ in range(k):
                rr, cc = cc, h - 1 - rr
                h, w = w, h
            rotated.append((rr, cc))

        # map back to original colors by order is tricky; instead use set heuristic.
        # Score by side occupancy: for each endpoint color, look at pixels of that color.
        old_colors = [grid_temp for grid_temp in []]
        # Rebuild with colors:
        pairs = []
        for (r, c) in obj["pixels"]:
            rr, cc = r - r0, c - c0
            for _ in range(k):
                rr, cc = cc, h - 1 - rr
                h, w = w, h
            pairs.append((rr, cc))
        # We lost color; do another rotation retaining color by indexing:
        rot_pairs = []
        H, W = oh, ow
        for idx, (r, c) in enumerate(obj["pixels"]):
            rr, cc = r - r0, c - c0
            for _ in range(k):
                rr, cc = cc, H - 1 - rr
                H, W = W, H
            rot_pairs.append((rr, cc))
        # color list aligned to obj pixels
        col_list = [None] * len(obj["pixels"])
        # We need original color; we can get from grid outside? Not here.
        # Therefore use a simpler heuristic: choose k so that endpoint colors are not
        # all on the opposite side. This is intentionally conservative.
        break

    return 0


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule (as inferred from the examples):

    1. Detect the background color and the colored wall strips.
    2. Extract the non-background, non-wall objects.
    3. For each object, infer the wall it should be associated with from
       the wall-colored endpoint pixels inside the object.
    4. Rotate the object so that those endpoint colors face the associated wall.
    5. Reposition the object toward that wall, centered in the available
       interior lane, while preserving the relative ordering of objects and
       avoiding overlaps.
    6. Keep the original border/wall strips unchanged.

    The implementation is conservative and modular; it falls back to the
    original grid if wall structure cannot be detected reliably.
    """
    if grid.size == 0:
        return grid.copy()

    # Work on a copy; never mutate input.
    out = grid.copy()
    h, w = grid.shape

    bg = Counter(grid.flatten()).most_common(1)[0][0]
    walls, wall_colors = _detect_walls(grid)

    # If we cannot find at least two distinct wall strips, the puzzle is not
    # in the expected family; return input unchanged.
    if len(walls) < 2:
        return out

    mask = _interior_mask(grid, wall_colors, bg)

    # Extract objects as connected components.
    objects = []
    seen = np.zeros_like(mask, dtype=bool)
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                comp = _connected_component(mask, i, j)
                for (r, c) in comp:
                    seen[r, c] = True
                if len(comp) >= 3:  # ignore tiny noise
                    objects.append({
                        "pixels": comp,
                        "colors": [grid[r, c] for r, c in comp]
                    })

    if not objects:
        return out

    # Determine a simple target side for each object.
    def target_side_for(obj):
        eps = {c for c in obj["colors"] if c in wall_colors}
        if not eps:
            return None
        # Prefer endpoint color that is not the most common interior color.
        for side, color in walls.items():
            if color in eps:
                return side
        return None

    # For this task family, objects are moved toward the wall whose color
    # appears as an endpoint. We use a safe translation: shift the object
    # toward that wall until it is at least one background cell away from
    # any wall strip and does not overlap other placed objects.
    placed = []

    def bbox(pixels):
        return _minimal_bbox(pixels)

    # Sort objects by top-to-bottom, then left-to-right to keep stable order.
    objects.sort(key=lambda o: (bbox(o["pixels"])[0], bbox(o["pixels"])[1]))

    def overlaps(pix, other):
        other_set = set(other)
        return any(p in other_set for p in pix)

    for obj in objects:
        side = target_side_for(obj)
        if side is None:
            continue

        r0, c0, r1, c1 = bbox(obj["pixels"])
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        pixels = [(r - r0, c - c0) for r, c in obj["pixels"]]
        colors = obj["colors"]

        # Try the four rotations. Choose the one that puts endpoint-colored
        # pixels closest to the selected side of the object's bounding box.
        best_k, best_pix, best_score = 0, pixels, float("inf")
        for k in range(4):
            H, W = obj_h, obj_w
            rot = []
            for idx, (r, c) in enumerate(pixels):
                rr, cc = r, c
                for _ in range(k):
                    rr, cc = cc, H - 1 - rr
                    H, W = W, H
                rot.append((rr, cc))
            Rh, Rw = W, H

            score = 0.0
            endpoint_pixels = []
            # Need colors after rotation: we can rotate colors by same index.
            for (rr, cc), col in zip(rot, [grid[r0 + r, c0 + c] for r, c in obj["pixels"]]):
                if col in wall_colors:
                    endpoint_pixels.append((rr, cc))

            if not endpoint_pixels:
                score = 100
            else:
                if side in ("top", "hmid"):
                    # For horizontal wall, vertical coordinate matters.
                    score = sum(rr for rr, _ in endpoint_pixels) / len(endpoint_pixels)
                elif side in ("bottom",):
                    score = sum((Rh - 1 - rr) for rr, _ in endpoint_pixels) / len(endpoint_pixels)
                elif side in ("left", "vmid"):
                    score = sum(cc for _, cc in endpoint_pixels) / len(endpoint_pixels)
                elif side in ("right",):
                    score = sum((Rw - 1 - cc) for _, cc in endpoint_pixels) / len(endpoint_pixels)
                else:
                    score = 100
            if score < best_score:
                best_score = score
                best_k, best_pix = k, rot

        H, W = obj_h, obj_w
        rot = []
        for (r, c) in pixels:
            rr, cc = r, c
            for _ in range(best_k):
                rr, cc = cc, H - 1 - rr
                H, W = W, H
            rot.append((rr, cc))
        Rh, Rw = W, H

        # Normalize to origin.
        rot = _translate_to_origin(rot)

        # Determine target lane.
        # We use the interior bounding box after removing wall strips.
        interior_rows = np.where(mask.any(axis=1))[0]
        interior_cols = np.where(mask.any(axis=0))[0]
        if len(interior_rows) == 0 or len(interior_cols) == 0:
            continue
        ir0, ir1 = interior_rows[0], interior_rows[-1]
        ic0, ic1 = interior_cols[0], interior_cols[-1]

        if side in ("top", "hmid"):
            target_r = ir0 + 1
            target_c = (ic0 + ic1) // 2 - Rw // 2
        elif side == "bottom":
            target_r = ir1 - Rh - 1
            target_c = (ic0 + ic1) // 2 - Rw // 2
        elif side == "left" or side == "vmid":
            target_r = (ir0 + ir1) // 2 - Rh // 2
            target_c = ic0 + 1
        elif side == "right":
            target_r = (ir0 + ir1) // 2 - Rh // 2
            target_c = ic1 - Rw - 1
        else:
            continue

        # Slide to avoid overlaps.
        def can_place(r, c):
            if r < 0 or c < 0 or r + Rh > h or c + Rw > w:
                return False
            pix = [(r + dr, c + dc) for dr, dc in rot]
            # cannot overwrite non-background original pixels that are not this object
            for (pr, pc) in pix:
                if out[pr, pc] != bg and not any(pr == r0 + a and pc == c0 + b for a, b in obj["pixels"]):
                    # allow overwriting background and already placed object pixels
                    if not any(pr == pr2 and pc == pc2 for pr2, pc2 in placed):
                        return False
            for other in placed:
                other_set = set(other)
                if any((pr, pc) in other_set for pr, pc in pix):
                    return False
            return True

        # Search a small neighborhood around target.
        placed_here = None
        for dr in range(0, h):
            for dc in range(0, w):
                if can_place(target_r + dr, target_c + dc):
                    placed_here = (target_r + dr, target_c + dc)
                    break
            if placed_here is not None:
                break
        if placed_here is None:
            continue

        pr, pc = placed_here
        pix = [(pr + dr, pc + dc) for dr, dc in rot]
        placed.append(pix)

        # First clear original object pixels, then draw rotated object.
        for (r, c) in obj["pixels"]:
            if bg == out[r, c] or True:
                # only clear if it is object pixel
                out[r, c] = bg
        for (r, c) in pix:
            # find color by matching position in rotated list
            pass

        # Rebuild colors: rotate colors with the same indices.
        H, W = obj_h, obj_w
        rot_with_colors = []
        for idx, (r, c) in enumerate(pixels):
            rr, cc = r, c
            for _ in range(best_k):
                rr, cc = cc, H - 1 - rr
                H, W = W, H
            col = grid[r0 + r, c0 + c]
            rot_with_colors.append((rr, cc, col))
        # normalize
        rr0, cc0 = min(a for a, _, _ in rot_with_colors), min(b for _, b, _ in rot_with_colors)
        for (rr, cc, col) in rot_with_colors:
            out[pr + (rr - rr0), pc + (cc - cc0)] = col

    return out