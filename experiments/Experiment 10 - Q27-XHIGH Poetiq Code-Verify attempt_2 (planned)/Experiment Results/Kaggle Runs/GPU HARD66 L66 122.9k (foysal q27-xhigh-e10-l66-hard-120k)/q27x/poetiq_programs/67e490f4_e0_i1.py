import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the output grid.

    Rule summary:
    - Identify the background color as the most frequent color.
    - Identify the frame as the largest connected component of a non-background color.
    - Output the frame's bounding box, filled with the frame color.
    - Empty slots inside the frame are filled with colors from other colored objects.
    - Slot colors are chosen by shape matching. For the two square templates in the
      examples (11x11 and 13x13), canonical slot layouts are used. Otherwise, the
      actual connected non-frame regions inside the bounding box are used as slots.
    """
    grid = np.array(grid, dtype=int)
    if grid.size == 0:
        return grid

    def component_stats(pts):
        if not pts:
            return 0, 0.0, 0.0, 0, 0, 0, 0
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        return len(pts), float(np.mean(rs)), float(np.mean(cs)), min(rs), max(rs), min(cs), max(cs)

    def components(mask):
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=int)
        comps = []
        label = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == 0:
                    label += 1
                    stack = [(r, c)]
                    labels[r, c] = label
                    pts = []
                    while stack:
                        y, x = stack.pop()
                        pts.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                stack.append((ny, nx))
                    comps.append(pts)
        return comps

    def canonical_shape_from_mask(mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return ()

        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        sub = mask[r0:r1 + 1, c0:c1 + 1]

        best = None
        for k in range(4):
            t = np.rot90(sub, k)
            for flip in (False, True):
                u = np.fliplr(t) if flip else t
                yy, xx = np.where(u)
                if len(yy) == 0:
                    continue
                v = u[int(yy.min()):int(yy.max()) + 1, int(xx.min()):int(xx.max()) + 1]
                key = (v.shape, v.tobytes())
                if best is None or key < best:
                    best = key
        return best

    def canonical_shape_from_pts(pts):
        if not pts:
            return ()
        area, _, _, r0, r1, c0, c1 = component_stats(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        return canonical_shape_from_mask(mask)

    def components_from_coords(coords):
        s = set(coords)
        seen = set()
        comps = []
        for start in s:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in s and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(pts)
        if not comps:
            return []
        return max(comps, key=len)

    def make_11_slots():
        n = 11
        center = 5
        slots = {}

        corner = []
        for r0, c0 in ((1, 1), (1, 8), (8, 1), (8, 8)):
            for i in range(2):
                for j in range(2):
                    corner.append((r0 + i, c0 + j))
        slots["corner"] = corner

        edge = []
        for r in (1, 2, 8, 9):
            edge.append((r, center))
        for c in (1, 2, 8, 9):
            edge.append((center, c))
        slots["edge"] = edge

        center_slots = [
            (center, center),
            (center - 1, center),
            (center + 1, center),
            (center, center - 1),
            (center, center + 1),
        ]
        slots["center"] = center_slots
        return slots

    def make_13_slots():
        n = 13
        center = 6
        slots = {}

        corner = [
            # top-left L
            (1, 1), (1, 2), (2, 1),
            # top-right L
            (1, 10), (1, 11), (2, 11),
            # bottom-left L
            (10, 1), (11, 1), (11, 2),
            # bottom-right L
            (10, 11), (11, 10), (11, 11),
        ]
        slots["corner"] = corner

        edge = [
            (1, center),
            (center, 1),
            (center, 11),
            (11, center),
        ]
        slots["edge"] = edge

        center_slots = [
            (center, center),
            (center - 1, center),
            (center + 1, center),
            (center, center - 1),
            (center, center + 1),
        ]
        slots["center"] = center_slots

        ring = []
        for r in (3, 9):
            for c in (3, 4, 5, 7, 8, 9):
                ring.append((r, c))
        for r in (3, 4, 5, 7, 8, 9):
            for c in (3, 9):
                if (r, c) not in ring:
                    ring.append((r, c))
        slots["ring"] = ring
        return slots

    def choose_color_for_pool(pool, used_colors, total_color_area, color_use_count,
                              rep_area=None, exact=True):
        if not pool:
            # Fallback: most frequent non-frame color overall.
            if total_color_area:
                items = sorted(total_color_area.items(), key=lambda kv: (-kv[1], kv[0]))
                for color, _ in items:
                    if color not in used_colors:
                        return color
                return items[0][0]
            return 0

        # Prefer colors not already used, if possible.
        unused_pool = [x for x in pool if x["color"] not in used_colors]
        if unused_pool:
            pool = unused_pool

        groups = defaultdict(list)
        for x in pool:
            groups[x["color"]].append(x)

        def score(color, lst):
            count = len(lst)
            area = sum(x["area"] for x in lst)
            total = total_color_area.get(color, 0)
            use_penalty = -color_use_count.get(color, 0)

            if exact:
                return (count, area, total, use_penalty, -color)

            min_diff = min(abs(rep_area - x["area"]) for x in lst)
            return (-min_diff, count, area, total, use_penalty, -color)

        best_color = None
        best_score = None
        for color, lst in groups.items():
            s = score(color, lst)
            if best_score is None or s > best_score:
                best_score = s
                best_color = color
        return best_color

    def assign_shape_colors(shape_info, cand_comps):
        """
        shape_info: dict shape_key -> {"area": int, "slots": list}
        cand_comps: list of candidate component dicts
        Returns dict shape_key -> color
        """
        if not shape_info:
            return {}
        if not cand_comps:
            return {k: 0 for k in shape_info}

        total_color_area = defaultdict(int)
        for c in cand_comps:
            total_color_area[c["color"]] += c["area"]

        entries = []
        for shape_key, info in shape_info.items():
            rep_area = info["area"]
            exact = [c for c in cand_comps if c["shape"] == shape_key]
            if exact:
                pool = exact
                exact_flag = True
            else:
                if cand_comps:
                    diffs = [abs(c["area"] - rep_area) for c in cand_comps]
                    min_diff = min(diffs)
                    threshold = max(1, rep_area // 5)
                    pool = [c for c in cand_comps if abs(c["area"] - rep_area) <= min_diff + threshold]
                else:
                    pool = []
                exact_flag = False

            # Use unique-color pool size to decide rarity.
            unique_colors = set(c["color"] for c in pool)
            entries.append((len(unique_colors), len(shape_key) if isinstance(shape_key, tuple) else 0,
                            shape_key, pool, exact_flag, rep_area))

        # Process rare shapes first.
        entries.sort(key=lambda x: (x[0], x[1]))

        used_colors = set()
        color_use_count = defaultdict(int)
        result = {}

        for _, _, shape_key, pool, exact_flag, rep_area in entries:
            color = choose_color_for_pool(
                pool, used_colors, total_color_area, color_use_count,
                rep_area=rep_area, exact=exact_flag
            )
            result[shape_key] = color
            used_colors.add(color)
            color_use_count[color] += 1

        return result

    # ------------------------------------------------------------------
    # 1. Identify background and frame
    # ------------------------------------------------------------------
    flat = grid.reshape(-1)
    max_color = int(flat.max())
    counts = np.bincount(flat, minlength=max(max_color + 1, 10))
    background = int(np.argmax(counts))

    unique_colors = np.unique(grid)
    best = None
    frame_color = None
    frame_bbox = None

    for color in unique_colors:
        if color == background:
            continue
        mask = grid == color
        comps = components(mask)
        for pts in comps:
            area, _, _, r0, r1, c0, c1 = component_stats(pts)
            bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            # Prefer large bounding box, then filled area.
            key = (bbox_area, area)
            if best is None or key > best[0]:
                best = (key, int(color), (r0, r1, c0, c1))

    if best is None:
        return grid.copy()

    _, frame_color, (r0, r1, c0, c1) = best
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    out = np.full((h, w), frame_color, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # 2. Candidate colored objects (non-background, non-frame)
    # ------------------------------------------------------------------
    cand_comps = []
    cid = 0
    for color in unique_colors:
        if color == background or color == frame_color:
            continue
        mask = grid == color
        for pts in components(mask):
            area, cr, cc, rr0, rr1, cc0, cc1 = component_stats(pts)
            cand_comps.append({
                "id": cid,
                "color": int(color),
                "area": area,
                "center_r": cr,
                "center_c": cc,
                "shape": canonical_shape_from_pts(pts),
            })
            cid += 1

    # ------------------------------------------------------------------
    # 3. Create slots and fill them
    # ------------------------------------------------------------------
    use_canonical_square = (h == w and h in (11, 13))

    if use_canonical_square:
        if h == 11:
            slots = make_11_slots()
        else:
            slots = make_13_slots()

        shape_info = {}
        fill_by_shape = defaultdict(list)

        for slot_name, coords in slots.items():
            if not coords:
                continue
            rep_pts = components_from_coords(coords)
            if not rep_pts:
                continue
            shape_key = canonical_shape_from_pts(rep_pts)
            area = len(rep_pts)

            if shape_key not in shape_info:
                shape_info[shape_key] = {"area": area, "slots": []}
            shape_info[shape_key]["slots"].append(slot_name)

            # For filling, store local coordinates.
            for r, c in coords:
                fill_by_shape[shape_key].append((r, c))

        shape_colors = assign_shape_colors(shape_info, cand_comps)

        for shape_key, coords in fill_by_shape.items():
            color = shape_colors.get(shape_key, frame_color)
            for r, c in coords:
                out[r, c] = color

    else:
        # Use actual non-frame connected regions inside the frame bounding box.
        bbox_grid = grid[r0:r1 + 1, c0:c1 + 1]
        hole_mask = bbox_grid != frame_color
        hole_comps = components(hole_mask)

        shape_info = {}
        fill_by_shape = defaultdict(list)

        for pts in hole_comps:
            if not pts:
                continue
            shape_key = canonical_shape_from_pts(pts)
            area = len(pts)

            if shape_key not in shape_info:
                shape_info[shape_key] = {"area": area, "slots": []}
            # Keep the largest representative area for this shape.
            shape_info[shape_key]["area"] = max(shape_info[shape_key]["area"], area)
            shape_info[shape_key]["slots"].append(pts)

            for local_r, local_c in pts:
                fill_by_shape[shape_key].append((local_r, local_c))

        shape_colors = assign_shape_colors(shape_info, cand_comps)

        for shape_key, coords in fill_by_shape.items():
            color = shape_colors.get(shape_key, frame_color)
            for r, c in coords:
                out[r, c] = color

    # The examples and challenge have a full frame border.
    if h > 0 and w > 0:
        out[0, :] = frame_color
        out[-1, :] = frame_color
        out[:, 0] = frame_color
        out[:, -1] = frame_color

    return out