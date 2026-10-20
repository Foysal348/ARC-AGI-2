import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the ARC task:

    1. Find the "frame": a colored component whose bounding-box border is
       completely filled by that color and which contains zero-pixel holes.
    2. Find the "marker" color: the non-frame color with the most disconnected
       single/small components (in the examples this is the scattered color).
    3. Other non-zero, non-frame, non-marker components are "pieces".
    4. Find an integer scale factor s such that pieces can be transformed
       (rotations/flips), placed into the frame's holes scaled by s, and their
       s-by-s block projection matches the hole mask.
    5. Output is the frame bounding box scaled by s, filled with the frame
       color, with the matching pieces placed into the corresponding holes.
    """
    if grid.size == 0:
        return np.array([], dtype=int)

    H, W = grid.shape

    def component_bbox(coords):
        rows = [c[0] for c in coords]
        cols = [c[1] for c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return r0, r1, c0, c1

    def connected_components(g):
        """Return components of same-color 4-connected non-zero pixels."""
        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if g[r, c] != 0 and not seen[r, c]:
                    color = g[r, c]
                    q = deque([(r, c)])
                    seen[r, c] = True
                    coords = []
                    while q:
                        cr, cc = q.popleft()
                        coords.append((cr, cc))
                        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if not seen[nr, nc] and g[nr, nc] == color:
                                    seen[nr, nc] = True
                                    q.append((nr, nc))
                    r0, r1, c0, c1 = component_bbox(coords)
                    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                    for rr, cc in coords:
                        mask[rr - r0, cc - c0] = True
                    comps.append({
                        "color": int(color),
                        "coords": coords,
                        "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                        "h": r1 - r0 + 1,
                        "w": c1 - c0 + 1,
                        "area": len(coords),
                        "mask": mask
                    })
        return comps

    comps = connected_components(grid)
    if not comps:
        return grid.copy()

    # ------------------------------------------------------------------
    # Choose the frame component: full bounding-box border and internal 0s.
    # ------------------------------------------------------------------
    frame_comp = None
    best_frame_score = -1

    for comp in comps:
        color = comp["color"]
        r0, r1, c0, c1 = comp["r0"], comp["r1"], comp["c0"], comp["c1"]
        h, w = comp["h"], comp["w"]
        if h < 3 or w < 3:
            continue

        # border must be all this color
        border_ok = True
        for cc in range(c0, c1 + 1):
            if grid[r0, cc] != color or grid[r1, cc] != color:
                border_ok = False
                break
        if border_ok:
            for rr in range(r0, r1 + 1):
                if grid[rr, c0] != color or grid[rr, c1] != color:
                    border_ok = False
                    break
        if not border_ok:
            continue

        # must contain at least one zero hole inside the bbox
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        holes = (sub == 0).sum()
        if holes == 0:
            continue

        score = h * w
        if score > best_frame_score:
            best_frame_score = score
            frame_comp = comp

    if frame_comp is None:
        # Fallback: largest bounding box if no perfect frame found.
        frame_comp = max(comps, key=lambda c: c["h"] * c["w"])

    frame_color = frame_comp["color"]
    fr0, fr1, fc0, fc1 = frame_comp["r0"], frame_comp["r1"], frame_comp["c0"], frame_comp["c1"]
    FH = fr1 - fr0 + 1
    FW = fc1 - fc0 + 1

    # ------------------------------------------------------------------
    # Identify marker color: non-frame color with most components.
    # ------------------------------------------------------------------
    color_info = {}
    for comp in comps:
        col = comp["color"]
        if col == frame_color:
            continue
        d = color_info.setdefault(col, {"count": 0, "size1": 0, "area": 0})
        d["count"] += 1
        d["area"] += comp["area"]
        if comp["area"] == 1:
            d["size1"] += 1

    marker_color = None
    if color_info:
        marker_color = max(
            color_info.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["size1"], -kv[1]["area"])
        )[0]

    # ------------------------------------------------------------------
    # Hole components inside frame bbox.
    # ------------------------------------------------------------------
    sub = grid[fr0:fr1 + 1, fc0:fc1 + 1]
    hole_mask = (sub == 0)

    hole_comps = []
    hseen = np.zeros(hole_mask.shape, dtype=bool)
    hr, hc = hole_mask.shape

    for r in range(hr):
        for c in range(hc):
            if hole_mask[r, c] and not hseen[r, c]:
                q = deque([(r, c)])
                hseen[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < hr and 0 <= nc < hc:
                            if hole_mask[nr, nc] and not hseen[nr, nc]:
                                hseen[nr, nc] = True
                                q.append((nr, nc))
                r0 = min(x[0] for x in cells)
                r1 = max(x[0] for x in cells)
                c0 = min(x[1] for x in cells)
                c1 = max(x[1] for x in cells)
                mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                for rr, cc in cells:
                    mask[rr - r0, cc - c0] = True
                hole_comps.append({
                    "r": r0, "c": c0,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "mask": mask,
                    "ones": mask.sum()
                })

    if not hole_comps:
        return np.full((FH, FW), frame_color, dtype=int)

    # ------------------------------------------------------------------
    # Object pieces: all components except frame and marker.
    # ------------------------------------------------------------------
    objects = []
    for comp in comps:
        if comp["color"] == frame_color or comp["color"] == marker_color:
            continue
        objects.append(comp)

    def symmetries(mask):
        res = []
        cur = mask
        for _ in range(4):
            res.append(cur)
            res.append(np.fliplr(cur))
            cur = np.rot90(cur)
        uniq = []
        seen = set()
        for m in res:
            t = tuple(m.astype(int).flatten())
            if t not in seen:
                seen.add(t)
                uniq.append(m)
        return uniq

    # Precompute transformed masks for every object.
    object_variants = []
    for oid, obj in enumerate(objects):
        variants = []
        for m in symmetries(obj["mask"]):
            variants.append((m, int(m.shape[0]), int(m.shape[1]), int(m.sum())))
        object_variants.append(variants)

    def find_placement(mask, th, tw, s, max_gap=1):
        """
        Try to place mask into an s*th x s*tw slot so that its s-by-s
        block projection equals the target hole mask.
        Returns (mask, dr, dc) or None.
        """
        target = None  # passed separately in caller? We'll inline target in closure below.
        return None

    def placement_score(mask, slot_h, slot_w, oh, ow, target_ones, s):
        dim_gap = abs(slot_h - oh) + abs(slot_w - ow)
        area_gap = abs(int(mask.sum()) - target_ones * s * s)
        return 1000 - 200 * dim_gap - 5 * area_gap

    def evaluate_scale(s):
        """
        Try to assign objects to holes at scale s.
        Returns (score, assignment).
        assignment: hole_index -> (object_index, mask, dr, dc)
        """
        if s < 2:
            return -10**9, {}

        # For each hole, collect valid object placements.
        hole_options = [[] for _ in hole_comps]

        for hi, hole in enumerate(hole_comps):
            th, tw = hole["h"], hole["w"]
            slot_h, slot_w = s * th, s * tw
            target = hole["mask"].astype(bool)
            target_ones = int(hole["ones"])

            for oi, obj in enumerate(objects):
                for m, oh, ow, marea in object_variants[oi]:
                    if oh > slot_h or ow > slot_w:
                        continue
                    # Pieces should be close to slot size; allow at most one
                    # missing pixel row/column (handles slightly cropped pieces).
                    if slot_h - oh > 1 or slot_w - ow > 1:
                        continue

                    best = None
                    best_key = None
                    # Because gap <= 1, only a few offsets are possible.
                    for dr in range(slot_h - oh + 1):
                        if dr > 1:
                            break
                        for dc in range(slot_w - ow + 1):
                            if dc > 1:
                                continue
                            down = np.zeros((th, tw), dtype=bool)
                            # Project object pixels into s x s blocks.
                            idxs = np.argwhere(m)
                            for i, j in idxs:
                                br = (dr + i) // s
                                bc = (dc + j) // s
                                if 0 <= br < th and 0 <= bc < tw:
                                    down[br, bc] = True
                            if np.array_equal(down, target):
                                key = (dr + dc, dr, dc)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best = (m, dr, dc)
                    if best is not None:
                        m, dr, dc = best
                        sc = placement_score(m, slot_h, slot_w, oh, ow, target_ones, s)
                        hole_options[hi].append((sc, oi, m, dr, dc))

        # Sort holes by fewest options to make DFS fast.
        hole_order = sorted(range(len(hole_comps)), key=lambda i: len(hole_options[i]))
        pos = {orig: idx for idx, orig in enumerate(hole_order)}

        best_score = -10**9
        best_assign = {}
        used = set()

        def dfs(idx, score, assign):
            nonlocal best_score, best_assign
            if idx == len(hole_order):
                if score > best_score:
                    best_score = score
                    best_assign = assign.copy()
                return

            hi = hole_order[idx]
            # Option: leave this hole empty.
            dfs(idx + 1, score, assign)

            # Try each object in descending score order.
            opts = sorted(hole_options[hi], key=lambda x: x[0], reverse=True)
            for sc, oi, m, dr, dc in opts:
                if oi in used:
                    continue
                used.add(oi)
                assign[hi] = (oi, m, dr, dc)
                dfs(idx + 1, score + sc, assign)
                del assign[hi]
                used.remove(oi)

        dfs(0, 0, {})
        return best_score, best_assign

    # ------------------------------------------------------------------
    # Choose scale factor.
    # ------------------------------------------------------------------
    max_s = max(FH, FW, 3)
    best_s = 2
    best_total = -10**9
    best_final_assign = {}

    for s in range(2, max_s + 1):
        score, assign = evaluate_scale(s)
        if score > best_total:
            best_total = score
            best_s = s
            best_final_assign = assign

    # ------------------------------------------------------------------
    # Build output.
    # ------------------------------------------------------------------
    s = best_s
    out = np.full((FH * s, FW * s), frame_color, dtype=int)

    for hi, data in best_final_assign.items():
        oi, m, dr, dc = data
        hole = hole_comps[hi]
        color = objects[oi]["color"]

        orow = hole["r"] * s + dr
        ocol = hole["c"] * s + dc

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    rr = orow + i
                    cc = ocol + j
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                        out[rr, cc] = color

    return out