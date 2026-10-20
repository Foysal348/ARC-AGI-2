import numpy as np
import math

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by finding the largest "band" component,
    locating notches in it, and stacking the other components into those
    notches, interlocking overlaps with larger color winning.
    """
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # ------------------------------------------------------------
    # Connected components of non-zero pixels
    # ------------------------------------------------------------
    def get_components():
        visited = np.zeros((H, W), dtype=bool)
        comps = []

        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0 or visited[r, c]:
                    continue

                color = int(grid[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                pts = []

                while stack:
                    cr, cc = stack.pop()
                    pts.append((cr, cc))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and int(grid[nr, nc]) == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                if not pts:
                    continue

                rmin = min(p[0] for p in pts)
                rmax = max(p[0] for p in pts)
                cmin = min(p[1] for p in pts)
                cmax = max(p[1] for p in pts)
                h = rmax - rmin + 1
                w = cmax - cmin + 1

                mask = np.zeros((h, w), dtype=bool)
                for rr, cc in pts:
                    mask[rr - rmin, cc - cmin] = True

                comps.append({
                    "color": color,
                    "points": pts,
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                    "h": h,
                    "w": w,
                    "area": len(pts),
                    "crow": sum(p[0] for p in pts) / float(len(pts)),
                    "ccol": sum(p[1] for p in pts) / float(len(pts)),
                    "mask": mask,
                })

        return comps

    comps = get_components()
    if not comps:
        return grid.copy()

    # Largest component is treated as the band
    bar = max(comps, key=lambda x: x["area"])
    bar_color = bar["color"]
    other = [c for c in comps if c is not bar]

    out = np.zeros_like(grid)
    bar_mask = np.zeros((H, W), dtype=bool)
    for r, c in bar["points"]:
        bar_mask[r, c] = True
        out[r, c] = bar_color

    if not other:
        return out

    # ------------------------------------------------------------
    # Surface arrays for horizontal band
    # ------------------------------------------------------------
    def compute_horizontal_surfaces():
        topmost = np.full(W, -1, dtype=int)
        bottommost = np.full(W, -1, dtype=int)

        for c in range(W):
            idx = np.where(bar_mask[:, c])[0]
            if len(idx) > 0:
                topmost[c] = int(idx[0])
                bottommost[c] = int(idx[-1])

        valid = topmost >= 0
        if not np.any(valid):
            return topmost, bottommost, 0, 0

        baseline_top = int(np.min(topmost[valid]))
        baseline_bottom = int(np.max(bottommost[valid]))
        return topmost, bottommost, baseline_top, baseline_bottom

    # ------------------------------------------------------------
    # Surface arrays for vertical band
    # ------------------------------------------------------------
    def compute_vertical_surfaces():
        leftmost = np.full(H, -1, dtype=int)
        rightmost = np.full(H, -1, dtype=int)

        for r in range(H):
            idx = np.where(bar_mask[r, :])[0]
            if len(idx) > 0:
                leftmost[r] = int(idx[0])
                rightmost[r] = int(idx[-1])

        valid = leftmost >= 0
        if not np.any(valid):
            return leftmost, rightmost, 0, 0

        base_left = int(np.min(leftmost[valid]))
        base_right = int(np.max(rightmost[valid]))
        return leftmost, rightmost, base_left, base_right

    # ------------------------------------------------------------
    # Find horizontal notch groups
    # ------------------------------------------------------------
    def find_horizontal_notches():
        segments = []

        for r in range(bar["rmin"], bar["rmax"] + 1):
            c = 0
            while c < W:
                if not bar_mask[r, c]:
                    s = c
                    while c < W and not bar_mask[r, c]:
                        c += 1
                    e = c - 1

                    # A usable notch is flanked by band cells on both sides
                    if s > 0 and e < W - 1 and bar_mask[r, s - 1] and bar_mask[r, e + 1]:
                        segments.append((r, s, e))
                else:
                    c += 1

        if not segments:
            return []

        segments.sort(key=lambda x: (x[0], x[1]))
        groups = []

        for seg in segments:
            r, s, e = seg
            match = None

            for g in groups:
                row_ok = any(abs(r - gr) <= 2 for gr in g["rows"])
                col_ok = s <= g["cmax"] + 2 and e >= g["cmin"] - 2
                if row_ok and col_ok:
                    match = g
                    break

            if match is None:
                groups.append({
                    "rows": [r],
                    "cmin": s,
                    "cmax": e,
                    "max_len": e - s + 1,
                    "max_seg": (r, s, e),
                })
            else:
                match["rows"].append(r)
                if e - s + 1 > match["max_len"]:
                    match["max_len"] = e - s + 1
                    match["max_seg"] = (r, s, e)
                match["cmin"] = min(match["cmin"], s)
                match["cmax"] = max(match["cmax"], e)

        res = []
        for g in groups:
            rr, cs, ce = g["max_seg"]
            center = (cs + ce) / 2.0
            avg_r = sum(g["rows"]) / float(len(g["rows"]))
            direction = "top" if avg_r < (bar["rmin"] + bar["rmax"]) / 2.0 else "bottom"
            res.append({
                "dir": direction,
                "c0": cs,
                "c1": ce,
                "center": center,
                "max_len": g["max_len"],
            })

        return res

    # ------------------------------------------------------------
    # Find vertical notch groups
    # ------------------------------------------------------------
    def find_vertical_notches(leftmost, rightmost, base_left, base_right):
        res = []

        # left edge notches
        cur = None
        for r in range(bar["rmin"], bar["rmax"] + 1):
            if leftmost[r] >= 0 and leftmost[r] > base_left:
                if cur is None:
                    cur = [r, r]
                else:
                    if r == cur[1] + 1:
                        cur[1] = r
                    else:
                        res.append({"side": "left", "r0": cur[0], "r1": cur[1],
                                    "max_len": cur[1] - cur[0] + 1,
                                    "center": (cur[0] + cur[1]) / 2.0})
                        cur = [r, r]
            else:
                if cur is not None:
                    res.append({"side": "left", "r0": cur[0], "r1": cur[1],
                                "max_len": cur[1] - cur[0] + 1,
                                "center": (cur[0] + cur[1]) / 2.0})
                    cur = None
        if cur is not None:
            res.append({"side": "left", "r0": cur[0], "r1": cur[1],
                        "max_len": cur[1] - cur[0] + 1,
                        "center": (cur[0] + cur[1]) / 2.0})

        # right edge notches
        cur = None
        for r in range(bar["rmin"], bar["rmax"] + 1):
            if rightmost[r] >= 0 and rightmost[r] < base_right:
                if cur is None:
                    cur = [r, r]
                else:
                    if r == cur[1] + 1:
                        cur[1] = r
                    else:
                        res.append({"side": "right", "r0": cur[0], "r1": cur[1],
                                    "max_len": cur[1] - cur[0] + 1,
                                    "center": (cur[0] + cur[1]) / 2.0})
                        cur = [r, r]
            else:
                if cur is not None:
                    res.append({"side": "right", "r0": cur[0], "r1": cur[1],
                                "max_len": cur[1] - cur[0] + 1,
                                "center": (cur[0] + cur[1]) / 2.0})
                    cur = None
        if cur is not None:
            res.append({"side": "right", "r0": cur[0], "r1": cur[1],
                        "max_len": cur[1] - cur[0] + 1,
                        "center": (cur[0] + cur[1]) / 2.0})

        return res

    # ------------------------------------------------------------
    # Candidate notch positions of a desired length
    # ------------------------------------------------------------
    def horizontal_candidates(g, L):
        res = []
        if g["max_len"] >= L and g["max_len"] <= L + 1:
            for s in range(g["c0"], g["c1"] - L + 2):
                e = s + L - 1
                res.append((g["dir"], (s + e) / 2.0, s, e))
        elif g["max_len"] >= 1 and L - g["max_len"] <= 1:
            res.append((g["dir"], g["center"], g["c0"], g["c1"]))
        return res

    def vertical_candidates(g, L):
        res = []
        if g["max_len"] >= L and g["max_len"] <= L + 1:
            for s in range(g["r0"], g["r1"] - L + 2):
                e = s + L - 1
                res.append(((s + e) / 2.0, s, e))
        elif g["max_len"] >= 1 and L - g["max_len"] <= 1:
            res.append((g["center"], g["r0"], g["r1"]))
        return res

    # ------------------------------------------------------------
    # Anchor positions
    # ------------------------------------------------------------
    def anchor_horizontal(direction, c0, c1, topmost, bottommost, baseline_top, baseline_bottom):
        if direction == "top":
            vals = topmost[c0:c1 + 1]
            vals = vals[vals >= 0]
            if len(vals) == 0:
                return int(baseline_top)
            surface = int(np.max(vals))
            if surface > baseline_top:
                return max(0, surface - 1)
            return int(baseline_top)
        else:
            vals = bottommost[c0:c1 + 1]
            vals = vals[vals >= 0]
            if len(vals) == 0:
                return int(baseline_bottom)
            surface = int(np.min(vals))
            if surface < baseline_bottom:
                return surface
            return int(baseline_bottom)

    def anchor_vertical(side, r0, r1, leftmost, rightmost, base_left, base_right):
        if side == "left":
            vals = [leftmost[r] for r in range(r0, r1 + 1) if leftmost[r] >= 0]
            if not vals:
                return int(base_left)
            surface = max(vals)
            if surface > base_left:
                return max(0, surface - 1)
            return int(base_left)
        else:
            vals = [rightmost[r] for r in range(r0, r1 + 1) if rightmost[r] >= 0]
            if not vals:
                return int(base_right)
            surface = min(vals)
            if surface < base_right:
                return min(W - 1, surface + 1)
            return int(base_right)

    # ------------------------------------------------------------
    # Overlap scoring for interlocking
    # ------------------------------------------------------------
    def overlap_score(prev_bbox, prev_mask, next_bbox, next_mask):
        r0 = max(prev_bbox[0], next_bbox[0])
        r1 = min(prev_bbox[2], next_bbox[2])
        c0 = max(prev_bbox[1], next_bbox[1])
        c1 = min(prev_bbox[3], next_bbox[3])

        if r0 > r1 or c0 > c1:
            return 0, 0

        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        fit = 0
        conf = 0

        for r in range(r0, r1 + 1):
            pr = r - prev_bbox[0]
            nr = r - next_bbox[0]
            for c in range(c0, c1 + 1):
                pc = c - prev_bbox[1]
                nc = c - next_bbox[1]
                p = bool(prev_mask[pr, pc])
                n = bool(next_mask[nr, nc])
                if p and n:
                    conf += 1
                elif (not p) and n:
                    fit += 1

        return fit - conf, area

    def clamp_bbox(top, left, bottom, right):
        if top < 0:
            shift = -top
            top += shift
            bottom += shift
        if bottom >= H:
            shift = bottom - H + 1
            top -= shift
            bottom = H - 1
        if left < 0:
            shift = -left
            left += shift
            right += shift
        if right >= W:
            shift = right - W + 1
            left -= shift
            right = W - 1

        top = max(0, top)
        bottom = min(H - 1, bottom)
        left = max(0, left)
        right = min(W - 1, right)
        return top, left, bottom, right

    # ------------------------------------------------------------
    # Overlay one component
    # ------------------------------------------------------------
    def overlay_component(comp, top, left):
        h, w = comp["h"], comp["w"]
        m = comp["mask"]
        color = comp["color"]

        for i in range(h):
            r = top + i
            if r < 0 or r >= H:
                continue
            row = m[i]
            if not np.any(row):
                continue
            cols = np.nonzero(row)[0]
            for cc in cols:
                c = left + cc
                if c < 0 or c >= W:
                    continue
                cur = out[r, c]
                if cur == 0 or cur == bar_color or color > cur:
                    out[r, c] = color

    # ------------------------------------------------------------
    # Place a stack of components
    # ------------------------------------------------------------
    def place_stack(st):
        orient = st["orient"]
        direction = st["dir"]
        center = st["center"]
        anchor = st["anchor"]

        if orient == "H":
            comps = sorted(st["comps"], key=lambda x: (abs(x["ccol"] - center), x["crow"], x["ccol"]))
        else:
            comps = sorted(st["comps"], key=lambda x: (abs(x["crow"] - center), x["ccol"], x["row"]))

        prev = None
        prev_mask = None

        for comp in comps:
            h = comp["h"]
            w = comp["w"]

            if orient == "H":
                left = int(math.floor(center)) - (w - 1) // 2
            else:
                top = int(math.floor(center)) - (h - 1) // 2

            if prev is None:
                if direction == "top":
                    bottom = anchor
                    top = bottom - h + 1
                elif direction == "bottom":
                    top = anchor
                    bottom = top + h - 1
                elif direction == "left":
                    right = anchor
                    left = right - w + 1
                else:  # right
                    left = anchor
                    right = left + w - 1

            else:
                if direction == "top":
                    e = prev[0]
                    span = min(prev[3] - prev[0] + 1, h) + 2
                    best = None
                    for t in range(e - span, e + span + 1):
                        bottom = t
                        top = bottom - h + 1
                        if top < 0 or bottom >= H:
                            continue
                        right = left + w - 1
                        if right >= W:
                            continue
                        bbox = (top, left, bottom, right)
                        score, area = overlap_score(prev, prev_mask, bbox, comp["mask"])
                        if area == 0:
                            continue
                        dist = abs(t - e)
                        if best is None or score > best[0] or (score == best[0] and dist < best[1]):
                            best = (score, dist, t)

                    if best is None:
                        t = e
                        bottom = t
                        top = bottom - h + 1
                    else:
                        t = best[2]
                        bottom = t
                        top = bottom - h + 1

                elif direction == "bottom":
                    e = prev[2]
                    span = min(prev[3] - prev[0] + 1, h) + 2
                    best = None
                    for t in range(e - span, e + span + 1):
                        top = t
                        bottom = top + h - 1
                        if top < 0 or bottom >= H:
                            continue
                        right = left + w - 1
                        if right >= W:
                            continue
                        bbox = (top, left, bottom, right)
                        score, area = overlap_score(prev, prev_mask, bbox, comp["mask"])
                        if area == 0:
                            continue
                        dist = abs(t - e)
                        if best is None or score > best[0] or (score == best[0] and dist < best[1]):
                            best = (score, dist, t)

                    if best is None:
                        t = e
                        top = t
                        bottom = top + h - 1
                    else:
                        t = best[2]
                        top = t
                        bottom = top + h - 1

                elif direction == "left":
                    e = prev[1]
                    span = min(prev[3] - prev[1] + 1, w) + 2
                    best = None
                    for t in range(e - span, e + span + 1):
                        right = t
                        left = right - w + 1
                        if left < 0 or right >= W:
                            continue
                        bottom = top + h - 1
                        if top < 0 or bottom >= H:
                            continue
                        bbox = (top, left, bottom, right)
                        score, area = overlap_score(prev, prev_mask, bbox, comp["mask"])
                        if area == 0:
                            continue
                        dist = abs(t - e)
                        if best is None or score > best[0] or (score == best[0] and dist < best[1]):
                            best = (score, dist, t)

                    if best is None:
                        t = e
                        right = t
                        left = right - w + 1
                    else:
                        t = best[2]
                        right = t
                        left = right - w + 1

                else:  # right
                    e = prev[3]
                    span = min(prev[3] - prev[1] + 1, w) + 2
                    best = None
                    for t in range(e - span, e + span + 1):
                        left = t
                        right = left + w - 1
                        if left < 0 or right >= W:
                            continue
                        bottom = top + h - 1
                        if top < 0 or bottom >= H:
                            continue
                        bbox = (top, left, bottom, right)
                        score, area = overlap_score(prev, prev_mask, bbox, comp["mask"])
                        if area == 0:
                            continue
                        dist = abs(t - e)
                        if best is None or score > best[0] or (score == best[0] and dist < best[1]):
                            best = (score, dist, t)

                    if best is None:
                        t = e
                        left = t
                        right = left + w - 1
                    else:
                        t = best[2]
                        left = t
                        right = left + w - 1

            top, left, bottom, right = clamp_bbox(top, left, bottom, right)
            if top > bottom or left > right:
                continue

            overlay_component(comp, top, left)
            prev = (top, left, bottom, right)
            prev_mask = comp["mask"]

    # ------------------------------------------------------------
    # Build stacks based on orientation
    # ------------------------------------------------------------
    stacks = []

    if bar["w"] >= bar["h"]:
        # Horizontal band
        topmost, bottommost, baseline_top, baseline_bottom = compute_horizontal_surfaces()
        hgroups = find_horizontal_notches()

        width_groups = {}
        for comp in other:
            width_groups.setdefault(comp["w"], []).append(comp)

        for w, comps in width_groups.items():
            L = max(1, w - 2)
            best = None

            for g in hgroups:
                for direction, center, c0, c1 in horizontal_candidates(g, L):
                    score = sum(abs(c["ccol"] - center) for c in comps)
                    if best is None or score < best[0] - 1e-9:
                        best = (score, direction, center, c0, c1)

            if best is None:
                direction = "bottom"
                center = bar["ccol"]
                c0 = c1 = int(round(center))
            else:
                _, direction, center, c0, c1 = best

            anchor = anchor_horizontal(direction, c0, c1,
                                       topmost, bottommost,
                                       baseline_top, baseline_bottom)

            stacks.append({
                "orient": "H",
                "dir": direction,
                "center": center,
                "anchor": anchor,
                "comps": comps,
            })

    else:
        # Vertical band
        leftmost, rightmost, base_left, base_right = compute_vertical_surfaces()
        vgroups = find_vertical_notches(leftmost, rightmost, base_left, base_right)

        side_height_groups = {}
        bar_cx = (bar["cmin"] + bar["cmax"]) / 2.0
        for comp in other:
            side = "left" if comp["ccol"] < bar_cx else "right"
            side_height_groups.setdefault((side, comp["h"]), []).append(comp)

        for (side, h), comps in side_height_groups.items():
            L = max(1, h - 2)
            best = None

            for g in vgroups:
                if g["side"] != side:
                    continue
                for center, r0, r1 in vertical_candidates(g, L):
                    score = sum(abs(c["crow"] - center) for c in comps)
                    if best is None or score < best[0] - 1e-9:
                        best = (score, center, r0, r1)

            if best is None:
                center = bar["crow"]
                r0 = r1 = int(round(center))
            else:
                _, center, r0, r1 = best

            anchor = anchor_vertical(side, r0, r1,
                                     leftmost, rightmost,
                                     base_left, base_right)

            stacks.append({
                "orient": "V",
                "dir": side,
                "center": center,
                "anchor": anchor,
                "comps": comps,
            })

    # Merge stacks that target the same notch
    merged = {}
    for st in stacks:
        key = (st["orient"], st["dir"], round(st["center"], 2))
        if key not in merged:
            merged[key] = st
        else:
            merged[key]["comps"].extend(st["comps"])
    stacks = list(merged.values())

    for st in stacks:
        place_stack(st)

    return out