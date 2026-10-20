import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move objects containing a marker color so that their special point aligns
    with the matching isolated border marker.

    The implementation uses line segments:
      - border single pixels are markers;
      - objects are 8-connected non-background components;
      - the special color is the rarest marker color present in the object;
      - maximal horizontal/vertical frame segments are built;
      - segments containing the special point move;
      - segments in 2-core cycles move rigidly;
      - other segments move only if needed to keep intersections/avoid bad overlaps.
    """
    from collections import defaultdict, deque

    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = grid.copy()

    nonbg = grid != bg
    comp_id = np.full((H, W), -1, dtype=int)
    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # ---------------------------------------------------------------
    # 1. Find 8-connected non-background components.
    # ---------------------------------------------------------------
    all_comps = []
    for r in range(H):
        for c in range(W):
            if nonbg[r, c] and comp_id[r, c] == -1:
                cid = len(all_comps)
                q = deque([(r, c)])
                comp_id[r, c] = cid
                pixels = []

                while q:
                    rr, cc = q.popleft()
                    pixels.append((rr, cc))

                    for dr, dc in dirs8:
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W and
                                nonbg[nr, nc] and comp_id[nr, nc] == -1):
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))

                color_counts = {}
                for rr, cc in pixels:
                    v = int(grid[rr, cc])
                    color_counts[v] = color_counts.get(v, 0) + 1

                touches_border = any(
                    rr == 0 or rr == H - 1 or cc == 0 or cc == W - 1
                    for rr, cc in pixels
                )

                all_comps.append((pixels, color_counts, touches_border))

    # ---------------------------------------------------------------
    # 2. Markers are isolated border pixels.
    # ---------------------------------------------------------------
    markers = defaultdict(list)
    object_comps = []

    for pixels, ccounts, border in all_comps:
        if len(pixels) == 1 and border:
            r, c = pixels[0]
            markers[int(grid[r, c])].append((r, c))
        else:
            object_comps.append((pixels, ccounts, border))

    # ---------------------------------------------------------------
    # Helper functions.
    # ---------------------------------------------------------------
    def classify_marker_axis(r, c):
        left_right = (c == 0 or c == W - 1)
        top_bottom = (r == 0 or r == H - 1)

        if left_right and not top_bottom:
            return "v"          # align row
        if top_bottom and not left_right:
            return "h"          # align column
        if left_right and top_bottom:
            return "h"          # corner: choose one consistently
        return None

    def get_anchor(pixels, special_color):
        special_pixels = [(r, c) for r, c in pixels if int(grid[r, c]) == special_color]
        if not special_pixels:
            return None
        if len(special_pixels) == 1:
            return special_pixels[0]

        cr = sum(r for r, c in pixels) / len(pixels)
        cc = sum(c for r, c in pixels) / len(pixels)
        ar = int(np.floor(cr + 0.5))
        ac = int(np.floor(cc + 0.5))
        ar = min(max(ar, 0), H - 1)
        ac = min(max(ac, 0), W - 1)
        return (ar, ac)

    def choose_target(anchor, special_color):
        best = None
        for mr, mc in markers.get(special_color, []):
            axis = classify_marker_axis(mr, mc)
            if axis is None:
                continue

            if axis == "v":
                target = (mr, anchor[1])
                side = "left" if mc == 0 else ("right" if mc == W - 1 else None)
            else:
                target = (anchor[0], mc)
                side = "top" if mr == 0 else ("bottom" if mr == H - 1 else None)

            dist = abs(target[0] - anchor[0]) + abs(target[1] - anchor[1])
            if best is None or dist < best[3]:
                best = (target, axis, side, dist)

        return best

    def is_centered_3x3_frame(pixels, anchor):
        rs = [r for r, c in pixels]
        cs = [c for r, c in pixels]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        if r1 - r0 != 2 or c1 - c0 != 2:
            return False
        if anchor != (r0 + 1, c0 + 1):
            return False

        pset = set(pixels)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = anchor[0] + dr, anchor[1] + dc
                if (nr, nc) not in pset:
                    return False
        return True

    def apply_rigid(pixels, dr, dc):
        for r, c in pixels:
            out[r, c] = bg

        for r, c in pixels:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = int(grid[r, c])

    def build_segments(pixels_set, special_color):
        frame_colors = set()
        for r, c in pixels_set:
            v = int(grid[r, c])
            if v != bg and v != special_color:
                frame_colors.add(v)

        segs = []

        # Horizontal segments.
        for F in frame_colors:
            for r in range(H):
                c = 0
                while c < W:
                    if (r, c) in pixels_set and (int(grid[r, c]) == F or int(grid[r, c]) == special_color):
                        start = c
                        c2 = c
                        hasF = False
                        hasS = False

                        while c2 < W and (r, c2) in pixels_set and (
                                int(grid[r, c2]) == F or int(grid[r, c2]) == special_color):
                            v = int(grid[r, c2])
                            if v == F:
                                hasF = True
                            elif v == special_color:
                                hasS = True
                            c2 += 1

                        if hasF:
                            frame_cells = [(r, cc) for cc in range(start, c2) if int(grid[r, cc]) == F]
                            segs.append({
                                "orient": "h",
                                "fixed": r,
                                "c1": start,
                                "c2": c2 - 1,
                                "color": F,
                                "frame_cells": frame_cells,
                                "contains": hasS,
                            })

                        c = c2
                    else:
                        c += 1

        # Vertical segments.
        for F in frame_colors:
            for c in range(W):
                r = 0
                while r < H:
                    if (r, c) in pixels_set and (int(grid[r, c]) == F or int(grid[r, c]) == special_color):
                        start = r
                        r2 = r
                        hasF = False
                        hasS = False

                        while r2 < H and (r2, c) in pixels_set and (
                                int(grid[r2, c]) == F or int(grid[r2, c]) == special_color):
                            v = int(grid[r2, c])
                            if v == F:
                                hasF = True
                            elif v == special_color:
                                hasS = True
                            r2 += 1

                        if hasF:
                            frame_cells = [(rr, c) for rr in range(start, r2) if int(grid[rr, c]) == F]
                            segs.append({
                                "orient": "v",
                                "fixed": c,
                                "r1": start,
                                "r2": r2 - 1,
                                "color": F,
                                "frame_cells": frame_cells,
                                "contains": hasS,
                            })

                        r = r2
                    else:
                        r += 1

        for i, s in enumerate(segs):
            s["id"] = i
        return segs

    def apply_segment_move(pixels, segs, moved, dr, dc, anchor, special_color):
        for r, c in pixels:
            out[r, c] = bg

        # Static segments first.
        for s in segs:
            if s["id"] not in moved:
                for r, c in s["frame_cells"]:
                    out[r, c] = s["color"]

        # Moved segments.
        for s in segs:
            if s["id"] in moved:
                for r, c in s["frame_cells"]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = s["color"]

        sr, sc = anchor[0] + dr, anchor[1] + dc
        if 0 <= sr < H and 0 <= sc < W:
            out[sr, sc] = special_color

    # ---------------------------------------------------------------
    # Process each object.
    # ---------------------------------------------------------------
    for pixels, ccounts, _ in object_comps:
        candidate_specials = [col for col in ccounts if col in markers]
        if not candidate_specials:
            continue

        special_color = min(candidate_specials, key=lambda col: (ccounts[col], col))
        anchor = get_anchor(pixels, special_color)
        if anchor is None:
            continue

        target_info = choose_target(anchor, special_color)
        if target_info is None:
            continue

        target, axis, side, _ = target_info
        dr = target[0] - anchor[0]
        dc = target[1] - anchor[1]

        # Extra one-cell shift for centered 3x3 frames with side markers.
        if axis == "v" and side in ("left", "right"):
            if is_centered_3x3_frame(pixels, anchor):
                extra = -1 if side == "left" else 1
                if 0 <= anchor[1] + extra < W and all(0 <= c + extra < W for _, c in pixels):
                    dc += extra

        special_pixels = [(r, c) for r, c in pixels if int(grid[r, c]) == special_color]
        applied = False

        if len(special_pixels) == 1:
            pixels_set = set(pixels)
            segs = build_segments(pixels_set, special_color)

            if segs:
                final_special = (anchor[0] + dr, anchor[1] + dc)

                # Implications: if segment i moves and j stays is invalid, i -> j.
                n = len(segs)
                imp = [set() for _ in range(n)]
                adj = [set() for _ in range(n)]

                def seg_coords(s, move):
                    sdr = dr if move else 0
                    sdc = dc if move else 0
                    if s["orient"] == "h":
                        return (s["fixed"] + sdr, s["c1"] + sdc, s["c2"] + sdc)
                    else:
                        return (s["fixed"] + sdc, s["r1"] + sdr, s["r2"] + sdr)

                def hv_intersect(h, v, mh, mv):
                    hrow, hc1, hc2 = seg_coords(h, mh)
                    vcol, vr1, vr2 = seg_coords(v, mv)
                    return vr1 <= hrow <= vr2 and hc1 <= vcol <= hc2

                def parallel_overlap_invalid(s1, s2, m1, m2):
                    if s1["orient"] != s2["orient"] or s1["color"] != s2["color"]:
                        return False

                    if s1["orient"] == "h":
                        r1, c1a, c1b = seg_coords(s1, m1)
                        r2, c2a, c2b = seg_coords(s2, m2)
                        if r1 != r2:
                            return False
                        lo = max(c1a, c2a)
                        hi = min(c1b, c2b)
                        if lo > hi:
                            return False
                        length = hi - lo + 1
                        if length > 1:
                            return True
                        return (r1, lo) != final_special
                    else:
                        c1, r1a, r1b = seg_coords(s1, m1)
                        c2, r2a, r2b = seg_coords(s2, m2)
                        if c1 != c2:
                            return False
                        lo = max(r1a, r2a)
                        hi = min(r1b, r2b)
                        if lo > hi:
                            return False
                        length = hi - lo + 1
                        if length > 1:
                            return True
                        return (lo, c1) != final_special

                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = segs[i], segs[j]

                        if a["orient"] != b["orient"]:
                            if a["orient"] == "h":
                                h, v = a, b
                                hi, vi = i, j
                            else:
                                h, v = b, a
                                hi, vi = j, i

                            if hv_intersect(h, v, False, False):
                                adj[hi].add(vi)
                                adj[vi].add(hi)

                                if not hv_intersect(h, v, True, False):
                                    imp[hi].add(vi)
                                if not hv_intersect(h, v, False, True):
                                    imp[vi].add(hi)

                        else:
                            if parallel_overlap_invalid(a, b, True, False):
                                imp[i].add(j)
                            if parallel_overlap_invalid(a, b, False, True):
                                imp[j].add(i)

                # 2-core: closed line structures move rigidly.
                if n:
                    in_core = [True] * n
                    degree = [len(adj[i]) for i in range(n)]
                    q = deque([i for i, d in enumerate(degree) if d <= 1])

                    while q:
                        u = q.popleft()
                        if not in_core[u]:
                            continue
                        in_core[u] = False
                        for v in adj[u]:
                            if in_core[v]:
                                degree[v] -= 1
                                if degree[v] <= 1:
                                    q.append(v)

                    seen = set()
                    for i in range(n):
                        if in_core[i] and i not in seen:
                            stack = [i]
                            seen.add(i)
                            core_comp = []
                            while stack:
                                u = stack.pop()
                                core_comp.append(u)
                                for v in adj[u]:
                                    if in_core[v] and v not in seen:
                                        seen.add(v)
                                        stack.append(v)

                            for a in core_comp:
                                for b in core_comp:
                                    if a != b:
                                        imp[a].add(b)

                required = [s["id"] for s in segs if s["contains"]]
                if required:
                    moved = set(required)
                    stack = list(required)

                    while stack:
                        u = stack.pop()
                        for v in imp[u]:
                            if v not in moved:
                                moved.add(v)
                                stack.append(v)

                    apply_segment_move(pixels, segs, moved, dr, dc, anchor, special_color)
                    applied = True

        if not applied:
            apply_rigid(pixels, dr, dc)

    # Restore markers in case objects moved over them.
    for color, positions in markers.items():
        for r, c in positions:
            out[r, c] = color

    return out