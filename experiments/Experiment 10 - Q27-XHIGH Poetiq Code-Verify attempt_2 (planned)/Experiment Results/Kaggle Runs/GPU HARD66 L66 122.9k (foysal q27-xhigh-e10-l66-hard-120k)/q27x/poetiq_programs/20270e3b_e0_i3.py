import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Join two 4-colored objects whose sides are marked by 7 pixels.

    The 7 pixels indicate which side of each object should be attached to
    the other object.  One object is moved so that its marked side becomes
    adjacent to the marked side of the other object.  All 7 pixels are
    removed, and the result is cropped to the bounding box of color 4.
    """

    def crop_to_4(g):
        """Crop to the bounding box of all 4 pixels."""
        if g.size == 0:
            return g.copy()
        if not np.any(g == 4):
            return g.copy()
        rows = np.where((g == 4).any(axis=1))[0]
        cols = np.where((g == 4).any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return g.copy()
        return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()

    a = np.array(grid, dtype=int)
    if a.size == 0:
        return a.copy()
    if a.ndim != 2:
        return a.copy()

    H, W = a.shape

    # If there are no markers, nothing needs to be joined.
    if not np.any(a == 7):
        return a.copy()

    # Background is the most common color that is neither 4 nor 7.
    vals, counts = np.unique(a, return_counts=True)
    bg_mask = (vals != 4) & (vals != 7)
    if np.any(bg_mask):
        bg_idx = np.where(bg_mask)[0]
        bg = int(vals[bg_idx[np.argmax(counts[bg_idx])]])
    else:
        bg = 0

    # Base grid with markers removed.
    base = a.copy()
    base[base == 7] = bg

    # If there are no 4 objects, just return marker-removed grid.
    if not np.any(a == 4):
        return base

    # Find 4-connected components of color 4.
    comp_id = np.full((H, W), -1, dtype=int)
    comps = []
    bboxes = []
    moves4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(H):
        for c in range(W):
            if a[r, c] == 4 and comp_id[r, c] == -1:
                cid = len(comps)
                cells = []
                stack = [(r, c)]
                comp_id[r, c] = cid

                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in moves4:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < H
                            and 0 <= ny < W
                            and a[nx, ny] == 4
                            and comp_id[nx, ny] == -1
                        ):
                            comp_id[nx, ny] = cid
                            stack.append((nx, ny))

                rs = [x for x, _ in cells]
                cs = [y for _, y in cells]
                comps.append(cells)
                bboxes.append((min(rs), min(cs), max(rs), max(cs)))

    ncomps = len(comps)

    # Assign each 7 marker to every adjacent 4 component.
    comp_markers = [[] for _ in comps]
    for r in range(H):
        for c in range(W):
            if a[r, c] == 7:
                touched = set()
                for dx, dy in moves4:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 4:
                        cid = comp_id[nr, nc]
                        if cid >= 0:
                            touched.add(cid)
                for cid in touched:
                    comp_markers[cid].append((r, c))

    # Side directions are component-to-marker directions.
    dir_vectors = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    opposite = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }
    all_sides = ["top", "bottom", "left", "right"]

    def is_outside(r, c):
        return r < 0 or r >= H or c < 0 or c >= W

    def compute_side_info(cid, markers):
        """
        Determine likely marked sides for one component.

        A side is preferred when a marker has the component on one side and
        background/outside on the opposite side.
        """
        scores = {s: 0.0 for s in all_sides}
        marker_sets = {s: [] for s in all_sides}
        r1, c1, r2, c2 = bboxes[cid]

        # Primary method: local background/outside neighbor.
        for r, c in markers:
            comp_adj = set()
            for name, (dx, dy) in dir_vectors.items():
                nr, nc = r + dx, c + dy
                if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 4 and comp_id[nr, nc] == cid:
                    # direction from marker to component
                    comp_adj.add(name)

            for out_name in all_sides:
                in_name = opposite[out_name]
                if in_name in comp_adj:
                    dx, dy = dir_vectors[out_name]
                    nr, nc = r + dx, c + dy
                    if is_outside(nr, nc):
                        scores[out_name] += 1.0
                        if (r, c) not in marker_sets[out_name]:
                            marker_sets[out_name].append((r, c))
                    else:
                        val = a[nr, nc]
                        if val != 4 and val != 7:
                            scores[out_name] += 1.0
                            if (r, c) not in marker_sets[out_name]:
                                marker_sets[out_name].append((r, c))

        # Fallback: markers completely outside component bbox.
        if not any(scores.values()):
            mrs = [r for r, _ in markers]
            mcs = [c for _, c in markers]
            if max(mrs) < r1:
                scores["top"] = 10.0
                marker_sets["top"] = list(markers)
            if min(mrs) > r2:
                scores["bottom"] = 10.0
                marker_sets["bottom"] = list(markers)
            if max(mcs) < c1:
                scores["left"] = 10.0
                marker_sets["left"] = list(markers)
            if min(mcs) > c2:
                scores["right"] = 10.0
                marker_sets["right"] = list(markers)

        # Fallback: use all component adjacencies.
        if not any(scores.values()):
            for r, c in markers:
                for name, (dx, dy) in dir_vectors.items():
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 4 and comp_id[nr, nc] == cid:
                        out_name = opposite[name]
                        scores[out_name] += 0.5
                        if (r, c) not in marker_sets[out_name]:
                            marker_sets[out_name].append((r, c))

        # Final fallback: centroid direction.
        if not any(scores.values()):
            mr = float(np.mean([r for r, _ in markers]))
            mc = float(np.mean([c for _, c in markers]))
            cr = (r1 + r2) / 2.0
            cc = (c1 + c2) / 2.0

            if mr < cr:
                scores["top"] = 1.0
            elif mr > cr:
                scores["bottom"] = 1.0
            if mc < cc:
                scores["left"] = 1.0
            elif mc > cc:
                scores["right"] = 1.0
            if not any(scores.values()):
                scores["top"] = 1.0

            for s in all_sides:
                if not marker_sets[s]:
                    marker_sets[s] = list(markers)

        return {
            s: {"score": scores[s], "markers": marker_sets[s] if marker_sets[s] else list(markers)}
            for s in all_sides
            if scores[s] > 0
        }

    side_info = [{} for _ in comps]
    for cid in range(ncomps):
        if comp_markers[cid]:
            side_info[cid] = compute_side_info(cid, comp_markers[cid])

    marked = [cid for cid in range(ncomps) if side_info[cid]]
    if len(marked) < 2:
        return crop_to_4(base)

    def side_main_ref(side, markers):
        """
        Reference coordinate along the marked side's main axis.

        For top/bottom sides this is the marker row closest to the object.
        For left/right sides this is the marker column closest to the object.
        """
        if not markers:
            return 0
        if side in ("top", "bottom"):
            rows = [r for r, _ in markers]
            return max(rows) if side == "top" else min(rows)
        else:
            cols = [c for _, c in markers]
            return max(cols) if side == "left" else min(cols)

    def side_perp_ref(side, markers):
        """Reference coordinate perpendicular to the side, using top/left edge."""
        if not markers:
            return 0
        if side in ("top", "bottom"):
            return min(c for _, c in markers)
        else:
            return min(r for r, _ in markers)

    def get_info(cid, side):
        if side in side_info[cid]:
            return side_info[cid][side]
        return {"score": 0.0, "markers": list(comp_markers[cid])}

    def choose_source(ci, cj, si, sj, axis):
        """
        Choose which of the two components to move.

        Prefer the smaller object.  For ties, prefer the lower object in
        vertical joins or the righter object in horizontal joins.
        """
        ni = len(comps[ci])
        nj = len(comps[cj])

        if ni < nj:
            return ci, cj
        if ni > nj:
            return cj, ci

        if axis == "v":
            ri = side_main_ref(si, get_info(ci, si)["markers"])
            rj = side_main_ref(sj, get_info(cj, sj)["markers"])
            if ri != rj:
                return (ci, cj) if ri > rj else (cj, ci)
        else:
            ci_ref = side_main_ref(si, get_info(ci, si)["markers"])
            cj_ref = side_main_ref(sj, get_info(cj, sj)["markers"])
            if ci_ref != cj_ref:
                return (ci, cj) if ci_ref > cj_ref else (cj, ci)

        # Final tie-breaker: lower/right bounding box.
        bi = bboxes[ci]
        bj = bboxes[cj]
        if axis == "v":
            if bi[2] != bj[2]:
                return (ci, cj) if bi[2] > bj[2] else (cj, ci)
        else:
            if bi[3] != bj[3]:
                return (ci, cj) if bi[3] > bj[3] else (cj, ci)

        return ci, cj

    def compute_offset(src, src_side, tgt, tgt_side):
        """
        Translation that places the source object's marked edge adjacent to
        the target object's marked edge.
        """
        src_markers = get_info(src, src_side)["markers"]
        tgt_markers = get_info(tgt, tgt_side)["markers"]
        axis = "v" if src_side in ("top", "bottom") else "h"

        if axis == "v":
            src_main = side_main_ref(src_side, src_markers)
            tgt_main = side_main_ref(tgt_side, tgt_markers)
            d_src = -1 if src_side == "top" else 1
            dr = tgt_main - src_main + d_src
            dc = side_perp_ref(tgt_side, tgt_markers) - side_perp_ref(src_side, src_markers)
        else:
            src_main = side_main_ref(src_side, src_markers)
            tgt_main = side_main_ref(tgt_side, tgt_markers)
            d_src = -1 if src_side == "left" else 1
            dc = tgt_main - src_main + d_src
            dr = side_perp_ref(tgt_side, tgt_markers) - side_perp_ref(src_side, src_markers)

        return dr, dc

    def estimate_area(src, dr, dc):
        """Approximate final 4-bounding-box area, used as a tie-breaker."""
        min_r = min_c = None
        max_r = max_c = None

        for cid, (r1, c1, r2, c2) in enumerate(bboxes):
            if cid == src:
                rr1, rr2 = r1 + dr, r2 + dr
                cc1, cc2 = c1 + dc, c2 + dc
            else:
                rr1, rr2 = r1, r2
                cc1, cc2 = c1, c2

            if min_r is None or rr1 < min_r:
                min_r = rr1
            if max_r is None or rr2 > max_r:
                max_r = rr2
            if min_c is None or cc1 < min_c:
                min_c = cc1
            if max_c is None or cc2 > max_c:
                max_c = cc2

        if min_r is None:
            return 0
        return (max_r - min_r + 1) * (max_c - min_c + 1)

    best = None
    best_score = None

    # Consider all marked pairs and all opposite-side assignments.
    for i in range(len(marked)):
        for j in range(i + 1, len(marked)):
            ci, cj = marked[i], marked[j]

            for si in all_sides:
                sj = opposite[si]

                info_i = get_info(ci, si)
                info_j = get_info(cj, sj)

                score = info_i["score"] + info_j["score"]
                axis = "v" if si in ("top", "bottom") else "h"

                src, tgt = choose_source(ci, cj, si, sj, axis)
                src_side = si if src == ci else sj
                tgt_side = sj if src == ci else si

                dr, dc = compute_offset(src, src_side, tgt, tgt_side)
                area = estimate_area(src, dr, dc)

                key = (score, -area, -abs(dr + dc))
                if best_score is None or key > best_score:
                    best_score = key
                    best = (src, src_side, tgt, tgt_side, dr, dc)

    if best is None:
        return crop_to_4(base)

    src, src_side, tgt, tgt_side, dr, dc = best
    src_cells = comps[src]

    if not src_cells:
        return crop_to_4(base)

    # Remove source object from its original location.
    work = base.copy()
    for r, c in src_cells:
        work[r, c] = bg

    # Canvas large enough for original grid plus moved source object.
    min_r, max_r = 0, H - 1
    min_c, max_c = 0, W - 1
    for r, c in src_cells:
        nr, nc = r + dr, c + dc
        min_r = min(min_r, nr)
        max_r = max(max_r, nr)
        min_c = min(min_c, nc)
        max_c = max(max_c, nc)

    new_H = max_r - min_r + 1
    new_W = max_c - min_c + 1
    new_grid = np.full((new_H, new_W), bg, dtype=int)

    off_r = -min_r
    off_c = -min_c
    new_grid[off_r:off_r + H, off_c:off_c + W] = work

    # Place moved source object.
    for r, c in src_cells:
        nr, nc = off_r + r + dr, off_c + c + dc
        if 0 <= nr < new_H and 0 <= nc < new_W:
            new_grid[nr, nc] = 4

    return crop_to_4(new_grid)