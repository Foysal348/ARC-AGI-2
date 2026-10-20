import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by finding an "anchor" component whose bounding-box notch matches
    a second component (the "filler"). A third adjacent component (the "extra")
    is moved together with the filler.

    Rule:
    1. Background is the most frequent color.
    2. Extract 4-connected non-background components.
    3. For each anchor, take the background cells inside its bounding box,
       crop them to a minimal mask: the notch.
    4. If another component matches this notch under a dihedral symmetry, it is
       the filler. A third component adjacent to the filler is the extra.
    5. The anchor stays fixed. The filler is placed into the anchor notch.
       The extra is placed next to the anchor.
       - If the filler was placed without rotation/reflection, the extra is
         translated by the same delta.
       - Otherwise, the extra is transformed/aligned to the side suggested by
         its original relative position to the filler, using the anchor's
         feature line for alignment.
    6. If the notch is bottom-heavy, swap the output colors of filler and extra.

    A small fallback handles the exact three-piece assembly style if no exact
    notch match is present.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    bg = int(np.bincount(grid.ravel()).argmax())
    if not np.any(grid != bg):
        return grid

    def crop_mask(m):
        """Crop a boolean mask to its minimal non-empty bounding box."""
        if m.size == 0:
            return np.zeros((0, 0), dtype=bool)
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros((0, 0), dtype=bool)
        return m[np.ix_(rows, cols)]

    def all_sims(mask):
        """
        Return all 8 dihedral symmetries of a mask, each as:
        (cropped_mask, rot, flip)
        rot is counterclockwise 90-degree count, flip is horizontal flip.
        """
        res = []
        seen = set()
        for rot in range(4):
            for flip in (False, True):
                m = np.rot90(mask, k=rot)
                if flip:
                    m = np.fliplr(m)
                m = crop_mask(m)
                if m.size == 0:
                    continue
                key = (m.shape, m.tobytes(), rot, flip)
                if key not in seen:
                    seen.add(key)
                    res.append((m, rot, flip))
        return res

    def get_components():
        """Extract 4-connected non-background components."""
        visited = np.zeros((H, W), dtype=bool)
        comps = []
        for r0 in range(H):
            for c0 in range(W):
                if visited[r0, c0] or grid[r0, c0] == bg:
                    continue
                color = int(grid[r0, c0])
                stack = [(r0, c0)]
                visited[r0, c0] = True
                pix = []
                while stack:
                    r, c = stack.pop()
                    pix.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                rs = [p[0] for p in pix]
                cs = [p[1] for p in pix]
                rmin, rmax = min(rs), max(rs)
                cmin, cmax = min(cs), max(cs)
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                mask = np.zeros((h, w), dtype=bool)
                for r, c in pix:
                    mask[r - rmin, c - cmin] = True
                comps.append({
                    "id": len(comps),
                    "color": color,
                    "pixels": pix,
                    "r0": int(rmin),
                    "c0": int(cmin),
                    "h": int(h),
                    "w": int(w),
                    "mask": mask,
                    "area": len(pix),
                    "center_r": float(np.mean(rs)),
                    "center_c": float(np.mean(cs)),
                })
        return comps

    comps = get_components()
    if not comps:
        return grid

    def get_notch(mask):
        """
        Return (notch_mask, offset_r, offset_c), where notch_mask is the
        background cells inside the component bbox, cropped to its own bbox.
        """
        comp = ~mask
        rows = np.any(comp, axis=1)
        cols = np.any(comp, axis=0)
        if not rows.any() or not cols.any():
            return None
        sub = comp[np.ix_(rows, cols)]
        off_r = int(np.argmax(rows))
        off_c = int(np.argmax(cols))
        return sub, off_r, off_c

    for c in comps:
        c["syms"] = all_sims(c["mask"])
        c["notch"] = get_notch(c["mask"])

    def bboxes_touch(a, b):
        """True if bounding boxes touch or are separated by at most 1 cell."""
        dr = max(0, a["r0"] - (b["r0"] + b["h"]), b["r0"] - (a["r0"] + a["h"]))
        dc = max(0, a["c0"] - (b["c0"] + b["w"]), b["c0"] - (a["c0"] + a["w"]))
        return max(dr, dc) <= 1

    def bottom_heavy(mask):
        """True if the mass of the mask is below its vertical center."""
        if mask.size == 0 or mask.sum() == 0:
            return False
        h = mask.shape[0]
        row_counts = mask.sum(axis=1)
        centroid = float(np.sum(np.arange(h) * row_counts)) / float(mask.sum())
        return centroid > (h - 1) / 2.0 + 1e-9

    def transform_vector(dr, dc, rot, flip):
        """Apply the same rotation/flip convention to a 2D vector."""
        dr = float(dr)
        dc = float(dc)
        if rot == 0:
            vr, vc = dr, dc
        elif rot == 1:
            vr, vc = -dc, dr
        elif rot == 2:
            vr, vc = -dr, -dc
        else:
            vr, vc = dc, -dr
        if flip:
            vr, vc = vr, -vc
        return vr, vc

    def order_sides(vr, vc):
        """Order side names by agreement with transformed vector."""
        order = ["right", "left", "down", "up"]
        dirs = {
            "down": (1, 0),
            "up": (-1, 0),
            "right": (0, 1),
            "left": (0, -1),
        }
        scored = []
        for name, (dr, dc) in dirs.items():
            proj = dr * vr + dc * vc
            scored.append((proj, order.index(name), name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [name for _, _, name in scored]

    def feature_line(mask, axis, notch=None, prefer=None):
        """
        Choose a feature line of the anchor:
        - among partially filled lines, the one with the fewest cells.
        - ties are broken by the notch center if possible.
        """
        if axis == "row":
            counts = mask.sum(axis=1)
            full = mask.shape[1]
            n = mask.shape[0]
        else:
            counts = mask.sum(axis=0)
            full = mask.shape[0]
            n = mask.shape[1]

        partial = np.where((counts > 0) & (counts < full))[0]
        if len(partial) == 0:
            return n // 2

        minc = counts[partial].min()
        idxs = partial[counts[partial] == minc]

        if prefer in ("down", "right"):
            return int(idxs[-1])
        if prefer in ("up", "left"):
            return int(idxs[0])

        if notch is not None:
            nmask, off_r, off_c = notch
            if axis == "row":
                ref = off_r + nmask.shape[0] // 2
            else:
                ref = off_c + nmask.shape[1] // 2
            return int(min(idxs, key=lambda i: (abs(i - ref), i)))

        return int(idxs[0])

    def can_place(out, mask, r, c, allow_overlap=False):
        """Check bounds and, unless allowed, no overlap with non-background."""
        mh, mw = mask.shape
        if mh == 0 or mw == 0:
            return True
        if r < 0 or c < 0 or r + mh > H or c + mw > W:
            return False
        if not allow_overlap:
            region = out[r:r + mh, c:c + mw]
            if np.any(mask & (region != bg)):
                return False
        return True

    def place_mask(out, mask, r, c, color, allow_overlap=False):
        """Place mask if valid."""
        if not can_place(out, mask, r, c, allow_overlap=allow_overlap):
            return False
        sub = out[r:r + mask.shape[0], c:c + mask.shape[1]]
        sub[mask] = color
        return True

    def union_area(anchor, r, c, mh, mw):
        """Area of bounding box containing anchor and placed mask."""
        min_r = min(anchor["r0"], r)
        max_r = max(anchor["r0"] + anchor["h"], r + mh)
        min_c = min(anchor["c0"], c)
        max_c = max(anchor["c0"] + anchor["w"], c + mw)
        return (max_r - min_r) * (max_c - min_c)

    def positions_for_side(side, mh, mw, anchor, gap):
        """Yield candidate positions on a side of the anchor bbox."""
        if side == "right":
            c = anchor["c0"] + anchor["w"] + gap
            if c < 0 or c + mw > W:
                return
            r_start = max(0, anchor["r0"] - mh + 1)
            r_end = min(H - mh, anchor["r0"] + anchor["h"] - 1)
            for r in range(r_start, r_end + 1):
                yield r, c
        elif side == "left":
            c = anchor["c0"] - mw - gap
            if c < 0 or c + mw > W:
                return
            r_start = max(0, anchor["r0"] - mh + 1)
            r_end = min(H - mh, anchor["r0"] + anchor["h"] - 1)
            for r in range(r_start, r_end + 1):
                yield r, c
        elif side == "down":
            r = anchor["r0"] + anchor["h"] + gap
            if r < 0 or r + mh > H:
                return
            c_start = max(0, anchor["c0"] - mw + 1)
            c_end = min(W - mw, anchor["c0"] + anchor["w"] - 1)
            for c in range(c_start, c_end + 1):
                yield r, c
        elif side == "up":
            r = anchor["r0"] - mh - gap
            if r < 0 or r + mh > H:
                return
            c_start = max(0, anchor["c0"] - mw + 1)
            c_end = min(W - mw, anchor["c0"] + anchor["w"] - 1)
            for c in range(c_start, c_end + 1):
                yield r, c

    def search_extra(out, anchor, filler, extra, rot, flip, extra_color, notch=None):
        """
        Search for the best transformed placement of the extra piece.
        Returns (mask, r, c, score) or None.
        """
        vr, vc = transform_vector(
            extra["center_r"] - filler["center_r"],
            extra["center_c"] - filler["center_c"],
            rot,
            flip
        )
        sides = order_sides(vr, vc)
        pref_rot = (4 - rot) % 4

        for allow_overlap in (False, True):
            best = None
            best_score = -10**18

            for side_idx, side in enumerate(sides):
                if side in ("right", "left"):
                    feat_idx = feature_line(anchor["mask"], "row", notch=notch)
                    feat_global = anchor["r0"] + feat_idx
                else:
                    feat_idx = feature_line(anchor["mask"], "col", notch=notch)
                    feat_global = anchor["c0"] + feat_idx

                for gap in range(0, 3):
                    for sm, srot, sflip in extra["syms"]:
                        mh, mw = sm.shape
                        if mh == 0 or mw == 0:
                            continue

                        for r, c in positions_for_side(side, mh, mw, anchor, gap):
                            if not can_place(out, sm, r, c, allow_overlap=allow_overlap):
                                continue

                            overlap_count = 0
                            if allow_overlap:
                                region = out[r:r + mh, c:c + mw]
                                overlap_count = int(np.sum(sm & (region != bg)))

                            score = 0

                            if side_idx == 0:
                                score += 500
                            elif side_idx == 1:
                                score += 200

                            score += (2 - gap) * 50

                            if srot == pref_rot and not sflip:
                                score += 1000
                            elif srot == pref_rot:
                                score += 500
                            if not sflip:
                                score += 100

                            if side in ("right", "left"):
                                row_counts = sm.sum(axis=1)
                                maxc = row_counts.max()
                                if maxc > 0:
                                    dense = np.where(row_counts == maxc)[0]
                                    dist = min(abs(r + d - feat_global) for d in dense)
                                    if dist == 0:
                                        score += 1000
                                    else:
                                        score += max(0, 300 - 100 * dist)
                            else:
                                col_counts = sm.sum(axis=0)
                                maxc = col_counts.max()
                                if maxc > 0:
                                    dense = np.where(col_counts == maxc)[0]
                                    dist = min(abs(c + d - feat_global) for d in dense)
                                    if dist == 0:
                                        score += 1000
                                    else:
                                        score += max(0, 300 - 100 * dist)

                            score -= union_area(anchor, r, c, mh, mw) * 5

                            if allow_overlap:
                                score -= 1000 * overlap_count

                            if score > best_score:
                                best_score = score
                                best = (sm, r, c, score)

            if best is not None:
                return best

        return None

    def make_plan(anchor, filler, extra, rot, flip, penalty=0):
        """Build a candidate output for one (anchor, filler, extra, symmetry)."""
        out = grid.copy()

        for r, c in filler["pixels"]:
            out[r, c] = bg
        for r, c in extra["pixels"]:
            out[r, c] = bg

        nmask, off_r, off_c = anchor["notch"]
        fr = anchor["r0"] + off_r
        fc = anchor["c0"] + off_c

        swap = bottom_heavy(nmask)
        fill_color = extra["color"] if swap else filler["color"]
        extra_color = filler["color"] if swap else extra["color"]

        if not place_mask(out, nmask, fr, fc, fill_color, allow_overlap=False):
            if not place_mask(out, nmask, fr, fc, fill_color, allow_overlap=True):
                return None

        static_score = anchor["area"] * 1000 + filler["area"] * 100 + extra["area"] * 10
        if rot == 0 and not flip:
            static_score += 200

        identity = (rot == 0 and not flip)
        if identity:
            dr = fr - filler["r0"]
            dc = fc - filler["c0"]
            er = extra["r0"] + dr
            ec = extra["c0"] + dc
            if can_place(out, extra["mask"], er, ec, allow_overlap=False):
                place_mask(out, extra["mask"], er, ec, extra_color, allow_overlap=False)
                return {"score": static_score + 10000 - penalty, "out": out}

        best = search_extra(
            out,
            anchor,
            filler,
            extra,
            rot,
            flip,
            extra_color,
            notch=anchor["notch"]
        )
        if best is None:
            return None

        emask, er, ec, escore = best
        if not place_mask(out, emask, er, ec, extra_color, allow_overlap=True):
            return None

        return {"score": static_score + escore - penalty, "out": out}

    best_plan = None
    best_score = -10**18

    # Pass 1: extra must be adjacent to filler.
    for anchor in comps:
        if anchor["notch"] is None:
            continue
        nmask, _, _ = anchor["notch"]
        narea = int(nmask.sum())

        for filler in comps:
            if filler["id"] == anchor["id"] or filler["area"] != narea:
                continue

            matches = []
            for sm, rot, flip in filler["syms"]:
                if sm.shape == nmask.shape and np.array_equal(sm, nmask):
                    matches.append((rot, flip))

            if not matches:
                continue

            for extra in comps:
                if extra["id"] in (anchor["id"], filler["id"]):
                    continue
                if not bboxes_touch(filler, extra):
                    continue

                for rot, flip in matches:
                    plan = make_plan(anchor, filler, extra, rot, flip, penalty=0)
                    if plan is not None and plan["score"] > best_score:
                        best_score = plan["score"]
                        best_plan = plan

    # Pass 2: if no adjacent extra was found, allow a looser choice.
    if best_plan is None:
        for anchor in comps:
            if anchor["notch"] is None:
                continue
            nmask, _, _ = anchor["notch"]
            narea = int(nmask.sum())

            for filler in comps:
                if filler["id"] == anchor["id"] or filler["area"] != narea:
                    continue

                matches = []
                for sm, rot, flip in filler["syms"]:
                    if sm.shape == nmask.shape and np.array_equal(sm, nmask):
                        matches.append((rot, flip))

                if not matches:
                    continue

                for extra in comps:
                    if extra["id"] in (anchor["id"], filler["id"]):
                        continue

                    for rot, flip in matches:
                        plan = make_plan(anchor, filler, extra, rot, flip, penalty=50000)
                        if plan is not None and plan["score"] > best_score:
                            best_score = plan["score"]
                            best_plan = plan

    if best_plan is not None:
        return best_plan["out"]

    # Fallback for the simple three-component assembly style.
    def fallback_plan():
        if len(comps) != 3:
            return None

        anchor = max(comps, key=lambda x: x["area"])
        others = [c for c in comps if c["id"] != anchor["id"]]
        if len(others) < 2:
            return None

        filler = min(others, key=lambda x: x["area"])
        extra = max(others, key=lambda x: x["area"])

        out = grid.copy()
        for r, c in filler["pixels"]:
            out[r, c] = bg
        for r, c in extra["pixels"]:
            out[r, c] = bg

        notch = anchor["notch"]
        swap = bottom_heavy(notch[0]) if notch is not None else False
        fill_color = extra["color"] if swap else filler["color"]
        extra_color = filler["color"] if swap else extra["color"]

        # Choose the best way to place the smaller piece over/around the anchor.
        best_f = None
        best_f_score = -10**18

        for fmask, frot, fflip in filler["syms"]:
            mh, mw = fmask.shape
            if mh == 0 or mw == 0:
                continue

            r_start = max(0, anchor["r0"] - 2)
            r_end = min(H - mh, anchor["r0"] + anchor["h"] - mh + 2)
            c_start = max(0, anchor["c0"] - 2)
            c_end = min(W - mw, anchor["c0"] + anchor["w"] - mw + 2)

            for r in range(r_start, r_end + 1):
                for c in range(c_start, c_end + 1):
                    if not can_place(out, fmask, r, c, allow_overlap=True):
                        continue

                    region = out[r:r + mh, c:c + mw]
                    fill = int(np.sum(fmask & (region == bg)))
                    if fill == 0:
                        continue
                    overlap = int(np.sum(fmask & (region != bg)))

                    r1 = max(r, anchor["r0"])
                    r2 = min(r + mh, anchor["r0"] + anchor["h"])
                    c1 = max(c, anchor["c0"])
                    c2 = min(c + mw, anchor["c0"] + anchor["w"])

                    inside = 0
                    if r1 < r2 and c1 < c2:
                        inside = int(fmask[r1 - r:r2 - r, c1 - c:c2 - c].sum())
                    if inside == 0:
                        continue

                    outside = int(fmask.sum()) - inside

                    score = 10 * fill - 5 * overlap - 10 * outside

                    if notch is not None:
                        nmask2, off_r, off_c = notch
                        nr1 = max(r, anchor["r0"] + off_r)
                        nr2 = min(r + mh, anchor["r0"] + off_r + nmask2.shape[0])
                        nc1 = max(c, anchor["c0"] + off_c)
                        nc2 = min(c + mw, anchor["c0"] + off_c + nmask2.shape[1])
                        if nr1 < nr2 and nc1 < nc2:
                            nfill = int(fmask[nr1 - r:nr2 - r, nc1 - c:nc2 - c].sum())
                            score += 5 * nfill

                    if frot == 1 and not fflip:
                        score += 1000
                    elif frot == 3 and not fflip:
                        score += 800
                    elif frot == 0 and not fflip:
                        score += 500

                    score -= union_area(anchor, r, c, mh, mw) * 2

                    if score > best_f_score:
                        best_f_score = score
                        best_f = (fmask, frot, fflip, r, c)

        if best_f is None:
            return None

        fmask, frot, fflip, fr, fc = best_f
        if not place_mask(out, fmask, fr, fc, fill_color, allow_overlap=True):
            return None

        best_e = search_extra(
            out,
            anchor,
            filler,
            extra,
            frot,
            fflip,
            extra_color,
            notch=anchor["notch"]
        )

        if best_e is None:
            # Graceful fallback: restore extra if it cannot be placed.
            for r, c in extra["pixels"]:
                out[r, c] = extra["color"]
            return {"out": out}

        emask, er, ec, _ = best_e
        if not place_mask(out, emask, er, ec, extra_color, allow_overlap=True):
            for r, c in extra["pixels"]:
                out[r, c] = extra["color"]

        return {"out": out}

    plan = fallback_plan()
    if plan is not None:
        return plan["out"]

    return grid