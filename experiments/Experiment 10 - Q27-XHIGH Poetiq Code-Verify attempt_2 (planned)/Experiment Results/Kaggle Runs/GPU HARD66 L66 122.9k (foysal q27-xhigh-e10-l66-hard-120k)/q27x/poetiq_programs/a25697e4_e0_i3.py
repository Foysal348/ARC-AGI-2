import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the puzzle by identifying a large "anchor" color, extracting the
    empty notch inside its bounding box, filling that notch with one of the
    other pieces, and moving the remaining piece next to the notch.

    Rule inferred from the examples:
    1. Background is the most frequent color.
    2. Group all non-background cells by color.  The largest non-background
       color is the anchor.
    3. The anchor's hole/notch is the set of background cells inside the
       anchor's bounding box, cropped to its own bounding box.
    4. One of the non-anchor pieces matches that notch under a grid symmetry.
       That piece is the filler.
    5. The other non-anchor piece is the extra piece.
    6. If the notch is bottom-heavy, swap the output colors of filler and extra;
       otherwise keep their original colors.
    7. Remove filler and extra from the input.
    8. Put the filler into the anchor notch using the matching symmetry.
    9. Put the extra next to the anchor/filler.  If the filler was translated
       only, the extra is translated by the same delta.  Otherwise, the extra
       is rotated/reflected consistently and then placed on the best adjacent
       side of the anchor, aligned with the anchor's sparse feature line.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    bg = int(np.bincount(grid.ravel()).argmax())

    def crop_mask(m):
        """Crop boolean mask to its minimal non-empty bounding box."""
        if m.size == 0:
            return np.zeros((0, 0), dtype=bool)
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros((0, 0), dtype=bool)
        return m[np.ix_(rows, cols)]

    def apply_op(mask, rot, flip):
        """Apply rotation (0-3 ccw), optional left-right flip, then crop."""
        if mask.size == 0:
            return np.zeros((0, 0), dtype=bool)
        m = np.rot90(mask, k=rot)
        if flip:
            m = np.fliplr(m)
        return crop_mask(m)

    def symmetries(mask):
        """Return distinct cropped symmetries of mask with their operation."""
        res = []
        seen = set()
        for rot in range(4):
            for flip in (False, True):
                m = apply_op(mask, rot, flip)
                if m.size == 0:
                    continue
                key = (m.shape, m.tobytes())
                if key not in seen:
                    seen.add(key)
                    res.append((m, rot, flip))
        return res

    # Gather all non-background cells by color.  A "piece" may be disconnected
    # in the input; we treat all cells of the same color as one group.
    colors = {}
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != bg:
                colors.setdefault(val, []).append((r, c))

    if not colors:
        return grid

    def make_mask(pixels):
        rmin = min(p[0] for p in pixels)
        rmax = max(p[0] for p in pixels)
        cmin = min(p[1] for p in pixels)
        cmax = max(p[1] for p in pixels)
        mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=bool)
        for r, c in pixels:
            mask[r - rmin, c - cmin] = True
        return mask, rmin, cmin

    pieces = []
    for color, pixels in colors.items():
        mask, r0, c0 = make_mask(pixels)
        pieces.append({
            "color": color,
            "pixels": pixels,
            "mask": mask,
            "area": len(pixels),
            "r0": r0,
            "c0": c0,
            "h": mask.shape[0],
            "w": mask.shape[1],
        })

    if len(pieces) < 2:
        return grid

    anchor = max(pieces, key=lambda x: x["area"])
    others = [p for p in pieces if p["color"] != anchor["color"]]

    if len(others) < 2:
        # Fallback: if there is not a clear three-color puzzle, do nothing.
        return grid

    # Notch: background cells inside anchor bbox, cropped.
    anchor_mask = anchor["mask"]
    complement = ~anchor_mask
    if not complement.any():
        return grid

    nmask, noff_r, noff_c = crop_mask(complement), None, None
    rows = np.any(complement, axis=1)
    cols = np.any(complement, axis=0)
    noff_r = int(np.argmax(rows))
    noff_c = int(np.argmax(cols))
    nmask = complement[np.ix_(rows, cols)]

    # Match each other piece to the notch under a symmetry.
    best = None
    for other in others:
        for sm, rot, flip in symmetries(other["mask"]):
            if sm.shape == nmask.shape and np.array_equal(sm, nmask):
                # Prefer no-op transformations and pieces with closer area.
                score = 0
                if rot == 0 and not flip:
                    score += 1000
                score += 100 - abs(other["area"] - int(nmask.sum()))
                if best is None or score > best[0]:
                    best = (score, other, rot, flip)

    if best is None:
        # Graceful fallback: no notch match, leave grid unchanged.
        return grid

    _, filler, rot, flip = best
    extra = others[0] if others[1]["color"] == filler["color"] else others[1]
    # In case colors are identical for the two non-anchor groups (should not
    # happen), choose the smaller one as filler if it matches.
    if extra["color"] == filler["color"]:
        if filler["area"] > extra["area"]:
            filler, extra = extra, filler

    out = grid.copy()

    # Remove the moving pieces.
    for r, c in filler["pixels"]:
        out[r, c] = bg
    for r, c in extra["pixels"]:
        out[r, c] = bg

    # Color swap if notch is bottom-heavy.
    def bottom_heavy(mask):
        if mask.size == 0 or mask.sum() == 0:
            return False
        h = mask.shape[0]
        row_counts = mask.sum(axis=1)
        centroid = float(np.sum(np.arange(h) * row_counts)) / float(mask.sum())
        return centroid > (h - 1) / 2.0

    swap = bottom_heavy(nmask)
    fill_color = extra["color"] if swap else filler["color"]
    extra_color = filler["color"] if swap else extra["color"]

    # Place filler into notch.
    filler_mask = apply_op(filler["mask"], rot, flip)
    fr = anchor["r0"] + noff_r
    fc = anchor["c0"] + noff_c

    def can_place(mask, r, c):
        mh, mw = mask.shape
        if mh == 0 or mw == 0:
            return False
        if r < 0 or c < 0 or r + mh > H or c + mw > W:
            return False
        region = out[r:r + mh, c:c + mw]
        return not np.any(mask & (region != bg))

    def place(mask, r, c, color):
        if not can_place(mask, r, c):
            return False
        out[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = color
        return True

    if not place(filler_mask, fr, fc, fill_color):
        # Should not happen, but gracefully keep original if impossible.
        return grid

    # Place extra piece.
    identity = (rot == 0 and not flip)
    placed = False

    if identity:
        dr = fr - filler["r0"]
        dc = fc - filler["c0"]
        er = extra["r0"] + dr
        ec = extra["c0"] + dc
        if place(extra["mask"], er, ec, extra_color):
            placed = True

    if not placed:
        # Transform the extra by the same operation used for the filler.
        candidate = apply_op(extra["mask"], rot, flip)

        def feature_row(mask):
            counts = mask.sum(axis=1)
            full = mask.shape[1]
            partial = np.where((counts > 0) & (counts < full))[0]
            if len(partial):
                mn = counts[partial].min()
                idxs = partial[counts[partial] == mn]
                return int(idxs[0])
            return mask.shape[0] // 2

        def feature_col(mask):
            counts = mask.sum(axis=0)
            full = mask.shape[0]
            partial = np.where((counts > 0) & (counts < full))[0]
            if len(partial):
                mn = counts[partial].min()
                idxs = partial[counts[partial] == mn]
                return int(idxs[0])
            return mask.shape[1] // 2

        def try_side(side, mask):
            mh, mw = mask.shape
            if side == "right":
                c = anchor["c0"] + anchor["w"]
                if c + mw > W:
                    return False
                feat = feature_row(anchor["mask"])
                target = anchor["r0"] + feat
                # Prefer vertical positions overlapping the anchor and aligned
                # to the anchor's sparse row.
                for r in range(max(0, anchor["r0"] - mh + 1),
                               min(H - mh, anchor["r0"] + anchor["h"]) + 1):
                    if can_place(mask, r, c):
                        if abs(r + mh // 2 - target) <= 1:
                            return place(mask, r, c, extra_color)
                for r in range(max(0, anchor["r0"] - mh + 1),
                               min(H - mh, anchor["r0"] + anchor["h"]) + 1):
                    if can_place(mask, r, c):
                        return place(mask, r, c, extra_color)
            elif side == "left":
                c = anchor["c0"] - mw
                if c < 0:
                    return False
                feat = feature_row(anchor["mask"])
                target = anchor["r0"] + feat
                for r in range(max(0, anchor["r0"] - mh + 1),
                               min(H - mh, anchor["r0"] + anchor["h"]) + 1):
                    if can_place(mask, r, c):
                        if abs(r + mh // 2 - target) <= 1:
                            return place(mask, r, c, extra_color)
                for r in range(max(0, anchor["r0"] - mh + 1),
                               min(H - mh, anchor["r0"] + anchor["h"]) + 1):
                    if can_place(mask, r, c):
                        return place(mask, r, c, extra_color)
            elif side == "down":
                r = anchor["r0"] + anchor["h"]
                if r + mh > H:
                    return False
                feat = feature_col(anchor["mask"])
                target = anchor["c0"] + feat
                for c in range(max(0, anchor["c0"] - mw + 1),
                               min(W - mw, anchor["c0"] + anchor["w"]) + 1):
                    if can_place(mask, r, c):
                        if abs(c + mw // 2 - target) <= 1:
                            return place(mask, r, c, extra_color)
                for c in range(max(0, anchor["c0"] - mw + 1),
                               min(W - mw, anchor["c0"] + anchor["w"]) + 1):
                    if can_place(mask, r, c):
                        return place(mask, r, c, extra_color)
            elif side == "up":
                r = anchor["r0"] - mh
                if r < 0:
                    return False
                feat = feature_col(anchor["mask"])
                target = anchor["c0"] + feat
                for c in range(max(0, anchor["c0"] - mw + 1),
                               min(W - mw, anchor["c0"] + anchor["w"]) + 1):
                    if can_place(mask, r, c):
                        if abs(c + mw // 2 - target) <= 1:
                            return place(mask, r, c, extra_color)
                for c in range(max(0, anchor["c0"] - mw + 1),
                               min(W - mw, anchor["c0"] + anchor["w"]) + 1):
                    if can_place(mask, r, c):
                        return place(mask, r, c, extra_color)
            return False

        # Try the side suggested by the transformed relative position.
        vr = extra["center_r"] - filler["center_r"] if "center_r" in extra else 0
        vc = extra["center_c"] - filler["center_c"] if "center_c" in extra else 0
        if rot == 0:
            tr, tc = vr, vc
        elif rot == 1:
            tr, tc = -vc, vr
        elif rot == 2:
            tr, tc = -vr, -vc
        else:
            tr, tc = vc, -vr
        if flip:
            tr, tc = tr, -tc

        side_order = []
        if tc > 0:
            side_order.append("right")
        if tc < 0:
            side_order.append("left")
        if tr > 0:
            side_order.append("down")
        if tr < 0:
            side_order.append("up")
        if not side_order:
            side_order = ["right", "down", "left", "up"]

        for side in side_order:
            if try_side(side, candidate):
                placed = True
                break

        # Robust final fallback: try all symmetries and all sides.
        if not placed:
            for sm, _, _ in symmetries(extra["mask"]):
                for side in ["right", "down", "left", "up"]:
                    if try_side(side, sm):
                        placed = True
                        break
                if placed:
                    break

    return out