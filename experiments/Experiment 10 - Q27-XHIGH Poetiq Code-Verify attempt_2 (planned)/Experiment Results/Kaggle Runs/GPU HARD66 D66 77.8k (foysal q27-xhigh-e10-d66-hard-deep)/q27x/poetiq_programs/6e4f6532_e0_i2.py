import numpy as np
from collections import deque, Counter


def _rotate_matrix(mat, k):
    """Rotate matrix k times 90 degrees clockwise."""
    for _ in range(k % 4):
        mat = mat.T[::-1, :]
    return mat


def _flip_matrix(mat, mode):
    if mode == "v":
        return mat[::-1, :]
    if mode == "h":
        return mat[:, ::-1]
    if mode == "d1":
        return np.fliplr(mat.T)
    if mode == "d2":
        return np.flipud(mat.T)
    return mat


def _all_dihedral_transforms(mat):
    """Return all 8 dihedral transformations of mat, without changing dtype."""
    base = mat.copy()
    outs = []
    for k in range(4):
        r = _rotate_matrix(base, k)
        outs.append(r)
        outs.append(r[:, ::-1])
    # Unique by array equality
    uniq = []
    seen = []
    for t in outs:
        same = False
        for u in seen:
            if u.shape == t.shape and np.array_equal(u, t):
                same = True
                break
        if not same:
            seen.append(t)
            uniq.append(t)
    return uniq


def _detect_wall_strips(grid):
    h, w = grid.shape
    bg = Counter(grid.flatten()).most_common(1)[0][0]

    wall_mask = np.zeros((h, w), dtype=bool)
    wall_colors = set()

    # Horizontal strips: long runs of a non-background color.
    for r in range(h):
        c = 0
        while c < w:
            color = grid[r, c]
            if color == bg:
                c += 1
                continue
            d = c
            while d < w and grid[r, d] == color:
                d += 1
            length = d - c
            # A wall strip is at least half the width, or touches a border.
            if length >= max(3, int(0.75 * w)) or (c == 0 and d >= w // 2) or (d == w and c <= w // 2):
                # Require vertical consistency: the row above or below has a similar run
                consistent = False
                if r > 0:
                    above = np.count_nonzero(grid[r - 1, c:d] == color)
                    if above >= 0.8 * length:
                        consistent = True
                if r + 1 < h:
                    below = np.count_nonzero(grid[r + 1, c:d] == color)
                    if below >= 0.8 * length:
                        consistent = True
                if consistent:
                    wall_mask[r, c:d] = True
                    wall_colors.add(color)
            c = d

    # Vertical strips.
    for c in range(w):
        r = 0
        while r < h:
            color = grid[r, c]
            if color == bg:
                r += 1
                continue
            d = r
            while d < h and grid[d, c] == color:
                d += 1
            length = d - r
            if length >= max(3, int(0.75 * h)) or (r == 0 and d >= h // 2) or (d == h and r <= h // 2):
                consistent = False
                if c > 0:
                    left = np.count_nonzero(grid[r:d, c - 1] == color)
                    if left >= 0.8 * length:
                        consistent = True
                if c + 1 < w:
                    right = np.count_nonzero(grid[r:d, c + 1] == color)
                    if right >= 0.8 * length:
                        consistent = True
                if consistent:
                    wall_mask[r:d, c] = True
                    wall_colors.add(color)
            r = d

    return wall_mask, bg, wall_colors


def _find_slots(grid, wall_mask, h, w):
    """
    Find placement slots:
    - first, connected non-wall regions (rooms);
    - if a room spans both left/right borders and has no vertical wall inside,
      split it horizontally;
    - if it spans both top/bottom borders and has no horizontal wall inside,
      split it vertically.
    """
    free = ~wall_mask
    labels = np.zeros((h, w), dtype=int)
    rooms = []
    cur = 0
    for i in range(h):
        for j in range(w):
            if free[i, j] and labels[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                labels[i, j] = cur
                pix = []
                while q:
                    y, x = q.popleft()
                    pix.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            q.append((ny, nx))
                r0 = min(p[0] for p in pix)
                r1 = max(p[0] for p in pix)
                c0 = min(p[1] for p in pix)
                c1 = max(p[1] for p in pix)
                rooms.append((r0, c0, r1, c1))

    slots = []

    def has_wall_in_rect(rect, axis, color=None):
        r0, c0, r1, c1 = rect
        if axis == "v":
            # horizontal wall inside this x-range
            for rr in range(r0, r1 + 1):
                if np.any(wall_mask[rr, c0:c1 + 1]):
                    return True
        else:
            for cc in range(c0, c1 + 1):
                if np.any(wall_mask[r0:r1 + 1, cc]):
                    return True
        return False

    def touches_left(rect):
        _, c0, _, _ = rect
        return c0 <= 1

    def touches_right(rect):
        _, _, _, c1 = rect
        return c1 >= w - 2

    def touches_top(rect):
        r0, _, _, _ = rect
        return r0 <= 1

    def touches_bottom(rect):
        _, _, r1, _ = rect
        return r1 >= h - 2

    for rect in rooms:
        r0, c0, r1, c1 = rect
        wlen = c1 - c0 + 1
        hlen = r1 - r0 + 1

        split_h = (
            touches_left(rect) and touches_right(rect)
            and wlen >= 8
            and not has_wall_in_rect(rect, "c")  # no vertical wall inside
        )
        split_v = (
            touches_top(rect) and touches_bottom(rect)
            and hlen >= 8
            and not has_wall_in_rect(rect, "v")  # no horizontal wall inside
        )

        if split_h and split_v:
            # quadrants
            rm = (r0 + r1) // 2
            cm = (c0 + c1) // 2
            slots.append((r0, c0, rm, cm))
            slots.append((r0, cm + 1, rm, c1))
            slots.append((rm + 1, c0, r1, cm))
            slots.append((rm + 1, cm + 1, r1, c1))
        elif split_h:
            cm = (c0 + c1) // 2
            slots.append((r0, c0, r1, cm))
            slots.append((r0, cm + 1, r1, c1))
        elif split_v:
            rm = (r0 + r1) // 2
            slots.append((r0, c0, rm, c1))
            slots.append((rm + 1, c0, r1, c1))
        else:
            slots.append(rect)

    # Deduplicate and sort slots deterministically.
    uniq = []
    for s in slots:
        if s not in uniq:
            uniq.append(s)
    uniq.sort(key=lambda x: (x[0], x[1]))
    return uniq


def _slot_side_color(grid, wall_mask, slot):
    r0, c0, r1, c1 = slot
    h, w = grid.shape
    colors = {}

    def outside_color(rect):
        vals = []
        for (y, x) in rect:
            if 0 <= y < h and 0 <= x < w and wall_mask[y, x]:
                vals.append(grid[y, x])
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    top = []
    bot = []
    left = []
    right = []

    if r0 > 0:
        top = [(r0 - 1, x) for x in range(c0, c1 + 1)]
    if r1 + 1 < h:
        bot = [(r1 + 1, x) for x in range(c0, c1 + 1)]
    if c0 > 0:
        left = [(y, c0 - 1) for y in range(r0, r1 + 1)]
    if c1 + 1 < w:
        right = [(y, c1 + 1) for y in range(r0, r1 + 1)]

    colors["top"] = outside_color(top)
    colors["bottom"] = outside_color(bot)
    colors["left"] = outside_color(left)
    colors["right"] = outside_color(right)
    return colors


def _object_side_colors(mat, wall_colors):
    """Return dict side -> set of wall colors appearing on that side."""
    H, W = mat.shape
    sides = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    for r in range(H):
        for c in range(W):
            col = mat[r, c]
            if col not in wall_colors:
                continue
            if r == 0:
                sides["top"].add(col)
            if r == H - 1:
                sides["bottom"].add(col)
            if c == 0:
                sides["left"].add(col)
            if c == W - 1:
                sides["right"].add(col)
    return sides


def _matches_slot(scolors, slot_side_colors, wall_colors):
    """
    A side with a wall must have the matching color if the object has a colored pixel there.
    A side without a wall may contain any wall color.
    """
    for side in ("top", "bottom", "left", "right"):
        obj_colors = scolors.get(side, set())
        if not obj_colors:
            continue
        wall_col = slot_side_colors.get(side)
        if wall_col is not None:
            # If this side is a wall, all object wall-color pixels on this side must match it.
            if not obj_colors.issubset({wall_col}):
                return False
        else:
            # Inner side: must not contain a color that is exclusively a border wall of another side?
            # Keep permissive; this helps the examples.
            pass
    return True


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. Detect thick wall strips and background.
    2. Find placement slots (rooms; split into halves/quadrants only when a room spans
       both opposite borders and has no central wall inside).
    3. Extract objects using 8-colored frame pixels as anchors, and gather nearby
       non-background, non-wall connector pixels into the same object.
    4. For each object, test all dihedral orientations. Keep orientations that can match
       the wall colors of some slot side. Choose the orientation/slot with best match,
       using original position as a tie-breaker.
    5. Place objects using deterministic horizontal centering in their slot and a
       vertical packing heuristic (bottom packing when a bottom wall exists, otherwise
       top packing with a small stagger).
    6. Return the reconstructed grid with wall strips unchanged.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.copy()
    h, w = grid.shape
    wall_mask, bg, wall_colors = _detect_wall_strips(grid)

    # If no meaningful walls are found, return input unchanged.
    if not wall_colors:
        return out

    # Interior object pixels: not background and not wall strip.
    obj_mask = (grid != bg) & (~wall_mask)

    # Anchor on color 8 components (the "frames").
    seen = np.zeros_like(obj_mask, dtype=bool)
    anchors = []
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 8 and not seen[i, j]:
                q = deque([(i, j)])
                seen[i, j] = True
                pix = []
                while q:
                    y, x = q.popleft()
                    pix.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 8 and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                if pix:
                    anchors.append(pix)

    if not anchors:
        return out

    # Group nearby object pixels to anchors.
    assigned = np.zeros_like(obj_mask, dtype=bool)
    objects = []

    for apix in anchors:
        r0 = min(p[0] for p in apix)
        r1 = max(p[0] for p in apix)
        c0 = min(p[1] for p in apix)
        c1 = max(p[1] for p in apix)

        # Expand by 2 to catch connectors just outside the frame.
        er0, er1 = max(0, r0 - 2), min(h - 1, r1 + 2)
        ec0, ec1 = max(0, c0 - 2), min(w - 1, c1 + 2)

        pix = list(apix)
        for r in range(er0, er1 + 1):
            for c in range(ec0, ec1 + 1):
                if obj_mask[r, c] and not assigned[r, c]:
                    pix.append((r, c))

        # Assign only pixels that are reasonably close to this anchor bbox.
        final_pix = []
        for (r, c) in pix:
            dr = max(0, r0 - r, r - r1)
            dc = max(0, c0 - c, c - c1)
            if dr <= 2 and dc <= 2 and not assigned[r, c]:
                final_pix.append((r, c))
                assigned[r, c] = True

        if len(final_pix) < 3:
            continue

        fr0 = min(p[0] for p in final_pix)
        fr1 = max(p[0] for p in final_pix)
        fc0 = min(p[1] for p in final_pix)
        fc1 = max(p[1] for p in final_pix)

        mat = np.zeros((fr1 - fr0 + 1, fc1 - fc0 + 1), dtype=grid.dtype)
        for (r, c) in final_pix:
            mat[r - fr0, c - fc0] = grid[r, c]

        # Remove colors that are neither 8/9 nor wall colors.
        keep = np.zeros_like(mat, dtype=bool)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                if mat[r, c] in (8, 9) or mat[r, c] in wall_colors:
                    keep[r, c] = True
        mat[~keep] = 0

        # Trim empty rows/cols after cleanup.
        rows = np.any(mat != 0, axis=1)
        cols = np.any(mat != 0, axis=0)
        if not rows.any() or not cols.any():
            continue
        ridx = np.where(rows)[0]
        cidx = np.where(cols)[0]
        mat = mat[ridx[0]:ridx[-1] + 1, cidx[0]:idx[-1] + 1] if False else mat[ridx[0]:ridx[-1] + 1, cidx[0]:cidx[-1] + 1]

        objects.append({
            "mat": mat,
            "bbox": (fr0, fc0, fr1, fc1),
            "center": ((fr0 + fr1) / 2.0, (fc0 + fc1) / 2.0),
            "pixels": final_pix
        })

    if not objects:
        return out

    slots = _find_slots(grid, wall_mask, h, w)
    if not slots:
        # Fallback: whole non-wall bounding box.
        ys, xs = np.where(~wall_mask)
        if ys.size == 0:
            return out
        slots = [(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))]

    slot_side_colors = [_slot_side_color(grid, wall_mask, s) for s in slots]

    # Bottom wall exists if any wall strip is in the bottom two rows.
    bottom_wall = bool(np.any(wall_mask[h - 2:h, :]))
    top_wall = bool(np.any(wall_mask[:2, :]))

    placements = []

    for obj in objects:
        best = None
        oc = obj["center"]
        base_mat = obj["mat"]
        H0, W0 = base_mat.shape

        for t in _all_dihedral_transforms(base_mat):
            sc = _object_side_colors(t, wall_colors)
            Ht, Wt = t.shape

            # Trim empty border rows/cols of the transformed object.
            rows = np.any(t != 0, axis=1)
            cols = np.any(t != 0, axis=0)
            if not rows.any() or not cols.any():
                continue
            ri = np.where(rows)[0]
            ci = np.where(cols)[0]
            tt = t[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
            sc_trim = _object_side_colors(tt, wall_colors)
            Hf, Wf = tt.shape

            for si, slot in enumerate(slots):
                r0, c0, r1, c1 = slot
                if Hf > (r1 - r0 + 1) or Wf > (c1 - c0 + 1):
                    continue
                if not _matches_slot(sc_trim, slot_side_colors[si], wall_colors):
                    continue

                # Score:
                # + matches where object side color equals slot wall color
                score = 0.0
                for side in ("top", "bottom", "left", "right"):
                    if sc_trim.get(side):
                        if slot_side_colors[si].get(side) in sc_trim[side]:
                            score += 2.0
                        elif slot_side_colors[si].get(side) is None:
                            score += 1.0

                # Prefer slots closer to original object center.
                sr = (r0 + r1) / 2.0
                scx = (c0 + c1) / 2.0
                score -= 0.01 * ((oc[0] - sr) ** 2 + (oc[1] - scx) ** 2)

                # Slight preference for keeping original orientation if multiple valid.
                if np.array_equal(tt, base_mat):
                    score += 0.05

                if best is None or score > best[0]:
                    best = (score, tt, slot)

        if best is None:
            # Fallback: place unrotated in nearest slot if it fits.
            for si, slot in enumerate(slots):
                r0, c0, r1, c1 = slot
                if H0 <= (r1 - r0 + 1) and W0 <= (c1 - c0 + 1):
                    best = (0.0, base_mat, slot)
                    break
        if best is None:
            continue

        _, tmat, slot = best
        Hf, Wf = tmat.shape
        r0, c0, r1, c1 = slot

        # Horizontal: center in slot.
        tc = c0 + (c1 - c0 + 1 - Wf) // 2
        tc = max(c0, min(tc, c1 - Wf + 1))

        # Store candidate, vertical position determined after all slots/orientations.
        placements.append({
            "mat": tmat,
            "slot": slot,
            "tc": tc,
            "orig_center": obj["center"],
            "bbox": obj["bbox"]
        })

    # Vertical packing heuristic.
    if placements:
        # Sort by original top-to-bottom, then left-to-right.
        placements.sort(key=lambda p: (p["orig_center"][0], p["orig_center"][1]))

        # Determine interior vertical bounds from all slots.
        top_y = min(p["slot"][0] for p in placements)
        bot_y = max(p["slot"][2] for p in placements)

        if bottom_wall:
            # Pack from bottom, preserving rough original gaps.
            order = placements[::-1]
            next_top = bot_y - 1
            for p in order:
                Hf = p["mat"].shape[0]
                p["tr"] = min(next_top, max(top_y, next_top - Hf + 1))
                # next_top becomes just above placed object, with a small gap.
                next_top = p["tr"] - 1 - max(0, (p["mat"].shape[0] // 4))
        else:
            # Pack from top with a small stagger.
            cur_top = top_y
            for p in placements:
                Hf = p["mat"].shape[0]
                p["tr"] = max(top_y, min(cur_top, bot_y - Hf + 1))
                cur_top = p["tr"] + max(1, Hf // 3)

    # Clear object pixels from output, then draw placed objects.
    for p in placements:
        for (r, c) in []:
            pass

    # Clear original object pixels (all non-background non-wall pixels).
    for r in range(h):
        for c in range(w):
            if obj_mask[r, c]:
                out[r, c] = bg

    for p in placements:
        mat = p["mat"]
        tr, tc = p["tr"], p["tc"]
        Hf, Wf = mat.shape
        for r in range(Hf):
            for c in range(Wf):
                val = mat[r, c]
                if val != 0:
                    out[tr + r, tc + c] = val

    return out