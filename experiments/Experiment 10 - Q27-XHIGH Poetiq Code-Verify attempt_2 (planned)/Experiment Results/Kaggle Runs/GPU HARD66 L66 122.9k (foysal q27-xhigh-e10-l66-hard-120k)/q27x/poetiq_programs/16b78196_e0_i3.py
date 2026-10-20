import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC solver for the “band with interlocking pieces” task.

    The input contains one large “band” component and several smaller
    components.  The band has one or more notches/slots on its long edge.
    Smaller components are moved into compatible notches, preserving their
    exact shapes, and stacked so that adjacent pieces interlock:
    in every overlapped row (for a horizontal band) or column (for a
    vertical band), the two masks are disjoint and their union is the whole
    tower width/height.

    The implementation:
      1. finds connected components,
      2. chooses the largest component as the band,
      3. detects band orientation,
      4. detects slots/notches in the band edge,
      5. groups pieces whose cross dimension equals slot_width + 2,
      6. finds the adjacent piece for each slot,
      7. builds the interlocked stack by directed compatibility edges,
      8. draws the band, then the stack, and fills the notch cavity with
         the color of the adjacent piece.
    """
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # ------------------------------------------------------------
    # Connected components, 4-connected, same color.
    # ------------------------------------------------------------
    def components():
        seen = np.zeros((H, W), dtype=bool)
        comps = []

        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0 or seen[r, c]:
                    continue
                color = int(grid[r, c])
                stack = [(r, c)]
                seen[r, c] = True
                pts = []

                while stack:
                    cr, cc = stack.pop()
                    pts.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not seen[nr, nc] and int(grid[nr, nc]) == color:
                                seen[nr, nc] = True
                                stack.append((nr, nc))

                rmin = min(r for r, _ in pts)
                rmax = max(r for r, _ in pts)
                cmin = min(c for _, c in pts)
                cmax = max(c for _, c in pts)
                h = rmax - rmin + 1
                w = cmax - cmin + 1

                mask = np.zeros((h, w), dtype=bool)
                for rr, cc in pts:
                    mask[rr - rmin, cc - cmin] = True

                comps.append({
                    "color": color,
                    "rmin": rmin,
                    "cmin": cmin,
                    "rmax": rmax,
                    "cmax": cmax,
                    "h": h,
                    "w": w,
                    "area": len(pts),
                    "mask": mask,
                    "crow": sum(r for r, _ in pts) / float(len(pts)),
                    "ccol": sum(c for _, c in pts) / float(len(pts)),
                })
        return comps

    comps = components()
    if not comps:
        return np.zeros_like(grid)

    band = max(comps, key=lambda x: x["area"])
    band_color = band["color"]
    pieces = [c for c in comps if c is not band]

    out = np.zeros((H, W), dtype=int)
    out[band["rmin"]:band["rmax"] + 1, band["cmin"]:band["cmax"] + 1] = 0

    band_mask = np.zeros((H, W), dtype=bool)
    for r in range(band["rmin"], band["rmax"] + 1):
        for c in range(band["cmin"], band["cmax"] + 1):
            if grid[r, c] == band_color:
                band_mask[r, c] = True
    out[band_mask] = band_color

    if not pieces:
        return out

    # ------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------
    def row_tuple(row):
        return tuple(1 if x else 0 for x in row)

    def full_row(w):
        return (1,) * w

    def complementary_rows(row_a, row_b):
        a = row_tuple(row_a)
        b = row_tuple(row_b)
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x and y:
                return False
            if not x and not y:
                return False
        return True

    def compatible_above(above, below, d):
        """
        above is placed with top offset 0, below with top offset d > 0.
        All overlapping rows must be complementary and cover the full width.
        """
        ha, wa = above["h"], above["w"]
        hb, wb = below["h"], below["w"]
        if wa != wb or d <= 0:
            return False

        count = min(ha - d, hb)
        if count <= 0:
            return False

        for k in range(count):
            if not complementary_rows(above["mask"][d + k], below["mask"][k]):
                return False
        return True

    def valid_offsets(above, below):
        """All positive offsets where above/below interlock."""
        res = []
        max_d = min(above["h"], below["h"] + 1)
        for d in range(1, max_d + 1):
            if compatible_above(above, below, d):
                res.append(d)
        return res

    # ------------------------------------------------------------
    # Slot detection.
    # ------------------------------------------------------------
    band_h = band["rmax"] - band["rmin"] + 1
    band_w = band["cmax"] - band["cmin"] + 1
    orientation = "horizontal" if band_w >= band_h else "vertical"

    def intervals(mask_1d):
        """Intervals of True values in a 1-D bool array."""
        res = []
        i = 0
        n = len(mask_1d)
        while i < n:
            if mask_1d[i]:
                s = i
                while i < n and mask_1d[i]:
                    i += 1
                res.append((s, i - 1))
            else:
                i += 1
        return res

    def find_horizontal_slots():
        slots = []
        top_edge = band["rmin"]
        bot_edge = band["rmax"]

        # Top slots: zeros in the top edge row, flanked by band cells.
        if top_edge < H:
            row = band_mask[top_edge, :]
            for s, e in intervals(row):
                if s > 0 and e < W - 1 and band_mask[top_edge, s - 1] and band_mask[top_edge, e + 1]:
                    slots.append({
                        "side": "top",
                        "orient": "horizontal",
                        "edge": top_edge,
                        "start": s,
                        "end": e,
                        "len": e - s + 1,
                    })

        # Bottom slots.
        if bot_edge < H:
            row = band_mask[bot_edge, :]
            for s, e in intervals(row):
                if s > 0 and e < W - 1 and band_mask[bot_edge, s - 1] and band_mask[bot_edge, e + 1]:
                    slots.append({
                        "side": "bottom",
                        "orient": "horizontal",
                        "edge": bot_edge,
                        "start": s,
                        "end": e,
                        "len": e - s + 1,
                    })

        return slots

    def find_vertical_slots():
        slots = []
        left_edge = band["cmin"]
        right_edge = band["cmax"]

        if left_edge < W:
            col = band_mask[:, left_edge]
            for s, e in intervals(col):
                if s > 0 and e < H - 1 and band_mask[s - 1, left_edge] and band_mask[e + 1, left_edge]:
                    slots.append({
                        "side": "left",
                        "orient": "vertical",
                        "edge": left_edge,
                        "start": s,
                        "end": e,
                        "len": e - s + 1,
                    })

        if right_edge < W:
            col = band_mask[:, right_edge]
            for s, e in intervals(col):
                if s > 0 and e < H - 1 and band_mask[s - 1, right_edge] and band_mask[e + 1, right_edge]:
                    slots.append({
                        "side": "right",
                        "orient": "vertical",
                        "edge": right_edge,
                        "start": s,
                        "end": e,
                        "len": e - s + 1,
                    })

        return slots

    slots = find_horizontal_slots() if orientation == "horizontal" else find_vertical_slots()

    if not slots:
        # Fallback: if slot detection fails, leave the input unchanged
        # except band extraction is not useful; return original grid.
        return grid.copy()

    # ------------------------------------------------------------
    # Piece assignment.
    # ------------------------------------------------------------
    def cross_dim(piece):
        return piece["w"] if orientation == "horizontal" else piece["h"]

    # For each slot, candidate pieces are those with cross_dim == slot.len + 2.
    # To avoid using a piece twice, process slots deterministically and
    # greedily by best total distance to original piece centers.
    used_piece_ids = set()
    assignments = []

    # Give pieces stable ids.
    for i, p in enumerate(pieces):
        p["id"] = i

    # Precompute candidate chains for each slot and each possible anchor piece.
    def piece_interface_rows(p, cross_w, slot_len):
        """
        Rows of piece p that can be the band-edge row.
        The row must have ones exactly in the central slot_len columns and
        zeros in the two outer columns.
        """
        Wtower = cross_w
        L = slot_len
        good = []
        for r in range(p["h"]):
            row = p["mask"][r]
            ok = True
            for c in range(Wtower):
                val = bool(row[c])
                if 0 <= c - 0 < Wtower:
                    # central interval starts at 1 (one side cell)
                    if 1 <= c <= L:
                        if not val:
                            ok = False
                            break
                    else:
                        if val:
                            ok = False
                            break
            if ok:
                good.append(r)
        return good

    def cavity_ok_horizontal(piece, anchor_row, slot):
        """
        Check that rows of the anchor piece that extend inward from the band
        do not overlap band cells outside the slot cavity.
        """
        side = slot["side"]
        edge = slot["edge"]
        c0 = slot["start"] - 1
        L = slot["len"]
        Wtower = L + 2

        # inward direction
        if side == "top":
            # piece is above band; rows after anchor_row extend downward into base.
            for rr in range(anchor_row + 1, piece["h"]):
                grid_r = edge + (rr - anchor_row)
                if grid_r < 0 or grid_r >= H:
                    return False
                for cc in range(Wtower):
                    if piece["mask"][rr, cc]:
                        gc = c0 + cc
                        if gc < 0 or gc >= W:
                            return False
                        if band_mask[grid_r, gc]:
                            return False
        else:  # bottom
            # piece is below band; rows before anchor_row extend upward into base.
            for rr in range(0, anchor_row):
                grid_r = edge - (anchor_row - rr)
                if grid_r < 0 or grid_r >= H:
                    return False
                for cc in range(Wtower):
                    if piece["mask"][rr, cc]:
                        gc = c0 + cc
                        if gc < 0 or gc >= W:
                            return False
                        if band_mask[grid_r, gc]:
                            return False
        return True

    def cavity_ok_vertical(piece, anchor_col, slot):
        side = slot["side"]
        edge = slot["edge"]
        r0 = slot["start"] - 1
        L = slot["len"]
        Htower = L + 2

        if side == "left":
            # piece is left of band; columns after anchor_col extend right into base.
            for cc in range(anchor_col + 1, piece["w"]):
                grid_c = edge + (cc - anchor_col)
                if grid_c < 0 or grid_c >= W:
                    return False
                for rr in range(Htower):
                    if piece["mask"][rr, cc]:
                        gr = r0 + rr
                        if gr < 0 or gr >= H:
                            return False
                        if band_mask[gr, grid_c]:
                            return False
        else:  # right
            # piece is right of band; columns before anchor_col extend left into base.
            for cc in range(0, anchor_col):
                grid_c = edge - (anchor_col - cc)
                if grid_c < 0 or grid_c >= W:
                    return False
                for rr in range(Htower):
                    if piece["mask"][rr, cc]:
                        gr = r0 + rr
                        if gr < 0 or gr >= H:
                            return False
                        if band_mask[gr, grid_c]:
                            return False
        return True

    def vertical_order_top_to_bottom(start_piece, remaining, side):
        """
        Build a stack ordered from band outward.
        side='top': start_piece is lowest (closest to band); return order low -> high.
        side='bottom': start_piece is highest (closest to band); return order high -> low.
        """
        # Try to build a path starting from start_piece.
        # state: (last_piece, order)
        best = None

        def dist_score(order):
            # Sum distance between each piece's original center and approximate final center.
            # For slot assignment score only; exact final position not yet known.
            # We prefer pieces that were originally closer along the band axis.
            total = 0.0
            if side == "top":
                # order low to high: last is outer top.  Approx row increases? We'll just use original rows.
                for idx, p in enumerate(order):
                    total += abs(p["crow"])
            else:
                for idx, p in enumerate(order):
                    total += abs(p["crow"])
            return total

        def dfs(last, order, used):
            nonlocal best
            if len(order) == 1 + len(remaining):
                score = dist_score(order)
                if best is None or score < best[0]:
                    best = (score, order[:])
                return

            # candidates for next outward piece
            for p in remaining:
                if p["id"] in used:
                    continue

                if side == "top":
                    # p will be above last
                    if valid_offsets(p, last):
                        nused = used | {p["id"]}
                        dfs(p, order + [p], nused)
                else:
                    # p will be below last
                    if valid_offsets(last, p):
                        nused = used | {p["id"]}
                        dfs(p, order + [p], nused)

        dfs(start_piece, [start_piece], {start_piece["id"]})

        if best is None:
            return None
        return best[1]

    def vertical_offsets(order, side):
        """
        Return list of offsets between consecutive pieces.
        side='top': order low -> high; offset_i = top(order[i]) - top(order[i-1])? 
        For top side: order[0] is lowest, order[1] is above it.
        If A=order[i] above B=order[i-1], d = B.top - A.top.
        For bottom side: order[0] is highest, order[1] is below it.
        If A=order[i-1] above B=order[i], d = B.top - A.top.
        """
        offs = []
        if side == "top":
            for i in range(1, len(order)):
                above = order[i]
                below = order[i - 1]
                vals = valid_offsets(above, below)
                if not vals:
                    return None
                offs.append(vals[0])
        else:
            for i in range(1, len(order)):
                above = order[i - 1]
                below = order[i]
                vals = valid_offsets(above, below)
                if not vals:
                    return None
                offs.append(vals[0])
        return offs

    def horizontal_order_left_to_right(start_piece, remaining, side):
        # side='left': start is leftmost (closest), return left -> right.
        # side='right': start is rightmost (closest), return right -> left.
        best = None

        def dist_score(order):
            return sum(abs(p["ccol"]) for p in order)

        def dfs(last, order, used):
            nonlocal best
            if len(order) == 1 + len(remaining):
                score = dist_score(order)
                if best is None or score < best[0]:
                    best = (score, order[:])
                return

            for p in remaining:
                if p["id"] in used:
                    continue
                if side == "left":
                    # p is to the right of last: last above? For horizontal band vertical orientation:
                    # We use compatibility by treating rows as columns: transpose masks.
                    above_like = make_vertical_transpose(last)
                    below_like = make_vertical_transpose(p)
                    if valid_offsets(above_like, below_like):
                        nused = used | {p["id"]}
                        dfs(p, order + [p], nused)
                else:
                    above_like = make_vertical_transpose(p)
                    below_like = make_vertical_transpose(last)
                    if valid_offsets(above_like, below_like):
                        nused = used | {p["id"]}
                        dfs(p, order + [p], nused)

        dfs(start_piece, [start_piece], {start_piece["id"]})
        if best is None:
            return None
        return best[1]

    def make_vertical_transpose(piece):
        m = piece["mask"]
        h, w = m.shape
        # Transpose so vertical dimension becomes height for compatibility routine.
        tm = np.zeros((w, h), dtype=bool)
        tm[:] = m.T
        return {
            "h": w,
            "w": h,
            "mask": tm,
        }

    def horizontal_offsets(order, side):
        offs = []
        if side == "left":
            for i in range(1, len(order)):
                # previous is left, current is right.  Treat left as above in transposed space.
                A = make_vertical_transpose(order[i - 1])
                B = make_vertical_transpose(order[i])
                vals = valid_offsets(A, B)
                if not vals:
                    return None
                offs.append(vals[0])
        else:
            for i in range(1, len(order)):
                # previous is right (closest), current is left (farther).
                # In transposed vertical terms, current is above previous.
                A = make_vertical_transpose(order[i])
                B = make_vertical_transpose(order[i - 1])
                vals = valid_offsets(A, B)
                if not vals:
                    return None
                offs.append(vals[0])
        return offs

    # Evaluate all slots and choose best assignment greedily.
    slot_plans = []

    for slot in slots:
        L = slot["len"]
        tower_cross = L + 2
        cand = [p for p in pieces if p["id"] not in used_piece_ids and cross_dim(p) == tower_cross]
        if not cand:
            continue

        side = slot["side"]
        best_plan = None

        for anchor in cand:
            if orientation == "horizontal":
                anchor_rows = piece_interface_rows(anchor, tower_cross, L)
            else:
                # Interface columns for vertical orientation.
                anchor_rows = []
                for c in range(anchor["w"]):
                    col = anchor["mask"][:, c]
                    ok = True
                    for r in range(tower_cross):
                        val = bool(col[r])
                        if 1 <= r <= L:
                            if not val:
                                ok = False
                                break
                        else:
                            if val:
                                ok = False
                                break
                    if ok:
                        anchor_rows.append(c)
                # rename for uniformity
                anchor_rows = anchor_rows

            if not anchor_rows:
                continue

            # Choose the best anchor interface row/col that fits cavity.
            # Prefer the interface closest to the band (smallest inward depth).
            chosen_anchor_interface = None
            if orientation == "horizontal":
                for r in anchor_rows:
                    if cavity_ok_horizontal(anchor, r, slot):
                        chosen_anchor_interface = r
                        break
            else:
                for c in anchor_rows:
                    if cavity_ok_vertical(anchor, c, slot):
                        chosen_anchor_interface = c
                        break

            if chosen_anchor_interface is None:
                continue

            # Try to build full stack.
            remaining = [p for p in cand if p is not anchor]
            if orientation == "horizontal":
                order = vertical_order_top_to_bottom(anchor, remaining, side)
            else:
                order = horizontal_order_left_to_right(anchor, remaining, side)

            if order is None:
                # If not all pieces in this candidate group can be chained,
                # still allow a single-piece stack if anchor alone works.
                if len(cand) > 1:
                    continue
                order = [anchor]

            if orientation == "horizontal":
                offs = vertical_offsets(order, side) if len(order) > 1 else []
            else:
                offs = horizontal_offsets(order, side) if len(order) > 1 else []

            if offs is None:
                continue

            # Score plan by total original distance to the slot center.
            if orientation == "horizontal":
                slot_center = (slot["start"] + slot["end"]) / 2.0
                score = sum(abs(p["ccol"] - slot_center) for p in order)
            else:
                slot_center = (slot["start"] + slot["end"]) / 2.0
                score = sum(abs(p["crow"] - slot_center) for p in order)

            plan = {
                "slot": slot,
                "anchor": anchor,
                "anchor_interface": chosen_anchor_interface,
                "order": order,
                "offsets": offs,
                "score": score,
            }

            if best_plan is None or plan["score"] < best_plan["score"]:
                best_plan = plan

        if best_plan is not None:
            slot_plans.append(best_plan)

    # Choose non-overlapping piece plans.  Since examples use disjoint piece sets,
    # sort by score and keep plans whose pieces are unused.
    slot_plans.sort(key=lambda p: p["score"])
    chosen_plans = []
    for plan in slot_plans:
        if all(p["id"] not in used_piece_ids for p in plan["order"]):
            chosen_plans.append(plan)
            for p in plan["order"]:
                used_piece_ids.add(p["id"])

    # ------------------------------------------------------------
    # Draw chosen stacks.
    # ------------------------------------------------------------
    def draw_mask(color, mask, top, left):
        h, w = mask.shape
        for i in range(h):
            rr = top + i
            if rr < 0 or rr >= H:
                continue
            row = mask[i]
            nz = np.nonzero(row)[0]
            for j in nz:
                cc = left + j
                if cc < 0 or cc >= W:
                    continue
                cur = out[rr, cc]
                if cur == 0 or cur == band_color or color > cur:
                    out[rr, cc] = color

    def fill_cavity(plan):
        """
        Fill band cavity zeros connected to the slot with the anchor color.
        We only fill zeros in the band's bounding box that lie in the tower
        cross interval and on the inward side of the slot edge.
        """
        slot = plan["slot"]
        color = plan["anchor"]["color"]
        side = slot["side"]
        edge = slot["edge"]
        L = slot["len"]

        if orientation == "horizontal":
            c0 = slot["start"]
            c1 = slot["end"] + 1
            # Look inward for a bounded depth, say up to band thickness.
            if side == "top":
                r0 = edge
                r1 = min(H, band["rmax"] + 1)
            else:
                r0 = max(0, band["rmin"])
                r1 = edge + 1
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if 0 <= r < H and 0 <= c < W and not band_mask[r, c] and out[r, c] == 0:
                        out[r, c] = color
        else:
            r0 = slot["start"]
            r1 = slot["end"] + 1
            if side == "left":
                c0 = edge
                c1 = min(W, band["cmax"] + 1)
            else:
                c0 = max(0, band["cmin"])
                c1 = edge + 1
            for c in range(c0, c1):
                for r in range(r0, r1):
                    if 0 <= r < H and 0 <= c < W and not band_mask[r, c] and out[r, c] == 0:
                        out[r, c] = color

    for plan in chosen_plans:
        slot = plan["slot"]
        side = slot["side"]
        L = slot["len"]
        order = plan["order"]
        offsets = plan["offsets"]
        anchor_interface = plan["anchor_interface"]

        if orientation == "horizontal":
            Wtower = L + 2
            c_left = slot["start"] - 1
            edge = slot["edge"]

            # Determine absolute top for anchor.
            if side == "top":
                # anchor row anchor_interface lies at edge (band top edge)
                anchor_top = edge - anchor_interface
            else:
                # anchor row anchor_interface lies at edge (band bottom edge)
                anchor_top = edge - anchor_interface

            tops = []
            top = anchor_top
            tops.append(top)

            if side == "top":
                # order low -> high; offsets: for i=1, order[i] is above order[i-1]
                # offset d = below.top - above.top => above.top = below.top - d
                for d in offsets:
                    top = tops[-1] - d
                    tops.append(top)
            else:
                # order high -> low; offsets: below.top - above.top = d
                for d in offsets:
                    top = tops[-1] + d
                    tops.append(top)

            for p, t in zip(order, tops):
                draw_mask(p["color"], p["mask"], t, c_left)

            # Fill cavity after pieces so any unfilled notch zeros get anchor color.
            fill_cavity(plan)

        else:
            Htower = L + 2
            r_top = slot["start"] - 1
            edge = slot["edge"]

            # anchor column anchor_interface lies at edge
            if side == "left":
                anchor_left = edge - anchor_interface
            else:
                anchor_left = edge - anchor_interface

            lefts = []
            left = anchor_left
            lefts.append(left)

            if side == "left":
                # order left -> right; offsets in transposed sense: right.left - left.left = d
                for d in offsets:
                    left = lefts[-1] + d
                    lefts.append(left)
            else:
                # order right -> left; offsets in transposed sense: left.left = right.left - d
                for d in offsets:
                    left = lefts[-1] - d
                    lefts.append(left)

            for p, l in zip(order, lefts):
                draw_mask(p["color"], p["mask"], r_top, l)

            fill_cavity(plan)

    # If any pieces were not assigned, leave them in their original positions
    # as a graceful fallback.
    for p in pieces:
        if p["id"] in used_piece_ids:
            continue
        top = p["rmin"]
        left = p["cmin"]
        draw_mask(p["color"], p["mask"], top, left)

    return out