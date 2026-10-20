import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC transformation:

    1. The largest 4-connected non-zero component is the "barrier".
    2. All other non-zero components are movable shapes.
    3. The barrier has slots/notches on its long outer edge:
       - horizontal barrier -> vertical slots, matching shape width
       - vertical barrier   -> horizontal slots, matching shape height
    4. Slots are first "activated" by shapes that lie on the solid side of
       the slot. Shapes on the open side are then inserted only into active
       slots.
    5. Within each slot, shapes are packed from the solid side toward the
       open side:
         - solid-side shapes first, nearest to the barrier first
         - open-side shapes second, farthest from the barrier first
       Each shape is placed at the first collision-free position when moving
       from the barrier edge outward.
    6. The barrier is kept unchanged except that zero cells inside its cavity
       may be occupied by placed shapes.
    """

    if grid is None or grid.size == 0:
        return np.array(grid, copy=True)

    grid = np.array(grid, dtype=np.uint8)
    h, w = grid.shape
    out = np.zeros((h, w), dtype=np.uint8)

    def components(mask):
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)])
                    seen[r, c] = True
                    comp = [(r, c)]
                    while q:
                        y, x = q.popleft()
                        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                                comp.append((ny, nx))
                    comps.append(comp)
        return comps

    # Largest component is the barrier.
    best_comp = None
    for color in np.unique(grid):
        if color == 0:
            continue
        for comp in components(grid == color):
            if best_comp is None or len(comp) > len(best_comp):
                best_comp = comp

    if best_comp is None:
        return out

    barrier_color = int(grid[best_comp[0][0], best_comp[0][1]])
    barrier_mask = (grid == barrier_color)
    ys, xs = np.where(barrier_mask)
    b_top, b_bot = int(ys.min()), int(ys.max())
    b_left, b_right = int(xs.min()), int(xs.max())
    b_height = b_bot - b_top + 1
    b_width = b_right - b_left + 1

    # Extract all movable shapes.
    shapes = []
    used = barrier_mask.copy()
    for color in np.unique(grid):
        if color == 0 or color == barrier_color:
            continue
        mask = grid == color
        for comp in components(mask):
            if any(used[r, c] for r, c in comp):
                continue
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            pat = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                pat[r - r0, c - c0] = True
                used[r, c] = True
            shapes.append({
                "id": len(shapes),
                "color": int(color),
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "h": r1 - r0 + 1, "w": c1 - c0 + 1,
                "pat": pat,
                "cr": (r0 + r1) / 2.0,
                "cc": (c0 + c1) / 2.0,
                "assigned_slot": None,
            })

    horizontal_barrier = b_height <= b_width
    slots = []

    # Detect slots on the barrier's outer edge.
    if horizontal_barrier:
        # Vertical slots: zero run in top/bottom edge row flanked by barrier.
        for edge_row, pref_open in ((b_top, "up"), (b_bot, "down")):
            row = grid[edge_row]
            c = 0
            while c < w:
                if row[c] == 0:
                    c2 = c
                    while c2 < w and row[c2] == 0:
                        c2 += 1
                    c0 = c - 1
                    c1 = c2
                    if c0 >= 0 and c1 < w and row[c0] == barrier_color and row[c1] == barrier_color:
                        width = c1 - c0 + 1
                        if width >= 2:
                            above = below = 0
                            for cc in range(c0, c1 + 1):
                                if edge_row > 0 and grid[edge_row - 1, cc] == barrier_color:
                                    above += 1
                                if edge_row + 1 < h and grid[edge_row + 1, cc] == barrier_color:
                                    below += 1
                            if above > below:
                                open_dir = "down"
                            elif below > above:
                                open_dir = "up"
                            else:
                                open_dir = pref_open
                            slots.append({
                                "orient": "v",
                                "c0": c0, "c1": c1, "width": width,
                                "edge_row": edge_row,
                                "open": open_dir,
                            })
                    c = c2
                else:
                    c += 1
    else:
        # Horizontal slots: zero run in left/right edge column flanked by barrier.
        for edge_col, pref_open in ((b_left, "left"), (b_right, "right")):
            col = grid[:, edge_col]
            r = 0
            while r < h:
                if col[r] == 0:
                    r2 = r
                    while r2 < h and col[r2] == 0:
                        r2 += 1
                    r0 = r - 1
                    r1 = r2
                    if r0 >= 0 and r1 < h and grid[r0, edge_col] == barrier_color and grid[r1, edge_col] == barrier_color:
                        height = r1 - r0 + 1
                        if height >= 2:
                            left = right = 0
                            for rr in range(r0, r1 + 1):
                                if edge_col > 0 and grid[rr, edge_col - 1] == barrier_color:
                                    left += 1
                                if edge_col + 1 < w and grid[rr, edge_col + 1] == barrier_color:
                                    right += 1
                            if left > right:
                                open_dir = "right"
                            elif right > left:
                                open_dir = "left"
                            else:
                                open_dir = pref_open
                            slots.append({
                                "orient": "h",
                                "r0": r0, "r1": r1, "height": height,
                                "edge_col": edge_col,
                                "open": open_dir,
                            })
                    r = r2
                else:
                    r += 1

    if not slots:
        out[barrier_mask] = barrier_color
        return out

    def matches(shape, slot):
        if slot["orient"] == "v":
            return shape["w"] == slot["width"]
        return shape["h"] == slot["height"]

    def shape_side(shape, slot):
        """Return 'solid' or 'open' side for this shape relative to slot."""
        if slot["orient"] == "v":
            if slot["open"] == "down":
                if shape["r1"] < b_top:
                    return "solid"
                if shape["r0"] > b_bot:
                    return "open"
                return "solid" if shape["cr"] < slot["edge_row"] else "open"
            else:  # up
                if shape["r0"] > b_bot:
                    return "solid"
                if shape["r1"] < b_top:
                    return "open"
                return "solid" if shape["cr"] > slot["edge_row"] else "open"
        else:
            if slot["open"] == "right":
                if shape["c1"] < b_left:
                    return "solid"
                if shape["c0"] > b_right:
                    return "open"
                return "solid" if shape["cc"] < slot["edge_col"] else "open"
            else:  # left
                if shape["c0"] > b_right:
                    return "solid"
                if shape["c1"] < b_left:
                    return "open"
                return "solid" if shape["cc"] > slot["edge_col"] else "open"

    def perp_dist(shape, slot):
        if slot["orient"] == "v":
            if shape["r1"] < b_top:
                return b_top - shape["r1"]
            if shape["r0"] > b_bot:
                return shape["r0"] - b_bot
            return 0
        else:
            if shape["c1"] < b_left:
                return b_left - shape["c1"]
            if shape["c0"] > b_right:
                return shape["c0"] - b_right
            return 0

    def parallel_dist(shape, slot):
        if slot["orient"] == "v":
            c0, c1 = slot["c0"], slot["c1"]
            if shape["c1"] < c0:
                return c0 - shape["c1"]
            if shape["c0"] > c1:
                return shape["c0"] - c1
            return 0
        else:
            r0, r1 = slot["r0"], slot["r1"]
            if shape["r1"] < r0:
                return r0 - shape["r1"]
            if shape["r0"] > r1:
                return shape["r0"] - r1
            return 0

    # Assign solid-side shapes first. They activate slots.
    slot_active = {id(s): False for s in slots}
    slot_shapes = {id(s): [] for s in slots}

    for sh in shapes:
        best = None
        for slot in slots:
            if not matches(sh, slot):
                continue
            if shape_side(sh, slot) != "solid":
                continue
            score = (perp_dist(sh, slot), parallel_dist(sh, slot), sh["color"], sh["id"])
            if best is None or score < best[0]:
                best = (score, slot)
        if best is not None:
            slot = best[1]
            sh["assigned_slot"] = slot
            slot_shapes[id(slot)].append(sh)
            slot_active[id(slot)] = True

    # Assign open-side shapes to active slots only.
    for sh in shapes:
        if sh["assigned_slot"] is not None:
            continue
        best = None
        for slot in slots:
            if not matches(sh, slot):
                continue
            if shape_side(sh, slot) != "open":
                continue
            if not slot_active[id(slot)]:
                continue
            score = (perp_dist(sh, slot), parallel_dist(sh, slot), sh["color"], sh["id"])
            if best is None or score < best[0]:
                best = (score, slot)
        if best is not None:
            slot = best[1]
            sh["assigned_slot"] = slot
            slot_shapes[id(slot)].append(sh)
            slot_active[id(slot)] = True

    out[barrier_mask] = barrier_color

    def can_place_v(slot, sh, top):
        c0 = slot["c0"]
        pat = sh["pat"]
        if top < 0 or top + sh["h"] > h:
            return False
        for i in range(sh["h"]):
            for j in range(sh["w"]):
                if pat[i, j] and out[top + i, c0 + j] != 0:
                    return False
        return True

    def place_v(slot, sh, top):
        c0 = slot["c0"]
        pat = sh["pat"]
        for i in range(sh["h"]):
            for j in range(sh["w"]):
                if pat[i, j]:
                    out[top + i, c0 + j] = sh["color"]

    def can_place_h(slot, sh, left):
        r0 = slot["r0"]
        pat = sh["pat"]
        if left < 0 or left + sh["w"] > w:
            return False
        for i in range(sh["h"]):
            for j in range(sh["w"]):
                if pat[i, j] and out[r0 + i, left + j] != 0:
                    return False
        return True

    def place_h(slot, sh, left):
        r0 = slot["r0"]
        pat = sh["pat"]
        for i in range(sh["h"]):
            for j in range(sh["w"]):
                if pat[i, j]:
                    out[r0 + i, left + j] = sh["color"]

    for slot in slots:
        list_sh = slot_shapes.get(id(slot), [])
        if not list_sh:
            continue

        solid = [sh for sh in list_sh if shape_side(sh, slot) == "solid"]
        open_side = [sh for sh in list_sh if shape_side(sh, slot) == "open"]

        solid.sort(key=lambda sh: (perp_dist(sh, slot), parallel_dist(sh, slot), sh["color"], sh["id"]))
        open_side.sort(key=lambda sh: (-perp_dist(sh, slot), -parallel_dist(sh, slot), sh["color"], sh["id"]))
        ordered = solid + open_side

        if slot["orient"] == "v":
            # Vertical slot: fixed columns, move along rows.
            edge = slot["edge_row"]
            for sh in ordered:
                placed = False
                if slot["open"] == "down":
                    # Open side is below. Start with shape's bottom touching edge.
                    start = edge - sh["h"] + 1
                    start = max(0, start)
                    for top in range(start, h - sh["h"] + 1):
                        if can_place_v(slot, sh, top):
                            place_v(slot, sh, top)
                            placed = True
                            break
                else:  # up
                    # Open side is above. Start with shape's top at edge.
                    start = min(edge, h - sh["h"])
                    for top in range(start, -1, -1):
                        if can_place_v(slot, sh, top):
                            place_v(slot, sh, top)
                            placed = True
                            break
                if not placed:
                    # Safe fallback: place as close as possible without overlap.
                    if slot["open"] == "down":
                        for top in range(h - sh["h"], -1, -1):
                            if can_place_v(slot, sh, top):
                                place_v(slot, sh, top)
                                break
                    else:
                        for top in range(0, h - sh["h"] + 1):
                            if can_place_v(slot, sh, top):
                                place_v(slot, sh, top)
                                break
        else:
            # Horizontal slot: fixed rows, move along columns.
            edge = slot["edge_col"]
            for sh in ordered:
                placed = False
                if slot["open"] == "right":
                    # Open side is right. Start with shape's left edge touching edge.
                    start = edge - sh["w"] + 1
                    start = max(0, start)
                    for left in range(start, w - sh["w"] + 1):
                        if can_place_h(slot, sh, left):
                            place_h(slot, sh, left)
                            placed = True
                            break
                else:  # left
                    # Open side is left. Start with shape's right edge at edge.
                    start = min(edge, w - sh["w"])
                    for left in range(start, -1, -1):
                        if can_place_h(slot, sh, left):
                            place_h(slot, sh, left)
                            placed = True
                            break
                if not placed:
                    if slot["open"] == "right":
                        for left in range(w - sh["w"], -1, -1):
                            if can_place_h(slot, sh, left):
                                place_h(slot, sh, left)
                                break
                    else:
                        for left in range(0, w - sh["w"] + 1):
                            if can_place_h(slot, sh, left):
                                place_h(slot, sh, left)
                                break

    return out