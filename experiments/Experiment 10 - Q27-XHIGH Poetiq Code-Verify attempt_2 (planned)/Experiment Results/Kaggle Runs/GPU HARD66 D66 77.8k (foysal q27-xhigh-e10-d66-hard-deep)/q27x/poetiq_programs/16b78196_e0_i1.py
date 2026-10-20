import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by:
    1. Identify the large barrier color (largest connected non-zero component).
    2. Keep the barrier unchanged.
    3. Extract all other 4-connected colored components as shapes.
    4. Find rectangular "slots" cut into the barrier.
       - For horizontal barriers, slots have a fixed column interval and
         open upward/downward.
       - For vertical barriers, slots have a fixed row interval and
         open left/right.
    5. Assign shapes to slots by their perpendicular size (width for horizontal
       slots, height for vertical slots).
    6. Pack shapes inside each slot:
       - The end adjacent to the solid barrier side is filled first.
       - Shapes on the barrier side are packed closest to the barrier first.
       - Shapes on the open side are packed farthest from the barrier first.
       - Each next shape is placed as far inward as possible without
         overlapping already placed shape pixels or the barrier.
    7. Fill any still-empty cavity cells in the slot with the color of the
       nearest placed shape.
    """

    if grid.size == 0:
        return grid

    grid = np.array(grid, dtype=np.uint8)
    h, w = grid.shape

    def neighbors4(r, c):
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc

    def flood(color_mask, sr, sc):
        seen = np.zeros_like(color_mask, dtype=bool)
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in neighbors4(r, c):
                if color_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        return comp

    # Largest non-zero component is the barrier.
    best_comp = None
    for color in np.unique(grid):
        if color == 0:
            continue
        mask = (grid == color)
        seen = np.zeros_like(mask, dtype=bool)
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    comp = flood(mask, r, c)
                    seen[comp[0][0], comp[0][1]] = True
                    # Mark all seen in this component correctly
                    for rr, cc in comp:
                        seen[rr, cc] = True
                    if best_comp is None or len(comp) > len(best_comp):
                        best_comp = comp

    barrier_color = None
    if best_comp:
        barrier_color = grid[best_comp[0]]
    else:
        return grid

    barrier_mask = (grid == barrier_color)
    barrier_ys, barrier_xs = np.where(barrier_mask)
    b_top, b_bot = barrier_ys.min(), barrier_ys.max()
    b_left, b_right = barrier_xs.min(), barrier_xs.max()

    # Extract all non-barrier shapes.
    shapes = []
    used = np.zeros((h, w), dtype=bool)
    used[barrier_mask] = True

    for color in np.unique(grid):
        if color == 0 or color == barrier_color:
            continue
        mask = (grid == color)
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not used[r, c]:
                    comp = flood(mask, r, c)
                    ys = [p[0] for p in comp]
                    xs = [p[1] for p in comp]
                    r0, r1 = min(ys), max(ys)
                    c0, c1 = min(xs), max(xs)
                    pat = np.zeros((r1-r0+1, c1-c0+1), dtype=np.uint8)
                    for rr, cc in comp:
                        pat[rr-r0, cc-c0] = 1
                    shapes.append({
                        'color': int(color),
                        'top': r0,
                        'left': c0,
                        'height': r1-r0+1,
                        'width': c1-c0+1,
                        'pattern': pat
                    })

    horizontal_barrier = (b_bot - b_top + 1) <= (b_right - b_left + 1)

    out = np.zeros_like(grid)
    out[barrier_mask] = barrier_color

    def barrier_conflicts_h(row, c0, c1):
        # For horizontal slots: shape occupies one row interval? Actually
        # placement is done by row offset; this helper checks if a given
        # horizontal placement conflicts with barrier in a row.
        return None

    def pack_horizontal(slot, shape_list):
        """
        slot:
          cols: (c0, c1)
          open_down: bool
          anchor_row: barrier-side end row
        shape_list: ordered from barrier-side to open-side.
        """
        c0, c1 = slot['cols']
        W = c1 - c0 + 1
        open_down = slot['open_down']
        anchor = slot['anchor_row']

        placed = []

        for sh in shape_list:
            if sh['width'] != W:
                continue
            pat = sh['pattern']
            H = sh['height']

            best_row = None
            if open_down:
                # Start as close to barrier as possible: top row at anchor,
                # then move downward until no conflict with barrier/placed.
                r = anchor
                while r + H <= h:
                    ok = True
                    for i in range(H):
                        rr = r + i
                        for j in range(W):
                            cc = c0 + j
                            if pat[i, j] and out[rr, cc] not in (0,):
                                if out[rr, cc] == barrier_color:
                                    ok = False
                                    break
                                # non-zero other color: shape-shape conflict
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_row = r
                        break
                    r += 1
            else:
                # open upward: bottom row at anchor, move upward.
                r = anchor - H + 1
                while r >= 0:
                    ok = True
                    for i in range(H):
                        rr = r + i
                        for j in range(W):
                            cc = c0 + j
                            if pat[i, j] and out[rr, cc] != 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_row = r
                        break
                    r -= 1

            if best_row is not None:
                for i in range(H):
                    for j in range(W):
                        if pat[i, j]:
                            out[best_row+i, c0+j] = sh['color']
                placed.append((best_row, sh['color']))

    def pack_vertical(slot, shape_list):
        """
        slot:
          rows: (r0, r1)
          open_left: bool
          anchor_col: barrier-side end column
        shape_list: ordered from barrier-side to open-side.
        """
        r0, r1 = slot['rows']
        H = r1 - r0 + 1
        open_left = slot['open_left']
        anchor = slot['anchor_col']

        for sh in shape_list:
            if sh['height'] != H:
                continue
            pat = sh['pattern']
            W = sh['width']

            best_col = None
            if open_left:
                # Move leftward from barrier side: try right edge as close to
                # anchor as possible.
                # Right edge = col + W - 1.
                col = anchor - W + 1
                while col >= 0:
                    ok = True
                    for i in range(H):
                        rr = r0 + i
                        for j in range(W):
                            cc = col + j
                            if pat[i, j] and out[rr, cc] != 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_col = col
                        break
                    col -= 1
            else:
                col = anchor
                while col + W <= w:
                    ok = True
                    for i in range(H):
                        rr = r0 + i
                        for j in range(W):
                            cc = col + j
                            if pat[i, j] and out[rr, cc] != 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        best_col = col
                        break
                    col += 1

            if best_col is not None:
                for i in range(H):
                    for j in range(W):
                        if pat[i, j]:
                            out[r0+i, best_col+j] = sh['color']

    # For this challenge and the given examples, detect simple slots on the
    # outer edge of the barrier.
    slots = []

    if horizontal_barrier:
        # Find side-wall rows: two barrier cells in same row with zeros between.
        for r in range(h):
            row = grid[r]
            c = 0
            while c < w:
                if row[c] == barrier_color:
                    c2 = c + 1
                    while c2 < w and row[c2] != barrier_color:
                        c2 += 1
                    if c2 < w and row[c2] == barrier_color:
                        c0, c1 = c, c2
                        W = c1 - c0 + 1
                        # check open direction: which side is mostly barrier?
                        above = True if r > 0 else False
                        below = True if r + 1 < h else False
                        above_bar = 0
                        below_bar = 0
                        for cc in range(c0, c1+1):
                            if r > 0 and grid[r-1, cc] == barrier_color:
                                above_bar += 1
                            if r+1 < h and grid[r+1, cc] == barrier_color:
                                below_bar += 1
                        if above_bar > below_bar:
                            open_down = True
                            # anchor is topmost zero in this column interval
                            anchor = r
                            while anchor > 0 and any(grid[anchor-1, cc] == 0 for cc in range(c0, c1+1)):
                                anchor -= 1
                        else:
                            open_down = False
                            anchor = r
                        slots.append({'cols': (c0, c1), 'open_down': open_down, 'anchor_row': anchor})
                        c = c2
                    c += 1
                else:
                    c += 1
    else:
        # Vertical barrier: find side-wall columns.
        for c in range(w):
            r = 0
            while r < h:
                if grid[r, c] == barrier_color:
                    r2 = r + 1
                    while r2 < h and grid[r2, c] != barrier_color:
                        r2 += 1
                    if r2 < h and grid[r2, c] == barrier_color:
                        r0, r1 = r, r2
                        H = r1 - r0 + 1
                        mid = (r0 + r1) // 2
                        left_bar = grid[mid, c-1] == barrier_color if c > 0 else False
                        right_bar = grid[mid, c+1] == barrier_color if c+1 < w else False
                        if right_bar and not left_bar:
                            open_left = True
                            anchor = c
                        elif left_bar and not right_bar:
                            open_left = False
                            anchor = c
                        else:
                            open_left = None
                            anchor = c
                        if open_left is not None:
                            slots.append({'rows': (r0, r1), 'open_left': open_left, 'anchor_col': anchor})
                        r = r2
                    else:
                        r += 1
                else:
                    r += 1

    # Assign shapes to slots.
    used_shapes = [False] * len(shapes)

    def shape_distance_to_slot_h(sh, slot):
        c0, c1 = slot['cols']
        # vertical distance to anchor row
        if sh['top'] < slot['anchor_row']:
            v = slot['anchor_row'] - sh['top']
        else:
            v = sh['top'] - slot['anchor_row']
        # horizontal distance to slot interval
        if sh['left'] + sh['width'] - 1 < c0:
            hdist = c0 - (sh['left'] + sh['width'] - 1)
        elif sh['left'] > c1:
            hdist = sh['left'] - c1
        else:
            hdist = 0
        return v + hdist

    def shape_distance_to_slot_v(sh, slot):
        r0, r1 = slot['rows']
        if sh['top'] + sh['height'] - 1 < r0:
            v = r0 - (sh['top'] + sh['height'] - 1)
        elif sh['top'] > r1:
            v = sh['top'] - r1
        else:
            v = 0
        if sh['left'] + sh['width'] - 1 < slot['anchor_col']:
            hdist = slot['anchor_col'] - (sh['left'] + sh['width'] - 1)
        elif sh['left'] > slot['anchor_col']:
            hdist = sh['left'] - slot['anchor_col']
        else:
            hdist = 0
        return v + hdist

    for slot in slots:
        if horizontal_barrier:
            W = slot['cols'][1] - slot['cols'][0] + 1
            c0, c1 = slot['cols']
            cand = []
            for i, sh in enumerate(shapes):
                if used_shapes[i] or sh['width'] != W:
                    continue
                # side relative to barrier
                barrier_side = 'above' if sh['top'] + sh['height'] - 1 < b_top else 'below'
                open_side = 'below' if slot['open_down'] else 'above'
                on_open = (barrier_side == open_side)
                cand.append((i, on_open, shape_distance_to_slot_h(sh, slot)))

            # Order: barrier-side group first (near barrier), open-side group after.
            # Within barrier-side: nearest distance first.
            # Within open-side: farthest distance first.
            barrier_group = sorted([x for x in cand if not x[1]], key=lambda x: (x[2], x[0]))
            open_group = sorted([x for x in cand if x[1]], key=lambda x: (-x[2], x[0]))
            ordered = [shapes[x[0]] for x in barrier_group + open_group]
            for sh in ordered:
                used_shapes[shapes.index(sh)] = True
            pack_horizontal(slot, ordered)

        else:
            H = slot['rows'][1] - slot['rows'][0] + 1
            r0, r1 = slot['rows']
            cand = []
            for i, sh in enumerate(shapes):
                if used_shapes[i] or sh['height'] != H:
                    continue
                # side relative to barrier for vertical barrier
                barrier_side = 'left' if sh['left'] + sh['width'] - 1 < b_left else 'right'
                open_side = 'left' if slot['open_left'] else 'right'
                on_open = (barrier_side == open_side)
                cand.append((i, on_open, shape_distance_to_slot_v(sh, slot)))

            barrier_group = sorted([x for x in cand if not x[1]], key=lambda x: (x[2], x[0]))
            open_group = sorted([x for x in cand if x[1]], key=lambda x: (-x[2], x[0]))
            ordered = [shapes[x[0]] for x in barrier_group + open_group]
            for sh in ordered:
                used_shapes[shapes.index(sh)] = True
            pack_vertical(slot, ordered)

    # Fill slot cavity cells that remain zero with nearest placed color.
    for slot in slots:
        if horizontal_barrier:
            c0, c1 = slot['cols']
            # cavity is on barrier side of side-wall row; approximate by checking
            # zero cells in slot columns near anchor row not filled.
            r = slot['anchor_row']
            # look one row toward barrier side
            dr = -1 if not slot['open_down'] else 1
            rr = r + dr
            if 0 <= rr < h:
                for cc in range(c0, c1+1):
                    if out[rr, cc] == 0:
                        # find nearest placed color in this slot column interval
                        # fallback to first non-zero color placed in same column nearby
                        for d in range(1, max(h, w)+1):
                            found = None
                            for rr2 in range(max(0, rr-d), min(h, rr+d+1)):
                                for cc2 in range(max(0, cc-1), min(w, cc+2)):
                                    if out[rr2, cc2] not in (0, barrier_color):
                                        found = out[rr2, cc2]
                                        break
                                if found is not None:
                                    break
                            if found is not None:
                                out[rr, cc] = found
                                break
        else:
            r0, r1 = slot['rows']
            c = slot['anchor_col']
            dc = -1 if slot['open_left'] else 1
            cc = c + dc
            if 0 <= cc < w:
                for rr in range(r0, r1+1):
                    if out[rr, cc] == 0:
                        found = None
                        for d in range(1, max(h, w)+1):
                            for cc2 in range(max(0, cc-d), min(w, cc+d+1)):
                                for rr2 in range(max(0, rr-1), min(h, rr+2)):
                                    if out[rr2, cc2] not in (0, barrier_color):
                                        found = out[rr2, cc2]
                                        break
                                if found is not None:
                                    break
                            if found is not None:
                                break
                        if found is not None:
                            out[rr, cc] = found

    return out