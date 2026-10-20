import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move all non-barrier objects into the cavities formed by the largest
    connected colored component (the barrier).

    The transformation works as follows:
    1. The largest 4-connected non-zero component is treated as the barrier.
    2. All other non-zero components are treated as movable shapes.
    3. If the barrier is horizontally elongated, cavities are vertical slots
       cut into the top/bottom edge of the barrier.
       If it is vertically elongated, cavities are horizontal slots cut into
       the left/right edge.
    4. Each shape is assigned to the best cavity that can accommodate its
       perpendicular dimension (width for horizontal barrier, height for
       vertical barrier).
    5. Inside each cavity, shapes are ordered:
         - first, shapes originally on the solid/barrier side, nearest first;
         - then, shapes originally on the open side, farthest first.
    6. Shapes are packed into the cavity from the barrier side outward,
       preserving their original orientation.  Placement uses the earliest
       position that does not overlap already placed shapes.
    7. The barrier remains present unless overwritten by a shape inside the
       active cavity rectangle.
    """
    if grid.size == 0:
        return grid

    grid = np.array(grid, copy=True)
    if grid.dtype != np.uint8:
        grid = grid.astype(np.uint8)

    h, w = grid.shape

    def components(mask):
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)])
                    seen[r, c] = True
                    comp = []
                    while q:
                        y, x = q.popleft()
                        comp.append((y, x))
                        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                    comps.append(comp)
        return comps

    # Largest non-zero component is the barrier.
    best = None
    for color in np.unique(grid):
        if color == 0:
            continue
        for comp in components(grid == color):
            if best is None or len(comp) > len(best[0]):
                best = (comp, int(color))

    if best is None:
        return grid

    barrier_comp, barrier_color = best
    barrier_mask = (grid == barrier_color)

    barrier_ys, barrier_xs = np.where(barrier_mask)
    b_top, b_bot = int(barrier_ys.min()), int(barrier_ys.max())
    b_left, b_right = int(barrier_xs.min()), int(barrier_xs.max())
    b_height = b_bot - b_top + 1
    b_width = b_right - b_left + 1

    horizontal_barrier = b_height <= b_width

    # Extract movable shapes.
    shapes = []
    used = barrier_mask.copy()
    for color in np.unique(grid):
        if color == 0 or color == barrier_color:
            continue
        mask = (grid == color)
        seen = np.zeros_like(mask, dtype=bool)
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)])
                    seen[r, c] = True
                    comp = []
                    while q:
                        y, x = q.popleft()
                        comp.append((y, x))
                        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))

                    ys = [p[0] for p in comp]
                    xs = [p[1] for p in comp]
                    r0, r1 = min(ys), max(ys)
                    c0, c1 = min(xs), max(xs)
                    pat = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
                    for yy, xx in comp:
                        pat[yy-r0, xx-c0] = True

                    shapes.append({
                        'color': int(color),
                        'r0': r0, 'r1': r1,
                        'c0': c0, 'c1': c1,
                        'h': r1-r0+1,
                        'w': c1-c0+1,
                        'pat': pat,
                        'center': (0.5*(r0+r1), 0.5*(c0+c1))
                    })

    out = np.zeros_like(grid)
    out[barrier_mask] = barrier_color

    def count_kind(row, c0, c1):
        """For a horizontal slot candidate, classify what is above/below."""
        above_zero = above_bar = 0
        below_zero = below_bar = 0
        if row > 0:
            for c in range(c0, c1+1):
                if grid[row-1, c] == 0:
                    above_zero += 1
                elif grid[row-1, c] == barrier_color:
                    above_bar += 1
        if row+1 < h:
            for c in range(c0, c1+1):
                if grid[row+1, c] == 0:
                    below_zero += 1
                elif grid[row+1, c] == barrier_color:
                    below_bar += 1
        return above_zero, above_bar, below_zero, below_bar

    def count_kind_v(col, r0, r1):
        """For a vertical slot candidate, classify what is left/right."""
        left_zero = left_bar = 0
        right_zero = right_bar = 0
        if col > 0:
            for r in range(r0, r1+1):
                if grid[r, col-1] == 0:
                    left_zero += 1
                elif grid[r, col-1] == barrier_color:
                    left_bar += 1
        if col+1 < w:
            for r in range(r0, r1+1):
                if grid[r, col+1] == 0:
                    right_zero += 1
                elif grid[r, col+1] == barrier_color:
                    right_bar += 1
        return left_zero, left_bar, right_zero, right_bar

    slots = []

    if horizontal_barrier:
        # Detect horizontal barrier vertical slots:
        # a row of the barrier with a zero run between two barrier cells.
        for r in range(max(0, b_top-1), min(h, b_bot+2)):
            c = 0
            while c < w:
                if grid[r, c] == barrier_color:
                    c2 = c + 1
                    while c2 < w and grid[r, c2] != barrier_color:
                        c2 += 1
                    if c2 < w and grid[r, c2] == barrier_color:
                        c0, c1 = c, c2
                        width = c1 - c0 + 1
                        if width >= 2:
                            az, ab, bz, bb = count_kind(r, c0, c1)
                            if az > bz and az >= 1:
                                open_dir = 'up'
                            elif bz > az and bz >= 1:
                                open_dir = 'down'
                            else:
                                open_dir = None
                            if open_dir is not None:
                                slots.append({
                                    'orient': 'h',
                                    'c0': c0, 'c1': c1,
                                    'width': width,
                                    'mouth_row': r,
                                    'open': open_dir
                                })
                        c = c2
                    c += 1
                else:
                    c += 1
    else:
        # Detect vertical barrier horizontal slots:
        # a column of the barrier with a zero run between two barrier cells.
        for c in range(max(0, b_left-1), min(w, b_right+2)):
            r = 0
            while r < h:
                if grid[r, c] == barrier_color:
                    r2 = r + 1
                    while r2 < h and grid[r2, c] != barrier_color:
                        r2 += 1
                    if r2 < h and grid[r2, c] == barrier_color:
                        r0, r1 = r, r2
                        height = r1 - r0 + 1
                        if height >= 2:
                            lz, lb, rz, rb = count_kind_v(c, r0, r1)
                            if lz > rz and lz >= 1:
                                open_dir = 'left'
                            elif rz > lz and rz >= 1:
                                open_dir = 'right'
                            else:
                                open_dir = None
                            if open_dir is not None:
                                slots.append({
                                    'orient': 'v',
                                    'r0': r0, 'r1': r1,
                                    'height': height,
                                    'mouth_col': c,
                                    'open': open_dir
                                })
                        r = r2
                    r += 1
                else:
                    r += 1

    if not slots:
        # Fallback: leave barrier unchanged and remove shapes.
        return out

    # Assign shapes to slots.
    assigned = {id(slot): [] for slot in slots}

    def shape_slot_distance(shape, slot):
        if slot['orient'] == 'h':
            # Distance between shape vertical interval and mouth row.
            if shape['r1'] < slot['mouth_row']:
                vr = slot['mouth_row'] - shape['r1']
            elif shape['r0'] > slot['mouth_row']:
                vr = shape['r0'] - slot['mouth_row']
            else:
                vr = 0

            if shape['c1'] < slot['c0']:
                hc = slot['c0'] - shape['c1']
            elif shape['c0'] > slot['c1']:
                hc = shape['c0'] - slot['c1']
            else:
                hc = 0
            return vr + hc
        else:
            if shape['r1'] < slot['r0']:
                vr = slot['r0'] - shape['r1']
            elif shape['r0'] > slot['r1']:
                vr = shape['r0'] - slot['r1']
            else:
                vr = 0

            if shape['c1'] < slot['mouth_col']:
                hc = slot['mouth_col'] - shape['c1']
            elif shape['c0'] > slot['mouth_col']:
                hc = shape['c0'] - slot['mouth_col']
            else:
                hc = 0
            return vr + hc

    for sh in shapes:
        best_slot = None
        best_score = None
        for slot in slots:
            if slot['orient'] == 'h':
                fit = sh['w'] <= slot['width']
                dim_diff = slot['width'] - sh['w']
            else:
                fit = sh['h'] <= slot['height']
                dim_diff = slot['height'] - sh['h']
            if not fit:
                continue
            dist = shape_slot_distance(sh, slot)
            # Prefer exact perpendicular size, then closeness.
            score = (0 if dim_diff == 0 else 1, dim_diff, dist)
            if best_score is None or score < best_score:
                best_score = score
                best_slot = slot
        if best_slot is not None:
            assigned[id(best_slot)].append(sh)

    def shape_side(shape, slot):
        """Return whether a shape is on the barrier side of the slot."""
        if slot['orient'] == 'h':
            # Shape is above/below the barrier band.
            if shape['r1'] < b_top:
                side = 'above'
            elif shape['r0'] > b_bot:
                side = 'below'
            else:
                side = 'above' if shape['center'][0] < 0.5*(b_top+b_bot) else 'below'
            # For a slot opening downward, barrier side is above.
            # For a slot opening upward, barrier side is below.
            barrier_side = 'above' if slot['open'] == 'down' else 'below'
            return side == barrier_side
        else:
            if shape['c1'] < b_left:
                side = 'left'
            elif shape['c0'] > b_right:
                side = 'right'
            else:
                side = 'left' if shape['center'][1] < 0.5*(b_left+b_right) else 'right'
            barrier_side = 'right' if slot['open'] == 'left' else 'left'
            return side == barrier_side

    def distance_to_barrier(shape, slot):
        if slot['orient'] == 'h':
            if shape['r1'] < b_top:
                return b_top - shape['r1']
            if shape['r0'] > b_bot:
                return shape['r0'] - b_bot
            return 0
        else:
            if shape['c1'] < b_left:
                return b_left - shape['c1']
            if shape['c0'] > b_right:
                return shape['c0'] - b_right
            return 0

    def can_place(slot, sh, origin):
        """
        origin: (r, c) top-left for horizontal slots, (r, c) top-left for vertical slots.
        Checks only against already placed shapes, not barrier.
        """
        r, c = origin
        pat = sh['pat']
        sh_h, sh_w = sh['h'], sh['w']
        if r < 0 or c < 0 or r + sh_h > h or c + sh_w > w:
            return False

        # For shapes already placed, conflict if both are non-zero.
        for i in range(sh_h):
            for j in range(sh_w):
                if pat[i, j]:
                    if out[r+i, c+j] != 0 and out[r+i, c+j] != barrier_color:
                        return False
        return True

    def place_shape(slot, sh, origin):
        r, c = origin
        pat = sh['pat']
        for i in range(sh['h']):
            for j in range(sh['w']):
                if pat[i, j]:
                    out[r+i, c+j] = sh['color']

    def anchor_origin(slot, sh):
        """
        Initial origin for the first shape: one row/col on the barrier side
        of the mouth, aligned to the slot interval.
        """
        if slot['orient'] == 'h':
            c0 = slot['c0']
            if slot['open'] == 'down':
                r = slot['mouth_row'] - 1
            else:
                r = slot['mouth_row'] + 1 - sh['h']
            return r, c0
        else:
            r0 = slot['r0']
            if slot['open'] == 'right':
                c = slot['mouth_col'] - 1
            else:
                c = slot['mouth_col'] + 1 - sh['w']
            return r0, c

    def pack_slot(slot, shape_list):
        if not shape_list:
            return

        barrier_side = [sh for sh in shape_list if shape_side(sh, slot)]
        open_side = [sh for sh in shape_list if not shape_side(sh, slot)]

        barrier_side.sort(key=lambda sh: (distance_to_barrier(sh, slot), sh['color'], sh['r0'], sh['c0']))
        open_side.sort(key=lambda sh: (-distance_to_barrier(sh, slot), sh['color'], sh['r0'], sh['c0']))

        ordered = barrier_side + open_side

        for sh in ordered:
            r, c = anchor_origin(slot, sh)

            if slot['orient'] == 'h':
                step_r = 1 if slot['open'] == 'down' else -1
                step_c = 0
                limit = h if slot['open'] == 'down' else 0
            else:
                step_r = 0
                step_c = 1 if slot['open'] == 'right' else -1
                limit = w if slot['open'] == 'right' else 0

            placed = False
            while True:
                if can_place(slot, sh, (r, c)):
                    place_shape(slot, sh, (r, c))
                    placed = True
                    break

                r += step_r
                c += step_c
                if (slot['orient'] == 'h' and (r > limit or r < 0)):
                    break
                if (slot['orient'] == 'v' and (c > limit or c < 0)):
                    break

            if not placed:
                # If the exact anchor packing failed, try a safe fallback:
                # put the shape as close as possible along the open axis.
                if slot['orient'] == 'h':
                    c = slot['c0']
                    for rr in range(max(0, slot['mouth_row'] - 1), min(h, h - sh['h'] + 1)):
                        if can_place(slot, sh, (rr, c)):
                            place_shape(slot, sh, (rr, c))
                            break
                else:
                    r = slot['r0']
                    for cc in range(max(0, slot['mouth_col'] - 1), min(w, w - sh['w'] + 1)):
                        if can_place(slot, sh, (r, cc)):
                            place_shape(slot, sh, (r, cc))
                            break

    for slot in slots:
        pack_slot(slot, assigned.get(id(slot), []))

    return out