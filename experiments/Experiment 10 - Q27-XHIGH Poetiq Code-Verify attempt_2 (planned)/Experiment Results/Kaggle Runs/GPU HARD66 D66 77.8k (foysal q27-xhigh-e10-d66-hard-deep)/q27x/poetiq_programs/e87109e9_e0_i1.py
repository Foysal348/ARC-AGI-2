import numpy as np
from collections import Counter, deque

def _connected_components(mask):
    """Return bounding boxes of 4-connected components in a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []

    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                q = deque([(i, j)])
                seen[i, j] = True
                r0 = r1 = i
                c0 = c1 = j
                while q:
                    r, c = q.popleft()
                    r0 = min(r0, r)
                    r1 = max(r1, r)
                    c0 = min(c0, c)
                    c1 = max(c1, c)

                    for nr, nc in ((r + 1, c), (r - 1, c),
                                   (r, c + 1), (r, c - 1)):
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))

                boxes.append((r0, r1, c0, c1))
    return boxes


def _legend_bits(grid):
    """
    Read the top legend.

    The legend row contains groups of four cells separated by 5s.
    In each group, the non-zero, non-5 color appears either on the left
    or on the right side.  This is used as the 'turn' bit:
        left  -> turn left relative to the outward direction
        right -> turn right relative to the outward direction
    """
    bits = {}

    # Use the first non-border legend row (usually row 1).
    row = None
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r])
        if 5 in vals and 0 in vals and len(vals) > 2:
            row = grid[r]
            break
    if row is None:
        return bits

    h, w = row.shape

    # Find group intervals between vertical 5 separators.
    groups = []
    i = 0
    while i < w:
        if row[i] == 5:
            i += 1
            continue
        j = i
        while j < w and row[j] != 5:
            j += 1
        groups.append((i, j))
        i = j

    for a, b in groups:
        for c in range(a, b):
            v = row[c]
            if v not in (0, 5):
                # Offset within the group; left/right is enough for these tasks.
                offset = c - a
                bits[int(v)] = "right" if offset >= 2 else "left"
                break
    return bits


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Remove the top 5-legend frame and add 8 corridors to the main map.

    The top frame is a legend.  The main map contains a central 8 block,
    a background color, and up to four side blocks (left/right/top/bottom).
    For each side block, a corridor is drawn along the side facing the
    central 8.  The legend tells which way the corridor turns:
        legend color on the right -> turn right from the outward direction
        legend color on the left  -> turn left from the outward direction

    Corridors are as thick as the central 8 block.
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape

    # 1. Remove the top legend frame: find the second all-5 row.
    cut = None
    all5_rows = [r for r in range(H) if np.all(grid[r] == 5)]
    if len(all5_rows) >= 2:
        cut = all5_rows[1] + 1
    else:
        # Fallback: no top frame.
        cut = 0

    main = grid[cut:].copy()
    mh, mw = main.shape

    # If there is no main area, return what we have.
    if mh == 0 or mw == 0:
        return grid

    # 2. Background color is the most frequent color in the main area.
    bg = Counter(main.flatten()).most_common(1)[0][0]

    # 3. Central 8 block.
    s_boxes = [b for b in _connected_components(main == 8)]
    if not s_boxes:
        return grid[cut:]

    sr0, sr1, sc0, sc1 = max(s_boxes, key=lambda b: (b[1] - b[0]) * (b[3] - b[2]))
    th = sr1 - sr0 + 1  # thickness for horizontal lines
    tw = sc1 - sc0 + 1  # thickness for vertical lines

    # 4. Legend bits.
    bits = _legend_bits(grid)

    # 5. Non-background, non-8 components in main area.
    comps = []
    colors = np.unique(main)
    for c in colors:
        if c in (bg, 8):
            continue
        for (r0, r1, c0, c1) in _connected_components(main == c):
            comps.append((int(c), r0, r1, c0, c1))

    # 6. Assign components to left/right/top/bottom sides.
    center_r = (sr0 + sr1) / 2.0
    center_c = (sc0 + sc1) / 2.0

    sides = {}
    for c, r0, r1, c0, c1 in comps:
        cr = (r0 + r1) / 2.0
        cc = (c0 + c1) / 2.0

        overlaps_r = (r0 <= sr1 and r1 >= sr0)
        overlaps_c = (c0 <= sc1 and c1 >= sc0)

        if overlaps_c and not overlaps_r:
            side = "T" if cr < center_r else "B"
        elif overlaps_r and not overlaps_c:
            side = "L" if cc < center_c else "R"
        elif not overlaps_r and not overlaps_c:
            dr = abs(cr - center_r)
            dc = abs(cc - center_c)
            if dc > dr:
                side = "L" if cc < center_c else "R"
            else:
                side = "T" if cr < center_r else "B"
        else:
            # Both overlap: use the more extreme direction, or keep the best
            # existing side if this is clearly not a side block.
            if r0 < sr0:
                side = "T"
            elif r1 > sr1:
                side = "B"
            elif c0 < sc0:
                side = "L"
            elif c1 > sc1:
                side = "R"
            else:
                continue

        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        dist = abs(cr - center_r) + abs(cc - center_c)

        # Prefer the component that is most strongly on that side / largest.
        if side not in sides:
            sides[side] = (c, r0, r1, c0, c1, area, dist)
        else:
            old_c, old_r0, old_r1, old_c0, old_c1, old_area, old_dist = sides[side]
            # Keep the one closer to the center for this side.
            if dist < old_dist:
                sides[side] = (c, r0, r1, c0, c1, area, dist)

    def road_coord(side, c, r0, r1, c0, c1):
        if side == "L":
            x = c1 + 1
            x = max(0, min(x, mw - tw))
            return x, "V"
        if side == "R":
            x = c0 - tw
            x = max(0, min(x, mw - tw))
            return x, "V"
        if side == "T":
            y = r1 + 1
            y = max(0, min(y, mh - th))
            return y, "H"
        if side == "B":
            y = r0 - th
            y = max(0, min(y, mh - th))
            return y, "H"
        return 0, "V"

    def branch_dir(side, color):
        b = bits.get(color, "right")
        if side == "L":
            return "N" if b == "right" else "S"
        if side == "R":
            return "S" if b == "right" else "N"
        if side == "T":
            return "E" if b == "right" else "W"
        if side == "B":
            return "E" if b == "left" else "W"
        return "N"

    def can_paint(r, c):
        return 0 <= r < mh and 0 <= c < mw and main[r, c] in (bg, 8)

    def paint_v(x, y0, y1):
        x0 = max(0, x)
        x1 = min(mw, x + tw)
        y0 = max(0, y0)
        y1 = min(mh, y1)
        for r in range(y0, y1):
            for c in range(x0, x1):
                if can_paint(r, c):
                    main[r, c] = 8

    def paint_h(y, x0, x1):
        y0 = max(0, y)
        y1 = min(mh, y + th)
        x0 = max(0, x0)
        x1 = min(mw, x1)
        for r in range(y0, y1):
            for c in range(x0, x1):
                if can_paint(r, c):
                    main[r, c] = 8

    # 7. Build the central cross and side corridors.

    # Default central cross limits.
    h_left = 0
    h_right = mw
    v_top = 0
    v_bottom = mh

    roads = {}

    for side, data in sides.items():
        c, r0, r1, c0, c1, _, _ = data
        pos, kind = road_coord(side, c, r0, r1, c0, c1)
        d = branch_dir(side, c)
        roads[side] = (pos, kind, d)

        if side == "L":
            h_left = min(h_left, pos)
            # Vertical corridor on the right side of the left block.
            if d == "N":
                paint_v(pos, 0, sr0 + th)
            else:
                paint_v(pos, sr0, mh)

        elif side == "R":
            h_right = max(h_right, pos + tw)
            # Vertical corridor on the left side of the right block.
            if d == "N":
                paint_v(pos, 0, sr0 + th)
            else:
                paint_v(pos, sr0, mh)

        elif side == "T":
            v_top = min(v_top, pos)
            # Horizontal corridor below the top block.
            # Choose the vertical road inside the top block closest to center,
            # then extend in the requested direction to the edge.
            candidate_xs = []
            if "L" in roads:
                candidate_xs.append(roads["L"][0])
            candidate_xs.append(sc0)
            if "R" in roads:
                candidate_xs.append(roads["R"][0])

            # Keep only roads that touch the top block's horizontal span.
            valid = [x for x in candidate_xs if c0 <= x + tw - 1 and c1 >= x]
            if not valid:
                valid = [sc0]

            if d == "W":
                x_start = max(valid)
                paint_h(pos, 0, x_start + tw)
            else:
                x_start = min(valid)
                paint_h(pos, x_start, mw)

        elif side == "B":
            v_bottom = max(v_bottom, pos + th)
            # Horizontal corridor above the bottom block.
            candidate_xs = []
            if "L" in roads:
                candidate_xs.append(roads["L"][0])
            candidate_xs.append(sc0)
            if "R" in roads:
                candidate_xs.append(roads["R"][0])

            valid = [x for x in candidate_xs if c0 <= x + tw - 1 and c1 >= x]
            if not valid:
                valid = [sc0]

            if d == "W":
                x_start = max(valid)
                paint_h(pos, x_start, mw)
            else:
                x_start = min(valid)
                paint_h(pos, x_start, mw)

    # Central horizontal line through the 8 block.
    paint_h(sr0, h_left, h_right)

    # Central vertical line through the 8 block.
    paint_v(sc0, v_top, v_bottom)

    # 8. Return the transformed main map.
    return main