import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input by:
    1. Removing the top 5-colored legend frame.
    2. Finding the central 8-block; its thickness is the road thickness.
    3. Finding the four side blocks (left/right/top/bottom) in the main map.
    4. Reading left/right bits from the legend for each block color.
    5. Drawing 8-colored road strips:
       - a central vertical strip through the central 8-block,
       - a central horizontal strip through the central 8-block,
       - side strips adjacent to each side block, with endpoints chosen
         from the legend bit and from the other active strips.

    The rules below are inferred from the examples and match all of them.
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def connected_components(mask):
        """4-connected components; returns bounding boxes (r0, r1, c0, c1)."""
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        boxes = []

        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue

                q = deque([(r, c)])
                seen[r, c] = True
                r0 = r1 = r
                c0 = c1 = c

                while q:
                    cr, cc = q.popleft()
                    r0 = min(r0, cr)
                    r1 = max(r1, cr)
                    c0 = min(c0, cc)
                    c1 = max(c1, cc)

                    for nr, nc in ((cr + 1, cc), (cr - 1, cc),
                                   (cr, cc + 1), (cr, cc - 1)):
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))

                boxes.append((r0, r1, c0, c1))

        return boxes

    def parse_legend_bits(grid_2d, all5_rows):
        """
        Legend rows lie between the first and second all-5 rows.
        Each 4-wide cell contains one color. If the color is in the left
        half, bit='L'; if in the right half, bit='R'.
        """
        bits = {}
        if len(all5_rows) < 2:
            return bits

        for r in range(all5_rows[0] + 1, all5_rows[1]):
            row = grid_2d[r]
            i = 0
            while i < W:
                if row[i] == 5:
                    i += 1
                    continue

                j = i
                while j < W and row[j] != 5:
                    j += 1

                a, b = i, j
                for c in range(a, b):
                    v = int(row[c])
                    if v not in (0, 5):
                        if c - a < (b - a) // 2:
                            bits[v] = 'L'
                        else:
                            bits[v] = 'R'
                        break

                i = j

        return bits

    def assign_side(r0, r1, c0, c1, h, w):
        """Assign a block to T/B/L/R by closest edge, with tie-breaking."""
        d = {
            'T': r0,
            'B': h - 1 - r1,
            'L': c0,
            'R': w - 1 - c1,
        }
        m = min(d.values())
        cand = [k for k, v in d.items() if v == m]

        if len(cand) == 1:
            return cand[0]

        height = r1 - r0 + 1
        width = c1 - c0 + 1

        # If the block is horizontally long, prefer top/bottom.
        if height < width:
            for s in ('T', 'B'):
                if s in cand:
                    return s
        else:
            for s in ('L', 'R'):
                if s in cand:
                    return s

        return cand[0]

    def clamp_row(y):
        return max(0, min(int(y), mh - t))

    def clamp_col(x):
        return max(0, min(int(x), mw - t))

    # ------------------------------------------------------------------
    # 1. Remove the top legend frame.
    # ------------------------------------------------------------------
    all5_rows = [r for r in range(H) if np.all(grid[r] == 5)]
    if len(all5_rows) >= 2:
        cut = all5_rows[1] + 1
    elif len(all5_rows) == 1:
        cut = all5_rows[0] + 1
    else:
        cut = 0

    if cut >= H:
        return grid

    main = grid[cut:].copy()
    mh, mw = main.shape
    if mh == 0 or mw == 0:
        return main

    bits = parse_legend_bits(grid, all5_rows)

    # ------------------------------------------------------------------
    # 2. Background and central 8 block.
    # ------------------------------------------------------------------
    vals, counts = np.unique(main, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    eight_boxes = connected_components(main == 8)
    if not eight_boxes:
        return main

    cr0, cr1, cc0, cc1 = max(
        eight_boxes,
        key=lambda b: (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
    )

    t = min(cr1 - cr0 + 1, cc1 - cc0 + 1)
    if t <= 0:
        return main

    cv_x = clamp_col(cc0)          # central vertical road left column
    ch_y = clamp_row(cr0)          # central horizontal road top row

    # ------------------------------------------------------------------
    # 3. Side blocks.
    # ------------------------------------------------------------------
    side_boxes = {}

    for color in np.unique(main):
        if color == bg or color == 8:
            continue

        for r0, r1, c0, c1 in connected_components(main == color):
            side = assign_side(r0, r1, c0, c1, mh, mw)
            if side is None:
                continue

            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            # Store: (color, r0, r1, c0, c1, area)
            if side not in side_boxes or area > side_boxes[side][5]:
                side_boxes[side] = (int(color), r0, r1, c0, c1, area)

    L = side_boxes.get('L')
    R = side_boxes.get('R')
    T = side_boxes.get('T')
    B = side_boxes.get('B')

    LV_x = clamp_col(L[4] + 1) if L is not None else None
    RV_x = clamp_col(R[3] - t) if R is not None else None
    TH_y = clamp_row(T[2] + 1) if T is not None else None
    BH_y = clamp_row(B[1] - t) if B is not None else None

    # ------------------------------------------------------------------
    # 4. Central strips.
    # ------------------------------------------------------------------
    # Central horizontal strip: stop at left/right blocks if they align.
    CH_x0 = 0
    CH_x1 = mw - 1

    if L is not None and L[1] <= ch_y and ch_y + t - 1 <= L[2]:
        CH_x0 = LV_x
    if R is not None and R[1] <= ch_y and ch_y + t - 1 <= R[2]:
        CH_x1 = RV_x + t - 1

    # Central vertical strip: stop at top/bottom blocks if they align.
    CV_y0 = 0
    CV_y1 = mh - 1

    if T is not None and T[3] <= cv_x and cv_x + t - 1 <= T[4]:
        CV_y0 = TH_y
    if B is not None and B[3] <= cv_x and cv_x + t - 1 <= B[4]:
        CV_y1 = BH_y + t - 1

    # ------------------------------------------------------------------
    # 5. Top and bottom side strips.
    # ------------------------------------------------------------------
    # For top strips, anchor to the rightmost valid vertical line.
    # For bottom strips, anchor to the leftmost valid vertical line.
    # Valid means the whole strip width lies inside the block's horizontal span.
    TH_int = None
    if T is not None:
        candidates = []
        if L is not None:
            candidates.append(LV_x)
        candidates.append(cv_x)
        if R is not None:
            candidates.append(RV_x)

        valid = []
        partial = []

        for x in candidates:
            if x is None:
                continue
            if x >= T[3] and x + t - 1 <= T[4]:
                valid.append(x)
            elif x < T[4] and x + t - 1 > T[3]:
                partial.append(x)

        if valid:
            anchor_x = max(set(valid))
        elif partial:
            anchor_x = max(set(partial))
        else:
            anchor_x = cv_x

        bit = bits.get(T[0], 'L')
        if bit == 'L':       # west
            x0 = 0
            x1 = anchor_x + t - 1
        else:                # east
            x0 = anchor_x
            x1 = mw - 1

        TH_int = (TH_y, x0, x1)

    BH_int = None
    if B is not None:
        candidates = []
        if L is not None:
            candidates.append(LV_x)
        candidates.append(cv_x)
        if R is not None:
            candidates.append(RV_x)

        valid = []
        partial = []

        for x in candidates:
            if x is None:
                continue
            if x >= B[3] and x + t - 1 <= B[4]:
                valid.append(x)
            elif x < B[4] and x + t - 1 > B[3]:
                partial.append(x)

        if valid:
            anchor_x = min(set(valid))
        elif partial:
            anchor_x = min(set(partial))
        else:
            anchor_x = cv_x

        bit = bits.get(B[0], 'L')
        if bit == 'L':       # east
            x0 = anchor_x
            x1 = mw - 1
        else:                # west
            x0 = 0
            x1 = anchor_x + t - 1

        BH_int = (BH_y, x0, x1)

    # ------------------------------------------------------------------
    # 6. Left and right side strips.
    # ------------------------------------------------------------------
    def horizontal_crosses(x, h_int):
        """True if horizontal strip h_int fully covers vertical strip x..x+t-1."""
        return h_int is not None and h_int[1] <= x and h_int[2] >= x + t - 1

    LV_int = None
    if L is not None:
        x = LV_x

        crosses = []
        if horizontal_crosses(x, TH_int):
            crosses.append((TH_y, TH_y + t - 1))
        if horizontal_crosses(x, (ch_y, CH_x0, CH_x1)):
            crosses.append((ch_y, ch_y + t - 1))
        if horizontal_crosses(x, BH_int):
            crosses.append((BH_y, BH_y + t - 1))

        if not crosses:
            # Fallback: connect to central horizontal if possible.
            if CH_x0 <= x and CH_x1 >= x + t - 1:
                crosses = [(ch_y, ch_y + t - 1)]

        bit = bits.get(L[0], 'L')

        if bit == 'R':       # direction north
            y0 = TH_y if (T is not None and T[3] <= x + t - 1 and T[4] >= x) else 0
            if crosses:
                y1 = max(end for _, end in crosses)
            else:
                y1 = ch_y + t - 1
        else:                # direction south
            if crosses:
                y0 = min(start for start, _ in crosses)
            else:
                y0 = ch_y

            if horizontal_crosses(x, BH_int) or (
                B is not None and B[3] <= x + t - 1 and B[4] >= x
            ):
                y1 = BH_y + t - 1
            else:
                y1 = mh - 1

        LV_int = (x, y0, y1)

    RV_int = None
    if R is not None:
        x = RV_x

        crosses = []
        if horizontal_crosses(x, TH_int):
            crosses.append((TH_y, TH_y + t - 1))
        if horizontal_crosses(x, (ch_y, CH_x0, CH_x1)):
            crosses.append((ch_y, ch_y + t - 1))
        if horizontal_crosses(x, BH_int):
            crosses.append((BH_y, BH_y + t - 1))

        if not crosses:
            if CH_x0 <= x and CH_x1 >= x + t - 1:
                crosses = [(ch_y, ch_y + t - 1)]

        bit = bits.get(R[0], 'L')

        # Right roads always start from the top edge/top block boundary.
        y0 = TH_y if (T is not None and T[3] <= x + t - 1 and T[4] >= x) else 0

        if bit == 'L':       # direction north: stop at central horizontal.
            if CH_x0 <= x and CH_x1 >= x + t - 1:
                y1 = ch_y + t - 1
            elif crosses:
                y1 = min(end for _, end in crosses)
            else:
                y1 = ch_y + t - 1
        else:                # direction south: continue to southernmost crossing.
            if crosses:
                y1 = max(end for _, end in crosses)
            elif B is not None and B[3] <= x + t - 1 and B[4] >= x:
                y1 = BH_y + t - 1
            else:
                y1 = mh - 1

        RV_int = (x, y0, y1)

    # ------------------------------------------------------------------
    # 7. Paint roads only on background or existing 8 cells.
    # ------------------------------------------------------------------
    out = main.copy()

    def paint_v(x, y0, y1):
        y0 = max(0, int(y0))
        y1 = min(mh - 1, int(y1))
        if y0 > y1:
            y0, y1 = y1, y0

        c0 = max(0, int(x))
        c1 = min(mw - 1, int(x) + t - 1)

        for r in range(y0, y1 + 1):
            for c in range(c0, c1 + 1):
                if out[r, c] == bg or out[r, c] == 8:
                    out[r, c] = 8

    def paint_h(y, x0, x1):
        x0 = max(0, int(x0))
        x1 = min(mw - 1, int(x1))
        if x0 > x1:
            x0, x1 = x1, x0

        r0 = max(0, int(y))
        r1 = min(mh - 1, int(y) + t - 1)

        for r in range(r0, r1 + 1):
            for c in range(x0, x1 + 1):
                if out[r, c] == bg or out[r, c] == 8:
                    out[r, c] = 8

    # Paint central strips.
    paint_v(cv_x, CV_y0, CV_y1)
    paint_h(ch_y, CH_x0, CH_x1)

    # Paint side strips.
    if TH_int is not None:
        paint_h(TH_int[0], TH_int[1], TH_int[2])

    if BH_int is not None:
        paint_h(BH_int[0], BH_int[1], BH_int[2])

    if LV_int is not None:
        paint_v(LV_int[0], LV_int[1], LV_int[2])

    if RV_int is not None:
        paint_v(RV_int[0], RV_int[1], RV_int[2])

    return out