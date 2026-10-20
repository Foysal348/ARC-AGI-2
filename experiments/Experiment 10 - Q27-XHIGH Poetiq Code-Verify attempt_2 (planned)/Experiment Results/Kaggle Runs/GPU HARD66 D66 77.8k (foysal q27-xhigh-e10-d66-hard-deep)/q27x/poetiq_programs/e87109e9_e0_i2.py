import numpy as np
from collections import Counter, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Remove the top 5-frame legend and draw 8-colored roads in the main map.

    The transformation works as follows:
    1. Cut off the top legend frame.
    2. Find the central 8 block; its size gives the road thickness.
    3. Find the background color and the side blocks (top/bottom/left/right).
    4. Read the legend bits:
       - color in left part of its legend cell  => "left"
       - color in right part of its legend cell => "right"
    5. For each side block, convert its bit to a turning direction
       relative to the outward direction from the center:
         left bit  => turn left
         right bit => turn right
    6. Draw:
       - a core horizontal road near the center,
       - a core vertical road through the center,
       - side roads attached to the side blocks, extending in the
         appropriate turn direction and clipped by colored obstacles.

    The implementation is intentionally conservative: roads are painted
    only on background or existing 8 cells, so they never overwrite side
    blocks except where the examples show an intended touch/overlap.
    """

    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Remove the top legend frame.
    #    In all tasks it is:
    #      all-5 border row
    #      4 legend rows containing 5 separators
    #      all-5 border row
    # ------------------------------------------------------------------
    all5_rows = []
    for r in range(H):
        if np.all(grid[r] == 5):
            all5_rows.append(r)

    cut = 0
    if len(all5_rows) >= 2:
        cut = all5_rows[1] + 1
    elif len(all5_rows) == 1:
        cut = all5_rows[0] + 1

    if cut >= H:
        cut = 0

    main = grid[cut:].copy()
    mh, mw = main.shape
    if mh == 0 or mw == 0:
        return main

    # ------------------------------------------------------------------
    # 2. Legend bits.
    #    The legend rows are between the two all-5 border rows.
    #    Each legend cell has width 4, separated by two 5-columns.
    #    A color in the leftmost cell of its group means "left",
    #    in the rightmost cell means "right".
    # ------------------------------------------------------------------
    legend_bits = {}

    if len(all5_rows) >= 2:
        for r in range(all5_rows[0] + 1, all5_rows[1]):
            row = grid[r]
            if row is None:
                continue

            i = 0
            while i < W:
                if row[i] == 5:
                    i += 1
                    continue

                j = i
                while j < W and row[j] != 5:
                    j += 1

                a, b = i, j
                if b - a >= 4:
                    # The non-background, non-5 color in this cell.
                    for c in range(a, b):
                        v = int(row[c])
                        if v not in (0, 5):
                            # left half vs right half is enough for these tasks
                            if c - a < (b - a) // 2:
                                legend_bits[v] = "L"
                            else:
                                legend_bits[v] = "R"
                            break

                i = j

    # ------------------------------------------------------------------
    # 3. Background color in the main map.
    # ------------------------------------------------------------------
    counts = Counter(main.flatten().tolist())
    # 8 is the road color; background is normally the most common color.
    bg = counts.most_common(1)[0][0] if counts else 0

    # ------------------------------------------------------------------
    # 4. Connected components / bounding boxes.
    # ------------------------------------------------------------------
    def components(mask):
        """4-connected components; returns bounding boxes (r0, r1, c0, c1)."""
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
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

    # Central 8 block.
    c8_boxes = components(main == 8)
    if not c8_boxes:
        return main

    # Choose the largest 8 component as the central block.
    sr0, sr1, sc0, sc1 = max(
        c8_boxes,
        key=lambda b: (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
    )

    # Road thickness is the central block thickness.
    th = sr1 - sr0 + 1
    tw = sc1 - sc0 + 1

    # Use a square thickness; in tasks these are equal.
    t = max(1, min(th, tw))

    # Core vertical road is aligned with the left column of the central block.
    x_core = sc0
    x_core = max(0, min(x_core, mw - t))

    # ------------------------------------------------------------------
    # 5. Side blocks.
    # ------------------------------------------------------------------
    side_boxes = {}

    nonbg = (main != bg) & (main != 8)
    colors = np.unique(main)

    for color in colors:
        if color == bg or color == 8:
            continue

        for (r0, r1, c0, c1) in components(main == color):
            cr = (r0 + r1) / 2.0
            cc = (c0 + c1) / 2.0
            ccr = (sr0 + sr1) / 2.0
            ccc = (sc0 + sc1) / 2.0

            # Distances from this component to each side of the central block.
            dL = max(0, sc0 - c1)
            dR = max(0, c0 - sc1)
            dT = max(0, sr0 - r1)
            dB = max(0, r0 - sr1)

            # Pick the most dominant side.
            best = max(
                (("L", dL), ("R", dR), ("T", dT), ("B", dB)),
                key=lambda x: x[1]
            )

            side, dist = best
            if dist <= 0:
                continue

            # If several components claim a side, keep the closest/largest one.
            if side not in side_boxes:
                side_boxes[side] = (int(color), r0, r1, c0, c1)
            else:
                old_c, or0, or1, oc0, oc1 = side_boxes[side]
                old_dist = max(0, (or0 + or1) / 2.0 - ccr,
                               (cc + oc0 + oc1) / 2.0 - ccc)
                # Prefer closer to center; on tie, larger area.
                area_new = (r1 - r0 + 1) * (c1 - c0 + 1)
                area_old = (or1 - or0 + 1) * (oc1 - oc0 + 1)

                if dist < old_dist or (dist == old_dist and area_new > area_old):
                    side_boxes[side] = (int(color), r0, r1, c0, c1)

    # ------------------------------------------------------------------
    # 6. Determine the horizontal core road y.
    #
    #    In the examples, the horizontal core is usually the top row of the
    #    central 8 block.  When both left/right arms turn the same vertical
    #    way, it shifts toward that common direction, snapped to the side
    #    block center closest to the central block.
    # ------------------------------------------------------------------
    def turn_for_side(side, bit):
        if side == "L":
            return "S" if bit == "L" else "N"
        if side == "R":
            return "N" if bit == "L" else "S"
        if side == "T":
            return "W" if bit == "L" else "E"
        if side == "B":
            return "E" if bit == "L" else "W"
        return "N"

    def bit_for(color):
        return legend_bits.get(color, "L")

    arms = {}
    for side, data in side_boxes.items():
        color, r0, r1, c0, c1 = data
        bit = bit_for(color)
        turn = turn_for_side(side, bit)
        arms[side] = {
            "color": color,
            "box": (r0, r1, c0, c1),
            "bit": bit,
            "turn": turn,
        }

    y_core = sr0

    if "L" in arms and "R" in arms:
        tL = arms["L"]["turn"]
        tR = arms["R"]["turn"]

        if tL == tR:
            # Both vertical arms turn the same way.
            cands = []
            for side in ("L", "R"):
                r0, r1, c0, c1 = arms[side]["box"]
                cy = (r0 + r1) / 2.0
                ccy = (sr0 + sr1) / 2.0
                cands.append((abs(cy - ccy), int(np.floor(cy))))

            if cands:
                cands.sort()
                y_candidate = cands[0][1]

                # Clamp so the horizontal strip stays adjacent/connected
                # to the central block.
                y_min = max(0, sr0 - t + 1)
                y_max = min(mh - t, sr1 + 1)
                y_candidate = max(y_min, min(y_max, y_candidate))
                y_core = y_candidate

    y_core = max(0, min(y_core, mh - t))

    # Vertical core x is fixed to the central block left edge.
    x_core = max(0, min(x_core, mw - t))

    # ------------------------------------------------------------------
    # 7. Paint helpers.
    # ------------------------------------------------------------------
    def paint_v(x, y0, y1):
        x0 = max(0, x)
        x1 = min(mw, x + t)
        y0 = max(0, y0)
        y1 = min(mh, y1)

        for r in range(y0, y1):
            for c in range(x0, x1):
                if main[r, c] in (bg, 8):
                    main[r, c] = 8

    def paint_h(y, x0, x1):
        y0 = max(0, y)
        y1 = min(mh, y + t)
        x0 = max(0, x0)
        x1 = min(mw, x1)

        for r in range(y0, y1):
            for c in range(x0, x1):
                if main[r, c] in (bg, 8):
                    main[r, c] = 8

    # ------------------------------------------------------------------
    # 8. Side road coordinates.
    # ------------------------------------------------------------------
    line = {}

    if "L" in arms:
        r0, r1, c0, c1 = arms["L"]["box"]
        x = c1 + 1
        x = max(0, min(x, mw - t))
        line["L"] = x

    if "R" in arms:
        r0, r1, c0, c1 = arms["R"]["box"]
        x = c0 - t
        x = max(0, min(x, mw - t))
        line["R"] = x

    if "T" in arms:
        r0, r1, c0, c1 = arms["T"]["box"]
        y = r1 + 1
        y = max(0, min(y, mh - t))
        line["T"] = y

    if "B" in arms:
        r0, r1, c0, c1 = arms["B"]["box"]
        y = r0 - t
        y = max(0, min(y, mh - t))
        line["B"] = y

    # ------------------------------------------------------------------
    # 9. Draw core roads.
    #    They extend until blocked by non-background/non-8 pixels.
    # ------------------------------------------------------------------
    paint_v(x_core, 0, mh)
    paint_h(y_core, 0, mw)

    # ------------------------------------------------------------------
    # 10. Draw vertical side roads (left/right).
    # ------------------------------------------------------------------
    for side in ("L", "R"):
        if side not in arms:
            continue

        x = line[side]
        turn = arms[side]["turn"]

        if turn == "N":
            # From top edge down to the core horizontal road / central block.
            # Stop just below the core horizontal strip, but not before the
            # central block bottom if that is closer.
            stop = max(y_core + t, sr1 + 1)
            paint_v(x, 0, stop)
        else:  # S
            # From core horizontal road / central top down to bottom edge.
            start = min(y_core, sr0)
            paint_v(x, start, mh)

    # ------------------------------------------------------------------
    # 11. Draw horizontal side roads (top/bottom).
    #
    #     A horizontal side road uses the leftmost valid vertical line
    #     among:
    #       - left side road,
    #       - core vertical,
    #       - right side road
    #     that overlaps the side block horizontally.
    #
    #     If the turn is E, paint from that line to the right edge.
    #     If the turn is W, paint from the left edge to that line.
    # ------------------------------------------------------------------
    def vertical_lines_that_overlap(block_c0, block_c1):
        xs = []

        if "L" in line:
            x = line["L"]
            if x + t - 1 >= block_c0 and x <= block_c1:
                xs.append(x)

        # Core vertical is always a valid connector if it overlaps.
        if x_core + t - 1 >= block_c0 and x_core <= block_c1:
            xs.append(x_core)

        if "R" in line:
            x = line["R"]
            if x + t - 1 >= block_c0 and x <= block_c1:
                xs.append(x)

        if not xs:
            xs.append(x_core)

        return min(xs)

    for side in ("T", "B"):
        if side not in arms:
            continue

        y = line[side]
        r0, r1, c0, c1 = arms[side]["box"]
        turn = arms[side]["turn"]

        x_start = vertical_lines_that_overlap(c0, c1)

        if turn == "E":
            paint_h(y, x_start, mw)
        else:  # W
            paint_h(y, 0, x_start + t)

    return main