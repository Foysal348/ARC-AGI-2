import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the lower "scene" by drawing thick gray (8) rays from the gray
    anchor block.

    The top strip encodes, for each colored rectangle, a left/right bit.
    The lower grid contains:
      - a background color,
      - one gray 8 block,
      - colored rectangular blocks whose colors appear in the top strip.

    From the gray block, launch four thick rays: north, east, south, west.
    A ray travels straight until it hits a colored rectangle or the border.
    When it hits a rectangle, it turns:
      - left  if the top-strip marker for that color is on the left,
      - right if the top-strip marker for that color is on the right,
    relative to the direction the ray was traveling.

    The ray continues in the new direction until it hits another rectangle or
    the border.  The union of all ray cells is color 8.  The original colored
    rectangles are preserved.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1) Remove the fixed top control strip.
    #    It has the form:
    #      row 0: all 5
    #      rows 1-4: control cells
    #      row 5: all 5
    # ------------------------------------------------------------------
    def is_all_five(row):
        return np.all(row == 5)

    top_end = None
    for i in range(1, H):
        if is_all_five(grid[i]):
            top_end = i
            break

    if top_end is not None and top_end >= 5:
        main = grid[top_end + 1:].copy()
        top_rows = grid[1:top_end]
    else:
        # Fallback if the top strip is absent or malformed.
        main = grid.copy()
        top_rows = np.empty((0, W), dtype=grid.dtype)

    Hm, Wm = main.shape

    # ------------------------------------------------------------------
    # 2) Parse the top strip: color -> bit (0 = left, 1 = right).
    # ------------------------------------------------------------------
    color_bit = {}

    if top_rows.size:
        # Find the four horizontal control slots (runs of non-5 cells).
        row_sample = top_rows[0]
        slots = []
        in_slot = False
        start = 0
        for c in range(W):
            if row_sample[c] != 5:
                if not in_slot:
                    in_slot = True
                    start = c
            else:
                if in_slot:
                    slots.append((start, c - 1))
                    in_slot = False
        if in_slot:
            slots.append((start, W - 1))

        # There should normally be four slots.  If parsing failed, use the
        # known regular layout for these tasks.
        if len(slots) < 4:
            slots = [(1, 4), (7, 10), (13, 16), (19, 22)]

        for c0, c1 in slots:
            if c0 >= W or c1 >= W:
                continue
            block = top_rows[:, c0:c1 + 1]

            # Find the nonzero, non-5 color in this slot.
            mask = (block != 0) & (block != 5)
            if not np.any(mask):
                continue

            vals = block[mask]
            # If multiple values appear, take the most frequent.
            col, _ = Counter(vals.tolist()).most_common(1)[0]

            # Find the column offset where this color appears.
            col_positions = np.where(block == col)[1]
            if col_positions.size == 0:
                continue
            offset = int(np.mean(col_positions))
            width = c1 - c0 + 1
            bit = 0 if offset < width // 2 else 1
            color_bit[int(col)] = bit

    # ------------------------------------------------------------------
    # 3) Find the gray anchor block (color 8) and its thickness.
    # ------------------------------------------------------------------
    key_mask = (main == 8)
    if not np.any(key_mask):
        return main

    key_rows, key_cols = np.where(key_mask)
    kr0, kr1 = int(key_rows.min()), int(key_rows.max())
    kc0, kc1 = int(key_cols.min()), int(key_cols.max())
    t_r = kr1 - kr0 + 1
    t_c = kc1 - kc0 + 1

    # ------------------------------------------------------------------
    # 4) Find colored rectangles from the top-strip colors.
    # ------------------------------------------------------------------
    # Background is the most frequent color in the main scene.
    bg_color = Counter(main.flatten().tolist()).most_common(1)[0][0]

    blocks = []
    for color, bit in color_bit.items():
        if color == 8:
            continue

        mask = (main == color)
        if not np.any(mask):
            continue

        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        # Ignore huge "background-like" boxes if a top color accidentally
        # matches the background.
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        count = int(mask.sum())
        if color == bg_color and area > 0.5 * Hm * Wm and count > 0.5 * Hm * Wm:
            continue

        blocks.append({
            "r0": r0, "r1": r1,
            "c0": c0, "c1": c1,
            "bit": bit,
        })

    # ------------------------------------------------------------------
    # 5) Ray simulation.
    # ------------------------------------------------------------------
    out = main.copy()
    out[kr0:kr1 + 1, kc0:kc1 + 1] = 8

    def paint(r0, r1, c0, c1):
        r0 = max(0, r0)
        r1 = min(Hm - 1, r1)
        c0 = max(0, c0)
        c1 = min(Wm - 1, c1)
        if r0 > r1 or c0 > c1:
            return
        # Never overwrite existing colored rectangles.
        # Paint background and gray only.
        sub = out[r0:r1 + 1, c0:c1 + 1]
        safe = np.ones(sub.shape, dtype=bool)
        for b in blocks:
            safe &= (sub != b["color"] if "color" in b else np.ones(sub.shape, dtype=bool))
        # Blocks do not currently store color explicitly; add it below.
        sub[safe] = 8
        out[r0:r1 + 1, c0:c1 + 1] = sub

    # Store color in each block for safe painting.
    for color, bit in color_bit.items():
        for b in blocks:
            if (main == color).any():
                rr, cc = np.where(main == color)
                if b["r0"] == rr.min() and b["r1"] == rr.max() \
                   and b["c0"] == cc.min() and b["c1"] == cc.max():
                    b["color"] = int(color)
                    break

    # Fix paint() to use b["color"] if present.
    def paint_safe(r0, r1, c0, c1):
        r0 = max(0, r0)
        r1 = min(Hm - 1, r1)
        c0 = max(0, c0)
        c1 = min(Wm - 1, c1)
        if r0 > r1 or c0 > c1:
            return
        sub = out[r0:r1 + 1, c0:c1 + 1].copy()
        safe = np.ones(sub.shape, dtype=bool)
        for b in blocks:
            if "color" in b:
                safe &= (sub != b["color"])
        sub[safe] = 8
        out[r0:r1 + 1, c0:c1 + 1] = sub

    # Directions: 0=N, 1=E, 2=S, 3=W
    def turn_dir(d, bit):
        # bit 0 => left, bit 1 => right
        if bit == 1:
            return (d + 1) % 4
        return (d - 1) % 4

    visited = set()

    def hit_block_W(rb0, rb1, front_col):
        best = None
        for b in blocks:
            if b["r0"] <= rb1 and b["r1"] >= rb0:
                if b["c1"] < front_col:
                    if best is None or b["c1"] > best["c1"]:
                        best = b
        return best

    def hit_block_E(rb0, rb1, front_col):
        best = None
        for b in blocks:
            if b["r0"] <= rb1 and b["r1"] >= rb0:
                if b["c0"] > front_col:
                    if best is None or b["c0"] < best["c0"]:
                        best = b
        return best

    def hit_block_N(cb0, cb1, front_row):
        best = None
        for b in blocks:
            if b["c0"] <= cb1 and b["c1"] >= cb0:
                if b["r1"] < front_row:
                    if best is None or b["r1"] < best["r1"]:
                        best = b
        return best

    def hit_block_S(cb0, cb1, front_row):
        best = None
        for b in blocks:
            if b["c0"] <= cb1 and b["c1"] >= cb0:
                if b["r0"] > front_row:
                    if best is None or b["r0"] < best["r0"]:
                        best = b
        return best

    def emit_W(rb0, rb1, front_col, depth=0):
        if depth > 100:
            return
        state = ("W", rb0, rb1, front_col)
        if state in visited:
            return
        visited.add(state)

        b = hit_block_W(rb0, rb1, front_col)
        if b is None:
            stop_col = 0
        else:
            stop_col = b["c1"] + 1

        if stop_col <= front_col:
            paint_safe(rb0, rb1, stop_col, front_col)

        if b is not None:
            td = turn_dir(3, b["bit"])  # facing West
            # New vertical band immediately right of the hit block.
            nc0 = b["c1"] + 1
            nc1 = nc0 + t_c - 1
            if nc0 < 0 or nc1 >= Wm:
                return
            if td == 0:  # North
                emit_N(rb0, nc0, nc1, depth + 1)
            elif td == 2:  # South
                emit_S(rb1, nc0, nc1, depth + 1)

    def emit_E(rb0, rb1, front_col, depth=0):
        if depth > 100:
            return
        state = ("E", rb0, rb1, front_col)
        if state in visited:
            return
        visited.add(state)

        b = hit_block_E(rb0, rb1, front_col)
        if b is None:
            stop_col = Wm - 1
        else:
            stop_col = b["c0"] - 1

        if stop_col >= front_col:
            paint_safe(rb0, rb1, front_col, stop_col)

        if b is not None:
            td = turn_dir(1, b["bit"])  # facing East
            # New vertical band immediately left of the hit block.
            nc1 = b["c0"] - 1
            nc0 = nc1 - t_c + 1
            if nc1 < 0 or nc0 >= Wm:
                return
            if td == 0:  # North
                emit_N(rb0, nc0, nc1, depth + 1)
            elif td == 2:  # South
                emit_S(rb1, nc0, nc1, depth + 1)

    def emit_N(cb0, cb1, front_row, depth=0):
        if depth > 100:
            return
        state = ("N", cb0, cb1, front_row)
        if state in visited:
            return
        visited.add(state)

        b = hit_block_N(cb0, cb1, front_row)
        if b is None:
            stop_row = 0
        else:
            stop_row = b["r1"] + 1

        if stop_row <= front_row:
            paint_safe(stop_row, front_row, cb0, cb1)

        if b is not None:
            td = turn_dir(0, b["bit"])  # facing North
            # New horizontal band immediately below the hit block.
            nr0 = b["r1"] + 1
            nr1 = nr0 + t_r - 1
            if nr0 < 0 or nr1 >= Hm:
                return
            if td == 3:  # West
                emit_W(nr0, nr1, cb0 - 1, depth + 1)
            elif td == 1:  # East
                emit_E(nr0, nr1, cb1 + 1, depth + 1)

    def emit_S(cb0, cb1, front_row, depth=0):
        if depth > 100:
            return
        state = ("S", cb0, cb1, front_row)
        if state in visited:
            return
        visited.add(state)

        b = hit_block_S(cb0, cb1, front_row)
        if b is None:
            stop_row = Hm - 1
        else:
            stop_row = b["r0"] - 1

        if stop_row >= front_row:
            paint_safe(front_row, stop_row, cb0, cb1)

        if b is not None:
            td = turn_dir(2, b["bit"])  # facing South
            # New horizontal band immediately above the hit block.
            nr1 = b["r0"] - 1
            nr0 = nr1 - t_r + 1
            if nr1 < 0 or nr0 >= Hm:
                return
            if td == 3:  # West
                emit_W(nr0, nr1, cb0 - 1, depth + 1)
            elif td == 1:  # East
                emit_E(nr0, nr1, cb1 + 1, depth + 1)

    # Launch the four initial rays from the gray anchor.
    if kc0 > 0:
        emit_W(kr0, kr1, kc0 - 1)
    if kc1 + 1 < Wm:
        emit_E(kr0, kr1, kc1 + 1)
    if kr0 > 0:
        emit_N(kc0, kc1, kr0 - 1)
    if kr1 + 1 < Hm:
        emit_S(kc0, kc1, kr1 + 1)

    return out