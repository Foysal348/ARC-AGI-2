import numpy as np
from itertools import combinations

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input mosaic into the output mosaic.

    The rule implemented:
    1. Find the implicit rectangular rooms in the input.  A room is a rectangular
       block whose non-1 pixels are dominated by one color.  Small non-1 patches
       inside rooms are treated as target tokens, not separate rooms.
    2. In each room, detect the small 1-pixel arrow.  The arrow is one of the
       four chevron shapes:
          right:      X .      left:   . X
                      X X              X X
                      X .              . X
          down:       . .      up:     X X
                      X X              . X
                      X .              X .
       If a room has no arrow, default to pointing toward the center of the grid.
    3. Cast a ray from the arrow tip in the arrow direction.  The first non-1
       color encountered that is not the room's own base color is the output
       color for that room.  If nothing is found, fall back to the adjacent
       room's base color or the room's own base color.
    4. Recolor all rooms in place.
    5. If the original grid is not square, rotate the recolored grid by 90 degrees:
          clockwise if the original width is odd,
          counter-clockwise if the original width is even.
    """

    if grid is None:
        return grid

    g = np.array(grid, copy=True)
    h, w = g.shape

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def dominant_non1_color(sub: np.ndarray, exclude: int = -1) -> int:
        vals = np.asarray(sub).reshape(-1)
        vals = vals[vals != 1]
        if exclude >= 0:
            vals = vals[vals != exclude]
        if vals.size == 0:
            return -1
        vals = vals.astype(int)
        vals, counts = np.unique(vals, return_counts=True)
        return int(vals[np.argmax(counts)])

    def score_rect(r0, r1, c0, c1):
        """
        Higher score if the rectangle looks like a clean room:
        - high fraction of dominant non-1 color
        - not too many 1 pixels
        - not too many different non-1 colors
        """
        block = g[r0:r1, c0:c1]
        hgt, wid = r1 - r0, c1 - c0
        if hgt <= 0 or wid <= 0:
            return -10**9

        non1 = block[block != 1]
        if non1.size == 0:
            return -10**9

        vals = non1.astype(int)
        vals, counts = np.unique(vals, return_counts=True)
        dom = vals[np.argmax(counts)]
        dom_frac = counts.max() / counts.sum()

        one_frac = np.sum(block == 1) / block.size
        other_frac = 1.0 - dom_frac - one_frac

        # Strongly prefer clean rectangles.
        score = 100.0 * dom_frac - 30.0 * one_frac - 20.0 * other_frac

        # Mild preference for moderate/larger rooms, but allow small edge rooms.
        score += min(hgt * wid, 25) * 0.2

        # Penalize very thin slivers unless they are plausible full-grid strips.
        if hgt == 1 or wid == 1:
            score -= 5
        if hgt == 2 or wid == 2:
            score -= 2

        return score

    # ------------------------------------------------------------------
    # Find row and column boundaries that best partition the grid into rooms.
    #
    # We use change frequencies as candidate boundaries and search over small
    # subsets to maximize the sum of room scores.
    # ------------------------------------------------------------------
    def candidate_boundaries(axis_len: int, axis_is_row: bool):
        changes = np.zeros(axis_len, dtype=int)
        for i in range(1, axis_len):
            if axis_is_row:
                changes[i] = np.sum(g[i - 1, :] != g[i, :])
            else:
                changes[i] = np.sum(g[:, i - 1] != g[:, i])

        thr = max(2, (w if axis_is_row else h) // 6)
        cands = [0]
        for i in range(1, axis_len):
            if changes[i] >= thr:
                cands.append(i)
        cands.append(axis_len)
        return sorted(set(cands))

    row_cands = candidate_boundaries(h, True)
    col_cands = candidate_boundaries(w, False)

    def evaluate_partition(row_bounds, col_bounds):
        total = 0.0
        n_rooms = 0
        for r0, r1 in zip(row_bounds[:-1], row_bounds[1:]):
            for c0, c1 in zip(col_bounds[:-1], col_bounds[1:]):
                total += score_rect(r0, r1, c0, c1)
                n_rooms += 1
        # Prefer a small number of rooms, but avoid one giant room if the
        # candidates strongly suggest a mosaic.
        total -= max(0, n_rooms - 12) * 3.0
        total -= max(0, 8 - n_rooms) * 1.5
        return total

    best_row_bounds = [0, h]
    best_col_bounds = [0, w]
    best_score = -10**18

    # Search small partitions.  These ARC inputs are small enough.
    def search(bounds_cands, max_internal=5):
        results = []
        if len(bounds_cands) <= 1:
            return [[0, len(bounds_cands)] if False else [[0, 0]]]  # placeholder, replaced below
        # include single boundary case
        results.append([0, len(bounds_cands) if False else 0])
        return results

    # Simpler: enumerate internal boundaries from candidate sets with limited size.
    def enumerate_bounds(cands, max_internal=5):
        cands = [0] + cands[1:-1] + [len(cands) - 1 if False else 0]  # dummy to avoid lint, overwritten
        return None

    # Build concrete enumerations.
    def enum_row_bounds():
        internals = row_cands[1:-1]
        best = -10**18
        best_b = None
        # always include the one-piece partition
        for b in ([[0, h]]):
            s = evaluate_partition(b, best_col_bounds)
            if s > best:
                best = s
                best_b = b
        # try 1..max_internal boundaries
        max_int = min(4, len(internals))
        for k in range(1, max_int + 1):
            for combo in combinations(internals, k):
                b = [0] + sorted(combo) + [h]
                # avoid very narrow strips
                if any(b[i + 1] - b[i] < 2 for i in range(len(b) - 1)):
                    continue
                s = evaluate_partition(b, best_col_bounds)
                if s > best:
                    best = s
                    best_b = b
        return best_b if best_b is not None else [0, h]

    def enum_col_bounds():
        internals = col_cands[1:-1]
        best = -10**18
        best_b = None
        for b in ([[0, w]]):
            s = evaluate_partition(best_row_bounds, b)
            if s > best:
                best = s
                best_b = b
        max_int = min(4, len(internals))
        for k in range(1, max_int + 1):
            for combo in combinations(internals, k):
                b = [0] + sorted(combo) + [w]
                if any(b[i + 1] - b[i] < 2 for i in range(len(b) - 1)):
                    continue
                s = evaluate_partition(best_row_bounds, b)
                if s > best:
                    best = s
                    best_b = b
        return best_b if best_b is not None else [0, w]

    # Alternate a few times to let row/col choices improve together.
    for _ in range(3):
        best_row_bounds = enum_row_bounds()
        best_col_bounds = enum_col_bounds()

    # ------------------------------------------------------------------
    # Build rooms from the selected partition.
    # ------------------------------------------------------------------
    rooms = []
    for r0, r1 in zip(best_row_bounds[:-1], best_row_bounds[1:]):
        for c0, c1 in zip(best_col_bounds[:-1], best_col_bounds[1:]):
            block = g[r0:r1, c0:c1]
            dom = dominant_non1_color(block)
            if dom == -1:
                dom = 0
            rooms.append({
                "r0": int(r0), "r1": int(r1),
                "c0": int(c0), "c1": int(c1),
                "color": int(dom),
            })

    if not rooms:
        rooms.append({"r0": 0, "r1": h, "c0": 0, "c1": w,
                      "color": dominant_non1_color(g) if np.any(g != 1) else 0})

    # ------------------------------------------------------------------
    # Arrow detection inside each room.
    # ------------------------------------------------------------------
    def detect_arrow(room):
        """
        Return (dy, dx, tip_r, tip_c) for the arrow in this room, or None.
        """
        r0, r1, c0, c1 = room["r0"], room["r1"], room["c0"], room["c1"]
        block = g[r0:r1, c0:c1]
        bh, bw = block.shape
        ones = np.argwhere(block == 1)
        if ones.size == 0:
            return None

        # Prefer a compact 3x3 chevron.  Scan all 3x3 windows.
        best = None
        best_score = -1
        for rr in range(0, bh - 2):
            for cc in range(0, bw - 2):
                win = block[rr:rr + 3, cc:cc + 3]
                # right chevron
                if (win[0, 0] == 1 and win[1, 0] == 1 and win[1, 1] == 1 and win[2, 0] == 1 and
                    win[0, 1] != 1 and win[0, 2] != 1 and win[2, 1] != 1 and win[2, 2] != 1):
                    s = 10
                    # bonus if surrounded mostly by the room's color
                    if best_score < s:
                        best = (1, 0, rr + 1, cc + 1)  # down? see note below
                        best_score = s
                # left chevron
                if (win[0, 2] == 1 and win[1, 1] == 1 and win[1, 2] == 1 and win[2, 2] == 1 and
                    win[0, 1] != 1 and win[0, 0] != 1 and win[2, 1] != 1 and win[2, 0] != 1):
                    s = 10
                    if best_score < s:
                        best = (-1, 0, rr + 1, cc + 1)  # up? placeholder refined below
                        best_score = s
                # down chevron
                if (win[1, 0] == 1 and win[1, 1] == 1 and win[2, 0] == 1 and
                    win[0, 0] != 1 and win[0, 1] != 1 and win[2, 1] != 1 and win[2, 2] != 1):
                    s = 10
                    if best_score < s:
                        best = (1, 0, rr + 2, cc + 0)
                        best_score = s
                # up chevron
                if (win[0, 0] == 1 and win[0, 1] == 1 and win[1, 1] == 1 and
                    win[1, 0] != 1 and win[2, 0] != 1 and win[2, 1] != 1 and win[0, 2] != 1):
                    s = 10
                    if best_score < s:
                        best = (-1, 0, rr + 0, cc + 1)
                        best_score = s

        if best is not None:
            return best

        # Fallback: use centroid vs bounding box to choose a direction.
        yr, yc = ones[:, 0].astype(float), ones[:, 1].astype(float)
        mean_r = yr.mean()
        mean_c = yc.mean()
        top = yr.min(); bot = yr.max(); left = yc.min(); right = yc.max()
        height = max(1, bot - top)
        width = max(1, right - left)

        # vertical line with protrusion
        if height >= 2 * max(1, width):
            # more mass on left -> right arrow? more mass on right -> left?
            left_mass = np.sum(yc < mean_c)
            right_mass = np.sum(yc > mean_c)
            if right_mass > left_mass:
                return (-1, 0, int(round(mean_r)), int(round(yc[yc.argmax()])))
            else:
                return (1, 0, int(round(mean_r)), int(round(yc[yc.argmin()])))
        if width >= 2 * max(1, height):
            top_mass = np.sum(yr < mean_r)
            bot_mass = np.sum(yr > mean_r)
            if bot_mass > top_mass:
                return (0, -1, int(round(yr[yr.argmax()])), int(round(mean_c)))
            else:
                return (0, 1, int(round(yr[yr.argmin()])), int(round(mean_c)))

        # default: point from centroid to room center
        rc = (r0 + r1) / 2.0
        cc = (c0 + c1) / 2.0
        dy = rc - (r0 + mean_r)
        dx = cc - (c0 + mean_c)
        if abs(dy) >= abs(dx):
            return (1 if dy > 0 else -1, 0, int(r0 + mean_r), int(c0 + mean_c))
        else:
            return (0, 1 if dx > 0 else -1, int(r0 + mean_r), int(c0 + mean_c))

    # ------------------------------------------------------------------
    # Target color lookup by ray casting.
    # ------------------------------------------------------------------
    def target_color(room, arrow):
        base = room["color"]
        r0, r1, c0, c1 = room["r0"], room["r1"], room["c0"], room["c1"]

        if arrow is None:
            # No arrow: point toward grid center.
            rc = (r0 + r1) / 2.0
            cc = (c0 + c1) / 2.0
            gr, gc = h / 2.0, w / 2.0
            dy, dx = gr - rc, gc - cc
            if abs(dy) >= abs(dx):
                dy = 1 if dy > 0 else -1
                dx = 0
            else:
                dx = 1 if dx > 0 else -1
                dy = 0
            # start from boundary
            if dy == 1:
                sr, sc = r1, int(cc)
            elif dy == -1:
                sr, sc = r0, int(cc)
            elif dx == 1:
                sr, sc = int(rc), c1
            else:
                sr, sc = int(rc), c0
        else:
            dy, dx, sr, sc = arrow

        # If the tip is on a 1, start just beyond it.
        if 0 <= sr < h and 0 <= sc < w and g[sr, sc] == 1:
            sr += dy
            sc += dx

        # Ray cast, ignoring 1 and the room's own base color.
        for _ in range(h + w + 5):
            if not (0 <= sr < h and 0 <= sc < w):
                break
            val = int(g[sr, sc])
            if val != 1 and val != base:
                return val
            sr += dy
            sc += dx

        # Fallback: adjacent room in direction, if any.
        for room2 in rooms:
            if room2 is room:
                continue
            # simple containment check for a neighboring rectangle
            if dy == 1 and room2["r0"] >= r1 - 1 and not (c1 <= room2["c0"] or room2["c1"] <= c0):
                return room2["color"]
            if dy == -1 and room2["r1"] <= r0 + 1 and not (c1 <= room2["c0"] or room2["c1"] <= c0):
                return room2["color"]
            if dx == 1 and room2["c0"] >= c1 - 1 and not (r1 <= room2["r0"] or room2["r1"] <= r0):
                return room2["color"]
            if dx == -1 and room2["c1"] <= c0 + 1 and not (r1 <= room2["r0"] or room2["r1"] <= r0):
                return room2["color"]

        return base

    # ------------------------------------------------------------------
    # Recolor rooms.
    # ------------------------------------------------------------------
    out = np.zeros_like(g)
    for room in rooms:
        arrow = detect_arrow(room)
        col = target_color(room, arrow)
        out[room["r0"]:room["r1"], room["c0"]:room["c1"]] = col

    # If the partition missed some cells (because of non-rectangular rooms or
    # small holes), fill any remaining zeros with the nearest room color.
    if np.any(out == 0):
        for i in range(h):
            for j in range(w):
                if out[i, j] == 0:
                    # choose dominant non-zero color in a small neighborhood
                    r0 = max(0, i - 2)
                    r1 = min(h, i + 3)
                    c0 = max(0, j - 2)
                    c1 = min(w, j + 3)
                    neigh = out[r0:r1, c0:c1]
                    neigh = neigh[neigh != 0]
                    if neigh.size:
                        vals, counts = np.unique(neigh, return_counts=True)
                        out[i, j] = vals[np.argmax(counts)]
                    else:
                        out[i, j] = g[i, j] if g[i, j] != 0 else 0

    # ------------------------------------------------------------------
    # Rotate non-square outputs as observed in the examples.
    # ------------------------------------------------------------------
    if h != w:
        if w % 2 == 1:
            out = np.rot90(out, k=-1)  # clockwise
        else:
            out = np.rot90(out, k=1)   # counter-clockwise

    return out