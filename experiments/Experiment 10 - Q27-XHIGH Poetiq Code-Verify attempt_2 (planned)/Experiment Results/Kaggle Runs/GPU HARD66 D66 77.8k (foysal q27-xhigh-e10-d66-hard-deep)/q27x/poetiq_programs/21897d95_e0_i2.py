import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the output grid.

    The input is interpreted as a rectangular array of “rooms”.  Each room has
    a dominant background color and may contain a small arrow made from 1-pixels.
    Sometimes the arrow contains a special colored pixel; that special color is
    the color sent by the arrow.  Otherwise the room's background color is sent.

    The rooms are arranged into a regular block grid.  The transformation is:
      1. Infer the regular room/block grid from the input.
      2. For each room, detect arrow directions from its 1-pixel pattern.
      3. Propagate colors through the directed room graph until stable.
      4. Transpose the resulting room-color matrix and expand it using the
         transposed room dimensions.

    This implementation is deliberately heuristic but is tuned to the ARC
    patterns in the examples: rectangular rooms, small T/diagonal arrow glyphs,
    possible multiple arms per glyph, and color flow through receiving rooms.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    BG = 0

    # ------------------------------------------------------------------
    # Utility: dominant color in a rectangle, ignoring 1 and, if possible, 0.
    # ------------------------------------------------------------------
    def dominant_color(rect: np.ndarray) -> int:
        mask = rect != 1
        if not np.any(mask):
            return 1
        vals = rect[mask]
        if vals.size == 0:
            return 1
        # Prefer non-zero colors when available.
        nz = vals[vals != 0]
        if nz.size > 0:
            vals = nz
        vals = vals.tolist()
        # Count colors.
        counts = {}
        for v in vals:
            counts[int(v)] = counts.get(int(v), 0) + 1
        if not counts:
            return 0
        # Deterministic tie-break by color value.
        return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    # ------------------------------------------------------------------
    # Infer block cuts.
    #
    # The correct cuts are usually lines where the visible color structure
    # changes, especially near rows/columns containing 1-pixel arrows.
    # We score candidate cuts by a combination of:
    #   - number of color changes across the cut (ignoring 1)
    #   - proximity to 1-pixels
    #   - stability of the color above/left and below/right
    # Then we choose a small set of separated cuts.
    # ------------------------------------------------------------------
    def choose_cuts(scores, end, min_spacing=2, max_cuts=12):
        if not scores:
            return [0, end]
        # Keep only reasonably strong candidates.
        best = max(score for _, score in scores)
        if best <= 0:
            return [0, end]
        thresh = max(4, best * 0.18)
        chosen = sorted([pos for pos, score in scores if score >= thresh])
        if not chosen:
            return [0, end]

        filtered = []
        last = 0
        for pos in chosen:
            if pos - last >= min_spacing:
                filtered.append(pos)
                last = pos
            if len(filtered) >= max_cuts:
                break
        return [0] + filtered + [end]

    def infer_cuts():
        row_scores = []
        col_scores = []

        # Row cuts.
        for r in range(1, H):
            top = grid[r - 1, :]
            bot = grid[r, :]
            valid = (top != 1) & (bot != 1)
            changes = int(np.sum(valid & (top != bot)))

            # How many 1s are near this line?
            near = np.zeros(H, dtype=int)
            for rr in range(max(0, r - 1), min(H, r + 2)):
                near[rr] += 1
            near_ones = int(np.sum((grid == 1) * near[:, None]))

            # Consistency: fraction of columns where either side is mostly one color.
            # A cut is better if it separates coherent bands.
            score = changes * 2 + near_ones * 3
            row_scores.append((r, score))

        # Column cuts.
        for c in range(1, W):
            left = grid[:, c - 1]
            right = grid[:, c]
            valid = (left != 1) & (right != 1)
            changes = int(np.sum(valid & (left != right)))

            near = np.zeros(W, dtype=int)
            for cc in range(max(0, c - 1), min(W, c + 2)):
                near[cc] += 1
            near_ones = int(np.sum((grid == 1) * near[None, :]))

            score = changes * 2 + near_ones * 3
            col_scores.append((c, score))

        rcuts = choose_cuts(row_scores, H)
        ccuts = choose_cuts(col_scores, W)

        # Very small grids / no useful cuts.
        if len(rcuts) < 2:
            rcuts = [0, H]
        if len(ccuts) < 2:
            ccuts = [0, W]

        return rcuts, ccuts

    row_cuts, col_cuts = infer_cuts()
    R = len(row_cuts) - 1
    C = len(col_cuts) - 1

    # ------------------------------------------------------------------
    # Extract room information.
    # ------------------------------------------------------------------
    room_base = np.zeros((R, C), dtype=int)
    room_special = np.zeros((R, C), dtype=int)  # 0 means none
    room_arms = [[set() for _ in range(C)] for _ in range(R)]

    # Directions: up, down, left, right, and four diagonals.
    DIRS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    def arm_directions_for_room(r0, r1, c0, c1):
        sub = grid[r0:r1, c0:c1]
        h, w = sub.shape
        ones = (sub == 1)
        if not np.any(ones):
            return set()

        # Connected components of 1s.
        seen = np.zeros_like(sub, dtype=bool)
        arms = set()
        centers = []

        for i in range(h):
            for j in range(w):
                if ones[i, j] and not seen[i, j]:
                    q = deque([(i, j)])
                    seen[i, j] = True
                    pix = []
                    while q:
                        y, x = q.popleft()
                        pix.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if ones[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = True
                                    q.append((ny, nx))
                    centers.append(pix)

        # For each 1-cluster, infer possible directions.
        for pix in centers:
            ys = [p[0] for p in pix]
            xs = [p[1] for p in pix]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            hh = y1 - y0 + 1
            ww = x1 - x0 + 1
            cy = (sum(ys) / max(1, len(ys)))
            cx = (sum(xs) / max(1, len(xs)))

            # If the cluster is a single pixel, treat it as pointing to the
            # nearest edge of the room.
            if hh == 1 and ww == 1:
                if cy <= 1:
                    arms.add((-1, 0))
                elif cy >= h - 2:
                    arms.add((1, 0))
                if cx <= 1:
                    arms.add((0, -1))
                elif cx >= w - 2:
                    arms.add((0, 1))
                continue

            # Build local binary pattern.
            pat = np.zeros((hh, ww), dtype=int)
            for y, x in pix:
                pat[y - y0, x - x0] = 1

            # T-right: vertical line with a right protrusion.
            if hh >= 3 and ww == 2:
                if pat[0, 0] and pat[1, 0] and pat[1, 1] and pat[2, 0]:
                    arms.add((0, 1))
                    arms.add((-1, 0))
                    arms.add((1, 0))
                    continue

            # T-left: vertical line with a left protrusion.
            if hh >= 3 and ww == 2:
                if pat[0, 1] and pat[1, 0] and pat[1, 1] and pat[2, 1]:
                    arms.add((0, -1))
                    arms.add((-1, 0))
                    arms.add((1, 0))
                    continue

            # Down T: horizontal line with bottom protrusion.
            if ww >= 3 and hh == 2:
                if pat[0, 0] and pat[0, 1] and pat[0, 2] and pat[1, 1]:
                    arms.add((1, 0))
                    arms.add((0, -1))
                    arms.add((0, 1))
                    continue

            # Up T: horizontal line with top protrusion.
            if ww >= 3 and hh == 2:
                if pat[1, 0] and pat[1, 1] and pat[1, 2] and pat[0, 1]:
                    arms.add((-1, 0))
                    arms.add((0, -1))
                    arms.add((0, 1))
                    continue

            # 2x2 anti-diagonal: often used for down-left / up-right flow.
            if hh == 2 and ww == 2:
                if pat[0, 1] and pat[1, 0]:
                    arms.add((1, -1))
                    arms.add((-1, 1))
                    arms.add((0, -1))
                    continue
                if pat[0, 0] and pat[1, 1]:
                    arms.add((1, 1))
                    arms.add((-1, -1))
                    continue

            # Generic: add directions for arms near the centroid that touch
            # room edges or form clear protrusions.
            for dy, dx in DIRS:
                # Sample a few cells ahead of the centroid.
                steps = [1, 2, 3]
                hit = False
                for s in steps:
                    ny = int(round(cy + dy * s))
                    nx = int(round(cx + dx * s))
                    if 0 <= ny < h and 0 <= nx < w and pat[min(ny, hh - 1), min(nx, ww - 1)]:
                        hit = True
                        break
                if hit:
                    arms.add((dy, dx))

        # If no directions were found, default to nearest edges.
        if not arms:
            # Use first cluster center.
            ys = [p[0] for p in centers[0]]
            xs = [p[1] for p in centers[0]]
            cy = sum(ys) / max(1, len(ys))
            cx = sum(xs) / max(1, len(xs))
            if cy <= h / 2:
                arms.add((-1, 0))
            else:
                arms.add((1, 0))
            if cx <= w / 2:
                arms.add((0, -1))
            else:
                arms.add((0, 1))

        return arms

    # Special color: a non-1, non-dominant color that appears inside the room
    # and is adjacent to at least one 1-pixel.
    def special_color_in_room(r0, r1, c0, c1, base):
        sub = grid[r0:r1, c0:c1]
        h, w = sub.shape
        best = 0
        best_score = 0

        for i in range(h):
            for j in range(w):
                v = int(sub[i, j])
                if v == 1 or v == 0 or v == base:
                    continue
                # Check adjacency (8-neighborhood) to a 1-pixel.
                adj1 = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = i + dy, j + dx
                        if 0 <= ny < h and 0 <= nx < w and sub[ny, nx] == 1:
                            adj1 = True
                            break
                    if adj1:
                        break
                if not adj1:
                    continue
                # Score: prefer isolated/rare colors.
                cnt = int(np.sum(sub == v))
                # Rare but not too rare.
                score = 100 - cnt
                if score > best_score:
                    best_score = score
                    best = v
        return best

    for i in range(R):
        for j in range(C):
            r0, r1 = row_cuts[i], row_cuts[i + 1]
            c0, c1 = col_cuts[j], col_cuts[j + 1]
            sub = grid[r0:r1, c0:c1]
            base = dominant_color(sub)
            room_base[i, j] = base
            room_special[i, j] = special_color_in_room(r0, r1, c0, c1, base)
            room_arms[i][j] = arm_directions_for_room(r0, r1, c0, c1)

    # ------------------------------------------------------------------
    # Color propagation.
    #
    # Each room starts with:
    #   initial color = special if present, otherwise base.
    #
    # Then, repeatedly, each room forwards its current color to neighboring
    # rooms in all detected arm directions.  This lets colors flow through
    # chains of rooms.  To avoid endless mutual overwriting, use a bounded
    # number of iterations and process rooms in a deterministic order.
    # ------------------------------------------------------------------
    color = np.zeros((R, C), dtype=int)
    for i in range(R):
        for j in range(C):
            color[i, j] = room_special[i, j] if room_special[i, j] != 0 else room_base[i, j]

    # If a room has no arms, it should still be able to keep/receive color.
    # Iterate a fixed number of times; this is enough for the regular room
    # graphs in the examples.
    for _ in range(20):
        changed = False
        # Row-major order gives a deterministic priority.
        for i in range(R):
            for j in range(C):
                cur = int(color[i, j])
                if cur == 0:
                    continue
                for dy, dx in room_arms[i][j]:
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < R and 0 <= nj < C:
                        if color[ni, nj] != cur:
                            color[ni, nj] = cur
                            changed = True
        if not changed:
            break

    # Rooms with no detected arrows and no flow should keep their base color.
    for i in range(R):
        for j in range(C):
            if color[i, j] == 0:
                color[i, j] = room_base[i, j]

    # ------------------------------------------------------------------
    # Build output by transposing the room-color matrix.
    # ------------------------------------------------------------------
    # Output block rows correspond to input block columns.
    # Output block columns correspond to input block rows.
    out_r = C
    out_c = R

    out_row_sizes = [col_cuts[j + 1] - col_cuts[j] for j in range(C)]
    out_col_sizes = [row_cuts[i + 1] - row_cuts[i] for i in range(R)]

    out = np.full((sum(out_row_sizes), sum(out_col_sizes)), BG, dtype=grid.dtype)

    r = 0
    for oi in range(out_r):
        c = 0
        for oj in range(out_c):
            h = out_row_sizes[oi]
            w = out_col_sizes[oj]
            val = int(color[oj, oi])  # transpose
            out[r:r + h, c:c + w] = val
            c += w
        r += out_row_sizes[oi]

    return out