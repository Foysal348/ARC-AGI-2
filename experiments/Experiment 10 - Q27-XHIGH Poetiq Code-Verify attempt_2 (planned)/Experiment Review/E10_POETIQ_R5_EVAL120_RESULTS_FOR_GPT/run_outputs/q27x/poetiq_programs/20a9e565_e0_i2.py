import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the rectangle marked by the two L-shaped 5-pixel corner markers.

    The 5 pixels define the output bounding box.  The content of that box is
    completed from the visible non-5 pattern.  The implementation handles the
    task family seen in the examples:
      * a 2-cell-thick vertical strip (Example 1 style),
      * a 3-cell-thick vertical ladder (Example 2 style),
      * a 3-cell-thick horizontal bracket / C-shape (Example 3 style),
      * a 2-cell-thick horizontal strip,
      * wider horizontal bracket completions.
    """
    if grid.size == 0:
        return grid.copy()

    H_in, W_in = grid.shape
    mask5 = (grid == 5)
    if not mask5.any():
        return np.zeros_like(grid)

    # The output box is exactly the bounding box of all 5 pixels.
    rs = np.where(mask5.any(axis=1))[0]
    cs = np.where(mask5.any(axis=0))[0]
    r0, r1 = int(rs[0]), int(rs[-1])
    c0, c1 = int(cs[0]), int(cs[-1])
    H, W = r1 - r0 + 1, c1 - c0 + 1
    out = np.zeros((H, W), dtype=grid.dtype)

    # Non-background, non-marker colors.
    flat = grid[(grid != 0) & (grid != 5)]
    if flat.size == 0:
        return out
    vals, counts = np.unique(flat, return_counts=True)
    palette = [int(v) for v in vals if v != 0]
    if not palette:
        return out

    # ---------------------------------------------------------------
    # Connected components of non-5, non-0 cells.
    # ---------------------------------------------------------------
    def get_components(g, exclude=5):
        Hc, Wc = g.shape
        comp_mask = (g != 0) & (g != exclude)
        seen = np.zeros_like(comp_mask, dtype=bool)
        comps = []
        for i in range(Hc):
            for j in range(Wc):
                if comp_mask[i, j] and not seen[i, j]:
                    q = deque([(i, j)])
                    seen[i, j] = True
                    cells = []
                    while q:
                        r, c = q.popleft()
                        cells.append((r, c))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < Hc and 0 <= nc < Wc and comp_mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    if len(cells) < 2:
                        continue
                    rmin = min(r for r, _ in cells)
                    rmax = max(r for r, _ in cells)
                    cmin = min(c for _, c in cells)
                    cmax = max(c for _, c in cells)
                    col_counts = {}
                    for r, c in cells:
                        col_counts[int(g[r, c])] = col_counts.get(int(g[r, c]), 0) + 1
                    color = max(col_counts, key=col_counts.get)
                    comps.append({
                        "color": int(color),
                        "rmin": rmin, "rmax": rmax,
                        "cmin": cmin, "cmax": cmax,
                        "h": rmax - rmin + 1,
                        "w": cmax - cmin + 1,
                    })
        return comps

    comps = get_components(grid)

    # ---------------------------------------------------------------
    # Small helpers.
    # ---------------------------------------------------------------
    def detect_cycle(seq):
        """Return the shortest period that matches seq, or seq if none."""
        if len(seq) <= 1:
            return list(seq)
        for p in range(1, len(seq) // 2 + 1):
            ok = True
            for i in range(len(seq) - p):
                if seq[i] != seq[i + p]:
                    ok = False
                    break
            if ok:
                return seq[:p]
        return list(seq)

    def fit_arithmetic(pos):
        """If positions form an arithmetic progression, return (first, step)."""
        if len(pos) < 2:
            return None
        pos = sorted(pos)
        diffs = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
        step = int(round(np.median(diffs)))
        if step <= 0:
            return None
        if max(abs(d - step) for d in diffs) <= 1:
            return pos[0], step
        return None

    def side_mass(g, r0, r1, c0, c1):
        """Determine which side of the 5-box has the visible pattern."""
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        m = max(10, min(h, w) * 2)
        non5 = (g != 0) & (g != 5)

        left = non5[r0:r1 + 1, max(0, c0 - m):c0]
        right = non5[r0:r1 + 1, c1 + 1:c1 + 1 + m]
        above = non5[max(0, r0 - m):r0, c0:c1 + 1]
        below = non5[r1 + 1:r1 + 1 + m, c0:c1 + 1]

        score = {
            "left": int(left.sum()),
            "right": int(right.sum()),
            "above": int(above.sum()),
            "below": int(below.sum()),
        }
        best = max(score, key=score.get)
        if score[best] == 0:
            return None
        return best

    def dominant_near(g, r0, r1, c0, c1, side):
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        m = max(10, min(h, w) * 2)
        non5 = (g != 0) & (g != 5)
        if side == "left":
            win = non5[r0:r1 + 1, max(0, c0 - m):c0]
        elif side == "right":
            win = non5[r0:r1 + 1, c1 + 1:c1 + 1 + m]
        elif side == "above":
            win = non5[max(0, r0 - m):r0, c0:c1 + 1]
        else:
            win = non5[r1 + 1:r1 + 1 + m, c0:c1 + 1]

        region = g[win]
        region = region[(region != 0) & (region != 5)]
        if region.size == 0:
            return None
        v, c = np.unique(region, return_counts=True)
        return int(v[np.argmax(c)])

    def select_single_color(axis, direction_positive, target_pos, min_width=1, min_height=1):
        """
        Choose the next color in a 1D color sequence.
        axis is 'x' or 'y'; direction_positive means the target is after the visible objects.
        """
        side = "left" if axis == "x" and direction_positive else ("right" if axis == "x" else ("above" if direction_positive else "below"))
        filtered = []
        for comp in comps:
            if comp["w"] < min_width or comp["h"] < min_height:
                continue
            # Keep components reasonably close to the target band.
            if axis == "x":
                if comp["rmax"] < r0 - max(10, H) or comp["rmin"] > r1 + max(10, H):
                    continue
                pos = comp["cmin"]
            else:
                if comp["cmax"] < c0 - max(10, W) or comp["cmin"] > c1 + max(10, W):
                    continue
                pos = comp["rmin"]
            filtered.append((pos, comp["color"]))

        if not filtered:
            return dominant_near(grid, r0, r1, c0, c1, side) or palette[0]

        filtered.sort()
        positions = [p for p, _ in filtered]
        colors = [col for _, col in filtered]

        # If the positions are an arithmetic progression, use that index.
        prog = fit_arithmetic(positions)
        if prog is not None:
            first, step = prog
            idx = int(round((target_pos - first) / step))
            # Only trust the index if it is not absurdly far.
            if -2 <= idx <= len(positions) + 2:
                cyc = detect_cycle(colors)
                return cyc[idx % len(cyc)]

        # Otherwise, if the target is beyond the last visible object, continue the sequence.
        if direction_positive and target_pos > positions[-1]:
            cyc = detect_cycle(colors)
            idx = len(positions)
            return cyc[idx % len(cyc)]
        if (not direction_positive) and target_pos < positions[0]:
            cyc = detect_cycle(colors)
            idx = -1
            return cyc[idx % len(cyc)]

        # Two-color alternation fallback: pick the other color from the nearest visible color.
        if len(set(palette)) == 2:
            nearest = colors[-1] if direction_positive else colors[0]
            for col in palette:
                if col != nearest:
                    return col

        return dominant_near(grid, r0, r1, c0, c1, side) or palette[0]

    # ---------------------------------------------------------------
    # Special selector for a 2-row horizontal strip.
    # Uses horizontal color bands when present (Challenge #1 style).
    # ---------------------------------------------------------------
    def select_2row_color():
        row_dom = []
        for r in range(H_in):
            vals_r, cnt_r = np.unique(grid[r][(grid[r] != 0) & (grid[r] != 5)], return_counts=True)
            if vals_r.size:
                row_dom.append((r, int(vals_r[np.argmax(cnt_r)])))

        bands = []
        for r, col in row_dom:
            if bands and r == bands[-1][1] + 1 and col == bands[-1][2]:
                bands[-1][1] = r
            else:
                bands.append([r, r, col])

        bands = [b for b in bands if b[1] - b[0] + 1 >= 2]
        if len(bands) >= 2:
            tops = [b[0] for b in bands]
            colors = [b[2] for b in bands]
            prog = fit_arithmetic(tops)
            if prog is not None:
                first, step = prog
                best_t, best_idx = None, None
                # A 2-row marker box can be the top/middle or middle/bottom rows of a 3-row band.
                for t in (r0 - 2, r0 - 1, r0, r0 + 1):
                    if t < 0:
                        continue
                    x = (t - first) / step
                    idx = int(round(x))
                    if abs(x - idx) < 0.05:
                        best_t, best_idx = t, idx
                        break
                if best_t is not None:
                    cyc = detect_cycle(colors)
                    return cyc[best_idx % len(cyc)]

        # Fallback: infer from the side containing the pattern.
        side = side_mass(grid, r0, r1, c0, c1)
        if side == "below":
            return select_single_color("y", False, r0, min_width=1, min_height=1)
        if side == "above":
            return select_single_color("y", True, r0, min_width=1, min_height=1)
        if side == "left":
            return select_single_color("x", True, c0, min_width=1, min_height=1)
        if side == "right":
            return select_single_color("x", False, c0, min_width=1, min_height=1)
        return palette[0]

    # ---------------------------------------------------------------
    # Special selector for a 2-column vertical strip (Example 1 style).
    # Finds the nearest visible 2-column strip and extrapolates its 2x2 tile cycle.
    # ---------------------------------------------------------------
    def select_2col_cycle():
        best_strip = None
        best_score = 0

        # Prefer the closest clean 2-column strip to the left or right.
        for dx in list(range(-1, -12, -1)) + list(range(1, 13)):
            col = c0 + dx
            if col < 0 or col + 1 >= W_in:
                continue
            strip = grid[r0:r0 + H, col:col + 2]
            strip = strip[(strip != 0) & (strip != 5)]
            total = int(strip.size)
            if total == 0:
                continue
            left_count = int((grid[r0:r0 + H, col] != 0).sum())
            right_count = int((grid[r0:r0 + H, col + 1] != 0).sum())
            score = int(strip.sum())
            if min(left_count, right_count) >= max(2, H // 5) and score > best_score:
                best_score = score
                best_strip = (col, col + 1)

        if best_strip is None:
            # Fallback: cycle through the palette in the order they appear.
            order = []
            for r in range(H_in):
                for c in range(W_in):
                    v = int(grid[r, c])
                    if v not in (0, 5) and v not in order:
                        order.append(v)
                    if len(order) >= 3:
                        break
                if len(order) >= 3:
                    break
            return order or palette, "bl"

        col, _ = best_strip
        strip = grid[r0:r0 + H, col:col + 2]
        mains = []
        first_pos = None

        for i in range(0, H - 1, 2):
            tile = strip[i:i + 2, :]
            tvals, tcnt = np.unique(tile, return_counts=True)
            tvals = [int(v) for v in tvals if v not in (0, 5)]
            if not tvals:
                continue
            tcnt = [int(c) for c in tcnt if int(tvals[np.where(tcnt == c)[0][0]]) not in (0, 5)] if False else tcnt
            # Simpler: count non-zero colors.
            cc = {}
            for r in range(tile.shape[0]):
                for c in range(tile.shape[1]):
                    v = int(tile[r, c])
                    if v not in (0, 5):
                        cc[v] = cc.get(v, 0) + 1
            main = max(cc, key=cc.get)
            mains.append(main)

            if first_pos is None:
                if tile[1, 0] != 0 and tile[1, 0] != main:
                    first_pos = "bl"
                elif tile[1, 1] != 0 and tile[1, 1] != main:
                    first_pos = "br"
                else:
                    first_pos = "bl"

        if len(mains) >= 3:
            cyc = detect_cycle(mains)
        elif len(mains) >= 2:
            cyc = mains
        else:
            cyc = palette[:3] or palette

        return cyc, (first_pos or "bl")

    # ---------------------------------------------------------------
    # Generators.
    # ---------------------------------------------------------------
    def gen_vertical_2col(cyc, pos0):
        n = H // 2
        for i in range(n):
            main = cyc[i % len(cyc)]
            diff = cyc[(i + 1) % len(cyc)] if i < n - 1 else 0
            if i % 2 == 0:
                pos = pos0
            else:
                pos = "br" if pos0 == "bl" else "bl"

            out[2 * i, 0] = main
            out[2 * i, 1] = main
            if pos == "bl":
                out[2 * i + 1, 0] = diff
                out[2 * i + 1, 1] = main
            else:
                out[2 * i + 1, 0] = main
                out[2 * i + 1, 1] = diff

        if H % 2 == 1:
            main = cyc[n % len(cyc)]
            out[H - 1, 0] = main
            out[H - 1, 1] = main

    def gen_vertical_ladder(color):
        for r in range(H):
            if r % 2 == 0:
                out[r, :] = color
            else:
                out[r, 0] = color
                out[r, -1] = color

    def gen_horizontal_2row(color):
        out[0, :] = color
        for c in range(0, W, 2):
            out[1, c] = color
            if c + 1 < W:
                out[1, c + 1] = 0

    def gen_horizontal_bracket(color):
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color

    # ---------------------------------------------------------------
    # Dispatch on the output box shape.
    # ---------------------------------------------------------------
    if W == 2 and H > 2:
        cyc, pos0 = select_2col_cycle()
        gen_vertical_2col(cyc, pos0)

    elif H == 2 and W > 2:
        color = select_2row_color()
        gen_horizontal_2row(color)

    elif W == 3 and H > 3:
        # Vertical 3-wide ladder: choose color from the horizontal sequence of ladders.
        filtered = []
        for comp in comps:
            if 2 <= comp["w"] <= 4 and comp["h"] >= 3:
                filtered.append((comp["cmin"], comp["color"]))
        if filtered:
            filtered.sort()
            positions = [p for p, _ in filtered]
            colors = [col for _, col in filtered]
            prog = fit_arithmetic(positions)
            if prog is not None:
                first, step = prog
                idx = int(round((c0 - first) / step))
                cyc = detect_cycle(colors)
                color = cyc[idx % len(cyc)]
            else:
                color = select_single_color("x", True, c0, min_width=2, min_height=3)
        else:
            color = select_single_color("x", True, c0, min_width=2, min_height=3)
        gen_vertical_ladder(color)

    elif H == 3 and W > 3:
        # Horizontal 3-high bracket / C-shape.
        color = select_single_color("x", True, c0, min_width=3, min_height=1)
        gen_horizontal_bracket(color)

    elif H < W:
        # Wider horizontal completion: use a bracket of the next color.
        side = side_mass(grid, r0, r1, c0, c1)
        if side == "left":
            color = select_single_color("x", True, c0, min_width=1, min_height=1)
        elif side == "right":
            color = select_single_color("x", False, c0, min_width=1, min_height=1)
        elif side == "above":
            color = select_single_color("y", True, r0, min_width=1, min_height=1)
        elif side == "below":
            color = select_single_color("y", False, r0, min_width=1, min_height=1)
        else:
            color = palette[0]
        gen_horizontal_bracket(color)

    elif W < H:
        # Taller vertical completion.  If not already handled, use a vertical ladder-like bracket.
        side = side_mass(grid, r0, r1, c0, c1)
        if side == "left":
            color = select_single_color("x", True, c0, min_width=1, min_height=1)
        elif side == "right":
            color = select_single_color("x", False, c0, min_width=1, min_height=1)
        elif side == "above":
            color = select_single_color("y", True, r0, min_width=1, min_height=1)
        elif side == "below":
            color = select_single_color("y", False, r0, min_width=1, min_height=1)
        else:
            color = palette[0]

        # For 3-wide boxes this should not reach here; otherwise make a vertical bracket.
        out[:, 0] = color
        out[0, :] = color
        out[-1, :] = color
        if W >= 3:
            out[:, -1] = color

    else:
        # Square or degenerate case: fill with a simple bracket in the dominant nearby color.
        side = side_mass(grid, r0, r1, c0, c1)
        color = dominant_near(grid, r0, r1, c0, c1, side) or palette[0]
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color
        out[:, -1] = color

    return out