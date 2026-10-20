import numpy as np
from collections import Counter, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular pieces into one complete frame.

    The input contains several rectangular pieces on a background color.
    The correct pieces form a single larger rectangle whose outer border is
    one solid “frame” color.  The transformation:
      1. Detects the background color.
      2. Extracts one rectangular piece from each connected non-background
         component, choosing the largest valid all-non-background rectangle.
      3. Infers the frame color from full edges of the pieces.
      4. Searches for a tiling of the pieces into a rectangle whose perimeter
         is entirely the frame color.
      5. Places the reassembled rectangle in the original canvas.  If there is
         a uniquely largest piece, that piece is centered in the canvas;
         otherwise the whole assembled rectangle is centered.
    """

    H, W = grid.shape
    out = np.full((H, W), 0, dtype=grid.dtype)
    if H == 0 or W == 0:
        return out

    # Background is the most frequent color.
    bg = int(Counter(grid.ravel()).most_common(1)[0][0])
    nonbg = grid != bg

    # ------------------------------------------------------------------
    # Connected components of non-background cells.
    # ------------------------------------------------------------------
    comp_id = -np.ones((H, W), dtype=int)
    components = []
    cid = 0
    for i in range(H):
        for j in range(W):
            if nonbg[i, j] and comp_id[i, j] == -1:
                q = deque([(i, j)])
                comp_id[i, j] = cid
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                        if 0 <= nr < H and 0 <= nc < W and nonbg[nr, nc] and comp_id[nr, nc] == -1:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))
                components.append(cells)
                cid += 1

    # ------------------------------------------------------------------
    # For each component, choose the largest rectangular sub-block that has
    # no background and both dimensions at least 3.
    # ------------------------------------------------------------------
    def rect_is_full(r0, r1, c0, c1):
        # inclusive bounds
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            return False
        return bool(np.all(nonbg[r0:r1+1, c0:c1+1]))

    def is_maximal(r0, r1, c0, c1):
        if r0 > 0 and rect_is_full(r0-1, r1, c0, c1):
            return False
        if r1 + 1 < H and rect_is_full(r0, r1+1, c0, c1):
            return False
        if c0 > 0 and rect_is_full(r0, r1, c0-1, c1):
            return False
        if c1 + 1 < W and rect_is_full(r0, r1, c0, c1+1):
            return False
        return True

    tiles = []
    for cells in components:
        if not cells:
            continue
        rmin = min(r for r, _ in cells)
        rmax = max(r for r, _ in cells)
        cmin = min(c for _, c in cells)
        cmax = max(c for _, c in cells)

        best = None
        for r0 in range(rmin, rmax + 1):
            for r1 in range(r0, rmax + 1):
                h = r1 - r0 + 1
                if h < 3:
                    continue
                for c0 in range(cmin, cmax + 1):
                    for c1 in range(c0, cmax + 1):
                        w = c1 - c0 + 1
                        if w < 3:
                            continue
                        if rect_is_full(r0, r1, c0, c1) and is_maximal(r0, r1, c0, c1):
                            area = h * w
                            if best is None or area > best[0]:
                                best = (area, r0, r1, c0, c1)
        if best is not None:
            area, r0, r1, c0, c1 = best
            tiles.append({
                'h': r1 - r0 + 1,
                'w': c1 - c0 + 1,
                'pattern': grid[r0:r1+1, c0:c1+1].copy(),
                'pos': (r0, c0),
            })

    if not tiles:
        # Fallback: if no proper piece can be found, return background.
        return np.full((H, W), bg, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # Infer likely frame color: the color that completes full edges often.
    # ------------------------------------------------------------------
    def edge_full(tile, color, side):
        p = tile['pattern']
        if side == 'top':
            return bool(np.all(p[0, :] == color))
        if side == 'bottom':
            return bool(np.all(p[-1, :] == color))
        if side == 'left':
            return bool(np.all(p[:, 0] == color))
        if side == 'right':
            return bool(np.all(p[:, -1] == color))
        return False

    edge_scores = Counter()
    for t in tiles:
        for color in Counter(t['pattern'].ravel()).keys():
            if color == bg:
                continue
            for side in ('top', 'bottom', 'left', 'right'):
                if edge_full(t, color, side):
                    edge_scores[int(color)] += 1

    if edge_scores:
        frame_color = int(edge_scores.most_common(1)[0][0])
    else:
        frame_color = int(Counter(grid.ravel()).most_common(2)[1][0])

    # ------------------------------------------------------------------
    # Packing search: place tiles exactly into an H x W rectangle.
    # The perimeter of the assembled rectangle must be frame_color.
    # ------------------------------------------------------------------
    def can_place_frame(t, r, c, h_box, w_box, fc):
        if r == 0 and not edge_full(t, fc, 'top'):
            return False
        if r + t['h'] == h_box and not edge_full(t, fc, 'bottom'):
            return False
        if c == 0 and not edge_full(t, fc, 'left'):
            return False
        if c + t['w'] == w_box and not edge_full(t, fc, 'right'):
            return False
        return True

    def solve_packing(tile_list, fc, max_solutions=1):
        if not tile_list:
            return []
        area = sum(t['h'] * t['w'] for t in tile_list)
        max_h = max(t['h'] for t in tile_list)
        max_w = max(t['w'] for t in tile_list)

        solutions = []
        # Try all plausible dimensions that fit in the original grid.
        for h in range(1, H + 1):
            if area % h != 0:
                continue
            w = area // h
            if w > W:
                continue
            if h < max_h or w < max_w:
                continue

            placed = [False] * len(tile_list)
            occupied = np.zeros((h, w), dtype=bool)
            placements = []

            # Use largest-first heuristic.
            order = sorted(range(len(tile_list)), key=lambda i: -tile_list[i]['h'] * tile_list[i]['w'])

            def backtrack(idx):
                if len(solutions) >= max_solutions:
                    return
                if idx == len(order):
                    if not np.any(~occupied):
                        solutions.append(list(placements))
                    return

                # Find first empty cell in row-major order.
                target = None
                for r in range(h):
                    for c in range(w):
                        if not occupied[r, c]:
                            target = (r, c)
                            break
                    if target is not None:
                        break
                if target is None:
                    if not np.any(~occupied):
                        solutions.append(list(placements))
                    return
                tr, tc = target

                i = order[idx]
                t = tile_list[i]
                th, tw = t['h'], t['w']

                # Try all placements covering the target cell.
                for dy in range(th):
                    for dx in range(tw):
                        r = tr - dy
                        c = tc - dx
                        if r < 0 or c < 0 or r + th > h or c + tw > w:
                            continue
                        if not can_place_frame(t, r, c, h, w, fc):
                            continue
                        if np.any(occupied[r:r+th, c:c+tw]):
                            continue

                        placements.append((i, r, c))
                        occupied[r:r+th, c:c+tw] = True
                        placed[i] = True
                        backtrack(idx + 1)
                        placed[i] = False
                        occupied[r:r+th, c:c+tw] = False
                        placements.pop()
                        if len(solutions) >= max_solutions:
                            return

            backtrack(0)
            if solutions:
                break
        return solutions

    # Try all tiles first; if that fails, try dropping low-scoring tiles.
    def tile_score(t):
        s = 0
        for side in ('top', 'bottom', 'left', 'right'):
            if edge_full(t, frame_color, side):
                s += 1
        s += min(4, t['h'] * t['w'] // 9)
        return s

    tiles_by_score = sorted(tiles, key=lambda t: (tile_score(t), t['h'] * t['w']))
    final_solution = None
    final_frame = frame_color

    # Keep a copy of original order for position cost later.
    original_tiles = tiles[:]
    for drop_count in range(0, min(3, len(tiles) - 3 + 1)):
        # Try subsets formed by dropping the lowest-scored tiles.
        # For simplicity, drop first, then pairs, etc., among low-score tiles.
        if drop_count == 0:
            candidates = [tiles]
        else:
            low = tiles_by_score[:drop_count + 2]
            combos = []
            def gen(idx, acc):
                if len(acc) == drop_count:
                    combos.append(acc)
                    return
                for i in range(idx, len(low)):
                    acc.append(low[i])
                    gen(i + 1, acc)
                    acc.pop()
            gen(0, [])
            candidates = []
            for dropped in combos:
                dset = set(id(x) for x in dropped)
                sub = [t for t in tiles if id(t) not in dset]
                if len(sub) >= 3:
                    candidates.append(sub)

        for subset in candidates:
            sols = solve_packing(subset, final_frame, max_solutions=1)
            if sols:
                # Record which tile indices in `subset` were used.
                subset_ids = [id(t) for t in subset]
                final_solution = sols[0]
                final_used_tiles = subset
                break
        if final_solution is not None:
            break

    if final_solution is None:
        # Last resort: use the single largest component as the output.
        best = max(tiles, key=lambda t: t['h'] * t['w'])
        final_used_tiles = [best]
        final_solution = [(0, 0, 0)]
        frame_color = best['pattern'][0, 0]

    # Build assembled image.
    placed_set = set((r, c) for _, r, c in final_solution)
    min_r = min(r for _, r, _ in final_solution) if final_solution else 0
    max_r = max(r + final_used_tiles[i]['h'] for i, r, _ in final_solution) if final_solution else 1
    min_c = min(c for _, _, c in final_solution) if final_solution else 0
    max_c = max(c + final_used_tiles[i]['w'] for i, _, c in final_solution) if final_solution else 1
    final_h = max_r - min_r
    final_w = max_c - min_c
    final_img = np.full((final_h, final_w), bg, dtype=grid.dtype)

    for i, r, c in final_solution:
        t = final_used_tiles[i]
        final_img[r - min_r:r - min_r + t['h'], c - min_c:c - min_c + t['w']] = t['pattern']

    # ------------------------------------------------------------------
    # Place the final assembled rectangle in the original canvas.
    # ------------------------------------------------------------------
    if final_h > H or final_w > W:
        # Should not happen, but crop if needed.
        final_img = final_img[:H, :W]
        return final_img

    # Determine allowed offsets.
    if len(final_used_tiles) > 1:
        areas = [t['h'] * t['w'] for t in final_used_tiles]
        max_area = max(areas)
        if areas.count(max_area) == 1:
            anchor = next(t for t in final_used_tiles if t['h'] * t['w'] == max_area)
            anchor_rel = next((r, c) for i, r, c in final_solution
                              if final_used_tiles[i] is anchor)
            anchor_h, anchor_w = anchor['h'], anchor['w']
            anchor_pos = anchor['pos']

            # Center the anchor piece.
            top_opts = {(H - anchor_h) // 2, (H - anchor_h + 1) // 2}
            left_opts = {(W - anchor_w) // 2, (W - anchor_w + 1) // 2}
            candidate_offsets = []
            for R in top_opts:
                for C in left_opts:
                    a_r = R + anchor_rel[0]
                    a_c = C + anchor_rel[1]
                    # Final image must fit.
                    if 0 <= R and R + final_h <= H and 0 <= C and C + final_w <= W:
                        # Prefer offsets that keep anchor reasonably centered.
                        candidate_offsets.append((R, C))
            if not candidate_offsets:
                candidate_offsets = [(0, 0)]
        else:
            top_opts = {(H - final_h) // 2, (H - final_h + 1) // 2}
            left_opts = {(W - final_w) // 2, (W - final_w + 1) // 2}
            candidate_offsets = [(R, C) for R in top_opts for C in left_opts
                                 if 0 <= R and R + final_h <= H and 0 <= C and C + final_w <= W]
            if not candidate_offsets:
                candidate_offsets = [(0, 0)]
    else:
        top_opts = {(H - final_h) // 2, (H - final_h + 1) // 2}
        left_opts = {(W - final_w) // 2, (W - final_w + 1) // 2}
        candidate_offsets = [(R, C) for R in top_opts for C in left_opts
                             if 0 <= R and R + final_h <= H and 0 <= C and C + final_w <= W]
        if not candidate_offsets:
            candidate_offsets = [(0, 0)]

    # Choose the offset minimizing movement from the input piece positions.
    def placement_cost(R, C):
        cost = 0
        for i, r, c in final_solution:
            t = final_used_tiles[i]
            orow, ocol = R + r - min_r, C + c - min_c
            ir, ic = t['pos']
            cost += abs(ir - orow) + abs(ic - ocol)
        return cost

    best_offset = min(candidate_offsets, key=placement_cost)
    R, C = best_offset

    out[R:R+final_h, C:C+final_w] = final_img
    return out