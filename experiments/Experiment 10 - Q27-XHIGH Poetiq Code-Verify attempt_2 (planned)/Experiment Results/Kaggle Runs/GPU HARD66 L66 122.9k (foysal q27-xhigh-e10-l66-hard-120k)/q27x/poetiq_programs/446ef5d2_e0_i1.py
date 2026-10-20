import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered framed pieces into one framed rectangle.

    The rule inferred from the examples:
      - The most frequent non-background color is the frame color.
      - Color 4 and the background color are treated as empty.
      - Connected non-empty components are rectangular pieces.
      - Each piece is anchored either to the top edge, bottom edge, or both
        (if it spans the whole final height).
      - The pieces are packed into the smallest rectangle whose outer border
        is entirely the frame color.
      - If several local packings have the same minimal area, the one that best
        preserves relative input positions is preferred.
      - The final object is placed in the original canvas by keeping the piece
        that is in the same input/final corner fixed, if possible; otherwise
        it is centered.
    """
    grid = np.array(grid, dtype=int)
    H_in, W_in = grid.shape

    # Background is the most frequent color. Color 4 is always treated as empty.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    empty_colors = np.array([bg, 4], dtype=int)
    mask = ~np.isin(grid, empty_colors)

    if not mask.any():
        return np.full((H_in, W_in), bg, dtype=int)

    # ------------------------------------------------------------------
    # Extract connected components (pieces)
    # ------------------------------------------------------------------
    visited = np.zeros((H_in, W_in), dtype=bool)
    raw_pieces = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(H_in):
        for c in range(W_in):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                min_r = max_r = r
                min_c = max_c = c
                while stack:
                    cr, cc = stack.pop()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H_in and 0 <= nc < W_in:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                raw_pieces.append((min_r, max_r, min_c, max_c))

    if not raw_pieces:
        return np.full((H_in, W_in), bg, dtype=int)

    # Frame color: prefer colors that form full sides of pieces, then frequency.
    nonempty_vals = grid[mask]
    fvals, fcounts = np.unique(nonempty_vals, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(fvals, fcounts)}
    side_score = {c: 0 for c in freq}

    for min_r, max_r, min_c, max_c in raw_pieces:
        sub = grid[min_r:max_r + 1, min_c:max_c + 1]
        # Count how many side rows/columns are constant for each color present.
        side_vals = set(int(v) for v in np.unique(sub[0])) | set(int(v) for v in np.unique(sub[-1])) | \
                    set(int(v) for v in np.unique(sub[:, 0])) | set(int(v) for v in np.unique(sub[:, -1]))
        for col in side_vals:
            if col not in side_score:
                continue
            if np.all(sub[0, :] == col):
                side_score[col] += 1
            if np.all(sub[-1, :] == col):
                side_score[col] += 1
            if np.all(sub[:, 0] == col):
                side_score[col] += 1
            if np.all(sub[:, -1] == col):
                side_score[col] += 1

    frame = max(freq.keys(), key=lambda c: (side_score.get(c, 0), freq[c]))

    # Build piece records.
    pieces = []
    for min_r, max_r, min_c, max_c in raw_pieces:
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        arr = grid[min_r:max_r + 1, min_c:max_c + 1].copy()
        arr[np.isin(arr, empty_colors)] = frame

        pieces.append({
            'h': h,
            'w': w,
            'arr': arr,
            'top': bool(np.all(arr[0, :] == frame)),
            'bottom': bool(np.all(arr[-1, :] == frame)),
            'left': bool(np.all(arr[:, 0] == frame)),
            'right': bool(np.all(arr[:, -1] == frame)),
            'in_r': min_r,
            'in_c': min_c,
            'cr': min_r + (h - 1) / 2.0,
            'cc': min_c + (w - 1) / 2.0,
        })

    n = len(pieces)

    # Single piece: keep it where it is.
    if n == 1:
        p = pieces[0]
        out = np.full((H_in, W_in), bg, dtype=int)
        r = max(0, min(p['in_r'], H_in - p['h']))
        c = max(0, min(p['in_c'], W_in - p['w']))
        out[r:r + p['h'], c:c + p['w']] = p['arr']
        return out

    # ------------------------------------------------------------------
    # Helper: quick necessary feasibility for a candidate final size.
    # ------------------------------------------------------------------
    def quick_ok(H, W):
        full_w = 0
        only_top_w = 0
        only_bottom_w = 0
        both_w = 0
        has_top = has_bottom = has_left = has_right = False

        for p in pieces:
            if p['h'] > H or p['w'] > W:
                return False
            if p['h'] == H:
                if not (p['top'] and p['bottom']):
                    return False
                full_w += p['w']
                has_top = has_bottom = True
            else:
                if not (p['top'] or p['bottom']):
                    return False
                if p['top'] and p['bottom']:
                    both_w += p['w']
                    has_top = has_bottom = True
                elif p['top']:
                    only_top_w += p['w']
                    has_top = True
                else:
                    only_bottom_w += p['w']
                    has_bottom = True
            if p['left']:
                has_left = True
            if p['right']:
                has_right = True

        if not (has_top and has_bottom and has_left and has_right):
            return False

        # Safe lower bound on width.
        nonfull_total = only_top_w + only_bottom_w + both_w
        w_lb = full_w + max(only_top_w, only_bottom_w, (nonfull_total + 1) // 2)
        return W >= w_lb

    # ------------------------------------------------------------------
    # Backtracking layout solver for a fixed H, W.
    # y is restricted to top (0), bottom (H-h), or full height (0 with h=H).
    # ------------------------------------------------------------------
    def find_layout(H, W, need_best=False, node_limit=20000):
        opts = []
        for p in pieces:
            if p['h'] > H or p['w'] > W:
                return None
            if p['h'] == H:
                if not (p['top'] and p['bottom']):
                    return None
                opts.append([('full', 0)])
            else:
                ls = []
                if p['top']:
                    ls.append(('top', 0))
                if p['bottom']:
                    ls.append(('bottom', H - p['h']))
                if not ls:
                    return None
                opts.append(ls)

        # Scale input centers to final local coordinates for tie-breaking cost.
        min_cr = min(p['cr'] for p in pieces)
        max_cr = max(p['cr'] for p in pieces)
        min_cc = min(p['cc'] for p in pieces)
        max_cc = max(p['cc'] for p in pieces)
        span_cr = max(1e-9, max_cr - min_cr)
        span_cc = max(1e-9, max_cc - min_cc)
        for p in pieces:
            p['sr'] = (p['cr'] - min_cr) / span_cr * H
            p['sc'] = (p['cc'] - min_cc) / span_cc * W

        top_occ = 0
        bottom_occ = 0
        placed_top = []
        placed_bottom = []
        placed = []
        unplaced = list(range(n))
        flags = [False, False, False, False]  # top, bottom, left, right
        nodes = 0
        found = False
        best = None
        best_cost = float('inf')

        def can_touch(side):
            for pid in unplaced:
                p = pieces[pid]
                for lane, _ in opts[pid]:
                    if side == 0 and lane in ('top', 'full'):
                        return True
                    if side == 1 and lane in ('bottom', 'full'):
                        return True
                    if side == 2 and p['left']:
                        return True
                    if side == 3 and p['right']:
                        return True
            return False

        def valid_placements(pid):
            p = pieces[pid]
            w = p['w']
            cand = {0, W - w}
            for (_, _, x2, _, w2, _) in placed:
                e1 = x2
                e2 = x2 + w2
                for e in (e1, e2):
                    if 0 <= e <= W - w:
                        cand.add(e)
                    if 0 <= e - w <= W - w:
                        cand.add(e - w)

            res = []
            for lane, y in opts[pid]:
                for x in cand:
                    if x == 0 and not p['left']:
                        continue
                    if x + w == W and not p['right']:
                        continue

                    bit = ((1 << w) - 1) << x
                    if lane in ('top', 'full') and (top_occ & bit):
                        continue
                    if lane in ('bottom', 'full') and (bottom_occ & bit):
                        continue

                    ok = True
                    if lane == 'top':
                        for x2, w2, h2 in placed_bottom:
                            if max(x, x2) < min(x + w, x2 + w2) and p['h'] + h2 > H:
                                ok = False
                                break
                    elif lane == 'bottom':
                        for x2, w2, h2 in placed_top:
                            if max(x, x2) < min(x + w, x2 + w2) and p['h'] + h2 > H:
                                ok = False
                                break
                    if not ok:
                        continue

                    cost = ((y + p['h'] / 2.0 - p['sr']) ** 2 +
                            (x + p['w'] / 2.0 - p['sc']) ** 2)
                    res.append((cost, lane, x, y, bit))
            return res

        def search(current_cost):
            nonlocal nodes, found, best, best_cost, top_occ, bottom_occ
            if found:
                return
            nodes += 1
            if nodes > node_limit:
                return

            if not unplaced:
                if all(flags):
                    best = [tuple(x) for x in placed]
                    best_cost = current_cost
                    if not need_best:
                        found = True
                return

            for s in range(4):
                if not flags[s] and not can_touch(s):
                    return

            # MRV: choose the unplaced piece with fewest valid placements.
            chosen = None
            chosen_pls = None
            min_cnt = 10**9
            for pid in unplaced:
                pls = valid_placements(pid)
                if not pls:
                    return
                if len(pls) < min_cnt:
                    min_cnt = len(pls)
                    chosen = pid
                    chosen_pls = pls
                    if min_cnt == 1:
                        break

            chosen_pls.sort(key=lambda t: t[0])

            for cost, lane, x, y, bit in chosen_pls:
                if need_best and best is not None and current_cost + cost >= best_cost:
                    continue

                old_top_occ = top_occ
                old_bottom_occ = bottom_occ
                old_flags = flags[:]

                if lane in ('top', 'full'):
                    top_occ |= bit
                if lane in ('bottom', 'full'):
                    bottom_occ |= bit

                if lane == 'top':
                    placed_top.append((x, pieces[chosen]['w'], pieces[chosen]['h']))
                elif lane == 'bottom':
                    placed_bottom.append((x, pieces[chosen]['w'], pieces[chosen]['h']))

                placed.append((chosen, lane, x, y, pieces[chosen]['w'], pieces[chosen]['h']))
                if x == 0:
                    flags[2] = True
                if x + pieces[chosen]['w'] == W:
                    flags[3] = True
                if y == 0:
                    flags[0] = True
                if y + pieces[chosen]['h'] == H:
                    flags[1] = True

                unplaced.remove(chosen)

                search(current_cost + cost)

                unplaced.append(chosen)
                placed.pop()
                if lane == 'top':
                    placed_top.pop()
                elif lane == 'bottom':
                    placed_bottom.pop()

                flags[:] = old_flags
                top_occ = old_top_occ
                bottom_occ = old_bottom_occ

                if found:
                    return

        search(0.0)
        if found or best is not None:
            return best
        return None

    # ------------------------------------------------------------------
    # Choose placement of the local object in the original canvas.
    # ------------------------------------------------------------------
    def choose_placement(H, W, placements):
        # placements: list of (pid, y, x)
        in_rs = [pieces[pid]['cr'] for pid, y, x in placements]
        in_cs = [pieces[pid]['cc'] for pid, y, x in placements]
        med_in_r = float(np.median(in_rs))
        med_in_c = float(np.median(in_cs))

        fin_rs = [y + pieces[pid]['h'] / 2.0 for pid, y, x in placements]
        fin_cs = [x + pieces[pid]['w'] / 2.0 for pid, y, x in placements]
        med_fin_r = float(np.median(fin_rs))
        med_fin_c = float(np.median(fin_cs))

        in_quad = []
        for i, (pid, y, x) in enumerate(placements):
            q = ('T' if in_rs[i] <= med_in_r else 'B') + ('L' if in_cs[i] <= med_in_c else 'R')
            in_quad.append(q)

        candidates = []
        for i, (pid, y, x) in enumerate(placements):
            fq = ('T' if fin_rs[i] <= med_fin_r else 'B') + ('L' if fin_cs[i] <= med_fin_c else 'R')
            if in_quad[i] == fq:
                R = pieces[pid]['in_r'] - y
                C = pieces[pid]['in_c'] - x
                if 0 <= R <= H_in - H and 0 <= C <= W_in - W:
                    disp = 0.0
                    for pid2, y2, x2 in placements:
                        fr = R + y2
                        fc = C + x2
                        disp += (pieces[pid2]['in_r'] - fr) ** 2 + (pieces[pid2]['in_c'] - fc) ** 2
                    candidates.append((disp, R, C))

        if candidates:
            _, R, C = min(candidates)
            return R, C

        R = (H_in - H) // 2
        C = (W_in - W) // 2
        R = max(0, min(R, max(0, H_in - H)))
        C = max(0, min(C, max(0, W_in - W)))
        return R, C

    # ------------------------------------------------------------------
    # Search candidate final sizes in increasing area.
    # ------------------------------------------------------------------
    max_h = max(p['h'] for p in pieces)
    max_w = max(p['w'] for p in pieces)
    sum_w = sum(p['w'] for p in pieces)

    top_h_max = max([p['h'] for p in pieces if p['top']], default=0)
    bottom_h_max = max([p['h'] for p in pieces if p['bottom']], default=0)
    H_upper = max(max_h, top_h_max + bottom_h_max)

    pairs = []
    for H in range(max_h, H_upper + 1):
        for W in range(max_w, sum_w + 1):
            if quick_ok(H, W):
                pairs.append((H * W, H, W))
    pairs.sort()

    best_area = None
    feasible_pairs = []

    i = 0
    while i < len(pairs):
        area = pairs[i][0]
        current = []
        while i < len(pairs) and pairs[i][0] == area:
            current.append((pairs[i][1], pairs[i][2]))
            i += 1

        for H, W in current:
            sol = find_layout(H, W, need_best=False, node_limit=20000)
            if sol is not None:
                best_area = area
                feasible_pairs.append((H, W))
        if best_area is not None:
            break

    # Fallback: if the restricted search failed, try a larger height range.
    if best_area is None:
        pairs = []
        H_upper2 = sum(p['h'] for p in pieces)
        for H in range(max_h, H_upper2 + 1):
            for W in range(max_w, sum_w + 1):
                if quick_ok(H, W):
                    pairs.append((H * W, H, W))
        pairs.sort()
        i = 0
        while i < len(pairs):
            area = pairs[i][0]
            current = []
            while i < len(pairs) and pairs[i][0] == area:
                current.append((pairs[i][1], pairs[i][2]))
                i += 1
            for H, W in current:
                sol = find_layout(H, W, need_best=False, node_limit=50000)
                if sol is not None:
                    best_area = area
                    feasible_pairs = [(H, W)]
            if best_area is not None:
                break

    if best_area is None:
        # Last resort: return background with 4 removed.
        out = grid.copy()
        out[out == 4] = bg
        return out

    # For all pairs of the minimal area, find the best local packing.
    best_final = None
    best_disp = float('inf')

    for H, W in feasible_pairs:
        sol = find_layout(H, W, need_best=True, node_limit=100000)
        if sol is None:
            continue
        placements = [(pid, y, x) for (pid, lane, x, y, _, _) in sol]
        R, C = choose_placement(H, W, placements)

        disp = 0.0
        for pid, y, x in placements:
            fr = R + y
            fc = C + x
            disp += (pieces[pid]['in_r'] - fr) ** 2 + (pieces[pid]['in_c'] - fc) ** 2

        if disp < best_disp:
            best_disp = disp
            best_final = (H, W, placements, R, C)

    H, W, placements, R, C = best_final

    # Build the local object, then paste into the original canvas.
    obj = np.full((H, W), frame, dtype=int)
    for pid, y, x in placements:
        p = pieces[pid]
        obj[y:y + p['h'], x:x + p['w']] = p['arr']

    out = np.full((H_in, W_in), bg, dtype=int)
    R = max(0, min(R, max(0, H_in - H)))
    C = max(0, min(C, max(0, W_in - W)))
    hh = min(H, H_in - R)
    ww = min(W, W_in - C)
    out[R:R + hh, C:C + ww] = obj[:hh, :ww]
    return out