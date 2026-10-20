import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered frame fragments into one framed rectangle.

    Observed rule from the examples:
      * The most frequent color is the background.  Color 4 is also treated as
        transparent/empty when it is not the main background.
      * Non-empty connected components are rectangular fragments.  Empty cells
        inside a fragment's bounding box are filled with the frame color.
      * The frame color is the dominant non-background color that forms complete
        sides of fragments.
      * Fragments are translated, not rotated, into the smallest rectangle that
        can contain them without overlap.  A fragment may touch an outer side of
        the final rectangle only if that side of the fragment is completely frame.
      * Missing cells inside the final rectangle are filled with the frame color.
      * Among possible packings/translations, the solution is chosen to preserve
        the relative input positions, with a strong preference for keeping a
        fragment in the same relative quadrant/corner.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H_in, W_in = grid.shape

    # Background and secondary empty color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    empty_set = {bg, 4}
    empty_arr = np.array(sorted(empty_set), dtype=int)

    non_bg_mask = ~np.isin(grid, empty_arr)
    if not non_bg_mask.any():
        return np.full((H_in, W_in), bg, dtype=int)

    # ------------------------------------------------------------------
    # Connected components of non-empty cells.
    # ------------------------------------------------------------------
    visited = np.zeros((H_in, W_in), dtype=bool)
    bboxes = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(H_in):
        for c in range(W_in):
            if non_bg_mask[r, c] and not visited[r, c]:
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
                            if non_bg_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                bboxes.append((min_r, max_r, min_c, max_c))

    if not bboxes:
        return np.full((H_in, W_in), bg, dtype=int)

    # ------------------------------------------------------------------
    # Determine frame color.
    # ------------------------------------------------------------------
    non_bg_cells = grid[non_bg_mask]
    fvals, fcounts = np.unique(non_bg_cells, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(fvals, fcounts)}

    side_score = {c: 0 for c in freq}
    for min_r, max_r, min_c, max_c in bboxes:
        sub = grid[min_r:max_r + 1, min_c:max_c + 1]
        sides = (sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1])
        flat = np.concatenate([s.ravel() for s in sides])
        side_colors = set(int(v) for v in np.unique(flat) if int(v) not in empty_set)
        for col in side_colors:
            for s in sides:
                if np.all((s == col) | np.isin(s, empty_arr)):
                    side_score[col] += 1

    frame = max(freq.keys(), key=lambda c: (side_score.get(c, 0), freq[c]))

    # ------------------------------------------------------------------
    # Build fragment records.
    # ------------------------------------------------------------------
    pieces = []
    for min_r, max_r, min_c, max_c in bboxes:
        arr = grid[min_r:max_r + 1, min_c:max_c + 1].copy()
        arr[np.isin(arr, empty_arr)] = frame

        # Pure frame rectangles are redundant because gaps are filled with frame.
        if np.all(arr == frame):
            continue

        h, w = arr.shape
        can_top = bool(np.all(arr[0, :] == frame))
        can_bottom = bool(np.all(arr[-1, :] == frame))
        can_left = bool(np.all(arr[:, 0] == frame))
        can_right = bool(np.all(arr[:, -1] == frame))

        pieces.append({
            'arr': arr,
            'h': h,
            'w': w,
            'area': h * w,
            'top': can_top,
            'bottom': can_bottom,
            'left': can_left,
            'right': can_right,
            'side_count': int(can_top) + int(can_bottom) + int(can_left) + int(can_right),
            'in_r': min_r,
            'in_c': min_c,
            'cr': min_r + (h - 1) / 2.0,
            'cc': min_c + (w - 1) / 2.0,
        })

    if not pieces:
        return np.full((H_in, W_in), bg, dtype=int)

    if len(pieces) == 1:
        p = pieces[0]
        out = np.full((H_in, W_in), bg, dtype=int)
        r = max(0, min(p['in_r'], H_in - p['h']))
        c = max(0, min(p['in_c'], W_in - p['w']))
        out[r:r + p['h'], c:c + p['w']] = p['arr']
        return out

    pieces.sort(key=lambda p: (-p['side_count'], -p['area'], p['in_r'], p['in_c']))
    n = len(pieces)

    max_h = max(p['h'] for p in pieces)
    max_w = max(p['w'] for p in pieces)
    sum_h = sum(p['h'] for p in pieces)
    sum_w = sum(p['w'] for p in pieces)
    sum_area = sum(p['area'] for p in pieces)

    # Individual edge-clearing lower bounds.
    H_low = max(p['h'] + (0 if p['top'] else 1) + (0 if p['bottom'] else 1) for p in pieces)
    W_low = max(p['w'] + (0 if p['left'] else 1) + (0 if p['right'] else 1) for p in pieces)
    H_low = max(H_low, max_h)
    W_low = max(W_low, max_w)

    H_upper = min(H_in, sum_h)
    W_upper = min(W_in, sum_w)

    if H_low > H_upper or W_low > W_upper:
        # Very defensive fallback: use whole grid.
        H_low, W_low = 1, 1
        H_upper, W_upper = H_in, W_in

    base_order = sorted(range(n), key=lambda i: (
        -pieces[i]['side_count'], -pieces[i]['area'], pieces[i]['in_r'], pieces[i]['in_c']
    ))
    area_order = sorted(range(n), key=lambda i: (
        -pieces[i]['area'], -pieces[i]['side_count'], pieces[i]['in_r'], pieces[i]['in_c']
    ))
    pos_order = sorted(range(n), key=lambda i: (pieces[i]['in_r'], pieces[i]['in_c']))
    greedy_orders = [base_order, area_order, pos_order]

    def prepare_scaled(H, W):
        min_cr = min(p['cr'] for p in pieces)
        max_cr = max(p['cr'] for p in pieces)
        min_cc = min(p['cc'] for p in pieces)
        max_cc = max(p['cc'] for p in pieces)
        span_cr = max(1e-9, max_cr - min_cr)
        span_cc = max(1e-9, max_cc - min_cc)
        for p in pieces:
            p['sr'] = (p['cr'] - min_cr) / span_cr * H
            p['sc'] = (p['cc'] - min_cc) / span_cc * W

    def layout_cost(placed):
        cost = 0.0
        for pid, x, y in placed:
            p = pieces[pid]
            dr = y + p['h'] / 2.0 - p['sr']
            dc = x + p['w'] / 2.0 - p['sc']
            cost += dr * dr + dc * dc
        return cost

    def quick_ok(H, W):
        if H * W < sum_area:
            return False
        for p in pieces:
            h, w = p['h'], p['w']
            if h > H or w > W:
                return False
            min_y = 0 if p['top'] else 1
            max_y = H - h if p['bottom'] else H - h - 1
            if min_y > max_y:
                return False
            min_x = 0 if p['left'] else 1
            max_x = W - w if p['right'] else W - w - 1
            if min_x > max_x:
                return False
        return True

    def place_occ(occ, x, y, w, h, positive=True):
        mask = ((1 << w) - 1) << x
        if positive:
            for yy in range(y, y + h):
                occ[yy] |= mask
        else:
            for yy in range(y, y + h):
                occ[yy] &= ~mask

    def overlaps(occ, x, y, w, h):
        mask = ((1 << w) - 1) << x
        for yy in range(y, y + h):
            if occ[yy] & mask:
                return True
        return False

    def generate_positions(pid, occ, placed, H, W, relaxed=False, limit=12):
        p = pieces[pid]
        h, w = p['h'], p['w']
        max_x = W - w
        max_y = H - h

        xset = set()
        yset = set()

        if p['left']:
            xset.add(0)
        if p['right']:
            xset.add(max_x)
        if p['top']:
            yset.add(0)
        if p['bottom']:
            yset.add(max_y)

        if not xset:
            if max_x <= 30 or relaxed:
                xset.update(range(max_x + 1))
            else:
                xset.update([max_x // 2, max_x // 3, (2 * max_x) // 3])

        if not yset:
            if max_y <= 30 or relaxed:
                yset.update(range(max_y + 1))
            else:
                yset.update([max_y // 2, max_y // 3, (2 * max_y) // 3])

        for qid, px, py in placed:
            q = pieces[qid]
            pw, ph = q['w'], q['h']

            for e in (px, px + pw):
                for v in (e, e - w):
                    if 0 <= v <= max_x:
                        xset.add(v)

            for e in (py, py + ph):
                for v in (e, e - h):
                    if 0 <= v <= max_y:
                        yset.add(v)

        def valid_x(v):
            return (0 <= v <= max_x and
                    not (v == 0 and not p['left']) and
                    not (v == max_x and not p['right']))

        def valid_y(v):
            return (0 <= v <= max_y and
                    not (v == 0 and not p['top']) and
                    not (v == max_y and not p['bottom']))

        xset = {v for v in xset if valid_x(v)}
        yset = {v for v in yset if valid_y(v)}

        if not xset or not yset:
            return []

        tx = p['sc']
        ty = p['sr']
        xlist = sorted(xset, key=lambda v: abs((v + w / 2.0) - tx))
        ylist = sorted(yset, key=lambda v: abs((v + h / 2.0) - ty))

        if not relaxed:
            xlist = xlist[:limit]
            ylist = ylist[:limit]
        else:
            xlist = xlist[:min(25, len(xlist))]
            ylist = ylist[:min(25, len(ylist))]

        res = []
        for x in xlist:
            for y in ylist:
                if overlaps(occ, x, y, w, h):
                    continue
                dr = y + h / 2.0 - p['sr']
                dc = x + w / 2.0 - p['sc']
                res.append((dr * dr + dc * dc, x, y))

        res.sort(key=lambda t: t[0])
        return res

    def greedy_pack(H, W, order):
        occ = [0] * H
        placed = []
        for pid in order:
            p = pieces[pid]
            cand = generate_positions(pid, occ, placed, H, W, relaxed=False, limit=12)
            if not cand:
                return None
            _, x, y = cand[0]
            place_occ(occ, x, y, p['w'], p['h'], True)
            placed.append((pid, x, y))
        return placed

    def dfs_pack(H, W, need_best=False, node_limit=20000, initial=None):
        best = initial
        best_cost = layout_cost(initial) if initial else float('inf')
        nodes = 0
        unplaced0 = base_order[:]

        def dfs(occ, placed, unplaced, cur_cost):
            nonlocal nodes, best, best_cost
            if nodes >= node_limit:
                return
            nodes += 1

            if need_best and cur_cost >= best_cost:
                return

            if not unplaced:
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best = placed[:]
                return

            chosen = -1
            chosen_cands = None
            min_len = 10 ** 9

            for pid in unplaced:
                cands = generate_positions(pid, occ, placed, H, W, relaxed=False, limit=12)
                if not cands:
                    return
                if len(cands) < min_len:
                    min_len = len(cands)
                    chosen = pid
                    chosen_cands = cands
                    if min_len == 1:
                        break

            for c, x, y in chosen_cands[:20]:
                if need_best and cur_cost + c >= best_cost:
                    continue

                p = pieces[chosen]
                place_occ(occ, x, y, p['w'], p['h'], True)
                idx = unplaced.index(chosen)
                unplaced.pop(idx)
                placed.append((chosen, x, y))

                dfs(occ, placed, unplaced, cur_cost + c)

                placed.pop()
                unplaced.append(chosen)
                place_occ(occ, x, y, p['w'], p['h'], False)

                if not need_best and best is not None:
                    return

        dfs([0] * H, [], unplaced0[:], 0.0)
        return best

    def find_layout(H, W, need_best=False, node_limit=20000):
        prepare_scaled(H, W)

        best_greedy = None
        best_greedy_cost = float('inf')
        for order in greedy_orders:
            cand = greedy_pack(H, W, order)
            if cand is not None:
                cst = layout_cost(cand)
                if cst < best_greedy_cost:
                    best_greedy_cost = cst
                    best_greedy = cand

        if best_greedy is not None and not need_best:
            return best_greedy

        initial = best_greedy
        if node_limit > 0:
            best_dfs = dfs_pack(H, W, need_best=True if initial is not None else False,
                                node_limit=node_limit, initial=initial)
            if best_dfs is not None:
                return best_dfs
        return best_greedy

    # ------------------------------------------------------------------
    # Search for the smallest feasible rectangle.
    # ------------------------------------------------------------------
    top_h = max([p['h'] for p in pieces if p['top']], default=0)
    bottom_h = max([p['h'] for p in pieces if p['bottom']], default=0)
    left_w = max([p['w'] for p in pieces if p['left']], default=0)
    right_w = max([p['w'] for p in pieces if p['right']], default=0)

    H_struct = max(H_low, max_h, top_h + bottom_h, max_h + 3)
    W_struct = max(W_low, max_w, left_w + right_w, max_w + 3)

    def search_phase(H_end, W_max, max_pairs, dfs_pairs):
        pairs = []
        for H in range(H_low, min(H_end, H_upper) + 1):
            minW = max(W_low, (sum_area + H - 1) // H)
            for W in range(minW, min(W_max, W_upper) + 1):
                if quick_ok(H, W):
                    pairs.append((H * W, H, W))

        pairs.sort()
        feasible = []
        i = 0
        tried = 0
        used_dfs = 0

        while i < len(pairs) and tried < max_pairs:
            area = pairs[i][0]
            group = []
            while i < len(pairs) and pairs[i][0] == area:
                group.append((pairs[i][1], pairs[i][2]))
                i += 1
            group.sort()

            for H, W in group:
                if tried >= max_pairs:
                    break
                tried += 1

                placed = find_layout(H, W, need_best=False, node_limit=0)
                if placed is None and used_dfs < dfs_pairs:
                    placed = find_layout(H, W, need_best=False, node_limit=6000)
                    used_dfs += 1

                if placed is not None:
                    feasible.append((H, W, placed))

            if feasible:
                break

        return feasible

    feasible = search_phase(H_struct, W_struct, max_pairs=35, dfs_pairs=6)
    if not feasible:
        feasible = search_phase(H_upper, W_upper, max_pairs=90, dfs_pairs=12)

    # ------------------------------------------------------------------
    # Choose global placement.  Prefer fixing a piece that is in the same
    # relative quadrant as in the input.
    # ------------------------------------------------------------------
    def choose_offset(H, W, placed):
        in_rs = [pieces[pid]['cr'] for pid, _, _ in placed]
        in_cs = [pieces[pid]['cc'] for pid, _, _ in placed]
        fin_rs = [y + pieces[pid]['h'] / 2.0 for pid, _, y in placed]
        fin_cs = [x + pieces[pid]['w'] / 2.0 for pid, x, _ in placed]

        med_in_r = float(np.median(in_rs))
        med_in_c = float(np.median(in_cs))
        med_fin_r = float(np.median(fin_rs))
        med_fin_c = float(np.median(fin_cs))

        def quadrant(r, c, mr, mc):
            return ('T' if r <= mr else 'B') + ('L' if c <= mc else 'R')

        candidates = []
        for i, (pid, x, y) in enumerate(placed):
            q_in = quadrant(in_rs[i], in_cs[i], med_in_r, med_in_c)
            q_fin = quadrant(fin_rs[i], fin_cs[i], med_fin_r, med_fin_c)
            if q_in == q_fin:
                R = pieces[pid]['in_r'] - y
                C = pieces[pid]['in_c'] - x
                if 0 <= R <= H_in - H and 0 <= C <= W_in - W:
                    disp = 0
                    for pid2, x2, y2 in placed:
                        p2 = pieces[pid2]
                        dr = R + y2 - p2['in_r']
                        dc = C + x2 - p2['in_c']
                        disp += dr * dr + dc * dc
                    candidates.append((disp, R, C))

        if candidates:
            disp, R, C = min(candidates)
            return disp, R, C, True

        # Fallback: minimize total translation over all valid offsets.
        best = None
        for R in range(max(0, H_in - H) + 1):
            for C in range(max(0, W_in - W) + 1):
                disp = 0
                for pid, x, y in placed:
                    p = pieces[pid]
                    dr = R + y - p['in_r']
                    dc = C + x - p['in_c']
                    disp += dr * dr + dc * dc
                if best is None or disp < best[0]:
                    best = (disp, R, C)

        if best is not None:
            return best[0], best[1], best[2], False

        R = max(0, (H_in - H) // 2)
        C = max(0, (W_in - W) // 2)
        return 0, R, C, False

    if feasible:
        final_candidates = []
        for H, W, placed in feasible[:6]:
            # Improve the local packing slightly.
            improved = find_layout(H, W, need_best=True, node_limit=30000)
            if improved is not None:
                placed = improved

            disp, R, C, matched = choose_offset(H, W, placed)
            local = layout_cost(placed)
            score = (0 if matched else 1, disp, local, H, W)
            final_candidates.append((score, H, W, placed, R, C))

        _, H, W, placed, R, C = min(final_candidates, key=lambda t: t[0])
    else:
        # Defensive fallback: pack into a larger rectangle if the main search
        # did not find anything within the time/size limits.
        H = min(H_in, max(H_low, sum_h))
        W = min(W_in, max(W_low, sum_w))
        H = max(H, H_low)
        W = max(W, W_low)

        placed = find_layout(H, W, need_best=False, node_limit=20000)
        if placed is None:
            # Last-resort shelf packing with a one-cell frame margin.
            H = min(H_in, max(H_low, sum_h + 2))
            W = min(W_in, max(W_low, sum_w + 2))
            prepare_scaled(H, W)
            occ = [0] * H
            placed = []
            x = 1
            y = 1
            shelf_h = 0

            for pid in base_order:
                p = pieces[pid]
                if x + p['w'] > W - 1:
                    x = 1
                    y += shelf_h + 1
                    shelf_h = 0

                if y + p['h'] > H - 1:
                    y = max(1, H - p['h'])
                    x = 1

                while True:
                    ok = not overlaps(occ, x, y, p['w'], p['h'])
                    if ok:
                        break
                    x += 1
                    if x + p['w'] > W - 1:
                        x = 1
                        y += p['h'] + 1

                place_occ(occ, x, y, p['w'], p['h'], True)
                placed.append((pid, x, y))
                shelf_h = max(shelf_h, p['h'])
                x += p['w'] + 1

        H = min(H, H_in)
        W = min(W, W_in)
        R = max(0, (H_in - H) // 2)
        C = max(0, (W_in - W) // 2)

    # ------------------------------------------------------------------
    # Build the final framed object and paste it into the original canvas.
    # ------------------------------------------------------------------
    obj = np.full((H, W), frame, dtype=int)
    for pid, x, y in placed:
        p = pieces[pid]
        if y + p['h'] > H or x + p['w'] > W:
            continue
        obj[y:y + p['h'], x:x + p['w']] = p['arr']

    out = np.full((H_in, W_in), bg, dtype=int)
    R = max(0, min(R, max(0, H_in - H)))
    C = max(0, min(C, max(0, W_in - W)))
    out[R:R + H, C:C + W] = obj
    return out