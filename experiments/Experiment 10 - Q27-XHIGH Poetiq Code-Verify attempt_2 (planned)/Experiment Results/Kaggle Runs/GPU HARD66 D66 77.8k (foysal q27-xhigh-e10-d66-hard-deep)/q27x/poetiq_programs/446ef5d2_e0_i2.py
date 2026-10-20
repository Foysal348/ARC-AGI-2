import numpy as np
from collections import Counter, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular pieces into one compact frame.

    The input is interpreted as containing several rectangular "pieces"
    scattered on a background color.  A few extraneous pixels may be attached
    to pieces; the algorithm extracts the largest clean non-background
    rectangle from each connected component.

    The transformation:
      1. Detect the background color.
      2. Extract one clean rectangular piece per connected non-background
         component.
      3. Infer the common frame color from full edges of the pieces.
      4. Find the smallest rectangle that can contain all pieces so that every
         piece touching the outer border has a full frame-color edge there.
         Empty cells in the assembled rectangle are filled with the frame
         color.
      5. Place the assembled rectangle in the original canvas so that the
         largest piece is centered when possible; otherwise center the whole
         assembled rectangle.  Ties are broken by minimizing movement from the
         original piece positions.
    """

    H, W = grid.shape
    if H == 0 or W == 0:
        return np.empty_like(grid)

    # Background is the most frequent color.
    bg = int(Counter(grid.ravel()).most_common(1)[0][0])
    nonbg = grid != bg

    # ---------------------------------------------------------------
    # Connected components of non-background cells (4-connectivity).
    # ---------------------------------------------------------------
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
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if 0 <= nr < H and 0 <= nc < W and nonbg[nr, nc] and comp_id[nr, nc] == -1:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))
                components.append(cells)
                cid += 1

    if not components:
        return np.full((H, W), bg, dtype=grid.dtype)

    # ---------------------------------------------------------------
    # For a component, find the largest all-non-background rectangle.
    # ---------------------------------------------------------------
    def largest_full_rect(cells):
        rmin = min(r for r, _ in cells)
        rmax = max(r for r, _ in cells)
        cmin = min(c for _, c in cells)
        cmax = max(c for _, c in cells)

        h0 = rmax - rmin + 1
        w0 = cmax - cmin + 1

        # Prefix sum of non-background cells in the component's bounding box.
        pref = np.zeros((h0 + 1, w0 + 1), dtype=int)
        for i in range(h0):
            row = nonbg[rmin + i, cmin:cmax + 1].astype(int)
            pref[i + 1, 1:w0 + 1] = pref[i, 1:w0 + 1] + row

        best = None
        # Try all rectangles.  Grids are small in ARC tasks.
        for r0 in range(h0):
            for r1 in range(r0, h0):
                h = r1 - r0 + 1
                if h < 2:
                    continue
                for c0 in range(w0):
                    for c1 in range(c0, w0):
                        w = c1 - c0 + 1
                        if w < 2:
                            continue
                        area = h * w
                        s = (pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] -
                             pref[r1 + 1, c0] + pref[r0, c0])
                        if s == area:
                            key = (area, h, w)
                            if best is None or key > best[0]:
                                best = (key, rmin + r0, rmin + r1, cmin + c0, cmin + c1)

        if best is None:
            return None
        _, r0, r1, c0, c1 = best
        return {
            'h': r1 - r0 + 1,
            'w': c1 - c0 + 1,
            'pos': (r0, c0),
            'pattern': grid[r0:r1 + 1, c0:c1 + 1].copy(),
            'full': set(),
        }

    raw_tiles = []
    for cells in components:
        t = largest_full_rect(cells)
        if t is not None:
            raw_tiles.append(t)

    # Ignore tiny accidental fragments if enough real tiles remain.
    tiles = [t for t in raw_tiles if t['h'] >= 2 and t['w'] >= 2 and t['h'] * t['w'] >= 4]
    if len(tiles) < 2:
        tiles = raw_tiles
    if not tiles:
        return np.full((H, W), bg, dtype=grid.dtype)

    # ---------------------------------------------------------------
    # Infer the frame color: the non-background color that appears most
    # often as a complete edge of a tile.
    # ---------------------------------------------------------------
    def infer_frame_color():
        scores = Counter()
        for t in tiles:
            p = t['pattern']
            colors = set(p.ravel().tolist())
            colors.discard(bg)
            for col in colors:
                if int(col) == bg:
                    continue
                if np.all(p[0, :] == col):
                    scores[int(col)] += 1
                if np.all(p[-1, :] == col):
                    scores[int(col)] += 1
                if np.all(p[:, 0] == col):
                    scores[int(col)] += 1
                if np.all(p[:, -1] == col):
                    scores[int(col)] += 1

        if scores:
            return int(scores.most_common(1)[0][0])

        # Fallback: second most common color in the whole grid.
        cnt = Counter(grid.ravel())
        cnt.pop(bg, None)
        if cnt:
            return int(cnt.most_common(1)[0][0])
        return bg

    frame = infer_frame_color()

    # Mark which edges of each tile are completely the frame color.
    def mark_full_edges():
        for t in tiles:
            p = t['pattern']
            full = set()
            if np.all(p[0, :] == frame):
                full.add('top')
            if np.all(p[-1, :] == frame):
                full.add('bottom')
            if np.all(p[:, 0] == frame):
                full.add('left')
            if np.all(p[:, -1] == frame):
                full.add('right')
            t['full'] = full

    mark_full_edges()

    # Prefer pieces that actually contribute a frame edge.  If this removes
    # everything, fall back to all extracted pieces.
    usable = [t for t in tiles if t['full']]
    if not usable:
        usable = tiles

    # ---------------------------------------------------------------
    # Packing search.
    #
    # For a candidate assembled rectangle h x w, every tile must be placed
    # inside it without overlap.  If a tile touches an outer side, the
    # corresponding edge of the tile must be a full frame-color edge.
    # Cells not covered by tiles are filled with the frame color.
    # ---------------------------------------------------------------

    def valid_position(tile, h, w, r, c):
        th, tw = tile['h'], tile['w']
        if r < 0 or c < 0 or r + th > h or c + tw > w:
            return False

        full = tile['full']

        # If a boundary side is used, it must be full frame color.
        if r == 0 and 'top' not in full:
            return False
        if r + th == h and 'bottom' not in full:
            return False
        if c == 0 and 'left' not in full:
            return False
        if c + tw == w and 'right' not in full:
            return False

        # Tiles with a full frame edge are expected to touch the border at
        # least on one of those edges.  Tiles with no full edge cannot touch
        # the border.
        touches = False
        if r == 0 and 'top' in full:
            touches = True
        if r + th == h and 'bottom' in full:
            touches = True
        if c == 0 and 'left' in full:
            touches = True
        if c + tw == w and 'right' in full:
            touches = True

        if full:
            return touches
        else:
            return not touches

    def make_mask(h, w, tile, r, c):
        th, tw = tile['h'], tile['w']
        m = 0
        for dr in range(th):
            base = (r + dr) * w
            for dc in range(tw):
                m |= 1 << (base + c + dc)
        return m

    def find_layout_for_size(h, w, limit=100):
        n = len(usable)
        if n == 0:
            return None

        total_area = sum(t['h'] * t['w'] for t in usable)
        if h * w < total_area:
            return None

        valid = []
        for t in usable:
            positions = []
            for r in range(h - t['h'] + 1):
                for c in range(w - t['w'] + 1):
                    if valid_position(t, h, w, r, c):
                        positions.append((r, c, make_mask(h, w, t, r, c)))
            if not positions:
                return None
            valid.append(positions)

        # Place constrained tiles first.
        order = sorted(range(n), key=lambda i: (len(valid[i]), -usable[i]['h'] * usable[i]['w']))

        all_mask = (1 << (h * w)) - 1
        pos = [None] * n
        best = {'score': None, 'layout': None}
        found = 0

        def score_layout(layout):
            # Determine final canvas offset.
            # Prefer centering the unique largest tile; otherwise center the
            # whole assembly.  Then choose the option that minimizes movement.
            areas = [usable[i]['h'] * usable[i]['w'] for i, _, _ in layout]
            max_area = max(areas)
            max_indices = [i for i, a in enumerate(areas) if a == max_area]

            if len(max_indices) == 1:
                anchor = max_indices[0]
                tile_i, r, c = layout[anchor]
                ah, aw = usable[tile_i]['h'], usable[tile_i]['w']
                r_opts = {(H - ah) // 2, (H - ah + 1) // 2}
                c_opts = {(W - aw) // 2, (W - aw + 1) // 2}
                cand = []
                for rr in r_opts:
                    for cc in c_opts:
                        R = rr - r
                        C = cc - c
                        if 0 <= R <= H - h and 0 <= C <= W - w:
                            cand.append((R, C))
            else:
                r_opts = {(H - h) // 2, (H - h + 1) // 2}
                c_opts = {(W - w) // 2, (W - w + 1) // 2}
                cand = []
                for R in r_opts:
                    for C in c_opts:
                        if 0 <= R <= H - h and 0 <= C <= W - w:
                            cand.append((R, C))

            # Fallback: best translation by median if centering is impossible.
            if not cand:
                rt = [usable[i]['pos'][0] - r for i, r, _ in layout]
                ct = [usable[i]['pos'][1] - c for i, _, c in layout]
                rt.sort()
                ct.sort()
                med_r = rt[len(rt) // 2]
                med_c = ct[len(ct) // 2]
                R = max(0, min(H - h, med_r))
                C = max(0, min(W - w, med_c))
                cand = [(R, C)]

            best_off = None
            best_score = None
            for R, C in cand:
                cost = 0
                fixed = 0
                moved = 0
                for tile_i, r, c in layout:
                    orow = R + r
                    ocol = C + c
                    ir, ic = usable[tile_i]['pos']
                    d = abs(ir - orow) + abs(ic - ocol)
                    cost += d
                    if d == 0:
                        fixed += 1
                    else:
                        moved += 1
                sc = (cost, moved, -fixed, R, C)
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_off = (R, C)

            return best_score, best_off

        def dfs(occupied, filler, touched, remaining_area, remaining_count):
            nonlocal found, best

            if found >= limit:
                return

            if remaining_count == 0:
                # All sides should be touched by at least one piece.
                if touched == 15:
                    layout = [(i, pos[i][0], pos[i][1]) for i in range(n)]
                    sc, off = score_layout(layout)
                    cur = (sc, 0)
                    if best['score'] is None or cur < (best['score'], 0):
                        best['score'] = sc
                        best['layout'] = layout
                        best['offset'] = off
                    found += 1
                return

            free = all_mask & ~(occupied | filler)
            if free == 0:
                return

            # Need enough free cells for all remaining tiles.
            if free.bit_count() < remaining_area:
                return

            first = (free & -free).bit_length() - 1

            # Try to cover the first free cell with an unused tile.
            tried = False
            for idx in order:
                if pos[idx] is not None:
                    continue
                tile = usable[idx]
                for r, c, m in valid[idx]:
                    if ((m >> first) & 1) == 0:
                        continue
                    if m & occupied:
                        continue
                    if m & filler:
                        continue

                    ntouched = touched
                    if r == 0:
                        ntouched |= 1
                    if r + tile['h'] == h:
                        ntouched |= 2
                    if c == 0:
                        ntouched |= 4
                    if c + tile['w'] == w:
                        ntouched |= 8

                    pos[idx] = (r, c)
                    dfs(occupied | m, filler, ntouched, remaining_area - tile['h'] * tile['w'], remaining_count - 1)
                    pos[idx] = None

                    if found >= limit:
                        return
                    tried = True

            # If no tile can cover this cell, make it frame filler.
            if not tried:
                filler_bit = 1 << first
                dfs(occupied, filler | filler_bit, touched, remaining_area, remaining_count)

        dfs(0, 0, 0, total_area, n)

        if best['layout'] is None:
            return None
        return best['layout'], best['offset']

    # Enumerate assembled rectangle sizes by increasing area.
    max_h = min(H, sum(t['h'] for t in usable))
    max_w = min(W, sum(t['w'] for t in usable))
    min_h = max(t['h'] for t in usable)
    min_w = max(t['w'] for t in usable)
    total_area = sum(t['h'] * t['w'] for t in usable)

    dims = []
    for h in range(min_h, max_h + 1):
        for w in range(min_w, max_w + 1):
            if h * w >= total_area:
                dims.append((h, w))
    dims.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))

    solution = None
    for h, w in dims:
        res = find_layout_for_size(h, w)
        if res is not None:
            solution = (h, w, res[0], res[1])
            break

    # ---------------------------------------------------------------
    # Fallbacks if the full packing search fails.
    # ---------------------------------------------------------------
    if solution is None:
        # Very conservative fallback: center the largest extracted piece.
        best_tile = max(tiles, key=lambda t: (t['h'] * t['w'], t['h'], t['w']))
        th, tw = best_tile['h'], best_tile['w']
        R = max(0, min(H - th, (H - th) // 2))
        C = max(0, min(W - tw, (W - tw) // 2))
        out = np.full((H, W), bg, dtype=grid.dtype)
        out[R:R + th, C:C + tw] = best_tile['pattern']
        return out

    h, w, layout, offset = solution
    R, C = offset

    # Build the assembled rectangle: frame color in unfilled cells.
    assembly = np.full((h, w), frame, dtype=grid.dtype)
    for tile_i, r, c in layout:
        t = usable[tile_i]
        assembly[r:r + t['h'], c:c + t['w']] = t['pattern']

    out = np.full((H, W), bg, dtype=grid.dtype)
    out[R:R + h, C:C + w] = assembly
    return out