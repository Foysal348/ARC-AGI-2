import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the ARC task by detecting local anchor/filler/extra component groups.

    The inferred rule:
    - Find connected non-background components.
    - An anchor is a component whose bounding-box complement has the same shape
      (up to grid symmetry) as another component.
    - That other component is the filler and is placed into the anchor complement.
    - A third component adjacent to the filler is the extra.
    - If the filler is placed by translation only, the extra is translated by the
      same delta.
    - If the filler was rotated/reflected, the extra is transformed by the same
      operation followed by a vertical flip, then placed adjacent to the anchor
      on the side suggested by its original relative position.
    - If the anchor complement is bottom-heavy, the two non-anchor colors are swapped.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    bg = int(np.bincount(grid.ravel()).argmax())
    out = grid.copy()

    def crop_mask(m):
        """Crop a boolean mask to its minimal non-empty bounding box."""
        if m.size == 0:
            return np.zeros((0, 0), dtype=bool)
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros((0, 0), dtype=bool)
        return m[np.ix_(rows, cols)]

    def apply_op(mask, rot, flip):
        """Apply rot (0-3) counterclockwise, optional horizontal flip, and crop."""
        if mask.size == 0:
            return np.zeros((0, 0), dtype=bool)
        m = np.rot90(mask, k=rot)
        if flip:
            m = np.fliplr(m)
        return crop_mask(m)

    def all_sims(mask):
        """Return all distinct cropped grid symmetries of a mask."""
        res = []
        seen = set()
        for rot in range(4):
            for flip in (False, True):
                sm = apply_op(mask, rot, flip)
                if sm.size == 0:
                    continue
                key = (sm.shape, sm.tobytes())
                if key not in seen:
                    seen.add(key)
                    res.append((sm, rot, flip))
        return res

    def get_components(g):
        """Extract 4-connected same-color non-background components."""
        visited = np.zeros((H, W), dtype=bool)
        comps = []
        for r0 in range(H):
            for c0 in range(W):
                if visited[r0, c0] or g[r0, c0] == bg:
                    continue
                color = g[r0, c0]
                stack = [(r0, c0)]
                visited[r0, c0] = True
                pix = []
                while stack:
                    r, c = stack.pop()
                    pix.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and g[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                rs = [p[0] for p in pix]
                cs = [p[1] for p in pix]
                rmin, rmax = min(rs), max(rs)
                cmin, cmax = min(cs), max(cs)
                mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=bool)
                for r, c in pix:
                    mask[r - rmin, c - cmin] = True
                comps.append({
                    'color': int(color),
                    'pixels': pix,
                    'r0': int(rmin),
                    'c0': int(cmin),
                    'h': int(rmax - rmin + 1),
                    'w': int(cmax - cmin + 1),
                    'mask': mask,
                    'area': len(pix),
                    'center_r': float(np.mean(rs)),
                    'center_c': float(np.mean(cs)),
                })
        return comps

    comps = get_components(grid)
    if not comps:
        return out

    def get_complement(comp):
        """Cropped complement inside the component bbox, plus offset inside bbox."""
        comp_mask = ~comp['mask']
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        off_r = int(np.argmax(rows))
        off_c = int(np.argmax(cols))
        cropped = comp_mask[np.ix_(rows, cols)]
        return cropped, off_r, off_c

    def is_bottom_heavy(comp_crop):
        """True if the cropped mask's mass is below its vertical center."""
        if comp_crop.size == 0 or comp_crop.sum() == 0:
            return False
        h = comp_crop.shape[0]
        row_counts = comp_crop.sum(axis=1)
        centroid_r = float(np.sum(np.arange(h) * row_counts)) / float(comp_crop.sum())
        return centroid_r > (h - 1) / 2.0

    def match_filler(filler, comp_crop):
        """Return a symmetry of filler matching comp_crop, or None."""
        for sm, rot, flip in all_sims(filler['mask']):
            if sm.shape == comp_crop.shape and np.array_equal(sm, comp_crop):
                return sm, rot, flip
        return None

    def bbox_touch(a, b):
        """Chebyshev gap <= 1 between bounding boxes."""
        dr = max(0, a['r0'] - (b['r0'] + b['h']), b['r0'] - (a['r0'] + a['h']))
        dc = max(0, a['c0'] - (b['c0'] + b['w']), b['c0'] - (a['c0'] + a['w']))
        return max(dr, dc) <= 1

    n = len(comps)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            adj[i][j] = bbox_touch(comps[i], comps[j])

    comp_infos = [get_complement(c) for c in comps]

    # Find all plausible (anchor, filler, extra) groups.
    candidates = []
    for fi in range(n):
        for ei in range(n):
            if fi == ei or not adj[fi][ei]:
                continue
            for ai in range(n):
                if ai == fi or ai == ei:
                    continue
                ci = comp_infos[ai]
                if ci is None:
                    continue
                comp_crop, off_r, off_c = ci
                if comp_crop.sum() != comps[fi]['area']:
                    continue
                match = match_filler(comps[fi], comp_crop)
                if match is None:
                    continue
                sym_mask, rot, flip = match

                score = comps[ai]['area'] * 10 + comps[fi]['area']
                if comps[ai]['area'] <= comps[fi]['area']:
                    score -= 20
                if adj[ai][fi] or adj[ai][ei]:
                    score -= 10
                if comps[ai]['color'] not in (comps[fi]['color'], comps[ei]['color']):
                    score += 2

                candidates.append({
                    'A': ai,
                    'F': fi,
                    'E': ei,
                    'sym_mask': sym_mask,
                    'rot': rot,
                    'flip': flip,
                    'off_r': off_r,
                    'off_c': off_c,
                    'score': score,
                })

    candidates.sort(key=lambda x: -x['score'])
    used = set()
    groups = []
    for g in candidates:
        ids = {g['A'], g['F'], g['E']}
        if ids & used:
            continue
        used |= ids
        groups.append(g)

    def feature_line(mask, axis):
        """
        Choose a feature line in the anchor mask: a partial line with the fewest cells.
        Fallback to the middle if none exists.
        """
        if axis == 'row':
            counts = mask.sum(axis=1)
            full = mask.shape[1]
            n_lines = mask.shape[0]
        else:
            counts = mask.sum(axis=0)
            full = mask.shape[0]
            n_lines = mask.shape[1]

        partial = np.where((counts > 0) & (counts < full))[0]
        if len(partial) > 0:
            min_count = counts[partial].min()
            idxs = partial[counts[partial] == min_count]
            return int(idxs[0])
        return n_lines // 2

    def dense_line(mask, axis, prefer=None):
        """Choose the densest line in a mask, resolving ties by preference."""
        if axis == 'row':
            counts = mask.sum(axis=1)
        else:
            counts = mask.sum(axis=0)
        maxc = counts.max()
        idxs = np.where(counts == maxc)[0]
        if len(idxs) == 1:
            return int(idxs[0])
        if axis == 'row':
            if prefer == 'up':
                return int(idxs[0])
            if prefer == 'down':
                return int(idxs[-1])
        else:
            if prefer == 'left':
                return int(idxs[0])
            if prefer == 'right':
                return int(idxs[-1])
        return int(idxs[0])

    def put(mask, r, c, color):
        """Place mask at (r,c) if inside grid and not overlapping non-background."""
        mh, mw = mask.shape
        if r < 0 or c < 0 or r + mh > H or c + mw > W:
            return False
        region = out[r:r + mh, c:c + mw]
        if np.any(mask & (region != bg)):
            return False
        region[mask] = color
        return True

    def side_positions(e_mask, anchor, extra, side):
        """
        Generate candidate positions for e_mask on a given side of the anchor bbox,
        sorted by compactness, secondary-direction agreement, and dense-line alignment.
        """
        eh, ew = e_mask.shape
        pos = []

        if side in ('right', 'left'):
            c = anchor['c0'] + anchor['w'] if side == 'right' else anchor['c0'] - ew
            if c < 0 or c + ew > W:
                return
            feat = feature_line(anchor['mask'], 'row')
            desired_dense = anchor['r0'] + feat
            dr = extra['center_r'] - anchor['center_r']
            prefer = 'up' if dr < 0 else ('down' if dr > 0 else None)
            dense = dense_line(e_mask, 'row', prefer)

            min_t = anchor['r0'] - eh + 1
            max_t = anchor['r0'] + anchor['h'] - 1
            for t in range(max(0, min_t), min(H, max_t + 1)):
                if t + eh > H:
                    continue
                # Need vertical overlap with anchor for side contact.
                if not (t < anchor['r0'] + anchor['h'] and t + eh > anchor['r0']):
                    continue
                rmin = min(t, anchor['r0'])
                rmax = max(t + eh, anchor['r0'] + anchor['h'])
                width = max(c + ew, anchor['c0'] + anchor['w']) - min(c, anchor['c0'])
                area = (rmax - rmin) * width
                dist = abs((t + dense) - desired_dense)
                sec = 0
                if dr < 0 and (t + eh / 2.0) > anchor['center_r']:
                    sec = 1
                if dr > 0 and (t + eh / 2.0) < anchor['center_r']:
                    sec = 1
                pos.append(((area, sec, dist), t, c))
        else:
            r = anchor['r0'] + anchor['h'] if side == 'down' else anchor['r0'] - eh
            if r < 0 or r + eh > H:
                return
            feat = feature_line(anchor['mask'], 'col')
            desired_dense = anchor['c0'] + feat
            dc = extra['center_c'] - anchor['center_c']
            prefer = 'left' if dc < 0 else ('right' if dc > 0 else None)
            dense = dense_line(e_mask, 'col', prefer)

            min_c = anchor['c0'] - ew + 1
            max_c = anchor['c0'] + anchor['w'] - 1
            for c in range(max(0, min_c), min(W, max_c + 1)):
                if c + ew > W:
                    continue
                # Need horizontal overlap with anchor for side contact.
                if not (c < anchor['c0'] + anchor['w'] and c + ew > anchor['c0']):
                    continue
                cmin = min(c, anchor['c0'])
                cmax = max(c + ew, anchor['c0'] + anchor['w'])
                height = max(r + eh, anchor['r0'] + anchor['h']) - min(r, anchor['r0'])
                area = height * (cmax - cmin)
                dist = abs((c + dense) - desired_dense)
                sec = 0
                if dc < 0 and (c + ew / 2.0) > anchor['center_c']:
                    sec = 1
                if dc > 0 and (c + ew / 2.0) < anchor['center_c']:
                    sec = 1
                pos.append(((area, sec, dist), r, c))

        pos.sort(key=lambda x: x[0])
        for _, r, c in pos:
            yield r, c

    def attempt_placement(e_mask, anchor, extra, color):
        """Try to place e_mask on the best side of anchor."""
        dr = extra['center_r'] - anchor['center_r']
        dc = extra['center_c'] - anchor['center_c']
        sides = []
        if dc > 0:
            sides.append(('right', abs(dc)))
        if dc < 0:
            sides.append(('left', abs(dc)))
        if dr > 0:
            sides.append(('down', abs(dr)))
        if dr < 0:
            sides.append(('up', abs(dr)))
        if not sides:
            sides = [('right', 0), ('left', 0), ('down', 0), ('up', 0)]
        sides.sort(key=lambda x: -x[1])

        for side, _ in sides:
            for r, c in side_positions(e_mask, anchor, extra, side):
                if put(e_mask, r, c, color):
                    return True
        return False

    # Apply selected groups.
    for g in groups:
        A = comps[g['A']]
        F = comps[g['F']]
        E = comps[g['E']]

        comp_crop = get_complement(A)[0]
        swap = is_bottom_heavy(comp_crop)

        # Remove moving pieces from their original locations.
        for r, c in F['pixels']:
            out[r, c] = bg
        for r, c in E['pixels']:
            out[r, c] = bg

        # Place filler into anchor complement.
        fill_color = E['color'] if swap else F['color']
        fr = A['r0'] + g['off_r']
        fc = A['c0'] + g['off_c']
        put(g['sym_mask'], fr, fc, fill_color)

        # Place extra piece.
        extra_color = F['color'] if swap else E['color']
        identity = (g['rot'] == 0 and not g['flip'])

        placed = False
        if identity:
            delta_r = fr - F['r0']
            delta_c = fc - F['c0']
            r = E['r0'] + delta_r
            c = E['c0'] + delta_c
            if put(E['mask'], r, c, extra_color):
                placed = True
            else:
                if attempt_placement(E['mask'], A, E, extra_color):
                    placed = True
        else:
            base = apply_op(E['mask'], g['rot'], g['flip'])
            e_mask = crop_mask(np.flipud(base))
            if e_mask.size == 0:
                e_mask = E['mask']
            if attempt_placement(e_mask, A, E, extra_color):
                placed = True
            else:
                # Robust fallback: try all symmetries of the extra piece.
                for sm, _, _ in all_sims(E['mask']):
                    if attempt_placement(sm, A, E, extra_color):
                        placed = True
                        break

        # If placement still failed, leave the extra piece in its original place
        # as a graceful fallback.
        if not placed:
            for r, c in E['pixels']:
                out[r, c] = E['color']

    return out