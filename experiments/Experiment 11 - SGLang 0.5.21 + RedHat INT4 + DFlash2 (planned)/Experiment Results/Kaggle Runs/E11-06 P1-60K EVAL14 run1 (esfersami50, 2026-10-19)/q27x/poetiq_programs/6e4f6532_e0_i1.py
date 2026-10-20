import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    h, w = g.shape
    if h < 2 or w < 2:
        return g.copy()

    out = g.copy()

    # Background is the most frequent color that is not part of frame bars.
    # Frame bars are uniform 2-pixel-wide full rows/columns.
    def is_bar_row(r, c):
        return all(g[r, cc] == c for cc in range(w)) and all(g[r + 1, cc] == c for cc in range(w)) if r + 1 < h else False

    def is_bar_col(c, col):
        return all(g[rr, c] == col for rr in range(h)) and all(g[rr, c + 1] == col for rr in range(h)) if c + 1 < w else False

    frame = np.zeros_like(g, dtype=bool)
    for r in range(0, h - 1, 2):
        if is_bar_row(r, g[r, 0]):
            frame[r:r + 2, :] = True
    for c in range(0, w - 1, 2):
        if is_bar_col(c, g[0, c]):
            frame[:, c:c + 2] = True

    # If the frame detection leaves most cells unframed, use the border color set.
    if frame.sum() == 0:
        # Fallback: assume 2-wide borders on all sides if present.
        if h >= 4 and w >= 4:
            if all(g[0, :] == g[1, :]) and all(g[-2, :] == g[-1, :]):
                frame[0:2, :] = True
                frame[-2:, :] = True
            if all(g[:, 0] == g[:, 1]) and all(g[:, -2] == g[:, -1]):
                frame[:, 0:2] = True
                frame[:, -2:] = True

    bg = 0
    if not frame.any():
        vals, counts = np.unique(g, return_counts=True)
        bg = vals[np.argmax(counts)]
    else:
        vals, counts = np.unique(g[~frame], return_counts=True)
        bg = vals[np.argmax(counts)] if len(vals) else 0

    # Border color -> global direction(s).
    color_dirs = {}
    for c in range(0, w - 1, 2):
        if is_bar_col(c, g[0, c]):
            col = g[0, c]
            color_dirs.setdefault(col, set()).add('L' if c <= w // 4 else 'R' if c >= 3 * w // 4 else 'M')
    for r in range(0, h - 1, 2):
        if is_bar_row(r, g[r, 0]):
            col = g[r, 0]
            color_dirs.setdefault(col, set()).add('U' if r <= h // 4 else 'D' if r >= 3 * h // 4 else 'M')

    # Find components of non-background, non-frame cells that contain 8.
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if not seen[i, j] and g[i, j] != bg and not frame[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != bg and not frame[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if any(g[y, x] == 8 for y, x in cells):
                    comps.append(cells)

    if not comps:
        return out

    def bbox(cells):
        ys = [y for y, x in cells]
        xs = [x for y, x in cells]
        return min(ys), min(xs), max(ys), max(xs)

    def extract(cells):
        y0, x0, y1, x1 = bbox(cells)
        sub = np.full((y1 - y0 + 1, x1 - x0 + 1), bg, dtype=g.dtype)
        for y, x in cells:
            sub[y - y0, x - x0] = g[y, x]
        return sub

    def side_tags(mat):
        tags = {}
        nonbg = mat != bg
        if not nonbg.any():
            return tags
        top = np.where(nonbg[0])[0]
        bottom = np.where(nonbg[-1])[0]
        left = np.where(nonbg[:, 0])[0]
        right = np.where(nonbg[:, -1])[0]
        for x in top:
            c = mat[0, x]
            if c != bg:
                tags.setdefault(c, set()).add('U')
        for x in bottom:
            c = mat[-1, x]
            if c != bg:
                tags.setdefault(c, set()).add('D')
        for y in left:
            c = mat[y, 0]
            if c != bg:
                tags.setdefault(c, set()).add('L')
        for y in right:
            c = mat[y, -1]
            if c != bg:
                tags.setdefault(c, set()).add('R')
        return tags

    def rot90(mat, k):
        m = np.array(mat)
        for _ in range(k % 4):
            m = np.rot90(m, 1)
        return m

    def reflect_v(mat):
        return np.flipud(np.array(mat))

    def reflect_h(mat):
        return np.fliplr(np.array(mat))

    def side_of_color_after(mat, color):
        res = set()
        nonbg = mat != bg
        if not nonbg.any():
            return res
        top = np.where(nonbg[0])[0]
        bottom = np.where(nonbg[-1])[0]
        left = np.where(nonbg[:, 0])[0]
        right = np.where(nonbg[:, -1])[0]
        for x in top:
            if mat[0, x] == color:
                res.add('U')
        for x in bottom:
            if mat[-1, x] == color:
                res.add('D')
        for y in left:
            if mat[y, 0] == color:
                res.add('L')
        for y in right:
            if mat[y, -1] == color:
                res.add('R')
        return res

    def transformations():
        yield 0, lambda m: m
        yield 1, lambda m: rot90(m, 1)
        yield 2, lambda m: rot90(m, 2)
        yield 3, lambda m: rot90(m, 3)
        yield 4, lambda m: reflect_v(m)
        yield 5, lambda m: reflect_h(m)
        yield 6, lambda m: reflect_v(reflect_h(m))
        yield 7, lambda m: rot90(reflect_v(m), 1)

    def target_dirs_for_component(cells):
        tags = side_tags(extract(cells))
        y0, x0, y1, x1 = bbox(cells)
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        target = {}
        for c, dirs in tags.items():
            cdirs = color_dirs.get(c, set())
            # Prefer outer directions matching the component's relative position.
            cand = []
            for d in dirs:
                if d in cdirs:
                    cand.append(d)
            if not cand:
                # If only middle divider color, choose side away from it based on component side.
                if 'M' in cdirs:
                    # Determine if component is left/right or up/down of the middle bar of that color.
                    # Use simple heuristic: choose direction toward nearest same-color frame segment if possible.
                    pass
                continue
            target[c] = cand[0]
        return target

    # In both provided examples, components are transformed independently and
    # placed into slots by a simple deterministic order:
    # - components with an outer frame color keep that outer side.
    # - otherwise they are placed by size/order.
    # The exact challenge grids follow the same component transformation rule:
    # rotate/flip so each frame-colored tag points to its frame side.
    processed = []
    for cells in comps:
        mat = extract(cells)
        tags = side_tags(mat)
        if not tags:
            processed.append((mat, 0, 0))
            continue
        # Find a transformation that makes every tag color point to a matching frame direction.
        best = None
        for idx, tf in transformations():
            tmat = tf(mat)
            ok = True
            for c, _ in tags.items():
                td = side_of_color_after(tmat, c)
                fd = color_dirs.get(c, set())
                # A middle divider color can be accepted on either side.
                if not fd:
                    continue
                if not (td & fd or 'M' in fd):
                    ok = False
                    break
            if ok:
                best = (idx, tmat)
                break
        if best is None:
            # Fallback: vertical flip if top/bottom tags exist, otherwise identity.
            best = (0, mat)
        processed.append((best[1], best[0], cells))

    # Place components back.  For the training pattern, the slot is chosen by
    # the original component position in the frame-defined area, then transformed.
    # A simple exact reproduction is to clear component cells and repaint each
    # transformed component at its original bounding-box top-left, adjusted to
    # stay inside the frame.
    #
    # This is the rule implied by the examples: the object geometry is the only
    # thing changed, and its anchor is the top-left of the original component.
    for y, x in [c for c in comps for c in [tuple(c)] if False]:
        pass

    # Clear all component cells first.
    for y in range(h):
        for x in range(w):
            if not frame[y, x] and g[y, x] != bg:
                out[y, x] = bg

    for tmat, idx, cells in processed:
        y0, x0, y1, x1 = bbox(cells)
        th, tw = tmat.shape
        # Keep same top-left anchor when possible.
        ny, nx = y0, x0
        if ny + th >= h or nx + tw >= w:
            ny = max(0, min(h - th, y0))
            nx = max(0, min(w - tw, x0))
        # Do not overwrite frame.
        for yy in range(th):
            for xx in range(tw):
                val = tmat[yy, xx]
                if val == bg:
                    continue
                ay, ax = ny + yy, nx + xx
                if 0 <= ay < h and 0 <= ax < w and not frame[ay, ax]:
                    out[ay, ax] = val

    return out