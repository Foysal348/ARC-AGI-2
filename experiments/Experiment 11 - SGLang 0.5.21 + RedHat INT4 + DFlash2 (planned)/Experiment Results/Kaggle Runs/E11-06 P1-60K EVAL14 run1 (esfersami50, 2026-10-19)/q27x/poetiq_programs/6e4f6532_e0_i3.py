import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()

    # Identify frame bars (2-wide uniform rows/cols)
    frame = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        if np.array_equal(g[r, :], g[r + 1, :]) and len(np.unique(g[r, :])) == 1:
            frame[r:r + 2, :] = True
    for c in range(w - 1):
        if np.array_equal(g[:, c], g[:, c + 1]) and len(np.unique(g[:, c])) == 1:
            frame[:, c:c + 2] = True

    # Background
    nf = g[~frame]
    if nf.size > 0:
        vals, cnts = np.unique(nf, return_counts=True)
        bg = vals[np.argmax(cnts)]
    else:
        bg = 0

    # Frame color -> side mapping
    color_sides = {}
    top_c = g[0, 0] if (h >= 2 and np.array_equal(g[0, :], g[1, :]) and len(np.unique(g[0, :])) == 1) else None
    bot_c = g[h - 1, 0] if (h >= 2 and np.array_equal(g[h - 2, :], g[h - 1, :]) and len(np.unique(g[h - 2, :])) == 1) else None
    left_c = g[0, 0] if (w >= 2 and np.array_equal(g[:, 0], g[:, 1]) and len(np.unique(g[:, 0])) == 1) else None
    right_c = g[0, w - 1] if (w >= 2 and np.array_equal(g[:, w - 2], g[:, w - 1]) and len(np.unique(g[:, w - 2])) == 1) else None

    if top_c is not None:
        color_sides.setdefault(top_c, set()).add('U')
    if bot_c is not None:
        color_sides.setdefault(bot_c, set()).add('D')
    if left_c is not None:
        color_sides.setdefault(left_c, set()).add('L')
    if right_c is not None:
        color_sides.setdefault(right_c, set()).add('R')

    mid_vc, mid_hc = set(), set()
    for c in range(2, w - 1, 2):
        if np.array_equal(g[:, c], g[:, c + 1]) and len(np.unique(g[:, c])) == 1:
            cv = g[0, c]
            if cv != left_c and cv != right_c and cv != top_c and cv != bot_c:
                mid_vc.add(cv)
                color_sides.setdefault(cv, set()).add('M')
    for r in range(2, h - 1, 2):
        if np.array_equal(g[r, :], g[r + 1, :]) and len(np.unique(g[r, :])) == 1:
            rv = g[r, 0]
            if rv != left_c and rv != right_c and rv != top_c and rv != bot_c:
                mid_hc.add(rv)
                color_sides.setdefault(rv, set()).add('M')

    # Find components
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if not visited[i, j] and g[i, j] != bg and not frame[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and g[ny, nx] != bg and not frame[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if any(g[y, x] == 8 for y, x in cells):
                    comps.append(cells)

    if not comps:
        return out

    # Helper functions
    def extract(cells):
        ys = [y for y, _ in cells]
        xs = [x for _, x in cells]
        y0, x0 = min(ys), min(xs)
        y1, x1 = max(ys), max(xs)
        m = np.full((y1 - y0 + 1, x1 - x0 + 1), bg)
        for y, x in cells:
            m[y - y0, x - x0] = g[y, x]
        return m

    def border_tags(m):
        tags = {}
        hh, ww = m.shape
        for c in range(ww):
            if m[0, c] != bg: tags.setdefault(m[0, c], set()).add('U')
            if m[-1, c] != bg: tags.setdefault(m[-1, c], set()).add('D')
        for r in range(hh):
            if m[r, 0] != bg: tags.setdefault(m[r, 0], set()).add('L')
            if m[r, -1] != bg: tags.setdefault(m[r, -1], set()).add('R')
        return tags

    def dihedral(m):
        yield m
        yield np.rot90(m, 1)
        yield np.rot90(m, 2)
        yield np.rot90(m, 3)
        yield np.fliplr(m)
        yield np.flipud(m)
        yield np.rot90(np.fliplr(m), 1)
        yield np.rot90(np.fliplr(m), 2)

    def orient_score(m):
        tags = border_tags(m)
        score = 0
        for col, sds in tags.items():
            if col in color_sides:
                allowed = color_sides[col]
                if 'M' in allowed:
                    score += 2 * len(sds)
                else:
                    for s in sds:
                        score += 2 if s in allowed else -2
        return score

    # Orient each component
    oriented = []
    for cells in comps:
        m = extract(cells)
        best, bs = m, -999
        for tm in dihedral(m):
            s = orient_score(tm)
            if s > bs:
                bs, best = s, tm
        oriented.append(best)

    # Clear non-frame, non-bg
    for i in range(h):
        for j in range(w):
            if not frame[i, j] and g[i, j] != bg:
                out[i, j] = bg

    # Interior bounds
    top = 0
    while top < h and frame[top, 0]: top += 1
    bot = h
    while bot > 0 and frame[bot - 1, 0]: bot -= 1
    left = 0
    while left < w and frame[0, left]: left += 1
    right = w
    while right > 0 and frame[0, right - 1]: right -= 1

    # Find middle bar positions
    mid_v = None
    for c in range(left, right - 1, 2):
        if c + 1 < right and np.array_equal(g[top:bot, c], g[top:bot, c + 1]) and len(np.unique(g[top:bot, c])) == 1:
            cv = g[top, c]
            if cv in mid_vc:
                mid_v = c
                break
    mid_h = None
    for r in range(top, bot - 1, 2):
        if r + 1 < bot and np.array_equal(g[r, left:right], g[r + 1, left:right]) and len(np.unique(g[r, left:right])) == 1:
            rv = g[r, left]
            if rv in mid_hc:
                mid_h = r
                break

    # Assign each oriented component to a region and place it
    # Determine region for each based on primary tag (smallest frame color in border)
    def primary_region(m):
        tags = border_tags(m)
        best_c = None
        for c in sorted(tags.keys()):
            if c in color_sides:
                best_c = c
                break
        if best_c is None:
            return 'M'
        sds = color_sides[best_c]
        if 'L' in sds: return 'L'
        if 'R' in sds: return 'R'
        if 'U' in sds: return 'U'
        if 'D' in sds: return 'D'
        return 'M'

    regions = {}
    if mid_v is not None:
        regions['L'] = (top, left, bot, mid_v)
        regions['R'] = (top, mid_v + 2, bot, right)
        regions['M'] = (top, left, bot, right)
    elif mid_h is not None:
        regions['U'] = (top, left, mid_h, right)
        regions['D'] = (mid_h + 2, left, bot, right)
        regions['M'] = (top, left, bot, right)
    else:
        mr = (top + bot) // 2
        mc = (left + right) // 2
        regions['U'] = (top, left, mr, right)
        regions['D'] = (mr, left, bot, right)
        regions['L'] = (top, left, bot, mc)
        regions['R'] = (top, mc, bot, right)
        regions['M'] = (top, left, bot, right)

    # Sort by primary region for deterministic placement
    items = [(primary_region(m), m) for m in oriented]
    order = {'U': 0, 'L': 1, 'R': 2, 'D': 3, 'M': 4}
    items.sort(key=lambda x: order.get(x[0], 4))

    # If two objects go to same region, try to split: use secondary tag
    # Place all
    used_regions = {}
    for reg, m in items:
        mh, mw = m.shape
        if reg in regions:
            r0, c0, r1, c1 = regions[reg]
        else:
            r0, c0, r1, c1 = regions.get('M', (top, left, bot, right))

        # If region already used, try the other half
        key = (r0, c0, r1, c1)
        if key in used_regions:
            # Try opposite region
            if reg == 'L' and 'R' in regions:
                r0, c0, r1, c1 = regions['R']
            elif reg == 'R' and 'L' in regions:
                r0, c0, r1, c1 = regions['L']
            elif reg == 'U' and 'D' in regions:
                r0, c0, r1, c1 = regions['D']
            elif reg == 'D' and 'U' in regions:
                r0, c0, r1, c1 = regions['U']
            else:
                r0, c0, r1, c1 = regions.get('M', (top, left, bot, right))
        used_regions[(r0, c0, r1, c1)] = True

        ny = r0 + (r1 - r0 - mh) // 2
        nx = c0 + (c1 - c0 - mw) // 2
        ny = max(r0, min(ny, r1 - mh))
        nx = max(c0, min(nx, c1 - mw))

        for yy in range(mh):
            for xx in range(mw):
                v = m[yy, xx]
                if v == bg:
                    continue
                ay, ax = ny + yy, nx + xx
                if 0 <= ay < h and 0 <= ax < w and not frame[ay, ax]:
                    out[ay, ax] = v

    return out