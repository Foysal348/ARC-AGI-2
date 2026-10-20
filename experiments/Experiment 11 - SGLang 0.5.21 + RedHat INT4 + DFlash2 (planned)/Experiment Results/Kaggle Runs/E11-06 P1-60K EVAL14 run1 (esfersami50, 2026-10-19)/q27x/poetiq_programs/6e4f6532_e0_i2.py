import numpy as np

def transform(grid: np.ndarray) -> np.ndarray):
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()

    # Identify frame bars
    frame = np.zeros((h, w), dtype=bool)
    # Full rows
    for r in range(0, h - 1):
        if np.all(g[r, :] == g[r + 1, :]) and np.all(g[r, :]):
            if len(np.unique(g[r, :])) == 1:
                frame[r:r + 2, :] = True
    # Full columns
    for c in range(0, w - 1):
        if len(np.unique(g[:, c])) == 1 and len(np.unique(g[:, c + 1])) == 1:
            if np.all(g[:, c] == g[:, c + 1]):
                frame[:, c:c + 2] = True

    # Background
    nf = ~frame
    if nf.any():
        vals, cnts = np.unique(g[nf], return_counts=True)
        bg = vals[np.argmax(cnts)]
    else:
        bg = 0

    # Map frame color -> direction(s)
    color_dir = {}
    if h >= 2 and np.all(g[0, :] == g[1, :]):
        color_dir.setdefault(g[0, 0], set()).add('U')
    if h >= 2 and np.all(g[h-2, :] == g[h-1, :]):
        color_dir.setdefault(g[h-1, 0], set()).add('D')
    if w >= 2 and np.all(g[:, 0] == g[:, 1]):
        color_dir.setdefault(g[0, 0], set()).add('L')
    if w >= 2 and np.all(g[:, w-2] == g[:, w-1]):
        color_dir.setdefault(g[0, w-1], set()).add('R')
    # Middle vertical bars
    for c in range(2, w - 1, 2):
        if len(np.unique(g[:, c])) == 1 and len(np.unique(g[:, c+1])) == 1:
            if np.all(g[:, c] == g[:, c+1]):
                cv = g[0, c]
                if cv not in color_dir:
                    color_dir[cv] = {'M'}
    # Middle horizontal bars
    for r in range(2, h - 1, 2):
        if len(np.unique(g[r, :])) == 1 and len(np.unique(g[r+1, :])) == 1:
            if np.all(g[r, :] == g[r+1, :]):
                rv = g[r, 0]
                if rv not in color_dir:
                    color_dir[rv] = {'M'}

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
                    for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny,nx] and g[ny,nx] != bg and not frame[ny,nx]:
                            visited[ny,nx] = True
                            stack.append((ny,nx))
                comps.append(cells)

    # Keep only ship components (contain 8)
    ships = [c for c in comps if any(g[y,x] == 8 for y,x in c)]

    # Clear all non-frame, non-bg cells
    for i in range(h):
        for j in range(w):
            if not frame[i,j] and g[i,j] != bg:
                out[i,j] = bg

    if not ships:
        return out

    def extract(cells):
        ys = [y for y,_ in cells]
        xs = [x for x,_ in [(y,x) for y,x in cells] for x in [x]]
        y0, y1 = min(ys), max(ys)
        x0, x1 = min(xs), max(xs)
        m = np.full((y1-y0+1, x1-x0+1), bg)
        for y, x in cells:
            m[y-y0, x-x0] = g[y, x]
        return m

    def get_tags(m):
        """Return dict: color -> set of sides where that color appears on the border of m"""
        tags = {}
        hh, ww = m.shape
        for c in range(ww):
            if m[0, c] != bg:
                tags.setdefault(m[0,c], set()).add('U')
            if m[-1, c] != bg:
                tags.setdefault(m[-1,c], set()).add('D')
        for r in range(hh):
            if m[r, 0] != bg:
                tags.setdefault(m[r,0], set()).add('L')
            if m[r, -1] != bg:
                tags.setdefault(m[r,-1], set()).add('R')
        return tags

    def all_transforms(m):
        """Yield (transformed_matrix, description) for all 8 dihedral transforms"""
        yield m, 0
        yield np.rot90(m, -1), 1   # 90 CW
        yield np.rot90(m, -2), 2    # 180
        yield np.rot90(m, -3), 3    # 90 CCW
        yield np.flipud(m), 4
        yield np.fliplr(m), 5
        yield np.flipud(np.fliplr(m)), 6
        yield np.rot90(np.flipud(m), -1), 7

    def tag_sides(m):
        """For each non-bg color on the border, return which sides it appears on"""
        hh, ww = m.shape
        res = {}
        for c in range(ww):
            v = m[0, c]
            if v != bg: res.setdefault(v, set()).add('U')
            v = m[-1, c]
            if v != bg: res.setdefault(v, set()).add('D')
        for r in range(hh):
            v = m[r, 0]
            if v != bg: res.setdefault(v, set()).add('L')
            v = m[r, -1]
            if v != bg: res.setdefault(v, set()).add('R')
        return res

    def is_valid_orient(m):
        """Check if all frame-colored tags are on the correct side"""
        sides = tag_sides(m)
        for color, sds in sides.items():
            if color in color_dir:
                allowed = color_dir[color]
                if 'M' in allowed:
                    continue  # middle bar: accept either side
                # For non-middle, the tag side must match the frame direction
                for sd in sds:
                    if sd not in allowed and 'M' not in allowed:
                        # Check if this color is actually a frame color that should be on a specific side
                        if sd not in allowed:
                            return False
        return True

    # For each ship, find the correct orientation
    oriented = []
    for cells in ships:
        m = extract(cells)
        best = None
        for tm, idx in all_transforms(m):
            if is_valid_orient(tm):
                best = tm
                break
        if best is None:
            best = m  # fallback: keep original
        oriented.append(best)

    # Determine placement region for each ship based on primary tag (smallest frame color)
    def primary_tag(m):
        sides = tag_sides(m)
        for color in sorted(sides.keys()):
            if color in color_dir:
                return color
        return None

    def dir_for_color(c):
        if c not in color_dir:
            return None
        d = color_dir[c]
        if 'U' in d: return 'U'
        if 'D' in d: return 'D'
        if 'L' in d: return 'L'
        if 'R' in d: return 'R'
        if 'M' in d: return 'M'
        return None

    # Determine regions
    # Identify frame boundaries
    top_bar = 0
    while top_bar < h and frame[top_bar, 0]:
        top_bar += 1
    bot_bar = h
    while bot_bar > 0 and frame[bot_bar-1, 0]:
        bot_bar -= 1
    left_bar = 0
    while left_bar < w and frame[0, left_bar]:
        left_bar += 1
    right_bar = w
    while right_bar > 0 and frame[0, right_bar-1]:
        right_bar -= 1

    # Find middle vertical bars
    mid_vcols = []
    for c in range(left_bar, right_bar - 1, 2):
        if frame[top_bar, c] and len(np.unique(g[:, c])) == 1 and len(np.unique(g[:, c+1])) == 1:
            if np.all(g[:, c] == g[:, c+1]):
                cv = g[0, c]
                if cv != g[0, 0] and cv != g[0, w-1]:
                    mid_vcols.append(c)

    # Find middle horizontal bars
    mid_hrows = []
    for r in range(top_bar, bot_bar - 1, 2):
        if frame[r, left_bar] and len(np.unique(g[r, :])) == 1 and len(np.unique(g[r+1, :])) == 1:
            if np.all(g[r, :] == g[r+1, :]):
                rv = g[r, 0]
                if rv != g[0, 0] and rv != g[h-1, 0]:
                    mid_hrows.append(r)

    # Sort ships by primary tag color and place
    ship_info = []
    for i, m in enumerate(oriented):
        pt = primary_tag(m)
        d = dir_for_color(pt) if pt is not None else None
        ship_info.append((pt if pt is not None else 999, d, m))
    ship_info.sort(key=lambda x: x[0])

    # Determine regions for placement
    if mid_vcols:
        # Left and right regions
        regions = {}
        regions['L'] = (top_bar, left_bar, bot_bar, mid_vcols[0])
        regions['R'] = (top_bar, mid_vcols[0] + 2, bot_bar, right_bar)
        regions['M'] = (top_bar, left_bar, bot_bar, right_bar)
    elif mid_hrows:
        regions = {}
        regions['U'] = (top_bar, left_bar, mid_hrows[0], right_bar)
        regions['D'] = (mid_hrows[0] + 2, left_bar, bot_bar, right_bar)
        regions['M'] = (top_bar, left_bar, bot_bar, right_bar)
    else:
        regions = {}
        mid_r = (top_bar + bot_bar) // 2
        mid_c = (left_bar + right_bar) // 2
        regions['U'] = (top_bar, left_bar, mid_r, right_bar)
        regions['D'] = (mid_r, left_bar, bot_bar, right_bar)
        regions['L'] = (top_bar, left_bar, bot_bar, mid_c)
        regions['R'] = (top_bar, mid_c, bot_bar, right_bar)
        regions['M'] = (top_bar, left_bar, bot_bar, right_bar)

    # Place ships
    for pt, d, m in ship_info:
        mh, mw = m.shape
        if d in regions:
            r0, c0, r1, c1 = regions[d]
        elif d is None:
            r0, c0, r1, c1 = regions.get('M', (top_bar, left_bar, bot_bar, right_bar))
        else:
            r0, c0, r1, c1 = regions.get('M', (top_bar, left_bar, bot_bar, right_bar))

        # Center the ship in the region
        ny = r0 + (r1 - r0 - mh) // 2
        nx = c0 + (c1 - c0 - mw) // 2
        ny = max(r0, min(ny, r1 - mh))
        nx = max(c0, min(nx, c1 - mw))

        for yy in range(mh):
            for xx in range(mw):
                val = m[yy, xx]
                if val == bg:
                    continue
                ay, ax = ny + yy, nx + xx
                if 0 <= ay < h and 0 <= ax < w and not frame[ay, ax]:
                    out[ay, ax] = val

    return out