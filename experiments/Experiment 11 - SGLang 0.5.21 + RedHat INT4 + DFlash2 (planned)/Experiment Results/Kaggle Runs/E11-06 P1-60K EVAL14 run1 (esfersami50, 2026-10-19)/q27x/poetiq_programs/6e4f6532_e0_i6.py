import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()

    # Detect 2-px uniform bars (frame and middle bars)
    row_bars = []
    for r in range(h - 1):
        if len(np.unique(g[r])) == 1 and len(np.unique(g[r + 1])) == 1 and g[r, 0] == g[r + 1, 0]:
            row_bars.append((r, r + 1, int(g[r, 0])))
    col_bars = []
    for c in range(w - 1):
        if len(np.unique(g[:, c])) == 1 and len(np.unique(g[:, c + 1])) == 1 and g[0, c] == g[0, c + 1]:
            col_bars.append((c, c + 1, int(g[0, c])))

    frame = np.zeros((h, w), dtype=bool)
    for r0, r1, _ in row_bars:
        frame[r0:r1 + 1, :] = True
    for c0, c1, _ in col_bars:
        frame[:, c0:c1 + 1] = True

    nf = g[~frame]
    vals, cnts = np.unique(nf, return_counts=True)
    bg = int(vals[np.argmax(cnts)]) if nf.size else 0

    # Build side->color map from frame bars (excluding middle bars)
    # Identify middle bars: bars that are NOT at the grid boundary
    mid_cols = set()
    for c0, c1, cv in col_bars:
        if c0 > 0 and c1 < w - 1:
            mid_cols.add(cv)
    mid_rows = set()
    for r0, r1, cv in row_bars:
        if r0 > 0 and r1 < h - 1:
            mid_rows.add(cv)

    side_color = {}
    for r0, r1, cv in row_bars:
        if r0 == 0:
            side_color.setdefault(cv, set()).add('U')
        if r1 == h - 1:
            side_color.setdefault(cv, set()).add('D')
    for c0, c1, cv in col_bars:
        if c0 == 0:
            side_color.setdefault(cv, set()).add('L')
        if c1 == w - 1:
            side_color.setdefault(cv, set()).add('R')

    # Find connected components (non-bg, non-frame)
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if not vis[i, j] and g[i, j] != bg and not frame[i, j]:
                q, cells = deque([(i, j)]), []
                vis[i, j] = True
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not vis[ny, nx] and g[ny, nx] != bg and not frame[ny, nx]:
                            vis[ny, nx] = True
                            q.append((ny, nx))
                if any(g[y, x] == 8 for y, x in cells):
                    comps.append(cells)

    if not comps:
        return out

    # Clear all non-frame, non-bg cells
    for i in range(h):
        for j in range(w):
            if not frame[i, j] and g[i, j] != bg:
                out[i, j] = bg

    # Determine interior bounds
    top = 0
    while top < h and frame[top, 0]:
        top += 1
    bot = h
    while bot > 0 and frame[bot - 1, 0]:
        bot -= 1
    left = 0
    while left < w and frame[0, left]:
        left += 1
    right = w
    while right > 0 and frame[0, right - 1]:
        right -= 1

    # Find middle bar positions (for region splitting)
    mid_v_col = None
    for c0, c1, cv in col_bars:
        if c0 > left and c1 < right and cv in mid_cols:
            mid_v_col = c0
            break
    mid_h_row = None
    for r0, r1, cv in row_bars:
        if r0 > top and r1 < bot and cv in mid_rows:
            mid_h_row = r0
            break

    # Define regions based on middle bars
    def get_regions():
        regs = []
        if mid_v_col is not None:
            # Split into left and right
            regs.append(('L', top, left, bot, mid_v_col))
            regs.append(('R', top, mid_v_col + 2, bot, right))
        elif mid_h_row is not None:
            regs.append(('U', top, left, mid_h_row, right))
            regs.append(('D', mid_h_row + 2, left, bot, right))
        else:
            # No middle bar: single region
            regs.append(('M', top, left, bot, right))
        return regs

    regions = get_regions()

    def dihedral(m):
        yield m
        yield np.rot90(m, 1)
        yield np.rot90(m, 2)
        yield np.rot90(m, 3)
        yield np.fliplr(m)
        yield np.flipud(m)
        yield np.rot90(np.fliplr(m), 1)
        yield np.rot90(np.fliplr(m), 2)

    def border_tags(m):
        tags = {}
        hh, ww = m.shape
        for c in range(ww):
            if m[0, c] != bg:
                tags.setdefault(int(m[0, c]), set()).add('U')
            if m[-1, c] != bg:
                tags.setdefault(int(m[-1, c]), set()).add('D')
        for r in range(hh):
            if m[r, 0] != bg:
                tags.setdefault(int(m[r, 0]), set()).add('L')
            if m[r, -1] != bg:
                tags.setdefault(int(m[r, -1]), set()).add('R')
        return tags

    # For each component, determine its region and orientation
    placed_regions = set()
    results = []

    for cells in comps:
        ys = [y for y, _ in cells]
        xs = [x for _, x in cells]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        m = np.full((y1 - y0 + 1, x1 - x0 + 1), bg, dtype=g.dtype)
        for y, x in cells:
            m[y - y0, x - x0] = g[y, x]

        # Determine which region this component belongs to (by centroid)
        cy, cx = np.mean(ys), np.mean(xs)
        best_reg = None
        for reg_name, r0, c0, r1, c1 in regions:
            if r0 <= cy < r1 and c0 <= cx < c1:
                best_reg = reg_name
                break
        if best_reg is None:
            # Fallback: find the region whose center is closest
            best_d = 1e9
            for reg_name, r0, c0, r1, c1 in regions:
                d = (cy - (r0 + r1) / 2) ** 2 + (cx - (c0 + c1) / 2) ** 2
                if d < best_d:
                    best_d = d
                    best_reg = reg_name

        # Try all 8 orientations, score by how well border tags match region sides
        # Region side colors: L->left frame color, R->right, U->top, D->bottom
        # For split regions, the "inner" side faces the middle bar
        reg_sides = {}
        for reg_name, r0, c0, r1, c1 in regions:
            if reg_name == best_reg:
                # Determine which sides this region has
                if best_reg == 'L':
                    reg_sides = {'L': 'frame_L', 'R': 'mid', 'U': 'frame_U', 'D': 'frame_D'}
                elif best_reg == 'R':
                    reg_sides = {'L': 'mid', 'R': 'frame_R', 'U': 'frame_U', 'D': 'frame_D'}
                elif best_reg == 'U':
                    reg_sides = {'U': 'frame_U', 'D': 'mid', 'L': 'frame_L', 'R': 'frame_R'}
                elif best_reg == 'D':
                    reg_sides = {'U': 'mid', 'D': 'frame_D', 'L': 'frame_L', 'R': 'frame_R'}
                else:
                    reg_sides = {'U': 'frame_U', 'D': 'frame_D', 'L': 'frame_L', 'R': 'frame_R'}
                break

        # Get actual frame colors
        frame_U = side_color.get([c for c, s in side_color.items() if 'U' in s][0], None) if any('U' in s for s in side_color.values()) else None
        frame_D = side_color.get([c for c, s in side_color.items() if 'D' in s][0], None) if any('D' in s for s in side_color.values()) else None
        frame_L = side_color.get([c for c, s in side_color.items() if 'L' in s][0], None) if any('L' in s for s in side_color.values()) else None
        frame_R = side_color.get([c for c, s in side_color.items() if 'R' in s][0], None) if any('R' in s for s in side_color.values()) else None

        # Score: for each orientation, check if border colors match expected side colors
        def score_orient(tm):
            tags = border_tags(tm)
            s = 0
            for col, sds in tags.items():
                if col == bg:
                    continue
                if col in (8, 9):
                    continue  # 8 and 9 are structural, not tags
                # col is a tag color - check if it matches a frame side
                for side in sds:
                    if side == 'U' and col == frame_U:
                        s += 3
                    elif side == 'D' and col == frame_D:
                        s += 3
                    elif side == 'L' and col == frame_L:
                        s += 3
                    elif side == 'R' and col == frame_R:
                        s += 3
                    elif side == 'U' and col in mid_rows:
                        s += 2
                    elif side == 'D' and col in mid_rows:
                        s += 2
                    elif side == 'L' and col in mid_cols:
                        s += 2
                    elif side == 'R' and col in mid_cols:
                        s += 2
            return s

        best_m, best_s = m, -999
        for tm in dihedral(m):
            s = score_orient(tm)
            if s > best_s:
                best_s, best_m = s, tm

        results.append((best_reg, best_m))

    # Place each oriented component in its region (centered)
    # Sort by region for deterministic placement
    reg_order = {'U': 0, 'L': 1, 'R': 2, 'D': 3, 'M': 4}
    results.sort(key=lambda x: reg_order.get(x[0], 9))

    used = {}
    for reg_name, m in results:
        mh, mw = m.shape
        # Find the region bounds
        r0, c0, r1, c1 = None, None, None, None
        for rn, rr0, rc0, rr1, rc1 in regions:
            if rn == reg_name:
                r0, c0, r1, c1 = rr0, rc0, rr1, rc1
                break
        if r0 is None:
            r0, c0, r1, c1 = top, left, bot, right

        # If region already used, try alternate
        key = (r0, c0, r1, c1)
        if key in used:
            alt = None
            if reg_name == 'L' and 'R' in [x[0] for x in regions]:
                for rn, rr0, rc0, rr1, rc1 in regions:
                    if rn == 'R':
                        alt = (rr0, rc0, rr1, rc1)
            elif reg_name == 'R' and 'L' in [x[0] for x in regions]:
                for rn, rr0, rc0, rr1, rc1 in regions:
                    if rn == 'L':
                        alt = (rr0, rc0, rr1, rc1)
            elif reg_name == 'U' and 'D' in [x[0] for x in regions]:
                for rn, rr0, rc0, rr1, rc1 in regions:
                    if rn == 'D':
                        alt = (rr0, rc0, rr1, rc1)
            elif reg_name == 'D' and 'U' in [x[0] for x in regions]:
                for rn, rr0, rc0, rr1, rc1 in regions:
                    if rn == 'U':
                        alt = (rr0, rc0, rr1, rc1)
            if alt:
                r0, c0, r1, c1 = alt
                key = (r0, c0, r1, c1)

        used[key] = True

        ny = r0 + (r1 - r0 - mh) // 2
        nx = c0 + (c1 - c0 - mw) // 2
        ny = max(r0, min(ny, r1 - mh))
        nx = max(c0, min(nx, c1 - mw))

        for yy in range(mh):
            for xx in range(mw):
                v = int(m[yy, xx])
                if v == bg:
                    continue
                ay, ax = ny + yy, nx + xx
                if 0 <= ay < h and 0 <= ax < w and not frame[ay, ax]:
                    out[ay, ax] = v

    return out