def transform(grid):
    """
    Transform the input grid by re-routing each small "attached" object so that
    it becomes a chamfered connector between the line it is attached to and the
    nearest other object in front of that line.

    The rule inferred from the examples:
    1. Identify connected non-background components.
    2. A "moving" component is a non-straight component that is edge-adjacent
       to a straight line component.
    3. For each moving component:
       - The attached straight line is the anchor line.
       - The target is the nearest non-moving component other than the anchor.
       - The starting cell is the moving component's cell adjacent to the anchor
         that is closest to the target.
       - Draw the first N cells of a canonical connector path, where N is the
         original number of cells of the moving component.
    4. The canonical path is:
       - chamfer diagonally/straightly toward the target's outer offset contour;
       - follow that contour from near to far endpoint;
       - continue straight/diagonally away.
    """
    import numpy as np
    from collections import deque

    # Convert to a mutable list of lists.
    if isinstance(grid, np.ndarray):
        g = [list(row) for row in grid.tolist()]
    else:
        g = [list(row) for row in grid]

    H = len(g)
    if H == 0:
        return np.array(g, dtype=int)
    W = len(g[0])
    if W == 0:
        return np.array(g, dtype=int)

    # Background is the most frequent color.
    color_count = {}
    for r in range(H):
        for c in range(W):
            color_count[g[r][c]] = color_count.get(g[r][c], 0) + 1
    background = max(color_count, key=lambda x: color_count[x])

    def inside(r, c):
        return 0 <= r < H and 0 <= c < W

    # ------------------------------------------------------------
    # Connected components of non-background cells (8-connected).
    # ------------------------------------------------------------
    class Comp:
        __slots__ = ("idx", "color", "cells", "count", "rmin", "rmax", "cmin", "cmax")

        def __init__(self, idx, color, cells):
            self.idx = idx
            self.color = color
            self.cells = cells
            self.count = len(cells)
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            self.rmin = min(rs)
            self.rmax = max(rs)
            self.cmin = min(cs)
            self.cmax = max(cs)

    comp_id = [[-1] * W for _ in range(H)]
    comps = []
    dirs8 = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(H):
        for c in range(W):
            if g[r][c] == background or comp_id[r][c] != -1:
                continue
            color = g[r][c]
            cid = len(comps)
            stack = [(r, c)]
            comp_id[r][c] = cid
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in dirs8:
                    nr, nc = cr + dr, cc + dc
                    if inside(nr, nc) and comp_id[nr][nc] == -1 and g[nr][nc] == color:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            comps.append(Comp(cid, color, cells))

    if not comps:
        return np.array(g, dtype=int)

    # ------------------------------------------------------------
    # Line detection: horizontal or vertical contiguous segment.
    # ------------------------------------------------------------
    line_info = []
    for comp in comps:
        rows = {r for r, _ in comp.cells}
        cols = {c for _, c in comp.cells}
        if len(rows) == 1 and len(cols) == comp.count:
            y = next(iter(rows))
            line_info.append(("h", y, comp.cmin, comp.cmax))
        elif len(cols) == 1 and len(rows) == comp.count:
            x = next(iter(cols))
            line_info.append(("v", x, comp.rmin, comp.rmax))
        else:
            line_info.append(None)

    def is_anchor_line(cid):
        return line_info[cid] is not None and comps[cid].count >= 2

    def line_endpoints(info):
        if info is None:
            return []
        if info[0] == "v":
            _, x, rmin, rmax = info
            return [(rmin, x), (rmax, x)]
        else:
            _, y, cmin, cmax = info
            return [(y, cmin), (y, cmax)]

    def cell_cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def comp_min_cheb(A, B):
        # Exact minimum Chebyshev distance between component cells.
        best = 10**9
        if A.count * B.count <= 20000:
            for p in A.cells:
                for q in B.cells:
                    d = cell_cheb(p, q)
                    if d < best:
                        best = d
                        if best == 0:
                            return 0
        else:
            # Fallback: bounding-box distance is safe enough for large cases.
            dx = 0
            if A.cmax < B.cmin:
                dx = B.cmin - A.cmax
            elif B.cmax < A.cmin:
                dx = A.cmin - B.cmax
            dy = 0
            if A.rmax < B.rmin:
                dy = B.rmin - A.rmax
            elif B.rmax < A.rmin:
                dy = A.rmin - B.rmax
            best = max(dx, dy)
        return best

    def cell_comp_min_cheb(cell, comp):
        best = 10**9
        for p in comp.cells:
            d = cell_cheb(cell, p)
            if d < best:
                best = d
        return best

    # ------------------------------------------------------------
    # Identify moving components attached to anchor lines.
    # ------------------------------------------------------------
    moving_candidates = []

    for cid, comp in enumerate(comps):
        if is_anchor_line(cid):
            continue

        adjacent_lines = []
        for r, c in comp.cells:
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if inside(nr, nc):
                    nid = comp_id[nr][nc]
                    if nid != -1 and nid != cid and is_anchor_line(nid):
                        if nid not in adjacent_lines:
                            adjacent_lines.append(nid)

        if not adjacent_lines:
            continue

        best_A = None
        best_key = None
        for A_id in adjacent_lines:
            A = comps[A_id]
            anchors = []
            for r, c in comp.cells:
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if inside(nr, nc) and comp_id[nr][nc] == A_id:
                        anchors.append((r, c))
                        break
            if not anchors:
                continue

            eps = line_endpoints(line_info[A_id])
            score = min(cell_cheb(a, e) for a in anchors for e in eps)
            key = (score, -A.count, A_id)
            if best_key is None or key < best_key:
                best_key = key
                best_A = A_id

        if best_A is not None:
            moving_candidates.append((cid, best_A))

    moving_ids = {cid for cid, _ in moving_candidates}

    # ------------------------------------------------------------
    # Path helpers.
    # ------------------------------------------------------------
    def simple_chamfer_path(start, end):
        """Diagonal-first 8-neighbor path from start to end."""
        path = [start]
        r, c = start
        er, ec = end
        steps = 0
        while (r, c) != (er, ec):
            dr = er - r
            dc = ec - c
            if dr != 0 and dc != 0:
                r += 1 if dr > 0 else -1
                c += 1 if dc > 0 else -1
            elif dr != 0:
                r += 1 if dr > 0 else -1
            else:
                c += 1 if dc > 0 else -1
            path.append((r, c))
            steps += 1
            if steps > 5000:
                break
        return path

    def bfs_path(start, end, forbidden):
        """Shortest 8-neighbor path avoiding forbidden cells, if possible."""
        if start not in forbidden and end in forbidden:
            return None
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == end:
                break
            r, c = cur
            nbrs = []
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if not inside(nr, nc):
                    continue
                cell = (nr, nc)
                if cell in prev or cell in forbidden:
                    continue
                d = max(abs(nr - end[0]), abs(nc - end[1]))
                # Prefer diagonal moves slightly for a chamfered feel.
                pref = 0 if dr != 0 and dc != 0 else 1
                nbrs.append((d, pref, dr, dc, cell))
            nbrs.sort()
            for _, _, dr, dc, cell in nbrs:
                prev[cell] = cur
                q.append(cell)
        if end not in prev:
            return None
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def path_between(start, end, forbidden):
        """Canonical path if possible, otherwise BFS fallback."""
        if start == end:
            return [start]
        sp = simple_chamfer_path(start, end)
        if all(p not in forbidden for p in sp[1:]):
            return sp
        if inside(start[0], start[1]) and inside(end[0], end[1]):
            bp = bfs_path(start, end, forbidden)
            if bp is not None:
                return bp
        return sp

    def build_contour(target, orient, side, along_dir):
        """
        Build the outer offset contour of target on the chosen side.
        For vertical anchor: side is -1 left / +1 right, along_dir is row direction.
        For horizontal anchor: side is -1 above / +1 below, along_dir is column direction.
        """
        if orient == "v":
            rows = {}
            for r, c in target.cells:
                rows.setdefault(r, []).append(c)

            row_off = {}
            for r, cs in rows.items():
                row_off[r] = min(cs) - 1 if side < 0 else max(cs) + 1

            sorted_rows = sorted(rows.keys())
            if not sorted_rows:
                return []
            if along_dir == 1:
                start_r, end_r = sorted_rows[0], sorted_rows[-1]
            else:
                start_r, end_r = sorted_rows[-1], sorted_rows[0]

            pts = []
            last_c = None
            for r in range(start_r, end_r + along_dir, along_dir):
                if r in row_off:
                    last_c = row_off[r]
                elif last_c is None:
                    nr = min(rows.keys(), key=lambda x: abs(x - r))
                    last_c = row_off[nr]
                pts.append((r, last_c))

            cleaned = [pts[0]]
            for p in pts[1:]:
                if p == cleaned[-1]:
                    continue
                if abs(p[0] - cleaned[-1][0]) <= 1 and abs(p[1] - cleaned[-1][1]) <= 1:
                    cleaned.append(p)
                else:
                    # Connect with a small chamfer path; forbidden is not known here,
                    # but these points are already outside target.
                    path = simple_chamfer_path(cleaned[-1], p)
                    cleaned.extend(path[1:])
            return cleaned

        else:  # horizontal anchor
            cols = {}
            for r, c in target.cells:
                cols.setdefault(c, []).append(r)

            col_off = {}
            for c, rs in cols.items():
                col_off[c] = min(rs) - 1 if side < 0 else max(rs) + 1

            sorted_cols = sorted(cols.keys())
            if not sorted_cols:
                return []
            if along_dir == 1:
                start_c, end_c = sorted_cols[0], sorted_cols[-1]
            else:
                start_c, end_c = sorted_cols[-1], sorted_cols[0]

            pts = []
            last_r = None
            for c in range(start_c, end_c + along_dir, along_dir):
                if c in col_off:
                    last_r = col_off[c]
                elif last_r is None:
                    nc = min(cols.keys(), key=lambda x: abs(x - c))
                    last_r = col_off[nc]
                pts.append((last_r, c))

            cleaned = [pts[0]]
            for p in pts[1:]:
                if p == cleaned[-1]:
                    continue
                if abs(p[0] - cleaned[-1][0]) <= 1 and abs(p[1] - cleaned[-1][1]) <= 1:
                    cleaned.append(p)
                else:
                    path = simple_chamfer_path(cleaned[-1], p)
                    cleaned.extend(path[1:])
            return cleaned

    def generate_path(M, A, B, start):
        N = M.count
        info = line_info[A.idx]
        orient = info[0]

        # Perpendicular side relative to anchor line.
        if orient == "v":
            _, x, rmin, rmax = info
            if start[1] > x:
                side = 1
            elif start[1] < x:
                side = -1
            else:
                side = 1 if (B.cmin + B.cmax) // 2 >= x else -1

            # Direction along the anchor line toward target.
            a_mid = (rmin + rmax) / 2.0
            if B.rmax < rmin:
                along = -1
            elif B.rmin > rmax:
                along = 1
            else:
                b_mid = (B.rmin + B.rmax) / 2.0
                if b_mid < a_mid:
                    along = -1
                elif b_mid > a_mid:
                    along = 1
                else:
                    along = -1 if start[0] < a_mid else 1

        else:  # horizontal
            _, y, cmin, cmax = info
            if start[0] > y:
                side = 1
            elif start[0] < y:
                side = -1
            else:
                side = 1 if (B.rmin + B.rmax) // 2 >= y else -1

            a_mid = (cmin + cmax) / 2.0
            if B.cmax < cmin:
                along = -1
            elif B.cmin > cmax:
                along = 1
            else:
                b_mid = (B.cmin + B.cmax) / 2.0
                if b_mid < a_mid:
                    along = -1
                elif b_mid > a_mid:
                    along = 1
                else:
                    along = -1 if start[1] < a_mid else 1

        contour = build_contour(B, orient, side, along)
        if not contour:
            return []

        # All non-background cells except the moving component itself are obstacles
        # while generating the ideal connector.
        M_set = set(M.cells)
        forbidden = set()
        for r in range(H):
            for c in range(W):
                if g[r][c] != background and (r, c) not in M_set:
                    forbidden.add((r, c))

        path = [start]

        # Phase 1: approach the first contour point.
        p1 = path_between(start, contour[0], forbidden)
        path.extend(p1[1:])
        if len(path) >= N:
            return path[:N]

        # Phase 2: follow the target's outer contour.
        for pt in contour[1:]:
            if len(path) >= N:
                break
            pc = path_between(path[-1], pt, forbidden)
            path.extend(pc[1:])

        # Phase 3: exit away from the target.
        if len(path) < N:
            if orient == "v":
                exit_dir = (along, -side)
            else:
                exit_dir = (side, -along)

            r, c = path[-1]
            while len(path) < N:
                nr = r + exit_dir[0]
                nc = c + exit_dir[1]
                if not inside(nr, nc):
                    break
                cell = (nr, nc)
                if cell in forbidden:
                    break
                path.append(cell)
                r, c = nr, nc

        return path[:N]

    # ------------------------------------------------------------
    # Create tasks for all moving components.
    # ------------------------------------------------------------
    tasks = []

    for M_id, A_id in moving_candidates:
        M = comps[M_id]
        A = comps[A_id]

        # Choose target: nearest non-moving component other than A.
        B_id = None
        best_dist = 10**9

        # Prefer not to target another moving component.
        for C_id in range(len(comps)):
            if C_id == M_id or C_id == A_id or C_id in moving_ids:
                continue
            d = comp_min_cheb(M, comps[C_id])
            if d < best_dist:
                best_dist = d
                B_id = C_id

        # Fallback if that was impossible.
        if B_id is None:
            for C_id in range(len(comps)):
                if C_id == M_id or C_id == A_id:
                    continue
                d = comp_min_cheb(M, comps[C_id])
                if d < best_dist:
                    best_dist = d
                    B_id = C_id

        if B_id is None:
            continue
        B = comps[B_id]

        # Anchor cell: moving component's cell edge-adjacent to A, closest to B.
        anchors = []
        for r, c in M.cells:
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if inside(nr, nc) and comp_id[nr][nc] == A_id:
                    anchors.append((r, c))
                    break

        if not anchors:
            for r, c in M.cells:
                for dr, dc in dirs8:
                    nr, nc = r + dr, c + dc
                    if inside(nr, nc) and comp_id[nr][nc] == A_id:
                        anchors.append((r, c))
                        break

        if not anchors:
            anchors = [min(M.cells, key=lambda p: comp_min_cheb(Comp(-1, -1, [p]), A))]

        start = min(anchors, key=lambda p: cell_comp_min_cheb(p, B))
        path = generate_path(M, A, B, start)
        if path:
            tasks.append((M, path))

    if not tasks:
        return np.array(g, dtype=int)

    # ------------------------------------------------------------
    # Apply changes: erase moving components, then draw connectors.
    # ------------------------------------------------------------
    out = [row[:] for row in g]

    for M, _ in tasks:
        for r, c in M.cells:
            out[r][c] = background

    for M, path in tasks:
        for r, c in path:
            if not inside(r, c):
                break
            # Keep stationary objects intact.
            if out[r][c] != background and out[r][c] != M.color:
                break
            out[r][c] = M.color

    return np.array(out, dtype=int)