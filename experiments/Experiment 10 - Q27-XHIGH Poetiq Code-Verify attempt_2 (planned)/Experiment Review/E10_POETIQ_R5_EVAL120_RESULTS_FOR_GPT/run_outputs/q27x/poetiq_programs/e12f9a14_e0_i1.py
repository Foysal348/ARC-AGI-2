import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input by detecting payload regions enclosed by wall regions.
    Each missing wall cell on the boundary of a payload's 1-cell expanded box
    emits a ray in the outward direction (or diagonal for a corner gap).

    Rays are routed as straight 8-direction segments.  When two rays of different
    colors would intersect or cross, both are truncated just before the conflict
    and then continue in the normalized sum of their previous directions.
    This reproduces the examples: perpendicular rays bend into a shared diagonal,
    and crossing diagonal rays bend into their common straight component.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Background is the most common color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Wall color is the most common non-background color.
    nonbg_counts = {}
    for v, c in zip(vals, counts):
        v = int(v)
        if v != bg:
            nonbg_counts[v] = int(c)
    wall = max(nonbg_counts, key=nonbg_counts.get) if nonbg_counts else -1

    # Payload colors are all non-background colors except the wall color.
    payload_colors = [v for v, c in nonbg_counts.items() if v != wall]

    # Find connected payload components (4-connected).
    comp_id = -1
    payload_comps = []
    seen = np.zeros((h, w), dtype=bool)
    for color in payload_colors:
        for r in range(h):
            for c in range(w):
                if seen[r, c] or grid[r, c] != color:
                    continue
                seen[r, c] = True
                q = deque([(r, c)])
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                if cells:
                    rs = [x[0] for x in cells]
                    cs = [x[1] for x in cells]
                    payload_comps.append((color, min(rs), max(rs), min(cs), max(cs)))

    # Obstacles for ray extension: any original non-background cell.
    obstacle = (grid != bg)

    def extend_to_boundary(r, c, dr, dc):
        """Extend from (r, c) in direction (dr, dc) until just before boundary/obstacle."""
        er, ec = r, c
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w and not obstacle[nr, nc]:
            er, ec = nr, nc
            nr, nc = nr + dr, nc + dc
        return er, ec

    def cell_at(seg, k):
        r, c = seg['start']
        dr, dc = seg['dir']
        return (r + dr * k, c + dc * k)

    def seg_len(seg):
        return max(abs(seg['end'][0] - seg['start'][0]),
                   abs(seg['end'][1] - seg['start'][1]))

    def normalize_dir(dr, dc):
        nr = 0 if dr == 0 else (1 if dr > 0 else -1)
        nc = 0 if dc == 0 else (1 if dc > 0 else -1)
        if nr == 0 and nc == 0:
            return None
        return (nr, nc)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[1] - o[1])

    def on_segment(o, a, b):
        return (min(o[0], b[0]) <= a[0] <= max(o[0], b[0]) and
                min(o[1], b[1]) <= a[1] <= max(o[1], b[1]))

    def segments_intersect(p1, p2, p3, p4):
        d1 = cross(p1, p2, p3)
        d2 = cross(p1, p2, p4)
        d3 = cross(p3, p4, p1)
        d4 = cross(p3, p4, p2)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        if d1 == 0 and on_segment(p1, p3, p2):
            return True
        if d2 == 0 and on_segment(p1, p4, p2):
            return True
        if d3 == 0 and on_segment(p3, p1, p4):
            return True
        if d4 == 0 and on_segment(p3, p2, p4):
            return True
        return False

    def segment_cells(seg):
        k = seg_len(seg)
        cells = []
        for i in range(k + 1):
            cells.append(cell_at(seg, i))
        return cells

    def conflict_info(seg_a, seg_b):
        """
        Return None if no conflict, else:
        (turn_index_a, turn_index_b, new_dr, new_dc)
        """
        la = seg_len(seg_a)
        lb = seg_len(seg_b)

        # Shared cell conflict.
        cells_a = segment_cells(seg_a)
        cells_b = set(segment_cells(seg_b))
        for i, cell in enumerate(cells_a):
            if cell in cells_b:
                # first occurrence in b
                j = None
                for jj, cell_b in enumerate(segment_cells(seg_b)):
                    if cell_b == cell:
                        j = jj
                        break
                ti_a = max(0, i - 1)
                ti_b = max(0, j - 1)
                dr_a, dc_a = seg_a['dir']
                dr_b, dc_b = seg_b['dir']
                new = normalize_dir(dr_a + dr_b, dc_a + dc_b)
                if new is None:
                    return (ti_a, ti_b, None, None)
                return (ti_a, ti_b, new[0], new[1])

        # Diagonal crossing conflict (no shared cell).
        for i in range(la):
            a0 = cell_at(seg_a, i)
            a1 = cell_at(seg_a, i + 1)
            for j in range(lb):
                b0 = cell_at(seg_b, j)
                b1 = cell_at(seg_b, j + 1)
                if segments_intersect(a0, a1, b0, b1):
                    dr_a, dc_a = seg_a['dir']
                    dr_b, dc_b = seg_b['dir']
                    new = normalize_dir(dr_a + dr_b, dc_a + dc_b)
                    if new is None:
                        return (i, j, None, None)
                    return (i, j, new[0], new[1])
        return None

    # Build initial rays from valid gaps.
    paths = []
    for color, pr0, pr1, pc0, pc1 in payload_comps:
        br0, br1 = max(0, pr0 - 1), min(h - 1, pr1 + 1)
        bc0, bc1 = max(0, pc0 - 1), min(w - 1, pc1 + 1)

        # Perimeter of the expanded box.
        perim = []
        for c in range(bc0, bc1 + 1):
            perim.append((br0, c))
            perim.append((br1, c))
        for r in range(br0 + 1, br1):
            perim.append((r, bc0))
            perim.append((r, bc1))

        seen_gap = set()
        for r, c in perim:
            if (r, c) in seen_gap:
                continue
            if grid[r, c] != bg:
                continue

            # Direction outward from the box side(s).
            dr = 0
            dc = 0
            if r == br0:
                dr = -1
            elif r == br1:
                dr = 1
            if c == bc0:
                dc = -1
            elif c == bc1:
                dc = 1
            if dr == 0 and dc == 0:
                continue

            # The inward cell must be the payload itself.
            ir, ic = r - dr, c - dc
            if not (0 <= ir < h and 0 <= ic < w) or grid[ir, ic] != color:
                continue

            seen_gap.add((r, c))
            end = extend_to_boundary(r, c, dr, dc)
            if end != (r, c):
                paths.append({
                    'color': color,
                    'segments': [{'start': (r, c), 'dir': (dr, dc), 'end': end}]
                })

    # Iteratively resolve conflicts between different-colored segments.
    max_iter = 500
    for _ in range(max_iter):
        resolved = False
        n = len(paths)
        for i in range(n):
            if resolved:
                break
            for j in range(i + 1, n):
                if resolved:
                    break
                if paths[i]['color'] == paths[j]['color']:
                    continue

                conf = None
                si = sj = None
                for ai, sega in enumerate(paths[i]['segments']):
                    for bj, segb in enumerate(paths[j]['segments']):
                        info = conflict_info(sega, segb)
                        if info is not None:
                            conf = info
                            si = ai
                            sj = bj
                            break
                    if conf:
                        break
                if not conf:
                    continue

                ti_a, ti_b, ndr, ndc = conf
                seg_a = paths[i]['segments'][si]
                seg_b = paths[j]['segments'][sj]

                # Truncate both segments at their turn cells.
                end_a = cell_at(seg_a, ti_a)
                end_b = cell_at(seg_b, ti_b)
                paths[i]['segments'][si]['end'] = end_a
                paths[j]['segments'][sj]['end'] = end_b

                # Append deflected segments if possible.
                if ndr is not None:
                    end_na = extend_to_boundary(end_a[0], end_a[1], ndr, ndc)
                    if end_na != end_a:
                        paths[i]['segments'].append({
                            'start': end_a,
                            'dir': (ndr, ndc),
                            'end': end_na
                        })

                    end_nb = extend_to_boundary(end_b[0], end_b[1], ndr, ndc)
                    if end_nb != end_b:
                        paths[j]['segments'].append({
                            'start': end_b,
                            'dir': (ndr, ndc),
                            'end': end_nb
                        })

                resolved = True
                break

    # Render final grid.
    out = grid.copy()
    for p in paths:
        for seg in p['segments']:
            k = seg_len(seg)
            for t in range(k + 1):
                r, c = cell_at(seg, t)
                out[r, c] = p['color']

    return out