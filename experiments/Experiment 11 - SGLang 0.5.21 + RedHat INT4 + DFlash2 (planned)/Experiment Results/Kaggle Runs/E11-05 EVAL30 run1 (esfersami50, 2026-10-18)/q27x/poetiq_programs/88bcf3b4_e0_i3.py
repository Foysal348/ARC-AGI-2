import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing two anchor objects and one rough connector.

    The rule inferred from the examples:
      - The background is the most common color.
      - Foreground components are extracted as 8-connected same-color components.
      - Anchor components are straight/symmetric line-like objects (vertical,
        horizontal, point, diamond, or short compact shape).
      - The remaining rough component is replaced by a smooth connector of the
        same color between the two anchors.

    The connector is generated in one of two ways:
      1. U-connector: if one anchor is the compact/line "center" anchor and the
         other is farther away, draw a symmetric parabolic arc around the center
         anchor.  The arc touches the center anchor on the side determined by
         the original rough path and passes through a point adjacent to the
         target anchor.
      2. L-connector: if the anchors form a simple corner connection (e.g. point
         to line, or line to line at a boundary), draw a smooth quarter-arc
         using a quadratic Bezier-like path.

    The implementation is intentionally heuristic but tuned to reproduce the
    given examples and the two challenge grids.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.copy()
    rows, cols = grid.shape

    # ------------------------------------------------------------------
    # Background and components
    # ------------------------------------------------------------------
    bg = int(np.bincount(grid.ravel()).argmax())
    visited = np.zeros((rows, cols), dtype=bool)

    def get_component(r, c):
        color = int(grid[r, c])
        q = deque([(r, c)])
        visited[r, c] = True
        pix = [(r, c)]
        while q:
            rr, cc = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr, nc] and int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            pix.append((nr, nc))
                            q.append((nr, nc))
        return color, pix

    comps = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and int(grid[r, c]) != bg:
                comps.append(get_component(r, c))

    if len(comps) < 3:
        return out

    # ------------------------------------------------------------------
    # Component descriptors
    # ------------------------------------------------------------------
    def bbox(pix):
        rs = [p[0] for p in pix]
        cs = [p[1] for p in pix]
        return min(rs), max(rs), min(cs), max(cs)

    def centroid(pix):
        rs = [p[0] for p in pix]
        cs = [p[1] for p in pix]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    def is_line_like(pix):
        r0, r1, c0, c1 = bbox(pix)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 1 or w == 1:
            return True
        # compact diamond / point
        if h <= 3 and w <= 3 and len(pix) <= 5:
            return True
        # almost perfectly horizontal/vertical
        row_counts = {}
        col_counts = {}
        for r, c in pix:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1
        if max(row_counts.values()) / len(pix) >= 0.9 and h <= 3:
            return True
        if max(col_counts.values()) / len(pix) >= 0.9 and w <= 3:
            return True
        return False

    descs = []
    for color, pix in comps:
        r0, r1, c0, c1 = bbox(pix)
        cr, cc = centroid(pix)
        descs.append({
            'color': color,
            'pix': pix,
            'bbox': (r0, r1, c0, c1),
            'cent': (cr, cc),
            'h': r1 - r0 + 1,
            'w': c1 - c0 + 1,
            'line': is_line_like(pix)
        })

    # Split anchors and paths.
    anchors = [d for d in descs if d['line']]
    paths = [d for d in descs if not d['line']]

    # In some challenge grids, there may be more than one path component.
    # Process each path independently against the best pair of anchors.
    for path in paths:
        if not anchors:
            continue

        pc = path['cent']
        path_color = path['color']

        # Remove original path pixels.
        for r, c in path['pix']:
            out[r, c] = bg

        # Choose the two anchors nearest to the path centroid, but prefer
        # anchors that are spatially coherent with the path bounding box.
        scored = []
        for a in anchors:
            ac = a['cent']
            dist = np.hypot(ac[0] - pc[0], ac[1] - pc[1])
            # slight preference for anchors overlapping path bbox
            r0, r1, c0, c1 = path['bbox']
            ar0, ar1, ac0, ac1 = a['bbox']
            overlap = max(0, min(r1, ar1) - max(r0, ar0) + 1) * \
                      max(0, min(c1, ac1) - max(c0, ac0) + 1)
            score = dist - 0.01 * overlap
            scored.append((score, a))
        scored.sort()
        if len(scored) < 2:
            continue
        A = scored[0][1]
        B = scored[1][1]

        # Make B the smaller/center anchor.
        def extent(d):
            return max(d['h'], d['w'])
        if extent(A) < extent(B):
            A, B = B, A

        ac = A['cent']
        bc = B['cent']

        # Determine orientation: mostly vertical or horizontal.
        dr = abs(ac[0] - bc[0])
        dc = abs(ac[1] - bc[1])
        vertical = dc < dr

        # Determine side based on path position relative to the line joining anchors.
        if vertical:
            mid_c = (ac[1] + bc[1]) / 2.0
            side = -1 if pc[1] < mid_c else 1
        else:
            mid_r = (ac[0] + bc[0]) / 2.0
            side = -1 if pc[0] < mid_r else 1

        # Helper to draw a U-shaped symmetric parabola.
        def draw_U(center, target, side):
            # center is the anchor being wrapped; target is the other anchor.
            cr0, cr1, cc0, cc1 = center['bbox']
            tr0, tr1, tc0, tc1 = target['bbox']

            if vertical:
                # side = -1 left, +1 right
                if side < 0:
                    vertex_c = cc0 - 1
                    target_c = tc0 - 1
                else:
                    vertex_c = cc1 + 1
                    target_c = tc1 + 1

                # Vertex row is the middle of the center anchor.
                vr = (cr0 + cr1) / 2.0
                # Pick the endpoint of target closest to the vertex row.
                if abs(tr0 - vr) <= abs(tr1 - vr):
                    target_r = float(tr0)
                else:
                    target_r = float(tr1)

                # If target is above, use top endpoint; if below, use bottom endpoint.
                if target_r < vr:
                    target_r = float(tr0)
                else:
                    target_r = float(tr1)

                d_r = target_r - vr
                d_c = target_c - vertex_c
                if d_r == 0 or d_c == 0:
                    # fallback: straight offset line
                    r0 = int(round(min(vr, target_r)))
                    r1 = int(round(max(vr, target_r)))
                    for r in range(max(0, r0), min(rows, r1 + 1)):
                        c = int(np.clip(round(vertex_c), 0, cols - 1))
                        if 0 <= c < cols:
                            out[r, c] = path_color
                    return

                # Draw full symmetric parabola: c = vc + k*(r - vr)^2
                # where k = dc / d_r^2
                k = d_c / (d_r * d_r)

                # Determine drawing range: from target to mirror, but keep within
                # reasonable bounds and avoid overwriting anchors.
                max_d = abs(int(round(d_r))) + 1
                r_min = int(np.floor(vr) - max_d - 2)
                r_max = int(np.ceil(vr) + max_d + 2)

                for r in range(max(0, r_min), min(rows, r_max + 1)):
                    val = vertex_c + k * ((r - vr) ** 2)
                    # Round in a way that keeps the curve on the correct side.
                    if side < 0:
                        c = int(np.ceil(val))
                    else:
                        c = int(np.floor(val))
                    c = int(np.clip(c, 0, cols - 1))
                    if 0 <= r < rows and 0 <= c < cols and out[r, c] == bg:
                        out[r, c] = path_color

                # Ensure the straight side segment along the center anchor is present.
                if side < 0:
                    seg_c = cc0 - 1
                else:
                    seg_c = cc1 + 1
                if 0 <= seg_c < cols:
                    for r in range(cr0, cr1 + 1):
                        if out[r, seg_c] == bg:
                            out[r, seg_c] = path_color

            else:
                # horizontal orientation
                if side < 0:
                    vertex_r = cr0 - 1
                    target_r = tr0 - 1
                else:
                    vertex_r = cr1 + 1
                    target_r = tr1 + 1

                vc = (cc0 + cc1) / 2.0
                if abs(tc0 - vc) <= abs(tc1 - vc):
                    target_c = float(tc0)
                else:
                    target_c = float(tc1)
                if target_c < vc:
                    target_c = float(tc0)
                else:
                    target_c = float(tc1)

                d_c = target_c - vc
                d_r = target_r - vertex_r
                if d_c == 0 or d_r == 0:
                    c0 = int(round(min(vc, target_c)))
                    c1 = int(round(max(vc, target_c)))
                    for c in range(max(0, c0), min(cols, c1 + 1)):
                        r = int(np.clip(round(vertex_r), 0, rows - 1))
                        if 0 <= r < rows:
                            out[r, c] = path_color
                    return

                k = d_r / (d_c * d_c)
                max_d = abs(int(round(d_c))) + 1
                c_min = int(np.floor(vc) - max_d - 2)
                c_max = int(np.ceil(vc) + max_d + 2)

                for c in range(max(0, c_min), min(cols, c_max + 1)):
                    val = vertex_r + k * ((c - vc) ** 2)
                    if side < 0:
                        r = int(np.ceil(val))
                    else:
                        r = int(np.floor(val))
                    r = int(np.clip(r, 0, rows - 1))
                    if 0 <= r < rows and 0 <= c < cols and out[r, c] == bg:
                        out[r, c] = path_color

                if side < 0:
                    seg_r = cr0 - 1
                else:
                    seg_r = cr1 + 1
                if 0 <= seg_r < rows:
                    for c in range(cc0, cc1 + 1):
                        if out[seg_r, c] == bg:
                            out[seg_r, c] = path_color

        # Helper to draw an L-shaped quarter arc (point-to-line style).
        def draw_L(center, target, side):
            cr0, cr1, cc0, cc1 = center['bbox']
            tr0, tr1, tc0, tc1 = target['bbox']

            if vertical:
                # start from the center anchor side, end at target side.
                if side < 0:
                    start_c = cc0 - 1
                    end_c = tc0 - 1
                else:
                    start_c = cc1 + 1
                    end_c = tc1 + 1

                # choose start row near the endpoint of center closest to target
                if target['cent'][0] > center['cent'][0]:
                    start_r = cr1
                else:
                    start_r = cr0

                if target['cent'][0] > center['cent'][0]:
                    end_r = tr0
                else:
                    end_r = tr1

                # Add a small vertical/horizontal lead-in for smoother appearance.
                dr = end_r - start_r
                dc = end_c - start_c

                # Draw a rounded quarter arc using a quadratic Bezier.
                p0 = (float(start_r), float(start_c))
                p2 = (float(end_r), float(end_c))
                if dr * dc >= 0:
                    p1 = (float(end_r), float(start_c))
                else:
                    p1 = (float(start_r), float(end_c))

                # sample enough points
                steps = max(32, abs(int(end_r) - int(start_r)) * 8)
                last = None
                for i in range(steps + 1):
                    t = i / steps
                    r = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t * t * p2[0]
                    c = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t * t * p2[1]
                    ri = int(np.clip(round(r), 0, rows - 1))
                    ci = int(np.clip(round(c), 0, cols - 1))
                    if last is None or (abs(ri - last[0]) + abs(ci - last[1])) <= 2:
                        if 0 <= ri < rows and 0 <= ci < cols and out[ri, ci] == bg:
                            out[ri, ci] = path_color
                        last = (ri, ci)
            else:
                if side < 0:
                    start_r = cr0 - 1
                    end_r = tr0 - 1
                else:
                    start_r = cr1 + 1
                    end_r = tr1 + 1

                start_c = int(round(center['cent'][1]))
                end_c = int(round(target['cent'][1]))

                p0 = (float(start_c), float(start_r))
                p2 = (float(end_c), float(end_r))
                if (end_c - start_c) * (end_r - start_r) >= 0:
                    p1 = (float(end_c), float(start_r))
                else:
                    p1 = (float(start_c), float(end_r))

                steps = max(32, abs(int(end_c) - int(start_c)) * 8)
                last = None
                for i in range(steps + 1):
                    t = i / steps
                    c = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t * t * p2[0]
                    r = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t * t * p2[1]
                    ci = int(np.clip(round(c), 0, cols - 1))
                    ri = int(np.clip(round(r), 0, rows - 1))
                    if last is None or (abs(ri - last[1]) + abs(ci - last[0])) <= 2:
                        if 0 <= ri < rows and 0 <= ci < cols and out[ri, ci] == bg:
                            out[ri, ci] = path_color
                        last = (ci, ri)

        # Choose U vs L heuristic:
        # - Use U if center anchor is not a point and has extent >= 2 and is not
        #   touching the grid boundary in a way that would make the U degenerate.
        # - Otherwise use L.
        cr0, cr1, cc0, cc1 = B['bbox']
        is_point = (B['h'] == 1 and B['w'] == 1)
        touches_boundary = (cr0 == 0 or cr1 == rows - 1 or cc0 == 0 or cc1 == cols - 1)

        # For vertical line anchors near a boundary, still use U unless the
        # center is a point.  The boundary case in Example 5 is handled by
        # clipping; in the final output the topmost cell is omitted by the
        # "only draw if background" rule when the parabola rounds onto the anchor.
        if not is_point:
            draw_U(B, A, side)
        else:
            draw_L(B, A, side)

    return out