import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Replace rough non-straight components by smooth parabolic connectors
    between two nearby anchor components.

    The implemented rule matches the given examples:
      - The background is the most frequent color.
      - Straight line/point components are treated as anchors.
      - A rough component of another color is a "path" component.
      - The path component is removed and replaced by a smooth 8-connected
        parabolic arc of the same color, connecting the two anchors.

    For each triple of components:
      1. Determine whether the connection is mainly vertical or horizontal.
      2. Select the shorter/point-like anchor as the "center" anchor and the
         other as the "target" anchor.
      3. Determine the side (left/right for vertical, top/bottom for horizontal)
         from the position of the original rough component.
      4. Draw a quadratic/parabolic connector tangent to the center anchor and
         reaching the target anchor.
    """

    if grid.size == 0:
        return grid.copy()

    out = grid.copy()
    rows, cols = grid.shape

    # --------------------------------------------------------------------
    # Component extraction (8-connected, by color)
    # --------------------------------------------------------------------
    bg = int(np.bincount(grid.ravel()).argmax())
    visited = np.zeros((rows, cols), dtype=bool)

    def component(start_r, start_c):
        color = grid[start_r, start_c]
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        pix = [(start_r, start_c)]
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            pix.append((nr, nc))
                            q.append((nr, nc))
        return color, pix

    components = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and grid[r, c] != bg:
                components.append(component(r, c))

    if len(components) < 3:
        return out

    # --------------------------------------------------------------------
    # Component descriptors
    # --------------------------------------------------------------------
    def bbox(pix):
        rs = [p[0] for p in pix]
        cs = [p[1] for p in pix]
        return min(rs), max(rs), min(cs), max(cs)

    def centroid(pix):
        rs = [p[0] for p in pix]
        cs = [p[1] for p in pix]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    def is_straight_line(pix, tol=0.10):
        r0, r1, c0, c1 = bbox(pix)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 1 or w == 1:
            return True, ('H' if h == 1 else 'V')
        # Almost-horizontal: most cells on one row
        row_counts = {}
        col_counts = {}
        for r, c in pix:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1
        if max(row_counts.values()) / len(pix) >= 0.95 and h <= 3:
            return True, 'H'
        if max(col_counts.values()) / len(pix) >= 0.95 and w <= 3:
            return True, 'V'
        return False, None

    descriptors = []
    for color, pix in components:
        r0, r1, c0, c1 = bbox(pix)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        cr, cc = centroid(pix)
        straight, orient = is_straight_line(pix)
        descriptors.append({
            "color": color,
            "pix": pix,
            "bbox": (r0, r1, c0, c1),
            "h": h,
            "w": w,
            "size": len(pix),
            "cent": (cr, cc),
            "straight": straight,
            "orient": orient,
            "extent": max(h, w) if straight else (min(h, w) + 0.2 * max(h, w))
        })

    anchors = [d for d in descriptors if d["straight"]]
    paths = [d for d in descriptors if not d["straight"]]
    if not paths or len(anchors) < 2:
        return out

    # --------------------------------------------------------------------
    # Geometry helpers
    # --------------------------------------------------------------------
    def side_offset_v(anchor, side):
        """side: -1 left, +1 right"""
        r0, r1, c0, c1 = anchor["bbox"]
        return c0 - 1 if side < 0 else c1 + 1

    def side_offset_h(anchor, side):
        """side: -1 top, +1 bottom"""
        r0, r1, c0, c1 = anchor["bbox"]
        return r0 - 1 if side < 0 else r1 + 1

    def endpoint_row_v(anchor, target_row):
        r0, r1, c0, c1 = anchor["bbox"]
        if target_row >= r0 and target_row <= r1:
            return target_row
        return r1 if target_row > r1 else r0

    def endpoint_col_h(anchor, target_col):
        r0, r1, c0, c1 = anchor["bbox"]
        if target_col >= c0 and target_col <= c1:
            return target_col
        return c1 if target_col > c1 else c0

    def draw_parabola_v(center_anchor, target_anchor, side, center_row, end_row):
        """Draw vertical-orientation parabola: col = f(row)."""
        if end_row == center_row:
            return
        R = abs(end_row - center_row)
        pc = side_offset_v(center_anchor, side)
        pe = side_offset_v(target_anchor, side)

        # Draw the segment from the center toward the target.
        r_start = center_row
        r_end = end_row
        step = 1 if r_end >= r_start else -1
        for i, r in enumerate(range(r_start, r_end + step, step)):
            t = (r - center_row) / float(R)
            c = pc + (pe - pc) * t * t
            cc = int(np.clip(np.rint(c), 0, cols - 1))
            out[r, cc] = center_anchor["color"] if False else path_color

        # Draw the symmetric extension on the far side of the center anchor,
        # only when it stays inside the grid and does not overlap the center
        # anchor's own row span (matches the examples).
        far_step = 1 if end_row < center_row else -1  # away from target
        # Determine far extent: symmetric distance R, clipped by grid.
        far_rows = []
        r = center_row + far_step
        while 0 <= r < rows:
            if (far_step > 0 and r <= center_anchor["bbox"][1]) or \
               (far_step < 0 and r >= center_anchor["bbox"][0]):
                # In most examples, do not draw through the center anchor rows.
                # But allow the immediate row if it is the anchor's edge and
                # there is open space beyond it (small shapes in examples 2/4).
                if far_step > 0:
                    if r == center_anchor["bbox"][1] + 1:
                        pass
                    else:
                        break
                else:
                    if r == center_anchor["bbox"][0] - 1:
                        pass
                    else:
                        break
            far_rows.append(r)
            r += far_step
            if abs(r - center_row) > R + 1:
                break

        for r in far_rows:
            if r == center_row:
                continue
            t = (r - center_row) / float(R)
            c = pc + (pe - pc) * t * t
            cc = int(np.clip(np.rint(c), 0, cols - 1))
            out[r, cc] = path_color

    def draw_parabola_h(center_anchor, target_anchor, side, center_col, end_col):
        """Draw horizontal-orientation parabola: row = f(col)."""
        if end_col == center_col:
            return
        R = abs(end_col - center_col)
        pc = side_offset_h(center_anchor, side)
        pe = side_offset_h(target_anchor, side)

        c_start = center_col
        c_end = end_col
        step = 1 if c_end >= c_start else -1
        for c in range(c_start, c_end + step, step):
            t = (c - center_col) / float(R)
            r = pc + (pe - pc) * t * t
            rr = int(np.clip(np.rint(r), 0, rows - 1))
            out[rr, c] = path_color

        far_step = 1 if end_col < center_col else -1
        far_cols = []
        c = center_col + far_step
        while 0 <= c < cols:
            if (far_step > 0 and c <= center_anchor["bbox"][3]) or \
               (far_step < 0 and c >= center_anchor["bbox"][2]):
                if far_step > 0:
                    if c == center_anchor["bbox"][3] + 1:
                        pass
                    else:
                        break
                else:
                    if c == center_anchor["bbox"][2] - 1:
                        pass
                    else:
                        break
            far_cols.append(c)
            c += far_step
            if abs(c - center_col) > R + 1:
                break

        for c in far_cols:
            if c == center_col:
                continue
            t = (c - center_col) / float(R)
            r = pc + (pe - pc) * t * t
            rr = int(np.clip(np.rint(r), 0, rows - 1))
            out[rr, c] = path_color

    # --------------------------------------------------------------------
    # For each rough path, find the best pair of anchors and replace it.
    # --------------------------------------------------------------------
    processed_paths = set()

    def component_key(d):
        return (d["color"], tuple(sorted(d["pix"])))

    for path in paths:
        path_color = path["color"]
        path_cent = path["cent"]
        path_bbox = path["bbox"]

        # Find the two nearest straight anchors to the path.
        candidate_anchors = []
        for a in anchors:
            ac = a["cent"]
            dist = np.hypot(ac[0] - path_cent[0], ac[1] - path_cent[1])
            candidate_anchors.append((dist, a))
        candidate_anchors.sort()

        if len(candidate_anchors) < 2:
            continue

        # Choose the pair that is geometrically plausible and nearest.
        best_pair = None
        best_score = None
        for i in range(len(candidate_anchors)):
            for j in range(i + 1, len(candidate_anchors)):
                a1 = candidate_anchors[i][1]
                a2 = candidate_anchors[j][1]
                d1 = candidate_anchors[i][0]
                d2 = candidate_anchors[j][0]
                # Prefer anchors on opposite sides of the path or aligned.
                mid = ((a1["cent"][0] + a2["cent"][0]) / 2.0,
                       (a1["cent"][1] + a2["cent"][1]) / 2.0)
                score = (d1 + d2) * 0.5
                # Slight preference for shorter combined extents.
                score += 0.01 * (a1["extent"] + a2["extent"])
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (a1, a2)

        if best_pair is None:
            continue

        A, B = best_pair
        # Make B the smaller/center anchor.
        if A["extent"] < B["extent"]:
            A, B = B, A

        # Remove the original path component.
        for r, c in path["pix"]:
            out[r, c] = bg

        # Decide vertical vs horizontal orientation.
        a_cent = A["cent"]
        b_cent = B["cent"]
        dr = abs(a_cent[0] - b_cent[0])
        dc = abs(a_cent[1] - b_cent[1])

        # If both anchors are horizontal, or mostly separated horizontally, use horizontal mode.
        if dc >= dr and (A["orient"] == "H" or B["orient"] == "H" or A["w"] >= A["h"] and B["w"] >= B["h"]):
            # Horizontal mode: row = f(col)
            # Side: -1 top, +1 bottom
            side = -1 if path_cent[0] < (a_cent[0] + b_cent[0]) / 2.0 else 1

            center_col = int(np.clip(np.rint(b_cent[1]), 0, cols - 1))
            end_col = endpoint_col_h(A, center_col)

            # If the target endpoint is not well separated, use bbox edge.
            if end_col == center_col:
                r0a, r1a, c0a, c1a = A["bbox"]
                r0b, r1b, c0b, c1b = B["bbox"]
                if a_cent[1] >= center_col:
                    end_col = c0a
                else:
                    end_col = c1a

            draw_parabola_h(B, A, side, center_col, end_col)

        else:
            # Vertical mode: col = f(row)
            # Side: -1 left, +1 right
            side = -1 if path_cent[1] < (a_cent[1] + b_cent[1]) / 2.0 else 1

            center_row = int(np.clip(np.rint(b_cent[0]), 0, rows - 1))
            end_row = endpoint_row_v(A, center_row)

            if end_row == center_row:
                r0a, r1a, c0a, c1a = A["bbox"]
                if a_cent[0] >= center_row:
                    end_row = r0a
                else:
                    end_row = r1a

            draw_parabola_v(B, A, side, center_row, end_row)

        processed_paths.add(component_key(path))

    return out