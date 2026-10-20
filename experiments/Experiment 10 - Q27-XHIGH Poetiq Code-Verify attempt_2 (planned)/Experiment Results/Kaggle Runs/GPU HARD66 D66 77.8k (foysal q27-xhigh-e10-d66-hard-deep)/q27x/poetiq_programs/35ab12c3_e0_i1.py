import numpy as np
from functools import cmp_to_key

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by drawing implied line shapes for each color.

    Rules inferred from the examples:
    1. Group all non-zero pixels by color.
    2. For every color with at least two pixels:
       - Order the pixels cyclically by angle around their centroid.
       - For pixels with the same angle, order them from nearest to farthest
         from the centroid.
       - Connect consecutive pixels in this cyclic order.
       - Draw a segment only if it is horizontal, vertical, or a 45-degree
         diagonal.
    3. For a color with exactly one pixel:
       - If it is 8-neighbor adjacent to a pixel of another multi-pixel color,
         treat the single pixel as a shifted copy of that anchor pixel.
       - Shift the entire drawn shape of that other color by the same offset.
       - Stamp the shifted shape in the single color, only over empty cells.
    4. Original pixels are always preserved and are never overwritten by
       another color.
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return arr.copy()

    out = arr.astype(int, copy=True)
    rows, cols = out.shape
    if rows == 0 or cols == 0:
        return out

    # Collect coordinates for every non-zero color.
    colors = {}
    for r, c in zip(*np.where(arr != 0)):
        color = int(arr[r, c])
        if color != 0:
            colors.setdefault(color, []).append((r, c))

    if not colors:
        return out

    def is_allowed_segment(p, q):
        """Return True if p-q is horizontal, vertical, or 45-degree diagonal."""
        dr = q[0] - p[0]
        dc = q[1] - p[1]
        return dr == 0 or dc == 0 or abs(dr) == abs(dc)

    def add_segment(shape, active, p, q):
        """
        Add an allowed straight segment to the shape set.
        If the segment is drawn, mark its endpoints as active points.
        """
        if not is_allowed_segment(p, q):
            return

        active.add(p)
        active.add(q)

        dr = q[0] - p[0]
        dc = q[1] - p[1]

        if dr == 0:
            r = p[0]
            for c in range(min(p[1], q[1]), max(p[1], q[1]) + 1):
                shape.add((r, c))
        elif dc == 0:
            c = p[1]
            for r in range(min(p[0], q[0]), max(p[0], q[0]) + 1):
                shape.add((r, c))
        else:
            steps = abs(dr)
            sr = 1 if dr > 0 else -1
            sc = 1 if dc > 0 else -1
            for k in range(steps + 1):
                shape.add((p[0] + k * sr, p[1] + k * sc))

    def angle_order(points):
        """
        Sort points cyclically by polar angle around their centroid.

        The sort is exact: vectors are scaled by the number of points so that
        centroid coordinates need not be fractional.
        """
        n = len(points)
        if n <= 2:
            return list(points)

        sum_r = sum(r for r, c in points)
        sum_c = sum(c for r, c in points)

        # For each point, store integer vector from centroid and the point.
        data = []
        for r, c in points:
            vx = c * n - sum_c
            vy = r * n - sum_r
            data.append((vx, vy, r, c))

        def half(vx, vy):
            # Split into two half-planes for polar sorting.
            # This orders angles from -pi to pi.
            if vy < 0 or (vy == 0 and vx > 0):
                return 0
            return 1

        def cmp(a, b):
            vx1, vy1, r1, c1 = a
            vx2, vy2, r2, c2 = b

            h1 = half(vx1, vy1)
            h2 = half(vx2, vy2)
            if h1 != h2:
                return -1 if h1 < h2 else 1

            cross = vx1 * vy2 - vy1 * vx2
            if cross > 0:
                return -1
            if cross < 0:
                return 1

            # Same angle: nearest to centroid first.
            d1 = vx1 * vx1 + vy1 * vy1
            d2 = vx2 * vx2 + vy2 * vy2
            if d1 != d2:
                return -1 if d1 < d2 else 1

            # Deterministic final tie-breaker.
            if r1 != r2:
                return -1 if r1 < r2 else 1
            if c1 != c2:
                return -1 if c1 < c2 else 1
            return 0

        ordered = sorted(data, key=cmp_to_key(cmp))
        return [(r, c) for _, _, r, c in ordered]

    def compute_shape(points):
        """
        Compute the full drawn shape for a color.

        Returns:
            shape: set of all cells belonging to the drawn shape.
            active: set of input points incident to at least one drawn segment.
        """
        shape = set(points)
        active = set()

        if len(points) < 2:
            return shape, active

        ordered = angle_order(points)
        n = len(ordered)

        if n == 2:
            add_segment(shape, active, ordered[0], ordered[1])
        else:
            for i in range(n):
                add_segment(shape, active, ordered[i], ordered[(i + 1) % n])

        return shape, active

    # Compute all shapes and active endpoint sets.
    shapes = {}
    actives = {}
    for color, points in colors.items():
        shape, active = compute_shape(points)
        shapes[color] = shape
        actives[color] = active

    # Draw all normal multi-pixel / single-pixel shapes first.
    # Only fill empty cells or the same color, preserving existing pixels.
    for color in sorted(shapes):
        for r, c in shapes[color]:
            if 0 <= r < rows and 0 <= c < cols:
                if out[r, c] == 0 or out[r, c] == color:
                    out[r, c] = color

    def shift_score(color, offset):
        """
        Score how many new empty cells would be filled by shifting this color's
        shape by offset.
        """
        or_, oc = offset
        score = 0
        for r, c in shapes.get(color, ()):
            nr, nc = r + or_, c + oc
            if 0 <= nr < rows and 0 <= nc < cols and out[nr, nc] == 0:
                score += 1
        return score

    # Find shifting tasks for single-pixel colors.
    single_tasks = []

    for color in sorted(colors):
        pts = colors[color]
        if len(pts) != 1:
            continue

        sr, sc = pts[0]
        best = None

        for other, other_pts in colors.items():
            if other == color or len(other_pts) < 2:
                continue

            for ar, ac in other_pts:
                # The single pixel must be 8-neighbor adjacent to an anchor.
                if max(abs(sr - ar), abs(sc - ac)) != 1:
                    continue

                offset = (sr - ar, sc - ac)
                score = shift_score(other, offset)
                active_pref = 0 if (ar, ac) in actives.get(other, set()) else 1

                # Prefer a candidate that fills more cells, then active anchor,
                # then larger source color, then deterministic order.
                key = (-score, active_pref, -len(other_pts), other, ar, ac)
                if best is None or key < best[0]:
                    best = (key, other, offset, score)

        if best is not None:
            _, other, offset, score = best
            single_tasks.append((score, color, other, offset))

    # Apply single-color shifted shapes. Larger-score tasks first if conflicts occur.
    single_tasks.sort(key=lambda t: (-t[0], t[1]))

    for score, color, other, offset in single_tasks:
        or_, oc = offset
        for r, c in shapes[other]:
            nr, nc = r + or_, c + oc
            if 0 <= nr < rows and 0 <= nc < cols:
                if out[nr, nc] == 0 or out[nr, nc] == color:
                    out[nr, nc] = color

    return out