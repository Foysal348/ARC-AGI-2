import numpy as np
from collections import deque

def _components(grid):
    """Return 8-connected foreground components."""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return []

    bg = np.bincount(grid.ravel()).argmax()
    visited = np.zeros((rows, cols), dtype=bool)
    comps = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == bg or visited[r, c]:
                continue

            color = grid[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            pix = [(r, c)]

            while q:
                x, y = q.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if not visited[nx, ny] and grid[nx, ny] == color:
                                visited[nx, ny] = True
                                pix.append((nx, ny))
                                q.append((nx, ny))

            comps.append((color, pix))

    return comps


def _bbox(pix):
    rs = [r for r, c in pix]
    cs = [c for r, c in pix]
    return min(rs), max(rs), min(cs), max(cs)


def _straightness(pix):
    if len(pix) < 2:
        return 1.0
    rs = [r for r, c in pix]
    cs = [c for r, c in pix]
    h = max(rs) - min(rs)
    w = max(cs) - min(cs)
    if h == 0:
        return 1.0
    if w == 0:
        return 1.0
    # Prefer components that look like straight lines.
    return max(h, w) / max(1, min(h, w))


def _curve_points(p0, c, p1, steps):
    """Sample a quadratic Bezier curve from p0 to p1 with control c."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        r = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * c[0] + t * t * p1[0]
        ccol = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * c[1] + t * t * p1[1]
        pts.append((int(round(r)), int(round(ccol))))
    return pts


def _draw_curve(out, color, anchors, path_color, grid_shape):
    """
    Draw the replacement Bezier curve for the current local instance.

    The anchors are the two components that remain fixed.
    The curve color is the color of the component being replaced.
    """
    if len(anchors) != 2:
        return out

    # Order anchors top-to-bottom, left-to-right if needed.
    a, b = anchors
    ac = _centroid(a[1])
    bc = _centroid(b[1])
    if ac[0] > bc[0] or (ac[0] == bc[0] and ac[1] > bc[1]):
        a, b = b, a

    ar0, ar1, ac0, ac1 = _bbox(a[1])
    br0, br1, bc0, bc1 = _bbox(b[1])

    # Choose the side where the curve should lie.
    # Use the midpoint between anchors and compare with each anchor midpoint.
    mid_r = (ar0 + ar1 + br0 + br1) / 2
    mid_c = (ac0 + ac1 + bc0 + bc1) / 2
    left_anchor_mid = ((ac0 + ac1) / 2, (ar0 + ar1) / 2)
    right_anchor_mid = ((bc0 + bc1) / 2, (br0 + br1) / 2)

    # Heuristic side selection:
    # If the first anchor is left of the second, curve usually lies left.
    # If the first anchor is right of the second, curve usually lies right.
    left_mid_c = (ac0 + ac1) / 2
    right_mid_c = (bc0 + bc1) / 2

    if left_mid_c <= right_mid_c:
        side = -1  # left
    else:
        side = 1   # right

    # Determine start, control, end points.
    # The following rules reproduce the examples:
    #
    # 1. If the anchors are vertically separated, the curve usually ends at the
    #    top of the lower anchor on the selected side.
    # 2. The curve starts near the upper anchor. If the upper anchor is short or
    #    the gap is small, it starts above it; otherwise it starts near its
    #    middle/lower side.
    # 3. The control point bends the curve to the selected side.

    # End point: adjacent to lower anchor.
    if side == -1:
        end = (br0, bc0 - 1)
    else:
        end = (br0, bc1 + 1)

    # Start point.
    if ar1 <= br0:
        # Upper anchor is above lower anchor.
        upper_len = ar1 - ar0 + 1
        gap = br0 - ar1

        if upper_len <= 2 or gap <= 3:
            # Start above or near the upper anchor.
            start_r = max(0, ar0 - 2)
            if side == -1:
                start_c = ac0 - 1
            else:
                start_c = ac1 + 1
        else:
            # Start near the middle/lower part of the upper anchor.
            start_r = (ar0 + ar1) // 2
            if side == -1:
                start_c = ac0 - 1
            else:
                start_c = ac1 + 1

        # Control point.
        ctrl_r = (start_r + end[0]) // 2
        if side == -1:
            ctrl_c = min(ac0, bc0) - 2
        else:
            ctrl_c = max(ac1, bc1) + 2

        # Adjust control to keep the curve inside grid and reasonably close.
        if side == -1 and ac0 > bc0:
            ctrl_c = ac0 - 2
        if side == 1 and ac1 < bc1:
            ctrl_c = bc1 + 2

        start = (start_r, start_c)
        ctrl = (ctrl_r, ctrl_c)
    else:
        # Fallback: anchors overlap in rows; use a vertical-ish connector.
        if side == -1:
            start = (min(ar0, br0), min(ac0, bc0) - 1)
            end = (max(ar1, br1), min(ac0, bc0) - 1)
            ctrl = ((start[0] + end[0]) // 2, start[1] - 2)
        else:
            start = (min(ar0, br0), max(ac1, bc1) + 1)
            end = (max(ar1, br1), max(ac1, bc1) + 1)
            ctrl = ((start[0] + end[0]) // 2, start[1] + 2)

    # Clamp points to grid.
    rows, cols = grid_shape
    def clamp(pt):
        r, c = pt
        r = max(0, min(rows - 1, r))
        c = max(0, min(cols - 1, c))
        return r, c

    start = clamp(start)
    end = clamp(end)
    ctrl = clamp(ctrl)

    # Sample the curve row-wise from start to end.
    steps = abs(end[0] - start[0])
    if steps == 0:
        steps = 1

    pts = _curve_points(start, ctrl, end, steps)

    for r, c in pts:
        if 0 <= r < rows and 0 <= c < cols:
            out[r, c] = color

    return out


def _centroid(pix):
    rs = [r for r, c in pix]
    cs = [c for r, c in pix]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the inferred ARC rule.

    The rule:
    - Keep the fixed anchor objects unchanged.
    - Replace the movable foreground object by a smooth Bezier curve of the
      same color that connects the anchors.
    """
    if grid.size == 0:
        return grid.copy()

    out = grid.copy()
    rows, cols = grid.shape
    bg = np.bincount(grid.ravel()).argmax()

    comps = _components(grid)
    if len(comps) < 3:
        return out

    # Score each component as a possible anchor.
    scored = []
    for color, pix in comps:
        r0, r1, c0, c1 = _bbox(pix)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        straight = _straightness(pix)
        # Anchor-like if small/straight/point/line/block.
        # Path-like if elongated and irregular.
        anchor_score = 0
        if h <= 2 or w <= 2:
            anchor_score += 3
        if straight >= 1.0:
            anchor_score += 2
        if len(pix) <= 9:
            anchor_score += 1
        if h == 1 or w == 1:
            anchor_score += 3

        scored.append((anchor_score, color, pix, (r0, r1, c0, c1)))

    scored.sort(reverse=True)

    # Choose top two as anchors, the next one as the path to replace.
    if len(scored) < 3:
        return out

    anchors = [scored[0], scored[1]]
    path = scored[2]
    path_color = path[1]

    # Remove the path component from output.
    for r, c in path[2]:
        out[r, c] = bg

    # Keep anchors unchanged (they already are).
    # Draw the new curve.
    out = _draw_curve(out, path_color, anchors, path_color, (rows, cols))

    return out