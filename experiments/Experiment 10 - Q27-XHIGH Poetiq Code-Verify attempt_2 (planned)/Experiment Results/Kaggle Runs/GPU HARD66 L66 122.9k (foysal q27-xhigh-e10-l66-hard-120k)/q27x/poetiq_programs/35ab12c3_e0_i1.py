import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. Treat every non-zero cell as a coloured dot.
    2. For each colour with at least two dots, connect dots that are aligned in
       one of the 8 grid directions (horizontal, vertical, or 45-degree
       diagonal). For every dot, only its two closest aligned same-colour dots
       are considered, so the resulting shapes are the visible orthodiagonal
       polylines / polygons in the examples.
    3. A single isolated dot casts an offset copy of the nearest aligned
       multi-dot colour's polyline(s). The offset vector is from the anchor
       dot to the isolated dot.
    4. Offset copies are drawn first and do not overwrite existing dots.
       The original colour's segments are drawn afterwards, so they take
       priority where copies overlap.

    This reproduces the main behaviour in the provided examples and handles
    the challenge input by drawing the 3-polygon, the 6/1 vertical lines,
    the offset 4 and 7 vertical lines, and leaving non-aligned singleton /
    non-aligned multi-dot clusters unchanged.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    out = g.copy()

    # Collect points by colour.
    points = {}
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v != 0:
                points.setdefault(v, []).append((r, c))

    def aligned(a, b):
        dr = a[0] - b[0]
        dc = a[1] - b[1]
        return dr == 0 or dc == 0 or abs(dr) == abs(dc)

    def dist8(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def draw_segment(a, b, color, overwrite=True):
        """Draw an 8-connected segment if it is horizontal, vertical, or 45-degree."""
        if not aligned(a, b):
            return

        r1, c1 = a
        r2, c2 = b

        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)
        steps = max(abs(r2 - r1), abs(c2 - c1))

        r, c = r1, c1
        for _ in range(steps + 1):
            if 0 <= r < H and 0 <= c < W:
                if overwrite or out[r, c] == 0 or out[r, c] == color:
                    out[r, c] = color
            r += dr
            c += dc

    # Build the main segments for every colour that has at least two aligned dots.
    primary_segments = {}
    for color, pts in points.items():
        if len(pts) < 2:
            continue

        segs = set()
        for i, p in enumerate(pts):
            candidates = []
            for j, q in enumerate(pts):
                if i != j and aligned(p, q):
                    candidates.append((dist8(p, q), j))

            if not candidates:
                continue

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, j in candidates[:2]:
                a, b = pts[i], pts[j]
                key = (a, b) if a <= b else (b, a)
                segs.add(key)

        if segs:
            primary_segments[color] = list(segs)

    def component_segments(pts, segs, start):
        """Return only the segments in the connected component containing start."""
        if start not in pts:
            return []

        adj = {p: [] for p in pts}
        for a, b in segs:
            adj[a].append(b)
            adj[b].append(a)

        if not adj[start]:
            return []

        stack = [start]
        comp = set()
        while stack:
            p = stack.pop()
            if p in comp:
                continue
            comp.add(p)
            for nb in adj[p]:
                if nb not in comp:
                    stack.append(nb)

        return [seg for seg in segs if seg[0] in comp and seg[1] in comp]

    # Handle singleton colours by offset-copying the nearest aligned multi-dot colour.
    for color, pts in points.items():
        if len(pts) != 1:
            continue

        p = pts[0]
        candidates = []

        for other_color, other_pts in points.items():
            if other_color == color or other_color not in primary_segments:
                continue
            for q in other_pts:
                if aligned(p, q):
                    candidates.append((dist8(p, q), other_color, q))

        if not candidates:
            continue

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        for _, anchor_color, anchor_pt in candidates:
            comp_segs = component_segments(
                points[anchor_color],
                primary_segments[anchor_color],
                anchor_pt
            )
            if not comp_segs:
                continue

            vr = p[0] - anchor_pt[0]
            vc = p[1] - anchor_pt[1]

            for a, b in comp_segs:
                ta = (a[0] + vr, a[1] + vc)
                tb = (b[0] + vr, b[1] + vc)
                draw_segment(ta, tb, color, overwrite=False)
            break

    # Redraw primary colour segments last so they have priority over offset copies.
    for color in sorted(primary_segments.keys(), reverse=True):
        for a, b in primary_segments[color]:
            draw_segment(a, b, color, overwrite=True)

    return out