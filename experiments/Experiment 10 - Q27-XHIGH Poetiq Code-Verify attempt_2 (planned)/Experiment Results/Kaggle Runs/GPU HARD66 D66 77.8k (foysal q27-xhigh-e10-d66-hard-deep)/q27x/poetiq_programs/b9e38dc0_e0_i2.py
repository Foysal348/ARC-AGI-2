import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC transform inferred from the examples:

    * The most frequent color is the background.
    * The most frequent non-background color is the boundary/wall color.
    * Another small color is the "paint" color that must spread.
    * Remaining non-background colors are solid obstacles.

    The paint spreads from its initial pixels into background cells.
    The spread is directional:
        - Direction D is from the paint centroid toward the wall centroid.
        - The paint may move in direction D and laterally, but not opposite D.
    Walls and obstacles block the spread.

    In addition, the paint leaks through the wall opening that faces D.
    The leak is approximated by projecting the opening endpoints of the wall
    to the D edge and filling the resulting trapezoidal projection.

    Obstacles cast a straight shadow in direction D; shadowed background
    cells are not painted.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # ------------------------------------------------------------------
    # Color identification
    # ------------------------------------------------------------------
    color_counts = Counter(grid.ravel().tolist())
    bg = color_counts.most_common(1)[0][0]

    non_bg = {c: n for c, n in color_counts.items() if c != bg}
    if not non_bg:
        return grid.copy()

    wall = max(non_bg, key=lambda c: non_bg[c])

    # Paint color: prefer the color whose centroid is on the side of the wall
    # that points toward a grid edge, and which is not the wall.
    wall_coords = np.argwhere(grid == wall)
    wall_cy, wall_cx = wall_coords[:, 0].mean(), wall_coords[:, 1].mean()

    candidate_colors = [c for c in non_bg if c != wall]
    if not candidate_colors:
        return grid.copy()

    # Choose the candidate closest to the wall but not the wall.
    def centroid(c):
        ys, xs = np.argwhere(grid == c).T
        return ys.mean(), xs.mean()

    paint = None
    best_score = None
    for c in candidate_colors:
        cy, cx = centroid(c)
        d = abs(cy - wall_cy) + abs(cx - wall_cx)
        # Prefer smaller count if distances are similar, but main signal is distance.
        score = (-d, -color_counts[c])
        if best_score is None or score < best_score:
            best_score = score
            paint = c

    # ------------------------------------------------------------------
    # Direction: paint -> wall, dominant cardinal direction.
    # ------------------------------------------------------------------
    py, px = centroid(paint)
    dy = wall_cy - py
    dx = wall_cx - px

    if abs(dy) >= abs(dx):
        # up or down
        D = (0, 1) if dy > 0 else (0, -1)
    else:
        # left or right
        D = (1, 0) if dx > 0 else (-1, 0)

    # Lateral directions perpendicular to D.
    if D == (0, 1):       # down
        laterals = [(-1, 0), (1, 0)]
    elif D == (0, -1):    # up
        laterals = [(-1, 0), (1, 0)]
    elif D == (1, 0):     # right
        laterals = [(0, -1), (0, 1)]
    else:                 # left
        laterals = [(0, -1), (0, 1)]

    moves = [D] + laterals

    # ------------------------------------------------------------------
    # Build blocked mask: walls and obstacles.  Paint pixels are passable.
    # ------------------------------------------------------------------
    blocked = np.zeros((h, w), dtype=bool)
    for c in non_bg:
        if c != paint:
            blocked |= (grid == c)

    # ------------------------------------------------------------------
    # Shadows cast by obstacles in direction D.
    # ------------------------------------------------------------------
    shadow = np.zeros((h, w), dtype=bool)
    obstacle_colors = [c for c in non_bg if c not in (wall, paint)]

    for c in obstacle_colors:
        for r, col in zip(*np.argwhere(grid == c)):
            nr, nc = r + D[0], col + D[1]
            while 0 <= nr < h and 0 <= nc < w:
                if shadow[nr, nc]:
                    break
                shadow[nr, nc] = True
                if blocked[nr, nc]:
                    break
                nr += D[0]
                nc += D[1]

    blocked_for_flood = blocked | shadow

    # ------------------------------------------------------------------
    # Directional flood fill from paint pixels.
    # ------------------------------------------------------------------
    filled = np.zeros((h, w), dtype=bool)
    q = deque()

    for r, col in zip(*np.argwhere(grid == paint)):
        filled[r, col] = True
        q.append((r, col))

    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if not filled[nr, nc] and not blocked_for_flood[nr, nc]:
                    filled[nr, nc] = True
                    q.append((nr, nc))

    # ------------------------------------------------------------------
    # Find the wall opening endpoints: wall pixels with 8-neighbor degree 1.
    # Use them to add a projected leak region in direction D.
    # ------------------------------------------------------------------
    wall_mask = (grid == wall)
    endpoints = []

    for r, c in zip(*np.argwhere(wall_mask)):
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and wall_mask[nr, nc]:
                    deg += 1
        if deg <= 1:
            endpoints.append((r, c))

    # If there are at least two endpoints, choose the pair whose midpoint is
    # furthest from the paint in direction D (the opening facing D).
    if len(endpoints) >= 2:
        def opening_score(e):
            cy, cx = (e[0] + e[1][0]) / 2.0, (e[1][1] + e[3][1]) / 2.0
            # Project difference from paint center onto D.
            if D == (0, 1):
                return cy - py
            if D == (0, -1):
                return py - cy
            if D == (1, 0):
                return cx - px
            return px - cx

        best_pair = None
        best_val = None
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                val = opening_score(endpoints[i]) + opening_score(endpoints[j])
                if best_val is None or val > best_val:
                    best_val = val
                    best_pair = (endpoints[i], endpoints[j])

        # Add the trapezoidal projection of the opening to the D edge.
        p1, p2 = best_pair

        if D == (0, 1):   # down
            # Project endpoints to bottom edge.
            def project_col(r1, c1, r2, c2, target_r):
                if r2 == r1:
                    return c1
                t = (target_r - r1) / (r2 - r1)
                return c1 + t * (c2 - c1)

            # Use a pseudo-second point one row down to create extension.
            # Simpler: for every row below the endpoints, linearly expand
            # between the two endpoint columns.
            base_r = max(p1[0], p2[0])
            c1, c2 = min(p1[1], p2[1]), max(p1[1], p2[1])

            for r in range(base_r, h):
                # Expand by one cell per two rows, but never beyond grid.
                spread = (r - base_r) // 2
                left = max(0, c1 - spread)
                right = min(w - 1, c2 + spread)
                for c in range(left, right + 1):
                    if grid[r, c] == bg and not blocked[r, c] and not shadow[r, c]:
                        filled[r, c] = True

        elif D == (0, -1): # up
            base_r = min(p1[0], p2[0])
            c1, c2 = min(p1[1], p2[1]), max(p1[1], p2[1])

            for r in range(base_r, -1, -1):
                spread = (base_r - r) // 2
                left = max(0, c1 - spread)
                right = min(w - 1, c2 + spread)
                for c in range(left, right + 1):
                    if grid[r, c] == bg and not blocked[r, c] and not shadow[r, c]:
                        filled[r, c] = True

        elif D == (1, 0):  # right
            base_c = max(p1[1], p2[1])
            r1, r2 = min(p1[0], p2[0]), max(p1[0], p2[1]) if False else max(p1[0], p2[0])

            for c in range(base_c, w):
                spread = (c - base_c) // 2
                top = max(0, r1 - spread)
                bottom = min(h - 1, r2 + spread)
                for r in range(top, bottom + 1):
                    if grid[r, c] == bg and not blocked[r, c] and not shadow[r, c]:
                        filled[r, c] = True

        else:              # left
            base_c = min(p1[1], p2[1])
            r1, r2 = min(p1[0], p2[0]), max(p1[0], p2[0])

            for c in range(base_c, -1, -1):
                spread = (base_c - c) // 2
                top = max(0, r1 - spread)
                bottom = min(h - 1, r2 + spread)
                for r in range(top, bottom + 1):
                    if grid[r, c] == bg and not blocked[r, c] and not shadow[r, c]:
                        filled[r, c] = True

    # ------------------------------------------------------------------
    # Write output: paint only background cells in the filled mask.
    # ------------------------------------------------------------------
    out = grid.copy()
    out[filled & (grid == bg)] = paint
    return out