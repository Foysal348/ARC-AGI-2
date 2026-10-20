import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extend interior colors outward through openings in their frame boundary.

    The algorithm:
      1. Detect the background color as the most frequent color.
      2. Detect the frame color as the non-background color that most often
         touches both background and other non-background colors.
      3. Treat all remaining non-background colors as interior colors.
      4. For each 4-connected interior component:
           - take its bounding box and expand it by 1 cell;
           - background cells on that expanded boundary are openings;
           - side openings emit a straight perpendicular ray;
           - corner openings emit a diagonal outward ray.
      5. Rays are drawn in top-to-bottom, left-to-right order and stop before
         overwriting any non-background cell already present.
    """
    if grid is None or grid.size == 0:
        return grid.copy()

    grid = np.asarray(grid)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    out = grid.copy()

    # ------------------------------------------------------------
    # Color roles
    # ------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg_colors = [int(v) for v in vals if int(v) != bg]
    if not nonbg_colors:
        return out

    # Choose the frame color.  A good frame color touches both background
    # and interior colors.  Frequency is used as a small tie-breaker.
    frame = int(nonbg_colors[0])
    best_score = -1

    for f in nonbg_colors:
        f = int(f)
        mask = grid == f
        freq = int(np.sum(mask))

        adj_bg = 0
        adj_other = 0

        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nv = int(grid[nr, nc])
                    if nv == bg:
                        adj_bg += 1
                    elif nv != f:
                        adj_other += 1

        score = adj_bg + adj_other + freq / 10000.0
        if score > best_score:
            best_score = score
            frame = f

    interior_colors = [int(v) for v in nonbg_colors if int(v) != frame]
    if not interior_colors:
        return out

    # ------------------------------------------------------------
    # Find interior components
    # ------------------------------------------------------------
    components = []

    for color in interior_colors:
        color = int(color)
        mask = grid == color
        visited = np.zeros((H, W), dtype=bool)

        for r in range(H):
            for c in range(W):
                if not mask[r, c] or visited[r, c]:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                pixels = []

                while stack:
                    pr, pc = stack.pop()
                    pixels.append((pr, pc))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                if pixels:
                    rs = [p[0] for p in pixels]
                    cs = [p[1] for p in pixels]
                    components.append({
                        "color": color,
                        "pixels": pixels,
                        "rmin": min(rs),
                        "rmax": max(rs),
                        "cmin": min(cs),
                        "cmax": max(cs),
                    })

    # Draw earlier components first: top to bottom, then left to right.
    components.sort(
        key=lambda comp: (
            comp["rmin"] + comp["rmax"],
            comp["cmin"] + comp["cmax"],
            comp["color"],
        )
    )

    # ------------------------------------------------------------
    # Ray shooting helper
    # ------------------------------------------------------------
    def shoot(r: int, c: int, dr: int, dc: int, color: int) -> None:
        while 0 <= r < H and 0 <= c < W:
            if out[r, c] != bg:
                break
            out[r, c] = color
            r += dr
            c += dc

    # ------------------------------------------------------------
    # Process each component
    # ------------------------------------------------------------
    for comp in components:
        color = comp["color"]

        rmin = comp["rmin"]
        rmax = comp["rmax"]
        cmin = comp["cmin"]
        cmax = comp["cmax"]

        # One-cell expanded boundary, clipped to the grid.
        r0 = max(0, rmin - 1)
        r1 = min(H - 1, rmax + 1)
        c0 = max(0, cmin - 1)
        c1 = min(W - 1, cmax + 1)

        # ----- corner openings -----
        corners = set()
        for rr in (r0, r1):
            for cc in (c0, c1):
                if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == bg:
                    corners.add((rr, cc))

        for rr, cc in sorted(corners):
            dr = 0
            dc = 0

            if rr == r0 and r0 != r1:
                dr = -1
            elif rr == r1 and r0 != r1:
                dr = 1

            if cc == c0 and c0 != c1:
                dc = -1
            elif cc == c1 and c0 != c1:
                dc = 1

            if dr != 0 or dc != 0:
                shoot(rr, cc, dr, dc, color)

        # ----- side openings -----
        # Helper to process one straight side segment.
        # coords are the varying coordinates of background cells on that side.
        def process_side(coords, fixed, dr, dc):
            if not coords:
                return

            runs = []
            current = [coords[0]]

            for x in coords[1:]:
                if x == current[-1] + 1:
                    current.append(x)
                else:
                    runs.append(current)
                    current = [x]
            runs.append(current)

            for run in runs:
                # A single isolated side cell is the cleanest opening.
                # For a longer isolated side gap, use its middle cell.
                if len(run) == 1:
                    x = run[0]
                else:
                    x = run[len(run) // 2]

                if dr == -1 or dr == 1:  # top / bottom side
                    shoot(fixed, x, dr, dc, color)
                else:                    # left / right side
                    shoot(x, fixed, dr, dc, color)

        # Top side, excluding corners.
        if r0 < r1 and c0 + 1 < c1:
            top_coords = []
            for cc in range(c0 + 1, c1):
                if grid[r0, cc] == bg:
                    top_coords.append(cc)
            process_side(top_coords, r0, -1, 0)

        # Bottom side, excluding corners.
        if r0 < r1 and c0 + 1 < c1:
            bottom_coords = []
            for cc in range(c0 + 1, c1):
                if grid[r1, cc] == bg:
                    bottom_coords.append(cc)
            process_side(bottom_coords, r1, 1, 0)

        # Left side, excluding corners.
        if c0 < c1 and r0 + 1 < r1:
            left_coords = []
            for rr in range(r0 + 1, r1):
                if grid[rr, c0] == bg:
                    left_coords.append(rr)
            process_side(left_coords, c0, 0, -1)

        # Right side, excluding corners.
        if c0 < c1 and r0 + 1 < r1:
            right_coords = []
            for rr in range(r0 + 1, r1):
                if grid[rr, c1] == bg:
                    right_coords.append(rr)
            process_side(right_coords, c1, 0, 1)

    return out