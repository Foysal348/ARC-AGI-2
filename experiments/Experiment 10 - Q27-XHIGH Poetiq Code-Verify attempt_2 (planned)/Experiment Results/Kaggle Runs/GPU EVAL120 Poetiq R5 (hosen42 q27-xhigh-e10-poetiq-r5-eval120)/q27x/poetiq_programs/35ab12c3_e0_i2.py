import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete multi-pixel color shapes and copy adjacent single-pixel strokes.

    Rules:
    1. For every color with at least two input pixels:
       - Compute the convex hull of that color's pixels.
       - Draw only hull edges that are horizontal, vertical, or 45-degree diagonal.
    2. For every color with exactly one input pixel:
       - If it is adjacent to a completed stroke component of another color,
         shift that component by the offset from the neighboring stroke cell
         to the singleton cell.
       - Draw the shifted component only on cells that are empty or already
         the singleton color.

    Existing non-zero pixels are never overwritten by other colors.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    h, w = grid.shape
    out = grid.copy()

    if h == 0 or w == 0:
        return out

    # Collect input pixels by color.
    points_by_color = {}
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v != 0:
                points_by_color.setdefault(v, []).append((r, c))

    # ------------------------------------------------------------------
    # Convex hull helpers.
    # Points are stored as (row, col), but hull geometry uses x=col, y=row.
    # ------------------------------------------------------------------
    def cross(o, a, b):
        """Cross product for points (row, col) interpreted as x=col, y=row."""
        return (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1])

    def convex_hull(pts):
        """Monotone chain convex hull. Returns hull vertices as (row, col)."""
        pts_sorted = sorted(set(pts), key=lambda p: (p[1], p[0]))
        if len(pts_sorted) <= 1:
            return pts_sorted

        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return hull

    def is_straight(p, q):
        """True if p-q is horizontal, vertical, or 45-degree diagonal."""
        dr = abs(p[0] - q[0])
        dc = abs(p[1] - q[1])
        return dr == 0 or dc == 0 or dr == dc

    def segment_cells(p, q):
        """Yield all grid cells on a straight horizontal/vertical/45-degree segment."""
        r1, c1 = p
        r2, c2 = q

        if r1 == r2:
            step = 1 if c2 >= c1 else -1
            for c in range(c1, c2 + step, step):
                yield (r1, c)
        elif c1 == c2:
            step = 1 if r2 >= r1 else -1
            for r in range(r1, r2 + step, step):
                yield (r, c1)
        elif abs(r2 - r1) == abs(c2 - c1):
            step_r = 1 if r2 >= r1 else -1
            step_c = 1 if c2 >= c1 else -1
            n = abs(r2 - r1)
            for t in range(n + 1):
                yield (r1 + step_r * t, c1 + step_c * t)
        else:
            return

    def draw_cells(cells, color):
        """Draw cells, never overwriting existing non-zero pixels."""
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                if out[r, c] == 0 or out[r, c] == color:
                    out[r, c] = color

    # ------------------------------------------------------------------
    # Stage 1: compute and draw completed primary strokes.
    # ------------------------------------------------------------------
    stroke_masks = {}

    for color in sorted(points_by_color):
        pts = points_by_color[color]
        if len(pts) < 2:
            continue

        cells = set()
        hull = convex_hull(pts)

        if len(hull) >= 2:
            n = len(hull)
            for i in range(n):
                p = hull[i]
                q = hull[(i + 1) % n]
                if is_straight(p, q):
                    cells.update(segment_cells(p, q))

        stroke_masks[color] = cells

    # Draw all primary strokes.
    for color in sorted(stroke_masks):
        draw_cells(stroke_masks[color], color)

    # ------------------------------------------------------------------
    # Split each stroke mask into connected components (8-neighbor).
    # This lets a singleton copy only the specific stroke it touches.
    # ------------------------------------------------------------------
    def connected_components(cells):
        remaining = set(cells)
        comps = []

        while remaining:
            start = remaining.pop()
            comp = {start}
            stack = [start]

            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in remaining:
                            remaining.remove((nr, nc))
                            comp.add((nr, nc))
                            stack.append((nr, nc))

            comps.append(comp)

        return comps

    color_cell_to_comp = {}
    for color, mask in stroke_masks.items():
        comp_map = {}
        if mask:
            comps = connected_components(mask)
            for comp in comps:
                for cell in comp:
                    comp_map[cell] = comp
        color_cell_to_comp[color] = comp_map

    # ------------------------------------------------------------------
    # Stage 2: handle singleton colors by copying an adjacent stroke.
    # ------------------------------------------------------------------
    for color in sorted(points_by_color):
        pts = points_by_color[color]
        if len(pts) != 1:
            continue

        r, c = pts[0]
        candidates = []

        def consider_neighbor(nr, nc, is_input, dr, dc):
            base = int(grid[nr, nc]) if is_input else int(out[nr, nc])

            if base == 0 or base == color:
                return

            comp = color_cell_to_comp.get(base, {}).get((nr, nc))
            if comp is None:
                return

            # Count how many shifted cells would actually be visible.
            new_count = 0
            for pr, pc in comp:
                rr, cc = pr + dr, pc + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if out[rr, cc] == 0 or out[rr, cc] == color:
                        new_count += 1

            is_4 = 1 if (dr == 0 or dc == 0) else 0

            # Prefer:
            # 1. input-neighbor over already-drawn-neighbor,
            # 2. 4-neighbor over diagonal neighbor,
            # 3. more visible shifted cells,
            # 4. larger stroke component,
            # 5. deterministic color/vector tie-breaking.
            score = (
                1 if is_input else 0,
                is_4,
                new_count,
                len(comp),
                -base,
                dr,
                dc,
            )
            candidates.append((score, base, (dr, dc), comp))

        # First, try neighbors that exist in the original input.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    consider_neighbor(nr, nc, True, dr, dc)

        # Fallback: if the singleton is adjacent only to a completed stroke
        # cell (not an original input pixel), use the current grid state.
        if not candidates:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        consider_neighbor(nr, nc, False, dr, dc)

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, _, vector, comp = candidates[0]
            shifted = [(pr + vector[0], pc + vector[1]) for pr, pc in comp]
            draw_cells(shifted, color)

    return out