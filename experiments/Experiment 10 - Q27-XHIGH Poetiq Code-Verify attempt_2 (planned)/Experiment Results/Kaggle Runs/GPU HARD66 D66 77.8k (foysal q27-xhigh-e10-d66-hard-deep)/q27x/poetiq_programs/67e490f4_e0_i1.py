import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a large frame with matching shapes found elsewhere
    in the grid.

    The frame is chosen as the non-background connected component whose
    bounding box contains the most enclosed background cells.  Each enclosed
    background component inside the frame is treated as a hole.  For each hole,
    all connected non-background components outside the frame are compared
    by shape (up to rotation, with reflection fallback).  The color chosen is
    the color with the greatest total cell count among matching shapes;
    ties are broken by the smaller color value.
    """
    grid = np.array(grid, dtype=int)

    if grid.ndim != 2:
        return grid

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid

    # Background: most frequent color.  On a tie, prefer colors seen on border.
    vals, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    bg_candidates = vals[counts == max_count].tolist()

    if len(bg_candidates) == 1:
        bg = int(bg_candidates[0])
    else:
        border = np.concatenate([
            grid[0, :],
            grid[-1, :],
            grid[:, 0],
            grid[:, -1]
        ])
        bvals, bcounts = np.unique(border, return_counts=True)
        border_counts = {int(v): int(c) for v, c in zip(bvals, bcounts)}
        bg = min(bg_candidates, key=lambda c: (-border_counts.get(int(c), 0), int(c)))

    # Flood fill background from the grid border.  Background cells not reached
    # are enclosed background cells.
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    def try_add_border_cell(r, c):
        if 0 <= r < h and 0 <= c < w and grid[r, c] == bg and not visited[r, c]:
            visited[r, c] = True
            q.append((r, c))

    for r in range(h):
        try_add_border_cell(r, 0)
        try_add_border_cell(r, w - 1)

    for c in range(w):
        try_add_border_cell(0, c)
        try_add_border_cell(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h and 0 <= nc < w
                and not visited[nr, nc]
                and grid[nr, nc] == bg
            ):
                visited[nr, nc] = True
                q.append((nr, nc))

    enclosed_bg = (grid == bg) & (~visited)

    # Label all non-background connected components.
    comp_id = -np.ones((h, w), dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg and comp_id[r, c] == -1:
                color = int(grid[r, c])
                cid = len(components)
                comp_id[r, c] = cid
                stack = [(r, c)]
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h and 0 <= nc < w
                            and comp_id[nr, nc] == -1
                            and grid[nr, nc] == color
                        ):
                            comp_id[nr, nc] = cid
                            stack.append((nr, nc))

                rmin = min(rr for rr, _ in cells)
                rmax = max(rr for rr, _ in cells)
                cmin = min(cc for _, cc in cells)
                cmax = max(cc for _, cc in cells)

                components.append((color, cells, (rmin, rmax, cmin, cmax)))

    if not components:
        return grid.copy()

    # Choose the frame component: maximize enclosed background inside its bbox.
    best_frame = None
    best_score = None

    for color, cells, (r0, r1, c0, c1) in components:
        enclosed_count = int(enclosed_bg[r0:r1 + 1, c0:c1 + 1].sum())
        bg_inside = int((grid[r0:r1 + 1, c0:c1 + 1] == bg).sum())
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)

        if enclosed_count == 0 and bg_inside == 0:
            continue

        score = (
            1 if enclosed_count > 0 else 0,
            enclosed_count,
            bg_inside,
            bbox_area,
            len(cells)
        )

        if best_score is None or score > best_score:
            best_score = score
            best_frame = (color, cells, (r0, r1, c0, c1))

    if best_frame is None:
        return grid.copy()

    frame_color, _, (r0, r1, c0, c1) = best_frame
    out_h = r1 - r0 + 1
    out_w = c1 - c0 + 1

    # Extract hole components inside the frame bbox.
    hole_mask = enclosed_bg[r0:r1 + 1, c0:c1 + 1].copy()
    if not hole_mask.any():
        hole_mask = (grid[r0:r1 + 1, c0:c1 + 1] == bg)

    holes = []
    hole_visited = np.zeros(hole_mask.shape, dtype=bool)

    for r in range(out_h):
        for c in range(out_w):
            if hole_mask[r, c] and not hole_visited[r, c]:
                stack = [(r, c)]
                hole_visited[r, c] = True
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < out_h and 0 <= nc < out_w
                            and hole_mask[nr, nc]
                            and not hole_visited[nr, nc]
                        ):
                            hole_visited[nr, nc] = True
                            stack.append((nr, nc))

                holes.append(cells)

    # Candidate shapes: non-background, non-frame-color components outside frame.
    candidates = []
    for color, cells, (cr0, cr1, cc0, cc1) in components:
        if color == bg or color == frame_color:
            continue

        # Exclude components whose bbox overlaps the frame bbox.
        if cr0 <= r1 and cr1 >= r0 and cc0 <= c1 and cc1 >= c0:
            continue

        candidates.append((color, cells))

    def normalize(coords):
        if not coords:
            return ()
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        return tuple(sorted((r - min_r, c - min_c) for r, c in coords))

    def rotate90(coords):
        if not coords:
            return ()
        height = max(r for r, _ in coords) + 1
        return normalize((c, height - 1 - r) for r, c in coords)

    def canonical_shape(coords, reflect=False):
        """Return a canonical key for a shape under rotations (and optionally reflections)."""
        s = set(coords)
        if not s:
            return ()

        variants = [s]
        if reflect:
            width = max(c for _, c in s) + 1
            variants.append(set((r, width - 1 - c) for r, c in s))

        best = None
        for variant in variants:
            cur = normalize(variant)
            for _ in range(4):
                if best is None or cur < best:
                    best = cur
                cur = rotate90(cur)

        return best

    def pick_color(color_stats):
        """Choose color by total cells, then component count, then smaller color."""
        if not color_stats:
            return None
        return max(
            color_stats.items(),
            key=lambda item: (item[1][0], item[1][1], -item[0])
        )[0]

    # Aggregate candidate shapes.
    stats_rot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    stats_rot_ref = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    area_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    overall_stats = defaultdict(lambda: [0, 0])

    for color, cells in candidates:
        n = len(cells)

        key_rot = canonical_shape(cells, reflect=False)
        stats_rot[key_rot][color][0] += n
        stats_rot[key_rot][color][1] += 1

        key_rot_ref = canonical_shape(cells, reflect=True)
        stats_rot_ref[key_rot_ref][color][0] += n
        stats_rot_ref[key_rot_ref][color][1] += 1

        area_stats[n][color][0] += n
        area_stats[n][color][1] += 1

        overall_stats[color][0] += n
        overall_stats[color][1] += 1

    # Build output.
    output = np.full((out_h, out_w), frame_color, dtype=grid.dtype)

    for hole_cells in holes:
        chosen_color = None

        key_rot = canonical_shape(hole_cells, reflect=False)
        if key_rot in stats_rot:
            chosen_color = pick_color(stats_rot[key_rot])

        if chosen_color is None:
            key_rot_ref = canonical_shape(hole_cells, reflect=True)
            if key_rot_ref in stats_rot_ref:
                chosen_color = pick_color(stats_rot_ref[key_rot_ref])

        # Fallbacks if exact shape match is unavailable.
        if chosen_color is None and len(hole_cells) in area_stats:
            chosen_color = pick_color(area_stats[len(hole_cells)])

        if chosen_color is None and overall_stats:
            chosen_color = pick_color(overall_stats)

        if chosen_color is None:
            chosen_color = frame_color

        for r, c in hole_cells:
            output[r, c] = chosen_color

    return output