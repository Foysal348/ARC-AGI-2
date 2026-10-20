import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find the frame component that encloses background cells, crop its bounding
    box, and fill each enclosed background hole with the color of the best
    matching external shape.

    Matching is by exact shape up to rotation. Reflection is used only as a
    fallback if no rotated match exists. If several colors match the same hole
    shape, the color with the largest total number of matching cells is chosen;
    ties are broken by more matching components, then by smaller color value.
    Holes with no match are left as background.
    """
    grid = np.array(grid, dtype=int)

    if grid.ndim != 2:
        return grid

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # -------------------------------------------------------------------------
    # 1. Background color: most common color on the border, with overall count
    #    as a tie-breaker.
    # -------------------------------------------------------------------------
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    bvals, bcounts = np.unique(border, return_counts=True)
    border_counts = {int(v): int(c) for v, c in zip(bvals, bcounts)}

    ovals, ocounts = np.unique(grid, return_counts=True)
    overall_counts = {int(v): int(c) for v, c in zip(ovals, ocounts)}

    bg = min(
        (int(v) for v in bvals),
        key=lambda c: (-border_counts[c], -overall_counts.get(c, 0), c)
    )

    # -------------------------------------------------------------------------
    # 2. Flood-fill background from the border. Remaining background cells are
    #    enclosed holes.
    # -------------------------------------------------------------------------
    exterior = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == bg and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == bg and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not exterior[nr, nc]
                and grid[nr, nc] == bg
            ):
                exterior[nr, nc] = True
                q.append((nr, nc))

    enclosed_bg = (grid == bg) & (~exterior)

    # -------------------------------------------------------------------------
    # 3. Label all non-background connected components.
    # -------------------------------------------------------------------------
    label = -np.ones((h, w), dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg and label[r, c] == -1:
                color = int(grid[r, c])
                cid = len(components)
                label[r, c] = cid

                stack = [(r, c)]
                cells = []
                rmin, rmax = r, r
                cmin, cmax = c, c

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    if cr < rmin:
                        rmin = cr
                    if cr > rmax:
                        rmax = cr
                    if cc < cmin:
                        cmin = cc
                    if cc > cmax:
                        cmax = cc

                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and label[nr, nc] == -1
                            and grid[nr, nc] == color
                        ):
                            label[nr, nc] = cid
                            stack.append((nr, nc))

                components.append((color, cells, (rmin, rmax, cmin, cmax), cid))

    if not components:
        return grid.copy()

    # -------------------------------------------------------------------------
    # 4. Choose the frame: the non-background component whose bounding box
    #    contains the most enclosed background cells.
    # -------------------------------------------------------------------------
    best_frame = None
    best_score = None

    for color, cells, (r0, r1, c0, c1), cid in components:
        enclosed = int(enclosed_bg[r0:r1 + 1, c0:c1 + 1].sum())
        bg_inside = int((grid[r0:r1 + 1, c0:c1 + 1] == bg).sum())
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)

        score = (
            1 if enclosed > 0 else 0,
            enclosed,
            bg_inside,
            bbox_area,
            len(cells)
        )

        if best_score is None or score > best_score:
            best_score = score
            best_frame = (color, cells, (r0, r1, c0, c1), cid)

    if best_frame is None or best_score[0] == 0:
        return grid.copy()

    frame_color, frame_cells, (r0, r1, c0, c1), frame_id = best_frame
    out_h = r1 - r0 + 1
    out_w = c1 - c0 + 1

    # -------------------------------------------------------------------------
    # 5. Extract hole components inside the frame bounding box.
    # -------------------------------------------------------------------------
    hole_mask = enclosed_bg[r0:r1 + 1, c0:c1 + 1]
    holes = []

    if hole_mask.any():
        hole_visited = np.zeros((out_h, out_w), dtype=bool)

        for r in range(out_h):
            for c in range(out_w):
                if hole_mask[r, c] and not hole_visited[r, c]:
                    stack = [(r, c)]
                    hole_visited[r, c] = True
                    cells = []

                    while stack:
                        cr, cc = stack.pop()
                        cells.append((cr, cc))

                        for dr, dc in dirs:
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < out_h
                                and 0 <= nc < out_w
                                and hole_mask[nr, nc]
                                and not hole_visited[nr, nc]
                            ):
                                hole_visited[nr, nc] = True
                                stack.append((nr, nc))

                    holes.append(cells)

    # -------------------------------------------------------------------------
    # 6. Shape canonicalization under rotation (and optional reflection).
    # -------------------------------------------------------------------------
    def normalize(coords):
        coords = tuple(coords)
        if not coords:
            return ()
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        return tuple(sorted((r - min_r, c - min_c) for r, c in coords))

    def rotate90(coords):
        coords = tuple(coords)
        if not coords:
            return ()
        height = max(r for r, _ in coords) + 1
        return normalize((c, height - 1 - r) for r, c in coords)

    def mirror(coords):
        coords = tuple(coords)
        if not coords:
            return ()
        width = max(c for _, c in coords) + 1
        return normalize((r, width - 1 - c) for r, c in coords)

    def canonical_shape(coords, reflect=False):
        base = normalize(coords)
        if not base:
            return ()

        variants = [base]
        if reflect:
            variants.append(mirror(base))

        best_key = None
        for variant in variants:
            cur = variant
            for _ in range(4):
                if best_key is None or cur < best_key:
                    best_key = cur
                cur = rotate90(cur)

        return best_key

    def pick_color(color_stats):
        if not color_stats:
            return None
        return max(
            color_stats.items(),
            key=lambda kv: (kv[1][0], kv[1][1], -kv[0])
        )[0]

    # -------------------------------------------------------------------------
    # 7. Candidate pieces: non-background components outside the frame bbox,
    #    excluding the frame color itself.
    # -------------------------------------------------------------------------
    candidates = []

    for color, cells, _, cid in components:
        if cid == frame_id or color == bg or color == frame_color:
            continue

        inside_frame = False
        for rr, cc in cells:
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                inside_frame = True
                break

        if not inside_frame:
            candidates.append((color, cells))

    # Aggregate matching stats by canonical shape.
    stats_rot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    stats_ref = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for color, cells in candidates:
        n = len(cells)

        key_rot = canonical_shape(cells, reflect=False)
        stats_rot[key_rot][color][0] += n
        stats_rot[key_rot][color][1] += 1

        key_ref = canonical_shape(cells, reflect=True)
        stats_ref[key_ref][color][0] += n
        stats_ref[key_ref][color][1] += 1

    # -------------------------------------------------------------------------
    # 8. Build output: frame color base, holes initially background, then fill
    #    matched holes.
    # -------------------------------------------------------------------------
    out = np.full((out_h, out_w), frame_color, dtype=grid.dtype)
    out[hole_mask] = bg

    for cells in holes:
        chosen = None

        key_rot = canonical_shape(cells, reflect=False)
        if key_rot in stats_rot:
            chosen = pick_color(stats_rot[key_rot])

        if chosen is None:
            key_ref = canonical_shape(cells, reflect=True)
            if key_ref in stats_ref:
                chosen = pick_color(stats_ref[key_ref])

        if chosen is not None:
            for rr, cc in cells:
                out[rr, cc] = chosen

    return out