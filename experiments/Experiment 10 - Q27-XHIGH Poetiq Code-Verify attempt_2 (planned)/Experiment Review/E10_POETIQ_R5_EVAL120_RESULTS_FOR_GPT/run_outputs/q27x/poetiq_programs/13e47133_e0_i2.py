import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform each non-wall connected region into concentric rectangle layers.

    Rule inferred from the examples:
      * The most frequent color is the background.
      * The wall color is the non-background color forming long line segments.
      * Non-wall cells are grouped into 4-connected components.
      * For each component:
          - If it contains no marker cells, leave it as background.
          - Marker cells are the non-background, non-wall cells inside the component.
          - The marker positions define a color sequence for layers 0, 1, 2, ...
          - Layers are filled cyclically.

    The key correction over a simple bounding-box solution:
      * The component is anchored at the corner nearest the first marker.
      * Internal walls on the two "far" sides of that anchor can act as
        additional boundaries and reduce the layer distance.
      * Internal walls on the two "near" sides are ignored except when a cell
        is immediately adjacent to a wall, in which case that cell is layer 0.
    """

    if grid.size == 0:
        return grid.copy()

    g = np.array(grid, dtype=int)
    H, W = g.shape

    # ------------------------------------------------------------------
    # 1. Background: most frequent color.
    # ------------------------------------------------------------------
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # 2. Wall color: strongest straight-line non-background color.
    # ------------------------------------------------------------------
    def line_score(c: int):
        mask = (g == c)
        score = 0

        # Horizontal runs.
        for r in range(H):
            run = 0
            for cc in range(W + 1):
                if cc < W and mask[r, cc]:
                    run += 1
                else:
                    if run >= 2:
                        score += run * run
                    run = 0

        # Vertical runs.
        for cc in range(W):
            run = 0
            for rr in range(H + 1):
                if rr < H and mask[rr, cc]:
                    run += 1
                else:
                    if run >= 2:
                        score += run * run
                    run = 0

        return score

    wall = -1
    best_score = 0
    for c in vals:
        c = int(c)
        if c == bg:
            continue
        s = line_score(c)
        if s > best_score:
            best_score = s
            wall = c

    # Fallback if no clear wall is found.
    if wall == -1:
        order = np.argsort(-counts)
        for idx in order:
            c = int(vals[idx])
            if c != bg:
                wall = c
                break
        if wall == -1:
            wall = bg

    # ------------------------------------------------------------------
    # 3. Precompute nearest wall distances in four directions.
    #    dist_*[r, c] is the number of non-wall cells between (r, c)
    #    and the next wall cell in that direction, not including the wall.
    #    If no wall exists in that direction, use a sentinel -1.
    # ------------------------------------------------------------------
    INF = 10**9

    down = np.full((H, W), -1, dtype=int)
    up = np.full((H, W), -1, dtype=int)
    right = np.full((H, W), -1, dtype=int)
    left = np.full((H, W), -1, dtype=int)

    for c in range(W):
        last_wall = -1
        for r in range(H - 1, -1, -1):
            if g[r, c] == wall:
                last_wall = r
            else:
                if last_wall != -1:
                    down[r, c] = last_wall - r - 1
        last_wall = H
        for r in range(H):
            if g[r, c] == wall:
                last_wall = r
            else:
                if last_wall != H:
                    up[r, c] = r - last_wall - 1

    for r in range(H):
        last_wall = -1
        for c in range(W - 1, -1, -1):
            if g[r, c] == wall:
                last_wall = c
            else:
                if last_wall != -1:
                    right[r, c] = last_wall - c - 1
        last_wall = W
        for c in range(W):
            if g[r, c] == wall:
                last_wall = c
            else:
                if last_wall != W:
                    left[r, c] = c - last_wall - 1

    # ------------------------------------------------------------------
    # 4. Label 4-connected components of non-wall cells.
    # ------------------------------------------------------------------
    non_wall = (g != wall)
    labels = -np.ones((H, W), dtype=int)
    components = []
    comp_id = 0

    for r in range(H):
        for c in range(W):
            if non_wall[r, c] and labels[r, c] == -1:
                q = deque([(r, c)])
                labels[r, c] = comp_id
                cells = []

                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if non_wall[nr, nc] and labels[nr, nc] == -1:
                                labels[nr, nc] = comp_id
                                q.append((nr, nc))

                components.append(cells)
                comp_id += 1

    # ------------------------------------------------------------------
    # 5. Output starts as background, with walls preserved.
    # ------------------------------------------------------------------
    out = np.full((H, W), bg, dtype=int)
    out[g == wall] = wall

    # ------------------------------------------------------------------
    # 6. Helper: effective layer for a cell in a component.
    # ------------------------------------------------------------------
    def effective_layer(r: int, c: int,
                        r0: int, r1: int, c0: int, c1: int,
                        anchor_top: bool, anchor_left: bool) -> int:
        """
        Layer distance to the relevant effective boundary.

        anchor_top / anchor_left define the anchor corner.
        The two far sides may be truncated by internal walls.
        The two near sides always use the component bounding box.
        Adjacent to any wall => layer 0.
        """
        # Immediate adjacency to a wall forces outermost layer.
        if r > 0 and g[r - 1, c] == wall:
            return 0
        if r + 1 < H and g[r + 1, c] == wall:
            return 0
        if c > 0 and g[r, c - 1] == wall:
            return 0
        if c + 1 < W and g[r, c + 1] == wall:
            return 0

        # Vertical directions.
        if anchor_top:
            near_v = r - r0
            far_d = down[r, c]
            if far_d == -1 or far_d > (r1 - r):
                far_v = r1 - r
            else:
                far_v = far_d
        else:
            near_v = r1 - r
            far_d = up[r, c]
            if far_d == -1 or far_d > (r - r0):
                far_v = r - r0
            else:
                far_v = far_d

        # Horizontal directions.
        if anchor_left:
            near_h = c - c0
            far_d = right[r, c]
            if far_d == -1 or far_d > (c1 - c):
                far_h = c1 - c
            else:
                far_h = far_d
        else:
            near_h = c1 - c
            far_d = left[r, c]
            if far_d == -1 or far_d > (c - c0):
                far_h = c - c0
            else:
                far_h = far_d

        return min(near_v, far_v, near_h, far_h)

    # ------------------------------------------------------------------
    # 7. Fill each component.
    # ------------------------------------------------------------------
    for cells in components:
        if not cells:
            continue

        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        # Marker cells inside this component.
        markers = [(r, c, int(g[r, c])) for r, c in cells
                   if g[r, c] != bg and g[r, c] != wall]

        if not markers:
            continue

        # Determine anchor corner from the first/outermost marker.
        # Use simple bounding-box layer to choose the outermost marker(s).
        def bbox_layer(r, c):
            return min(r - r0, r1 - r, c - c0, c1 - c)

        outer_layer = min(bbox_layer(r, c) for r, c, _ in markers)
        outer_markers = [(r, c, col) for r, c, col in markers
                         if bbox_layer(r, c) == outer_layer]

        # Choose one outer marker to define the anchor.
        # Prefer topmost, then leftmost/rightmost extreme.
        chosen = min(outer_markers, key=lambda x: (x[0], abs(x[1] - (c0 + c1) / 2)))
        cr, cc, _ = chosen

        anchor_top = (cr - r0) <= (r1 - cr)
        anchor_left = (cc - c0) <= (c1 - cc)

        # Build the color sequence by effective layer of markers.
        max_layer = min(r1 - r0, c1 - c0) // 2

        layer_to_colors = {}
        for r, c, col in markers:
            L = effective_layer(r, c, r0, r1, c0, c1, anchor_top, anchor_left)
            if 0 <= L <= max_layer:
                layer_to_colors.setdefault(L, {})
                layer_to_colors[L][col] = layer_to_colors[L].get(col, 0) + 1

        seq = []
        started = False

        for L in range(max_layer + 1):
            counts_in_layer = layer_to_colors.get(L)
            if counts_in_layer:
                if not started:
                    seq.extend([bg] * L)
                    started = True

                # Dominant marker color on this layer; tie by smaller color.
                chosen_col = min(counts_in_layer.items(),
                                 key=lambda kv: (-kv[1], kv[0]))[0]
                seq.append(chosen_col)
            else:
                if started:
                    break

        if not seq:
            continue

        # Paint all cells in this component using the cyclic layer sequence.
        for r, c in cells:
            L = effective_layer(r, c, r0, r1, c0, c1, anchor_top, anchor_left)
            out[r, c] = seq[L % len(seq)]

    return out