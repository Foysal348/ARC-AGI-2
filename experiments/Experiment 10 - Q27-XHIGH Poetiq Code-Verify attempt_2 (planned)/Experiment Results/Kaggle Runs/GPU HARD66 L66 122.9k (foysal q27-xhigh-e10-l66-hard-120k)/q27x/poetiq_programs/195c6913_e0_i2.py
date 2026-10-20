import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the inferred rule:

    - Isolated 2x2 blocks encode a repeating color sequence and a marker color.
    - Single border pixels of the first sequence color are anchors.
    - A thick field color is skeletonized.
    - Anchors are connected to the skeleton.
    - The resulting path is colored by repeating the sequence according to
      distance from anchors.
    - Terminal path cells are colored with the marker color.
    - The original 2x2 legend blocks are removed.
    """

    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H < 2 or W < 2:
        return g

    out = g.copy()

    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def border_background():
        border = np.concatenate([
            out[0, :].astype(int),
            out[-1, :].astype(int),
            out[:, 0].astype(int),
            out[:, -1].astype(int)
        ])
        vals, counts = np.unique(border, return_counts=True)
        return int(vals[np.argmax(counts)])

    def find_isolated_2x2_blocks(exclude_colors=None):
        exclude = set(exclude_colors or [])
        blocks = []

        for r in range(H - 1):
            for c in range(W - 1):
                v = int(out[r, c])
                if v in exclude:
                    continue

                if not (out[r, c] == v and out[r, c + 1] == v and
                        out[r + 1, c] == v and out[r + 1, c + 1] == v):
                    continue

                # Skip if the 2x2 is part of a larger vertical block.
                if r > 0 and out[r - 1, c] == v and out[r - 1, c + 1] == v:
                    continue
                if r + 2 < H and out[r + 2, c] == v and out[r + 2, c + 1] == v:
                    continue

                # Skip if the 2x2 is part of a larger horizontal block.
                if c > 0 and out[r, c - 1] == v and out[r + 1, c - 1] == v:
                    continue
                if c + 2 < W and out[r, c + 2] == v and out[r + 1, c + 2] == v:
                    continue

                blocks.append((r, c, v))

        return blocks

    def component_sets(mask):
        """Return a list of 4-connected components as sets of (r,c)."""
        if not mask.any():
            return []

        label = np.full((H, W), -1, dtype=int)
        comps = []

        for r in range(H):
            for c in range(W):
                if not mask[r, c] or label[r, c] != -1:
                    continue

                cid = len(comps)
                q = deque([(r, c)])
                label[r, c] = cid
                cells = set()

                while q:
                    x, y = q.popleft()
                    cells.add((x, y))

                    for dx, dy in dirs4:
                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny) and mask[nx, ny] and label[nx, ny] == -1:
                            label[nx, ny] = cid
                            q.append((nx, ny))

                comps.append(cells)

        return comps

    def build_graph(cells):
        """Build a 4-neighbor graph for a set of grid cells."""
        cells = set(cells)
        graph = {v: [] for v in cells}

        for (r, c) in cells:
            for dr, dc in dirs4:
                nb = (r + dr, c + dc)
                if nb in cells:
                    graph[(r, c)].append(nb)

        return graph

    def dilate_mask(mask):
        """8-neighbor dilation."""
        res = mask.copy()
        for r in range(H):
            for c in range(W):
                if mask[r, c]:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if in_bounds(nr, nc):
                                res[nr, nc] = True
        return res

    def erode_keep_border(mask):
        """Erosion with border treated as foreground, so boundary survives."""
        if not mask.any():
            return mask.copy()

        pad = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=1)
        res = np.zeros((H, W), dtype=bool)

        for r in range(H):
            for c in range(W):
                if pad[r:r + 3, c:c + 3].all():
                    res[r, c] = True

        return res

    def binary_closing_keep_border(mask):
        """Small closing: dilate then erode, preserving border cells."""
        if not mask.any():
            return mask.copy()
        return erode_keep_border(dilate_mask(mask))

    def zhang_suen_skeleton(mask):
        """Zhang-Suen thinning for binary masks."""
        if not mask.any():
            return np.zeros((H, W), dtype=bool)

        X = np.pad(mask.astype(np.uint8), 2, mode="constant", constant_values=0)
        h, w = X.shape

        while True:
            changed = False

            for step in (1, 2):
                delete = []

                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        if X[i, j] != 1:
                            continue

                        b2 = X[i, j - 1]
                        b3 = X[i + 1, j - 1]
                        b4 = X[i + 1, j]
                        b5 = X[i + 1, j + 1]
                        b6 = X[i, j + 1]
                        b7 = X[i - 1, j + 1]
                        b8 = X[i - 1, j]
                        b9 = X[i - 1, j - 1]

                        B = int(b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9)

                        A = (
                            int(b2 == 0 and b3 == 1) +
                            int(b3 == 0 and b4 == 1) +
                            int(b4 == 0 and b5 == 1) +
                            int(b5 == 0 and b6 == 1) +
                            int(b6 == 0 and b7 == 1) +
                            int(b7 == 0 and b8 == 1) +
                            int(b8 == 0 and b9 == 1) +
                            int(b9 == 0 and b2 == 1)
                        )

                        if 2 <= B <= 6 and A == 1:
                            if step == 1 and (b2 * b4 * b6) == 0 and (b4 * b6 * b8) == 0:
                                delete.append((i, j))
                            elif step == 2 and (b2 * b4 * b8) == 0 and (b2 * b6 * b8) == 0:
                                delete.append((i, j))

                if delete:
                    for i, j in delete:
                        X[i, j] = 0
                    changed = True

            if not changed:
                break

        return (X[2:-2, 2:-2] == 1)

    # ------------------------------------------------------------------
    # 1. Detect legend blocks: sequence + marker
    # ------------------------------------------------------------------

    bg_candidate = border_background()

    key_blocks = find_isolated_2x2_blocks(exclude_colors={bg_candidate})

    # Fallback: if too restrictive, try without excluding background.
    if not key_blocks:
        key_blocks = find_isolated_2x2_blocks(exclude_colors=set())

    if not key_blocks:
        return out

    key_cells = set()
    for r, c, _ in key_blocks:
        for dr in (0, 1):
            for dc in (0, 1):
                key_cells.add((r + dr, c + dc))

    top_row = min(r for r, c, v in key_blocks)
    seq_blocks = sorted(
        [b for b in key_blocks if b[0] == top_row],
        key=lambda x: x[1]
    )

    sequence = [int(v) for _, _, v in seq_blocks]

    other_blocks = [b for b in key_blocks if b[0] != top_row]
    if other_blocks:
        marker_block = max(other_blocks, key=lambda x: (x[0], x[1]))
        marker_color = int(marker_block[2])
    else:
        # Fallback: choose a color present in key blocks but not in sequence.
        marker_color = -1
        for _, _, v in key_blocks:
            v = int(v)
            if v not in sequence:
                marker_color = v
                break

        # Last fallback: any color not background/sequence.
        if marker_color == -1:
            for v in np.unique(out):
                v = int(v)
                if v != bg_candidate and v not in sequence:
                    marker_color = v
                    break

    if not sequence or marker_color == -1:
        return out

    anchor_color = int(sequence[0])

    # ------------------------------------------------------------------
    # 2. Detect anchors: single border pixels of anchor_color
    # ------------------------------------------------------------------

    anchors = []

    def is_key_cell(r, c):
        return (r, c) in key_cells

    # Prefer true border cells.
    for r in range(H):
        for c in (0, W - 1):
            if in_bounds(r, c) and int(out[r, c]) == anchor_color and not is_key_cell(r, c):
                anchors.append((r, c))

    for c in range(W):
        for r in (0, H - 1):
            if in_bounds(r, c) and int(out[r, c]) == anchor_color and not is_key_cell(r, c):
                if (r, c) not in anchors:
                    anchors.append((r, c))

    # Fallback: first two columns, useful if border detection missed them.
    if not anchors:
        for r in range(H):
            for c in range(min(2, W)):
                if int(out[r, c]) == anchor_color and not is_key_cell(r, c):
                    anchors.append((r, c))
                    break

    if not anchors:
        return out

    # ------------------------------------------------------------------
    # 3. Infer field color from anchors
    # ------------------------------------------------------------------

    neighbor_counts = {}
    for (r, c) in anchors:
        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                continue
            if is_key_cell(nr, nc):
                continue

            v = int(out[nr, nc])
            if v == anchor_color or v == bg_candidate:
                continue
            if v in sequence:
                continue
            if v == marker_color:
                continue

            neighbor_counts[v] = neighbor_counts.get(v, 0) + 1

    def largest_component_area(color):
        mask = (out == color)
        comps = component_sets(mask)
        if not comps:
            return 0
        return max(len(x) for x in comps)

    if neighbor_counts:
        field_color = max(neighbor_counts.items(), key=lambda kv: (kv[1], largest_component_area(kv[0])))[0]
    else:
        # Fallback: largest non-background component.
        best_color = -1
        best_area = 0
        for v in np.unique(out):
            v = int(v)
            if v == bg_candidate or v in sequence or v == marker_color:
                continue
            area = largest_component_area(v)
            if area > best_area:
                best_area = area
                best_color = v

        if best_color == -1:
            return out

        field_color = best_color

    if field_color == bg_candidate:
        # Choose the largest component that is not the border background.
        best_color = -1
        best_area = 0
        for v in np.unique(out):
            v = int(v)
            if v == bg_candidate or v in sequence or v == marker_color:
                continue
            area = largest_component_area(v)
            if area > best_area:
                best_area = area
                best_color = v
        if best_color != -1:
            field_color = best_color

    # Final background: most common color outside field/legend colors.
    legend_colors = set(sequence) | {marker_color}
    final_bg = bg_candidate

    vals, counts = np.unique(out, return_counts=True)
    candidates = [(int(v), int(cnt)) for v, cnt in zip(vals, counts)
                  if int(v) != field_color and int(v) not in legend_colors]
    if candidates:
        final_bg = max(candidates, key=lambda x: x[1])[0]
    else:
        final_bg = bg_candidate

    # ------------------------------------------------------------------
    # 4. Remove legend blocks from output
    # ------------------------------------------------------------------

    for r, c, _ in key_blocks:
        out[r:r + 2, c:c + 2] = final_bg

    # ------------------------------------------------------------------
    # 5. Build work mask and skeleton
    # ------------------------------------------------------------------

    main_mask = (out == field_color)
    for (r, c) in key_cells:
        main_mask[r, c] = False

    base_mask = main_mask.copy()
    for (r, c) in anchors:
        base_mask[r, c] = True

    work_mask = binary_closing_keep_border(base_mask)

    # Do not let legend cells become part of the path.
    for (r, c) in key_cells:
        work_mask[r, c] = False

    # Keep anchors in the work mask for connectivity.
    for (r, c) in anchors:
        work_mask[r, c] = True

    skeleton = zhang_suen_skeleton(work_mask)

    # If skeleton is empty, fall back to the work mask itself, but only cells
    # in components that touch anchors.
    if not skeleton.any():
        work_comps = component_sets(work_mask)
        fallback = set()
        for comp in work_comps:
            if any(a in comp for a in anchors):
                fallback.update(comp)
        skeleton_cells = fallback
    else:
        skeleton_cells = set()
        for comp in component_sets(skeleton):
            skeleton_cells.update(comp)

    # ------------------------------------------------------------------
    # 6. Connect anchors to the best skeleton component
    # ------------------------------------------------------------------

    path_cells = set(anchors)

    skel_comps = component_sets(skeleton)

    def max_distance_to_terminal_in_component(comp):
        graph = build_graph(comp)
        terminals = [v for v, ns in graph.items() if len(ns) == 1]
        if not terminals:
            terminals = [next(iter(comp))]

        maxd = {v: 0 for v in comp}

        for t in terminals:
            dist = {t: 0}
            q = deque([t])
            while q:
                x, y = q.popleft()
                d = dist[(x, y)]
                for nb in graph.get((x, y), []):
                    if nb not in dist:
                        dist[nb] = d + 1
                        q.append(nb)
                        if d + 1 > maxd.get(nb, 0):
                            maxd[nb] = d + 1

        return maxd

    # Precompute max distance to terminal for each skeleton component.
    comp_info = []
    for comp in skel_comps:
        comp = {cell for cell in comp if cell not in key_cells}
        if not comp:
            continue
        maxd = max_distance_to_terminal_in_component(comp)
        comp_info.append((comp, maxd))

    def add_straight_line(path, r1, c1, r2, c2):
        if r1 == r2:
            lo, hi = sorted((c1, c2))
            for c in range(lo, hi + 1):
                path.add((r1, c))
        elif c1 == c2:
            lo, hi = sorted((r1, r2))
            for r in range(lo, hi + 1):
                path.add((r, c1))

    def l_path_cells_h_first(r1, c1, r2, c2):
        cells = set()
        add_straight_line(cells, r1, c1, r1, c2)
        add_straight_line(cells, r1, c2, r2, c2)
        return cells

    def l_path_cells_v_first(r1, c1, r2, c2):
        cells = set()
        add_straight_line(cells, r1, c1, r2, c1)
        add_straight_line(cells, r2, c1, r2, c2)
        return cells

    def connection_cost(cells):
        cost = 0
        for (r, c) in cells:
            if not work_mask[r, c]:
                cost += 1
            if (r, c) in key_cells:
                cost += 100
        return cost

    for (ar, ac) in anchors:
        if not comp_info:
            continue

        best_cell = None
        best_score = -1e18

        for comp, maxd in comp_info:
            for cell in comp:
                cr, cc = cell
                l1 = abs(ar - cr) + abs(ac - cc)
                score = maxd.get(cell, 0) - 0.25 * l1 + 0.001 * len(comp)
                if score > best_score:
                    best_score = score
                    best_cell = cell

        if best_cell is None:
            continue

        tr, tc = best_cell

        if ar == tr:
            add_straight_line(path_cells, ar, ac, tr, tc)
        elif ac == tc:
            add_straight_line(path_cells, ar, ac, tr, tc)
        else:
            h_first = l_path_cells_h_first(ar, ac, tr, tc)
            v_first = l_path_cells_v_first(ar, ac, tr, tc)

            cost_h = connection_cost(h_first)
            cost_v = connection_cost(v_first)

            # Prefer horizontal first for left-border anchors, vertical first
            # for top-border anchors, when costs are similar.
            if cost_h < cost_v or (cost_h == cost_v and ac == 0):
                path_cells.update(h_first)
            elif cost_v < cost_h or (cost_h == cost_v and ar == 0):
                path_cells.update(v_first)
            else:
                path_cells.update(h_first)

        # Add the chosen skeleton component.
        for comp, _ in comp_info:
            if best_cell in comp:
                path_cells.update(comp)
                break

    # Keep only path components that contain at least one anchor.
    if path_cells:
        path_bool = np.zeros((H, W), dtype=bool)
        for (r, c) in path_cells:
            path_bool[r, c] = True

        good_components = []
        for comp in component_sets(path_bool):
            if any(a in comp for a in anchors):
                good_components.append(comp)

        new_path = set()
        for comp in good_components:
            new_path.update(comp)

        path_cells = new_path

    if not path_cells:
        return out

    path_mask = np.zeros((H, W), dtype=bool)
    for (r, c) in path_cells:
        path_mask[r, c] = True

    # ------------------------------------------------------------------
    # 7. Color path by BFS distance from anchors, repeating sequence
    # ------------------------------------------------------------------

    L = len(sequence)
    dist = np.full((H, W), -1, dtype=int)
    q = deque()

    for (r, c) in anchors:
        if path_mask[r, c]:
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and path_mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    for (r, c) in path_cells:
        d = dist[r, c]
        if d < 0:
            d = 0
        out[r, c] = sequence[d % L]

    # ------------------------------------------------------------------
    # 8. Mark terminal path cells
    # ------------------------------------------------------------------

    final_graph = build_graph(path_cells)
    terminals = [v for v, ns in final_graph.items() if len(ns) == 1]

    original_main_mask = (g == field_color)

    def work_degree(cell):
        r, c = cell
        cnt = 0
        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and work_mask[nr, nc]:
                cnt += 1
        return cnt

    for t in terminals:
        if t in anchors:
            continue

        r, c = t
        ns = final_graph.get(t, [])

        if not ns:
            pos = t
        else:
            nr, nc = ns[0]
            dr = r - nr
            dc = c - nc
            away = (r + dr, c + dc)

            # If the terminal is already outside the original field, it is
            # likely the marker cell itself.
            if not original_main_mask[r, c]:
                pos = t
            else:
                wdeg = work_degree(t)

                # If it looks like a narrow cap in the work mask, mark it.
                if wdeg <= 2:
                    pos = t
                # Otherwise, if there is an open cell directly beyond the
                # terminal, place the marker there.
                elif in_bounds(*away) and away not in path_cells and away not in key_cells:
                    pos = away
                else:
                    pos = t

        if in_bounds(*pos) and pos not in key_cells:
            out[pos] = marker_color

    # ------------------------------------------------------------------
    # 9. Final cleanup: ensure legend blocks are erased
    # ------------------------------------------------------------------

    for r, c, _ in key_blocks:
        out[r:r + 2, c:c + 2] = final_bg

    return out.astype(int, copy=False)