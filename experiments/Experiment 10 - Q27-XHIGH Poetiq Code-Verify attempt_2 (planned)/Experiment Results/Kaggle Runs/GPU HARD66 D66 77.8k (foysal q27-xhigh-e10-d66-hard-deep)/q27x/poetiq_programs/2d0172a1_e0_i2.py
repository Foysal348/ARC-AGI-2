import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract the main room/floor-plan structure from the input.

    The non-background color is treated as walls.  The algorithm:
      1. Finds the cyclic 4-connected core of the wall color.  This keeps
         closed wall loops and removes dangling branches/noise.
      2. Finds enclosed background "rooms" (background components not
         connected to the grid border).
      3. Assigns each enclosed room to the innermost wall loop that
         surrounds it.
      4. Selects the largest surrounding loop as the main room.
      5. Crops to the main loop's cyclic core bounding box.
      6. Draws each main/nested loop as a clean rectangular border around
         the bounding box of the background region it encloses.
      7. Preserves small isolated wall dots inside the main room.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # -------------------------
    # Choose background/wall
    # -------------------------
    all_colors, all_counts = np.unique(grid, return_counts=True)
    overall_bg = int(all_colors[np.argmax(all_counts)])

    border_vals = np.concatenate((
        grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
    ))
    border_colors, _ = np.unique(border_vals, return_counts=True)

    # Prefer the overall majority color if it appears on the border.
    if len(border_colors) and overall_bg in border_colors:
        bg = overall_bg
    elif len(border_colors):
        # Otherwise use the most common border color.
        bvals, bcounts = np.unique(border_vals, return_counts=True)
        bg = int(bvals[np.argmax(bcounts)])
    else:
        bg = overall_bg

    bg_mask = (grid == bg)
    wall_mask = (grid != bg)

    if not np.any(wall_mask):
        return grid.copy()

    # Wall color is the most frequent non-background color.
    wall_vals, wall_cnts = np.unique(grid[wall_mask], return_counts=True)
    wall = int(wall_vals[np.argmax(wall_cnts)])

    # -------------------------
    # Helper: connected components
    # -------------------------
    def _label_components(mask):
        labels = np.zeros((h, w), dtype=int)
        comps = []
        next_id = 0

        for r in range(h):
            for c in range(w):
                if not mask[r, c] or labels[r, c] != 0:
                    continue

                next_id += 1
                q = deque([(r, c)])
                labels[r, c] = next_id
                cells = []
                min_r = max_r = r
                min_c = max_c = c

                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))

                    if rr < min_r:
                        min_r = rr
                    if rr > max_r:
                        max_r = rr
                    if cc < min_c:
                        min_c = cc
                    if cc > max_c:
                        max_c = cc

                    for nr, nc in ((rr - 1, cc), (rr + 1, cc),
                                   (rr, cc - 1), (rr, cc + 1)):
                        if (0 <= nr < h and 0 <= nc < w and
                                mask[nr, nc] and labels[nr, nc] == 0):
                            labels[nr, nc] = next_id
                            q.append((nr, nc))

                comps.append((next_id, cells, (min_r, min_c, max_r, max_c), len(cells)))

        return labels, comps

    # -------------------------
    # Helper: 2-core (remove all tree-like branches)
    # -------------------------
    def _two_core(mask):
        core = mask.copy()
        deg = np.zeros((h, w), dtype=np.int16)

        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                d = 0
                if r > 0 and mask[r - 1, c]:
                    d += 1
                if r + 1 < h and mask[r + 1, c]:
                    d += 1
                if c > 0 and mask[r, c - 1]:
                    d += 1
                if c + 1 < w and mask[r, c + 1]:
                    d += 1
                deg[r, c] = d

        q = deque()
        for r in range(h):
            for c in range(w):
                if core[r, c] and deg[r, c] <= 1:
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            if not core[r, c]:
                continue

            core[r, c] = False

            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and core[nr, nc]:
                    deg[nr, nc] -= 1
                    if deg[nr, nc] <= 1:
                        q.append((nr, nc))

        return core

    # -------------------------
    # Helper: background connected to the outside
    # -------------------------
    def _outside_background(mask):
        outside = np.zeros((h, w), dtype=bool)
        q = deque()

        for r in range(h):
            for c in (0, w - 1):
                if mask[r, c] and not outside[r, c]:
                    outside[r, c] = True
                    q.append((r, c))

        for c in range(w):
            for r in (0, h - 1):
                if mask[r, c] and not outside[r, c]:
                    outside[r, c] = True
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not outside[nr, nc]:
                    outside[nr, nc] = True
                    q.append((nr, nc))

        return outside

    # -------------------------
    # Label wall components and cyclic cores
    # -------------------------
    orig_labels, orig_comps = _label_components(wall_mask)
    core_mask = _two_core(wall_mask)
    core_labels, core_comps = _label_components(core_mask)

    def _fallback():
        """Graceful fallback: crop to the largest original wall component."""
        if not orig_comps:
            return np.full((1, 1), bg, dtype=grid.dtype)

        _, cells, bbox, _ = max(orig_comps, key=lambda x: x[3])
        r0, c0, r1, c1 = bbox
        out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=grid.dtype)

        use_cells = [(r, c) for (r, c) in cells if core_mask[r, c]]
        if not use_cells:
            use_cells = cells

        for r, c in use_cells:
            out[r - r0, c - c0] = wall
        return out

    if not core_comps:
        return _fallback()

    # -------------------------
    # Enclosed background holes
    # -------------------------
    outside = _outside_background(bg_mask)
    hole_mask = bg_mask & ~outside
    _, hole_comps = _label_components(hole_mask)

    # -------------------------
    # Assign holes to surrounding cyclic wall loops
    # -------------------------
    label_to_ci = {comp[0]: i for i, comp in enumerate(core_comps)}
    assigned_holes = [[] for _ in core_comps]

    for _, hole_cells, hbbox, harea in hole_comps:
        hr0, hc0, hr1, hc1 = hbbox

        # Core components adjacent to this hole.
        adjacent_core_labels = set()
        for r, c in hole_cells:
            if r > 0 and core_labels[r - 1, c] != 0:
                adjacent_core_labels.add(core_labels[r - 1, c])
            if r + 1 < h and core_labels[r + 1, c] != 0:
                adjacent_core_labels.add(core_labels[r + 1, c])
            if c > 0 and core_labels[r, c - 1] != 0:
                adjacent_core_labels.add(core_labels[r, c - 1])
            if c + 1 < w and core_labels[r, c + 1] != 0:
                adjacent_core_labels.add(core_labels[r, c + 1])

        if not adjacent_core_labels:
            continue

        best_score = None
        best_ci = None

        # First try strict bbox containment: the hole must be inside the loop.
        for ci, (_, _, cbbox, carea) in enumerate(core_comps):
            cid = core_comps[ci][0]
            if cid not in adjacent_core_labels:
                continue

            cr0, cc0, cr1, cc1 = cbbox
            if cr0 < hr0 and cr1 > hr1 and cc0 < hc0 and cc1 > hc1:
                bbox_area = (cr1 - cr0 + 1) * (cc1 - cc0 + 1)
                score = (bbox_area, carea, ci)
                if best_score is None or score < best_score:
                    best_score = score
                    best_ci = ci

        # Fallback: non-strict containment, but still require adjacency.
        if best_ci is None:
            hole_bbox_area = (hr1 - hr0 + 1) * (hc1 - hc0 + 1)
            for ci, (_, _, cbbox, carea) in enumerate(core_comps):
                cid = core_comps[ci][0]
                if cid not in adjacent_core_labels:
                    continue

                cr0, cc0, cr1, cc1 = cbbox
                bbox_area = (cr1 - cr0 + 1) * (cc1 - cc0 + 1)
                if (cr0 <= hr0 and cr1 >= hr1 and
                        cc0 <= hc0 and cc1 >= hc1 and
                        bbox_area > hole_bbox_area):
                    score = (bbox_area, carea, ci)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_ci = ci

        if best_ci is not None:
            assigned_holes[best_ci].append((hbbox, harea))

    # -------------------------
    # Build candidate rectangular loops
    # -------------------------
    rect_infos = [None] * len(core_comps)

    for ci, holes in enumerate(assigned_holes):
        if not holes:
            continue

        hr0 = min(h[0][0] for h in holes)
        hr1 = max(h[0][2] for h in holes)
        hc0 = min(h[0][1] for h in holes)
        hc1 = max(h[0][3] for h in holes)

        # Rectangular border area after expanding the interior by one pixel.
        rect_area = (hr1 - hr0 + 3) * (hc1 - hc0 + 3)
        rect_infos[ci] = (hr0, hc0, hr1, hc1, rect_area)

    valid_roots = [i for i, info in enumerate(rect_infos) if info is not None]
    if not valid_roots:
        return _fallback()

    # Main room: largest clean rectangular loop.
    root_ci = max(
        valid_roots,
        key=lambda i: (
            rect_infos[i][4],
            (core_comps[i][2][2] - core_comps[i][2][0] + 1) *
            (core_comps[i][2][3] - core_comps[i][2][1] + 1),
            core_comps[i][3]
        )
    )

    # -------------------------
    # Create output canvas from the main loop's cyclic core bbox
    # -------------------------
    r0, c0, r1, c1 = core_comps[root_ci][2]
    out_h = r1 - r0 + 1
    out_w = c1 - c0 + 1
    out = np.full((out_h, out_w), bg, dtype=grid.dtype)

    hr0, hc0, hr1, hc1, _ = rect_infos[root_ci]

    # Interior region of the main room, relative to the output canvas.
    ir0 = hr0 - r0
    ir1 = hr1 - r0
    ic0 = hc0 - c0
    ic1 = hc1 - c0

    ir0 = max(0, ir0)
    ir1 = min(out_h - 1, ir1)
    ic0 = max(0, ic0)
    ic1 = min(out_w - 1, ic1)

    if ir0 <= ir1 and ic0 <= ic1:
        br0 = max(0, ir0 - 1)
        br1 = min(out_h - 1, ir1 + 1)
        bc0 = max(0, ic0 - 1)
        bc1 = min(out_w - 1, ic1 + 1)
    else:
        br0, br1, bc0, bc1 = 0, out_h - 1, 0, out_w - 1

    if not (0 <= br0 <= br1 < out_h and 0 <= bc0 <= bc1 < out_w):
        return _fallback()

    # Draw main rectangular wall border.
    out[br0, bc0:bc1 + 1] = wall
    out[br1, bc0:bc1 + 1] = wall
    out[br0:br1 + 1, bc0] = wall
    out[br0:br1 + 1, bc1] = wall

    # Interior of the main drawn border.
    inter_r0 = br0 + 1
    inter_r1 = br1 - 1
    inter_c0 = bc0 + 1
    inter_c1 = bc1 - 1
    has_interior = inter_r0 <= inter_r1 and inter_c0 <= inter_c1

    # -------------------------
    # Draw nested rectangular loops inside the main room
    # -------------------------
    nested_rects = []

    if has_interior:
        for ci, info in enumerate(rect_infos):
            if ci == root_ci or info is None:
                continue

            nr0, nc0, nr1, nc1, _ = info

            rr0 = nr0 - r0
            rr1 = nr1 - r0
            cc0 = nc0 - c0
            cc1 = nc1 - c0

            if rr0 < 0 or rr1 >= out_h or cc0 < 0 or cc1 >= out_w:
                continue

            rb0 = max(0, rr0 - 1)
            rb1 = min(out_h - 1, rr1 + 1)
            rc0 = max(0, cc0 - 1)
            rc1 = min(out_w - 1, cc1 + 1)

            # Keep only loops fully inside the main room's interior.
            if rb0 >= inter_r0 and rb1 <= inter_r1 and rc0 >= inter_c0 and rc1 <= inter_c1:
                rect_area = (rb1 - rb0 + 1) * (rc1 - rc0 + 1)
                nested_rects.append((rb0, rb1, rc0, rc1, rect_area))

        # Draw larger nested rooms first.
        nested_rects.sort(key=lambda x: x[4], reverse=True)

        for rb0, rb1, rc0, rc1, _ in nested_rects:
            out[rb0, rc0:rc1 + 1] = wall
            out[rb1, rc0:rc1 + 1] = wall
            out[rb0:rb1 + 1, rc0] = wall
            out[rb0:rb1 + 1, rc1] = wall

    # -------------------------
    # Draw isolated wall dots inside the main room
    # -------------------------
    DOT_MAX_AREA = 4

    if has_interior:
        core_orig_ids = [set() for _ in core_comps]
        for ci, comp in enumerate(core_comps):
            for r, c in comp[1]:
                core_orig_ids[ci].add(int(orig_labels[r, c]))

        root_orig_ids = core_orig_ids[root_ci]

        # Original components that correspond to drawn rectangular loops.
        loop_orig_ids = set()
        for ci, info in enumerate(rect_infos):
            if info is not None:
                loop_orig_ids.update(core_orig_ids[ci])

        for oid, cells, obbox, oarea in orig_comps:
            if oid in root_orig_ids:
                continue
            if oid in loop_orig_ids:
                continue

            # Prefer core pixels if present; this lets small thick dots simplify.
            core_cells = [(r, c) for r, c in cells if core_mask[r, c]]

            if core_cells:
                if len(core_cells) > DOT_MAX_AREA:
                    continue
                dr0 = min(r for r, _ in core_cells)
                dr1 = max(r for r, _ in core_cells)
                dc0 = min(c for _, c in core_cells)
                dc1 = max(c for _, c in core_cells)
            else:
                if oarea > DOT_MAX_AREA:
                    continue
                dr0, dc0, dr1, dc1 = obbox

            rr = (dr0 + dr1) // 2 - r0
            cc = (dc0 + dc1) // 2 - c0

            if (inter_r0 <= rr <= inter_r1 and
                    inter_c0 <= cc <= inter_c1 and
                    0 <= rr < out_h and 0 <= cc < out_w):
                if out[rr, cc] == bg:
                    out[rr, cc] = wall

    return out