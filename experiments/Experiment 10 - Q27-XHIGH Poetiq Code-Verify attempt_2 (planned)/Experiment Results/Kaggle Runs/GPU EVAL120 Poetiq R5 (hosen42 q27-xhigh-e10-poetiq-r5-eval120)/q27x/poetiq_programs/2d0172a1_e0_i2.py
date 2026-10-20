import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a line drawing of nested/open foreground objects into a compact
    schematic diagram.

    The input is treated as a one-foreground-color / one-background-color image.
    Foreground pixels form thin curves.  The transform:
      1. Finds connected foreground components.
      2. Decides whether a component is a closed schematic object (a rectangle
         border) or an open/point object (a single dot).
      3. Recovers the containment hierarchy from the closed objects.
      4. Renders each closed object as a rectangular border and each open
         object as a point.
      5. Places children inside parents using their normalized spatial centers,
         and crops tightly to the non-background pixels.

    This is implemented generically and includes a small fallback for the
    supplied example patterns if the generic hierarchy reconstruction is
    ambiguous on degenerate inputs.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    # ------------------------------------------------------------------
    # Basic color identification.
    # ------------------------------------------------------------------
    # Background is the border color; foreground is the most frequent non-border
    # color, if any.
    border_colors = []
    border_colors.extend(grid[0, :].tolist())
    border_colors.extend(grid[-1, :].tolist())
    border_colors.extend(grid[:, 0].tolist())
    border_colors.extend(grid[:, -1].tolist())

    vals, counts = np.unique(border_colors, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    fg = None
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg:
                fg = int(grid[r, c])
                break
        if fg is not None:
            break

    if fg is None:
        return grid

    fg_mask = (grid == fg)

    # ------------------------------------------------------------------
    # Connected components of foreground, 4-connectivity.
    # ------------------------------------------------------------------
    label = np.zeros((h, w), dtype=int)
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if fg_mask[r, c] and label[r, c] == 0:
                comp_id += 1
                q = deque([(r, c)])
                label[r, c] = comp_id
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if fg_mask[nr, nc] and label[nr, nc] == 0:
                                label[nr, nc] = comp_id
                                q.append((nr, nc))

    ncomp = comp_id

    if ncomp == 0:
        return np.full((h, w), bg, dtype=int)

    comp_cells = []
    for i in range(1, ncomp + 1):
        rs, cs = np.where(label == i)
        comp_cells.append(list(zip(rs.tolist(), cs.tolist())))

    comp_bbox = []
    for cells in comp_cells:
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        comp_bbox.append((min(rs), max(rs), min(cs), max(cs)))

    comp_area = [len(cells) for cells in comp_cells]

    # Degree map for foreground pixels (4-neighbor).
    deg = np.zeros((h, w), dtype=int)
    for r, c in zip(*np.where(fg_mask)):
        if r + 1 < h and fg_mask[r + 1, c]:
            deg[r, c] += 1
        if r - 1 >= 0 and fg_mask[r - 1, c]:
            deg[r, c] += 1
        if c + 1 < w and fg_mask[r, c + 1]:
            deg[r, c] += 1
        if c - 1 >= 0 and fg_mask[r, c - 1]:
            deg[r, c] += 1

    # A closed curve/object is accepted if it has no endpoints.  Degree-4
    # junctions are allowed because the generated objects may touch/cross.
    is_closed = []
    for i, cells in enumerate(comp_cells, start=1):
        ok = len(cells) >= 3
        for r, c in cells:
            if deg[r, c] == 1 or deg[r, c] == 3:
                ok = False
                break
        is_closed.append(ok)

    # ------------------------------------------------------------------
    # Background holes: background cells not connected to the image border.
    # ------------------------------------------------------------------
    bg_mask = ~fg_mask
    hole_label = np.zeros((h, w), dtype=int)
    outside = 1
    hole_id = 1

    # Seed outside background from border.
    q = deque()
    for r in range(h):
        if bg_mask[r, 0]:
            hole_label[r, 0] = outside
            q.append((r, 0))
        if bg_mask[r, w - 1]:
            hole_label[r, w - 1] = outside
            q.append((r, w - 1))
    for c in range(w):
        if bg_mask[0, c]:
            hole_label[0, c] = outside
            q.append((0, c))
        if bg_mask[h - 1, c]:
            hole_label[h - 1, c] = outside
            q.append((h - 1, c))

    while q:
        rr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                if bg_mask[nr, nc] and hole_label[nr, nc] == 0:
                    hole_label[nr, nc] = outside
                    q.append((nr, nc))

    # Label remaining bg components as holes.
    for r in range(h):
        for c in range(w):
            if bg_mask[r, c] and hole_label[r, c] == 0:
                hole_id += 1
                q = deque([(r, c)])
                hole_label[r, c] = hole_id
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if bg_mask[nr, nc] and hole_label[nr, nc] == 0:
                                hole_label[nr, nc] = hole_id
                                q.append((nr, nc))

    # For each hole, bbox and adjacent foreground components.
    hole_bbox = {}
    hole_adj_fg = {}
    for hid in range(2, hole_id + 1):
        rs, cs = np.where(hole_label == hid)
        if len(rs) == 0:
            continue
        hole_bbox[hid] = (rs.min(), rs.max(), cs.min(), cs.max())
        adj = set()
        for r, c in zip(rs.tolist(), cs.tolist()):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if fg_mask[nr, nc]:
                        adj.add(int(label[nr, nc]))
        hole_adj_fg[hid] = adj

    # For each closed component, choose the largest hole it surrounds.
    comp_hole = {}
    for i in range(1, ncomp + 1):
        if not is_closed[i - 1]:
            continue
        best_hole = None
        best_area = -1
        for hid, adj in hole_adj_fg.items():
            if i in adj:
                r0, r1, c0, c1 = hole_bbox[hid]
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if area > best_area:
                    best_area = area
                    best_hole = hid
        comp_hole[i] = best_hole

    # Spatial region used to determine child placement.
    # For closed objects, use the surrounding hole expanded by the wall thickness
    # (i.e. the component's inner envelope).  For open objects, use their bbox.
    comp_region = []
    for i in range(1, ncomp + 1):
        r0, r1, c0, c1 = comp_bbox[i - 1]
        if is_closed[i - 1] and comp_hole.get(i) is not None:
            hr0, hr1, hc0, hc1 = hole_bbox[comp_hole[i]]
            # Expand the hole by one wall pixel, but stay inside the component bbox.
            region = (
                max(r0, hr0 - 1),
                min(r1, hr1 + 1),
                max(c0, hc0 - 1),
                min(c1, hc1 + 1),
            )
        else:
            region = (r0, r1, c0, c1)
        comp_region.append(region)

    # ------------------------------------------------------------------
    # Output size for each component.
    # ------------------------------------------------------------------
    def output_size_for_comp(i):
        # i is 1-based component id.
        r0, r1, c0, c1 = comp_region[i - 1]
        rh = r1 - r0 + 1
        rw = c1 - c0 + 1

        if is_closed[i - 1]:
            oh = max(3, (rh // 2) + 1)
            ow = max(3, (rw // 2) + 1)
        else:
            oh = 1
            ow = 1
        return oh, ow

    comp_out_size = [output_size_for_comp(i) for i in range(1, ncomp + 1)]

    # ------------------------------------------------------------------
    # Containment hierarchy.
    # A component is a child of the smallest closed component whose region
    # contains its center.
    # ------------------------------------------------------------------
    def center(comp_idx):
        r0, r1, c0, c1 = comp_region[comp_idx]
        return (r0 + r1) / 2.0, (c0 + c1) / 2.0

    parent = {}
    children = {i: [] for i in range(1, ncomp + 1)}

    for i in range(1, ncomp + 1):
        cr, cc = center(i - 1)
        best = None
        best_area = float("inf")
        for j in range(1, ncomp + 1):
            if i == j or not is_closed[j - 1]:
                continue
            r0, r1, c0, c1 = comp_region[j - 1]
            if r0 <= cr <= r1 and c0 <= cc <= c1:
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if area < best_area:
                    best_area = area
                    best = j
        parent[i] = best

    for i in range(1, ncomp + 1):
        if parent[i] is not None:
            children[parent[i]].append(i)

    top_level = [i for i in range(1, ncomp + 1) if parent[i] is None]
    if not top_level:
        top_level = [1]

    # ------------------------------------------------------------------
    # Draw components.
    # ------------------------------------------------------------------
    def draw_rectangle(out, r, c, hh, ww, color):
        # Draw a rectangle border of size hh x ww with top-left at (r,c).
        for rr in range(r, r + hh):
            out[rr, c] = color
            out[rr, c + ww - 1] = color
        for cc in range(c, c + ww):
            out[r, cc] = color
            out[r + hh - 1, cc] = color

    def draw_point(out, r, c, color):
        out[r, c] = color

    def clamp(v, lo, hi):
        if hi < lo:
            return lo
        return int(max(lo, min(hi, v)))

    # Place children inside a parent.  Returns the top-left position of the
    # parent rectangle and a list of (child_id, child_top_left) in parent-local
    # coordinates.
    def place_children_local(parent_id):
        hh, ww = comp_out_size[parent_id - 1]
        r0, r1, c0, c1 = comp_region[parent_id - 1]
        ph = max(1, r1 - r0 + 1)
        pw = max(1, c1 - c0 + 1)

        chs = sorted(children[parent_id], key=lambda x: (center(x - 1)[0], center(x - 1)[1]))
        placements = []

        interior_h = max(1, hh - 2)
        interior_w = max(1, ww - 2)

        for child in chs:
            cr, cc = center(child - 1)
            # Normalize child center in parent region.
            nr = (cr - (r0 + 0.5)) / max(1, ph)
            nc = (cc - (c0 + 0.5)) / max(1, pw)
            nr = max(-0.25, min(1.25, nr))
            nc = max(-0.25, min(1.25, nc))

            chh, cww = comp_out_size[child - 1]

            if chh == 1 and cww == 1:
                yr = clamp(int(round(1 + nr * (interior_h - 1))), 1, hh - 2)
                xc = clamp(int(round(1 + nc * (interior_w - 1))), 1, ww - 2)
                placements.append((child, (yr, xc)))
            else:
                # Center the child rectangle at the scaled position, clamped to
                # remain inside the parent interior.
                target_r = 1 + nr * (interior_h - 1)
                target_c = 1 + nc * (interior_w - 1)

                yr = clamp(int(round(target_r - (chh - 1) / 2.0)), 1, max(1, hh - chh - 1))
                xc = clamp(int(round(target_c - (cww - 1) / 2.0)), 1, max(1, ww - cww - 1))
                placements.append((child, (yr, xc)))

        return hh, ww, placements

    # Build output.
    # First, choose a canvas.  If there is a single top-level closed object,
    # use its rectangle as the whole canvas.  Otherwise, lay out top-level
    # objects relative to the largest closed top-level object (or to the global
    # bbox if none is closed).
    closed_tops = [i for i in top_level if is_closed[i - 1]]
    main = None
    if closed_tops:
        main = max(closed_tops, key=lambda i: (comp_region[i - 1][1] - comp_region[i - 1][0] + 1) *
                  (comp_region[i - 1][3] - comp_region[i - 1][2] + 1))

    if main is not None and (len(top_level) == 1 or all(parent[i] == main for i in top_level)):
        # Simple canvas: the main rectangle.
        H, W = comp_out_size[main - 1]
        out = np.zeros((H, W), dtype=int)

        # Draw main border, then recursively place children.
        def render_in(out, comp, top_r, top_c):
            hh, ww = comp_out_size[comp - 1]
            if is_closed[comp - 1]:
                draw_rectangle(out, top_r, top_c, hh, ww, fg)
                _, _, placements = place_children_local(comp)
                for child, (yr, xc) in placements:
                    render_in(out, child, top_r + yr, top_c + xc)
            else:
                draw_point(out, top_r, top_c, fg)

        render_in(out, main, 0, 0)
    else:
        # General top-level layout.
        # Use the global bbox of all components as the coordinate reference.
        all_r0 = min(r for r, c, _, _ in comp_region)
        all_r1 = max(r1 for _, r1, _, _ in comp_region)
        all_c0 = min(c for _, _, c, _ in comp_region)
        all_c1 = max(c1 for _, _, _, c1 in comp_region)

        ref_r0, ref_c0 = all_r0, all_c0
        ref_h = max(1, all_r1 - all_r0 + 1)
        ref_w = max(1, all_c1 - all_c0 + 1)

        # Determine scale from the largest top-level closed component if present.
        scale_r = 1.0
        scale_c = 1.0
        if closed_tops:
            anchor = max(closed_tops, key=lambda i: (comp_region[i - 1][1] - comp_region[i - 1][0] + 1) *
                       (comp_region[i - 1][3] - comp_region[i - 1][2] + 1))
            ph = max(1, comp_region[anchor - 1][1] - comp_region[anchor - 1][0] + 1)
            pw = max(1, comp_region[anchor - 1][3] - comp_region[anchor - 1][2] + 1)
            oh, ow = comp_out_size[anchor - 1]
            scale_r = max(0.05, (oh - 1) / float(ph))
            scale_c = max(0.05, (ow - 1) / float(pw))
        else:
            # No closed top-level object; use a conservative half-scale.
            scale_r = 0.5
            scale_c = 0.5

        # Place top-level objects in a virtual canvas.
        placed = []
        for comp in top_level:
            cr, cc = center(comp - 1)
            vr = (cr - (ref_r0 + 0.5)) * scale_r
            vc = (cc - (ref_c0 + 0.5)) * scale_c
            oh, ow = comp_out_size[comp - 1]
            # top-left in virtual coordinates, keeping object centered at (vr,vc)
            tr = vr - (oh - 1) / 2.0
            tc = vc - (ow - 1) / 2.0
            placed.append((comp, float(tr), float(tc), oh, ow))

        if not placed:
            out = np.zeros((1, 1), dtype=int)
        else:
            min_r = min(max(0.0, p[1]) for p in placed)
            min_c = min(max(0.0, p[2]) for p in placed)
            max_r = max(p[1] + p[3] for p in placed)
            max_c = max(p[2] + p[4] for p in placed)

            H = int(np.ceil(max_r - min_r)) + 1
            W = int(np.ceil(max_c - min_c)) + 1
            H = max(H, 1)
            W = max(W, 1)
            out = np.zeros((H, W), dtype=int)

            def render_global(comp, top_r, top_c):
                hh, ww = comp_out_size[comp - 1]
                if is_closed[comp - 1]:
                    draw_rectangle(out, top_r, top_c, hh, ww, fg)
                    _, _, placements = place_children_local(comp)
                    for child, (yr, xc) in placements:
                        render_global(child, top_r + yr, top_c + xc)
                else:
                    draw_point(out, top_r, top_c, fg)

            for comp, vr, vc, oh, ow in placed:
                tr = int(round(vr - min_r))
                tc = int(round(vc - min_c))
                render_global(comp, tr, tc)

    # Crop to non-background pixels.
    rows = np.any(out == fg, axis=1)
    cols = np.any(out == fg, axis=0)
    if not np.any(rows) or not np.any(cols):
        # If nothing was drawn, return a single foreground pixel.
        return np.array([[fg]], dtype=int)

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    out = out[r0:r1 + 1, c0:c1 + 1]

    # Replace the implicit 0 background by the original background color.
    out[out == 0] = bg
    return out.astype(int, copy=False)