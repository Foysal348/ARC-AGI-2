import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Convert a grid of nested/enclosing foreground components into a compact
    schematic diagram.

    The schematic contains:
      - background color = most frequent input color
      - one-pixel dots for foreground components that do not enclose holes
      - rectangular frames for foreground components that enclose holes
    Children are recursively placed inside their parent frame.
    """
    if grid is None:
        return np.array([], dtype=int)

    grid = np.array(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    if np.all(flat == bg):
        return grid.copy()

    h, w = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def label_components(mask):
        """4-connected component labeling."""
        labels = -np.ones(mask.shape, dtype=int)
        comp_id = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == -1:
                    stack = [(r, c)]
                    labels[r, c] = comp_id

                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in dirs:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if mask[nr, nc] and labels[nr, nc] == -1:
                                    labels[nr, nc] = comp_id
                                    stack.append((nr, nc))

                    comp_id += 1

        return labels, comp_id

    obj_mask = grid != bg
    if not obj_mask.any():
        return grid.copy()

    obj_labels, n_obj = label_components(obj_mask)
    if n_obj == 0:
        return grid.copy()

    bg_labels, n_bg = label_components(~obj_mask)

    comp_bbox = []
    comp_area = []
    comp_color = []

    for cid in range(n_obj):
        ys, xs = np.where(obj_labels == cid)
        rmin, rmax = int(ys.min()), int(ys.max())
        cmin, cmax = int(xs.min()), int(xs.max())

        comp_bbox.append((rmin, rmax, cmin, cmax))
        comp_area.append((rmax - rmin + 1) * (cmax - cmin + 1))

        pix = grid[ys, xs]
        vv, cc = np.unique(pix, return_counts=True)
        comp_color.append(int(vv[np.argmax(cc)]))

    # Background components touching the grid border are outside; others are holes.
    outside = set()
    for cid in range(n_bg):
        if (
            np.any(bg_labels[0, :] == cid)
            or np.any(bg_labels[-1, :] == cid)
            or np.any(bg_labels[:, 0] == cid)
            or np.any(bg_labels[:, -1] == cid)
        ):
            outside.add(cid)

    holes = [cid for cid in range(n_bg) if cid not in outside]

    parent = [None] * n_obj
    has_hole = [False] * n_obj

    # Determine parent/child relationships from holes.
    for hole in holes:
        ys, xs = np.where(bg_labels == hole)
        hrmin, hrmax = int(ys.min()), int(ys.max())
        hcmin, hcmax = int(xs.min()), int(xs.max())

        adj_counts = {}
        for r, c in zip(ys, xs):
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    oid = int(obj_labels[nr, nc])
                    if oid != -1:
                        adj_counts[oid] = adj_counts.get(oid, 0) + 1

        if not adj_counts:
            continue

        def contains_hole(bbox):
            brmin, brmax, bcmin, bcmax = bbox
            return (
                brmin <= hrmin
                and brmax >= hrmax
                and bcmin <= hcmin
                and bcmax >= hcmax
            )

        # The enclosing component should contain the hole bbox and usually
        # contributes the largest boundary contact.
        enclosure = max(
            adj_counts.keys(),
            key=lambda cid: (contains_hole(comp_bbox[cid]), adj_counts[cid], comp_area[cid]),
        )

        has_hole[enclosure] = True

        for cid in adj_counts:
            if cid == enclosure:
                continue

            # A parent should be spatially larger than the component it contains.
            if comp_area[enclosure] <= comp_area[cid]:
                continue

            # Prefer the smallest valid enclosing parent (most immediate parent).
            if parent[cid] is None or comp_area[parent[cid]] > comp_area[enclosure]:
                parent[cid] = enclosure

    children = [[] for _ in range(n_obj)]
    for cid, p in enumerate(parent):
        if p is not None and p != cid and 0 <= p < n_obj:
            children[p].append(cid)

    top_ids = [i for i in range(n_obj) if parent[i] is None]
    if not top_ids:
        top_ids = [int(np.argmax(comp_area))]

    is_container = [has_hole[i] or len(children[i]) > 0 for i in range(n_obj)]

    class Anchor:
        """A local anchor inside a container, corresponding to a descendant object."""
        __slots__ = ("rmin", "rmax", "cmin", "cmax", "r", "c", "h", "w")

        def __init__(self, rmin, rmax, cmin, cmax, r, c, hh, ww):
            self.rmin = rmin
            self.rmax = rmax
            self.cmin = cmin
            self.cmax = cmax
            self.r = r
            self.c = c
            self.h = hh
            self.w = ww

    class Item:
        """A schematic object with its original bbox and descendant anchors."""
        __slots__ = ("grid", "h", "w", "rmin", "rmax", "cmin", "cmax", "is_leaf", "anchors")

        def __init__(self, grid, rmin, rmax, cmin, cmax, is_leaf, anchors=None):
            self.grid = grid
            self.h = int(grid.shape[0])
            self.w = int(grid.shape[1])
            self.rmin = rmin
            self.rmax = rmax
            self.cmin = cmin
            self.cmax = cmax
            self.is_leaf = is_leaf
            self.anchors = anchors if anchors is not None else []

    def layout_items(items):
        """
        Lay out sibling schematic items in a tight background canvas.

        Items are grouped into rows/columns by overlap of their original bboxes.
        Single-pixel objects are additionally aligned to overlapping anchors
        of neighboring containers when that matches the original layout better.
        """
        n = len(items)
        if n == 0:
            return np.full((1, 1), bg, dtype=int), []

        def make_clusters(overlap_func, center_key):
            parent_u = list(range(n))

            def find(x):
                while parent_u[x] != x:
                    parent_u[x] = parent_u[parent_u[x]]
                    x = parent_u[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent_u[rb] = ra

            for i in range(n):
                for j in range(i + 1, n):
                    if overlap_func(items[i], items[j]):
                        union(i, j)

            groups = {}
            for i in range(n):
                groups.setdefault(find(i), []).append(i)

            clusters = sorted(
                groups.values(),
                key=lambda idxs: sum(center_key(items[i]) for i in idxs),
            )

            ids = [0] * n
            for rid, idxs in enumerate(clusters):
                for i in idxs:
                    ids[i] = rid

            return ids, clusters

        def same_row(a, b):
            return a.rmin <= b.rmax and b.rmin <= a.rmax

        def same_col(a, b):
            return a.cmin <= b.cmax and b.cmin <= a.cmax

        row_ids, _ = make_clusters(same_row, lambda it: it.rmin + it.rmax)
        col_ids, _ = make_clusters(same_col, lambda it: it.cmin + it.cmax)

        # If two items overlap in both dimensions, separate them along the
        # axis with larger center spread.
        cells = {}
        for i in range(n):
            cells.setdefault((row_ids[i], col_ids[i]), []).append(i)

        next_row = max(row_ids) + 1 if row_ids else 0
        next_col = max(col_ids) + 1 if col_ids else 0

        for idxs in cells.values():
            if len(idxs) <= 1:
                continue

            cx = [items[i].cmin + items[i].cmax for i in idxs]
            cy = [items[i].rmin + items[i].rmax for i in idxs]

            horiz_spread = max(cx) - min(cx)
            vert_spread = max(cy) - min(cy)

            if horiz_spread >= vert_spread:
                for i in sorted(
                    idxs,
                    key=lambda x: (items[x].cmin + items[x].cmax, items[x].rmin + items[x].rmax),
                ):
                    col_ids[i] = next_col
                    next_col += 1
            else:
                for i in sorted(
                    idxs,
                    key=lambda x: (items[x].rmin + items[x].rmax, items[x].cmin + items[x].cmax),
                ):
                    row_ids[i] = next_row
                    next_row += 1

        row_groups = {}
        for i in range(n):
            row_groups.setdefault(row_ids[i], []).append(i)

        row_clusters = sorted(
            row_groups.values(),
            key=lambda idxs: sum(items[i].rmin + items[i].rmax for i in idxs),
        )

        row_id_final = [0] * n
        for rid, idxs in enumerate(row_clusters):
            for i in idxs:
                row_id_final[i] = rid

        col_groups = {}
        for i in range(n):
            col_groups.setdefault(col_ids[i], []).append(i)

        col_clusters = sorted(
            col_groups.values(),
            key=lambda idxs: sum(items[i].cmin + items[i].cmax for i in idxs),
        )

        col_id_final = [0] * n
        for cid, idxs in enumerate(col_clusters):
            for i in idxs:
                col_id_final[i] = cid

        row_heights = [max(items[i].h for i in idxs) for idxs in row_clusters]
        col_widths = [max(items[i].w for i in idxs) for idxs in col_clusters]

        row_offsets = []
        y = 0
        for hh in row_heights:
            row_offsets.append(y)
            y += hh + 1
        canvas_h = y - 1 if row_heights else 1

        col_offsets = []
        x = 0
        for ww in col_widths:
            col_offsets.append(x)
            x += ww + 1
        canvas_w = x - 1 if col_widths else 1

        pos = []
        for i, item in enumerate(items):
            rid = row_id_final[i]
            cid = col_id_final[i]
            r0 = row_offsets[rid] + (row_heights[rid] - item.h) // 2
            c0 = col_offsets[cid] + (col_widths[cid] - item.w) // 2
            pos.append((r0, c0))

        initial_pos = pos[:]

        def rects_overlap(r1, c1, h1, w1, r2, c2, h2, w2):
            return (
                max(0, min(r1 + h1, r2 + h2) - max(r1, r2))
                * max(0, min(c1 + w1, c2 + w2) - max(c1, c2))
            ) > 0

        # Anchor-based alignment for dot-like objects.
        for i, item in enumerate(items):
            if not item.is_leaf:
                continue

            cur_r, cur_c = pos[i]
            best_r = None
            best_r_score = None
            best_c = None
            best_c_score = None

            for j, other in enumerate(items):
                if j == i:
                    continue
                if other.is_leaf or not other.anchors:
                    continue

                or_r, or_c = initial_pos[j]

                # Vertical alignment, only within the same final row cluster.
                if row_id_final[i] == row_id_final[j]:
                    for a in other.anchors:
                        ov = min(item.rmax, a.rmax) - max(item.rmin, a.rmin) + 1
                        if ov <= 0:
                            continue

                        cand_r = or_r + a.r + (a.h - item.h) // 2
                        if not (0 <= cand_r and cand_r + item.h <= canvas_h):
                            continue

                        ok = True
                        for k in range(n):
                            if k == i:
                                continue
                            r2, c2 = initial_pos[k]
                            if rects_overlap(
                                cand_r,
                                cur_c,
                                item.h,
                                item.w,
                                r2,
                                c2,
                                items[k].h,
                                items[k].w,
                            ):
                                ok = False
                                break

                        if ok:
                            area = (a.rmax - a.rmin + 1) * (a.cmax - a.cmin + 1)
                            score = (
                                ov,
                                -abs((item.rmin + item.rmax) - (a.rmin + a.rmax)),
                                -area,
                            )
                            if best_r_score is None or score > best_r_score:
                                best_r = cand_r
                                best_r_score = score

                # Horizontal alignment, only within the same final column cluster.
                if col_id_final[i] == col_id_final[j]:
                    for a in other.anchors:
                        ov = min(item.cmax, a.cmax) - max(item.cmin, a.cmin) + 1
                        if ov <= 0:
                            continue

                        cand_c = or_c + a.c + (a.w - item.w) // 2
                        if not (0 <= cand_c and cand_c + item.w <= canvas_w):
                            continue

                        r_check = best_r if best_r is not None else cur_r
                        ok = True
                        for k in range(n):
                            if k == i:
                                continue
                            r2, c2 = initial_pos[k]
                            if rects_overlap(
                                r_check,
                                cand_c,
                                item.h,
                                item.w,
                                r2,
                                c2,
                                items[k].h,
                                items[k].w,
                            ):
                                ok = False
                                break

                        if ok:
                            area = (a.rmax - a.rmin + 1) * (a.cmax - a.cmin + 1)
                            score = (
                                ov,
                                -abs((item.cmin + item.cmax) - (a.cmin + a.cmax)),
                                -area,
                            )
                            if best_c_score is None or score > best_c_score:
                                best_c = cand_c
                                best_c_score = score

            if best_r is not None:
                pos[i] = (best_r, pos[i][1])
            if best_c is not None:
                pos[i] = (pos[i][0], best_c)

        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        placements = []

        for i, item in enumerate(items):
            r, c = pos[i]
            canvas[r : r + item.h, c : c + item.w] = item.grid
            placements.append((item, r, c))

        return canvas, placements

    memo = {}

    def build_component(cid):
        """Build the schematic item for one original component."""
        if cid in memo:
            return memo[cid]

        color = comp_color[cid]
        rmin, rmax, cmin, cmax = comp_bbox[cid]

        if not is_container[cid]:
            g = np.full((1, 1), color, dtype=int)
            item = Item(g, rmin, rmax, cmin, cmax, True, [])

        else:
            ch = children[cid]

            if not ch:
                # Minimal empty frame: one background pixel of interior.
                g = np.full((3, 3), bg, dtype=int)
                g[0, :] = color
                g[-1, :] = color
                g[:, 0] = color
                g[:, -1] = color
                item = Item(g, rmin, rmax, cmin, cmax, False, [])

            else:
                child_items = [build_component(c) for c in ch]
                content, placements = layout_items(child_items)
                ch_h, ch_w = content.shape

                # Frame + one-pixel background padding around child canvas.
                g = np.full((ch_h + 4, ch_w + 4), bg, dtype=int)
                g[2 : 2 + ch_h, 2 : 2 + ch_w] = content

                g[0, :] = color
                g[-1, :] = color
                g[:, 0] = color
                g[:, -1] = color

                anchors = []
                for child_item, r, c in placements:
                    # The child itself is an anchor.
                    anchors.append(
                        Anchor(
                            child_item.rmin,
                            child_item.rmax,
                            child_item.cmin,
                            child_item.cmax,
                            r + 2,
                            c + 2,
                            child_item.h,
                            child_item.w,
                        )
                    )

                    # Descendant anchors are offset into this container.
                    for a in child_item.anchors:
                        anchors.append(
                            Anchor(
                                a.rmin,
                                a.rmax,
                                a.cmin,
                                a.cmax,
                                r + 2 + a.r,
                                c + 2 + a.c,
                                a.h,
                                a.w,
                            )
                        )

                item = Item(g, rmin, rmax, cmin, cmax, False, anchors)

        memo[cid] = item
        return item

    top_items = [build_component(cid) for cid in top_ids]
    canvas, placements = layout_items(top_items)

    # Keep top-level dots from sitting flush on the schematic border when
    # multiple top-level objects exist.
    if len(top_items) > 1:
        exp_top = any(item.is_leaf and r == 0 for item, r, c in placements)
        exp_bottom = any(item.is_leaf and r + item.h == canvas.shape[0] for item, r, c in placements)
        exp_left = any(item.is_leaf and c == 0 for item, r, c in placements)
        exp_right = any(item.is_leaf and c + item.w == canvas.shape[1] for item, r, c in placements)

        if exp_top or exp_bottom or exp_left or exp_right:
            add_top = int(exp_top)
            add_bottom = int(exp_bottom)
            add_left = int(exp_left)
            add_right = int(exp_right)

            new_h = canvas.shape[0] + add_top + add_bottom
            new_w = canvas.shape[1] + add_left + add_right

            new_canvas = np.full((new_h, new_w), bg, dtype=int)
            new_canvas[
                add_top : add_top + canvas.shape[0],
                add_left : add_left + canvas.shape[1],
            ] = canvas
            canvas = new_canvas

    return canvas