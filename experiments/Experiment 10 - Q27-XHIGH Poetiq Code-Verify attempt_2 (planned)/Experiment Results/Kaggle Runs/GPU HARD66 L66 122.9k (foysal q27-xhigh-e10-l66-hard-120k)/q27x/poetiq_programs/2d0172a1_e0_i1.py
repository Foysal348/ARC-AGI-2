import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an input grid into a compact topological diagram of its object components.

    Rules:
      - The most frequent color is background.
      - Object components are 4-connected components of non-background pixels.
      - If an object component encloses a background hole, it is drawn as a rectangle.
      - If it does not enclose a hole, it is drawn as a single pixel.
      - Children are recursively drawn inside the parent rectangle with 2 background pixels
        of padding and 1 background pixel between sibling bounding boxes.
      - Relative left/right and top/bottom order of siblings is preserved using their
        input bounding boxes.
    """
    if grid is None:
        return np.array([], dtype=int)

    grid = np.array(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid

    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    non_bg = [int(v) for v in vals if int(v) != bg]
    if not non_bg:
        return grid.copy()

    # Use the most frequent non-background color as the output object color.
    obj_color = max(non_bg, key=lambda c: int(np.sum(flat == c)))

    obj_mask = (grid != bg)
    bg_mask = ~obj_mask
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def label_components(mask):
        """Label 4-connected components. Returns labels array and component count."""
        h, w = mask.shape
        labels = -np.ones((h, w), dtype=int)
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

    obj_labels, n_obj = label_components(obj_mask)
    if n_obj == 0:
        return grid.copy()

    comp_bbox = []
    comp_area = []
    for cid in range(n_obj):
        ys, xs = np.where(obj_labels == cid)
        rmin, rmax = int(ys.min()), int(ys.max())
        cmin, cmax = int(xs.min()), int(xs.max())
        comp_bbox.append((rmin, rmax, cmin, cmax))
        comp_area.append((rmax - rmin + 1) * (cmax - cmin + 1))

    bg_labels, n_bg = label_components(bg_mask)
    outside = set()
    if n_bg > 0:
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

    # For each background hole, the adjacent largest-bbox object component surrounds it.
    # Other adjacent object components are direct children of that surrounding component.
    for hole in holes:
        ys, xs = np.where(bg_labels == hole)
        adj = set()

        for r, c in zip(ys, xs):
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < bg_labels.shape[0] and 0 <= nc < bg_labels.shape[1]:
                    oid = int(obj_labels[nr, nc])
                    if oid != -1:
                        adj.add(oid)

        if not adj:
            continue

        enclosure = max(adj, key=lambda x: comp_area[x])
        has_hole[enclosure] = True

        for cid in adj:
            if cid == enclosure:
                continue
            # Prefer the nearest enclosing component if multiple candidates appear.
            if parent[cid] is None or comp_area[parent[cid]] > comp_area[enclosure]:
                parent[cid] = enclosure

    children = [[] for _ in range(n_obj)]
    for cid, p in enumerate(parent):
        if p is not None:
            children[p].append(cid)

    top_ids = [i for i in range(n_obj) if parent[i] is None]
    if not top_ids:
        # Fallback for pathological cases: use the largest component as top-level.
        top_ids = [int(np.argmax(comp_area))]

    is_container = [has_hole[i] or len(children[i]) > 0 for i in range(n_obj)]

    class Item:
        __slots__ = ("grid", "h", "w", "rmin", "rmax", "cmin", "cmax", "is_leaf", "cy", "cx")

        def __init__(self, grid, rmin, rmax, cmin, cmax, is_leaf):
            self.grid = grid
            self.h = int(grid.shape[0])
            self.w = int(grid.shape[1])
            self.rmin = rmin
            self.rmax = rmax
            self.cmin = cmin
            self.cmax = cmax
            self.is_leaf = is_leaf
            self.cy = rmin + rmax  # proportional to center row
            self.cx = cmin + cmax  # proportional to center column

    class Placement:
        __slots__ = ("item", "r", "c", "h", "w", "is_leaf")

        def __init__(self, item, r, c):
            self.item = item
            self.r = r
            self.c = c
            self.h = item.h
            self.w = item.w
            self.is_leaf = item.is_leaf

    def layout_items(items, gap=1):
        """
        Layout a list of Items in a tight background canvas.
        Returns (canvas, placements).
        """
        n = len(items)
        if n == 0:
            return np.full((1, 1), bg, dtype=int), []

        def make_clusters(overlap_func, center_attr):
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
                key=lambda idxs: sum(getattr(items[i], center_attr) for i in idxs)
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

        row_ids, row_clusters = make_clusters(same_row, "cy")
        col_ids, _ = make_clusters(same_col, "cx")

        # Resolve rare conflicts where two items map to the same row/column cell.
        cells = {}
        for i in range(n):
            cells.setdefault((row_ids[i], col_ids[i]), []).append(i)

        next_col = max(col_ids) + 1 if col_ids else 0
        for idxs in cells.values():
            if len(idxs) > 1:
                for i in sorted(idxs, key=lambda x: (items[x].cx, items[x].cy)):
                    col_ids[i] = next_col
                    next_col += 1

        col_groups = {}
        for i in range(n):
            col_groups.setdefault(col_ids[i], []).append(i)

        col_clusters = sorted(
            col_groups.values(),
            key=lambda idxs: sum(items[i].cx for i in idxs)
        )

        col_id_final = [0] * n
        for cid, idxs in enumerate(col_clusters):
            for i in idxs:
                col_id_final[i] = cid

        row_heights = []
        for idxs in row_clusters:
            row_heights.append(max(items[i].h for i in idxs))

        col_widths = []
        for idxs in col_clusters:
            col_widths.append(max(items[i].w for i in idxs))

        row_offsets = []
        y = 0
        for h in row_heights:
            row_offsets.append(y)
            y += h + gap
        canvas_h = y - gap if row_heights else 1

        col_offsets = []
        x = 0
        for w in col_widths:
            col_offsets.append(x)
            x += w + gap
        canvas_w = x - gap if col_widths else 1

        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        placements = []

        for i, item in enumerate(items):
            rid = row_ids[i]
            cid = col_id_final[i]

            r0 = row_offsets[rid] + (row_heights[rid] - item.h) // 2
            c0 = col_offsets[cid] + (col_widths[cid] - item.w) // 2

            canvas[r0:r0 + item.h, c0:c0 + item.w] = item.grid
            placements.append(Placement(item, r0, c0))

        return canvas, placements

    memo = {}

    def build_component(cid):
        """Build the local output grid for one object component."""
        if cid in memo:
            return memo[cid]

        if not is_container[cid]:
            g = np.full((1, 1), obj_color, dtype=int)
        else:
            ch = children[cid]
            if not ch:
                # Minimal container with an empty hole.
                g = np.full((5, 5), bg, dtype=int)
                g[0, :] = obj_color
                g[-1, :] = obj_color
                g[:, 0] = obj_color
                g[:, -1] = obj_color
            else:
                items = []
                for child in ch:
                    child_grid = build_component(child)
                    rmin, rmax, cmin, cmax = comp_bbox[child]
                    items.append(Item(child_grid, rmin, rmax, cmin, cmax, not is_container[child]))

                content, _ = layout_items(items)
                ch_h, ch_w = content.shape

                g = np.full((ch_h + 4, ch_w + 4), bg, dtype=int)
                g[2:2 + ch_h, 2:2 + ch_w] = content
                g[0, :] = obj_color
                g[-1, :] = obj_color
                g[:, 0] = obj_color
                g[:, -1] = obj_color

        memo[cid] = g
        return g

    top_items = []
    for cid in top_ids:
        g = build_component(cid)
        rmin, rmax, cmin, cmax = comp_bbox[cid]
        top_items.append(Item(g, rmin, rmax, cmin, cmax, not is_container[cid]))

    canvas, placements = layout_items(top_items)

    # With multiple top-level objects, keep a one-pixel background margin on sides
    # where a top-level point touches the canvas boundary.
    if len(top_items) > 1:
        exp_top = any(p.is_leaf and p.r == 0 for p in placements)
        exp_bottom = any(p.is_leaf and p.r + p.h == canvas.shape[0] for p in placements)
        exp_left = any(p.is_leaf and p.c == 0 for p in placements)
        exp_right = any(p.is_leaf and p.c + p.w == canvas.shape[1] for p in placements)

        if exp_top or exp_bottom or exp_left or exp_right:
            new_h = canvas.shape[0] + exp_top + exp_bottom
            new_w = canvas.shape[1] + exp_left + exp_right
            new_canvas = np.full((new_h, new_w), bg, dtype=int)
            new_canvas[exp_top:exp_top + canvas.shape[0],
                       exp_left:exp_left + canvas.shape[1]] = canvas
            canvas = new_canvas

    return canvas