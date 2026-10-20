import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    - The input has a background color, a monochrome "mask"/stencil object,
      and a colored "tile" object.
    - The mask object is the one made from a single non-background color.
    - The tile object is the other non-background object, usually multicolored.
    - The output is the mask's bounding box scaled by the tile's size:
      every mask pixel is replaced by the full tile pattern; mask-background
      pixels become background.
    """
    g = np.array(grid, dtype=int)
    if g.size == 0:
        return g.copy()

    H, W = g.shape

    # ------------------------------------------------------------------
    # 1. Determine the background color.
    #    Prefer the dominant border color when the border is clearly
    #    background; otherwise fall back to the globally most common color.
    # ------------------------------------------------------------------
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    bvals, bcounts = np.unique(border, return_counts=True)
    border_bg = int(bvals[np.argmax(bcounts)])

    if bcounts.max() > len(border) // 2:
        bg = border_bg
    else:
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(counts)])

    nonbg = g != bg
    if not nonbg.any():
        return np.full((H, W), bg, dtype=g.dtype)

    # ------------------------------------------------------------------
    # Small helpers.
    # ------------------------------------------------------------------
    def bbox_area(bbox):
        r1, r2, c1, c2 = bbox
        return (r2 - r1 + 1) * (c2 - c1 + 1)

    def is_contained(a, b):
        ar1, ar2, ac1, ac2 = a
        br1, br2, bc1, bc2 = b
        return br1 <= ar1 and br2 >= ar2 and bc1 <= ac1 and bc2 >= ac2

    def bbox_from_mask(mask):
        rs, cs = np.where(mask)
        if rs.size == 0:
            return None
        return (int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max()))

    # ------------------------------------------------------------------
    # 2. Find 8-connected components of non-background pixels.
    #    The tile is usually one multi-colored component.
    #    The mask may be one or several same-color components.
    # ------------------------------------------------------------------
    def get_components(mask):
        visited = np.zeros((H, W), dtype=bool)
        comps = []

        for i in range(H):
            for j in range(W):
                if not mask[i, j] or visited[i, j]:
                    continue

                stack = [(i, j)]
                visited[i, j] = True
                cells = []
                rmin, rmax = H, -1
                cmin, cmax = W, -1

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    if r < rmin: rmin = r
                    if r > rmax: rmax = r
                    if c < cmin: cmin = c
                    if c > cmax: cmax = c

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < H and 0 <= nc < W and
                                mask[nr, nc] and not visited[nr, nc]):
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                colors = {int(g[r, c]) for r, c in cells}
                comps.append({
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                    "count": len(cells),
                    "colors": colors
                })

        return comps

    # ------------------------------------------------------------------
    # Primary selection method:
    # use connected components. The multi-colored component is the tile.
    # The mask color is a color appearing only in single-color components
    # and not contained inside the tile's bounding box.
    # ------------------------------------------------------------------
    def try_component_selection():
        comps = get_components(nonbg)
        if not comps:
            return None

        complex_comps = [c for c in comps if len(c["colors"]) > 1]
        if not complex_comps:
            return None

        tile_comp = max(
            complex_comps,
            key=lambda c: (
                len(c["colors"]),
                c["count"],
                bbox_area((c["rmin"], c["rmax"], c["cmin"], c["cmax"]))
            )
        )
        tile_bbox = (tile_comp["rmin"], tile_comp["rmax"],
                     tile_comp["cmin"], tile_comp["cmax"])

        color_comps = {}
        for comp in comps:
            for col in comp["colors"]:
                color_comps.setdefault(col, []).append(comp)

        candidates = []
        for col, ccomps in color_comps.items():
            # If this color is part of a multi-color component,
            # it belongs to the tile, not the mask.
            if any(len(cc["colors"]) > 1 for cc in ccomps):
                continue

            rmin = min(cc["rmin"] for cc in ccomps)
            rmax = max(cc["rmax"] for cc in ccomps)
            cmin = min(cc["cmin"] for cc in ccomps)
            cmax = max(cc["cmax"] for cc in ccomps)
            count = sum(cc["count"] for cc in ccomps)
            bbox = (rmin, rmax, cmin, cmax)
            contained = is_contained(bbox, tile_bbox)

            candidates.append((col, bbox, count, contained))

        # Prefer a mask candidate outside the tile's bounding box.
        outside = [x for x in candidates if not x[3]]
        if not outside:
            return None

        mask_color, mask_bbox, _, _ = max(
            outside,
            key=lambda x: (x[2], bbox_area(x[1]))
        )

        # The tile is everything non-background except the chosen mask color.
        tile_mask = nonbg & (g != mask_color)
        if not tile_mask.any():
            return None

        tile_bbox = bbox_from_mask(tile_mask)
        mask_bbox = bbox_from_mask(g == mask_color)

        if tile_bbox is None or mask_bbox is None:
            return None

        return mask_color, mask_bbox, tile_bbox

    # ------------------------------------------------------------------
    # Fallback selection method:
    # group colors whose bounding boxes overlap. A multi-color group is
    # the tile; a separate single-color group is the mask.
    # ------------------------------------------------------------------
    def try_color_group_selection():
        colors = [int(c) for c in np.unique(g[nonbg])]
        if len(colors) < 2:
            return None

        bboxes = {}
        counts = {}
        for c in colors:
            rs, cs = np.where(g == c)
            bboxes[c] = (int(rs.min()), int(rs.max()),
                         int(cs.min()), int(cs.max()))
            counts[c] = int(len(rs))

        parent = {c: c for c in colors}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def bboxes_intersect(a, b):
            r1, r2, c1, c2 = a
            r3, r4, c3, c4 = b
            return r1 <= r4 and r2 >= r3 and c1 <= c4 and c2 >= c3

        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                a, b = colors[i], colors[j]
                if bboxes_intersect(bboxes[a], bboxes[b]):
                    union(a, b)

        groups = {}
        for c in colors:
            groups.setdefault(find(c), []).append(c)

        infos = []
        for cols in groups.values():
            rmin = min(bboxes[c][0] for c in cols)
            rmax = max(bboxes[c][1] for c in cols)
            cmin = min(bboxes[c][2] for c in cols)
            cmax = max(bboxes[c][3] for c in cols)
            count = sum(counts[c] for c in cols)
            bbox = (rmin, rmax, cmin, cmax)
            infos.append({
                "colors": set(cols),
                "bbox": bbox,
                "count": count,
                "area": bbox_area(bbox)
            })

        multi = [info for info in infos if len(info["colors"]) > 1]

        if multi:
            tile = max(
                multi,
                key=lambda x: (len(x["colors"]), x["count"], x["area"])
            )
            singles = [
                info for info in infos
                if info is not tile and len(info["colors"]) == 1
            ]
            if not singles:
                return None

            outside = [
                info for info in singles
                if not is_contained(info["bbox"], tile["bbox"])
            ]
            candidates = outside if outside else singles
            mask = max(candidates, key=lambda x: (x["count"], x["area"]))
            mask_color = next(iter(mask["colors"]))
        else:
            # All groups are single-color. Use a conservative heuristic:
            # the larger group is usually the tile, the other is the mask.
            if len(infos) < 2:
                return None

            sorted_infos = sorted(
                infos,
                key=lambda x: (x["count"], x["area"]),
                reverse=True
            )
            tile = sorted_infos[0]
            mask = sorted_infos[1]

            # If the first choice looks like a mask because it has holes,
            # and the second does not, swap them.
            holes_tile = tile["area"] - tile["count"]
            holes_mask = mask["area"] - mask["count"]
            if holes_mask == 0 and holes_tile > 0:
                tile, mask = mask, tile

            mask_color = next(iter(mask["colors"]))

        tile_mask = nonbg & (g != mask_color)
        if not tile_mask.any():
            return None

        tile_bbox = bbox_from_mask(tile_mask)
        mask_bbox = bbox_from_mask(g == mask_color)

        if tile_bbox is None or mask_bbox is None:
            return None

        return mask_color, mask_bbox, tile_bbox

    # ------------------------------------------------------------------
    # Choose the best interpretation.
    # ------------------------------------------------------------------
    result = try_component_selection()
    if result is None:
        result = try_color_group_selection()

    if result is None:
        return np.full((H, W), bg, dtype=g.dtype)

    mask_color, mask_bbox, tile_bbox = result

    # ------------------------------------------------------------------
    # 3. Extract the binary mask and the colored tile.
    # ------------------------------------------------------------------
    mr1, mr2, mc1, mc2 = mask_bbox
    mask_sub = g[mr1:mr2 + 1, mc1:mc2 + 1]
    mask_pattern = mask_sub == mask_color

    # Safety fallback if something unexpected happened.
    if not mask_pattern.any():
        mask_pattern = mask_sub != bg

    tr1, tr2, tc1, tc2 = tile_bbox
    tile_sub = g[tr1:tr2 + 1, tc1:tc2 + 1]

    mh, mw = mask_pattern.shape
    th, tw = tile_sub.shape

    # ------------------------------------------------------------------
    # 4. Build the output by stamping the tile for every mask pixel.
    # ------------------------------------------------------------------
    out = np.full((mh * th, mw * tw), bg, dtype=g.dtype)

    for r in range(mh):
        for c in range(mw):
            if mask_pattern[r, c]:
                out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile_sub

    return out