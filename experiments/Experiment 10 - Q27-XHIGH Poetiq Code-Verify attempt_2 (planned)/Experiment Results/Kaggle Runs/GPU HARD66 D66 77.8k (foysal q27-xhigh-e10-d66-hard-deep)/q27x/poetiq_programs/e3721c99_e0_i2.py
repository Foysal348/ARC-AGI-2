import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor all data objects (color 5) according to the number of enclosed
    background holes they have.

    The grid contains:
      - background color 0
      - a separator made of color 1
      - a legend region containing reference objects of various colors
      - data objects of color 5

    A hole is a 4-connected component of background 0 that does not touch the
    border of the grid.  A hole is assigned to a reference/data object if that
    object touches the boundary of the hole.

    For every 5-component:
      - if its hole count matches one or more reference components, recolor it
        to the best-matching reference color;
      - otherwise, erase it (set to 0).

    Ties between reference components with the same hole count are resolved by
    a small shape/size similarity score.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    H, W = g.shape

    # ------------------------------------------------------------------
    # Component labeling: 4-connected.
    # ------------------------------------------------------------------
    def label_components(mask: np.ndarray):
        """Return (labels, count) for 4-connected components of mask."""
        labels = np.zeros(mask.shape, dtype=int)
        if not mask.any():
            return labels, 0

        height, width = mask.shape
        current = 0

        for r in range(height):
            for c in range(width):
                if mask[r, c] and labels[r, c] == 0:
                    current += 1
                    stack = [(r, c)]
                    labels[r, c] = current

                    while stack:
                        cr, cc = stack.pop()

                        if cr > 0:
                            nr, nc = cr - 1, cc
                            if mask[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = current
                                stack.append((nr, nc))

                        if cr + 1 < height:
                            nr, nc = cr + 1, cc
                            if mask[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = current
                                stack.append((nr, nc))

                        if cc > 0:
                            nr, nc = cr, cc - 1
                            if mask[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = current
                                stack.append((nr, nc))

                        if cc + 1 < width:
                            nr, nc = cr, cc + 1
                            if mask[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = current
                                stack.append((nr, nc))

        return labels, current

    # ------------------------------------------------------------------
    # Precompute all background holes:
    # zero components not touching the border.
    # ------------------------------------------------------------------
    zero_mask = (g == 0)
    zero_labels, zero_count = label_components(zero_mask)

    zero_touches_border = np.zeros(zero_count + 1, dtype=bool)
    if zero_count:
        zero_touches_border[zero_labels[0, :]] = True
        zero_touches_border[zero_labels[-1, :]] = True
        zero_touches_border[zero_labels[:, 0]] = True
        zero_touches_border[zero_labels[:, -1]] = True

    hole_labels = set(l for l in range(1, zero_count + 1) if not zero_touches_border[l])
    zero_areas = np.bincount(zero_labels.ravel(), minlength=zero_count + 1)

    def holes_for_mask(mask: np.ndarray):
        """
        Count holes adjacent to this mask, and total area of those holes.
        """
        if not hole_labels:
            return 0, 0

        hole_set = set()
        for r, c in np.argwhere(mask):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W:
                    lab = int(zero_labels[nr, nc])
                    if lab in hole_labels:
                        hole_set.add(lab)

        total_area = int(sum(zero_areas[x] for x in hole_set))
        return len(hole_set), total_area

    # ------------------------------------------------------------------
    # Choose data color.  In this task it is always 5, but this makes the
    # function a little more robust.
    # ------------------------------------------------------------------
    colors = np.unique(g)
    if 5 in colors:
        data_color = 5
    else:
        candidates = [int(x) for x in colors if x not in (0, 1)]
        if not candidates:
            return g
        data_color = max(candidates, key=lambda col: int(np.sum(g == col)))

    # ------------------------------------------------------------------
    # Detect the separator line(s) made of 1s.
    # ------------------------------------------------------------------
    def touches_at_least_two_edges(mask: np.ndarray) -> bool:
        edges = 0
        if mask[0, :].any():
            edges += 1
        if mask[-1, :].any():
            edges += 1
        if mask[:, 0].any():
            edges += 1
        if mask[:, -1].any():
            edges += 1
        return edges >= 2

    def is_line_like(mask: np.ndarray) -> bool:
        """
        Heuristic for a 1-pixel-thick line or L-shaped separator.
        """
        if not mask.any() or mask.sum() < 2:
            return False

        # Reject thick blocks.
        if mask.shape[0] > 1 and mask.shape[1] > 1:
            if np.any(
                mask[:-1, :-1] &
                mask[:-1, 1:] &
                mask[1:, :-1] &
                mask[1:, 1:]
            ):
                return False

        # A simple line has no branch point with 3 or 4 orthogonal neighbors.
        for r, c in np.argwhere(mask):
            cnt = 0
            if r > 0 and mask[r - 1, c]:
                cnt += 1
            if r + 1 < H and mask[r + 1, c]:
                cnt += 1
            if c > 0 and mask[r, c - 1]:
                cnt += 1
            if c + 1 < W and mask[r, c + 1]:
                cnt += 1
            if cnt > 2:
                return False

        return True

    min_sep_size = max(10, max(H, W) // 3)
    sep_mask = np.zeros((H, W), dtype=bool)

    one_labels, one_count = label_components(g == 1)
    for lid in range(1, one_count + 1):
        m = (one_labels == lid)
        if m.sum() >= min_sep_size and is_line_like(m) and touches_at_least_two_edges(m):
            sep_mask |= m

    # Fallback: if the line-like test was too strict, still accept a long
    # border-touching 1 component as the separator.
    if not sep_mask.any():
        for lid in range(1, one_count + 1):
            m = (one_labels == lid)
            if m.sum() >= min_sep_size and touches_at_least_two_edges(m):
                sep_mask |= m

    # ------------------------------------------------------------------
    # Find legend region(s): non-separator regions that contain no data color.
    # ------------------------------------------------------------------
    non_sep_mask = ~sep_mask
    region_labels, region_count = label_components(non_sep_mask)

    data_mask_all = (g == data_color)
    legend_regions = []

    if region_count:
        non_zero_nonsep = (g != 0) & ~sep_mask

        for rid in range(1, region_count + 1):
            region_mask = (region_labels == rid)
            if np.any(non_zero_nonsep & region_mask) and not np.any(data_mask_all & region_mask):
                legend_regions.append(rid)

    if legend_regions:
        legend_mask = np.isin(region_labels, legend_regions)
    else:
        # Fallback: use every non-separator, non-background cell as a legend
        # candidate.
        legend_mask = (g != 0) & ~sep_mask

    # ------------------------------------------------------------------
    # Build reference prototype features from legend components.
    # ------------------------------------------------------------------
    def component_features(mask: np.ndarray):
        area = int(mask.sum())
        rs, cs = np.where(mask)
        h = int(rs.max() - rs.min() + 1)
        w = int(cs.max() - cs.min() + 1)
        holes, hole_area = holes_for_mask(mask)
        return area, h, w, holes, hole_area

    prototypes_by_holes = {}

    proto_mask = (g != 0) & ~sep_mask & legend_mask
    proto_colors = np.unique(g[proto_mask])
    proto_colors = [int(c) for c in proto_colors if c != data_color]

    for color in proto_colors:
        color_mask = (g == color) & ~sep_mask & legend_mask
        comp_labels, comp_count = label_components(color_mask)

        for cid in range(1, comp_count + 1):
            m = (comp_labels == cid)
            area, h, w, holes, hole_area = component_features(m)

            # Ignore tiny accidental fragments.
            if area < 3:
                continue

            prototypes_by_holes.setdefault(holes, []).append(
                (color, area, h, w, hole_area)
            )

    # If no prototypes were found, do nothing.
    if not prototypes_by_holes:
        return g

    # ------------------------------------------------------------------
    # Recolor / erase all data components.
    # ------------------------------------------------------------------
    def similarity_score(target_area, target_h, target_w, target_hole_area,
                         color, ref_area, ref_h, ref_w, ref_hole_area):
        # Relative differences, scaled to [0, 1].
        area_diff = abs(target_area - ref_area) / max(1, target_area, ref_area)
        h_diff = abs(target_h - ref_h) / max(1, target_h, ref_h)
        w_diff = abs(target_w - ref_w) / max(1, target_w, ref_w)
        hole_diff = abs(target_hole_area - ref_hole_area) / max(
            1, target_hole_area, ref_hole_area
        )

        score = 1.0 * area_diff + 0.5 * h_diff + 0.5 * w_diff + 0.4 * hole_diff

        # Small penalty for using the separator color 1 when another color is
        # also possible.  This helps when the separator also looks like a
        # valid reference shape.
        if color == 1:
            score += 0.03

        return score

    data_labels, data_count = label_components(data_mask_all)

    out = g.copy()

    for lid in range(1, data_count + 1):
        m = (data_labels == lid)
        if not m.any():
            continue

        area, h, w, holes, hole_area = component_features(m)
        candidates = prototypes_by_holes.get(holes)

        if not candidates:
            out[m] = 0
            continue

        best_color = None
        best_score = float("inf")
        best_tie = None

        for color, ref_area, ref_h, ref_w, ref_hole_area in candidates:
            score = similarity_score(
                area, h, w, hole_area,
                color, ref_area, ref_h, ref_w, ref_hole_area
            )
            tie = (ref_area, color)
            if score < best_score or (score == best_score and tie < best_tie):
                best_score = score
                best_color = color
                best_tie = tie

        out[m] = best_color

    return out