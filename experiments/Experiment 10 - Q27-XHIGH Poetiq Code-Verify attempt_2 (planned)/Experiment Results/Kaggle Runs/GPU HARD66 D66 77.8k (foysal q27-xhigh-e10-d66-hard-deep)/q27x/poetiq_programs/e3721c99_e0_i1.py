import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every 4-connected component of color 5 according to the number
    of holes it contains.

    The legend is identified as the side of the separating line of 1s that
    contains no 5s. Non-5 components in that legend provide the color for
    each hole count. A hole is a 4-connected component of the complement
    of an object that does not touch the grid border.
    """
    g = np.array(grid, dtype=int, copy=False)
    out = g.copy()

    if g.size == 0:
        return out

    H, W = g.shape

    # The data objects are always color 5 in this task.
    if not np.any(g == 5):
        return out

    def label_components(mask):
        """4-connected component labeling. Returns labels array and count."""
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

    def touches_at_least_two_edges(mask):
        """Return True if the component touches at least two grid borders."""
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

    def is_line_like(mask):
        """
        Heuristic for a 1-pixel-thick line or L-shaped separator.
        Such a component has no 2x2 block and every pixel has <= 2
        orthogonal neighbors inside the component.
        """
        if not mask.any():
            return False
        if mask.sum() < 2:
            return False

        # Exclude small solid blocks.
        if mask.shape[0] > 1 and mask.shape[1] > 1:
            if np.any(mask[:-1, :-1] &
                      mask[:-1, 1:] &
                      mask[1:, :-1] &
                      mask[1:, 1:]):
                return False

        coords = np.argwhere(mask)
        for r, c in coords:
            cnt = 0
            if r > 0 and mask[r - 1, c]:
                cnt += 1
            if r + 1 < mask.shape[0] and mask[r + 1, c]:
                cnt += 1
            if c > 0 and mask[r, c - 1]:
                cnt += 1
            if c + 1 < mask.shape[1] and mask[r, c + 1]:
                cnt += 1
            if cnt > 2:
                return False

        return True

    def count_holes(obj_mask):
        """
        Count holes in one object.
        A hole is a connected component of the complement that does not
        touch the outer border.
        """
        complement = ~obj_mask
        labels, max_label = label_components(complement)

        if max_label == 0:
            return 0

        boundary_labels = set(labels[0, :].tolist())
        boundary_labels.update(labels[-1, :].tolist())
        boundary_labels.update(labels[:, 0].tolist())
        boundary_labels.update(labels[:, -1].tolist())
        boundary_labels.discard(0)

        holes = 0
        for lab in range(1, max_label + 1):
            if lab not in boundary_labels:
                holes += 1
        return holes

    # ------------------------------------------------------------------
    # 1. Find the separating line of 1s.
    # ------------------------------------------------------------------
    sep_mask = np.zeros((H, W), dtype=bool)

    one_labels, one_count = label_components(g == 1)
    min_sep_size = max(5, max(H, W) // 10)

    for lid in range(1, one_count + 1):
        m = (one_labels == lid)
        if (m.sum() >= min_sep_size and
            is_line_like(m) and
            touches_at_least_two_edges(m)):
            sep_mask |= m

    # Fallback: if line-like test was too strict, still accept a large
    # border-touching 1 component as the separator.
    if not sep_mask.any():
        for lid in range(1, one_count + 1):
            m = (one_labels == lid)
            if m.sum() >= min_sep_size and touches_at_least_two_edges(m):
                sep_mask |= m

    # ------------------------------------------------------------------
    # 2. Determine the legend region.
    #    It is the non-separator region containing no 5s and at least one
    #    non-5 prototype.
    # ------------------------------------------------------------------
    non_sep_mask = ~sep_mask
    region_labels, _ = label_components(non_sep_mask)
    max_region = int(region_labels.max())

    if max_region == 0:
        return out

    target_mask = (g == 5)
    legend_regions = list(range(1, max_region + 1))

    if sep_mask.any():
        if target_mask.any():
            target_region_counts = np.bincount(
                region_labels[target_mask],
                minlength=max_region + 1
            )
        else:
            target_region_counts = np.zeros(max_region + 1, dtype=int)

        zero_target_regions = [
            r for r in range(1, max_region + 1)
            if target_region_counts[r] == 0
        ]

        non_target_nonzero = (g != 0) & (g != 5) & ~sep_mask
        candidates = []

        for r in zero_target_regions:
            if np.any(non_target_nonzero & (region_labels == r)):
                candidates.append(r)

        if candidates:
            legend_regions = candidates

    legend_region_mask = np.isin(region_labels, legend_regions)

    # ------------------------------------------------------------------
    # 3. Build mapping: number_of_holes -> legend_color.
    # ------------------------------------------------------------------
    proto_mask = (g != 0) & (g != 5) & ~sep_mask
    mapping = {}

    if proto_mask.any():
        for color in np.unique(g[proto_mask]):
            color = int(color)
            color_mask = (g == color) & ~sep_mask
            comp_labels, comp_count = label_components(color_mask)

            for lid in range(1, comp_count + 1):
                m = (comp_labels == lid)

                # Keep only prototype components in the legend region.
                if not np.any(legend_region_mask & m):
                    continue

                holes = count_holes(m)
                area = int(m.sum())

                # If multiple prototypes have the same hole count,
                # prefer the larger one.
                if holes not in mapping or area > mapping[holes][0]:
                    mapping[holes] = (area, color)

    # ------------------------------------------------------------------
    # 4. Recolor all 5 components using the hole-count mapping.
    # ------------------------------------------------------------------
    data_mask = target_mask & ~sep_mask
    data_labels, data_count = label_components(data_mask)

    for lid in range(1, data_count + 1):
        m = (data_labels == lid)
        holes = count_holes(m)

        if holes in mapping:
            out[m] = mapping[holes][1]

    return out