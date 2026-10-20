import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes in the large mask with colors from outside pieces.

    The transformation works as follows:
      1. Identify the background color from the grid border.
      2. Find connected components of non-background colors.
      3. Treat the largest non-background component as the mask/template.
      4. Crop to its bounding box.
      5. Treat all non-mask cells inside that box as holes.
      6. Match each hole component to outside components by shape
         (rotation and reflection invariant).
      7. For each shape, choose the color whose number of matching
         outside components is closest to the number of holes of that shape.
         If there is a tie, choose the color closest to the holes.
      8. Fill the corresponding holes with that color.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # ------------------------------------------------------------------
    # 1. Background color: most common border color.
    # ------------------------------------------------------------------
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    bvals, bcounts = np.unique(border, return_counts=True)
    max_border_count = bcounts.max()
    border_ties = bvals[bcounts == max_border_count]

    if len(border_ties) == 1:
        bg = int(border_ties[0])
    else:
        all_vals, all_counts = np.unique(grid, return_counts=True)
        overall_count = {int(v): int(c) for v, c in zip(all_vals, all_counts)}
        bg = int(max(border_ties,
                     key=lambda v: (overall_count.get(int(v), 0), -int(v))))

    # ------------------------------------------------------------------
    # 2. Connected components of non-background colors.
    # ------------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for i in range(h):
        for j in range(w):
            if not visited[i, j] and int(grid[i, j]) != bg:
                color = int(grid[i, j])
                stack = [(i, j)]
                visited[i, j] = True
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < h and 0 <= nc < w and
                                not visited[nr, nc] and
                                int(grid[nr, nc]) == color):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                components.append((color, cells))

    if not components:
        return grid.copy()

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    def bbox_area(cells):
        r0, r1, c0, c1 = bbox(cells)
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    # ------------------------------------------------------------------
    # 3. Largest non-background component is the mask.
    # ------------------------------------------------------------------
    mask_idx = max(
        range(len(components)),
        key=lambda idx: (
            len(components[idx][1]),
            bbox_area(components[idx][1]),
            components[idx][0]
        )
    )

    mask_color, mask_cells = components[mask_idx]
    r0, r1, c0, c1 = bbox(mask_cells)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    rect = grid[r0:r1 + 1, c0:c1 + 1].copy()

    # Outside pieces are all other non-background components,
    # except components fully inside the mask bounding box.
    outside_pieces = []
    for idx, (color, cells) in enumerate(components):
        if idx == mask_idx:
            continue

        cr0, cr1, cc0, cc1 = bbox(cells)
        if cr0 >= r0 and cr1 <= r1 and cc0 >= c0 and cc1 <= c1:
            continue

        outside_pieces.append((color, cells))

    # ------------------------------------------------------------------
    # Shape canonicalization under rotations and reflections.
    # ------------------------------------------------------------------
    def normalize(pts):
        min_r = min(r for r, _ in pts)
        min_c = min(c for _, c in pts)
        return frozenset((r - min_r, c - min_c) for r, c in pts)

    def shape_key(pts):
        if not pts:
            return ()

        base = list(pts)
        shapes = []

        for mirror in (False, True):
            cur = list(base)
            if mirror:
                cur = [(r, -c) for r, c in cur]

            for _ in range(4):
                shapes.append(tuple(sorted(normalize(cur))))
                cur = [(c, -r) for r, c in cur]

        return min(shapes)

    def centroid(cells):
        avg_r = sum(r for r, _ in cells) / len(cells)
        avg_c = sum(c for _, c in cells) / len(cells)
        return avg_r, avg_c

    def proximity_score(piece_centroids, hole_centroids):
        if not piece_centroids or not hole_centroids:
            return float("inf")

        total = 0.0
        for hr, hc in hole_centroids:
            dmin = min(
                (hr - pr) * (hr - pr) + (hc - pc) * (hc - pc)
                for pr, pc in piece_centroids
            )
            total += dmin

        return total / len(hole_centroids)

    # ------------------------------------------------------------------
    # 4. Index outside pieces by shape.
    # ------------------------------------------------------------------
    candidates_by_shape = {}
    for color, cells in outside_pieces:
        key = shape_key(cells)
        by_color = candidates_by_shape.setdefault(key, {})
        info = by_color.setdefault(int(color), {"count": 0, "centroids": []})
        info["count"] += 1
        info["centroids"].append(centroid(cells))

    # ------------------------------------------------------------------
    # 5. Find hole components inside the mask bounding box.
    # ------------------------------------------------------------------
    holes_mask = rect != mask_color
    hole_groups = {}

    if np.any(holes_mask):
        visited_holes = np.zeros((H, W), dtype=bool)

        for i in range(H):
            for j in range(W):
                if not holes_mask[i, j] or visited_holes[i, j]:
                    continue

                stack = [(i, j)]
                visited_holes[i, j] = True
                cells = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))

                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W and
                                holes_mask[nr, nc] and
                                not visited_holes[nr, nc]):
                            visited_holes[nr, nc] = True
                            stack.append((nr, nc))

                key = shape_key(cells)
                group = hole_groups.setdefault(
                    key,
                    {"count": 0, "area": 0, "centroids": [], "cells_list": []}
                )

                group["count"] += 1
                group["area"] += len(cells)

                ar, ac = centroid(cells)
                group["centroids"].append((r0 + ar, c0 + ac))
                group["cells_list"].append(cells)

    # ------------------------------------------------------------------
    # 6. Choose fill color for each hole shape group.
    # ------------------------------------------------------------------
    out = np.full((H, W), int(mask_color), dtype=grid.dtype)

    if hole_groups:
        def fallback_color(group):
            """Used only if no exact shape match exists."""
            if not outside_pieces:
                return None

            target_area = group["area"] // max(1, group["count"])
            best_color = None
            best_score = None

            for color, cells in outside_pieces:
                area_diff = abs(len(cells) - target_area)
                prox = proximity_score([centroid(cells)], group["centroids"])
                score = (area_diff, prox, int(color))

                if best_score is None or score < best_score:
                    best_score = score
                    best_color = int(color)

            return best_color

        fill_color_by_shape = {}

        for key, group in hole_groups.items():
            candidates = candidates_by_shape.get(key, {})

            if candidates:
                target_count = group["count"]
                best_color = None
                best_score = None

                for color, info in candidates.items():
                    count_diff = abs(info["count"] - target_count)
                    prox = proximity_score(info["centroids"], group["centroids"])

                    # Primary: closest number of matching pieces.
                    # Secondary: closer to the holes.
                    # Tertiary: more matching pieces, then smaller color.
                    score = (count_diff, prox, -info["count"], int(color))

                    if best_score is None or score < best_score:
                        best_score = score
                        best_color = int(color)

            else:
                best_color = fallback_color(group)

            if best_color is None:
                best_color = int(bg)

            fill_color_by_shape[key] = best_color

        # Fill all holes.
        for key, group in hole_groups.items():
            color = fill_color_by_shape[key]
            for cells in group["cells_list"]:
                for r, c in cells:
                    out[r, c] = color

    return out