import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Make each non-background object left-right symmetric by deleting cells
    that do not have a mirror counterpart across the object's vertical axis.

    For each connected non-background region:
      - consider every possible vertical axis (integer or half-integer column)
      - count how many cells would remain if we kept only cells whose mirror
        cell in the same row is also present
      - choose the axis that retains the most cells
      - if tied, prefer a more compact result and use deterministic fallbacks

    The background is detected from the grid border / most frequent color and
    is left unchanged.
    """
    grid = np.array(grid, dtype=int)

    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Detect background color.
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    border_counts = dict(zip(*np.unique(border, return_counts=True)))
    bg = int(max(border_counts, key=border_counts.get))

    all_counts = dict(zip(*np.unique(grid, return_counts=True)))
    if max(all_counts.values()) > all_counts.get(bg, 0):
        bg = int(max(all_counts, key=all_counts.get))

    out = np.full((H, W), bg, dtype=grid.dtype)
    visited = np.zeros((H, W), dtype=bool)

    # 8-neighborhood connectivity is used so that line-like shapes connected
    # diagonally are treated as one object.
    directions = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ]

    def symmetric_core(coords):
        """
        Given coordinates of one object, return the coordinates that remain
        after removing non-mirrored cells.
        """
        if not coords:
            return []

        rows = {}
        for r, c in coords:
            rows.setdefault(r, set()).add(c)

        min_c = min(min(cols) for cols in rows.values())
        max_c = max(max(cols) for cols in rows.values())

        # centroid_S == 2 * centroid_col
        sum_c = sum(c for cols in rows.values() for c in cols)
        centroid_S = 2.0 * sum_c / len(coords)

        best_kept = []
        best_score = None

        # S = 2 * axis_col.
        # Even S means an integer column axis, odd S means a half-integer axis.
        for S in range(2 * min_c, 2 * max_c + 1):
            kept = []
            total = 0
            paired = 0

            for r, cols in rows.items():
                for c in cols:
                    mirror_c = S - c
                    if mirror_c in cols:
                        kept.append((r, c))
                        total += 1
                        if mirror_c != c:
                            paired += 1

            if kept:
                min_r = min(r for r, _ in kept)
                max_r = max(r for r, _ in kept)
                min_ckept = min(c for _, c in kept)
                max_ckept = max(c for _, c in kept)
                width_k = max_ckept - min_ckept + 1
                height_k = max_r - min_r + 1
                rows_k = len(set(r for r, _ in kept))
            else:
                width_k = 0
                height_k = 0
                rows_k = 0

            dist = abs(S - centroid_S)

            # Primary: keep as many cells as possible.
            # Secondary: prefer a compact, stable symmetric core.
            score = (
                total,
                -width_k,
                rows_k,
                -height_k,
                paired,
                -dist,
                -S,
            )

            if best_score is None or score > best_score:
                best_score = score
                best_kept = kept

        return best_kept

    # Process each non-background connected component.
    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg or visited[r, c]:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))

                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            kept = symmetric_core(coords)
            for rr, cc in kept:
                out[rr, cc] = color

    return out