import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete objects by mirroring their non-2 cells across the 2 cells.

    The background color is taken to be the most frequent color.
    Non-background cells are grouped into 8-connected components.

    For each component that contains color 2:
      - a single 2 is a center of 180-degree rotation;
      - several 2s forming a straight horizontal, vertical, or diagonal line
        form a mirror line;
      - otherwise each non-2 cell is rotated 180 degrees about the nearest 2.

    Mirrored cells are written only over background cells (or over the same
    color), so original objects are preserved.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    # Identify background color: most common color, preferring top-left if tied.
    colors, counts = np.unique(grid, return_counts=True)
    max_count = counts.max()
    possible_backgrounds = colors[counts == max_count]
    if grid[0, 0] in possible_backgrounds:
        bg = int(grid[0, 0])
    else:
        bg = int(possible_backgrounds[0])

    out = grid.copy()
    H, W = grid.shape
    non_background = grid != bg

    def place(r: int, c: int, color: int) -> None:
        """Place a mirrored cell if it is inside and not over different object."""
        if 0 <= r < H and 0 <= c < W:
            if grid[r, c] == bg or grid[r, c] == color:
                out[r, c] = color

    if not non_background.any():
        return out

    visited = np.zeros((H, W), dtype=bool)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for start_r in range(H):
        for start_c in range(W):
            if not non_background[start_r, start_c] or visited[start_r, start_c]:
                continue

            # Collect one 8-connected component of non-background cells.
            stack = [(start_r, start_c)]
            visited[start_r, start_c] = True
            component = []

            while stack:
                r, c = stack.pop()
                component.append((r, c))

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and non_background[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            twos = [(r, c) for r, c in component if grid[r, c] == 2]
            if not twos:
                continue

            payload = [(r, c) for r, c in component if grid[r, c] != 2]
            if not payload:
                continue

            # Single 2: 180-degree rotation around it.
            if len(twos) == 1:
                r0, c0 = twos[0]
                for r, c in payload:
                    place(2 * r0 - r, 2 * c0 - c, grid[r, c])
                continue

            # Multiple 2s forming a straight line: reflection across that line.
            rows = {r for r, _ in twos}
            cols = {c for _, c in twos}

            # Horizontal line.
            if len(rows) == 1:
                r0 = next(iter(rows))
                for r, c in payload:
                    place(2 * r0 - r, c, grid[r, c])
                continue

            # Vertical line.
            if len(cols) == 1:
                c0 = next(iter(cols))
                for r, c in payload:
                    place(r, 2 * c0 - c, grid[r, c])
                continue

            # Main diagonal line: r - c = constant.
            main_diags = {r - c for r, c in twos}
            if len(main_diags) == 1:
                k = next(iter(main_diags))
                for r, c in payload:
                    place(c + k, r - k, grid[r, c])
                continue

            # Anti-diagonal line: r + c = constant.
            anti_diags = {r + c for r, c in twos}
            if len(anti_diags) == 1:
                k = next(iter(anti_diags))
                for r, c in payload:
                    place(k - c, k - r, grid[r, c])
                continue

            # Fallback: rotate each payload cell 180 degrees about nearest 2.
            for r, c in payload:
                r2, c2 = min(
                    twos,
                    key=lambda rc: (
                        abs(rc[0] - r) + abs(rc[1] - c),
                        (rc[0] - r) ** 2 + (rc[1] - c) ** 2,
                    ),
                )
                place(2 * r2 - r, 2 * c2 - c, grid[r, c])

    return out