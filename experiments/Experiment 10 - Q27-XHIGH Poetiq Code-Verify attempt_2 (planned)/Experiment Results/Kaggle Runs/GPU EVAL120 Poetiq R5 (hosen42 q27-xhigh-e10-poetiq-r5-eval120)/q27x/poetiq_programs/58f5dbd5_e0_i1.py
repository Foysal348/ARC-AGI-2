import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by finding solid 5x5 colored squares that have a matching
    3x3 stencil of the same color.

    For each matching color:
      - keep the 5x5 square
      - use the 3x3 stencil as a hole mask in the center of the square
        (cells that are colored in the stencil become background in the output)

    The output is the one-cell-padded bounding box of all kept 5x5 squares.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()

    # ------------------------------------------------------------------
    # Background color
    # ------------------------------------------------------------------
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    border_vals, border_counts = np.unique(border, return_counts=True)
    bg = int(border_vals[np.argmax(border_counts)])

    # If the border is not clearly one color, fall back to global mode.
    if border_counts.max() * 2 < len(border):
        all_vals, all_counts = np.unique(grid, return_counts=True)
        bg = int(all_vals[np.argmax(all_counts)])

    # ------------------------------------------------------------------
    # Find solid 5x5 non-background components.
    # A valid component has:
    #   - bounding box exactly 5x5
    #   - exactly 25 cells
    #   - all cells the same color
    # ------------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    in_large = np.zeros((h, w), dtype=bool)
    large_rects = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True

            cells = []
            min_r = max_r = r
            min_c = max_c = c

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                min_r = min(min_r, rr)
                max_r = max(max_r, rr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if (
                max_r - min_r + 1 == 5
                and max_c - min_c + 1 == 5
                and len(cells) == 25
            ):
                large_rects.append((min_r, min_c, color))
                for rr, cc in cells:
                    in_large[rr, cc] = True

    if not large_rects:
        return np.full((h, w), bg, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # For each large-square color, extract its 3x3 stencil.
    # The stencil is the bounding box of all remaining cells of that color.
    # ------------------------------------------------------------------
    def get_pattern(color: int):
        mask = (grid == color) & (~in_large)
        if not np.any(mask):
            return None

        rs, cs = np.nonzero(mask)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())

        # The stencil must occupy exactly a 3x3 box.
        if r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3:
            pat = np.zeros((3, 3), dtype=bool)
            pat[rs - r0, cs - c0] = True
            return pat

        return None

    patterns = {}
    for color in {col for _, _, col in large_rects}:
        pat = get_pattern(color)
        if pat is not None:
            patterns[color] = pat

    # Keep only large squares whose color has a valid stencil.
    selected = [(r, c, col) for r, c, col in large_rects if col in patterns]

    if not selected:
        return np.full((h, w), bg, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # Build output as the one-cell-padded bounding box of selected squares.
    # ------------------------------------------------------------------
    r_min = min(r for r, c, col in selected)
    r_max = max(r + 4 for r, c, col in selected)
    c_min = min(c for r, c, col in selected)
    c_max = max(c + 4 for r, c, col in selected)

    out = np.full((r_max - r_min + 3, c_max - c_min + 3), bg, dtype=grid.dtype)

    base_r = r_min - 1
    base_c = c_min - 1

    for r, c, col in selected:
        orow = r - base_r
        ocol = c - base_c

        # Fill the whole 5x5 square with its color first.
        out[orow:orow + 5, ocol:ocol + 5] = col

        # Apply the 3x3 stencil as a hole mask in the center.
        pat = patterns[col]
        for i in range(3):
            for j in range(3):
                if pat[i, j]:
                    out[orow + 1 + i, ocol + 1 + j] = bg

    return out