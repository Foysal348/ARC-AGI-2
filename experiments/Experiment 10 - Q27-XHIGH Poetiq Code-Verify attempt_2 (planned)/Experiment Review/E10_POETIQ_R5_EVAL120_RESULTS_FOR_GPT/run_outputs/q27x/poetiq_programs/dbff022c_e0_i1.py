import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed zero regions whose surrounding color is a key in the
    solid rectangular color table.

    The color table is the connected solid rectangle of nonzero cells.
    If it is wider than/tall as tall as it is tall, the top row maps to
    the bottom row. If it is taller than wide, the left column maps to
    the right column. Enclosed holes surrounded only by a mapped key color
    are painted with the mapped value.
    """
    if grid.ndim != 2:
        return grid.copy()

    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return grid.copy()

    out = grid.copy()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def find_components(mask: np.ndarray) -> np.ndarray:
        """Label 4-connected components of mask."""
        labels = np.zeros((rows, cols), dtype=int)
        comp_id = 0

        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and labels[r, c] == 0:
                    comp_id += 1
                    stack = [(r, c)]
                    labels[r, c] = comp_id

                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in dirs:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if mask[nr, nc] and labels[nr, nc] == 0:
                                    labels[nr, nc] = comp_id
                                    stack.append((nr, nc))

        return labels

    # ------------------------------------------------------------------
    # 1. Find the solid multicolor rectangle used as the color table.
    # ------------------------------------------------------------------
    nonzero = grid != 0
    nz_labels = find_components(nonzero)
    max_label = int(nz_labels.max()) if nz_labels.size else 0

    candidates = []
    for comp in range(1, max_label + 1):
        ys, xs = np.where(nz_labels == comp)
        if ys.size == 0:
            continue

        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        ch = bottom - top + 1
        cw = right - left + 1

        if ch >= 2 and cw >= 2:
            sub = grid[top:bottom + 1, left:right + 1]
            # A valid table is a completely filled rectangle with multiple colors.
            if np.all(sub != 0) and len(np.unique(sub)) >= 2:
                candidates.append((ch * cw, top, left, ch, cw))

    if not candidates:
        return out

    # Prefer a table with one side equal to 2, then the largest one.
    candidates.sort(key=lambda x: (
        0 if (x[3] == 2 or x[4] == 2) else 1,
        -x[0],
        x[1],
        x[2]
    ))

    _, top, left, ch, cw = candidates[0]
    bottom = top + ch - 1
    right = left + cw - 1
    legend = grid[top:bottom + 1, left:right + 1]

    # ------------------------------------------------------------------
    # 2. Build the color mapping from the table.
    # ------------------------------------------------------------------
    mapping = {}

    if ch <= cw:
        # Wide/square: top row keys -> bottom row values.
        for c in range(cw):
            key = int(legend[0, c])
            val = int(legend[-1, c])
            if key != 0 and key not in mapping:
                mapping[key] = val
    else:
        # Tall: left column keys -> right column values.
        for r in range(ch):
            key = int(legend[r, 0])
            val = int(legend[r, -1])
            if key != 0 and key not in mapping:
                mapping[key] = val

    if not mapping:
        return out

    # ------------------------------------------------------------------
    # 3. Find zeros that are not connected to the outside border.
    # ------------------------------------------------------------------
    visited = np.zeros((rows, cols), dtype=bool)
    stack = []

    # Seed flood fill with border zeros.
    for r in range(rows):
        if grid[r, 0] == 0 and not visited[r, 0]:
            visited[r, 0] = True
            stack.append((r, 0))
        if grid[r, cols - 1] == 0 and not visited[r, cols - 1]:
            visited[r, cols - 1] = True
            stack.append((r, cols - 1))

    for c in range(cols):
        if grid[0, c] == 0 and not visited[0, c]:
            visited[0, c] = True
            stack.append((0, c))
        if grid[rows - 1, c] == 0 and not visited[rows - 1, c]:
            visited[rows - 1, c] = True
            stack.append((rows - 1, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr, nc] and grid[nr, nc] == 0:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    holes = (grid == 0) & (~visited)

    # ------------------------------------------------------------------
    # 4. For each enclosed hole, if it is surrounded by a single mapped
    #    color, paint the whole hole with the mapped value.
    # ------------------------------------------------------------------
    for r in range(rows):
        for c in range(cols):
            if not holes[r, c]:
                continue

            cells = []
            neighbor_colors = set()
            stack = [(r, c)]
            holes[r, c] = False

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        val = int(grid[nr, nc])
                        if val == 0:
                            if holes[nr, nc]:
                                holes[nr, nc] = False
                                stack.append((nr, nc))
                        else:
                            neighbor_colors.add(val)

            if len(neighbor_colors) == 1:
                color = next(iter(neighbor_colors))
                if color in mapping:
                    fill = mapping[color]
                    for rr, cc in cells:
                        out[rr, cc] = fill

    return out