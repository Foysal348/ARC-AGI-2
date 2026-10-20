import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each connected component of the puzzle color according to the
    number of holes in that component.

    The examples show that small reference shapes provide a mapping from
    "number of enclosed holes" to color.  The large pattern is drawn in one
    color (usually 5).  Every connected component of that color is replaced
    by the color whose reference shape has the same number of holes.

    Holes are counted as 4-connected regions not connected to the grid border
    when the component is treated as a barrier.
    """
    g = np.array(grid, dtype=int)
    if g.size == 0:
        return g.copy()

    H, W = g.shape
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def same_color_component_mask(arr, sr, sc, color):
        """Return a boolean mask for the 4-connected component of a color."""
        if not (0 <= sr < H and 0 <= sc < W) or arr[sr, sc] != color:
            return np.zeros((H, W), dtype=bool)

        mask = np.zeros((H, W), dtype=bool)
        mask[sr, sc] = True
        stack = [(sr, sc)]

        while stack:
            y, x = stack.pop()
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < H
                    and 0 <= nx < W
                    and not mask[ny, nx]
                    and arr[ny, nx] == color
                ):
                    mask[ny, nx] = True
                    stack.append((ny, nx))

        return mask

    # Remove the separator line if present.
    # A separator is a full row or full column of one nonzero color.
    # We remove the entire connected component containing that line, which
    # handles cases where the separator has an attached arm.
    sep_mask = np.zeros((H, W), dtype=bool)

    for r in range(H):
        row = g[r]
        if row.size and row[0] != 0 and np.all(row == row[0]):
            color = int(row[0])
            sep_mask |= same_color_component_mask(g, r, 0, color)

    for c in range(W):
        col = g[:, c]
        if col.size and col[0] != 0 and np.all(col == col[0]):
            color = int(col[0])
            sep_mask |= same_color_component_mask(g, 0, c, color)

    work = g.copy()
    if sep_mask.any():
        work[sep_mask] = 0

    if not work.any():
        return g.copy()

    # The puzzle color is the most frequent nonzero color after separator removal.
    flat = work.ravel()
    flat = flat[flat != 0]
    if flat.size == 0:
        return g.copy()

    values, counts = np.unique(flat, return_counts=True)
    puzzle_color = int(values[np.argmax(counts)])

    # Label all same-color connected components in the working grid.
    visited = np.zeros((H, W), dtype=bool)
    components = []

    for y in range(H):
        for x in range(W):
            color = work[y, x]
            if color != 0 and not visited[y, x]:
                mask = np.zeros((H, W), dtype=bool)
                stack = [(y, x)]
                visited[y, x] = True

                while stack:
                    cy, cx = stack.pop()
                    mask[cy, cx] = True

                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and not visited[ny, nx]
                            and work[ny, nx] == color
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                components.append((int(color), mask))

    def count_holes(mask):
        """
        Count enclosed holes in a boolean mask.

        All cells not in the mask are treated as background.  First flood-fill
        background connected to the grid border.  Every remaining background
        connected component is one hole.
        """
        if not mask.any():
            return 0

        bg = ~mask
        seen = np.zeros((H, W), dtype=bool)
        stack = []

        # Seed flood fill from all border background cells.
        for x in range(W):
            if bg[0, x] and not seen[0, x]:
                seen[0, x] = True
                stack.append((0, x))
            if bg[H - 1, x] and not seen[H - 1, x]:
                seen[H - 1, x] = True
                stack.append((H - 1, x))

        for y in range(H):
            if bg[y, 0] and not seen[y, 0]:
                seen[y, 0] = True
                stack.append((y, 0))
            if bg[y, W - 1] and not seen[y, W - 1]:
                seen[y, W - 1] = True
                stack.append((y, W - 1))

        # Flood fill outside background.
        while stack:
            y, x = stack.pop()

            ny = y - 1
            if ny >= 0 and bg[ny, x] and not seen[ny, x]:
                seen[ny, x] = True
                stack.append((ny, x))

            ny = y + 1
            if ny < H and bg[ny, x] and not seen[ny, x]:
                seen[ny, x] = True
                stack.append((ny, x))

            nx = x - 1
            if nx >= 0 and bg[y, nx] and not seen[y, nx]:
                seen[y, nx] = True
                stack.append((y, nx))

            nx = x + 1
            if nx < W and bg[y, nx] and not seen[y, nx]:
                seen[y, nx] = True
                stack.append((y, nx))

        # Count remaining background components.
        holes = 0
        for y in range(H):
            for x in range(W):
                if bg[y, x] and not seen[y, x]:
                    holes += 1
                    stack = [(y, x)]
                    seen[y, x] = True

                    while stack:
                        cy, cx = stack.pop()

                        ny = cy - 1
                        if ny >= 0 and bg[ny, cx] and not seen[ny, cx]:
                            seen[ny, cx] = True
                            stack.append((ny, cx))

                        ny = cy + 1
                        if ny < H and bg[ny, cx] and not seen[ny, cx]:
                            seen[ny, cx] = True
                            stack.append((ny, cx))

                        nx = cx - 1
                        if nx >= 0 and bg[cy, nx] and not seen[cy, nx]:
                            seen[cy, nx] = True
                            stack.append((cy, nx))

                        nx = cx + 1
                        if nx < W and bg[cy, nx] and not seen[cy, nx]:
                            seen[cy, nx] = True
                            stack.append((cy, nx))

        return holes

    # Build mapping from hole count to reference color.
    hole_to_color = {}
    hole_to_area = {}
    puzzle_components = []

    for color, mask in components:
        if color == puzzle_color:
            puzzle_components.append(mask)
        else:
            holes = count_holes(mask)
            area = int(mask.sum())

            # If several references have the same hole count, keep the simplest
            # one deterministically.
            if holes not in hole_to_color or area < hole_to_area.get(holes, 10**18):
                hole_to_color[holes] = color
                hole_to_area[holes] = area

    if not hole_to_color:
        return g.copy()

    # Recolor puzzle components.
    out = g.copy()
    for mask in puzzle_components:
        holes = count_holes(mask)
        if holes in hole_to_color:
            out[mask] = hole_to_color[holes]

    return out