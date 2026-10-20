import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Remove (replace with 7) every non-7 connected component that contains
    all of the key colors found in the top-left corner inside the 3-marker.
    """
    g = np.array(grid, dtype=int, copy=True)
    if g.ndim != 2:
        return g

    h, w = g.shape
    bg = 7
    marker = 3

    marker_mask = (g == marker)
    if not marker_mask.any():
        return g

    # Find the corner of the top-left 3-marker.
    row_counts = marker_mask.sum(axis=1)
    col_counts = marker_mask.sum(axis=0)

    if row_counts.max() == 0 or col_counts.max() == 0:
        return g

    # Use the bottom-most / right-most maximal marker lines as the corner.
    h_row = int(np.where(row_counts == row_counts.max())[0][-1])
    v_col = int(np.where(col_counts == col_counts.max())[0][-1])

    # Key colors are the non-zero, non-background, non-marker colors
    # in the open area above-left of the marker corner.
    key_colors = set()
    for r in range(h_row):
        for c in range(v_col):
            val = int(g[r, c])
            if val not in (0, bg, marker):
                key_colors.add(val)

    if not key_colors:
        return g

    # Protect the marker frame itself from being erased.
    protected = np.zeros((h, w), dtype=bool)
    protected[:h_row + 1, :v_col + 1] = marker_mask[:h_row + 1, :v_col + 1]

    visited = np.zeros((h, w), dtype=bool)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True

            cells = []
            colors = set()
            has_protected = False

            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                colors.add(int(g[x, y]))
                if protected[x, y]:
                    has_protected = True

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if not visited[nx, ny] and g[nx, ny] != bg:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

            # Preserve the marker frame, otherwise erase target components.
            if has_protected:
                continue

            if key_colors.issubset(colors):
                for x, y in cells:
                    g[x, y] = bg

    return g