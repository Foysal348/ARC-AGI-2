import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid as follows:

    1. Find the full vertical column of 5s separating the left and right panels.
    2. In the left panel, shift every 4-connected component of 0s to the right
       until the right edge of its bounding box touches the right border of
       the left panel.
    3. In the right panel, fill 2s in every row where the shifted left panel
       has an enclosed 6 cell immediately to the left of its right border.
       Enclosed means not connected to any border of the left panel through 6s.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Find the vertical separator line of 5s.
    sep = None
    for c in range(W):
        if np.all(grid[:, c] == 5):
            sep = c
            break

    # Fallback for robustness, though the examples always have a full line.
    if sep is None:
        counts = (grid == 5).sum(axis=0)
        if counts.max() == 0:
            return grid.copy()
        sep = int(np.argmax(counts))

    left_w = sep
    right_w = W - sep - 1

    if left_w <= 0:
        return grid.copy()

    bg = 6
    obj = 0

    left_in = grid[:, :left_w]
    left_out = np.full((H, left_w), bg, dtype=int)

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Shift each 0-component right by its bounding-box distance to the right edge.
    visited = np.zeros((H, left_w), dtype=bool)

    for i in range(H):
        for j in range(left_w):
            if left_in[i, j] != obj or visited[i, j]:
                continue

            queue = [(i, j)]
            visited[i, j] = True
            head = 0
            max_col = j

            while head < len(queue):
                r, c = queue[head]
                head += 1

                if c > max_col:
                    max_col = c

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < left_w:
                        if not visited[nr, nc] and left_in[nr, nc] == obj:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            shift = (left_w - 1) - max_col

            if shift > 0:
                for r, c in queue:
                    left_out[r, c + shift] = obj
            elif shift == 0:
                for r, c in queue:
                    left_out[r, c] = obj
            else:
                # Defensive fallback; should not happen for valid inputs.
                for r, c in queue:
                    nc = c + shift
                    if 0 <= nc < left_w:
                        left_out[r, nc] = obj

    # Flood-fill all outside 6-cells in the shifted left panel.
    outside = np.zeros((H, left_w), dtype=bool)
    queue = []

    def try_seed(r, c):
        if 0 <= r < H and 0 <= c < left_w:
            if left_out[r, c] == bg and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    for r in range(H):
        try_seed(r, 0)
        try_seed(r, left_w - 1)

    for c in range(left_w):
        try_seed(0, c)
        try_seed(H - 1, c)

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in dirs:
            try_seed(r + dr, c + dc)

    # Build final output.
    out = grid.copy()
    out[:, :left_w] = left_out
    out[:, sep] = 5

    if right_w > 0:
        out[:, sep + 1:] = bg

        if left_w >= 2:
            edge = left_w - 1
            inner = left_w - 2

            for r in range(H):
                if (
                    left_out[r, edge] == obj
                    and left_out[r, inner] == bg
                    and not outside[r, inner]
                ):
                    out[r, sep + 1:] = 2

    return out