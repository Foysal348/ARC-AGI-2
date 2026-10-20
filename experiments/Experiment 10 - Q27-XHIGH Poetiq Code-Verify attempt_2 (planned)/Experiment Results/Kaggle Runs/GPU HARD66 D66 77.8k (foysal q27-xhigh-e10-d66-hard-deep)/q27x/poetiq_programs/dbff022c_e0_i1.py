import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed zero regions using a color mapping inferred from a solid
    multi-color rectangular key.

    Key orientation:
      - If the key is at least as wide as it is tall, map top -> bottom.
      - If the key is taller than wide, map left -> right.

    A hole is a 4-connected zero component not connected to the grid border.
    If the hole touches only one border color, and that color has a mapping,
    the hole is filled with the mapped color.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def label_components(mask):
        """Label 4-connected nonzero components."""
        labels = np.zeros((H, W), dtype=int)
        next_label = 0

        for r in range(H):
            for c in range(W):
                if mask[r, c] and labels[r, c] == 0:
                    next_label += 1
                    labels[r, c] = next_label
                    stack = [(r, c)]

                    while stack:
                        rr, cc = stack.pop()
                        for dr, dc in dirs:
                            nr, nc = rr + dr, cc + dc
                            if (
                                0 <= nr < H
                                and 0 <= nc < W
                                and mask[nr, nc]
                                and labels[nr, nc] == 0
                            ):
                                labels[nr, nc] = next_label
                                stack.append((nr, nc))

        return labels, next_label

    # ---------------------------------------------------------------
    # 1. Find the key: a solid rectangular component with multiple colors.
    # ---------------------------------------------------------------
    labels, n_labels = label_components(grid != 0)

    best_score = None
    best_rect = None

    for lab in range(1, n_labels + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue

        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1

        # Must be a rectangle at least 2x2, completely filled.
        if bh < 2 or bw < 2:
            continue
        if bh * bw != ys.size:
            continue

        colors = set(grid[ys, xs].tolist())
        if len(colors) < 2:
            continue

        # Prefer the usual key shape: one side has length 2.
        # Then prefer more colors and larger area.
        score = (min(bh, bw) == 2, len(colors), bh * bw)
        if best_score is None or score > best_score:
            best_score = score
            best_rect = (r0, c0, bh, bw)

    # ---------------------------------------------------------------
    # 2. Build color mapping from the key.
    # ---------------------------------------------------------------
    mapping = {}

    if best_rect is not None:
        r0, c0, bh, bw = best_rect
        key = grid[r0:r0 + bh, c0:c0 + bw]

        if bh <= bw:
            # Horizontal or square key: top -> bottom.
            for r in range(bh - 1):
                for c in range(bw):
                    src = int(key[r, c])
                    dst = int(key[r + 1, c])
                    if src != 0 and dst != 0 and src not in mapping:
                        mapping[src] = dst
        else:
            # Vertical key: left -> right.
            for r in range(bh):
                for c in range(bw - 1):
                    src = int(key[r, c])
                    dst = int(key[r, c + 1])
                    if src != 0 and dst != 0 and src not in mapping:
                        mapping[src] = dst

    if not mapping:
        return out

    # ---------------------------------------------------------------
    # 3. Find zero regions connected to the border (background zeros).
    # ---------------------------------------------------------------
    background = np.zeros((H, W), dtype=bool)
    stack = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 and (
                r == 0 or r == H - 1 or c == 0 or c == W - 1
            ):
                background[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < H
                and 0 <= nc < W
                and not background[nr, nc]
                and grid[nr, nc] == 0
            ):
                background[nr, nc] = True
                stack.append((nr, nc))

    # ---------------------------------------------------------------
    # 4. Fill enclosed zero components when bordered by a mapped color.
    # ---------------------------------------------------------------
    visited = background.copy()

    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 and not visited[r, c]:
                cells = []
                stack = [(r, c)]
                visited[r, c] = True

                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))

                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and not visited[nr, nc]
                            and grid[nr, nc] == 0
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                border_colors = set()
                for rr, cc in cells:
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != 0:
                            border_colors.add(int(grid[nr, nc]))

                if len(border_colors) == 1:
                    border_color = border_colors.pop()
                    if border_color in mapping:
                        fill_color = mapping[border_color]
                        for rr, cc in cells:
                            out[rr, cc] = fill_color

    return out