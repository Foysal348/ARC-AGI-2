import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Replace each same-color line fragment by a straight segment.

    - Background is the most frequent color.
    - Same-color 8-connected components are treated as objects.
    - Infer the common direction among: right, down, up-right, down-right.
    - Order objects by their perpendicular coordinate.
    - The middle object's pixel count is the target segment length.
    - Redraw each object as a segment of that length from its anchor
      (minimum projection along the chosen direction).
    """
    arr = np.array(grid, dtype=int)
    if arr.ndim != 2:
        return arr

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr.copy()

    # Background color
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    if not np.any(arr != bg):
        return arr.copy()

    # Find 8-connected components for each non-background color
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    components = []
    for c in vals:
        c = int(c)
        if c == bg:
            continue

        mask = (arr == c)
        visited = np.zeros((H, W), dtype=bool)

        for i in range(H):
            for j in range(W):
                if mask[i, j] and not visited[i, j]:
                    stack = [(i, j)]
                    visited[i, j] = True
                    pixels = []

                    while stack:
                        r, col = stack.pop()
                        pixels.append((r, col))

                        for dr, dc in neighbors:
                            nr, nc = r + dr, col + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))

                    components.append((c, pixels))

    if not components:
        return np.full((H, W), bg, dtype=arr.dtype)

    # Candidate canonical directions:
    # right, down, up-right, down-right
    candidates = [(0, 1), (1, 0), (-1, 1), (1, 1)]

    def perpendicular(d):
        if d == (0, 1):    # right
            return (1, 0)  # row
        if d == (1, 0):    # down
            return (0, 1)  # column
        if d == (-1, 1):   # up-right
            return (1, 1)  # row + col
        if d == (1, 1):    # down-right
            return (-1, 1) # col - row

    # Choose direction that makes components lie closest to straight lines
    best_key = None
    best_d = candidates[0]

    for idx, d in enumerate(candidates):
        n = perpendicular(d)
        n0, n1 = n

        total_span = 0
        total_sq_span = 0
        zero_span_components = 0

        for _, pixels in components:
            values = [r * n0 + c * n1 for r, c in pixels]
            span = max(values) - min(values)
            total_span += span
            total_sq_span += span * span
            if span == 0:
                zero_span_components += 1

        key = (total_span, total_sq_span, -zero_span_components, idx)
        if best_key is None or key < best_key:
            best_key = key
            best_d = d

    d = best_d
    d0, d1 = d
    n = perpendicular(d)
    n0, n1 = n

    # Determine anchor and ordering coordinate for each component
    info = []
    for color, pixels in components:
        projs = [r * d0 + c * d1 for r, c in pixels]
        min_proj = min(projs)

        anchors = [
            (r, c) for r, c in pixels
            if r * d0 + c * d1 == min_proj
        ]

        # If multiple pixels tie, choose the one with smallest perpendicular
        # coordinate, then row, then column.
        anchor = min(
            anchors,
            key=lambda x: (x[0] * n0 + x[1] * n1, x[0], x[1])
        )

        perp_val = anchor[0] * n0 + anchor[1] * n1
        proj_val = anchor[0] * d0 + anchor[1] * d1

        info.append((color, pixels, anchor, len(pixels), perp_val, proj_val))

    # Order objects perpendicular to the line direction
    info.sort(key=lambda x: (x[4], x[5], x[2][0], x[2][1]))

    # Target length is the size of the middle object
    target_len = max(1, info[len(info) // 2][3])

    # Redraw
    out = np.full((H, W), bg, dtype=arr.dtype)

    for color, _, anchor, _, _, _ in info:
        ar, ac = anchor
        for t in range(target_len):
            r = ar + t * d0
            c = ac + t * d1
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = color

    return out