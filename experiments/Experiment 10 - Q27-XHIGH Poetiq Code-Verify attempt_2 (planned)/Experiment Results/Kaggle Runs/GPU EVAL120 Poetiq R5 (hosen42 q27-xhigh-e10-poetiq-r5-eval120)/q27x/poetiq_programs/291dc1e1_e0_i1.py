import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the bordered input grid into the stacked centered output.

    Rules inferred from the examples:
    1. Ignore the L-shaped border made of 0, 1, and 2.
    2. Find all 4-connected non-8 components inside the border.
    3. The 0 corner is the origin. The border color 2 defines the primary
       ordering axis, and color 1 defines the secondary ordering axis.
    4. Read components by primary axis first, then secondary axis.
       - Horizontal direction is away from the 0 corner:
         left-to-right if 0 is top-left, right-to-left if 0 is top-right.
       - Vertical direction is always top-to-bottom.
    5. If the primary axis is horizontal (top border is 2), pieces are stored
       rotated clockwise, so rotate them counterclockwise. For safety, tall
       pieces are also rotated counterclockwise.
    6. Stack the processed pieces vertically, centered, with 8 as background.
    """
    grid = np.array(grid, dtype=int, copy=True)

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    # Determine which side has the vertical border, based on the 0 corner.
    if grid[0, 0] == 0:
        side = "left"
        inside_start, inside_end = 1, W
    elif grid[0, W - 1] == 0:
        side = "right"
        inside_start, inside_end = 0, W - 1
    else:
        # Fallback for unexpected inputs.
        zeros = np.argwhere(grid == 0)
        if zeros.size:
            side = "left" if zeros[0, 1] < W / 2 else "right"
            inside_start, inside_end = (1, W) if side == "left" else (0, W - 1)
        else:
            side = "left"
            inside_start, inside_end = 0, W

    # Read the two border colors: top edge and side edge.
    top_color = -1
    side_color = -1

    if side == "left":
        for c in range(1, W):
            v = int(grid[0, c])
            if v != 0 and v != 8:
                top_color = v
                break

        for r in range(1, H):
            v = int(grid[r, 0])
            if v != 0 and v != 8:
                side_color = v
                break
    else:
        for c in range(W - 2, -1, -1):
            v = int(grid[0, c])
            if v != 0 and v != 8:
                top_color = v
                break

        for r in range(1, H):
            v = int(grid[r, W - 1])
            if v != 0 and v != 8:
                side_color = v
                break

    # Primary axis is the axis labeled with 2.
    primary = None
    if top_color == 2:
        primary = "horizontal"
    elif side_color == 2:
        primary = "vertical"
    elif top_color == 1:
        primary = "vertical"
    elif side_color == 1:
        primary = "horizontal"

    # Mask of interior cells that are not background 8.
    mask = np.zeros((H, W), dtype=bool)
    if H > 1 and inside_end > inside_start:
        mask[1:, inside_start:inside_end] = grid[1:, inside_start:inside_end] != 8

    def label_components(m: np.ndarray):
        """Label 4-connected components of a boolean mask."""
        h, w = m.shape
        labels = np.zeros((h, w), dtype=int)
        next_label = 1

        for i in range(h):
            for j in range(w):
                if m[i, j] and labels[i, j] == 0:
                    labels[i, j] = next_label
                    stack = [(i, j)]

                    while stack:
                        y, x = stack.pop()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if (
                                0 <= ny < h
                                and 0 <= nx < w
                                and m[ny, nx]
                                and labels[ny, nx] == 0
                            ):
                                labels[ny, nx] = next_label
                                stack.append((ny, nx))

                    next_label += 1

        return labels, next_label - 1

    labels, count = label_components(mask)

    comps = []
    for lab in range(1, count + 1):
        ys, xs = np.where(labels == lab)
        min_r, max_r = int(ys.min()), int(ys.max())
        min_c, max_c = int(xs.min()), int(xs.max())

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        piece = grid[min_r:max_r + 1, min_c:max_c + 1].copy()

        comps.append({
            "piece": piece,
            "height": height,
            "width": width,
            "center_r": (min_r + max_r) / 2.0,
            "center_c": (min_c + max_c) / 2.0,
        })

    if not comps:
        return np.zeros((0, 0), dtype=grid.dtype)

    # If border colors were not clear, infer primary axis from piece shapes.
    if primary is None:
        tall = sum(1 for c in comps if c["height"] > c["width"])
        wide = sum(1 for c in comps if c["width"] > c["height"])
        primary = "horizontal" if tall >= wide else "vertical"

    # Horizontal direction is away from the 0 corner.
    h_dir = 1 if side == "left" else -1

    # Sort by primary axis, then secondary axis.
    if primary == "horizontal":
        ordered = sorted(
            comps,
            key=lambda c: (c["center_c"] * h_dir, c["center_r"])
        )
    else:
        ordered = sorted(
            comps,
            key=lambda c: (c["center_r"], c["center_c"] * h_dir)
        )

    pieces = []
    for comp in ordered:
        piece = comp["piece"]
        h, w = piece.shape

        # Restore orientation for vertically stored pieces.
        if (primary == "horizontal" and h >= w) or (h > w):
            piece = np.rot90(piece, k=1)  # counterclockwise

        pieces.append(piece)

    max_w = max(p.shape[1] for p in pieces)
    total_h = sum(p.shape[0] for p in pieces)

    out = np.full((total_h, max_w), 8, dtype=grid.dtype)

    y = 0
    for p in pieces:
        ph, pw = p.shape
        left = (max_w - pw) // 2
        out[y:y + ph, left:left + pw] = p
        y += ph

    return out