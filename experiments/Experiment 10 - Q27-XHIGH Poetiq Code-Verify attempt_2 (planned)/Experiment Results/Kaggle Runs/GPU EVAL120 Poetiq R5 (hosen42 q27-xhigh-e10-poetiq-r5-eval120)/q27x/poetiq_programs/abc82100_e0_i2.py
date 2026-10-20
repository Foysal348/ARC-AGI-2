import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Infer and apply the transformation used in the examples.

    The observed rule is:
      * Colour 8 is a mask / obstacle and is always removed.
      * The meaningful non-zero, non-8 pixels form one or more "active"
        structures.
      * The output is a checkerboard projection of those structures onto the
        borders of the grid.
      * The colour used for each projection is normally the colour of the
        structure on the opposite side of the grid.  If no opposite structure
        exists, the structure's own colour is used, with a small fallback to
        a nearby singleton colour when that makes the examples consistent.

    The implementation is intentionally conservative:
      1. Split the non-zero, non-8 pixels into 8-connected components.
      2. For each component, determine which border it is closest to.
      3. Look for a matching component on the opposite border and take its
         dominant colour.  If absent, use the component's own dominant colour.
      4. Project the component toward that border with cardinal and diagonal
         rays, stopping at the border or at original 8 cells.  The reached
         cells are written as a checkerboard using parity of coordinates.

    This compact rule matches the visible behaviour in the supplied examples:
      - straight side lines become opposite-side checkerboard bands;
      - top/bottom shapes become border checkerboard projections;
      - 8 pixels are erased and block projection.
    """

    if grid is None:
        return np.zeros((0, 0), dtype=np.uint8)

    H, W = grid.shape
    if H == 0 or W == 0:
        return np.zeros_like(grid)

    out = np.zeros((H, W), dtype=np.uint8)

    # Active pixels are all non-background, non-8 pixels.
    active = (grid != 0) & (grid != 8)
    if not active.any():
        return out

    # ---------------------------------------------------------------
    # 8-connected component labelling for active pixels.
    # ---------------------------------------------------------------
    label = -np.ones((H, W), dtype=int)
    components = []
    current = 0

    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for r in range(H):
        for c in range(W):
            if active[r, c] and label[r, c] == -1:
                stack = [(r, c)]
                label[r, c] = current
                cells = []

                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))

                    for dr, dc in dirs8:
                        nx, ny = x + dr, y + dc
                        if 0 <= nx < H and 0 <= ny < W:
                            if active[nx, ny] and label[nx, ny] == -1:
                                label[nx, ny] = current
                                stack.append((nx, ny))

                components.append(cells)
                current += 1

    # ---------------------------------------------------------------
    # Helper: dominant colour of a component.
    # ---------------------------------------------------------------
    def dominant_color(cells):
        counts = {}
        for r, c in cells:
            val = int(grid[r, c])
            counts[val] = counts.get(val, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # ---------------------------------------------------------------
    # Helper: classify a component by its closest side.
    # ---------------------------------------------------------------
    def side_of(cells):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        cr = sum(rows) / len(rows)
        cc = sum(cols) / len(cols)

        d_top = cr
        d_bottom = (H - 1) - cr
        d_left = cc
        d_right = (W - 1) - cc

        m = min(d_top, d_bottom, d_left, d_right)
        if m == d_top:
            return "top"
        if m == d_bottom:
            return "bottom"
        if m == d_left:
            return "left"
        return "right"

    opposite = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left"
    }

    # Precompute side and dominant colour for every component.
    comp_side = []
    comp_color = []
    for cells in components:
        comp_side.append(side_of(cells))
        comp_color.append(dominant_color(cells))

    # All singleton active colours, used only as a fallback.
    singleton_colors = []
    for r in range(H):
        for c in range(W):
            if active[r, c]:
                val = int(grid[r, c])
                # Count this value globally.
                same = int(np.count_nonzero((grid == val) & active))
                if same == 1 and val not in singleton_colors:
                    singleton_colors.append(val)

    # ---------------------------------------------------------------
    # Project each component to its closest side.
    # ---------------------------------------------------------------
    for idx, cells in enumerate(components):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rmin, rmax = min(rows), max(rows)
        cmin, cmax = min(cols), max(cols)
        height = rmax - rmin + 1
        width = cmax - cmin + 1

        side = comp_side[idx]
        own_color = comp_color[idx]

        # Choose output colour:
        #   1. opposite-side component colour if it exists
        #   2. otherwise own colour
        #   3. if own colour is 8/0, fall back to a singleton colour
        target_color = None
        opp = opposite[side]
        for jdx, other_side in enumerate(comp_side):
            if jdx == idx:
                continue
            if other_side == opp:
                target_color = comp_color[jdx]
                break

        if target_color is None or target_color in (0, 8):
            target_color = own_color

        if target_color in (0, 8):
            if singleton_colors:
                target_color = singleton_colors[0]
            else:
                continue

        # Rays are chosen to generate the checkerboard bands seen in the
        # examples.  Horizontal components project vertically and diagonally;
        # vertical components project horizontally and diagonally.
        if side == "top":
            rays = [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1)]
        elif side == "bottom":
            rays = [(1, 0), (1, -1), (1, 1), (0, -1), (0, 1)]
        elif side == "left":
            rays = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)]
        else:  # right
            rays = [(0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)]

        cell_set = set(cells)

        for r0, c0 in cells:
            for dr, dc in rays:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < H and 0 <= c < W:
                    # Do not cross original 8 cells.
                    if grid[r, c] == 8:
                        break

                    # Do not cross other active cells of the same component.
                    if (r, c) in cell_set:
                        break

                    # Write a checkerboard cell only when parity matches the
                    # border orientation.  This reproduces the alternating
                    # patterns in the examples.
                    if side in ("top", "bottom"):
                        if (r + c) % 2 == 0:
                            out[r, c] = target_color
                    else:
                        if (r - c) % 2 == 0:
                            out[r, c] = target_color

                    r += dr
                    c += dc

    return out