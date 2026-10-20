import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    The non-8 pixels are treated as directional "seeds" around a central 8-object.
    For each side of the central object (top / bottom / left / right), the seeds on
    that side are projected outward in the three outward diagonal/orthogonal rays.
    Only cells whose parity matches the seed parity are painted.  The paint color
    is taken from the opposite side of the central 8-object (the key color on the
    other side).  All 8 pixels are removed.
    """
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    out = np.zeros_like(grid)

    eight = (grid == 8)
    seeds = np.argwhere((grid != 0) & ~eight)
    if len(seeds) == 0:
        return out

    # Center of the 8-object, or grid center if no 8 exists.
    if eight.any():
        ys, xs = np.where(eight)
        cr, cc = float(ys.mean()), float(xs.mean())
    else:
        cr, cc = (H - 1) / 2.0, (W - 1) / 2.0

    cr_i, cc_i = int(round(cr)), int(round(cc))

    # Collect seeds by side relative to the central point.
    top, bottom, left, right = [], [], [], []
    for r, c in seeds:
        v = grid[r, c]
        if r < cr_i:
            top.append((r, c, v))
        elif r > cr_i:
            bottom.append((r, c, v))
        if c < cc_i:
            left.append((r, c, v))
        elif c > cc_i:
            right.append((r, c, v))

    def majority_color(cells):
        if not cells:
            return None
        vals = [v for _, _, v in cells]
        return max(set(vals), key=vals.count)

    # Main color for each side.
    c_top = majority_color(top)
    c_bottom = majority_color(bottom)
    c_left = majority_color(left)
    c_right = majority_color(right)

    # Paint one side.
    def paint_side(cells, paint_color, side):
        if not cells or paint_color is None:
            return

        # Use the parity of the seeds on this side.
        parities = [((r + c) & 1) for r, c, _ in cells]
        parity = max(set(parities), key=parities.count)

        # If the side has only one unique seed color and that color differs from
        # the paint color, we project it.  If the paint color is the same as the
        # side color, skip to avoid redundant overwriting.
        side_colors = {v for _, _, v in cells}
        if len(side_colors) == 1 and next(iter(side_colors)) == paint_color:
            return

        # For each seed, project in the three outward directions.
        for r0, c0, _ in cells:
            if ((r0 + c0) & 1) != parity:
                continue

            if side == "top":
                dirs = [(-1, -1), (-1, 0), (-1, 1)]
            elif side == "bottom":
                dirs = [(1, -1), (1, 0), (1, 1)]
            elif side == "left":
                dirs = [(-1, -1), (0, -1), (1, -1)]
            else:  # right
                dirs = [(-1, 1), (0, 1), (1, 1)]

            for dr, dc in dirs:
                r, c = r0, c0
                while 0 <= r < H and 0 <= c < W:
                    if eight[r, c]:
                        break
                    if ((r + c) & 1) == parity:
                        out[r, c] = paint_color

                    # Stop if moving inward toward the 8-object would cross it.
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break

                    # Keep rays on the outward side of the central object.
                    if side == "top" and nr > r:
                        break
                    if side == "bottom" and nr < r:
                        break
                    if side == "left" and nc > c:
                        break
                    if side == "right" and nc < c:
                        break

                    # Optional: stop at a strong central 8 block.
                    if eight[nr, nc]:
                        break

                    r, c = nr, nc

    # Paint opposite-side key colors:
    # top side is painted with bottom side's color, etc.
    paint_side(top, c_bottom, "top")
    paint_side(bottom, c_top, "bottom")
    paint_side(left, c_right, "left")
    paint_side(right, c_left, "right")

    return out