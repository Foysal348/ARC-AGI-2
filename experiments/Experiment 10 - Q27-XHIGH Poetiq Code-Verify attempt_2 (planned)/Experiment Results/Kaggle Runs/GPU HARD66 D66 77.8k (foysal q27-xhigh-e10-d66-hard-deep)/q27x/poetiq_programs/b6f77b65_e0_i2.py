import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. The top-left cell contains the "key" color.
    2. Remove every other cell having that key color, keeping only (0, 0).
    3. The remaining colored cells are treated as vertical "column stacks":
       in each column, maximal consecutive non-zero runs are separated by zeros.
    4. These runs fall downward as rigid blocks, from bottom to top.
       A block can only move downward until it would leave the grid or overlap
       a block that has already been placed.

    This matches the supplied examples: the top-left color is erased, and the
    remaining line/segment blocks settle downward column by column.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Work on a copy.
    out = grid.copy()

    # Key color is the top-left pixel.
    key = out[0, 0]

    # Remove all key-colored cells except the key pixel itself.
    if key != 0:
        key_mask = (out == key)
        key_mask[0, 0] = False
        out[key_mask] = 0

    # Ensure the key pixel itself is preserved exactly.
    out[0, 0] = key

    # Process each column independently.
    for x in range(W):
        col = out[:, x]

        # If this column is entirely zero, nothing to do.
        if not np.any(col):
            continue

        # Build maximal vertical non-zero runs in this column.
        # Each run is represented by its top and bottom rows.
        runs = []
        y = 0
        while y < H:
            if col[y] != 0:
                top = y
                while y + 1 < H and col[y + 1] != 0:
                    y += 1
                bottom = y
                runs.append((top, bottom))
            y += 1

        if not runs:
            continue

        # Clear the column, then refill it with settled blocks.
        # We will place blocks from bottom to top.
        new_col = np.zeros_like(col)

        # For each run, we know its original height and colors.
        # We need to decide the largest downward shift that does not collide
        # with already placed lower blocks or the bottom border.
        placed_blocks = []  # list of (top, bottom) in final positions, bottom-up

        # Process bottommost original run first.
        for top, bottom in reversed(runs):
            height = bottom - top + 1

            # Extract the original colors of this run.
            block_colors = col[top:bottom + 1].copy()

            # Maximum downward shift limited by grid bottom.
            max_shift = H - 1 - bottom

            # Find the lowest position that does not overlap any placed block.
            best_shift = max_shift
            # Try all shifts from largest to smallest; first valid wins.
            for shift in range(max_shift, -1, -1):
                nt = top + shift
                nb = bottom + shift

                ok = True
                for pt, pb in placed_blocks:
                    # Overlap check in this column.
                    if nt <= pb and pt <= nb:
                        ok = False
                        break
                if ok:
                    best_shift = shift
                    break

            final_top = top + best_shift
            final_bottom = bottom + best_shift

            new_col[final_top:final_bottom + 1] = block_colors
            placed_blocks.append((final_top, final_bottom))

        out[:, x] = new_col

    # Preserve top-left key color one final time.
    out[0, 0] = key

    return out