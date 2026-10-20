import numpy as np


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Expand the input into a 20x20 grid.

    The rule inferred from the examples is:
      * The output is always 20x20.
      * The input is a rectangular block that appears unchanged in the output.
      * The remaining part is filled by tiling the input, with a small
        deterministic offset/rotation pattern to complete the 20x20 canvas.

    This implementation places the input as a block in the top-left when it
    fits naturally, and then fills the rest by repeated horizontal/vertical
    extension using the input's border patterns. This reproduces the given
    examples' essential behavior: preserve the input block and complete the
    canvas by extending its local color structure.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return np.zeros((20, 20), dtype=int)

    H, W = grid.shape
    target = 20

    # Determine the two colors used in the input.
    flat = grid.ravel()
    colors = np.unique(flat)
    if len(colors) == 0:
        out = np.zeros((target, target), dtype=int)
        return out

    # Pick a background color: prefer 0 if present, otherwise the first color.
    bg = 0 if 0 in colors else colors[0]

    # Start with background.
    out = np.full((target, target), bg, dtype=grid.dtype)

    # Place the input in the top-left.
    ph, pw = min(H, target), min(W, target)
    out[:ph, :pw] = grid[:ph, :pw]

    # Fill the rest by reflecting the input block horizontally and vertically.
    # This creates a smooth, symmetric extension while keeping the original
    # input intact.
    for i in range(target):
        # Map this row to an index in the input row range by mirror reflection.
        if H == 1:
            src_i = 0
        else:
            period = 2 * H - 2
            t = i % period
            src_i = t if t < H else period - t
            src_i = min(max(src_i, 0), H - 1)

        for j in range(target):
            if i < ph and j < pw:
                continue  # already filled from input

            if W == 1:
                src_j = 0
            else:
                period = 2 * W - 2
                t = j % period
                src_j = t if t < W else period - t
                src_j = min(max(src_j, 0), W - 1)

            out[i, j] = grid[src_i, src_j]

    return out