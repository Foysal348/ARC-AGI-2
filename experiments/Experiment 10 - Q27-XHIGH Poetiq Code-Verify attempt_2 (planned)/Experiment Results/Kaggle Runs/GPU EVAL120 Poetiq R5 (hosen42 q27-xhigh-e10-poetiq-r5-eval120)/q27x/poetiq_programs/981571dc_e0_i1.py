import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the grid by assuming it is the top-left crop of a larger grid
    that is symmetric under horizontal and vertical reflection.

    For an H x W grid, visible mirror pairs are:
        column c <-> (W + 1 - c) for c >= 2
        row r    <-> (H + 1 - r) for r >= 2

    The first two rows and first two columns have no visible mirror.

    Zero cells are treated as unknown. Conflicts are resolved with priority:
        top-left, top-right, bottom-left, bottom-right.

    This is implemented by:
        1. enforcing horizontal symmetry row-wise, with left priority;
        2. enforcing vertical symmetry column-wise, with top priority.
    """
    out = np.array(grid, copy=True)
    h, w = out.shape

    # Horizontal symmetry.
    # For each column pair (c, w + 1 - c), keep the left value if it is nonzero;
    # otherwise copy from the right. If both are nonzero, left wins.
    for c in range(2, w):
        mc = w + 1 - c
        if mc >= w or mc <= c:
            continue

        left = out[:, c]
        right = out[:, mc]

        out[:, c] = np.where(left != 0, left, right)
        out[:, mc] = out[:, c]

    # Vertical symmetry.
    # For each row pair (r, h + 1 - r), keep the top value if it is nonzero;
    # otherwise copy from the bottom. If both are nonzero, top wins.
    for r in range(2, h):
        mr = h + 1 - r
        if mr >= h or mr <= r:
            continue

        top = out[r, :]
        bottom = out[mr, :]

        out[r, :] = np.where(top != 0, top, bottom)
        out[mr, :] = out[r, :]

    return out