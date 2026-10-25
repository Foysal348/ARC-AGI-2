import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor all 4-colored plus shapes to color 8.

    A plus shape is defined locally as:
        . 4 .
        4 4 4
        . 4 .
    where the center and its four orthogonal neighbors are all 4.
    The five cells of each detected plus are changed to 8.
    """
    arr = np.asarray(grid)

    # Guard against invalid or too-small grids.
    if arr.ndim != 2:
        return arr.copy()

    out = arr.copy()

    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return out

    # Find all possible center cells of a plus made of 4s.
    # This boolean mask has shape (height - 2, width - 2).
    centers = (
        (arr[1:-1, 1:-1] == 4) &  # center
        (arr[:-2, 1:-1] == 4) &   # up
        (arr[2:, 1:-1] == 4) &    # down
        (arr[1:-1, :-2] == 4) &   # left
        (arr[1:-1, 2:] == 4)      # right
    )

    # For each detected center, recolor the center and its four arms.
    for r, c in np.argwhere(centers):
        r += 1
        c += 1

        out[r, c] = 8
        out[r - 1, c] = 8
        out[r + 1, c] = 8
        out[r, c - 1] = 8
        out[r, c + 1] = 8

    return out