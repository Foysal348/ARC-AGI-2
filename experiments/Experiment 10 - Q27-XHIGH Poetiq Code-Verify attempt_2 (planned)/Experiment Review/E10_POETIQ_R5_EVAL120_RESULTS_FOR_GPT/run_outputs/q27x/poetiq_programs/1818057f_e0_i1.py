import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor all 5-cell plus-shaped patterns made of 4s to 8.

    A valid plus pattern is a cell with value 4 whose four orthogonal
    neighbors (up, down, left, right) are also 4. When such a pattern is
    found, the center and its four neighbors are changed to 8.

    Overlapping patterns are handled naturally. If the grid is too small
    to contain a plus pattern, or if no such pattern exists, the grid is
    returned unchanged.
    """
    grid = np.asarray(grid)
    result = grid.copy()

    # Handle invalid or too-small inputs gracefully.
    if grid.ndim != 2:
        return result

    h, w = grid.shape
    if h < 3 or w < 3:
        return result

    # In all examples, the pattern color is 4 and the replacement color is 8.
    target = 4
    if not np.any(grid == target):
        return result

    # Boolean mask where the center of a plus pattern would be.
    # This has shape (h - 2, w - 2).
    centers = (
        (grid[1:-1, 1:-1] == target) &  # center
        (grid[:-2, 1:-1] == target) &   # up
        (grid[2:, 1:-1] == target) &    # down
        (grid[1:-1, :-2] == target) &   # left
        (grid[1:-1, 2:] == target)      # right
    )

    if not np.any(centers):
        return result

    # Convert mask indices to original grid coordinates.
    r, c = np.nonzero(centers)
    r += 1
    c += 1

    # Recolor the center and the four arms of every detected plus.
    result[r, c] = 8
    result[r - 1, c] = 8
    result[r + 1, c] = 8
    result[r, c - 1] = 8
    result[r, c + 1] = 8

    return result