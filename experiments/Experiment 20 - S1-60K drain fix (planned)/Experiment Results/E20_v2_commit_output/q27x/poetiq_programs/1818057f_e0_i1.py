import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every cardinal cross of 4s to 8.

    A cross is defined by a center cell of value 4 whose four orthogonal
    neighbors (up, down, left, right) are also 4. The center and those
    four neighbors are changed to 8. All other cells remain unchanged.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return grid

    result = grid.copy()
    height, width = grid.shape

    if height < 3 or width < 3:
        return result

    is_four = grid == 4

    # Centers of valid crosses:
    # center, up, down, left, and right must all be 4.
    centers = (
        is_four[1:-1, 1:-1] &
        is_four[0:-2, 1:-1] &
        is_four[2:, 1:-1] &
        is_four[1:-1, 0:-2] &
        is_four[1:-1, 2:]
    )

    # Mark the center and its four arms.
    mark = np.zeros((height, width), dtype=bool)

    mark[1:-1, 1:-1] |= centers  # center
    mark[0:-2, 1:-1] |= centers  # up
    mark[2:, 1:-1]   |= centers  # down
    mark[1:-1, 0:-2] |= centers  # left
    mark[1:-1, 2:]   |= centers  # right

    # Only original 4 cells should become 8.
    result[mark & is_four] = 8

    return result