import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every orthogonal plus sign made of 4s to color 8.

    A valid plus sign is a 5-cell pattern:
        . 4 .
        4 4 4
        . 4 .
    where all five cells in the input are 4. The output changes exactly
    those five cells to 8. If plus signs overlap, the union of their cells
    is changed to 8.
    """
    grid = np.asarray(grid)

    # Ensure the input is 2D; otherwise return a copy unchanged.
    if grid.ndim != 2:
        return grid.copy()

    result = grid.copy()
    h, w = grid.shape

    # A plus sign needs at least 3 rows and 3 columns.
    if h < 3 or w < 3:
        return result

    source_color = 4
    target_color = 8

    is_source = (grid == source_color)
    mark = np.zeros((h, w), dtype=bool)

    # Check every possible interior center of a plus sign.
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (
                is_source[r, c] and
                is_source[r - 1, c] and
                is_source[r + 1, c] and
                is_source[r, c - 1] and
                is_source[r, c + 1]
            ):
                # Mark the center and its four orthogonal neighbors.
                mark[r, c] = True
                mark[r - 1, c] = True
                mark[r + 1, c] = True
                mark[r, c - 1] = True
                mark[r, c + 1] = True

    # Change only originally-source cells that belong to at least one plus sign.
    result[mark & is_source] = target_color

    return result