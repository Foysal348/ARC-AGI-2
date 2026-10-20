import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find all plus/cross shapes where a center cell and its 4 orthogonal neighbors
    (up, down, left, right) all have value 4, and change those cells to 8.
    
    A valid cross centered at (r, c) requires:
        grid[r-1, c] == 4, grid[r+1, c] == 4,
        grid[r, c-1] == 4, grid[r, c+1] == 4,
        grid[r, c] == 4
    All 5 cells are then set to 8.
    """
    result = grid.copy()
    rows, cols = grid.shape
    to_change = np.zeros((rows, cols), dtype=bool)
    
    # Check each potential cross center (must not be on the border)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid[r, c] == 4 and
                grid[r - 1, c] == 4 and
                grid[r + 1, c] == 4 and
                grid[r, c - 1] == 4 and
                grid[r, c + 1] == 4):
                # Mark all 5 cells of the cross
                to_change[r, c] = True
                to_change[r - 1, c] = True
                to_change[r + 1, c] = True
                to_change[r, c - 1] = True
                to_change[r, c + 1] = True
    
    result[to_change] = 8
    return result