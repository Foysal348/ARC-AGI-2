import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill zero regions in the grid using 180-degree rotational symmetry.
    
    The complete grid satisfies: output[r][c] == output[H-1-r][W-1-c].
    Zero values in the input indicate missing data. Each zero at (r, c) 
    is filled with the value from the rotationally symmetric position 
    (H-1-r, W-1-c).
    
    This is verified across all provided examples where the output grid 
    exhibits 180-degree rotational symmetry and all zeros from the input 
    are filled accordingly.
    """
    grid = grid.copy()
    H, W = grid.shape
    
    # Identify zero positions
    zero_mask = (grid == 0)
    
    # The 180-degree rotation maps (r,c) -> (H-1-r, W-1-c)
    # We fill each zero with the value at its rotated position
    rotated = grid[::-1, ::-1]
    
    # Vectorized fill: where grid is 0, take value from rotated grid
    grid[zero_mask] = rotated[zero_mask]
    
    return grid