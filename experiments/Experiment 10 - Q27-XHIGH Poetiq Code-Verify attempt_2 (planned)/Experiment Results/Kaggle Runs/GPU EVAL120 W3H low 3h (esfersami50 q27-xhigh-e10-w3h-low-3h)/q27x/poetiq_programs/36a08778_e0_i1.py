import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each connected component of 2s (4-connectivity), draw a rectangular border
    at distance 1 from the component. Only background cells (7) are changed to 6.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    mask = (grid == 2)
    labeled, num_features = ndimage.label(mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
    
    for i in range(1, num_features + 1):
        component_cells = np.where(labeled == i)
        min_r, max_r = component_cells[0].min(), component_cells[0].max()
        min_c, max_c = component_cells[1].min(), component_cells[1].max()
        
        exp_min_r = max(0, min_r - 1)
        exp_max_r = min(rows - 1, max_r + 1)
        exp_min_c = max(0, min_c - 1)
        exp_max_c = min(cols - 1, max_c + 1)
        
        for c in range(exp_min_c, exp_max_c + 1):
            if result[exp_min_r, c] == 7:
                result[exp_min_r, c] = 6
            if result[exp_max_r, c] == 7:
                result[exp_max_r, c] = 6
        for r in range(exp_min_r, exp_max_r + 1):
            if result[r, exp_min_c] == 7:
                result[r, exp_min_c] = 6
            if result[r, exp_max_c] == 7:
                result[r, exp_max_c] = 6
    
    return result