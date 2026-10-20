import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the 0-block to the lowest valid position within the 1-shape.
    A position is valid if all cells in the block area are 'fillable' (1, 0, or 2).
    The original 0 position is excluded. Among valid columns at the best row,
    the one closest to the original column is chosen.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find the 0 block
    zero_positions = np.argwhere(grid == 0)
    if len(zero_positions) == 0:
        return result
    
    z_min_r, z_min_c = zero_positions.min(axis=0)
    z_max_r, z_max_c = zero_positions.max(axis=0)
    block_h = z_max_r - z_min_r + 1
    block_w = z_max_c - z_min_c + 1
    
    # Fillable: cells that are 1, 0, or 2
    fillable = (grid == 1) | (grid == 0) | (grid == 2)
    
    # Find the lowest valid row (scanning from bottom), excluding original position
    best_row = z_min_r
    best_col = z_min_c
    found = False
    
    for r in range(rows - block_h, -1, -1):
        valid_cols = []
        for c in range(cols - block_w + 1):
            if r == z_min_r and c == z_min_c:
                continue
            block_region = fillable[r:r+block_h, c:c+block_w]
            if block_region.all():
                valid_cols.append(c)
        if valid_cols:
            best_col = min(valid_cols, key=lambda c: abs(c - z_min_c))
            best_row = r
            found = True
            break
    
    if not found:
        return result
    
    # Fill old 0 position with 1
    result[z_min_r:z_max_r+1, z_min_c:z_max_c+1] = 1
    # Place 0 at new position
    result[best_row:best_row+block_h, best_col:best_col+block_w] = 0
    
    return result