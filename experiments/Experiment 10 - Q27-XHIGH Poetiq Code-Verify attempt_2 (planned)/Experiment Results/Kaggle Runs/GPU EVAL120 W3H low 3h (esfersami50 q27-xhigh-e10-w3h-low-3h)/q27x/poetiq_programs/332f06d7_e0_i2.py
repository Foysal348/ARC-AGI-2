import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the 0-block to the position closest to the 2-block center
    where the block fits within the fillable region (cells that are 1, 0, or 2).
    The old 0 position is filled with 1s. The 2-block remains unchanged.
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
    
    # Find the 2 block (the attractor)
    two_positions = np.argwhere(grid == 2)
    if len(two_positions) == 0:
        return result
    
    t_min_r, t_min_c = two_positions.min(axis=0)
    t_max_r, t_max_c = two_positions.max(axis=0)
    target_r = (t_min_r + t_max_r) / 2.0
    target_c = (t_min_c + t_max_c) / 2.0
    
    # Fillable: cells that are 1, 0, or 2
    fillable = (grid == 1) | (grid == 0) | (grid == 2)
    
    # Find the position closest to 2-block center where the 0-block fits
    best_dist = float('inf')
    best_row = z_min_r
    best_col = z_min_c
    found = False
    
    for r in range(rows - block_h + 1):
        for c in range(cols - block_w + 1):
            if r == z_min_r and c == z_min_c:
                continue
            if fillable[r:r+block_h, c:c+block_w].all():
                center_r = r + (block_h - 1) / 2.0
                center_c = c + (block_w - 1) / 2.0
                dist = (center_r - target_r) ** 2 + (center_c - target_c) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_row = r
                    best_col = c
                    found = True
    
    # Fallback: allow non-background cells if strict check fails
    if not found:
        non_bg = grid != 3
        for r in range(rows - block_h + 1):
            for c in range(cols - block_w + 1):
                if r == z_min_r and c == z_min_c:
                    continue
                if non_bg[r:r+block_h, c:c+block_w].all():
                    center_r = r + (block_h - 1) / 2.0
                    center_c = c + (block_w - 1) / 2.0
                    dist = (center_r - target_r) ** 2 + (center_c - target_c) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_row = r
                        best_col = c
                        found = True
    
    if not found:
        return result
    
    # Fill old 0 position with 1
    result[z_min_r:z_max_r+1, z_min_c:z_max_c+1] = 1
    # Place 0-block at new position
    result[best_row:best_row+block_h, best_col:best_col+block_w] = 0
    
    return result