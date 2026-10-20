import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each box (rectangle bordered by 0s), find numbered pixels inside.
    Each number is projected to its nearest corner: placed at the corner
    and one step beyond in each of the two axes away from the number.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find the background color (most common on the border)
    border_vals = np.concatenate([
        grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
    ])
    bg_color = np.bincount(border_vals).argmax()
    
    # Find connected components of 0s (box borders)
    zero_mask = (grid == 0).astype(int)
    labeled, num_features = ndimage.label(zero_mask)
    
    for i in range(1, num_features + 1):
        component = (labeled == i)
        rows_idx, cols_idx = np.where(component)
        
        r_min, r_max = rows_idx.min(), rows_idx.max()
        c_min, c_max = cols_idx.min(), cols_idx.max()
        
        # Check if this component forms a rectangular border
        # Top edge: all cells in row r_min from c_min to c_max should be 0
        top_ok = np.all(grid[r_min, c_min:c_max+1] == 0)
        # Bottom edge
        bottom_ok = np.all(grid[r_max, c_min:c_max+1] == 0)
        # Left edge
        left_ok = np.all(grid[r_min:r_max+1, c_min] == 0)
        # Right edge
        right_ok = np.all(grid[r_min:r_max+1, c_max] == 0)
        
        if not (top_ok and bottom_ok and left_ok and right_ok):
            continue
        
        # Find the interior of the box
        int_r_min, int_r_max = r_min + 1, r_max - 1
        int_c_min, int_c_max = c_min + 1, c_max - 1
        
        if int_r_min > int_r_max or int_c_min > int_c_max:
            continue
        
        # Find non-zero, non-background pixels in the interior
        interior = grid[int_r_min:int_r_max+1, int_c_min:int_c_max+1]
        for dr in range(interior.shape[0]):
            for dc in range(interior.shape[1]):
                val = interior[dr, dc]
                if val == 0 or val == bg_color:
                    continue
                
                r = int_r_min + dr
                c = int_c_min + dc
                
                # Find the nearest corner
                corners = [
                    (r_min, c_min),  # top-left
                    (r_min, c_max),  # top-right
                    (r_max, c_min),  # bottom-left
                    (r_max, c_max),  # bottom-right
                ]
                
                # Calculate Manhattan distance to each corner
                dists = [abs(r - cr) + abs(c - cc) for cr, cc in corners]
                nearest_idx = np.argmin(dists)
                corner_r, corner_c = corners[nearest_idx]
                
                # Determine direction away from the number
                dr_sign = np.sign(r - corner_r)  # positive if number is below corner
                dc_sign = np.sign(c - corner_c)  # positive if number is right of corner
                
                # Place at corner
                if 0 <= corner_r < rows and 0 <= corner_c < cols:
                    result[corner_r, corner_c] = val
                
                # Place one step beyond corner vertically (away from number)
                vr = corner_r - dr_sign if dr_sign != 0 else corner_r
                if dr_sign != 0 and 0 <= vr < rows and 0 <= corner_c < cols:
                    result[vr, corner_c] = val
                
                # Place one step beyond corner horizontally (away from number)
                hc = corner_c - dc_sign if dc_sign != 0 else corner_c
                if dc_sign != 0 and 0 <= corner_r < rows and 0 <= hc < cols:
                    result[corner_r, hc] = val
    
    return result