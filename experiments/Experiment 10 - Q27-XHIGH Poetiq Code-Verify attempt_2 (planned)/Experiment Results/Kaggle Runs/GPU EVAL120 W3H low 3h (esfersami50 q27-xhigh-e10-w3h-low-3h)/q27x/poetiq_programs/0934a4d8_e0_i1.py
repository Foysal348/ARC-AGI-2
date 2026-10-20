import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find the rectangular block of 8s in the grid and fill it using
    the vertical mirror symmetry of the grid (Row[r] == Row[H-1-r]).
    Return the filled sub-block as the output.
    """
    grid = grid.copy()
    H, W = grid.shape
    
    # Find the bounding box of 8s
    mask = (grid == 8)
    rows_with_8 = np.where(mask.any(axis=1))[0]
    cols_with_8 = np.where(mask.any(axis=0))[0]
    
    r_start, r_end = rows_with_8[0], rows_with_8[-1]
    c_start, c_end = cols_with_8[0], cols_with_8[-1]
    
    # Fill using vertical mirror: Row[r] should equal Row[H-1-r]
    # For each row in the 8-block, get values from the mirror row
    for r in range(r_start, r_end + 1):
        mirror_r = H - 1 - r
        for c in range(c_start, c_end + 1):
            if grid[r, c] == 8:
                if grid[mirror_r, c] != 8:
                    grid[r, c] = grid[mirror_r, c]
    
    # Handle rows where the mirror row also has 8s (8-block straddles center)
    # These rows mirror each other within the 8-block
    for r in range(r_start, r_end + 1):
        mirror_r = H - 1 - r
        if r != mirror_r:
            for c in range(c_start, c_end + 1):
                if grid[r, c] == 8 and grid[mirror_r, c] != 8:
                    grid[r, c] = grid[mirror_r, c]
    
    # If there are still 8s remaining, use the most common value from 
    # the same column in non-8 rows as fallback
    for c in range(c_start, c_end + 1):
        col_values = []
        for r in range(H):
            if grid[r, c] != 8:
                col_values.append(grid[r, c])
        if col_values:
            from collections import Counter
            most_common = Counter(col_values).most_common(1)[0][0]
            for r in range(r_start, r_end + 1):
                if grid[r, c] == 8:
                    grid[r, c] = most_common
    
    # Extract and return the filled sub-block
    return grid[r_start:r_end+1, c_start:c_end+1]