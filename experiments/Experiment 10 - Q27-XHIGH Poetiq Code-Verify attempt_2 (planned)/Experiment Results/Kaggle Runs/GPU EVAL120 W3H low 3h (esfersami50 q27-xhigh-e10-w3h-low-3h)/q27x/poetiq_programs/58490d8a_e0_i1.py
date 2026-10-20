import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
    1. Finding the 0-region (bounding box of all 0 cells).
    2. Identifying marker colors within the 0-region (non-zero pixels).
    3. Counting 8-connected components of each marker color outside the 0-region.
    4. Placing that many pixels of each color in the output at the marker's row,
       starting at the marker's column and going right with spacing of 2.
    """
    grid = grid.copy()
    H, W = grid.shape
    
    # Step 1: Find the 0-region (bounding box of all 0 cells)
    zero_mask = (grid == 0)
    if not np.any(zero_mask):
        return grid
    
    rows_with_zeros = np.where(zero_mask.any(axis=1))[0]
    cols_with_zeros = np.where(zero_mask.any(axis=0))[0]
    r_min, r_max = rows_with_zeros[0], rows_with_zeros[-1]
    c_min, c_max = cols_with_zeros[0], cols_with_zeros[-1]
    
    region_height = r_max - r_min + 1
    region_width = c_max - c_min + 1
    region = grid[r_min:r_max+1, c_min:c_max+1]
    
    # Step 2: Find markers (non-zero pixels) in the 0-region
    markers = {}  # color -> (relative_row, relative_col)
    for r in range(region_height):
        for c in range(region_width):
            val = int(region[r, c])
            if val != 0:
                if val not in markers:
                    markers[val] = (r, c)
    
    # Step 3: Create blocked mask for the 0-region
    blocked = np.zeros((H, W), dtype=bool)
    blocked[r_min:r_max+1, c_min:c_max+1] = True
    
    # Step 4: Count 8-connected components for each marker color (outside 0-region)
    def count_components(color):
        visited = np.zeros((H, W), dtype=bool)
        count = 0
        for r in range(H):
            for c in range(W):
                if grid[r, c] == color and not visited[r, c] and not blocked[r, c]:
                    count += 1
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.popleft()
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < H and 0 <= nc < W:
                                    if (grid[nr, nc] == color and 
                                        not visited[nr, nc] and 
                                        not blocked[nr, nc]):
                                        visited[nr, nc] = True
                                        queue.append((nr, nc))
        return count
    
    # Step 5: Build output grid
    output = np.zeros((region_height, region_width), dtype=grid.dtype)
    for color, (mr, mc) in markers.items():
        count = count_components(color)
        for i in range(count):
            col = mc + 2 * i
            if mr < region_height and col < region_width:
                output[mr, col] = color
    
    return output