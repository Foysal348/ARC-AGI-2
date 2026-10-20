import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Each special cell (non-0, non-1) emits beams in all 4 diagonal directions.
    Beams extend until they hit another special cell or the grid edge.
    Beam cells take the source cell's value.
    If multiple beams reach the same cell, the nearest source wins (ties broken by value).
    """
    rows, cols = grid.shape
    result = grid.copy()
    
    # Find all special cells
    special = np.argwhere((grid != 0) & (grid != 1))
    
    if len(special) == 0:
        return result
    
    # Four diagonal directions: (dr, dc)
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # For each cell, track the best (nearest, highest-value) source
    # We'll collect all beam assignments with their distances
    best_value = {}  # (row, col) -> (distance, value)
    
    for (r, c) in special:
        val = grid[r, c]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            dist = 1
            while 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != 0 and grid[nr, nc] != 1:
                    # Hit a special cell, stop
                    break
                # This cell gets a beam from (r, c)
                key = (nr, nc)
                if key not in best_value:
                    best_value[key] = (dist, val)
                else:
                    old_dist, old_val = best_value[key]
                    if dist < old_dist or (dist == old_dist and val > old_val):
                        best_value[key] = (dist, val)
                nr += dr
                nc += dc
                dist += 1
    
    # Apply beam values to the result
    for (nr, nc), (dist, val) in best_value.items():
        if result[nr, nc] == 0 or result[nr, nc] == 1:
            result[nr, nc] = val
    
    return result